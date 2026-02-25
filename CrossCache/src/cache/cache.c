#include "device.h"
#include "inner.h"

#include "mempool.h"
#include "hashmap.h"
#include "hashfunc.h"
#include "threadpool.h"
#include "sema.h"
#include "atomic.h"
#include "log.h"
#include "sysdef.h"

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define PATH_LEN 128
#define LKUP_MAP_SCALES 4096

#define KV_H_TO_D false
#define KV_D_TO_H true

enum cache_op
{
    OP_STORE = 0,
    OP_LOAD,
};

const char *op_to_str[] = {
    [OP_STORE] = "STORE",
    [OP_LOAD] = "LOAD",
};

struct cache_manager {
    uint32_t chunk_size;
    uint64_t pool_size;
    hashmap_t *model_rt_map;

    threadpool_t *io;
    threadpool_t *copy;

    adaptor_t *adaptor;

    char *local_dir;
    int dirfd;
};

struct cache_manager g_cache_mgr;

struct model_rt_key {
    char *model_tag;    /* model unique tag, like model name */
    uint8_t tp_size;    /* model running number of TP */
    uint8_t tp_id;  /* TP id */
};

struct model_rt_entry {
    struct model_rt_key key;
    uint32_t block_num;     /* total block_num/num_pages for this model */
    uint32_t block_size;    /* page_size for the running model */
    uint32_t objsize;
    uint8_t element_size;   /* how many bytes the kv element occupies */
    hashmap_t *lkup_map;
    mempool_t *mempool;
    uint64_t *smapping;     /* global slot mapping in memory */
    uint64_t pool_size;
    void *host_pinned_addr;     /* cache memory pool */
    void *dev_addr;     /* cache memory pool (device accessiable address) */
    int32_t ref;
    hashlink_t node;
};

static int rt_cmp(void *first, void *second)
{
    struct model_rt_key *key1 = (struct model_rt_key *)first;
    struct model_rt_key *key2 = (struct model_rt_key *)second;

    if (strcmp(key1->model_tag, key2->model_tag))
        return -1;
    if (key1->tp_size != key2->tp_size)
        return -1;
    return (key1->tp_id == key2->tp_id) ? 0 : -1;
}

static uint32_t rt_hash(void *arg)
{
    struct model_rt_key *key = (struct model_rt_key *)arg;

    return hashstr(key->model_tag, strlen(key->model_tag));
}

static void rt_inc(void *args, hashlink_t *data)
{
    struct model_rt_entry *entry =
        container_of(data, struct model_rt_entry, node);

        atomic_s32_inc(&entry->ref);
}

static int rt_dec(void *args, hashlink_t *data)
{
    struct model_rt_entry *entry =
        container_of(data, struct model_rt_entry, node);
    int32_t ref = atomic_s32_dec(&entry->ref);

    if (ref < 0) {
        log_fatal("model runtime entry ref:%d is invalid", ref);
        sys_assert(0);
    }
    return (ref > 0) ? -1 : 0;
}

struct cache_key {
    uint8_t key[32];
};

struct cache_object {
    struct cache_key key;
    void *addr;

    int32_t ref;
    uint64_t ts;

    void *private;
    hashlink_t node;
};

static inline uint64_t _get_curr_time()
{
    struct timespec now = { 0 };

    (void)clock_gettime(CLOCK_MONOTONIC_COARSE, &now);
    return ((uint64_t)now.tv_sec * 1000000000UL + (uint64_t)now.tv_nsec);
}

static inline int key_cmp(void *first, void *second)
{
    struct cache_key *key1 = (struct cache_key *)first;
    struct cache_key *key2 = (struct cache_key *)second;

    return memcmp(key1->key, key2->key, 32);
}

static inline uint32_t key_hash(void *args)
{
    struct cache_key *key = (struct cache_key *)args;
    uint64_t hash = hashstr((const char *)key->key, 32);

    return (uint32_t)hash;
}

static inline void key_access(void *args, hashlink_t *data)
{
    struct cache_object *obj = container_of(data, struct cache_object, node);
    obj->ts = _get_curr_time();
}

enum pipeline_status
{
    PIPELINE_START = 0,
    DISK_TO_MEM,
    MEM_TO_DISK,
    MEM_TO_DEVICE,
    DEVICE_TO_MEM,
    PIPELINE_END,
};

const char *status_to_str[] = {
    [PIPELINE_START] = "START",
    [DISK_TO_MEM] = "DISK->MEM",
    [MEM_TO_DISK] = "MEM->DISK",
    [MEM_TO_DEVICE] = "MEM->DEVICE",
    [DEVICE_TO_MEM] = "DEVICE->MEM",
    [PIPELINE_END] = "END",
};

struct pipeline_ctx {
    enum pipeline_status status;

    struct cache_key *key;
    struct cache_object *memobj;
    uint64_t *block_ids;
    uint32_t block_num;

    uint16_t num_layers;
    uint8_t element_size;
    uint32_t hidden_dim_size;
    uint32_t total_block_num;
    uint32_t block_size;

    uint8_t node_id;
    void *k_ptrs;
    void *v_ptrs;

    void *cb_data;
    void (*cb_func)(int, void *);
};

struct batch_ctx {
    int ret;
    int notback;
    sema_t sem;
};

static inline void _batch_done_cb(int ret, void *arg)
{
    struct batch_ctx *ctx = (struct batch_ctx *)arg;

    if (ret < 0)
        ctx->ret = ret;

    if (atomic_s32_dec(&ctx->notback) == 0)
        sema_up(ctx->sem);
}

/*****************************************************************
******************************************************************/

#define CACHEKEY_TO_NAME(_key, _name)                       \
    do                                                      \
    {                                                       \
        char *_p = _name;                                   \
        for (int i = 0; i < sizeof(*(_key)); i++)       \
            _p += sprintf(_p, "%02x", _key->key[i]);        \
        *_p = '\0';                                         \
    } while(0)

int lookup_file(struct cache_key *key, struct stat *stbuf)
{
    char filename[65];
    char path[256];
    int ret;

    CACHEKEY_TO_NAME(key, filename);
    sprintf(path, "%s/%s", g_cache_mgr.local_dir, filename);
    ret = stat(path, stbuf);
    log_debug("lookup from disk for keypath:%s, ret:%d, errno:%d", path, ret, errno);
    return ret;
}

int write_to_file(struct cache_key *key, void *buffer, uint32_t osize)
{
    char filename[65];
    ssize_t ret;

    CACHEKEY_TO_NAME(key, filename);
    int fd = openat(g_cache_mgr.dirfd, filename, O_CREAT|O_WRONLY|O_TRUNC|O_DIRECT, 0666);
    if (fd < 0) {
        log_error("open(w) kv file:%s failed", filename);
        return fd;
    }

    ret = write(fd, buffer, osize);
    log_debug("write to disk keypath:%s, osize:%u, wsize:%lld, err:%d", filename, osize, ret, (osize != ret) ? errno : 0);
    close(fd);
    return 0;
}

int read_from_file(struct cache_key *key, void *buffer, uint32_t osize)
{
    char filename[65];
    ssize_t ret;

    CACHEKEY_TO_NAME(key, filename);
    int fd = openat(g_cache_mgr.dirfd, filename, O_RDONLY|O_DIRECT);
    if (fd < 0) {
        log_error("open(r) kv file:%s failed", filename);
        return fd;
    }

    ret = read(fd, buffer, osize);
    log_debug("read to disk keypath:%s, osize:%u, rsize:%lld, err:%d", filename, osize, ret, (osize != ret) ? errno : 0);
    close(fd);
    return 0;
}

/*****************************************************************
******************************************************************/

static void free_cache_object(struct cache_object *memobj)
{
    struct model_rt_entry *entry = (struct model_rt_entry *)memobj->private;

    mempool_free(entry->mempool, memobj->addr);
    free(memobj);
}

static int _cache_object_evictable(void *args, hashlink_t *data)
{
    struct cache_object *obj = container_of(data, struct cache_object, node);
    uint64_t *count = (uint64_t *)args;

    uint64_t cur_time = _get_curr_time();
    // 3s as the timeout
    if (cur_time - obj->ts > 1000000000UL * 3) {
        *count = *count + 1;
        free_cache_object(obj);
        return 0;
    }
    return -1;
}

static void evict_cache_object(hashmap_t *map)
{
    uint64_t evict_count = 0;
    uint64_t depth = 10;

#define MAX_EVICT_COUNT 256
    while (evict_count < MAX_EVICT_COUNT && depth > 0) {
        hashmap_eviction(map, depth, &evict_count, _cache_object_evictable);
        depth >>= 1;
    }
    log_info("eviction count:%lu, depth:%lu", evict_count, depth);
}

static struct cache_object *alloc_cache_object(struct cache_key *key,
                                        struct model_rt_entry *entry)
{
    struct cache_object *memobj;

    memobj = (struct cache_object *)malloc(sizeof(struct cache_object));
    if (!memobj) {
        log_error("alloc cache object failed");
        return NULL;
    }
retry:
    memobj->addr = (struct cache_object *)mempool_alloc(entry->mempool);
    if (!memobj->addr) {
        log_warn("mem object begin eviction");
        evict_cache_object(entry->lkup_map);
        goto retry;
    }
    memobj->private = entry;
    memobj->ref = 1;
    memobj->ts = _get_curr_time();
    memcpy(&memobj->key, key, sizeof(struct cache_key));
    memobj->node.key = &memobj->key;
    list_init(&memobj->node.list_node);
    return memobj;
}

static uint64_t *create_slot_mapping(uint32_t num_blocks, uint32_t block_size)
{
    uint64_t *smapping = malloc(num_blocks * block_size * sizeof(uint64_t));
    if (!smapping)
        return NULL;

    uint32_t i, j;
    for (i = 0; i < num_blocks; i++)
        for (j = 0; j < block_size; j++)
            smapping[i * block_size + j] = j + i * block_size;

#ifdef _DEBUG
    printf("======== SLOT MAPPING <%u,%u> ========\n", num_blocks, block_size);
    for (i = 0; i < num_blocks; i++) {
        for (j = 0; j < block_size; j++) {
            printf("%lu ", smapping[i * block_size + j]);
        }
        printf("\n");
    }
#endif
    return smapping;
}

/*
 * generate the slot mapping for each chunk, and reback the device addr
 */
static void *get_slot_mapping(uint8_t nid, uint64_t *smapping, uint64_t *bids, uint32_t bnum,
                        uint64_t num_blocks, uint32_t block_size)
{
    uint32_t i, tokens_sm = bnum * block_size * sizeof(uint64_t);
    void *sm_haddr, *sm_daddr;
    uint64_t *slot_mapping, bid;

    sm_haddr = malloc(tokens_sm);
    if (!sm_haddr)
        return NULL;
    slot_mapping = (uint64_t *)sm_haddr;
    for (i = 0; i < bnum; i++) {
        bid = bids[i];
        memcpy(slot_mapping + i * block_size,
            smapping + bid * block_size,
            block_size * sizeof(uint64_t));
#ifdef _DEBUG
        printf("Generate slot mapping for bid %lu: ", bid);
        for (int j = 0; j < block_size; j++) {
            printf("%lu ", smapping[bid * block_size + j]);
        }
        printf("\n");
#endif
    }
    sm_daddr = wrapper_ptr(g_cache_mgr.adaptor, nid, sm_haddr, tokens_sm);
    free(sm_haddr);
    return sm_daddr;
}

static int transfer_with_device(bool direction, struct pipeline_ctx *pctx,
                        struct cache_object *memobj)
{
    struct model_rt_entry *entry = (struct model_rt_entry *)memobj->private;
    struct transfer_params params;
    int ret;

    /* get the device accessiable address */
    params.caches = (memobj->addr - entry->host_pinned_addr) + entry->dev_addr;
    params.caches_h = memobj->addr;
    log_debug("transfer cache haddr:%p, daddr:%p", params.caches_h, params.caches);
    params.keys_ptrs = pctx->k_ptrs;
    params.value_ptrs = pctx->v_ptrs;
    params.page_buffer_size = pctx->block_size * pctx->total_block_num;
    params.slot_mapping = get_slot_mapping(pctx->node_id, entry->smapping, pctx->block_ids,
                                pctx->block_num, pctx->total_block_num, pctx->block_size);
    params.devid = pctx->node_id;
    params.caches_element_size = pctx->element_size;
    params.num_layers = pctx->num_layers;
    params.num_tokens = g_cache_mgr.chunk_size;
    params.hidden_dim_size = pctx->hidden_dim_size;
    params.direction = direction;
    ret = transfer_kvcache(g_cache_mgr.adaptor, &params);
    unwrapper_ptr(g_cache_mgr.adaptor, params.slot_mapping);
    return ret;
}

static void pipeline_io(void *args);

static void pipeline_copy(void *args)
{
    struct pipeline_ctx *pctx = (struct pipeline_ctx *)args;
    struct cache_object *memobj = pctx->memobj;
    int ret;

    log_debug("memobj haddr:%p in copy stream, status:%s", memobj->addr, status_to_str[pctx->status]);
    if (pctx->status == PIPELINE_START) {
        /* STEP1: device to memory: STEP2: memory to disk */
        ret = transfer_with_device(KV_D_TO_H, pctx, memobj);
        if (ret) {
            if (pctx->cb_func)
                pctx->cb_func(ret, pctx->cb_data);
            free(pctx);
            return;
        }
        pctx->status = DEVICE_TO_MEM;
        log_debug("memobj haddr:%p from device done, status:%s", memobj->addr, status_to_str[pctx->status]);
        threadpool_submit(g_cache_mgr.io, pctx, pipeline_io);
        return;
    }

    sys_assert(pctx->status == DISK_TO_MEM);
    if (pctx->status == DISK_TO_MEM) {
        ret = transfer_with_device(KV_H_TO_D, pctx, memobj);
        pctx->status = PIPELINE_END;
        log_debug("memobj haddr:%p to device done, status:%s", memobj->addr, status_to_str[pctx->status]);
        if (pctx->cb_func)
            pctx->cb_func(ret, pctx->cb_data);
        free(pctx);
    }
}

static void pipeline_io(void *args)
{
    struct pipeline_ctx *pctx = (struct pipeline_ctx *)args;
    struct cache_object *memobj = pctx->memobj;
    struct model_rt_entry *entry = memobj->private;
    hashlink_t *data;
    int ret;

    log_debug("memobj haddr:%p in io stream, status:%s", memobj->addr, status_to_str[pctx->status]);
    if (pctx->status == PIPELINE_START) {
        /* STEP1: disk to memory; STEP2: memory to device */
        ret = read_from_file(&memobj->key, memobj->addr, entry->objsize);
        if (ret) {
            if (pctx->cb_func)
                pctx->cb_func(ret, pctx->cb_data);
            free(pctx);
            return;
        }
        if (EEXIST == hashmap_insert(entry->lkup_map, &memobj->node, &data, NULL, NULL)) {
            log_debug("load from disk done and insert conflict, just free for addr:%p", memobj->addr);
            /* for multi thread insert, we just use the exist one */
            free_cache_object(memobj);
            memobj = container_of(data, struct cache_object, node);
            pctx->memobj = memobj;
        }

        /* copy data to device */
        pctx->status = DISK_TO_MEM;
        log_debug("read done and submit to copy stream for memobj haddr:%p, status:%s", memobj->addr, status_to_str[pctx->status]);
        threadpool_submit(g_cache_mgr.copy, pctx, pipeline_copy);
        return;
    }

    sys_assert(pctx->status == DEVICE_TO_MEM);
    if (pctx->status == DEVICE_TO_MEM) {
        ret = write_to_file(&memobj->key, memobj->addr, entry->objsize);
        if (ret == 0 &&
                EEXIST == hashmap_insert(entry->lkup_map, &memobj->node, &data, NULL, NULL)) {
            log_debug("store to disk done and insert conflict, just free for addr:%p", memobj->addr);
            /* for multi thread insert, we just use the exist one */
            free_cache_object(memobj);
            memobj = container_of(data, struct cache_object, node);
        }
        pctx->status = PIPELINE_END;
        log_debug("write done(%d) and memobj haddr:%p are in cache, status:%s", ret, memobj->addr, status_to_str[pctx->status]);
        if (pctx->cb_func)
            pctx->cb_func(ret, pctx->cb_data);
        free(pctx);
    }
}

static int cache_op(struct worker_context *wctx, uint8_t *keys,
                uint32_t key_len, uint64_t *bids, uint32_t bnum, int op)
{
    struct model_rt_key rt_key;
    struct model_rt_entry *entry;
    struct cache_object *memobj;
    hashlink_t *data;
    int ret, i, kpos;

    int blocks_per_chunk = g_cache_mgr.chunk_size / wctx->block_size;
    int key_num = bnum / blocks_per_chunk;

    /* find out which kinds of model for this request */
    rt_key.model_tag = wctx->model.model_name;
    rt_key.tp_size = wctx->world_size;
    rt_key.tp_id = wctx->rank_id;
    ret = hashmap_search(g_cache_mgr.model_rt_map, &rt_key, &data, NULL, NULL);
    if (ret != EEXIST)
        return -ENOENT;
    entry = container_of(data, struct model_rt_entry, node);
    log_debug("find rt model[%s,%u,%u],op:%s", rt_key.model_tag, rt_key.tp_size, rt_key.tp_id, op_to_str[op]);

    /* submit the copy request for each key for this model in parallel */
    struct batch_ctx ctx = {0, 1};
    sema_init(ctx.sem);
    for (i = 0, kpos = 0; i < key_num; i++, kpos += key_len) {
        struct pipeline_ctx *pctx = (struct pipeline_ctx *)malloc(sizeof(struct pipeline_ctx));
        if (!pctx) {
            log_error("pipeline ctx alloc failed");
            break;
        }
        pctx->key = (struct cache_key *)(keys + kpos);
        pctx->status = PIPELINE_START;
        pctx->block_ids = bids + (i * blocks_per_chunk);
        pctx->block_num = blocks_per_chunk;
        pctx->cb_data = &ctx;
        pctx->num_layers = wctx->model.num_layers;
        pctx->element_size = wctx->model.element_size;
        pctx->hidden_dim_size = wctx->model.hidden_dim_size;
        pctx->total_block_num = wctx->num_blocks;
        pctx->block_size = wctx->block_size;
        pctx->node_id = wctx->node_id;
        pctx->k_ptrs = wctx->k_ptrs;
        pctx->v_ptrs = wctx->v_ptrs;
        pctx->cb_func = _batch_done_cb;     /* callback for this batch */
        if (op == OP_LOAD) {
            ret = hashmap_search(entry->lkup_map, pctx->key, &data, NULL, key_access);
            if (ret == EEXIST) {
                /* fast path for memory cache */
                memobj = container_of(data, struct cache_object, node);
                log_debug("load hit the memory cache, just copy, haddr:%p", memobj->addr);
                pctx->memobj = memobj;
                pctx->status = DISK_TO_MEM;
                (void)atomic_s32_inc(&ctx.notback);
                threadpool_submit(g_cache_mgr.copy, pctx, pipeline_copy);
                continue;
            }
        }
        memobj = alloc_cache_object(pctx->key, entry);
        if (!memobj) {
            log_error("alloc memobj failed in store");
            free(pctx);
            break;
        }
        pctx->memobj = memobj;
        log_debug("new memobj, haddr:%p, op:%s", memobj->addr, op_to_str[op]);

        /* submit task into pipeline workers */
        (void)atomic_s32_inc(&ctx.notback);
        if (op == OP_STORE) {
            threadpool_submit(g_cache_mgr.copy, pctx, pipeline_copy);
        } else {
            threadpool_submit(g_cache_mgr.io, pctx, pipeline_io);
        }
    }

    if (atomic_s32_dec(&ctx.notback) > 0)
        sema_down(ctx.sem);
    sema_fini(ctx.sem);
    return ctx.ret;
}

/*****************************************************************
******************************************************************/

int cache_register(struct worker_context *wctx)
{
    struct model_rt_key key;
    struct model_rt_entry *entry;
    void *haddr, *daddr, *dptrs;
    uint32_t chunk_size_in_bytes;
    int ret;

    dptrs = open_ipc_keys(g_cache_mgr.adaptor, wctx->node_id, wctx->per_len,
                    wctx->model.num_layers, wctx->k_raw_ptrs);
    if (!dptrs)
        return -ENOMEM;
    wctx->k_ptrs = dptrs;
    if (wctx->v_raw_ptrs) {
        dptrs = open_ipc_keys(g_cache_mgr.adaptor, wctx->node_id, wctx->per_len,
                    wctx->model.num_layers, wctx->v_raw_ptrs);
        if (!dptrs) {
            ret = -ENOMEM;
            goto unpin_k;
        }
        wctx->v_ptrs = dptrs;
    }

    chunk_size_in_bytes = wctx->per_token_size * g_cache_mgr.chunk_size;
    sys_assert((chunk_size_in_bytes % 4096) == 0);
    key.model_tag = wctx->model.model_name;
    key.tp_size = wctx->world_size;
    key.tp_id = wctx->rank_id;
    ret = hashmap_search(g_cache_mgr.model_rt_map, &key, NULL, NULL, rt_inc);
    if (ret == EEXIST)
        goto unpin_v;
    ret = -ENOMEM;
    entry = (struct model_rt_entry *)malloc(sizeof(struct model_rt_entry));
    if (!entry)
        goto err_search;
    entry->key.model_tag = strdup(wctx->model.model_name);
    if (!entry->key.model_tag)
        goto free_entry;
    entry->key.tp_size = wctx->world_size;
    entry->key.tp_id = wctx->rank_id;
    /* create lookup hashmap for each sharing domain */
    ret = hashmap_create(LKUP_MAP_SCALES, key_cmp, key_hash, &entry->lkup_map);
    if (ret)
        goto free_tag;
    /* alloc pinned address for each sharing domain */
    uint64_t m_count = (g_cache_mgr.pool_size /  chunk_size_in_bytes);
    uint64_t size = m_count * chunk_size_in_bytes;
    haddr = alloc_pinned_ptr(g_cache_mgr.adaptor, wctx->node_id, size, 0, &daddr);
    if (!haddr)
        goto free_lkupmap;
    sys_assert(((uint64_t)haddr % 4096) == 0);
    entry->pool_size = size;
    entry->objsize = chunk_size_in_bytes;
    entry->element_size = wctx->model.element_size;
    entry->host_pinned_addr = haddr;
    entry->dev_addr = daddr;
    entry->smapping = create_slot_mapping(wctx->num_blocks, wctx->block_size);
    if (!entry->smapping)
        goto free_pinned_mem;
    sys_assert(entry->smapping != NULL);
    entry->mempool = mempool_create(chunk_size_in_bytes, m_count, haddr);
    if (!entry->mempool)
        goto free_slotmapping;
    entry->node.key = &entry->key;
    entry->ref = 1;
    hashmap_insert(g_cache_mgr.model_rt_map, &entry->node, NULL, NULL, NULL);
    log_debug("register done rt model[%s,%u,%u] hbase:%p, dbase:%p, chunkbytes:%u", key.model_tag, key.tp_size, key.tp_id, haddr, daddr, chunk_size_in_bytes);
    return 0;
free_slotmapping:
    free(entry->smapping);
free_pinned_mem:
    free_pinned_ptr(g_cache_mgr.adaptor, entry->host_pinned_addr, size);
free_lkupmap:
    hashmap_destroy(entry->lkup_map, NULL, NULL);
free_tag:
    free(entry->key.model_tag);
free_entry:
    free(entry);
err_search:
    (void)hashmap_delete(g_cache_mgr.model_rt_map, &key, NULL, NULL, rt_dec);
unpin_v:
    if (wctx->v_ptrs)
        close_ipc_keys(g_cache_mgr.adaptor, wctx->v_ptrs);
unpin_k:
    close_ipc_keys(g_cache_mgr.adaptor, wctx->k_ptrs);
    return ret;
}

void cache_unregister(struct worker_context *wctx)
{
    struct model_rt_key key;
    struct model_rt_entry *entry;
    hashlink_t *data = NULL;

    key.model_tag = wctx->model.model_name;
    key.tp_size = wctx->world_size;
    key.tp_id = wctx->rank_id;
    if (0 != hashmap_delete(g_cache_mgr.model_rt_map, &key, &data, NULL, rt_dec))
        return;

    entry = container_of(data, struct model_rt_entry, node);
    mempool_destroy(entry->mempool);
    free(entry->smapping);
    free_pinned_ptr(g_cache_mgr.adaptor, entry->host_pinned_addr, entry->pool_size);
    hashmap_destroy(entry->lkup_map, NULL, NULL);
    free(entry->key.model_tag);
    free(entry);
    close_ipc_keys(g_cache_mgr.adaptor, wctx->k_ptrs);
    if (wctx->v_ptrs)
        close_ipc_keys(g_cache_mgr.adaptor, wctx->v_ptrs);
    reset_device(g_cache_mgr.adaptor, wctx->node_id);
    log_debug("unregister done rt model[%s,%u,%u]", key.model_tag, key.tp_size, key.tp_id);
}

int cache_lookup(struct worker_context *wctx, uint8_t *keys,
                uint32_t key_num, uint32_t key_len)
{
    struct model_rt_key rt_key;
    struct model_rt_entry *entry;
    hashlink_t *exist_data;
    struct cache_key *key;
    int ret, i, pos;

    /* find the rt entry for each model */
    rt_key.model_tag = wctx->model.model_name;
    rt_key.tp_size = wctx->world_size;
    rt_key.tp_id = wctx->rank_id;
    ret = hashmap_search(g_cache_mgr.model_rt_map, &rt_key, &exist_data, NULL, NULL);
    if (ret != EEXIST)
        return 0;
    entry = container_of(exist_data, struct model_rt_entry, node);
    log_debug("find rt model[%s,%u,%u]", rt_key.model_tag, rt_key.tp_size, rt_key.tp_id);

    for (i = 0, pos = 0; i < key_num; i++, pos += key_len) {
        /* handle for each key sequence */
        key = (struct cache_key *)(keys + pos);
        /* hit in memory: search and keep reference in scheduler context. TODO. */
        ret = hashmap_search(entry->lkup_map, key, &exist_data, NULL, key_access);
        if (ret == EEXIST)
            continue;
        /* lookup from local dcache */
        struct stat st;
        ret = lookup_file(key, &st);
        if (ret)
            break;
    }
    return pos;
}

int cache_store(struct worker_context *wctx, uint8_t *keys,
            uint32_t key_len, uint64_t *bids, uint32_t bnum)
{
    return cache_op(wctx, keys, key_len, bids, bnum, OP_STORE);
}

int cache_load(struct worker_context *wctx, uint8_t *keys,
            uint32_t key_len, uint64_t *bids, uint32_t bnum)
{
    return cache_op(wctx, keys, key_len, bids, bnum, OP_LOAD);
}

int cache_init(uint32_t chunk_size, uint64_t pool_size, const char *root_dir)
{
    int ret;

    g_cache_mgr.chunk_size = chunk_size;
    g_cache_mgr.pool_size = pool_size;
#ifdef _NPU
    g_cache_mgr.adaptor = npu_adaptor_create();
#elif _NPUSDK
    g_cache_mgr.adaptor = npu_sdk_adaptor_create();
#else
    g_cache_mgr.adaptor = mem_adaptor_create();
#endif
    sys_assert(g_cache_mgr.adaptor != 0);
    ret = hashmap_create(8, rt_cmp, rt_hash, &g_cache_mgr.model_rt_map);
    sys_assert(ret == 0);

    g_cache_mgr.io = threadpool_create("CaIO", 4);
    g_cache_mgr.copy = threadpool_create("CaCopy", 4);

    g_cache_mgr.local_dir = strdup(root_dir);
    if (0 != access(root_dir, 0)) {
        ret = mkdir(root_dir, S_IRWXU | S_IRGRP | S_IXGRP);
        sys_assert(ret == 0);
    }
    g_cache_mgr.dirfd = open(root_dir, O_RDONLY|O_DIRECTORY);
    sys_assert(g_cache_mgr.dirfd > 0);
    return 0;
}

void cache_exit()
{
    close(g_cache_mgr.dirfd);
    free(g_cache_mgr.local_dir);
    threadpool_destroy(g_cache_mgr.copy);
    threadpool_destroy(g_cache_mgr.io);
    hashmap_destroy(g_cache_mgr.model_rt_map, NULL, NULL);
#ifdef _NPU
    npu_adaptor_destroy(g_cache_mgr.adaptor);
#elif _NPUSDK
    npu_sdk_adaptor_destroy(g_cache_mgr.adaptor);
#else
    mem_adaptor_destroy(g_cache_mgr.adaptor);
#endif
}