#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <zmq.h>
#include <getopt.h>

#include "spinlock.h"
#include "threadpool.h"
#include "log.h"
#include "protocol.h"
#include "inner.h"
#include "cache.h"
#include "sysdef.h"

#define MAX_NODE_NUM 8
#define INVALID_NODE_ID MAX_NODE_NUM

const char *cop_to_str[] = {
    [CACHE_REGISTER] = "REGISTER",
    [CACHE_UNREGISTER] = "UNREGISTER",
    [CACHE_LOOKUP] = "LOOKUP",
    [CACHE_STORE] = "STORE",
    [CACHE_LOAD] = "LOAD",
    [CACHE_GET_INFO] = "GETINFO",
    [CACHE_MAX] = "UNKOWN",
};

struct cache_config {
    uint32_t chunk_size;
    uint32_t port;
    uint64_t pool_size;
    const char *local_path;
};

struct cache_server {
    struct cache_config config;
    struct worker_context wcontext[MAX_NODE_NUM];

    /* for zeromq */
    void *context;
    void *socket;
    spinlock_t rsp_lock;

    threadpool_t *common_thp;
    threadpool_t *load_thp;
    threadpool_t *store_thp;
};

struct cache_server g_server;

/*****************************************************************
******************************************************************/

static void _free_wcontext(struct worker_context *wctx)
{
    cache_unregister(wctx);
    wctx->node_id = INVALID_NODE_ID;
    free(wctx->model.model_name);
    wctx->model.model_name = NULL;
    wctx->k_raw_ptrs = NULL;
    wctx->v_raw_ptrs = NULL;
    wctx->k_ptrs = NULL;
    wctx->v_ptrs = NULL;
}

static int do_register(struct worker_context *wctx)
{
    struct worker_context *gwctx;
    int ret;

    if (wctx->node_id >= MAX_NODE_NUM)
        return -ERANGE;

    gwctx = &g_server.wcontext[wctx->node_id];
    if (gwctx->node_id != INVALID_NODE_ID) {
        log_warn("worker:%u updated", wctx->node_id);
        _free_wcontext(gwctx);
    }

    ret = -ENOMEM;
    gwctx->model.model_name = strdup(wctx->model.model_name);
    if (!gwctx->model.model_name)
        goto out;
    gwctx->model.element_size = wctx->model.element_size;
    gwctx->model.num_layers = wctx->model.num_layers;
    gwctx->model.hidden_dim_size = wctx->model.hidden_dim_size;
    gwctx->world_size = wctx->world_size;
    gwctx->rank_id = wctx->rank_id;
    gwctx->per_token_size = wctx->per_token_size;
    gwctx->num_blocks = wctx->num_blocks;
    gwctx->block_size = wctx->block_size;
    ret = cache_register(wctx);
    if (ret)
        goto free_str;

    gwctx->k_raw_ptrs = wctx->k_raw_ptrs;
    gwctx->k_ptrs = wctx->k_ptrs;
    gwctx->v_raw_ptrs = wctx->v_raw_ptrs;
    gwctx->v_ptrs = wctx->v_ptrs;
    gwctx->node_id = wctx->node_id;
    log_debug("nodeid:%u register done,kvptr:%p", gwctx->node_id, gwctx->k_ptrs);
    return 0;
free_str:
    free(gwctx->model.model_name);
    gwctx->model.model_name = NULL;
out:
    return ret;
}

static int do_unregister(int nid)
{
    struct worker_context *gwctx;

    if (nid >= MAX_NODE_NUM)
        return -ERANGE;
    /* release the worker context */
    gwctx = &g_server.wcontext[nid];
    if (gwctx->node_id == INVALID_NODE_ID) {
        log_warn("worker:%u has not registered", nid);
        return 0;
    }
    _free_wcontext(gwctx);
    return 0;
}

static int do_lookup(uint8_t *keys, uint32_t key_num, uint32_t key_len,
            char *name, uint8_t world_size)
{
    int hitted;
    struct worker_context wctx;

    /* only pass to lookup model */
    wctx.model.model_name = name;
    wctx.world_size = world_size;
    // default use the first one: because tp 0 is enough
    wctx.rank_id = 0;

    hitted = cache_lookup(&wctx, keys, key_num, key_len);
    log_debug("[do_lookup] model[%s,%u] knum:%u,klen:%u,hitted pos:%d", name, world_size, key_num, key_len, hitted);
    return (hitted / key_len) * g_server.config.chunk_size;
}

static int do_load(int nid, uint8_t *keys, uint32_t key_len, uint32_t key_num,
            uint64_t *block_ids, uint32_t block_num)
{
    struct worker_context *wctx;

    if (nid >= MAX_NODE_NUM)
        return -ERANGE;
    wctx = &g_server.wcontext[nid];
    if (wctx->node_id == INVALID_NODE_ID) {
        log_warn("worker:%u has not been registered", nid);
        return -EINVAL;
    }

#ifdef _DEBUG
    printf("LOAD bids:[ ");
    for (int i = 0; i < block_num; i++) {
        printf("%lu ", block_ids[i]);
    }
    printf("]\n");
#endif
    sys_assert((key_num * g_server.config.chunk_size == block_num * wctx->block_size));
    return cache_load(wctx, keys, key_len, block_ids, block_num);
}

static int do_store(int nid, uint8_t *keys, uint32_t key_len, uint32_t key_num,
            uint64_t *block_ids, uint32_t block_num)
{
    struct worker_context *wctx;
    
    if (nid >= MAX_NODE_NUM)
        return -ERANGE;
    wctx = &g_server.wcontext[nid];
    if (wctx->node_id == INVALID_NODE_ID) {
        log_warn("worker:%u has not been registered", nid);
        return -EINVAL;
    }

#ifdef _DEBUG
    printf("STORE bids:[ ");
    for (int i = 0; i < block_num; i++) {
        printf("%lu ", block_ids[i]);
    }
    printf("]\n");
#endif
    sys_assert((key_num * g_server.config.chunk_size == block_num * wctx->block_size));
    return cache_store(wctx, keys, key_len, block_ids, block_num);
}

/*****************************************************************
******************************************************************/

struct request_ctx {
    void *id;
    int id_len;
    void *body;
};

#define ALLOC_REQ_CTX(_ctx, _id, _id_len, _header)             \
    do {                                                         \
        _ctx = malloc(sizeof(struct request_ctx));               \
        if (_ctx) {                                              \
            _ctx->id = _id;                                       \
            _ctx->id_len = _id_len;                                 \
            _ctx->body = _header;                                   \
        }                                                         \
    } while(0)

#define FREE_REQ_CTX(_ctx)                                       \
    do {                                                         \
        if (_ctx) {                                              \
            free(_ctx->id);                                       \
            free(_ctx->body);                                       \ 
            free(_ctx);                                           \
        }                                                         \
    } while(0)

static void reply(int ret, void *id, int id_len, uint32_t uid)
{
    struct msg_response rsp;

    rsp.ok = (uint32_t)ret;
    rsp.uid = uid;
    /* zmq socket is not thread-safe */
    spinlock_lock(&g_server.rsp_lock);
    zmq_send(g_server.socket, id, id_len, ZMQ_SNDMORE);
    zmq_send(g_server.socket, &rsp, sizeof(struct msg_response), 0);
    spinlock_unlock(&g_server.rsp_lock);
    log_debug("send response for uid:%u, ok:%d", rsp.uid, ret);
}

static uint8_t dtype_to_bytes(uint8_t flag)
{
    if (flag & KV_DTYPE_1BYTE)
        return 1;
    if (flag & KV_DTYPE_2BYTE)
        return 2;
    if (flag & KV_DTYPE_4BYTE)
        return 4;
    return 2;
}

static void process_register(void *arg)
{
    struct request_ctx *ctx = (struct request_ctx *)arg;
    struct msg_header *msg = ctx->body;
    struct msg_register *reg = (struct msg_register *)msg->data;

    int klen_in_bytes = reg->num_layers * reg->first_key_in_bytes;
    void *ptr = (void *)reg->data;
    int ret;

    struct worker_context wctx = {};
    int dtype = dtype_to_bytes(reg->kv_flags);  /* assume bfloat takes 2 Bytes */

    if (msg->version != CACHE_VERSION) {
        log_error("supported version is not matched");
        ret = -EINVAL;
        goto resp;
    }
    wctx.node_id = msg->node_id;
    wctx.world_size = reg->world_size;
    wctx.rank_id = reg->rank_id;
    wctx.num_blocks = reg->num_blocks;
    wctx.block_size = reg->block_size;
    wctx.model.model_name = (char *)ptr;
    wctx.per_len = reg->first_key_in_bytes;

    ptr = (void *)((uint8_t *)ptr + strlen(wctx.model.model_name) + 1);
    wctx.k_raw_ptrs = ptr;
    if (reg->kv_flags & K_V_MASK) {
        /* kv shape * kv type, in memory */
        wctx.per_token_size = 2 * reg->num_layers * reg->hidden_dim_size * dtype;
        wctx.v_raw_ptrs = wctx.k_raw_ptrs + klen_in_bytes;
    } else {
        wctx.per_token_size = reg->num_layers * reg->hidden_dim_size * dtype;
    }

    wctx.model.element_size = dtype;
    wctx.model.num_layers = reg->num_layers;
    wctx.model.hidden_dim_size = reg->hidden_dim_size;

    log_debug("[Register] nid:%u,rid:%u,nblocks:%u,bsize:%u,kbytes:%u,flags:%o,dtype:%d,per_token_size:%u,nlay:%u,mname:%s",
        wctx.node_id, wctx.rank_id, wctx.num_blocks, wctx.block_size, reg->first_key_in_bytes,
        reg->kv_flags, dtype, wctx.per_token_size, reg->num_layers, wctx.model.model_name);
    ret = do_register(&wctx);
resp:
    reply(ret, ctx->id, ctx->id_len, msg->uid);
    FREE_REQ_CTX(ctx);
}

static void process_unregister(void *arg)
{
    struct request_ctx *ctx = (struct request_ctx *)arg;
    struct msg_header *msg = ctx->body;

    log_debug("[Unregister] nid:%u", msg->node_id);
    int ret = do_unregister(msg->node_id);
    reply(ret, ctx->id, ctx->id_len, msg->uid);
    FREE_REQ_CTX(ctx);
}

static void process_lookup(void *arg)
{
    struct request_ctx *ctx = (struct request_ctx *)arg;
    struct msg_header *msg = ctx->body;
    struct msg_common *comm = (struct msg_common *)msg->data;
    uint8_t *keys = (uint8_t *)comm->data;
    struct msg_lookup *lkbody = (struct msg_lookup *)(keys + comm->keys_num * comm->key_len);
    uint8_t world_size = lkbody->world_size;
    char *model_name = (char *)lkbody->data;

    log_debug("[Lookup] nid:%u,knum:%u,klen:%u", msg->node_id, comm->keys_num, comm->key_len);
    int ret = do_lookup(keys, comm->keys_num, comm->key_len, model_name, world_size);
    reply(ret, ctx->id, ctx->id_len, msg->uid);
    FREE_REQ_CTX(ctx);
}

static void process_load(void *arg)
{
    struct request_ctx *ctx = (struct request_ctx *)arg;
    struct msg_header *msg = ctx->body;
    struct msg_common *comm = (struct msg_common *)msg->data;
    uint8_t *keys = (uint8_t *)comm->data;
    uint8_t *bids = (uint8_t *)comm->data + (comm->keys_num * comm->key_len);
    uint32_t bnum = (msg->len - sizeof(struct msg_common) - (bids - keys)) / sizeof(uint64_t);

    log_debug("[Load] nid:%u,knum:%u,klen:%u,bnum:%u", msg->node_id, comm->keys_num, comm->key_len, bnum);
    int ret = do_load(msg->node_id, keys, comm->key_len, comm->keys_num, (uint64_t *)bids, bnum);
    reply(ret, ctx->id, ctx->id_len, msg->uid);
    FREE_REQ_CTX(ctx);
}

static void process_store(void *arg)
{
    struct request_ctx *ctx = (struct request_ctx *)arg;
    struct msg_header *msg = ctx->body;
    struct msg_common *comm = (struct msg_common *)msg->data;
    uint8_t *keys = (uint8_t *)comm->data;
    uint8_t *bids = (uint8_t *)comm->data + (comm->keys_num * comm->key_len);
    uint32_t bnum = (msg->len - sizeof(struct msg_common) - (bids - keys)) / sizeof(uint64_t);

    log_debug("[Store] nid:%u,knum:%u,klen:%u,bnum:%u", msg->node_id, comm->keys_num, comm->key_len, bnum);
    int ret = do_store(msg->node_id, keys, comm->key_len, comm->keys_num, (uint64_t *)bids, bnum);
    reply(ret, ctx->id, ctx->id_len, msg->uid);
    FREE_REQ_CTX(ctx);
}

static void handle_request(struct msg_header *header, void *id, int id_len)
{
    struct request_ctx *ctx;

    ALLOC_REQ_CTX(ctx, id, id_len, header);
    if (!ctx) {
        log_error("alloc for request context failed");
        free(header);
        free(id);
        return;
    }

    switch (header->op) { 
    case CACHE_REGISTER:
        threadpool_submit(g_server.common_thp, ctx, process_register);
        break;
    case CACHE_UNREGISTER:
        threadpool_submit(g_server.common_thp, ctx, process_unregister);
        break;
    case CACHE_LOOKUP:
        threadpool_submit(g_server.common_thp, ctx, process_lookup);
        break;
    case CACHE_STORE:
        threadpool_submit(g_server.store_thp, ctx, process_store);
        break;
    case CACHE_LOAD:
        threadpool_submit(g_server.load_thp, ctx, process_load);
        break;
    default:
        log_error("Unknown operations:%d", header->op);
        FREE_REQ_CTX(ctx);
        break;
    }
}

/*****************************************************************
******************************************************************/

int init_server_resource()
{
    for (int i = 0; i < MAX_NODE_NUM; i++)
        g_server.wcontext[i].node_id = INVALID_NODE_ID;

        spinlock_init(&g_server.rsp_lock);
        g_server.common_thp = threadpool_create("CommThp", 2);
        if (!g_server.common_thp) {
            log_error("create common thp failed");
            return -1;
        }

        g_server.load_thp = threadpool_create("LoadThp", 4);
        if (!g_server.load_thp) {
            log_error("create load thp failed");
            return -1;
        }

        g_server.store_thp = threadpool_create("StoreThp", 4);
        if (!g_server.store_thp) {
            log_error("create store thp failed");
            return -1;
        }
        return cache_init(g_server.config.chunk_size, g_server.config.pool_size,
                g_server.config.local_path);
}

void exit_server_resource()
{
    cache_exit();
    threadpool_destroy(g_server.store_thp);
    threadpool_destroy(g_server.load_thp);
    threadpool_destroy(g_server.common_thp);
    spinlock_destroy(&g_server.rsp_lock);
}

void cache_init_configure(void)
{
    g_server.config.port = 5555;
    g_server.config.chunk_size = 256;
    g_server.config.pool_size = 4 * 1024UL * 1024UL * 1024UL;
    g_server.config.local_path = "/var/log/crosscache";
}

static struct option long_options[] =
{
    {"help", no_argument, NULL, 'h'},
    {"port", required_argument, 0, 'P'},
    {"cachedir", required_argument, 0, 1},
    {"chunksize", required_argument, 0, 2},
    {0, 0, 0, 0},
};

static void usage(int argc, char **argv)
{
    printf("Usage: %s [options]\n", argv[0]);
    printf("Options:\n");
    printf("  -h, --help                   Print this help message\n");
    printf("  -P, --port <port>            Specify the listening port (default: 5555)\n");
    printf("  --chunksize <size>           Specify the chunk size (default: 256)\n")
    printf("  --cachedir <dir>             Specify the local dir for k/v files (default: /var/log/crosscache)\n");
}

static int cache_parse_options_cfg(int argc, char **argv)
{
    int opt;
    char *endptr;

    while ((opt = getopt_long(argc, argv, "hP:",
                long_options, NULL)) != - 1) {
        switch (opt) {
        case 'h':
            usage(argc, argv);
            exit(0);
        case 'P':
            g_server.config.port = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid port: %s\n", optarg);
                return -EINVAL;
            }
            break;
        case 1:
            g_server.config.local_path = optarg;
            break;
        case 2:
            g_server.config.chunk_size = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid chunksize: %s\n", optarg);
                return -EINVAL;
            }
            break;
        default:
            return -EINVAL;
        }
    }

    return 0;
}

int main(int argc, char *argv[])
{
    cache_init_configure();
    int ret = cache_parse_options_cfg(argc, argv);
    if (ret != 0) {
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return ret;
    }
    ret = init_server_resource();
    if (ret)
        goto exit;

    g_server.context = zmq_ctx_new();
    g_server.socket = zmq_socket(g_server.context, ZMQ_ROUTER);
    spinlock_init(&g_server.rsp_lock);
    char addr[20];
    sprintf(addr, "tcp://*:%u", g_server.config.port);
    ret = zmq_bind(g_server.socket, addr);
    if (ret) {
        log_error("Failed to bind socket:%s", zmq_strerror(errno));
        goto err;
    }

    log_info("CrossCache listening on tcp://*:%u, chunk_size:%u, cachedir:%s",
              g_server.config.port, g_server.config.chunk_size, g_server.config.local_path);
    int timeout = 5000;
    zmq_setsockopt(g_server.socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_pollitem_t items[] = {{g_server.socket, 0, ZMQ_POLLIN, 0}};
    while (1) {
        int rc = zmq_poll(items, 1, 1000);
        if (rc < 0) break;
        if (rc == 0) continue;
        if (!(items[0].revents & ZMQ_POLLIN))
            continue;

        zmq_msg_t id, body;

        /* first frame: client identity */
        zmq_msg_init(&id);
        zmq_msg_init(&body);
        zmq_msg_recv(&id, g_server.socket, 0);
        zmq_msg_recv(&body, g_server.socket, 0);

        int id_len = zmq_msg_size(&id);
        char *identity = malloc(id_len + 1);
        memcpy(identity, zmq_msg_data(&id), id_len);
        identity[id_len] = '\0';

        size_t body_size = zmq_msg_size(&body);
        char *body_str = malloc(body_size);
        memcpy(body_str, zmq_msg_data(&body), body_size);

        zmq_msg_close(&id);
        zmq_msg_close(&body);

        struct msg_header *header = (struct msg_header *)body_str;
        log_debug("Get Msg op:%s, node_id:%u, len:%u, uid:%u",
            cop_to_str[header->op >= CACHE_MAX ? CACHE_MAX : header->op],
            header->node_id, header->len, header->uid);
        handle_request(header, identity, id_len);
    }
err:
    zmq_close(g_server.socket);
    zmq_ctx_destroy(g_server.context);
exit:
    exit_server_resource();
    return ret;
}