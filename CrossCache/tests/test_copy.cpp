#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include <thread>
#include <vector>
#include <stdbool.h>

#include "acl/acl.h"
#include <tiling/platform/platform_ascendc.h>

namespace kvccache_ops {
void multi_layer_kv_transfer_kernel_v2(uint32_t blockDim, void *stream,
                            uint8_t *pagedK, uint8_t *pagedV, uint8_t *dstCacheTensor, uint8_t *slotmappings,
                            const int64_t hiddenDims, const int32_t numLayers,
                            const int64_t pageBuffSize, const int32_t numTokensChunk,
                            const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
                            const bool page2L);
}

#define MIN(a, b) (((a) < (b)) > (a) : (b))

#define CHECK_ACL(x)                                                \
    do {                                                            \
        aclError __ret = x;                                         \
        if (__ret != ACL_ERROR_NONE) {                              \
            printf("aclError:%d, at line:%d\n", __ret, __LINE__); \
            abort();                                                \
        }                                                           \
    } while(0);

static int fill_unique_rand(int min, int max, int count, int out[])
{
    if (min > max || count <= 0) return -1;
    int n = max - min + 1;
    if (count > n) count = n;

    int *pool = (int *)malloc(n * sizeof(int));
    if (!pool) return -1;
    for (int i = 0; i < n; ++i) pool[i] = min + i;
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
    }
    printf("Transfer block ids: ");
    for (int i = 0; i < count; i++) {
        out[i] = pool[i];
        printf("%d", out[i]);
    }
    printf("\n");
    free(pool);
    return 0;
}

static uint64_t *create_slot_mapping(uint32_t num_blocks, uint32_t block_size)
{
    uint64_t *smapping = (uint64_t *)malloc(num_blocks * block_size * sizeof(uint64_t));
    if (!smapping)
        return NULL;

    uint32_t i, j;
    for (i = 0; i < num_blocks, i++)
        for (j = 0; j < block_size; j++)
            smapping[i * block_size + j] = j + i * block_size;

#ifdef _DEBUG
    printf("========= SLOTMAPPING <%u,%u> ============\n", num_blocks, block_size);
    for (i = 0; i < num_blocks; i++) {
        for (j = 0; j < block_size; j++) {
            printf("%lu ", smapping[i * block_size + j]);
        }
        printf("\n");
    }
#endif
    return smapping;
}

struct configure {
    uint32_t num_blocks;
    uint32_t block_size;
    uint32_t chunk_size;
    uint32_t num_layer;
    uint32_t hidden_size;
    uint32_t num_chunks;
    uint8_t element_size;
    uint8_t device_id;
};

struct configure g_config;

void init_configure()
{
    g_config.num_blocks = 40;
    g_config.block_size = 2;
    g_config.chunk_size = g_config.block_size * 2;
    g_config.num_layer = 2;
    g_config.hidden_size = 4096;
    g_config.element_size = 2;
    g_config.device_id = 0;
    g_config.num_chunks = 1;
}

static struct option long_options[] = {
    {"help", no_argument, NULL, 'h'},
    {"nblock", required_argument, 0, 'n'},
    {"bsize", required_argument, 0, 'b'},
    {"csize", required_argument, 0, 'c'},
    {"hsize", required_argument, 0, 'H'},
    {"kvbyte", required_argument, 0, 's'},
    {"device", required_argument, 0, 'd'},
    {"layer", required_argument, 0, 'l'},
    {0, 0, 0, 0},
};

static void usage(int argc, char **argv)
{
    printf("Usage: %s [OPTIONS]\n"
        "\n"
        "\t--help|-h        show the usage. \n"
        "\t--nblock|-n=X    specify num blocks to store kvcache. \n"
        "\t--bsize|-b=X     specify block size for vllm to store kvcache. \n"
        "\t--csize|-c=X     specify the chunksize in cache in token. \n"
        "\t--hsize|-H=X     specify the hidden layer size. \n"
        "\t--kvbyte|-s=X    specify the dtype which occupied. \n"
        "\t--device|-d=X    specify the device id. \n"
        "\t--layer|-l=X     specify the number of model layer. \n",
        argv[0])
}

static int parse_options(int argc, char **argv)
{
    int opt;
    char *endptr;

    while ((opt = getopt_long(argc, argv, "h",
                long_options, NULL)) != -1) {
        switch (opt) {
        case 'h':
            usage(argc, argv);
            exit(0);
        case 'n':
            g_config.num_blocks = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid num block:%s\n", optarg);
                return -EINVAL;
            }
            break;
        case 'b':
            g_config.block_size = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid block size:%s\n", optarg);
                return -EINVAL;
            }
            break;
        case 'H':
            g_config.hidden_size = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid hidden size:%s\n", optarg);
                return -EINVAL;
            }
            break;
        case 's':
            g_config.element_size = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid element size:%s\n", optarg);
                return -EINVAL;
            }
            break;
        case 'd':
            g_config.device_id = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid device id:%s\n", optarg);
                return -EINVAL;
            }
            break;
        case 'l':
            g_config.num_layer = strtoul(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "invalid num layer:%s\n", optarg);
                return -EINVAL;
            }
            break;
        default:
            return -EINVAL;
        }
    }

    if ((g_config.chunk_size % g_config.block_size) != 0) {
        fprintf(stderr, "chunk size needs multiple of block size\n");
        return -EINVAL;
    }

    return 0;
}

#ifdef _MULTITHREAD
static void copy_worker(int layerId, uint64_t *ptrs, uint64_t *_kptr, uint64_t *_vptr, void *chost, bool direction)
{
    uint32_t cpos, ppos, copy_len;
    int slot;
    uint32_t chunk_size = g_config.chunk_size;
    uint32_t hidden_size = g_config.hidden_size;
    uint32_t num_layers = g_config.num_layer;
    uint8_t element_size = g_config.element_size;
    copy_len = element_size * hidden_size;
    for (int tokenId = 0; tokenId < chunk_size; tokenId++) {
        cpos = layerId * chunk_size * hidden_size * element_size
            + tokenId * hidden_size * element_size;
        slot = ptrs[tokenId];
        ppos = slot * copy_len;
        if (direction) {
            CHECK_ACL(aclrtMemcpy((char *)chost + cpos, copy_len, (char *)(_kptr[layerId]) + ppos, copy_len, ACL_MEMCPY_DEVICE_TO_HOST));
            cpos += num_layers * chunk_size * hidden_size * element_size;
            CHECK_ACL(aclrtMemcpy((char *)chost + cpos, copy_len, (char *)(_vptr[layerId]) + ppos, copy_len, ACL_MEMCPY_DEVICE_TO_HOST));
        } else {
            CHECK_ACL(aclrtMemcpy((char *)(_kptr[layerId]) + ppos, copy_len, (char *)chost + cpos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
            cpos += num_layers * chunk_size * hidden_size * element_size;
            CHECK_ACL(aclrtMemcpy((char *)(_vptr[layerId]) + ppos, copy_len, (char *)chost + cpos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
}
#endif

int main(int argc, char *argv[])
{
#ifdef __KERNEL_FUNC__
    printf("========= TEST WITH DATACOPY =========\n");
#else
    printf("========= TEST WITH aclrtMemcpy ==========\n");
#endif
    init_configure();
    int ret = parse_options(argc, argv);
    if (ret) {
        fprintf(stderr, "Try '%s --help' for more information. \n", argv[0]);
        return ret;
    }
    uint32_t num_blocks = g_config.num_blocks;
    uint32_t block_size = g_config.block_size;
    uint32_t chunk_size = g_config.chunk_size;
    uint32_t num_layers = g_config.num_layer;
    uint32_t hidden_size = g_config.hidden_size;
    uint8_t element_size = g_config.element_size;
    uint32_t deviceId = g_config.device_id;
    uint32_t page_buffer_size = num_blocks * block_size;
    uint32_t k_buffer_size = num_blocks * block_size * hidden_size * element_size;
    uint32_t c_block_size = 2 * num_layers * hidden_size * element_size * chunk_size;
    int ptrs_size = num_layers * sizeof(uint64_t);
    // allocate transfer block
    int bnum = chunk_size / block_size * g_config.num_blocks;
    int _size = bnum * sizeof(int);
    int *block_ids = (int *)malloc(_size);
    ret = fill_unique_rand(0, num_blocks, bnum, block_ids);
    if (ret) {
        fprintf(stderr, "generate block ids failed\n");
        return ret;
    }

    CHECK_ACL(aclInit(NULL));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *sm, *smdev, *chost, *cdev, *kptrs, *kdevptrs, *vptrs, *vdevptrs, *tmp;
    uint64_t *gsm;

    // allocate the global slot mapping
    gsm = create_slot_mapping(num_blocks, block_size);
    if (!gsm) {
        fprintf(stderr, "global sm alloc failed\n");
        return -1;
    }

    // allocate cache object
    CHECK_ACL(aclrtMallocHost(&chost, c_block_size));
    CHECK_ACL(aclrtHostRegister(chost, c_block_size, ACL_HOST_REGISTER_MAPPED, &cdev));

    printf("PTRS gsm(m):%p,ksz:%u,kptrssz:%u,chost(h):%p,cdev:%p,cbs:%u\n",
        gsm, k_buffer_size, ptrs_size, chost, cdev, c_block_size);
    // alloc kptrs, vptrs
    CHECK_ACL(aclrtMallocHost(&kptrs, ptrs_size));
    CHECK_ACL(aclrtHostRegister(kptrs, ptrs_size, ACL_HOST_REGISTER_MAPPED, &kdevptrs));
    CHECK_ACL(aclrtMallocHost(&vptrs, ptrs_size));
    CHECK_ACL(aclrtHostRegister(vptrs, ptrs_size, ACL_HOST_REGISTER_MAPPED, &vdevptrs));

    // allocate the k, v tensor
    void *k, *v;
    for (int i = 0; i < num_layers; ++i) {
        CHECK_ACL(aclrtMalloc(&k, k_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&v, k_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST));
        *((uint64_t)kptrs + i) = (uint64_t)k;
        *((uint64_t)vptrs + i) = (uint64_t)v;
    }
    uint64_t *_kptr = (uint64_t *)kptrs;
    uint64_t *_vptr = (uint64_t *)vptrs;
#ifdef _DEBUG
    for (int i = 0; i < num_layers; i++) {
        printf("layer:%d,kdev(d):%p,vdev(d):%p\n", i, _kptr[i], _vptr[i]);
    }
#endif
    // initialize cache object value
    tmp = malloc(c_block_size);
    memset(tmp, 'C', c_block_size  - block_size);
    memset((char *)tmp + block_size, 'D', block_size);
    memcpy(chost, tmp, c_block_size);
    char out[10];

    // alloocate the slotmapping for blocks
    int sm_size = bnum * block_size * sizeof(uint64_t);
    CHECK_ACL(aclrtMallocHost(&sm, sm_size));
    CHECK_ACL(aclrtHostRegister(sm, sm_size, ACL_HOST_REGISTER_MAPPED, &smdev));
#ifdef _DEBUG
    printf("PTRS sm(h):%p,smdev:%p,ssz:%u\n", sm, smdev, sm_size);
#endif
    uint64_t *ptrs = (uint64_t *)sm;
    for (int i = 0; i < bnum; i++) {
        int bid = block_ids[i];
        memcpy(ptrs + i * block_size, gsm + bid * block_size, block_size * sizeof(uint64_t));
#ifdef _DEBUG
        printf("Generate slot mapping for bid %lu", bid);
        for (int j = 0; j < block_size; j++) {
            printf("%lu ", ptrs[i * block_size + j]);
        }
        printf("\n");
#endif
    }

#ifdef __KERNEL_FUNC__
    CHECK_ACL(aclrtMalloc(&kdevptrs, ptrs_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&vdevptrs, ptrs_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(kdevptrs, ptrs_size, kptrs, ptrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(vdevptrs, ptrs_size, vptrs, ptrs_size, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMalloc(&smdev, sm_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(smdev, sm_size, sm, sm_size, ACL_MEMCPY_HOST_TO_DEVICE));
#endif

    memcpy(out, chost, 10);
    printf("Before Transfer, READ cache content[first 10 bytes]:\t%s\n", out);
    struct timeval tv;
    uint64_t start, end;
    gettimeofday(&tv, NULL);
    start = (unsigned long)tv.tv_sec * 1000000UL + tv.tv_usec;
#ifdef __KERNEL_FUNC__
    const char *socName = aclrtGetSocName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aiv_num = MIN(num_layers, 4);
    int32_t numBuffsOnDev = 2;
    int64_t baseBuffSize = numBuffsOnDev * hidden_size * element_size;
    if ((int64_t)ubSize < baseBuffSize) {
        printf("(device:%u) ubSize is too small for single token, ubSize:%lu, bs:%ld\n", deviceId, ubSize, baseBuffSize);
        return -1;
    }
    int32_t maxTokensPerLoop = (ubSize / baseBuffSize) - 1;
    maxTokensPerLoop = static_cast<int32_t>(std::min(maxTokensPerLoop, static_cast<int32_t>(chunk_size)));
    int64_t totalPerLoopBuffer = static<int64_t>(maxTokensPerLoop) * baseBuffSize;
    if ((int64_t)ubSize < totalPerLoopBuffer) {
        printf("(device:%u) per Loop Buffer size:%ld exceed ubsize:%lu\n", deviceId, totalPerLoopBuffer,ubSize);
        return -1;
    }

    // using double bufs mean we actually want to allocate half of this per round.
    int64_t singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;
#ifdef _DEBUG
    printf("(device:%u) kptrs:0x%llx,vptrs:0x%llx,caches:0x%llx,slot:0x%llx,hdim:%d,psize:%ld,tokens:%d,loopBuffer:%d,perToken:%d,basebuffersize:%u,ubsize:%ld\n",
        deviceId, (void *)kdevptrs, (void *)vdevptrs, (void *)cdev, (void *)smdev, hidden_size, page_buffer_size, chunk_size, singlePerLoopBuffer, maxTokensPerLoop, baseBuffSize, ubSize);
#endif
    kvccache_ops::multi_layer_kv_transfer_kernel_v2(aiv_num, stream, (uint8_t *)kdevptrs, (uint8_t *)vdevptrs, (uint8_t *)cdev,
                            (uint8_t *)smdev, hidden_size, num_layers, page_buffer_size, chunk_size,
                            singlePerLoopBuffer, maxTokensPerLoop, false);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    printf("=====>CopyIn Done!\n");
    memcpy(out, chost, 10);
    printf("READ cache content[first 10 bytes]:\t%s\n", out);
    CHECK_ACL(aclrtMemset(chost, c_block_size, 'W', c_block_size));
    memcpy(out, chost, 10);
    printf("UPDATE cache content[first 10 bytes]:\t%s\n", out);

    // copy OUT
    kvccache_ops::multi_layer_kv_transfer_kernel_v2(aiv_num, stream, (uint8_t *)kdevptrs, (uint8_t *)vdevptrs, (uint8_t *)cdev,
                            (uint8_t *)smdev, hidden_size, num_layers, page_buffer_size, chunk_size,
                            singlePerLoopBuffer, maxTokensPerLoop, true);
    CHECK_ACL(aclrtSynchronizeStream(stream));
#else
    uint32_t cpos, ppos, copy_len;
#ifdef _DEBUG
    uint32_t total_len = 0;
#endif
    int slot;
    copy_len = element_size * hidden_size;
    memcpy(out, chost, 10);
    printf("REAQD cache content[first 10 bytes]:\t%s\n", out);
    // copy IN
#ifdef _MULTITHREAD
    std::vector<std::thread> thds_in;
    thds_in.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        thds_in.emplace_back(copy_worker, i, ptrs, _kptr, _vptr, chost, false);
    }
    for (auto& t: thds_in) t.join();
#else
    for (int layerId = 0; layerId < num_layers; layerId++) {
        for (int tokenId = 0; tokenId < chunk_size; tokenId++) {
            // copy key
            cpos = layerId * chunk_size * hidden_size * element_size
                + tokenId * hidden_size * element_size;
            slot = ptrs[tokenId];
            ppos = slot * copy_len;
            CHECK_ACL(aclrtMemcpy((char *)_kptr[layerId] + ppos, copy_len, (char *)chost + cpos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
#ifdef _DEBUG
            printf("COPY RANGE K host[%u-%u]-->dev[%u-%u]\n", cpos, cpos + copy_len, ppos, ppos + copy_len);
            total_len += copy_len;
#endif
            // copy value
            cpos += num_layers * chunk_size * hidden_size * element_size;
            CHECK_ACL(aclrtMemcpy((char *)_vptr[layerId] + ppos, copy_len, (char *)chost + cpos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
#ifdef _DEBUG
            printf("COPY RANGE V host[%u-%u]-->dev[%u-%u]\n", cpos, cpos + copy_len, ppos, ppos + copy_len);
            total_len += copy_len;
#endif
        }
    }
#endif
#ifdef _DEBUG
    printf("Copy in total length:%u\n", total_len);
    total_len = 0;
#endif
    printf("=======>CopyIn Done!\n");
    memcpy(out, chost, 10);
    printf("READ cache content[first 10 bytes]:\t%s\n", out);
    CHECK_ACL(aclrtMemset(chost, c_block_size, 'W', c_block_size));
    memcpy(out, chost, 10);
    printf("UPDATE cache content[first 10 bytes]:\t%s\n", out);

    // copy OUT
#ifdef _MULTITHREAD
    std::vector<std::thread> thds_out;
    thds_out.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        thds_out.emplace_back(copy_worker, i, ptrs, _kptr, _vptr, chost, true);
    }
    for (auto& t: thds_out) t.join();
#else
    for (int layerId = 0; layerId < num_layers; layerId++) {
        for (int tokenId = 0; tokenId < chunk_size; tokenId++) {
            // copy key
            cpos = layerId * chunk_size * hidden_size * element_size
                + tokenId * hidden_size * element_size;
            slot = ptrs[tokenId];
            ppos = slot * copy_len;
            CHECK_ACL(aclrtMemcpy((char *)chost + cpos, copy_len, (char *)_kptr[layerId] + ppos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
#ifdef _DEBUG
            printf("COPY RANGE K dev[%u-%u]-->host[%u-%u]\n", ppos, ppos + copy_len, cpos, cpos + copy_len);
            total_len += copy_len;
#endif
            // copy value
            cpos += num_layers * chunk_size * hidden_size * element_size;
            CHECK_ACL(aclrtMemcpy((char *)chost + cpos, copy_len, (char *)_vptr[layerId] + ppos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE));
#ifdef _DEBUG
            printf("COPY RANGE V dev[%u-%u]-->host[%u-%u]\n", ppos, ppos + copy_len, cpos, cpos + copy_len);
            total_len += copy_len;
#endif
        }
    }
#endif
#ifdef _DEBUG
    printf("Copy out total length:%u\n", total_len);
#endif
#endif
    gettimeofday(&tv, NULL);
    end = (unsigned long)tv.tv_sec * 1000000UL + tv.tv_sec;

    printf("=======>CopyOut Done!\n");
    memcpy(out, chost, 10);
    printf("READ cache content[first 10 bytes]:\t%s\n", out);
    memcpy(out, tmp, 10);
    printf("READ original cache content[first 10 bytes]:\t%s\n", out);
    // compare
    if (memcmp(chost, tmp, c_block_size) == 0) {
        printf("[PASS] After Transfer, is the same!!!\n");
    } else {
        printf("[FAIL] After Transfer, not the same!!!\n");
    }

    printf("---------------- Total cost: %u (us) ---------------\n", end - start);

    CHECK_ACL(aclrtHostUnregister(chost));
    CHECK_ACL(aclrtFreeHost(chost));
    for (int i = 0; i < num_layers; i++) {
        CHECK_ACL(aclrtFree((void *)_kptr[i]));
        CHECK_ACL(aclrtFree((void *)_vptr[i]));
    }
    CHECK_ACL(aclrtHostUnregister(kptrs));
    CHECK_ACL(aclrtFreeHost(kptrs));
    CHECK_ACL(aclrtHostUnregister(vptrs));
    CHECK_ACL(aclrtFreeHost(vptrs));
    CHECK_ACL(aclrtHostUnregister(sm));
    CHECK_ACL(aclrtFreeHost(sm));
#ifdef __KERNEL_FUNC__
    CHECK_ACL(aclrtFree(kdevptrs));
    CHECK_ACL(aclrtFree(vdevptrs));
    CHECK_ACL(aclrtFree(smdev));
#endif
    free(gsm);
    free(tmp);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    aclFinalize();
    return 0;
}