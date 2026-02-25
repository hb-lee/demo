#include "npu.h"
#include "adaptor.h"
#include "device.h"

#include "log.h"

#include <errno.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <tiling/platform/platform_ascendc.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <new>

class NPUAdaptor : public Adaptor
{
public:
    NPUAdaptor()
    {
        log_debug("NPUAdaptor init");
        if (setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7", 1) != 0) {
            log_error("set ASCEND_RT_VISIBLE_DEVICES failed");
            throw std::bad_alloc();
        }
        aclError err = aclInit(NULL);
        if (err != ACL_SUCCESS)
            throw std::bad_alloc();
    }

    ~NPUAdaptor()
    {
        log_debug("NPUAdaptor exit");
        aclFinalize();
    }

    void *AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr);
    void FreePinnedPtr(void *addr, uint64_t size);
    void *WrapperPtr(uint8_t nid, void *addr, uint64_t size);
    void UnWrapperPtr(void *daddr);
    void *OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data);
    void CloseIPCKeys(void *daddr);
    int TransferKVCache(struct transfer_params *params);
    void Reset(uint8_t nid);
};

void* NPUAdaptor::AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr)
{
    void *addr, *devptr;
    char *buf;
    aclError err;

    if (size % 4096) {
        log_error("device(%u) alloc pin addr size should be 4k-aligned, now is:%lu", size);
        return NULL;
    }
    err = aclrtSetDevice(nid);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt set device failed, err:%d", nid, err);
        return NULL;
    }
    /* Allocate the locked memory */
    addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        log_error("device(%u) alloc pin addr failed, size:%lu, errno:%d", nid, size, errno);
        return NULL;
    }
    buf = (char *)addr;
    for (uint64_t i = 0; i < size; i += 4096) {
        buf[i] = '0';
    }
    if (mlock(addr, size) != 0) {
        log_error("device(%u) lock memory failed, size:%lu, errno:%d", nid, size, errno);
        munmap(addr, size);
        return NULL;
    }
    err = aclrtHostRegister(addr, size, ACL_HOST_REGISTER_MAPPED, &devptr);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt register memory failed, err:%d", nid, err);
        munlock(addr, size);
        munmap(addr, size);
        return NULL;
    }
    if (daddr != NULL)
        *daddr = devptr;
    log_debug("device(%u) alloc pinned ptr size:%llu done(0x%llx,0x%llx).", nid, size, addr, devptr);
    return addr;
}

void NPUAdaptor::FreePinnedPtr(void *addr, uint64_t size)
{
    log_debug("free pinned ptr :0x%llx", addr);
    aclrtHostUnregister(addr);
    munlock(addr, size);
    munmap(addr, size);
}

void* NPUAdaptor::WrapperPtr(uint8_t nid, void *addr, uint64_t size)
{
    void *daddr;
    aclError err;

    err = aclrtSetDevice(nid);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt set device failed, err:%d", nid, err);
        return NULL;
    }
    err = aclrtMalloc(&daddr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt alloc(%lu) failed, err:%d", nid, size, err);
        return NULL;
    }
    err = aclrtMemcpy(daddr, size, addr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt copy(%lu) failed, err:%d", nid, size, err);
        aclrtFree(daddr);
        daddr = NULL;
    }
    log_debug("device(%u) Wrapper addr:%p into %p", nid, addr, daddr);
    return daddr;
}

void NPUAdaptor::UnWrapperPtr(void *daddr)
{
    log_debug("UnWrapper addr:%p", daddr);
    aclrtFree(daddr);
}

void* NPUAdaptor::OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data)
{
    void *addr, *daddr;
    aclError err;
    int i, pos, size = num * sizeof(uint64_t);

    addr = malloc(size);
    if (!addr)
        return NULL;
    err = aclrtSetDevice(nid);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt set device failed, err:%d", nid, err);
        free(addr);
        return NULL;
    }
    /* import the key to dev addr */
    for (i = 0, pos = 0; i < num; i++) {
        void *devptr;
        char buffer[256] = {0};

        memcpy(buffer, (char *)data + pos, per_len);
        pos += per_len;
        err = aclrtIpcMemImportByKey(&devptr, buffer, 0);
        if (err != ACL_SUCCESS) {
            log_error("device(%u) import key:%d failed for key:%s, err:%d", nid, i, buffer, err);
            free(addr);
            return NULL;
        }
        log_debug("device(%u) aclrt import key:[%d] %s to addr:0x%llx", nid, i, buffer, (uint64_t)devptr);
        *((uint64_t *)addr + i) = (uint64_t)devptr;
    }
    /* move dev addr array to device memory */
    err = aclrtMalloc(&daddr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt malloc failed, size:%lu, err:%d", nid, size, err);
        free(addr);
        return NULL;
    }
    log_debug("device(%u) aclrt alloc size:%d done(0x%llx).", nid, size, daddr);
    err = aclrtMemcpy(daddr, size, addr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt copy(%lu) failed, err:%d", nid, size, err);
        aclrtFree(daddr);
        free(addr);
        return NULL;
    }
    free(addr);
    log_debug("device(%u) OpenIPCKeys at:%p", nid, daddr);
    return daddr;
}

void NPUAdaptor::CloseIPCKeys(void *daddr)
{
    log_debug("close ipckey :0x%llx", daddr);
    aclrtFree(daddr);
}

int NPUAdaptor::TransferKVCache(struct transfer_params *params)
{
    int num_layers = params->num_layers;
    int num_tokens = params->num_tokens;
    int hidden_dims = params->hidden_dim_size;
    uint8_t *caches = (uint8_t *)params->caches;
    uint8_t *key_ptrs = (uint8_t *)params->keys_ptrs;
    uint8_t *value_ptrs = (uint8_t *)params->value_ptrs;
    uint8_t *slot_mapping_ptr = (uint8_t *)params->slot_mapping;
    int64_t page_buffer_size = params->page_buffer_size;
    bool direction = params->direction;

    aclError err = aclrtSetDevice(params->devid);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt set device failed, err:%d", params->devid, err);
        return -1;
    }
    aclrtStream stream;
    err = aclrtCreateStream(&stream);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt create stream failed, err:%d", params->devid, err);
        return -1;
    }

    const char *socName = aclrtGetSocName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    /* set default AI core number */
    uint32_t aiv_num = (uint32_t)std::min(num_layers, 4);

    int32_t numBuffsOnDev = 2;
    int64_t baseBuffSize = numBuffsOnDev * hidden_dims * params->caches_element_size;
    if ((int64_t)ubSize < baseBuffSize) {
        log_error("(device:%u) ubSize is too small for single token, ubSize:%lu, bs:%ld", params->devid, ubSize, baseBuffSize);
        aclrtDestroyStream(stream);
        return -1;
    }
    int32_t maxTokensPerLoop = (ubSize / baseBuffSize) - 1;
    maxTokensPerLoop = static_cast<int32_t>(std::min(maxTokensPerLoop, static_cast<int32_t>(num_tokens)));
    int64_t totalPerLoopBuffer = static_cast<int64_t>(maxTokensPerLoop) * baseBuffSize;
    if ((int64_t)ubSize < totalPerLoopBuffer) {
        log_error("(device:%u) per Loop buffer size:%ld exceed ubsize:%lu", params->devid, totalPerLoopBuffer,ubSize);
        aclrtDestroyStream(stream);
        return -1;
    }

    // using double buffs mean we actually want to allocate half of this per round.
    int64_t singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;
    log_debug("(device:%u) ops param:kptrs:0x%llx,slotmapping:0x%llx,caches:0x%llx,hdim:%d,psize:%ld,tokens:%d,loopBuffer:%d,perToken:%d,basebuffersize:%u,ubsize:%ld",
        params->devid, (void *)key_ptrs, (void *)slot_mapping_ptr, (void *)caches, hidden_dims, page_buffer_size, num_tokens, singlePerLoopBuffer, maxTokensPerLoop, baseBufferSize, ubSize);
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(aiv_num, stream, key_ptrs, value_ptrs, caches,
                            slot_mapping_ptr, hidden_dims, num_layers, page_buffer_size, num_tokens,
                            singlePerLoopBuffer, maxTokensPerLoop, direction);
    err = aclrtSynchronizeStream(stream);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt sync stream failed, err:%d", params->devid, err);
        aclrtDestroyStream(stream);
        return -1;
    }
    log_debug("(device:%u) transfer done, direction:%s", params->devid, direction ? "D->H" : "H->D");
    aclrtDestroyStream(stream);
    return 0;
}

void NPUAdaptor::Reset(uint8_t nid)
{
    log_debug("reset npu:%u", nid);
    aclrtResetDeviceForce(nid);
}

adaptor_t *npu_adaptor_create()
{
    try
    {
        return new NPUAdaptor();
    }
    catch (std::bad_alloc &)
    {
        log_error("npu adaptor create failed");
        return nullptr;
    }
}

void npu_adaptor_destroy(adaptor_t *adaptor)
{
    delete adaptor;
}