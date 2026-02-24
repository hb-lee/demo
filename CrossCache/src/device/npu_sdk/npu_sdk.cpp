#include "adaptor.h"
#include "device.h"

#include "log.h"
#include <string.h>

#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <new>

class NPUSDKAdaptor : public Adaptor
{
public:
    NPUSDKAdaptor()
    {
        log_debug("NPUSDKAdaptor init");
        if (setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7", 1) != 0) {
            log_error("set ASCEND_RT_VISIBLE_DEVICES failed");
            throw std::bad_alloc();
        }
        aclError err = aclInit(NULL);
        if (err != ACL_SUCCESS)
            throw std::bad_alloc();
    }

    ~NPUSDKAdaptor()
    {
        log_debug("NPUSDKAdaptor exit");
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

void* NPUSDKAdaptor::AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr)
{
    void *addr, *devptr;
    char *buf;
    aclError err;

    if (size % 4096) {
        log_error("nid(%u) alloc addr size should be 4k-aligned, now is:%lu", size);
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
        log_error("device(%u) alloc pin addr failed, size:%u, errno:%d", nid, size, errno);
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

void NPUSDKAdaptor::FreePinnedPtr(void *addr, uint64_t size)
{
    log_debug("free pinned ptr :0x%llx", addr);
    aclrtHostUnregister(addr);
    munlock(addr, size);
    munmap(addr, size);
}

void* NPUSDKAdaptor::WrapperPtr(uint8_t nid, void *addr, uint64_t size)
{
    void *daddr;

    daddr = malloc(size);
    if (!daddr)
        return NULL;
    memcpy(daddr, addr, size);
    return daddr;
}

void NPUSDKAdaptor::UnWrapperPtr(void *daddr)
{
    free(daddr);
}

void* NPUSDKAdaptor::OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data)
{
    void *addr;
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
    return addr;
}

void NPUSDKAdaptor::CloseIPCKeys(void *daddr)
{
    log_debug("close ptr :0x%llx", daddr);
    free(daddr);
}

int NPUSDKAdaptor::TransferKVCache(struct transfer_params *params)
{
    int layerIdx, tokenIdx;
    int64_t slot, *slotmapping;
    uint32_t cpos, copy_len;
    uint64_t ppos, *key_ptrs, *val_ptrs;
    uint8_t *caches;
    std::string out_str;
    aclError err;

    err = aclrtSetDevice(params->devid);
    if (err != ACL_SUCCESS) {
        log_error("device(%u) aclrt set device failed, err:%d", params->devid, err);
        return -1;
    }

    caches = (uint8_t *)params->caches;
    key_ptrs = (uint64_t *)params->key_ptrs;
    val_ptrs = (uint64_t *)params->value_ptrs;
    slotmapping = (int64_t *)params->slot_mapping;

    /* move length for each epoch */
    copy_len = params->caches_element_size *params->hidden_dim_size;
    if (params->direction)
        out_str = std::string("========== TO cache (uint: " + std::to_string(copy_len) +
        ") ==========\n");
    else
        out_str = std::string("========== FROM cache (uint: " + std::to_string(copy_len) +
        ") ==========\n");
    for (layerIdx = 0; layerIdx < params->num_layers, layerIdx++) {
        out_str += std::string("Layer:") + std::to_string(layerIdx) +
                std::string(" [ ");
        for (tokenIdx = 0; tokenIdx < params->num_tokens, tokenIdx++) {
            /* copy key cache */
            cpos = layerIdx * params->num_tokens * copy_len
                + tokenIdx * copy_len;

            slot = slotmapping[tokenIdx];
            ppos = slot * copy_len;
            if (params->direction) {
                err = aclrtMemcpy(caches + cpos, (uint8_t *)key_ptrs[layerIdx] + ppos, copy_len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (err != ACL_SUCCESS) {
                    log_error("device(%u) aclrt memcpy failed, err:%d", params->devid, err);
                    return -1;
                }
                out_str += std::string("K:") + std::to_string(ppos) + std::string("-") + std::to_string(ppos + copy_len) +
                    std::string("->") + std::string(cpos) + std::string("-") + std::to_string(cpos + copy_len) + std::string(" ");
            } else {
                err = aclrtMemcpy((uint8_t *)key_ptrs[layerIdx] + ppos, caches + cpos, copy_len, ACL_MEMCPY_HOST_TO_DEVICE);
                if (err != ACL_SUCCESS) {
                    log_error("device(%u) aclrt memcpy failed, err:%d", params->devid, err);
                    return -1;
                }
                out_str += std::string("K:") + std::to_string(cpos) + std::string("-") + std::to_string(cpos + copy_len) +
                    std::string("->") + std::string(ppos) + std::string("-") + std::to_string(ppos + copy_len) + std::string(" ");
            }
            /* copy value cache if needed */
            if (val_ptrs) {
                cpos += params->num_layers * params->num_tokens * copy_len;
                if (params->direction) {
                    err = aclrtMemcpy(caches + cpos, (uint8_t *)val_ptrs[layerIdx] + ppos, copy_len, ACL_MEMCPY_DEVICE_TO_HOST);
                    if (err != ACL_SUCCESS) {
                        log_error("device(%u) aclrt memcpy failed, err:%d", params->devid, err);
                        return -1;
                    }
                    out_str += std::string("V:") + std::to_string(ppos) + std::string("-") + std::to_string(ppos + copy_len) +
                        std::string("->") + std::string(cpos) + std::string("-") + std::string(cpos + copy_len) + std::string(" ");
                } else {
                    err = aclrtMemcpy((uint8_t *)val_ptrs[layerIdx] + ppos, caches + cpos, copy_len);
                    if (err != ACL_SUCCESS) {
                        log_error("device(%u) aclrt memcpy failed, err:%d", params->devid, err);
                        return -1;
                    }
                    out_str += std::string("V:") + std::to_string(cpos) + std::string("-") + std::string(cpos + copy_len) +
                        std::string("->") + std::string(ppos) + std::string("-") + std::string(ppos + copy_len) + std::string(" ");
                }
            }
        }
        out_str += std::string("]\n");
    }
    printf("%s\n", out_str.c_str());
    return 0;
}

void NPUSDKAdaptor::Reset(uint8_t nid)
{
    log_debug("reset npu:%u", nid);
    aclrtResetDeviceForce(nid);
}

adaptor_t *npu_sdk_adaptor_create()
{
    try
    {
        return new NPUSDKAdaptor();
    }
    catch (std::bad_alloc &)
    {
        log_error("npu sdk adaptor create failed");
        return nullptr;
    }
}

void npu_sdk_adaptor_destroy(adaptor_t *adaptor)
{
    delete adaptor;
}