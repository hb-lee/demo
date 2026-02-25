#include "adaptor.h"
#include "device.h"

#include "log.h"
#include "sysdef.h"

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <new>

class MemAdaptor : public Adaptor
{
public:
    MemAdaptor()
    {
        log_info("MemAdaptor init");
    }

    ~MemAdaptor()
    {
        log_info("MemAdaptor exit");
    }

    void *AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr);
    void FreePinnedPtr(void *addr, uint64_t size);
    void *WrapperPtr(uint8_t nid, void *addr, uint64_t size);
    void UnWrapperPtr(void *daddr);
    void *OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data);
    void CloseIPCKeys(void *daddr);
    int TransferKVCache(struct transfer_params *params);
    void Reset(uint8_t nid) {}
};

void* MemAdaptor::AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr)
{
    void *addr;

    if (size % 4096) {
        log_error("nid(%u) alloc addr size should be 4k-aligned, now is:%lu", size);
        return NULL;
    }
    addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        log_error("nid(%u) alloc addr failed, size:%lu, errno:%d", nid, size, errno);
        return NULL;
    }
    *daddr = addr;
    log_info("nid:%u, alloc memory mem at:%p, size:%lu", nid, addr, size);
    return addr;
}

void MemAdaptor::FreePinnedPtr(void *addr, uint64_t size)
{
    munmap(addr, size);
}

void* MemAdaptor::WrapperPtr(uint8_t nid, void *addr, uint64_t size)
{
    void *daddr;

    daddr = malloc(size);
    if (!daddr)
        return NULL;
    memcpy(daddr, addr, size);
    log_debug("original addr:%p, wrapper addr:%p", addr, daddr);
    return daddr;
}

void MemAdaptor::UnWrapperPtr(void *daddr)
{
    free(daddr);
}

void* MemAdaptor::OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data)
{
    char shm[10];
    int i, fd;
    void *va;
    uint64_t *ptrs, *ori_ptrs;

    if (num > 10) {
        log_error("Fake Mode does not support more than 10 layers");
        return NULL;
    }
    ptrs = (uint64_t *)malloc(num * per_len);
    if (!ptrs)
        return NULL;
    ori_ptrs = (uint64_t *)data;
    for (i = 0; i < num; i++) {
        sprintf(shm, "/fakekv:%d", i);
        fd = shm_open(shm, O_CREAT | O_RDWR, 0666);
        sys_assert(fd > 0);
        ftruncate(fd, 10 * 1024 * 1024); // 10M
        va = mmap(NULL, 10 * 1024 * 1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        log_debug("nid:%u, layer:%d,ori kptr:0x%llx,fake kvptr:%p,fake shmfile:%s,limit:10MB",
            nid, i, ori_ptrs[i], va, shm);
        ptrs[i] = (uint64_t)va;
    }
    log_debug("IPCKeys array addr:0x%llx", ptrs);
    return ptrs;
}

void MemAdaptor::CloseIPCKeys(void *daddr)
{
    free(daddr);
}

int MemAdaptor::TransferKVCache(struct transfer_params *params)
{
    int layerIdx, tokenIdx;
    int64_t slot, *slotmaping;
    uint32_t cpos, copy_len;
    uint64_t ppos, *key_ptrs, *val_ptrs;
    uint8_t *caches;
    std::string out_str;

    caches = (uint8_t *)params->caches;
    key_ptrs = (uint64_t *)params->keys_ptrs;
    val_ptrs = (uint64_t *)params->value_ptrs;
    slotmaping = (int64_t *)params->slot_mapping;

    /* move length for each epoch */
    copy_len = params->caches_element_size *params->hidden_dim_size;
    if (params->direction)
        out_str = std::string("========== TO cache (unit: " + std::to_string(copy_len) +
        ") ==========\n");
    else
        out_str = std::string("========== FROM cache (unit: " + std::to_string(copy_len) +
        ") ==========\n");
    for (layerIdx = 0; layerIdx < params->num_layers; layerIdx++) {
        out_str += std::string("Layer:") + std::to_string(layerIdx) +
                std::string(" [ ");
        for (tokenIdx = 0; tokenIdx < params->num_tokens; tokenIdx++) {
            /* copy key cache */
            cpos = layerIdx * params->num_tokens * copy_len
                + tokenIdx * copy_len;

            slot = slotmaping[tokenIdx];
            ppos = slot * copy_len;
            if (params->direction) {
                memcpy(caches + cpos, (uint8_t *)key_ptrs[layerIdx] + ppos, copy_len);
                out_str += std::string("K:") + std::to_string(ppos) +
                    std::string("->") + std::to_string(cpos) + std::string(" ");
            } else {
                memcpy((uint8_t *)key_ptrs[layerIdx] + ppos, caches + cpos, copy_len);
                out_str += std::string("K:") + std::to_string(cpos) +
                    std::string("->") + std::to_string(ppos) + std::string(" ");
            }
            /* copy value cache if needed */
            if (val_ptrs) {
                cpos += params->num_layers * params->num_tokens * copy_len;
                if (params->direction) {
                    memcpy(caches + cpos, (uint8_t *)val_ptrs[layerIdx] + ppos, copy_len);
                    out_str += std::string("V:") + std::to_string(ppos) +
                        std::string("->") + std::to_string(cpos) + std::string(" ");
                } else {
                    memcpy((uint8_t *)val_ptrs[layerIdx] + ppos, caches + cpos, copy_len);
                    out_str += std::string("V:") + std::to_string(cpos) +
                        std::string("->") + std::to_string(ppos) + std::string(" ");
                }
            }
        }
        out_str += std::string("]\n");
    }
    printf("%s\n", out_str.c_str());
    return 0;
}

adaptor_t *mem_adaptor_create()
{
    try
    {
        return new MemAdaptor();
    }
    catch (std::bad_alloc &)
    {
        log_error("mem adaptor create failed");
        return nullptr;
    }
}

void mem_adaptor_destroy(adaptor_t *adaptor)
{
    delete adaptor;
}