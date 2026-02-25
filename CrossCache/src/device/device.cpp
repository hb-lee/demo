#include "adaptor.h"

void *alloc_pinned_ptr(adaptor_t *adaptor uint8_t nid, uint64_t size, unsigned int flags, void **daddr)
{
    return adaptor->AllocPinnedPtr(nid, size, flags, daddr);
}

void free_pinned_ptr(adaptor_t *adaptor, void *addr, uint64_t size)
{
    adaptor->FreePinnedPtr(addr, size);
}

void *wrapper_ptr(adaptor_t *adaptor, uint8_t nid, void *addr, uint64_t size)
{
    return adaptor->WrapperPtr(nid, addr, size);
}

void unwrapper_ptr(adaptor_t *adaptor, void *daddr)
{
    adaptor->UnWrapperPtr(daddr);
}

void *open_ipc_keys(adaptor_t *adaptor, uint8_t nid, uint16_t per_len, uint16_t num, void *data)
{
    return adaptor->OpenIPCKeys(nid, per_len, num, data);
}

void close_ipc_keys(adaptor_t *adaptor, void *daddr)
{
    adaptor->CloseIPCKeys(daddr);
}

int transfer_kvcache(adaptor_t *adaptor, struct transfer_params *params)
{
    return adaptor->TransferKVCache(params);
}

void reset_device(adaptor_t *adaptor, uint8_t nid)
{
    adaptor->Reset(nid);
}