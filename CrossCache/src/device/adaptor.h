#ifndef __ADAPTOR_H__
#define __ADAPTOR_H__

#include "device.h"
#include <cstdint>

class Adaptor
{
public:
    virtual ~Adaptor() {};

    virtual void*
        AllocPinnedPtr(uint8_t nid, uint64_t size, unsigned int flags, void **daddr) = 0;

    virtual void
        FreePinnedPtr(void *addr, uint64_t size) = 0;

    virtual void*
        WrapperPtr(uint8_t nid, void *addr, uint64_t size) = 0;

    virtual void
        UnWrapperPtr(void *daddr) = 0;

    virtual void*
        OpenIPCKeys(uint8_t nid, uint16_t per_len, uint16_t num, void *data) = 0;

    virtual void
        CloseIPCKeys(void *daddr) = 0;

    virtual int
        TransferKVCache(struct transfer_params *params) = 0;

    virtual void
        Reset(uint8_t nid) = 0;
};

#endif