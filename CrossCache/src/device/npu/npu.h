#ifndef __NPU_H__
#define __NPU_H__

#include <cstdint>

namespace kvcache_ops {
    void multi_layer_kv_transfer_kernel_v2(uint32_t blockDim, void *stream,
                            uint8_t *pagedK, uint8_t *pagedV, uint8_t *dstCacheTensor, uint8_t *slotmappings,
                            const int64_t hiddenDims, const int32_t numLayers,
                            const int64_t pageBuffSize, const int32_t numTokensChunk,
                            const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
                            const bool page2L);
}

#endif