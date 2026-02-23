#ifndef __DEVICE_H__
#define __DEVICE_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEV_PTR

struct transfer_params {
    void *caches, *caches_h;        /* buffer in memory side (Pinned Memory) */

    void DEV_PTR *key_ptrs;
    void DEV_PTR *value_ptrs;
    uint64_t page_buffer_size;  /* buffer size in token (num_pages * page_size) in vllm */
    void DEV_PTR *slot_mapping;

    uint8_t devid;
    uint8_t caches_element_size;    /* key/value element size */

    uint16_t num_layers;    /* num layers */
    uint16_t num_tokens;    /* num tokens in this caches */
    uint32_t hidden_dim_size;   /* hidden dim size */
    bool direction;
};

typedef struct Adaptor adaptor_t;

#ifdef _NPU
adaptor_t *npu_adaptor_create(void);
void npu_adaptor_destroy(adaptor_t *adaptor);
#elif _NPUSDK
adaptor_t *npu_sdk_adaptor_create(void);
void npu_sdk_adaptor_destroy(adaptor_t *adaptor);
#else
adaptor_t *mem_adaptor_create(void);
void mem_adaptor_destroy(adaptor_t *adaptor);
#endif

void *alloc_pinned_ptr(adaptor_t *adaptor, uint8_t nid, uint64_t size, unsigned int flags, void **daddr);
void free_pinned_ptr(adaptor_t *adaptor, void *addr, uint64_t size);
void *wrapper_ptr(adaptor_t *adaptor, uint8_t nid, void *addr, uint64_t size);
void unwrapper_ptr(adaptor_t *adaptor, void *daddr);
void *open_ipc_keys(adaptor_t *adaptor, uint8_t nid, uint16_t per_len, uint16_t num, void *data);
void close_ipc_keys(adaptor_t *adaptor, void *daddr);
int transfer_kvcache(adaptor_t *adaptor, struct transfer_params *params);
void reset_device(adaptor_t *adaptor, uint8_t nid);
#ifdef __cplusplus
}
#endif

#endif