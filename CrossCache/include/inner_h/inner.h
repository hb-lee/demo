#ifndef __INNER_H__
#define __INNER_H__

#include <stdint.h>

struct model_info {
    char *model_name;
    uint8_t element_size;
    uint8_t num_layers;
    uint32_t hidden_dim_size;
};

struct worker_context {
    struct model_info model;
    uint8_t node_id;
    uint8_t world_size;
    uint8_t rank_id;
    uint32_t per_token_size;
    uint32_t num_blocks;
    uint32_t block_size;

    // IPC key
    uint16_t per_len;   /* num layers means the keys num */

    void *k_raw_ptrs;   /* raw va recorded in host memory for client/server */
    void *v_raw_ptrs;
    // OUT
    void *k_ptrs;   /* va recorded in device memory in server side */
    void *v_ptrs;
};

#endif