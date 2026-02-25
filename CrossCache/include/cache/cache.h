#ifndef __CACHE_H__
#define __CACHE_H__

#include "inner.h"

#include <stdint.h>

int cache_register(struct worker_context *wctx);

void cache_unregister(struct worker_context *wctx);

int cache_lookup(struct worker_context *wctx, uint8_t *keys,
                 uint32_t key_num, uint32_t key_len);

int cache_store(struct worker_context *wctx, uint8_t *keys,
                uint32_t key_len, uint64_t *bids, uint32_t bnum);

int cache_load(struct worker_context *wctx, uint8_t *keys,
               uint32_t key_len, uint64_t *bids, uint32_t bnum);

int cache_init(uint32_t chunk_size, uint64_t pool_size, const char *root_dir);

void cache_exit();

#endif
