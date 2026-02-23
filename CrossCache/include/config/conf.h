#ifndef __CONF_H__
#define __CONF_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BASIC_PORT "Basic/Port"
#define BASIC_CHUNKSZ "Basic/ChunkSize"
#define BASIC_CACHEDIR "Basic/CacheDir"

#define REQ_LWORKER "Request/LoadWorker"
#define REQ_SWORKER "Request/StoreWorker"
#define REQ_CWORKER "Request/CommWorker"

#define CACHE_IOWORKER "Cache/IOWorker"
#define CACHE_CYWORKER "Cache/CopyWorker"
#define CACHE_LKUPSCALE "Cache/LookupMapScale"

int config_init(const char *file);
void config_exit();

uint32_t config_get_u32(const char *key);
uint64_t config_get_u64(const char *key);
const char *config_get_string(const char *key);

#ifdef __cplusplus
}
#endif

#endif