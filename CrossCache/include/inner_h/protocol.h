#ifndef __PROTOCOL_H__
#define __PROTOCOL_H__

#define CACHE_VERSION 1

#define K_V_MASK    0x01
#define KV_DTYPE_1BYTE  0x02
#define KV_DTYPE_2BYTE  0x04
#define KV_DTYPE_4BYTE  0x08

enum cache_opcode {
    CACHE_REGISTER  = 0,
    CACHE_UNREGISTER,
    CACHE_LOOKUP,
    CACHE_STORE,
    CACHE_LOAD,
    CACHE_GET_INFO,
    CACHE_MAX,
};

#pragma pack(push, 1)

/* message header */
struct msg_header {
    uint8_t version;
    uint8_t op;
    uint8_t node_id;
    uint8_t reserved;
    uint32_t len;
    uint32_t uid;
    uint8_t data[];
};

/* message body: register */
struct msg_register {
    uint8_t world_size;
    uint8_t rank_id;
    uint8_t num_layers;
    uint8_t kv_flags;
    uint32_t num_blocks;
    uint32_t hidden_dim_size;
    uint32_t first_key_in_bytes;
    uint8_t data[]; /* ${name},k1,k2,k3,v1,v2,v3... */
};

/* message body: lookup, load, store */
struct msg_common {
    uint32_t keys_num;
    uint32_t key_len;   /* blockid len default is sizeof(uint64_t) */
    uint8_t data[]; /* key1|key2|..., blockid,... */
};

/* lookup extra body */
struct msg_lookup {
    uint8_t world_size;
    uint8_t data[]; /* ${name} */
};

/* message response */
struct msg_response {
    uint32_t uid;
    uint32_t ok;    /* for lookup, < -4096 means the success hit count */
};

#pragma pack(pop)

#endif