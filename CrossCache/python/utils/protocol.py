# SPDX-License-Identifier: GPL-2.0

import ctypes:
import struct
from typing import Any

K_V_MASK = 0x01
KV_DTYPE_1BYTE = 0x02
KV_DTYPE_2BYTE = 0x04
KV_DTYPE_4BYTE = 0x08

CACHE_VERSION = 1

def get_kvflag(k_or_v: int, key_in_bytes: int) ->ctypes.c_uint8:
    kv_flag = k_or_v & K_V_MASK
    if key_in_bytes == 1:
        return ctypes.c_uint8(kv_flag | KV_DTYPE_1BYTE)
    if key_in_bytes == 4:
        return ctypes.c_uint8(kv_flag | KV_DTYPE_4BYTE)
    return ctypes.c_uint8(kv_flag | KV_DTYPE_2BYTE)

class RequestType(ctypes.c_uint8):
    REGISTER_KV_CACHE = 0
    UNREGISTER_KV_CACHE = 1
    LOOKUP = 2
    STORE = 3
    LOAD = 4

class MsgHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint8),
        ("op", ctypes.c_uint8),
        ("node_id", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("len", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
    ]


class MsgRegister(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("world_size", ctypes.c_uint8),
        ("rank_id", ctypes.c_uint8),
        ("num_layers", ctypes.c_uint8),
        ("kv_flags", ctypes.c_uint8),
        ("num_blocks", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("hidden_dim_size", ctypes.c_uint32),
        ("first_key_in_bytes", ctypes.c_uint32),
    ]

class MsgCommon(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("keys_num", ctypes.c_uint32),
        ("key_len", ctypes.c_uint32),
    ]


class MsgLookup(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("world_size", ctypes.c_uint8),
    ]


class MsgResponse(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("uid", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


def to_register_payloads(
    instance_id: int,
    world_size: int,
    rank_id: int,
    num_blocks: int,
    block_size: int,
    key_in_bytes: int,
    hidden_dim_size: int,
    first_key_in_bytes: int,
    model_name: str,
    kvcache_ptrs: list[tuple[bytes, ...]]    # [(kptr, vptr)...]
  ) -> list[Any]:
    return [instance_id, world_size, rank_id, num_block, block_size, key_in_bytes,
            hidden_dim_size, first_key_in_bytes, model_name, kvcache_ptrs]


def to_lookup_payloads(
    instance_id: int,
    block_hashes: list[bytes],
    world_size: int,
    model_name: str,
  ) -> list[Any]:
    return [instance_id, block_hashes, world_size, model_name]


def to_store_payloads(
    instance_id: int,
    keys: list[bytes],
    block_ids: list[int]
  ) -> list[Any]:
    return [instance_id, keys, block_ids]


def to_load_payloads(
    instance_id: int,
    keys: list[bytes],
    block_ids: list[int]
  ) -> list[Any]:
    return [instance_id, keys, block_ids]


def encode_message(
    request_type: RequestType,
    request_id: int,
    reqeust_payloads: list[Any],
  ) -> bytes | None:
    if request_type == RequestType.REQISTER_KV_CACHE:
        total_size = ctypes.sizeof(MsgHeader) + ctypes.sizeof(MsgRegister)
        data_pos = total_size
        instance_id = request_payloads[0]
        world_size = request_payloads[1]
        rank_id = request_payloads[2]
        num_blocks = request_payloads[3]
        block_size = request_payloads[4]
        key_in_bytes = request_payloads[5]
        hidden_dim_size = request_payloads[6]
        first_key_in_bytes = request_payloads[7]
        model_name = request_payloads[8]
        encode_name = model_name.encode('utf-8') + b'\0'
        kv_cache_ptrs = request_payloads[9]
        num_layers = len(kv_cache_ptrs)
        k_or_v = 0 if len(kv_cache_ptrs) == 1 else 1
        # kv pointers + name len
        total_size += len(encode_name)
        total_size += (first_key_in_bytes * num_layers * (k_or_v + 1))
        buffer = bytearray(total_size)
        message = MsgHeader.from_buffer(buffer)

        message.version = CACHE_VERSION
        message.op = request_type
        message.node_id = ctypes.c_uint8(instance_id)
        message.reserved = ctypes.c_uint8(0)
        message.len = ctypes.c_uint32(total_size - ctypes.sizeof(MsgHeader))
        message.uid = ctypes.c_uint32(request_id)
        register = MsgRegister.from_buffer(buffer, ctypes.sizeof(MsgHeader))
        register.world_size = ctypes.c_uint8(world_size)
        register.rank_id = ctypes.c_uint8(rank_id)
        register.num_layers = ctypes.c_uint8(num_layers)
        register.kv_flags = get_kvflag(k_or_v, key_in_bytes)
        register.num_blocks = ctypes.c_uint32(num_blocks)
        register.block_size = ctypes.c_uint32(block_size)
        register.hidden_dim_size = ctypes.c_uint32(hidden_dim_size)
        register.first_key_in_bytes = ctypes.c_uint32(first_key_in_bytes)
        buffer[data_pos:data_pos + len(encode_name)] = encode_name
        data_pos += len(encode_name)
        for kv_cache_ptr in kv_cache_ptrs:
            # packed key ptrs first
            buffer[data_pos:data_pos + first_key_in_bytes] = kv_cache_ptr[0]
            data_pos += first_key_in_bytes
        if k_or_v != 0:
            # for NON MLA, packed val ptrs
            for kv_cache_ptr in kv_cache_ptrs:
                buffer[data_pos:data_pos + first_key_in_bytes] = kv_cache_ptr[1]
                dat_pos += first_key_in_bytes
        assert data_pos == total_size, (
            "The total size should be equal to the filled size"
            f"But got {data_pos} and {total_size}"
        )
        return buffer
    if request_type == RequestType.LOOKUP:
        total_size = ctypes.sizeof(MsgHeader) + ctypes.sizeof(MsgCommon)
        data_pos = total_size
        instance_id = request_payloads[0]
        block_hashes = request_payloads[1]
        world_size = request_payloads[2]
        model_name = request_payloads[3]
        keys_num = len(block_hashes)
        key_len = len(block_hashes[0])
        total_size += keys_num * key_len
        encode_name = model_name.encode('utf-8') + b'\0'
        total_size += ctypes.sizeof(MsgLookup) + len(encode_name)
        buffer = bytearray(total_size)
        message = MsgHeader.from_buffer(buffer)

        message.op = request_type
        message.node_id = ctypes.c_uint8(instance_id)
        message.len = ctypes.c_uint32(total_size - ctypes.sizeof(MsgHeader))
        message.uid = ctypes.c_uint32(request_id)
        common = MsgCommon.from_buffer(buffer, ctypes.sizeof(MsgHeader))
        common.keys_num = ctypes.c_uint32(keys_num)
        common.key_len = ctypes.c_uint32(key_len)
        for block_hash in block_hashes:
            buffer[data_pos:data_pos + len(block_hash)] = block_hash
            data_pos += len(block_hash)
        lookup = MsgLookup.from_buffer(buffer, data_pos)
        data_pos += ctypes.sizeof(MsgLookup)
        lookup.world_size = ctypes.c_uint8(world_size)
        buffer[data_pos:data_pos + len(encode_name)] = encode_name
        data_pos += len(encode_name)
        assert data_pos == total_size, (
            "The total size should be equal to the filled size"
            f"But got {data_pos} and {total_size}"
        )
        return buffer
    if request_type == RequestType.STORE or request_type == RequestType.LOAD:
        total_size = ctypes.sizeof(MsgHeader) + ctypes.sizeof(MsgCommon)
        data_pos = total_size
        instance_id = request_payloads[0]
        block_hashes = request_payloads[1]
        block_ids = request_payloads[2]
        keys_num = len(block_hashes)
        key_len = len(block_hashes[0])
        total_size += keys_num * key_len
        total_size += len(block_ids) * 8
        buffer = bytearray(total_size)
        message = MsgHeader.from_buffer(buffer)

        message.op = request_type
        message.node_id = ctypes.c_uint8(instance_id)
        message.len = ctypes.c_uint32(total_size - ctypes.sizeof(MsgHeader))
        message.uid = ctypes.c_uint32(request_id)
        common = MsgCommon.from_buffer(buffer, ctypes.sizeof(MsgHeader))
        common.keys_num = ctypes.c_uint32(keys_num)
        common.key_len = ctypes.c_uint32(key_len)
        for block_hash in block_hashes:
            buffer[data_pos:data_pos + len(block_hash)] = block_hash
            data_pos += len(block_hash)
        for block_id in block_ids:
            buffer[data_pos:data_pos + 8] = ctypes.c_uint64(block_id)
            data_pos += 8
        assert data_pos == total_size, (
            "The total size should be equal to the filled size"
            f"But got {data_pos} and {total_size}"
        )
        return buffer

    return None


def decode_message(message: list[bytes]) -> tuple[int, int]:
    if len(message) < ctypes.sizeof(MsgResponse):
        raise Exception("response is not matched")
    writable_buffer = bytearray(message)
    response = MsgResponse.from_buffer(writable_buffer)
    return response.uid, response.status
