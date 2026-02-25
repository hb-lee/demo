# SPDX-License-Identifier: GPL-2.0

import torch
import zmq
import hashlib

# import torch_npu
# import ipckey_utils

from utils.mq import MessageQueueClient, MessagingFuture
from utils.protocol import (
    RequestType,
    to_register_payloads,
    to_lookup_payloads,
    to_store_payloads,
    to_load_payloads,
)


class FakeConnector(object):
    def __init__(self):
        server_host = "tcp://localhost"
        server_port = 5555

        server_url = f"{server_host}:{server_port}"
        zmq_context = zmq.Context.instance()
        self.client = MessageQueueClient(server_url, zmq_context)
        self.world_size = 1
        self.rank_id = 0
        self.num_blocks = 40
        self.block_size = 2
        self.k_or_v = 2
        self.num_layers = 2
        self.model_name = "Qwen-2.5"
        self.hidden_dim_size = 4096
        self.instance_id = 0  # means on npu 0
        # ipckey_utils.ipckey_init()

    def dump_info(self, op):
        print("======== %s ======" % op)
        print("nid:%d, wsize:%d rid:%d, nblock:%d, bsize:%d, dimsize:%d mname:%s" %
              (self.instance_id, self.world_size, self.rank_id, self.num_blocks, self.block_size,
        self.hidden_dim_size, self.model_name))

    def get_kv_cache_ptrs(self, kvcaches: dict[str, tuple[torch.Tensor, ...]]) -> tuple[list[tuple[bytes, ...]], int]:
        kv_caches_ptrs: list[tuple[bytes, ...]] = []
        kv_caches_ptrs = [(tensor[0].data_ptr().to_bytes(8, 'little'), tensor[1].data_ptr().to_bytes(8, 'little')) for tensor in kvcaches.values()]
        return kv_caches_ptrs, 8

    def get_kv_cache_ptrs_npu(self, kvcaches: dict[str, tuple[torch.Tensor, ...]]) -> tuple[list[tuple[bytes, ...]], int]:
        kv_caches_ptrs: list[tuple[bytes, ...]] = []
        for tensor in kvcaches.values():
            k_bytes = ipckey_utils.ipckey_from_tensor(tensor[0])
            v_bytes = ipckey_utils.ipckey_from_tensor(tensor[1])
            kv_caches_ptrs.append((k_bytes, v_bytes))
            print("K bytes, V bytes:", k_bytes, v_bytes)
        return kv_caches_ptrs, 64

    def register(self):
        print("Registering kv caches!")
        kv_caches_raw: dict[str, (torch.Tensor, torch.Tensor)] = {}
        for layer in ["layer1", "layer2"]:
            tensor_k = torch.zeros(
                [self.num_blocks, self.block_size, self.hidden_dim_size], dtype=torch.bfloat16
            )
            tensor_v = torch.zeros(
                [self.num_blocks, self.block_size, self.hidden_dim_size], dtype=torch.bfloat16
            )
            '''
            tensor_k = torch_npu.empty_with_format(
                [self.num_blocks, self.block_size, self.hidden_dim_size],
                dtype=torch.bfloat16, device='npu:0', acl_format=2, base_addr_aligned_kb=2048
            )
            tensor_v = torch_npu.empty_with_format(
                [self.num_blocks, self.block_size, self.hidden_dim_size],
                dtype=torch.bfloat16, device='npu:0', acl_format=2, base_addr_aligned_kb=2048
            )
            '''
            kv_caches_raw[layer] = (tensor_k, tensor_v)

        kv_cache_ptrs, first_key_in_bytes = self.get_kv_cache_ptrs(kv_caches_raw)
        # kv_cache_ptrs, first_key_in_bytes = self.get_kv_cache_ptrs_npu(kv_caches_raw)
        payloads = to_register_payloads(self.instance_id, self.world_size, self.rank_id, self.num_blocks, self.block_size,
                        2, self.hidden_dim_size, first_key_in_bytes, self.model_name, kv_cache_ptrs)
        self.dump_info("Register")
        for item in kv_cache_ptrs:
            print("kptrs:%d, vptrs:%d" % (int.from_bytes(item[0], byteorder='little'), int.from_bytes(item[1], byteorder='little')))
        future = self.client.submit_request(
            RequestType.REGISTER_KV_CACHE,
            payloads
        )
        result = future.result()
        print("Register kv cache finished! result:%s" % str(result))

    def lookup(self):
        print("Lookup")
        prompt_str1 = b"I am a test case"
        prompt_str2 = b"and I want be a client"
        block_hashes = [hashlib.sha256(prompt_str1).digest(), hashlib.sha256(prompt_str2).digest()]
        self.dump_info("Lookup")
        future = self.client.submit_request(
            RequestType.LOOKUP,
            to_lookup_payloads(self.instance_id, block_hashes, self.world_size, self.model_name)
        )
        result = future.result()
        print("Lookup finished! result:%s" % str(result))

    def store(self):
        print("Store")
        prompt_str1 = b"I am a test case"
        prompt_str2 = b"and I want be a client"
        keys = [hashlib.sha256(prompt_str1).digest(), hashlib.sha256(prompt_str2).digest()]
        block_ids = [1, 3, 4, 7]
        self.dump_info("Store")
        future = self.client.submit_request(
            RequestType.STORE,
            to_store_payloads(self.instance_id, keys, block_ids)
        )
        result = future.result()
        print("Store finished! result:%s" % str(result))

    def load(self):
        print("Load")
        prompt_str1 = b"I am a test case"
        prompt_str2 = b"and I want be a client"
        keys = [hashlib.sha256(prompt_str1).digest(), hashlib.sha256(prompt_str2).digest()]
        block_ids = [1, 3, 4, 7]
        self.dump_info("Load")
        future = self.client.submit_request(
            RequestType.LOAD,
            to_load_payloads(self.instance_id, keys, block_ids)
        )
        result = future.result()
        print("Load finished! result:%s" % str(result))


connector = FakeConnector()
connector.register()
print("------------- lookup first ----------")
connector.lookup()
connector.store()

print("------------- lookup second ----------")
connector.lookup()
connector.load()
