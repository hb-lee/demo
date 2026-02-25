import torch
import zmq
import ctypes
from collections.abc import Iterable
from itertools import islice
from ctypes import c_uint8, POINTER, cast

from dataclasses import dataclass
from vllm.logger import init_logger

import torch_npu
from . import ipckey_utils

from vllm.distributed.kv_transfer.kv_connector.v1.crosscache.utils.mq import MessageQueueClient, MessagingFuture
from vllm.distributed.kv_transfer.kv_connector.v1.crosscache.utils.protocol import (
    RequestType,
    to_register_payloads,
    to_lookup_payloads,
    to_store_payloads,
    to_load_payloads,
)

logger = init_logger(__name__)


def bytes_to_uint8_ptr(data: bytes):
    c_str = ctypes.c_char_p(data)
    uint8_ptr = cast(c_str, POINTER(c_uint8))
    return uint8_ptr


def striding_block_hashes(
    block_hashes: list[bytes],
    blocks_in_chunk,
) -> Iterable[bytes]:
    """ Striding the block hashes to get the block hashes for each chunk.
    For example, if blocks_in_chunk is 16, then we will get the block hashes
    for the 16th, 32nd, 48th, ... blocks.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


@dataclass
class LoadStoreOp:
    block_hashes: list[bytes]
    block_ids: list[int]

    def __len__(self) -> int:
        return len(self.block_hashes)

    def __post_init__(self):
        assert len(self.block_hashes) == len(self.block_ids), (
            "The number of block hashes should be equal to the number of block ids "
            f"But got {len(self.block_hashes)} and {len(self.block_ids)}"
        )


class SchedulerAdapter(object):
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int,
        kv_rank: int,
        vllm_block_size: int,
    ):
        self.mq_client = MessageQueueClient(server_url, context)
        self.instance_id = 0
        self.model_name = model_name
        self.world_size = world_size
        self.worker_id = kv_rank

        self.chunk_size = 256  # Cache chunk
        self.blocks_in_chunk = self.chunk_size // vllm_block_size
        self.lookup_futures: dict[str, MessagingFuture] = {}

    def try_lookup(self, request_id: str, block_hashes: list[bytes]):
        if request_id in self.lookup_futures:
            return

        logger.debug("Scheduler lookup for req:%s", request_id)
        # [2183 2129 3099 238 | 3289 3894 84809 3902] --> [2183 3289]
        s = striding_block_hashes(block_hashes, self.blocks_in_chunk)
        keys = [block_hash for block_hash in s]
        future = self.mq_client.submit_request(
            RequestType.LOOKUP,
            to_lookup_payloads(self.instance_id, keys, self.world_size, self.model_name),
        )
        self.lookup_futures[request_id] = future

    def check_lookup_result(self, request_id: str) -> int | None:
        assert request_id in self.lookup_futures, (
            f"Lookup request for request_id={request_id} has not been submitted"
        )

        future = self.lookup_futures[request_id]
        if not future.query():
            return None

        result = future.result()
        logger.debug("Scheduler lookup result:%d", result)
        return result

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup future for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self.lookup_futures.pop(request_id, None)

    def num_blocks_per_chunk(self) -> int:
        return self.blocks_in_chunk


class WorkerAdapter(object):
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int,
        kv_rank: int,
        vllm_block_size: int,
    ):
        self.mq_client = MessageQueueClient(server_url, context)
        self.instance_id = 0
        self.model_name = model_name
        self.world_size = world_size
        self.worker_id = kv_rank

        self.chunk_size = 256  # Cache chunk
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

        self.store_futures: dict[str, MessagingFuture] = {}
        self.load_futures: dict[str, MessagingFuture] = {}

        self.finished_stores: set[str] = set()
        self.previously_finished: set[str] = set()

    def _block_hashes_to_keys(
        self, block_hashes: list[bytes]
    ) -> list[bytes]:
        """Convert block hashes to IPC cache engine keys"""
        s = striding_block_hashes(block_hashes, self.blocks_in_chunk)
        return [block_hash for block_hash in s]

    def register(self, kv_caches: dict[str, torch.Tensor]):
        logger.debug("Worker registering kv caches")
        kv_caches_ptrs: list[tuple[bytes, ...]] = []
        for tensor in kv_caches.values():
            k_bytes = ipckey_utils.ipckey_from_tensor(tensor[0])
            v_bytes = ipckey_utils.ipckey_from_tensor(tensor[1])
            logger.debug("K bytes:%s, V bytes:%s", k_bytes, v_bytes)
            kv_caches_ptrs.append((k_bytes, v_bytes))
        _, kv_tensor = next(iter(kv_caches.items()))
        logger.debug("ndim:%d,npu:%s", kv_tensor[0].ndim, kv_tensor[0].device)
        num_blocks = kv_tensor[0].shape[0]
        block_size = kv_tensor[0].shape[1]
        num_heads = kv_tensor[0].shape[2]
        head_size = kv_tensor[0].shape[3]
        element_size = kv_tensor[0].element_size()

        logger.debug("num_blocks:%d,block_size:%d,num_heads:%d,head_size:%d,element_size:%d", num_blocks, block_size,
                     num_heads, head_size, element_size)

        payloads = to_register_payloads(self.instance_id, self.world_size, self.worker_id, num_blocks, block_size,
                                        element_size, num_heads * head_size, 64, self.model_name, kv_caches_ptrs)
        future = self.mq_client.submit_request(
            RequestType.REGISTER_KV_CACHE, payloads
        )
        result = future.result()
        logger.debug("Worker register kv cache finished! result:%s", str(result))

    def unregister(self):
        logger.debug("Worker unregistering kv caches")
        future = self.mq_client.submit_request(
            RequestType.UNREGISTER_KV_CACHE, [self.instance_id]
        )
        future.result()
        self.mq_client.close()

    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp]
    ):
        keys = []
        block_ids = []
        for op in ops:
            keys.extend(self._block_hashes_to_keys(op.block_hashes))
            block_ids.extend(op.block_ids)
        logger.debug("Worker store for req:%s, keys:%s", request_ids, keys)
        future = self.mq_client.submit_request(
            RequestType.STORE,
            to_store_payloads(self.instance_id, keys, block_ids),
        )
        self.store_futures[request_ids[0]] = (future, request_ids[1:])

    def load_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp]
    ):
        keys = []
        block_ids = []
        for op in ops:
            keys.extend(self._block_hashes_to_keys(op.block_hashes))
            block_ids.extend(op.block_ids)
        logger.debug("Worker load for req:%s, keys:%s", request_ids, keys)
        future = self.mq_client.submit_request(
            RequestType.LOAD,
            to_load_payloads(self.instance_id, keys, block_ids),
        )
        self.load_futures[request_ids[0]] = (future, request_ids[1:])

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        logger.debug("Worker check finished for ids:%s", finished_req_ids)
        finished_stores = set()
        finished_loads = set()
        for request_id, (future, other_reqs) in self.store_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)
            if result != 0:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )
            logger.info("Store request for request_id=%s finished", request_id)

        for request_id, (future, other_reqs) in self.load_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_loads.add(request_id)
            finished_loads.update(other_reqs)

            if result != 0:
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    result,
                )
            logger.info("Retrieve request for request_id=%s finished", request_id)

        logger.debug("store futures:%s, load futures:%s, finished stores:%s prevfinished:%s", self.store_futures, self.load_futures, self.finished_stores, self.previously_finished)
        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_loads:
            self.load_futures.pop(request_id, None)

        # Update the internal states
        self.finished_stores.update(finished_stores)

        ret_stores = set()
        for req_id in finished_req_ids:
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)

        # Calculate the final finished stores
        ret_stores.update(self._update_and_get_finished_store())
        logger.debug("store futures:%s, load futures:%s, finished stores:%s prevfinished:%s", self.store_futures, self.load_futures, self.finished_stores, self.previously_finished)
        ret_stores = ret_stores.intersection(finished_req_ids)

        logger.debug("ret_store:%s, finished_loads:%s", ret_stores,  finished_loads)
        return ret_stores, finished_loads
