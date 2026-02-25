# SPDX-License-Identifier: Apache-2.0
import torch
import enum
import zmq

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from vllm.distributed.kv_transfer.kv_connector.v1.crosscache.adaptor import LoadStoreOp, SchedulerAdapter, WorkerAdapter

logger = init_logger(__name__)

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


def reformat_block_ids(block_ids: tuple[list[int], ...] | None) -> list[int]:
    if block_ids is None:
        return []
    assert isinstance(block_ids, tuple), (
        f"Expected block_ids to be a tuple of lists, but got {type(block_ids)}"
    )

    if len(block_ids) > 1:
        raise RuntimeError(
            "CrossCacheConnector only works without hybrid kv cache manager. "
            "Please pass --disable-hybrid-kv-cache-manager when starting vllm"
        )

    return block_ids[0]


def create_scheduler_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig
) -> SchedulerAdapter:
    # TODO: have a helper function to calculate the correct rank and
    # world size for the MLA and other models
    return SchedulerAdapter(
        server_url,
        zmq_context,
        vllm_config.model_config.model,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.cache_config.block_size,
    )


def create_worker_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig
) -> WorkerAdapter:
    # TODO: have a helper function to calculate the correct rank and
    # world size for the MLA and other models
    return WorkerAdapter(
        server_url,
        zmq_context,
        vllm_config.model_config.model,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.cache_config.block_size,
    )


def convert_block_hashes_to_bytes(
    block_hashes: list["BlockHash"],
) -> list[bytes]:
    return cast(list[bytes], block_hashes)


class CacheRequestState(enum.Enum):
    """
    State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()


@dataclass
class CacheRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only lists to track the token ids and block hashes
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList["BlockHash"]

    # Block ids and hashes will be updated at update_states_after_alloc and
    # during the generation
    allocated_block_ids: list[int] = field(default_factory=list)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving half-full blocks.
    num_scheduled_tokens: int = 0

    # Number of blocks stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_blocks: int = 0

    # Staging load operation -- save vllm and cache hit tokens during lookup
    num_vllm_hit_blocks: int = 0
    num_cache_hit_blocks: int = 0

    # Main state
    state: CacheRequestState = CacheRequestState.PREFETCHING

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = []
        self.num_stored_blocks = 0
        self.num_vllm_hit_blocks = 0
        self.num_cache_hit_blocks = 0
        self.state = CacheRequestState.PREFETCHING

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_cache_hit_blocks > self.num_vllm_hit_blocks
            and self.state != CacheRequestState.READY
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == CacheRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_scheduled_tokens(self, num_new_tokens: int):
        self.num_scheduled_tokens += num_new_tokens

    def increase_num_stored_blocks(self, num_new_blocks: int):
        """Increase the number of stored blocks for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_blocks += num_new_blocks

    def append_block_ids(
        self,
        new_block_ids: list[int],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        self.allocated_block_ids.extend(new_block_ids)

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"CacheRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks={len(self.allocated_block_ids)}, "
            f"num_stored_blocks={self.num_stored_blocks}, "
            f"vllm_hit_blocks={self.num_vllm_hit_blocks}, "
            f"cache_hit_blocks={self.num_cache_hit_blocks}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class CacheRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp

    @staticmethod
    def GetStoreMetadata(
        tracker: CacheRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "CacheRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a Cache data chunk
            vllm_block_size: the block size used in vLLM
        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        min_available_blocks = min(
            len(tracker.block_hashes),
            len(tracker.allocated_block_ids),
            tracker.num_scheduled_tokens // vllm_block_size,
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            block_hashes = convert_block_hashes_to_bytes(
                tracker.block_hashes[start:end]
            )
            block_ids = tracker.allocated_block_ids[start:end]

            ret = CacheRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=LoadStoreOp(block_hashes=block_hashes, block_ids=block_ids),
            )

            # Update the request tracker
            tracker.increase_num_stored_blocks(end - start)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: CacheRequestTracker,
        blocks_in_chunk: int,
    ) -> "CacheRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a Cache data chunk
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_blocks |
        # | cache chunk 1   | cache chunk 2   |
        #                     |  need to retrieve |

        start = tracker.num_vllm_hit_blocks // blocks_in_chunk * blocks_in_chunk
        end = tracker.num_cache_hit_blocks
        assert end % blocks_in_chunk == 0, (
            "The number of Cache hit blocks should be a multiple of the "
            "number of blocks in a cache chunk. "
        )
        assert len(tracker.block_hashes) >= end, (
            "The number of block hashes should be greater than or equal to the "
            "number of Cache hit blocks. "
        )
        if end > start:
            block_hashes = convert_block_hashes_to_bytes(
                tracker.block_hashes[start:end]
            )
            block_ids = tracker.allocated_block_ids[start:end]

            ret = CacheRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=LoadStoreOp(block_hashes=block_hashes, block_ids=block_ids),
            )
            return ret

        return None


class CacheConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.requests: list[CacheRequestMetadata] = []

    def add_request_metadata(self, request_metadata: CacheRequestMetadata):
        self.requests.append(request_metadata)

    def __len__(self):
        return len(self.requests)

    # For debugging
    def __str__(self):
        request_strs = []
        for req_meta in self.requests:
            request_strs.append(
                f"RequestMetadata(request_id={req_meta.request_id}, "
                f"direction={req_meta.direction}, "
                f"num_blocks={len(req_meta.op)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return "[" + "\n".join(request_strs) + "]"

    def __repr__(self):
        return self.__str__()


class CrossCacheConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        server_host = vllm_config.kv_transfer_config.get_from_extra_config(
            "cache.server.host", "tcp://localhost"
        )
        server_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "cache.server.port", 5555
        )

        server_url = f"{server_host}:{server_port}"
        zmq_context = zmq.Context.instance()
        if self.role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter = create_scheduler_adapter(
                server_url, zmq_context, vllm_config
            )
            self.request_trackers: dict[str, CacheRequestTracker] = {}
        elif self.role == KVConnectorRole.WORKER:
            self.worker_adapter = create_worker_adapter(
                server_url, zmq_context, vllm_config
            )
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")
        self.vllm_block_size = vllm_config.cache_config.block_size

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================
    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Register kv caches!")
        self.worker_adapter.register(kv_caches)
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        return

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any
    ) -> None:
        logger.debug("Worker start load kv...")
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CacheConnectorMetadata)

        request_ids = []
        ops = []

        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)

        if len(request_ids) > 0:
            logger.info(
                "HERE! SUBMITTING THE BATCHED RETRIEVE REQUESTS %s", request_ids
            )
            self.worker_adapter.load_requests(
                request_ids, ops
            )

    def wait_for_save(self):
        logger.debug("Worker wait for save...")
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CacheConnectorMetadata)

        request_ids = []
        ops = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)

        if len(request_ids) > 0:
            self.worker_adapter.store_requests(request_ids, ops)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        val = self.worker_adapter.get_finished(finished_req_ids)
        # logger.error("Finished req ids: %s, %s", val[0], val[1])
        return val

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        if hasattr(self, "worker_adapter"):
            self.worker_adapter.unregister()
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        logger.debug("enter get_num_new_matched_tokens (computed:%d)...", num_computed_tokens)
        tracker = self._get_or_create_request_tracker(request)
        logger.debug("get_num_new_matched_tokens tracker:%s", tracker)

        self.scheduler_adapter.try_lookup(
            request.request_id, convert_block_hashes_to_bytes(request.block_hashes)
        )

        ret = self.scheduler_adapter.check_lookup_result(request.request_id)
        if ret is None:
            return None, True

        if ret == 0:
            return 0, False

        assert (
            ret % (self.scheduler_adapter.num_blocks_per_chunk() * self.vllm_block_size)
            == 0
        )

        # Update num stored blocks for the tracker
        num_vllm_blocks = num_computed_tokens // self.vllm_block_size
        num_cache_blocks = ret // self.vllm_block_size
        tracker.increase_num_stored_blocks(num_cache_blocks)

        # Save the vllm and cache hit tokens
        tracker.num_vllm_hit_blocks = num_vllm_blocks
        tracker.num_cache_hit_blocks = num_cache_blocks

        need_to_load = max(0, ret - num_computed_tokens)
        logger.debug(
            "vLLM hit is: %d, Need to load is %d", num_computed_tokens, need_to_load
        )
        return need_to_load, need_to_load > 0

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        logger.debug("enter update_state_after_alloc (external:%d)...", num_external_tokens)
        # NOTE: the `blocks` are NEW BLOCKS allocated for this request.
        tracker = self._get_request_tracker(request.request_id)
        logger.debug("update_state_after_alloc tracker:%s", tracker)
        block_ids = reformat_block_ids(blocks.get_block_ids())
        logger.debug("block ids:%s", str(block_ids))

        # No matter we need to retrieve or not, we need to update
        # the block ids into the tracker
        tracker.append_block_ids(block_ids)

        # Update the state of the tracker
        condition = tracker.needs_retrieve()
        if tracker.state == CacheRequestState.PREFETCHING:
            # If need to retrieve, change to WAITING_FOR_LOAD
            # Otherwise, change to READY
            tracker.state = (
                CacheRequestState.WAITING_FOR_LOAD
                if condition
                else CacheRequestState.READY
            )
            # Clean up lookup future in scheduler adapter
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
        logger.debug("update_state_after_alloc(done) tracker:%s", tracker)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = CacheConnectorMetadata()

        self._process_retrieve_requests(metadata)
        self._process_new_requests(scheduler_output, metadata)
        self._process_cached_requests(scheduler_output, metadata)

        if len(metadata) > 0:
            logger.debug("Final connector metadata: %s", metadata)

        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)
        return True, None

    ##############################
    # Helper functions
    ##############################
    def _process_retrieve_requests(
        self,
        metadata: CacheConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        for request_tracker in self.request_trackers.values():
            if request_tracker.state != CacheRequestState.WAITING_FOR_LOAD:
                continue
            r_metadata = CacheRequestMetadata.GetRetrieveMetadata(
                request_tracker, blocks_per_chunk
            )
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)
            request_tracker.state = CacheRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: CacheConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)

            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = CacheRequestMetadata.GetStoreMetadata(
                request_tracker, blocks_per_chunk, self.vllm_block_size
            )
            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: CacheConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids
            new_block_ids = reformat_block_ids(cached_reqs.new_block_ids[idx])
            request_tracker.append_block_ids(new_block_ids)

            # Update new scheduled tokens
            num_new_tokens = cached_reqs.num_computed_tokens[idx]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = CacheRequestMetadata.GetStoreMetadata(
                request_tracker, blocks_per_chunk, self.vllm_block_size
            )

            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _get_request_tracker(self, request_id: str) -> CacheRequestTracker:
        assert request_id in self.request_trackers, (
            f"Request tracker for request_id {request_id} not found. "
        )
        return self.request_trackers[request_id]

    def _get_or_create_request_tracker(
        self, request: "Request"
    ) -> CacheRequestTracker:
        request_id = request.request_id
        if request_id not in self.request_trackers:
            new_tracker = CacheRequestTracker(request)
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _cleanup_request_tracker(self, request_id: str) -> None:
        """
        Clean up request tracker and associated lookup future for a request.
        This should be called when a request is finished to prevent memory leak.
        """
        # Clean up request tracker
        if self.request_trackers.pop(request_id, None):
            logger.debug(
                "[KVConnector] Cleaned up request_tracker for request %s",
                request_id,
            )
