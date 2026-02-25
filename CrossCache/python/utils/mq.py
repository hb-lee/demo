# SPDX-License-Identifier: GPL-2.0

import queue
import threading
import zmq

from dataclasses import dataclass
from typing import Any, TypeVar

from vllm.distributed.kv_transfer.kv_connector.v1.crosscache.utils.futures import MessagingFuture
from vllm.distributed.kv_transfer.kv_connector.v1.crosscache.utils.protocol import RequestType, encode_message, decode_message

RequestUID = int

T = TypeVar("T")

class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any]
        request_type: RequestType
        request_payloads: list[Any]

    def __init__(self, server_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(server_url)

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # sender and receiver thread
        self.is_finished = threading.Event()
        self.sender_queue: queue.Queue = queue.Queue()
        self.sender_condition = threading.Condition()
        self.sender = threading.Thread(target=self._sender_loop, daemon=True)
        self.receiver = threading.Thread(target=self._receiver_loop, daemon=True)

        # Pending jobs' futures
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}
        self.pending_id = 0
        self.pending_lock = threading.Lock()
        self.sender.start()
        self.receiver.start()

    def _sender_loop(self):
        while not self.is_finished.is_set():
            with self.sender_condition:
                if self.sender_queue.empty():
                    self.sender_condition.wait()

            try:
                # handler the request from queue
                while wrapped_request := self.sender_queue.get_nowait():
                    with self.pending_lock:
                        request_uid = self.pending_id
                        self.pending_id += 1
                        self.pending_futures[request_uid] = wrapped_request.future

                    b_payloads = encode_message(
                        wrapped_request.request_type, request_uid, wrapped_request.request_payloads)
                    self.socket.send(b_payloads)

            except queue.Empty:
                pass

    def _receiver_loop(self):
        while not self.is_finished.is_set():
            events = dict(self.poller.poll(1000))
            if self.socket in events and events[self.socket] == zmq.POLLIN:
                message = self.socket.recv()
                request_uid, result = decode_message(message)

                if request_uid in self.pending_futures:
                    with self.pending_lock:
                        future = self.pending_futures.pop(request_uid)
                    if result is not None:
                        future.set_result(result)
                    else:
                        future.set_result(None)

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
    ) -> MessagingFuture[T]:
        """
        Submit a request to the server.

        Args:
            request_type (RequestType): The type of the request.
            request_payloads (list[Any]): The payloads of the request.

        Returns:
            MessagingFuture[T]: A future that will hold the response.
        """
        future: MessagingFuture[T] = MessagingFuture()
        self.sender_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=0,  # set uid while sending
                future=future,
                request_type=request_type,
                request_payloads=request_payloads,
            )
        )
        with self.sender_condition:
            self.sender_condition.notify()
        return future

    def close(self) -> None:
        self.is_finished.set()
        self.sender.join()
        self.receiver.join()
        self.socket.close()
