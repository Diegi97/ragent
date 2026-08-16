import asyncio
import logging
import os
import queue
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from typing import Any

from pydantic import BaseModel, Field

from data_pipelines.pipelines.deep_search_task_generation.config import (
    DEFAULT_RETRIEVER_WORKER_PORT,
)
from ragent_core.retrievers import TurbopufferRetriever
from ragent_core.retrievers.document import RetrievalResult

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
RETRIEVER_WORKER_HOST = "127.0.0.1"
RETRIEVER_AUTHKEY_ENV = "RAGENT_RETRIEVER_AUTHKEY"
_STOP = object()


class RetrieverWorkerConfig(BaseModel):
    retriever_namespace: str = Field(default="default", min_length=1)
    retriever_device: str | None = None
    rerank_threshold: float = Field(default=3.0, ge=0.0)
    port: int = Field(default=DEFAULT_RETRIEVER_WORKER_PORT, ge=1, le=65535)

    def public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class RetrievalRequest:
    protocol_version: int
    client_id: str
    request_id: str
    query: str
    table_name: str
    top_k: int


@dataclass(frozen=True)
class RetrievalError:
    type: str
    message: str


@dataclass(frozen=True)
class RetrievalResponse:
    protocol_version: int
    client_id: str
    request_id: str
    results: tuple[dict[str, Any], ...]
    error: RetrievalError | None


class RetrieverWorkerRemoteError(RuntimeError):
    def __init__(self, error: RetrievalError) -> None:
        super().__init__(f"Retriever worker failed: {error.type}: {error.message}")
        self.error_type = error.type


def authkey_from_environment() -> bytes:
    value = os.getenv(RETRIEVER_AUTHKEY_ENV, "")
    if not value:
        raise ValueError(f"{RETRIEVER_AUTHKEY_ENV} is not set")
    return value.encode("utf-8")


class RetrieverBroker:
    """In-memory FIFO broker with one sequential retriever consumer."""

    def __init__(
        self,
        retriever: TurbopufferRetriever,
        config: RetrieverWorkerConfig,
    ) -> None:
        self._retriever = retriever
        self._config = config
        self._requests: queue.Queue[RetrievalRequest | object] = queue.Queue(maxsize=0)
        self._responses: dict[str, queue.Queue[RetrievalResponse]] = {}
        self._clients_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._accepting = True
        self._active_request_id: str | None = None
        self._completed_requests = 0
        self._failed_requests = 0
        self._started_at = time.time()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="deep-search-tasks-retriever",
            daemon=False,
        )

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        with self._state_lock:
            self._accepting = False
        self._stopping.set()
        self._requests.put(_STOP)
        self._worker.join()

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            accepting = self._accepting
            active_request_id = self._active_request_id
            completed_requests = self._completed_requests
            failed_requests = self._failed_requests
        with self._clients_lock:
            connected_clients = len(self._responses)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": "ready" if accepting else "stopping",
            "host": RETRIEVER_WORKER_HOST,
            "port": self._config.port,
            "pid": os.getpid(),
            "uptime_seconds": max(0.0, time.time() - self._started_at),
            "queued_requests": self._requests.qsize(),
            "active_request_id": active_request_id,
            "connected_clients": connected_clients,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "config": self._config.public_dict(),
        }

    def register_client(self, client_id: str) -> None:
        if not client_id:
            raise ValueError("client_id must not be empty")
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("Retriever worker is stopping")
            with self._clients_lock:
                if client_id in self._responses:
                    raise ValueError(
                        f"Retriever client is already registered: {client_id}"
                    )
                self._responses[client_id] = queue.Queue(maxsize=0)

    def unregister_client(self, client_id: str) -> None:
        with self._clients_lock:
            self._responses.pop(client_id, None)

    def submit(self, request: RetrievalRequest) -> None:
        self._validate_request(request)
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("Retriever worker is stopping")
            with self._clients_lock:
                if request.client_id not in self._responses:
                    raise ValueError(
                        f"Retriever client is not registered: {request.client_id}"
                    )
                self._requests.put(request)

    def get_response(
        self,
        client_id: str,
        timeout: float = 1.0,
    ) -> RetrievalResponse | None:
        with self._clients_lock:
            response_queue = self._responses.get(client_id)
        if response_queue is None:
            raise ValueError(f"Retriever client is not registered: {client_id}")
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def _validate_request(request: RetrievalRequest) -> None:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        if request.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                "Retriever protocol mismatch: "
                f"client={request.protocol_version}, worker={PROTOCOL_VERSION}"
            )
        if not request.client_id or not request.request_id:
            raise ValueError("client_id and request_id must not be empty")
        if not request.query or not request.table_name:
            raise ValueError("query and table_name must not be empty")
        if request.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            request = self._requests.get()
            if request is _STOP:
                return
            if not isinstance(request, RetrievalRequest):
                logger.error("Discarding invalid retriever queue item: %r", request)
                continue
            self._process_request(request)

    def _process_request(self, request: RetrievalRequest) -> None:
        with self._state_lock:
            self._active_request_id = request.request_id
        logger.info(
            "Processing retrieval request %s for client %s (top_k=%d)",
            request.request_id,
            request.client_id,
            request.top_k,
        )

        try:
            results = self._retrieve(request)
            serialized_results = tuple(result.to_dict() for result in results)
        except Exception as exc:
            logger.exception("Retriever request %s failed", request.request_id)
            response = RetrievalResponse(
                protocol_version=PROTOCOL_VERSION,
                client_id=request.client_id,
                request_id=request.request_id,
                results=(),
                error=RetrievalError(
                    type=type(exc).__name__,
                    message=str(exc),
                ),
            )
            with self._state_lock:
                self._failed_requests += 1
        else:
            response = RetrievalResponse(
                protocol_version=PROTOCOL_VERSION,
                client_id=request.client_id,
                request_id=request.request_id,
                results=serialized_results,
                error=None,
            )
            with self._state_lock:
                self._completed_requests += 1
            logger.info("Completed retrieval request %s", request.request_id)
        finally:
            with self._state_lock:
                self._active_request_id = None

        with self._clients_lock:
            response_queue = self._responses.get(request.client_id)
        if response_queue is None:
            logger.warning(
                "Discarding response %s because client %s disconnected",
                request.request_id,
                request.client_id,
            )
            return
        response_queue.put(response)

    def _retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        reranker = self._retriever.reranker
        if reranker is None:
            raise RuntimeError("Retriever worker requires a CrossEncoder reranker")
        # top_k is a candidate/result ceiling, not a requested result count.
        # The CrossEncoder scores up to this many fused chunks and its relevance
        # threshold decides how many of them are actually returned.
        reranker.top_rerank = request.top_k
        return self._retriever.retrieve(
            request.query,
            table_name=request.table_name,
            top_k=request.top_k,
        )


_manager_broker: RetrieverBroker | None = None


def _get_manager_broker() -> RetrieverBroker:
    if _manager_broker is None:
        raise RuntimeError("Retriever broker has not been initialized")
    return _manager_broker


class RetrieverManager(BaseManager):
    pass


RetrieverManager.register(
    "get_broker",
    callable=_get_manager_broker,
    exposed=(
        "health",
        "register_client",
        "unregister_client",
        "submit",
        "get_response",
    ),
)


def load_worker_retriever(config: RetrieverWorkerConfig) -> TurbopufferRetriever:
    return TurbopufferRetriever.load_index(
        namespace=config.retriever_namespace,
        device=config.retriever_device,
        rerank_threshold=config.rerank_threshold,
        top_rerank=1,
        rerank_batch_size=8,
    )


def serve_retriever_worker(config: RetrieverWorkerConfig) -> None:
    global _manager_broker

    authkey = authkey_from_environment()
    retriever = load_worker_retriever(config)
    broker = RetrieverBroker(retriever, config)
    manager = RetrieverManager(
        address=(RETRIEVER_WORKER_HOST, config.port),
        authkey=authkey,
    )
    server = manager.get_server()

    def stop_on_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    _manager_broker = broker
    broker.start()
    logger.info(
        "Retriever worker ready on %s:%d (pid=%d)",
        RETRIEVER_WORKER_HOST,
        config.port,
        os.getpid(),
    )
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping retriever worker after the active request finishes.")
    finally:
        broker.stop()
        _manager_broker = None
        signal.signal(signal.SIGTERM, previous_sigterm)


class AsyncRetrieverWorkerClient:
    def __init__(self, client_id: str, port: int) -> None:
        if not client_id:
            raise ValueError("client_id must not be empty")
        self.client_id = client_id
        self.port = port
        self._manager: RetrieverManager | None = None
        self._broker: Any = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[list[RetrievalResult]]] = {}
        self._closing = False
        self._dispatcher_error: str | None = None
        self.worker_info: dict[str, Any] | None = None

    async def __aenter__(self) -> "AsyncRetrieverWorkerClient":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._broker is not None:
            raise RuntimeError("Retriever worker client is already started")
        self._closing = False
        self._dispatcher_error = None

        authkey = authkey_from_environment()

        def connect() -> tuple[RetrieverManager, Any, dict[str, Any]]:
            manager = RetrieverManager(
                address=(RETRIEVER_WORKER_HOST, self.port),
                authkey=authkey,
            )
            manager.connect()
            broker = manager.get_broker()
            info = dict(broker.health())
            if info.get("protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError(
                    "Retriever protocol mismatch: "
                    f"client={PROTOCOL_VERSION}, "
                    f"worker={info.get('protocol_version')}"
                )
            broker.register_client(self.client_id)
            return manager, broker, info

        self._manager, self._broker, self.worker_info = await asyncio.to_thread(connect)
        self._dispatcher = asyncio.create_task(
            self._dispatch_responses(),
            name=f"retriever-responses-{self.client_id}",
        )

    async def close(self) -> None:
        if self._broker is None:
            return
        self._closing = True
        broker = self._broker
        try:
            await asyncio.to_thread(broker.unregister_client, self.client_id)
        except Exception:
            logger.warning("Failed to unregister retriever client", exc_info=True)
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        self._fail_pending(RuntimeError("Retriever worker client closed"))
        self._dispatcher = None
        self._broker = None
        self._manager = None

    async def health(self) -> dict[str, Any]:
        if self._broker is None:
            raise RuntimeError("Retriever worker client is not started")
        if self._dispatcher_error is not None:
            raise RuntimeError(self._dispatcher_error)
        return dict(await asyncio.to_thread(self._broker.health))

    async def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        if self._broker is None or self._closing:
            raise RuntimeError("Retriever worker client is not running")
        if self._dispatcher_error is not None:
            raise RuntimeError(self._dispatcher_error)
        request_id = uuid.uuid4().hex
        future: asyncio.Future[list[RetrievalResult]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        request = RetrievalRequest(
            protocol_version=PROTOCOL_VERSION,
            client_id=self.client_id,
            request_id=request_id,
            query=query,
            table_name=table_name,
            top_k=top_k,
        )
        try:
            await asyncio.to_thread(self._broker.submit, request)
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _dispatch_responses(self) -> None:
        try:
            while not self._closing:
                response = await asyncio.to_thread(
                    self._broker.get_response,
                    self.client_id,
                    0.5,
                )
                if response is None:
                    continue
                if not isinstance(response, RetrievalResponse):
                    raise TypeError("Retriever worker returned an invalid response")
                if response.protocol_version != PROTOCOL_VERSION:
                    raise RuntimeError("Retriever response protocol mismatch")
                if response.client_id != self.client_id:
                    raise RuntimeError(
                        "Retriever response was routed to the wrong client: "
                        f"expected={self.client_id}, actual={response.client_id}"
                    )
                future = self._pending.pop(response.request_id, None)
                if future is None or future.cancelled():
                    continue
                if response.error is not None:
                    future.set_exception(RetrieverWorkerRemoteError(response.error))
                else:
                    future.set_result(
                        [RetrievalResult(**result) for result in response.results]
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                logger.exception("Retriever response dispatcher stopped unexpectedly")
                self._dispatcher_error = f"Lost connection to retriever worker: {exc}"
                self._fail_pending(RuntimeError(self._dispatcher_error))

    def _fail_pending(self, exc: BaseException) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)
