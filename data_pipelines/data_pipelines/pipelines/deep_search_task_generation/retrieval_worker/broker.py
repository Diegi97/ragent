import logging
import os
import queue
import threading
import time
from typing import Any, cast

from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.contracts import (
    PROTOCOL_VERSION,
    RETRIEVER_WORKER_HOST,
    RetrievalError,
    RetrievalRequest,
    RetrievalResponse,
    WorkerHealth,
    WorkerStatus,
)
from ragent_core.retrievers.document import RetrievalResult
from ragent_core.retrievers.retriever import TurbopufferRetriever

logger = logging.getLogger(__name__)
_STOP = object()


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
        return WorkerHealth.model_validate(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": WorkerStatus.READY if accepting else WorkerStatus.STOPPING,
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
        ).model_dump(mode="json")

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
        for name in ("client_id", "request_id", "query", "table_name"):
            value = getattr(request, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(request.top_k) is not int or request.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            request = self._requests.get()
            if request is _STOP:
                return
            self._process_request(cast(RetrievalRequest, request))

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
