import asyncio
import logging
import uuid
from typing import Any

from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.contracts import (
    PROTOCOL_VERSION,
    RETRIEVER_WORKER_HOST,
    RetrievalRequest,
    RetrievalResponse,
    RetrieverWorkerRemoteError,
    WorkerHealth,
    authkey_from_environment,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.transport import (
    RetrieverManager,
)
from ragent_core.retrievers.document import RetrievalResult

logger = logging.getLogger(__name__)


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
        self.worker_info: WorkerHealth | None = None

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

        def connect() -> tuple[RetrieverManager, Any, WorkerHealth]:
            manager = RetrieverManager(
                address=(RETRIEVER_WORKER_HOST, self.port),
                authkey=authkey,
            )
            manager.connect()
            broker = manager.get_broker()
            info = WorkerHealth.model_validate(broker.health())
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

    async def health(self) -> WorkerHealth:
        if self._broker is None:
            raise RuntimeError("Retriever worker client is not started")
        if self._dispatcher_error is not None:
            raise RuntimeError(self._dispatcher_error)
        return WorkerHealth.model_validate(await asyncio.to_thread(self._broker.health))

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
                future = self._pending.get(response.request_id)
                if future is None or future.done():
                    continue
                if response.error is not None:
                    future.set_exception(RetrieverWorkerRemoteError(response.error))
                else:
                    future.set_result(
                        [
                            RetrievalResult.from_dict(result)
                            for result in response.results
                        ]
                    )
                self._pending.pop(response.request_id, None)
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
