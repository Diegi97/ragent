import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker import (
    client as client_module,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.broker import (
    RetrieverBroker,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.client import (
    AsyncRetrieverWorkerClient,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.contracts import (
    PROTOCOL_VERSION,
    RETRIEVER_AUTHKEY_ENV,
    RETRIEVER_WORKER_HOST,
    RetrievalRequest,
    RetrievalResponse,
    WorkerStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.transport import (
    RetrieverManager,
)
from ragent_core.retrievers.document import RetrievalResult


class Retriever:
    def __init__(self):
        self.reranker = SimpleNamespace(top_rerank=0)
        self.queries = []

    def retrieve(self, query, **kwargs):
        self.queries.append(query)
        if query == "fail":
            raise ValueError("retrieval failed")
        return [RetrievalResult(id=0, content=query)]


def test_broker_fifo_failure_isolation_and_shutdown():
    retriever = Retriever()
    broker = RetrieverBroker(retriever, RetrieverWorkerConfig())
    broker.register_client("client")
    broker.start()
    try:
        for index, query in enumerate(["first", "fail", "last"]):
            broker.submit(
                RetrievalRequest(
                    PROTOCOL_VERSION, "client", str(index), query, "corpus", 5
                )
            )
        responses = [broker.get_response("client", timeout=2) for _ in range(3)]
        assert [response.request_id for response in responses] == ["0", "1", "2"]
        assert responses[1].error.message == "retrieval failed"
        assert responses[2].results[0]["content"] == "last"
        assert broker.health()["completed_requests"] == 2
        assert retriever.reranker.top_rerank == 5
    finally:
        broker.stop()
    assert broker.health()["status"] == WorkerStatus.STOPPING


class MalformedBroker:
    def __init__(self):
        self.responses = queue.Queue()

    def submit(self, request):
        self.responses.put(
            RetrievalResponse(
                PROTOCOL_VERSION,
                request.client_id,
                request.request_id,
                ({"score": 1.0},),
                None,
            )
        )

    def get_response(self, client_id, timeout):
        try:
            return self.responses.get(timeout=timeout)
        except queue.Empty:
            return None

    def unregister_client(self, client_id):
        pass


@pytest.mark.asyncio
async def test_malformed_response_fails_its_own_future_without_hanging():
    client = AsyncRetrieverWorkerClient("test", RetrieverWorkerConfig().port)
    client._broker = MalformedBroker()
    client._dispatcher = asyncio.create_task(client._dispatch_responses())
    try:
        with pytest.raises(RuntimeError, match="Lost connection to retriever worker"):
            await asyncio.wait_for(client.retrieve("q", "corpus", 1), timeout=2)
        assert not client._pending
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_fails_pending_requests():
    class IdleBroker(MalformedBroker):
        def submit(self, request):
            pass

    client = AsyncRetrieverWorkerClient("test", RetrieverWorkerConfig().port)
    client._broker = IdleBroker()
    request = asyncio.create_task(client.retrieve("q", "corpus", 1))
    while not client._pending:
        await asyncio.sleep(0)
    await client.close()
    with pytest.raises(RuntimeError, match="closed"):
        await request


@pytest.mark.asyncio
async def test_manager_transport_round_trip_and_cancelled_request_isolation(
    monkeypatch,
):

    started = threading.Event()
    release = threading.Event()

    class ControlledRetriever(Retriever):
        def retrieve(self, query, **kwargs):
            if query == "hold":
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release retrieval")
            return super().retrieve(query, **kwargs)

    authkey = b"test-manager-key"
    monkeypatch.setenv(RETRIEVER_AUTHKEY_ENV, authkey.decode())
    broker = RetrieverBroker(ControlledRetriever(), RetrieverWorkerConfig())
    manager = RetrieverManager.serving(
        broker, address=(RETRIEVER_WORKER_HOST, 0), authkey=authkey
    )
    server = manager.get_server()

    def serve():
        try:
            server.serve_forever()
        except SystemExit:
            pass

    broker.start()
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        async with (
            AsyncRetrieverWorkerClient("first", server.address[1]) as first,
            AsyncRetrieverWorkerClient("second", server.address[1]) as second,
        ):
            assert (await first.health()).connected_clients == 2
            request = asyncio.create_task(first.retrieve("hold", "source", 2))
            assert await asyncio.to_thread(started.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            release.set()
            result = await asyncio.wait_for(
                second.retrieve("after cancellation", "source", 2), 3
            )
            assert result[0].content == "after cancellation"
            assert not first._pending
    finally:
        release.set()
        await asyncio.to_thread(broker.stop)
        server.stop_event.set()
        server.listener.close()
        await asyncio.to_thread(thread.join, 2)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_incompatible_worker_rejected_before_client_registration(monkeypatch):

    broker = RetrieverBroker(Retriever(), RetrieverWorkerConfig())
    health = broker.health()
    health["protocol_version"] = PROTOCOL_VERSION - 1
    registered = []

    class IncompatibleManager:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            pass

        def get_broker(self):
            return SimpleNamespace(
                health=lambda: health,
                register_client=registered.append,
            )

    monkeypatch.setenv(RETRIEVER_AUTHKEY_ENV, "test-manager-key")
    monkeypatch.setattr(client_module, "RetrieverManager", IncompatibleManager)
    client = AsyncRetrieverWorkerClient("incompatible", RetrieverWorkerConfig().port)
    with pytest.raises(ValueError, match="Retriever protocol mismatch"):
        await client.start()
    assert registered == []
    assert client._broker is None
    assert client._dispatcher is None
