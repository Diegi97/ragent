from multiprocessing.managers import BaseManager
from typing import Any

BROKER_OPERATIONS = (
    "health",
    "register_client",
    "unregister_client",
    "submit",
    "get_response",
)


class RetrieverManager(BaseManager):
    @classmethod
    def serving(
        cls, broker: Any, *, address: tuple[str, int], authkey: bytes
    ) -> "RetrieverManager":
        # The server runs in this process, so its callable can capture the broker.
        # A separate registry keeps client construction free of server globals.
        class ServerManager(cls):
            pass

        ServerManager.register(
            "get_broker", callable=lambda: broker, exposed=BROKER_OPERATIONS
        )
        return ServerManager(address=address, authkey=authkey)


RetrieverManager.register("get_broker", exposed=BROKER_OPERATIONS)
