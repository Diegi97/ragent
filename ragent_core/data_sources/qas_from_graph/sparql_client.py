from functools import wraps, cache
from typing import Any, Dict, List
from SPARQLWrapper import SPARQLWrapper, JSON

from ragent_core.data_sources.qas_from_graph.config import SPARQL_ENDPOINT, SPARQL_GRAPH


# -----------------------------
# SPARQL client wrapper
# -----------------------------


class SparqlClient:
    """Thin wrapper for SPARQLWrapper with logging.

    Provides a run(query) -> bindings list method.
    """

    def __init__(self, endpoint: str = SPARQL_ENDPOINT, graph: str = SPARQL_GRAPH):
        self._sparql = SPARQLWrapper(endpoint, graph)
        self._sparql.setReturnFormat(JSON)

    def run(self, query: str) -> List[Dict[str, Any]]:
        self._sparql.setQuery(query)
        results = self._sparql.query().convert()
        return results["results"]["bindings"]

    @cache
    def fetch_triples(self, ref: str) -> List[Dict[str, Any]]:
        q = f"SELECT ?p ?o WHERE {{ <{ref}> ?p ?o . }}"
        rows = self.run(q)
        return rows
