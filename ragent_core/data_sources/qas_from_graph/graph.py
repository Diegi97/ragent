import copy
import json
import os
from typing import Any, Dict, List, Tuple, Optional, ClassVar
import re
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice

from networkx.classes.reportviews import DiDegreeView

from ragent_core.data_sources.qas_from_graph.config import MAX_EXPANSION_TRIES, SPARQL_ENDPOINT, SPARQL_GRAPH
from ragent_core.data_sources.qas_from_graph.entity import EntityNodeData, EntityProcessor, EntityBuilder
from ragent_core.data_sources.qas_from_graph.sparql_client import SparqlClient
from logging import getLogger

from ragent_core.data_sources.qas_from_graph.sparql_generator import SparqlBuilder

logger = getLogger(__name__)

# -----------------------------
# GraphStore (networkx wrapper)
# -----------------------------

class GraphStore:
    def __init__(self, g = None):
        self.graph = g
        if g is None:
            self.graph = nx.DiGraph()

    def add_node(self, label: str, metadata: EntityNodeData):
        self.graph.add_node(label, metadata=metadata)

    def has_node(self, label: str) -> bool:
        return self.graph.has_node(label)

    def add_edge(self, src: str, dst: str, predicate_meta: EntityNodeData):
        self.graph.add_edge(src, dst, metadata=predicate_meta)

    def get_random_node(self) -> Tuple[str, Dict[str, Any]]:
        items = list(self.graph.nodes.items())
        return choice(items) if items else (None, None)

    def out_degree(self, node: str) -> DiDegreeView:
        return self.graph.out_degree(node)

    def in_degree(self, node: str) -> DiDegreeView:
        return self.graph.in_degree(node)

    def is_leaf(self, node: str) -> bool:
        return self.out_degree(node) + self.in_degree(node) == 1

    def get_node_item(self, node_name) -> Tuple[str, EntityNodeData]:
        attrs = nx.get_node_attributes(self.graph, "metadata")
        return node_name, attrs[node_name]

    def duplicate(self):
        return GraphStore(copy.deepcopy(self.graph))

    def copy_from(self, gs: "GraphStore"):
        self.graph = copy.deepcopy(gs.graph)

    def get_leafs(self):
        return [n for n in self.graph.nodes() if self.is_leaf(n)]
        
    def to_dict(self, light=False) -> Dict[str, Any]:
        """Convert the GraphStore to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the GraphStore.
        """
        # Get the basic graph structure using NetworkX's node_link_data
        graph_data = nx.node_link_data(self.graph)
        
        # Convert node metadata (EntityNodeData objects) to dictionaries
        for node in graph_data["nodes"]:
            if "metadata" in node:
                node["metadata"] = node["metadata"].to_dict(light=light)
        
        # Convert edge metadata (EntityNodeData objects) to dictionaries
        for link in graph_data["links"]:
            if "metadata" in link:
                link["metadata"] = link["metadata"].to_dict(light=light)
        
        return {
            "graph": graph_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphStore':
        graph_data = data.get("graph", {})
        
        for node in graph_data.get("nodes", []):
            if "metadata" in node and node["metadata"] is not None:
                node["metadata"] = EntityNodeData.from_dict(node["metadata"])
        
        for link in graph_data.get("links", []):
            if "metadata" in link and link["metadata"] is not None:
                link["metadata"] = EntityNodeData.from_dict(link["metadata"])
        
        graph = nx.node_link_graph(graph_data, directed=True)
        
        return cls(graph)


# -----------------------------
# GraphExpander using EntityRepository and caches
# -----------------------------

class GraphExpander:
    def __init__(self):
        self.sparql_builder = SparqlBuilder()
        self.entity_builder = EntityBuilder(SparqlClient())

    def build_random_seed(self, force=False) -> EntityNodeData:
        attempts = 0
        while True:
            attempts += 1
            logger.info(f"Gathering random entity --- Attempt: {attempts}")
            entity = self._gather_random_entity()
            ref = EntityProcessor.extract_ref(entity)
            node = self.entity_builder.build_entity(ref, raw_bindings=entity)
            if node.is_valid():
                return node
            if not force and attempts >= MAX_EXPANSION_TRIES:
                raise MaxRandomSearchReached()

    def _gather_random_entity(self) -> List[Dict[str, Any]]:
        q_total = "SELECT COUNT(?s) as ?total WHERE {?s ?p ?o}"
        total = int(self.entity_builder.client.run(q_total)[0]["total"]["value"])
        random_offset = choice(range(0, total))

        q = f"""SELECT ?s ?p ?o
        WHERE {{
          {{ SELECT ?s WHERE {{ ?s ?p ?o . }} OFFSET {random_offset} LIMIT 1 }}
          ?s ?p ?o .
        }}"""
        return self.entity_builder.client.run(q)

    def init_graph(self, graph_store: GraphStore, force: bool = False) -> None:
        root = self.build_random_seed(force=force)
        graph_store.add_node(root.label, root)
        logger.info(f"Initialized graph --- Root: {root.label}. Ref.: {root.ref}")

    @staticmethod
    def exclude_prop(prop: str) -> bool:
        # TODO: check if the literal value is actually a label from another node and discard: example:
        #  - "Antonio Román" (en) (<https://dbpedia.org/page/A_Dog_in_Space> dbp:director) == <https://dbpedia.org/page/Antonio_Rom%C3%A1n>
        exclude_literal_props = ["name", "label", "abstract", "page"]
        for sec in exclude_literal_props:
            if sec in prop:
                return True
        return False

    def expand_from_node(self, graph_store: GraphStore, node: str, include_literals: bool = True, copy: bool = False, exclude: List[EntityNodeData] = None) -> tuple[GraphStore, EntityNodeData] | None:
        if exclude is None:
            exclude =  []
        _graph_store = graph_store
        if copy:
            _graph_store = graph_store.duplicate()

        node_label, node_meta = _graph_store.get_node_item(node)

        triples = node_meta.entity_triples[:]
        random.shuffle(list(triples))  # pick randomly instead of ordered iteration
        for ref, p_uri, o in triples:
            is_uri = o.get("type") == "uri" and o.get("value", "").startswith("http://dbpedia.org")
            is_literal = include_literals and o.get("type") == "literal"

            if not (is_uri or is_literal) or self.exclude_prop(p_uri):
                continue

            target_ref = o["value"]

            # build entity differently depending on type
            if is_uri:
                new_node = self.entity_builder.build_entity(target_ref)
            else:  # literal
                new_node = EntityNodeData(
                    ref=target_ref,
                    label=target_ref,
                    entity_type_ref=None,
                    label_entity_type="Literal"
                )

            if new_node in exclude or _graph_store.has_node(new_node.label) or not new_node.is_valid():
                continue

            _graph_store.add_node(new_node.label, new_node)

            pred_meta = self.entity_builder.get_predicate_meta(p_uri)
            if pred_meta is None:
                pred_meta = EntityNodeData(ref=p_uri, label=None, entity_type_ref=None, label_entity_type=None)

            _graph_store.add_edge(node_label, new_node.label, pred_meta)
            logger.info(f"Added edge --- From: {node_label}. Pred {p_uri}. To: {new_node.label}")

            return _graph_store, new_node  # expansion succeeded
        return None

    # ------------------------------ Policies ------------------------------

    def random_expand(self, graph_store: GraphStore, times: int = 1, copy: bool = False) -> GraphStore:
        _graph_store = graph_store
        if copy:
            _graph_store = graph_store.duplicate()

        for _ in range(times):
            node_item, answer_attrs = _graph_store.get_random_node()
            self.expand_from_node(_graph_store, node_item)
        return _graph_store

    def width_expand_until_single_result(
        self,
        graph_store: GraphStore,
        node: str,
        width: int = 0,
        include_literals: bool = True
    ):
        if width < 0:
            raise ValueError("width must be >= 0")

        # Get all possible triples for expansion
        node_label, node_meta = graph_store.get_node_item(node)
        triples = node_meta.entity_triples[:]
        triples = [t for t in triples if not self.exclude_prop(t[1])]
        random.shuffle(triples)

        # Build candidate nodes from triples
        candidates = []
        for ref, p_uri, o in triples:
            is_uri = o.get("type") == "uri" and o.get("value", "").startswith("http://dbpedia.org")
            is_literal = include_literals and o.get("type") == "literal"

            if not (is_uri or is_literal):
                continue

            target_ref = o["value"]
            if is_uri:
                new_node = self.entity_builder.build_entity(target_ref)
            else:
                new_node = EntityNodeData(
                    ref=target_ref,
                    label=target_ref,
                    entity_type_ref=None,
                    label_entity_type="Literal"
                )

            if not new_node.is_valid() or graph_store.has_node(new_node.label):
                continue

            candidates.append((new_node, p_uri))

        if not candidates:
            return False
            # raise WidthExpansionFailed("No valid candidates to expand from node.")

        search_range = range(1, len(candidates) + 1) if width == 0 else range(1, width + 1)

        for k in search_range:
            for combo in combinations(candidates, k):
                temp_graph = graph_store.duplicate()
                added_nodes = []

                for new_node, p_uri in combo:
                    temp_graph.add_node(new_node.label, new_node)
                    pred_meta = self.entity_builder.get_predicate_meta(p_uri)
                    if pred_meta is None:
                        pred_meta = EntityNodeData(ref=p_uri, label=None, entity_type_ref=None, label_entity_type=None)
                    temp_graph.add_edge(node_label, new_node.label, pred_meta)
                    added_nodes.append(new_node)

                    # Evaluate SPARQL
                    query_str = self.sparql_builder.build_count_from_node(temp_graph, node)
                    total_answers = int(self.entity_builder.client.run(query_str)[0]["count"]["value"])

                    if total_answers == 1 and width > 0 and k < width:
                        break
                    if total_answers == 1:
                        graph_store.copy_from(temp_graph)
                        logger.info(f"Expansion succeeded with {len(added_nodes)} nodes: {[n.label for n in added_nodes]}")
                        return True

        return False
        # If we reach here, no combination produced exactly 1 answer
        # raise WidthExpansionFailed(
        #     f"No combination of {'minimal nodes' if width == 0 else width} nodes "
        #     "produces exactly 1 response."
        # )



    def _depth_expand(
        self,
        graph_store: GraphStore,
        node: str,
        depth: int = 3,
        include_literals: bool = True,
    ) -> bool:
        current_node = node
        current_depth = 0

        logger.info(f"Starting depth expansion from node '{node}' up to depth {depth}")
        visited_nodes = []

        _graph_store = graph_store.duplicate()

        while current_depth < depth:
            expand_result = self.expand_from_node(
                _graph_store,
                current_node,
                exclude=visited_nodes,
                include_literals=include_literals,
                copy=False,
            )

            if expand_result is None and current_node == node:
                raise LocalDepthExpansionFailed()
            elif expand_result is None:
                current_node = node
                current_depth = 0
                _graph_store = graph_store.duplicate()
                continue

            _, new_node = expand_result
            if current_node == node:
                visited_nodes.append(new_node)
            current_node = new_node.label
            current_depth += 1

        graph_store.copy_from(_graph_store)
        return True

    def depth_expand(
        self,
        graph_store: GraphStore,
        node: str,
        depth: int = 3,
        include_literals: bool = True,
    ) -> bool:
        success = False
        for leaf in graph_store.get_leafs():
            if leaf == node:
                continue
            try:
                success = self._depth_expand(graph_store, leaf, depth=depth, include_literals=include_literals)
                break
            except LocalDepthExpansionFailed as e:
                continue
        return success

# -----------------------------
# Visualization
# -----------------------------

class GraphVisualizer:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def draw(self, figsize: Tuple[int, int] = (8, 8)) -> None:
        g = self.graph_store.graph
        plt.figure(figsize=figsize)
        # pos = nx.kamada_kawai_layout(g)
        pos = nx.spring_layout(g)
        # pos = nx.bipartite_layout(g)
        # pos = nx.circular_layout(g)
        # pos = nx.shell_layout(g)
        # pos = nx.planar_layout(g)


        labels = {}
        for node, data in g.nodes(data=True):
            md = data.get("metadata")
            if isinstance(md, EntityNodeData):
                labels[node] = f"[{md.label_entity_type}]\n\n{node}"
            else:
                labels[node] = node

        nx.draw(
            g,
            pos,
            with_labels=True,
            labels=labels,
            node_size=1000,
            node_color="lightblue",
            font_size=7,
            font_weight="bold",
            arrows=True,
        )

        edge_labels = {}
        for u, v, d in g.edges(data=True):
            if u == v:
                continue
            md = d.get("metadata")
            if isinstance(md, EntityNodeData) and md.label:
                edge_labels[(u, v)] = md.label
            elif isinstance(md, EntityNodeData) and md.ref:
                frag = md.ref.rsplit("/", 1)[-1].split("#")[-1]
                frag = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", frag)
                edge_labels[(u, v)] = frag.replace("_", " ")

        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red')
        plt.axis('off')
        plt.show()

# ----------------------- Exceptions ----------------------
class MaxRandomSearchReached(Exception):
    pass

class WidthExpansionFailed(Exception):
    pass

class LocalDepthExpansionFailed(Exception):
    pass

class TooFewNodesSuffice(Exception):
    pass