from __future__ import annotations
from typing import List, Tuple, Set
import re

from ragent_core.data_sources.qas_from_graph.entity import EntityNodeData

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ragent_core.data_sources.qas_from_graph.graph import GraphStore



# -----------------------------
# PathCollector & QABuilder remain mostly the same but use repo/store
# -----------------------------

class PathCollector:
    def __init__(self, max_depth: int = 50, max_paths: int = 500):
        self.max_depth = max_depth
        self.max_paths = max_paths

    def _get_edge_label(self, edata: dict) -> str:
        md = edata.get('metadata') if isinstance(edata, dict) else None
        if isinstance(md, EntityNodeData) and md.label:
            return md.label.strip()
        if isinstance(md, EntityNodeData) and md.ref:
            frag = md.ref.rsplit('/', 1)[-1].split('#')[-1]
            frag = re.sub(r'([a-z0-9])([A-Z])', r'\\1 \\2', frag)
            frag = frag.replace('_', ' ')
            return frag
        return 'related to'

    def collect_paths(self, graph_store: GraphStore, start: str) -> List[List[Tuple[str, str, str, dict]]]:
        paths: List[List[Tuple[str, str, str, dict]]] = []

        def dfs(current: str, visited: Set[str], cur_path: List[Tuple[str, str, str, dict]], depth: int):
            if depth > self.max_depth or len(paths) >= self.max_paths:
                return

            traversals = [
                (graph_store.graph.out_edges(current, data=True), lambda ed, node: (self._get_edge_label(ed), node, 'forward', ed)),
                (graph_store.graph.in_edges(current, data=True), lambda ed, node: (f"object of {self._get_edge_label(ed)}", node, 'backward', ed)),
            ]

            for edges, step_builder in traversals:
                for src, tgt, edata in edges:
                    neighbor = tgt if src == current else src
                    if neighbor in visited:
                        continue

                    step = step_builder(edata, neighbor)
                    new_path = cur_path + [step]

                    if graph_store.is_leaf(neighbor):
                        paths.append(new_path)
                    else:
                        visited.add(neighbor)
                        dfs(neighbor, visited, new_path, depth + 1)
                        visited.remove(neighbor)

        dfs(start, {start}, [], 0)
        return paths