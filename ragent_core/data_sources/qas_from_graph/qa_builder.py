from typing import List, Optional, Tuple, Set, Union

from ragent_core.data_sources.qas_from_graph.entity import EntityNodeData
from ragent_core.data_sources.qas_from_graph.graph import GraphStore
from ragent_core.data_sources.qas_from_graph.path_collector import PathCollector


class QABuilder:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.collector = PathCollector(graph_store)

    @staticmethod
    def _indefinite(word: Optional[str]) -> str:
        return 'an' if word and word[0].lower() in 'aeiou' else 'a'

    def _get_node_type_label(self, node: str) -> str:
        meta: EntityNodeData = self.graph_store.graph.nodes[node].get('metadata')
        return meta.label_entity_type if isinstance(meta, EntityNodeData) and meta.label_entity_type else 'Thing'

    def _path_to_fragment(self, path: List[Tuple[str, str, str, dict]]) -> str:
        parts: List[str] = []
        for phrase, node, direction, edata in path:
            if self.graph_store.is_leaf(node):
                meta: EntityNodeData = self.graph_store.graph.nodes[node].get('metadata')
                node_label = meta.label if isinstance(meta, EntityNodeData) and meta.label else node
                parts.append(f"{phrase} {node_label}")
            else:
                node_type = self._get_node_type_label(node)
                article = self._indefinite(node_type)
                parts.append(f"{phrase} {article} {node_type}")
        return ' '.join(parts).strip()

    def build_from_node(self, answer_node: str) -> Tuple[str, str]:
        if answer_node not in self.graph_store.graph.nodes:
            raise ValueError('Node not in graph')
        answer_type = self._get_node_type_label(answer_node)
        paths = self.collector.collect_paths(answer_node)
        fragments = [self._path_to_fragment(p) for p in paths if p]
        fragments = [f for f in fragments if f.strip()]
        if fragments:
            question = f"What {answer_type} " + " and ".join(fragments) + "?"
        else:
            question = f"What {answer_type} ?"
        return question, answer_node
