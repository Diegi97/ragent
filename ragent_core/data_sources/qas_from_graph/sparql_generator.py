from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ragent_core.data_sources.qas_from_graph.entity import EntityNodeData
from ragent_core.data_sources.qas_from_graph.path_collector import PathCollector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ragent_core.data_sources.qas_from_graph.graph import GraphStore

# -----------------------------
# Declarative SPARQL AST + Renderer
# -----------------------------

@dataclass
class TriplePattern:
    s: str
    p: str
    o: str
    optional: bool = False

@dataclass
class OptionalBlock:
    patterns: List[Union['TriplePattern', 'OptionalBlock']]

@dataclass
class UnionBlock:
    branches: List[List[Union['TriplePattern', OptionalBlock]]]

class SparqlQueryAST:
    def __init__(self):
        self.patterns: List[Union[TriplePattern, OptionalBlock, UnionBlock]] = []
        self.prefixes: Dict[str, str] = {
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }
        self.select_vars: List[str] = ['?ans', '?ansLabel']
        self.limit: Optional[int] = None

    def add(self, pat: Union[TriplePattern, OptionalBlock, UnionBlock]):
        self.patterns.append(pat)

    def set_limit(self, limit: int):
        self.limit = limit

    def render(self) -> str:
        lines: List[str] = []
        for pfx, uri in self.prefixes.items():
            lines.append(f"PREFIX {pfx}: <{uri}>")
        lines.append(f"SELECT DISTINCT {' '.join(self.select_vars)} WHERE {{")

        def render_pat(p):
            if isinstance(p, TriplePattern):
                return f"  {p.s} {p.p} {p.o} ."
            elif isinstance(p, OptionalBlock):
                inner = "\n".join(render_pat(x) for x in p.patterns)
                return f"  OPTIONAL {{\n{inner}\n  }}"
            elif isinstance(p, UnionBlock):
                branches_str = []
                for branch in p.branches:
                    b = "\n".join(render_pat(x) for x in branch)
                    branches_str.append(f"{{\n{b}\n }}")
                return "  UNION\n".join(branches_str)
            else:
                return ""

        for pat in self.patterns:
            lines.append(render_pat(pat))

        lines.append('}')
        if self.limit is not None:
            lines.append(f"LIMIT {self.limit}")
        return "\n".join(lines)

# -----------------------------
# SparqlGenerator using AST
# -----------------------------

class SparqlBuilder:
    def __init__(self):
        self.collector = PathCollector()

    @staticmethod
    def _escape_literal(s: Optional[str]) -> str:
        if s is None:
            return ""
        return s.replace('\\', '\\\\').replace('"', '\\"')

    def _pred_uri_from_edata(self, edata: dict) -> Optional[str]:
        md = edata.get("metadata") if isinstance(edata, dict) else None
        if isinstance(md, EntityNodeData) and md.ref:
            return md.ref
        return None

    def _type_uri_from_meta(self, meta: EntityNodeData) -> Optional[str]:
        return meta.entity_type_ref if isinstance(meta, EntityNodeData) else None

    def _build_ast_for_node(self, graph_store: GraphStore, node: str, limit: Optional[int] = None) -> SparqlQueryAST:
        """
        Internal helper to build the SPARQL AST from a node.
        Handles both URI and literal neighbors.
        """
        paths = self.collector.collect_paths(graph_store, node)

        ast = SparqlQueryAST()
        if limit is not None:
            ast.set_limit(limit)

        var_counter = 0
        var_map: Dict[Tuple[int, str], str] = {}

        for p_idx, path in enumerate(paths):
            cur_var = '?ans'
            branch_patterns: List[TriplePattern] = []

            for phrase, node_label, direction, edata in path:
                key = (p_idx, node_label)
                if key not in var_map:
                    var_counter += 1
                    var_map[key] = f"?v{var_counter}"
                neighbor_var = var_map[key]

                pred_uri = self._pred_uri_from_edata(edata)

                # --- decide if neighbor is literal or uri ---
                meta = graph_store.graph.nodes[node_label].get('metadata')
                is_literal = isinstance(meta, EntityNodeData) and meta.label_entity_type == "Literal"

                if pred_uri:
                    if is_literal:
                        # direct literal binding instead of variable
                        lit_val = self._escape_literal(meta.label or meta.ref)
                        if direction == 'forward':
                            branch_patterns.append(TriplePattern(cur_var, f"<{pred_uri}>", f"\"{lit_val}\"@en"))
                        else:
                            branch_patterns.append(TriplePattern(f"\"{lit_val}\"@en", f"<{pred_uri}>", cur_var))
                    else:
                        # normal URI object
                        if direction == 'forward':
                            branch_patterns.append(TriplePattern(cur_var, f"<{pred_uri}>", neighbor_var))
                        else:
                            branch_patterns.append(TriplePattern(neighbor_var, f"<{pred_uri}>", cur_var))
                else:
                    # edge has no predicate URI, fallback to label-based representation
                    edge_label = phrase
                    if edge_label.startswith('object of '):
                        edge_label = edge_label[len('object of '):]
                    var_counter += 1
                    pvar = f"?p{var_counter}"
                    if direction == 'forward':
                        branch_patterns.append(TriplePattern(cur_var, pvar, neighbor_var))
                    else:
                        branch_patterns.append(TriplePattern(neighbor_var, pvar, cur_var))
                    edge_label_esc = self._escape_literal(edge_label)
                    branch_patterns.append(TriplePattern(pvar, 'rdfs:label', f'"{edge_label_esc}"@en'))

                # --- type or label triples (only for non-literals) ---
                if not is_literal:
                    if graph_store.is_leaf(node_label):
                        if isinstance(meta, EntityNodeData) and meta.label:
                            lbl = self._escape_literal(meta.label)
                            branch_patterns.append(TriplePattern(neighbor_var, 'rdfs:label', f'"{lbl}"@en'))
                    else:
                        type_uri = self._type_uri_from_meta(meta)
                        if type_uri:
                            branch_patterns.append(TriplePattern(neighbor_var, 'a', f"<{type_uri}>"))
                        elif isinstance(meta, EntityNodeData) and meta.label_entity_type:
                            type_var = f"?t{var_counter+1}"
                            branch_patterns.append(TriplePattern(neighbor_var, 'a', type_var))
                            lbl = self._escape_literal(meta.label_entity_type)
                            branch_patterns.append(TriplePattern(type_var, 'rdfs:label', f'"{lbl}"@en'))

                cur_var = neighbor_var if not is_literal else cur_var

            # merge branch patterns into AST
            for bp in branch_patterns:
                ast.add(bp)

        # answer type restriction (only if not literal)
        ans_meta: EntityNodeData = graph_store.graph.nodes[node].get('metadata')
        if isinstance(ans_meta, EntityNodeData) and ans_meta.label_entity_type != "Literal":
            ans_type_uri = self._type_uri_from_meta(ans_meta)
            if ans_type_uri:
                ast.add(TriplePattern('?ans', 'a', f"<{ans_type_uri}>"))

        return ast

    def build_from_node(self, graph_store: GraphStore, node: str, limit: int = 50) -> str:
        ast = self._build_ast_for_node(graph_store, node, limit=limit)

        # Add labels and lang filter
        ast.add(TriplePattern('?ans', 'rdfs:label', '?ansLabel'))
        ast.add(TriplePattern('FILTER(LANGMATCHES(LANG(?ansLabel), "EN"))', '', ''))

        return ast.render()

    def build_count_from_node(self, graph_store: GraphStore, node: str) -> str:
        ast = self._build_ast_for_node(graph_store, node, limit=None)

        # Replace select_vars with COUNT version
        ast.select_vars = ["(COUNT(DISTINCT ?ans) as ?count)"]

        # No label/language filters needed for count
        return ast.render()
