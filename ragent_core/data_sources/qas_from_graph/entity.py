from dataclasses import dataclass, field
from functools import cache
from typing import Tuple

from typing import Any, Dict, List, Optional

from ragent_core.data_sources.qas_from_graph.sparql_client import SparqlClient


# -----------------------------
# Entity extraction utilities
# -----------------------------

class EntityProcessor:
    LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    DBPEDIA_ONTOLOGY_PREFIX = "http://dbpedia.org/ontology/"

    @staticmethod
    def extract_label(bindings: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not bindings:
            return None
        for item in bindings:
            p = item.get("p", {}).get("value")
            o = item.get("o", {})
            if p == EntityProcessor.LABEL_PREDICATE and o.get("xml:lang") == "en":
                return o.get("value")
        return None

    @staticmethod
    def extract_ref(bindings: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if bindings and isinstance(bindings, list) and "s" in bindings[0]:
            return bindings[0]["s"]["value"]
        return None

    @staticmethod
    def find_first_ontology_type(bindings: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not bindings:
            return None
        for item in bindings:
            p = item.get("p", {}).get("value")
            o = item.get("o", {})
            if p == EntityProcessor.RDF_TYPE:
                val = o.get("value")
                if val and val.startswith(EntityProcessor.DBPEDIA_ONTOLOGY_PREFIX):
                    return val
        return None

# -----------------------------
# Domain: EntityNodeData (pure data holder)
# -----------------------------

@dataclass(frozen=True)
class EntityNodeData:
    """Pure data holder representing an entity/resource fetched from SPARQL.

    Does NOT perform network calls. All enrichment should be done in a repository.
    
    This class can be serialized to and deserialized from a dictionary format using
    the to_dict() and from_dict() methods.
    """

    ref: Optional[str]
    label: Optional[str]
    entity_type_ref: Optional[str]
    label_entity_type: Optional[str]
    entity_triples: Tuple[Tuple[str, str, Dict[str, Any]], ...] = field(default_factory=tuple)

    def is_valid(self) -> bool:
        return self.label is not None and self.label_entity_type is not None
        
    def to_dict(self, light=False) -> Dict[str, Any]:
        """Convert the EntityNodeData to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the EntityNodeData.
        """
        # Convert entity_triples to a serializable format
        serialized_triples = []
        for ref, pred, obj in self.entity_triples:
            serialized_triples.append({
                "ref": ref,
                "pred": pred,
                "obj": obj
            })
            
        return {
            "ref": self.ref,
            "label": self.label,
            "entity_type_ref": self.entity_type_ref,
            "label_entity_type": self.label_entity_type,
            "entity_triples": [] if light else serialized_triples,
            "light": light
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityNodeData':
        """Create an EntityNodeData from a dictionary.
        
        Args:
            data: A dictionary representation of an EntityNodeData.
            
        Returns:
            A new EntityNodeData instance.
        """
        # Convert serialized triples back to the original format
        triples = []
        # TODO: check if light and then re-request triples
        for triple_dict in data.get("entity_triples", []):
            triples.append((
                triple_dict["ref"],
                triple_dict["pred"],
                triple_dict["obj"]
            ))
            
        return cls(
            ref=data.get("ref"),
            label=data.get("label"),
            entity_type_ref=data.get("entity_type_ref"),
            label_entity_type=data.get("label_entity_type"),
            entity_triples=tuple(triples)
        )


# -----------------------------
# EntityRepository: infrastructure -> domain mapping + caches
# -----------------------------

class EntityBuilder:
    """Responsible for fetching triples from SPARQL and constructing EntityNodeData.

    Caches fetched triples, resolved type labels and predicate metadata.
    """

    def __init__(self, client: SparqlClient):
        self.client = client

    def get_type_label(self, type_ref: str) -> Optional[str]:
        rows = self.client.fetch_triples(type_ref)
        label = EntityProcessor.extract_label(rows)
        return label

    def get_predicate_meta(self, pred_ref: str) -> Optional[EntityNodeData]:
        rows = self.client.fetch_triples(pred_ref)
        label = EntityProcessor.extract_label(rows)
        pred = EntityNodeData(ref=pred_ref, label=label, entity_type_ref=None, label_entity_type=None,
                              entity_triples=tuple((pred_ref, "p", r) for r in rows))
        return pred

    def build_entity(self, ref: Optional[str], raw_bindings: Optional[List[Dict[str, Any]]] = None) -> EntityNodeData:
        """Construct an EntityNodeData for a subject ref.

        If raw_bindings are provided, they will be used; otherwise triples will be fetched.
        """
        if not ref and raw_bindings:
            ref = EntityProcessor.extract_ref(raw_bindings)

        if not ref:
            raise ValueError("ref required to build entity")

        rows = raw_bindings if raw_bindings is not None else self.client.fetch_triples(ref)

        label = EntityProcessor.extract_label(rows)
        entity_type_ref = EntityProcessor.find_first_ontology_type(rows)
        label_entity_type = self.get_type_label(entity_type_ref) if entity_type_ref else None

        # capture triples as an immutable tuple for the dataclass
        triples_t = tuple((ref, item.get("p", {}).get("value"), item.get("o", {})) for item in rows)

        node = EntityNodeData(ref=ref, label=label, entity_type_ref=entity_type_ref,
                              label_entity_type=label_entity_type, entity_triples=triples_t)
        return node
