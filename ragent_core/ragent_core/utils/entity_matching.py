import re
import unicodedata
from typing import Dict, List, Tuple

from rapidfuzz import fuzz, process


def match_entities_in_text(
    text: str,
    entities: List[str],
    max_entities: int = 50,
    max_ngram: int = 5,
    min_fuzzy_score_single: int = 97,
    min_fuzzy_score_multi: int = 90,
) -> List[str]:
    """
    Return entities that appear in `text` using:
      1. exact normalized matching
      2. fuzzy matching fallback if exact matches < max_entities

    Parameters
    ----------
    text : str
        Paragraph/text to search in.
    entities : List[str]
        List of entity strings.
    max_entities : int
        Max number of entities to return.
    max_ngram : int
        Max token span length used in fuzzy fallback.
    min_fuzzy_score_single : int
        Minimum fuzzy score for 1-token spans/entities.
    min_fuzzy_score_multi : int
        Minimum fuzzy score for multi-token spans/entities.

    Returns
    -------
    List[str]
        Matched entities. Exact matches come first, then fuzzy matches.
    """

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKC", s).lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def token_count(s: str) -> int:
        if not s:
            return 0
        return len(s.split())

    def exact_in_text(text_norm: str, entity_norm: str) -> bool:
        # Word-boundary-ish exact matching on normalized text
        # Works well after punctuation normalization.
        pattern = rf"(?<!\w){re.escape(entity_norm)}(?!\w)"
        return re.search(pattern, text_norm) is not None

    def build_candidate_spans(text_norm: str, max_n: int) -> List[str]:
        tokens = text_norm.split()
        spans = set()

        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens), i + max_n) + 1):
                span = " ".join(tokens[i:j])
                # Skip very short/noisy spans unless they are meaningful
                if len(span) >= 3:
                    spans.add(span)

        return list(spans)

    # Deduplicate entities while preserving first occurrence order
    seen_raw = set()
    entities_unique = []
    for e in entities:
        if e and e not in seen_raw:
            seen_raw.add(e)
            entities_unique.append(e)

    if not text or not entities_unique:
        return []

    text_norm = normalize(text)
    if not text_norm:
        return []

    # Precompute normalized entities
    # If multiple original entities normalize to the same string,
    # keep the first one.
    norm_to_entity: Dict[str, str] = {}
    entity_token_counts: Dict[str, int] = {}

    for entity in entities_unique:
        entity_norm = normalize(entity)
        if entity_norm and entity_norm not in norm_to_entity:
            norm_to_entity[entity_norm] = entity
            entity_token_counts[entity_norm] = token_count(entity_norm)

    normalized_entities = list(norm_to_entity.keys())

    # ------------------------------------------------------------------
    # 1) Exact normalized matching
    # ------------------------------------------------------------------
    exact_matches: List[str] = []
    matched_entities = set()

    # Longer entities first helps avoid noisy short matches dominating
    for entity_norm in sorted(normalized_entities, key=lambda x: (-len(x), x)):
        original_entity = norm_to_entity[entity_norm]
        if exact_in_text(text_norm, entity_norm):
            exact_matches.append(original_entity)
            matched_entities.add(original_entity)

    if len(exact_matches) >= max_entities:
        return exact_matches[:max_entities]

    # ------------------------------------------------------------------
    # 2) Fuzzy fallback
    #    Only if exact matches are fewer than max_entities
    # ------------------------------------------------------------------
    candidate_spans = build_candidate_spans(text_norm, max_ngram)
    if not candidate_spans:
        return exact_matches

    # Group normalized entities by token count to reduce bad comparisons
    entities_by_tokens: Dict[int, List[str]] = {}
    for entity_norm in normalized_entities:
        n = entity_token_counts[entity_norm]
        entities_by_tokens.setdefault(n, []).append(entity_norm)

    fuzzy_candidates: List[Tuple[str, float]] = []
    already_matched_norms = {normalize(e) for e in exact_matches}

    for span in candidate_spans:
        span_tokens = token_count(span)

        # Compare only against entities with similar token length
        possible_lengths = {span_tokens - 1, span_tokens, span_tokens + 1}
        search_space = []
        for n in possible_lengths:
            if n in entities_by_tokens:
                search_space.extend(entities_by_tokens[n])

        if not search_space:
            continue

        score_cutoff = (
            min_fuzzy_score_single if span_tokens == 1 else min_fuzzy_score_multi
        )

        # WRatio is a good general-purpose scorer
        match = process.extractOne(
            span,
            search_space,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )

        if not match:
            continue

        matched_norm, score, _ = match
        if matched_norm in already_matched_norms:
            continue

        original_entity = norm_to_entity[matched_norm]
        if original_entity in matched_entities:
            continue

        fuzzy_candidates.append((original_entity, score))

    # Keep best fuzzy score per entity
    best_fuzzy_score: Dict[str, float] = {}
    for entity, score in fuzzy_candidates:
        if entity not in best_fuzzy_score or score > best_fuzzy_score[entity]:
            best_fuzzy_score[entity] = score

    fuzzy_matches_sorted = [
        entity
        for entity, _ in sorted(best_fuzzy_score.items(), key=lambda x: (-x[1], x[0]))
    ]

    result = exact_matches[:]
    for entity in fuzzy_matches_sorted:
        if entity not in matched_entities:
            result.append(entity)
            matched_entities.add(entity)
        if len(result) >= max_entities:
            break

    return result
