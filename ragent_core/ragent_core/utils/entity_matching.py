import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = _NON_WORD_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _maximum_ratio_for_lengths(left: int, right: int) -> float:
    """Maximum fuzz.ratio score possible for strings of these lengths."""
    if left <= 0 or right <= 0:
        return 0.0
    return 200.0 * min(left, right) / (left + right)


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    terminal: str | None = None


class EntityMatcher:
    """Reusable normalized exact matcher with a restricted fuzzy fallback.

    Entity normalization and lookup indexes are built once. Exact matching uses a
    token trie, so matching a passage does not scan or compile a regular expression
    for every known entity.

    Fuzzy matching is deliberately restricted to plausible candidates:

    * single-token entities are compared only with complete passage tokens;
    * multi-token entities must have an approximately matching token in the passage
      before their full name is compared with the passage.

    This retains typo tolerance without comparing every passage n-gram with every
    entity or filling the result with unrelated best-effort fuzzy matches.
    """

    def __init__(self, entities: Iterable[str]) -> None:
        norm_to_entity: dict[str, str] = {}
        for entity in entities:
            if not entity:
                continue
            normalized = _normalize_text(entity)
            if normalized and normalized not in norm_to_entity:
                norm_to_entity[normalized] = entity

        self._normalized_entities = tuple(norm_to_entity)
        self._entities_by_normalized = norm_to_entity
        self.entities = tuple(norm_to_entity.values())
        self._normalized_index = {
            normalized: index
            for index, normalized in enumerate(self._normalized_entities)
        }
        self._token_counts = tuple(
            len(normalized.split()) for normalized in self._normalized_entities
        )
        self._exact_rank = {
            normalized: rank
            for rank, normalized in enumerate(
                sorted(
                    self._normalized_entities, key=lambda value: (-len(value), value)
                )
            )
        }

        self._trie = _TrieNode()
        self._max_entity_tokens = 0
        single_entities_by_length: dict[int, list[str]] = defaultdict(list)
        multi_entity_indices_by_token: dict[str, list[int]] = defaultdict(list)

        for index, normalized in enumerate(self._normalized_entities):
            tokens = normalized.split()
            self._max_entity_tokens = max(self._max_entity_tokens, len(tokens))
            node = self._trie
            for token in tokens:
                node = node.children.setdefault(token, _TrieNode())
            node.terminal = normalized

            if len(tokens) == 1:
                single_entities_by_length[len(normalized)].append(normalized)
            else:
                for token in set(tokens):
                    multi_entity_indices_by_token[token].append(index)

        self._single_entities_by_length = {
            length: tuple(values)
            for length, values in single_entities_by_length.items()
        }
        self._multi_entity_indices_by_token = {
            token: tuple(indices)
            for token, indices in multi_entity_indices_by_token.items()
        }
        self._multi_entity_tokens_by_length: dict[int, tuple[str, ...]] = {}
        tokens_by_length: dict[int, list[str]] = defaultdict(list)
        for token in self._multi_entity_indices_by_token:
            tokens_by_length[len(token)].append(token)
        self._multi_entity_tokens_by_length = {
            length: tuple(values) for length, values in tokens_by_length.items()
        }

    def __len__(self) -> int:
        return len(self._normalized_entities)

    def _find_exact(self, tokens: list[str]) -> list[str]:
        matched: set[str] = set()
        for start in range(len(tokens)):
            node = self._trie
            stop = min(len(tokens), start + self._max_entity_tokens)
            for index in range(start, stop):
                node = node.children.get(tokens[index])
                if node is None:
                    break
                if node.terminal is not None:
                    matched.add(node.terminal)
        return sorted(matched, key=self._exact_rank.__getitem__)

    @staticmethod
    def _possible_length_choices(
        choices_by_length: dict[int, tuple[str, ...]],
        query_length: int,
        score_cutoff: int,
    ) -> list[str]:
        choices: list[str] = []
        for choice_length, values in choices_by_length.items():
            if _maximum_ratio_for_lengths(query_length, choice_length) >= score_cutoff:
                choices.extend(values)
        return choices

    def _find_fuzzy(
        self,
        normalized_text: str,
        tokens: list[str],
        exact: set[str],
        max_ngram: int,
        min_fuzzy_score_single: int,
        min_fuzzy_score_multi: int,
    ) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        unique_tokens = {token for token in tokens if len(token) >= 3}

        # A single-token entity must resemble a complete token. This avoids the
        # partial-substring false positives produced by comparing short names with
        # arbitrary multi-token spans.
        for token in unique_tokens:
            choices = self._possible_length_choices(
                self._single_entities_by_length,
                len(token),
                min_fuzzy_score_single,
            )
            for normalized, score, _ in process.extract(
                token,
                choices,
                scorer=fuzz.ratio,
                score_cutoff=min_fuzzy_score_single,
                limit=None,
            ):
                if normalized in exact:
                    continue
                entity_index = self._normalized_index[normalized]
                scores[entity_index] = max(scores.get(entity_index, 0.0), score)

        # Build the multi-token shortlist through a precomputed token index. Token
        # matching is fuzzy so a candidate can still be found when a name contains
        # a typo, but the expensive full-name comparison only sees the shortlist.
        candidate_indices: set[int] = set()
        for token in unique_tokens:
            token_choices = self._possible_length_choices(
                self._multi_entity_tokens_by_length,
                len(token),
                min_fuzzy_score_multi,
            )
            for entity_token, _, _ in process.extract(
                token,
                token_choices,
                scorer=fuzz.ratio,
                score_cutoff=min_fuzzy_score_multi,
                limit=None,
            ):
                candidate_indices.update(
                    self._multi_entity_indices_by_token[entity_token]
                )

        max_fuzzy_tokens = max_ngram + 1
        candidate_indices = {
            index
            for index in candidate_indices
            if self._token_counts[index] <= max_fuzzy_tokens
            and self._normalized_entities[index] not in exact
        }
        candidate_names = [
            self._normalized_entities[index] for index in sorted(candidate_indices)
        ]
        for normalized, score, _ in process.extract(
            normalized_text,
            candidate_names,
            scorer=fuzz.partial_ratio,
            score_cutoff=min_fuzzy_score_multi,
            limit=None,
        ):
            entity_index = self._normalized_index[normalized]
            scores[entity_index] = max(scores.get(entity_index, 0.0), score)

        return sorted(
            (
                (self._entities_by_normalized[self._normalized_entities[index]], score)
                for index, score in scores.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )

    def match(
        self,
        text: str,
        max_entities: int = 50,
        max_ngram: int = 5,
        min_fuzzy_score_single: int = 97,
        min_fuzzy_score_multi: int = 90,
    ) -> list[str]:
        """Return normalized exact and high-confidence fuzzy entity matches."""
        if not text or not self._normalized_entities or max_entities <= 0:
            return []

        normalized_text = _normalize_text(text)
        if not normalized_text:
            return []
        tokens = normalized_text.split()
        exact_normalized = self._find_exact(tokens)
        exact_entities = [
            self._entities_by_normalized[normalized]
            for normalized in exact_normalized[:max_entities]
        ]
        if len(exact_entities) >= max_entities:
            return exact_entities

        fuzzy = self._find_fuzzy(
            normalized_text,
            tokens,
            set(exact_normalized),
            max_ngram,
            min_fuzzy_score_single,
            min_fuzzy_score_multi,
        )
        matched = set(exact_entities)
        result = list(exact_entities)
        for entity, _ in fuzzy:
            if entity in matched:
                continue
            result.append(entity)
            matched.add(entity)
            if len(result) >= max_entities:
                break
        return result


def match_entities_in_text(
    text: str,
    entities: list[str],
    max_entities: int = 50,
    max_ngram: int = 5,
    min_fuzzy_score_single: int = 97,
    min_fuzzy_score_multi: int = 90,
) -> list[str]:
    """Match entities in one text.

    Repeated callers should construct :class:`EntityMatcher` once and call
    :meth:`EntityMatcher.match` to reuse the normalized trie and fuzzy indexes.
    """
    return EntityMatcher(entities).match(
        text,
        max_entities=max_entities,
        max_ngram=max_ngram,
        min_fuzzy_score_single=min_fuzzy_score_single,
        min_fuzzy_score_multi=min_fuzzy_score_multi,
    )
