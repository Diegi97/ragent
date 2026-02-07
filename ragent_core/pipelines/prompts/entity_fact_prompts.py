from __future__ import annotations

import re
from dataclasses import dataclass

from ragent_core.types import QA


@dataclass(frozen=True)
class ExtractedFact:
    statement: str
    doc_ids: list[int]
    fact_id: int = 0


FACT_EXTRACTION_PROMPT = """You will be extracting factual information about a specific target entity from a set of source documents. Your goal is to identify distinct, explicitly stated facts that are relevant to the target entity.

Here are the source documents you will be working with:

<documents>
{DOCUMENTS}
</documents>

Here is the target entity you should focus on:

<target_entity>
{ENTITY}
</target_entity>

Your task is to extract facts about the target entity following these rules:

1. **Explicit statements only**: Extract only facts that are directly and explicitly stated in the documents. Do not infer, interpret, or use outside knowledge.

2. **Relevance**: Only extract facts that are directly relevant to the target entity.

3. **Standalone statements**: Each fact must be a short, complete statement that can be understood on its own without additional context.

4. **Document attribution**: Every fact must list ALL document IDs where it appears or is supported. Use the document identifiers provided in the source documents.

5. **No duplication**: If the same fact appears in multiple documents, include it only once and list all supporting document IDs in the `<doc_ids>` tag.

6. **Handle disagreements**: If documents contain conflicting information about the same aspect of the target entity, choose the fact that appears better supported based on:
   - Specificity and detail
   - Consistency with other evidence in the documents
   - Temporal recency (if dates are available)

7. **No invention**: Do not create facts or combine information in ways not explicitly stated in the source material.

Provide your final answer in the following XML format:

<facts>
  <fact>
    <statement>fact statement</statement>
    <doc_ids>1,2,3</doc_ids>
  </fact>
</facts>

If no relevant facts can be extracted from the documents about the target entity, return an empty <facts> tag.

Your final output should contain only the facts tags with the extracted information. Do not include your thinking process in the final answer."""


FACT_REFINEMENT_PROMPT = """You will be extracting NEW factual information about a specific target entity from a new batch of source documents. A set of facts has already been extracted from previously processed documents. Your goal is to identify only facts that are not yet covered by the existing list.

Here is the target entity you should focus on:

<target_entity>
{ENTITY}
</target_entity>

Here are the facts that have already been extracted from previously processed documents:

<current_facts>
{CURRENT_FACTS}
</current_facts>

Here are the new source documents to process:

<additional_documents>
{ADDITIONAL_DOCUMENTS}
</additional_documents>

Your task is to extract ONLY NEW facts from the additional documents following these rules:

1. **New facts only**: Do NOT repeat or rewrite facts that are already in the current fact list. Only output facts that provide genuinely new information not covered by any existing entry.

2. **Duplicate detection**: A fact from the additional documents is considered a duplicate if it is semantically equivalent to an existing fact, even if worded differently. Skip it entirely — do not include it in your output.

3. **Additional support for existing facts**: If an additional document provides further evidence for an already-listed fact, output that fact's `fact_id` along with ONLY the new document IDs. Do not repeat the document IDs already listed in the current facts. The system will merge the IDs automatically.

4. **Correcting an existing fact**: If an additional document contradicts an existing fact and the new information is clearly better supported (based on specificity, consistency, or recency), output the corrected statement using the SAME `fact_id` as the incorrect fact. The system will replace the old statement. Only do this when the correction is well-evidenced — do not speculatively override existing facts.

5. **Explicit statements only**: Extract only facts that are directly and explicitly stated in the additional documents. Do not infer, interpret, or use outside knowledge.

6. **Relevance**: Only extract facts that are directly relevant to the target entity.

7. **Standalone statements**: Each fact must be a short, complete statement that can be understood on its own without additional context. Do not use pronouns that require the reader to look up another fact.

8. **No invention**: Do not create facts, infer beyond what is explicitly stated, or combine information in ways not directly supported by the source material.

Provide your final answer in the following XML format:

<facts>
  <fact>
    <fact_id>existing id OR leave empty for new facts</fact_id>
    <statement>fact statement</statement>
    <doc_ids>1,2,3</doc_ids>
  </fact>
</facts>

If no new facts can be extracted from the additional documents, return an empty <facts> tag.

Your final output should contain only the facts tags with the new information. Do not include your thinking process in the final answer."""


FACT_TO_QA_PROMPT = """You are generating a question-answer pair designed to train an LLM agent that autonomously navigates, retrieves, and synthesizes information from multiple interdependent sources. The QA pair must require agentic, multi-step search to resolve — it should NOT be answerable from a single document or through simple lookup.

Target entity:
<entity>{ENTITY}</entity>

Fact bank (each fact is grounded in specific source documents):
<facts>
{FACTS}
</facts>

Generation parameters:
<complexity_target>{COMPLEXITY_TARGET}</complexity_target>

---

## Question Design Principles

Your question MUST satisfy these core requirements:

### 1. Multi-Hop Reasoning
The question must require 2 or more sequential reasoning steps, where each step builds on information discovered in a previous step. A searcher should need to:
- Find an initial piece of information from one source
- Use that result to formulate a follow-up query to a different source
- Synthesize findings across sources to produce the final answer

### 2. Complex Query Decomposition
The question should be naturally decomposable into atomic sub-queries where:
- Each sub-query addresses a distinct information need
- Sub-queries are **interdependent** (answering one requires the result of another)
- The sub-queries form a reasoning chain or graph, not a simple parallel lookup

### 3. Cross-Source Integration
The answer must require synthesizing information from multiple, distinct source documents. No single document should contain the complete answer.

### 4. Disambiguation Challenges (when the facts allow)
Where possible, incorporate ambiguity that requires contextual understanding to resolve — e.g., temporal qualifiers ("before/after X"), entity disambiguation, or conditional relationships.

---

## Question Type Guide

Vary the style across these categories based on what the facts naturally support:

- **Comparative Analysis**: Compare attributes, outcomes, or metrics across different contexts, time periods, or entities mentioned in the facts.
  Pattern: "How does [attribute A from doc X] compare to [attribute B from doc Y], and what explains the difference?"

- **Temporal Reasoning**: Questions whose answer depends on understanding chronological order, causation chains, or temporal relationships across facts.
  Pattern: "What event preceded [outcome Z], and how did it contribute to [result W]?"

- **Multi-Entity Relationships**: Trace connections between entities through intermediate facts that bridge them.
  Pattern: "What is the relationship between [entity A's action] and [entity B's outcome]?"

- **Aggregation with Context**: Combine quantitative or qualitative information from multiple facts while applying a condition or threshold found in another fact.
  Pattern: "Which [items] mentioned across [sources] satisfy the condition described in [fact N]?"

- **Causal Chain Reconstruction**: Trace a chain of causes and effects across multiple documents to explain a final outcome.
  Pattern: "What sequence of events led from [initial cause] to [final outcome]?"

---

## Complexity Levels

**If complexity_target is "complex":**
- Prefer question types that involve 3+ reasoning hops, cross-entity connections, or require resolving contradictions/ambiguities across sources.
- The question should be one that a naive keyword search would fail to answer directly.
- Use as many source documents as naturally needed to answer the question thoroughly.

**If complexity_target is "simple":**
- The question must still require at least 2 reasoning steps (bridging information across documents), but the reasoning chain can be shorter and more direct.
- Prefer straightforward comparative or temporal questions with a clear 2-hop structure.
- Use the minimum number of sources needed to create a valid multi-hop question.

---

## Quality Criteria

DO:
- Write questions that are self-contained and understandable without seeing the fact bank.
- Ground the answer entirely and only in the provided facts — never hallucinate or add external knowledge.
- Make the answer detailed enough to be verifiable against the source documents.
- Use only document IDs that appear in the fact bank.
- Ensure the question naturally requires consulting multiple sources (the multi-source need should arise organically from the question, not be forced).

DO NOT:
- Write questions answerable from a single fact or single document.
- Write questions that are just paraphrases of a single fact statement.
- Write trivial "list" or "enumerate" questions (e.g., "What are three facts about X?").
- Include meta-references to the fact bank, document IDs, or this prompt in the question text.
- Add chain-of-thought reasoning or commentary in the output.
- Write questions where the multi-source requirement feels artificial or contrived.

---

Generate exactly ONE QA pair. Return only this XML:
<qa>
  <question>question text</question>
  <answer>answer text</answer>
  <doc_ids>1,2,3</doc_ids>
</qa>
"""


def parse_extracted_facts(text: str) -> list[ExtractedFact]:
    fact_blocks = re.findall(r"<fact>(.*?)</fact>", text, re.DOTALL)
    facts: list[ExtractedFact] = []
    seen_statements: set[str] = set()

    for block in fact_blocks:
        statement_match = re.search(r"<statement>(.*?)</statement>", block, re.DOTALL)
        doc_ids_match = re.search(r"<doc_ids?>(.*?)</doc_ids?>", block, re.DOTALL)
        fact_id_match = re.search(r"<fact_id>(\d+)</fact_id>", block)

        if not statement_match or not doc_ids_match:
            continue

        statement = statement_match.group(1).strip()
        if not statement:
            continue

        doc_ids = sorted(set(int(v) for v in re.findall(r"-?\d+", doc_ids_match.group(1))))
        if not doc_ids:
            continue

        fact_id = int(fact_id_match.group(1)) if fact_id_match else 0

        key = statement.lower()
        if key in seen_statements:
            continue
        seen_statements.add(key)
        facts.append(ExtractedFact(statement=statement, doc_ids=doc_ids, fact_id=fact_id))

    return facts


def parse_fact_grounded_qas(text: str) -> list[QA]:
    pattern = (
        r"<qa>\s*<question>(.*?)</question>\s*<answer>(.*?)</answer>\s*<doc_ids>(.*?)</doc_ids>\s*</qa>"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    qas: list[QA] = []

    for question_raw, answer_raw, doc_ids_raw in matches:
        question = question_raw.strip()
        answer = answer_raw.strip()
        if not question or not answer:
            continue
        doc_ids = [int(value) for value in re.findall(r"-?\d+", doc_ids_raw)]
        qas.append(QA(question=question, answer=answer, doc_ids=doc_ids, info={}))

    return qas
