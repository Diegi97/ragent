import re

from ragent_core.types import QA

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


def parse_fact_grounded_qas(text: str) -> list[QA]:
    pattern = r"<qa>\s*<question>(.*?)</question>\s*<answer>(.*?)</answer>\s*<doc_ids>(.*?)</doc_ids>\s*</qa>"
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
