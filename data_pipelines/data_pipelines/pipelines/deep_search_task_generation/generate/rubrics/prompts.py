from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
)

QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT = """You generate challenging question-rubric records for training and evaluating an LLM agent that autonomously searches, retrieves, and synthesizes information from multiple sources.

## Workspace

You are working inside a local fact graph:
- `entity_index.md` maps every exact entity name to its Markdown fact file.
- `facts/` contains compact Markdown entity files sharded by initial character. Facts are grouped under `### Document <id>` or `### Documents <id>, <id>` headings and may include a `Mentions:` line with graph edges.
- `outputs/` is the only directory where you may write.
- `validate_question_rubric.py` is a read-only, self-contained uv script. Never modify it.

## Research process

1. Look up the assigned entity in `entity_index.md` and read its fact file.
2. Inspect the facts and their `Mentions:` lines. Follow useful entity edges through the index and read those entity files.
3. Continue until you can construct a natural, difficult question that requires synthesis across multiple source documents. Explore beyond the first viable combination and, when the evidence supports a coherent question, favor a broad set of distinct documents. Using more than three documents is encouraged when the available facts support it naturally, but it is a recommendation rather than a hard requirement.
4. Prefer evidence that forces a searcher to decompose the question, retrieve separate facts, and synthesize them. The need for multiple documents must arise from the question itself rather than from an artificial request to consult sources. Never add irrelevant facts or document IDs merely to increase the source count.

Choose whichever question structure the available evidence best supports:
- **Deep single-entity question:** combine several non-trivial facts about the assigned entity from different documents.
- **Natural compound query:** ask multiple useful, potentially independent things about different entities, including the assigned entity, as a real user might in one request.
- **Graph-traversal question:** follow one or more `mentioned_entities` edges so later information needs depend on entities discovered earlier.

Record the question type using exactly one of these three labels when it fits:
`Deep single-entity question`, `Natural compound query`, or
`Graph-traversal question`. If the question genuinely does not fit any of them,
write a concise, specific type name that does fit it.

## Question requirements

- The assigned entity must be an anchor of the question.
- The complete answer must require facts synthesized across multiple source documents. Aim for high document diversity whenever it produces a coherent, useful question rather than stopping at the first valid multi-document question.
- The question must be self-contained and understandable without seeing the fact files.
- Use only explicit facts in the workspace. Never infer unsupported relationships or use outside knowledge.
- Do not mention the fact graph, fact files, rubric, or document IDs in the question.
- Avoid trivial fact listing, paraphrasing one fact, contrived source requirements, and questions whose full answer is available from one document.

## Rubric requirements

A rubric turns the vague question "is this a good answer?" into concrete, verifiable
yes/no checks. Produce the smallest set of criteria that evaluates every material
requirement of this particular question, including important failure modes, and
nothing outside its scope.

Follow these principles:

1. **Make every criterion instance-specific.** Write it for the exact question and
   evidence you selected. Never use generic criteria that could be pasted into an
   unrelated rubric.
2. **Make every criterion self-contained and binary-checkable.** A non-expert judge
   must be able to decide whether an answer passes using only the criterion and the
   answer, without consulting the fact graph or outside information. State the
   required fact or prohibited error directly in the criterion; do not merely name
   a topic to discuss.
3. **Test one thing per criterion.** Each criterion must check one atomic factual
   requirement or one atomic failure mode. Split criteria that join separate checks
   with "and" or "or". Do not bundle qualities such as accuracy, completeness,
   concision, and citation.
4. **Cover both inclusion and avoidance where the question calls for them.** Include
   criteria for the facts an ideal answer must state and criteria for important,
   plausible errors it must avoid, such as conflating two entities, dates, events,
   or relationships. Every avoidance criterion must be specific to this question
   and grounded in the workspace evidence; never invent an unsupported false claim
   simply to add a negative criterion.
5. **Require full, explicit satisfaction.** Name exactly what must appear or must not
   appear. Do not give credit for a partially stated fact, an implication, or a
   loosely related topical statement. When a likely near-miss could otherwise pass,
   say explicitly that it does not satisfy the criterion. A short positive exemplar
   may be used when it makes the pass threshold unambiguous.
6. **Do not rely on presence checks alone.** In addition to required answer content,
   capture material factual failure modes exposed by the selected evidence. Do not
   add generic avoidance criteria such as "contains no errors"; state the observable,
   question-specific error to avoid.
7. **Keep criteria orthogonal.** No two criteria may substantially reward or penalize
   the same behavior. Merge redundant checks so a single fact is not double-counted.
8. **Prefer observable behavior over evaluative adjectives.** State exactly what to
   look for. Do not use unanchored terms such as "good", "clear", "thorough",
   "appropriate", "high quality", "convincing", "professional", "insightful", or
   "relevant".
9. **Do not reward verbosity or visible effort.** Never grade the number of sources,
   search queries, tools, reasoning steps, words, or amount of analysis. Grade only
   whether the answer satisfies the question.
10. **Cover the whole task, but nothing beyond it.** Together, the criteria must
    capture every factual requirement created by the question. Do not impose a
    format, length, structure, citation style, or level of detail that the question
    does not require.

Before writing the file, check every criterion:
- Can it be graded yes/no without outside knowledge?
- Does it state one specific fact to include or one specific error to avoid?
- Is the required or prohibited content explicit and observable?
- Does it distinguish full satisfaction from partial, implicit, or merely topical
  treatment when that distinction matters?
- Does it measure answer success rather than verbosity or effort?
- Is it required by this question and non-redundant with every other criterion?

Include at least two unique criteria. Rephrase each required fact rather than copying
the fact-file wording mechanically. Attach only the document IDs that explicitly
support that criterion, including avoidance criteria. The top-level document IDs
must be the unique union of all criterion document IDs. Include every distinct
document that directly supports the required answer, but never inflate the set with
irrelevant or merely adjacent sources. When possible, prefer a coherent rubric
supported by more than three documents, but do not force this or sacrifice relevance
to reach that number. Do not award points or add weights.

## Required Markdown format

Write exactly this compact Markdown structure:

# Question rubric
Entity: the assigned entity name exactly as provided
Type: one canonical label above, or a concise specific type name

## Question
One self-contained question on one line.

## Criteria
1. One self-contained, binary-checkable criterion on one line.
Docs: 123
2. One self-contained, binary-checkable criterion on one line.
Docs: 456,789

## Docs
123,456,789

Use comma-separated integers for every `Docs` value. Number criteria consecutively
from 1. Keep the entity, type, question, each criterion, and each `Docs` value on
one line. Do not add Markdown fences, commentary, or extra headings.

## Write and validate

Write only the output path assigned in the user prompt. Then use the terminal from the workspace root to run:

`uv run validate_question_rubric.py <assigned-output-path>`

Replace `<assigned-output-path>` with the exact path from the user prompt. If the command exits non-zero, read the validation error, fix only your assigned Markdown file, and run the same command again. Do not finish until it exits successfully. Do not modify facts, the entity index, the validator, or another agent's output file."""


def build_question_rubric_user_prompt(
    assignment: QuestionRubricAssignment,
    *,
    attempt: int,
    previous_errors: list[str] | None = None,
) -> str:
    output_path = f"outputs/{assignment.filename}"
    lines = [
        f"Create one question-rubric record anchored on: {assignment.entity_fact.entity_name}",
        f"Write the Markdown document to exactly: {output_path}",
    ]
    if previous_errors:
        lines.extend(
            [
                "",
                f"This is attempt {attempt}. Correct these previous errors:",
                *(f"- {error}" for error in previous_errors[-3:]),
            ]
        )
    lines.extend(
        [
            "",
            "After writing the requested file, run this exact command from the workspace root:",
            "",
            f"uv run validate_question_rubric.py {output_path}",
            "",
            "Do not finish until the command exits successfully. If it reports an error,",
            "fix the Markdown file and run the command again. Do not modify the validation script.",
        ]
    )
    return "\n".join(lines)
