import re
from collections.abc import Iterable

from data_pipelines.pipelines.deep_search_task_generation.facts import ExtractedFact

FACT_EXTRACTION_PROMPT = """You will be extracting factual information from one or more text passages in order to build a knowledge graph around a single **target entity**.

There are TWO distinct groups of entities in this task. Do not confuse them:

- **Target entity**: "{ENTITY}". This is the main subject of the extraction. You extract facts that are *about* it — i.e. facts that describe, define, configure, parameterize, relate, or otherwise pertain to this entity. The target entity does **not** need to be named verbatim inside every fact statement (see rules below), because many facts about it appear as arguments, attributes, sub-blocks, or exported values of a resource/data-source whose subject *is* the target entity.

- **Linked entities**: the ones listed in the `<entities>` block below. These are *other* named entities that, together with the target entity, form the nodes of a knowledge graph. Your job is to surface every explicit connection between the target entity and these linked entities. Facts that connect the target entity to one or more linked entities are especially valuable — they are the cross-entity edges that downstream multi-hop reasoning depends on, so never drop them.

Here are the retrieved chunks/passages. Each passage contains chunk content, not necessarily a full document. The <doc_id> identifies the original source document that the chunk came from:

<passages>
{PASSAGE}
</passages>

Here is the list of known linked entities (use these for the `mentioned_entities` field only):

<entities>
{ENTITIES}
</entities>

Extract ALL facts from these passages that are about the target entity "{ENTITY}", following these rules:

1. **Explicit statements only**: Extract only facts that are directly and explicitly stated in the passages. Do not infer, interpret, or use outside knowledge.

2. **Inclusion criterion — "about the target entity"**: A fact qualifies if it is stated in a passage whose subject is the target entity, even when the target entity's name is not repeated in that particular sentence. Concretely, treat as facts about "{ENTITY}":
   - The description/definition of a resource, data source, class, function, or page whose subject *is* the target entity.
   - Any argument, attribute, parameter, property, configuration field, sub-block, exported value, or returned field of such a resource/data source, even if the bullet only names the field (e.g. "`disaster_recovery` - (Optional) Specify if an Oracle Data Guard configuration is created...").
   - Any explicit relationship between the target entity and another named thing.
   Do NOT require the literal string "{ENTITY}" to appear in the fact statement. The literal-name requirement applies only to `mentioned_entities` (rule 8), not to inclusion.

3. **Preserve original relationships**: Write each fact exactly as the passage states it. Keep the original subject, verb, and object order. Do NOT rephrase a statement to make "{ENTITY}" the subject if it is not the subject in the source text.
   - Example — Source says "X extends Y's requirements" → Write "X extends Y's requirements", NOT "Y is extended by X" or "Y extends X's requirements".
   - Example — Source says "A reports to B" → Write "A reports to B", NOT "B is reported to by A".
   - For arguments/attributes whose bullet describes the field, keep the field name in the statement (e.g. "The `disaster_recovery` argument specifies if an Oracle Data Guard configuration is created...") so the fact stays standalone and grounded.

4. **Standalone statements**: Each fact must be a short, complete statement that can be understood on its own without additional context. Use full entity names rather than pronouns.

5. **Exhaustive extraction**: Extract every relevant fact you can find. Do not summarize or combine multiple facts into one. Prefer more granular, atomic facts over broad summaries. When a passage lists many arguments/attributes of the target entity, extract one fact per argument/attribute rather than collapsing them.

6. **No invention**: Do not create facts or combine information in ways not explicitly stated in the passages.

7. **Source documents**: For each fact, include the source document ID(s) that explicitly support the statement. Use only doc_id values from the provided <document> tags. If a fact is stated in exactly one passage, include that single doc_id.

8. **Mentioned entities (knowledge-graph edges)**: For each fact, list the **linked entities** from the `<entities>` block above whose name (or a clear direct reference to it) **literally appears in the fact statement text**.
   - Do NOT include "{ENTITY}" (the target entity) itself — only list *other* entities.
   - Only include entities from the `<entities>` list above; copy their names exactly as written.
   - "Literally appears" means the entity's name is present in the fact statement you wrote. If a fact statement mentions "Oracle Data Guard" and "Oracle Data Guard" is in the list, include it. If the fact statement does not name any linked entity, leave `<mentioned_entities>` empty.
   - Do NOT include entities that are merely related or co-occur in the same document but are not named in the fact statement.
   - These `mentioned_entities` are the cross-entity edges of the knowledge graph — prioritize surfacing facts that produce non-empty `mentioned_entities`, because they enable multi-hop questions that span documents.

Provide your final answer in the following XML format:

<facts>
  <fact>
    <statement>fact statement</statement>
    <doc_ids>123,456</doc_ids>
    <mentioned_entities>Entity One, Entity Two</mentioned_entities>
  </fact>
</facts>

If no relevant facts can be extracted from the passages, return an empty <facts> tag.

Your final output should contain only the facts tags with the extracted information. Do not include your thinking process in the final answer."""


def _parse_mentioned_entities(raw: str, exclude: str = "") -> list[str]:
    exclude_key = exclude.strip().lower()
    entities: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen or key == exclude_key:
            continue
        seen.add(key)
        entities.append(name)
    return entities


def parse_extracted_facts(text: str, entity_name: str = "") -> list[ExtractedFact]:
    root = re.search(r"<facts>(.*?)</facts>", text, re.DOTALL)
    if root is None:
        raise ValueError("Fact response requires a complete <facts> root")
    text = root.group(1)
    fact_blocks = re.findall(r"<fact>(.*?)</fact>", text, re.DOTALL)
    if re.sub(r"<fact>.*?</fact>", "", text, flags=re.DOTALL).strip():
        raise ValueError("Fact response contains malformed fact blocks")
    facts: list[ExtractedFact] = []
    seen_statements: set[str] = set()

    for block in fact_blocks:
        statement_match = re.search(r"<statement>(.*?)</statement>", block, re.DOTALL)
        if not statement_match:
            raise ValueError("Extracted fact is missing its statement")

        statement = statement_match.group(1).strip()
        if not statement:
            raise ValueError("Extracted fact statement must not be blank")

        doc_ids_match = re.search(r"<doc_ids?>(.*?)</doc_ids?>", block, re.DOTALL)
        doc_ids = (
            sorted(
                set(
                    int(doc_id_text)
                    for doc_id_text in re.findall(r"-?\d+", doc_ids_match.group(1))
                )
            )
            if doc_ids_match
            else []
        )

        fact_id_match = re.search(r"<fact_id>(\d+)</fact_id>", block)
        fact_id = int(fact_id_match.group(1)) if fact_id_match else 0

        entities_match = re.search(
            r"<mentioned_entities?>(.*?)</mentioned_entities?>", block, re.DOTALL
        )
        mentioned_entities = (
            _parse_mentioned_entities(entities_match.group(1), exclude=entity_name)
            if entities_match
            else []
        )

        statement_key = statement.lower()
        if statement_key in seen_statements:
            continue
        seen_statements.add(statement_key)
        facts.append(
            ExtractedFact(
                statement=statement,
                doc_ids=doc_ids,
                fact_id=fact_id,
                mentioned_entities=mentioned_entities,
            )
        )

    return facts


def format_documents(
    contents: Iterable[str], titles: Iterable[str], doc_ids: Iterable[int]
) -> str:
    return "".join(
        "<document>\n"
        f"<doc_id>{doc_id}</doc_id>\n"
        f"<title>{title}</title>\n"
        f"<content>{content}</content>\n"
        "</document>\n"
        for content, title, doc_id in zip(contents, titles, doc_ids)
    )
