import logging
import re
from dataclasses import dataclass, field

from ragent_core.types import Concept

DATA_SOURCE_DESCRIPTION_SECTION = """\

## Data Source Context
The following description provides context about the data source you are exploring. Use this information to better understand the domain, terminology, and structure of the content you will encounter:

<data_source_description>
{description}
</data_source_description>

Keep this context in mind when forming concepts, questions, and answers. Use common sense to avoid overgeneralizing and only rely on information that is supported by the provided documents.
"""


def format_prompt_with_description(
    base_prompt: str,
    data_source_description: str | None = None,
) -> str:
    """Append the corpus description when one is available."""
    if not data_source_description:
        return base_prompt
    return base_prompt.rstrip() + DATA_SOURCE_DESCRIPTION_SECTION.format(
        description=data_source_description.strip()
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFact:
    statement: str
    doc_ids: list[int]
    fact_id: int = 0
    mentioned_entities: list[str] = field(default_factory=list)


ENTITY_EXTRACTOR_PROMPT = """You are tasked with extracting named entities from a document. Your goal is to identify concrete, specific entities — things that have a distinct identity and could be looked up or referenced by name. Extract entities using the same language as the document.

Here is the document you will be working with:

<document>
<doc_id>{DOC_ID}</doc_id>
<title>{TITLE}</title>
<content>{CONTENT}</content>
</document>

**What counts as an entity:**
- **People**: Named individuals (e.g., "Ada Lovelace", "Satya Nadella")
- **Organizations & teams**: Companies, departments, working groups, committees (e.g., "European Central Bank", "Platform Engineering Team")
- **Projects & products**: Named software, initiatives, frameworks, tools (e.g., "Kubernetes", "Apollo 11", "GitLab CI/CD")
- **Locations & regions**: Named places, geographic areas (e.g., "Silicon Valley", "European Union")
- **Systems & infrastructure**: Named technical systems, platforms, protocols (e.g., "OAuth 2.0", "PostgreSQL", "REST API")
- **Events & milestones**: Named occurrences, releases, incidents (e.g., "Sprint Review", "v2.0 Release")
- **Policies, standards & processes**: Named regulations, frameworks, methodologies (e.g., "GDPR", "Scrum", "ISO 27001")
- **Domain-specific named things**: Named roles with institutional specificity, named metrics, named programs (e.g., "Chief Technology Officer", "Net Promoter Score", "Onboarding Program")

**What does NOT count as an entity:**
- Generic concepts or abstract ideas: "innovation", "leadership", "data quality"
- Common nouns or descriptors: "database", "meeting", "report", "strategy"
- Broad categories: "machine learning techniques", "economic policies", "management practices"
- Adjective phrases or vague labels: "effective communication", "best practices", "key findings"
- **The overarching subject of the document collection**: If the data source context (provided below) indicates that all documents belong to a specific organization, product, or domain (e.g., a company handbook, a product's documentation), do not extract that organization or product itself as an entity. It would appear in nearly every document and is too ubiquitous to support targeted question generation. Focus on more specific entities within the corpus instead.

**Extraction rules:**
- Use the **full, canonical name** of each entity. Write "World Health Organization" not "WHO", "United States of America" not "USA", unless the abbreviated form is the universally recognized name (e.g., "NASA", "NATO").
- Entity names should be short and precise — typically 1-5 words. Do not include explanatory clauses.
- Extract a **maximum of 5 entities**. Prioritize entities that are most distinctive and would support targeted question generation.
- If the document contains no clear named entities, return an empty `<entities>` tag.

Organize your output in the following structure:

<entities>
  <entity>
    <name>Entity name with proper capitalization</name>
  </entity>
</entities>

Your final output should only include the <entities> section with the structured list of entities. Do not include your thought process or any other content outside of this section."""


def parse_entities(text: str, data_source: str, doc_id: int) -> list[Concept]:
    """Parse entities from XML output produced by ENTITY_EXTRACTOR_PROMPT.

    Returns Concept objects so the downstream pipeline stays compatible.
    ``doc_id`` is supplied by the caller (the prompt no longer asks the LLM
    to echo back the document ID).
    """
    entities: list[Concept] = []

    pattern = r"<entity>\s*<name>(.*?)</name>\s*</entity>"
    matches = re.findall(pattern, text, re.DOTALL)

    for name_raw in matches:
        name = name_raw.strip()
        if not name:
            continue

        entities.append(
            Concept(
                name=name,
                data_source=data_source,
                doc_id=doc_id,
            )
        )

    return entities


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
    fact_blocks = re.findall(r"<fact>(.*?)</fact>", text, re.DOTALL)
    facts: list[ExtractedFact] = []
    seen_statements: set[str] = set()

    for block in fact_blocks:
        statement_match = re.search(r"<statement>(.*?)</statement>", block, re.DOTALL)
        if not statement_match:
            continue

        statement = statement_match.group(1).strip()
        if not statement:
            continue

        doc_ids_match = re.search(r"<doc_ids?>(.*?)</doc_ids?>", block, re.DOTALL)
        doc_ids = (
            sorted(set(int(v) for v in re.findall(r"-?\d+", doc_ids_match.group(1))))
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

        key = statement.lower()
        if key in seen_statements:
            continue
        seen_statements.add(key)
        facts.append(
            ExtractedFact(
                statement=statement,
                doc_ids=doc_ids,
                fact_id=fact_id,
                mentioned_entities=mentioned_entities,
            )
        )

    return facts
