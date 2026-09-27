import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import unescape

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
        name = unescape(name_raw).strip()
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

- **Target entity**: "{ENTITY}". Extract facts about this specific entity. Retrieved passages may concern unrelated or similarly named entities.

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

**Identity check — apply before extracting any facts:**
- Similar names or shared attributes do not establish identity. Do not assume typos, aliases, or equivalence.
- Resolve abbreviations, pronouns, and generic labels only when the supplied context from the same source document clearly identifies the target. Do not transfer local definitions between documents.
- For other entities, extract only explicit relationships with the target. Omit facts whose subject is uncertain; return `<facts></facts>` if none qualify.

1. **Explicit statements only**: Extract only facts that are directly and explicitly stated in the passages. Do not infer, interpret, or use outside knowledge.

2. **Inclusion criterion — "about the target entity"**: After the identity check passes, extract explicit descriptions, attributes, actions, obligations, and relationships of "{ENTITY}". The name need not be repeated in every sentence when its reference is unambiguous within the supplied same-document context. A document merely mentioning the target does not make every statement in that document a fact about it.

3. **Preserve original relationships**: Write each fact exactly as the passage states it. Keep the original subject, verb, and object order. Do NOT rephrase a statement to make "{ENTITY}" the subject if it is not the subject in the source text.
   - Example — Source says "X extends Y's requirements" → Write "X extends Y's requirements", NOT "Y is extended by X" or "Y extends X's requirements".
   - Example — Source says "A reports to B" → Write "A reports to B", NOT "B is reported to by A".
   - Resolve a pronoun or abbreviated name to a full name only when the same-document evidence establishes that identity. Never replace a different named subject with the target.

4. **Standalone statements**: Write short, complete statements using supported names and context. Preserve dates, conditions, and uncertainty. Do not guess missing context.

5. **Exhaustive extraction**: Extract every relevant fact you can find. Do not summarize or combine multiple facts into one. Prefer more granular, atomic facts over broad summaries. When a passage lists many arguments/attributes of the target entity, extract one fact per argument/attribute rather than collapsing them.

6. **No invention**: Do not create facts or combine information in ways not explicitly stated in the passages.

7. **Source documents**: For each fact, include the source document ID(s) that explicitly support the statement. Use only doc_id values from the provided <document> tags. If a fact is stated in exactly one passage, include that single doc_id.

8. **Mentioned entities (knowledge-graph edges)**: For each fact, list the **linked entities** from the `<entities>` block above whose name (or a clear direct reference to it) **literally appears in the fact statement text**.
   - Do NOT include "{ENTITY}" (the target entity) itself — only list *other* entities.
   - Only include entities from the `<entities>` list above; copy their names exactly as written.
   - Put each name in its own `<entity>` tag inside `<mentioned_entities>`. Commas may be part of a name; never use commas to separate entities.
   - "Literally appears" means the entity's name is present in the fact statement you wrote. If a fact statement mentions "Oracle Data Guard" and "Oracle Data Guard" is in the list, include it. If the fact statement does not name any linked entity, leave `<mentioned_entities>` empty.
   - Do NOT include entities that are merely related or co-occur in the same document but are not named in the fact statement.
   - These `mentioned_entities` are the cross-entity edges of the knowledge graph — prioritize surfacing facts that produce non-empty `mentioned_entities`, because they enable multi-hop questions that span documents.

Provide your final answer in the following XML format:

<facts>
  <fact>
    <statement>fact statement</statement>
    <doc_ids>123,456</doc_ids>
    <mentioned_entities>
      <entity>Example Company, Inc.</entity>
      <entity>Entity Two</entity>
    </mentioned_entities>
  </fact>
</facts>

If no relevant facts can be extracted from the passages, return an empty <facts> tag.

Your final output should contain only the facts tags with the extracted information. Do not include your thinking process in the final answer."""


def _parse_mentioned_entities(
    raw: str,
    exclude: str = "",
    known_entities: Sequence[str] | None = None,
) -> list[str]:
    """Read entity tags, or recover legacy comma lists using request names.

    Without a catalog a legacy comma list is ambiguous, so retain it as one
    value rather than inventing entities from pieces of a legal name.
    """
    exclude_key = exclude.strip().casefold()
    canonical = (
        {
            unescape(name).strip().casefold(): unescape(name).strip()
            for name in known_entities
            if name.strip()
        }
        if known_entities is not None
        else None
    )
    tagged = re.findall(r"<entity>(.*?)</entity>", raw, re.DOTALL)
    if tagged:
        names = [unescape(name).strip() for name in tagged]
    elif canonical is not None:
        # Longest names win over prefixes such as "Pinnacle Trust Bank".
        # Include the target so its comma-containing name is consumed whole
        # before it is excluded below.
        candidates = set(canonical.values())
        if exclude:
            candidates.add(exclude)
        alternatives = "|".join(
            re.escape(name) for name in sorted(candidates, key=lambda n: (-len(n), n))
        )
        names = (
            re.findall(
                rf"(?:^|,)\s*({alternatives})(?=\s*(?:,|$))",
                unescape(raw).strip(),
                re.IGNORECASE,
            )
            if alternatives
            else []
        )
    else:
        names = [unescape(raw).strip()]

    entities: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name:
            continue
        key = name.casefold()
        if key in seen or key == exclude_key:
            continue
        if canonical is not None:
            if key not in canonical:
                continue
            name = canonical[key]
        seen.add(key)
        entities.append(name)
    return entities


def parse_extracted_facts(
    text: str,
    entity_name: str = "",
    known_entities: Sequence[str] | None = None,
) -> list[ExtractedFact]:
    fact_blocks = re.findall(r"<fact>(.*?)</fact>", text, re.DOTALL)
    facts: list[ExtractedFact] = []
    seen_statements: set[str] = set()

    for block in fact_blocks:
        statement_match = re.search(r"<statement>(.*?)</statement>", block, re.DOTALL)
        if not statement_match:
            continue

        statement = unescape(statement_match.group(1)).strip()
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
            _parse_mentioned_entities(
                entities_match.group(1),
                exclude=entity_name,
                known_entities=known_entities,
            )
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
