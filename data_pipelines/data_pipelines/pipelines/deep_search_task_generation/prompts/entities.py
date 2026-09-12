import re

from data_pipelines.pipelines.deep_search_task_generation.prepare.models import Concept

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
