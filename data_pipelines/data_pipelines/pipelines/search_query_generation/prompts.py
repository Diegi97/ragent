import xml.etree.ElementTree as ET
from collections.abc import Sequence
from xml.sax.saxutils import escape, quoteattr

from data_pipelines.pipelines.search_query_generation.models import RetrievalChunk


def build_generate_query_messages(
    positive: RetrievalChunk,
) -> list[dict[str, str]]:
    user_prompt = f"""
Generate one search-tool query that an autonomous research agent could issue
to retrieve the target chunk.

First, silently identify one minimal, independently retrievable fact, event,
attribute, or relation that is explicitly supported by the target chunk.
Then write a query seeking that information.

Requirements:

1. The query must express exactly one information need.
   - Prefer one entity plus one attribute, relation, event, date, or value.
   - Do not ask for multiple facts, comparisons, aggregations, explanations,
     or a broad summary of the chunk.

2. The query should be suitable for retrieval.
   - A user issuing the query should reasonably expect the target chunk to
     rank among the best results.
   - Include the entity and any necessary discriminating constraints.
   - Include only constraints supported by the chunk or its metadata.

3. Write the query as a realistic agent search input.
   - It may be a concise keyword phrase or a short natural-language question.
   - Do not mention "the target chunk", "the text above", or similar references.
   - Avoid ambiguous pronouns when a specific entity can be named.

4. Do not reveal the answer in the query.

5. The query must be answerable using the target chunk alone and use the same
   language as the target chunk.

Return exactly this XML schema:

<result>
  <query>query text</query>
</result>

Target chunk:
{chunk_xml("target_chunk", positive)}
""".strip()
    return [{"role": "user", "content": user_prompt}]


def parse_generate_query_response(content: str) -> str:
    return required_text(extract_result_root(content), "query")


def build_contrastive_narrow_messages(
    query: str,
    positive: RetrievalChunk,
    candidates: Sequence[RetrievalChunk],
) -> list[dict[str, str]]:
    candidate_xml = "\n\n".join(
        chunk_xml("chunk", candidate, rank=index)
        for index, candidate in enumerate(candidates, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You create discriminative retrieval training queries. Return only "
                "XML. Do not wrap the XML in Markdown. Do not return JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Rewrite the query so it is answerable only by the target chunk.

Add the smallest necessary disambiguator to avoid the candidate chunks.
Keep the query natural and in the same language as the target chunk.
Select hard negatives only from the provided candidate chunk ids.
If the target cannot be made distinct from the candidates, set <keep>false</keep>.

Return exactly this XML schema:
<result>
  <keep>true</keep>
  <query>rewritten query text</query>
  <hard_negatives>
    <chunk id="candidate_chunk_id"/>
  </hard_negatives>
</result>

Original query:
{xml_text(query)}

Target chunk:
{chunk_xml("target_chunk", positive)}

Candidate chunks in relevance order:
<candidates>
{candidate_xml}
</candidates>
""",
        },
    ]


def parse_contrastive_narrow_response(content: str) -> tuple[bool, str, list[str]]:
    root = extract_result_root(content)
    keep = required_text(root, "keep").lower()
    if keep not in {"true", "yes", "1"}:
        return False, "", []
    query = required_text(root, "query")
    hard_negative_ids = [
        element.attrib["id"]
        for element in root.findall("./hard_negatives/chunk")
        if element.attrib.get("id")
    ]
    return True, query, hard_negative_ids


def extract_result_root(content: str) -> ET.Element:
    start = content.find("<result")
    end = content.rfind("</result>")
    if start < 0 or end < 0:
        raise ValueError("LLM response did not contain a <result> XML element.")
    return ET.fromstring(content[start : end + len("</result>")])


def required_text(root: ET.Element, tag: str) -> str:
    element = root.find(tag)
    value = element.text.strip() if element is not None and element.text else ""
    if not value:
        raise ValueError(f"LLM XML response is missing <{tag}> text.")
    return value


def xml_text(value: str) -> str:
    return escape(value or "")


def chunk_xml(tag: str, chunk: RetrievalChunk, rank: int | None = None) -> str:
    rank_xml = f"\n  <rank>{rank}</rank>" if rank is not None else ""
    return (
        f"<{tag} id={quoteattr(str(chunk.id))}>"
        f"{rank_xml}\n"
        f"  <title>{xml_text(chunk.title)}</title>\n"
        f"  <content>{xml_text(chunk.text)}</content>\n"
        f"</{tag}>"
    )
