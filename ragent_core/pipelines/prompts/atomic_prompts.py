import re
import logging

from ragent_core.types import QA, Concept

logger = logging.getLogger(__name__)


class PostprocessingError(Exception):
    """Raised when an LLM response fails validation during post-processing."""


# Shoutout to the Anthropic's prompt generation tool.

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
    """
    Append the data source description section to a prompt if provided.

    Args:
        base_prompt: The prompt template or fully formatted prompt.
        data_source_description: Optional description of the data source.

    Returns:
        The prompt with the description section appended if provided.
    """
    if not data_source_description:
        return base_prompt

    description_section = DATA_SOURCE_DESCRIPTION_SECTION.format(
        description=data_source_description.strip()
    )

    return base_prompt.rstrip() + description_section


CONCEPT_EXTRACTOR_PROMPT = """You are tasked with extracting concepts of varying importance levels from a set of documents to facilitate the creation of a question-answer generator. Your goal is to identify concepts across different importance levels - important, medium importance, and less important - extracting them with proper capitalization and formatting for clarity and consistency, using the same language as the documents.

Here are the documents you will be working with:

<documents>
{DOCUMENTS}
</documents>

Please follow these steps:

1. Read through each document carefully.

2. For each document, extract concepts at three importance levels:
   - **Important concepts**: Central ideas, key terms, main themes, or critical information
   - **Medium importance concepts**: Supporting ideas, secondary terms, or moderately relevant information
   - **Less important concepts**: Minor details, specific examples, or peripheral information

   Extract these concepts as SHORT, CLEAR TERMS - preferably 1-3 words. Focus on nouns, key phrases, or technical terms rather than full sentences.

3. You do not need to extract ALL concepts from the documents - focus on a diverse representative sample across the importance levels.

4. **Extract a maximum of 5 concepts per document**. Prioritize the most significant and useful concepts for each document.

5. Concepts can appear in just one document - there is no requirement for cross-document appearance.

6. **IMPORTANT**: Focus on SPECIFIC, MEANINGFUL concepts that would be useful for generating targeted questions and answers. Avoid overly broad, generic, or vague terms that could apply to many contexts.

   **AVOID concepts like:**
   - Generic terms: "Estadísticas", "Datos", "Documentos", "Información", "Resultados", "Métodos"
   - Overly broad categories: "Estudios sobre dietas", "Investigaciones científicas", "Teorías económicas", "Tecnologías modernas"
   - Vague descriptors: "Falso estudio", "Buenos resultados", "Estrategias efectivas", "Herramientas útiles"

   **PREFER specific concepts like:**
   - Concrete terms: "Flexibilidad metabólica", "Inteligencia artificial", "Blockchain", "Realismo mágico"
   - Specific techniques/methods: "Ayuno intermitente", "Metodología ágil", "Machine learning", "Análisis FODA"
   - Precise technical/domain terms: "Cuerpos cetónicos", "Algoritmo de consenso", "Sesgo de confirmación", "Narrativa omnisciente"

7. Extract all concepts in the same language as the documents. If the documents are in Spanish, extract concepts in Spanish. If they're in English, extract concepts in English, and so on.

8. Organize your output in the following structure:

<concepts>
  <concept>
    <name>Concept with proper capitalization and formatting</name>
    <importance>important|medium|less_important</importance>
    <doc_id>Document ID where this concept appears</doc_id>
  </concept>
  <!-- Repeat for each concept -->
</concepts>

Your final output should only include the <concepts> section with the structured list of concepts. Do not include your thought process or any other content outside of this section."""


def parse_concepts(text: str, data_source: str) -> list[Concept]:
    """
    Extract concepts from text using regex to parse the XML-like format.

    Args:
        text: Text containing concepts in XML format as produced by the concept extractor prompt
        data_source: Data source identifier to assign to extracted concepts

    Returns:
        List of Concept objects extracted from the text
    """
    concepts = []

    # Regex pattern to match concept blocks
    concept_pattern = r"<concept>\s*<name>(.*?)</name>\s*<importance>(.*?)</importance>\s*<doc_id>(\d+)</doc_id>\s*</concept>"

    # Find all concept matches
    matches = re.findall(concept_pattern, text, re.DOTALL)

    for match in matches:
        name = match[0].strip()
        importance = match[1].strip()
        doc_id = match[2].strip()

        # Validate importance level
        if importance not in ["important", "medium", "less_important"]:
            continue

        # Create Concept object with empty QAs list
        concept = Concept(
            name=name,
            importance=importance,
            data_source=data_source,
            doc_id=int(doc_id),
        )
        concepts.append(concept)

    return concepts


QUESTION_GENERATION_PROMPT = """You will be generating multiple self-contained questions for a question-answer system. All questions must be based on the same source document but focused on different concepts. Your goal is to produce one distinct question for each concept provided, ensuring every question can stand on its own without referencing the document or the concept list.

Here is the source document that provides context for every concept:

<source_document>
{SOURCE_DOCUMENT}
</source_document>

Here are the concepts you must cover, in order:

<concepts>
{CONCEPTS}
</concepts>

Your task is to generate one self-contained question for each concept using the same language as the source document. Each question should be specific enough to be useful but broad enough to capture relevant information that would likely be found in documents similar to the source document provided.

**IMPORTANT - Questions must be self-contained**: Each question must be fully understandable on its own without requiring prior knowledge of the source document. Avoid vague references like:
- "the study" → Instead specify what the study is about (e.g., "What problems are present in studies that relate red meat consumption to premature death?")
- "this method", "that approach" → Be specific about what method or approach
- "the research", "the experiment" → Describe what the research/experiment examined
- "it", "this", "that" when referring to concepts from the document → Use the full concept name
- **NEVER reference "the document", "del documento", "from the document", "in the text", "according to the source", or any similar meta-reference to the source material itself**

Make sure someone reading only the question (without the source document or concept) can understand what is being asked.

**CRITICAL - Questions must be ATOMIC (single-focus)**: Each question should ask about ONE thing only. Avoid compound questions that ask multiple things or have multiple clauses connected by "and"/"y"/"or"/"o".

Common signs of NON-ATOMIC questions to AVOID:
- Questions with "and what", "y qué", "and how", "y cómo" connecting multiple inquiries
- Questions asking about both X AND its effects/impacts/consequences
- Questions asking about relationships AND their implications
- Questions with multiple question marks or multiple interrogative words
- Questions listing multiple aspects (e.g., "effects on X, Y, and Z")

If a concept naturally leads to multiple questions, pick the MOST DIRECT and FUNDAMENTAL aspect to ask about.

Examples of BAD questions (compound, multi-focus):
- "What problems does the study present?" ❌ (vague)
- "How does this method work?" ❌ (vague)
- "¿Cómo utilizan las estrategias de marketing la publicidad para hacer que los alimentos parezcan saludables, y qué impacto tiene eso en las decisiones de compra?" ❌ (asks about HOW they use it AND the impact)
- "¿Cómo se relaciona la exposición al frío con la liberación de dopamina y qué efectos podría tener esta neurotransmisión?" ❌ (asks about the relationship AND the effects)
- "What are the benefits and risks of intermittent fasting?" ❌ (asks about two things)

Examples of GOOD questions (atomic, single-focus, self-contained):
- "What problems are present in studies that relate red meat consumption to premature death?" ✓
- "How does intermittent fasting affect insulin sensitivity?" ✓
- "¿Cómo utilizan las estrategias de marketing la publicidad para hacer que los alimentos ultraprocesados parezcan saludables?" ✓ (focused only on HOW)
- "¿Cómo se relaciona la exposición al frío con la liberación de dopamina?" ✓ (focused only on the relationship)
- "What foods have anti-cancer properties?" ✓

**Additional Guidelines:**
1. Produce exactly one question per concept. Do not omit or merge concepts.
2. Preserve the order of concepts as provided.
3. Use precise language that clearly references the concept itself.
4. Each question must be independently understandable.

Return your response in the following XML format, preserving the order of the provided concepts:

<questions>
  <qa>
    <concept>Concept Name</concept>
    <question>The self-contained question</question>
  </qa>
  <!-- one <qa> block for each concept in order -->
</questions>

Your final response should only include the XML shown above.
"""


def extract_batch_qas_from_text(
    text: str,
    *,
    doc_id: int,
    concepts: list[Concept],
) -> list[QA]:
    """
    Extract multiple QA entries from model output containing XML tags.

    Args:
        text: Text containing batched QA pairs in XML format as produced by the batch question generation prompt.
        doc_id: Document ID that all QAs relate to.
        concepts: Ordered list of concepts expected in the output.

    Returns:
        List of QA objects ordered to align with the provided concepts.
    """

    qa_pattern = r"<qa>\s*<concept>(.*?)</concept>\s*<question>(.*?)</question>\s*</qa>"
    matches = re.findall(qa_pattern, text, re.DOTALL)

    if not matches:
        raise PostprocessingError("No QA pairs found in the batch question response")

    concept_map: dict[str, Concept] = {concept.name: concept for concept in concepts}
    qa_map: dict[str, QA] = {}

    for concept_name_raw, question_raw in matches:
        concept_name = concept_name_raw.strip()
        question = question_raw.strip()

        if not concept_name or not question:
            continue

        concept = concept_map.get(concept_name)
        if concept is None:
            continue

        qa_map[concept_name] = QA(
            question=question,
            answer="",
            doc_ids=[doc_id],
            info={"concept": concept_name},
        )

    missing_concepts = [
        concept.name for concept in concepts if concept.name not in qa_map
    ]

    if missing_concepts:
        # Just a warning
        logger.warning(f"Missing questions for concepts: {', '.join(missing_concepts)}")

    return [qa_map[concept.name] for concept in concepts]


ANSWER_GENERATION_PROMPT = """You are tasked with generating a focused, direct answer to a question based on a set of source documents. You will be provided with documents from a data source and a specific question to answer using information from those documents.

<documents>
{DOCUMENTS}
</documents>

<question>
{QUESTION}
</question>

Your task is to synthesize a focused answer to the question using only the information provided in the documents above. Follow these guidelines:

1. **Answer ONLY what is asked**: Read the question carefully and answer precisely what it asks. Do not expand into related topics, explanations of how to achieve something if not asked, or background information unless directly necessary to answer the specific question.

2. **Thoroughly analyze all documents**: Read through all the provided documents carefully to identify relevant information that helps answer the question.

3. **Extract relevant information**: Identify key facts, concepts, examples, and details from the documents that directly relate to the question being asked. Discard information that is interesting but not directly answering the question.

4. **Synthesize a focused answer**: Combine the relevant information from multiple documents to create a complete, well-structured answer. Do not simply copy text verbatim - instead, synthesize and organize the information logically. Keep your answer concise and on-topic.

5. **Stay grounded in the source material**: Only use information that is explicitly stated or can be reasonably inferred from the provided documents. Do not add external knowledge or make claims not supported by the source material.

6. **Handle conflicting information**: If you find contradictory information across documents, acknowledge this in your answer and present both perspectives when relevant.

7. **Structure your response clearly**: Organize your answer in a logical flow that directly addresses the question. Use clear paragraphs and transitions between ideas.

8. **Be direct and concise**: Answer the question directly without adding superfluous details. If the question asks "What role does X play?", focus on explaining the role, not on how to improve X or other tangentially related topics.

9. **Match the language**: Generate your answer in the same language as the source documents.

10. **Do not reference document IDs or indices**: Never mention specific document numbers, IDs, or indices (e.g., "document 94", "documento 88", "doc 253") in your answer. Integrate information seamlessly without citing which document it came from.

**Important reminders**:
- If the question asks "What is X?", explain what X is, not how to use X or improve X.
- If the question asks "What role does X play?", explain the role, not a comprehensive guide about X.
- If the question asks "How does X work?", explain the mechanism, not all the benefits of X.
- Avoid adding lists of tips, recommendations, or how-to information unless the question explicitly asks for them.

If the question cannot be adequately answered based on the provided documents, clearly state what information is missing or insufficient.

Your final response should contain only your synthesized answer to the question based on the documents provided. Do not include references to "the documents" or "according to the sources" - write as if you are directly providing the information to someone asking the question."""


ANSWER_REFINMENT_PROMPT = """You are tasked with refining an existing answer to a question by incorporating information from additional documents. You will be given a question, a current answer, and additional documents that may contain relevant information to improve the answer.

Here is the question:
<question>
{QUESTION}
</question>

Here is the current answer that needs to be refined:
<current_answer>
{CURRENT_ANSWER}
</current_answer>

Here are additional documents that may contain relevant information:
<additional_docs>
{ADDITIONAL_DOCS}
</additional_docs>

Your task is to refine and improve the current answer by incorporating relevant information from the additional documents. Follow these guidelines:

1. **Answer ONLY what is asked**: Before adding any information, re-read the question carefully. Only add information that directly answers what the question asks. Do not expand the answer into related topics, how-to guides, or tangential information.

2. **Preserve accurate information**: Keep all correct and relevant information from the current answer.

3. **Add new relevant information**: Incorporate any new facts, details, examples, or insights from the additional documents that help answer the specific question more completely. Be critical - just because information is in the additional documents doesn't mean it should be added if it doesn't directly answer the question.

4. **Remove superfluous information**: If the current answer contains information that, while related, doesn't directly answer the question asked, remove it during refinement. This includes:
   - How-to information when the question asks "what" or "what role"
   - Extensive lists of tips or recommendations when not explicitly asked
   - Background information that goes beyond what's needed to understand the answer
   - Related but tangential topics

5. **Resolve contradictions**: If the additional documents contradict information in the current answer, use the following approach:
   - First, attempt to determine which source is more recent based on dates, version numbers, or other temporal indicators in the content. If recency can be established, prioritize information from the most recent source.
   - If recency cannot be determined, preserve the information from the current answer. Since the answer is refined iteratively from the most important documents (highest relevance to the query) to the least important, the existing answer already reflects information from more relevant sources.

6. **Maintain coherence**: Ensure the refined answer flows logically and reads as a cohesive response rather than a collection of disconnected facts.

7. **Stay focused and concise**: Only include information that directly relates to answering the question. Do not add tangential details. Prefer a shorter, more focused answer over a longer, more comprehensive one.

8. **Match the language**: Maintain the same language as the current answer and source documents throughout your refinement.

9. **Do not reference document IDs or indices**: Never mention specific document numbers, IDs, or indices (e.g., "document 94", "documento 88", "doc 253") in your refined answer. Remove any such references that may exist in the current answer and integrate all information seamlessly without citing which document it came from.

**Important reminders**:
- If the question asks "What is X?", focus on what X is, not how to use or improve X.
- If the question asks "What role does X play?", explain the role, not a comprehensive guide about X.
- If the question asks "How does X work?", explain the mechanism, not all the benefits or applications of X.
- Be ruthless in cutting information that doesn't directly answer the question, even if it's interesting or related.

If the additional documents do not contain any relevant information that would improve the current answer, simply return the current answer unchanged (or with superfluous information removed if present).

If the additional documents are entirely irrelevant to the question, state that no refinements were made and return the original answer (or a trimmed version if the original contained superfluous information).

Your response should contain only the refined answer - do not include explanations of what changes you made or reasoning about your refinements."""
