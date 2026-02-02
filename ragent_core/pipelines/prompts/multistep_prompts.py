"""Prompt templates and utilities for the multi-step QA composition pipeline."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_RECIPE_KEY = "breadth_multihop_logic_search"

DATA_SOURCE_DESCRIPTION_SECTION = """\

## Data Source Context
The following description provides context about the data source you are exploring. Use this information to better understand the domain, terminology, and structure of the content you will encounter:

<data_source_description>
{description}
</data_source_description>

Use this context to craft questions and answers that make sense for the domain, and avoid assumptions not supported by the provided findings.
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


BREADTH_MULTIHOP_QUESTION_PROMPT = """You will be given a set of simple, atomic question-answer pairs from a knowledge base. Your task is to create one new complex question that requires multi-hop reasoning to answer, where the answer can only be obtained by chaining together multiple inference steps using the information provided in the simple Q&A pairs.

<simple_qa_pairs>
{qa_pairs}
</simple_qa_pairs>

Your goal is to create a question where:
- The answer requires combining information from at least 2-3 of the simple Q&A pairs (you don't need to use all pairs, only the relevant ones)
- Each reasoning step depends on the output of the previous step
- The question cannot be answered by looking at just one simple Q&A pair
- The answer must be completely derivable using only the information contained in the provided Q&A pairs - do not ask for information that cannot be obtained by combining the given answers
- The question should be self-contained and conversational, understandable on its own without requiring context from the original Q&A pairs
- The question should sound natural and realistic (not artificially contrived)
- Write your question in the SAME LANGUAGE as the provided Q&A pairs. If the pairs are in Spanish, write in Spanish. If they are in English, write in English. Match the language exactly.

Good complex questions often involve:
- Finding connections between entities mentioned in different Q&A pairs
- Requiring temporal reasoning (what happened before/after)
- Needing to combine quantitative information
- Asking about relationships that emerge from combining multiple facts
- Requiring comparison or ranking based on multiple pieces of information

Avoid creating questions that:
- Can be answered directly from a single Q&A pair
- Require external knowledge not provided in the simple Q&A pairs
- Ask for information that cannot be derived by combining the provided answers
- Are overly convoluted or unnatural
- Have ambiguous answers
- Reference the Q&A pairs themselves or require prior context to understand

Examples of multi-hop reasoning:
- If Q&A pair 1 says "John works at Company X" and Q&A pair 2 says "Company X is located in Boston", a good complex question might be "Where does John work?" (requiring you to first find John's company, then find that company's location)
- If one pair gives a person's birth year and another gives the current year, you could ask about their age
- If different pairs mention related events with dates, you could ask about their chronological order

<scratchpad>
Before creating your complex question, think through:
1. Which Q&A pairs contain information that can be meaningfully connected? (You don't need to use all of them)
2. What entities, relationships, or concepts appear across the selected pairs?
3. What logical connections can you make between different pieces of information?
4. What would be a natural, self-contained question that requires chaining these connections together?
5. Can you trace through the reasoning steps needed to answer your proposed question using only the provided information?
6. Is the answer fully derivable from combining the provided answers, without requiring any external knowledge?
</scratchpad>

Create one complex question that demonstrates multi-hop reasoning. Your final output should contain only the complex question you've created, written inside <complex_question> tags.
"""

BREADTH_THEMATIC_QUESTION_PROMPT = """You will be given a set of simple, atomic question-answer pairs from a knowledge base. These pairs are clustered together because they share similar themes or topics. Your task is to create one new broad, thematic question that requires synthesizing information across multiple Q&A pairs to provide a comprehensive answer.

<simple_qa_pairs>
{qa_pairs}
</simple_qa_pairs>

Your goal is to create a question where:
- The answer requires gathering and synthesizing information from at least 2-3 of the simple Q&A pairs (you don't need to use all pairs, only the relevant ones)
- The question asks for general information about a theme, topic, or concept rather than a specific fact
- The question encourages a comprehensive response that brings together related pieces of information
- The answer must be completely derivable by synthesizing the information contained in the provided Q&A pairs - do not ask for information that cannot be obtained from the given answers
- The question should be self-contained and conversational, understandable on its own without requiring context from the original Q&A pairs
- The question should sound natural, like something a curious person would genuinely ask
- Write your question in the SAME LANGUAGE as the provided Q&A pairs. If the pairs are in Spanish, write in Spanish. If they are in English, write in English. Match the language exactly.

Good thematic questions often ask about:
- Overview or explanation of a concept, process, or phenomenon
- Characteristics, features, or properties of something
- Benefits, advantages, or purposes of something
- Ways to do something or approaches to a problem
- Types, categories, or examples of something
- Relationships between related concepts or entities
- General patterns or trends across multiple data points

Avoid creating questions that:
- Ask for a single specific fact that can be answered by one Q&A pair alone
- Require external knowledge not provided in the simple Q&A pairs
- Ask for information that cannot be derived by combining and synthesizing the provided answers
- Are overly narrow or technical when a broader question would better capture the theme
- Reference the Q&A pairs themselves or require prior context to understand
- Have ambiguous scope or unclear expectations

Examples of thematic questions:
- If multiple Q&A pairs discuss different vitamins and their functions, ask "What are the main benefits of different vitamins for health?"
- If pairs mention various features of a product, ask "What features does [product] offer?"
- If pairs describe different aspects of a historical event, ask "What were the key aspects of [event]?"
- If pairs explain steps in a process, ask "How does [process] work?"

<scratchpad>
Before creating your thematic question, think through:
1. What is the common theme or topic that connects the Q&A pairs?
2. Which Q&A pairs contain complementary information about this theme? (You don't need to use all of them)
3. What broader question would naturally elicit answers that draw from multiple pairs?
4. Is your question asking for synthesis and overview rather than a single specific fact?
5. Can the question be answered comprehensively by combining information from the selected Q&A pairs?
6. Is the question self-contained and natural-sounding, without referencing the structure of the Q&A pairs?
7. What language are the Q&A pairs written in, and am I matching that language exactly?
</scratchpad>

Create one thematic question that requires synthesizing information from multiple Q&A pairs. Your final output should contain only the question you've created, written inside <thematic_question> tags.
"""

RANDOM_WALK_QUESTION_PROMPT = """You will be given a set of simple, atomic question-answer pairs from a knowledge base. These pairs highlight concepts that are linked together through shared relationships.

<simple_qa_pairs>
{qa_pairs}
</simple_qa_pairs>

Your goal is to craft a question that explores interconnected concepts through a natural random walk, asking about a topic and its related neighbors without imposing a strict linear progression. The question should feel like following connections organically from one related concept to another, creating a subgraph of connected information nodes.

Style hints:
- Frame the question as exploring connected concepts rather than following a predetermined path.
- Use phrases that suggest connection and exploration: "and its relationship to...", "which connects to...", "along with related aspects of...".
- Avoid rigid sequencing phrases like "first...then...finally"—let the connections guide the flow.
- Ask for factual information about interconnected topics, not theoretical synthesis.
- The question should feel like traversing a knowledge graph, where each concept naturally leads to related neighbors.

Examples of effective questions:
- "What is dopamine's role in the brain's reward system, and how does this relate to addiction mechanisms and the impact on decision-making processes?"
- "How do neural networks use backpropagation, what is the connection to gradient descent optimization, and how does this relate to overfitting in model training?"
- "What causes coral bleaching, how does this connect to ocean acidification, and what are the related impacts on marine biodiversity in reef ecosystems?"

Guidelines to follow:
- Combine information from at least 2-3 Q&A pairs—do not base the question on a single pair.
- Let the question feel like an exploratory walk: each clause should lead naturally to the next related concept.
- Avoid rigid sequencing language like "first...then...finally". Instead, use connective phrases such as "and how does... relate to..." or "along with".
- Keep the focus on factual information that can be answered using only the provided Q&A pairs.
- Ensure the question is fully self-contained.
- Write your question in the SAME LANGUAGE as the provided Q&A pairs. If the pairs are in Spanish, write in Spanish. If they are in English, write in English. Match the language exactly.

Output only the final question, wrapped in <question> tags.
"""

MULTI_TOPIC_QUESTION_PROMPT = """You will be given a set of simple, atomic question-answer pairs from a knowledge base. These pairs may cover unrelated topics.

<simple_qa_pairs>
{qa_pairs}
</simple_qa_pairs>

Craft a single user query that naturally bundles two or three unrelated sub-questions. Keep each sub-question distinct and separate—do NOT merge them, combine them, or create artificial connections between them. Simply present them as separate asks in one query.

Style hints:
- Let the disjoint curiosity feel organic, as if the user stacked requests in one breath.
- Do NOT invent bridges, relationships, or shared context between the topics.
- Use transitional phrases like "Also,", "Additionally,", "On another note,", or "Separately" to signal topic shifts.
- Ensure each sub-question remains independently understandable.

Examples of effective bundled questions:
- "What are the benefits of meditation for stress reduction? Also, what is the chemical composition of aspirin?"
- "How does photosynthesis work in C4 plants? And separately, what are the key architectural features of Gothic cathedrals?"
- "Can you explain the differences between supervised and unsupervised learning? On another note, what causes the aurora borealis?"

Guidelines to follow:
- Each sub-question must rely on a different Q&A pair, and together they should cover at least 2-3 pairs.
- Keep the sub-questions clearly separated using natural transitions like "Also," "Additionally," or "On another note".
- Do not invent or imply connections between the sub-questions—let them remain independent curiosities.
- Make the phrasing conversational, factual, and answerable strictly from the given Q&A pairs.
- Write your question in the SAME LANGUAGE as the provided Q&A pairs. If the pairs are in Spanish, write in Spanish. If they are in English, write in English. Match the language exactly.

Output only the final bundled question, wrapped in <question> tags.
"""

QUESTION_PROMPTS: dict[str, str] = {
    "breadth_multihop_logic_search": BREADTH_MULTIHOP_QUESTION_PROMPT,
    "breadth_thematic_synthesis": BREADTH_THEMATIC_QUESTION_PROMPT,
    "random_walk_chain": RANDOM_WALK_QUESTION_PROMPT,
    "multi_topic_queries": MULTI_TOPIC_QUESTION_PROMPT,
}

ANSWER_GOALS: dict[str, str] = {
    "breadth_multihop_logic_search": (
        "Explain the answer by stepping through the contributing facts so the reasoning chain is explicit."
    ),
    "breadth_thematic_synthesis": (
        "Deliver an organized summary that weaves together the complementary details into a cohesive explanation."
    ),
    "random_walk_chain": (
        "Present the information by following the natural connections between concepts so the flow feels like exploring a network of related ideas."
    ),
    "multi_topic_queries": (
        "Provide distinct answers for each unrelated sub-question, using clear transitions so the reader can navigate the topics independently."
    ),
    "atomic_replay": (
        "Restate the original atomic answer clearly and concisely without additional synthesis."
    ),
}

MULTISTEP_ANSWER_PROMPT = """You are tasked with presenting multiple research findings in a single, cohesive answer that responds to a complex, multi-step question. Your goal is to create a natural, human-friendly explanation that presents the documented information, following a specific organization strategy.

Here is the synthesis strategy you should follow:

<recipe_config>
{recipe_config}
</recipe_config>

Here is the question you need to answer:

<final_question>
{final_question}
</final_question>

Here are the supporting findings from the multi-step research process:

<supporting_findings>
{supporting_findings}
</supporting_findings>

Your task is to present these findings according to the recipe configuration. Follow these guidelines:

1. **Language Consistency**: Write the answer in the SAME LANGUAGE as the question and supporting findings. If they are in Spanish, write in Spanish. If they are in English, write in English. Match the language exactly.
2. **Follow the organization intent**: Use the recipe configuration to guide how you present the information, but don't be overly rigid - let the content shape the natural flow.
3. **Create organic flow**: Write in a way that sounds like a knowledgeable person presenting factual information. Avoid mechanical language. For example, instead of just listing facts ("Fact A. Fact B."), try to connect them: "Fact A, which then leads to Fact B." or "Building on Fact A, it's also important to consider Fact B."
4. **Synthesize and Merge**: When multiple findings address the same aspect of the question, synthesize them into a single, coherent point. If findings overlap, merge them to avoid redundancy.
5. **Handle Multi-Topic Queries**: If the question bundles unrelated topics (as in the "Multi-Topic Query Bundle" recipe), structure your answer with clear separation for each topic. Use paragraphs, headings, or transitional phrases to signal to the reader that you are switching topics. The goal is clarity and separation, not forced cohesion.
6. **Reference evidence naturally**: Present the supporting findings using natural phrasing like "Research shows that...", "Studies indicate that...", "Evidence suggests that..." without citing specific document IDs.
7. **Maintain accuracy**: Only include information supported by the provided findings. Do not introduce speculation, creative integration, or facts beyond what is given.
8. **Present, don't synthesize**: Your role is to present factual information, not to create new frameworks or hypothetical integrations. Stick to what is documented.
9. **Address the full question**: Cover all aspects requested while maintaining coherence and avoiding redundancy.

Write your synthesized answer inside <answer> tags. Your response should contain only the final synthesized answer - do not include analysis of the recipe configuration or commentary on the synthesis process.
"""


class PostprocessingError(Exception):
    """Raised when an LLM response fails validation during post-processing."""


def build_question_prompt(
    recipe: str,
    sub_questions: list[str],
    sub_answers: list[str],
    rng: Any = None,
    data_source_description: str | None = None,
) -> str:
    """Construct the question synthesis prompt for a grouped multi-step QA bundle.

    Returns:
        A tuple of (prompt, temperature) where temperature is sampled from the recipe's range.
    """
    qa_pair_blocks = []
    for question, answer in zip(sub_questions, sub_answers):
        qa_pair_blocks.append(
            "<qa_pair>\n"
            f"  <question>{question}</question>\n"
            f"  <answer>{answer}</answer>\n"
            "</qa_pair>"
        )
    qa_pairs_formatted = "\n".join(qa_pair_blocks)

    template = QUESTION_PROMPTS.get(recipe, QUESTION_PROMPTS[DEFAULT_RECIPE_KEY])
    prompt = template.format(qa_pairs=qa_pairs_formatted)
    return format_prompt_with_description(prompt, data_source_description)


def build_answer_prompt(
    recipe: str,
    final_question: str,
    sub_questions: list[str],
    sub_answers: list[str],
    rng: Any = None,
    data_source_description: str | None = None,
) -> str:
    """Construct the answer synthesis prompt using the final question and supporting findings.

    Returns:
        A tuple of (prompt, temperature) where temperature is sampled from the recipe's range.
    """
    answer_goal = ANSWER_GOALS.get(recipe, ANSWER_GOALS[DEFAULT_RECIPE_KEY])
    recipe_label = recipe.replace("_", " ").title()
    recipe_config = f"Recipe: {recipe_label}\nAnswer Goal: {answer_goal}"

    # Format supporting findings
    findings_lines = []
    for idx, (question, answer) in enumerate(zip(sub_questions, sub_answers), start=1):
        findings_lines.append(
            f"<finding>\n  <index>{idx}</index>\n  <sub_question>{question}</sub_question>\n"
            f"  <sub_answer>{answer}</sub_answer>\n</finding>"
        )
    supporting_findings = "\n".join(findings_lines)

    prompt = MULTISTEP_ANSWER_PROMPT.format(
        recipe_config=recipe_config,
        final_question=final_question,
        supporting_findings=supporting_findings,
    )

    return format_prompt_with_description(prompt, data_source_description)


def extract_question_from_text(response: str) -> str:
    """Extract the final question from the LLM response."""
    candidate_tags = ("question", "complex_question", "thematic_question")
    for tag in candidate_tags:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            question = match.group(1).strip()
            if not question:
                raise PostprocessingError("Generated question is empty")
            return question
    raise PostprocessingError(
        "No supported question tag (<question>, <complex_question>, <thematic_question>) found in the LLM response"
    )


def extract_answer_from_text(response: str) -> str:
    """Extract the final answer from the LLM response."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if not match:
        raise PostprocessingError("No <answer> tag found in the LLM response")
    answer = match.group(1).strip()
    if not answer:
        raise PostprocessingError("Generated answer is empty")
    return answer


__all__ = [
    "ANSWER_GOALS",
    "build_question_prompt",
    "build_answer_prompt",
    "extract_question_from_text",
    "extract_answer_from_text",
    "format_prompt_with_description",
]
