import verifiers as vf

PROMPT = """You are tasked with evaluating the correctness of an AI-generated response to a given question. Your goal is to determine whether the generated response is correct according to the provided ground truth answer.

Here is the question:
<question>
{question}
</question>

Here is the ground truth answer:
<ground_truth>
{answer}
</ground_truth>

Here is the generated response:
<generated_response>
{response}
</generated_response>

Compare the generated response to the ground truth answer, considering only the information relevant to the question asked. Treat the ground truth as authoritative and do not use outside knowledge in your evaluation.

Guidelines for determining correctness:
1. The generated response must contain all important information from the ground truth.
2. It must not contradict the ground truth in any way.
3. Focus on semantic meaning rather than exact wording.
4. Ignore differences in casing, punctuation, articles, whitespace, and diacritics.
5. Accept synonyms, aliases, acronyms, singular/plural forms, and lemmas.
6. Normalize numbers and accept mathematically equivalent forms and unit conversions.
7. Be lenient with date/time format variants that denote the same moment.
8. Hedging and guessing are acceptable if the core information is correct.
9. The generated response only needs to address the information requested in the question, even if the ground truth contains additional details.
10. Do not penalize omissions of information that can be clearly inferred from the question.
11. For numerical answers, require exact matches unless approximation is implied. Allow ±1–2% or the nearest rounding step for approximate values.
12. Extra information is allowed if it doesn't add specific, unsupported factual claims.
13. For identifiers, codes, emails, exact titles, file paths, or option letters, require an exact match (case-insensitive unless case is semantically meaningful).
14. If the ground truth indicates no answer is available, the generated response should convey a similar meaning.

Examples:
Q: "Where is OpenAI headquartered?"
Ground truth: "San Francisco, California."
Generated: "San Francisco." → CORRECT

Q: "Give two Python web frameworks."
Ground truth: "Django; Flask."
Generated: "Django, Flask, Rails." → INCORRECT

Q: "What's 3/8 as a decimal?"
Ground truth: "0.375."
Generated: "0.38" → INCORRECT
Generated: "0.3750" → CORRECT

Q: "What is the user's email address?"
Ground truth: "Not provided."
Generated: "alice@example.com" → INCORRECT
Generated: "The email address is not provided in the document." → CORRECT

After carefully comparing the generated response to the ground truth, provide your evaluation. First, explain your reasoning, then give your final judgment.

Write your explanation inside <explanation> tags, followed by your judgment (either CORRECT or INCORRECT) inside <judgment> tags.

Your final output should only include the explanation and judgment tags, without any additional text or repetition of the input."""

RESPONSE_PARSER = vf.XMLParser(
    fields=["explanation", "judgment"],
    answer_field="judgment",
)