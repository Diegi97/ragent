prompt = """You are tasked with extracting concepts of varying importance levels from a set of documents to facilitate the creation of a question-answer generator. Your goal is to identify concepts across different importance levels - important, medium importance, and less important - extracting them literally as they appear in the text.

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

4. Concepts can appear in just one document - there is no requirement for cross-document appearance.

5. Organize your output in the following structure. For the 'aliases' field, provide a comma-separated list of alternative names, synonyms, or related terms that could refer to the same concept. Keep aliases short and clear as well. For example, if 'dolor' is a concept, aliases might include 'pain', 'sufrimiento'. If 'ferritina' appears, aliases could be 'ferritin', 'hierro sérico'. If 'ayuno intermitente' is extracted, aliases could be 'intermittent fasting', 'IF':

<concepts>
  <concept>
    <name>Exact text as it appears in document</name>
    <importance>important|medium|less_important</importance>
    <aliases>Alternative Name 1, Alternative Name 2, Alternative Name 3</aliases>
    <document>Document number where this concept appears</document>
  </concept>
  <!-- Repeat for each concept -->
</concepts>

6. After the concepts list, provide a brief summary of the overall themes represented across all importance levels.

Your final output should only include the <concepts> section with the structured list of concepts and the brief summary of themes. Do not include your thought process or any other content outside of these elements."""