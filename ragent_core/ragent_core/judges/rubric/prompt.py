from ragent_core.judges.criteria import criterion_id

RUBRIC_PROMPT = """Given a task, a response, and grading criteria, determine whether the response satisfies each criterion.

Task:
```
{question}
```

Response:
```
{response}
```

Criteria:
{criteria}

The only allowed verdict values are {positive_verdict} and {negative_verdict}.

For each criterion ID, provide a brief explanation of your assessment and the verdict that best matches it.

Do not forget the required outer <criteria> root: the response must begin with <criteria> and end with </criteria>.
Never return bare <criterion> elements without that outer root.

Return exactly one evaluation per criterion, using each criterion's exact ID. Each
<criterion> must contain exactly one <id>, one <reason>, and one <verdict> child, in
that order. Always use the <reason> tag for the explanation; never use <explanation>,
<rationale>, or any other substitute. The <verdict> text must be exactly
{positive_verdict} or {negative_verdict}.

The number of <criterion> children must equal the number of supplied criteria: one
supplied criterion requires one child, and four supplied criteria require four
children. For example, a batch of two criteria has this XML structure:

<criteria>
  <criterion>
    <id>{example_first_id}</id>
    <reason>Brief explanation of the assessment</reason>
    <verdict>{positive_verdict}</verdict>
  </criterion>
  <criterion>
    <id>{example_second_id}</id>
    <reason>Brief explanation of the assessment</reason>
    <verdict>{negative_verdict}</verdict>
  </criterion>
</criteria>
""".replace("{example_first_id}", criterion_id(1)).replace(
    "{example_second_id}", criterion_id(2)
)
