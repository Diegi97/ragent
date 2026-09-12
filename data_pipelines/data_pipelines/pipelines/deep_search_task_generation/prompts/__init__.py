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
