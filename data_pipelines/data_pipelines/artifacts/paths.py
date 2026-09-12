import re


def output_slug(value: str, *, fallback: str) -> str:
    """Bound an artifact directory label while retaining readable source names."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or fallback)[:80]
