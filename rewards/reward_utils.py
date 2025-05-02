import re
from typing import List


def completions_to_lst(completions) -> List[str]:
    if (
        isinstance(completions, list)
        and all(isinstance(c, list) for c in completions)
        and all(
            len(c) > 0 and isinstance(c[0], dict) and "content" in c[0]
            for c in completions
        )
    ):
        return [completion[0]["content"] for completion in completions]
    elif isinstance(completions, list) and all(isinstance(c, str) for c in completions):
        return completions
    else:
        try:
            # Attempt basic conversion for other iterable types
            return [str(c) for c in completions]
        except Exception as e:
            # Log the error type and the problematic data structure
            # import logging
            # logging.error(f"Could not convert completions of type {type(completions)}: {e}")
            # Depending on strictness, either raise error or return empty list
            raise ValueError(
                f"Could not convert completions to a list of strings. Input type: {type(completions)}"
            )


def extract_tag_content(text: str, tag: str, strip_content: bool = True) -> str | None:
    """Extracts content between the first occurrence of <tag> and </tag>."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_index = text.find(start_tag)
    if start_index == -1:
        return None
    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    if end_index == -1:
        return None  # Tag opened but not closed properly
    if strip_content:
        return text[start_index:end_index].strip()  # Return stripped content
    else:
        return text[start_index:end_index]  # Return unstripped content
