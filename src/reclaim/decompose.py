from typing import List, Optional

from pydantic import BaseModel

from .openai_client import OpenAIChat
from .prompts import (
    DOC_TO_ATOMIC_CLAIMS_PROMPT,
    DOC_TO_INDEPEDENT_SENTENCES_PROMPT,
    DOC_TO_SENTENCES_PROMPT,
    SENTENCES_TO_CLAIMS_PROMPT,
)


def doc2sentences(
    doc: str,
    mode: str = "independent_sentences",
    model: str = "gpt-4.1",
    system_role: str = "You are good at decomposing and decontextualizing text.",
    num_retries: int = 5,
    schema: Optional[BaseModel] = None,
) -> List[str]:
    """
    Decompose a document into sentences or claims using an OpenAI prompt.

    The function is a thin wrapper around :class:`OpenAIChat` that:
    - Chooses a prompt template based on ``mode``.
    - Formats the document into the prompt.
    - Invokes the chat model (optionally with a schema) to obtain structured
      text slices or claims. Parsing is delegated to Pydantic when a schema
      is provided, otherwise raw strings are returned.

    Args:
        doc: Source document to split.
        mode: Extraction strategy; controls which prompt is used.
        model: OpenAI model name.
        system_role: System prompt passed to the chat model.
        num_retries: Number of attempts if the request fails.
        schema: Optional Pydantic schema to parse structured responses.

    Returns:
        Parsed result from the LLM, often a list of strings or a Pydantic model.
    """
    if mode == "sentences":
        prompt = DOC_TO_SENTENCES_PROMPT
    elif mode == "independent_sentences":
        prompt = DOC_TO_INDEPEDENT_SENTENCES_PROMPT
    elif mode == "claims":
        prompt = SENTENCES_TO_CLAIMS_PROMPT
    elif mode == "atomic_claims":
        prompt = DOC_TO_ATOMIC_CLAIMS_PROMPT
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Render the selected prompt with the source document.
    user_input = prompt.format(doc=doc).strip()
    chat = OpenAIChat(openai_model=model)
    results = chat.ask(
        message=user_input,
        schema=schema,
        system_role=system_role,
    )
    return results
