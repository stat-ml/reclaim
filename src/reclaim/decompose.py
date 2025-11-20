from typing import List, Optional

from pydantic import BaseModel

from .openai_client import gpt
from .prompts import (
    DOC_TO_ATOMIC_CLAIMS_PROMPT,
    DOC_TO_INDEPEDENT_SENTENCES_PROMPT,
    DOC_TO_SENTENCES_PROMPT,
    SENTENCES_TO_CLAIMS_PROMPT,
)


def doc2sentences(
    doc: str,
    mode: str = "independent_sentences",
    model: str = "gpt-4o",
    system_role: str = "You are good at decomposing and decontextualizing text.",
    num_retries: int = 5,
    schema: Optional[BaseModel] = None,
) -> List[str]:
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

    results = None
    user_input = prompt.format(doc=doc).strip()
    for _ in range(num_retries):
        try:
            results = gpt(user_input, model=model, system_role=system_role, schema=schema)
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
    return results
