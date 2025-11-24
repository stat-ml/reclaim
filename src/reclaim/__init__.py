"""ReClaim core package."""

from importlib import metadata
from typing import List

from .decompose import doc2sentences
from .extract_claims import Claim, ClaimModel, ClaimsExtractor
from .annotate_claims import ClaimsAnnotator
from .openai_client import OpenAIChat

try:
    __version__ = metadata.version("reclaim")
except metadata.PackageNotFoundError:  # pragma: no cover - resolved at runtime
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "extract_claims",
    "extract_and_align_claims",
    "batch_extract_and_align_claims",
    "doc2sentences",
    "Claim",
]


def extract_claims(text: str, model: str = "gpt-4o") -> List[Claim]:
    """
    Extract atomic claims from plain text.
    """
    result = doc2sentences(doc=text, mode="atomic_claims", model=model, schema=ClaimModel)
    claim_texts = result.claims if isinstance(result, ClaimModel) else result
    return [
        Claim(
            claim_text=claim_text,
            decoded_claim="",
            sentence="",
            aligned_token_ids=[],
        )
        for claim_text in claim_texts
    ]


def extract_and_align_claims(
    text,
    tokens,
    tokenizer,
    openai_model: str = "gpt-4o",
    progress_bar: bool = True,
    n_threads: int = 1,
):
    """
    Extract and align claims with token-level provenance from model output tokens.
    """
    extractor = ClaimsExtractor(
        openai_chat=OpenAIChat(openai_model=openai_model),
        progress_bar=progress_bar,
        n_threads=n_threads,
    )
    return extractor.claims_from_text(text, tokens, tokenizer)


def batch_extract_and_align_claims(
    texts: List[str],
    tokens: List[List[int]],
    tokenizer,
    openai_model: str = "gpt-4o",
    progress_bar: bool = True,
    n_threads: int = 1,
) -> List[List[Claim]]:
    """
    Batch extract and align claims with token-level provenance from model output tokens.
    """
    extractor = ClaimsExtractor(
        openai_chat=OpenAIChat(openai_model=openai_model),
        progress_bar=progress_bar,
        n_threads=n_threads,
    )

    return extractor.batch_claims_from_texts(texts, tokens, tokenizer)


def annotate_claims(
    claims: List[str],
    contexts: List[str],
    openai_model: str = "gpt-4o",
    progress_bar: bool = True,
    n_threads: int = 1,
):
    """
    Annotate claims with labels.
    """
    annotator = ClaimsAnnotator(
        openai_model=openai_model,
        progress_bar=progress_bar,
        n_threads=n_threads,
    )

    return annotator.annotate_claims(claims, contexts)
