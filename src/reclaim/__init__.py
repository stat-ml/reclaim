"""ReClaim core package."""

from importlib import metadata
from typing import List, Optional

from .decompose import doc2sentences
from .extract_claims import (
    Claim,
    ClaimModel,
    ClaimPostprocessingConfig,
    ClaimsExtractor,
)
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
    "ClaimPostprocessingConfig",
]


def _default_postprocess_config(
    override: Optional[ClaimPostprocessingConfig],
    enable_defaults: bool,
) -> Optional[ClaimPostprocessingConfig]:
    """
    Resolve a postprocessing config:
    - honor explicit override;
    - if allowed, provide a sensible default bundle;
    - otherwise, return None to disable all extras.
    """
    if override is not None:
        return override
    if not enable_defaults:
        return None
    return ClaimPostprocessingConfig(
        rewrite_pronouns=True,
        sanitize_with_llm=True,
        split_non_atomic=True,
        dedupe_with_encoder=True,
        dedupe_with_cosine=True,
    )


def extract_claims(
    text: str,
    model: str = "gpt-4.1",
    postprocess_config: Optional[ClaimPostprocessingConfig] = None,
    enable_default_postprocessing: bool = True,
) -> List[Claim]:
    """
    Extract atomic claims from plain text.

    By default, enables post-processing (pronoun rewrite, LLM sanitization,
    encoder + BoW dedupe). Pass a custom ClaimPostprocessingConfig or set
    enable_default_postprocessing=False to skip these steps.
    """
    result = doc2sentences(doc=text, mode="atomic_claims", model=model, schema=ClaimModel)
    claim_texts = result.claims if isinstance(result, ClaimModel) else result
    chat = OpenAIChat(openai_model=model)
    config = _default_postprocess_config(postprocess_config, enable_default_postprocessing)
    extractor = ClaimsExtractor(
        openai_chat=chat,
        postprocess_config=config,
    )
    claim_texts = extractor.postprocess_claims(claim_texts, text)
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
    openai_model: str = "gpt-4.1",
    progress_bar: bool = True,
    n_threads: int = 1,
    postprocess_config: Optional[ClaimPostprocessingConfig] = None,
    enable_default_postprocessing: bool = True,
):
    """
    Extract and align claims with token-level provenance from model output tokens.

    By default, enables post-processing (pronoun rewrite, LLM sanitization,
    encoder + BoW dedupe). Pass a custom ClaimPostprocessingConfig or set
    enable_default_postprocessing=False to skip these steps.
    """
    config = _default_postprocess_config(postprocess_config, enable_default_postprocessing)
    extractor = ClaimsExtractor(
        openai_chat=OpenAIChat(openai_model=openai_model),
        progress_bar=progress_bar,
        n_threads=n_threads,
        postprocess_config=config,
    )
    return extractor.claims_from_text(text, tokens, tokenizer)


def batch_extract_and_align_claims(
    texts: List[str],
    tokens: List[List[int]],
    tokenizer,
    openai_model: str = "gpt-4.1",
    progress_bar: bool = True,
    n_threads: int = 1,
    postprocess_config: Optional[ClaimPostprocessingConfig] = None,
    enable_default_postprocessing: bool = True,
) -> List[List[Claim]]:
    """
    Batch extract and align claims with token-level provenance from model output tokens.

    By default, enables post-processing (pronoun rewrite, LLM sanitization,
    encoder + BoW dedupe). Pass a custom ClaimPostprocessingConfig or set
    enable_default_postprocessing=False to skip these steps.
    """
    config = _default_postprocess_config(postprocess_config, enable_default_postprocessing)
    extractor = ClaimsExtractor(
        openai_chat=OpenAIChat(openai_model=openai_model),
        progress_bar=progress_bar,
        n_threads=n_threads,
        postprocess_config=config,
    )

    return extractor.batch_claims_from_texts(texts, tokens, tokenizer)


def annotate_claims(
    claims: List[str],
    contexts: List[str],
    openai_model: str = "gpt-4.1",
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
