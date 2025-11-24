from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .claim_level_prompts import ANNOTATION_PROMPT
from .openai_client import OpenAIChat

class ClaimsAnnotator:
    """Annotate claims for faithfulness and factuality using an LLM backend."""

    def __init__(
            self,
            openai_model: str = "gpt-4o",
            progress_bar: bool = True,
            n_threads: int = 1,
    ):
        """
        Initialize the annotator.

        Args:
            openai_model: Model name passed to the OpenAI client.
            progress_bar: Whether to show tqdm progress bars during annotation.
            n_threads: Number of worker threads for concurrent annotation.
        """
        self.openai_chat = OpenAIChat(openai_model=openai_model)
        self.progress_bar = progress_bar
        self.n_threads = n_threads


    def annotate_claims(
        self,
        claims: list[str],
        contexts: list[str],
        language: str = "en",
    ):
        """
        Annotate many claims in parallel against their contexts.

        Args:
            claims: Claims to evaluate.
            contexts: Context strings aligned with each claim.
            language: Language code selecting the prompt template.

        Returns:
            List of label tuples produced by the model.
        """
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            labels = list(
                tqdm(
                    executor.map(
                        self._annotate_claim,
                        claims,
                        contexts,
                        [language] * len(claims),
                    ),
                    total=len(claims),
                    desc="Annotating claims",
                    disable=not self.progress_bar,
                )
            )

        return labels

    def _annotate_claim(
        self,
        claim: str,
        context: str,
        language: str = "en",
    ):
        """
        Annotate a single claim against a context string.

        Args:
            claim: Claim text to judge.
            context: Reference context for evaluation.
            language: Language code selecting the prompt template.

        Returns:
            Model response containing faithfulness and factuality labels.
        """
        prompt = ANNOTATION_PROMPT[language]
        return self.openai_chat.ask(prompt.format(context=context, claim=claim))
