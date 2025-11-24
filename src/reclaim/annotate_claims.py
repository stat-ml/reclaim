from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .claim_level_prompts import ANNOTATION_PROMPT
from .openai_client import OpenAIChat

class ClaimsAnnotator:
    def __init__(
            self,
            openai_model: str = "gpt-4o",
            progress_bar: bool = True,
            n_threads: int = 1,
    ):
        self.openai_chat = OpenAIChat(openai_model=openai_model)
        self.progress_bar = progress_bar
        self.n_threads = n_threads


    def annotate_claims(
        self,
        claims: list[str],
        contexts: list[str],
        language: str = "en",
    ):
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
        prompt = ANNOTATION_PROMPT[language]
        return self.openai_chat.ask(prompt.format(context=context, claim=claim))
