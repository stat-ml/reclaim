import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from tqdm import tqdm

from .claim_level_prompts import CLAIM_EXTRACTION_PROMPTS, MATCHING_PROMPTS
from .decompose import doc2sentences
from .openai_client import OpenAIChat
from .prompts import MATCHING_PROMPT

log = logging.getLogger("lm_polygraph")


def replace_quotes(text: str) -> str:
    """Normalize single-quoted spans to double quotes for matching stability."""
    return re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', text)


@dataclass
class Claim:
    """
    Structured claim with optional token-level alignment information.

    - claim_text: normalized claim text returned by the LLM.
    - decoded_claim: text reconstructed from the aligned tokens; useful for debugging tokenizer drift.
    - sentence: source sentence used for alignment.
    - aligned_token_ids: indices into the model output tokens that support the claim.
    """
    claim_text: str
    decoded_claim: str
    sentence: str
    aligned_token_ids: List[int]


class ClaimModel(BaseModel):
    """Pydantic model describing a list of extracted claims."""
    claims: List[str]


class ClaimSentence(BaseModel):
    """Sentence with related words used for alignment."""
    sentence: str
    related_words: List[str]


class ClaimSentences(BaseModel):
    """Wrapper for a list of `ClaimSentence` objects."""
    sentences: List[ClaimSentence]


class ClaimsExtractor:
    """
    Extract claims from model-generated text and align them back to token spans.

    The extractor drives a two-stage LLM workflow:
    1) Claim discovery: ask the model to enumerate claims in the text.
    2) Claim-to-text alignment: for each claim, ask the model for supporting words,
       convert them to character masks, and map those characters to token indices.

    Alignment is character-driven: we build a caret mask marking matched characters,
    then walk decoded tokens to flag any token that overlaps a caret. This approach
    is tokenizer-agnostic and resilient to subword boundaries.
    """

    def __init__(
        self,
        openai_chat: OpenAIChat,
        sent_separators: str = ".?!。？！\n",
        language: str = "en",
        progress_bar: bool = False,
        extraction_prompts: Dict[str, str] = CLAIM_EXTRACTION_PROMPTS,
        matching_prompts: Dict[str, str] = MATCHING_PROMPTS,
        n_threads: int = 1,
    ):
        """
        Initialize the extractor with prompt templates and chat backend.

        Args:
            openai_chat: Chat client used for LLM calls.
            sent_separators: Characters treated as sentence boundaries.
            language: Language code used to pick prompts.
            progress_bar: Whether to display tqdm progress bars.
            extraction_prompts: Templates used for claim extraction.
            matching_prompts: Templates used for aligning claims to text spans.
            n_threads: Maximum worker threads for batch operations.
        """
        self.language = language
        self.openai_chat = openai_chat
        self.sent_separators = sent_separators
        self.progress_bar = progress_bar
        self.extraction_prompts = extraction_prompts
        self.matching_prompts = matching_prompts
        self.n_threads = n_threads

    def batch_claims_from_texts(
        self,
        texts: List[str],
        tokens: List[List[int]],
        tokenizer,
    ) -> Dict[str, List]:
        """
        Extract and align claims for multiple texts concurrently.

        Args:
            texts: Raw generated texts to analyze.
            tokens: Token id sequences corresponding to each text.
            tokenizer: Tokenizer matching the token ids.

        Returns:
            List of aligned claims per input text, preserving order.
        """

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            claims = list(
                tqdm(
                    executor.map(
                        self.claims_from_text,
                        texts,
                        tokens,
                        [tokenizer] * len(texts),
                    ),
                    total=len(texts),
                    desc="Extracting claims",
                    disable=not self.progress_bar,
                )
            )

        return claims

    def claims_from_text(self, text: str, tokens: List[int], tokenizer) -> List[Claim]:
        """
        Extract claims from a single text and align them to token ids.

        Workflow overview:
        - Ask the LLM to enumerate claims from ``text``.
        - For each claim, ask the LLM for related words within the text and build
          a character-level mask of those words.
        - Convert character masks to token indices so callers can attribute claims
          to spans in the generated output.

        Args:
            text: Model-generated text to analyze.
            tokens: Token ids that produced the text.
            tokenizer: Tokenizer used to decode token ids.

        Returns:
            List of `Claim` objects with token-level provenance.
        """
        start_time = time.time()
        stupid_claims = [
            "Mary is a five-year old girl.",
            "Mary likes playing piano.",
            "Mary doesn't like cookies.",
        ]
        # Guardrail cleanup: drop known garbage claims sometimes returned by the prompt.

        sent_list = []
        match_str_list = []

        uniq_sentences: List[str] = []
        claim_list = doc2sentences(doc=text, mode="claims", schema=ClaimModel).claims

        for s in stupid_claims:
            if s in claim_list:
                claim_list.remove(s)

        missed_text = (
            'Original text:\n' +
            text +
            '\n\nExtracted claims:\n' +
            '\n-'.join(claim_list)
        )
        missed_claims = doc2sentences(
            doc=missed_text,
            mode="enrich_claims",
            schema=ClaimModel
            ).claims

        claimed_text = ' '.join(claim_list + missed_claims)

        claim_list = doc2sentences(
            doc=claimed_text,
            mode="claims",
            schema=ClaimModel
        ).claims
        for s in stupid_claims:
            if s in claim_list:
                claim_list.remove(s)

        print("claim_list:", claim_list)

        final_claims: List[Claim] = []
        claims_from_sent: Dict[str, List[str]] = {}
        match_str_dict: Dict[str, Dict[str, str]] = {}

        N_TRIES = 10
        MAX_MATCHED_WORDS = 30

        for claim in claim_list:
            match_str = None
            # The matching prompt can be flaky; retry a handful of times to
            # coax usable word lists and spans from the LLM.
            for _ in range(N_TRIES):
                try:
                    match_str, sent = self.match_claim(text, claim, MAX_MATCHED_WORDS)
                    if match_str is not None:
                        break
                except Exception as e:
                    print("Error: " + repr(e))
                    continue
            if match_str is not None:
                match_str_list.append(match_str)
                sent_list.append(sent)

                if sent not in uniq_sentences:
                    uniq_sentences.append(sent)
                    claims_from_sent[sent] = [claim]
                    match_str_dict[sent] = {claim: match_str}
                else:
                    claims_from_sent[sent].append(claim)
                    match_str_dict[sent][claim] = match_str

        for s in tqdm(uniq_sentences, desc="Unique sentences"):
            sent_start_token_idx, sent_end_token_idx = 0, 0
            sent_start_idx, sent_end_idx = 0, 0

            # Identify the exact character span of the sentence in the full text.
            while not text[sent_start_idx:].startswith(s):
                sent_start_idx += 1
            while not text[:sent_end_idx].endswith(s):
                sent_end_idx += 1

            # Walk decoded prefixes until we cover the sentence boundaries; this
            # yields the token window corresponding to the sentence.
            while len(tokenizer.decode(tokens[:sent_start_token_idx])) < sent_start_idx:
                sent_start_token_idx += 1
            while len(tokenizer.decode(tokens[:sent_end_token_idx])) < sent_end_idx:
                sent_end_token_idx += 1

            claims = claims_from_sent[s]
            sent_tokens = tokens[sent_start_token_idx:sent_end_token_idx]

            for claim in tqdm(claims, desc="Claims"):
                if match_str_dict[s][claim] is not None:
                    match_string = match_str_dict[s][claim][
                        sent_start_idx:sent_end_idx
                    ]
                    aligned_token_ids = self._align(s, match_string, sent_tokens, tokenizer)
                    if len(aligned_token_ids) == 0:
                        # If alignment fails, skip rather than emitting partial provenance.
                        continue

                    for i in range(len(aligned_token_ids)):
                        aligned_token_ids[i] += sent_start_token_idx

                    decoded_claim = tokenizer.decode([tokens[i] for i in aligned_token_ids])
                    print("claim:", claim)
                    print("decoded_claim:", decoded_claim)

                    final_claims.append(
                        Claim(
                            claim_text=claim,
                            decoded_claim=decoded_claim,
                            sentence=s,
                            aligned_token_ids=aligned_token_ids,
                        )
                    )

        print("len(claim_list):", len(claim_list))
        print("len(final_claims):", len(final_claims))
        print("--- %s seconds ---" % (time.time() - start_time))

        return final_claims

    def _claims_from_sentence(
        self,
        sent: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[Claim]:
        """
        Extract claims from a sentence and align them within that sentence.

        Args:
            sent: Sentence to process.
            sent_tokens: Token ids covering the sentence span.
            tokenizer: Tokenizer that produced the token ids.

        Returns:
            Claims aligned to spans inside the provided sentence.
        """
        extracted_claims = self.openai_chat.ask(
            self.extraction_prompts[self.language].format(sent=sent)
        )
        claims = []
        for claim_text in extracted_claims.split("\n"):
            if not claim_text.startswith("- "):
                continue
            if "there aren't any claims" in claim_text.lower():
                continue
            claim_text = claim_text[2:].strip()
            # Ask the model to surface specific words in the sentence that back the claim.
            chat_ask = self.matching_prompts[self.language].format(
                sent=sent,
                claim=claim_text,
            )
            match_words = self.openai_chat.ask(chat_ask)
            if self.language == "zh":
                match_words = match_words.strip().split(" ")
            else:
                match_words = match_words.strip().split(",")
            match_words = list(map(lambda x: x.strip(), match_words))
            if self.language == "zh":
                match_string = self._match_string_zh(sent, match_words)
            else:
                match_string = self._match_string(sent, match_words)
            if match_string is None:
                continue
            aligned_token_ids = self._align(sent, match_string, sent_tokens, tokenizer)
            if len(aligned_token_ids) == 0:
                continue
            claims.append(
                Claim(
                    claim_text=claim_text,
                    sentence=sent,
                    aligned_token_ids=aligned_token_ids,
                    decoded_claim="",
                )
            )
        return claims

    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
        """
        Build a mask string marking where each match word appears in the sentence.

        Args:
            sent: Lowercased sentence text.
            match_words: Ordered words expected to appear in the sentence.

        Returns:
            Mask string with carets over matched characters, or raises if unmatched.
        """
        sent_pos = 0
        match_words_pos = 0
        match_str = ""
        while sent_pos < len(sent):
            check_boundaries = False
            # Only consider matches when current position is at a word boundary.
            if sent_pos == 0 or not sent[sent_pos - 1].isalpha():
                check_boundaries = True
            if check_boundaries and match_words_pos < len(match_words):
                cur_match_word = match_words[match_words_pos]
                right_idx = sent_pos + len(cur_match_word)
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()
                if check_boundaries and sent[sent_pos:].startswith(cur_match_word):
                    len_w = len(cur_match_word)
                    sent_pos += len_w
                    match_str += "^" * len_w
                    match_words_pos += 1
                    continue
            sent_pos += 1
            match_str += " "

        if match_words_pos < len(match_words):
            print("match_words:", match_words)
            print("sent:", sent)
            raise Exception("Not all words were matched, but loop ended.")

        return match_str

    def _match_string_zh(self, sent: str, match_words: List[str]) -> Optional[str]:
        """
        Build a mask string for Chinese text where match words are contiguous spans.

        The matching strategy is simpler than the English path: it expects each
        item in ``match_words`` to be a contiguous sequence of characters and
        walks the sentence linearly to mark those spans with carets.

        Args:
            sent: Sentence text.
            match_words: Ordered list of character spans to match.

        Returns:
            Mask string with carets on matched characters, or None if not all found.
        """
        last = 0
        last_match = 0
        match_str = ""

        for char in sent:
            if last_match < len(match_words) and char == match_words[last_match][last]:
                match_str += "^"
                last += 1
                if last == len(match_words[last_match]):
                    last = 0
                    last_match += 1
            else:
                match_str += " "

        if last_match < len(match_words):
            return None

        return match_str

    def _align(
        self,
        sent: str,
        match_str: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[int]:
        """
        Convert a character-level match mask into aligned token indices.

        Args:
            sent: Sentence text corresponding to the tokens.
            match_str: Mask with carets marking matched characters.
            sent_tokens: Token ids for the sentence.
            tokenizer: Tokenizer used to decode token ids.

        Returns:
            Token indices within `sent_tokens` that overlap matched characters.

        The decoder reads token-by-token, advancing the character pointer by the
        decoded token length whenever a prefix matches. When a token includes any
        caret in the mask, it is considered part of the claim's provenance.
        """
        sent_pos = 0
        cur_token_i = 0
        aligned_token_ids = []
        while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
            cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
            if len(cur_token_text) == 0:
                cur_token_i += 1
                continue
            if sent[sent_pos:].startswith(cur_token_text):
                # If any part of the decoded token overlaps a matched character,
                # consider the token aligned to the claim.
                if any(
                    t == "^"
                    for t in match_str[sent_pos : sent_pos + len(cur_token_text)]
                ):
                    aligned_token_ids.append(cur_token_i)
                cur_token_i += 1
                sent_pos += len(cur_token_text)
            else:
                sent_pos += 1
        return aligned_token_ids

    def match_claim(self, text: str, claim: str, max_parsed_words: int):
        """
        Find the best matching sentence for a claim and construct a match mask.

        Args:
            text: Full generated text.
            claim: Claim string to align.
            max_parsed_words: Maximum allowed matched words to avoid noisy matches.

        Returns:
            Tuple of (mask over full text, best matching sentence).
        """
        prompt = MATCHING_PROMPT
        q = prompt.format(text=text, claim=claim)
        res = self.openai_chat.ask(q, schema=ClaimSentences)

        text_pos = 0
        match_string = list(" " * len(text))
        sents: List[str] = []
        curr_len_pars = 0
        best_sent = text
        for r in res.sentences:
            sent, words = r.sentence, r.related_words
            sent, words = sent.strip(), [word.strip() for word in words]
            if sent.startswith('"') and sent.endswith('"'):
                sent = sent[1:-1]
            if sent not in text[text_pos:]:
                sent = replace_quotes(sent)
            assert sent in text[text_pos:]
            # Advance through the text to avoid matching earlier duplicate substrings.
            text_pos += text[text_pos:].find(sent)
            if "No related words" in words:
                continue
            parsed_words = []
            for w in words:
                if w.startswith('"') and w.endswith('"'):
                    w = w[1:-1]
                parsed_words.append(w.lower())
            if len(parsed_words) > max_parsed_words:
                raise Exception(
                    f"Error while matching {parsed_words} in {sent}: too many matched words, expected no more than {max_parsed_words}"
                )

            ms = self._match_string(sent.lower(), parsed_words)

            print("-----------------")
            print("claim:", claim)
            print("ms:  ", ms)
            print("sent:", sent)
            print("parsed_words:", parsed_words)
            print("-----------------")

            # Keep the sentence with the richest match as the best candidate.
            if len(parsed_words) > curr_len_pars:
                curr_len_pars = len(parsed_words)
                best_sent = sent

            sents.append(sent)
            match_string[text_pos : text_pos + len(sent)] = list(ms)

        return "".join(match_string), best_sent
