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
from .stat_calculator import StatCalculator

log = logging.getLogger("lm_polygraph")


def replace_quotes(text: str) -> str:
    return re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', text)


@dataclass
class Claim:
    claim_text: str
    decoded_claim: str
    sentence: str
    aligned_token_ids: List[int]


class ClaimModel(BaseModel):
    claims: List[str]


class ClaimSentence(BaseModel):
    sentence: str
    related_words: List[str]


class ClaimSentences(BaseModel):
    sentences: List[ClaimSentence]


class ClaimLabels(BaseModel):
    faithful: bool
    factual: bool
    explanation: str


class ClaimsExtractor(StatCalculator):
    """
    Extracts claims from the text of the model generation.
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
        super().__init__()
        log.info(f"Initializing ClaimsExtractor with language={language}")
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
        start_time = time.time()
        stupid_claims = [
            "Mary is a five-year old girl.",
            "Mary likes playing piano.",
            "Mary doesn't like cookies.",
        ]

        sent_list = []
        match_str_list = []

        uniq_sentences: List[str] = []
        claim_list = doc2sentences(doc=text, mode="claims", schema=ClaimModel).claims

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

            while not text[sent_start_idx:].startswith(s):
                sent_start_idx += 1
            while not text[:sent_end_idx].endswith(s):
                sent_end_idx += 1

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
        sent_pos = 0
        match_words_pos = 0
        match_str = ""
        while sent_pos < len(sent):
            check_boundaries = False
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
        sent_pos = 0
        cur_token_i = 0
        aligned_token_ids = []
        while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
            cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
            if len(cur_token_text) == 0:
                cur_token_i += 1
                continue
            if sent[sent_pos:].startswith(cur_token_text):
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
            print("ms:", ms)
            print("sent:", sent)
            print("parsed_words:", parsed_words)
            print("-----------------")

            if len(parsed_words) > curr_len_pars:
                curr_len_pars = len(parsed_words)
                best_sent = sent

            sents.append(sent)
            match_string[text_pos : text_pos + len(sent)] = list(ms)

        return "".join(match_string), best_sent
