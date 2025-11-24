ReClaim API Reference
=====================

This reference is written for Read the Docs and mirrors the current code in ``src/reclaim``. It outlines the public surface area and the most relevant internal helpers so new contributors can understand how text is decomposed, aligned, and annotated.

Top-Level Package
-----------------

.. module:: reclaim

.. py:function:: extract_claims(text: str, model: str = "gpt-4o") -> List[Claim]

   Extract atomic claims from plain text using the LLM prompt flow in ``doc2sentences``. Returns ``Claim`` objects with only ``claim_text`` populated; alignment fields are left empty.

.. py:function:: extract_and_align_claims(text, tokens, tokenizer, openai_model: str = "gpt-4o", progress_bar: bool = True, n_threads: int = 1)

   Extract claims from generated text and align each claim back to the token ids that produced it. Internally builds a :class:`~reclaim.extract_claims.ClaimsExtractor` wired to an :class:`~reclaim.openai_client.OpenAIChat` instance.

.. py:function:: batch_extract_and_align_claims(texts: List[str], tokens: List[List[int]], tokenizer, openai_model: str = "gpt-4o", progress_bar: bool = True, n_threads: int = 1) -> List[List[Claim]]

   Batch wrapper over ``ClaimsExtractor.batch_claims_from_texts`` to process multiple model outputs in parallel. Returns a list of claim lists aligned to each input text.

.. py:function:: annotate_claims(claims: List[str], contexts: List[str], openai_model: str = "gpt-4o", progress_bar: bool = True, n_threads: int = 1)

   Annotate claims for faithfulness and factuality relative to their contexts by delegating to :class:`~reclaim.annotate_claims.ClaimsAnnotator`.

Data Models
-----------

.. module:: reclaim.extract_claims

.. py:class:: Claim(claim_text: str, decoded_claim: str, sentence: str, aligned_token_ids: List[int])

   Dataclass representing a single claim. ``claim_text`` holds the normalized claim, ``decoded_claim`` carries the tokenizer-decoded span, ``sentence`` stores the sentence that supported the claim, and ``aligned_token_ids`` lists token indices aligned to the claim.

.. py:class:: ClaimModel

   Pydantic model with ``claims: List[str]`` used to parse claim lists from LLM responses.

.. py:class:: ClaimSentence

   Pydantic model with ``sentence`` and ``related_words`` used when aligning claims to text spans.

.. py:class:: ClaimSentences

   Wrapper Pydantic model with ``sentences: List[ClaimSentence]``.

.. py:class:: ClaimLabels

   Pydantic model holding ``faithful``, ``factual``, and ``explanation`` attributes for annotation results.

Extraction and Alignment
------------------------

.. py:class:: ClaimsExtractor(openai_chat: OpenAIChat, sent_separators: str = ".?!。？！\\n", language: str = "en", progress_bar: bool = False, extraction_prompts: Dict[str, str] = CLAIM_EXTRACTION_PROMPTS, matching_prompts: Dict[str, str] = MATCHING_PROMPTS, n_threads: int = 1)

   High-level orchestrator for claim extraction and alignment.

   - Initializes with prompt templates, language, and threading options.
   - Uses :class:`~reclaim.openai_client.OpenAIChat` to call OpenAI for both claim extraction and matching.

   .. py:method:: batch_claims_from_texts(texts: List[str], tokens: List[List[int]], tokenizer) -> Dict[str, List]

      Runs :meth:`claims_from_text` across many texts concurrently via a ``ThreadPoolExecutor``. Returns aligned claims per input text, preserving order.

   .. py:method:: claims_from_text(text: str, tokens: List[int], tokenizer) -> List[Claim]

      Extract claims from a single generated text, identify supporting sentences, and align matched spans back to token indices. Filters out some known low-quality claims before alignment. Produces ``Claim`` objects with sentence provenance and token alignment.

   .. py:method:: _claims_from_sentence(sent: str, sent_tokens: List[int], tokenizer) -> List[Claim]

      Internal helper to extract claims from an individual sentence and align them within that sentence using prompt-driven matching.

   .. py:method:: _match_string(sent: str, match_words: List[str]) -> Optional[str]

      Build a caret mask over ``sent`` marking the positions of ordered ``match_words`` when they appear at word boundaries. Raises if not all words are found.

   .. py:method:: _match_string_zh(sent: str, match_words: List[str]) -> Optional[str]

      Chinese-specific matcher that marks contiguous spans of characters corresponding to each match word; returns ``None`` if any span is missing.

   .. py:method:: _align(sent: str, match_str: str, sent_tokens: List[int], tokenizer) -> List[int]

      Convert a character-level match mask into token indices by decoding tokens and checking for overlap with carets in ``match_str``.

   .. py:method:: match_claim(text: str, claim: str, max_parsed_words: int)

      Query the LLM for sentences and related words supporting ``claim``, construct a full-text caret mask, and pick the sentence with the richest match. Returns the mask and the best supporting sentence.

Annotation
----------

.. module:: reclaim.annotate_claims

.. py:class:: ClaimsAnnotator(openai_model: str = "gpt-4o", progress_bar: bool = True, n_threads: int = 1)

   Thin wrapper that labels claims for faithfulness and factuality.

   .. py:method:: annotate_claims(claims: list[str], contexts: list[str], language: str = "en")

      Annotate many claims in parallel against their paired contexts, returning model responses (typically boolean label pairs).

   .. py:method:: _annotate_claim(claim: str, context: str, language: str = "en")

      Internal worker invoked in the thread pool; sends the annotation prompt and returns the parsed result.

Text Decomposition
------------------

.. module:: reclaim.decompose

.. py:function:: doc2sentences(doc: str, mode: str = "independent_sentences", model: str = "gpt-4o", system_role: str = "You are good at decomposing and decontextualizing text.", num_retries: int = 5, schema: Optional[BaseModel] = None) -> List[str]

   Core utility that asks an LLM to decompose a document. The ``mode`` selects the appropriate prompt (plain sentences, independent sentences, claims, or atomic claims). Retries failed requests up to ``num_retries`` and optionally parses structured outputs via a Pydantic schema.

OpenAI Client Utilities
-----------------------

.. module:: reclaim.openai_client

.. py:function:: gpt_single_try(user_input: str, model: str = "gpt-4o", system_role: str = "You are a helpful assistant.", schema: Optional[BaseModel] = None)

   Send a single chat completion or ``responses.parse`` call. Returns raw string content or a parsed schema instance.

.. py:function:: gpt(user_input: str, model: str = "gpt-4o", system_role: str = "You are a helpful assistant.", num_retries: int = 3, waiting_time: int = 1, schema: Optional[BaseModel] = None)

   Convenience wrapper that retries ``gpt_single_try`` on ``openai`` errors with a short sleep between attempts.

.. py:class:: OpenAIChat(openai_model: str = "gpt-4o", base_url: Optional[str] = None, cache_path: str = "~/.cache", timeout: int = 600, max_tokens: Optional[int] = None, rewrite_cache: bool = False)

   Lightweight chat client used across extraction and annotation.

   - Reads ``OPENAI_API_KEY`` from the environment.
   - Stores cache path metadata (no cache implementation included here).
   - Supports optional base URL, timeout, and completion length limits.

   .. py:method:: ask(message: str, schema: Optional[BaseModel] = None)

      Send a prompt to OpenAI with a fixed system role. If ``schema`` is provided, uses ``responses.parse``; otherwise returns the string content while filtering common boilerplate refusals.

   .. py:method:: _send_request(messages, schema: Optional[BaseModel] = None)

      Internal request dispatcher with progressive backoff across several wait intervals.

Stat Calculator
---------------

.. module:: reclaim.stat_calculator

.. py:class:: StatCalculator

   Minimal placeholder included for compatibility with upstream projects. Implements ``meta_info`` to return empty metadata and no-op initialization.
