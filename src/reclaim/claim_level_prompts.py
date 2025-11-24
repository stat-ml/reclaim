from .prompts import MATCHING_PROMPT

# Lightweight prompt dictionaries keyed by language to mirror original structure.
CLAIM_EXTRACTION_PROMPTS = {
    "en": "List all atomic claims from the following sentence. Return each claim on a new line starting with '- '. Sentence: {sent}",
}

MATCHING_PROMPTS = {
    "en": MATCHING_PROMPT,
}

ANNOTATION_PROMPT = {
    "en": """
Evaluate the given claim using two criteria: **faithfulness** and **factuality**.

- **Faithfulness** assesses how accurately the claim reflects the *context document*. Assign one of the following labels:
  - "faithful" — The claim is directly supported by the context.
  - "unfaithful-contra" — The claim directly contradicts the context.
  - "unfaithful-neutral" — The claim is not present in or supported by the context.

- **Factuality** assesses the truth of the claim *independently of the context*, based on the most up-to-date and reliable sources of knowledge available to humanity. Assign one of the following labels:
  - "True" — The claim is factually correct.
  - "False" — The claim is factually incorrect.
  - "unverifiable" — The truth of the claim cannot be determined with current knowledge.

Return your answer in the exact format:
("faithfulness label", "factuality label")


Context Document: {context}  


Claim: {claim}  


Label:
"""
}
