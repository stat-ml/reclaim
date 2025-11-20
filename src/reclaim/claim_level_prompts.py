from .prompts import MATCHING_PROMPT

# Lightweight prompt dictionaries keyed by language to mirror original structure.
CLAIM_EXTRACTION_PROMPTS = {
    "en": "List all atomic claims from the following sentence. Return each claim on a new line starting with '- '. Sentence: {sent}",
}

MATCHING_PROMPTS = {
    "en": MATCHING_PROMPT,
}
