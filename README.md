# ReClaim

ReClaim is a Python library for decomposing complex texts into atomic claims, enabling structured analysis and verification of information. A claim is a minimal unit of information that can be independently evaluated for truthfulness.

ReClaim allows to extract claims both from human-written texts and from LLM-generated content. For LLM-generated content, ReClaim can also map each claim back to the specific tokens in the original text that support it. Extracted claims are decontextualized to ensure they can be further processed without reference to the source text.


## Requirements
- Python 3.10 or newer.
- `uv` 0.4+ for managing the virtual environment (recommended, but any PEP 517 build tool works).

## Installation
```bash
# via uv (recommended)
uv pip install reclaim

# or via pip
pip install reclaim
```

## Usage
For simple claim extraction from text, use the `extract_claims` function:

```python
from reclaim import extract_claims

text = """
The Eiffel Tower is located in Paris. It was completed in 1889 and stands at a height of 324 meters.
"""

claims = extract_claims(text)

for claim in claims:
    print(f"Claim: {claim.claim_text}")
```

This will output:
```
Claim: The Eiffel Tower is located in Paris.
Claim: The Eiffel Tower was completed in 1889.
Claim: The Eiffel Tower stands at a height of 324 meters.
```

For LLM-generated texts with token-level provenance, use the `extract_and_align_claims` function:

```python
from reclaim import extract_and_align_claims
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Tell me bio of Albert Einstein."
model_out = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], max_new_tokens=200)

model_tokens = model_out[0]
model_text = tokenizer.decode(model_tokens, skip_special_tokens=True)

claims = extract_and_align_claims(model_out, tokenizer)
```

## Local development
```bash
uv sync               # create the virtual environment
uv run python -m reclaim  # invoke claim CLI entrypoint
uv run pytest         # run your tests (add them under tests/)
```

## Documentation
The `docs/` directory contains high-level notes and can be extended with your preferred documentation generator. Start with [`docs/index.md`](docs/index.md) and grow the content alongside the ReClaim feature set.
