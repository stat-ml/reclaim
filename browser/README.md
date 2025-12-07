# Claim Browser

A small local web app for browsing retrievals and their extracted claims.

## Prerequisites

- Python 3.10+
- Install project with tokenizer extras (needed for the Llama 3.1 tokenizer):
  ```bash
  pip install .[llm]
  ```
  If you prefer a virtual environment, create/activate it first.

## Run the browser

```bash
python browser/server.py
```

What happens:
- Starts a local server at `http://localhost:5678`
- Opens your default browser automatically
- Serves the static UI from `browser/static`
- Exposes a `/decode` endpoint the UI calls to reconstruct text and token offsets

## Load data

1) Click “Load JSON file” in the top bar.
2) Select an outputs file matching the structure of `contest/outputs.json`:
   - Top-level object or array of retrievals.
   - Each retrieval includes:
     - `retrieval` (string)
     - `greedy_tokens` (list of token IDs for `meta-llama/Llama-3.1-70B-Instruct`)
     - `claims` (list) where each claim has `aligned_token_ids` pointing into `greedy_tokens` and `claim_text` (or `decoded_claim`).
   - Optional: `question`, `label`, `id`.

## Using the UI

- **Sidebar:** shows all retrievals; click to select.
- **Retrieval text:** tokenized and decoded via Llama tokenizer; claim tokens are highlighted.
- **Claim list:** hover a claim to isolate its spans in the retrieval; scrollable so you can hover while keeping the retrieval in view.
- If a claim references token IDs outside the available range, those tokens are ignored gracefully.

## Troubleshooting

- If the tokenizer download fails (no network), ensure the model is already cached locally or try again with network access.
- If the page doesn’t open automatically, open `http://localhost:5678` manually.
