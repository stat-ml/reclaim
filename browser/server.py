import json
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    # Keep a clear message so users know how to install extras.
    raise SystemExit(
        "Transformers is required to run the claim browser. "
        "Install the optional LLM extras with `pip install .[llm]`."
    ) from exc


DEFAULT_PORT = 5678
STATIC_DIR = Path(__file__).parent / "static"


def _validate_token_ids(raw_ids: Iterable) -> List[int]:
    token_ids = list(raw_ids)
    if not all(isinstance(t, int) for t in token_ids):
        raise ValueError("token_ids must be a list of integers")
    return token_ids


def decode_tokens(tokenizer, token_ids: List[int]) -> Tuple[str, List[dict]]:
    """
    Decode tokens and return the reconstructed text plus per-token offsets.

    Offsets are built by progressively decoding the prefix of tokens. This
    preserves spacing decisions made by the tokenizer.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    fragments: List[str] = []
    offsets: List[dict] = []

    prev_text = ""
    for idx in range(len(tokens)):
        partial_text = tokenizer.convert_tokens_to_string(tokens[: idx + 1])
        piece = partial_text[len(prev_text) :]
        fragments.append(piece)
        prev_text = partial_text

    cursor = 0
    for idx, piece in enumerate(fragments):
        start = cursor
        cursor += len(piece)
        offsets.append(
            {
                "index": idx,
                "start": start,
                "end": cursor,
                "text": piece,
                "token_id": token_ids[idx],
                "token": tokens[idx],
            }
        )

    full_text = "".join(fragments)
    return full_text, offsets


class ClaimBrowserHandler(SimpleHTTPRequestHandler):
    tokenizer = None

    def do_POST(self) -> None:  # pragma: no cover - runtime path
        if self.path != "/decode":
            self.send_error(404, "Unknown endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            data = json.loads(payload)
            token_ids = _validate_token_ids(data.get("token_ids", []))
        except Exception as exc:  # broad to simplify response
            self.send_error(400, f"Invalid request: {exc}")
            return

        try:
            text, offsets = decode_tokens(self.tokenizer, token_ids)
        except Exception as exc:  # pragma: no cover - runtime path
            self.send_error(500, f"Failed to decode tokens: {exc}")
            return

        body = json.dumps({"text": text, "tokens": offsets}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_tokenizer():
    print("Loading tokenizer meta-llama/Llama-3.1-70B-Instruct ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-70B-Instruct"
    )
    return tokenizer


def main(port: int = DEFAULT_PORT) -> None:  # pragma: no cover - runtime entrypoint
    if not STATIC_DIR.exists():
        raise SystemExit(f"Static assets not found in {STATIC_DIR}")

    tokenizer = _load_tokenizer()
    ClaimBrowserHandler.tokenizer = tokenizer

    handler_class = partial(ClaimBrowserHandler, directory=str(STATIC_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)

    url = f"http://localhost:{port}"
    print(f"Serving claim browser at {url}")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover - CLI execution
    chosen_port = DEFAULT_PORT
    if len(sys.argv) == 2:
        try:
            chosen_port = int(sys.argv[1])
        except ValueError:
            raise SystemExit("Port must be an integer")
    main(port=chosen_port)
