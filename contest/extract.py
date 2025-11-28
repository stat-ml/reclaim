import json
from transformers import AutoTokenizer
from dataclasses import asdict

from reclaim import extract_and_align_claims

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

with open("Meta-Llama-3.1-70B-Instruct-Turbo.json", "r") as f:
    data = json.load(f)
    for _id, row in data.items():
        claims = extract_and_align_claims(
            text=row["output"],
            tokens=row["greedy_tokens"],
            tokenizer=tokenizer,
            openai_model="gpt-5.1",
            progress_bar=True,
            n_threads=1,
        )
        row.update({"claims": [asdict(claim) for claim in claims]}) 

with open("outputs.json", "w") as f:
    json.dump(data, f, indent=2)
