import json
from transformers import AutoTokenizer

from reclaim import extract_and_align_claims

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

results = []
with open "inputs.json", "r") as f:
    reader = json.load(f)
    for row in reader:
        result = {}

        claims = extract_and_align_claims(
            text=row["text"],
            tokens=row["tokens"],
            tokenizer=tokenizer,
            openai_model="gpt-5.1",
            progress_bar=True,
            n_threads=1,
        )

        result['text'] = row['text']
        result['tokens'] = row['tokens']
        result['claims'] = [claim.to_dict() for claim in claims]

        results.append(result)

with open("outputs.json", "w") as f:
    json.dump(results, f, indent=2)
