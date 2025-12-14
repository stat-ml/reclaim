#### Claim Extraction Contest

Prepare environment:
```bash
git clone https://github.com/stat-ml/reclaim
cd reclaim

git fetch origin contest
git checkout contest
cd contest

pip install -r requirements.txt
```

`Meta-Llama-3.1-70B-Instruct-Turbo.json` contains data for contest. Each object has the following structure:
```json
{
    "question": "<question text>",
    "retrieval": "<retrieved context>",
    "input": "<full input to the model>",
    "output": "<model response to a question>"
    "greedy_tokens": [...tokenized model response...]
}
```

Run extraction:
```bash
python extract.py
```

Extraction enriches each object with a list of claims. Each claim has the following structure:
```json
{
    "claim_text": "<extracted claim text>",
    "decoded_claim": "<words in model response related to the claim>",
    "sentence": "<full sentence from model response containing the claim>",
    "aligned_token_ids": [...token ids in model response related to the claim...]
}
```

Your goal is to modify the library code to improve the quality of extracted claims.

Submission should be a pull request to this repository with your changes to the code. The pull request should include a brief description of the changes you made and why you believe they improve claim extraction. Ensure your code produces the same output format as described above.
