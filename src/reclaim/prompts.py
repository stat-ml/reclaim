DOC_TO_ATOMIC_CLAIMS_PROMPT = """
You extract *atomic*, self-contained claims from the given text.

Rules:
- A claim is one fact (single predicate); do not bundle multiple facts with "and", commas, or enumerations.
- Each claim must be decontextualized: replace pronouns (he/she/they/it/this/that/these/those/there) and vague references with the concrete entity from the text.
- Each claim must be informative and complete (subject + predicate + object/complement when needed). Drop fragments like "He began his career" or "is considered".
- Avoid duplicates or near-duplicates; keep the most specific phrasing.
- If no atomic claims exist, return an empty list.

Return JSON exactly in the form:
{{"claims": ["claim 1", "claim 2", "..."]}}
No code fences, no extra keys, no prose.

Text:
{doc}
"""


DOC_TO_SENTENCES_PROMPT = """
Your task is to perform sentence segmentation. 
Let's define a function named split(input:str).
The return value should be a list of strings, where each string should be a sentence.
For example, if a user call process("Mary is a five-year old girl. She likes playing piano. She doesn't like cookies.").
You should return a python list without any other words, 
["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

split("{doc}")
"""


SENTENCES_TO_CLAIMS_PROMPT = """
Decompose the text into *atomic, decontextualized* claims.

Requirements:
- One fact per claim; split conjunctions/enumerations into separate claims.
- Replace pronouns and vague references with the specific entity from the text so the claim stands alone.
- Keep claims informative and complete (subject + predicate + complement as needed). Drop non-informative fragments.
- Remove duplicates/near-duplicates; keep the most specific version.

Return JSON exactly in the form:
{{"claims": ["claim 1", "claim 2", "..."]}}
No code fences or extra text.

Text:
{doc}
"""


DOC_TO_INDEPEDENT_SENTENCES_PROMPT = """
Your task is to perform sentence segmentation and de-contextualization. 
Let's define a function named process(input:str).
The return value should be a list of strings, where each string should be a decontextualized sentence.
For example, if a user call process("Mary is a five-year old girl. She likes playing piano. She doesn't like cookies.").
You should return a python list without any other words, 
["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

process("{doc}")
"""

PRONOUN_REWRITE_PROMPT = """
Rewrite the claim so it is fully self-contained and does not rely on pronouns or vague references. Use the text to resolve the referents. If it cannot be rewritten, return DROP.

Text:
{text}

Claim:
{claim}

Return only the rewritten claim, or DROP.
"""

SANITIZE_CLAIM_PROMPT = """
You are checking a claim for quality. The claim must be:
- one atomic fact (single predicate; no "and"/enumerations),
- decontextualized (explicit subject, no pronouns like he/she/they/it/this/that/these/those/there),
- informative and complete (no bare fragments like "was 13 years old" without who, where, when; no "is considered" without the thing).

If the claim violates these rules, rewrite it into one valid atomic claim using the text for context. If it cannot be made valid, return DROP.

Text:
{text}

Claim:
{claim}

Return only the rewritten claim, or DROP.
"""

SPLIT_NON_ATOMIC_PROMPT = """
Split the claim into atomic, decontextualized claims if it contains multiple facts.
- Each output claim must be one fact with a single predicate.
- Replace pronouns with explicit entities from the text so each claim stands alone.
- If the claim is already atomic, return it as a single-element list.
- If you cannot produce valid atomic claims, return an empty list.

Return JSON exactly as: {{"claims": ["claim 1", "claim 2", "..."]}}
No code fences or extra text.

Text:
{text}

Claim:
{claim}
"""


MATCHING_PROMPT = """
Task: Analyze the given text and the claim (which was extracted from the text). For each sentence in the text:
1. Copy the sentence exactly as it appears in the text.
2. Identify the words from the sentence that are related to the claim, in the same order they appear in the sentence. If no words are related, output "No related words". Each word should be present in the output exactly as it appears in the sentence, including capitalization and punctuation. Don't expand abbreviations or change forms of the words.

Example:

Text:
"Sure! Here are brief explanations of each type of network topology mentioned in the passages:

1. Linear Bus: In a Linear Bus topology, all network nodes are connected to a common transmission medium via two endpoints. This topology is simple to install and maintain but can be prone to single points of failure, meaning that if one node fails, the entire network will be affected.
2. Distributed Bus: In a Distributed Bus topology, all network nodes are connected to a shared transmission medium via multiple endpoints, creating a branched structure. This topology is similar to the Linear Bus topology but offers greater redundancy and reliability due to the multiple connections.
3. Star: In a Star topology, each computer or device is connected to a central hub using a point-to-point connection. This topology is easy to manage and maintain, and the central hub provides a single point of failure, making it easier to troubleshoot issues. However, this topology can become expensive as more devices need to be connected to the central hub.
4. Mesh: In a Mesh topology, each device is directly connected to every other device in the network, forming a web-like structure. This topology offers high redundancy and reliability as there are multiple paths for data to travel between devices. Additionally, mesh networks can self-heal, meaning that if one path becomes unavailable, data can be rerouted through another path.

Unable to answer based on given passages. The passages do not provide detailed descriptions of the different types of topology networks beyond their basic definitions."

Claim:
"Distributed Bus topology connects all network nodes to a shared transmission medium via multiple endpoints."

Answer:
Sentence: "Sure! Here are brief explanations of each type of network topology mentioned in the passages:"
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "1. Linear Bus: In a Linear Bus topology, all network nodes are connected to a common transmission medium via two endpoints."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "This topology is simple to install and maintain but can be prone to single points of failure, meaning that if one node fails, the entire network will be affected."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "2. Distributed Bus: In a Distributed Bus topology, all network nodes are connected to a shared transmission medium via multiple endpoints, creating a branched structure."
Related words from this sentence (same order they appear in the sentence): "Distributed", "Bus", "topology", "all", "network", "nodes", "are", "connected", "to", "a", shared", "transmission", "medium", "via", multiple", "endpoints"

Sentence: "This topology is similar to the Linear Bus topology but offers greater redundancy and reliability due to the multiple connections."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "3. Star: In a Star topology, each computer or device is connected to a central hub using a point-to-point connection."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "This topology is easy to manage and maintain, and the central hub provides a single point of failure, making it easier to troubleshoot issues."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "However, this topology can become expensive as more devices need to be connected to the central hub."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "4. Mesh: In a Mesh topology, each device is directly connected to every other device in the network, forming a web-like structure."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "This topology offers high redundancy and reliability as there are multiple paths for data to travel between devices."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "Additionally, mesh networks can self-heal, meaning that if one path becomes unavailable, data can be rerouted through another path."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "Unable to answer based on given passages."
Related words from this sentence (same order they appear in the sentence): No related words

Sentence: "The passages do not provide detailed descriptions of the different types of topology networks beyond their basic definitions."
Related words from this sentence (same order they appear in the sentence): No related words

Now analyze the following text using this format:

Text:
{text}

Claim:
{claim}
"""
