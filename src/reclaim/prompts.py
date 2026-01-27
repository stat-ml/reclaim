DOC_TO_ATOMIC_CLAIMS_PROMPT = """
Your task is to extract atomic claims from a given text. Each claim must meet the following criteria:
1. Informative: Each claim must convey factual information about the subject matter, avoiding generic or irrelevant statements (e.g., "I will provide a balanced response" or "This type of network is simple").
2. Context-independent: Each claim must be understandable and verifiable on its own, without requiring additional context (e.g., avoid claims like "This type of network is simple" without specifying the network type).
3. De-duplicated: Avoid repeating the same information in different wordings.
4. Precise: Focus on clear and specific information, avoiding vague or overly broad statements.
Define a function named decompose(input: str). The function should return only a Python list containing strings of atomic claims. Do not include any extra formatting, code blocks, or labels like "python" in your response. For example:
If the input text is:
"Mary is a five-year-old girl. She likes playing piano and doesn't like cookies."
The output should be:
["Mary is a five-year-old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]  
Example Input:
"Linear Bus Topology involves connecting all network nodes to a single cable. This design makes it easy to install but difficult to troubleshoot, especially in large networks."  
Example Output:
["Linear Bus Topology involves connecting all network nodes to a single cable.", "Linear Bus Topology is easy to install.", "Linear Bus Topology is difficult to troubleshoot.", "Linear Bus Topology is unsuitable for large networks."]  

Important Notes:
1. The output must be a valid Python list with no additional text, code blocks, or formatting. 
2. The response must consist of only the list.

Process the following text according to these rules:
decompose("{doc}")
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
Your task is to decompose the text into atomic claims.

Atomic claims must be:
- Context-independent (understandable/verifiable on their own)
- Fully atomic (one fact per claim, no bundled facts)
- Exhaustive (capture all relevant factual content from the text)
- Entailed by the text (do not add background knowledge or “obvious” facts)

### Guidelines for Decomposition:

1) **Atomicity**
Break down each statement into the smallest possible unit of factual information. Avoid bundling multiple facts in one claim.

Examples:
- Instead of: "The heart pumps blood through the body and regulates oxygen supply to tissues."
- Output: ["The heart pumps blood through the body.", "The heart regulates oxygen supply to tissues."]

- Instead of: "Gravity causes objects to fall to the ground and keeps planets in orbit around the sun."
- Output: ["Gravity causes objects to fall to the ground.", "Gravity keeps planets in orbit around the sun."]

Note: Split conjunctions only when each resulting claim remains meaningful and truth-preserving.  
For example:
- "Alice bought apples and oranges." → ["Alice bought apples.", "Alice bought oranges."]
- "Alice and Bob met." → ["Alice and Bob met."]  (Do not split into nonsense like "Alice met.")

2) **Exhaustiveness**
Ensure all relevant facts from the original text are captured in the claims. Do not omit significant information.

3) **Context-Independent**
Each claim must be understandable and verifiable on its own without requiring additional context.
- Replace vague references (“this”, “that”, “it”, “they”, “the former”, “the company”, etc.) with the specific entity when it is identifiable from the text.
- If the text does not provide enough information to resolve a reference, keep the original wording but make the claim as standalone as possible.

Avoid vague claims like: "This process is important for life."

4) **Inter-claim Independence**
Each claim must stand alone and not depend on other claims for interpretation.

Example:
- Original: "The evidence for undisclosed ingredients in vials includes the presence of graphene compounds. This evidence was later confirmed in further samples."
- Incorrect: ["The evidence for undisclosed ingredients in vials includes the presence of graphene compounds.", "This evidence was later confirmed in further samples."]
- Correct: ["The evidence for undisclosed ingredients in vials includes the presence of graphene compounds.", "Further samples confirmed the presence of graphene compounds in vials."]

5) **Precise and Unambiguous**
Use specific phrasing and preserve meaning exactly. Do not “improve” the author’s certainty or scope.

Preserve exactly:
- Quantifiers: all, some, many, few, only, exactly, at least, at most
- Negation: not, no, none, never
- Comparatives/superlatives: larger than, highest, most, among, ranked #3
- Units: kg, %, USD, km², etc.
- Relationship type: causes vs correlates vs is associated with vs suggests

6) **Attribution vs Truth (reported speech)**
If the text attributes a claim to a source (e.g., “X said/claimed/reported/suggested…”), produce a claim about the attribution.
Do NOT automatically convert attributed claims into unqualified facts unless the author clearly asserts them as true.

Examples:
- "A report claims the outage began in May."
  → ["A report claims the outage began in May."]  (Do not output "The outage began in May." unless asserted as fact.)

- "The WHO reported that the outbreak began in May."
  → ["The WHO reported that the outbreak began in May."]

7) **Modality and Uncertainty**
Preserve modality/uncertainty (may, might, could, likely, allegedly, reportedly, suggests). Do not upgrade uncertain statements into certain ones.

Example:
- "The drug may reduce symptoms."
  → ["The drug may reduce symptoms."]

8) **Time Handling**
Preserve all temporal qualifiers: dates, time ranges, “currently”, “formerly”, “as of”, “later”, “before”, “after”.
- If a reference time anchor is explicitly provided in the text, you may resolve relative time (“then”, “now”, “today”) into that anchor.
- If no anchor is provided, do not invent dates. Keep relative time terms.

Example:
- "In 2020, Alice moved to Paris. She now lives in Berlin."
  → ["Alice moved to Paris in 2020.", "Alice lives in Berlin now."]

9) **Conditionals and Counterfactuals**
Keep conditional/counterfactual statements as conditionals. Do not assert the antecedent or consequent as fact.

Examples:
- "If it rains tomorrow, the game will be canceled."
  → ["If it rains tomorrow, the game will be canceled."]

- "Had the policy passed, taxes would have fallen."
  → ["If the policy had passed, taxes would have fallen."]

10) **Negation vs “no evidence”**
Do not convert “lack of evidence” into a negated world claim.

Example:
- "There is no evidence that X causes Y."
  → ["There is no evidence that X causes Y."]  (Not: "X does not cause Y.")

11) **Numbers, ranges, and approximation**
Preserve numeric qualifiers (about/approximately, between, more than, less than).
For ranges, you may represent them as separate lower/upper-bound claims if that improves atomicity.

Examples:
- "Sales were between $2M and $3M."
  → ["Sales were at least $2M.", "Sales were at most $3M."]

- "About 10% of users churned."
  → ["About 10% of users churned."]

12) **No added knowledge / no bridging**
Claims must be entailed by the text. Do not add background facts, common knowledge, or “obvious” implications.

Example:
- If the text says "Paris has 2.1 million residents", do NOT add "Paris is in France."

13) **Output format**
Return ONLY a Python list of strings (e.g., ["...", "..."]). No extra formatting, no code blocks, no labels.

### Example:
Input:
"Mary is a five-year-old girl. She likes playing piano and doesn't like cookies."
Output:
["Mary is a five-year-old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]

### Now, decompose the following text into atomic claims:
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
