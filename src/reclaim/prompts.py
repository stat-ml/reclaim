# DOC_TO_ATOMIC_CLAIMS_PROMPT = """
# Your task is to extract atomic claims from a given text. Each claim must meet the following criteria:
# 1. Informative: Each claim must convey factual information about the subject matter, avoiding generic or irrelevant statements (e.g., "I will provide a balanced response" or "This type of network is simple").
# 2. Context-independent: Each claim must be understandable and verifiable on its own, without requiring additional context (e.g., avoid claims like "This type of network is simple" without specifying the network type).
# 3. De-duplicated: Avoid repeating the same information in different wordings.
# 4. Precise: Focus on clear and specific information, avoiding vague or overly broad statements.
# Define a function named decompose(input: str). The function should return only a Python list containing strings of atomic claims. Do not include any extra formatting, code blocks, or labels like "python" in your response. For example:
# If the input text is:
# "Mary is a five-year-old girl. She likes playing piano and doesn't like cookies."
# The output should be:
# ["Mary is a five-year-old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]  
# Example Input:
# "Linear Bus Topology involves connecting all network nodes to a single cable. This design makes it easy to install but difficult to troubleshoot, especially in large networks."  
# Example Output:
# ["Linear Bus Topology involves connecting all network nodes to a single cable.", "Linear Bus Topology is easy to install.", "Linear Bus Topology is difficult to troubleshoot.", "Linear Bus Topology is unsuitable for large networks."]  

# Important Notes:
# 1. The output must be a valid Python list with no additional text, code blocks, or formatting. 
# 2. The response must consist of only the list.

# Process the following text according to these rules:
# decompose("{doc}")
# """

# DOC_TO_ATOMIC_CLAIMS_PROMPT = """
# You are given a piece of text that contains factual claims. Extract all atomic, meaningful, self-contained claims. A claim asserts something as true or false and can be verified by humans.

# Extraction Rules:
# 1. Atomic — each claim must express exactly one factual idea.
# 2. Meaningful — each claim must convey substantive information (avoid trivial or generic statements).
# 3. Self-contained — each claim must stand alone without context from the original text. Resolve all coreferences (pronouns, “this,” “it,” etc.).
# 4. Precise — claims must be specific and unambiguous.
# 5. Non-duplicative — do not repeat the same fact in different wording.
# 6. Concise — preferably fewer than 15 words, but clarity is more important.

# Output Rules:
# - You MUST output ONLY a Python list of strings.
# - No code blocks, no explanations, no labels.
# - The output MUST start with '['.

# Few-Shot Examples:

# Example 1:
# [text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches the Monte Carlo Masters final for the first time. Berdych will face either Rafael Nadal or Novak Djokovic in the final.
# [response]:
# ["Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday.",
#  "Tomas Berdych is the sixth seed.",
#  "Tomas Berdych reached the Monte Carlo Masters final for the first time.",
#  "Tomas Berdych will face Rafael Nadal or Novak Djokovic in the final."]

# Example 2:
# [text]: Tinder only displays the last 34 photos, but users can easily see more. The firm also said it had improved its mutual friends feature.
# [response]:
# ["Tinder displays only the last 34 photos.",
#  "Tinder users can view more than the last 34 photos.",
#  "Tinder said it improved its mutual-friends feature."]

# Example 3:
# [text]: Linear Bus Topology involves connecting all network nodes to a single cable. This design makes it easy to install but difficult to troubleshoot, especially in large networks.
# [response]:
# ["Linear Bus Topology connects all network nodes to a single cable.",
#  "Linear Bus Topology is easy to install.",
#  "Linear Bus Topology is difficult to troubleshoot.",
#  "Linear Bus Topology performs poorly in large networks."]

# Now complete the following:
# [text]: {doc}
# [response]:
# """


DOC_TO_ATOMIC_CLAIMS_PROMPT = """
You are given a piece of text that contains factual claims. Extract all atomic, meaningful, self-contained claims. A claim asserts something as true or false and can be verified by humans.

Extraction Rules:
1. Atomic — each claim must express exactly one factual idea.
2. Meaningful — each claim must convey substantive information (avoid trivial or generic statements).
3. Self-contained — each claim must stand alone without context from the original text. Resolve all coreferences (pronouns, “this,” “it,” etc.).
4. Precise — claims must be specific and unambiguous.
5. Non-duplicative — do not repeat the same fact in different wording.
6. Concise — preferably fewer than 15 words, but clarity is more important.
7. Avoid vague bucket claims:
   - Do NOT include claims that only say things like “other serious health issues”, “various problems”, “many negative effects”, “other damage”, or “other impacts” when they do not add new, specific information.
   - You may keep a vague harm claim like “X causes serious health issues” ONLY if this is the main information given and there are no more specific harms that you already extracted.

Output Rules:
- You MUST output ONLY a Python list of strings.
- No code blocks, no explanations, no labels.
- The output MUST start with '['.

Few-Shot Examples:

Example 1:
[text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches the Monte Carlo Masters final for the first time. Berdych will face either Rafael Nadal or Novak Djokovic in the final.
[response]:
["Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday.",
 "Tomas Berdych is the sixth seed.",
 "Tomas Berdych reached the Monte Carlo Masters final for the first time.",
 "Tomas Berdych will face Rafael Nadal or Novak Djokovic in the final."]

Example 2:
[text]: Tinder only displays the last 34 photos, but users can easily see more. The firm also said it had improved its mutual friends feature.
[response]:
["Tinder displays only the last 34 photos.",
 "Tinder users can view more than the last 34 photos.",
 "Tinder said it improved its mutual-friends feature."]

Example 3:
[text]: Linear Bus Topology involves connecting all network nodes to a single cable. This design makes it easy to install but difficult to troubleshoot, especially in large networks.
[response]:
["Linear Bus Topology connects all network nodes to a single cable.",
 "Linear Bus Topology is easy to install.",
 "Linear Bus Topology is difficult to troubleshoot.",
 "Linear Bus Topology performs poorly in large networks."]

Now complete the following:
[text]: {doc}
[response]:
"""


DOC_TO_ATOMIC_CLAIMS_SYSTEM_PROMPT = """Your role is to extract factual atomic claims from text. You must follow all output rules strictly. You always respond ONLY with a Python list of strings. No explanations, no formatting, no code blocks. Each claim must be atomic, meaningful, self-contained, precise, concise, and non-duplicated. Resolve all pronouns and vague references so each claim stands alone. Examples will be provided. Learn the pattern from them and generalize it reliably. Never break the output format."""


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
Claims should be a context-independent, fully atomic, representing one fact. Atomic claims are simple, indivisible facts that do not bundle multiple pieces of information together.

### Guidelines for Decomposition:
1. **Atomicity**: Break down each statement into the smallest possible unit of factual information. Avoid grouping multiple facts in one claim. For example:
   - Instead of: "Photosynthesis in plants converts sunlight, carbon dioxide, and water into glucose and oxygen."
   - Output: ["Photosynthesis in plants converts sunlight into glucose.", "Photosynthesis in plants converts carbon dioxide into glucose.", "Photosynthesis in plants converts water into glucose.", "Photosynthesis in plants produces oxygen."]

   - Instead of: "The heart pumps blood through the body and regulates oxygen supply to tissues."
   - Output: ["The heart pumps blood through the body.", "The heart regulates oxygen supply to tissues."]

   - Instead of: "Gravity causes objects to fall to the ground and keeps planets in orbit around the sun."
   - Output: ["Gravity causes objects to fall to the ground.", "Gravity keeps planets in orbit around the sun."]

2. **Context-Independent**: Each claim must be understandable and verifiable on its own without requiring additional context or references to other claims. Avoid vague claims like "This process is important for life."

3. **Precise and Unambiguous**: Ensure the claims are specific and avoid combining related ideas that can stand independently.

4. **No Formatting**: The response must be a Python list of strings without any extra formatting, code blocks, or labels like "python".

### Example:
If the input text is:
"Mary is a five-year-old girl. She likes playing piano and doesn't like cookies."
Extracted claims should be:
"Mary is a five-year-old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."

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
