"""Shared model identifiers and prompts for the Qwen demos."""

QWEN3_TEXT_MODEL = "Qwen/Qwen3-8B"
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"

DEFAULT_HATEFUL_MEME_SYSTEM_PROMPT = (
    "You are a careful research assistant for hateful meme classification. "
    "Analyze visual content, meme text, target groups, stereotypes, and implied meaning. "
    "Return concise JSON with whether the meme is hateful (Answer yes or no), label, confidence (0-1), target_group, evidence, and explanation."
)

SIMPLE_HATEFUL_MEME_SYSTEM_PROMPT = (
    "You are a careful research assistant for hateful meme classification. "
    "Analyze visual content, meme text, target groups, stereotypes, and implied meaning. "
    "Return only one label, whether the meme is hateful (true) or not hateful (false). Output only true/false, nothing else."
)

DEFINITION_HATEFUL_MEME_SYSTEM_PROMPT = (
    "You are going to carefully examine a meme and determine if it is hateful or not. Carefully analyze the visual content, meme text, context etc, but be sure to allow for free speech and humor."
    "Use the following definition of hateful speech: a communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity fa"
    "Return only one label, whether the meme is hateful (true) or not hateful (false). Output only true/false, nothing else."
)
DEFAULT_EMBEDDING_INSTRUCTION = (
    "Represent the meme text and image for hateful meme classification, "
    "similar example retrieval, prototype matching, and contrastive learning."
)
