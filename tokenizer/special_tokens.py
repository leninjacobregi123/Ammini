"""Special tokens shared by the tokenizer, data prep, training and the app.

Kept in one place so the tokenizer is trained with exactly the tokens the
chat template needs, and nothing drifts between pretraining and finetuning.
"""

PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end|>"

SPECIAL_TOKENS = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT, END_TURN]

# Default persona used to turn single-turn instruction data into chat-shaped
# examples, and shown to the user as the system prompt in the app. Written in
# Malayalam since the whole point is a Malayalam-native assistant.
DEFAULT_SYSTEM_PROMPT = (
    "നീ ഒരു സഹായകരമായ AI അസിസ്റ്റന്റാണ്. മലയാളത്തിൽ വ്യക്തമായും ചുരുക്കമായും മറുപടി നൽകുക."
)


def format_chat(turns, tokenizer_eos_appended_by_caller=False):
    """turns: list of (role, text) pairs, role in {"system","user","assistant"}.
    Returns the flat training/prompt string using the special-token template:

        <|system|>
        {system text}<|end|>
        <|user|>
        {user text}<|end|>
        <|assistant|>
        {assistant text}<|end|>
    """
    role_tag = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
    parts = []
    for role, text in turns:
        parts.append(f"{role_tag[role]}\n{text.strip()}{END_TURN}\n")
    return "".join(parts)


def build_training_example(instruction: str, input_text: str, output: str,
                            system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Turns one Alpaca-style (instruction, input, output) triple into a chat
    example, returning (prompt_text, full_text). prompt_text is everything up
    to and including the "<|assistant|>\n" tag (used to mask the loss so the
    model isn't trained to predict the prompt it was given)."""
    user_text = instruction.strip()
    if input_text and input_text.strip():
        user_text = f"{user_text}\n{input_text.strip()}"

    prompt_text = format_chat([("system", system_prompt), ("user", user_text)])
    prompt_text += f"{ASSISTANT}\n"
    full_text = prompt_text + f"{output.strip()}{END_TURN}\n"
    return prompt_text, full_text
