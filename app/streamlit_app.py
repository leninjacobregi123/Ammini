"""Streamlit chat UI for the instruction-tuned MalayaLM checkpoint --
equivalent role to the book's ch05/06_user_interface, ch06/04_user_interface,
ch07/06_user_interface bonus apps (which use Chainlit), rebuilt in Streamlit
per preference, and driving our own from-scratch model instead of a
downloaded GPT-2 checkpoint.

Run (inside the docker container, on Shannon):
    streamlit run app/streamlit_app.py -- --checkpoint checkpoints/instruct/malayalam_assistant.pt
"""
import argparse
import sys
from pathlib import Path

import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.checkpoint import load_model  # noqa: E402
from tokenizer.special_tokens import (  # noqa: E402
    format_chat, ASSISTANT, END_TURN, DEFAULT_SYSTEM_PROMPT,
)
from tokenizers import Tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/instruct/malayalam_assistant.pt")
    ap.add_argument("--tokenizer", default="tokenizer/malayalam_tokenizer.json")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    # streamlit passes its own args before "--", ignore anything unrecognized
    args, _ = ap.parse_known_args()
    return args


@st.cache_resource
def get_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    tok = Tokenizer.from_file(tokenizer_path)
    model, cfg, system_prompt = load_model(checkpoint_path)
    return model, tok, cfg, (system_prompt or DEFAULT_SYSTEM_PROMPT)


def generate_reply(model, tok, cfg, system_prompt, history, user_text, max_new_tokens, temperature, top_k):
    device = next(model.parameters()).device
    turns = [("system", system_prompt)]
    for role, text in history:
        turns.append((role, text))
    turns.append(("user", user_text))
    prompt = format_chat(turns) + f"{ASSISTANT}\n"

    ids = tok.encode(prompt).ids
    ids = ids[-(cfg.context_length - max_new_tokens):]  # keep room to generate
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    eos_id = tok.token_to_id(END_TURN)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, eos_id=eos_id)
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids)
    return text.split(END_TURN)[0].strip()


def main():
    args = parse_args()
    st.set_page_config(page_title="Malayalam Assistant", page_icon="🤖")
    st.title("🤖 മലയാളം AI അസിസ്റ്റന്റ്")
    st.caption("A from-scratch Llama+MoE model, pretrained and instruction-tuned entirely on Malayalam.")

    if not Path(args.checkpoint).exists():
        st.error(f"Checkpoint not found: {args.checkpoint}\n\n"
                 "Run pretraining + finetune/instruction_finetune.py first, or pass "
                 "--checkpoint pointing at your .pt file.")
        st.stop()

    model, tok, cfg, system_prompt = get_model_and_tokenizer(args.checkpoint, args.tokenizer)

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, text in st.session_state.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)

    user_text = st.chat_input("നിങ്ങളുടെ സന്ദേശം ഇവിടെ ടൈപ്പ് ചെയ്യുക...")
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("ചിന്തിക്കുന്നു..."):
                reply = generate_reply(
                    model, tok, cfg, system_prompt, st.session_state.history, user_text,
                    args.max_new_tokens, args.temperature, args.top_k,
                )
            st.markdown(reply)

        st.session_state.history.append(("user", user_text))
        st.session_state.history.append(("assistant", reply))

    with st.sidebar:
        st.subheader("Settings")
        st.write(f"Context length: {cfg.context_length}")
        st.write(f"Layers: {cfg.n_layers}, d_model: {cfg.d_model}, experts: {cfg.n_experts}")
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()


if __name__ == "__main__":
    main()
