"""
Reference runner for superwhisper/s1-mini using real HuggingFace `transformers`,
loaded from the local weights/ dir (no network needed once weights/ is populated).

Two things this script is for:

1. `normalize()` -- the real intended usage (chat template, control line,
   enable_thinking=False, greedy decode) straight from the model card's
   Quickstart. Use this to see what the model is actually supposed to do.

2. `raw_logits()` -- a single forward pass on a fixed prompt's *raw* token
   ids (no chat template), dumped to a .npy file. This is the actual M3 proof
   target: our C++ model_forward() runs the exact same raw-token forward pass,
   so its logits should match this file within float tolerance.
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)


def load():
    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, torch_dtype="auto")
    model.eval()
    return tok, model


def normalize(tok, model, transcript, styling="semi-formal", structure="prose", context="general"):
    control = f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{control}\n{transcript}"},
    ]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def raw_logits(tok, model, prompt, out_path):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    print("token ids:", ids[0].tolist())

    with torch.no_grad():
        out = model(input_ids=ids)

    logits = out.logits[0].to(torch.float32).numpy()  # [seq_len, vocab_size]
    print("logits shape:", logits.shape)

    np.save(out_path, logits)
    print(f"saved logits to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normalize", "raw_logits"], default="raw_logits")
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "hf_logits.npy"))
    args = parser.parse_args()

    tok, model = load()

    if args.mode == "raw_logits":
        raw_logits(tok, model, args.prompt, args.out)
    else:
        raw = "so um i need to like send the the report by uh friday no wait make that thursday"
        print(normalize(tok, model, raw))
