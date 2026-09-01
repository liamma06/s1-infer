"""
Times the official HuggingFace `transformers` + PyTorch path on the exact same
prompt/format s1-infer's demo.exe uses, for a direct TTFT/TPOT comparison.

Forces float32 (not the model's native bfloat16) -- bf16 on a CPU without
AVX512-BF16 support is often software-emulated and pathologically slow, which
would make this an unfair "PyTorch is terrible" result rather than a real
comparison against a properly-run official baseline.

Manual greedy decode loop (not model.generate()) so we can time each step
individually, same TTFT/TPOT split demo.exe reports.
"""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)

TRANSCRIPT = "So hey, I think we should probably go to the store later today."
NUM_TOKENS_TO_GENERATE = 16  # match the s1-infer demo.exe runs this session


def build_prompt(tok, transcript, styling="semi-formal", structure="prose", context="general"):
    control = f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{control}\n{transcript}"},
    ]
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(os.cpu_count())

    text = build_prompt(tok, TRANSCRIPT)
    inputs = tok(text, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"prompt tokens: {input_ids.shape[1]}")

    step_ms = []
    generated = []

    with torch.no_grad():
        # step 0: full prompt forward pass (this is TTFT)
        start = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        next_id = out.logits[0, -1].argmax().item()
        step_ms.append((time.perf_counter() - start) * 1000.0)
        generated.append(next_id)
        past = out.past_key_values

        for _ in range(NUM_TOKENS_TO_GENERATE - 1):
            start = time.perf_counter()
            step_input = torch.tensor([[next_id]])
            out = model(input_ids=step_input, past_key_values=past, use_cache=True)
            next_id = out.logits[0, -1].argmax().item()
            step_ms.append((time.perf_counter() - start) * 1000.0)
            generated.append(next_id)
            past = out.past_key_values

            if next_id in (tok.eos_token_id,):
                break

    ttft = step_ms[0]
    tpot = sum(step_ms[1:]) / len(step_ms[1:]) if len(step_ms) > 1 else 0.0
    total = sum(step_ms)

    print(f"generated {len(generated)} tokens")
    print(f"response: {tok.decode(generated, skip_special_tokens=True)!r}")
    print(f"TTFT: {ttft:.2f} ms")
    print(f"TPOT: {tpot:.2f} ms/token")
    print(f"total: {total:.2f} ms ({len(generated) / (total / 1000.0):.4f} tok/s)")


if __name__ == "__main__":
    main()
