"""
Numeric comparison between our C++ model_forward() output (scripts/cpp_logits.bin,
raw float32 dumped by src/demo.cpp) and the HF `transformers` reference
(scripts/hf_logits.npy, from scripts/hf_reference.py --mode raw_logits).

Both must have been generated on the *same prompt* for this to mean anything.
"""

import os

import numpy as np

SCRIPTS_DIR = os.path.dirname(__file__)
VOCAB_SIZE = 151936


def main():
    hf = np.load(os.path.join(SCRIPTS_DIR, "hf_logits.npy"))

    cpp_flat = np.fromfile(os.path.join(SCRIPTS_DIR, "cpp_logits.bin"), dtype=np.float32)
    cpp = cpp_flat.reshape(-1, VOCAB_SIZE)

    print("hf shape: ", hf.shape)
    print("cpp shape:", cpp.shape)

    if hf.shape != cpp.shape:
        print("SHAPE MISMATCH -- can't compare further. Did you run both on the same prompt?")
        return

    diff = np.abs(hf - cpp)
    print(f"max abs diff:  {diff.max():.6f}")
    print(f"mean abs diff: {diff.mean():.6f}")

    hf_argmax = hf.argmax(axis=-1)
    cpp_argmax = cpp.argmax(axis=-1)
    print("hf  argmax per position: ", hf_argmax.tolist())
    print("cpp argmax per position: ", cpp_argmax.tolist())
    print("argmax match per position:", (hf_argmax == cpp_argmax).tolist())

    # BF16-derived weights + fp32 accumulation differences mean this won't be
    # bit-exact, but a correct implementation should be very close.
    tolerance = 0.5
    if diff.max() < tolerance:
        print(f"\nPASS -- max diff {diff.max():.6f} is within tolerance ({tolerance})")
    else:
        print(f"\nFAIL -- max diff {diff.max():.6f} exceeds tolerance ({tolerance})")
        worst_pos, worst_vocab = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"worst mismatch at position {worst_pos}, vocab id {worst_vocab}: "
              f"hf={hf[worst_pos, worst_vocab]:.4f} cpp={cpp[worst_pos, worst_vocab]:.4f}")


if __name__ == "__main__":
    main()
