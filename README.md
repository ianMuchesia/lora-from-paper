# lora-from-paper

A from-scratch implementation of **LoRA (Low-Rank Adaptation of Large Language Models)** — the fine-tuning technique from the 2021 Microsoft Research paper by Hu et al.

The goal is the same as `mini-llm-api`: read the paper, understand the math, implement it in PyTorch, and document what you learn. Project is ongoing.

---

## What LoRA Does
Fine-tuning a large model means updating every parameter — expensive in memory and compute. LoRA freezes the original weights `W` and injects two small trainable matrices `B` and `A` beside them:

```
y = xW + xBA
```

- `W` is frozen (no gradients)
- `B` is `(in_features, rank)` — initialized to zeros
- `A` is `(rank, out_features)` — initialized randomly
- `rank` is tiny (e.g., 4, 8, 16) — far smaller than the original dimensions

The update `BA` is low-rank, meaning far fewer parameters to train. At `rank=8` on a 4096×4096 weight matrix, you're training ~65K parameters instead of ~16M.

---

## Project Structure

| Path | Contents |
| :--- | :--- |
| `src/lora_layer.py` | Core `LoRALayer` — toy implementation |
| `src/linear.py` | Vanilla `Linear` layer for comparison |
| `src/lora_gpt_layer.py` | `LoRAGPTLayer` — handles GPT-2's `Conv1D` quirk, includes merge/unmerge |
| `src/lora_transformer.py` | `inject_lora()` — injects LoRA into real GPT-2 attention layers |
| `src/small_model.py` | `ToyModel` — MLP + LoRALayer for rank sweep experiments |
| `src/train_small_model.py` | Training loop, rank sweep (r=4, 8, 16), result export |
| `utils/` | Helper utilities |
| `notebooks/paper_notes.ipynb` | Paper walkthrough and math experiments |
| `math-notes/` | Derivations, memory analysis, low-rank factorization notes |
| `experiments/` | Rank sweep results |

---

## Experiments: Rank Sweep
Trained `ToyModel` (20→64→2) with LoRA at different ranks across 10 epochs. Results saved in `experiments/`:

- `rank_4_results.txt` — fewest trainable params, simplest adapter
- `rank_8_results.txt` — balanced
- `rank_16_results.txt` — most expressive adapter

Each file logs frozen vs trainable parameter counts, final loss, and accuracy.

---

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers

# Run toy rank sweep
python src/train_small_model.py

# Inject LoRA into real GPT-2
python src/lora_transformer.py
```

---

## Study Notes
See [NOTES.md](NOTES.md) for conceptual breakdowns of the math, the memory efficiency trick, and implementation decisions.
