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
| `src/lora_layer.py` | Core `LoRALayer` implementation |
| `src/linear.py` | Vanilla `Linear` layer for comparison |
| `notebooks/paper_notes.ipynb` | Paper walkthrough and math experiments |
| `math-notes/` | Derivations and memory analysis |
| `experiments/` | Comparison runs (to be added) |

---

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.in
```

---

## Study Notes
See [NOTES.md](NOTES.md) for conceptual breakdowns of the math and implementation decisions.
