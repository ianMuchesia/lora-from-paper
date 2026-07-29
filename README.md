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
| `src/lora_gpt_layer.py` | `LoRAGPTLayer` — handles GPT-2's `Conv1D`, includes merge/unmerge |
| `src/lora_transformer.py` | `inject_lora()` — injects LoRA into real GPT-2 attention layers |
| `src/train.py` | Full training loop on Alpaca dataset |
| `src/data.py` | Data loader and preprocessing |
| `src/format_prompt.py` | Prompt formatter for instruction-tuning format |
| `src/pre_tuned_gpt.py` | Baseline GPT-2 generation before fine-tuning |
| `src/post_tuned_gpt.py` | GPT-2 generation after LoRA fine-tuning |
| `src/compare_results.py` | Side-by-side baseline vs LoRA output comparison |
| `src/small_model.py` | `ToyModel` — MLP + LoRALayer for rank sweep |
| `src/train_small_model.py` | Rank sweep training (r=4, 8, 16) |
| `utils/` | Helper utilities (dataset downloader etc.) |
| `notebooks/paper_notes.ipynb` | Paper walkthrough and math |
| `math-notes/` | Derivations, FLOP/memory analysis |
| `experiments/` | Results, rank analysis, LoRA vs full fine-tuning reports |

---

## Experiments
- **Rank sweep** (toy model, r=4/8/16): confirms rank directly controls trainable param count; frozen backbone unchanged.
- **LoRA vs Full Fine-tuning on GPT-2** (`experiments/lora_vs_full_finetuning.md`): accuracy plateaued at ~54.6% across ranks — confirms low intrinsic dimension hypothesis from the paper.
- **Rank analysis** (`experiments/lora_rank_analysis.md`): parameter compression math — at `d=768, r=8` → 48× fewer trainable params than full fine-tuning.
- **Memory paradox:** LoRA showed *higher* peak GPU memory than full fine-tuning in experiments. LoRA doesn't skip the forward pass — it adds extra activation tensors (`x @ B @ A`). The optimizer memory saving is real, but activation memory can be higher.

---

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers datasets

# Toy rank sweep
python src/train_small_model.py

# Full LoRA fine-tuning on Alpaca
python src/train.py

# Compare baseline vs fine-tuned output
python src/compare_results.py
```

---

## Study Notes
See [NOTES.md](NOTES.md) for conceptual breakdowns of the math, the memory efficiency trick, and implementation decisions.




