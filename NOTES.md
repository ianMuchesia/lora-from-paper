# Notes: LoRA from Paper

## 1. What Problem LoRA Solves

Full fine-tuning updates every weight in the model. For a 7B parameter model, that means storing a full copy of gradients and optimizer states — often 3–4x the size of the model itself. LoRA sidesteps this by keeping the original weights frozen and only training two small matrices injected beside them.

## 2. The Core Math

For a pre-trained weight matrix `W` of shape `(in_features, out_features)`, LoRA adds a low-rank update:

```
y = xW + xBA
```

Where:

- `W` — frozen, shape `(in_features, out_features)`, no gradients
- `B` — trainable, shape `(in_features, rank)`, initialized to **zeros**
- `A` — trainable, shape `(rank, out_features)`, initialized randomly

**Why B is initialized to zeros:** At the start of fine-tuning, `BA = 0`. This means the LoRA layer produces exactly the same output as the original frozen layer. Training starts from the same point as the pre-trained model, not from a random perturbation.

## 3. Why Not Compute `W + BA` Directly?

Mathematically, `x(W + BA)` is correct. But computing it that way forces PyTorch to first materialize the full `BA` matrix in GPU memory before multiplying by `x`.

If `W` is 4096×4096, then `BA` is also 4096×4096 — you've just allocated another 16M-parameter tensor. The whole memory advantage of LoRA is gone.

The fix is the **associative property**:

```
x(BA) = (xB)A
```

Compute `xB` first — result is `(batch, rank)`, tiny. Then multiply by `A` — result is `(batch, out_features)`. Peak intermediate memory is `(batch, rank)` instead of `(in_features, out_features)`. This is why the implementation writes `(x @ B) @ A` not `x @ (B @ A)`.

## 4. Parameter Count vs Full Fine-tuning

At `rank=8` on a 4096×4096 weight matrix:

- Full fine-tuning: 4096 × 4096 = **16,777,216 parameters**
- LoRA: (4096 × 8) + (8 × 4096) = **65,536 parameters** — ~256x fewer

The larger the model, the more dramatic the saving.

## 5. What Gets Trained, What Stays Frozen

```python
self.W = nn.Parameter(torch.randn(in_features, out_features))
self.W.requires_grad = False   # frozen — never updated

self.A = nn.Parameter(torch.randn(rank, out_features))  # trainable
self.B = nn.Parameter(torch.zeros(in_features, rank))   # trainable
```

Only `A` and `B` accumulate gradients. `W` is fixed at its pre-trained values throughout.

## 6. CrossEntropyLoss — Classification vs Regression

CrossEntropyLoss does not work like regression (where you predict a number and compare it to another number). In classification, the model's job is to **pick a category**, not measure a quantity.

**How it actually works:**

- Targets are category indices — just an answer key e.g. `[1, 0, 0, 1]`. Not scores.
- The model's final layer outputs one raw score per class. With `out_features=2`, you get two numbers per sample: confidence for class 0, confidence for class 1.
- `CrossEntropyLoss` looks at the target index, finds the model's score at that index, and asks: _is that score higher than the others?_ If yes, loss is low.

**The teacher/student analogy:**

- Answer key says: correct answer is category `0`.
- Model outputs: `[4.5, 1.2]` (score 4.5 for category 0, 1.2 for category 1).
- Loss checks: is 4.5 the highest? Yes — low loss.
- You never provide a "target score" because CrossEntropyLoss does that comparison internally.

**Rule of thumb:**

- Regression → model predicts a number, target is a number, loss is the difference.
- Classification → model predicts scores per class, target is just the correct index, loss checks if the right index got the highest score.

## 7. Navigating Model Modules — Parent vs Child

When you call `model.named_modules()`, PyTorch returns names as dot-separated paths, like folder paths on a computer. For example `transformer.h.0.attn.c_attn`:

- `transformer.h.0.attn` is the path to the parent module
- `c_attn` is the specific child layer you want to replace

To split this reliably:

```python
child = name.split(".")[-1]        # last segment = the layer to replace
parent = ".".join(name.split(".")[:-1])  # everything before = path to parent
parent_module = model.get_submodule(parent)
setattr(parent_module, child, new_layer)  # swap the layer in-place
```

This works for any model — LLaMA, BERT, GPT-2. The dot path is a PyTorch standard, not Hugging Face-specific.

## 8. GPT-2's Conv1D Quirk

GPT-2 uses a custom `Conv1D` class instead of `nn.Linear`. They are mathematically identical — `Conv1D` on a sequence does the same computation as a linear layer. Hugging Face kept it because they directly ported OpenAI's original 2019 code.

The dimension naming is reversed from `nn.Linear`:

- `Conv1D(nf=2304, nx=768)` → `nx` is input (768 = GPT-2 hidden dim), `nf` is output (2304 = 768 × 3 for Q, K, V projections)

When writing `LoRAGPTLayer`, use `module.nx` as `in_features` and `module.nf` as `out_features`.

## 9. Hugging Face Model Caching

When you call `AutoModelForCausalLM.from_pretrained("gpt2")`, Hugging Face downloads and caches the model automatically:

- **Linux/macOS:** `~/.cache/huggingface/hub/`
- **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`

GPT-2 base: ~500MB on disk (124M parameters), ~1GB in RAM when loaded. Don't download manually from the web UI — the Python call handles all files (config, weights, tokenizer) in one shot and verifies integrity.

## 10. Weight Merging — LoRA at Inference Time

During training: `y = xW + (x @ B) @ A` — two separate computations.

At inference, you can merge the adapter back into the original weight to remove the overhead:

```python
def merge_weights(self):
    with torch.no_grad():
        self.W += (self.B @ self.A)  # bake the update into W permanently
        self.merged = True
```

After merging: `y = xW` — same result, one computation, no extra memory. You can also `unmerge_weights()` to restore the original `W` by subtracting `B @ A`. This is how production LoRA deployments work — train with adapters, merge before serving.

## 11. Repetition Loops — Why GPT-2 Gets Stuck
Baseline GPT-2 without fine-tuning immediately falls into repetition loops — `"### Response: / The following is a list... / ### Response: / The following is a list..."` — it doesn't even understand it should answer once and stop.

The root cause: without instruction-tuning, the model has no concept of "answer format." It just predicts the next likely token, and once it hits a high-probability pattern it loops.

The fix is in generation, not training:
```python
model.generate(..., repetition_penalty=1.2, temperature=0.7)
```
Also: a 124M param model doesn't have enough capacity to memorize world facts — it will hallucinate ("Orange and Green are primary colors") even after fine-tuning. That's a scale limitation, not a LoRA bug.

## 12. What LoRA Actually Fixed — Qualitative Finding
After fine-tuning with LoRA at any rank, every output showed a clear behavioral shift: real attempts at answers, on-topic, correctly formatted, no immediate degeneration. That's the adapter teaching the model **instruction-following behavior and format** — a visible, real win even when the accuracy numbers looked flat.

**Rank comparison by eye:**
| Rank | Observation |
| :---: | :--- |
| 4 | Real answers, but in-response phrase repetition; some circular reasoning |
| 8 | Slightly more fluent, still repeats phrases; factually off on some samples |
| 16 | Least repetition inside single responses; closest in structure to target answers |

There **is** a real quality gradient with rank — higher rank produces more fluent, less repetitive outputs. But the token-level accuracy metric wasn't sensitive enough to catch it.

## 13. Why Token-Level Accuracy Was a Bad Metric Here
Token-level next-token prediction accuracy rewards getting the **exact next word right** against the target. A response can be coherent, on-topic, and grammatically correct but still score poorly if it's worded differently from the training target.

Example: model says _"Eat a balanced diet that includes healthy fats and protein"_, target says _"Maintain a diet rich in vegetables and whole grains"_ — both are valid answers, but zero overlap in tokens = low accuracy score.

**The real lesson:** flat accuracy ≠ flat output quality. The metric was measuring something narrower than "is this a good response." This is why ROUGE, BLEU, or human eval exist alongside accuracy — no single metric tells the whole story.

## 14. How My Results Line Up With the Paper
**Hu et al. 2021's headline claim:** the "intrinsic rank" needed to adapt a pretrained model is surprisingly low. They show diminishing returns going from small rank (r=1, r=4) to large rank — most of the useful adaptation lives in a low-rank subspace.

**What I got:** accuracy stayed nearly flat across r=4/8/16, and LoRA matched full fine-tuning accuracy (54.6–54.7% vs 54.86%). That's not contradicting the paper — it's the paper's own conclusion at small scale.

**Why my numbers aren't a 1:1 match to the paper:**
- **Scale** — their key results come from GPT-3 175B and RoBERTa-large, not GPT-2-small 124M. Smaller models have less redundancy for rank to express differently.
- **Metric** — they used task-specific metrics (ROUGE, exact match, GLUE scores). Raw next-token accuracy is coarser and hides qualitative differences.
- **Single seed** — differences of 0.1–0.3% val accuracy across ranks are within noise from random initialization and data shuffling. You'd need multiple seeds to distinguish real signal from variance.
- **Small dataset** — 3,000 examples and a few epochs is a smaller regime than published LoRA results; convergence behavior doesn't have to mirror fully-converged large-scale experiments.

**The honest framing:** results are broadly consistent with LoRA's core claim. The flat accuracy metric just didn't surface the more subtle quality trend that qualitative review did — which is itself a finding worth stating in a report.

