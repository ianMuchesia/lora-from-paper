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
- `CrossEntropyLoss` looks at the target index, finds the model's score at that index, and asks: *is that score higher than the others?* If yes, loss is low.

**The teacher/student analogy:**
- Answer key says: correct answer is category `0`.
- Model outputs: `[4.5, 1.2]` (score 4.5 for category 0, 1.2 for category 1).
- Loss checks: is 4.5 the highest? Yes — low loss.
- You never provide a "target score" because CrossEntropyLoss does that comparison internally.

**Rule of thumb:**
- Regression → model predicts a number, target is a number, loss is the difference.
- Classification → model predicts scores per class, target is just the correct index, loss checks if the right index got the highest score.

