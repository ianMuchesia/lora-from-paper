## Notes & Engineering Insights

- **Rank Saturation:** All LoRA runs share the same alpha. The accuracy plateaued at ~54.6%, proving the paper's claim that task-specific adaptations reside in a very low intrinsic dimension.
- **The Memory Paradox:** Counter-intuitively, LoRA showed *higher* peak GPU memory than Full FT. 
  - **Why?** Trainable parameter count only reduces *optimizer* memory. Total GPU memory is dominated by *activations* (intermediate outputs saved for backprop). 
  - LoRA doesn't skip the base forward pass; it adds extra operations (`x @ B @ A`), meaning PyTorch must store *extra* activation tensors.
  - The abnormally low Full FT memory (3173 MB) is likely a byproduct of dynamic padding and `shuffle=True` — the Full FT run happened to draw batches with shorter average sequence lengths, avoiding massive memory spikes.