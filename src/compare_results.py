"""
Day 7 — reads the results JSON from each experiment and builds
experiments/lora_vs_full_finetuning.md automatically.

Usage:
    python compare_results.py --results_dir experiments --out experiments/lora_vs_full_finetuning.md
"""
import argparse
import json
import os


EXPERIMENTS = ["full_finetune", "rank_4", "rank_8", "rank_16"]
DISPLAY_NAMES = {
    "full_finetune": "Full FT",
    "rank_4": "LoRA r=4",
    "rank_8": "LoRA r=8",
    "rank_16": "LoRA r=16",
}


def load_results(results_dir):
    results = {}
    for name in EXPERIMENTS:
        path = os.path.join(results_dir, f"{name}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: missing {path}, skipping from table")
    return results


def build_markdown(results):
    lines = [
        "# LoRA vs Full Fine-Tuning\n",
        "| Method | Val Accuracy | Trainable Params | Avg Epoch Time (s) | Peak GPU Memory (MB) |",
        "|---|---|---|---|---|",
    ]
    for name in EXPERIMENTS:
        if name not in results:
            continue
        r = results[name]
        acc = f"{r['final_val_accuracy']:.2f}%"
        params = f"{r['trainable_params']:,}"
        time_s = f"{r['avg_train_time_sec']:.1f}"
        mem = f"{r['peak_gpu_memory_mb']:.1f}" if r.get("peak_gpu_memory_mb") else "N/A"
        lines.append(f"| {DISPLAY_NAMES[name]} | {acc} | {params} | {time_s} | {mem} |")

    lines.append("\n## Notes\n")
    lines.append("- All LoRA runs share the same alpha, so trainable-parameter and "
                  "memory differences are attributable to rank alone.")
    lines.append("- Full FT uses a lower learning rate than the LoRA runs "
                  "(see configs) since all weights are trainable.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments")
    parser.add_argument("--out", default="experiments/lora_vs_full_finetuning.md")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    markdown = build_markdown(results)

    with open(args.out, "w") as f:
        f.write(markdown)

    print(f"Wrote comparison table to {args.out}\n")
    print(markdown)
