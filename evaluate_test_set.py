import gc
import os
import torch
import pandas as pd
import numpy as np
import evaluate
import wandb
from tqdm import tqdm
from src.qwen_summarizer import QwenSummarizer
from src.data_pipeline import parse_stringified_list


def run_evaluation(summarizer, samples_df, model_label, max_new_tokens=150):
    """
    Generate summaries for all samples and compute ROUGE + BERTScore.
    Returns dict of metrics and list of (reference, prediction) pairs.
    """
    predictions = []
    references = []

    print(f"\nGenerating summaries with {model_label}...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=model_label):
        ref = row["summary"]
        pred = summarizer.summarize(row["abstract"], max_new_tokens=max_new_tokens)
        predictions.append(pred)
        references.append(ref)

    # Compute ROUGE
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # Compute BERTScore (use distilbert for speed; swap to roberta-large for publication-grade)
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    avg_bert_f1 = float(np.mean(bert_results["f1"]))

    metrics = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_f1": avg_bert_f1,
    }

    return metrics, predictions, references


def build_comparison_table(abstracts, refs, vanilla_preds, lora_preds):
    """Build a W&B table with side-by-side comparison columns."""
    table = wandb.Table(
        columns=[
            "Abstract",
            "Reference Summary",
            "Vanilla Qwen Prediction",
            "LoRA Qwen Prediction",
        ]
    )
    for abs_, ref, van, lor in zip(abstracts, refs, vanilla_preds, lora_preds):
        table.add_data(
            abs_[:500] + "..." if len(abs_) > 500 else abs_,
            ref[:500] + "..." if len(ref) > 500 else ref,
            van[:500] + "..." if len(van) > 500 else van,
            lor[:500] + "..." if len(lor) > 500 else lor,
        )
    return table


def main():
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    TEST_CSV = "data/test.csv"
    LORA_PATH = "checkpoints/best_model"
    MAX_SAMPLES = 10 
    MAX_NEW_TOKENS = 150

    print(f"Loading test dataset from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)

    # Parse source (abstract) and target (summary)
    df["abstract"] = df["source"].apply(parse_stringified_list)
    df["summary"] = df["target"].apply(parse_stringified_list)

    # Use a subset for speed (full test set is 619; BERTScore is heavy)
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        samples_df = df.head(MAX_SAMPLES).copy().reset_index(drop=True)
    else:
        samples_df = df.copy().reset_index(drop=True)

    print(f"Evaluating on {len(samples_df)} test samples.")

    # ------------------------------------------------------------------
    # Initialize W&B
    # ------------------------------------------------------------------
    wandb.init(
        project="paper-summarizer",
        name="eval-vanilla-vs-lora",
        job_type="evaluation",
        config={
            "test_samples": len(samples_df),
            "lora_path": LORA_PATH,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )

    # ------------------------------------------------------------------
    # 1. Vanilla Qwen (No LoRA)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1/2  Loading VANILLA Qwen 3.5-2B (no LoRA)")
    print("=" * 50)

    vanilla_summarizer = QwenSummarizer(use_lora=False)
    vanilla_metrics, vanilla_preds, references = run_evaluation(
        vanilla_summarizer, samples_df, "Vanilla Qwen", max_new_tokens=MAX_NEW_TOKENS
    )

    del vanilla_summarizer
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Fine-Tuned Qwen (with LoRA)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2/2  Loading FINE-TUNED Qwen 3.5-2B (with LoRA)")
    print("=" * 50)

    lora_available = os.path.isdir(LORA_PATH)
    if lora_available:
        lora_summarizer = QwenSummarizer(
            use_lora=True, lora_weights_path=LORA_PATH
        )
        lora_metrics, lora_preds, _ = run_evaluation(
            lora_summarizer, samples_df, "LoRA Qwen", max_new_tokens=MAX_NEW_TOKENS
        )

        del lora_summarizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"WARNING: LoRA checkpoint not found at '{LORA_PATH}'. Skipping LoRA evaluation.")
        lora_metrics = {k: 0.0 for k in vanilla_metrics}
        lora_preds = ["[LoRA not trained yet]"] * len(samples_df)

    # ------------------------------------------------------------------
    # 3. Log aggregate metrics to W&B
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("AGGREGATE METRICS")
    print("=" * 50)

    for metric_name in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
        v = vanilla_metrics[metric_name]
        l = lora_metrics[metric_name]
        delta = l - v
        print(f"  {metric_name:15s}  Vanilla: {v:.4f}  |  LoRA: {l:.4f}  |  Δ: {delta:+.4f}")

        wandb.log({
            f"test/vanilla_{metric_name}": v,
            f"test/lora_{metric_name}": l,
            f"test/delta_{metric_name}": delta,
        })

    # ------------------------------------------------------------------
    # 4. Build and log side-by-side comparison table
    # ------------------------------------------------------------------
    abstracts = samples_df["abstract"].tolist()
    comparison_table = build_comparison_table(
        abstracts, references, vanilla_preds, lora_preds
    )
    wandb.log({"test/comparison_table": comparison_table})

    # ------------------------------------------------------------------
    # 5. Log a few full-text samples to the console for quick inspection
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS (first 3)")
    print("=" * 50)

    for i in range(min(3, len(samples_df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"ABSTRACT:       {abstracts[i][:250]}...")
        print(f"REFERENCE:      {references[i]}")
        print(f"VANILLA QWEN:   {vanilla_preds[i]}")
        if lora_available:
            print(f"LORA QWEN:      {lora_preds[i]}")

    wandb.finish()
    print("\nDone! Results logged to W&B.")


if __name__ == "__main__":
    main()
