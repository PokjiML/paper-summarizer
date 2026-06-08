"""
Generate an HTML report comparing Qwen Vanilla vs Qwen+LoRA responses
on test samples, with inline diffs and metrics.
"""
import gc
import os
import torch
import pandas as pd
import numpy as np
import evaluate
from tqdm import tqdm
from datetime import datetime
import difflib
import html

from src.qwen_summarizer import QwenSummarizer
from src.data_pipeline import parse_stringified_list
from peft import PeftModel, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# HTML template with modern CSS (no external deps)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen Vanilla vs LoRA — Comparison Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f4f6f9; color: #1a1a2e; padding: 20px; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.8rem; color: #16213e; }
  .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 0.9rem; }

  /* Metrics cards */
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 30px; }
  .metric-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
  .metric-card .values { display: flex; justify-content: center; align-items: baseline; gap: 10px; }
  .metric-card .vanilla { font-size: 1.4rem; font-weight: 700; color: #e74c3c; }
  .metric-card .lora    { font-size: 1.4rem; font-weight: 700; color: #2ecc71; }
  .metric-card .delta   { font-size: 0.85rem; font-weight: 600; padding: 2px 8px; border-radius: 12px; }
  .delta-pos { background: #d4edda; color: #155724; }
  .delta-neg { background: #f8d7da; color: #721c24; }

  /* Sample cards */
  .sample-card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .sample-card h3 { font-size: 1rem; color: #16213e; margin-bottom: 12px; border-bottom: 2px solid #eef0f5; padding-bottom: 8px; }
  .field { margin-bottom: 12px; }
  .field-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }
  .field-content { background: #f8f9fb; border-radius: 6px; padding: 10px 14px; font-size: 0.88rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
  .field-content.abstract { max-height: 120px; overflow-y: auto; font-size: 0.82rem; color: #555; }
  .field-content.ref { border-left: 3px solid #3498db; }
  .field-content.vanilla { border-left: 3px solid #e74c3c; }
  .field-content.lora { border-left: 3px solid #2ecc71; }

  /* Diff table (mimicking difflib output but cleaner) */
  .diff-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; line-height: 1.55; margin-top: 6px; border-radius: 6px; overflow: hidden; }
  .diff-table td { padding: 4px 10px; vertical-align: top; }
  .diff-table .diff-num { width: 30px; text-align: right; color: #999; font-size: 0.75rem; user-select: none; }
  .diff-table .diff-sign { width: 20px; text-align: center; font-weight: bold; }
  .diff-row-equal  { background: #f9fafb; }
  .diff-row-delete { background: #ffeef0; }
  .diff-row-insert { background: #e6ffed; }
  .diff-row-delete .diff-sign { color: #cb2431; }
  .diff-row-insert .diff-sign { color: #22863a; }
  .diff-row-delete td.diff-text { color: #cb2431; }
  .diff-row-insert td.diff-text { color: #22863a; }

  .winner-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-left: 8px; }
  .winner-vanilla { background: #fde8e8; color: #c0392b; }
  .winner-lora    { background: #d4efdf; color: #1e8449; }
  .winner-tie     { background: #eaecee; color: #666; }

  .footer { text-align: center; color: #aaa; font-size: 0.75rem; margin-top: 30px; }
</style>
</head>
<body>
<div class="container">
  <h1>🔬 Qwen 3.5-2B: Vanilla vs LoRA Fine-Tuned</h1>
  <p class="subtitle">Generated on {timestamp} &nbsp;|&nbsp; {num_samples} test samples &nbsp;|&nbsp; max_new_tokens={max_tokens}</p>

  <div class="metrics-grid">
    {metrics_cards}
  </div>

  {sample_cards}

  <div class="footer">Paper Summarizer — Qwen 3.5-2B Comparison Report</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(predictions, references):
    """Compute ROUGE + BERTScore for a set of predictions."""
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)

    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
        device="cpu",
    )
    avg_bert_f1 = float(np.mean(bert_results["f1"]))

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_f1": avg_bert_f1,
    }


def generate_predictions(summarizer, samples_df, model_label, max_new_tokens):
    """Run summarizer over all rows, return list of predictions."""
    preds = []
    print(f"\nGenerating summaries with {model_label}...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=model_label):
        pred = summarizer.summarize(row["abstract"], max_new_tokens=max_new_tokens)
        preds.append(pred)
    return preds


def build_diff_html(text_a, text_b, label_a="Vanilla", label_b="LoRA"):
    """
    Build a clean side-by-side diff HTML table comparing two texts.
    Uses difflib.SequenceMatcher for word-level diffs.
    """
    # Split into words for finer diff
    words_a = text_a.split()
    words_b = text_b.split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    opcodes = matcher.get_opcodes()

    rows = []
    for tag, i1, i2, j1, j2 in opcodes:
        left = " ".join(words_a[i1:i2])
        right = " ".join(words_b[j1:j2])
        left_esc = html.escape(left)
        right_esc = html.escape(right)

        if tag == "equal":
            rows.append(f"""<tr class="diff-row-equal">
              <td class="diff-num">{i1}</td><td class="diff-text">{left_esc}</td>
              <td class="diff-num">{j1}</td><td class="diff-text">{right_esc}</td>
            </tr>""")
        elif tag == "delete":
            rows.append(f"""<tr class="diff-row-delete">
              <td class="diff-num">{i1}</td><td class="diff-text"><span class="diff-sign">−</span> {left_esc}</td>
              <td class="diff-num"></td><td class="diff-text"></td>
            </tr>""")
        elif tag == "insert":
            rows.append(f"""<tr class="diff-row-insert">
              <td class="diff-num"></td><td class="diff-text"></td>
              <td class="diff-num">{j1}</td><td class="diff-text"><span class="diff-sign">+</span> {right_esc}</td>
            </tr>""")
        elif tag == "replace":
            rows.append(f"""<tr class="diff-row-delete">
              <td class="diff-num">{i1}</td><td class="diff-text"><span class="diff-sign">−</span> {left_esc}</td>
              <td class="diff-num"></td><td class="diff-text"></td>
            </tr>""")
            rows.append(f"""<tr class="diff-row-insert">
              <td class="diff-num"></td><td class="diff-text"></td>
              <td class="diff-num">{j1}</td><td class="diff-text"><span class="diff-sign">+</span> {right_esc}</td>
            </tr>""")

    if not rows:
        rows.append(f"""<tr class="diff-row-equal">
          <td colspan="4" style="text-align:center;color:#999;">(identical)</td>
        </tr>""")

    return f"""<table class="diff-table">
      <colgroup><col style="width:40px"><col><col style="width:40px"><col></colgroup>
      <thead><tr style="background:#eef0f5;font-weight:600;font-size:0.75rem;">
        <th></th><th style="text-align:left;padding:4px 10px;">🔴 {html.escape(label_a)}</th>
        <th></th><th style="text-align:left;padding:4px 10px;">🟢 {html.escape(label_b)}</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def build_metrics_cards(vanilla_metrics, lora_metrics):
    """Generate HTML for the top metrics cards."""
    cards = []
    labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "bertscore_f1": "BERTScore F1",
    }
    for key, label in labels.items():
        v = vanilla_metrics[key]
        l = lora_metrics[key]
        delta = l - v
        cls = "delta-pos" if delta >= 0 else "delta-neg"
        cards.append(f"""<div class="metric-card">
          <div class="label">{label}</div>
          <div class="values">
            <span class="vanilla">{v:.4f}</span>
            <span style="color:#ccc;">→</span>
            <span class="lora">{l:.4f}</span>
            <span class="delta {cls}">{delta:+.4f}</span>
          </div>
        </div>""")
    return "\n".join(cards)


def build_sample_card(idx, abstract, ref, vanilla_pred, lora_pred, diff_html):
    """Generate HTML for a single sample comparison card."""
    abs_display = html.escape(abstract[:600] + ("..." if len(abstract) > 600 else ""))

    return f"""<div class="sample-card">
      <h3>📄 Sample #{idx + 1}</h3>

      <div class="field">
        <div class="field-label">📝 Abstract</div>
        <div class="field-content abstract">{abs_display}</div>
      </div>

      <div class="field">
        <div class="field-label">🎯 Reference Summary</div>
        <div class="field-content ref">{html.escape(ref)}</div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div class="field">
          <div class="field-label">🔴 Vanilla Qwen</div>
          <div class="field-content vanilla">{html.escape(vanilla_pred)}</div>
        </div>
        <div class="field">
          <div class="field-label">🟢 Qwen + LoRA</div>
          <div class="field-content lora">{html.escape(lora_pred)}</div>
        </div>
      </div>

      <div class="field" style="margin-top:4px;">
        <div class="field-label">🔍 Word-Level Diff (Vanilla vs LoRA)</div>
        {diff_html}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TEST_CSV = "data/test.csv"
    LORA_PATH = "checkpoints/best_model"
    MAX_SAMPLES = 5
    MAX_NEW_TOKENS = 150
    OUTPUT_HTML = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    print(f"Loading test dataset from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)
    df["abstract"] = df["source"].apply(parse_stringified_list)
    df["summary"] = df["target"].apply(parse_stringified_list)

    samples_df = df.head(MAX_SAMPLES).copy().reset_index(drop=True)
    print(f"Evaluating on {len(samples_df)} test samples.")

    # ----- 1. Load base model ONCE, then run Vanilla -----
    print("\n" + "=" * 50)
    print("1/2  Loading VANILLA Qwen 3.5-2B (no LoRA)")
    print("=" * 50)

    vanilla = QwenSummarizer(use_lora=False)
    vanilla_preds = generate_predictions(vanilla, samples_df, "Vanilla Qwen", MAX_NEW_TOKENS)
    vanilla_metrics = compute_metrics(vanilla_preds, samples_df["summary"].tolist())

    # ----- 2. Apply LoRA adapter to the SAME base model (no re-download) -----
    print("\n" + "=" * 50)
    print("2/2  Applying LoRA adapter to the same base model")
    print("=" * 50)

    # Reuse the tokenizer from vanilla
    lora_summarizer = QwenSummarizer.__new__(QwenSummarizer)
    lora_summarizer.model_id = vanilla.model_id
    lora_summarizer.device = vanilla.device
    lora_summarizer.tokenizer = vanilla.tokenizer
    lora_summarizer.use_4bit = vanilla.use_4bit

    # Apply LoRA to the base model (reuse the already-loaded model, no re-download)
    if vanilla.use_4bit and vanilla.device == "cuda":
        prepare_model_for_kbit_training(vanilla.model)
    lora_summarizer.model = PeftModel.from_pretrained(vanilla.model, LORA_PATH)
    lora_summarizer.model.eval()

    lora_preds = generate_predictions(lora_summarizer, samples_df, "LoRA Qwen", MAX_NEW_TOKENS)
    lora_metrics = compute_metrics(lora_preds, samples_df["summary"].tolist())

    del vanilla, lora_summarizer
    gc.collect()
    torch.cuda.empty_cache()

    # ----- 3. Print aggregate metrics -----
    print("\n" + "=" * 50)
    print("AGGREGATE METRICS")
    print("=" * 50)
    for key, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                        ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore F1")]:
        v = vanilla_metrics[key]
        l = lora_metrics[key]
        print(f"  {label:15s}  Vanilla: {v:.4f}  |  LoRA: {l:.4f}  |  Δ: {l-v:+.4f}")

    # ----- 4. Build HTML report -----
    print(f"\nBuilding HTML report → {OUTPUT_HTML} ...")

    sample_cards = []
    abstracts = samples_df["abstract"].tolist()
    references = samples_df["summary"].tolist()

    for i in range(len(samples_df)):
        diff_html = build_diff_html(vanilla_preds[i], lora_preds[i])
        card = build_sample_card(i, abstracts[i], references[i],
                                 vanilla_preds[i], lora_preds[i], diff_html)
        sample_cards.append(card)

    metrics_cards = build_metrics_cards(vanilla_metrics, lora_metrics)

    html_content = (
        HTML_TEMPLATE
        .replace("{timestamp}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        .replace("{num_samples}", str(len(samples_df)))
        .replace("{max_tokens}", str(MAX_NEW_TOKENS))
        .replace("{metrics_cards}", metrics_cards)
        .replace("{sample_cards}", "\n".join(sample_cards))
    )

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Report saved to: {os.path.abspath(OUTPUT_HTML)}")
    print(f"   Open it in your browser to see side-by-side diffs!")


if __name__ == "__main__":
    main()
