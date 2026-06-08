"""
Compare LoRA fine-tuned Qwen (simple prompt) vs Vanilla Qwen (well-crafted prompt).
Answers: "Did LoRA really help, or would a better prompt suffice?"
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
# Prompt templates
# ---------------------------------------------------------------------------
# LoRA was fine-tuned on this simple prompt — we keep it as-is
SIMPLE_PROMPT = None  # uses QwenSummarizer's default

# Vanilla gets a well-crafted prompt to see if prompting alone can match LoRA
CRAFTED_PROMPT = (
    "Summarize the following AI research paper abstract in exactly one concise paragraph. "
    "Be specific about the key contribution and findings. "
    "Do NOT use markdown, bullet points, numbered lists, or any special formatting. "
    "Output only plain text as a single paragraph with no line breaks.\n\n"
    "Abstract: {abstract}\n\n"
    "One-paragraph summary:"
)

# ---------------------------------------------------------------------------
# HTML template (same as visualize_comparison.py)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vanilla + Prompt vs LoRA — Comparison Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f4f6f9; color: #1a1a2e; padding: 20px; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.8rem; color: #16213e; }
  .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 0.9rem; }

  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 30px; }
  .metric-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
  .metric-card .values { display: flex; justify-content: center; align-items: baseline; gap: 10px; }
  .metric-card .vanilla { font-size: 1.4rem; font-weight: 700; color: #e67e22; }
  .metric-card .lora    { font-size: 1.4rem; font-weight: 700; color: #2ecc71; }
  .metric-card .delta   { font-size: 0.85rem; font-weight: 600; padding: 2px 8px; border-radius: 12px; }
  .delta-pos { background: #d4edda; color: #155724; }
  .delta-neg { background: #f8d7da; color: #721c24; }

  .sample-card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .sample-card h3 { font-size: 1rem; color: #16213e; margin-bottom: 12px; border-bottom: 2px solid #eef0f5; padding-bottom: 8px; }
  .field { margin-bottom: 12px; }
  .field-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }
  .field-content { background: #f8f9fb; border-radius: 6px; padding: 10px 14px; font-size: 0.88rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
  .field-content.abstract { max-height: 120px; overflow-y: auto; font-size: 0.82rem; color: #555; }
  .field-content.ref { border-left: 3px solid #3498db; }
  .field-content.prompt { border-left: 3px solid #e67e22; }
  .field-content.lora { border-left: 3px solid #2ecc71; }

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

  .footer { text-align: center; color: #aaa; font-size: 0.75rem; margin-top: 30px; }
</style>
</head>
<body>
<div class="container">
  <h1>🔬 Vanilla Qwen + Well-Crafted Prompt vs LoRA Fine-Tuned</h1>
  <p class="subtitle">Generated on {timestamp} &nbsp;|&nbsp; {num_samples} test samples &nbsp;|&nbsp; max_new_tokens={max_tokens}</p>
  <p class="subtitle" style="font-size:0.8rem;color:#999;">
    🟠 Vanilla prompt: <em>"Summarize in one concise paragraph, no markdown, plain text only"</em><br>
    🟢 LoRA prompt: <em>Simple "Summarize this abstract"</em> (matches training distribution)
  </p>

  <div class="metrics-grid">
    {metrics_cards}
  </div>

  {sample_cards}

  <div class="footer">Paper Summarizer — Prompt Engineering vs Fine-Tuning Comparison</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers (same logic as visualize_comparison.py)
# ---------------------------------------------------------------------------

def compute_metrics(predictions, references):
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


def generate_predictions(summarizer, samples_df, model_label, max_new_tokens, custom_prompt=None):
    preds = []
    print(f"\nGenerating summaries with {model_label}...")
    for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc=model_label):
        pred = summarizer.summarize(row["abstract"], max_new_tokens=max_new_tokens, custom_prompt=custom_prompt)
        preds.append(pred)
    return preds


def build_diff_html(text_a, text_b, label_a="Vanilla+Prompt", label_b="LoRA"):
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
        rows.append("""<tr class="diff-row-equal">
          <td colspan="4" style="text-align:center;color:#999;">(identical)</td>
        </tr>""")

    return f"""<table class="diff-table">
      <colgroup><col style="width:40px"><col><col style="width:40px"><col></colgroup>
      <thead><tr style="background:#eef0f5;font-weight:600;font-size:0.75rem;">
        <th></th><th style="text-align:left;padding:4px 10px;">🟠 {html.escape(label_a)}</th>
        <th></th><th style="text-align:left;padding:4px 10px;">🟢 {html.escape(label_b)}</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def build_metrics_cards(prompt_metrics, lora_metrics):
    cards = []
    labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "bertscore_f1": "BERTScore F1",
    }
    for key, label in labels.items():
        v = prompt_metrics[key]
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


def build_sample_card(idx, abstract, ref, prompt_pred, lora_pred, diff_html):
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
          <div class="field-label">🟠 Vanilla Qwen + Well-Crafted Prompt</div>
          <div class="field-content prompt">{html.escape(prompt_pred)}</div>
        </div>
        <div class="field">
          <div class="field-label">🟢 Qwen + LoRA (simple prompt)</div>
          <div class="field-content lora">{html.escape(lora_pred)}</div>
        </div>
      </div>

      <div class="field" style="margin-top:4px;">
        <div class="field-label">🔍 Word-Level Diff (Prompt vs LoRA)</div>
        {diff_html}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TEST_CSV = "data/test.csv"
    LORA_PATH = "checkpoints/best_model"
    MAX_SAMPLES = 10
    MAX_NEW_TOKENS = 150
    OUTPUT_HTML = f"comparison_prompt_vs_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    print(f"Loading test dataset from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)
    df["abstract"] = df["source"].apply(parse_stringified_list)
    df["summary"] = df["target"].apply(parse_stringified_list)

    samples_df = df.head(MAX_SAMPLES).copy().reset_index(drop=True)
    print(f"Evaluating on {len(samples_df)} test samples.")

    # ----- 1. Load base model ONCE -----
    print("\n" + "=" * 60)
    print("1/2  Vanilla Qwen + Well-Crafted Prompt (no LoRA)")
    print("=" * 60)

    vanilla = QwenSummarizer(use_lora=False)
    prompt_preds = generate_predictions(
        vanilla, samples_df, "Vanilla+Prompt", MAX_NEW_TOKENS,
        custom_prompt=CRAFTED_PROMPT,
    )
    prompt_metrics = compute_metrics(prompt_preds, samples_df["summary"].tolist())

    # ----- 2. Apply LoRA to the SAME base model -----
    print("\n" + "=" * 60)
    print("2/2  Qwen + LoRA (simple prompt — matches training)")
    print("=" * 60)

    lora_summarizer = QwenSummarizer.__new__(QwenSummarizer)
    lora_summarizer.model_id = vanilla.model_id
    lora_summarizer.device = vanilla.device
    lora_summarizer.tokenizer = vanilla.tokenizer
    lora_summarizer.use_4bit = vanilla.use_4bit

    if vanilla.use_4bit and vanilla.device == "cuda":
        prepare_model_for_kbit_training(vanilla.model)
    lora_summarizer.model = PeftModel.from_pretrained(vanilla.model, LORA_PATH)
    lora_summarizer.model.eval()

    lora_preds = generate_predictions(
        lora_summarizer, samples_df, "LoRA Qwen", MAX_NEW_TOKENS
    )
    lora_metrics = compute_metrics(lora_preds, samples_df["summary"].tolist())

    del vanilla, lora_summarizer
    gc.collect()
    torch.cuda.empty_cache()

    # ----- 3. Print aggregate metrics -----
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS: Vanilla+Prompt  vs  LoRA (simple prompt)")
    print("=" * 60)
    for key, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                        ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore F1")]:
        v = prompt_metrics[key]
        l = lora_metrics[key]
        winner = "LoRA wins 🟢" if l > v else "Prompt wins 🟠" if v > l else "Tie ⚪"
        print(f"  {label:15s}  Prompt: {v:.4f}  |  LoRA: {l:.4f}  |  Δ: {l-v:+.4f}  ({winner})")

    # ----- 4. Build HTML report -----
    print(f"\nBuilding HTML report → {OUTPUT_HTML} ...")

    sample_cards = []
    abstracts = samples_df["abstract"].tolist()
    references = samples_df["summary"].tolist()

    for i in range(len(samples_df)):
        diff_html = build_diff_html(prompt_preds[i], lora_preds[i])
        card = build_sample_card(i, abstracts[i], references[i],
                                 prompt_preds[i], lora_preds[i], diff_html)
        sample_cards.append(card)

    metrics_cards = build_metrics_cards(prompt_metrics, lora_metrics)

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
    print(f"   Open it in your browser to compare Prompt vs LoRA!")


if __name__ == "__main__":
    main()
