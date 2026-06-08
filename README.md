# Paper Summarizer

Fine-tuning Qwen 3.5 to summarize AI paper abstracts, with a comparison of LoRA fine-tuning vs just writing a better prompt.

## What's this about?

The core question: if you have a small LLM like Qwen 3.5 (2B params), does LoRA fine-tuning on a summarization dataset actually beat a well-written zero-shot prompt? This repo runs that experiment.

The model is fine-tuned on the [SciTLDR](https://huggingface.co/datasets/allenai/scitldr) dataset (arXiv AI paper abstracts → human summaries). Two variants are compared:

- **Vanilla Qwen + crafted prompt** — the base model with a detailed instruction telling it exactly how to format the summary
- **LoRA Qwen** — the base model fine-tuned with LoRA adapters on the simple prompt it was trained on

Results get dumped into an HTML report, and logged to W&B.

### Evaluation

Both variants are evaluated using:

- **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L) — measures n-gram overlap between generated and reference summaries
- **BERTScore** — computes semantic similarity using contextual embeddings (distilbert-base-uncased)

These metrics are computed during training (on the validation set) and in the final comparison scripts.

## Project structure

```
paper-summarizer/
├── main.py                  # Quick smoke test: fetch arXiv papers & summarize
├── test.py                  # (older test script)
├── src/
│   ├── qwen_summarizer.py   # QwenSummarizer class — loads model, handles LoRA
│   ├── data_pipeline.py     # CSV → tokenized dataset (prompt + target)
│   ├── train.py             # LoRA training loop with eval (loss, ROUGE, BERTScore)
│   ├── get_arxiv_api.py     # Fetch recent AI papers from arXiv
│   └── prompt.py            # Simple prompt template used for fine-tuning
├── evaluate_test_set.py     # Evaluate vanilla vs LoRA, push table to W&B
├── visualize_prompt_vs_lora.py  # Generate HTML comparison report
├── visualize_comparison.py  # (older HTML report script)
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── checkpoints/
    └── best_model/          # LoRA adapter weights + tokenizer
```

## Usage

### Quick smoke test

Fetches a few recent arXiv papers and summarizes them:

```bash
uv run python main.py
```

### Training

```bash
uv run python src/train.py
```

This fine-tunes Qwen 3.5 2B with LoRA, logs metrics to W&B, and saves the best checkpoint.

### Evaluate & compare

```bash
uv run python evaluate_test_set.py
uv run python visualize_prompt_vs_lora.py
```

The first generates predictions from both models and logs them to W&B. The second produces an HTML report with diff-highlighted side-by-side summaries.

## Key dependencies

- `transformers` + `peft` — model loading and LoRA
- `bitsandbytes` — 4-bit quantization
- `torch` — obviously
- `evaluate`, `rouge-score`, `bert-score` — evaluation metrics
- `wandb` — experiment tracking
- `arxiv` — arXiv API client
