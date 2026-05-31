import os
import sys

# Add the project root to the Python path to allow 'src' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import evaluate
import numpy as np
import wandb
from src.data_pipeline import prepare_training_data
from src.qwen_summarizer import QwenSummarizer

class SummarizationDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "prompt_text": item["prompt_text"],
            "original_summary": item["original_summary"]
        }

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Loss"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            
    return total_loss / len(dataloader)

def evaluate_generation(model, tokenizer, dataloader, device, max_new_tokens=150, sample_size=50):
    model.eval()
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    predictions = []
    references = []
    
    print(f"Generating summaries for {min(sample_size, len(dataloader.dataset))} validation examples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            if i >= sample_size:
                break
                
            # For generation, we only need the prompt text, not the padded train targets
            prompts = batch["prompt_text"]
            refs = batch["original_summary"]
            
            # Since generation might have different prompt lengths, we process batch size 1 here for simplicity/accuracy
            for prompt, ref in zip(prompts, refs):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]
                
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for evaluation
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Extract only generated tokens
                summary_tokens = output_tokens[0][input_length:]
                pred = tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()
                
                predictions.append(pred)
                references.append(ref)
                
    # Compute metrics
    rouge_results = rouge.compute(predictions=predictions, references=references)
    
    # Compute BERTScore (using distilbert for speed, but standard requires roberta-large)
    # We use a smaller model for BERTScore to speed up evaluation during training
    bert_results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    avg_bert_f1 = np.mean(bert_results["f1"])
    
    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_f1": avg_bert_f1,
        "predictions": predictions[:2], # save a few for inspection
        "references": references[:2]
    }

def train(epochs=5, batch_size=2, lr=2e-4, eval_every_n_epochs=1, patience=2, max_samples=None):
    wandb.init(
        project="paper-summarizer",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_samples": max_samples
        }
    )
    
    print("Preparing datasets...")
    train_data, tokenizer = prepare_training_data("data/train.csv", max_samples=max_samples)
    val_data, _ = prepare_training_data("data/val.csv", max_samples=max_samples)
    
    train_dataset = SummarizationDataset(train_data)
    val_dataset = SummarizationDataset(val_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # For generation, batch size 1 is simpler due to variable prompt lengths
    val_generation_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    # Initialize Model for Fine-Tuning
    summarizer = QwenSummarizer(use_lora=True)
    model = summarizer.model
    device = summarizer.device
    
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    
    best_bertscore = -1.0
    epochs_no_improve = 0
    save_dir = "checkpoints/best_model"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*30}\nEpoch {epoch}/{epochs}\n{'='*30}")
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train/loss": avg_train_loss, "epoch": epoch})
        
        if epoch % eval_every_n_epochs == 0:
            val_loss = evaluate_loss(model, val_dataloader, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Evaluate generation metrics on a subset to save time, evaluate on 50 batches
            metrics = evaluate_generation(model, tokenizer, val_generation_loader, device, sample_size=12)
            
            print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
            
            print("\nSample Generation:")
            print(f"Reference: {metrics['references'][0][:200]}...")
            print(f"Prediction: {metrics['predictions'][0][:200]}...")
            
            # Create a wandb Table for generated samples
            table = wandb.Table(columns=["Reference", "Prediction"])
            for ref, pred in zip(metrics['references'], metrics['predictions']):
                table.add_data(ref, pred)
                
            wandb.log({
                "val/loss": val_loss,
                "val/rouge1": metrics['rouge1'],
                "val/rouge2": metrics['rouge2'],
                "val/rougeL": metrics['rougeL'],
                "val/bertscore_f1": metrics['bertscore_f1'],
                "val/generations": table,
                "epoch": epoch
            })
            
            # Check for improvement and early stopping
            if metrics["bertscore_f1"] > best_bertscore:
                print(f"BERTScore improved from {best_bertscore:.4f} to {metrics['bertscore_f1']:.4f}. Saving model...")
                best_bertscore = metrics["bertscore_f1"]
                epochs_no_improve = 0
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                
                # Save best model to wandb
                artifact = wandb.Artifact("qwen-summarizer-best", type="model")
                artifact.add_dir(save_dir)
                wandb.log_artifact(artifact)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered! Best BERTScore: {best_bertscore:.4f}")
                    break
                    
    wandb.finish()

if __name__ == "__main__":
    train(epochs=5, batch_size=1, max_samples=1)