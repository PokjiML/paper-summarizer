import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import ast
import torch
from transformers import AutoTokenizer
from src.prompt import prompt_template

def parse_stringified_list(text_repr):
    """Parses a string representation of a list of sentences into a single string."""
    try:
        sentences = ast.literal_eval(text_repr)
        if isinstance(sentences, list):
            return " ".join(sentences)
        return text_repr
    except (ValueError, SyntaxError):
        return str(text_repr)

def prepare_training_data(csv_path, tokenizer_name="Qwen/Qwen3.5-2B", max_samples=None):
    """
    Reads the CSV, merges sentences, creates the prompt and target,
    and returns tokenized examples ready for training or generation.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    # Parse abstract (source) and summary (target)
    df['abstract'] = df['source'].apply(parse_stringified_list)
    df['summary'] = df['target'].apply(parse_stringified_list)
    
    # Initialize tokenizer
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenized_datasets = []
    
    print("Formatting and tokenizing data...")
    for _, row in df.iterrows():
        abstract = row['abstract']
        summary = row['summary']
        
        # Fill the user prompt with the abstract
        user_content = prompt_template.format(abstract).strip()
        
        # We prepare the messages format for Qwen's chat template
        # For training/fine-tuning, we concatenate user prompt + assistant response
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": summary}
        ]
        
        # Apply Qwen's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare inputs and labels for causal LM
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask out user instruction so model only learns to predict the summary.
        # This requires finding where the assistant's response starts.
        user_only_msgs = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(
            user_only_msgs, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
        prompt_len = prompt_encoded["input_ids"].shape[1]
        
        # Ignore loss on the prompt itself
        labels[:prompt_len] = -100 
        
        tokenized_datasets.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_text": prompt_text,  # for generation
            "original_abstract": abstract,
            "original_summary": summary
        })
        
    print(f"Prepared {len(tokenized_datasets)} training instances.")
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    # Test the pipeline
    dataset, tokenizer = prepare_training_data("data/train.csv", max_samples=2)
    
    print("\n--- Example 1 ---")
    print("PROMPT TEXT:")
    print(dataset[0]['prompt_text'])
    print("\nEXPECTED SUMMARY:")
    print(dataset[0]['original_summary'])
    print("\nInput IDs shape:", dataset[0]['input_ids'].shape)
    
