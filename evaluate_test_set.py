import gc
import torch
import pandas as pd
from src.qwen_summarizer import QwenSummarizer
from src.data_pipeline import parse_stringified_list

def main():
    print("Loading test dataset (test.csv)...")
    df = pd.read_csv("data/test.csv")
    
    # We will test on the first 3 examples
    sample_df = df.head(3).copy()
    
    # Parse source (abstract) and target (summary)
    sample_df['abstract'] = sample_df['source'].apply(parse_stringified_list)
    sample_df['summary'] = sample_df['target'].apply(parse_stringified_list)
    
    results = []
    
    # ---------------------------------------------
    # 1. Evaluate Vanilla Qwen (No LoRA)
    # ---------------------------------------------
    print("\n" + "="*50)
    print("Initializing Vanilla Qwen 3.5 2B (No LoRA)")
    print("="*50)
    vanilla_summarizer = QwenSummarizer(use_lora=False)
    
    vanilla_predictions = []
    for i, row in sample_df.iterrows():
        print(f"\nEvaluating Vanilla Qwen on Sample {i+1}...")
        pred = vanilla_summarizer.summarize(row['abstract'])
        vanilla_predictions.append(pred)
        
    # Free memory
    del vanilla_summarizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # ---------------------------------------------
    # 2. Evaluate Fine-Tuned Qwen (with LoRA)
    # ---------------------------------------------
    print("\n" + "="*50)
    print("Initializing Fine-Tuned Qwen 3.5 2B (With LoRA)")
    print("="*50)
    try:
        lora_summarizer = QwenSummarizer(
            use_lora=True, 
            lora_weights_path="checkpoints/best_model"
        )
        
        lora_predictions = []
        for i, row in sample_df.iterrows():
            print(f"\nEvaluating LoRA Qwen on Sample {i+1}...")
            pred = lora_summarizer.summarize(row['abstract'])
            lora_predictions.append(pred)
            
        # Free memory
        del lora_summarizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nError loading LoRA model (has it been trained yet?): {e}")
        lora_predictions = ["Error/Not trained"] * len(sample_df)
    
    # ---------------------------------------------
    # 3. Print Comparison
    # ---------------------------------------------
    print("\n\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    for i, row in sample_df.iterrows():
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"ABSTRACT:\n{row['abstract'][:300]}...")
        print(f"\nORIGINAL SUMMARY (Target):\n{row['summary']}")
        print(f"\nVANILLA QWEN PREDICTION:\n{vanilla_predictions[i]}")
        print(f"\nLORA QWEN PREDICTION:\n{lora_predictions[i]}")

if __name__ == "__main__":
    main()
