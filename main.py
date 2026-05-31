from src.get_arxiv_api import fetch_recent_ai_papers
from src.qwen_summarizer import QwenSummarizer

def main():
    print("Fetching recent AI papers from arXiv...")
    papers = fetch_recent_ai_papers(max_results=3)
    
    if not papers:
        print("No papers found.")
        return
        
    print(f"Found {len(papers)} papers. Initializing summarizer...")
    # You can set use_lora=True to continue fine-tuning or load a checkpoint
    summarizer = QwenSummarizer(use_lora=False)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{'='*50}")
        print(f"Paper {i}: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Published: {paper['published']}")
        print(f"{'='*50}")
        
        print("\nOriginal Abstract:")
        print(paper['abstract'])
        
        print("\nGenerating Summary...")
        summary = summarizer.summarize(paper['abstract'])
        print(f"\nAI Summary:\n{summary}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

