import arxiv

def fetch_recent_ai_papers(max_results=5):
    """
    Fetches the most recent AI papers from arXiv.
    Focuses on Computer Science categories like AI, Computation and Language,
    Computer Vision, and Machine Learning.
    """
    client = arxiv.Client()
    
    # Query for standard AI/ML categories
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.LG",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary.replace('\n', ' '),
            "url": result.pdf_url,
            "published": result.published
        })
        
    return papers
