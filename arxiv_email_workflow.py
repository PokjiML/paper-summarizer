import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
import arxiv
from src.qwen_summarizer import QwenSummarizer

STATE_FILE = "last_run_state.json"

def get_last_checked_date():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return datetime.fromisoformat(data["last_checked"])
    # Default to 24 hours ago if running for the first time
    return datetime.now(timezone.utc) - timedelta(days=1)

def save_last_checked_date(date: datetime):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_checked": date.isoformat()}, f)

def fetch_new_papers(last_checked, max_results=20):
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.LG",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    new_papers = []
    latest_date = last_checked

    for result in client.results(search):
        # Arxiv uses UTC timezone for published dates
        if result.published > last_checked:
            new_papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary.replace('\n', ' '),
                "url": result.pdf_url,
                "published": result.published
            })
            if result.published > latest_date:
                latest_date = result.published
                
    return new_papers, latest_date

def send_email(subject, body, sender_email, sender_password, receiver_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    # Email configuration - ideally set these as environment variables in your cloud deployment!
    sender_email = os.environ.get("SENDER_EMAIL", "your_email@gmail.com")
    sender_password = os.environ.get("SENDER_PASSWORD", "your_app_password")
    receiver_email = os.environ.get("RECEIVER_EMAIL", "your_email@gmail.com")
    
    last_checked = get_last_checked_date()
    print(f"Checking for new papers since {last_checked}...")
    
    papers, new_last_checked = fetch_new_papers(last_checked, max_results=30)
    print(f"Found {len(papers)} new papers.")
    
    if not papers:
        save_last_checked_date(new_last_checked)
        return

    print("Loading model for judging and summarization...")
    # NOTE: Set lora_weights_path to where you store your weights on the cloud
    summarizer = QwenSummarizer(use_lora=True, lora_weights_path="checkpoints/best_model")
    
    worthy_papers = []
    
    for i, paper in enumerate(papers, 1):
        print(f"\nProcessing {i}/{len(papers)}: {paper['title']}")
        
        is_worthy = summarizer.judge_paper(paper['title'], paper['abstract'])
        
        if is_worthy:
            print("=> Decision: WORTHY")
            summary = summarizer.summarize(paper['abstract'])
            worthy_papers.append({
                "title": paper['title'],
                "authors": ", ".join(paper['authors'][:3]) + (" et al." if len(paper['authors']) > 3 else ""),
                "published": paper['published'].strftime('%Y-%m-%d'),
                "url": paper['url'],
                "summary": summary
            })
        else:
            print("=> Decision: PASSED")
            
    if worthy_papers:
        # Build HTML Email
        html_body = "<h2>Daily AI Papers Summary</h2>"
        html_body += f"<p>Found {len(worthy_papers)} highly novel and impactful papers out of {len(papers)} recent submissions.</p><hr/>"
        
        for paper in worthy_papers:
            html_body += f"<h3><a href='{paper['url']}'>{paper['title']}</a></h3>"
            html_body += f"<p><b>Authors:</b> {paper['authors']} | <b>Date:</b> {paper['published']}</p>"
            html_body += f"<p><b>AI Summary:</b><br/>{paper['summary']}</p>"
            html_body += "<hr/>"
            
        print(f"\nSending email with {len(worthy_papers)} summaries...")
        send_email(
            subject=f"AI Papers Update ({datetime.now().strftime('%Y-%m-%d')})",
            body=html_body,
            sender_email=sender_email,
            sender_password=sender_password,
            receiver_email=receiver_email
        )
    else:
        print("No papers passed the novelty threshold today.")
        
    save_last_checked_date(new_last_checked)

if __name__ == "__main__":
    main()