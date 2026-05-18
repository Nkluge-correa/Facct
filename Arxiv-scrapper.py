import arxivscraper
import pandas as pd
from datetime import datetime
import time
import os
import random


CATEGORIES = ['cs.AI', 'cs.LG', 'stat.ML']   # used all relevant not just cs.ai as it was still quite fast 
START_YEAR = 2020
END_YEAR = 2026
TARGET_PER_CATEGORY = 1000 #Did not use as I did not create this logic in my script, I tried getting as many papers as possible instead of limiting it since it was not taking a lot of time
# I have tried to keep the taxonomy same as before for reasonable comparison
TAXONOMY = {
    "Authoritarianism": ["surveillance", "censorship", "predictive policing", "social scoring"],
    "Bias & Inequality": ["bias", "fairness", "discrimination", "inequality"],
    "Disempowerment": ["disempowerment", "labor displacement", "deskilling", "autonomy"],
    "Misinformation": ["misinformation", "deepfake", "deepfakes", "propaganda"],
    "Robustness": ["robustness", "adversarial", "red teaming"],
    "Extinction Risk": ["existential risk", "extinction risk", "loss of control"],
    "Privacy": ["privacy"],
    "Transparency": ["transparency", "explainable", "interpretability"],
    "Safety": ["safety", "safe ai"],
    "Authenticity": ["authenticity", "impersonation", "deepfake"],
    "Autonomy & Control": ["autonomy", "human oversight"],
    "Trustworthiness": ["trustworthy", "trustworthiness"],
}

os.makedirs("arxiv_taxonomy_papers", exist_ok=True)

all_papers = []
seen_ids = set()

def scrape_with_retry(category, date_from, date_until, keywords, max_retries=5):
    for attempt in range(max_retries):
        try:
            scraper = arxivscraper.Scraper(
                category=category,
                date_from=date_from,
                date_until=date_until,
                filters={'title': keywords, 'abstract': keywords},
                timeout=120
            )
            return scraper.scrape()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"      Failed after {max_retries} attempts: {e}")
                return []
            wait = (2 ** attempt) + random.uniform(1, 3)   # exponential backoff
            print(f"      Connection refused. Retrying in {wait:.1f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    return []



for year in range(START_YEAR, END_YEAR + 1):
    date_from = f"{year}-01-01"
    date_until = f"{year}-12-31"
    print(f"\n=== Year {year} ===")
    
    for cat_name, keywords in TAXONOMY.items():
        if not keywords:
            continue
            
        print(f"   → {cat_name}")
        
        for main_cat in CATEGORIES:
            papers = scrape_with_retry(main_cat, date_from, date_until, keywords)
            
            for paper in papers:
                pid = paper.get('id')
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    paper['taxonomy'] = cat_name
                    paper['year'] = year
                    paper['main_category'] = main_cat
                    all_papers.append(paper)
            
            print(f"      Found {len(papers)} papers (Total so far: {len(all_papers)})")
            time.sleep(4 + random.uniform(0, 2))   # previously it gave error due to continous requests so i did this

# ====================== SAVE ======================
df = pd.DataFrame(all_papers)
cols = ['taxonomy', 'year', 'main_category', 'id', 'title', 'authors', 
        'abstract', 'categories', 'created', 'url']

df = df[[c for c in cols if c in df.columns]]

df.to_csv('arxiv_taxonomy_papers_2.csv', index=False, encoding='utf-8')
print(f"\n Finished Total papers collected: {len(df)}")
print("Saved as: arxiv_taxonomy_papers_all.csv")
