import os
import requests
import time
from datetime import datetime
from textblob import TextBlob
import json

KEYWORDS = ['student stress', 'university burnout', 'exam anxiety', 'college life', 'tugas kuliah', 'stress akademik', 'burnout', '壓力', 'ストレス']
LANGUAGES = ['en', 'id', 'ms', 'ja', 'ko', 'zh']  # English, Indonesian, Malay, Japanese, Korean, Chinese

# Write output to the user's Desktop
DESKTOP = os.path.join(os.path.expanduser('~'), 'Desktop')
OUTPUT_FILE = os.path.join(DESKTOP, 'fake_scraped_data.jsonl')

def fake_twitter_scraper(query, limit=5, lang='en'):
    """Fetches fake tweet-like data from FakerAPI and formats it like real Twitter data."""
    url = f"https://fakerapi.it/api/v1/texts?_quantity={limit}&_locale={lang}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['data']
        fake_results = []
        for i, item in enumerate(data):
            fake_results.append({
                'source': 'twitter_fake',
                'id': f'fake_{i}_{int(time.time())}',
                'timestamp': datetime.now().isoformat(),
                'text': item['content'],
                'user': item['title'],
                'lang': lang,
                'veracity': 0
            })
        return fake_results
    except Exception as e:
        print(f"[FAKE SCRAPER ERROR] {e}")
        return []

def analyze_sentiment(text, lang='en'):
    try:
        # TextBlob sentiment only works well for English
        if lang != 'en':
            return None
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception:
        return None

if __name__ == '__main__':
    print(f"Writing output to: {OUTPUT_FILE}")
    all_results = []
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for keyword in KEYWORDS:
            for lang in LANGUAGES:
                query = f"{keyword} lang:{lang}"
                results = fake_twitter_scraper(query, limit=3, lang=lang)
                for r in results:
                    # Add sentiment analysis (only for English)
                    r['sentiment'] = analyze_sentiment(r['text'], lang=lang)
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    all_results.append(r)
    # Print sentiment summary
    sentiments = [r['sentiment'] for r in all_results if r['sentiment'] is not None]
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        print(f"Average sentiment (English only): {avg_sentiment:.3f}")
    else:
        print("No English sentiment data available.")
