# -*- coding: utf-8 -*-
"""
live_scraper.py - Continuous Data Collector
"""

import os
os.environ['SSL_CERT_FILE'] = r'C:\Users\Trise\anaconda3\envs\wellbeing\Lib\site-packages\certifi\cacert.pem'

import json
import time
import itertools
from datetime import datetime
import asyncio
from twikit import Client
from pathlib import Path

# --- CONFIGURATION ---
# Keywords for Twitter scraping
KEYWORDS = ['student stress', 'university burnout', 'exam anxiety', 'college life', 'tugas kuliah', 'stress akademik', 'burnout', '壓力', 'ストレス']

# Language codes for different regions
LANGUAGES = ['en', 'id', 'ms', 'ja', 'ko', 'zh'] # English, Indonesian, Malay, Japanese, Korean, Chinese

OUTPUT_FILE = 'scraped_data.jsonl'
SCRAPE_LIMIT_PER_CYCLE = 5
SLEEP_TIME_SECONDS = 10

USERNAME = 'ArchiveTrs60044'
EMAIL = 'trsarchive1@gmail.com'
PASSWORD = '2024:Twt'

# --- SCRAPING LOGIC ---

def ensure_output_file_exists():
    """Ensures the output file exists."""
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass

def is_valid_json_file(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except Exception:
        return False

def scrape_twitter_twikit(query, limit=5):
    """Scrapes Twitter for a given query using twikit (async). Only logs in if cookies.json is missing or invalid."""
    import os
    async def _scrape():
        client = Client('en-US')
        script_dir = Path(__file__).parent.resolve()
        cookies_file = str(script_dir / 'cookies.json')

        if not is_valid_json_file(cookies_file):
            if os.path.exists(cookies_file):
                # Ensure file is closed before deleting
                try:
                    os.remove(cookies_file)
                except Exception as e:
                    print(f"Warning: Could not delete cookies file: {e}")
            await client.login(
                auth_info_1=USERNAME,
                auth_info_2=EMAIL,
                password=PASSWORD,
                cookies_file=cookies_file
            )
        else:
            await client.load_cookies(cookies_file)
            await client.refresh_token()
        tweets = await client.search_tweet(query, 'Latest')
        results = []
        for i, tweet in enumerate(tweets):
            if i >= limit:
                break
            results.append({
                'source': 'twitter_twikit',
                'id': tweet.id,
                'timestamp': tweet.created_at,
                'text': tweet.text,
                'user': tweet.user.name,
                'lang': getattr(tweet, 'lang', None),
                'veracity': 1
            })
        return results
    return asyncio.run(_scrape())

if __name__ == '__main__':
    ensure_output_file_exists()
    query_cycle = itertools.cycle(zip(KEYWORDS, LANGUAGES))
    print("--- Starting Continuous Twitter Scraper (twikit, no Selenium) ---")
    while True:
        try:
            keyword, lang = next(query_cycle)
            full_query = f"{keyword} lang:{lang}"
            all_results = scrape_twitter_twikit(full_query, SCRAPE_LIMIT_PER_CYCLE)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for item in all_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Successfully scraped and saved {len(all_results)} tweets for query: {full_query}")
            print(f"\nCycle complete. Sleeping for {SLEEP_TIME_SECONDS} seconds...")
            time.sleep(SLEEP_TIME_SECONDS)
        except KeyboardInterrupt:
            print("\nScraper stopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Restarting cycle after sleep.")
            time.sleep(SLEEP_TIME_SECONDS)