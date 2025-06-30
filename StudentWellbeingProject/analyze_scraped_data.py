import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Path to the fake data file (update if needed)
input_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'scraped_data.jsonl')
output_file = 'scraped_data.jsonl'  # For compatibility with training_pipeline.py

sentiments = []
veracities = []
analyzer = SentimentIntensityAnalyzer()
processed_data = []

# Read and process all lines first
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        try:
            data = json.loads(line)
            if data.get('lang') == 'en':
                data['sentiment'] = analyzer.polarity_scores(data['text'])['compound']
            else:
                data['sentiment'] = None
            if data.get('sentiment') is not None:
                sentiments.append(data['sentiment'])
            if data.get('veracity') is not None:
                veracities.append(data['veracity'])
            processed_data.append(data)
        except Exception as e:
            print(f"[ERROR processing line] {e}")
            continue

# Write all processed data to output file, with error handling
try:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, data in enumerate(processed_data):
            try:
                print(f"Writing line {i+1}/{len(processed_data)}")
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            except OSError as e:
                print(f"[OSError writing line {i+1}] {e}")
                continue
except Exception as e:
    print(f"[OSError opening output file] {e}")

sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0.0
veracity_score = sum(veracities) / len(veracities) if veracities else 1.0

print(f"sentiment_score = {sentiment_score:.4f}")
print(f"veracity_score = {veracity_score:.4f}")
print(f"scraped_data.jsonl is ready for the training pipeline.")
