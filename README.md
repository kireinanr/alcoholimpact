
# Student Wellbeing Project

This project analyzes and predicts student wellbeing using scraped social media data and machine learning models.

## Project Structure

- `StudentWellbeingProject/` — Main project folder
  - `analyze_fake_scraped_data.py` — Analyze the scraped data
  - `cleanup_and_check.py` — Data cleaning and validation
  - `live_scraper.py` — Live data scraping (Twitter, etc.)
  - `log_updater.py` — Logging utility
  - `real_twitter_scraper_disabled.py` — (Disabled) real Twitter scraper
  - `recommendation_app.py` — Recommendation web app
  - `training_pipeline.py` — Model training pipeline
  - `data/` — Data storage
  - `MODEL/` — Trained models and scalers
  - `templates/WEB.HTML` — Web app template
  - `chromedriver.exe` — ChromeDriver for scraping
- `scraped_data.jsonl` — Scraped data (JSON Lines format)

## How to Use

1. **Install requirements**: Make sure you have Python 3.8+ and install required packages (see code for details).
2. **Run data analysis**: Use `analyze_fake_scraped_data.py` to analyze the data.
3. **Train models**: Use `training_pipeline.py` to train or retrain models.
4. **Run the app**: Use `recommendation_app.py` to start the web app.

## Notes
- Some scripts require ChromeDriver (`chromedriver.exe`) for web scraping.
- The real Twitter scraper is disabled for safety and compliance.
- Data files are in JSONL format (one JSON object per line).

## License
See `LICENSE.chromedriver` and `THIRD_PARTY_NOTICES.chromedriver` for ChromeDriver licensing.

