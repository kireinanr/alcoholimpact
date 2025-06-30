# -*- coding: utf-8 -*-
"""
training_pipeline.py - Revamped On-Demand Model Trainer
"""

# SECTION 1: SETUP AND IMPORTS
# ==============================================================================
import os
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import kagglehub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# SECTION 2: DATA LOADING AND ENRICHMENT
# ==============================================================================
def load_and_enrich_data():
    """
    Downloads, loads, and merges multiple Kaggle datasets to create an enriched
    feature set for training.
    """
    print("--- Starting Multi-Dataset Enrichment Process ---")
    datasets_to_load = { 'student_alcohol': 'uciml/student-alcohol-consumption', 'drug_use': 'tunguz/drug-use-by-age', 'life_expectancy': 'kumarajarshi/life-expectancy-who' }
    downloaded_paths = {}
    for name, handle in datasets_to_load.items():
        try:
            downloaded_paths[name] = kagglehub.dataset_download(handle)
        except Exception as e:
            print(f" -> FAILED to download {name}. Skipping.")
            downloaded_paths[name] = None
            
    if not downloaded_paths.get('student_alcohol'): return None
    primary_df_path = os.path.join(downloaded_paths['student_alcohol'], "student-mat.csv")
    
    try:
        with open(primary_df_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
        df_main = pd.read_csv(primary_df_path, sep=delimiter)
    except Exception as e:
        print(f"Failed to read the primary CSV file. Error: {e}")
        return None

    df_main.columns = [col.strip().lower() for col in df_main.columns]
    print("Cleaned column names:", df_main.columns.tolist())
    
    print(f"\nLoaded primary dataset with {len(df_main)} records.")
    
    if downloaded_paths.get('drug_use'):
        drug_use_path = os.path.join(downloaded_paths['drug_use'], "drug-use-by-age.csv")
        df_drug = pd.read_csv(drug_use_path)
        df_drug.columns = [col.strip().lower() for col in df_drug.columns]
        drug_use_subset = df_drug[df_drug['age'] == '17'] 
        if not drug_use_subset.empty: df_main['context_alcohol_use_rate_17'] = drug_use_subset['alcohol-use'].iloc[0]

    if downloaded_paths.get('life_expectancy'):
        life_exp_path = os.path.join(downloaded_paths['life_expectancy'], "Life Expectancy Data.csv")
        df_life = pd.read_csv(life_exp_path)
        df_life.columns = [col.strip().lower() for col in df_life.columns]
        df_main['context_avg_life_expectancy'] = df_life['life expectancy'].mean()
        
    print("\n--- Feature Enrichment Complete ---")
    return df_main

def load_scraped_data(file_path='scraped_data.jsonl'):
    print(f"Loading scraped data from {file_path}...")
    try:
        df = pd.read_json(file_path, lines=True)
        df.dropna(subset=['text'], inplace=True)
        if df.empty: return 0.0, 1.0
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment'] = df['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])
        # Calculate average veracity if present, else default to 1.0
        avg_veracity = df['veracity'].mean() if 'veracity' in df.columns else 1.0
        return df['sentiment'].mean(), avg_veracity
    except Exception:
        return 0.0, 1.0

# SECTION 3: MODEL TRAINING
# ==============================================================================
def train_and_save_model(df, sentiment_score, veracity_score):
    """
    Trains a RandomForest model and saves all necessary artifacts, including
    the crucial feature importances.
    """
    print("\n--- Starting Model Training Process ---")
    
    perf_file_name = 'model_performance.json'
    try:
        with open(perf_file_name, 'r') as f: old_performance = json.load(f)
    except FileNotFoundError: old_performance = {'version': 0, 'mse': 999}
    new_version = old_performance.get('version', 0) + 1

    df['social_sentiment'] = sentiment_score
    df['social_veracity'] = veracity_score
    df['total_alcohol'] = df['dalc'] + df['walc']
    df_processed = pd.get_dummies(df, drop_first=True)
    
    y = df_processed[['g3', 'health']]
    X = df_processed.drop(['g3', 'health', 'dalc', 'walc'], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training RandomForest model: Version {new_version}...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    model.fit(X_train, y_train)
    
    new_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"\n--- New Model (V{new_version}) Evaluation ---")
    print(f"Combined MSE (loss): {new_mse:.4f}")
    
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance_df.head(5))

    model_dir = Path.home() / "StudentWellbeingProjectModels"
    model_dir.mkdir(exist_ok=True)
    
    model_filename = model_dir / f'student_wellbeing_rf_v{new_version}.joblib'
    scaler_filename = model_dir / f'data_scaler_v{new_version}.joblib'
    columns_filename = model_dir / f'model_columns_v{new_version}.joblib'
    importance_filename = model_dir / f'feature_importance_v{new_version}.json'
    
    print(f"\nSaving artifacts to: {model_dir}")
    try:
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(feature_names.tolist(), columns_filename)
        feature_importance_df.to_json(importance_filename, orient='records')
        print("Model artifacts saved successfully.")
        return new_version, new_mse
    except Exception as e:
        print(f"AN ERROR OCCURRED DURING ARTIFACT SAVING: {e}")
        return None, None
        

# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    enriched_df = load_and_enrich_data()
    
    if enriched_df is not None:
        fresh_sentiment, avg_veracity = load_scraped_data()
        new_version, new_mse = train_and_save_model(enriched_df.copy(), fresh_sentiment, avg_veracity)
        
        if new_version is not None:
            print("\n" + "="*60)
            print("--- Training complete. PLEASE FOLLOW THE NEXT STEP CAREFULLY. ---")
            print("\n1. OPEN A NEW, SEPARATE TERMINAL WINDOW.")
            print("2. Navigate back to your project directory in the new terminal.")
            print("3. Run the following command EXACTLY as shown to finalize the log:")
            print("\n" + "="*20 + " RUN THIS COMMAND NEXT " + "="*20)
            print(f"python log_updater.py --version {new_version} --mse {new_mse}")
            print("="*60)
        else:
            print("\nTraining pipeline failed during artifact saving.")
            
    else:
        print("Could not create enriched dataset. Exiting pipeline.")
