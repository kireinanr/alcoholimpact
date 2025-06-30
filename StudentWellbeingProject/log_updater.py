# -*- coding: utf-8 -*-
"""
log_updater.py - Standalone Performance Log Writer

This script's ONLY purpose is to write the `model_performance.json` file.
It is called by the user after the main training pipeline is complete,
ensuring it runs in a clean process to avoid OS-level errors.

To run: `python log_updater.py --version <num> --mse <num>`
"""
import json
import argparse
from pathlib import Path

def update_performance_log(version, mse):
    """Reads the latest model files and writes the performance JSON log."""
    print("\n--- Updating Performance Log ---")
    perf_file_name = 'model_performance.json'
    
    # **FIX:** Define the model directory in the user's home folder.
    model_dir = Path.home() / "StudentWellbeingProjectModels"
    
    # Define paths to the artifacts that were just saved by the main script
    model_filename = model_dir / f'student_wellbeing_rf_v{version}.joblib'
    scaler_filename = model_dir / f'data_scaler_v{version}.joblib'
    columns_filename = model_dir / f'model_columns_v{version}.joblib'
    importance_filename = model_dir / f'feature_importance_v{version}.json'

    # Prepare the data for the JSON file
    performance_data_to_save = {
        'version': version,
        'mse': mse,
        'model_file': str(model_filename.resolve()),
        'scaler_file': str(scaler_filename.resolve()),
        'columns_file': str(columns_filename.resolve()),
        'importance_file': str(importance_filename.resolve())
    }
    
    # **FIX:** Define the output path for the log file to be inside the same robust directory.
    perf_file_path = model_dir / perf_file_name
    print(f"Writing final performance log to {perf_file_path}")
    
    # Write the file
    try:
        with open(perf_file_path, 'w') as f:
            json.dump(performance_data_to_save, f, indent=4)
        print("Performance log updated successfully.")
    except Exception as e:
        print(f"AN ERROR OCCURRED WRITING THE LOG FILE: {e}")


if __name__ == '__main__':
    # Set up to read arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True, help='Version number for the log update.')
    parser.add_argument('--mse', type=float, required=True, help='MSE for the log update.')
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    update_performance_log(args.version, args.mse)
