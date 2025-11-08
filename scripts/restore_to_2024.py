#!/usr/bin/env python3
"""
Restore all data files to end at 2024-12-31 to allow regeneration with smooth transitions
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

def main():
    # Get data paths
    paths = get_data_paths('Oil-and-Gas-Demo')
    data_dir = paths['base_data_path']
    cutoff_date = '2024-12-31'
    
    files = [
        'production_timeseries.parquet',
        'prices_timeseries.parquet', 
        'demand_timeseries.parquet',
        'maintenance_timeseries.parquet',
        'weather_timeseries.parquet'
    ]
    
    for filename in files:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"File not found: {filename}")
            continue
            
        print(f"Processing {filename}...")
        df = pd.read_parquet(file_path)
        original_rows = len(df)
        
        # Convert date column and filter
        df['date'] = pd.to_datetime(df['date'])
        filtered_df = df[df['date'] <= cutoff_date].copy()
        new_rows = len(filtered_df)
        
        # Save truncated data
        filtered_df.to_parquet(file_path)
        print(f"  Truncated from {original_rows} to {new_rows} rows (removed {original_rows - new_rows})")
        print(f"  Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
    
    print(f"\nAll files truncated to {cutoff_date}. Ready for smooth regeneration.")

if __name__ == "__main__":
    main()