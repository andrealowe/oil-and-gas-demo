#!/usr/bin/env python3
"""
Explore time series data structure for oil & gas forecasting models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

def explore_timeseries_data():
    """Explore the structure of all time series datasets"""
    
    # Get data paths
    paths = get_data_paths('Oil-and-Gas-Demo')
    data_dir = Path('/mnt/artifacts/Oil-and-Gas-Demo')  # Data is already in artifacts
    
    datasets = {
        'production': 'production_timeseries.parquet',
        'prices': 'prices_timeseries.parquet', 
        'demand': 'demand_timeseries.parquet',
        'maintenance': 'maintenance_timeseries.parquet'
    }
    
    print("Time Series Data Exploration for Oil & Gas Forecasting")
    print("=" * 60)
    
    for name, filename in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n{name.upper()} DATA ({filename})")
            print("-" * 40)
            
            df = pd.read_parquet(filepath)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show sample data
            print("\nSample data:")
            print(df.head())
            
            # Check for time column
            time_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
            if time_cols:
                print(f"Time column(s): {time_cols}")
                # Check frequency
                if len(df) > 1:
                    time_col = time_cols[0]
                    df_sorted = df.sort_values(time_col)
                    time_diff = pd.to_datetime(df_sorted[time_col]).diff().mode()
                    if len(time_diff) > 0:
                        print(f"Frequency: {time_diff.iloc[0]}")
            
            print(f"Data types:\n{df.dtypes}")
            print("\n" + "="*60)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    explore_timeseries_data()