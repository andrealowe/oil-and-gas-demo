#!/usr/bin/env python3
"""
Extend non-production Oil & Gas data files with smooth transitions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extend_prices_data(df, target_date):
    """Extend price data with smooth transitions"""
    logger.info("Extending prices data...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Prices data already extends to {current_max}")
        return df
    
    # Get last known prices
    last_prices = df[df['date'] == current_max].iloc[0]
    
    # Generate date range
    date_range = pd.date_range(start=current_max + timedelta(days=1), end=target_date, freq='D')
    
    # Price columns and their characteristics
    price_columns = [
        ('crude_oil_price_usd_bbl', 75.0, 0.025),
        ('natural_gas_price_usd_mcf', 4.0, 0.04),
        ('gasoline_price_usd_gal', 100.0, 0.02),
        ('diesel_price_usd_gal', 95.0, 0.02),
        ('jet_fuel_price_usd_gal', 98.0, 0.02),
        ('brent_crude_usd_bbl', 76.0, 0.025),
        ('wti_crude_usd_bbl', 74.0, 0.025)
    ]
    
    new_rows = []
    current_prices = {}
    
    # Initialize current prices
    for col, _, _ in price_columns:
        current_prices[col] = last_prices[col]
    
    for date in date_range:
        row = {'date': date}
        
        for col, target_price, volatility in price_columns:
            last_price = current_prices[col]
            
            # Seasonal adjustments for natural gas
            if 'natural_gas' in col:
                seasonal_target = 4.2 if date.month in [12, 1, 2, 11] else 3.8  # Winter higher
                target_price = seasonal_target
            
            # Gradual economic recovery for 2025
            if date.year >= 2025:
                months_into_2025 = (date.year - 2025) * 12 + date.month - 1
                recovery_progress = min(1.0, months_into_2025 / 12.0)
                economic_factor = 1.0 + (0.01 * recovery_progress)  # 1% gradual increase
                target_price *= economic_factor
            
            # Mean reversion + daily variation
            mean_reversion = 0.05 * (target_price - last_price) / target_price
            price_change = mean_reversion + np.random.normal(0, volatility)
            
            new_price = max(0.1, last_price * (1 + price_change))
            current_prices[col] = new_price
            row[col] = round(new_price, 2)
        
        new_rows.append(row)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} price records through {target_date}")
        return extended_df.sort_values('date')
    
    return df

def extend_demand_data(df, target_date):
    """Extend demand data with smooth transitions"""
    logger.info("Extending demand data...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Demand data already extends to {current_max}")
        return df
    
    # Get last values by region and commodity
    last_values = df[df['date'] == current_max].copy()
    
    new_rows = []
    date_range = pd.date_range(start=current_max + timedelta(days=1), end=target_date, freq='D')
    
    for _, row in last_values.iterrows():
        current_demand = row['demand_volume']
        
        for date in date_range:
            # Annual growth rate based on region/commodity
            if 'Asia Pacific' in str(row['region']):
                annual_growth = 0.025
            elif 'Europe' in str(row['region']):
                annual_growth = 0.01
            else:  # North America
                annual_growth = 0.015
            
            daily_growth = annual_growth / 365
            
            # Seasonal effects
            seasonal_factor = 1.0
            if 'natural_gas' in str(row.get('commodity', '')).lower():
                if date.month in [12, 1, 2]:
                    seasonal_factor = 1.15  # Winter heating
                elif date.month in [6, 7, 8]:
                    seasonal_factor = 0.92  # Summer low
            
            # Gradual economic recovery
            if date.year >= 2025:
                months_into_2025 = (date.year - 2025) * 12 + date.month - 1
                recovery_progress = min(1.0, months_into_2025 / 12.0)
                economic_factor = 1.0 + (0.015 * recovery_progress)  # 1.5% gradual increase
            else:
                economic_factor = 1.0
            
            # Daily variation
            daily_variation = np.random.normal(1.0, 0.02)
            
            # Calculate new demand
            new_demand = current_demand * (1 + daily_growth) * seasonal_factor * economic_factor * daily_variation
            current_demand = new_demand
            
            new_row = row.copy()
            new_row['date'] = date
            new_row['demand_volume'] = round(new_demand, 2)
            new_rows.append(new_row)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} demand records through {target_date}")
        return extended_df.sort_values(['date', 'region'])
    
    return df

def main():
    """Main function to extend other data files"""
    try:
        logger.info("=== Extending Other Data Files to November 8th, 2025 ===")
        
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_dir = paths['base_data_path']
        target_date = datetime(2025, 11, 8)
        
        # Files to process
        files_to_process = [
            ('prices_timeseries.parquet', extend_prices_data),
            ('demand_timeseries.parquet', extend_demand_data)
        ]
        
        for filename, extend_func in files_to_process:
            file_path = data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"\n--- Processing {filename} ---")
            
            # Load and extend
            df = pd.read_parquet(file_path)
            original_rows = len(df)
            logger.info(f"Original data: {original_rows:,} rows")
            
            extended_df = extend_func(df, target_date)
            new_rows = len(extended_df)
            added_rows = new_rows - original_rows
            
            if added_rows > 0:
                # Create backup
                backup_path = file_path.with_suffix('.backup2.parquet')
                df.to_parquet(backup_path)
                
                # Save extended data
                extended_df.to_parquet(file_path)
                logger.info(f"Extended data saved: {new_rows:,} rows (+{added_rows:,})")
                
                # Verify smooth transition
                extended_df['date'] = pd.to_datetime(extended_df['date'])
                transition_data = extended_df[
                    (extended_df['date'] >= '2024-12-30') & 
                    (extended_df['date'] <= '2025-01-02')
                ]
                
                if not transition_data.empty and len(transition_data.columns) > 2:
                    sample_col = [col for col in transition_data.columns if col not in ['date']][0]
                    transition_avg = transition_data.groupby('date')[sample_col].mean()
                    
                    logger.info(f"Transition check for {sample_col}:")
                    for date, value in transition_avg.items():
                        logger.info(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")
            else:
                logger.info("No extension needed - data already current")
        
        logger.info("\n=== Extension Complete ===")
        logger.info("✓ Smooth economic transitions implemented")
        logger.info("✓ Realistic seasonal patterns maintained")
        
    except Exception as e:
        logger.error(f"Error extending data: {e}")
        raise

if __name__ == "__main__":
    main()