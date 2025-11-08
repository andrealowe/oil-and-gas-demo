#!/usr/bin/env python3
"""
Efficient Oil & Gas Demo Data Extension to November 8th, 2025
Optimized for large datasets with progress tracking
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

def extend_production_data_efficient(df, target_date):
    """Efficiently extend production time series with vectorized operations"""
    logger.info("Extending production data efficiently...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Production data already extends to {current_max}")
        return df
    
    # Get unique facilities and their last known values
    last_values = df.groupby('facility_id').last().reset_index()
    facilities = last_values['facility_id'].unique()
    
    logger.info(f"Processing {len(facilities)} facilities...")
    
    # Generate date range
    date_range = pd.date_range(start=current_max + timedelta(days=1), end=target_date, freq='D')
    logger.info(f"Adding {len(date_range)} days of data...")
    
    # Create consistent decline rates per facility
    facility_decline_rates = pd.Series(
        np.random.uniform(0.02, 0.05, len(facilities)) / 365,
        index=facilities
    )
    
    new_data_list = []
    
    # Process in chunks to avoid memory issues
    chunk_size = 100
    for i in range(0, len(facilities), chunk_size):
        chunk_facilities = facilities[i:i+chunk_size]
        logger.info(f"Processing facilities {i+1}-{min(i+chunk_size, len(facilities))} of {len(facilities)}")
        
        for facility_id in chunk_facilities:
            facility_info = last_values[last_values['facility_id'] == facility_id].iloc[0]
            base_production = facility_info['oil_production_bpd']
            daily_decline = facility_decline_rates[facility_id]
            
            # Vectorized calculations for all dates for this facility
            days_from_start = np.arange(len(date_range))
            
            # Apply decline
            production_values = base_production * (1 - daily_decline) ** days_from_start
            
            # Add seasonal effects
            months = pd.Series(date_range).dt.month
            seasonal_factors = np.where(
                months.isin([12, 1, 2]), 0.93,  # Winter
                np.where(months.isin([6, 7, 8]), 1.07,  # Summer  
                1.0)  # Spring/Fall
            )
            
            # Add gradual economic recovery for 2025
            years = pd.Series(date_range).dt.year
            months_into_2025 = np.maximum(0, (years - 2025) * 12 + months - 1)
            recovery_progress = np.minimum(1.0, months_into_2025 / 12.0)
            economic_factors = 1.0 + (0.02 * recovery_progress)
            
            # Add daily variation
            daily_variations = np.random.normal(1.0, 0.05, len(date_range))
            
            # Combine all factors
            final_production = production_values * seasonal_factors * economic_factors * daily_variations
            final_production = np.maximum(0, final_production)  # Ensure non-negative
            
            # Create DataFrame for this facility
            facility_data = pd.DataFrame({
                'date': date_range,
                'facility_id': facility_id,
                'facility_type': facility_info['facility_type'],
                'region': facility_info['region'],
                'country': facility_info['country'],
                'oil_production_bpd': np.round(final_production, 2),
                'gas_production_mcfd': np.round(final_production * 0.6, 2),
                'water_cut_percent': np.round(
                    facility_info['water_cut_percent'] + np.random.normal(0, 1, len(date_range)), 1
                ),
                'gor': np.round(
                    facility_info['gor'] * np.random.uniform(0.95, 1.05, len(date_range)), 1
                ),
                'wellhead_pressure_psi': np.round(
                    facility_info['wellhead_pressure_psi'] * np.random.uniform(0.98, 1.02, len(date_range)), 1
                ),
                'flowing_tubing_pressure_psi': np.round(
                    facility_info['flowing_tubing_pressure_psi'] * np.random.uniform(0.98, 1.02, len(date_range)), 1
                )
            })
            
            new_data_list.append(facility_data)
    
    # Combine all new data
    if new_data_list:
        new_df = pd.concat(new_data_list, ignore_index=True)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_df)} production records through {target_date}")
        return extended_df.sort_values(['date', 'facility_id'])
    
    return df

def main():
    """Main function to extend production data efficiently"""
    try:
        logger.info("=== Efficient Data Extension to November 8th, 2025 ===")
        
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_dir = paths['base_data_path']
        target_date = datetime(2025, 11, 8)
        
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        
        # Process production data only (the largest file)
        filename = 'production_timeseries.parquet'
        file_path = data_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"Processing {filename}...")
        
        # Load original data
        df = pd.read_parquet(file_path)
        original_rows = len(df)
        logger.info(f"Original data: {original_rows:,} rows")
        
        # Extend data efficiently
        extended_df = extend_production_data_efficient(df, target_date)
        new_rows = len(extended_df)
        added_rows = new_rows - original_rows
        
        if added_rows > 0:
            # Create backup
            backup_path = file_path.with_suffix('.backup2.parquet')
            df.to_parquet(backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Save extended data
            extended_df.to_parquet(file_path)
            logger.info(f"Extended data saved: {new_rows:,} rows (+{added_rows:,})")
            
            # Verify date range
            extended_df['date'] = pd.to_datetime(extended_df['date'])
            date_range = f"{extended_df['date'].min().strftime('%Y-%m-%d')} to {extended_df['date'].max().strftime('%Y-%m-%d')}"
            logger.info(f"Date range: {date_range}")
            
            # Sample data around transition to verify smoothness
            transition_data = extended_df[
                (extended_df['date'] >= '2024-12-30') & 
                (extended_df['date'] <= '2025-01-02')
            ].groupby('date')['oil_production_bpd'].mean()
            
            logger.info("Production averages around New Year transition:")
            for date, avg_prod in transition_data.items():
                logger.info(f"  {date.strftime('%Y-%m-%d')}: {avg_prod:.2f} bpd")
                
        else:
            logger.info("No extension needed - data already current")
        
        logger.info("\n=== Production Data Extension Complete ===")
        logger.info("✓ Smooth economic transition implemented")
        logger.info("✓ Consistent facility-level decline rates")
        logger.info("✓ Realistic seasonal and daily variations")
        
    except Exception as e:
        logger.error(f"Error extending data: {e}")
        raise

if __name__ == "__main__":
    main()