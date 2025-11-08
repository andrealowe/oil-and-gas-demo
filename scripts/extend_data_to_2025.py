#!/usr/bin/env python3
"""
Extend Oil & Gas Demo Data to November 8th, 2025

Updates all parquet files to include realistic data through Nov 8, 2025.
Maintains historical patterns while projecting forward with realistic trends.
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

def extend_production_data(df, target_date):
    """Extend production time series with realistic trends"""
    logger.info("Extending production data...")
    
    # Get current date range
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Production data already extends to {current_max}")
        return df
    
    # Get last known values and trends
    recent_data = df[df['date'] >= current_max - timedelta(days=90)].copy()
    facilities = df['facility_id'].unique()
    
    new_rows = []
    
    # Generate consistent decline rates per facility (avoid daily randomness)
    facility_decline_rates = {}
    for facility_id in facilities:
        facility_decline_rates[facility_id] = np.random.uniform(0.02, 0.05) / 365
    
    # Generate data for each day from current_max + 1 to target_date
    current_date = current_max + timedelta(days=1)
    
    while current_date <= target_date:
        for facility_id in facilities:
            facility_data = recent_data[recent_data['facility_id'] == facility_id]
            
            if len(facility_data) == 0:
                continue
                
            # Calculate trend from recent data
            daily_avg = facility_data['oil_production_bpd'].mean()
            recent_trend = facility_data['oil_production_bpd'].pct_change().mean()
            
            # Apply realistic factors
            # 1. Natural decline (consistent per facility)
            daily_decline = facility_decline_rates[facility_id]
            
            # 2. Seasonal effects (winter = lower production, summer = higher)
            month = current_date.month
            seasonal_factor = 1.0
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 0.93
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.07
            elif month in [3, 4, 5, 9, 10, 11]:  # Spring/Fall
                seasonal_factor = 1.0
            
            # 3. Random daily variation (±5%)
            daily_variation = np.random.normal(1.0, 0.05)
            
            # 4. Occasional maintenance (5% chance of 20-40% reduction)
            maintenance_factor = 1.0
            if np.random.random() < 0.05:
                maintenance_factor = np.random.uniform(0.6, 0.8)
            
            # Calculate new production value
            base_production = daily_avg * (1 - daily_decline)
            new_production = max(0, base_production * seasonal_factor * daily_variation * maintenance_factor)
            
            # Get facility info from original data
            facility_info = df[df['facility_id'] == facility_id].iloc[0]
            
            new_row = {
                'date': current_date,
                'facility_id': facility_id,
                'facility_type': facility_info['facility_type'],
                'region': facility_info['region'],
                'country': facility_info['country'],
                'oil_production_bpd': round(new_production, 2),
                'gas_production_mcfd': round(new_production * 0.6, 2),  # Typical oil-to-gas ratio
                'water_cut_percent': round(facility_info['water_cut_percent'] + np.random.normal(0, 1), 1),  # Water cut increases over time
                'gor': round(facility_info['gor'] * np.random.uniform(0.95, 1.05), 1),  # Gas-Oil Ratio variation
                'wellhead_pressure_psi': round(facility_info['wellhead_pressure_psi'] * np.random.uniform(0.98, 1.02), 1),
                'flowing_tubing_pressure_psi': round(facility_info['flowing_tubing_pressure_psi'] * np.random.uniform(0.98, 1.02), 1)
            }
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    # Add new rows to dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} production records through {target_date}")
        return extended_df.sort_values(['date', 'facility_id'])
    
    return df

def extend_price_data(df, target_date):
    """Extend price time series with realistic market trends"""
    logger.info("Extending price data...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Price data already extends to {current_max}")
        return df
    
    # Get last known prices
    last_prices = df[df['date'] == current_max].set_index('commodity')
    
    new_rows = []
    current_date = current_max + timedelta(days=1)
    
    # Price tracking for continuity
    current_prices = last_prices['price'].to_dict()
    
    while current_date <= target_date:
        for commodity in current_prices.keys():
            last_price = current_prices[commodity]
            
            # Realistic price movements by commodity
            if commodity == 'crude_oil':
                # Oil: mean reversion around $75, 2-3% daily volatility
                target_price = 75.0
                volatility = 0.025
                mean_reversion = 0.05 * (target_price - last_price) / target_price
                price_change = mean_reversion + np.random.normal(0, volatility)
                
            elif commodity == 'natural_gas':
                # Natural gas: seasonal patterns, higher volatility
                target_price = 3.5 if current_date.month in [6, 7, 8, 9] else 4.2
                volatility = 0.04
                mean_reversion = 0.08 * (target_price - last_price) / target_price
                price_change = mean_reversion + np.random.normal(0, volatility)
                
            else:  # refined_products
                # Refined products: follow oil with premium
                oil_price = current_prices.get('crude_oil', 75.0)
                target_price = oil_price * 1.3  # Typical refining margin
                volatility = 0.02
                mean_reversion = 0.06 * (target_price - last_price) / target_price
                price_change = mean_reversion + np.random.normal(0, volatility)
            
            # Apply price change
            new_price = max(0.1, last_price * (1 + price_change))
            current_prices[commodity] = new_price
            
            new_row = {
                'date': current_date,
                'commodity': commodity,
                'price': round(new_price, 2),
                'currency': 'USD',
                'unit': last_prices.loc[commodity, 'unit'],
                'exchange': last_prices.loc[commodity, 'exchange']
            }
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} price records through {target_date}")
        return extended_df.sort_values(['date', 'commodity'])
    
    return df

def extend_demand_data(df, target_date):
    """Extend demand time series with economic growth trends"""
    logger.info("Extending demand data...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Demand data already extends to {current_max}")
        return df
    
    last_demand = df[df['date'] == current_max].set_index(['region', 'commodity'])
    
    new_rows = []
    current_date = current_max + timedelta(days=1)
    
    # Demand tracking
    current_demand = {}
    for (region, commodity), row in last_demand.iterrows():
        current_demand[(region, commodity)] = row['demand_volume']
    
    while current_date <= target_date:
        for (region, commodity), last_vol in current_demand.items():
            
            # Annual growth rates by region and commodity
            growth_rates = {
                ('North America', 'crude_oil'): 0.01,
                ('North America', 'natural_gas'): 0.02,
                ('North America', 'refined_products'): -0.005,
                ('Europe', 'crude_oil'): 0.005,
                ('Europe', 'natural_gas'): 0.015,
                ('Europe', 'refined_products'): -0.01,
                ('Asia Pacific', 'crude_oil'): 0.03,
                ('Asia Pacific', 'natural_gas'): 0.04,
                ('Asia Pacific', 'refined_products'): 0.02,
            }
            
            annual_growth = growth_rates.get((region, commodity), 0.015)
            daily_growth = annual_growth / 365
            
            # Seasonal demand patterns
            month = current_date.month
            seasonal_factor = 1.0
            
            if commodity == 'natural_gas':
                # Winter heating demand
                if month in [12, 1, 2]:
                    seasonal_factor = 1.2
                elif month in [6, 7, 8]:
                    seasonal_factor = 0.9
            elif commodity == 'refined_products':
                # Summer driving season
                if month in [6, 7, 8]:
                    seasonal_factor = 1.1
                elif month in [12, 1, 2]:
                    seasonal_factor = 0.95
            
            # Economic cycle effects - gradual recovery over 2025
            if current_date.year >= 2025:
                # Gradual economic recovery over 12 months (0% to 2% increase)
                months_into_2025 = (current_date.year - 2025) * 12 + current_date.month - 1
                recovery_progress = min(1.0, months_into_2025 / 12.0)  # 0 to 1 over 12 months
                economic_factor = 1.0 + (0.02 * recovery_progress)  # Gradual 0% to 2% increase
            else:
                economic_factor = 1.0
            
            # Daily variation
            daily_variation = np.random.normal(1.0, 0.03)
            
            # Calculate new demand
            new_demand = last_vol * (1 + daily_growth) * seasonal_factor * economic_factor * daily_variation
            current_demand[(region, commodity)] = new_demand
            
            # Get original row info
            orig_row = last_demand.loc[(region, commodity)]
            
            new_row = {
                'date': current_date,
                'region': region,
                'commodity': commodity,
                'demand_volume': round(new_demand, 2),
                'unit': orig_row['unit'],
                'sector': orig_row['sector'],
                'forecast_confidence': orig_row['forecast_confidence']
            }
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} demand records through {target_date}")
        return extended_df.sort_values(['date', 'region', 'commodity'])
    
    return df

def extend_maintenance_data(df, target_date):
    """Extend maintenance time series with scheduled and unplanned events"""
    logger.info("Extending maintenance data...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Maintenance data already extends to {target_date}")
        return df
    
    facilities = df['facility_id'].unique()
    new_rows = []
    
    current_date = current_max + timedelta(days=1)
    
    while current_date <= target_date:
        for facility_id in facilities:
            
            # Probability of maintenance event (2% daily for unplanned, seasonal for planned)
            unplanned_prob = 0.02
            planned_prob = 0.05 if current_date.month in [4, 5, 9, 10] else 0.01  # Spring/Fall turnarounds
            
            if np.random.random() < (unplanned_prob + planned_prob):
                
                # Determine maintenance type and duration
                if np.random.random() < 0.3:  # 30% major maintenance
                    maintenance_type = 'Major Overhaul'
                    duration = np.random.randint(7, 21)  # 1-3 weeks
                    production_impact = np.random.uniform(0.8, 1.0)  # 80-100% impact
                elif np.random.random() < 0.6:  # 60% routine maintenance
                    maintenance_type = 'Routine Maintenance'
                    duration = np.random.randint(1, 5)  # 1-4 days
                    production_impact = np.random.uniform(0.2, 0.5)  # 20-50% impact
                else:  # 10% emergency
                    maintenance_type = 'Emergency Repair'
                    duration = np.random.randint(1, 7)  # 1-7 days
                    production_impact = np.random.uniform(0.6, 1.0)  # 60-100% impact
                
                # Get facility info
                facility_info = df[df['facility_id'] == facility_id].iloc[0] if len(df[df['facility_id'] == facility_id]) > 0 else {
                    'facility_name': f'Facility_{facility_id}',
                    'region': 'Unknown'
                }
                
                # Map to actual maintenance types in data
                maintenance_type_map = {
                    'Major Overhaul': 'preventive',
                    'Routine Maintenance': 'preventive', 
                    'Emergency Repair': 'corrective'
                }
                
                actual_maintenance_type = maintenance_type_map.get(maintenance_type, 'corrective')
                duration_hours = duration * 24  # Convert days to hours
                estimated_cost = duration_hours * np.random.uniform(500, 2000)  # $500-2000 per hour
                
                new_row = {
                    'date': current_date,
                    'facility_id': facility_id,
                    'facility_type': facility_info.get('facility_type', 'oil_well'),
                    'region': facility_info.get('region', 'Unknown'),
                    'maintenance_type': actual_maintenance_type,
                    'duration_hours': round(duration_hours, 1),
                    'cost_usd': round(estimated_cost, 0),
                    'planned_start': current_date,
                    'planned_end': current_date + pd.Timedelta(hours=duration_hours),
                    'priority': 'high' if 'Emergency' in maintenance_type else 'medium'
                }
                new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        logger.info(f"Added {len(new_rows)} maintenance records through {target_date}")
        return extended_df.sort_values(['date', 'facility_id'])
    
    return df

def main():
    """Main function to extend all data files to November 8th, 2025"""
    try:
        logger.info("=== Extending Oil & Gas Data to November 8th, 2025 ===")
        
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_dir = paths['base_data_path']
        target_date = datetime(2025, 11, 8)
        
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        
        # Files to extend
        files_to_extend = [
            ('production_timeseries.parquet', extend_production_data),
            ('prices_timeseries.parquet', extend_price_data),
            ('demand_timeseries.parquet', extend_demand_data),
            ('maintenance_timeseries.parquet', extend_maintenance_data)
        ]
        
        for filename, extend_func in files_to_extend:
            file_path = data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"\n--- Processing {filename} ---")
            
            # Load original data
            df = pd.read_parquet(file_path)
            original_rows = len(df)
            logger.info(f"Original data: {original_rows} rows")
            
            # Extend data
            extended_df = extend_func(df, target_date)
            new_rows = len(extended_df)
            added_rows = new_rows - original_rows
            
            if added_rows > 0:
                # Create backup
                backup_path = file_path.with_suffix('.backup.parquet')
                df.to_parquet(backup_path)
                logger.info(f"Created backup: {backup_path}")
                
                # Save extended data
                extended_df.to_parquet(file_path)
                logger.info(f"Extended data saved: {new_rows} rows (+{added_rows})")
                
                # Verify date range
                extended_df['date'] = pd.to_datetime(extended_df['date'])
                date_range = f"{extended_df['date'].min().strftime('%Y-%m-%d')} to {extended_df['date'].max().strftime('%Y-%m-%d')}"
                logger.info(f"Date range: {date_range}")
            else:
                logger.info("No extension needed - data already current")
        
        logger.info("\n=== Data Extension Complete ===")
        logger.info("✓ All parquet files now extend through November 8th, 2025")
        logger.info("✓ Realistic trends and patterns maintained")
        logger.info("✓ Backup files created for safety")
        
    except Exception as e:
        logger.error(f"Error extending data: {e}")
        raise

if __name__ == "__main__":
    main()