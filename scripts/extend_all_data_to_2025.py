#!/usr/bin/env python3
"""
Extend All Oil & Gas Demo Data to November 8th, 2025

Updates all parquet files to include realistic data through Nov 8, 2025.
Handles all existing data files based on their actual structure.
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

def extend_production_timeseries(df, target_date):
    """Extend production_timeseries.parquet"""
    logger.info("Extending production timeseries...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Production data already extends to {current_max}")
        return df
    
    facilities = df['facility_id'].unique()
    new_rows = []
    
    current_date = current_max + timedelta(days=1)
    
    while current_date <= target_date:
        for facility_id in facilities:
            facility_data = df[df['facility_id'] == facility_id].tail(30)  # Last 30 days for trend
            
            if len(facility_data) == 0:
                continue
            
            # Get facility info
            facility_info = facility_data.iloc[-1]
            
            # Calculate trends and apply realistic factors
            avg_oil = facility_data['oil_production_bpd'].mean()
            avg_gas = facility_data['gas_production_mcfd'].mean()
            
            # Natural decline + seasonal + random variation
            annual_decline = 0.03  # 3% annually
            daily_decline = annual_decline / 365
            
            # Seasonal factor
            month = current_date.month
            seasonal_factor = 1.0
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 0.95
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.05
            
            # Random variation
            daily_variation = np.random.normal(1.0, 0.05)
            
            # Calculate new values
            new_oil = max(0, avg_oil * (1 - daily_decline) * seasonal_factor * daily_variation)
            new_gas = max(0, avg_gas * (1 - daily_decline) * seasonal_factor * daily_variation)
            
            new_row = {
                'date': current_date,
                'facility_id': facility_id,
                'facility_type': facility_info['facility_type'],
                'region': facility_info['region'],
                'country': facility_info['country'],
                'oil_production_bpd': round(new_oil, 2),
                'gas_production_mcfd': round(new_gas, 2),
                'water_cut_percent': min(95, facility_info['water_cut_percent'] + np.random.normal(0, 0.5)),
                'gor': facility_info['gor'] * (1 + np.random.normal(0, 0.02)),
                'wellhead_pressure_psi': facility_info['wellhead_pressure_psi'] * (1 + np.random.normal(0, 0.01)),
                'flowing_tubing_pressure_psi': facility_info['flowing_tubing_pressure_psi'] * (1 + np.random.normal(0, 0.01))
            }
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        return extended_df.sort_values(['date', 'facility_id'])
    
    return df

def extend_prices_timeseries(df, target_date):
    """Extend prices_timeseries.parquet"""
    logger.info("Extending prices timeseries...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Price data already extends to {current_max}")
        return df
    
    # Get last known prices
    last_row = df[df['date'] == current_max].iloc[0]
    
    new_rows = []
    current_date = current_max + timedelta(days=1)
    
    # Track current prices
    current_prices = {
        'crude_oil_price_usd_bbl': last_row['crude_oil_price_usd_bbl'],
        'natural_gas_price_usd_mcf': last_row['natural_gas_price_usd_mcf'],
        'gasoline_price_usd_gal': last_row['gasoline_price_usd_gal'],
        'diesel_price_usd_gal': last_row['diesel_price_usd_gal'],
        'jet_fuel_price_usd_gal': last_row['jet_fuel_price_usd_gal'],
        'brent_crude_usd_bbl': last_row['brent_crude_usd_bbl'],
        'wti_crude_usd_bbl': last_row['wti_crude_usd_bbl']
    }
    
    while current_date <= target_date:
        # Oil prices (mean reversion to $75-80)
        oil_target = 77.5
        oil_change = 0.05 * (oil_target - current_prices['crude_oil_price_usd_bbl']) / oil_target + np.random.normal(0, 0.025)
        current_prices['crude_oil_price_usd_bbl'] = max(10, current_prices['crude_oil_price_usd_bbl'] * (1 + oil_change))
        current_prices['wti_crude_usd_bbl'] = current_prices['crude_oil_price_usd_bbl'] * np.random.uniform(0.98, 1.02)
        current_prices['brent_crude_usd_bbl'] = current_prices['crude_oil_price_usd_bbl'] * np.random.uniform(1.02, 1.06)
        
        # Natural gas (seasonal patterns)
        gas_target = 4.5 if current_date.month in [12, 1, 2] else 3.2
        gas_change = 0.08 * (gas_target - current_prices['natural_gas_price_usd_mcf']) / gas_target + np.random.normal(0, 0.04)
        current_prices['natural_gas_price_usd_mcf'] = max(0.5, current_prices['natural_gas_price_usd_mcf'] * (1 + gas_change))
        
        # Refined products (follow crude with margins)
        oil_price = current_prices['crude_oil_price_usd_bbl']
        current_prices['gasoline_price_usd_gal'] = (oil_price / 42) * 1.3 + np.random.normal(0, 0.05)
        current_prices['diesel_price_usd_gal'] = (oil_price / 42) * 1.25 + np.random.normal(0, 0.03)
        current_prices['jet_fuel_price_usd_gal'] = (oil_price / 42) * 1.35 + np.random.normal(0, 0.04)
        
        new_row = {
            'date': current_date,
            **{k: round(v, 2) for k, v in current_prices.items()}
        }
        new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        return extended_df.sort_values('date')
    
    return df

def extend_demand_timeseries(df, target_date):
    """Extend demand_timeseries.parquet"""
    logger.info("Extending demand timeseries...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Demand data already extends to {current_max}")
        return df
    
    regions = df['region'].unique()
    new_rows = []
    
    current_date = current_max + timedelta(days=1)
    
    # Track current demand by region
    last_data = df[df['date'] == current_max].set_index('region')
    current_demand = {}
    for region in regions:
        if region in last_data.index:
            current_demand[region] = last_data.loc[region].to_dict()
    
    while current_date <= target_date:
        for region in regions:
            if region not in current_demand:
                continue
            
            last_values = current_demand[region]
            
            # Growth rates by region
            growth_factors = {
                'North America': 0.01,
                'Europe': 0.005,
                'Asia Pacific': 0.025,
                'Middle East': 0.02,
                'Africa': 0.03,
                'South America': 0.015
            }
            
            annual_growth = growth_factors.get(region, 0.015)
            daily_growth = annual_growth / 365
            
            # Seasonal factors
            month = current_date.month
            seasonal_oil = 1.05 if month in [6, 7, 8] else 0.98 if month in [12, 1, 2] else 1.0
            seasonal_gas = 1.2 if month in [12, 1, 2] else 0.9 if month in [6, 7, 8] else 1.0
            seasonal_gasoline = 1.1 if month in [6, 7, 8] else 0.95
            
            new_row = {
                'date': current_date,
                'region': region,
                'oil_demand_thousand_bpd': max(0, last_values['oil_demand_thousand_bpd'] * (1 + daily_growth) * seasonal_oil * np.random.normal(1, 0.02)),
                'gas_demand_thousand_mcfd': max(0, last_values['gas_demand_thousand_mcfd'] * (1 + daily_growth) * seasonal_gas * np.random.normal(1, 0.03)),
                'gasoline_demand_thousand_bpd': max(0, last_values['gasoline_demand_thousand_bpd'] * (1 + daily_growth) * seasonal_gasoline * np.random.normal(1, 0.02)),
                'diesel_demand_thousand_bpd': max(0, last_values['diesel_demand_thousand_bpd'] * (1 + daily_growth) * np.random.normal(1, 0.02)),
                'jet_fuel_demand_thousand_bpd': max(0, last_values['jet_fuel_demand_thousand_bpd'] * (1 + daily_growth) * np.random.normal(1, 0.03))
            }
            
            # Update tracking
            for key in ['oil_demand_thousand_bpd', 'gas_demand_thousand_mcfd', 'gasoline_demand_thousand_bpd', 'diesel_demand_thousand_bpd', 'jet_fuel_demand_thousand_bpd']:
                current_demand[region][key] = new_row[key]
            
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        return extended_df.sort_values(['date', 'region'])
    
    return df

def extend_maintenance_timeseries(df, target_date):
    """Extend maintenance_timeseries.parquet"""
    logger.info("Extending maintenance timeseries...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Maintenance data already extends to {current_max}")
        return df
    
    facilities = df['facility_id'].unique()
    facility_types = df[['facility_id', 'facility_type', 'region']].drop_duplicates().set_index('facility_id')
    
    new_rows = []
    current_date = current_max + timedelta(days=1)
    
    while current_date <= target_date:
        for facility_id in facilities:
            # Probability of maintenance (varies by season)
            base_prob = 0.01  # 1% daily base probability
            seasonal_prob = 0.03 if current_date.month in [4, 5, 9, 10] else 0.005  # Higher in spring/fall
            
            if np.random.random() < (base_prob + seasonal_prob):
                
                # Maintenance types and characteristics
                maintenance_types = [
                    ('Preventive', 8, 5000, 'high'),
                    ('Corrective', 24, 15000, 'medium'),
                    ('Emergency', 4, 8000, 'critical'),
                    ('Routine', 2, 1000, 'low'),
                    ('Overhaul', 168, 50000, 'medium')  # 1 week
                ]
                
                maint_type, hours, cost_base, priority = maintenance_types[np.random.randint(0, len(maintenance_types))]
                
                # Add randomness
                duration = max(1, hours + np.random.randint(-2, 3))
                cost = max(500, cost_base * np.random.uniform(0.7, 1.3))
                
                # Planned start/end (some maintenance is immediate)
                if maint_type == 'Emergency':
                    planned_start = current_date
                else:
                    planned_start = current_date + timedelta(days=np.random.randint(1, 7))
                
                planned_end = planned_start + timedelta(hours=duration)
                
                facility_info = facility_types.loc[facility_id] if facility_id in facility_types.index else {'facility_type': 'Well', 'region': 'Unknown'}
                
                new_row = {
                    'date': current_date,
                    'facility_id': facility_id,
                    'facility_type': facility_info['facility_type'],
                    'region': facility_info['region'],
                    'maintenance_type': maint_type,
                    'duration_hours': duration,
                    'cost_usd': round(cost, 2),
                    'planned_start': planned_start,
                    'planned_end': planned_end,
                    'priority': priority
                }
                new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        return extended_df.sort_values(['date', 'facility_id'])
    
    return df

def extend_weather_timeseries(df, target_date):
    """Extend weather_timeseries.parquet"""
    logger.info("Extending weather timeseries...")
    
    df['date'] = pd.to_datetime(df['date'])
    current_max = df['date'].max()
    
    if current_max >= target_date:
        logger.info(f"Weather data already extends to {current_max}")
        return df
    
    regions = df['region'].unique()
    new_rows = []
    
    current_date = current_max + timedelta(days=1)
    
    while current_date <= target_date:
        for region in regions:
            
            # Seasonal weather patterns by region
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday
            
            # Temperature patterns (simplified)
            if 'North' in region or 'Europe' in region:
                base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            elif 'Middle East' in region or 'Africa' in region:
                base_temp = 30 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            else:
                base_temp = 22 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            temperature = base_temp + np.random.normal(0, 3)
            
            # Humidity (inverse correlation with temperature)
            humidity = max(20, min(95, 70 - (temperature - 20) * 0.5 + np.random.normal(0, 10)))
            
            # Wind speed
            wind_speed = max(0, np.random.gamma(2, 5))
            
            # Precipitation (seasonal)
            precip_prob = 0.3 if month in [11, 12, 1, 2, 3] else 0.15
            precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0
            
            # Weather conditions
            if precipitation > 10:
                condition = 'Rainy'
            elif precipitation > 0:
                condition = 'Cloudy'
            elif wind_speed > 25:
                condition = 'Windy'
            else:
                condition = 'Clear'
            
            # Operation impact score (0-10, higher is worse)
            impact = 0
            if precipitation > 20:
                impact += 3
            if wind_speed > 30:
                impact += 2
            if temperature < 0 or temperature > 40:
                impact += 2
            if humidity > 90:
                impact += 1
            
            impact_score = min(10, impact + np.random.normal(0, 0.5))
            
            new_row = {
                'date': current_date,
                'region': region,
                'temperature_celsius': round(temperature, 1),
                'humidity_percent': round(humidity, 1),
                'wind_speed_kmh': round(wind_speed, 1),
                'precipitation_mm': round(precipitation, 1),
                'weather_condition': condition,
                'operation_impact_score': round(max(0, impact_score), 1)
            }
            new_rows.append(new_row)
        
        current_date += timedelta(days=1)
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        extended_df = pd.concat([df, new_df], ignore_index=True)
        return extended_df.sort_values(['date', 'region'])
    
    return df

def main():
    """Main function to extend all data files to November 8th, 2025"""
    try:
        logger.info("=== Extending All Oil & Gas Data to November 8th, 2025 ===")
        
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_dir = paths['base_data_path']
        target_date = datetime(2025, 11, 8)
        
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        
        # Time series files to extend (exclude static geospatial files)
        files_to_extend = [
            ('production_timeseries.parquet', extend_production_timeseries),
            ('prices_timeseries.parquet', extend_prices_timeseries),
            ('demand_timeseries.parquet', extend_demand_timeseries),
            ('maintenance_timeseries.parquet', extend_maintenance_timeseries),
            ('weather_timeseries.parquet', extend_weather_timeseries)
        ]
        
        # Summary stats
        total_added = 0
        
        for filename, extend_func in files_to_extend:
            file_path = data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"\n--- Processing {filename} ---")
            
            # Load original data
            df = pd.read_parquet(file_path)
            original_rows = len(df)
            logger.info(f"Original data: {original_rows:,} rows")
            
            # Check current date range
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                current_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                logger.info(f"Current range: {current_range}")
            
            # Extend data
            extended_df = extend_func(df, target_date)
            new_rows = len(extended_df)
            added_rows = new_rows - original_rows
            
            if added_rows > 0:
                # Create backup
                backup_path = file_path.with_suffix('.backup.parquet')
                df.to_parquet(backup_path)
                logger.info(f"Created backup: {backup_path.name}")
                
                # Save extended data
                extended_df.to_parquet(file_path)
                logger.info(f"✓ Extended and saved: {new_rows:,} rows (+{added_rows:,})")
                
                # Verify new date range
                if 'date' in extended_df.columns:
                    extended_df['date'] = pd.to_datetime(extended_df['date'])
                    new_range = f"{extended_df['date'].min().strftime('%Y-%m-%d')} to {extended_df['date'].max().strftime('%Y-%m-%d')}"
                    logger.info(f"New range: {new_range}")
                
                total_added += added_rows
            else:
                logger.info("✓ No extension needed - data already current")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 DATA EXTENSION COMPLETE!")
        logger.info(f"✓ All time series files now extend through {target_date.strftime('%B %d, %Y')}")
        logger.info(f"✓ Total new records added: {total_added:,}")
        logger.info("✓ Realistic trends and seasonal patterns applied")
        logger.info("✓ Backup files created for safety")
        logger.info("✓ Ready for dashboard and API use!")
        
    except Exception as e:
        logger.error(f"Error extending data: {e}")
        raise

if __name__ == "__main__":
    main()