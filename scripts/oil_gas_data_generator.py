"""
Oil and Gas Synthetic Data Generator

This script generates comprehensive synthetic datasets for oil and gas company dashboards:
1. Geospatial dataset: Wells, refineries, facilities with production and health metrics
2. Time series dataset: Production trends, forecasting, prices, and maintenance schedules

Usage:
    python scripts/oil_gas_data_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
from faker import Faker
import random
from typing import Dict, List, Tuple
import mlflow

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths, ensure_directories
from src.models.workflow_io import WorkflowIO

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
Faker.seed(42)

class OilGasDataGenerator:
    """Generate realistic synthetic data for oil and gas operations"""
    
    def __init__(self, project_name: str = "oil_gas_dashboards"):
        self.project_name = project_name
        self.fake = Faker()
        
        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:8768")
        mlflow.set_experiment(f"oil_gas_data_generation_{project_name}")
        
        # Set up directory paths
        self.paths = get_data_paths(project_name)
        self.directories = ensure_directories(project_name)
        
        # Define global operation regions
        self.regions = {
            'North America': {
                'lat_range': (25.0, 70.0),
                'lon_range': (-170.0, -50.0),
                'countries': ['USA', 'Canada', 'Mexico']
            },
            'Middle East': {
                'lat_range': (12.0, 42.0),
                'lon_range': (34.0, 63.0),
                'countries': ['Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Iraq', 'Iran']
            },
            'Europe': {
                'lat_range': (35.0, 71.0),
                'lon_range': (-10.0, 40.0),
                'countries': ['Norway', 'UK', 'Netherlands', 'Russia', 'Kazakhstan']
            },
            'South America': {
                'lat_range': (-55.0, 15.0),
                'lon_range': (-82.0, -35.0),
                'countries': ['Brazil', 'Venezuela', 'Colombia', 'Argentina']
            },
            'Africa': {
                'lat_range': (-35.0, 37.0),
                'lon_range': (-20.0, 55.0),
                'countries': ['Nigeria', 'Angola', 'Algeria', 'Libya', 'Egypt']
            },
            'Asia Pacific': {
                'lat_range': (-45.0, 55.0),
                'lon_range': (95.0, 180.0),
                'countries': ['China', 'Australia', 'Indonesia', 'Malaysia', 'Brunei']
            }
        }
        
        # Equipment types and their typical specifications
        self.equipment_types = {
            'drilling_rig': {
                'capacity_range': (50, 500),  # wells/year
                'maintenance_interval_days': 90,
                'failure_rate': 0.05
            },
            'pump_jack': {
                'capacity_range': (10, 100),  # barrels/day
                'maintenance_interval_days': 30,
                'failure_rate': 0.08
            },
            'compressor': {
                'capacity_range': (1000, 50000),  # MCF/day
                'maintenance_interval_days': 60,
                'failure_rate': 0.06
            },
            'separator': {
                'capacity_range': (100, 5000),  # barrels/day
                'maintenance_interval_days': 45,
                'failure_rate': 0.04
            },
            'pipeline_segment': {
                'capacity_range': (10000, 500000),  # barrels/day
                'maintenance_interval_days': 180,
                'failure_rate': 0.02
            }
        }
        
    def generate_geospatial_data(self, n_wells: int = 1500, n_refineries: int = 25, 
                                n_facilities: int = 200) -> pd.DataFrame:
        """
        Generate geospatial dataset with wells, refineries, and facilities
        
        Args:
            n_wells: Number of oil wells to generate
            n_refineries: Number of refineries to generate
            n_facilities: Number of other facilities to generate
            
        Returns:
            DataFrame with geospatial facility data
        """
        facilities = []
        facility_id = 1
        
        # Generate oil wells
        for i in range(n_wells):
            region = np.random.choice(list(self.regions.keys()))
            region_data = self.regions[region]
            country = np.random.choice(region_data['countries'])
            
            # Generate coordinates within region
            lat = np.random.uniform(region_data['lat_range'][0], region_data['lat_range'][1])
            lon = np.random.uniform(region_data['lon_range'][0], region_data['lon_range'][1])
            
            # Well production characteristics
            well_age_years = np.random.exponential(8)  # Exponential distribution for well age
            well_depth_ft = np.random.normal(8500, 2500)
            well_depth_ft = max(1000, well_depth_ft)  # Minimum depth
            
            # Production rates based on well characteristics
            base_oil_production = np.random.lognormal(4, 1)  # Log-normal distribution
            base_gas_production = np.random.lognormal(6, 1.2)
            
            # Age factor (older wells produce less)
            age_factor = np.exp(-well_age_years * 0.1)
            oil_production = base_oil_production * age_factor * np.random.uniform(0.7, 1.3)
            gas_production = base_gas_production * age_factor * np.random.uniform(0.7, 1.3)
            
            # Well status
            status_prob = np.random.random()
            if status_prob < 0.75:
                status = 'active'
            elif status_prob < 0.85:
                status = 'maintenance'
            else:
                status = 'inactive'
                oil_production = 0
                gas_production = 0
            
            # Equipment health metrics
            equipment_health = np.random.uniform(0.6, 1.0) if status == 'active' else np.random.uniform(0.2, 0.8)
            vibration_level = np.random.uniform(0.1, 2.5)
            temperature_c = np.random.uniform(15, 85)
            pressure_psi = np.random.uniform(100, 5000)
            
            # Environmental monitoring
            h2s_ppm = np.random.exponential(2) if status == 'active' else 0
            co2_ppm = np.random.normal(400, 50) + (h2s_ppm * 10)
            noise_db = np.random.uniform(45, 85) if status == 'active' else np.random.uniform(30, 45)
            
            facilities.append({
                'facility_id': f'WELL_{facility_id:06d}',
                'facility_type': 'oil_well',
                'facility_name': f'{country}_Well_{facility_id}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'region': region,
                'country': country,
                'status': status,
                'oil_production_bpd': round(oil_production, 2),
                'gas_production_mcfd': round(gas_production, 2),
                'well_depth_ft': round(well_depth_ft, 0),
                'well_age_years': round(well_age_years, 1),
                'equipment_health_score': round(equipment_health, 3),
                'vibration_level_mm_s': round(vibration_level, 2),
                'temperature_celsius': round(temperature_c, 1),
                'pressure_psi': round(pressure_psi, 0),
                'h2s_concentration_ppm': round(h2s_ppm, 3),
                'co2_concentration_ppm': round(co2_ppm, 1),
                'noise_level_db': round(noise_db, 1),
                'last_maintenance_date': self.fake.date_between(start_date='-180d', end_date='today'),
                'next_maintenance_date': self.fake.date_between(start_date='today', end_date='+90d')
            })
            facility_id += 1
        
        # Generate refineries
        for i in range(n_refineries):
            region = np.random.choice(list(self.regions.keys()))
            region_data = self.regions[region]
            country = np.random.choice(region_data['countries'])
            
            lat = np.random.uniform(region_data['lat_range'][0], region_data['lat_range'][1])
            lon = np.random.uniform(region_data['lon_range'][0], region_data['lon_range'][1])
            
            # Refinery characteristics
            capacity_bpd = np.random.uniform(50000, 500000)
            utilization = np.random.uniform(0.6, 0.95)
            current_throughput = capacity_bpd * utilization
            
            # Multiple product streams
            gasoline_yield = np.random.uniform(0.35, 0.45)
            diesel_yield = np.random.uniform(0.25, 0.35)
            jet_fuel_yield = np.random.uniform(0.10, 0.15)
            other_yield = 1 - gasoline_yield - diesel_yield - jet_fuel_yield
            
            facilities.append({
                'facility_id': f'REF_{facility_id:06d}',
                'facility_type': 'refinery',
                'facility_name': f'{country}_Refinery_{i+1}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'region': region,
                'country': country,
                'status': 'active',
                'capacity_bpd': round(capacity_bpd, 0),
                'current_throughput_bpd': round(current_throughput, 0),
                'utilization_rate': round(utilization, 3),
                'gasoline_production_bpd': round(current_throughput * gasoline_yield, 0),
                'diesel_production_bpd': round(current_throughput * diesel_yield, 0),
                'jet_fuel_production_bpd': round(current_throughput * jet_fuel_yield, 0),
                'other_products_bpd': round(current_throughput * other_yield, 0),
                'equipment_health_score': round(np.random.uniform(0.7, 0.98), 3),
                'energy_consumption_mwh': round(current_throughput * np.random.uniform(0.8, 1.2), 0),
                'co2_emissions_tons_day': round(current_throughput * np.random.uniform(0.3, 0.5), 1),
                'water_usage_gallons_day': round(current_throughput * np.random.uniform(1.5, 3.0), 0),
                'last_maintenance_date': self.fake.date_between(start_date='-90d', end_date='today'),
                'next_maintenance_date': self.fake.date_between(start_date='today', end_date='+120d')
            })
            facility_id += 1
        
        # Generate other facilities (storage, processing, terminals)
        facility_types = ['storage_tank', 'processing_plant', 'terminal', 'compressor_station', 'pipeline_station']
        
        for i in range(n_facilities):
            facility_type = np.random.choice(facility_types)
            region = np.random.choice(list(self.regions.keys()))
            region_data = self.regions[region]
            country = np.random.choice(region_data['countries'])
            
            lat = np.random.uniform(region_data['lat_range'][0], region_data['lat_range'][1])
            lon = np.random.uniform(region_data['lon_range'][0], region_data['lon_range'][1])
            
            # Facility-specific characteristics
            if facility_type == 'storage_tank':
                capacity = np.random.uniform(100000, 2000000)  # barrels
                current_level = capacity * np.random.uniform(0.1, 0.9)
                throughput = np.random.uniform(10000, 100000)  # bpd
            elif facility_type == 'processing_plant':
                capacity = np.random.uniform(20000, 200000)  # bpd
                current_level = 0  # N/A
                throughput = capacity * np.random.uniform(0.6, 0.9)
            elif facility_type == 'terminal':
                capacity = np.random.uniform(500000, 5000000)  # barrels
                current_level = capacity * np.random.uniform(0.2, 0.8)
                throughput = np.random.uniform(50000, 300000)  # bpd
            elif facility_type == 'compressor_station':
                capacity = np.random.uniform(50000, 500000)  # MCF/day
                current_level = 0  # N/A
                throughput = capacity * np.random.uniform(0.7, 0.95)
            else:  # pipeline_station
                capacity = np.random.uniform(100000, 1000000)  # bpd
                current_level = 0  # N/A
                throughput = capacity * np.random.uniform(0.8, 0.95)
            
            facilities.append({
                'facility_id': f'{facility_type.upper()[:3]}_{facility_id:06d}',
                'facility_type': facility_type,
                'facility_name': f'{country}_{facility_type.title()}_{i+1}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'region': region,
                'country': country,
                'status': np.random.choice(['active', 'maintenance'], p=[0.9, 0.1]),
                'capacity': round(capacity, 0),
                'current_level': round(current_level, 0),
                'throughput': round(throughput, 0),
                'equipment_health_score': round(np.random.uniform(0.6, 0.98), 3),
                'last_maintenance_date': self.fake.date_between(start_date='-120d', end_date='today'),
                'next_maintenance_date': self.fake.date_between(start_date='today', end_date='+150d')
            })
            facility_id += 1
        
        return pd.DataFrame(facilities)
    
    def generate_time_series_data(self, facilities_df: pd.DataFrame, 
                                 start_date: str = '2022-01-01', 
                                 end_date: str = '2024-12-31') -> Dict[str, pd.DataFrame]:
        """
        Generate time series datasets for production, prices, and forecasting
        
        Args:
            facilities_df: Geospatial facilities data
            start_date: Start date for time series
            end_date: End date for time series
            
        Returns:
            Dictionary containing different time series datasets
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # 1. Daily Production Data
        production_data = []
        oil_wells = facilities_df[facilities_df['facility_type'] == 'oil_well'].copy()
        
        for _, well in oil_wells.iterrows():
            base_oil_prod = well['oil_production_bpd']
            base_gas_prod = well['gas_production_mcfd']
            
            for date in date_range:
                # Seasonal factors
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)
                
                # Random daily variation
                daily_variation = np.random.normal(1, 0.05)
                
                # Equipment degradation over time
                days_since_start = (date - start_dt).days
                degradation_factor = np.exp(-days_since_start * 0.00005)
                
                # Maintenance impact
                if well['status'] == 'maintenance':
                    maintenance_factor = 0.1
                elif well['status'] == 'inactive':
                    maintenance_factor = 0
                else:
                    maintenance_factor = 1
                
                # Calculate actual production
                oil_prod = (base_oil_prod * seasonal_factor * daily_variation * 
                           degradation_factor * maintenance_factor)
                gas_prod = (base_gas_prod * seasonal_factor * daily_variation * 
                           degradation_factor * maintenance_factor)
                
                # Add some random outages
                if np.random.random() < 0.005:  # 0.5% chance of outage
                    oil_prod *= 0.1
                    gas_prod *= 0.1
                
                production_data.append({
                    'date': date,
                    'facility_id': well['facility_id'],
                    'facility_type': 'oil_well',
                    'region': well['region'],
                    'country': well['country'],
                    'oil_production_bpd': round(max(0, oil_prod), 2),
                    'gas_production_mcfd': round(max(0, gas_prod), 2),
                    'water_cut_percent': round(np.random.uniform(5, 40), 1),
                    'gor': round(np.random.uniform(500, 2000), 0),  # Gas-oil ratio
                    'wellhead_pressure_psi': round(np.random.uniform(100, 3000), 0),
                    'flowing_tubing_pressure_psi': round(np.random.uniform(50, 1500), 0)
                })
        
        production_df = pd.DataFrame(production_data)
        
        # 2. Price Data (Oil, Gas, Products)
        price_data = []
        base_oil_price = 70  # USD/barrel
        base_gas_price = 3.5  # USD/MCF
        
        for date in date_range:
            # Oil price with trend and volatility
            days_since_start = (date - start_dt).days
            trend = 0.02 * np.sin(days_since_start * 2 * np.pi / 365) + 0.001 * days_since_start
            volatility = np.random.normal(0, 0.03)
            oil_price = base_oil_price * (1 + trend + volatility)
            
            # Gas price correlated with oil but different seasonality
            gas_seasonal = 0.3 * np.sin((days_since_start + 90) * 2 * np.pi / 365)
            gas_volatility = np.random.normal(0, 0.05)
            gas_price = base_gas_price * (1 + 0.5 * trend + gas_seasonal + gas_volatility)
            
            # Refined product prices
            gasoline_price = oil_price * 1.4 + np.random.normal(0, 2)
            diesel_price = oil_price * 1.3 + np.random.normal(0, 1.5)
            jet_fuel_price = oil_price * 1.35 + np.random.normal(0, 1.8)
            
            price_data.append({
                'date': date,
                'crude_oil_price_usd_bbl': round(max(20, oil_price), 2),
                'natural_gas_price_usd_mcf': round(max(1, gas_price), 2),
                'gasoline_price_usd_gal': round(max(1, gasoline_price), 2),
                'diesel_price_usd_gal': round(max(1, diesel_price), 2),
                'jet_fuel_price_usd_gal': round(max(1, jet_fuel_price), 2),
                'brent_crude_usd_bbl': round(max(20, oil_price * 1.02), 2),
                'wti_crude_usd_bbl': round(max(20, oil_price * 0.98), 2)
            })
        
        price_df = pd.DataFrame(price_data)
        
        # 3. Demand Forecasting Data
        demand_data = []
        regions = list(self.regions.keys())
        
        for region in regions:
            for date in date_range:
                # Base demand varies by region
                region_populations = {
                    'North America': 500, 'Europe': 400, 'Asia Pacific': 800,
                    'Middle East': 200, 'Africa': 300, 'South America': 250
                }
                base_demand = region_populations[region] * 1000  # thousand bpd
                
                # Seasonal demand
                day_of_year = date.timetuple().tm_yday
                if region in ['North America', 'Europe']:
                    # Winter heating demand
                    seasonal = 1 + 0.2 * np.cos(2 * np.pi * (day_of_year - 365/4) / 365)
                else:
                    # Summer cooling demand
                    seasonal = 1 + 0.15 * np.cos(2 * np.pi * (day_of_year - 3*365/4) / 365)
                
                # Economic growth trend
                days_since_start = (date - start_dt).days
                growth_factor = 1 + 0.02 * (days_since_start / 365)
                
                # Random variation
                random_factor = np.random.normal(1, 0.03)
                
                oil_demand = base_demand * seasonal * growth_factor * random_factor
                gas_demand = base_demand * 0.8 * seasonal * growth_factor * random_factor
                
                demand_data.append({
                    'date': date,
                    'region': region,
                    'oil_demand_thousand_bpd': round(max(0, oil_demand), 1),
                    'gas_demand_thousand_mcfd': round(max(0, gas_demand), 1),
                    'gasoline_demand_thousand_bpd': round(oil_demand * 0.45, 1),
                    'diesel_demand_thousand_bpd': round(oil_demand * 0.28, 1),
                    'jet_fuel_demand_thousand_bpd': round(oil_demand * 0.12, 1)
                })
        
        demand_df = pd.DataFrame(demand_data)
        
        # 4. Equipment Maintenance Schedule
        maintenance_data = []
        all_facilities = facilities_df.copy()
        
        for _, facility in all_facilities.iterrows():
            # Generate maintenance events
            current_date = start_dt
            while current_date <= end_dt:
                # Determine maintenance type
                maintenance_types = ['preventive', 'corrective', 'predictive', 'shutdown']
                type_probs = [0.6, 0.2, 0.15, 0.05]
                maintenance_type = np.random.choice(maintenance_types, p=type_probs)
                
                # Maintenance duration and cost
                if maintenance_type == 'shutdown':
                    duration_hours = np.random.uniform(72, 240)  # 3-10 days
                    cost_factor = 5.0
                elif maintenance_type == 'corrective':
                    duration_hours = np.random.uniform(8, 48)  # 8 hours - 2 days
                    cost_factor = 3.0
                elif maintenance_type == 'predictive':
                    duration_hours = np.random.uniform(4, 24)  # 4-24 hours
                    cost_factor = 1.5
                else:  # preventive
                    duration_hours = np.random.uniform(2, 12)  # 2-12 hours
                    cost_factor = 1.0
                
                # Base cost depends on facility type
                base_costs = {
                    'oil_well': 15000,
                    'refinery': 150000,
                    'storage_tank': 25000,
                    'processing_plant': 75000,
                    'terminal': 50000,
                    'compressor_station': 40000,
                    'pipeline_station': 30000
                }
                
                base_cost = base_costs.get(facility['facility_type'], 20000)
                maintenance_cost = base_cost * cost_factor * np.random.uniform(0.7, 1.3)
                
                maintenance_data.append({
                    'date': current_date,
                    'facility_id': facility['facility_id'],
                    'facility_type': facility['facility_type'],
                    'region': facility['region'],
                    'maintenance_type': maintenance_type,
                    'duration_hours': round(duration_hours, 1),
                    'cost_usd': round(maintenance_cost, 0),
                    'planned_start': current_date,
                    'planned_end': current_date + timedelta(hours=duration_hours),
                    'priority': np.random.choice(['low', 'medium', 'high', 'critical'], 
                                               p=[0.3, 0.4, 0.25, 0.05])
                })
                
                # Next maintenance interval
                if maintenance_type == 'shutdown':
                    days_to_next = np.random.uniform(180, 365)
                elif maintenance_type == 'corrective':
                    days_to_next = np.random.uniform(90, 180)
                else:
                    days_to_next = np.random.uniform(30, 90)
                
                current_date += timedelta(days=days_to_next)
        
        maintenance_df = pd.DataFrame(maintenance_data)
        
        # 5. Weather Impact Data
        weather_data = []
        
        for region in regions:
            for date in date_range:
                # Seasonal temperature patterns
                day_of_year = date.timetuple().tm_yday
                
                # Base temperatures by region
                base_temps = {
                    'North America': 15, 'Europe': 12, 'Asia Pacific': 20,
                    'Middle East': 28, 'Africa': 25, 'South America': 22
                }
                
                base_temp = base_temps[region]
                seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                daily_temp = seasonal_temp + np.random.normal(0, 3)
                
                # Weather conditions
                humidity = np.random.uniform(30, 90)
                wind_speed = np.random.exponential(8)
                precipitation = np.random.exponential(2) if np.random.random() < 0.3 else 0
                
                # Extreme weather events
                is_extreme = np.random.random() < 0.02  # 2% chance
                if is_extreme:
                    extreme_type = np.random.choice(['hurricane', 'heat_wave', 'cold_snap', 'heavy_rain'])
                    if extreme_type == 'hurricane':
                        wind_speed *= 3
                        precipitation *= 5
                    elif extreme_type == 'heat_wave':
                        daily_temp += 10
                    elif extreme_type == 'cold_snap':
                        daily_temp -= 15
                    elif extreme_type == 'heavy_rain':
                        precipitation *= 8
                else:
                    extreme_type = 'normal'
                
                weather_data.append({
                    'date': date,
                    'region': region,
                    'temperature_celsius': round(daily_temp, 1),
                    'humidity_percent': round(humidity, 1),
                    'wind_speed_kmh': round(wind_speed, 1),
                    'precipitation_mm': round(precipitation, 1),
                    'weather_condition': extreme_type,
                    'operation_impact_score': round(np.random.uniform(0, 1), 2) if is_extreme else round(np.random.uniform(0.8, 1), 2)
                })
        
        weather_df = pd.DataFrame(weather_data)
        
        return {
            'production': production_df,
            'prices': price_df,
            'demand': demand_df,
            'maintenance': maintenance_df,
            'weather': weather_df
        }
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Validate data quality across all datasets
        
        Args:
            data: Dictionary of DataFrames to validate
            
        Returns:
            Dictionary of quality scores by dataset
        """
        quality_scores = {}
        
        for dataset_name, df in data.items():
            score_components = []
            
            # 1. Missing values score
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            missing_score = max(0, 1 - missing_ratio * 10)  # Penalize missing values
            score_components.append(missing_score)
            
            # 2. Data type consistency score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            type_consistency = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
            score_components.append(min(1.0, type_consistency * 2))
            
            # 3. Value range validation
            if dataset_name == 'production':
                # Production values should be non-negative
                negative_prod = (df[['oil_production_bpd', 'gas_production_mcfd']] < 0).sum().sum()
                range_score = max(0, 1 - negative_prod / len(df))
            elif dataset_name == 'prices':
                # Prices should be reasonable
                price_cols = [col for col in df.columns if 'price' in col.lower()]
                reasonable_prices = True
                for col in price_cols:
                    if df[col].min() < 0 or df[col].max() > 1000:  # Reasonable price bounds
                        reasonable_prices = False
                range_score = 1.0 if reasonable_prices else 0.5
            else:
                range_score = 1.0  # Default good score for other datasets
            
            score_components.append(range_score)
            
            # 4. Temporal consistency (for time series data)
            if 'date' in df.columns:
                date_diffs = df['date'].diff().dt.days
                expected_diff = 1  # Daily data
                temporal_consistency = (date_diffs == expected_diff).mean()
                score_components.append(temporal_consistency)
            else:
                score_components.append(1.0)  # Not applicable
            
            # Overall quality score
            quality_scores[dataset_name] = np.mean(score_components)
        
        return quality_scores
    
    def save_datasets(self, geospatial_df: pd.DataFrame, 
                     time_series_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Save all datasets with MLflow tracking
        
        Args:
            geospatial_df: Geospatial facilities data
            time_series_data: Dictionary of time series datasets
            
        Returns:
            Dictionary of saved file paths
        """
        saved_paths = {}
        
        with mlflow.start_run(run_name="oil_gas_data_generation") as run:
            mlflow.set_tag("data_type", "oil_gas_synthetic")
            mlflow.set_tag("project", self.project_name)
            
            # Save geospatial data
            geospatial_path = self.directories['data'] / 'geospatial_facilities.parquet'
            geospatial_df.to_parquet(geospatial_path, index=False)
            saved_paths['geospatial'] = str(geospatial_path)
            
            # Log geospatial metrics
            mlflow.log_metric("total_facilities", len(geospatial_df))
            mlflow.log_metric("oil_wells", len(geospatial_df[geospatial_df['facility_type'] == 'oil_well']))
            mlflow.log_metric("refineries", len(geospatial_df[geospatial_df['facility_type'] == 'refinery']))
            mlflow.log_metric("regions_covered", geospatial_df['region'].nunique())
            
            # Log sample data
            sample_data = geospatial_df.head(100).to_dict(orient='records')
            with open(self.directories['artifacts'] / 'geospatial_sample.json', 'w') as f:
                json.dump(sample_data, f, indent=2, default=str)
            mlflow.log_artifact(str(self.directories['artifacts'] / 'geospatial_sample.json'))
            mlflow.log_artifact(str(geospatial_path))
            
            # Save time series data
            for dataset_name, df in time_series_data.items():
                file_path = self.directories['data'] / f'{dataset_name}_timeseries.parquet'
                df.to_parquet(file_path, index=False)
                saved_paths[dataset_name] = str(file_path)
                
                # Log metrics for each dataset
                mlflow.log_metric(f"{dataset_name}_rows", len(df))
                mlflow.log_metric(f"{dataset_name}_columns", len(df.columns))
                
                if 'date' in df.columns:
                    date_range_days = (df['date'].max() - df['date'].min()).days
                    mlflow.log_metric(f"{dataset_name}_date_range_days", date_range_days)
                
                # Log sample data
                sample_data = df.head(100).to_dict(orient='records')
                sample_file = self.directories['artifacts'] / f"{dataset_name}_sample.json"
                with open(sample_file, 'w') as f:
                    json.dump(sample_data, f, indent=2, default=str)
                mlflow.log_artifact(str(sample_file))
                mlflow.log_artifact(str(file_path))
            
            # Validate data quality
            all_data = {'geospatial': geospatial_df, **time_series_data}
            quality_scores = self.validate_data_quality(all_data)
            
            for dataset_name, score in quality_scores.items():
                mlflow.log_metric(f"{dataset_name}_quality_score", score)
            
            overall_quality = np.mean(list(quality_scores.values()))
            mlflow.log_metric("overall_data_quality", overall_quality)
            
            # Create data summary report
            summary = {
                'generation_date': datetime.now().isoformat(),
                'project_name': self.project_name,
                'total_facilities': len(geospatial_df),
                'facility_breakdown': geospatial_df['facility_type'].value_counts().to_dict(),
                'regions_covered': geospatial_df['region'].unique().tolist(),
                'time_series_datasets': list(time_series_data.keys()),
                'quality_scores': quality_scores,
                'overall_quality': overall_quality,
                'saved_files': saved_paths
            }
            
            summary_path = self.directories['reports'] / 'data_generation_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            mlflow.log_artifact(str(summary_path))
            saved_paths['summary'] = str(summary_path)

            # Write to workflow outputs if directory exists (Flow mode)
            workflow_output = Path("/workflow/outputs/data_summary")
            if workflow_output.parent.exists():
                workflow_output.write_text(json.dumps(summary))
                print(f"✓ Wrote workflow output to {workflow_output}")

            mlflow.set_tag("generation_status", "success")

        return saved_paths

def main():
    """Main function to generate oil and gas synthetic datasets"""
    try:
        # Initialize generator
        generator = OilGasDataGenerator("Oil-and-Gas-Demo")

        print("🛢️  Oil & Gas Synthetic Data Generator")
        print("=" * 50)
        print(f"Project: {generator.project_name}")
        print(f"Data directory: {generator.directories['data']}")
        print(f"Artifacts directory: {generator.directories['artifacts']}")
        print()

        # Generate geospatial data
        print("📍 Generating geospatial facilities data...")
        geospatial_df = generator.generate_geospatial_data(
            n_wells=1500,
            n_refineries=25,
            n_facilities=200
        )
        print(f"   Generated {len(geospatial_df)} facilities across {geospatial_df['region'].nunique()} regions")

        # Generate time series data
        print("📈 Generating time series datasets...")
        time_series_data = generator.generate_time_series_data(
            geospatial_df,
            start_date='2022-01-01',
            end_date='2024-12-31'
        )

        for dataset_name, df in time_series_data.items():
            print(f"   {dataset_name}: {len(df):,} records")

        # Validate data quality
        print("✅ Validating data quality...")
        all_data = {'geospatial': geospatial_df, **time_series_data}
        quality_scores = generator.validate_data_quality(all_data)

        for dataset_name, score in quality_scores.items():
            print(f"   {dataset_name}: {score:.3f}")

        overall_quality = np.mean(list(quality_scores.values()))
        print(f"   Overall quality score: {overall_quality:.3f}")

        # Save datasets
        print("💾 Saving datasets...")
        saved_paths = generator.save_datasets(geospatial_df, time_series_data)

        print("📁 Files saved:")
        for dataset_name, path in saved_paths.items():
            print(f"   {dataset_name}: {path}")

        print()
        print("✨ Oil & Gas synthetic data generation completed successfully!")
        print(f"📊 MLflow experiment: oil_gas_data_generation_{generator.project_name}")
        print(f"🌐 MLflow UI: http://localhost:8768")

        return saved_paths

    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        # CRITICAL: Write error output for Flow execution
        # This ensures sidecar uploader has a file even if script fails
        workflow_output = Path("/workflow/outputs/data_summary")
        if workflow_output.parent.exists():
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'framework': 'data_generation',
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            workflow_output.write_text(json.dumps(error_data))
            print(f"✓ Wrote error output to {workflow_output}")
        raise

if __name__ == "__main__":
    saved_paths = main()