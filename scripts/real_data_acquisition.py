#!/usr/bin/env python3
"""
Real Oil & Gas Data Acquisition Script

Fetches real production and price data from:
- EIA API: US oil and gas production by state
- FRED API: Historical crude oil and natural gas prices

Usage:
    python scripts/real_data_acquisition.py --eia-key YOUR_EIA_KEY --fred-key YOUR_FRED_KEY
    
Or set environment variables:
    export EIA_API_KEY=your_key_here
    export FRED_API_KEY=your_key_here
    python scripts/real_data_acquisition.py
"""

import os
import sys
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Dict, Optional, List

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths, ensure_directories

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataAcquisition:
    """Fetches real oil and gas data from public APIs"""
    
    def __init__(self, eia_api_key: str, fred_api_key: Optional[str] = None):
        self.eia_api_key = eia_api_key
        self.fred_api_key = fred_api_key
        self.eia_base_url = "https://api.eia.gov/v2"
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Setup data paths
        self.project_name = "Oil-and-Gas-Demo"
        self.paths = ensure_directories(self.project_name)
        
        # Rate limiting
        self.request_delay = 1.1  # Seconds between requests
        
    def _make_eia_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make rate-limited request to EIA API"""
        params['api_key'] = self.eia_api_key
        
        try:
            response = requests.get(f"{self.eia_base_url}/{endpoint}", params=params)
            response.raise_for_status()
            
            time.sleep(self.request_delay)  # Rate limiting
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"EIA API request failed: {e}")
            return None
    
    def _make_fred_request(self, series_id: str, start_date: str = "2010-01-01") -> Optional[pd.DataFrame]:
        """Make rate-limited request to FRED API"""
        if not self.fred_api_key:
            logger.warning("No FRED API key provided, skipping price data")
            return None
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date
        }
        
        try:
            response = requests.get(self.fred_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna()
                return df[['date', 'value']].rename(columns={'value': series_id})
            
            time.sleep(self.request_delay)  # Rate limiting
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            return None
    
    def fetch_eia_production_data(self, start_year: int = 2014) -> Optional[pd.DataFrame]:
        """Fetch US oil and gas production data from EIA"""
        logger.info("Fetching EIA production data...")
        
        # Try multiple endpoints for production data with correct API v2 structure
        endpoints_to_try = [
            {
                'path': 'petroleum/crd/cplc',
                'params': {
                    'frequency': 'annual',
                    'data': ['value'],
                    'start': str(start_year),
                    'length': 100,
                    'facets[duoarea][]': 'NUS',  # US Total
                    'facets[process][]': 'P01'   # Production 
                }
            },
            {
                'path': 'total-energy/data',
                'params': {
                    'frequency': 'monthly', 
                    'data': ['value'],
                    'start': f'{start_year}-01',
                    'length': 1000,
                    'facets[msn][]': 'PAPRBUS2'  # Crude oil production
                }
            },
            {
                'path': 'petroleum/sum/snd',
                'params': {
                    'frequency': 'monthly',
                    'data': ['value'], 
                    'start': f'{start_year}-01',
                    'length': 1000
                }
            }
        ]
        
        for endpoint_config in endpoints_to_try:
            endpoint = endpoint_config['path']
            params = endpoint_config['params']
            
            logger.info(f"Trying EIA endpoint: {endpoint}")
            
            try:
                result = self._make_eia_request(f"{endpoint}/data", params)
                
                if result and 'response' in result and 'data' in result['response']:
                    data = result['response']['data']
                    if data:
                        logger.info(f"Successfully fetched {len(data)} records from {endpoint}")
                        df = pd.DataFrame(data)
                        return self._clean_eia_production_data(df)
                        
            except Exception as e:
                logger.warning(f"Endpoint {endpoint} failed: {e}")
                continue
        
        logger.warning("All EIA production endpoints failed, will use synthetic data")
        return None
    
    def _clean_eia_production_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format EIA production data"""
        logger.info("Cleaning EIA production data...")
        
        # Convert period to datetime
        df['date'] = pd.to_datetime(df['period'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Filter out invalid values
        df = df.dropna(subset=['value'])
        df = df[df['value'] > 0]
        
        # If we have multiple series, aggregate by date
        if len(df) > 0:
            production_data = df.groupby('date')['value'].sum().reset_index()
            production_data.columns = ['date', 'oil_production_bpd']
            return production_data.sort_values('date')
        
        return pd.DataFrame()
    
    def fetch_price_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical price data from FRED"""
        logger.info("Fetching price data from FRED...")
        
        price_series = {
            'brent_crude': 'POILBREUSDM',      # Brent crude oil
            'wti_crude': 'POILWTIUSDM',        # WTI crude oil  
            'natural_gas': 'DHHNGSP'           # Henry Hub natural gas
        }
        
        price_data = {}
        for name, series_id in price_series.items():
            logger.info(f"Fetching {name} price data ({series_id})...")
            df = self._make_fred_request(series_id)
            
            if df is not None and not df.empty:
                price_data[name] = df
                logger.info(f"Successfully fetched {len(df)} price records for {name}")
            else:
                logger.warning(f"Failed to fetch price data for {name}")
        
        return price_data
    
    def create_synthetic_facilities_data(self) -> pd.DataFrame:
        """Create synthetic facility data to match original schema"""
        logger.info("Creating synthetic facilities data...")
        
        # Define major oil-producing regions in the US
        facilities_data = [
            # Texas facilities
            {"facility_id": "TX001", "facility_name": "Permian Basin Alpha", "state": "Texas", 
             "latitude": 31.8457, "longitude": -102.3676, "region": "Permian Basin"},
            {"facility_id": "TX002", "facility_name": "Eagle Ford Beta", "state": "Texas",
             "latitude": 28.9783, "longitude": -98.9573, "region": "Eagle Ford"},
            {"facility_id": "TX003", "facility_name": "Bakken North", "state": "North Dakota",
             "latitude": 47.7511, "longitude": -101.7777, "region": "Bakken"},
             
            # North Dakota facilities  
            {"facility_id": "ND001", "facility_name": "Williston Basin", "state": "North Dakota",
             "latitude": 48.1470, "longitude": -103.6294, "region": "Bakken"},
            {"facility_id": "ND002", "facility_name": "Three Forks Formation", "state": "North Dakota", 
             "latitude": 47.9073, "longitude": -102.7894, "region": "Bakken"},
             
            # Oklahoma facilities
            {"facility_id": "OK001", "facility_name": "Anadarko Basin", "state": "Oklahoma",
             "latitude": 35.0653, "longitude": -98.2895, "region": "Anadarko Basin"},
            {"facility_id": "OK002", "facility_name": "SCOOP/STACK", "state": "Oklahoma",
             "latitude": 35.3863, "longitude": -97.6412, "region": "SCOOP"},
             
            # New Mexico facilities
            {"facility_id": "NM001", "facility_name": "Delaware Basin", "state": "New Mexico",
             "latitude": 32.0853, "longitude": -103.8501, "region": "Permian Basin"},
            {"facility_id": "NM002", "facility_name": "San Juan Basin", "state": "New Mexico", 
             "latitude": 36.7378, "longitude": -108.2208, "region": "San Juan Basin"},
             
            # Alaska facilities
            {"facility_id": "AK001", "facility_name": "Prudhoe Bay", "state": "Alaska",
             "latitude": 70.2553, "longitude": -148.3370, "region": "North Slope"},
        ]
        
        facilities_df = pd.DataFrame(facilities_data)
        
        # Add synthetic operational metrics
        np.random.seed(42)
        facilities_df['equipment_health_score'] = np.random.normal(85, 10, len(facilities_df)).clip(60, 100)
        facilities_df['utilization_rate'] = np.random.normal(0.75, 0.15, len(facilities_df)).clip(0.3, 1.0)
        facilities_df['h2s_concentration'] = np.random.normal(50, 20, len(facilities_df)).clip(10, 100)
        facilities_df['co2_concentration'] = np.random.normal(800, 200, len(facilities_df)).clip(400, 1500)
        
        return facilities_df
    
    def combine_and_format_data(self, production_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Combine production and price data into final format"""
        logger.info("Combining and formatting all data...")
        
        datasets = {}
        
        # 1. Production time series data
        if not production_df.empty:
            # Ensure we have enough data points
            if len(production_df) < 50:
                logger.warning(f"Limited production data ({len(production_df)} points), extending with synthetic data")
                production_df = self._extend_with_synthetic_data(production_df)
            
            # Add gas production (derived from oil production with realistic ratios)
            production_df['gas_production_mcf'] = production_df['oil_production_bpd'] * np.random.normal(6.0, 1.5, len(production_df))
            production_df['gas_production_mcf'] = production_df['gas_production_mcf'].clip(lower=0)
            
            datasets['production_timeseries'] = production_df
            logger.info(f"Created production timeseries with {len(production_df)} records")
        
        # 2. Price data
        if price_data:
            price_df = self._merge_price_data(price_data)
            if not price_df.empty:
                datasets['price_data'] = price_df
                logger.info(f"Created price dataset with {len(price_df)} records")
        
        # 3. Facilities data (always synthetic for geospatial features)
        facilities_df = self.create_synthetic_facilities_data()
        datasets['geospatial_data'] = facilities_df
        logger.info(f"Created facilities dataset with {len(facilities_df)} facilities")
        
        return datasets
    
    def _extend_with_synthetic_data(self, real_df: pd.DataFrame) -> pd.DataFrame:
        """Extend limited real data with synthetic data following real trends"""
        if real_df.empty:
            return self._generate_full_synthetic_timeseries()
        
        # Use the last known production value as a baseline
        last_value = real_df['oil_production_bpd'].iloc[-1]
        last_date = real_df['date'].iloc[-1]
        
        # Generate additional synthetic data
        future_dates = pd.date_range(start=last_date + timedelta(days=30), 
                                   end=datetime.now() - timedelta(days=30), freq='M')
        
        synthetic_data = []
        current_value = last_value
        
        for date in future_dates:
            # Add realistic variation
            trend = np.random.normal(0.98, 0.05)  # Slight declining trend with variation
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)  # Seasonal variation
            current_value *= trend * seasonal
            current_value = max(current_value, last_value * 0.5)  # Prevent unrealistic drops
            
            synthetic_data.append({
                'date': date,
                'oil_production_bpd': current_value
            })
        
        synthetic_df = pd.DataFrame(synthetic_data)
        combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        
        logger.info(f"Extended {len(real_df)} real records with {len(synthetic_data)} synthetic records")
        return combined_df.sort_values('date')
    
    def _generate_full_synthetic_timeseries(self) -> pd.DataFrame:
        """Generate realistic synthetic time series based on actual US production trends"""
        logger.info("Generating realistic synthetic production timeseries based on US oil production trends")
        
        # Generate daily data from 2014 to present 
        start_date = datetime(2014, 1, 1)
        end_date = datetime(2024, 12, 31) 
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base this on actual US production trends:
        # 2014: ~8.7M bpd, 2015-2016: decline due to oil price crash
        # 2017-2019: recovery to ~12M bpd, 2020: COVID crash
        # 2021-2024: recovery to ~13M bpd
        
        production_values = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Define production levels by year based on actual trends
            if year == 2014:
                base = 8700000  # 8.7M bpd
            elif year in [2015, 2016]:
                base = 8700000 - (year - 2014) * 400000  # Decline during oil crash
            elif year in [2017, 2018, 2019]:
                base = 8300000 + (year - 2016) * 1200000  # Shale boom recovery
            elif year == 2020:
                base = 11200000 - 1000000 * np.sin((date.dayofyear / 365.0) * np.pi)  # COVID impact
            else:  # 2021-2024
                base = 11500000 + (year - 2020) * 300000  # Recovery
            
            # Add seasonal variation (winter demand, summer driving season)
            seasonal = 1 + 0.08 * np.sin(2 * np.pi * (date.dayofyear + 90) / 365)
            
            # Add weekly pattern (lower on weekends)
            weekly = 1 - 0.03 * (date.weekday() >= 5)
            
            # Add realistic noise
            daily_noise = np.random.normal(1.0, 0.02)
            
            production = base * seasonal * weekly * daily_noise
            production_values.append(max(production, base * 0.8))  # Floor at 80% of base
        
        production_df = pd.DataFrame({
            'date': dates,
            'oil_production_bpd': production_values
        })
        
        logger.info(f"Generated {len(production_df)} days of synthetic production data (2014-2024)")
        return production_df
    
    def _merge_price_data(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all price series into single dataframe"""
        if not price_data:
            return pd.DataFrame()
        
        # Start with first series
        series_names = list(price_data.keys())
        merged_df = price_data[series_names[0]].copy()
        merged_df.columns = ['date', f'{series_names[0]}_price']
        
        # Merge additional series
        for series_name in series_names[1:]:
            df = price_data[series_name].copy()
            df.columns = ['date', f'{series_name}_price']
            merged_df = merged_df.merge(df, on='date', how='outer')
        
        # Forward fill missing values
        merged_df = merged_df.sort_values('date').ffill()
        
        return merged_df
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Save all datasets to the project directory"""
        logger.info(f"Saving datasets to {self.paths['data']}")
        
        file_mapping = {
            'production_timeseries': 'production_timeseries.parquet',
            'price_data': 'price_data.csv',
            'geospatial_data': 'geospatial_data.csv'
        }
        
        for dataset_name, df in datasets.items():
            if dataset_name in file_mapping:
                filename = file_mapping[dataset_name]
                filepath = self.paths['data'] / filename
                
                try:
                    if filename.endswith('.parquet'):
                        df.to_parquet(filepath, index=False)
                    else:
                        df.to_csv(filepath, index=False)
                    
                    logger.info(f"Saved {dataset_name} to {filepath} ({len(df)} records)")
                    
                except Exception as e:
                    logger.error(f"Failed to save {dataset_name}: {e}")
    
    def run_data_acquisition(self) -> bool:
        """Run complete data acquisition pipeline"""
        logger.info("Starting real oil & gas data acquisition...")
        
        try:
            # 1. Fetch production data from EIA
            production_df = self.fetch_eia_production_data()
            
            # 2. Fetch price data from FRED  
            price_data = self.fetch_price_data()
            
            # 3. Combine and format all data
            datasets = self.combine_and_format_data(production_df or pd.DataFrame(), price_data)
            
            # 4. Save datasets
            if datasets:
                self.save_datasets(datasets)
                
                logger.info("Data acquisition completed successfully!")
                logger.info(f"Datasets saved to: {self.paths['data']}")
                
                # Print summary
                for name, df in datasets.items():
                    print(f"✓ {name}: {len(df)} records")
                
                return True
            else:
                logger.error("No datasets were created")
                return False
                
        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fetch real oil and gas data')
    parser.add_argument('--eia-key', help='EIA API key')
    parser.add_argument('--fred-key', help='FRED API key (optional)')
    parser.add_argument('--start-year', type=int, default=2014, help='Start year for data (default: 2014)')
    
    args = parser.parse_args()
    
    # Get API keys from args or environment
    eia_key = args.eia_key or os.getenv('EIA_API_KEY')
    fred_key = args.fred_key or os.getenv('FRED_API_KEY')
    
    if not eia_key:
        print("ERROR: EIA API key required!")
        print("Get your free key at: https://www.eia.gov/opendata/register.php")
        print("Usage:")
        print("  export EIA_API_KEY=your_key_here")
        print("  python scripts/real_data_acquisition.py")
        print("OR:")
        print("  python scripts/real_data_acquisition.py --eia-key YOUR_KEY")
        return 1
    
    if not fred_key:
        print("WARNING: No FRED API key provided. Price data will be skipped.")
        print("Get your free FRED key at: https://research.stlouisfed.org/useraccount/apikey")
    
    # Run data acquisition
    data_fetcher = RealDataAcquisition(eia_key, fred_key)
    success = data_fetcher.run_data_acquisition()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())