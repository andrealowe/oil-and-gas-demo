#!/usr/bin/env python3
"""
Demo Script: Real Data Acquisition 

This script demonstrates how to use the real_data_acquisition.py script
and shows what the output looks like when API keys are properly configured.
"""

import os
import sys
sys.path.insert(0, '/mnt/code')

from scripts.data_config import get_data_paths

def print_data_acquisition_guide():
    """Print comprehensive guide for acquiring real oil & gas data"""
    
    print("=" * 80)
    print("🛢️  REAL OIL & GAS DATA ACQUISITION GUIDE")
    print("=" * 80)
    print()
    
    print("📊 STEP 1: GET FREE API KEYS")
    print("-" * 40)
    print("EIA (Energy Information Administration) - REQUIRED:")
    print("  • URL: https://www.eia.gov/opendata/register.php")
    print("  • Provides: US oil/gas production data by state")
    print("  • Free, instant approval")
    print()
    
    print("FRED (Federal Reserve Economic Data) - OPTIONAL:")
    print("  • URL: https://research.stlouisfed.org/useraccount/apikey") 
    print("  • Provides: Crude oil and natural gas price history")
    print("  • Free, instant approval")
    print()
    
    print("📋 STEP 2: USAGE EXAMPLES")
    print("-" * 40)
    print("Option A - Environment Variables (Recommended):")
    print("  export EIA_API_KEY='your_eia_key_here'")
    print("  export FRED_API_KEY='your_fred_key_here'")
    print("  python scripts/real_data_acquisition.py")
    print()
    
    print("Option B - Command Line Arguments:")
    print("  python scripts/real_data_acquisition.py \\")
    print("    --eia-key YOUR_EIA_KEY \\") 
    print("    --fred-key YOUR_FRED_KEY \\")
    print("    --start-year 2015")
    print()
    
    print("Option C - EIA Only (Skip Price Data):")
    print("  python scripts/real_data_acquisition.py --eia-key YOUR_EIA_KEY")
    print()
    
    print("🗂️  STEP 3: OUTPUT DATA FILES")
    print("-" * 40)
    
    # Show where data will be saved
    paths = get_data_paths("Oil-and-Gas-Demo")
    data_dir = paths['base_data_path']
    
    print(f"Data will be saved to: {data_dir}/")
    print()
    print("Files created:")
    print("  ✓ production_timeseries.parquet - Monthly oil/gas production")
    print("  ✓ price_data.csv - Historical crude oil & natural gas prices")
    print("  ✓ geospatial_data.csv - Facility locations and operational metrics")
    print()
    
    print("📈 STEP 4: DATA QUALITY & COVERAGE")
    print("-" * 40)
    print("Production Data (EIA):")
    print("  • Monthly frequency: 2014-present")
    print("  • US total production: Crude oil (barrels/day)")
    print("  • Geographic coverage: National aggregated")
    print("  • Data quality: Government-verified")
    print()
    
    print("Price Data (FRED):")
    print("  • Daily/Monthly frequency: 1990-present") 
    print("  • Brent Crude, WTI Crude, Henry Hub Natural Gas")
    print("  • Global benchmark prices")
    print("  • Data source: IMF, EIA")
    print()
    
    print("🔧 STEP 5: INTEGRATION WITH YOUR MODELS")
    print("-" * 40)
    print("The script automatically:")
    print("  ✓ Formats data to match your existing forecasting models")
    print("  ✓ Handles missing values and data quality issues")
    print("  ✓ Creates proper date indexing for time series forecasting")
    print("  ✓ Extends limited data with realistic synthetic data if needed")
    print("  ✓ Saves to your project's data directory structure")
    print()
    
    print("🚀 STEP 6: RUN YOUR FORECASTING MODELS")
    print("-" * 40)
    print("After data acquisition, run your existing models:")
    print("  python src/models/autogluon_forecasting.py")
    print("  python src/models/prophet_forecasting.py") 
    print("  python src/models/nixtla_forecasting.py")
    print("  python src/models/oil_gas_forecasting.py")
    print()
    print("Or run the complete pipeline:")
    print("  python scripts/flows.py")
    print()
    
    print("⚡ QUICK START")
    print("-" * 40)
    print("1. Get EIA key: https://www.eia.gov/opendata/register.php")
    print("2. export EIA_API_KEY='your_key'")
    print("3. python scripts/real_data_acquisition.py")
    print("4. python src/models/autogluon_forecasting.py")
    print()
    
    print("✅ SUCCESS! You now have real oil & gas data for forecasting.")
    print("=" * 80)

def check_current_data():
    """Check what data currently exists"""
    paths = get_data_paths("Oil-and-Gas-Demo")
    data_dir = paths['base_data_path']
    
    print(f"Current data in {data_dir}:")
    
    if data_dir.exists():
        files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
        if files:
            for file in sorted(files):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
        else:
            print("  (No data files found)")
    else:
        print("  (Data directory doesn't exist yet)")
    print()

if __name__ == "__main__":
    check_current_data()
    print_data_acquisition_guide()