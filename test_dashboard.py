#!/usr/bin/env python3
"""
Test script for oil & gas dashboards
Validates data loading and key components work properly
"""

import sys
sys.path.insert(0, '/mnt/code')
import pandas as pd
import numpy as np
from scripts.geospatial_dashboard import (
    load_geospatial_data, 
    create_facility_map, 
    create_sample_geospatial_data,
    display_kpi_metrics
)

def test_geospatial_dashboard():
    """Test geospatial dashboard components"""
    print("🧪 Testing Geospatial Dashboard Components...")
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        df = load_geospatial_data()
        print(f"   ✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for required columns
        required_cols = [
            'facility_id', 'facility_name', 'facility_type', 'region',
            'latitude', 'longitude', 'oil_production_bpd', 'equipment_health_score'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ Missing columns: {missing_cols}")
            return False
        else:
            print("   ✅ All required columns present")
        
        # Check for NaN values in critical columns
        print("2. Checking data quality...")
        critical_cols = ['latitude', 'longitude', 'equipment_health_score']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    print(f"   ⚠️  {col} has {null_count} null values")
                else:
                    print(f"   ✅ {col} is clean")
        
        # Test health status creation
        print("3. Testing health status calculation...")
        if 'equipment_health_score' in df.columns:
            df['health_status'] = df['equipment_health_score'].apply(
                lambda x: 'Excellent' if pd.notna(x) and x >= 0.8 
                else 'Good' if pd.notna(x) and x >= 0.6 
                else 'Fair' if pd.notna(x) and x >= 0.4 
                else 'Poor'
            )
            health_counts = df['health_status'].value_counts()
            print(f"   ✅ Health status distribution: {dict(health_counts)}")
        
        # Test facility map creation with small sample
        print("4. Testing facility map creation...")
        test_df = df.head(5).copy()
        
        # Ensure no NaN values in test data
        test_df = test_df.fillna({
            'oil_production_bpd': 50,
            'equipment_health_score': 0.7,
            'utilization_rate': 0.8
        })
        
        fig = create_facility_map(test_df)
        print("   ✅ Facility map created successfully")
        
        # Test equipment health dashboard
        print("5. Testing equipment health dashboard...")
        from scripts.geospatial_dashboard import create_equipment_health_dashboard
        health_fig = create_equipment_health_dashboard(test_df)
        print("   ✅ Equipment health dashboard created successfully")
        
        print("\n✅ All geospatial dashboard tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in geospatial dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_data():
    """Test sample data generation"""
    print("\n🧪 Testing Sample Data Generation...")
    
    try:
        sample_df = create_sample_geospatial_data()
        print(f"   ✅ Sample data created: {len(sample_df)} rows")
        print(f"   ✅ Columns: {list(sample_df.columns)}")
        
        # Check data types and ranges
        if 'latitude' in sample_df.columns:
            lat_range = (sample_df['latitude'].min(), sample_df['latitude'].max())
            print(f"   ✅ Latitude range: {lat_range}")
        
        if 'equipment_health_score' in sample_df.columns:
            health_range = (sample_df['equipment_health_score'].min(), sample_df['equipment_health_score'].max())
            print(f"   ✅ Health score range: {health_range}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in sample data: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Oil & Gas Dashboard Tests\n")
    
    success = True
    success &= test_sample_data()
    success &= test_geospatial_dashboard()
    
    if success:
        print("\n🎉 All tests passed! Dashboard should work properly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)