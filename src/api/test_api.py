#!/usr/bin/env python3
"""
Test script for Oil & Gas Model API
"""

import requests
import json
from datetime import datetime

def test_api():
    """Test the Oil & Gas Model API"""
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_facility = {
        "latitude": 25.7617,
        "longitude": -80.1918,
        "facility_type": "oil_well",
        "region": "North America",
        "status": "active",
        "oil_production_bpd": 150.0,
        "gas_production_mcfd": 500.0,
        "well_depth_ft": 8000.0,
        "well_age_years": 5.0,
        "vibration_level_mm_s": 2.1,
        "temperature_celsius": 28.0,
        "pressure_psi": 1200.0,
        "h2s_concentration_ppm": 10.0,
        "co2_concentration_ppm": 400.0,
        "noise_level_db": 68.0,
        "capacity_bpd": 200.0,
        "current_throughput_bpd": 180.0,
        "utilization_rate": 0.9,
        "days_since_maintenance": 45,
        "energy_consumption_mwh": 120.0,
        "co2_emissions_tons_day": 25.0,
        "water_usage_gallons_day": 8000.0
    }
    
    print("Oil & Gas Model API Test")
    print("=" * 40)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint: {response.status_code}")
        print(f"Health: {response.json()}")
        print()
    except Exception as e:
        print(f"Health endpoint failed: {e}")
    
    # Test single prediction
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_facility,
            headers={"Content-Type": "application/json"}
        )
        print(f"Single prediction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Equipment Health: {result.get('equipment_health_score', 'N/A')}")
            print(f"Production Efficiency: {result.get('production_efficiency_class', 'N/A')}")
            print(f"Environmental Risk: {result.get('environmental_risk_class', 'N/A')}")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Single prediction failed: {e}")
    
    # Test batch prediction
    try:
        batch_data = {
            "facilities": [test_facility, test_facility]  # Test with 2 identical facilities
        }
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Batch prediction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Batch summary: {result.get('summary', {})}")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Batch prediction failed: {e}")
    
    # Test individual model endpoints
    endpoints = [
        "/predict/equipment-health",
        "/predict/production-efficiency", 
        "/predict/environmental-risk"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.post(
                f"{base_url}{endpoint}",
                json=test_facility,
                headers={"Content-Type": "application/json"}
            )
            print(f"{endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"Result: {response.json()}")
            else:
                print(f"Error: {response.text}")
            print()
        except Exception as e:
            print(f"{endpoint} failed: {e}")

if __name__ == "__main__":
    # Note: This assumes the API server is running on localhost:8000
    print("Make sure to start the API server first with:")
    print("python src/api/oil_gas_model_api.py")
    print()
    test_api()