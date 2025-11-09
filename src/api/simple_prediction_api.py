#!/usr/bin/env python3
"""
Simple Oil & Gas Production Prediction API for Domino Model Endpoints
Simplified version with mock predictions that works reliably
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_production(facility_id="all_facilities", forecast_days=7, start_date=None):
    """
    Production prediction function for Domino Model API
    
    Args:
        facility_id (str): Facility ID or "all_facilities" 
        forecast_days (int): Days to forecast (1-365)
        start_date (str): Start date YYYY-MM-DD (optional)
    
    Returns:
        dict: Production forecast results
    """
    try:
        # Validate inputs
        forecast_days = int(forecast_days)
        if forecast_days < 1 or forecast_days > 365:
            raise ValueError("forecast_days must be between 1 and 365")
        
        # Parse start date or use tomorrow
        if start_date is None:
            start_dt = datetime.now() + timedelta(days=1)
        else:
            start_dt = datetime.strptime(str(start_date), '%Y-%m-%d')
        
        # Generate forecast dates
        forecast_dates = [
            (start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
            for i in range(forecast_days)
        ]
        
        # Mock production prediction based on facility
        if facility_id == "all_facilities":
            base_production = 52000.0  # Total production for all facilities
            facility_name = "All Facilities"
        else:
            base_production = 35.0     # Individual facility production
            facility_name = f"Facility {facility_id}"
        
        # Generate predictions with realistic patterns
        predictions = []
        current_production = base_production
        
        for i, date in enumerate(forecast_dates):
            # Apply realistic factors
            # 1. Natural decline (small daily decrease)
            decline_factor = 0.9998  # 0.02% daily decline
            
            # 2. Seasonal variation
            day_of_year = (start_dt + timedelta(days=i)).timetuple().tm_yday
            seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * day_of_year / 365)  # ±5% seasonal
            
            # 3. Daily random variation
            random_factor = np.random.normal(1.0, 0.03)  # ±3% daily variation
            
            # Calculate prediction
            current_production *= decline_factor
            predicted_value = current_production * seasonal_factor * random_factor
            predicted_value = max(0, predicted_value)  # Ensure non-negative
            
            # Create prediction entry
            predictions.append({
                'date': date,
                'facility_id': facility_id,
                'predicted_production_bpd': round(predicted_value, 2),
                'confidence_lower': round(predicted_value * 0.9, 2),
                'confidence_upper': round(predicted_value * 1.1, 2)
            })
        
        # Calculate summary statistics
        values = [p['predicted_production_bpd'] for p in predictions]
        
        # Prepare response
        response = {
            'model_info': {
                'name': 'simple_production_predictor',
                'version': '1.0',
                'type': 'mock_forecast_model'
            },
            'input_parameters': {
                'facility_id': facility_id,
                'forecast_days': forecast_days,
                'start_date': start_date or start_dt.strftime('%Y-%m-%d')
            },
            'predictions': predictions,
            'summary': {
                'facility_name': facility_name,
                'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}",
                'total_days': len(predictions),
                'average_daily_production': round(np.mean(values), 2),
                'total_forecasted_production': round(sum(values), 2),
                'starting_production': round(values[0], 2),
                'ending_production': round(values[-1], 2),
                'production_trend': round(((values[-1] - values[0]) / values[0]) * 100, 2)
            },
            'metadata': {
                'prediction_timestamp': datetime.now().isoformat(),
                'api_version': '1.0'
            }
        }
        
        logger.info(f"Generated production forecast for {facility_id} over {forecast_days} days")
        return response
        
    except Exception as e:
        logger.error(f"Error in production prediction: {e}")
        return {
            'error': str(e),
            'input_parameters': {
                'facility_id': facility_id,
                'forecast_days': forecast_days,
                'start_date': start_date
            },
            'timestamp': datetime.now().isoformat()
        }

# Example usage for testing
if __name__ == "__main__":
    print("=== Simple Production Prediction API Test ===")
    
    # Test 1: Basic forecast
    print("\n1. Basic 7-day forecast for all facilities:")
    result1 = predict_production()
    print(f"Summary: {result1['summary']}")
    
    # Test 2: Specific facility
    print("\n2. 14-day forecast for specific facility:")
    result2 = predict_production(facility_id="WELL_000123", forecast_days=14)
    print(f"Summary: {result2['summary']}")
    
    # Test 3: With start date
    print("\n3. 30-day forecast starting from specific date:")
    result3 = predict_production(forecast_days=30, start_date="2024-12-01")
    print(f"Summary: {result3['summary']}")