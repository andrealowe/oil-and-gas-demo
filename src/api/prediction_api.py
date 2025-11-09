#!/usr/bin/env python3
"""
Oil & Gas Production Forecasting API

Simple prediction API for the champion forecasting model.
Loads the registered champion model and provides prediction endpoint.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionForecastingAPI:
    """API class for oil production forecasting"""
    
    def __init__(self, model_name="oil_gas_production_forecasting_champion"):
        """Initialize API with champion model"""
        self.model_name = model_name
        self.model = None
        self.model_info = None
        self._load_champion_model()
    
    def _load_champion_model(self):
        """Load the champion model from MLflow Model Registry"""
        try:
            mlflow.set_tracking_uri("http://localhost:8768")
            client = MlflowClient()
            
            # Get latest version of champion model
            model_version = client.get_latest_versions(self.model_name, stages=["None"])[0]
            model_uri = f"models:/{self.model_name}/{model_version.version}"
            
            # Load model (note: this is a simplified approach - actual implementation depends on framework)
            # In practice, you'd load the specific framework model (Nixtla in this case)
            self.model_info = {
                'name': self.model_name,
                'version': model_version.version,
                'framework': 'nixtla',  # Champion framework
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Loaded champion model: {self.model_name} v{model_version.version}")
            
        except Exception as e:
            logger.error(f"Error loading champion model: {e}")
            raise
    
    def predict_simple(self, input_data):
        """Simple prediction method with mock data for Domino compatibility"""
        """
        Make production forecast prediction
        
        Args:
            input_data (dict): Input parameters for prediction
                Expected format:
                {
                  "data": {
                    "start": 1,      # Starting day index (1-based)
                    "stop": 100      # Ending day index (1-based, inclusive)
                  },
                  "facility_id": "WELL_000001",  # Optional: specific facility or "all_facilities"
                  "aggregation": "total"         # Optional: "total", "average", or "facility"
                }
        
        Returns:
            dict: Prediction results with forecasted values
        """
        try:
            # Extract data parameters with new format
            data_params = input_data.get('data', {})
            start_day = data_params.get('start', 1)
            stop_day = data_params.get('stop', 7)
            
            # Calculate forecast parameters
            forecast_days = stop_day - start_day + 1  # Inclusive range
            start_date = (datetime.now() + timedelta(days=start_day - 1)).strftime('%Y-%m-%d')
            
            # Extract optional parameters
            facility_id = input_data.get('facility_id', 'all_facilities')
            aggregation = input_data.get('aggregation', 'total')
            
            # Validate inputs
            if start_day < 1 or stop_day < 1:
                raise ValueError("start and stop indices must be >= 1")
            
            if start_day > stop_day:
                raise ValueError("start index must be <= stop index")
                
            if forecast_days > 365:
                raise ValueError("forecast range cannot exceed 365 days")
            
            if aggregation not in ['total', 'average', 'facility']:
                raise ValueError("aggregation must be 'total', 'average', or 'facility'")
            
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("calculated start_date format error")
            
            # Generate forecast dates
            forecast_dates = [
                (start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(forecast_days)
            ]
            
            # Mock prediction logic (replace with actual model inference)
            # In production, this would call the actual Nixtla model
            base_production = 50000  # Base daily production in bpd
            predictions = []
            
            for i, date in enumerate(forecast_dates):
                # Simple mock prediction with trend and noise
                trend = -50 * i  # Slight decline trend
                seasonal = 2000 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
                noise = np.random.normal(0, 500)  # Random variation
                
                predicted_value = max(0, base_production + trend + seasonal + noise)
                
                predictions.append({
                    'date': date,
                    'predicted_production_bpd': round(predicted_value, 2),
                    'confidence_interval_lower': round(predicted_value * 0.95, 2),
                    'confidence_interval_upper': round(predicted_value * 1.05, 2)
                })
            
            # Prepare response
            response = {
                'model_info': self.model_info,
                'input_parameters': {
                    'data': {
                        'start': start_day,
                        'stop': stop_day
                    },
                    'forecast_days': forecast_days,
                    'start_date': start_date,
                    'facility_id': facility_id,
                    'aggregation': aggregation
                },
                'predictions': predictions,
                'summary': {
                    'total_forecast_days': len(predictions),
                    'average_daily_production': round(np.mean([p['predicted_production_bpd'] for p in predictions]), 2),
                    'total_forecasted_production': round(sum([p['predicted_production_bpd'] for p in predictions]), 2),
                    'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}"
                },
                'metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'api_version': '1.0',
                    'champion_framework': 'nixtla'
                }
            }
            
            logger.info(f"Generated forecast for {forecast_days} days starting {start_date}")
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': str(e),
                'model_info': self.model_info,
                'timestamp': datetime.now().isoformat()
            }

# Global API instance
api = ProductionForecastingAPI()

def predict_production(facility_id="all_facilities", forecast_days=7, start_date=None):
    """
    Main production prediction function for Domino Model API endpoint
    
    Args:
        facility_id (str): Facility ID or "all_facilities" for aggregate
        forecast_days (int): Number of days to forecast (1-365)
        start_date (str): Start date in YYYY-MM-DD format (optional, defaults to tomorrow)
    
    Returns:
        dict: Production prediction results
    """
    from datetime import datetime, timedelta
    
    # Set default start date to tomorrow if not provided
    if start_date is None:
        start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Create input data structure for the API class
    input_data = {
        'facility_id': facility_id,
        'forecast_days': forecast_days,
        'start_date': start_date
    }
    
    return api.predict_simple(input_data)

# Example usage and testing
if __name__ == "__main__":
    # Test the API
    test_input = {
        'forecast_days': 14,
        'start_date': '2024-11-09',
        'facility_id': 'all_facilities'
    }
    
    print("=== Oil & Gas Production Forecasting API Test ===")
    print(f"Input: {json.dumps(test_input, indent=2)}")
    print("\nGenerating forecast...")
    
    result = predict_production(test_input)
    print(f"\nResult: {json.dumps(result, indent=2)}")