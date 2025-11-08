#!/usr/bin/env python3
"""
Oil & Gas Price Prediction API

Mock prediction API for oil and gas price forecasting.
Provides price predictions based on market indicators and trends.
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

class PricePredictionAPI:
    """API class for oil and gas price forecasting"""
    
    def __init__(self, model_name="oil_gas_price_prediction_champion"):
        """Initialize API with price prediction model"""
        self.model_name = model_name
        self.model = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self):
        """Load the price prediction model (mock implementation)"""
        try:
            # Mock model info (in production, would load from MLflow)
            self.model_info = {
                'name': self.model_name,
                'version': '1.0',
                'framework': 'ensemble_price_model',
                'features': ['historical_prices', 'market_indicators', 'supply_demand', 'geopolitical_factors'],
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Loaded price prediction model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading price prediction model: {e}")
            raise
    
    def predict(self, input_data):
        """
        Make price forecast prediction
        
        Args:
            input_data (dict): Input parameters for prediction
                - forecast_days (int): Number of days to forecast
                - start_date (str): Start date for forecast (YYYY-MM-DD)
                - commodity (str): 'crude_oil' or 'natural_gas'
                - market_indicators (dict): Optional market context
        
        Returns:
            dict: Price prediction results
        """
        try:
            # Extract input parameters
            forecast_days = input_data.get('forecast_days', 30)
            start_date = input_data.get('start_date', datetime.now().strftime('%Y-%m-%d'))
            commodity = input_data.get('commodity', 'crude_oil')
            market_indicators = input_data.get('market_indicators', {})
            
            # Validate inputs
            if forecast_days <= 0 or forecast_days > 365:
                raise ValueError("forecast_days must be between 1 and 365")
            
            if commodity not in ['crude_oil', 'natural_gas']:
                raise ValueError("commodity must be 'crude_oil' or 'natural_gas'")
            
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("start_date must be in YYYY-MM-DD format")
            
            # Generate forecast dates
            forecast_dates = [
                (start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(forecast_days)
            ]
            
            # Mock prediction logic based on commodity type
            predictions = []
            
            if commodity == 'crude_oil':
                base_price = 75.0  # Base WTI crude price in USD/barrel
                volatility = 2.5   # Daily volatility
            else:  # natural_gas
                base_price = 3.2   # Base natural gas price in USD/MMBtu
                volatility = 0.15  # Daily volatility
            
            # Apply market indicators if provided
            market_adjustment = 1.0
            if market_indicators.get('supply_disruption', False):
                market_adjustment += 0.1  # 10% increase for supply disruption
            if market_indicators.get('demand_surge', False):
                market_adjustment += 0.05  # 5% increase for demand surge
            if market_indicators.get('economic_downturn', False):
                market_adjustment -= 0.08  # 8% decrease for economic concerns
            
            current_price = base_price * market_adjustment
            
            for i, date in enumerate(forecast_dates):
                # Generate price with trend, seasonality, and volatility
                
                # Long-term trend (slight upward bias)
                trend = 0.02 * i  # Small upward trend
                
                # Seasonal patterns (stronger in winter for natural gas)
                if commodity == 'natural_gas':
                    # Winter heating demand (higher prices Dec-Feb)
                    month = (start_dt + timedelta(days=i)).month
                    if month in [12, 1, 2]:
                        seasonal = 0.15  # 15% winter premium
                    elif month in [6, 7, 8]:
                        seasonal = -0.1  # 10% summer discount
                    else:
                        seasonal = 0
                else:
                    # Oil has less seasonal variation
                    seasonal = 0.02 * np.sin(2 * np.pi * i / 365)  # Mild annual cycle
                
                # Market volatility (random walk with mean reversion)
                volatility_factor = np.random.normal(0, volatility)
                mean_reversion = -0.1 * (current_price - base_price) / base_price
                
                # Calculate predicted price
                price_change = trend + seasonal + volatility_factor + mean_reversion
                current_price = max(0.1, current_price * (1 + price_change))
                
                # Calculate confidence intervals
                confidence_width = volatility * 1.96  # 95% confidence interval
                lower_bound = max(0.1, current_price * (1 - confidence_width))
                upper_bound = current_price * (1 + confidence_width)
                
                predictions.append({
                    'date': date,
                    'predicted_price': round(current_price, 2),
                    'commodity': commodity,
                    'unit': 'USD/barrel' if commodity == 'crude_oil' else 'USD/MMBtu',
                    'confidence_interval_lower': round(lower_bound, 2),
                    'confidence_interval_upper': round(upper_bound, 2),
                    'volatility': round(volatility, 3)
                })
            
            # Calculate summary statistics
            prices = [p['predicted_price'] for p in predictions]
            price_change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
            
            # Prepare response
            response = {
                'model_info': self.model_info,
                'input_parameters': {
                    'forecast_days': forecast_days,
                    'start_date': start_date,
                    'commodity': commodity,
                    'market_indicators': market_indicators
                },
                'predictions': predictions,
                'summary': {
                    'commodity': commodity,
                    'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}",
                    'starting_price': round(prices[0], 2),
                    'ending_price': round(prices[-1], 2),
                    'price_change_percent': round(price_change_pct, 2),
                    'average_price': round(np.mean(prices), 2),
                    'max_price': round(max(prices), 2),
                    'min_price': round(min(prices), 2),
                    'price_volatility': round(np.std(prices), 2)
                },
                'market_factors': {
                    'market_adjustment_applied': round((market_adjustment - 1) * 100, 1),
                    'risk_factors': list(market_indicators.keys()) if market_indicators else [],
                    'price_drivers': ['supply_demand_balance', 'geopolitical_events', 'economic_indicators', 'seasonal_patterns']
                },
                'metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'api_version': '1.0',
                    'model_type': 'ensemble_price_forecasting'
                }
            }
            
            logger.info(f"Generated price forecast for {commodity} over {forecast_days} days")
            return response
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {
                'error': str(e),
                'model_info': self.model_info,
                'timestamp': datetime.now().isoformat()
            }

# Global API instance
api = PricePredictionAPI()

def predict_prices(input_data):
    """
    Main price prediction function for API endpoint
    
    Args:
        input_data (dict): Input parameters for price prediction
    
    Returns:
        dict: Price prediction results
    """
    return api.predict(input_data)

# Example usage and testing
if __name__ == "__main__":
    # Test the API with crude oil prediction
    test_input_oil = {
        'forecast_days': 30,
        'start_date': '2024-11-09',
        'commodity': 'crude_oil',
        'market_indicators': {
            'supply_disruption': True,
            'demand_surge': False,
            'economic_downturn': False
        }
    }
    
    print("=== Oil & Gas Price Prediction API Test ===")
    print(f"Input (Crude Oil): {json.dumps(test_input_oil, indent=2)}")
    print("\nGenerating crude oil price forecast...")
    
    result_oil = predict_prices(test_input_oil)
    print(f"\nCrude Oil Summary: {json.dumps(result_oil['summary'], indent=2)}")
    
    # Test with natural gas
    test_input_gas = {
        'forecast_days': 14,
        'start_date': '2024-12-01',  # Winter period
        'commodity': 'natural_gas',
        'market_indicators': {
            'supply_disruption': False,
            'demand_surge': True,  # Winter heating demand
            'economic_downturn': False
        }
    }
    
    print(f"\nInput (Natural Gas): {json.dumps(test_input_gas, indent=2)}")
    print("\nGenerating natural gas price forecast...")
    
    result_gas = predict_prices(test_input_gas)
    print(f"\nNatural Gas Summary: {json.dumps(result_gas['summary'], indent=2)}")