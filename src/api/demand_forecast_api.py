#!/usr/bin/env python3
"""
Oil & Gas Demand Forecasting API

Mock prediction API for oil and gas demand forecasting.
Provides demand predictions based on economic indicators and seasonal patterns.
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

class DemandForecastAPI:
    """API class for oil and gas demand forecasting"""
    
    def __init__(self, model_name="oil_gas_demand_forecast_champion"):
        """Initialize API with demand forecasting model"""
        self.model_name = model_name
        self.model = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self):
        """Load the demand forecasting model (mock implementation)"""
        try:
            # Mock model info (in production, would load from MLflow)
            self.model_info = {
                'name': self.model_name,
                'version': '1.0',
                'framework': 'multivariate_demand_model',
                'features': ['economic_indicators', 'weather_patterns', 'industrial_activity', 'transportation_trends'],
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Loaded demand forecast model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading demand forecast model: {e}")
            raise
    
    def predict(self, input_data):
        """
        Make demand forecast prediction
        
        Args:
            input_data (dict): Input parameters for prediction
                Expected format:
                {
                  "data": {
                    "start": 1,      # Starting day index (1-based)
                    "stop": 100      # Ending day index (1-based, inclusive)
                  },
                  "region": "north_america",     # Optional: region or "global"
                  "commodity": "crude_oil",      # Optional: commodity type
                  "economic_factors": {}         # Optional: economic context
                }
        
        Returns:
            dict: Demand forecast results
        """
        try:
            # Extract data parameters with new format
            data_params = input_data.get('data', {})
            start_day = data_params.get('start', 1)
            stop_day = data_params.get('stop', 30)
            
            # Calculate forecast parameters
            forecast_days = stop_day - start_day + 1  # Inclusive range
            start_date = (datetime.now() + timedelta(days=start_day - 1)).strftime('%Y-%m-%d')
            
            # Extract optional parameters
            region = input_data.get('region', 'global')
            commodity = input_data.get('commodity', 'crude_oil')
            economic_factors = input_data.get('economic_factors', {})
            
            # Validate inputs
            if start_day < 1 or stop_day < 1:
                raise ValueError("start and stop indices must be >= 1")
            
            if start_day > stop_day:
                raise ValueError("start index must be <= stop index")
                
            if forecast_days > 365:
                raise ValueError("forecast range cannot exceed 365 days")
            
            valid_regions = ['north_america', 'europe', 'asia_pacific', 'middle_east', 'south_america', 'africa', 'global']
            if region not in valid_regions:
                raise ValueError(f"region must be one of {valid_regions}")
            
            valid_commodities = ['crude_oil', 'natural_gas', 'refined_products', 'gasoline', 'diesel', 'jet_fuel']
            if commodity not in valid_commodities:
                raise ValueError(f"commodity must be one of {valid_commodities}")
            
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("calculated start_date format error")
            
            # Generate forecast dates
            forecast_dates = [
                (start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(forecast_days)
            ]
            
            # Mock prediction logic based on commodity and region
            predictions = []
            
            # Base demand levels by commodity and region (millions of barrels/day or bcf/day)
            base_demand = {
                'crude_oil': {
                    'north_america': 20.5,
                    'europe': 12.8,
                    'asia_pacific': 35.2,
                    'global': 101.5
                },
                'natural_gas': {
                    'north_america': 85.4,
                    'europe': 45.2,
                    'asia_pacific': 88.7,
                    'global': 395.8
                },
                'refined_products': {
                    'north_america': 18.2,
                    'europe': 11.5,
                    'asia_pacific': 28.9,
                    'global': 85.6
                }
            }
            
            current_demand = base_demand[commodity][region]
            
            # Apply economic factors
            economic_adjustment = 1.0
            gdp_growth = economic_factors.get('gdp_growth_rate', 0.025)  # 2.5% default
            industrial_activity = economic_factors.get('industrial_activity_index', 1.0)  # Baseline
            fuel_efficiency_trend = economic_factors.get('fuel_efficiency_improvement', 0.01)  # 1% annual improvement
            
            # Economic impact on demand
            economic_adjustment += gdp_growth * 0.8  # Demand elasticity to GDP
            economic_adjustment *= industrial_activity
            economic_adjustment -= fuel_efficiency_trend / 365 * forecast_days  # Efficiency improvement
            
            current_demand *= economic_adjustment
            
            for i, date in enumerate(forecast_dates):
                forecast_dt = start_dt + timedelta(days=i)
                
                # Seasonal patterns
                month = forecast_dt.month
                day_of_year = forecast_dt.timetuple().tm_yday
                
                if commodity == 'natural_gas':
                    # Strong winter heating demand
                    seasonal_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 45) / 365)
                elif commodity == 'refined_products':
                    # Summer driving season and winter heating
                    seasonal_factor = 1.0 + 0.15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
                else:  # crude_oil
                    # Moderate seasonal variation
                    seasonal_factor = 1.0 + 0.08 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
                
                # Weekly patterns (lower demand on weekends for some products)
                weekday = forecast_dt.weekday()
                if commodity == 'refined_products' and weekday >= 5:  # Weekend
                    weekly_factor = 0.95
                else:
                    weekly_factor = 1.0
                
                # Long-term trend
                if commodity == 'natural_gas':
                    trend = 0.03 * i / 365  # 3% annual growth
                elif commodity == 'refined_products':
                    trend = -0.01 * i / 365  # 1% annual decline (efficiency)
                else:  # crude_oil
                    trend = 0.015 * i / 365  # 1.5% annual growth
                
                # Random variation
                volatility = {
                    'crude_oil': 0.02,
                    'natural_gas': 0.05,
                    'refined_products': 0.03
                }[commodity]
                
                random_factor = 1.0 + np.random.normal(0, volatility)
                
                # Calculate predicted demand
                predicted_demand = current_demand * seasonal_factor * weekly_factor * (1 + trend) * random_factor
                predicted_demand = max(0.1, predicted_demand)  # Ensure positive demand
                
                # Calculate confidence intervals
                confidence_width = volatility * 1.96  # 95% confidence interval
                lower_bound = max(0.1, predicted_demand * (1 - confidence_width))
                upper_bound = predicted_demand * (1 + confidence_width)
                
                # Determine units based on commodity
                if commodity == 'natural_gas':
                    unit = 'bcf/day'  # Billion cubic feet per day
                else:
                    unit = 'mb/day'   # Million barrels per day
                
                predictions.append({
                    'date': date,
                    'predicted_demand': round(predicted_demand, 2),
                    'commodity': commodity,
                    'region': region,
                    'unit': unit,
                    'confidence_interval_lower': round(lower_bound, 2),
                    'confidence_interval_upper': round(upper_bound, 2),
                    'seasonal_factor': round(seasonal_factor, 3),
                    'week_day': weekday
                })
            
            # Calculate summary statistics
            demands = [p['predicted_demand'] for p in predictions]
            demand_change_pct = ((demands[-1] - demands[0]) / demands[0]) * 100
            
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
                    'region': region,
                    'commodity': commodity,
                    'economic_factors': economic_factors
                },
                'predictions': predictions,
                'summary': {
                    'commodity': commodity,
                    'region': region,
                    'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}",
                    'starting_demand': round(demands[0], 2),
                    'ending_demand': round(demands[-1], 2),
                    'demand_change_percent': round(demand_change_pct, 2),
                    'average_daily_demand': round(np.mean(demands), 2),
                    'peak_demand': round(max(demands), 2),
                    'minimum_demand': round(min(demands), 2),
                    'total_forecast_demand': round(sum(demands), 2),
                    'demand_volatility': round(np.std(demands), 2),
                    'unit': predictions[0]['unit']
                },
                'demand_drivers': {
                    'economic_adjustment_applied': round((economic_adjustment - 1) * 100, 1),
                    'key_factors': ['seasonal_patterns', 'economic_growth', 'industrial_activity', 'efficiency_trends'],
                    'economic_indicators': economic_factors if economic_factors else 'baseline_assumptions',
                    'seasonal_impact': 'high' if commodity == 'natural_gas' else 'moderate'
                },
                'metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'api_version': '1.0',
                    'model_type': 'multivariate_demand_forecasting'
                }
            }
            
            logger.info(f"Generated demand forecast for {commodity} in {region} over {forecast_days} days")
            return response
            
        except Exception as e:
            logger.error(f"Error in demand prediction: {e}")
            return {
                'error': str(e),
                'model_info': self.model_info,
                'timestamp': datetime.now().isoformat()
            }

# Global API instance
api = DemandForecastAPI()

def predict_demand(input_data):
    """
    Main demand prediction function for API endpoint
    
    Args:
        input_data (dict): Input parameters for demand prediction
    
    Returns:
        dict: Demand prediction results
    """
    return api.predict(input_data)

# Example usage and testing
if __name__ == "__main__":
    # Test the API with natural gas demand prediction
    test_input_gas = {
        'forecast_days': 30,
        'start_date': '2024-12-01',  # Winter period
        'region': 'north_america',
        'commodity': 'natural_gas',
        'economic_factors': {
            'gdp_growth_rate': 0.028,
            'industrial_activity_index': 1.05,
            'fuel_efficiency_improvement': 0.012
        }
    }
    
    print("=== Oil & Gas Demand Forecasting API Test ===")
    print(f"Input (Natural Gas): {json.dumps(test_input_gas, indent=2)}")
    print("\nGenerating natural gas demand forecast...")
    
    result_gas = predict_demand(test_input_gas)
    print(f"\nNatural Gas Summary: {json.dumps(result_gas['summary'], indent=2)}")
    
    # Test with crude oil demand
    test_input_oil = {
        'forecast_days': 90,
        'start_date': '2024-11-09',
        'region': 'asia_pacific',
        'commodity': 'crude_oil',
        'economic_factors': {
            'gdp_growth_rate': 0.045,  # Higher growth in Asia Pacific
            'industrial_activity_index': 1.1,
            'fuel_efficiency_improvement': 0.008
        }
    }
    
    print(f"\nInput (Crude Oil): {json.dumps(test_input_oil, indent=2)}")
    print("\nGenerating crude oil demand forecast...")
    
    result_oil = predict_demand(test_input_oil)
    print(f"\nCrude Oil Summary: {json.dumps(result_oil['summary'], indent=2)}")
    
    # Test with refined products
    test_input_refined = {
        'forecast_days': 60,
        'start_date': '2024-06-01',  # Summer driving season
        'region': 'global',
        'commodity': 'refined_products',
        'economic_factors': {
            'gdp_growth_rate': 0.025,
            'industrial_activity_index': 1.0,
            'fuel_efficiency_improvement': 0.015  # Higher efficiency improvement
        }
    }
    
    print(f"\nInput (Refined Products): {json.dumps(test_input_refined, indent=2)}")
    print("\nGenerating refined products demand forecast...")
    
    result_refined = predict_demand(test_input_refined)
    print(f"\nRefined Products Summary: {json.dumps(result_refined['summary'], indent=2)}")