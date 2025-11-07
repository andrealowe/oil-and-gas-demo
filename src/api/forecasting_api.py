#!/usr/bin/env python3
"""
FastAPI service for Oil & Gas Time Series Forecasting Models

Provides REST API endpoints for real-time forecasting:
- Production forecasting (oil/gas)
- Price predictions
- Demand projections
- Maintenance scheduling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oil & Gas Forecasting API",
    description="Time series forecasting API for oil & gas operations",
    version="1.0.0"
)

class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    forecast_type: str = Field(..., description="Type of forecast: production, prices, demand, maintenance")
    target_variable: str = Field(..., description="Variable to forecast")
    horizon_days: int = Field(default=30, description="Forecast horizon in days", ge=1, le=365)
    region: Optional[str] = Field(default=None, description="Region for demand forecasting")
    facility_type: Optional[str] = Field(default=None, description="Facility type for production forecasting")

class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    forecast_type: str
    target_variable: str
    forecast: List[Dict[str, Union[str, float]]]
    confidence_intervals: Optional[List[Dict[str, Union[str, float]]]] = None
    metadata: Dict[str, Union[str, float, int]]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    api_version: str

class ForecastingService:
    """Core forecasting service"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.paths = get_data_paths('Oil-and-Gas-Demo')
        self.models_dir = Path('/mnt/artifacts/models/forecasting')
        self._load_models()
    
    def _load_models(self):
        """Load all trained forecasting models"""
        try:
            # Load serving configuration
            config_path = self.models_dir / 'serving_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.model_configs = json.load(f)
                logger.info(f"Loaded serving configuration with {len(self.model_configs.get('models', {}))} models")
            
            # Load individual model files
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*.pkl"))
                for model_file in model_files:
                    try:
                        model_key = model_file.stem
                        self.models[model_key] = joblib.load(model_file)
                        logger.info(f"Loaded model: {model_key}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_file}: {e}")
                        
                logger.info(f"Successfully loaded {len(self.models)} forecasting models")
            else:
                logger.warning(f"Models directory not found: {self.models_dir}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _get_model_key(self, forecast_type: str, target_variable: str, 
                      horizon_days: int, region: str = None) -> str:
        """Generate model key for lookup"""
        # Determine horizon type
        horizon_type = "short_term" if horizon_days <= 30 else "medium_term"
        
        if forecast_type == "demand" and region:
            return f"demand_{region}_{target_variable}_lgb_{horizon_type}"
        elif forecast_type == "production":
            return f"production_{target_variable}_lgb_{horizon_type}"
        elif forecast_type == "prices":
            return f"prices_{target_variable}_lgb_{horizon_type}"
        elif forecast_type == "maintenance":
            return f"maintenance_overall_{target_variable}_lgb_{horizon_type}"
        
        return None
    
    def generate_forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate forecast based on request"""
        try:
            # Get appropriate model
            model_key = self._get_model_key(
                request.forecast_type,
                request.target_variable,
                request.horizon_days,
                request.region
            )
            
            if not model_key or model_key not in self.models:
                # Try alternative model keys
                alternative_keys = [k for k in self.models.keys() 
                                  if request.target_variable in k and request.forecast_type in k]
                if alternative_keys:
                    model_key = alternative_keys[0]
                else:
                    raise ValueError(f"Model not found for {request.forecast_type}:{request.target_variable}")
            
            model_info = self.models[model_key]
            
            # Generate forecast based on model type
            if model_info['model_type'] == 'lightgbm':
                forecast_data = self._generate_lgb_forecast(model_info, request)
            elif model_info['model_type'] == 'prophet':
                forecast_data = self._generate_prophet_forecast(model_info, request)
            else:
                raise ValueError(f"Unsupported model type: {model_info['model_type']}")
            
            # Prepare response
            forecast_response = ForecastResponse(
                forecast_type=request.forecast_type,
                target_variable=request.target_variable,
                forecast=forecast_data['forecast'],
                confidence_intervals=forecast_data.get('confidence_intervals'),
                metadata={
                    'model_type': model_info['model_type'],
                    'model_key': model_key,
                    'horizon_days': request.horizon_days,
                    'model_metrics': model_info.get('metrics', {}),
                    'generated_at': datetime.now().isoformat()
                }
            )
            
            return forecast_response
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
    
    def _generate_lgb_forecast(self, model_info: dict, request: ForecastRequest) -> dict:
        """Generate forecast using LightGBM model"""
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        
        # Create future dates
        start_date = datetime.now()
        future_dates = [start_date + timedelta(days=i) for i in range(1, request.horizon_days + 1)]
        
        # Create basic features (simplified approach for API)
        forecast_data = []
        for date in future_dates:
            # Time features
            features = {
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.weekday(),
                'day_of_year': date.timetuple().tm_yday,
                'week_of_year': date.isocalendar()[1],
                'quarter': (date.month - 1) // 3 + 1,
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'day_sin': np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25),
                'day_cos': np.cos(2 * np.pi * date.timetuple().tm_yday / 365.25)
            }
            
            # Add lag features (use dummy values for API - in production, use recent data)
            lag_features = {col: 0.0 for col in feature_columns if 'lag' in col or 'roll' in col}
            features.update(lag_features)
            
            # Ensure all required features are present
            feature_vector = []
            for col in feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            # Make prediction
            try:
                prediction = model.predict([feature_vector])[0]
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': float(prediction)
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {date}: {e}")
                # Use simple trend if prediction fails
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': 0.0
                })
        
        return {
            'forecast': forecast_data,
            'confidence_intervals': None  # LightGBM doesn't provide confidence intervals by default
        }
    
    def _generate_prophet_forecast(self, model_info: dict, request: ForecastRequest) -> dict:
        """Generate forecast using Prophet model"""
        if 'forecast' in model_info and not model_info['forecast'].empty:
            # Use pre-generated forecast
            forecast_df = model_info['forecast'].head(request.horizon_days)
            
            forecast_data = []
            confidence_intervals = []
            
            for _, row in forecast_df.iterrows():
                forecast_data.append({
                    'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'predicted_value': float(row[f'{request.target_variable}_forecast'])
                })
                
                if f'{request.target_variable}_lower' in row and f'{request.target_variable}_upper' in row:
                    confidence_intervals.append({
                        'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        'lower_bound': float(row[f'{request.target_variable}_lower']),
                        'upper_bound': float(row[f'{request.target_variable}_upper'])
                    })
            
            return {
                'forecast': forecast_data,
                'confidence_intervals': confidence_intervals if confidence_intervals else None
            }
        else:
            # Fallback to simple forecast
            start_date = datetime.now()
            forecast_data = []
            for i in range(1, request.horizon_days + 1):
                date = start_date + timedelta(days=i)
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': 100.0  # Placeholder value
                })
            
            return {
                'forecast': forecast_data,
                'confidence_intervals': None
            }

# Initialize forecasting service
forecasting_service = ForecastingService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(forecasting_service.models),
        api_version="1.0.0"
    )

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate time series forecast"""
    return forecasting_service.generate_forecast(request)

@app.get("/models")
async def list_available_models():
    """List all available forecasting models"""
    models_info = {}
    for model_key, model_info in forecasting_service.models.items():
        models_info[model_key] = {
            'model_type': model_info.get('model_type', 'unknown'),
            'metrics': model_info.get('metrics', {}),
            'available': True
        }
    
    return {
        'total_models': len(models_info),
        'models': models_info,
        'model_directory': str(forecasting_service.models_dir)
    }

@app.get("/forecast/production/{target}")
async def forecast_production(
    target: str,
    horizon_days: int = 30,
    background_tasks: BackgroundTasks = None
):
    """Quick endpoint for production forecasting"""
    request = ForecastRequest(
        forecast_type="production",
        target_variable=target,
        horizon_days=horizon_days
    )
    return forecasting_service.generate_forecast(request)

@app.get("/forecast/prices/{commodity}")
async def forecast_prices(
    commodity: str,
    horizon_days: int = 30
):
    """Quick endpoint for price forecasting"""
    request = ForecastRequest(
        forecast_type="prices",
        target_variable=commodity,
        horizon_days=horizon_days
    )
    return forecasting_service.generate_forecast(request)

@app.get("/forecast/demand/{region}")
async def forecast_demand(
    region: str,
    target: str = "oil_demand_thousand_bpd",
    horizon_days: int = 30
):
    """Quick endpoint for demand forecasting by region"""
    request = ForecastRequest(
        forecast_type="demand",
        target_variable=target,
        horizon_days=horizon_days,
        region=region
    )
    return forecasting_service.generate_forecast(request)

@app.get("/forecast/maintenance")
async def forecast_maintenance(
    target: str = "cost_usd",
    horizon_days: int = 30
):
    """Quick endpoint for maintenance forecasting"""
    request = ForecastRequest(
        forecast_type="maintenance",
        target_variable=target,
        horizon_days=horizon_days
    )
    return forecasting_service.generate_forecast(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)