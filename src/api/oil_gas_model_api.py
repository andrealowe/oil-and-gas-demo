#!/usr/bin/env python3
"""
Oil & Gas Geospatial Models API
FastAPI service for serving ML models for real-time predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import logging
from datetime import datetime
import uvicorn

# Add scripts directory to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Oil & Gas Geospatial Models API",
    description="REST API for oil and gas facility predictions",
    version="1.0.0"
)

# Get data paths
paths = get_data_paths("Oil-and-Gas-Demo")
MODELS_DIR = paths['artifacts_path'] / "models"

class FacilityInput(BaseModel):
    """Input schema for facility predictions"""
    
    # Location and identification
    latitude: float = Field(..., description="Facility latitude")
    longitude: float = Field(..., description="Facility longitude")
    facility_type: str = Field(..., description="Type of facility (oil_well, refinery, etc.)")
    region: str = Field(..., description="Geographic region")
    status: str = Field("active", description="Facility status")
    
    # Production metrics
    oil_production_bpd: Optional[float] = Field(0.0, description="Oil production barrels per day")
    gas_production_mcfd: Optional[float] = Field(0.0, description="Gas production thousand cubic feet per day")
    
    # Technical parameters
    well_depth_ft: Optional[float] = Field(5000.0, description="Well depth in feet")
    well_age_years: Optional[float] = Field(10.0, description="Well age in years")
    vibration_level_mm_s: Optional[float] = Field(2.5, description="Vibration level mm/s")
    temperature_celsius: Optional[float] = Field(25.0, description="Temperature in Celsius")
    pressure_psi: Optional[float] = Field(1000.0, description="Pressure in PSI")
    h2s_concentration_ppm: Optional[float] = Field(0.0, description="H2S concentration ppm")
    co2_concentration_ppm: Optional[float] = Field(500.0, description="CO2 concentration ppm")
    noise_level_db: Optional[float] = Field(70.0, description="Noise level in decibels")
    
    # Operational metrics
    capacity_bpd: Optional[float] = Field(10000.0, description="Capacity barrels per day")
    current_throughput_bpd: Optional[float] = Field(8000.0, description="Current throughput BPD")
    utilization_rate: Optional[float] = Field(0.8, description="Utilization rate 0-1")
    days_since_maintenance: Optional[int] = Field(30, description="Days since last maintenance")
    
    # Environmental metrics
    energy_consumption_mwh: Optional[float] = Field(100.0, description="Energy consumption MWh")
    co2_emissions_tons_day: Optional[float] = Field(50.0, description="CO2 emissions tons/day")
    water_usage_gallons_day: Optional[float] = Field(10000.0, description="Water usage gallons/day")

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    
    equipment_health_score: Optional[float] = Field(None, description="Equipment health 0-1")
    production_efficiency_class: Optional[str] = Field(None, description="Low/Medium/High")
    environmental_risk_class: Optional[str] = Field(None, description="Low/Medium/High")
    
    # Confidence scores
    equipment_health_confidence: Optional[float] = Field(None, description="Prediction confidence")
    production_efficiency_confidence: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    environmental_risk_confidence: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    
    # Metadata
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_versions: Dict[str, str] = Field(default_factory=dict)

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    facilities: List[FacilityInput] = Field(..., description="List of facilities to predict")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Union[int, float]] = Field(..., description="Batch summary statistics")

class ModelManager:
    """Manages loading and prediction with ML models"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Load model metadata
            metadata_path = MODELS_DIR / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {self.model_metadata}")
            
            # Load individual models
            model_files = {
                'equipment_health': MODELS_DIR / "equipment_health_model.pkl",
                'production_efficiency': MODELS_DIR / "production_efficiency_model.pkl",
                'environmental_risk': MODELS_DIR / "environmental_risk_model.pkl"
            }
            
            for model_name, model_path in model_files.items():
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded {model_name} model from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load {model_name} model: {e}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _prepare_features(self, input_data: FacilityInput) -> pd.DataFrame:
        """Convert input to feature DataFrame"""
        
        # Convert to dictionary
        data_dict = input_data.model_dump()
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])
        
        # Feature engineering (same as training)
        df['maintenance_overdue'] = (df['days_since_maintenance'] > 90).astype(int)
        df['latitude_bin'] = pd.cut(df['latitude'], bins=10, labels=False).fillna(0).astype(int)
        df['longitude_bin'] = pd.cut(df['longitude'], bins=10, labels=False).fillna(0).astype(int)
        df['production_ratio'] = df['oil_production_bpd'] / (df['gas_production_mcfd'] + 1)
        df['utilization_efficiency'] = df['utilization_rate'] * 0.8  # Default equipment health
        
        # Environmental stress
        df['environmental_stress'] = (
            df['h2s_concentration_ppm'] * 0.3 +
            df['co2_concentration_ppm'] * 0.2 +
            df['noise_level_db'] * 0.1 +
            df['vibration_level_mm_s'] * 0.4
        )
        
        # Age and depth interaction
        df['age_depth_interaction'] = df['well_age_years'] * df['well_depth_ft'] / 1000
        
        return df
    
    def predict_equipment_health(self, features: pd.DataFrame) -> Dict:
        """Predict equipment health score"""
        if 'equipment_health' not in self.models:
            raise ValueError("Equipment health model not available")
        
        try:
            model = self.models['equipment_health']
            prediction = model.predict(features)[0]
            
            # Ensure prediction is within valid range
            prediction = max(0.0, min(1.0, prediction))
            
            return {
                'score': float(prediction),
                'confidence': 0.85  # Placeholder confidence
            }
        except Exception as e:
            logger.error(f"Equipment health prediction error: {e}")
            raise
    
    def predict_production_efficiency(self, features: pd.DataFrame) -> Dict:
        """Predict production efficiency class"""
        if 'production_efficiency' not in self.models:
            raise ValueError("Production efficiency model not available")
        
        try:
            model = self.models['production_efficiency']
            prediction = model.predict(features)[0]
            
            # Get probability scores if available
            try:
                probabilities = model.predict_proba(features)[0]
                classes = model.classes_
                class_probs = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            except:
                class_probs = {prediction: 1.0}
            
            return {
                'class': str(prediction),
                'probabilities': class_probs
            }
        except Exception as e:
            logger.error(f"Production efficiency prediction error: {e}")
            raise
    
    def predict_environmental_risk(self, features: pd.DataFrame) -> Dict:
        """Predict environmental risk class"""
        if 'environmental_risk' not in self.models:
            raise ValueError("Environmental risk model not available")
        
        try:
            model = self.models['environmental_risk']
            prediction = model.predict(features)[0]
            
            # Get probability scores if available
            try:
                probabilities = model.predict_proba(features)[0]
                classes = model.classes_
                class_probs = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            except:
                class_probs = {prediction: 1.0}
            
            return {
                'class': str(prediction),
                'probabilities': class_probs
            }
        except Exception as e:
            logger.error(f"Environmental risk prediction error: {e}")
            raise
    
    def predict_all(self, input_data: FacilityInput) -> PredictionResponse:
        """Run all predictions for a facility"""
        
        # Prepare features
        features = self._prepare_features(input_data)
        
        response = PredictionResponse()
        
        # Equipment health prediction
        try:
            eq_health = self.predict_equipment_health(features)
            response.equipment_health_score = eq_health['score']
            response.equipment_health_confidence = eq_health['confidence']
        except Exception as e:
            logger.warning(f"Equipment health prediction failed: {e}")
        
        # Production efficiency prediction
        try:
            prod_eff = self.predict_production_efficiency(features)
            response.production_efficiency_class = prod_eff['class']
            response.production_efficiency_confidence = prod_eff['probabilities']
        except Exception as e:
            logger.warning(f"Production efficiency prediction failed: {e}")
        
        # Environmental risk prediction
        try:
            env_risk = self.predict_environmental_risk(features)
            response.environmental_risk_class = env_risk['class']
            response.environmental_risk_confidence = env_risk['probabilities']
        except Exception as e:
            logger.warning(f"Environmental risk prediction failed: {e}")
        
        # Add model version info
        response.model_versions = {
            name: "v1.0" for name in self.models.keys()
        }
        
        return response

# Initialize model manager
model_manager = ModelManager()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Oil & Gas Geospatial Models API",
        "version": "1.0.0",
        "models_available": list(model_manager.models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "available_models": list(model_manager.models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_facility(input_data: FacilityInput):
    """Predict all metrics for a single facility"""
    try:
        logger.info(f"Received prediction request for facility type: {input_data.facility_type}")
        prediction = model_manager.predict_all(input_data)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchPredictionInput):
    """Predict metrics for multiple facilities"""
    try:
        logger.info(f"Received batch prediction request for {len(batch_input.facilities)} facilities")
        
        predictions = []
        for facility in batch_input.facilities:
            prediction = model_manager.predict_all(facility)
            predictions.append(prediction)
        
        # Calculate summary statistics
        health_scores = [p.equipment_health_score for p in predictions if p.equipment_health_score is not None]
        
        summary = {
            "total_facilities": len(batch_input.facilities),
            "successful_predictions": len(predictions),
            "avg_equipment_health": float(np.mean(health_scores)) if health_scores else None,
            "high_risk_facilities": sum(1 for p in predictions if p.environmental_risk_class == "High"),
            "low_efficiency_facilities": sum(1 for p in predictions if p.production_efficiency_class == "Low")
        }
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "models": list(model_manager.models.keys()),
        "metadata": model_manager.model_metadata,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/equipment-health")
async def predict_equipment_health_only(input_data: FacilityInput):
    """Predict only equipment health score"""
    try:
        features = model_manager._prepare_features(input_data)
        result = model_manager.predict_equipment_health(features)
        return {
            "equipment_health_score": result['score'],
            "confidence": result['confidence'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Equipment health prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/production-efficiency")
async def predict_production_efficiency_only(input_data: FacilityInput):
    """Predict only production efficiency class"""
    try:
        features = model_manager._prepare_features(input_data)
        result = model_manager.predict_production_efficiency(features)
        return {
            "production_efficiency_class": result['class'],
            "probabilities": result['probabilities'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Production efficiency prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/environmental-risk")
async def predict_environmental_risk_only(input_data: FacilityInput):
    """Predict only environmental risk class"""
    try:
        features = model_manager._prepare_features(input_data)
        result = model_manager.predict_environmental_risk(features)
        return {
            "environmental_risk_class": result['class'],
            "probabilities": result['probabilities'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Environmental risk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "oil_gas_model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )