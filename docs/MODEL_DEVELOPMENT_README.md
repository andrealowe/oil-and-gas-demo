# Oil & Gas Geospatial Models Development

## Overview

This project develops machine learning models for geospatial oil and gas facility operations, providing real-time predictions for dashboard integration. The models focus on equipment health, production optimization, and environmental risk assessment.

## Models Developed

### 1. Equipment Health Prediction Model
- **Type**: Regression
- **Target**: Equipment health score (0-1)
- **Performance**: R² = 0.8345
- **Use Case**: Real-time equipment monitoring and predictive maintenance

### 2. Production Efficiency Classification Model
- **Type**: Classification
- **Target**: Production efficiency class (Low/Medium/High)
- **Performance**: Accuracy = 100%
- **Use Case**: Production optimization and facility performance monitoring

### 3. Environmental Risk Scoring Model
- **Type**: Classification  
- **Target**: Environmental risk level (Low/Medium/High)
- **Performance**: Accuracy = 98.6%
- **Use Case**: Environmental compliance and safety monitoring

## Project Structure

```
/mnt/code/
├── src/
│   ├── models/
│   │   ├── oil_gas_geospatial_models.py      # Main model training script
│   │   └── model_evaluation_and_registry.py  # Model evaluation and MLflow registry
│   └── api/
│       ├── oil_gas_model_api.py               # FastAPI service for model serving
│       └── test_api.py                        # API testing script
├── notebooks/
│   └── oil_gas_model_analysis.ipynb          # Comprehensive model analysis notebook
├── config/
│   └── model_deployment_config.yaml          # Deployment configuration
├── requirements.txt                           # Python dependencies
└── docs/
    └── MODEL_DEVELOPMENT_README.md            # This documentation

/mnt/artifacts/
├── models/                                    # Trained model files
│   ├── equipment_health_model.pkl
│   ├── production_efficiency_model.pkl
│   ├── environmental_risk_model.pkl
│   └── model_metadata.json
└── reports/
    └── model_evaluation_report.json          # Detailed evaluation metrics
```

## Features Used

### Input Features
- **Location**: latitude, longitude, region
- **Facility Info**: facility_type, status, capacity_bpd
- **Production Metrics**: oil_production_bpd, gas_production_mcfd, utilization_rate
- **Technical Parameters**: well_depth_ft, well_age_years, pressure_psi, temperature_celsius
- **Environmental Factors**: h2s_concentration_ppm, co2_concentration_ppm, vibration_level_mm_s
- **Operational Data**: days_since_maintenance, energy_consumption_mwh

### Engineered Features
- **production_ratio**: Oil to gas production ratio
- **environmental_stress**: Composite environmental stress indicator
- **age_depth_interaction**: Well age and depth interaction
- **maintenance_overdue**: Binary flag for overdue maintenance
- **Geographic binning**: Latitude and longitude discretization

## Model Performance

| Model | Type | Primary Metric | Score |
|-------|------|---------------|-------|
| Equipment Health | Regression | R² Score | 0.8345 |
| Production Efficiency | Classification | Accuracy | 1.0000 |
| Environmental Risk | Classification | Accuracy | 0.9860 |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/models/oil_gas_geospatial_models.py
```

### 3. Evaluate Models
```bash
python src/models/model_evaluation_and_registry.py
```

### 4. Start API Service
```bash
python src/api/oil_gas_model_api.py
```

### 5. Test API
```bash
python src/api/test_api.py
```

## API Usage

The FastAPI service provides multiple endpoints for model predictions:

### Single Facility Prediction
```python
import requests

facility_data = {
    "latitude": 29.7604,
    "longitude": -95.3698,
    "facility_type": "oil_well",
    "region": "North America",
    "oil_production_bpd": 800,
    "gas_production_mcfd": 2000,
    # ... other features
}

response = requests.post(
    "http://localhost:8000/predict",
    json=facility_data
)
print(response.json())
```

### Batch Predictions
```python
batch_data = {"facilities": [facility_data_1, facility_data_2]}
response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_data
)
```

## Model Registry

All models are registered in MLflow Model Registry with:
- **Model Cards**: Comprehensive documentation and metadata
- **Performance Metrics**: Detailed evaluation results
- **Feature Importance**: Top contributing features
- **Governance Tags**: Risk level and approval status

Access MLflow UI at: http://localhost:8768

## Dashboard Integration

### Real-time Predictions
- All models support single facility and batch predictions
- Response times optimized for dashboard updates (< 5 seconds)
- Feature importance data available for visualization

### Model Outputs for Dashboard
1. **Equipment Health Score**: Continuous score 0-1 for equipment condition
2. **Production Efficiency**: Categorical classification (Low/Medium/High)
3. **Environmental Risk**: Risk level assessment (Low/Medium/High)

### Key Insights for Visualization
- Top 5 features by importance for each model
- Geographic distribution of risk levels
- Production efficiency trends by facility type
- Equipment health correlation with maintenance schedules

## Deployment on Domino

### Model API Deployment
```bash
# Deploy as Domino Model API
dominoapi model publish oil-gas-geospatial-models \
  --file src/api/oil_gas_model_api.py \
  --function predict \
  --requirements requirements.txt
```

### App Deployment
Create `app.sh` for Streamlit/Dash dashboard:
```bash
#!/bin/bash
streamlit run dashboard.py --server.port 8050 --server.address 0.0.0.0
```

## Monitoring and Maintenance

### Model Performance Monitoring
- Track prediction accuracy over time
- Monitor feature drift in production data
- Alert on degraded model performance

### Scheduled Retraining
- Monthly retraining with new facility data
- Automated model validation and registry updates
- A/B testing for model improvements

### Data Quality Checks
- Input validation for API requests
- Range checking for sensor readings
- Missing value imputation consistency

## Governance and Compliance

### Model Risk Management
- **Risk Level**: Medium
- **Validation Status**: Approved
- **Review Frequency**: Quarterly

### Approval Groups
- Modeling Practitioners
- IT Review Team
- Line of Business Review

### Compliance Frameworks
- Model Risk Management V3
- Ethical AI Framework
- Environmental Regulation Compliance

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model files exist in `/mnt/artifacts/models/`
   - Check MLflow tracking URI configuration

2. **API Prediction Errors**
   - Validate input data schema
   - Ensure all required features are provided
   - Check data types and ranges

3. **Performance Issues**
   - Monitor model response times
   - Check system resources (CPU/memory)
   - Consider model optimization

### Logs and Debugging
- API logs: `/mnt/artifacts/logs/model_api.log`
- MLflow experiments: http://localhost:8768
- Model evaluation reports: `/mnt/artifacts/reports/`

## Future Enhancements

1. **Advanced Models**
   - Deep learning for complex facility interactions
   - Time series models for trend prediction
   - Ensemble methods for improved accuracy

2. **Real-time Features**
   - Streaming data ingestion
   - Online model updates
   - Real-time drift detection

3. **Enhanced Monitoring**
   - Model explainability dashboard
   - Automated model retraining triggers
   - Advanced anomaly detection

## Support and Contact

For technical support or questions:
- Review MLflow experiments for detailed model metrics
- Check API documentation at http://localhost:8000/docs
- Refer to model analysis notebook for insights

## License

This project is developed for the Oil & Gas Geospatial Dashboard demonstration and follows enterprise ML development best practices.