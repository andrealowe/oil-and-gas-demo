# Oil & Gas Production Forecasting Champion Model

## Model Overview
Champion forecasting model selected from AutoML comparison across AutoGluon, Prophet/NeuralProphet, Nixtla NeuralForecast, and Combined LightGBM+ARIMA frameworks.

## Performance
- **Metric**: Mean Absolute Error (MAE) - average magnitude of prediction errors
- **Selection**: Automated champion based on lowest MAE on holdout test set
- **Validation**: 80/20 temporal train-test split across historical production data

## Business Use Case
Daily oil production forecasting for operational planning and resource optimization.
- **Target**: Daily oil production (barrels per day)
- **Input**: Historical production time series data
- **Output**: Daily production forecasts with confidence intervals

## Technical Implementation
- **Training**: Parallel execution via Domino Flows with standardized configurations
- **Tracking**: MLflow experiment tracking with parent-child run relationships  
- **Frameworks**: 4 AutoML frameworks tested with 5+ configurations each
- **Selection**: Automated champion selection based on statistical performance

## Governance & Compliance
- **Risk Level**: Medium - impacts operational planning and resource allocation
- **Documentation**: Complete code/data lineage through Git and Domino Datasets
- **Monitoring**: Performance tracking via MAE and drift detection
- **Retraining**: Monthly or on performance degradation

## Performance Benchmarks
Target MAE thresholds by production scale:
- Large fields (10,000+ bpd): MAE < 5-10% of average daily production
- Medium fields (1,000-10,000 bpd): MAE < 3-7% of average daily production
- Small fields (<1,000 bpd): MAE < 2-5% of average daily production

## Deployment
- **Serving**: Domino Model API endpoints for real-time predictions
- **Monitoring**: Built-in drift detection and performance monitoring
- **Operations**: Version-controlled registry with rollback capabilities

## Limitations
- Dependent on historical data quality and completeness
- Does not account for unplanned maintenance or regulatory changes
- May require retraining when facilities are added/decommissioned
- Performance varies with seasonal production cycles

## Support
- **Model Owners**: Data Science Team (development), Operations Team (domain), MLOps Team (deployment)
- **Issues**: Report via Domino Model Monitoring dashboard
- **Emergency**: Contact on-call data science team

---
*Model Card v1.0 | Nov 2024 | Domino Data Lab*