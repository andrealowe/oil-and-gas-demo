# Oil & Gas Production Forecasting Champion Model

## Model Overview

This champion model represents the best-performing forecasting algorithm from a comprehensive AutoML comparison across four state-of-the-art time series frameworks: AutoGluon, Prophet/NeuralProphet, Nixtla NeuralForecast, and Combined LightGBM+ARIMA ensemble.

## Model Performance

- **Champion Framework**: Selected based on lowest Mean Absolute Error (MAE)
- **Evaluation Metric**: MAE (Mean Absolute Error) - measures average magnitude of prediction errors
- **Training Data**: Historical daily oil production aggregated across all facilities
- **Test Split**: 80/20 train-test split with temporal validation
- **Model Selection**: Automated champion selection from parallel framework comparison

## Business Context

### Use Case
Daily oil production forecasting for operational planning and resource optimization in oil & gas operations.

### Target Variable
- **Variable**: Daily oil production (barrels per day)
- **Granularity**: Daily aggregated production across all facilities
- **Forecast Horizon**: Configurable based on business requirements

### Performance Benchmarks
For oil production forecasting, target MAE thresholds by production scale:
- **Large fields (10,000+ bpd)**: MAE < 5-10% of average daily production
- **Medium fields (1,000-10,000 bpd)**: MAE < 3-7% of average daily production  
- **Small fields (<1,000 bpd)**: MAE < 2-5% of average daily production

## Technical Implementation

### AutoML Framework Comparison
The model was selected from parallel training across:

1. **AutoGluon TimeSeries**: 5 standardized configurations (fast, medium, high_quality, best_quality, interpretable)
2. **Prophet/NeuralProphet**: Traditional and neural prophet models with various seasonality parameters
3. **Nixtla NeuralForecast**: Advanced neural models (MLP, NBEATS, NHITS, LSTM, TFT) with StatsForecast fallback
4. **Combined LightGBM+ARIMA**: Ensemble model combining gradient boosting and time series techniques

### Data Pipeline
- **Source**: Production time series data with date and oil production columns
- **Preprocessing**: Daily aggregation across all facilities, data quality validation
- **Features**: Time-based features derived from historical production patterns
- **Validation**: Comprehensive data quality checks for completeness and consistency

### Model Training
- **Orchestration**: Domino Flows for parallel framework execution
- **Experiment Tracking**: MLflow with parent-child run relationships
- **Reproducibility**: Standardized configurations and random seeds across all frameworks
- **Champion Selection**: Automated based on MAE performance on holdout test set

## Model Governance & Compliance

### Model Risk Management
- **Model Category**: Operational forecasting model (non-regulatory)
- **Risk Level**: Medium - impacts operational planning and resource allocation
- **Validation**: Comprehensive backtesting with statistical significance testing
- **Monitoring**: Performance tracking via MAE and production vs. forecast comparisons

### Documentation Standards
- **Code Lineage**: Complete code versioning through Git integration
- **Data Lineage**: Tracked through Domino Datasets with versioning
- **Experiment Tracking**: Full MLflow logging of parameters, metrics, and artifacts
- **Reproducibility**: Containerized environment with dependency management

## Deployment & Operations

### Deployment Architecture
- **Model Serving**: Domino Model API endpoints for real-time predictions
- **Monitoring**: Built-in drift detection and performance monitoring
- **Retraining**: Automated retraining pipeline with champion/challenger framework
- **Rollback**: Version-controlled model registry for safe rollbacks

### Usage Guidelines
- **Input Format**: JSON with date range and facility identifiers
- **Output Format**: Daily production forecasts with confidence intervals
- **Latency**: Real-time inference for operational planning
- **Throughput**: Scalable for enterprise-level forecasting workloads

## Model Limitations & Considerations

### Known Limitations
- **Temporal Scope**: Model performance dependent on historical data quality and completeness
- **External Factors**: Does not account for unplanned maintenance, regulatory changes, or market disruptions
- **Facility Changes**: May require retraining when new facilities are added or decommissioned
- **Seasonal Patterns**: Performance may vary with seasonal production cycles

### Monitoring Requirements
- **Performance Drift**: Monitor MAE degradation over time
- **Data Drift**: Track changes in production patterns and facility operations
- **Concept Drift**: Validate model assumptions with changing operational conditions
- **Business Metrics**: Track forecast accuracy impact on operational decisions

## Contact & Support

### Model Owners
- **Data Science Team**: Primary model development and maintenance
- **Operations Team**: Business domain expertise and validation
- **MLOps Team**: Deployment, monitoring, and infrastructure support

### Support Process
- **Issues**: Report via Domino Model Monitoring dashboard alerts
- **Enhancements**: Submit requests through standard change management process
- **Emergency**: Contact on-call data science team for production issues

---

*Model Card Version: 1.0*  
*Last Updated: November 2024*  
*Framework: Domino Data Lab MLOps Platform*