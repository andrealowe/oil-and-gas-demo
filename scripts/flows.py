#!/usr/bin/env python3
"""
Domino Flows Configuration for Oil & Gas AutoML Forecasting Pipeline

This flow orchestrates the execution of multiple AutoML forecasting models in parallel,
followed by model comparison and champion selection.

Flow Structure:
1. Parallel execution of AutoML models:
   - AutoGluon TimeSeries
   - Prophet/NeuralProphet  
   - Nixtla NeuralForecast
   - Combined LightGBM+ARIMA
2. Sequential model comparison and registration

"""

from flytekit import workflow, task
from flytekit.types.file import FlyteFile
from typing import Dict, Any, List
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def oil_gas_automl_forecasting_workflow():
    """
    Oil & Gas AutoML Forecasting Workflow

    Steps:
    1. Generate synthetic oil & gas data (if not exists)
    2. Train multiple forecasting frameworks in parallel:
       - AutoGluon TimeSeries (multiple presets)
       - Prophet and NeuralProphet (multiple configurations)
       - Nixtla NeuralForecast (multiple neural models)
       - Combined LightGBM + ARIMA model
    3. Compare all results and register the best model
    """

    # Step 1: Data preparation - generate synthetic data
    # This ensures data exists before training tasks execute
    data_prep_task = DominoJobTask(
        name="Generate Oil & Gas Synthetic Data",
        domino_job_config=DominoJobConfig(
            Command="python scripts/oil_gas_data_generator.py"
        ),
        outputs={"data_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    # Execute data preparation first
    data_result = data_prep_task()

    # Step 2: Training tasks - run in parallel (depend on data preparation)
    # Each task produces training summary and model artifacts as outputs

    autogluon_task = DominoJobTask(
        name="Train AutoGluon TimeSeries Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/autogluon_forecasting.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    prophet_task = DominoJobTask(
        name="Train Prophet and NeuralProphet Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/prophet_forecasting.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    nixtla_task = DominoJobTask(
        name="Train Nixtla NeuralForecast Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/nixtla_forecasting.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    combined_model_task = DominoJobTask(
        name="Train Combined LightGBM + ARIMA Model",
        domino_job_config=DominoJobConfig(
            Command="python src/models/oil_gas_forecasting.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    # Execute all training tasks in parallel
    autogluon_result = autogluon_task()
    prophet_result = prophet_task()
    nixtla_result = nixtla_task()
    combined_result = combined_model_task()
    
    # Model comparison task - has inputs that depend on all training task outputs
    # This creates the sequential dependency after parallel training
    comparison_task = DominoJobTask(
        name="Compare Models and Register Champion",
        domino_job_config=DominoJobConfig(
            Command="python src/models/model_comparison.py"
        ),
        inputs={
            "autogluon_summary": FlyteFile,
            "prophet_summary": FlyteFile, 
            "nixtla_summary": FlyteFile,
            "combined_summary": FlyteFile
        },
        outputs={"comparison_results": FlyteFile},
        use_latest=True
    )
    
    # Execute comparison - depends on all training outputs
    comparison_result = comparison_task(
        autogluon_summary=autogluon_result["training_summary"],
        prophet_summary=prophet_result["training_summary"],
        nixtla_summary=nixtla_result["training_summary"], 
        combined_summary=combined_result["training_summary"]
    )
    
    return comparison_result

# Additional workflow for production forecasting using the champion model
@workflow  
def oil_gas_production_forecasting_workflow():
    """
    Production Forecasting Workflow using Champion Model
    
    Uses the registered champion model to generate production forecasts
    for operational planning and reporting.
    """
    
    # Data preparation task
    data_prep_task = DominoJobTask(
        name="Prepare Production Data",
        domino_job_config=DominoJobConfig(
            Command="python scripts/oil_gas_data_generator.py"
        ),
        outputs={"prepared_data": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    # Forecasting task using champion model
    forecast_task = DominoJobTask(
        name="Generate Production Forecasts", 
        domino_job_config=DominoJobConfig(
            Command="python src/api/oil_gas_model_api.py --mode=batch_forecast"
        ),
        inputs={
            "input_data": FlyteFile
        },
        outputs={"forecasts": FlyteFile},
        use_latest=True
    )
    
    # Dashboard update task
    dashboard_update_task = DominoJobTask(
        name="Update Forecasting Dashboard",
        domino_job_config=DominoJobConfig(
            Command="python scripts/forecasting_dashboard.py --mode=update"
        ),
        inputs={
            "forecast_data": FlyteFile
        },
        use_latest=True
    )
    
    # Execute tasks sequentially
    data_result = data_prep_task()
    forecast_result = forecast_task(input_data=data_result["prepared_data"])
    dashboard_result = dashboard_update_task(forecast_data=forecast_result["forecasts"])
    
    return dashboard_result

# Workflow for model retraining and validation
@workflow
def oil_gas_model_retraining_workflow():
    """
    Model Retraining Workflow
    
    Periodically retrains the champion model with new data and validates performance.
    If performance degrades, triggers the full AutoML comparison workflow.
    """
    
    # Data validation task
    data_validation_task = DominoJobTask(
        name="Validate New Training Data",
        domino_job_config=DominoJobConfig(
            Command="python src/data/feature_engineering.py --validate"
        ),
        outputs={"validation_results": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    # Champion model retraining
    retrain_task = DominoJobTask(
        name="Retrain Champion Model",
        domino_job_config=DominoJobConfig(
            Command="python src/models/model_evaluation_and_registry.py --retrain"
        ),
        inputs={
            "validation_data": FlyteFile
        },
        outputs={"retrain_results": FlyteFile},
        use_latest=True
    )
    
    # Performance validation
    validation_task = DominoJobTask(
        name="Validate Retrained Model Performance", 
        domino_job_config=DominoJobConfig(
            Command="python src/models/model_evaluation_and_registry.py --validate"
        ),
        inputs={
            "retrain_data": FlyteFile
        },
        outputs={"performance_results": FlyteFile},
        use_latest=True
    )
    
    # Execute tasks sequentially
    validation_result = data_validation_task()
    retrain_result = retrain_task(validation_data=validation_result["validation_results"])
    performance_result = validation_task(retrain_data=retrain_result["retrain_results"])
    
    return performance_result

# Monitoring workflow for model performance tracking
@workflow
def oil_gas_model_monitoring_workflow():
    """
    Model Monitoring Workflow
    
    Continuously monitors champion model performance and data drift.
    Triggers alerts and retraining when necessary.
    """
    
    # Data drift detection
    drift_detection_task = DominoJobTask(
        name="Detect Data Drift",
        domino_job_config=DominoJobConfig(
            Command="python src/monitoring/data_drift_monitor.py"
        ),
        outputs={"drift_report": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    # Model performance monitoring
    performance_monitoring_task = DominoJobTask(
        name="Monitor Model Performance",
        domino_job_config=DominoJobConfig(
            Command="python src/monitoring/model_performance_monitor.py"
        ),
        outputs={"performance_report": FlyteFile},
        use_latest=True,
        cache=True
    )
    
    # Alert generation
    alert_task = DominoJobTask(
        name="Generate Performance Alerts",
        domino_job_config=DominoJobConfig(
            Command="python src/monitoring/alert_generator.py"
        ),
        inputs={
            "drift_data": FlyteFile,
            "performance_data": FlyteFile
        },
        outputs={"alerts": FlyteFile},
        use_latest=True
    )
    
    # Execute monitoring tasks in parallel, then generate alerts
    drift_result = drift_detection_task()
    performance_result = performance_monitoring_task()
    
    alert_result = alert_task(
        drift_data=drift_result["drift_report"],
        performance_data=performance_result["performance_report"]
    )
    
    return alert_result

# Main comprehensive workflow that orchestrates all components
@workflow
def oil_gas_comprehensive_ml_pipeline():
    """
    Comprehensive ML Pipeline for Oil & Gas Analytics
    
    This is the master workflow that orchestrates the complete ML lifecycle:
    1. AutoML model training and comparison
    2. Production forecasting
    3. Model monitoring and retraining
    
    Can be scheduled to run periodically or triggered by events.
    """
    
    # Execute the AutoML comparison workflow first
    automl_result = oil_gas_automl_forecasting_workflow()
    
    # Generate production forecasts with the selected champion
    forecast_result = oil_gas_production_forecasting_workflow()
    
    # Monitor model performance
    monitoring_result = oil_gas_model_monitoring_workflow()
    
    return {
        "automl_comparison": automl_result,
        "production_forecasts": forecast_result,
        "monitoring_results": monitoring_result
    }

# Export the main workflow for Domino Flows
if __name__ == "__main__":
    # This script can be executed to register the workflows with Domino
    print("Domino Flows for Oil & Gas AutoML Forecasting Pipeline")
    print("=" * 60)
    print()
    print("Available Workflows:")
    print("1. oil_gas_automl_forecasting_workflow - Main AutoML comparison")
    print("   (Includes data generation step)")
    print("2. oil_gas_production_forecasting_workflow - Production forecasting")
    print("3. oil_gas_model_retraining_workflow - Model retraining")
    print("4. oil_gas_model_monitoring_workflow - Performance monitoring")
    print("5. oil_gas_comprehensive_ml_pipeline - Complete ML lifecycle")
    print()
    print("For Local Development/Testing:")
    print("Run data generator first to create datasets:")
    print("  python scripts/oil_gas_data_generator.py")
    print()
    print("Then run individual model training scripts:")
    print("  python src/models/autogluon_forecasting.py")
    print("  python src/models/prophet_forecasting.py")
    print("  python src/models/nixtla_forecasting.py")
    print("  python src/models/oil_gas_forecasting.py")
    print()
    print("To execute in Domino Flows:")
    print("1. Upload this file to your Domino project")
    print("2. Create a new Flow using the Domino UI")
    print("3. Select the desired workflow function")
    print("4. Configure compute environment and schedule")
    print()
    print("Key Features:")
    print("- Automatic data generation before training")
    print("- Parallel execution of AutoML frameworks")
    print("- Automatic model comparison and registration")
    print("- Champion model deployment")
    print("- Continuous monitoring and retraining")
    print("- Integration with MLflow experiment tracking")
    print("- Correct data path handling (/mnt/data/Oil-and-Gas-Demo/)")