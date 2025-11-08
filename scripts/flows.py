#!/usr/bin/env python3
"""
Domino Flows Configuration for Oil & Gas AutoML Forecasting Pipeline

This flow orchestrates the execution of multiple AutoML forecasting models in parallel,
followed by model comparison and champion selection.

Designed for read-only Domino Datasets where data is pre-loaded.
"""

from flytekit import workflow, task
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import Dict, Any, List, TypeVar, NamedTuple
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

# Define structured outputs for the workflow
class ForecastingResults(NamedTuple):
    data_summary: FlyteFile[TypeVar("json")]
    autogluon_summary: FlyteFile[TypeVar("json")]
    prophet_summary: FlyteFile[TypeVar("json")]
    nixtla_summary: FlyteFile[TypeVar("json")]
    combined_summary: FlyteFile[TypeVar("json")]
    models_directory: FlyteDirectory

@workflow
def oil_gas_automl_forecasting_workflow() -> ForecastingResults:
    """
    Oil & Gas AutoML Forecasting Workflow

    Steps:
    1. Check data quality and availability (read-only safe)
    2. Train multiple forecasting frameworks in parallel:
       - AutoGluon TimeSeries (multiple presets)
       - Prophet and NeuralProphet (multiple configurations)
       - Nixtla NeuralForecast (multiple neural models)
       - Combined LightGBM + ARIMA model
    3. Compare all results and register the best model
    
    Note: This workflow assumes data is pre-loaded via Domino Dataset
    """

    # Step 1: Data quality check
    # Verifies that required data files are accessible (read-only safe)
    # This task only checks data availability without attempting to write
    data_prep_task = DominoJobTask(
        name="Check Data Quality", 
        domino_job_config=DominoJobConfig(
            Command="python scripts/oil_gas_data_generator.py"
        ),
        outputs={"data_summary": FlyteFile[TypeVar("json")]},
        use_latest=True,
        cache=True
    )

    # Execute data preparation first
    data_result = data_prep_task()

    # Step 2: Training tasks - run in parallel (depend on data preparation)
    # Each task produces training summary and model artifacts as outputs
    # All tasks have data_prep dependency to ensure data exists before training

    autogluon_task = DominoJobTask(
        name="Train AutoGluon TimeSeries Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/autogluon_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile[TypeVar("json")]},  # Explicit dependency on data generation
        outputs={"training_summary": FlyteFile[TypeVar("json")]},
        use_latest=True,
        cache=True
    )

    prophet_task = DominoJobTask(
        name="Train Prophet and NeuralProphet Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/prophet_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile[TypeVar("json")]},  # Explicit dependency on data generation
        outputs={"training_summary": FlyteFile[TypeVar("json")]},
        use_latest=True,
        cache=True
    )

    nixtla_task = DominoJobTask(
        name="Train Nixtla NeuralForecast Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/nixtla_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile[TypeVar("json")]},  # Explicit dependency on data generation
        outputs={"training_summary": FlyteFile[TypeVar("json")]},
        use_latest=True,
        cache=True
    )

    combined_model_task = DominoJobTask(
        name="Train Combined LightGBM + ARIMA Model",
        domino_job_config=DominoJobConfig(
            Command="python src/models/oil_gas_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile[TypeVar("json")]},  # Explicit dependency on data generation
        outputs={"training_summary": FlyteFile[TypeVar("json")]},
        use_latest=True,
        cache=True
    )

    # Execute all training tasks in parallel - they all depend on data_result
    # This ensures data generation completes before any training task starts
    autogluon_result = autogluon_task(data_prep=data_result["data_summary"])
    prophet_result = prophet_task(data_prep=data_result["data_summary"])
    nixtla_result = nixtla_task(data_prep=data_result["data_summary"])
    combined_result = combined_model_task(data_prep=data_result["data_summary"])

    # Model comparison task - has inputs that depend on all training task outputs
    # This creates the sequential dependency after parallel training
    comparison_task = DominoJobTask(
        name="Compare Models and Register Champion",
        domino_job_config=DominoJobConfig(
            Command="python src/models/model_comparison.py"
        ),
        inputs={
            "autogluon_summary": FlyteFile[TypeVar("json")],
            "prophet_summary": FlyteFile[TypeVar("json")],
            "nixtla_summary": FlyteFile[TypeVar("json")],
            "combined_summary": FlyteFile[TypeVar("json")]
        },
        outputs={
            "comparison_results": FlyteFile[TypeVar("json")],
            "models_directory": FlyteDirectory
        },
        use_latest=True
    )

    # Execute comparison - depends on all training outputs
    comparison_result = comparison_task(
        autogluon_summary=autogluon_result["training_summary"],
        prophet_summary=prophet_result["training_summary"],
        nixtla_summary=nixtla_result["training_summary"],
        combined_summary=combined_result["training_summary"]
    )

    # Return structured results for artifact tracking
    return ForecastingResults(
        data_summary=data_result["data_summary"],
        autogluon_summary=autogluon_result["training_summary"],
        prophet_summary=prophet_result["training_summary"],
        nixtla_summary=nixtla_result["training_summary"],
        combined_summary=combined_result["training_summary"],
        models_directory=comparison_result["models_directory"]
    )

# Export the main workflow for Domino Flows
if __name__ == "__main__":
    # This script can be executed to register the workflow with Domino
    print("Domino Flows for Oil & Gas AutoML Forecasting Pipeline")
    print("=" * 60)
    print()
    print("Workflow: oil_gas_automl_forecasting_workflow")
    print("- Data quality check (read-only safe)")
    print("- Parallel execution of AutoML frameworks")
    print("- Automatic model comparison and registration")
    print("- Compatible with read-only Domino Datasets")
    print()
    print("Note: This workflow assumes data is pre-loaded via Domino Dataset")