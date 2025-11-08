#!/usr/bin/env python3
"""
Domino Flows Configuration for Oil & Gas AutoML Forecasting Pipeline

This workflow trains multiple forecasting models in parallel and compares results.
Follows Domino Flows best practices with simplified structure.
"""

from flytekit import workflow, task
from flytekit.types.file import FlyteFile
from typing import Dict, Any
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask


@workflow
def oil_gas_automl_forecasting_workflow():
    """
    Oil & Gas AutoML Forecasting Workflow

    Trains four forecasting frameworks in parallel and compares their results.
    The use_latest=True flag ensures the latest dataset snapshot is mounted.
    """

    # Step 1: Data quality check
    data_check_task = DominoJobTask(
        name="Check Data Quality",
        domino_job_config=DominoJobConfig(
            Command="python scripts/oil_gas_data_generator.py"
        ),
        outputs={"data_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    # Execute data check first
    data_result = data_check_task()

    # Step 2: Training tasks - run in parallel
    # Each task produces a training summary file as output
    autogluon_task = DominoJobTask(
        name="Train AutoGluon TimeSeries Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/autogluon_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile},
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    prophet_task = DominoJobTask(
        name="Train Prophet and NeuralProphet Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/prophet_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile},
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    nixtla_task = DominoJobTask(
        name="Train Nixtla NeuralForecast Models",
        domino_job_config=DominoJobConfig(
            Command="python src/models/nixtla_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile},
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    combined_task = DominoJobTask(
        name="Train Combined LightGBM + ARIMA Model",
        domino_job_config=DominoJobConfig(
            Command="python src/models/oil_gas_forecasting.py"
        ),
        inputs={"data_prep": FlyteFile},
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    # Execute training tasks in parallel - they all depend on data_result
    autogluon_result = autogluon_task(data_prep=data_result["data_summary"])
    prophet_result = prophet_task(data_prep=data_result["data_summary"])
    nixtla_result = nixtla_task(data_prep=data_result["data_summary"])
    combined_result = combined_task(data_prep=data_result["data_summary"])

    # Compare task - has inputs that depend on training task outputs
    # This creates the sequential dependency after parallel training
    compare_task = DominoJobTask(
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
        use_latest=True
    )

    # Execute comparison - depends on all training outputs
    comparison = compare_task(
        autogluon_summary=autogluon_result["training_summary"],
        prophet_summary=prophet_result["training_summary"],
        nixtla_summary=nixtla_result["training_summary"],
        combined_summary=combined_result["training_summary"]
    )

    return comparison


# Export the main workflow for Domino Flows
if __name__ == "__main__":
    print("Domino Flows for Oil & Gas AutoML Forecasting Pipeline")
    print("=" * 60)
    print()
    print("Workflow: oil_gas_automl_forecasting_workflow")
    print("- Data quality check")
    print("- Parallel training of AutoML frameworks")
    print("- Model comparison and champion selection")
    print("- Compatible with read-only Domino Datasets")