# workflow.py
"""
Simplified Domino Flow for Credit Card Fraud Detection Training

This workflow trains three classifiers in parallel and compares results.
Datasets are mounted automatically via use_latest=True flag.
"""

from flytekit import workflow, task
from flytekit.types.file import FlyteFile
from typing import Dict, Any
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask


@workflow
def credit_card_fraud_detection_workflow():
    """
    Credit Card Fraud Detection Training Workflow

    Trains three classifiers in parallel (AdaBoost, GaussianNB, XGBoost)
    and compares their results.

    The use_latest=True flag ensures the latest dataset snapshot is mounted.
    """

    # Training tasks - run in parallel
    # Each task produces a training summary file as output
    ada_task = DominoJobTask(
        name="Train AdaBoost Classifier",
        domino_job_config=DominoJobConfig(
            Command="python exercises/d_TrainingAndEvaluation/trainer_ada.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    gnb_task = DominoJobTask(
        name="Train GaussianNB Classifier",
        domino_job_config=DominoJobConfig(
            Command="python exercises/d_TrainingAndEvaluation/trainer_gnb.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    xgb_task = DominoJobTask(
        name="Train XGBoost Classifier",
        domino_job_config=DominoJobConfig(
            Command="python exercises/d_TrainingAndEvaluation/trainer_xgb.py"
        ),
        outputs={"training_summary": FlyteFile},
        use_latest=True,
        cache=True
    )

    # Execute training tasks in parallel
    ada_result = ada_task()
    gnb_result = gnb_task()
    xgb_result = xgb_task()

    # Compare task - has inputs that depend on training task outputs
    # This creates the sequential dependency after parallel training
    compare_task = DominoJobTask(
        name="Compare Training Results",
        domino_job_config=DominoJobConfig(
            Command="python exercises/d_TrainingAndEvaluation/compare.py"
        ),
        inputs={
            "ada_summary": FlyteFile,
            "gnb_summary": FlyteFile,
            "xgb_summary": FlyteFile
        },
        use_latest=True
    )

    # Execute comparison - depends on all training outputs
    comparison = compare_task(
        ada_summary=ada_result["training_summary"],
        gnb_summary=gnb_result["training_summary"],
        xgb_summary=xgb_result["training_summary"]
    )

    return comparison
