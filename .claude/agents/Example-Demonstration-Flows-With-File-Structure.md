---
name: Example-Demonstration-Flows
description: Reference documentation for demonstration workflows - not an executable agent
model: none
color: gray
---

### Quick Proof of Concept
```python
# Requirements favor speed
requirements = {
    "project": "quick_demo",
    "deployment_urgency": "urgent",
    "ui_complexity": "low"
}
# Front-end agent recommends: Gradio
# Files organized as:
# - Notebooks: /mnt/code/notebooks/
# - Scripts: /mnt/code/scripts/
# - Production code: /mnt/code/src/
# - Artifacts: /mnt/artifacts/
# - Data: /mnt/data/quick_demo/

# All agents automatically:
# 1. Use the standardized directory structure
# 2. Save notebooks to /mnt/code/notebooks/
# 3. Save scripts to /mnt/code/scripts/
# 4. Save production code to /mnt/code/src/
# 5. Save artifacts to /mnt/artifacts/
# 6. Save data to /mnt/data/{project}/
# 7. Update project-level requirements.txt
# 8. Register everything with MLflow
```

### Enterprise Dashboard with Full Pipeline
```python
# Complete pipeline example showing file organization
requirements = {
    "project": "customer_churn",
    "target_metric": "f1_score",
    "deployment_strategy": "canary",
    "expected_users": 100,
    "ui_complexity": "medium"
}

# Directory structure used:
# /mnt/code/
#   ├── src/                   # Production Python code
#   │   ├── api/              # API endpoints and serving
#   │   ├── models/           # Model training and preprocessing
#   │   ├── monitoring/       # Monitoring and dashboards
#   │   └── data/             # Data generation and processing
#   ├── scripts/              # Utility scripts and applications
#   ├── notebooks/            # Exploratory notebooks
#   ├── config/               # Configuration files
#   ├── tests/                # Test files
#   └── docs/                 # Documentation
#       └── business-analysis/ # Business requirements
#
# /mnt/artifacts/
#   ├── models/               # Saved model files
#   ├── reports/              # Analysis and validation reports
#   └── visualizations/       # Generated plots and charts
#
# /mnt/data/customer_churn/   # Project-specific data
#   ├── raw/                  # Raw data
#   ├── processed/            # Processed datasets
#   └── features/             # Feature store

# Each agent logs artifacts both locally and to MLflow
with mlflow.start_run(run_name="customer_churn_pipeline"):

    # Data Wrangler
    data = data_wrangler.acquire_data(specs)
    # Saves notebooks: /mnt/code/notebooks/data_wrangling.ipynb
    # Saves scripts: /mnt/code/scripts/generate_data.py
    # Saves code: /mnt/code/src/data/generate_*.py
    # Saves artifacts: /mnt/artifacts/data_profiles.json
    # Saves data: /mnt/data/customer_churn/raw/
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: data samples, profile reports

    # Data Scientist
    eda_results = data_scientist.perform_eda(data)
    # Saves notebooks: /mnt/code/notebooks/eda.ipynb
    # Saves scripts: /mnt/code/scripts/run_eda.py
    # Saves artifacts: /mnt/artifacts/visualizations/
    # Saves reports: /mnt/artifacts/reports/eda_report.json
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: profile report, plots, insights

    # Model Developer
    model = model_developer.develop_models(data)
    # Saves notebooks: /mnt/code/notebooks/model_training.ipynb
    # Saves code: /mnt/code/src/models/train_model.py
    # Saves code: /mnt/code/src/models/preprocess.py
    # Saves artifacts: /mnt/artifacts/models/best_model.pkl
    # Saves data: /mnt/data/customer_churn/processed/
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: registered models with signatures

    # MLOps Engineer
    deployment = mlops_engineer.deploy(model)
    # Saves code: /mnt/code/src/api/predict.py
    # Saves code: /mnt/code/src/monitoring/monitor.py
    # Saves configs: /mnt/code/config/deployment_config.yaml
    # Saves scripts: /mnt/code/start_api.sh
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: deployment specs, monitoring config

    # Front-End Developer (recommends Streamlit)
    frontend = frontend_developer.create_app(model, requirements)
    # Saves scripts: /mnt/code/scripts/app.py
    # Saves script: /mnt/code/app.sh (startup script)
    # Saves docs: /mnt/code/docs/FRONTEND_README.md
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: app code, Docker configs

    # Model Validator
    validation = model_validator.validate(model)
    # Saves notebooks: /mnt/code/notebooks/model_validation.ipynb
    # Saves tests: /mnt/code/tests/test_*.py
    # Saves scripts: /mnt/code/scripts/run_validation.py
    # Saves reports: /mnt/artifacts/reports/validation_report.json
    # Updates: /mnt/code/requirements.txt
    # Logs to MLflow: validation reports, test results
```

### File Naming Conventions

**Notebooks:** (in `/mnt/code/notebooks/`)
- `{project}_data_generation.ipynb`
- `{project}_eda.ipynb`
- `{project}_model_training.ipynb`
- `{project}_validation.ipynb`

**Production Code:** (in `/mnt/code/src/`)
- `src/api/predict.py` - API endpoints
- `src/models/train_model.py` - Training logic
- `src/models/preprocess.py` - Preprocessing
- `src/monitoring/monitor.py` - Monitoring
- `src/monitoring/dashboard.py` - Dashboards
- `src/data/generate_*.py` - Data generation

**Scripts:** (in `/mnt/code/scripts/`)
- `run_eda.py` - EDA execution
- `run_validation.py` - Validation execution
- `*_app.py` - Frontend applications

**Startup Scripts:** (in `/mnt/code/`)
- `app.sh` - Start frontend applications
- `start_api.sh` - Start API server
- `start_dashboard.sh` - Start monitoring dashboard

**Configuration:** (in `/mnt/code/config/`)
- `deployment_config.yaml`
- `monitoring_config.yaml`

**Tests:** (in `/mnt/code/tests/`)
- `test_functional.py`
- `test_performance.py`
- `test_fairness.py`
- `test_compliance.py`

**Documentation:** (in `/mnt/code/docs/`)
- `PROJECT_STATUS.md`
- `QUICK_RESUME.md`
- `FRONTEND_README.md`
- `business-analysis/` - Business analysis docs
