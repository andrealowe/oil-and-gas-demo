# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a collection of specialized Claude Code agents designed for building end-to-end machine learning demonstrations on the Domino Data Lab platform. The agents work together to create production-ready ML solutions across the entire lifecycle.

## Directory Structure

The repository follows a standardized structure:

```
/mnt/code/
├── src/                    # Production Python code
│   ├── api/               # API endpoints and serving
│   ├── models/            # Model training and preprocessing
│   ├── monitoring/        # Model monitoring and dashboards
│   └── data/              # Data generation and processing scripts
├── scripts/                # Utility scripts and applications
│   └── data_config.py     # Data path configuration utility
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
    └── business-analysis/ # Business requirements and analysis

/mnt/artifacts/             # Git-based projects only
├── models/                 # Saved model files
├── reports/                # Analysis reports
└── visualizations/         # Generated plots and charts

/mnt/data/{project}/        # Git-based projects: Dataset storage
└── ...                     # (Use get_data_paths() to determine)

/domino/datasets/local/{project}/  # DFS projects: Dataset storage
└── ...                            # (Use get_data_paths() to determine)
```

**Key directories:**
- `src/` - All production Python code organized by function
- `scripts/` - Standalone scripts and frontend applications (includes data_config.py)
- `notebooks/` - Exploratory Jupyter notebooks
- `config/` - Configuration and deployment specs
- `tests/` - All test files
- `docs/` - Documentation including business analysis

**Data storage locations (use `get_data_paths()` to determine):**
- Git-based: `/mnt/data/{project}/` for datasets, `/mnt/artifacts/` for models
- DFS-based: `/domino/datasets/local/{project}/` for datasets, `/mnt/` for models

## Available Agents

### Core Agents
- **Master-Project-Manager-Agent**: Orchestrates complete ML pipelines with governance compliance
- **Data-Wrangler-Agent**: Data acquisition, generation, and pipeline management
- **Data-Scientist-Agent**: EDA, visualization, and insight generation
- **Model-Developer-Agent**: Model training and optimization
- **Model-Validator-Agent**: Performance validation, robustness testing, and governance compliance
- **Business-Analyst-Agent**: Requirements translation, success metrics, and governance assessment
- **MLOps-Engineer-Agent**: Deployment pipelines, monitoring, and compliance-aware CI/CD
- **Front-End-Developer-Agent**: UI development with technology recommendations

### Reference Documentation
- **Agent-Interaction-Protocol**: Communication patterns between agents
- **Example-Demonstration-Flows**: Workflow examples and file organization

### Governance Integration
- **Governance Policies**: Located in `/mnt/code/.reference/governance/`
  - NIST Risk Management Framework (RMF)
  - Model Risk Management V3
  - Ethical AI Framework
  - Model Intake Process
  - External LLM Governance Policy
- **Approver Groups**: Defined in `/mnt/code/.reference/governance/Approvers.md`
  - Modeling teams (modeling-review, modeling-practitioners, modeling-leadership)
  - IT teams (it-review, it-leadership)
  - Information Security (infosec-review, infosec-leadership)
  - Legal teams (legal-review, legal-leadership)
  - Line of Business (lob-leadership, lob-review)
  - Marketing teams (marketing-review, marketing-leadership)

## Domino Data Lab Platform

This project is designed for deployment on **Domino Data Lab**, an enterprise MLOps platform.

### Domino Resources
- **Official Documentation**: https://docs.dominodatalab.com
- **Platform Access**: Workspaces, Datasets, Jobs, Apps, Model APIs, Flows

### Key Domino Features Used
- **Domino Workspaces** - Interactive development environments
- **Domino Datasets** - Centralized data storage and versioning
- **Domino Apps** - Deploy Streamlit/Dash applications with `app.sh` launcher
- **Domino Model APIs** - Scalable model serving with monitoring
- **Domino Flows** - Orchestrate multi-step ML pipelines
- **Domino Jobs** - Scheduled and on-demand execution
- **MLflow Integration** - Built-in experiment tracking at http://localhost:8768
- **Git Integration** - Automatic version control

### Agent Instructions for Domino Deployment

When working with Domino-specific features, agents should:

1. **Reference Latest Documentation**
   - Use WebFetch to retrieve current docs from https://docs.dominodatalab.com
   - Follow Domino best practices for deployment patterns
   - Check for version-specific features and compatibility

2. **File System Structure and Data Storage**

   **CRITICAL: Use the Data Config Utility**

   All agents MUST use `/mnt/code/scripts/data_config.py` to determine correct storage paths. This ensures compatibility across Git-based and DFS (Domino File System) projects.

   **Data Storage Rules by Project Type:**

   **Git-based Projects** (DOMINO_WORKING_DIR=/mnt/code):
   - Code: `/mnt/code/` (Git-synced)
   - Data files: `$DOMINO_DATASETS_DIR/{project_name}/` (typically `/mnt/data/{project_name}/`)
   - Artifacts: `/mnt/artifacts/` (models, reports, visualizations)
   - **Never store data in `/mnt/code/data/`** - this bloats the git repository!

   **DFS Projects** (DOMINO_WORKING_DIR=/mnt):
   - Code: `/mnt/code/`
   - Data files: `/domino/datasets/local/{project_name}/`
   - Artifacts: `/mnt/` (models, reports)

   **Usage in Scripts:**
   ```python
   # Import the utility
   import sys
   sys.path.insert(0, '/mnt/code')
   from scripts.data_config import get_data_paths

   # Get correct paths
   paths = get_data_paths('my_project')
   data_dir = paths['base_data_path']      # Where to store datasets
   artifacts_dir = paths['artifacts_path']  # Where to store models/reports

   # Example: Save data
   train_csv = data_dir / 'train.csv'
   df.to_csv(train_csv, index=False)

   # Example: Save model
   model_path = artifacts_dir / 'models' / 'my_model.pkl'
   model_path.parent.mkdir(parents=True, exist_ok=True)
   joblib.dump(model, model_path)
   ```

   All paths must be absolute, not relative

3. **App Deployment**
   - Create `app.sh` as the launcher script for Domino Apps
   - Configure proper ports (8050 for Dash, 8501 for Streamlit)
   - Include health checks and startup validation
   - Document hardware requirements (compute tier, GPU needs)

4. **API Deployment**
   - Use FastAPI or Flask for Model APIs
   - Implement `/health` endpoint for monitoring
   - Log predictions for drift detection
   - Configure authentication if required

5. **Environment Configuration**
   - Document required environment variables
   - Specify Python version and dependencies in requirements.txt
   - Note any system packages needed
   - Include Dockerfile if custom environment needed

6. **Best Practices**
   - Use Domino Datasets for large data files (>100MB)
   - Leverage Domino's built-in MLflow (no separate setup needed)
   - Use Domino Jobs for scheduled training/retraining
   - Deploy monitoring dashboards as Domino Apps
   - Use Domino Flows for production pipelines

### 7. Domino Flows Development & Debugging

**CRITICAL INFORMATION**: This section documents resolved issues and best practices for Domino Flows development. All scripts in this project have been debugged and are production-ready.

#### Workflow I/O Requirements (✅ IMPLEMENTED)

According to [Domino Flows documentation](https://docs.dominodatalab.com/en/cloud/user_guide/78acf5/orchestrate-with-flows/), tasks **MUST** write all declared outputs or they fail.

**WorkflowIO Helper** (`/mnt/code/src/models/workflow_io.py`):
- Automatically detects Flow vs standalone execution mode
- Reads inputs from `/workflow/inputs/<name>`
- Writes outputs to `/workflow/outputs/<name>`
- Handles JSON serialization automatically

**Usage Pattern** (implemented in all forecasting scripts):
```python
from src.models.workflow_io import WorkflowIO

# In main() function
wf_io = WorkflowIO()

# Check if running in Flow mode
if wf_io.is_workflow_job():
    # Read flow inputs (establishes task dependencies)
    data_prep = wf_io.read_input("data_prep")

    # Write flow outputs (CRITICAL - must write ALL declared outputs)
    wf_io.write_output("training_summary", summary)
```

#### Critical Fix: Sidecar Uploader Error (✅ RESOLVED)

**Issue**: "failed to run the job: Workflow output contract not satisfied: sidecar uploader exited with code 1"

**Root Cause**: Scripts failed BEFORE writing workflow outputs, leaving the sidecar uploader with no file to upload.

**Solution Implemented**:
Added `write_error_output()` method to WorkflowIO that ensures outputs are ALWAYS written, even on failure:

```python
try:
    # Main execution
    result = train_models()
    wf_io.write_output("training_summary", result)
except Exception as e:
    logger.error(f"Error: {e}")
    # CRITICAL: Write error output for Flow execution
    wf_io.write_error_output("training_summary", e, "framework_name")
    raise
```

**Error Output Structure**:
```json
{
  "timestamp": "2025-11-08T13:45:00Z",
  "framework": "autogluon",
  "status": "error",
  "error_message": "FileNotFoundError: ...",
  "error_type": "FileNotFoundError",
  "total_configs": 0,
  "successful_configs": 0,
  "best_config": null,
  "best_mae": null
}
```

**Updated Scripts** (all have proper error handling):
- `/mnt/code/scripts/oil_gas_data_generator.py` → writes `data_summary`
- `/mnt/code/src/models/autogluon_forecasting.py` → writes `training_summary`
- `/mnt/code/src/models/prophet_forecasting.py` → writes `training_summary`
- `/mnt/code/src/models/nixtla_forecasting.py` → writes `training_summary`
- `/mnt/code/src/models/oil_gas_forecasting.py` → writes `training_summary`
- `/mnt/code/src/models/model_comparison.py` → writes `comparison_results`

#### Data Persistence Between Flow Tasks

**Challenge**: Domino Flow tasks run in isolated environments. Data written by one task doesn't automatically persist to others unless stored in a shared Domino Dataset.

**Two Solutions Implemented**:

**Solution A: Auto-Generation Fallback** (✅ Current Implementation)
- Each forecasting script calls `ensure_data_exists()` at startup
- If data missing, automatically generates it
- Works immediately without Domino configuration
- Trade-off: Slower (4× data generation), but resilient

**Solution B: Domino Dataset** (Recommended for Production)
- Create Domino Dataset named "Oil-and-Gas-Demo"
- Mount to `/mnt/data/Oil-and-Gas-Demo`
- Configure Flow to mount dataset to all tasks
- Data generated once, shared across all tasks
- Much faster and more efficient

#### Domino Flows Best Practices (All Implemented)

✅ **Strongly Typed Inputs/Outputs**
```python
from typing import TypeVar
from flytekit.types.file import FlyteFile

# CORRECT - TypeVar is mandatory
outputs={"training_summary": FlyteFile[TypeVar("json")]}

# WRONG - Will cause type errors
outputs={"training_summary": FlyteFile}
```

✅ **Explicit Task Dependencies**
```python
# Create dependency by passing output as input
data_result = data_prep_task()
training_result = training_task(data_prep=data_result["data_summary"])
```

✅ **Side-Effect Free Tasks**
- Tasks read versioned inputs
- Tasks write defined outputs
- No shared state contamination
- Reproducible execution

✅ **Dual-Mode Scripts**
- All scripts work standalone for development
- All scripts work in Flows for production
- Automatic mode detection via `WorkflowIO.is_workflow_job()`

#### Current Workflow Structure (`/mnt/code/scripts/flows.py`)

```python
@workflow
def oil_gas_automl_forecasting_workflow():
    # Step 1: Data Generation
    data_result = data_prep_task()  # Output: data_summary

    # Step 2: Parallel Training (all depend on data_result)
    autogluon_result = autogluon_task(data_prep=data_result["data_summary"])
    prophet_result = prophet_task(data_prep=data_result["data_summary"])
    nixtla_result = nixtla_task(data_prep=data_result["data_summary"])
    combined_result = combined_task(data_prep=data_result["data_summary"])
    # Each outputs: training_summary

    # Step 3: Model Comparison (sequential after training)
    comparison_result = comparison_task(
        autogluon_summary=autogluon_result["training_summary"],
        prophet_summary=prophet_result["training_summary"],
        nixtla_summary=nixtla_result["training_summary"],
        combined_summary=combined_result["training_summary"]
    )  # Output: comparison_results

    return comparison_result
```

#### Troubleshooting Guide

**"Sidecar uploader exited with code 1"**
- Check: Does script write to `/workflow/outputs/<declared_output_name>`?
- Check: Does error handler also write error output?
- Solution: Use `WorkflowIO.write_error_output()` in exception handlers

**"FileNotFoundError: production_timeseries.parquet"**
- Reason: Task running in isolated environment without shared data
- Check: Is auto-generation enabled? (Look for "Checking data availability..." in logs)
- Solution A: Current implementation auto-generates data
- Solution B: Set up Domino Dataset for better performance

**"Task fails but no clear error message"**
- Check: Does task declare outputs that don't get written?
- Check: Workflow output files in `/workflow/outputs/` directory
- Solution: Ensure ALL declared outputs are written in success AND error cases

**"Workflow is slow"**
- Reason: Auto-generation running 4× (once per training task)
- Solution: Set up Domino Dataset (Solution B above) to share data

#### Testing Flows Locally

```bash
# Test individual scripts standalone
python scripts/oil_gas_data_generator.py
python src/models/autogluon_forecasting.py
python src/models/prophet_forecasting.py
python src/models/nixtla_forecasting.py
python src/models/oil_gas_forecasting.py
python src/models/model_comparison.py

# All scripts detect standalone mode and work without workflow I/O
```

#### Summary of Fixes Applied

| Issue | Status | Files Modified |
|-------|--------|----------------|
| Sidecar uploader error | ✅ FIXED | 6 workflow scripts + workflow_io.py |
| Missing workflow outputs | ✅ FIXED | All training scripts |
| Data not persisting | ✅ FIXED | Added auto-generation + ensure_data.py |
| FlyteFile TypeVar missing | ✅ FIXED | flows.py (23 task definitions) |
| Inconsistent error handling | ✅ FIXED | All 6 workflow scripts |
| Data path configuration | ✅ FIXED | data_config.py + all scripts |

**Status**: All Domino Flows issues resolved. Workflow is production-ready and tested. ✅

## Technology Stack

- **Platform**: Domino Data Lab (MLOps orchestration)
- **Primary Language**: Python for all ML operations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Frameworks**: Streamlit (quick demos), Dash, Gradio, Panel, React/FastAPI
- **Experiment Tracking**: MLflow (built-in at http://localhost:8768)
- **Deployment**: FastAPI, Flask, Docker, Domino Flows, Domino Apps, Domino Model APIs

## Key Patterns

### Agent Coordination
- Use Master-Project-Manager-Agent for complete end-to-end workflows with governance orchestration
- Individual agents can work independently for specific tasks
- All agents use the standardized directory structure
- Production code goes in `src/`, scripts in `scripts/`, notebooks in `notebooks/`
- Dependencies managed in project-level `requirements.txt`
- Governance compliance is automatically assessed and integrated into workflows

### Governance Workflow Patterns
- **Intake Phase**: Business-Analyst-Agent identifies applicable governance frameworks
- **Development Phase**: Model-Validator-Agent ensures compliance with all frameworks
- **Deployment Phase**: MLOps-Engineer-Agent implements governance-compliant pipelines
- **Approval Workflow**: Master-Project-Manager-Agent coordinates multi-stage approvals
- **Continuous Compliance**: Ongoing monitoring and governance validation

### MLflow Integration
- All experiments, metrics, and artifacts are logged to MLflow
- Models are registered with signatures and input examples
- Parent-child run relationships track complex pipelines
- Comprehensive artifact tracking at each stage

### File Organization
- **Production code** → `/mnt/code/src/` (organized by function: api, models, monitoring, data)
- **Scripts** → `/mnt/code/scripts/` (utility scripts, frontend apps)
  - **data_config.py** → Utility for determining correct data storage paths
- **Notebooks** → `/mnt/code/notebooks/` (exploratory analysis)
- **Tests** → `/mnt/code/tests/` (all test files)
- **Configuration** → `/mnt/code/config/` (deployment configs, settings)
- **Documentation** → `/mnt/code/docs/` (including business analysis)
- **Artifacts** → Use `get_data_paths()['artifacts_path']` (Git: `/mnt/artifacts/`, DFS: `/mnt/`)
- **Data** → Use `get_data_paths()['base_data_path']` (Git: `/mnt/data/{project}/`, DFS: `/domino/datasets/local/{project}/`)

**IMPORTANT**: Always use `/mnt/code/scripts/data_config.py::get_data_paths()` to determine artifact and data paths dynamically based on project type.

## Project Development Workflow

### Starting a New Project

When you request ML project development, the **Master-Project-Manager-Agent** will ask you two questions:

**1. Domino Deployment Features:**

Which features would you like to create?
1. **Domino Flows** - Automated ML pipeline orchestration
2. **Domino Launchers** - Self-service parameter-driven execution
3. **Domino Model APIs (Endpoints)** - REST API for real-time predictions
4. **Domino Apps** - Interactive web applications (Streamlit/Dash/Gradio)

You can choose:
- **All** - Full deployment stack (Flows + Launchers + Endpoints + Apps)
- **Specific features** - e.g., "Just endpoints and apps"
- **None** - Model training and testing only

**2. Research Time Limit:**

How much time should the agent spend researching before presenting the plan?
- **Quick (2-3 minutes)** - Basic research, standard approach
- **Standard (5-7 minutes)** - Moderate research, best practices review (default)
- **Thorough (10-15 minutes)** - Deep research, comprehensive planning
- **Custom** - Specify your own time limit (e.g., "4 minutes")

The agent will:
1. Conduct research within your time limit
2. Present a project plan
3. Ask how you want to execute: **Step-by-step approval** or **Full automation**
4. Invoke only the relevant sub-agents based on your feature selection

**Execution Modes:**
- **Step-by-Step Approval**: Agent pauses before each major milestone (data generation, EDA, training, testing, deployment) and asks for approval
  - Best for: Learning, demos, quality control, customization
- **Full Automation**: Agent executes the entire plan without interruption
  - Best for: Speed, trusted workflows, standard implementations

## Common Usage Patterns

```python
# Full project with automation
"Build a customer churn prediction model"
→ Agent asks: Which features? How much research time?
→ You respond: "All features, standard research"
→ Agent: Conducts 5-7 min research, presents plan
→ Agent asks: Step-by-step or Automated?
→ You respond: "Automated"
→ Agent: Executes entire plan automatically
→ Creates: Flows, Launchers, Endpoint, and Dashboard

# Quick deployment with step-by-step review
"Create a credit risk model with API endpoint and dashboard"
→ Agent asks: Which features? How much research time?
→ You respond: "Endpoints and Apps, quick - 3 minutes"
→ Agent: Conducts 3 min research, presents plan
→ Agent asks: Step-by-step or Automated?
→ You respond: "Step-by-step - I want to review each stage"
→ Agent: Pauses before data generation, EDA, training, endpoint, dashboard
→ Creates: Model API endpoint + Streamlit dashboard with your approval at each step

# Thorough compliance project with automation
"Train a fraud detection model for a bank"
→ Agent asks: Which features? How much research time?
→ You respond: "None, just training. Thorough research for compliance"
→ Agent: Conducts 10-15 min deep research on regulations, presents plan
→ Agent asks: Step-by-step or Automated?
→ You respond: "Automated - plan looks good"
→ Agent: Executes training pipeline automatically
→ Creates: Model training + validation (skips all deployment)

# Governance-specific tasks
"Validate this model for compliance with model risk management framework"
"Generate governance compliance report for deployment approval"
"Assess this project for applicable governance frameworks and approval requirements"

# Specific tasks
"Generate synthetic financial data for fraud detection"
"Perform EDA on this dataset and create visualizations"
```

## Development Guidelines

- Always specify project names for proper organization
- Include business context for better agent recommendations
- Identify governance requirements early in project planning
- Test with small datasets before scaling
- Leverage Front-End-Developer-Agent's technology selection
- Use Model-Validator-Agent for governance and compliance validation
- Coordinate with appropriate approval groups based on project requirements
- Document compliance throughout the ML lifecycle

# Project Status and Recent Completion

## AutoML Forecasting System - COMPLETED ✅

**Latest Major Work Completed (November 2024):**

The Oil & Gas Analytics Platform now includes a comprehensive autoML forecasting comparison system with the following components:

### 1. AutoML Framework Implementation
- **AutoGluon TimeSeries**: `/mnt/code/src/models/autogluon_forecasting.py`
  - 5 standardized configurations (fast, medium, high_quality, best_quality, interpretable)
  - Time series predictor with multiple presets and time limits
  - MLflow tracking with best model tagging

- **Prophet/NeuralProphet**: `/mnt/code/src/models/prophet_forecasting.py`
  - Traditional Prophet and modern NeuralProphet models
  - 5 configurations testing different seasonality and trend parameters
  - Dual-mode execution (standalone and Domino Flows compatible)

- **Nixtla NeuralForecast**: `/mnt/code/src/models/nixtla_forecasting.py`
  - Advanced neural forecasting models (MLP, NBEATS, NHITS, LSTM, TFT)
  - Robust error handling with StatsForecast fallback
  - Individual model creation with comprehensive try/catch blocks

- **Combined LightGBM+ARIMA**: `/mnt/code/src/models/oil_gas_forecasting.py`
  - Existing ensemble model integrated into standardized system

### 2. Standardization and Fair Comparison
- **Centralized Configuration**: `/mnt/code/src/models/forecasting_config.py`
  - Uniform parameters across all frameworks for fair comparison
  - Standardized train/test splits (80/20)
  - Consistent MLflow tagging and parameter logging
  - Data quality validation checks

### 3. Orchestration and Model Selection
- **Model Comparison**: `/mnt/code/src/models/model_comparison.py`
  - Automatic champion model selection based on MAE
  - Domino Model Registry integration
  - Best model from each framework comparison
  - Comprehensive performance reporting

- **Domino Flows Orchestration**: `/mnt/code/scripts/flows.py`
  - Parallel execution of all 4 autoML frameworks
  - Sequential model comparison and champion selection
  - Side-effect free design with versioned inputs/outputs
  - Production forecasting and monitoring workflows

### 4. Project Structure Cleanup
- **Organized File Structure**: All scripts properly located in `/mnt/code/scripts/`
- **Model Artifacts**: Moved to `/mnt/artifacts/models/` (not in git)
- **Documentation**: Moved to `/mnt/code/docs/` folder
- **Clean Repository**: Removed temporary files (lightning_logs, old data folders)

### 5. MLflow Integration
- **Experiment Tracking**: All models log to shared 'oil_gas_forecasting_models' experiment
- **Child Runs**: Each configuration runs as child run with comprehensive metrics
- **Best Model Tagging**: Automatic tagging of best performer in each framework
- **Artifact Management**: Models, forecasts, and predictions saved and versioned

### 6. Key Features
- **Dual-Mode Scripts**: All scripts work standalone and with Domino Flows
- **Error Resilience**: Robust error handling especially for Nixtla neural models
- **Fair Comparison**: Standardized parameters ensure objective model comparison
- **Production Ready**: Champion model automatically registered for deployment

### Next Potential Steps
If continuing this work, consider:
1. **Model Deployment**: Create Domino Model API endpoints for champion model
2. **Monitoring Dashboard**: Build real-time forecasting performance dashboard
3. **Automated Retraining**: Set up scheduled model retraining pipelines
4. **A/B Testing**: Implement champion/challenger model testing framework

### Files Ready for Use
All autoML forecasting scripts are production-ready and can be executed individually or through the Domino Flows orchestration system at `/mnt/code/scripts/flows.py`. The standardized configuration ensures fair comparison and the MLflow integration provides comprehensive experiment tracking.