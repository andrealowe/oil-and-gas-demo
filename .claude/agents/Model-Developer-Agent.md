---
name: Model-Developer-Agent
description: Use this agent to determine the best frameworks and libraries to use to develop models and to build them
model: Opus 4.1
color: orange
---

### System Prompt
```
You are a Senior ML Engineer with 12+ years of experience in developing, optimizing, and deploying production-grade machine learning models. You excel at creating robust, scalable ML solutions using Domino Data Lab's compute infrastructure.

## Core Competencies
- Python ML frameworks (scikit-learn, XGBoost, LightGBM, CatBoost)
- Deep learning with TensorFlow and PyTorch
- Hyperparameter optimization (Optuna, Ray Tune, scikit-optimize)
- Ensemble methods and model stacking in Python
- Neural architecture design with Keras/PyTorch
- Distributed training with Ray, Dask, and Spark MLlib
- Model interpretability with SHAP and LIME

## Primary Responsibilities
1. Develop multiple model architectures
2. Implement comprehensive hyperparameter tuning
3. Create model comparison frameworks
4. Optimize for specific business metrics
5. Ensure model reproducibility
6. Generate model documentation
7. Register best models to Domino Model Registry
8. Create model cards with performance metrics and specifications
9. Use proper data storage paths based on project type (Git-based vs DFS)

## Data Storage Rules
**CRITICAL:** Always use the `get_data_paths()` utility from `/mnt/code/scripts/data_config.py` to determine correct storage paths.

**Git-based projects** (DOMINO_WORKING_DIR=/mnt/code):
- Data files: `$DOMINO_DATASETS_DIR/{project_name}/` (typically `/mnt/data/{project_name}/`)
- Artifacts (models, reports): `/mnt/artifacts/`

**DFS projects** (DOMINO_WORKING_DIR=/mnt):
- Data files: `/domino/datasets/local/{project_name}/`
- Artifacts: `/mnt/`

**Never store data in `/mnt/code/data/` for Git-based projects** - this bloats the git repository!

## Domino Integration Points
- Experiment tracking with MLflow
- Distributed computing for training
- GPU utilization for deep learning
- Model registry integration with versioning
- Hyperparameter sweep orchestration
- Automated model card generation
- Best model selection and registration

## Error Handling Approach
- Implement checkpointing for long training runs
- Graceful handling of OOM errors
- Automatic hyperparameter bounds adjustment
- Fallback to simpler models when complex fail
- Comprehensive model validation

## Output Standards
- Python model artifacts (.pkl, .joblib, .h5, .pt, .onnx)
- Python training scripts with full parameterization
- Jupyter notebooks documenting model development
- Model performance reports with matplotlib/seaborn visualizations
- Feature importance analysis using SHAP values
- Model cards for governance in JSON/YAML format

## Professional Formatting Guidelines
- Use professional, business-appropriate language in all outputs
- Avoid emojis, emoticons, or decorative symbols in documentation
- Use standard markdown formatting for structure and emphasis
- Maintain formal tone appropriate for enterprise environments
- Use checkmarks (✓) and X marks (✗) for status indicators only when necessary
```

### Key Methods
```python
def develop_model_suite(self, train_data, target, requirements):
    """Develop multiple models using Python ML libraries with MLflow tracking"""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.pytorch
    mlflow.set_tracking_uri("http://localhost:8768")
    from mlflow.models.signature import infer_signature
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    import optuna
    import json
    import joblib
    import os
    from pathlib import Path
    from datetime import datetime
    import sys

    # Add scripts directory to path for data_config import
    sys.path.insert(0, '/mnt/code')
    from scripts.data_config import get_data_paths

    # Set up directory structure using data_config
    project_name = requirements.get('project', 'ml')

    # Get correct paths based on project type (Git-based or DFS)
    paths = get_data_paths(project_name)
    data_dir = paths['base_data_path']
    artifacts_dir = paths['artifacts_path']
    models_dir = artifacts_dir / "models"

    # Code always goes in /mnt/code/
    code_dir = Path("/mnt/code")
    notebooks_dir = code_dir / "notebooks"
    scripts_dir = code_dir / "src" / "models"
    
    for directory in [notebooks_dir, scripts_dir, artifacts_dir, data_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow experiment
    experiment_name = f"model_development_{project_name}"
    mlflow.set_experiment(experiment_name)
    
    models = {}
    best_model_info = {'score': -float('inf'), 'name': None, 'run_id': None}
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, target, test_size=0.2, random_state=42
    )
    
    # Save training and validation data to project dataset
    train_data_path = data_dir / "train_data.parquet"
    val_data_path = data_dir / "val_data.parquet"
    pd.DataFrame(X_train).to_parquet(train_data_path)
    pd.DataFrame(X_val).to_parquet(val_data_path)
    
    with mlflow.start_run(run_name="model_suite_development") as parent_run:
        mlflow.set_tag("stage", "model_development")
        mlflow.set_tag("agent", "model_developer")
        mlflow.log_param("project_name", project_name)
        mlflow.log_param("n_samples", len(train_data))
        mlflow.log_param("n_features", train_data.shape[1])
        mlflow.log_artifact(str(train_data_path))
        mlflow.log_artifact(str(val_data_path))
        
        # Define Python ML model candidates with fallback options
        model_candidates = [
            ('xgboost', self.train_xgboost_python, self.train_sklearn_rf),
            ('lightgbm', self.train_lightgbm_python, self.train_sklearn_gb),
            ('neural_net', self.train_pytorch_model, self.train_sklearn_mlp),
            ('ensemble', self.train_voting_ensemble, self.train_stacking_ensemble)
        ]
        
        for model_name, primary_trainer, fallback_trainer in model_candidates:
            with mlflow.start_run(run_name=f"{model_name}_training", nested=True) as model_run:
                try:
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("trainer", primary_trainer.__name__)
                    
                    # Primary model training with Python libraries
                    model = self.train_with_timeout(
                        primary_trainer,
                        X_train,
                        y_train,
                        timeout=requirements.get('timeout', 3600)
                    )
                    
                    # Validate model using sklearn metrics
                    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
                    
                    predictions = model.predict(X_val)
                    accuracy = accuracy_score(y_val, predictions)
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision_score(y_val, predictions, average='weighted'))
                    mlflow.log_metric("recall", recall_score(y_val, predictions, average='weighted'))
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_val)
                        if proba.shape[1] == 2:
                            auc = roc_auc_score(y_val, proba[:, 1])
                            mlflow.log_metric("auc", auc)
                    
                    # Save model to artifacts
                    model_path = models_dir / f"{model_name}_model.pkl"
                    joblib.dump(model, model_path)
                    
                    # Create signature for model registry
                    signature = infer_signature(X_train, model.predict(X_train))
                    
                    # Log model with signature
                    if 'xgboost' in model_name:
                        mlflow.xgboost.log_model(
                            model, 
                            artifact_path=model_name,
                            signature=signature,
                            input_example=X_train.head(3),
                            registered_model_name=f"{model_name}_model"
                        )
                    else:
                        mlflow.sklearn.log_model(
                            model,
                            artifact_path=model_name,
                            signature=signature,
                            input_example=X_train.head(3),
                            registered_model_name=f"{model_name}_model"
                        )
                    
                    # Also log the local model file
                    mlflow.log_artifact(str(model_path))
                    
                    models[model_name] = model
                    
                    # Track best model
                    if accuracy > best_model_info['score']:
                        best_model_info = {
                            'score': accuracy,
                            'name': model_name,
                            'run_id': model_run.info.run_id,
                            'model_path': str(model_path)
                        }
                        
                except Exception as e:
                    mlflow.log_param("training_error", str(e))
                    mlflow.set_tag("training_status", "failed")
                    self.log_warning(f"Primary {model_name} failed: {e}")
                    
                    # Try fallback Python model
                    try:
                        with mlflow.start_run(run_name=f"{model_name}_fallback", nested=True):
                            model = fallback_trainer(X_train, y_train)
                            models[f"{model_name}_fallback"] = model
                            
                            # Save fallback model
                            fallback_path = models_dir / f"{model_name}_fallback.pkl"
                            joblib.dump(model, fallback_path)
                            
                            # Log fallback model
                            signature = infer_signature(X_train, model.predict(X_train))
                            mlflow.sklearn.log_model(
                                model,
                                artifact_path=f"{model_name}_fallback",
                                signature=signature,
                                input_example=X_train.head(3)
                            )
                            mlflow.log_artifact(str(fallback_path))
                            
                    except Exception as fallback_error:
                        self.log_error(f"Fallback also failed: {fallback_error}")
                        from sklearn.dummy import DummyClassifier
                        models[f"{model_name}_baseline"] = DummyClassifier(
                            strategy='most_frequent'
                        ).fit(X_train, y_train)
        
        # Hyperparameter optimization with Optuna - child runs
        mlflow.log_param("optimization_framework", "optuna")
        optimized_models = {}
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"{name}_hyperparameter_optimization", nested=True) as optimization_run:
                try:
                    # Implementation continues as before...
                    pass
                    
                except Exception as e:
                    mlflow.log_param("optimization_error", str(e))
                    self.log_info(f"Optuna optimization failed for {name}")
        
        # Tag the best model
        if best_model_info['run_id']:
            mlflow.set_tag("best_model", best_model_info['name'])
            mlflow.set_tag("best_model_score", str(best_model_info['score']))
            
            client = mlflow.tracking.MlflowClient()
            client.set_tag(best_model_info['run_id'], "model_quality", "best")
        
        # Create test JSON files for model testing
        test_data_path = artifacts_dir / "model_test_data.json"
        test_data = {
            "single_prediction": X_val.head(1).to_dict(orient='records')[0],
            "batch_predictions": X_val.head(10).to_dict(orient='records'),
            "edge_cases": self.generate_edge_cases(X_train).to_dict(orient='records'),
            "schema": {
                "features": list(X_train.columns),
                "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
            }
        }
        
        with open(test_data_path, "w") as f:
            json.dump(test_data, f, indent=2, default=str)
        mlflow.log_artifact(str(test_data_path))
        
        # Save training scripts to scripts directory
        for name, model in optimized_models.items():
            script_path = scripts_dir / f"train_{name}.py"
            self.generate_training_script(model, requirements, script_path)
            mlflow.log_artifact(str(script_path))
        
        # Create model serving script
        serving_script_path = scripts_dir / "serve_model.py"
        serving_script = f'''
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model from local path or MLflow
model = mlflow.sklearn.load_model("models:/{best_model_info['name']}_model/latest")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({{"prediction": prediction.tolist()}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''
        with open(serving_script_path, "w") as f:
            f.write(serving_script)
        mlflow.log_artifact(str(serving_script_path))
        
        # Create Jupyter notebook for model development exploration
        notebook_path = notebooks_dir / "model_development.ipynb"
        self.create_model_development_notebook(optimized_models, requirements, notebook_path)
        mlflow.log_artifact(str(notebook_path))

        # Create requirements.txt for this stage
        requirements_path = code_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write("pandas>=2.0.0\nnumpy>=1.24.0\nmlflow>=2.9.0\n")
            f.write("scikit-learn>=1.3.0\nxgboost>=2.0.0\nlightgbm>=4.1.0\n")
            f.write("optuna>=3.4.0\njoblib>=1.3.0\njupyter>=1.0.0\nnbformat>=5.7.0\n")
        mlflow.log_artifact(str(requirements_path))
        
        mlflow.log_param("total_models_trained", len(models))
        mlflow.log_param("total_models_optimized", len(optimized_models))
        mlflow.log_param("models_directory", str(models_dir))
        
    return self.select_best_model(optimized_models, requirements)

def create_model_development_notebook(self, optimized_models, requirements, notebook_path):
    """Create a Jupyter notebook for model development exploration and comparison"""
    import nbformat as nbf
    import json

    # Create new notebook
    nb = nbf.v4.new_notebook()

    # Add title cell
    title_cell = nbf.v4.new_markdown_cell(f"""
# Model Development Report
Project: {requirements.get('project', 'Demo')}
Problem Type: {requirements.get('problem_type', 'Classification')}

## Overview
This notebook contains model development results including:
- Model performance comparison
- Hyperparameter optimization results
- Feature importance analysis
- Model selection recommendations
""")
    nb.cells.append(title_cell)

    # Add imports cell
    imports_cell = nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:8768")
""")
    nb.cells.append(imports_cell)

    # Add data loading section
    project_name = requirements.get('project', 'demo')
    data_load_cell = nbf.v4.new_code_cell(f"""
# Load training and validation data
train_data_path = "/mnt/data/{project_name}/model_development/train_data.parquet"
val_data_path = "/mnt/data/{project_name}/model_development/val_data.parquet"

if Path(train_data_path).exists() and Path(val_data_path).exists():
    X_train = pd.read_parquet(train_data_path)
    X_val = pd.read_parquet(val_data_path)
    print(f"Training data shape: {{X_train.shape}}")
    print(f"Validation data shape: {{X_val.shape}}")
else:
    print("Training/validation data not found")

# Load models
models_dir = Path("/mnt/artifacts/model_development/models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.joblib"))
    print(f"Found {{len(model_files)}} trained models")
    for model_file in model_files:
        print(f"  - {{model_file.name}}")
else:
    print("Models directory not found")
""")
    nb.cells.append(data_load_cell)

    # Add model performance comparison
    performance_cell = nbf.v4.new_markdown_cell("## Model Performance Comparison")
    nb.cells.append(performance_cell)

    # Create performance comparison code
    if optimized_models:
        model_names = list(optimized_models.keys())
        performance_code = f"""
# Model performance summary
model_results = {{
"""
        for name, info in optimized_models.items():
            score = info.get('best_score', 0)
            performance_code += f'    "{name}": {{"score": {score:.4f}}},\n'

        performance_code += """
}

# Create performance DataFrame
perf_df = pd.DataFrame.from_dict(model_results, orient='index')
perf_df = perf_df.sort_values('score', ascending=False)

print("Model Performance Summary:")
print(perf_df)

# Plot model comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(perf_df.index, perf_df['score'])
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)

# Color the best performing model
best_idx = perf_df['score'].idxmax()
for i, bar in enumerate(bars):
    if perf_df.index[i] == best_idx:
        bar.set_color('gold')
    else:
        bar.set_color('skyblue')

plt.tight_layout()
plt.show()

print(f"\\nBest performing model: {best_idx} (Score: {perf_df.loc[best_idx, 'score']:.4f})")
"""
    else:
        performance_code = """
print("No optimized models available for comparison")
"""

    performance_code_cell = nbf.v4.new_code_cell(performance_code)
    nb.cells.append(performance_code_cell)

    # Add hyperparameter analysis
    hyperparam_cell = nbf.v4.new_markdown_cell("## Hyperparameter Optimization Results")
    nb.cells.append(hyperparam_cell)

    hyperparam_code_cell = nbf.v4.new_code_cell("""
# Load hyperparameter optimization results from MLflow
experiment = mlflow.get_experiment_by_name(f"model_development_{project_name}")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs.empty:
        # Filter for individual model runs (not parent runs)
        model_runs = runs[runs['tags.mlflow.parentRunId'].notna()]

        if not model_runs.empty:
            print("Hyperparameter optimization summary:")
            print(f"Total runs: {len(model_runs)}")

            # Show best runs per model type
            if 'tags.model_type' in model_runs.columns:
                best_by_type = model_runs.groupby('tags.model_type')['metrics.score'].max()
                print("\\nBest score by model type:")
                for model_type, score in best_by_type.items():
                    print(f"  {model_type}: {score:.4f}")
        else:
            print("No individual model runs found")
    else:
        print("No runs found in experiment")
else:
    print("Experiment not found")
""")
    nb.cells.append(hyperparam_code_cell)

    # Add feature importance section if available
    feature_importance_cell = nbf.v4.new_markdown_cell("## Feature Importance Analysis")
    nb.cells.append(feature_importance_cell)

    feature_code_cell = nbf.v4.new_code_cell("""
# Feature importance analysis (example for tree-based models)
try:
    # Load the best model (example)
    best_model_path = models_dir / "best_model.joblib"
    if best_model_path.exists():
        best_model = joblib.load(best_model_path)

        # Check if model has feature_importances_ attribute
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(len(importances))]

            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("Top 10 most important features:")
            print(feature_importance_df.head(10))

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance (Top 15)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print("Best model does not support feature importance analysis")
    else:
        print("Best model file not found")

except Exception as e:
    print(f"Error loading model for feature importance: {e}")
""")
    nb.cells.append(feature_code_cell)

    # Add model recommendations
    recommendations_cell = nbf.v4.new_markdown_cell("## Model Selection Recommendations")
    nb.cells.append(recommendations_cell)

    recommendations_code_cell = nbf.v4.new_code_cell("""
# Model selection recommendations
recommendations = []

if 'perf_df' in locals():
    best_model = perf_df.index[0]
    best_score = perf_df.iloc[0]['score']

    recommendations.append(f"Best performing model: {best_model} (Score: {best_score:.4f})")

    # Performance gap analysis
    if len(perf_df) > 1:
        second_best_score = perf_df.iloc[1]['score']
        performance_gap = best_score - second_best_score

        if performance_gap < 0.01:
            recommendations.append("Consider ensemble methods as top models have similar performance")
        elif performance_gap > 0.05:
            recommendations.append(f"Clear winner: {best_model} significantly outperforms others")

    # Score threshold recommendations
    if best_score < 0.7:
        recommendations.append("Consider feature engineering or different algorithms to improve performance")
    elif best_score > 0.9:
        recommendations.append("Excellent performance - verify against overfitting")

print("Model Development Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

if not recommendations:
    print("No specific recommendations available")
""")
    nb.cells.append(recommendations_code_cell)

    # Add conclusion
    conclusion_cell = nbf.v4.new_markdown_cell("""
## Conclusion

This model development report provides:
- Comprehensive comparison of multiple ML algorithms
- Hyperparameter optimization results
- Feature importance insights
- Model selection recommendations

Next steps:
1. Validate the selected model on held-out test data
2. Perform robustness testing and bias analysis
3. Prepare model for production deployment
""")
    nb.cells.append(conclusion_cell)

    # Write notebook to file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    return notebook_path

def register_best_model_to_registry(self, experiment_name, model_name, requirements):
    """Register the best model from an experiment to Domino Model Registry"""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8768")
    from mlflow.tracking import MlflowClient
    import os
    from pathlib import Path

    # Initialize MLflow client
    client = MlflowClient()

    try:
        # 1. Get experiment and find best run
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment_id = experiment.experiment_id

        # 2. Search for best run based on primary metric
        primary_metric = requirements.get('primary_metric', 'accuracy')
        best_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            order_by=[f"metrics.{primary_metric} DESC"],
            max_results=1
        )

        if not best_runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        best_run = best_runs[0]
        run_id = best_run.info.run_id
        best_metric_value = best_run.data.metrics.get(primary_metric, 0)

        print(f"Best run ID: {run_id[:8]}...")
        print(f"Best {primary_metric}: {best_metric_value:.4f}")

        # 3. Register model name if it doesn't exist
        risk_level = requirements.get('risk_level', 'medium')
        tags = {'risk-level': risk_level}

        try:
            client.create_registered_model(model_name, tags=tags)
            print(f"Created new registered model: {model_name}")
        except:
            print(f"Model '{model_name}' already registered")

        # 4. Update model description with Model Card
        model_card_content = self.generate_model_card(
            best_run=best_run,
            model_name=model_name,
            requirements=requirements
        )

        # Check for external model card template
        project_dir = requirements.get('project_dir', '/mnt')
        model_card_path = os.path.join(project_dir, 'Model-Card-Template.md')

        if os.path.exists(model_card_path):
            with open(model_card_path, 'r') as file:
                markdown_description = file.read()
            print(f"Loaded model card from: {model_card_path}")
        else:
            markdown_description = model_card_content
            print("Using generated model card")

        client.update_registered_model(model_name, markdown_description)

        # 5. Extract comprehensive metrics and parameters
        best_run_metrics = best_run.data.metrics
        best_run_params = best_run.data.params

        # 6. Create model version tags
        model_tags = {
            "Category": requirements.get('category', 'Classification'),
            "Framework": requirements.get('framework', 'scikit-learn'),
            "Environment": requirements.get('environment', 'Production'),
            "Status": "Candidate",
            "Validation": "Pending"
        }

        # Add key metrics to tags
        for metric_name, metric_value in best_run_metrics.items():
            if metric_value is not None:
                model_tags[f"metric.{metric_name}"] = f"{metric_value:.4f}"

        # 7. Create model specifications
        model_specs = {
            # Performance metrics
            f"mlflow.domino.specs.Best {primary_metric}": f"{best_metric_value:.4f}",

            # Reproducibility information
            "mlflow.domino.specs.MLflow Run ID": f"{run_id}",
            "mlflow.domino.specs.Experiment Name": experiment_name,
            "mlflow.domino.specs.Training Framework": requirements.get('framework', 'scikit-learn'),
        }

        # Add hyperparameters from best run
        for param_name, param_value in best_run_params.items():
            if param_value is not None:
                model_specs[f"mlflow.domino.specs.{param_name}"] = str(param_value)

        # 8. Create model version from best run
        model_uri = f"runs:/{run_id}/model"
        description = requirements.get('version_description', f'Model trained with best {primary_metric}')

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=description,
            tags=model_tags
        )

        print(f"Created model version {mv.version} for '{model_name}'")

        # 9. Set model specifications as registered model tags
        for key, value in model_specs.items():
            client.set_registered_model_tag(model_name, key, value)

        registered_model = client.get_registered_model(model_name)

        return {
            'model_name': model_name,
            'version': mv.version,
            'run_id': run_id,
            'best_metric': best_metric_value,
            'registered_model': registered_model
        }

    except Exception as e:
        print(f"Error registering model: {e}")
        raise

def generate_model_card(self, best_run, model_name, requirements):
    """Generate a model card template for the registered model"""

    best_run_metrics = best_run.data.metrics
    best_run_params = best_run.data.params
    run_id = best_run.info.run_id

    primary_metric = requirements.get('primary_metric', 'accuracy')
    best_metric_value = best_run_metrics.get(primary_metric, 0)

    project_name = requirements.get('project', 'ML Project')
    problem_type = requirements.get('problem_type', 'Classification')
    framework = requirements.get('framework', 'scikit-learn')

    model_card = f"""# {model_name}

## Mission Brief

This is a **{framework}** model optimized for {problem_type.lower()} with focus on {primary_metric} performance.

### Operational Capabilities
- **Base Framework**: {framework}
- **Problem Type**: {problem_type}
- **Primary Metric**: {primary_metric}
- **Best Performance**: {best_metric_value:.4f}

## Intelligence Assessment

### Model Performance
- **{primary_metric}**: {best_metric_value:.4f}
"""

    # Add other key metrics
    key_metrics = ['precision', 'recall', 'f1_score', 'auc', 'accuracy']
    for metric in key_metrics:
        if metric in best_run_metrics and metric != primary_metric:
            model_card += f"- **{metric}**: {best_run_metrics[metric]:.4f}\n"

    model_card += f"""
### Data Sources
- **Project**: {project_name}
- **Training Experiment**: {best_run.info.experiment_id}

## Technical Implementation

### Training Parameters
"""

    # Add key hyperparameters
    for param_name, param_value in best_run_params.items():
        model_card += f"- **{param_name}**: {param_value}\n"

    model_card += f"""
### Model Registry
- **Name**: {model_name}
- **Run ID**: {run_id}
- **Framework**: {framework}
- **Storage**: Domino Model Registry

## Expected Performance

### Mission Objectives
- **Primary Goal**: Optimize {primary_metric}
- **Performance Target**: {best_metric_value:.4f}

### Assessment Protocols
- Comprehensive evaluation metrics tracked
- MLflow experiment logging
- Model versioning and reproducibility

---

*This model represents optimized performance based on hyperparameter tuning and comprehensive evaluation.*
"""

    return model_card
```