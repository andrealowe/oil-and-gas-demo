#!/usr/bin/env python3
"""
Model Evaluation and Registry Management for Oil & Gas Geospatial Models
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML evaluation libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Add scripts directory to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:8768")

class ModelEvaluator:
    """Comprehensive model evaluation and registry management"""
    
    def __init__(self, project_name="Oil-and-Gas-Demo"):
        self.project_name = project_name
        self.client = MlflowClient()
        
        # Get correct paths
        paths = get_data_paths(project_name)
        self.artifacts_dir = paths['artifacts_path']
        self.models_dir = self.artifacts_dir / "models"
        self.reports_dir = self.artifacts_dir / "reports"
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set experiment
        self.experiment_name = "oil_gas_geospatial_models"
        mlflow.set_experiment(self.experiment_name)
    
    def load_test_data(self, data_path):
        """Load and prepare test data"""
        df = pd.read_parquet(data_path)
        
        # Apply same preprocessing as training
        df = self._engineer_features(df)
        df = self._handle_missing_values(df)
        
        return df
    
    def _engineer_features(self, df):
        """Apply same feature engineering as training"""
        df = df.copy()
        
        # Create time-based features
        df['days_since_maintenance'] = df['days_since_maintenance'].fillna(0)
        df['maintenance_overdue'] = (df['days_since_maintenance'] > 90).astype(int)
        
        # Geographic clustering features
        df['latitude_bin'] = pd.cut(df['latitude'], bins=10, labels=False)
        df['longitude_bin'] = pd.cut(df['longitude'], bins=10, labels=False)
        
        # Production efficiency features
        df['production_ratio'] = df['oil_production_bpd'] / (df['gas_production_mcfd'] + 1)
        df['utilization_efficiency'] = df['utilization_rate'] * df['equipment_health_score']
        
        # Environmental stress indicators
        df['environmental_stress'] = (
            df['h2s_concentration_ppm'].fillna(0) * 0.3 +
            df['co2_concentration_ppm'].fillna(0) * 0.2 +
            df['noise_level_db'].fillna(0) * 0.1 +
            df['vibration_level_mm_s'].fillna(0) * 0.4
        )
        
        # Age and depth interactions
        df['age_depth_interaction'] = df['well_age_years'] * df['well_depth_ft'] / 1000
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values consistently with training"""
        df = df.copy()
        
        # Fill production-related missing values
        production_cols = [
            'oil_production_bpd', 'gas_production_mcfd', 'gasoline_production_bpd',
            'diesel_production_bpd', 'jet_fuel_production_bpd', 'other_products_bpd'
        ]
        
        for col in production_cols:
            df[col] = df[col].fillna(0)
        
        # Fill technical parameters with median values by facility type
        technical_cols = [
            'well_depth_ft', 'well_age_years', 'vibration_level_mm_s',
            'temperature_celsius', 'pressure_psi', 'h2s_concentration_ppm',
            'co2_concentration_ppm', 'noise_level_db'
        ]
        
        for col in technical_cols:
            df[col] = df.groupby('facility_type')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Fill capacity and throughput
        capacity_cols = ['capacity_bpd', 'current_throughput_bpd', 'capacity', 'current_level', 'throughput']
        for col in capacity_cols:
            df[col] = df.groupby('facility_type')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Fill utilization rate
        df['utilization_rate'] = df['utilization_rate'].fillna(0.5)
        
        # Fill energy and environmental metrics
        df['energy_consumption_mwh'] = df['energy_consumption_mwh'].fillna(df['energy_consumption_mwh'].median())
        df['co2_emissions_tons_day'] = df['co2_emissions_tons_day'].fillna(df['co2_emissions_tons_day'].median())
        df['water_usage_gallons_day'] = df['water_usage_gallons_day'].fillna(df['water_usage_gallons_day'].median())
        
        return df
    
    def evaluate_model(self, model_path, model_name, df, target_info):
        """Evaluate a single model comprehensively"""
        print(f"Evaluating {model_name} model...")
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare features based on model type
        target_col = target_info['target_col']
        model_type = target_info['model_type']
        
        if model_name == 'production_efficiency':
            df_work = df[df['utilization_rate'].notna()].copy()
            df_work['production_efficiency_class'] = pd.cut(
                df_work['utilization_rate'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            target_df = df_work
        elif model_name == 'environmental_risk':
            df_work = df.copy()
            df_work['environmental_risk_class'] = pd.cut(
                df_work['environmental_risk'],
                bins=[0, 0.33, 0.66, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            target_df = df_work
        else:
            target_df = df
        
        # Prepare features
        numeric_features = [
            'latitude', 'longitude', 'oil_production_bpd', 'gas_production_mcfd',
            'well_depth_ft', 'well_age_years', 'vibration_level_mm_s',
            'temperature_celsius', 'pressure_psi', 'h2s_concentration_ppm',
            'co2_concentration_ppm', 'noise_level_db', 'capacity_bpd',
            'current_throughput_bpd', 'utilization_rate', 'days_since_maintenance',
            'production_ratio', 'environmental_stress', 'age_depth_interaction'
        ]
        
        categorical_features = [
            'facility_type', 'region', 'status', 'latitude_bin', 'longitude_bin'
        ]
        
        # Filter existing features
        numeric_features = [f for f in numeric_features if f in target_df.columns]
        categorical_features = [f for f in categorical_features if f in target_df.columns]
        
        # Prepare X and y
        X = target_df[numeric_features + categorical_features].copy()
        y = target_df[target_col].copy()
        
        # Remove rows where target is missing
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print(f"No valid data for {model_name} evaluation")
            return None
        
        # Make predictions
        try:
            y_pred = model.predict(X)
            
            # Calculate metrics
            if model_type == 'regression':
                metrics = {
                    'r2_score': r2_score(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred)
                }
                
                print(f"  R2 Score: {metrics['r2_score']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                
            else:  # classification
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                }
                
                # Calculate per-class metrics
                precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
                metrics.update({
                    'precision_weighted': precision,
                    'recall_weighted': recall,
                    'f1_weighted': f1
                })
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
                print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
                print(f"  F1-score (weighted): {metrics['f1_weighted']:.4f}")
            
            # Generate feature importance if available
            feature_importance = self._get_feature_importance(model, X.columns)
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def _get_feature_importance(self, model, feature_columns):
        """Extract feature importance from trained model"""
        try:
            # Get the actual model from pipeline
            actual_model = model.named_steps['model']
            
            # Get feature importance
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
                
                # Get feature names after preprocessing
                preprocessor = model.named_steps['preprocessor']
                feature_names_transformed = []
                
                # Add numeric features
                numeric_features = preprocessor.transformers_[0][2]
                feature_names_transformed.extend(numeric_features)
                
                # Add categorical features (one-hot encoded)
                if len(preprocessor.transformers_) > 1:
                    categorical_transformer = preprocessor.transformers_[1][1]
                    if hasattr(categorical_transformer, 'get_feature_names_out'):
                        cat_features = categorical_transformer.get_feature_names_out()
                        feature_names_transformed.extend(cat_features)
                
                # Create importance dictionary
                importance_dict = {}
                for i, importance in enumerate(importances):
                    if i < len(feature_names_transformed):
                        importance_dict[feature_names_transformed[i]] = importance
                
                # Return top 10 features
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_importance[:10])
                
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None
    
    def create_evaluation_report(self, evaluation_results):
        """Create comprehensive evaluation report"""
        report = {
            'project': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'models': evaluation_results
        }
        
        # Save report
        report_path = self.reports_dir / 'model_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {report_path}")
        return report_path
    
    def register_models_to_registry(self, evaluation_results):
        """Register best models to MLflow Model Registry"""
        print("Registering models to MLflow Model Registry...")
        
        registered_models = {}
        
        for result in evaluation_results:
            if result is None:
                continue
                
            model_name = result['model_name']
            model_type = result['model_type']
            metrics = result['metrics']
            
            # Determine primary metric
            if model_type == 'regression':
                primary_metric = 'r2_score'
                metric_value = metrics['r2_score']
            else:
                primary_metric = 'accuracy'
                metric_value = metrics['accuracy']
            
            try:
                # Get the best run for this model from MLflow
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model_name}_model'",
                    order_by=[f"metrics.best_{primary_metric} DESC"],
                    max_results=1
                )
                
                if runs:
                    best_run = runs[0]
                    run_id = best_run.info.run_id
                    
                    # Create registered model name
                    registry_model_name = f"oil_gas_{model_name}_model"
                    
                    # Register model
                    try:
                        self.client.create_registered_model(
                            registry_model_name,
                            tags={'project': self.project_name, 'type': model_type}
                        )
                    except Exception:
                        pass  # Model already exists
                    
                    # Create model version
                    model_uri = f"runs:/{run_id}/{model_name}"
                    
                    model_version = self.client.create_model_version(
                        name=registry_model_name,
                        source=model_uri,
                        run_id=run_id,
                        description=f"Oil & Gas {model_name} model - {model_type}"
                    )
                    
                    # Add model version tags
                    version_tags = {
                        f'metric.{primary_metric}': f'{metric_value:.4f}',
                        'environment': 'production',
                        'stage': 'candidate',
                        'model_type': model_type
                    }
                    
                    for tag_key, tag_value in version_tags.items():
                        self.client.set_model_version_tag(
                            registry_model_name,
                            model_version.version,
                            tag_key,
                            tag_value
                        )
                    
                    # Generate model card
                    model_card = self._generate_model_card(
                        model_name, model_type, metrics, result.get('feature_importance', {})
                    )
                    
                    # Update model description
                    self.client.update_registered_model(
                        registry_model_name,
                        description=model_card
                    )
                    
                    registered_models[model_name] = {
                        'registry_name': registry_model_name,
                        'version': model_version.version,
                        'run_id': run_id,
                        'primary_metric': metric_value
                    }
                    
                    print(f"  Registered {registry_model_name} v{model_version.version}")
                    
            except Exception as e:
                print(f"Error registering {model_name}: {e}")
        
        return registered_models
    
    def _generate_model_card(self, model_name, model_type, metrics, feature_importance):
        """Generate model card for governance"""
        
        if model_type == 'regression':
            performance_section = f"""
### Performance Metrics
- **R2 Score**: {metrics['r2_score']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **MAE**: {metrics['mae']:.4f}
"""
        else:
            performance_section = f"""
### Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision (weighted)**: {metrics.get('precision_weighted', 0):.4f}
- **Recall (weighted)**: {metrics.get('recall_weighted', 0):.4f}
- **F1-score (weighted)**: {metrics.get('f1_weighted', 0):.4f}
"""
        
        feature_importance_section = ""
        if feature_importance:
            feature_importance_section = "\n### Top Features\n"
            for feature, importance in list(feature_importance.items())[:5]:
                feature_importance_section += f"- **{feature}**: {importance:.4f}\n"
        
        model_card = f"""# Oil & Gas {model_name.replace('_', ' ').title()} Model

## Model Overview
- **Model Type**: {model_type.title()}
- **Purpose**: Geospatial Oil & Gas facility analysis
- **Domain**: Energy sector operations
- **Framework**: Scikit-learn with preprocessing pipeline

{performance_section}

## Use Cases
- Real-time dashboard integration
- Facility monitoring and optimization
- Risk assessment and management
- Operational decision support

{feature_importance_section}

## Deployment Notes
- Requires preprocessing pipeline
- Suitable for batch and real-time inference
- Input validation recommended
- Monitor for data drift

## Governance
- **Risk Level**: Medium
- **Validation Status**: Tested
- **Approval Status**: Pending
- **Review Date**: {datetime.now().strftime('%Y-%m-%d')}
"""
        return model_card
    
    def run_complete_evaluation(self, data_path):
        """Run complete evaluation pipeline"""
        print("Oil & Gas Model Evaluation Pipeline")
        print("=" * 50)
        
        # Load data
        df = self.load_test_data(data_path)
        print(f"Loaded evaluation data: {df.shape}")
        
        # Define model configurations
        model_configs = [
            {
                'model_name': 'equipment_health',
                'model_path': self.models_dir / 'equipment_health_model.pkl',
                'target_info': {
                    'target_col': 'equipment_health_score',
                    'model_type': 'regression'
                }
            },
            {
                'model_name': 'production_efficiency',
                'model_path': self.models_dir / 'production_efficiency_model.pkl',
                'target_info': {
                    'target_col': 'production_efficiency_class',
                    'model_type': 'classification'
                }
            },
            {
                'model_name': 'environmental_risk',
                'model_path': self.models_dir / 'environmental_risk_model.pkl',
                'target_info': {
                    'target_col': 'environmental_risk_class',
                    'model_type': 'classification'
                }
            }
        ]
        
        # Evaluate all models
        evaluation_results = []
        for config in model_configs:
            if config['model_path'].exists():
                result = self.evaluate_model(
                    config['model_path'],
                    config['model_name'],
                    df,
                    config['target_info']
                )
                evaluation_results.append(result)
            else:
                print(f"Model not found: {config['model_path']}")
        
        # Create evaluation report
        report_path = self.create_evaluation_report(evaluation_results)
        
        # Register models to registry
        registered_models = self.register_models_to_registry(evaluation_results)
        
        print("\nModel Evaluation Summary:")
        print("-" * 30)
        for result in evaluation_results:
            if result:
                model_name = result['model_name']
                model_type = result['model_type']
                
                if model_type == 'regression':
                    metric = result['metrics']['r2_score']
                    print(f"{model_name}: R2 = {metric:.4f}")
                else:
                    metric = result['metrics']['accuracy']
                    print(f"{model_name}: Accuracy = {metric:.4f}")
        
        print(f"\nRegistered {len(registered_models)} models to MLflow Model Registry")
        print(f"Evaluation report: {report_path}")
        
        return evaluation_results, registered_models

def main():
    """Main function"""
    evaluator = ModelEvaluator("Oil-and-Gas-Demo")
    paths = get_data_paths("Oil-and-Gas-Demo")
    data_path = paths['base_data_path'] / "prepared_geospatial_data.parquet"

    evaluation_results, registered_models = evaluator.run_complete_evaluation(str(data_path))
    
    # Print registry information
    print("\nMLflow Model Registry Information:")
    for model_name, info in registered_models.items():
        print(f"  {info['registry_name']} v{info['version']} - {info['primary_metric']:.4f}")

if __name__ == "__main__":
    main()