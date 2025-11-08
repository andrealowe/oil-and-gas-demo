#!/usr/bin/env python3
"""
Oil & Gas Geospatial Machine Learning Models
Develops and trains models for equipment health, production optimization, and environmental risk
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:8768")

class OilGasGeospatialModels:
    """
    Machine Learning models for Oil & Gas geospatial dashboard
    """
    
    def __init__(self, project_name="Oil-and-Gas-Demo"):
        self.project_name = project_name
        
        # Get correct paths based on project type
        paths = get_data_paths(project_name)
        self.data_dir = paths['base_data_path']
        self.artifacts_dir = paths['artifacts_path']
        self.models_dir = self.artifacts_dir / "models"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        self.experiment_name = "oil_gas_geospatial_models"
        mlflow.set_experiment(self.experiment_name)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the geospatial dataset"""
        print(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_parquet(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _engineer_features(self, df):
        """Create additional features for modeling"""
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
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Fill production-related missing values with 0 for non-production facilities
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
        
        # Fill capacity and throughput with facility type medians
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
    
    def prepare_features(self, df, target_col, test_size=0.2):
        """Prepare features for modeling"""
        # Define feature categories
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
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Prepare X and y
        X = df[numeric_features + categorical_features].copy()
        y = df[target_col].copy()
        
        # Remove rows where target is missing
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        return X_train, X_test, y_train, y_test, numeric_features, categorical_features
    
    def create_preprocessor(self, numeric_features, categorical_features):
        """Create preprocessing pipeline"""
        # Numeric preprocessing
        numeric_transformer = StandardScaler()
        
        # Categorical preprocessing
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Combine preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def train_equipment_health_model(self, df):
        """Train equipment health prediction model (regression)"""
        print("Training Equipment Health Prediction Model...")
        
        with mlflow.start_run(run_name="equipment_health_model", nested=True) as run:
            # Prepare data
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = \
                self.prepare_features(df, 'equipment_health_score')
            
            mlflow.log_param("target", "equipment_health_score")
            mlflow.log_param("n_samples", len(X_train))
            mlflow.log_param("n_features", len(numeric_features) + len(categorical_features))
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(numeric_features, categorical_features)
            
            # Try multiple models
            models_to_try = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('-inf')
            best_model_name = None
            
            for name, model in models_to_try.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = pipeline.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    print(f"{name} - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                    
                    # Log metrics
                    mlflow.log_metric(f"{name}_r2", r2)
                    mlflow.log_metric(f"{name}_rmse", rmse)
                    mlflow.log_metric(f"{name}_mae", mae)
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = pipeline
                        best_model_name = name
                        
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if best_model:
                # Save best model
                model_path = self.models_dir / "equipment_health_model.pkl"
                joblib.dump(best_model, model_path)
                
                # Log model and metrics
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_r2", best_score)
                mlflow.sklearn.log_model(best_model, "equipment_health_model")
                mlflow.log_artifact(str(model_path))
                
                self.models['equipment_health'] = best_model
                print(f"Best model: {best_model_name} (R2: {best_score:.4f})")
                
                return best_model, best_score
            else:
                raise ValueError("No model could be trained successfully")
    
    def train_production_efficiency_model(self, df):
        """Train production efficiency classification model"""
        print("Training Production Efficiency Model...")
        
        with mlflow.start_run(run_name="production_efficiency_model", nested=True) as run:
            # Create production efficiency target based on utilization_rate
            df_work = df.copy()
            df_work = df_work[df_work['utilization_rate'].notna()]
            
            # Create classification target
            df_work['production_efficiency_class'] = pd.cut(
                df_work['utilization_rate'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            # Prepare data
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = \
                self.prepare_features(df_work, 'production_efficiency_class')
            
            mlflow.log_param("target", "production_efficiency_class")
            mlflow.log_param("n_samples", len(X_train))
            mlflow.log_param("n_features", len(numeric_features) + len(categorical_features))
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(numeric_features, categorical_features)
            
            # Try multiple models
            models_to_try = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('-inf')
            best_model_name = None
            
            for name, model in models_to_try.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    print(f"{name} - Accuracy: {accuracy:.4f}")
                    
                    # Log metrics
                    mlflow.log_metric(f"{name}_accuracy", accuracy)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = pipeline
                        best_model_name = name
                        
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if best_model:
                # Save best model
                model_path = self.models_dir / "production_efficiency_model.pkl"
                joblib.dump(best_model, model_path)
                
                # Log model and metrics
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_score)
                mlflow.sklearn.log_model(best_model, "production_efficiency_model")
                mlflow.log_artifact(str(model_path))
                
                self.models['production_efficiency'] = best_model
                print(f"Best model: {best_model_name} (Accuracy: {best_score:.4f})")
                
                return best_model, best_score
            else:
                raise ValueError("No model could be trained successfully")
    
    def train_environmental_risk_model(self, df):
        """Train environmental risk scoring model"""
        print("Training Environmental Risk Model...")
        
        with mlflow.start_run(run_name="environmental_risk_model", nested=True) as run:
            # Create environmental risk classification target
            df_work = df.copy()
            
            # Create classification target based on environmental_risk score
            df_work['environmental_risk_class'] = pd.cut(
                df_work['environmental_risk'],
                bins=[0, 0.33, 0.66, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            # Prepare data
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = \
                self.prepare_features(df_work, 'environmental_risk_class')
            
            mlflow.log_param("target", "environmental_risk_class")
            mlflow.log_param("n_samples", len(X_train))
            mlflow.log_param("n_features", len(numeric_features) + len(categorical_features))
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(numeric_features, categorical_features)
            
            # Try multiple models
            models_to_try = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('-inf')
            best_model_name = None
            
            for name, model in models_to_try.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    print(f"{name} - Accuracy: {accuracy:.4f}")
                    
                    # Log metrics
                    mlflow.log_metric(f"{name}_accuracy", accuracy)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = pipeline
                        best_model_name = name
                        
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if best_model:
                # Save best model
                model_path = self.models_dir / "environmental_risk_model.pkl"
                joblib.dump(best_model, model_path)
                
                # Log model and metrics
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_score)
                mlflow.sklearn.log_model(best_model, "environmental_risk_model")
                mlflow.log_artifact(str(model_path))
                
                self.models['environmental_risk'] = best_model
                print(f"Best model: {best_model_name} (Accuracy: {best_score:.4f})")
                
                return best_model, best_score
            else:
                raise ValueError("No model could be trained successfully")
    
    def optimize_hyperparameters(self, model_type, df):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters for {model_type} model...")
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            if model_type == 'equipment_health':
                # XGBoost hyperparameters for regression
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
                # Prepare data
                X_train, X_test, y_train, y_test, numeric_features, categorical_features = \
                    self.prepare_features(df, 'equipment_health_score')
                
            else:  # classification models
                if model_type == 'production_efficiency':
                    target_col = 'production_efficiency_class'
                    df_work = df.copy()
                    df_work = df_work[df_work['utilization_rate'].notna()]
                    df_work['production_efficiency_class'] = pd.cut(
                        df_work['utilization_rate'],
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                else:  # environmental_risk
                    target_col = 'environmental_risk_class'
                    df_work = df.copy()
                    df_work['environmental_risk_class'] = pd.cut(
                        df_work['environmental_risk'],
                        bins=[0, 0.33, 0.66, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                
                # XGBoost hyperparameters for classification
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)
                
                # Prepare data
                X_train, X_test, y_train, y_test, numeric_features, categorical_features = \
                    self.prepare_features(df_work, target_col)
            
            # Create preprocessor and pipeline
            preprocessor = self.create_preprocessor(numeric_features, categorical_features)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train and evaluate
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            if model_type == 'equipment_health':
                score = r2_score(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params, study.best_value
    
    def generate_feature_importance(self, model, feature_names):
        """Generate feature importance analysis"""
        try:
            # Get the actual model from pipeline
            actual_model = model.named_steps['model']
            
            # Get feature importance
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
            elif hasattr(actual_model, 'coef_'):
                importances = np.abs(actual_model.coef_).flatten()
            else:
                return None
            
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
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names_transformed[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error generating feature importance: {e}")
            return None
    
    def train_all_models(self, data_path):
        """Train all three models"""
        print("Starting comprehensive model training...")
        
        # Load data
        df = self.load_and_prepare_data(data_path)
        
        results = {}
        
        with mlflow.start_run(run_name="oil_gas_model_suite") as parent_run:
            mlflow.set_tag("project", self.project_name)
            mlflow.set_tag("stage", "model_development")
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            
            # Train equipment health model
            try:
                model, score = self.train_equipment_health_model(df)
                results['equipment_health'] = {'model': model, 'score': score, 'type': 'regression'}
                print(f"Equipment Health Model trained successfully (R2: {score:.4f})")
            except Exception as e:
                print(f"Failed to train equipment health model: {e}")
            
            # Train production efficiency model
            try:
                model, score = self.train_production_efficiency_model(df)
                results['production_efficiency'] = {'model': model, 'score': score, 'type': 'classification'}
                print(f"Production Efficiency Model trained successfully (Accuracy: {score:.4f})")
            except Exception as e:
                print(f"Failed to train production efficiency model: {e}")
            
            # Train environmental risk model
            try:
                model, score = self.train_environmental_risk_model(df)
                results['environmental_risk'] = {'model': model, 'score': score, 'type': 'classification'}
                print(f"Environmental Risk Model trained successfully (Accuracy: {score:.4f})")
            except Exception as e:
                print(f"Failed to train environmental risk model: {e}")
            
            # Log overall results
            mlflow.log_param("models_trained", len(results))
            for model_name, info in results.items():
                mlflow.log_metric(f"{model_name}_score", info['score'])
            
            # Save model metadata
            metadata = {
                'project': self.project_name,
                'experiment': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'models': {name: {'score': info['score'], 'type': info['type']} 
                          for name, info in results.items()}
            }
            
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path))
        
        return results

def main():
    """Main function to run model training"""
    print("Oil & Gas Geospatial Model Development")
    print("=" * 50)
    
    # Initialize model trainer
    trainer = OilGasGeospatialModels(project_name="Oil-and-Gas-Demo")

    # Data path - use correct data directory from config
    paths = get_data_paths("Oil-and-Gas-Demo")
    data_path = paths['base_data_path'] / "prepared_geospatial_data.parquet"
    
    # Train all models
    results = trainer.train_all_models(data_path)
    
    # Print summary
    print("\nModel Training Summary:")
    print("-" * 30)
    for model_name, info in results.items():
        score = info['score']
        model_type = info['type']
        metric = "R2" if model_type == "regression" else "Accuracy"
        print(f"{model_name}: {metric} = {score:.4f}")
    
    print(f"\nModels saved to: {trainer.models_dir}")
    print(f"MLflow experiment: {trainer.experiment_name}")

if __name__ == "__main__":
    main()