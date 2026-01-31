# src/model.py - Production-Ready Model Training & Evaluation

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune ML models with MLflow tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ModelTrainer with configuration."""
        self.config = config
        self.models_config = config.get('models', {})
        self.training_config = config.get('training', {})
        
    def get_model(self, model_name: str):
        """Get model instance based on name."""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name]
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter grid for model."""
        if model_name not in self.models_config:
            logger.warning(f"No config for {model_name}, using empty params")
            return {}
        
        return self.models_config[model_name].get('params', {})
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Train a model with optional hyperparameter tuning.
        
        Returns:
            Tuple of (best_model, best_params, cv_results)
        """
        logger.info(f"Training {model_name}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_training", nested=True):
            # Get base model
            model = self.get_model(model_name)
            
            # Log model type
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tune_hyperparameters", tune_hyperparameters)
            
            if tune_hyperparameters:
                # Get parameter grid
                param_grid = self.get_param_grid(model_name)
                
                if param_grid:
                    logger.info(f"Tuning hyperparameters for {model_name}")
                    
                    # GridSearchCV
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=self.training_config.get('cv_folds', 5),
                        scoring=self.training_config.get('scoring', 'neg_mean_squared_error'),
                        n_jobs=self.training_config.get('n_jobs', -1),
                        verbose=1
                    )
                    
                    # Fit
                    grid_search.fit(X_train, y_train)
                    
                    # Get best model and params
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    cv_results = {
                        'best_score': grid_search.best_score_,
                        'best_index': grid_search.best_index_
                    }
                    
                    # Log best parameters
                    for param, value in best_params.items():
                        mlflow.log_param(f"best_{param}", value)
                    
                    # Log CV score
                    mlflow.log_metric("cv_score", -grid_search.best_score_)  # Negate because scoring is negative
                    
                    logger.info(f"Best params: {best_params}")
                    logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
                    
                else:
                    logger.info(f"No param grid for {model_name}, training with defaults")
                    model.fit(X_train, y_train)
                    best_model = model
                    best_params = {}
                    cv_results = None
            else:
                # Train without tuning
                logger.info(f"Training {model_name} without hyperparameter tuning")
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}
                cv_results = None
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log top 10 features
                for idx, row in feature_importance.head(10).iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
            
            elif hasattr(best_model, 'coef_'):
                # Log coefficients for linear models
                coefficients = pd.DataFrame({
                    'feature': X_train.columns,
                    'coefficient': best_model.coef_
                }).sort_values('coefficient', key=abs, ascending=False)
                
                # Log top 10 coefficients
                for idx, row in coefficients.head(10).iterrows():
                    mlflow.log_metric(f"coef_{row['feature']}", row['coefficient'])
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            
            logger.info(f"✅ {model_name} training complete")
            
            return best_model, best_params, cv_results


class ModelEvaluator:
    """Evaluate and visualize model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ModelEvaluator with configuration."""
        self.config = config
    
    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with metrics and predictions
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"{model_name}_evaluation", nested=True):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Print metrics
        print(f"\n📊 {model_name.upper()} Evaluation Metrics:")
        print(f"  • RMSE: ${metrics['rmse']:,.2f}")
        print(f"  • MAE:  ${metrics['mae']:,.2f}")
        print(f"  • R²:   {metrics['r2']:.4f}")
        print(f"  • MAPE: {metrics['mape']:.2f}%")
        
        # Create visualizations
        self.plot_predictions(y_test, y_pred, model_name)
        self.plot_residuals(y_test, y_pred, model_name)
        
        return {
            'metrics': metrics,
            'predictions': y_pred
        }
    
    def plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
        plt.ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
        plt.title(f'Actual vs Predicted - {model_name.title()}', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot residuals analysis."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Residual Plot - {model_name.title()}', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title(f'Residuals Distribution - {model_name.title()}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, result in results.items():
            test_metrics = result.get('test_metrics', {})
            train_metrics = result.get('train_metrics', {})
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train RMSE': train_metrics.get('rmse', 0),
                'Test RMSE': test_metrics.get('rmse', 0),
                'Train MAE': train_metrics.get('mae', 0),
                'Test MAE': test_metrics.get('mae', 0),
                'Train R²': train_metrics.get('r2', 0),
                'Test R²': test_metrics.get('r2', 0),
                'Overfit (RMSE)': train_metrics.get('rmse', 0) - test_metrics.get('rmse', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test RMSE')
        
        return comparison_df


def load_model(model_uri: str):
    """Load a model from MLflow."""
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ Model loaded from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def register_model(
    model_name: str,
    run_id: str,
    model_registry_name: str
) -> str:
    """Register model to MLflow Model Registry."""
    try:
        model_uri = f"runs:/{run_id}/{model_name}_model"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
        )
        
        logger.info(f"✅ Model registered: {model_registry_name} v{model_version.version}")
        return model_version.version
        
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        raise