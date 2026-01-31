# src/utils.py - DATABRICKS PRODUCTION READY

import yaml
import logging
import pandas as pd
from pathlib import Path
import mlflow
from typing import Dict, Any, List, Union, Optional
import json
import tempfile

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Config loader
# -------------------------------------------------------------------
class ConfigLoader:
    """Load and manage configuration."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            config_path = Path(config_path)
            with config_path.open("r") as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Config loaded: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error in {config_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Config error {config_path}: {str(e)}")
            raise


# -------------------------------------------------------------------
# Data IO helpers
# -------------------------------------------------------------------
class DataLoader:
    """Handle data loading operations."""

    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV file into a DataFrame."""
        try:
            file_path = Path(file_path)
            df = pd.read_csv(file_path)
            logger.info(f"📊 Loaded {file_path.name}: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"❌ Empty CSV file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Load failed {file_path}: {str(e)}")
            raise

    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Save DataFrame to CSV."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            logger.info(f"💾 Saved {file_path.name}: {df.shape}")
        except Exception as e:
            logger.error(f"❌ Save failed {file_path}: {str(e)}")
            raise


# -------------------------------------------------------------------
# Databricks MLflow (SIMPLE & SAFE)
# -------------------------------------------------------------------
class MLflowLogger:
    """Databricks MLflow logging utilities."""

    @staticmethod
    def log_params_from_dict(params: Dict[str, Any]) -> None:
        """Log parameters from a dictionary."""
        if not params:
            logger.warning("⚠️ No params to log")
            return
        try:
            # MLflow has a limit on param value length (500 chars)
            # Truncate long values
            truncated_params = {
                k: str(v)[:500] if len(str(v)) > 500 else v 
                for k, v in params.items()
            }
            mlflow.log_params(truncated_params)
            logger.info(f"📝 Logged {len(params)} params")
        except mlflow.exceptions.MlflowException as e:
            logger.warning(f"⚠️ MLflow param logging: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Param logging: {str(e)}")
            raise

    @staticmethod
    def log_metrics_from_dict(metrics: Dict[str, float]) -> None:
        """Log metrics from a dictionary."""
        if not metrics:
            logger.warning("⚠️ No metrics to log")
            return
        try:
            # Ensure all values are numeric
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() 
                if isinstance(v, (int, float)) and not pd.isna(v)
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
                logger.info(f"📊 Logged {len(numeric_metrics)} metrics")
            else:
                logger.warning("⚠️ No valid numeric metrics to log")
        except mlflow.exceptions.MlflowException as e:
            logger.warning(f"⚠️ MLflow metric logging: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Metric logging: {str(e)}")
            raise

    @staticmethod
    def log_artifact_from_dict(data: Dict[str, Any], filename: str) -> None:
        """Log dictionary as JSON artifact (tempdir safe)."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / filename
                with tmp_path.open("w") as f:
                    json.dump(data, f, indent=2, default=str)  # default=str handles non-serializable objects
                mlflow.log_artifact(str(tmp_path))
            logger.info(f"📦 Artifact logged: {filename}")
        except Exception as e:
            logger.error(f"❌ Artifact {filename}: {str(e)}")
            raise

    @staticmethod
    def log_dataframe_as_artifact(df: pd.DataFrame, filename: str) -> None:
        """Log DataFrame as CSV artifact."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / filename
                df.to_csv(tmp_path, index=False)
                mlflow.log_artifact(str(tmp_path))
            logger.info(f"📦 DataFrame artifact logged: {filename}")
        except Exception as e:
            logger.error(f"❌ DataFrame artifact {filename}: {str(e)}")
            raise


# -------------------------------------------------------------------
# Data validation
# -------------------------------------------------------------------
class DataValidator:
    """Validate data quality."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
        """Fail-fast validation for required columns."""
        if df is None or df.empty:
            msg = "❌ DataFrame is None or empty"
            logger.error(msg)
            raise ValueError(msg)
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            msg = f"❌ Missing columns: {missing_cols}"
            logger.error(msg)
            raise ValueError(msg)
        logger.info("✅ All required columns validated")

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> pd.Series:
        """Check and log missing values."""
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
            logger.warning(f"⚠️ Missing values: {total_missing} ({missing_pct:.2f}%)")
            logger.warning(f"Columns with missing:\n{missing[missing > 0]}")
        else:
            logger.info("✅ No missing values")
        return missing

    @staticmethod
    def check_data_types(df: pd.DataFrame) -> pd.Series:
        """Log data types."""
        dtypes = df.dtypes
        logger.info(f"📋 Data types:\n{dtypes}")
        return dtypes

    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> int:
        """Check for duplicate rows."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠️ Found {duplicates} duplicate rows")
        else:
            logger.info("✅ No duplicate rows")
        return duplicates


# -------------------------------------------------------------------
# Databricks MLflow Setup (ULTRA-SIMPLE)
# -------------------------------------------------------------------
def setup_mlflow_databricks(config: Dict[str, Any]) -> None:
    """Databricks-only MLflow setup (no tracking_uri needed)."""
    try:
        mlflow_config = config.get('mlflow', {})
        experiment_name = mlflow_config.get('experiment_name')
        
        if not experiment_name:
            logger.warning("⚠️ No 'experiment_name' in config.mlflow - using default")
            return
        
        # Set experiment (creates if doesn't exist)
        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ MLflow experiment set: {experiment_name}")
        
        # Enable autologging if specified
        if mlflow_config.get('autolog', False):
            mlflow.sklearn.autolog()
            logger.info("✅ MLflow autologging enabled")
            
    except Exception as e:
        logger.warning(f"⚠️ MLflow setup issue: {e}")


# -------------------------------------------------------------------
# Project helpers
# -------------------------------------------------------------------
def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


def log_dataset_summary(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Log dataset info to MLflow."""
    try:
        target = config['preprocessing']['target']
        
        summary = {
            "dataset_rows": int(df.shape[0]),
            "dataset_columns": int(df.shape[1]),
            "memory_kb": float(df.memory_usage(deep=True).sum() / 1024),
        }
        
        # Add target statistics if target exists
        if target in df.columns:
            summary.update({
                "target_mean": float(df[target].mean()),
                "target_std": float(df[target].std()),
                "target_min": float(df[target].min()),
                "target_max": float(df[target].max()),
            })
        
        MLflowLogger.log_params_from_dict(summary)
        logger.info("📊 Dataset summary logged to MLflow")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not log dataset summary: {e}")


def create_run_name(model_name: str, timestamp: Optional[str] = None) -> str:
    """Create a standardized MLflow run name."""
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def safe_display(df: pd.DataFrame, max_rows: int = 10) -> None:
    """Safely display DataFrame (handles empty DataFrames)."""
    if df is None or df.empty:
        print("📭 Empty DataFrame - nothing to display")
        return
    
    try:
        # In Databricks, use display()
        display(df.head(max_rows))
    except NameError:
        # Fallback for non-Databricks environments
        print(df.head(max_rows))