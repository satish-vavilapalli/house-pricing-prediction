#!/usr/bin/env python3
"""
Get Latest Model Metrics
Retrieves metrics for the most recently trained model
"""

import os
import sys
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def get_latest_model_metrics(model_name, output_file=None):
    """
    Get metrics for the latest model version (regardless of stage).
    
    Args:
        model_name: Name of the registered model
        output_file: Optional file to save metrics JSON
        
    Returns:
        dict: Model metrics
    """
    print("=" * 60)
    print("RETRIEVING LATEST MODEL METRICS")
    print("=" * 60)
    
    try:
        client = MlflowClient()
        
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"❌ No versions found for model: {model_name}")
            sys.exit(1)
        
        # Sort by version number (descending) to get latest
        versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
        latest_version = versions[0]
        
        version_number = latest_version.version
        run_id = latest_version.run_id
        stage = latest_version.current_stage
        
        print(f"Model: {model_name}")
        print(f"Latest Version: {version_number}")
        print(f"Stage: {stage}")
        print(f"Run ID: {run_id}")
        print()
        
        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        # Extract key metrics
        result = {
            'model_name': model_name,
            'version': version_number,
            'stage': stage,
            'run_id': run_id,
            'rmse': metrics.get('test_rmse', metrics.get('rmse', 0)),
            'mae': metrics.get('test_mae', metrics.get('mae', 0)),
            'r2': metrics.get('test_r2', metrics.get('r2', 0)),
            'mape': metrics.get('test_mape', metrics.get('mape', 0)),
        }
        
        print("Metrics:")
        print(f"  RMSE: ${result['rmse']:,.2f}")
        print(f"  MAE:  ${result['mae']:,.2f}")
        print(f"  R²:   {result['r2']:.4f}")
        print(f"  MAPE: {result['mape']:.2f}%")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ Metrics saved to {output_file}")
        
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error retrieving metrics: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Get latest model metrics from MLflow')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--output-file', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    if 'DATABRICKS_HOST' in os.environ and 'DATABRICKS_TOKEN' in os.environ:
        mlflow.set_tracking_uri('databricks')
    
    get_latest_model_metrics(args.model_name, args.output_file)


if __name__ == '__main__':
    main()
