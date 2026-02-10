#!/usr/bin/env python3
"""
Get Model Metrics from MLflow
Retrieves performance metrics for a specific model version
"""

import os
import sys
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def get_model_metrics(model_name, stage='Production', output_file=None):
    """
    Get metrics for a model in a specific stage.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Staging, Production)
        output_file: Optional file to save metrics JSON
        
    Returns:
        dict: Model metrics
    """
    print("=" * 60)
    print("RETRIEVING MODEL METRICS")
    print("=" * 60)
    
    try:
        client = MlflowClient()
        
        # Get model version in specified stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not versions:
            print(f"❌ No model found in {stage} stage for {model_name}")
            sys.exit(1)
        
        model_version = versions[0]
        version_number = model_version.version
        run_id = model_version.run_id
        
        print(f"Model: {model_name}")
        print(f"Version: {version_number}")
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
    parser = argparse.ArgumentParser(description='Get model metrics from MLflow')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--stage', default='Production', 
                       choices=['Staging', 'Production'],
                       help='Model stage')
    parser.add_argument('--output-file', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    if 'DATABRICKS_HOST' in os.environ and 'DATABRICKS_TOKEN' in os.environ:
        mlflow.set_tracking_uri('databricks')
    
    get_model_metrics(args.model_name, args.stage, args.output_file)


if __name__ == '__main__':
    main()
