#!/usr/bin/env python3
"""
Promote Model in MLflow Registry
Manages model stage transitions (Staging → Production)
"""

import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def promote_model(model_name, version=None, stage='Production'):
    """
    Promote a model to a specific stage.
    
    Args:
        model_name: Name of the registered model
        version: Model version (if None, uses latest)
        stage: Target stage (Staging, Production, Archived)
    """
    print("=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)
    
    try:
        client = MlflowClient()
        
        # Get model version
        if version is None:
            # Get latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"❌ No versions found for model: {model_name}")
                sys.exit(1)
            
            # Sort by version number (descending)
            versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
            version = versions[0].version
            print(f"Using latest version: {version}")
        
        print(f"Model: {model_name}")
        print(f"Version: {version}")
        print(f"Target Stage: {stage}")
        print()
        
        # Get current model info
        model_version = client.get_model_version(model_name, version)
        current_stage = model_version.current_stage
        
        print(f"Current Stage: {current_stage}")
        
        if current_stage == stage:
            print(f"✅ Model is already in {stage} stage")
            sys.exit(0)
        
        # Archive current production model if promoting to production
        if stage == 'Production':
            print(f"\nArchiving current Production models...")
            production_versions = client.get_latest_versions(model_name, stages=['Production'])
            
            for pv in production_versions:
                if pv.version != version:
                    print(f"  Archiving version {pv.version}")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=pv.version,
                        stage='Archived'
                    )
        
        # Promote new model
        print(f"\nPromoting version {version} to {stage}...")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print()
        print("=" * 60)
        print("✅ MODEL PROMOTION SUCCESSFUL")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Version: {version}")
        print(f"Stage: {current_stage} → {stage}")
        print("=" * 60)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error promoting model: {e}")
        sys.exit(1)


def main():
    """Main promotion function."""
    parser = argparse.ArgumentParser(description='Promote model in MLflow Registry')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--version', type=int, help='Model version (default: latest)')
    parser.add_argument('--stage', default='Production', 
                       choices=['Staging', 'Production', 'Archived'],
                       help='Target stage')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI from environment
    if 'DATABRICKS_HOST' in os.environ and 'DATABRICKS_TOKEN' in os.environ:
        mlflow.set_tracking_uri('databricks')
    
    promote_model(args.model_name, args.version, args.stage)


if __name__ == '__main__':
    main()
