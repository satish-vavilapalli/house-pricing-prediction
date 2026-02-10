#!/usr/bin/env python3
"""
Rollback Model in MLflow Registry
Reverts Production to the previous stable version
"""

import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def rollback_model(model_name):
    """
    Rollback to the previous Production model.
    
    Args:
        model_name: Name of the registered model
    """
    print("=" * 60)
    print("MODEL ROLLBACK")
    print("=" * 60)
    
    try:
        client = MlflowClient()
        
        # Get all versions
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not all_versions:
            print(f"❌ No versions found for model: {model_name}")
            sys.exit(1)
        
        # Get current production version
        production_versions = [v for v in all_versions if v.current_stage == 'Production']
        
        if not production_versions:
            print(f"❌ No Production model found for {model_name}")
            sys.exit(1)
        
        current_prod = production_versions[0]
        current_version = int(current_prod.version)
        
        print(f"Current Production Version: {current_version}")
        
        # Get archived versions (potential rollback candidates)
        archived_versions = [v for v in all_versions if v.current_stage == 'Archived']
        
        if not archived_versions:
            print(f"❌ No archived versions available for rollback")
            sys.exit(1)
        
        # Sort by version number (descending) and get the most recent archived
        archived_versions = sorted(archived_versions, key=lambda x: int(x.version), reverse=True)
        rollback_to = archived_versions[0]
        rollback_version = int(rollback_to.version)
        
        print(f"Rollback Target Version: {rollback_version}")
        print()
        
        # Confirm rollback
        print(f"Rolling back from v{current_version} to v{rollback_version}...")
        
        # Archive current production
        client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage='Archived',
            archive_existing_versions=False
        )
        print(f"✅ Archived current production (v{current_version})")
        
        # Promote rollback version to production
        client.transition_model_version_stage(
            name=model_name,
            version=rollback_version,
            stage='Production',
            archive_existing_versions=False
        )
        print(f"✅ Promoted v{rollback_version} to Production")
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=rollback_version,
            description=f"Rolled back to this version from v{current_version}"
        )
        
        print()
        print("=" * 60)
        print("✅ ROLLBACK SUCCESSFUL")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Previous Production: v{current_version} (now Archived)")
        print(f"New Production: v{rollback_version}")
        print("=" * 60)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error during rollback: {e}")
        sys.exit(1)


def main():
    """Main rollback function."""
    parser = argparse.ArgumentParser(description='Rollback model in MLflow Registry')
    parser.add_argument('--model-name', required=True, help='Model name')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    if 'DATABRICKS_HOST' in os.environ and 'DATABRICKS_TOKEN' in os.environ:
        mlflow.set_tracking_uri('databricks')
    
    rollback_model(args.model_name)


if __name__ == '__main__':
    main()
