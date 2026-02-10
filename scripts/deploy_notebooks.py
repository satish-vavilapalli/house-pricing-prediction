#!/usr/bin/env python3
"""
Deploy Notebooks to Databricks
UPDATED for house-pricing-prediction/MLFLOW_sample structure
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"Running: {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(result.stderr)
        return False
    
    print(f"✅ {description} completed")
    if result.stdout:
        print(result.stdout)
    return True


def sync_notebooks(workspace_path, source_path='MLFLOW_sample/notebooks'):
    """
    Sync notebooks to Databricks workspace.
    
    Args:
        workspace_path: Target path in Databricks workspace
        source_path: Source directory containing notebooks (relative to repo root)
    """
    print("=" * 60)
    print("DEPLOYING NOTEBOOKS TO DATABRICKS")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Target: {workspace_path}")
    print()
    
    # Get project root (repo root, not MLFLOW_sample)
    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / source_path
    
    if not notebooks_dir.exists():
        print(f"❌ Source directory not found: {notebooks_dir}")
        print(f"   Looking for: {notebooks_dir}")
        print(f"   Current dir: {os.getcwd()}")
        sys.exit(1)
    
    # Find all notebooks
    notebooks = list(notebooks_dir.glob('*.ipynb')) + list(notebooks_dir.glob('*.py'))
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]
    
    if not notebooks:
        print("⚠️  No notebooks found to deploy")
        sys.exit(0)
    
    print(f"Found {len(notebooks)} notebook(s) to deploy")
    print()
    
    # Deploy each notebook
    success_count = 0
    failed_count = 0
    
    for notebook in sorted(notebooks):
        notebook_name = notebook.stem
        target_path = f"{workspace_path}/{notebook_name}"
        
        # Determine format
        if notebook.suffix == '.ipynb':
            format_type = 'JUPYTER'
        else:
            format_type = 'SOURCE'
        
        # Import notebook to Databricks
        cmd = f"databricks workspace import {notebook} {target_path} --language PYTHON --format {format_type} --overwrite"
        
        if run_command(cmd, f"Deploy {notebook.name}"):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Deployed: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print("=" * 60)
    
    if failed_count > 0:
        print("\n⚠️  Some notebooks failed to deploy")
        sys.exit(1)
    else:
        print("\n✅ All notebooks deployed successfully!")
        sys.exit(0)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy notebooks to Databricks')
    parser.add_argument('--workspace-path', required=True, help='Target workspace path')
    parser.add_argument('--source-path', default='MLFLOW_sample/notebooks', help='Source directory (relative to repo root)')
    
    args = parser.parse_args()
    
    # Sync notebooks
    sync_notebooks(args.workspace_path, args.source_path)


if __name__ == '__main__':
    main()
