#!/usr/bin/env python3
"""
Validate Databricks Notebooks
UPDATED for house-pricing-prediction/MLFLOW_sample structure
"""

import sys
import json
from pathlib import Path
import nbformat
from nbformat.validator import NotebookValidationError


def validate_notebook(notebook_path):
    """
    Validate a single notebook file.
    
    Args:
        notebook_path: Path to notebook file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Basic validation
        nbformat.validate(notebook)
        
        # Check for required cells
        if len(notebook.cells) == 0:
            print(f"❌ {notebook_path.name}: No cells found")
            return False
        
        print(f"✅ {notebook_path.name}: Valid")
        return True
        
    except NotebookValidationError as e:
        print(f"❌ {notebook_path.name}: Validation error - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ {notebook_path.name}: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ {notebook_path.name}: Error - {e}")
        return False


def main():
    """Main validation function."""
    print("=" * 60)
    print("VALIDATING NOTEBOOKS")
    print("=" * 60)
    
    # Get repo root (where the script is run from)
    project_root = Path(__file__).parent.parent
    
    # Notebooks are in MLFLOW_sample/notebooks
    notebooks_dir = project_root / 'MLFLOW_sample' / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        print(f"   Expected structure: house-pricing-prediction/MLFLOW_sample/notebooks/")
        sys.exit(1)
    
    # Find all notebook files
    notebooks = list(notebooks_dir.glob('*.ipynb')) + list(notebooks_dir.glob('*.py'))
    
    if not notebooks:
        print("⚠️  No notebook files found")
        print(f"   Looked in: {notebooks_dir}")
        sys.exit(0)
    
    print(f"\nFound {len(notebooks)} notebook(s)")
    print(f"Location: {notebooks_dir}")
    print()
    
    # Validate each notebook
    valid_count = 0
    invalid_count = 0
    
    for notebook in sorted(notebooks):
        # Skip checkpoint files
        if '.ipynb_checkpoints' in str(notebook):
            continue
            
        # Validate .ipynb files only
        if notebook.suffix == '.ipynb':
            if validate_notebook(notebook):
                valid_count += 1
            else:
                invalid_count += 1
        else:
            # .py files (Databricks notebooks) - just check they exist and are not empty
            if notebook.stat().st_size > 0:
                print(f"✅ {notebook.name}: Valid (Python notebook)")
                valid_count += 1
            else:
                print(f"❌ {notebook.name}: Empty file")
                invalid_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Valid: {valid_count}")
    print(f"❌ Invalid: {invalid_count}")
    print("=" * 60)
    
    if invalid_count > 0:
        print("\n❌ Validation failed!")
        sys.exit(1)
    else:
        print("\n✅ All notebooks are valid!")
        sys.exit(0)


if __name__ == '__main__':
    main()
