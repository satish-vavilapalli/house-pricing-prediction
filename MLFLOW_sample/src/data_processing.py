
#Data Processing Module for House Price Prediction
#Author: Satish
#Date: 2026-01-17
#Handles data loading, validation, preprocessing, and train-test splitting.


import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for house price prediction.
    
    Handles:
    - Data loading and validation
    - Train-test splitting
    - Feature scaling
    - Saving/loading preprocessors"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("✅ DataProcessor initialized")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"📂 Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that required columns exist in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("🔍 Validating data...")
        
        # Get required columns
        required_cols = (
            self.config['preprocessing']['numerical_features'] +
            self.config['preprocessing']['categorical_features'] +
            [self.config['preprocessing']['target']]
        )
        
        # Check for missing columns
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info("✅ Data validation passed")
        return True

    # ------------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------------
    def split_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("🔧 Splitting data into train and test sets...")
        
        # Get target column
        target = self.config['preprocessing']['target']
        
        # Separate features and target
        # Exclude 'Home' (ID column) and target
        exclude_cols = ['Home', target]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target]
        
        # Split
        test_size = self.config['preprocessing']['test_size']
        random_state = self.config['preprocessing']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"✅ Data split complete")
        logger.info(f"  • Training set: {X_train.shape[0]} samples")
        logger.info(f"  • Test set: {X_test.shape[0]} samples")
        logger.info(f"  • Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Scale numerical features using StandardScaler.
        
        Fits the scaler on training data and transforms both train and test sets.
        
        Args:
            X_train: Training features DataFrame
            X_test: Test features DataFrame
            
        Returns:
            Tuple of (scaled X_train, scaled X_test, fitted scaler)
        """
        logger.info("🔧 Scaling features using StandardScaler...")
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data only
        scaler.fit(X_train)
        
        # Transform both train and test
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"✅ Scaled {X_train_scaled.shape[1]} features")
        logger.info(f"  • Training set: {X_train_scaled.shape}")
        logger.info(f"  • Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, scaler

    # ------------------------------------------------------------------
    # Save/Load preprocessors
    # ------------------------------------------------------------------
    def save_preprocessor(
        self, 
        preprocessor: Any, 
        file_path: str
    ) -> None:
        """
        Save preprocessor (scaler, encoder, etc.) to file.
        
        Args:
            preprocessor: Preprocessor object to save
            file_path: Path to save the preprocessor
        """
        logger.info(f"💾 Saving preprocessor to: {file_path}")
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logger.info("✅ Preprocessor saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save preprocessor: {e}")
            raise

    def load_preprocessor(self, file_path: str) -> Any:
        """
        Load preprocessor from file.
        
        Args:
            file_path: Path to the preprocessor file
            
        Returns:
            Loaded preprocessor object
        """
        logger.info(f"📂 Loading preprocessor from: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("✅ Preprocessor loaded successfully")
            return preprocessor
        except Exception as e:
            logger.error(f"❌ Failed to load preprocessor: {e}")
            raise

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names (excluding target and ID).
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature names
        """
        target = self.config['preprocessing']['target']
        exclude_cols = ['Home', target]
        
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return feature_names

    def __repr__(self) -> str:
        """String representation of DataProcessor."""
        return f"DataProcessor(config_loaded=True)"