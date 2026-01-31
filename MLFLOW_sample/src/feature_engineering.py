"""
Feature Engineering Module for House Price Prediction
Author: Satish
Date: 2026-01-17

Creates domain-specific features from raw house data:
- Price per square foot
- Room ratios and totals
- Size-per-room metrics
- Interactions between key features
- Luxury indicators
- Categorical encoding
- Feature scaling
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for house price prediction.

    Creates domain-specific features from raw house data that capture
    important relationships and patterns for price prediction.
    Also handles categorical encoding and feature scaling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary containing feature engineering settings
        """
        self.config = config
        self.created_features = []
        self.label_encoders = {}
        self.scaler = None
        logger.info("✅ FeatureEngineer initialized")

    # ------------------------------------------------------------------
    # Main feature creation
    # ------------------------------------------------------------------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all configured engineered features.

        Features are created in-place and safely handle edge cases
        (division by zero, missing values).
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            DataFrame with original and new engineered features
        """
        df = df.copy()
        self.created_features = []
        
        logger.info("🔧 Starting feature engineering...")

        # Get configuration
        features_to_create = self.config.get("feature_engineering", {}).get("create_features", [])
        target_col = self.config["preprocessing"]["target"]

        # 1. Price per square foot (if target column is present)
        if "PricePerSqFt" in features_to_create and target_col in df.columns and "SqFt" in df.columns:
            df["PricePerSqFt"] = df[target_col] / np.maximum(df["SqFt"], 1)
            self.created_features.append("PricePerSqFt")
            logger.info("  ✓ Created: PricePerSqFt")

        # 2. Total rooms
        if "TotalRooms" in features_to_create and "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
            self.created_features.append("TotalRooms")
            logger.info("  ✓ Created: TotalRooms")
        elif "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            # Create even if not in config (needed for other features)
            df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
            self.created_features.append("TotalRooms")
            logger.info("  ✓ Created: TotalRooms (auto)")

        # 3. Bathroom to bedroom ratio
        if "BathBedroomRatio" in features_to_create and "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            df["BathBedroomRatio"] = df["Bathrooms"] / np.maximum(df["Bedrooms"], 1)
            self.created_features.append("BathBedroomRatio")
            logger.info("  ✓ Created: BathBedroomRatio")

        # 4. SqFt per room
        if "SqFtPerRoom" in features_to_create and "SqFt" in df.columns and "TotalRooms" in df.columns:
            df["SqFtPerRoom"] = df["SqFt"] / np.maximum(df["TotalRooms"], 1)
            self.created_features.append("SqFtPerRoom")
            logger.info("  ✓ Created: SqFtPerRoom")
        elif "SqFt" in df.columns and "TotalRooms" in df.columns:
            # Create by default
            df["SqFtPerRoom"] = df["SqFt"] / np.maximum(df["TotalRooms"], 1)
            self.created_features.append("SqFtPerRoom")
            logger.info("  ✓ Created: SqFtPerRoom (auto)")

        # 5. Offers per room
        if "OffersPerRoom" in features_to_create and "Offers" in df.columns and "TotalRooms" in df.columns:
            df["OffersPerRoom"] = df["Offers"] / np.maximum(df["TotalRooms"], 1)
            self.created_features.append("OffersPerRoom")
            logger.info("  ✓ Created: OffersPerRoom")
        elif "Offers" in df.columns and "TotalRooms" in df.columns:
            # Create by default
            df["OffersPerRoom"] = df["Offers"] / np.maximum(df["TotalRooms"], 1)
            self.created_features.append("OffersPerRoom")
            logger.info("  ✓ Created: OffersPerRoom (auto)")

        # 6. SqFt × Bedrooms interaction
        if "SqFt_Bedrooms" in features_to_create and "SqFt" in df.columns and "Bedrooms" in df.columns:
            df["SqFt_Bedrooms"] = df["SqFt"] * df["Bedrooms"]
            self.created_features.append("SqFt_Bedrooms")
            logger.info("  ✓ Created: SqFt_Bedrooms")
        elif "SqFt" in df.columns and "Bedrooms" in df.columns:
            # Create by default
            df["SqFt_Bedrooms"] = df["SqFt"] * df["Bedrooms"]
            self.created_features.append("SqFt_Bedrooms")
            logger.info("  ✓ Created: SqFt_Bedrooms (auto)")

        # 7. SqFt × Bathrooms interaction
        if "SqFt_Bathrooms" in features_to_create and "SqFt" in df.columns and "Bathrooms" in df.columns:
            df["SqFt_Bathrooms"] = df["SqFt"] * df["Bathrooms"]
            self.created_features.append("SqFt_Bathrooms")
            logger.info("  ✓ Created: SqFt_Bathrooms")
        elif "SqFt" in df.columns and "Bathrooms" in df.columns:
            # Create by default
            df["SqFt_Bathrooms"] = df["SqFt"] * df["Bathrooms"]
            self.created_features.append("SqFt_Bathrooms")
            logger.info("  ✓ Created: SqFt_Bathrooms (auto)")

        # 8. Brick premium indicator (binary feature for modeling)
        if "BrickPremium" in features_to_create and "Brick" in df.columns:
            df["BrickPremium"] = (df["Brick"].astype(str).str.lower() == "yes").astype(int)
            self.created_features.append("BrickPremium")
            logger.info("  ✓ Created: BrickPremium")
        elif "Brick" in df.columns:
            # Create by default
            df["BrickPremium"] = (df["Brick"].astype(str).str.lower() == "yes").astype(int)
            self.created_features.append("BrickPremium")
            logger.info("  ✓ Created: BrickPremium (auto)")

        # 9. Luxury home indicator (4+ bedrooms AND 3+ bathrooms)
        if "IsLuxury" in features_to_create and "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            df["IsLuxury"] = ((df["Bedrooms"] >= 4) & (df["Bathrooms"] >= 3)).astype(int)
            self.created_features.append("IsLuxury")
            logger.info("  ✓ Created: IsLuxury")

        # 10. SqFt per bedroom
        if "SqFtPerBedroom" in features_to_create and "SqFt" in df.columns and "Bedrooms" in df.columns:
            df["SqFtPerBedroom"] = df["SqFt"] / np.maximum(df["Bedrooms"], 1)
            self.created_features.append("SqFtPerBedroom")
            logger.info("  ✓ Created: SqFtPerBedroom")

        # 11. Has multiple offers indicator
        if "HasMultipleOffers" in features_to_create and "Offers" in df.columns:
            df["HasMultipleOffers"] = (df["Offers"] > 1).astype(int)
            self.created_features.append("HasMultipleOffers")
            logger.info("  ✓ Created: HasMultipleOffers")

        # 12. Offers per bedroom
        if "OffersPerBedroom" in features_to_create and "Offers" in df.columns and "Bedrooms" in df.columns:
            df["OffersPerBedroom"] = df["Offers"] / np.maximum(df["Bedrooms"], 1)
            self.created_features.append("OffersPerBedroom")
            logger.info("  ✓ Created: OffersPerBedroom")

        logger.info(
            f"✅ Feature engineering complete. Shape: {df.shape}. "
            f"Created {len(self.created_features)} features: {self.created_features}"
        )
        
        return df

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------
    def encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (encoded DataFrame, dict of label encoders)
        """
        df_encoded = df.copy()
        self.label_encoders = {}
        
        categorical_features = self.config['preprocessing']['categorical_features']
        
        logger.info(f"🔧 Encoding {len(categorical_features)} categorical features...")
        
        for col in categorical_features:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Handle missing values by converting to string first
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  ✓ Encoded {col}: {list(le.classes_)}")
            else:
                logger.warning(f"  ⚠️ Column {col} not found in DataFrame")
        
        logger.info(f"✅ Categorical encoding complete. Encoded {len(self.label_encoders)} features")
        
        return df_encoded, self.label_encoders

    def transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders (for test/new data).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        if not self.label_encoders:
            raise ValueError("No label encoders fitted. Call encode_categorical() first.")
        
        df_encoded = df.copy()
        
        logger.info(f"🔧 Transforming {len(self.label_encoders)} categorical features...")
        
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle missing values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                
                # Handle unseen categories
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                
                df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                logger.info(f"  ✓ Transformed {col}")
            else:
                logger.warning(f"  ⚠️ Column {col} not found in DataFrame")
        
        logger.info(f"✅ Categorical transformation complete")
        
        return df_encoded

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------
    def fit_scaler(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> StandardScaler:
        """
        Fit StandardScaler on numerical features.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from scaling (e.g., target, ID)
            
        Returns:
            Fitted StandardScaler
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Add default exclusions
        default_exclude = ['Home', self.config['preprocessing']['target']]
        exclude_cols = list(set(exclude_cols + default_exclude))
        
        # Get columns to scale
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"🔧 Fitting scaler on {len(cols_to_scale)} features...")
        
        self.scaler = StandardScaler()
        self.scaler.fit(df[cols_to_scale])
        
        logger.info(f"✅ Scaler fitted on {len(cols_to_scale)} features")
        
        return self.scaler

    def transform_features(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Add default exclusions
        default_exclude = ['Home', self.config['preprocessing']['target']]
        exclude_cols = list(set(exclude_cols + default_exclude))
        
        # Get columns to scale
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        
        df_scaled = df.copy()
        
        logger.info(f"🔧 Scaling {len(cols_to_scale)} features...")
        
        df_scaled[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        logger.info(f"✅ Features scaled")
        
        return df_scaled

    def fit_transform_features(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit scaler and transform features in one step.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        self.fit_scaler(df, exclude_cols)
        return self.transform_features(df, exclude_cols)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def select_features(
        self, df: pd.DataFrame, feature_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Select specific features (useful for debugging or feature selection).
        
        Args:
            df: Input DataFrame
            feature_list: List of feature names to select. If None, returns all.
            
        Returns:
            DataFrame with selected features
        """
        if feature_list is None:
            return df

        # Ensure selected features exist
        missing = set(feature_list) - set(df.columns)
        if missing:
            logger.warning(f"⚠️ Missing features in selection: {missing}")
            feature_list = [f for f in feature_list if f in df.columns]

        if not feature_list:
            raise ValueError("No valid features to select")

        df = df[feature_list]
        logger.info(f"✅ Selected {len(feature_list)} features")
        return df

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_feature_names(self, df: Optional[pd.DataFrame] = None, exclude_target: bool = True) -> List[str]:
        """
        Return list of all feature names (excluding target and ID columns).
        
        Args:
            df: Optional DataFrame to get features from
            exclude_target: Whether to exclude target variable
            
        Returns:
            List of feature names
        """
        if df is None:
            return self.created_features.copy()
        
        # Get columns to exclude
        exclude_cols = ['Home']  # ID column
        
        if exclude_target:
            target = self.config['preprocessing']['target']
            exclude_cols.append(target)
        
        # Get feature names
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return feature_names

    def get_created_features(self) -> List[str]:
        """
        Return list of all created feature names.
        
        Returns:
            List of created feature names
        """
        return self.created_features.copy()

    def get_label_encoders(self) -> Dict[str, LabelEncoder]:
        """
        Return dictionary of fitted label encoders.
        
        Returns:
            Dictionary of label encoders
        """
        return self.label_encoders.copy()

    def get_scaler(self) -> Optional[StandardScaler]:
        """
        Return fitted scaler.
        
        Returns:
            StandardScaler or None if not fitted
        """
        return self.scaler

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for created features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        created_cols = [col for col in self.created_features if col in df.columns]
        if not created_cols:
            logger.warning("⚠️ No created features found in DataFrame")
            return {}

        summary = df[created_cols].describe().to_dict()
        logger.info(f"✅ Generated summary for {len(created_cols)} created features")
        return summary

    def get_feature_importance_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all feature names for importance analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature names suitable for model training
        """
        return self.get_feature_names(df, exclude_target=True)

    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive statistics for all features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with feature statistics
        """
        stats = []
        
        target = self.config['preprocessing']['target']
        exclude_cols = ['Home', target]
        
        for col in df.columns:
            if col not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats.append({
                        'Feature': col,
                        'Type': 'Numeric',
                        'Mean': df[col].mean(),
                        'Median': df[col].median(),
                        'Std': df[col].std(),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Missing': df[col].isnull().sum(),
                        'Unique': df[col].nunique(),
                        'Created': col in self.created_features
                    })
                else:
                    stats.append({
                        'Feature': col,
                        'Type': 'Categorical',
                        'Mean': None,
                        'Median': None,
                        'Std': None,
                        'Min': None,
                        'Max': None,
                        'Missing': df[col].isnull().sum(),
                        'Unique': df[col].nunique(),
                        'Created': col in self.created_features
                    })
        
        stats_df = pd.DataFrame(stats)
        logger.info(f"✅ Generated statistics for {len(stats)} features")
        return stats_df

    # ------------------------------------------------------------------
    # Advanced feature creation
    # ------------------------------------------------------------------
    def create_interaction_features(
        self, df: pd.DataFrame, feature_pairs: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact.
                          If None, uses default pairs.
            
        Returns:
            DataFrame with interaction features added
        """
        df_new = df.copy()
        
        if feature_pairs is None:
            # Default interaction pairs
            feature_pairs = [
                ('SqFt', 'Bedrooms'),
                ('SqFt', 'Bathrooms'),
                ('Bedrooms', 'Bathrooms')
            ]
        
        logger.info(f"🔧 Creating {len(feature_pairs)} interaction features...")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_new.columns and feat2 in df_new.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                df_new[interaction_name] = df_new[feat1] * df_new[feat2]
                self.created_features.append(interaction_name)
                logger.info(f"  ✓ Created {interaction_name}")
        
        logger.info(f"✅ Interaction features created")
        
        return df_new

    def create_polynomial_features(
        self, df: pd.DataFrame, features: List[str], degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df: Input DataFrame
            features: List of feature names to create polynomials for
            degree: Polynomial degree (default: 2 for squared terms)
            
        Returns:
            DataFrame with polynomial features added
        """
        df_new = df.copy()
        
        logger.info(f"🔧 Creating polynomial features (degree={degree})...")
        
        for feat in features:
            if feat in df_new.columns:
                for d in range(2, degree + 1):
                    poly_name = f'{feat}_pow{d}'
                    df_new[poly_name] = df_new[feat] ** d
                    self.created_features.append(poly_name)
                    logger.info(f"  ✓ Created {poly_name}")
        
        logger.info(f"✅ Polynomial features created")
        
        return df_new

    def __repr__(self) -> str:
        """String representation of FeatureEngineer."""
        return (f"FeatureEngineer(created_features={len(self.created_features)}, "
                f"encoders={len(self.label_encoders)}, "
                f"scaler_fitted={self.scaler is not None})")