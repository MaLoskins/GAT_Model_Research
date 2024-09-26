# feature_aggregator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class FeatureAggregator:
    def __init__(self, df, embedding_column, categorical_columns=None):
        self.df = df
        self.embedding_column = embedding_column
        self.categorical_columns = categorical_columns or []
        # Initialize weights for categorical features to 1.0
        self.categorical_weights = {col: 1.0 for col in self.categorical_columns}
        # Extract embeddings
        self.embeddings = df[embedding_column].tolist()
        # Extract categorical data if any
        if self.categorical_columns:
            self.categorical_data = df[self.categorical_columns]
        else:
            self.categorical_data = None

    def set_categorical_weights(self, weights_dict):
        """
        Update weights for the categorical features.

        Args:
            weights_dict (dict): Dictionary mapping column names to weights.
        """
        for col, weight in weights_dict.items():
            if col in self.categorical_weights:
                self.categorical_weights[col] = weight
            else:
                raise ValueError(f"Column '{col}' not in categorical columns.")

    def aggregate_features(self):
        """
        Aggregate embeddings and weighted categorical features.

        Returns:
            np.ndarray: Aggregated feature vectors.
        """
        embeddings_array = np.array(self.embeddings)
        if self.categorical_data is not None and not self.categorical_data.empty:
            # One-hot encode the categorical features
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # Ensure categorical data is string type
            categorical_data_str = self.categorical_data.astype(str)
            encoded_categorical = ohe.fit_transform(categorical_data_str)

            # Apply weights to the encoded features
            feature_names = ohe.get_feature_names_out(self.categorical_columns)
            encoded_df = pd.DataFrame(encoded_categorical, columns=feature_names)

            for col in self.categorical_columns:
                weight = self.categorical_weights[col]
                # Get columns corresponding to this categorical feature
                feature_cols = [c for c in feature_names if c.startswith(col + '_')]
                encoded_df[feature_cols] *= weight

            # Concatenate embeddings with weighted categorical features
            final_features = np.hstack([embeddings_array, encoded_df.values])
        else:
            # No categorical features; return embeddings as final features
            final_features = embeddings_array
        return final_features
