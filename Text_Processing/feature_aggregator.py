import torch
import torch.nn as nn
import pandas as pd

class FeatureAggregatorSimple(nn.Module):
    def __init__(self, 
                 sentence_dim, 
                 categorical_columns, 
                 categorical_dims, 
                 categorical_embed_dim=64):
        """
        Simple concatenation of sentence embeddings and categorical features.

        Args:
            sentence_dim (int): Dimension of sentence embeddings (n).
            categorical_columns (list of str): List of categorical column names.
            categorical_dims (dict): Dictionary mapping categorical columns to number of categories.
            categorical_embed_dim (int, optional): Embedding dimension for each categorical feature. Defaults to 64.
        """
        super(FeatureAggregatorSimple, self).__init__()
        self.categorical_columns = categorical_columns
        self.categorical_embed_dim = categorical_embed_dim

        # Initialize embedding layers for each categorical feature
        self.embedding_layers = nn.ModuleDict()
        for col in categorical_columns:
            num_categories = categorical_dims[col]
            self.embedding_layers[col] = nn.Embedding(num_embeddings=num_categories, 
                                                      embedding_dim=categorical_embed_dim)
        
        # Linear layer to project combined categorical embeddings to sentence_dim
        self.project_categorical = nn.Linear(len(categorical_columns) * categorical_embed_dim, sentence_dim)
        
        # Initialize weights (optional but recommended)
        self._init_weights()
        
        # Initialize categorical weights to 1.0
        self.categorical_weights = {col: 1.0 for col in categorical_columns}
        
    def _init_weights(self):
        """
        Initialize weights for embedding layers and linear layer.
        """
        for emb in self.embedding_layers.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.project_categorical.weight)
        nn.init.zeros_(self.project_categorical.bias)
        
    def set_categorical_weights(self, weights_dict):
        """
        Set scalar weights for each categorical feature.

        Args:
            weights_dict (dict): Dictionary mapping categorical columns to scalar weights.
        """
        for col, weight in weights_dict.items():
            if col in self.categorical_weights:
                self.categorical_weights[col] = weight
            else:
                raise ValueError(f"Column '{col}' is not a valid categorical feature.")
        
    def forward(self, sentence_embeddings, categorical_data):
        """
        Forward pass to concatenate sentence embeddings with categorical features.

        Args:
            sentence_embeddings (torch.Tensor): Tensor of shape (N, sentence_dim).
            categorical_data (dict): Dictionary where keys are categorical column names and values are 
                                     integer-encoded tensors of shape (N,).

        Returns:
            torch.Tensor: Tensor of shape (N, 2 * sentence_dim).
        """
        embedded_cats = []
        for col in self.categorical_columns:
            # Get integer-encoded categorical data
            cat_tensor = categorical_data[col]  # Shape: (N,)
            # Pass through embedding layer
            emb = self.embedding_layers[col](cat_tensor)  # Shape: (N, categorical_embed_dim)
            # Apply scalar weight
            weight = self.categorical_weights[col]
            emb = emb * weight
            embedded_cats.append(emb)
        
        # Concatenate all embedded categorical features
        concatenated_cats = torch.cat(embedded_cats, dim=1)  # Shape: (N, num_cats * embed_dim)
        
        # Project concatenated categorical embeddings to match sentence_dim
        projected_cats = self.project_categorical(concatenated_cats)  # Shape: (N, sentence_dim)
        
        # Concatenate sentence embeddings with processed categorical features
        final_features = torch.cat([sentence_embeddings, projected_cats], dim=1)  # Shape: (N, 2 * sentence_dim)
        
        return final_features
