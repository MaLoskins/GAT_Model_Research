# embedding_module.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Text_Processing.get_embeddings import generate_embeddings
from Text_Processing.feature_aggregator import FeatureAggregatorSimple  # Ensure correct import path
from Visualisations.embedding_plotter import EmbeddingPlotter

class EmbeddingModule:
    def __init__(self):
        pass

    def generate_and_aggregate_embeddings(
        self,
        file_name=None,
        dataframe=None,
        embedding_method="bert",
        embedding_dim=None,
        target_column="text",
        mode="sentence",
        device="cuda",
        apply_dim_reduction=False,
        reduced_dim_size=100,
        embedding_column='embedding',
        debug=False
    ):
        """
        Generates embeddings for the specified text column.

        Parameters:
            file_name (str): Name of the CSV file without extension.
            embedding_method (str): Embedding method to use ('glove', 'word2vec', 'bert').
            embedding_dim (int): Dimension of embeddings. Set None for BERT.
            target_column (str): Column name containing text data.
            mode (str): Embedding mode ('sentence' or 'word').
            device (str): Device to use for computation ('cuda' or 'cpu').
            apply_dim_reduction (bool): Whether to apply dimensionality reduction.
            reduced_dim_size (int): Reduced dimension size if dimensionality reduction is applied.
            plot_embeddings (bool): Whether to plot the embeddings.
            plot_node_count (int): Number of nodes to plot.
            categorical_columns (list): List of categorical columns to include.
            embedding_column (str): Column name containing embedding data.
            debug (bool): Enable debug mode for verbose output.
            
        Returns:
            pd.DataFrame: DataFrame with embeddings.
            torch.Tensor: Tensor of shape (N, sentence_dim) containing sentence embeddings.
        """

        # Convert embedding_dim if None to 768 for BERT
        if embedding_method.lower() == "bert":
            embedding_dim = embedding_dim or 768

        if file_name is None and dataframe is None:
            raise ValueError("Please provide either 'file_name' or 'dataframe'.")
        
        if dataframe is None:
            # Define input and output paths
            input_file_path = f"CSV_Files/{file_name}.csv"
        elif dataframe is not None:
            input_dataframe = dataframe
            input_file_path = None     
        
        # Define output paths based on mode
        output_csv_path = f"CSV_Files/{file_name}_embeddings.csv"
        output_pkl_path = f"Pickle_Files/{file_name}_embeddings.pkl"
        
        if mode == "word":
            output_pkl_path = f"Pickle_Files/{file_name}_word-sentence_embeddings.pkl"
        
        # Define cache/model directories based on embedding method
        glove_cache_path = "Glove_Cache"
        word2vec_model_path = "Word2Vec_Model"
        bert_cache_dir = "Bert_Cache"
        
        # Make directories for embedding caches/models if they don't exist
        if embedding_method == "glove":
            os.makedirs("Glove_Cache", exist_ok=True)
        elif embedding_method == "word2vec":
            os.makedirs("Word2Vec_Model", exist_ok=True)
        elif embedding_method == "bert":
            os.makedirs("Bert_Cache", exist_ok=True)
        
        # Generate embeddings
        processed_df = generate_embeddings(
            embedding_method=embedding_method,
            embedding_dim=embedding_dim,
            target_column=target_column,
            dataframe=input_dataframe,
            input_file_path=input_file_path,
            output_csv_path=output_csv_path,
            output_pkl_path=output_pkl_path,
            glove_cache_path=glove_cache_path,
            word2vec_model_path=word2vec_model_path,
            bert_cache_dir=bert_cache_dir,
            device=device,
            apply_dim_reduction=apply_dim_reduction,
            reduced_dim_size=reduced_dim_size,
            mode=mode,
            debug=debug
        )
        

        # Load the DataFrame from the pickle file
        embeddings_pkl_path = output_pkl_path
        processed_df = pd.read_pickle(embeddings_pkl_path)
        
        # Access the 'embedding' column and convert it to a NumPy array
        embeddings_list = processed_df[embedding_column].tolist()
        
        if mode == "sentence":
            embeddings_array = np.array(embeddings_list)
        elif mode == "word":
            max_words = max(len(emb) for emb in embeddings_list)
            embedding_dim_size = processed_df[embedding_column][0].shape[1] if len(embeddings_list[0].shape) > 1 else processed_df[embedding_column][0].shape[0]
            embeddings_array = np.zeros((len(embeddings_list), max_words, embedding_dim_size))
        
            for i, emb in enumerate(embeddings_list):
                embeddings_array[i, :emb.shape[0], :] = emb
        
        if debug:
            print("Shape of DataFrame:", processed_df.shape)
            print(f"Shape of embeddings array in feature '{embedding_column}':", embeddings_array.shape)
        
        # Convert embeddings to torch tensor
        sentence_embeddings = torch.tensor(embeddings_array, dtype=torch.float32).to(device)
        
        if debug:
            print("Shape of sentence_embeddings tensor:", sentence_embeddings.shape)
        
        return processed_df, sentence_embeddings

    def aggregate_features(self, 
                        processed_df, 
                        embedding_column, 
                        categorical_columns, 
                        categorical_dims, 
                        sentence_dim=768):
        """
        Aggregates sentence embeddings with categorical features via concatenation.

        Args:
            processed_df (pd.DataFrame): DataFrame containing embeddings and categorical data.
            embedding_column (str): Column name containing sentence embeddings.
            categorical_columns (list of str): List of categorical column names to include.
            categorical_dims (dict): Dictionary mapping categorical columns to number of categories.
            sentence_dim (int): Dimension of sentence embeddings.

        Returns:
            torch.Tensor: Tensor of shape (N, 2 * sentence_dim) containing final features.
        """

        # Extract sentence embeddings as a torch.Tensor
        sentence_embeddings = torch.tensor(processed_df[embedding_column].tolist(), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare categorical data as a dictionary of tensors
        categorical_data = {}
        for col in categorical_columns:
            cat_values = processed_df[col].values
            max_index = cat_values.max()
            if max_index >= categorical_dims[col]:
                raise ValueError(f"Categorical column '{col}' has index {max_index} which exceeds its dimension {categorical_dims[col]}.")
            categorical_data[col] = torch.tensor(cat_values, dtype=torch.long).to(sentence_embeddings.device)

        # Initialize FeatureAggregatorSimple
        aggregator = FeatureAggregatorSimple(
            sentence_dim=sentence_dim,
            categorical_columns=categorical_columns,
            categorical_dims=categorical_dims,
            categorical_embed_dim=sentence_dim  # To ensure projected_cats has sentence_dim
        ).to(sentence_embeddings.device)

        # Optionally set weights for categorical features
        # Example: all weights set to 1.0
        weights_dict = {col: 1.0 for col in categorical_columns}
        aggregator.set_categorical_weights(weights_dict)

        aggregator.eval()  # Set to evaluation mode

        with torch.no_grad():
            final_features = aggregator(sentence_embeddings, categorical_data)  # Shape: (N, 2 * sentence_dim)

        return final_features