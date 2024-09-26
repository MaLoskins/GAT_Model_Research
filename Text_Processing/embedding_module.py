# embedding_module.py

import os
import pandas as pd
import numpy as np
from Text_Processing.get_embeddings import generate_embeddings
from Text_Processing.feature_aggregator import FeatureAggregator
from Visualisations.embedding_plotter import EmbeddingPlotter

def generate_and_aggregate_embeddings(
    file_name = None,
    dataframe =None,
    embedding_method="bert",
    embedding_dim=None,
    target_column="text",
    mode="sentence",
    device="cuda",
    apply_dim_reduction=False,
    reduced_dim_size=100,
    plot_embeddings=False,
    plot_node_count=10000,
    categorical_columns=None,
    embedding_column='embedding',
    debug=False
):
    """
    Generates embeddings for the specified text column and aggregates features.
    
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
        pd.DataFrame: DataFrame with aggregated features.
    """
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
    
    # Integrate FeatureAggregator
    if categorical_columns is not None and len(categorical_columns) == 0:
        categorical_columns = None
    
    # Create an instance of the FeatureAggregator
    aggregator = FeatureAggregator(df=processed_df, embedding_column=embedding_column, categorical_columns=categorical_columns)
    
    # Set weights for categorical features (optional)
    if categorical_columns:
        # You can specify weights here; for example:
        weights_dict = {col: 1.0 for col in categorical_columns}
        # Example of setting a specific weight
        # weights_dict['user_screen_name'] = 0.5
        aggregator.set_categorical_weights(weights_dict)
    
    # Perform the aggregation to get the final feature vectors
    final_features = aggregator.aggregate_features()
    
    if debug:
        print("Shape of final aggregated features:", final_features.shape)
    
    # Add aggregated features to the DataFrame
    # It's better to keep them as separate columns or save them externally
    # Here, we'll add them as separate columns
    # Note: Depending on the size, this might not be efficient
    # Alternatively, you can save them as separate files or use them directly for modeling
    # For demonstration, we'll skip adding them to the DataFrame
    
    # Plotting the Embeddings (if enabled)
    if plot_embeddings:
        plotter = EmbeddingPlotter(
            color_column='label',        # Ensure 'label' exists in your DataFrame
            text_column=target_column,   # Use the selected text column
            n=plot_node_count,           # Number of samples to plot
            name='embedding_visualization',
            renderer='browser'           # Or 'notebook' if using Jupyter
        )
    
        plotter.plot_updated_embeddings(processed_df)
    
    return processed_df, final_features
