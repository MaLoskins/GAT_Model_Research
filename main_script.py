# main_script.py
import os
import argparse
from Text_Processing.get_embeddings import generate_embeddings
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate Sentence or Word Embeddings.")
    parser.add_argument('--file_name', type=str, default="sydneysiege", help='Name of the CSV file without extension.')
    parser.add_argument('--embedding_method', type=str, default="bert", choices=['glove', 'word2vec', 'bert'], help='Embedding method to use.')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Dimension of embeddings. Set None for BERT.')
    parser.add_argument('--target_column', type=str, default="text", help='Column name containing text data.')
    parser.add_argument('--mode', type=str, default="sentence", choices=['sentence', 'word'], help='Embedding mode: sentence or word.')
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help='Device to use for computation.')
    parser.add_argument('--apply_dim_reduction', action='store_true', help='Whether to apply dimensionality reduction.')
    parser.add_argument('--reduced_dim_size', type=int, default=100, help='Reduced dimension size if dimensionality reduction is applied.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output.')

    args = parser.parse_args()

    # File name of the CSV file (without the extension)
    file_name = args.file_name

    # Specify parameters
    embedding_method = args.embedding_method  # Options: 'glove', 'word2vec', 'bert'
    embedding_dim = args.embedding_dim       # Set None for BERT; specify integer for others
    target_column = args.target_column
    mode = args.mode  # 'sentence' or 'word'
    device = args.device
    apply_dim_reduction = args.apply_dim_reduction
    reduced_dim_size = args.reduced_dim_size
    debug = args.debug

    # Define input and output paths
    input_file_path = f"CSV_Files/{file_name}.csv"

    # Make "CSV_Files" and "Pickle_Files" directories if they don't exist
    os.makedirs("CSV_Files", exist_ok=True)
    os.makedirs("Pickle_Files", exist_ok=True)

    # Define output paths based on mode
    output_csv_path = f"CSV_Files/{file_name}_embeddings.csv"
    output_pkl_path = f"Pickle_Files/{file_name}_embeddings.pkl"

    if mode == "word":
        output_pkl_path = f"Pickle_Files/{file_name}_word-sentence_embeddings.pkl"

    # Define cache/model directories based on embedding method
    glove_cache_path = "Glove_Cache"          # Required if using GloVe
    word2vec_model_path = "Word2Vec_Model"    # Required if using Word2Vec
    bert_cache_dir = "Bert_Cache"             # Optional

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
    embeddings_list = processed_df['embedding'].tolist()

    if mode == "sentence":
        # For sentence embeddings: [n x embedding_dim]
        embeddings_array = np.array(embeddings_list)
    elif mode == "word":
        # For word embeddings: list of [words_in_sentence x embedding_dim]
        # Determine the maximum number of words to pad sequences
        max_words = max(len(emb) for emb in embeddings_list)
        embedding_dim_size = processed_df['embedding'][0].shape[1] if len(embeddings_list[0].shape) > 1 else processed_df['embedding'][0].shape[0]
        embeddings_array = np.zeros((len(embeddings_list), max_words, embedding_dim_size))

        for i, emb in enumerate(embeddings_list):
            embeddings_array[i, :emb.shape[0], :] = emb

    # Check the shape of the embeddings array
    print("Shape of embeddings array:", embeddings_array.shape)

    from Visualisations.embedding_plotter import EmbeddingPlotter

    plotter = EmbeddingPlotter(
        color_column='label',        # The column to use for coloring
        text_column='tokenized_text',
        n=20000,                      # Number of samples to plot
        name='embedding_visualization',
        renderer='browser'           # Or 'notebook' if using Jupyter
    )

    plotter.plot_updated_embeddings(processed_df)

if __name__ == "__main__":
    main()
