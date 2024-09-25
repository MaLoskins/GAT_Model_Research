# get_embeddings.py

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from Text_Processing.text_preprocessor import TextPreprocessor
from Text_Processing.embedding_creator import EmbeddingCreator

def generate_embeddings(
    embedding_method,
    embedding_dim,
    target_column,
    input_file_path,
    output_csv_path,
    output_pkl_path,
    glove_cache_path=None,
    word2vec_model_path=None,
    bert_model_name="bert-base-uncased",
    bert_cache_dir=None,
    device="cuda",
    apply_dim_reduction=False,
    reduced_dim_size=100,
    mode="sentence",
    debug=False
):
    if debug:
        print("Starting embedding generation process...")

    # Dynamically set EMBEDDING_DIM based on EMBEDDING_METHOD if not provided
    if embedding_method.lower() == "bert":
        embedding_dim = embedding_dim or 768  # BERT's hidden size
    elif embedding_method.lower() in ["glove", "word2vec"]:
        if embedding_dim is None:
            raise ValueError(f"embedding_dim must be specified for {embedding_method}.")
    else:
        raise ValueError("Unsupported embedding_method. Choose from 'glove', 'word2vec', or 'bert'.")

    # Check for existing embeddings
    embeddings_exist = os.path.exists(output_csv_path) and os.path.exists(output_pkl_path)

    if not embeddings_exist:
        if debug:
            print("Embeddings do not exist. Generating new embeddings...")

        # Check if the input CSV file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"CSV file not found: {input_file_path}")

        # Read the CSV file
        if debug:
            print(f"Reading CSV data from {input_file_path}...")
        df = pd.read_csv(input_file_path)
        if debug:
            print("CSV data loaded.")

        # Initialize the TextPreprocessor
        preprocessor = TextPreprocessor(
            target_column=target_column,
            include_stopwords=True,
            remove_ats=True,
            word_limit=100
        )

        # Preprocess the DataFrame
        if debug:
            print("Preprocessing text data...")
        processed_df = preprocessor.clean_text(df)
        if debug:
            print("Text preprocessing completed.")

        # Initialize EmbeddingCreator
        embedding_creator = EmbeddingCreator(
            embedding_method=embedding_method,
            embedding_dim=embedding_dim,
            glove_cache_path=glove_cache_path,
            word2vec_model_path=word2vec_model_path,
            bert_model_name=bert_model_name,
            bert_cache_dir=bert_cache_dir,
            device=device
        )

        # Generate embeddings
        if debug:
            print("Generating embeddings...")
        if mode == "sentence":
            processed_df['embedding'] = processed_df['tokenized_text'].apply(
                lambda tokens: embedding_creator.get_embedding(tokens)
            )
        elif mode == "word":
            processed_df['embedding'] = processed_df['tokenized_text'].apply(
                lambda tokens: embedding_creator.get_word_embeddings(tokens)
            )
        else:
            raise ValueError("Unsupported mode. Choose 'sentence' or 'word'.")

        if debug:
            print("Embedding generation completed.")

        # Remove rows with zero embeddings (if any)
        initial_count = len(processed_df)
        if mode == "sentence":
            processed_df = processed_df[processed_df['embedding'].apply(lambda x: not np.all(x == 0))].reset_index(drop=True)
        elif mode == "word":
            processed_df = processed_df[processed_df['embedding'].apply(lambda x: x.shape[0] > 0)].reset_index(drop=True)
        removed_count = initial_count - len(processed_df)
        if removed_count > 0 and debug:
            print(f"Removed {removed_count} rows with zero embeddings.")

        # Apply PCA for Dimensionality Reduction (if enabled and mode is sentence)
        if apply_dim_reduction and mode == "sentence":
            if debug:
                print("Applying PCA for dimensionality reduction...")
            from sklearn.decomposition import PCA
            embeddings = np.vstack(processed_df['embedding'].values)
            pca = PCA(n_components=reduced_dim_size, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)
            processed_df['embedding'] = list(reduced_embeddings)
            if debug:
                print(f"PCA reduction completed. Embedding dimensions reduced to {reduced_dim_size}.")

        # Save Embeddings to CSV and Pickle
        if debug:
            print(f"Saving embeddings to {output_csv_path} and {output_pkl_path}...")
        processed_df.to_csv(output_csv_path, index=False)
        processed_df.to_pickle(output_pkl_path)
        if debug:
            print("Embeddings saved successfully.")
    else:
        if debug:
            print("Loading existing embeddings...")
        processed_df = pd.read_pickle(output_pkl_path)
        if debug:
            print("Embeddings loaded successfully.")

    if debug:
        print("Embedding generation process completed.")

    return processed_df
