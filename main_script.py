# main_script.py
import os
from Text_Processing.get_embeddings import generate_embeddings


# File name of the CSV file (without the extension)
file_name = "sydneysiege"

# Specify parameters
embedding_method = "bert"  # Options: 'glove', 'word2vec', 'bert'
embedding_dim = None       # Set None for BERT; specify integer for others
target_column = "text"
input_file_path = f"CSV_Files/{file_name}.csv"

# Make "CSV_Files" directory in the same directory as this script if it doesn't exist
os.makedirs("CSV_Files", exist_ok=True)
output_csv_path = f"CSV_Files/{file_name}_embeddings.csv"
# Make "CSV_Files" directory in the same directory as this script if it doesn't exist
os.makedirs("Pickle_Files", exist_ok=True)
output_pkl_path = f"Pickle_Files/{file_name}_embeddings.pkl"

glove_cache_path = "Glove_Cache"          # Required if using GloVe
word2vec_model_path = "Word2Vec_Model"     # Required if using Word2Vec
bert_cache_dir = "Bert_Cache"             # Optional
device = "cuda"                                   # 'cuda' or 'cpu'
apply_dim_reduction = False
reduced_dim_size = 100
debug = True  # Set to True to enable debug statements

# Make directories "Bert_Cache", "Glove_Cache", "Word2Vec_Model" in the same directory as this script if they don't exist
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
    debug=debug
)

# main_script.py

import pandas as pd
import numpy as np

# Load the DataFrame from the pickle file
embeddings_pkl_path = f"Pickle_Files/{file_name}_embeddings.pkl"
processed_df = pd.read_pickle(embeddings_pkl_path)

print(processed_df.columns)
# Access the 'embedding' column and convert it to a NumPy array
embeddings_list = processed_df['embedding'].tolist()

# Check if embeddings are lists or arrays and convert if necessary
embeddings_array = np.array(embeddings_list)

# Alternatively, use np.vstack if embeddings are arrays
# embeddings_array = np.vstack(embeddings_list)

# Check the shape of the embeddings array
print("Shape of embeddings array:", embeddings_array.shape)
