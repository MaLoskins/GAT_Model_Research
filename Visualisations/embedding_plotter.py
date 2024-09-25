# Visualisations/embedding_plotter.py

import plotly.graph_objs as go
import plotly.io as pio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

class EmbeddingPlotter:
    """
    A class to plot word embeddings in 3D space using t-SNE and Plotly.

    Features:
    - Samples a specified number of embeddings.
    - Reduces dimensionality to 3D using t-SNE.
    - Filters out outliers based on z-score.
    - Colors points based on a specified column.
    - Creates interactive 3D scatter plots.
    - Saves the plot as an HTML file.
    """

    def __init__(
        self,
        color_column='label',
        text_column='tokenized_text',  # Updated default to 'tokenized_text'
        n=1000,
        name='gat',
        word_limit=100,
        renderer='browser'
    ):
        """
        Initializes the EmbeddingPlotter.

        :param color_column: The column name to color the embeddings.
        :param text_column: The column name containing tokenized text.
        :param n: Number of samples to plot.
        :param name: Name used in plot title and output filename.
        :param word_limit: Maximum allowed length for words in tokens.
        :param renderer: Plotly renderer to use (default: 'browser').
        """
        self.color_column = color_column
        self.text_column = text_column
        self.n = n
        self.name = name
        self.word_limit = word_limit
        self.renderer = renderer

        # Set Plotly renderer
        pio.renderers.default = self.renderer

    def plot_updated_embeddings(self, df_embeddings):
        """
        Plots the embeddings in 3D space after reducing dimensionality with t-SNE.

        :param df_embeddings: A pandas DataFrame containing embeddings and other columns.
                              Must include 'embedding', color_column, and text_column.
        """
        start_time = time.time()

        # Validate required columns
        required_columns = ['embedding', self.color_column, self.text_column]
        for col in required_columns:
            if col not in df_embeddings.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")

        # Sample 'n' observations
        if len(df_embeddings) < self.n:
            print(f"[WARN] DataFrame has fewer than {self.n} rows. Using all available rows.")
            df_sampled = df_embeddings.copy()
        else:
            df_sampled = df_embeddings.sample(n=self.n, random_state=42)

        # Extract the embeddings and convert to a NumPy array
        try:
            embedding_matrix = np.vstack(df_sampled['embedding'].values)
        except Exception as e:
            raise ValueError(f"Error converting 'embedding' column to NumPy array: {e}")

        # Reduce the dimensionality to 3D using t-SNE
        print("Applying t-SNE for dimensionality reduction...")
        tsne = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
        reduced_embeddings_tsne = tsne.fit_transform(embedding_matrix)
        print("t-SNE reduction completed.")

        # Compute the mean and standard deviation of the reduced embeddings
        mean = np.mean(reduced_embeddings_tsne, axis=0)
        std = np.std(reduced_embeddings_tsne, axis=0)

        # Compute the z-score for each point
        z_scores = np.abs((reduced_embeddings_tsne - mean) / std)

        # Set a threshold for outliers (e.g., z-score > 3)
        threshold = 3  # adjust as needed
        mask = (z_scores < threshold).all(axis=1)

        # Filter the embeddings and other arrays
        filtered_embeddings = reduced_embeddings_tsne[mask]
        df_sampled_filtered = df_sampled[mask].reset_index(drop=True)

        # Prepare colors based on the specified column
        unique_values = df_sampled_filtered[self.color_column].unique()
        color_map = plt.cm.get_cmap('tab10', len(unique_values))

        # Create a mapping from unique values to colors
        color_dict = {value: color_map(i) for i, value in enumerate(unique_values)}

        # Apply the mapping to the DataFrame column
        colors = df_sampled_filtered[self.color_column].map(color_dict)

        # Convert colors to a list
        colors_list = colors.tolist()

        # Create hover text for each point
        hover_text = df_sampled_filtered.apply(
            lambda row: f"{self.color_column}: {row[self.color_column]}, Text: {' '.join(row[self.text_column])}",
            axis=1
        )

        # Create a trace for the 3D scatter plot
        trace = go.Scatter3d(
            x=filtered_embeddings[:, 0],
            y=filtered_embeddings[:, 1],
            z=filtered_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors_list,  # Color based on the specified column
                opacity=0.7
            ),
            text=hover_text.tolist(),  # Hover text
            hoverinfo='text'
        )

        # Set up the layout
        layout = go.Layout(
            title=f't-SNE - {self.name} Embeddings colored by {self.color_column}',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # Create the figure
        fig = go.Figure(data=[trace], layout=layout)

        # Show the plot
        pio.show(fig)

        # Save as HTML file
        output_html = f"interactive_plot_{self.name}_{self.color_column}.html"
        try:
            fig.write_html(output_html)
            print(f"[INFO] Plot saved as {output_html}")
        except Exception as e:
            print(f"[ERROR] Error saving plot as HTML: {e}")

        # End Timer with note after
        end_time = time.time()
        print(f"Time taken for {len(df_sampled_filtered)} points: {end_time - start_time:.2f} seconds")
