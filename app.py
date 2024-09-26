# app.py

import os
import io
import pandas as pd
import streamlit as st
import numpy as np
import networkx as nx
from utils import (
    encode_categorical_columns,
    run_processing,
    load_data,
    align_dataframes,
    create_binning_sliders,
    plot_density_plots,
    download_data
)
from Text_Processing.embedding_module import EmbeddingModule
from Visualisations.embedding_plotter import EmbeddingPlotter
from Binning.data_binner import DataBinner
from Graph_Processing.graph_creator import GraphCreator
from Graph_Processing.graph_visualizer import GraphVisualizer
import matplotlib.pyplot as plt

# Initialize EmbeddingModule
embedding_module = EmbeddingModule()

# ============================
# Configuration and Setup
# ============================

# Set Streamlit page configuration
st.set_page_config(
    page_title="🛠️ GAT Preprocessing Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default menu and footer for a cleaner interface
def hide_streamlit_style():
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_style, unsafe_allow_html=True)

hide_streamlit_style()

# ============================
# Initialize Session State
# ============================

# Initialize necessary session state variables
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if 'aggregated_features' not in st.session_state:
    st.session_state.aggregated_features = None

if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False

# New session state variables for Graph Processing
if 'graph_created' not in st.session_state:
    st.session_state.graph_created = False

if 'graph' not in st.session_state:
    st.session_state.graph = None

# ============================
# Main Application
# ===========================

st.title('🛠️ Data Processing and Binning Application')

# Sidebar for tab selection and corresponding options
with st.sidebar:
    st.header("📂 Upload & Settings")
    
    # Tab selection via radio buttons
    active_tab = st.radio(
        "🔀 Select Mode:",
        ("📊 Binning", "🔢 Generate Embeddings", "🌐 Graph Processing")
    )
    
    st.markdown("---")
    
    # Conditional Sidebar Content Based on Active Tab
    if active_tab == "📊 Binning":
        st.header("📂 Binning Settings")
        
        # File Upload for Binning
        uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'])
        
        # Output File Type Selection
        file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
        
        
        st.markdown("---")
        
        st.header("⚙️ Binning Options")
        
        # Binning Method Selection
        binning_method = st.selectbox('🔧 Select Binning Method', ['Quantile', 'Equal Width'])
        
        if binning_method == 'Quantile':
            st.warning("⚠️ **Note:** Using Quantile binning will prevent the output of 'Original Data' Density Plots due to granularity.")
        
    elif active_tab == "🔢 Generate Embeddings":
        st.header("🔠 Embedding Settings")
        
        # File Upload for Embedding Generation
        uploaded_embedding_file = st.file_uploader("📤 Upload Binned Data (CSV or Pickle)", type=['csv', 'pkl'])
        
        st.markdown("---")
        
        st.header("⚙️ Embedding Options")
        
        # Embedding Method Selection
        embedding_method = st.selectbox("🔧 Select Embedding Method", ["glove", "word2vec", "bert"], index=2)
        
        # Embedding Dimension Selection
        if embedding_method == "bert":
            embedding_dim = None  # Default BERT dimension
        else:
            embedding_dim = st.number_input("📏 Embedding Dimension", min_value=50, max_value=1024, value=100, step=50)
        
        # Embedding Mode Selection
        mode = st.selectbox("🔀 Embedding Mode", ["sentence", "word"], index=0)
        
        # Device Selection for Computation
        device = st.selectbox("⚙️ Device for Computation", ["cpu","cuda"], index=1)
        
        # Dimensionality Reduction Option
        apply_dim_reduction = st.checkbox("🧠 Apply Dimensionality Reduction (PCA)", value=False)
        reduced_dim_size = 100
        if apply_dim_reduction:
            reduced_dim_size = st.number_input("🔢 Reduced Dimension Size", min_value=50, max_value=500, value=100, step=50)
        
        # Embedding Plotting Option
        plot_embeddings = st.checkbox("📊 Plot Embeddings", value=False)
        if plot_embeddings:
            plot_node_count = st.number_input("📈 Number of Nodes to Plot", min_value=100, max_value=100000, value=10000, step=1000)
            raw_or_aggregated = st.selectbox("📊 Plot Raw or Aggregated Embeddings", ["Raw", "Aggregated"], index=1)
        else:
            plot_node_count = 10000  # Default value
            raw_or_aggregated = "Raw"  # Default value
        
        st.markdown("---")
        
        st.header("🗂️ Embedding Text Settings")
        
        # Text Column Selection will be handled in the main content area

    elif active_tab == "🌐 Graph Processing":
        st.header("🌐 Heterogeneous Graph Creation & Visualization")
        
        # No sidebar-specific settings for the graph tab
        st.markdown("### 📌 Instructions")
        st.write("""
            - Ensure that you have generated embeddings in the '🔢 Generate Embeddings' tab.
            - Select the source and target ID columns to establish relationships.
            - Specify the type of relationship (e.g., replies_to, mentions).
            - Create and visualize the graph.
        """)

# Main Content Area
if active_tab == "📊 Binning":
    # ... [Existing Binning Tab Code]
    pass  # For brevity, assuming the existing code is unchanged

elif active_tab == "🔢 Generate Embeddings":
    # ... [Existing Embedding Generation Tab Code]
    pass  # For brevity, assuming the existing code is unchanged

elif active_tab == "🌐 Graph Processing":
    st.header("🌐 Heterogeneous Graph Creation & Visualization")
    
    # Check if embeddings have been generated
    if st.session_state.get('embeddings_generated', False):
        st.info("🔄 **Embeddings are available. Proceed to create and visualize the graph.**")
        
        # Select columns for graph creation
        st.subheader("🔗 Select ID Columns for Graph Relationships")
        
        # Identify potential ID columns
        id_columns = st.session_state.processed_df.columns.tolist()
        
        # Allow user to select source and target ID columns
        source_id_col = st.selectbox("🔸 Select Source ID Column", options=id_columns, key="source_id")
        target_id_col = st.selectbox("🔹 Select Target ID Column", options=id_columns, key="target_id")
        
        # Allow user to specify relationship type
        relationship_type = st.text_input("🔗 Relationship Type", value="replies_to")
        
        # Visualization Method Selection
        visualization_method = st.selectbox(
            "📊 Select Visualization Method",
            options=["Agraph", "PyVis"],
            index=0,
            help="Choose between 'Agraph' for interactive visualization or 'PyVis' for handling larger graphs."
        )
        
        # Button to create graph
        if st.button("🚀 Create Heterogeneous Graph"):
            if source_id_col == target_id_col:
                st.error("❌ Source and Target ID columns must be different.")
            else:
                try:
                    # Identify rows with nulls in selected columns
                    null_rows = st.session_state.processed_df[
                        st.session_state.processed_df[[source_id_col, target_id_col]].isnull().any(axis=1)
                    ]
                    num_null_rows = null_rows.shape[0]
                    
                    if num_null_rows > 0:
                        st.warning(f"⚠️ Found {num_null_rows} rows with null values in selected columns. These rows will be removed.")
                        # Optionally, display the rows being removed
                        with st.expander("🔍 View Rows to be Removed"):
                            st.dataframe(null_rows)
                        # Drop the rows with nulls in selected columns
                        st.session_state.processed_df = st.session_state.processed_df.dropna(subset=[source_id_col, target_id_col])
                    
                    with st.spinner('Creating graph...'):
                        # Convert ID columns to string with prefixes
                        st.session_state.processed_df[source_id_col] = "source_" + st.session_state.processed_df[source_id_col].astype(str)
                        st.session_state.processed_df[target_id_col] = "target_" + st.session_state.processed_df[target_id_col].astype(str)
                        
                        # Create graph
                        graph_creator = GraphCreator(st.session_state.processed_df)
                        graph = graph_creator.create_graph(source_id_col, target_id_col, relationship_type)
                        st.session_state.graph = graph
                        st.session_state.graph_created = True
                    st.success("✅ Heterogeneous graph created successfully!")
                    
                    # Display graph statistics
                    st.markdown("### 📊 Graph Statistics")
                    st.write(f"**Number of Nodes:** {graph.number_of_nodes()}")
                    st.write(f"**Number of Edges:** {graph.number_of_edges()}")
                    st.write(f"**Number of Relationships:** {len(set(nx.get_edge_attributes(graph, 'relationship').values()))}")

                    # Identify and display external target IDs
                    internal_ids = set(st.session_state.processed_df[source_id_col].unique())
                    external_target_ids = st.session_state.processed_df[
                        ~st.session_state.processed_df[target_id_col].isin(internal_ids)
                    ][target_id_col].unique()
                    
                    st.write(f"**Number of External Target IDs:** {len(external_target_ids)}")
                    if len(external_target_ids) > 0:
                        st.write(f"**Sample External Target IDs:** {external_target_ids[:5]}")
    
                except Exception as e:
                    st.error(f"❌ Error during graph creation: {e}")
    
        # If graph is created, provide visualization option
        if st.session_state.get('graph_created', False):
            st.markdown("---")
            st.subheader("📈 Visualize the Heterogeneous Graph")
            
            # Optionally, allow user to select number of nodes to visualize
            node_limit = st.number_input("🔢 Number of Nodes to Visualize", min_value=10, max_value=10000, value=1000, step=10)
            
            # Warning for Agraph with large nodes
            if visualization_method == "Agraph" and node_limit > 1000:
                st.warning(
                    "⚠️ You have selected over 1000 nodes for Agraph visualization. "
                    "Unless you have high computational resources, please consider selecting **PyVis** for better performance."
                )
            
            # Button to visualize graph
            if st.button("📈 Visualize Graph"):
                try:
                    with st.spinner('Generating graph visualization...'):
                        # Extract a subgraph if the total nodes exceed the limit
                        if st.session_state.graph.number_of_nodes() > node_limit:
                            # Sample nodes
                            sampled_nodes = list(st.session_state.graph.nodes)[:node_limit]
                            subgraph = st.session_state.graph.subgraph(sampled_nodes).copy()
                        else:
                            subgraph = st.session_state.graph
                        
                        # Optionally, convert to simple DiGraph to handle multiple edges
                        if isinstance(subgraph, nx.MultiDiGraph):
                            subgraph = nx.DiGraph(subgraph)
                        
                        # Initialize GraphVisualizer
                        visualizer = GraphVisualizer(subgraph)
                        
                        # Choose visualization method
                        if visualization_method == "Agraph":
                            visualizer.visualize_agraph()
                        elif visualization_method == "PyVis":
                            visualizer.visualize_pyvis()
                        else:
                            st.error("❌ Unsupported visualization method selected.")
                    
                except Exception as e:
                    st.error(f"❌ Error during graph visualization: {e}")
    else:
        st.warning("⚠️ **Please generate embeddings first in the '🔢 Generate Embeddings' tab.**")
