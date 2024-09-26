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
    # Proceed only if a file is uploaded
    if uploaded_file is not None:
        with st.spinner('Loading and processing data...'):
            Data, error = load_data(file_type, uploaded_file)
        
        if error:
            st.error(error)
        else:
            # Display data preview
            st.subheader('📊 Data Preview (Post Processing)')
            st.dataframe(Data.head())
            
            # Define identifier columns to exclude from binning and plotting
            IDENTIFIER_COLUMNS = ['tweet_id', 'user_id']  # Update as necessary
            
            # Select columns to bin (only numeric and datetime, excluding identifiers)
            COLUMNS_THAT_CAN_BE_BINNED = Data.select_dtypes(include=['number', 'datetime', 'datetime64[ns, UTC]']).columns.tolist()
            COLUMNS_THAT_CAN_BE_BINNED = [col for col in COLUMNS_THAT_CAN_BE_BINNED if col not in IDENTIFIER_COLUMNS]
            
            selected_columns = st.multiselect('🔢 Select columns to bin', COLUMNS_THAT_CAN_BE_BINNED)
            
            if selected_columns:
                # Create binning sliders
                bins = create_binning_sliders(selected_columns, Data)
                
                # Binning process
                st.markdown("### 🔄 Binning Process")
                try:
                    with st.spinner('Binning data...'):
                        binner = DataBinner(Data, method=binning_method.lower())
                        binned_df, binned_columns = binner.bin_columns(bins)
                        
                        # Align DataFrames to include all columns
                        Data_aligned, binned_df_aligned = align_dataframes(Data, binned_df)
                        
                        # Extract the list of binned columns
                        binned_columns_list = [col for cols in binned_columns.values() for col in cols]
                except Exception as e:
                    st.error(f"Error during binning: {e}")
                    st.stop()
                
                st.success("✅ Binning completed successfully!")
                
                # Display binned columns categorization
                st.markdown("### 🗂️ Binned Columns Categorization")
                for dtype, cols in binned_columns.items():
                    if cols:
                        st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")
                
                # Generate density plots
                plot_density_plots(Data_aligned, binned_df_aligned, selected_columns)
                
                st.markdown("---")
                
                # Debugging: Verify the structure of binned_df_aligned
                st.write("### 📝 Binned DataFrame Structure")
                st.dataframe(binned_df_aligned.head())
                
                # Provide download options for data
                download_data(Data_aligned, binned_df_aligned, binned_columns_list)
            else:
                st.info("🔄 **Please select at least one non-binary column to bin.**")
    else:
        st.info("🔄 **Please upload a file to get started.**")

elif active_tab == "🔢 Generate Embeddings":
    # Proceed only if a file is uploaded
    if uploaded_embedding_file is not None:
        # Identify the file type based on extension
        file_extension = os.path.splitext(uploaded_embedding_file.name)[-1].lower()

        try:
            if file_extension == '.csv':
                # Load CSV file
                binned_data = pd.read_csv(uploaded_embedding_file)
            elif file_extension == '.pkl':
                # Load Pickle file
                binned_data = pd.read_pickle(uploaded_embedding_file)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Pickle file.")
            
            # Display a preview of the uploaded data
            st.subheader('📊 Uploaded Data Preview')
            st.dataframe(binned_data.head())

            # Select the target text column for embedding
            text_columns = binned_data.select_dtypes(include=['object', 'string']).columns.tolist()

            if not text_columns:
                st.error("❌ No suitable text columns found for embedding generation.")
            else:
                target_text_column = st.selectbox("📝 Select Text Column for Embedding", text_columns)

                # Proceed with embedding generation
                # Select categorical columns for feature aggregation
                categorical_columns = binned_data.select_dtypes(include=['category', 'object', 'int']).columns.tolist()
                
                # Remove the target_text_column from categorical_columns if present
                if target_text_column in categorical_columns:
                    categorical_columns.remove(target_text_column)
                
                if categorical_columns:
                    # Encode categorical columns
                    binned_data = encode_categorical_columns(binned_data, categorical_columns)
                    st.success("✅ Categorical columns encoded successfully!")
                else:
                    st.info("🔄 **No categorical columns available for encoding.**")
                
                # Ensure the target_text_column is of string type
                binned_data[target_text_column] = binned_data[target_text_column].astype(str)

                # Add text box for name of the file to be saved
                file_name = st.text_input("📝 Name of the file to be saved", value='Processed_Data')

                # Submit Button to generate embeddings and aggregate features
                if st.button("🚀 Generate Embeddings for Embedding Column"):
                    if not target_text_column:
                        st.error("❌ Please select a valid text column for embedding.")
                    else:
                        with st.spinner('Generating Initial embeddings...'):
                            try:
                                # Generate embeddings and aggregate features using the uploaded data
                                processed_df, final_features = embedding_module.generate_and_aggregate_embeddings(
                                    dataframe=binned_data,
                                    file_name=file_name,
                                    embedding_method=embedding_method,
                                    embedding_dim=embedding_dim,
                                    target_column=target_text_column,
                                    mode=mode,
                                    device=device,
                                    apply_dim_reduction=apply_dim_reduction,
                                    reduced_dim_size=reduced_dim_size,
                                    embedding_column='embedding',
                                    debug=False
                                )
                                # Store the final features in session state
                                st.session_state.embeddings_generated = True
                                st.session_state.final_features = final_features
                                st.session_state.processed_df = processed_df  # Store processed_df for aggregation
                                st.success("✅ Embeddings generated successfully!")

                                # Convert embeddings tensor to list for dataframe
                                st.session_state.processed_df['embedding'] = st.session_state.final_features.cpu().numpy().tolist()

                            except Exception as e:
                                st.error(f"❌ An error occurred during embedding generation: {e}")
                                
                # If embeddings are generated, show the initial embeddings preview
                if st.session_state.get('embeddings_generated', False):
                    st.markdown("#### 📈 Initial Embeddings Preview")
                    st.write(f"Shape of Embeddings: {st.session_state.final_features.shape}")

                    # Display the first few embeddings
                    st.write("First few embeddings:")
                    st.write(st.session_state.final_features[:5].cpu().numpy())

        except Exception as e:
            st.error(f"❌ Error during data processing: {e}")

        # Proceed only if embeddings are generated
        if uploaded_embedding_file is not None and st.session_state.get('embeddings_generated', False):
            if categorical_columns:
                selected_categorical_columns = st.multiselect("🔢 Select Categorical Columns for Feature Aggregation", categorical_columns)
            else:
                selected_categorical_columns = None
                st.info("🔄 **No categorical columns available for feature aggregation.**")
            
            # Feature Aggregation Section
            if selected_categorical_columns:
                # Compute categorical_dims as a dictionary mapping each selected categorical column to its number of unique categories
                categorical_dims = {col: binned_data[col].nunique()+1 for col in selected_categorical_columns}

                # Display the number of unique categories for each selected categorical column
                st.markdown("### 🧮 Categorical Columns Dimensions")
                for col, dim in categorical_dims.items():
                    st.write(f" - **{col}**: {dim} unique categories")

                st.markdown("---")
                
                st.markdown("### 🚀 Feature Aggregation")
                # Button to perform feature aggregation
                if st.button("🚀 Aggregate Features with Categorical Data"):
                    try:
                        with st.spinner('Aggregating features...'):
                            # Call the aggregate_features method from EmbeddingModule
                            aggregated_features = embedding_module.aggregate_features(
                                processed_df=st.session_state.processed_df,  # Processed DataFrame from embedding generation
                                embedding_column='embedding',  # Column name containing embedding vectors
                                categorical_columns=selected_categorical_columns,  # List of selected categorical columns
                                categorical_dims=categorical_dims,  # Dictionary of categorical dimensions
                                sentence_dim=embedding_dim or 768  # Embedding dimension (default to 768 if not set)
                            )
                            
                            # Convert the aggregated torch.Tensor to a NumPy array for easier handling in Streamlit
                            aggregated_features_np = aggregated_features.cpu().numpy()
                            
                            # Store the aggregated features in Streamlit's session state for later use
                            st.session_state.aggregated_features = aggregated_features_np

                            # Add aggregated features to the processed_df as a new column
                            # Convert the aggregated features to lists
                            st.session_state.processed_df['aggregated_features'] = st.session_state.aggregated_features.tolist()

                        st.success("✅ Features aggregated successfully!")

                        # Display a preview of the aggregated features
                        st.subheader("📈 Aggregated Features Preview")
                        st.write(f"Shape of Aggregated Features: {st.session_state.aggregated_features.shape}")
                        st.write("First few aggregated features:")
                        st.write(st.session_state.aggregated_features[:5])

                    except ValueError as ve:
                        st.error(f"❌ Value Error during feature aggregation: {ve}")
                    except IndexError as ie:
                        st.error(f"❌ Index Error during feature aggregation: {ie}")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred during feature aggregation: {e}")
            else:
                selected_categorical_columns = None
                st.info("🔄 **No categorical columns available for feature aggregation.**")

            # Plot Section
            if plot_embeddings:
                st.markdown("---")
                st.markdown("### 📈 Embedding Visualization")
                
                # Add a button to trigger plotting
                if st.button("📈 Plot Embeddings"):
                    # Determine which embedding column to plot
                    if raw_or_aggregated == "Aggregated" and st.session_state.processed_df is not None and 'aggregated_features' in st.session_state.processed_df.columns:
                        embedding_column = 'aggregated_features'
                    else:
                        embedding_column = 'embedding'
                    
                    # Initialize EmbeddingPlotter
                    plotter = EmbeddingPlotter(
                        color_column='label',        # Default value
                        text_column=target_text_column,   # Use the selected text column
                        n=plot_node_count,           # Number of samples to plot
                        name='embedding_visualization',
                        renderer='browser',        # Or 'notebook' if using Jupyter
                        embedding_column=embedding_column
                    )
                    
                    # Check if 'label' column exists, else allow user to select
                    if 'label' not in st.session_state.processed_df.columns:
                        color_column = st.selectbox(
                            "🔴 Select Column for Coloring Embeddings",
                            options=st.session_state.processed_df.columns.tolist(),
                            index=0
                        )
                        plotter.color_column = color_column
                    else:
                        color_column = plotter.color_column  # 'label'
                    
                    # Plot Embeddings
                    with st.spinner('Plotting embeddings...'):
                        try:
                            plotter.plot_updated_embeddings(st.session_state.processed_df)
                        except Exception as e:
                            st.error(f"❌ Error during plotting embeddings: {e}")

    # Comprehensive Download Section for Embedding Generation Tab
    if active_tab == "🔢 Generate Embeddings" and uploaded_embedding_file is not None and st.session_state.get('embeddings_generated', False):
        st.markdown("---")
        st.markdown("### 💾 Download Updated Dataframe")

        # Select download format
        download_format = st.selectbox("🔽 Select Download Format:", ["CSV", "Pickle"], key="download_format")

        # Prepare the dataframe for download
        df_to_download = st.session_state.processed_df.copy()

        # If aggregated features exist, ensure they're in a serializable format
        if 'aggregated_features' in df_to_download.columns:
            df_to_download['aggregated_features'] = df_to_download['aggregated_features'].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x
            )

        # Convert embeddings to string if they are list-like
        if 'embedding' in df_to_download.columns:
            df_to_download['embedding'] = df_to_download['embedding'].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x
            )

        # Add a download button
        if st.button("📥 Download Updated Dataframe"):
            try:
                if download_format == "CSV":
                    csv_data = df_to_download.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Updated Dataframe as CSV",
                        data=csv_data,
                        file_name='updated_data.csv',
                        mime='text/csv',
                    )
                    st.success("✅ Updated dataframe successfully downloaded as CSV!")
                elif download_format == "Pickle":
                    buffer = io.BytesIO()
                    df_to_download.to_pickle(buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download Updated Dataframe as Pickle",
                        data=buffer,
                        file_name='updated_data.pkl',
                        mime='application/octet-stream',
                    )
                    st.success("✅ Updated dataframe successfully downloaded as Pickle!")
            except Exception as e:
                st.error(f"❌ Error during dataframe download: {e}")

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
                        
                        # Display visualization
                        visualizer.visualize()
                    
                except Exception as e:
                    st.error(f"❌ Error during graph visualization: {e}")
    else:
        st.warning("⚠️ **Please generate embeddings first in the '🔢 Generate Embeddings' tab.**")
