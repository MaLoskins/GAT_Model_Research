import os
import io
import pandas as pd
import streamlit as st
import numpy as np
from Binning.data_binner import DataBinner
from Binning.density_plotter import DensityPlotter
from Binning.Process_Data import DataProcessor
from Text_Processing.embedding_module import generate_and_aggregate_embeddings
import matplotlib.pyplot as plt
import tempfile

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

# Create directory for processed data
os.makedirs('Processed_Data', exist_ok=True)

# ============================
# Utility Functions
# ============================

def run_processing(save_type='csv', output_filepath='Processed_Data.csv', file_path='Data.csv'):
    """
    Initializes and runs the data processing pipeline.
    """
    try:
        processor = DataProcessor(
            input_filepath=file_path,
            output_filepath=output_filepath,
            report_path='CSV_Files/Type_Conversion_Report.csv',
            return_category_mappings=False,
            mapping_directory='CSV_Files/Category_Mappings',
            parallel_processing=False,
            date_threshold=0.6,
            numeric_threshold=0.9,
            factor_threshold_ratio=0.2,
            factor_threshold_unique=500,
            dayfirst=True,
            log_level='INFO',
            log_file=None,
            convert_factors_to_int=True,
            date_format=None,  # Keep as None to retain datetime dtype
            save_type=save_type
        )
        processor.process()
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_data(file_type, uploaded_file):
    """
    Handles file upload, processing, and loading of the processed data.
    """
    if uploaded_file is None:
        return None, "No file uploaded!"

    try:
        # Define temporary file extension based on selected file type
        file_extension = file_type if file_type in ["pkl", "csv"] else "csv"

        # Use NamedTemporaryFile to handle file operations safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        # Define output paths based on file type
        output_map = {
            "pkl": ("pickle", "Pickle_Files/Processed_Data.pkl"),
            "csv": ("csv", "CSV_Files/Processed_Data.csv")
        }
        save_type, output_filepath = output_map.get(file_type, ("csv", "Processed_Data.csv"))

        run_processing(save_type=save_type, output_filepath=output_filepath, file_path=temp_file_path)

        # Load processed data
        if save_type == "pickle":
            Data = pd.read_pickle(output_filepath)
        else:
            # Read the type conversion report to get date columns
            report_df = pd.read_csv('CSV_Files/Type_Conversion_Report.csv')
            date_columns = report_df[report_df['Type'] == 'date']['Column'].tolist()
            Data = pd.read_csv(output_filepath, parse_dates=date_columns)

        # Clean up temporary file
        os.remove(temp_file_path)

        return Data, None
    except Exception as e:
        return None, f"Error loading data: {e}"

def align_dataframes(original_df, binned_df):
    """
    Ensures that the binned DataFrame includes all original columns along with the new binned columns.
    """
    try:
        # Identify any missing columns in binned_df that are present in original_df
        missing_cols = original_df.columns.difference(binned_df.columns)
        for col in missing_cols:
            binned_df[col] = original_df[col]
        
        # Return the original DataFrame and the binned DataFrame without filtering out new columns
        return original_df, binned_df
    except Exception as e:
        st.error(f"Error aligning dataframes: {e}")
        st.stop()

def create_binning_sliders(selected_columns, Data):
    """
    Creates sliders for binning configuration based on column types.
    """
    bins = {}
    st.markdown("### 📏 Binning Configuration")

    cols_per_row = 2
    num_cols = len(selected_columns)
    rows = (num_cols + cols_per_row - 1) // cols_per_row  # Ceiling division

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            current_col = row * cols_per_row + col_idx
            if current_col >= num_cols:
                break
            column = selected_columns[current_col]
            max_bins = Data[column].nunique()
            if pd.api.types.is_datetime64_any_dtype(Data[column]):
                default_bins = min(6, max_bins) if max_bins >= 2 else 1
                bins[column] = st.slider(
                    f'📏 {column} (Datetime)', 
                    min_value=1, 
                    max_value=max_bins, 
                    value=default_bins,
                    key=column
                )
            elif pd.api.types.is_integer_dtype(Data[column]):
                if max_bins > 2:
                    bins[column] = st.slider(
                        f'📏 {column} (Integer)', 
                        min_value=2, 
                        max_value=max_bins, 
                        value=min(10, max_bins),
                        key=column
                    )
                else:
                    st.write(f'📏 **{column} (Integer):** {max_bins} (Fixed)')
                    bins[column] = max_bins
            elif pd.api.types.is_float_dtype(Data[column]):
                if max_bins > 2:
                    bins[column] = st.slider(
                        f'📏 {column} (Float)', 
                        min_value=2, 
                        max_value=max_bins, 
                        value=min(10, max_bins),
                        key=column
                    )
                else:
                    st.write(f'📏 **{column} (Float):** {max_bins} (Fixed)')
                    bins[column] = max_bins
            else:
                # For any other type, provide a generic slider
                if max_bins > 1:
                    bins[column] = st.slider(
                        f'📏 {column}', 
                        min_value=1, 
                        max_value=max_bins, 
                        value=min(10, max_bins),
                        key=column
                    )
                else:
                    st.write(f'📏 **{column}:** {max_bins} (Fixed)')
                    bins[column] = max_bins
    return bins

def plot_density_plots(Data_aligned, binned_df_aligned, selected_columns):
    """
    Generates density plots for original and binned data.
    Excludes datetime columns from original data density plots.
    """
    st.markdown("### 📈 Density Plots")
    
    if len(selected_columns) > 1:
        # Identify columns suitable for original data density plots
        columns_orig = [
            col for col in selected_columns 
            if pd.api.types.is_categorical_dtype(Data_aligned[col]) or 
               pd.api.types.is_integer_dtype(Data_aligned[col])
        ]
        
        # Identify and notify about excluded columns
        excluded_cols = set(selected_columns) - set(columns_orig)
        if excluded_cols:
            st.warning(
                f"The following columns were excluded from original data density plots because they are neither categorical nor integer dtype: {', '.join(excluded_cols)}"
            )
        
        if columns_orig:
            # Create tabs for Original and Binned Data
            density_tab1, density_tab2 = st.tabs(["Original Data", "Binned Data"])
            
            with density_tab1:
                try:
                    plotter_orig = DensityPlotter(
                        dataframe=Data_aligned,
                        category_columns=columns_orig,
                        figsize=(15, 4),                     
                        save_path=None,
                        plot_style='ticks'
                    )
                    plotter_orig.plot_grid()
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting original data density: {e}")
            
            with density_tab2:
                try:
                    plotter_binned = DensityPlotter(
                        dataframe=binned_df_aligned,
                        category_columns=selected_columns,
                        figsize=(15, 4),                     
                        save_path=None,
                        plot_style='ticks'
                    )
                    plotter_binned.plot_grid()
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error plotting binned data density: {e}")
        else:
            st.info("🔄 **No suitable columns to plot for original data density.**")
    else:
        st.info("🔄 **Please select more than one column to display density plots.**")

def download_data(Data_aligned, binned_df_aligned, binned_columns_list):
    """
    Provides download buttons for either the entire updated DataFrame or only the binned features in CSV or Pickle format.

    Parameters:
        Data_aligned (pd.DataFrame): The entire updated DataFrame after binning.
        binned_df_aligned (pd.DataFrame): The binned DataFrame aligned with the original DataFrame.
        binned_columns_list (List[str]): List of columns that have been binned.
    """
    st.markdown("### 💾 Download Data")
    
    # Selection for Data to Download
    download_option = st.radio(
        "🔽 Select Data to Download:",
        ("Entire DataFrame", "Only Binned Features")
    )
    
    # Selection for File Type
    file_type_download = st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0)
    
    # Determine which DataFrame to download based on user selection
    if download_option == "Entire DataFrame":
        data_to_download = binned_df_aligned  # Updated to include binned columns
        file_name = 'updated_data'
    else:
        # Extract only the binned columns
        data_to_download = binned_df_aligned[binned_columns_list]
        file_name = 'binned_data'
    
    try:
        if file_type_download == 'csv':
            csv_data = data_to_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {file_name.replace('_', ' ').title()} as CSV",
                data=csv_data,
                file_name=f'{file_name}.csv',
                mime='text/csv',
            )
            st.success(f"✅ {file_name.replace('_', ' ').title()} successfully downloaded as CSV!")
        elif file_type_download == 'pkl':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_pkl:
                data_to_download.to_pickle(tmp_pkl.name)
                tmp_pkl.seek(0)
                pkl_data = tmp_pkl.read()
            st.download_button(
                label=f"📥 Download {file_name.replace('_', ' ').title()} as Pickle",
                data=pkl_data,
                file_name=f'{file_name}.pkl',
                mime='application/octet-stream',
            )
            os.remove(tmp_pkl.name)
            st.success(f"✅ {file_name.replace('_', ' ').title()} successfully downloaded as Pickle!")
    except Exception as e:
        st.error(f"Error during data download: {e}")

# ============================
# Main Application
# ============================

st.title('🛠️ Data Processing and Binning Application')

# Sidebar for inputs and options
with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv'])
    file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
    st.markdown("---")

    if file_type == 'csv':
        st.warning("⚠️ **Note:** Using CSV may result in loss of data types and categories. This will affect subsequent processes. Incompatible columns will be removed from binning as a result. Consider using Pickle for better preservation.")
    
    st.header("⚙️ Binning Options")
    binning_method = st.selectbox('🔧 Select Binning Method', ['Quantile', 'Equal Width'])
    if binning_method == 'Quantile':
        st.warning("⚠️ **Note:** Using Quantile binning will prevent the output of 'Original Data' Density Plots due to granularity.")
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.info("""
        This application allows you to upload a dataset, process and bin numerical and datetime columns, 
        assess data integrity post-binning, and visualize data distributions.
    """)

# Introduce Tabs in the Main Content Area # NEW:
tabs = st.tabs(["📊 Binning", "🔢 Generate Embeddings"])  # You can rename "Additional Features" as desired

# ============================
# Binning Tab Content
# ============================

with tabs[0]:  # 📊 Binning Tab
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
                        
                        # Ensure binned columns have unique names if necessary
                        # For example, append '_binned' to each binned column
                        # Uncomment and modify the following lines if needed:
                        # binned_df = binned_df.rename(columns=lambda x: f"{x}_binned" if x in selected_columns else x)
                        
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


# ============================
# Embedding Generation Tab Content
# ============================

with tabs[1]:  # 🔠 Embedding Generation Tab
    st.subheader("🔠 Embedding Generation")
    st.write("Upload a dataset to generate embeddings and aggregate features.")

    # Upload the processed (binned) file from the binning step (CSV or Pickle)
    uploaded_embedding_file = st.file_uploader("📤 Upload Binned Data (CSV or Pickle)", type=['csv', 'pkl'])

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

                # Embedding Options
                st.markdown("### ⚙️ Embedding Options")
                embedding_method = st.selectbox("🔧 Select Embedding Method", ["glove", "word2vec", "bert"], index=2)

                if embedding_method == "bert":
                    embedding_dim = None  # Default BERT dimension
                else:
                    embedding_dim = st.number_input("📏 Embedding Dimension", min_value=50, max_value=1024, value=100, step=50)

                mode = st.selectbox("🔀 Embedding Mode", ["sentence", "word"], index=0)
                device = st.selectbox("⚙️ Device for Computation", ["cpu","cuda"], index=1)
                apply_dim_reduction = st.checkbox("🧠 Apply Dimensionality Reduction (PCA)", value=False)
                reduced_dim_size = 100
                if apply_dim_reduction:
                    reduced_dim_size = st.number_input("🔢 Reduced Dimension Size", min_value=50, max_value=500, value=100, step=50)

                plot_embeddings = st.checkbox("📊 Plot Embeddings", value=False)
                if plot_embeddings:
                    plot_node_count = st.number_input("📈 Number of Nodes to Plot", min_value=100, max_value=100000, value=10000, step=1000)
                else:
                    plot_node_count = 10000  # Default value

                # Select categorical columns for feature aggregation
                st.markdown("### 🗂️ Feature Aggregations")
                categorical_columns = binned_data.select_dtypes(include=['category', 'object', 'int']).columns.tolist()
                if categorical_columns:
                    selected_categorical_columns = st.multiselect("🔢 Select Categorical Columns for Feature Aggregation", categorical_columns)
                else:
                    selected_categorical_columns = None
                    st.info("🔄 **No categorical columns available for feature aggregation.**")
                # Add text box for name of the file to be saved
                file_name = st.text_input("📝 Name of the file to be saved", value='Processed_Data')

                # Submit Button to generate embeddings and aggregate features
                if st.button("🚀 Generate Embeddings and Aggregate Features"):
                    if not target_text_column:
                        st.error("❌ Please select a valid text column for embedding.")
                    else:
                        with st.spinner('Generating embeddings and aggregating features...'):
                            try:
                                # Generate embeddings and aggregate features using the uploaded data
                                processed_df, final_features = generate_and_aggregate_embeddings(
                                    dataframe=binned_data,
                                    file_name = file_name,
                                    embedding_method=embedding_method,
                                    embedding_dim=embedding_dim,
                                    target_column=target_text_column,
                                    mode=mode,
                                    device=device,
                                    apply_dim_reduction=apply_dim_reduction,
                                    reduced_dim_size=reduced_dim_size,
                                    plot_embeddings=plot_embeddings,
                                    plot_node_count=plot_node_count,
                                    categorical_columns=selected_categorical_columns,
                                    embedding_column='embedding',
                                    debug=False
                                )
                                # Store the final features in session state
                                st.session_state.embeddings_generated = True
                                st.session_state.final_features = final_features
                                st.success("✅ Embeddings generated and features aggregated successfully!")

                            except Exception as e:
                                st.error(f"❌ An error occurred during embedding generation: {e}")
                                
                # If embeddings are generated, show the download options
                if st.session_state.get('embeddings_generated', False):
                    st.subheader("📈 Aggregated Features Preview")
                    st.write(f"Shape of Aggregated Features: {st.session_state.final_features.shape}")

                    st.markdown("### 💾 Download Aggregated Features")
                    download_feature_option = st.radio(
                        "🔽 Select Feature to Download:",
                        ("Final Aggregated Features (NumPy Array)", "Final Aggregated Features (CSV)")
                    )

                    if download_feature_option == "Final Aggregated Features (NumPy Array)":
                        buffer = io.BytesIO()
                        np.save(buffer, st.session_state.final_features)
                        buffer.seek(0)
                        st.download_button(
                            label="📥 Download Aggregated Features as Numpy Array (.npy)",
                            data=buffer,
                            file_name='final_features.npy',
                            mime='application/octet-stream'
                        )
                        st.info("🔄 **Note:** To load the Numpy array, use `numpy.load('final_features.npy')`.")

                    elif download_feature_option == "Final Aggregated Features (CSV)":
                        aggregated_df = pd.DataFrame(st.session_state.final_features)
                        csv_aggregated = aggregated_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Aggregated Features as CSV",
                            data=csv_aggregated,
                            file_name='final_features.csv',
                            mime='text/csv'
                        )

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    else:
        st.info("🔄 **Please upload a processed file (CSV or Pickle) from the binning step.**")



