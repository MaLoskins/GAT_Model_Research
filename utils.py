# utils.py

import os
import io
import pandas as pd
import streamlit as st
import numpy as np
import tempfile
from Binning.data_binner import DataBinner
from Binning.density_plotter import DensityPlotter
from Binning.Process_Data import DataProcessor
import matplotlib.pyplot as plt

def encode_categorical_columns(df, categorical_columns):
    """
    Encodes categorical columns in the DataFrame to integer codes starting from 0.

    Args:
        df (pd.DataFrame): The DataFrame containing categorical columns.
        categorical_columns (list of str): List of categorical column names to encode.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def run_processing(save_type='csv', output_filepath='CSV_Files/Processed_Data.csv', file_path='Data.csv'):
    """
    Initializes and runs the data processing pipeline.

    Args:
        save_type (str): Type to save the processed data ('csv' or 'pickle').
        output_filepath (str): Path to save the processed data.
        file_path (str): Path of the input data file.
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

    Args:
        file_type (str): Desired output file type ('csv' or 'pkl').
        uploaded_file (UploadedFile): Uploaded file object from Streamlit.

    Returns:
        tuple: (Processed DataFrame, Error message or None)
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
            "pkl": ("pickle", "CSV_Files/Processed_Data.pkl"),
            "csv": ("csv", "CSV_Files/Processed_Data.csv")
        }
        save_type, output_filepath = output_map.get(file_type, ("csv", "CSV_Files/Processed_Data.csv"))

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

    Args:
        original_df (pd.DataFrame): The original DataFrame before binning.
        binned_df (pd.DataFrame): The binned DataFrame.

    Returns:
        tuple: (Original DataFrame aligned, Binned DataFrame aligned)
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

    Args:
        selected_columns (list): List of column names selected for binning.
        Data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        dict: Dictionary mapping column names to the number of bins.
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

    Args:
        Data_aligned (pd.DataFrame): Original DataFrame aligned with binned DataFrame.
        binned_df_aligned (pd.DataFrame): Binned DataFrame aligned with original DataFrame.
        selected_columns (list): List of columns that were binned.
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

    Args:
        Data_aligned (pd.DataFrame): The entire updated DataFrame after binning.
        binned_df_aligned (pd.DataFrame): The binned DataFrame aligned with the original DataFrame.
        binned_columns_list (list): List of columns that have been binned.
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
