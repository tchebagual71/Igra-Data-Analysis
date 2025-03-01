import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback
import inspect

def inspect_sounding(sounding, sounding_index=0):
    """
    Diagnostic function to examine a sounding's structure and data.
    
    Parameters:
    -----------
    sounding : dict
        A single sounding dictionary
    sounding_index : int
        Index of the sounding (for display purposes)
    """
    print(f"\n--- Inspecting Sounding #{sounding_index} ---")
    
    # Check if the sounding has a header
    if 'header' not in sounding:
        print("ERROR: Sounding is missing header information")
        return
    
    # Check header information
    header = sounding['header']
    print("Header information:")
    for key, value in header.items():
        print(f"  {key}: {value}")
    
    # Check if the sounding has data
    if 'data' not in sounding:
        print("ERROR: Sounding has no data records")
        return
    
    # Check data
    data = sounding['data']
    print(f"Number of data levels: {len(data)}")
    
    # Analyze data structure and content
    if len(data) > 0:
        # Get all keys from the first record
        data_keys = list(data[0].keys())
        print(f"Data variables: {', '.join(data_keys)}")
        
        # Check for missing values
        missing_counts = {}
        for key in data_keys:
            missing_values = sum(1 for record in data if key not in record or record[key] is None or (isinstance(record[key], float) and np.isnan(record[key])))
            missing_counts[key] = missing_values
        
        print("\nMissing value counts:")
        for key, count in missing_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {key}: {count}/{len(data)} ({percentage:.1f}%)")
        
        # Example of values for key fields (first 3 levels)
        print("\nSample values (first 3 levels):")
        for i, record in enumerate(data[:3]):
            print(f"  Level {i+1}:")
            for key in data_keys:
                if key in record:
                    print(f"    {key}: {record[key]}")
                else:
                    print(f"    {key}: MISSING")
    else:
        print("WARNING: Sounding has empty data array")

def diagnose_plot_skewt_function(sounding, plot_skewt_func):
    """
    Diagnose issues with the plot_skewt function for a given sounding.
    
    Parameters:
    -----------
    sounding : dict
        A single sounding dictionary
    plot_skewt_func : function
        The plot_skewt function to test
    """
    print("\n--- Diagnosing plot_skewt Function ---")
    
    # Check if sounding has necessary data for a skewt plot
    if 'data' not in sounding or len(sounding['data']) == 0:
        print("ERROR: Sounding has no data for plot_skewt")
        return
    
    # Check for required variables in the sounding data
    required_vars = ['pressure', 'temperature']
    
    # Create a pandas DataFrame for easier analysis
    try:
        # Extract data to lists
        data_dict = {key: [] for key in sounding['data'][0].keys()}
        
        for record in sounding['data']:
            for key in data_dict.keys():
                if key in record:
                    data_dict[key].append(record[key])
                else:
                    data_dict[key].append(np.nan)
        
        df = pd.DataFrame(data_dict)
        
        # Check required variables
        print("Required variables for plot_skewt:")
        for var in required_vars:
            if var in df.columns:
                non_nan_count = df[var].notna().sum()
                print(f"  {var}: {non_nan_count}/{len(df)} valid values ({(non_nan_count/len(df))*100:.1f}%)")
            else:
                print(f"  {var}: MISSING from data")
        
        # Test the function with try/except to catch errors
        print("\nAttempting to call plot_skewt...")
        try:
            result = plot_skewt_func(sounding, "Test Plot")
            if result is None:
                print("plot_skewt returned None - likely insufficient valid data")
            else:
                print("plot_skewt executed successfully and returned a figure/axis tuple")
                # Close the figure to avoid displaying it
                plt.close(result[0])
        except Exception as e:
            print(f"ERROR in plot_skewt: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR during diagnosis: {str(e)}")

def inspect_dataframe(df, name="DataFrame"):
    """
    Inspect a pandas DataFrame for data quality and structure issues.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to inspect
    name : str
        Name of the DataFrame for display purposes
    """
    print(f"\n--- Inspecting {name} ---")
    
    if df is None:
        print(f"ERROR: {name} is None")
        return
    
    if df.empty:
        print(f"WARNING: {name} is empty (has 0 rows)")
        return
    
    # Basic info
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check for missing values
    missing_counts = df.isna().sum()
    print("\nMissing values per column:")
    for col, count in missing_counts.items():
        percentage = (count / len(df)) * 100
        if percentage > 0:
            print(f"  {col}: {count}/{len(df)} ({percentage:.1f}%)")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nWARNING: {duplicate_count} duplicate rows found ({(duplicate_count/len(df))*100:.1f}%)")
    
    # Check data types
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Show sample values for key columns
    key_columns = ['datetime', 'pressure', 'temperature'] if 'temperature' in df.columns else df.columns[:3]
    print("\nSample values for key columns:")
    for col in key_columns:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 0:
                sample = unique_values[:5] if len(unique_values) > 5 else unique_values
                print(f"  {col}: {sample}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"    Min: {df[col].min()}, Max: {df[col].max()}")

def fix_missing_plot_skewt_function():
    """
    Create a basic plot_skewt function if the original one is missing or problematic.
    Returns a working plot_skewt function.
    """
    def plot_skewt(sounding, title=None):
        """
        Create a basic Skew-T log-P diagram for a single sounding.
        
        Parameters:
        -----------
        sounding : dict
            Dictionary containing a single sounding's data and header
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        tuple or None
            (fig, ax) if successful, None if not enough data
        """
        if 'data' not in sounding or len(sounding['data']) == 0:
            print("No data in sounding")
            return None
            
        # Extract pressure and temperature data
        pressure_values = []
        temp_values = []
        
        for level in sounding['data']:
            if 'pressure' in level and 'temperature' in level:
                p = level['pressure']
                t = level['temperature']
                
                # Skip levels with missing data
                if (p is not None and not np.isnan(p) and 
                    t is not None and not np.isnan(t)):
                    pressure_values.append(p)
                    temp_values.append(t)
        
        # Check if we have enough data to plot
        if len(pressure_values) < 3:
            print("Not enough valid data for Skew-T plot")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Plot temperature profile
        ax.plot(temp_values, pressure_values, 'r-o', linewidth=2, label='Temperature')
        
        # Setup the plot
        ax.set_yscale('log')
        ax.invert_yaxis()  # Pressure decreases with height
        ax.set_ylim(1050, 100)  # Standard tropospheric range
        
        # Add labels and title
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Pressure (hPa)')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Basic Skew-T Diagram')
            
        ax.grid(True)
        ax.legend(loc='upper right')
        
        return fig, ax
        
    return plot_skewt

def find_sounding_with_valid_data(soundings, min_valid_levels=5):
    """
    Find the first sounding with sufficient valid temperature and pressure data.
    
    Parameters:
    -----------
    soundings : list
        List of soundings to search
    min_valid_levels : int
        Minimum number of valid levels required
        
    Returns:
    --------
    tuple
        (index, sounding) or (-1, None) if no valid sounding found
    """
    for i, sounding in enumerate(soundings):
        if 'data' not in sounding or len(sounding['data']) == 0:
            continue
        
        valid_levels = 0
        for level in sounding['data']:
            if ('pressure' in level and level['pressure'] is not None and not np.isnan(level['pressure']) and
                'temperature' in level and level['temperature'] is not None and not np.isnan(level['temperature'])):
                valid_levels += 1
                
        if valid_levels >= min_valid_levels:
            return i, sounding
    
    return -1, None

def analyze_file(file_path, data_type='raw'):
    """
    Analyze an IGRA data file and print diagnostics.
    
    Parameters:
    -----------
    file_path : str
        Path to the IGRA data file
    data_type : str
        Type of data: 'raw' or 'derived'
    """
    print(f"\n=== Analyzing {file_path} (type: {data_type}) ===")
    
    try:
        # Import required functions
        from igra_loader import load_igra_data_from_file, soundings_to_dataframe, plot_skewt
        
        # Load the data
        soundings = load_igra_data_from_file(file_path, data_type)
        print(f"Successfully loaded {len(soundings)} soundings")
        
        # Convert to dataframe
        df = soundings_to_dataframe(soundings)
        
        # Inspect the dataframe
        inspect_dataframe(df, f"{data_type.capitalize()} Data")
        
        # Find a sounding with sufficient data
        idx, valid_sounding = find_sounding_with_valid_data(soundings)
        
        if idx >= 0:
            print(f"\nFound sounding with valid data (index {idx})")
            inspect_sounding(valid_sounding, idx)
            
            # Test plot_skewt
            diagnose_plot_skewt_function(valid_sounding, plot_skewt)
        else:
            print("\nNo sounding found with sufficient valid data for plotting")
            
            # Inspect the first sounding anyway for diagnostic purposes
            if len(soundings) > 0:
                inspect_sounding(soundings[0], 0)
                
    except ImportError as e:
        print(f"ERROR importing required modules: {str(e)}")
        print("Make sure igra_loader.py is in the current directory or Python path")
        
    except Exception as e:
        print(f"ERROR during analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        data_type = sys.argv[2] if len(sys.argv) > 2 else 'raw'
        
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
            
        analyze_file(file_path, data_type)
        
    else:
        print("Usage: python igra_debug.py <path_to_igra_data_file> [data_type]")
        print("  data_type: 'raw' (default) or 'derived'")