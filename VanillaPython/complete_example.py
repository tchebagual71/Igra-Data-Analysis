import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import functions from the IGRA loader module
from igra_loader import (
    load_igra_data_from_file,
    soundings_to_dataframe,
    soundings_to_xarray,
    plot_skewt
)

def create_output_directory(directory="output"):
    """
    Create output directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Name of the output directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_file_paths():
    """
    Get file paths for the data files, using default locations if not found.
    
    Returns:
    --------
    tuple
        (raw_data_path, derived_data_path)
    """
    default_paths = {
        'raw': '/home/tdieckman/Igra-Data-Analysis/VanillaPython/data/USM0007479f-data-exampleportion.txt',
        'derived': '/home/tdieckman/Igra-Data-Analysis/VanillaPython/data/USM0007479f-drvd-exampleportion.txt'
    }
    
    # Check if files exist at default locations
    if not os.path.exists(default_paths['raw']):
        # Try relative paths
        default_paths['raw'] = 'data/USM0007479f-data-exampleportion.txt'
    
    if not os.path.exists(default_paths['derived']):
        # Try relative paths
        default_paths['derived'] = 'data/USM0007479f-drvd-exampleportion.txt'
    
    return default_paths['raw'], default_paths['derived']

def validate_soundings(soundings, data_type):
    """
    Validate loaded soundings and print diagnostic information.
    
    Parameters:
    -----------
    soundings : list
        List of soundings to validate
    data_type : str
        Type of data ('raw' or 'derived')
        
    Returns:
    --------
    bool
        True if soundings are valid, False otherwise
    """
    if not soundings:
        print(f"ERROR: No {data_type} soundings loaded!")
        return False
    
    print(f"Successfully loaded {len(soundings)} {data_type} soundings")
    
    # Check first sounding for basic structure
    if 'header' not in soundings[0]:
        print(f"ERROR: {data_type} soundings missing header information!")
        return False
    
    if 'data' not in soundings[0]:
        print(f"ERROR: {data_type} soundings have no data!")
        return False
    
    # Get date range
    start_date = soundings[0]['header'].get('datetime')
    end_date = soundings[-1]['header'].get('datetime')
    
    if start_date and end_date:
        print(f"Date range: {start_date} to {end_date}")
    
    # Check data points
    total_levels = sum(len(sounding['data']) for sounding in soundings)
    print(f"Total number of data levels: {total_levels}")
    
    return True

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

def main():
    """
    Main function to demonstrate IGRA data analysis.
    """
    print("IGRA Weather Balloon Data Analysis Example")
    print("=========================================")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    
    # Get file paths
    raw_data_path, derived_data_path = get_file_paths()
    print(f"Raw data file: {raw_data_path}")
    print(f"Derived data file: {derived_data_path}")
    
    # Load raw sounding data
    print("\nLoading raw data...")
    try:
        raw_soundings = load_igra_data_from_file(raw_data_path, 'raw')
        
        if not validate_soundings(raw_soundings, 'raw'):
            print("WARNING: Issues with raw data, but attempting to continue")
            
    except Exception as e:
        print(f"ERROR loading raw data: {str(e)}")
        raw_soundings = []
    
    # Load derived parameter data
    print("\nLoading derived data...")
    try:
        derived_soundings = load_igra_data_from_file(derived_data_path, 'derived')
        
        if not validate_soundings(derived_soundings, 'derived'):
            print("WARNING: Issues with derived data, but attempting to continue")
            
    except Exception as e:
        print(f"ERROR loading derived data: {str(e)}")
        derived_soundings = []
    
    # Convert to DataFrames and xarray Datasets
    print("\nConverting data to DataFrames and xarray Datasets...")
    
    raw_df = soundings_to_dataframe(raw_soundings) if raw_soundings else pd.DataFrame()
    derived_df = soundings_to_dataframe(derived_soundings) if derived_soundings else pd.DataFrame()
    
    raw_ds = soundings_to_xarray(raw_soundings) if raw_soundings else None
    derived_ds = soundings_to_xarray(derived_soundings) if derived_soundings else None
    
    # Print basic statistics
    if not raw_df.empty:
        print("\n=== Raw Data Summary ===")
        station_id = raw_df['station_id'].iloc[0] if 'station_id' in raw_df.columns else "Unknown"
        print(f"Station ID: {station_id}")
        
        if 'pressure' in raw_df.columns:
            print(f"Number of pressure levels: {raw_df['pressure'].nunique()}")
        
        if 'datetime' in raw_df.columns:
            print(f"Date range: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")
        
        print(f"Variables: {list(raw_df.columns)}")
        
        # Print first few rows
        print("\n=== Raw Data Preview ===")
        columns_to_show = ['station_id', 'datetime', 'pressure', 'temperature', 'wind_speed']
        columns_to_show = [col for col in columns_to_show if col in raw_df.columns]
        print(raw_df[columns_to_show].head())
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Skew-T plot for a sounding with valid data
    if raw_soundings:
        idx, valid_sounding = find_sounding_with_valid_data(raw_soundings)
        
        if idx >= 0:
            print(f"Creating Skew-T plot for sounding {idx}...")
            try:
                fig, skew = plot_skewt(valid_sounding, f"Sounding {valid_sounding['header']['datetime']}")
                if fig is not None:
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "skewt_plot.png"))
                    plt.close(fig)
                    print(f"Saved Skew-T plot to {os.path.join(output_dir, 'skewt_plot.png')}")
                else:
                    print("Could not create Skew-T plot - insufficient valid data")
            except Exception as e:
                print(f"ERROR creating Skew-T plot: {str(e)}")
        else:
            print("No sounding found with sufficient valid data for a Skew-T plot")
    
    # 2. Temperature profiles
    if not raw_df.empty and 'temperature' in raw_df.columns and 'pressure' in raw_df.columns:
        print("Creating temperature profiles plot...")
        try:
            plt.figure(figsize=(10, 8))
            
            # Get unique dates
            dates = raw_df['datetime'].unique()
            
            # Plot up to 5 profiles
            profiles_plotted = 0
            for date in dates[:10]:  # Try up to 10 dates
                profile = raw_df[(raw_df['datetime'] == date) & raw_df['temperature'].notna()]
                
                if len(profile) >= 5:  # Only plot if at least 5 valid levels
                    plt.plot(profile['temperature'], profile['pressure'], marker='o', label=str(date))
                    profiles_plotted += 1
                
                if profiles_plotted >= 5:
                    break
            
            if profiles_plotted > 0:
                plt.yscale('log')
                plt.gca().invert_yaxis()
                plt.ylim(1050, 100)
                plt.xlabel('Temperature (°C)')
                plt.ylabel('Pressure (hPa)')
                plt.title('Temperature Profiles')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "temperature_profiles.png"))
                plt.close()
                print(f"Saved temperature profiles plot to {os.path.join(output_dir, 'temperature_profiles.png')}")
            else:
                print("Could not create temperature profiles plot - insufficient valid data")
                
        except Exception as e:
            print(f"ERROR creating temperature profiles plot: {str(e)}")
    
    # 3. Wind profile barbs
    if not raw_df.empty and 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        print("Creating wind profile plot...")
        try:
            # Find a sounding with valid wind data
            wind_plot_created = False
            
            for date in raw_df['datetime'].unique()[:10]:  # Try up to 10 dates
                wind_data = raw_df[(raw_df['datetime'] == date) & 
                                 ~raw_df['wind_speed'].isna() & 
                                 ~raw_df['wind_direction'].isna()]
                
                if len(wind_data) >= 5:  # At least 5 levels with wind data
                    # Convert to u, v components
                    wind_data['u'] = -wind_data['wind_speed'] * np.sin(np.radians(wind_data['wind_direction']))
                    wind_data['v'] = -wind_data['wind_speed'] * np.cos(np.radians(wind_data['wind_direction']))
                    
                    plt.figure(figsize=(6, 10))
                    plt.barbs(
                        np.zeros_like(wind_data['pressure']),
                        wind_data['pressure'],
                        wind_data['u'],
                        wind_data['v']
                    )
                    plt.ylim(1050, 100)
                    plt.gca().invert_yaxis()
                    plt.title(f'Wind Profile - {date}')
                    plt.ylabel('Pressure (hPa)')
                    plt.grid(True)
                    plt.savefig(os.path.join(output_dir, "wind_profile.png"))
                    plt.close()
                    print(f"Saved wind profile plot to {os.path.join(output_dir, 'wind_profile.png')}")
                    wind_plot_created = True
                    break
            
            if not wind_plot_created:
                print("Could not create wind profile plot - insufficient valid wind data")
                
        except Exception as e:
            print(f"ERROR creating wind profile plot: {str(e)}")
    
    # 4. Time-height cross section of temperature using xarray
    if raw_ds is not None and 'temperature' in raw_ds:
        print("Creating temperature time-height cross section plot...")
        try:
            # Check if we have sufficient temperature data
            if not raw_ds['temperature'].isnull().all():
                plt.figure(figsize=(12, 8))
                
                # Use xarray's built-in plotting functionality
                raw_ds['temperature'].plot.contourf(
                    x='datetime', 
                    y='pressure', 
                    levels=np.arange(-80, 40, 5),
                    cmap='RdBu_r'
                )
                plt.gca().invert_yaxis()
                plt.title('Temperature Time-Height Cross Section')
                plt.yscale('log')
                plt.savefig(os.path.join(output_dir, "temperature_cross_section.png"))
                plt.close()
                print(f"Saved temperature cross section plot to {os.path.join(output_dir, 'temperature_cross_section.png')}")
            else:
                print("Could not create temperature cross section plot - insufficient valid data")
                
        except Exception as e:
            print(f"ERROR creating temperature cross section plot: {str(e)}")
    
    # 5. Save processed data to CSV
    print("\nSaving processed data to CSV files...")
    
    if not raw_df.empty:
        raw_df.to_csv(os.path.join(output_dir, "igra_raw_data.csv"), index=False)
        print(f"Saved raw data to {os.path.join(output_dir, 'igra_raw_data.csv')}")
    
    if not derived_df.empty:
        derived_df.to_csv(os.path.join(output_dir, "igra_derived_data.csv"), index=False)
        print(f"Saved derived data to {os.path.join(output_dir, 'igra_derived_data.csv')}")
    
    # 6. Try to save to NetCDF
    print("\nAttempting to save data to NetCDF files...")
    
    try:
        if raw_ds is not None:
            raw_ds.to_netcdf(os.path.join(output_dir, "igra_raw_data.nc"))
            print(f"Saved raw data to {os.path.join(output_dir, 'igra_raw_data.nc')}")
        
        if derived_ds is not None:
            derived_ds.to_netcdf(os.path.join(output_dir, "igra_derived_data.nc"))
            print(f"Saved derived data to {os.path.join(output_dir, 'igra_derived_data.nc')}")
    except Exception as e:
        print(f"WARNING: Could not save to NetCDF format: {str(e)}")
        print("This is not critical - CSV files were still created.")
    
    print("\nAnalysis complete!")
    
    return {
        'raw_df': raw_df,
        'derived_df': derived_df,
        'raw_ds': raw_ds,
        'derived_ds': derived_ds
    }

if __name__ == "__main__":
    main()