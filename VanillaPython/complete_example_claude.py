import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import tempfile
import io

# Import the IGRA data loading functions
# Assuming the previous code is saved as igra_loader.py
from igra_loader import (
    load_igra_data_from_text, 
    soundings_to_dataframe, 
    soundings_to_xarray, 
    plot_skewt
)

def main():
    """
    Main function to demonstrate loading and analyzing IGRA data.
    """
    # Use the sample data provided
    # Replace these with paths to your actual data files when using this code
    # raw_data_path = "USM00074794-data.txt"
    # derived_data_path = "USM00074794-drvd.txt"
    
    # For demonstration, we'll use the sample data as strings
    with open("/home/tdieckman/Igra-Data-Analysis/USM0007479f-data-exampleportion.txt", "r") as f:
        raw_data_text = f.read()
    
    with open("/home/tdieckman/Igra-Data-Analysis/USM0007479f-drvd-exampleportion.txt", "r") as f:
        derived_data_text = f.read()

    print("Loading IGRA data from sample texts...")
    
    # Load the raw sounding data
    raw_soundings = load_igra_data_from_text(raw_data_text, 'raw')
    raw_df = soundings_to_dataframe(raw_soundings)
    raw_ds = soundings_to_xarray(raw_soundings)
    
    # Load the derived parameters data
    derived_soundings = load_igra_data_from_text(derived_data_text, 'derived')
    derived_df = soundings_to_dataframe(derived_soundings)
    derived_ds = soundings_to_xarray(derived_soundings)
    
    print(f"Loaded {len(raw_soundings)} raw soundings and {len(derived_soundings)} derived soundings")
    
    # Basic data inspection
    print("\n=== Raw Data Summary ===")
    print(f"Station ID: {raw_df['station_id'].iloc[0]}")
    print(f"Number of pressure levels: {raw_df['pressure'].nunique()}")
    print(f"Date range: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")
    print(f"Variables: {list(raw_df.columns)}")
    
    # Print first few rows to see structure
    print("\n=== Raw Data Preview ===")
    print(raw_df[['station_id', 'datetime', 'pressure', 'temperature', 'wind_speed']].head())
    
    # Create some basic visualizations
    
    # 1. Skew-T plot for the first sounding
    # Modified to handle the case where plot_skewt returns None
    first_sounding = raw_soundings[0]
    skewt_result = plot_skewt(first_sounding, "Sample Sounding")
    
    if skewt_result is not None:
        fig, skew = skewt_result
        plt.tight_layout()
        plt.savefig("skewt_first_sounding.png")
        plt.close()
        print("Created Skew-T plot for the first sounding")
    else:
        print("Could not create Skew-T plot for the first sounding due to insufficient data")
        
        # Try to find a sounding with sufficient data
        print("Trying to find a suitable sounding for Skew-T plot...")
        for i, sounding in enumerate(raw_soundings):
            skewt_result = plot_skewt(sounding, f"Sounding {i+1}")
            if skewt_result is not None:
                fig, skew = skewt_result
                plt.tight_layout()
                plt.savefig("skewt_alternate_sounding.png")
                plt.close()
                print(f"Created Skew-T plot for sounding {i+1}")
                break
    
    # 2. Temperature profile comparison
    plt.figure(figsize=(10, 8))
    
    # Get unique dates and filter out rows with missing temperature data
    valid_dates = []
    for date in raw_df['datetime'].unique():
        profile = raw_df[(raw_df['datetime'] == date) & raw_df['temperature'].notna()]
        if len(profile) >= 5:  # Only consider profiles with at least 5 valid temperature readings
            valid_dates.append(date)
            plt.plot(profile['temperature'], profile['pressure'], marker='o', label=str(date))
    
    if valid_dates:
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.ylim(1050, 100)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Pressure (hPa)')
        plt.title('Temperature Profiles')
        plt.legend()
        plt.grid(True)
        plt.savefig("temperature_profiles.png")
        plt.close()
        print("Created temperature profile comparison plot")
    else:
        print("Could not create temperature profile plot due to insufficient data")
    
    # 3. Wind profile barbs for the first sounding with valid wind data
    if 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        # Find the first date with valid wind data
        wind_found = False
        for date in raw_df['datetime'].unique():
            wind_data = raw_df[(raw_df['datetime'] == date) & 
                              ~raw_df['wind_speed'].isna() & 
                              ~raw_df['wind_direction'].isna()]
            
            if len(wind_data) >= 3:  # At least 3 levels with wind data
                wind_found = True
                # Convert to u and v components
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
                plt.savefig("wind_profile.png")
                plt.close()
                print(f"Created wind profile barbs plot for {date}")
                break
        
        if not wind_found:
            print("Could not create wind profile plot due to insufficient wind data")
    
    # 4. Create xarray visualizations
    if raw_ds is not None:
        # Time-height cross section of temperature
        try:
            # Check if we have sufficient temperature data
            temp_data = raw_ds['temperature']
            if not temp_data.isnull().all():
                plt.figure(figsize=(12, 8))
                # Use xarray's built-in plotting functionality
                temp_data.plot.contourf(
                    x='datetime', 
                    y='pressure', 
                    levels=np.arange(-80, 40, 5),
                    cmap='RdBu_r'
                )
                plt.gca().invert_yaxis()
                plt.title('Temperature Time-Height Cross Section')
                plt.yscale('log')
                plt.savefig("temp_xarray_plot.png")
                plt.close()
                print("Created temperature time-height cross section plot")
            else:
                print("Could not create xarray temperature plot due to insufficient data")
        except Exception as e:
            print(f"Error creating xarray plot: {e}")
    
    # 5. Analyze atmospheric stability if derived parameters are available
    stability_plot_created = False
    if derived_df is not None and not derived_df.empty:
        # Extract stability parameters
        stability_cols = [col for col in ['cape', 'cin', 'lifted_index'] if col in derived_df.columns]
        
        if stability_cols:
            # Get one value per sounding
            stability_df = derived_df.drop_duplicates(subset=['datetime'])[['datetime'] + stability_cols]
            
            # Filter out rows with all NaN stability values
            stability_df = stability_df.dropna(subset=stability_cols, how='all')
            
            if not stability_df.empty:
                # Print summary
                print("\n=== Stability Parameters ===")
                for col in stability_cols:
                    if not stability_df[col].isna().all():
                        print(f"{col}: Mean = {stability_df[col].mean():.2f}, Min = {stability_df[col].min():.2f}, Max = {stability_df[col].max():.2f}")
                
                # Plot time series
                if len(stability_df) > 1:
                    plt.figure(figsize=(12, 6))
                    for col in stability_cols:
                        if not stability_df[col].isna().all():
                            plt.plot(stability_df['datetime'], stability_df[col], marker='o', label=col)
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.title('Stability Indices')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("stability_indices.png")
                    plt.close()
                    print("Created stability indices time series plot")
                    stability_plot_created = True
    
    if not stability_plot_created:
        print("Could not create stability indices plot due to insufficient data")
    
    # 6. Save to common formats for further analysis
    
    # Save to CSV
    raw_df.to_csv("igra_raw_data.csv", index=False)
    if derived_df is not None and not derived_df.empty:
        derived_df.to_csv("igra_derived_data.csv", index=False)
    print("Saved data to CSV files")
    
    # Save to NetCDF using xarray
    try:
        if raw_ds is not None:
            raw_ds.to_netcdf("igra_raw_data.nc")
        if derived_ds is not None:
            derived_ds.to_netcdf("igra_derived_data.nc")
        print("Saved data to NetCDF files")
    except Exception as e:
        print(f"Error saving to NetCDF: {e}")
        print("Saving to NetCDF failed, but CSV files were created successfully")
    
    print("\nCompleted data analysis and saved results.")
    print("Created the following files:")
    print("  - igra_raw_data.csv - CSV file with raw data")
    print("  - igra_derived_data.csv - CSV file with derived data")
    
    # List only files that were actually created
    file_status = {
        "skewt_first_sounding.png": os.path.exists("skewt_first_sounding.png"),
        "skewt_alternate_sounding.png": os.path.exists("skewt_alternate_sounding.png"),
        "temperature_profiles.png": os.path.exists("temperature_profiles.png"),
        "wind_profile.png": os.path.exists("wind_profile.png"),
        "temp_xarray_plot.png": os.path.exists("temp_xarray_plot.png"),
        "stability_indices.png": os.path.exists("stability_indices.png"),
        "igra_raw_data.nc": os.path.exists("igra_raw_data.nc"),
        "igra_derived_data.nc": os.path.exists("igra_derived_data.nc")
    }
    
    for file, exists in file_status.items():
        if exists:
            print(f"  - {file}")
    
    return {
        'raw_df': raw_df,
        'raw_ds': raw_ds,
        'derived_df': derived_df,
        'derived_ds': derived_ds
    }

if __name__ == "__main__":
    main()