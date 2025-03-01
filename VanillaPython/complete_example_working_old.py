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
    first_sounding = raw_soundings[0]
    fig, skew = plot_skewt(first_sounding, "Sample Sounding")
    plt.tight_layout()
    plt.savefig("skewt_first_sounding.png")
    plt.close()
    
    # 2. Temperature profile comparison
    plt.figure(figsize=(10, 8))
    
    for i, date in enumerate(raw_df['datetime'].unique()[:3]):
        profile = raw_df[raw_df['datetime'] == date]
        plt.plot(profile['temperature'], profile['pressure'], marker='o', label=str(date))
    
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
    
    # 3. Wind profile barbs for the first sounding
    if 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        first_date = raw_df['datetime'].iloc[0]
        wind_data = raw_df[raw_df['datetime'] == first_date].dropna(subset=['wind_speed', 'wind_direction'])
        
        if not wind_data.empty:
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
            plt.title(f'Wind Profile - {first_date}')
            plt.ylabel('Pressure (hPa)')
            plt.grid(True)
            plt.savefig("wind_profile.png")
            plt.close()
    
    # 4. Create xarray visualizations
    if raw_ds is not None:
        # Time-height cross section of temperature
        try:
            # Extract the temperature data
            temp_data = raw_ds['temperature']
            
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
        except Exception as e:
            print(f"Error creating xarray plot: {e}")
    
    # 5. Analyze atmospheric stability if derived parameters are available
    if derived_df is not None and not derived_df.empty:
        # Extract stability parameters
        stability_cols = [col for col in ['cape', 'cin', 'lifted_index'] if col in derived_df.columns]
        
        if stability_cols:
            # Get one value per sounding
            stability_df = derived_df.drop_duplicates(subset=['datetime'])[['datetime'] + stability_cols]
            
            # Print summary
            print("\n=== Stability Parameters ===")
            for col in stability_cols:
                print(f"{col}: Mean = {stability_df[col].mean():.2f}, Min = {stability_df[col].min():.2f}, Max = {stability_df[col].max():.2f}")
            
            # Plot time series
            if len(stability_df) > 1:
                plt.figure(figsize=(12, 6))
                for col in stability_cols:
                    plt.plot(stability_df['datetime'], stability_df[col], marker='o', label=col)
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title('Stability Indices')
                plt.legend()
                plt.grid(True)
                plt.savefig("stability_indices.png")
                plt.close()
    
    # 6. Save to common formats for further analysis
    
    # Save to CSV
    raw_df.to_csv("igra_raw_data.csv", index=False)
    if derived_df is not None and not derived_df.empty:
        derived_df.to_csv("igra_derived_data.csv", index=False)
    
    # Save to NetCDF using xarray
    if raw_ds is not None:
        raw_ds.to_netcdf("igra_raw_data.nc")
    if derived_ds is not None:
        derived_ds.to_netcdf("igra_derived_data.nc")
    
    print("\nCompleted data analysis and saved results.")
    print("Created the following files:")
    print("  - skewt_first_sounding.png - Skew-T diagram of the first sounding")
    print("  - temperature_profiles.png - Temperature profile comparison")
    print("  - wind_profile.png - Wind barbs profile")
    print("  - temp_xarray_plot.png - Temperature time-height cross section")
    print("  - stability_indices.png - Stability indices time series (if available)")
    print("  - igra_raw_data.csv - CSV file with raw data")
    print("  - igra_derived_data.csv - CSV file with derived data")
    print("  - igra_raw_data.nc - NetCDF file with raw data")
    print("  - igra_derived_data.nc - NetCDF file with derived data")
    
    return {
        'raw_df': raw_df,
        'raw_ds': raw_ds,
        'derived_df': derived_df,
        'derived_ds': derived_ds
    }

if __name__ == "__main__":
    main()
