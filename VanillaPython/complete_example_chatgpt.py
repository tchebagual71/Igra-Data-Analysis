import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

from igra_loader import (
    load_igra_data_from_text, 
    soundings_to_dataframe, 
    soundings_to_xarray, 
    plot_skewt
)

def create_output_directory():
    """
    Creates a unique output directory for each run inside 'output/'.
    Returns the directory path.
    """
    base_output_dir = "output"
    os.makedirs(base_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(run_output_dir)
    
    return run_output_dir

def main():
    """
    Main function to load, analyze, and save IGRA data with organized output.
    """
    output_dir = create_output_directory()
    print(f"Output files will be saved in: {output_dir}")

    # Load the sample data
    with open("/home/tdieckman/Igra-Data-Analysis/USM0007479f-data-exampleportion.txt", "r") as f:
        raw_data_text = f.read()
    
    with open("/home/tdieckman/Igra-Data-Analysis/USM0007479f-drvd-exampleportion.txt", "r") as f:
        derived_data_text = f.read()

    print("Loading IGRA data...")
    
    raw_soundings = load_igra_data_from_text(raw_data_text, 'raw')
    raw_df = soundings_to_dataframe(raw_soundings)
    raw_ds = soundings_to_xarray(raw_soundings)
    
    derived_soundings = load_igra_data_from_text(derived_data_text, 'derived')
    derived_df = soundings_to_dataframe(derived_soundings)
    derived_ds = soundings_to_xarray(derived_soundings)
    
    print(f"Loaded {len(raw_soundings)} raw soundings and {len(derived_soundings)} derived soundings")
    
    # Save Data
    raw_df.to_csv(os.path.join(output_dir, "igra_raw_data.csv"), index=False)
    if derived_df is not None and not derived_df.empty:
        derived_df.to_csv(os.path.join(output_dir, "igra_derived_data.csv"), index=False)

    if raw_ds is not None:
        raw_ds.to_netcdf(os.path.join(output_dir, "igra_raw_data.nc"))
    if derived_ds is not None:
        derived_ds.to_netcdf(os.path.join(output_dir, "igra_derived_data.nc"))

    print("Saved data to CSV and NetCDF files.")

    # Skew-T plot
    first_sounding = raw_soundings[0]
    skewt_result = plot_skewt(first_sounding, "Sample Sounding")
    
    if skewt_result:
        fig, skew = skewt_result
        plt.savefig(os.path.join(output_dir, "skewt_first_sounding.png"))
        plt.close()
        print("Created Skew-T plot for first sounding.")
    else:
        print("Could not create Skew-T plot for first sounding. Trying an alternate sounding...")
        for i, sounding in enumerate(raw_soundings):
            skewt_result = plot_skewt(sounding, f"Sounding {i+1}")
            if skewt_result:
                fig, skew = skewt_result
                plt.savefig(os.path.join(output_dir, "skewt_alternate_sounding.png"))
                plt.close()
                print(f"Created Skew-T plot for sounding {i+1}")
                break

    # Temperature profile plot
    plt.figure(figsize=(10, 8))
    for date in raw_df['datetime'].unique():
        profile = raw_df[(raw_df['datetime'] == date) & raw_df['temperature'].notna()]
        if len(profile) >= 5:
            plt.plot(profile['temperature'], profile['pressure'], marker='o', label=str(date))

    if plt.gca().lines:
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Pressure (hPa)')
        plt.title('Temperature Profiles')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "temperature_profiles.png"))
        plt.close()
        print("Created temperature profile plot.")
    
    # Wind profile
    wind_found = False
    if 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        for date in raw_df['datetime'].unique():
            wind_data = raw_df[(raw_df['datetime'] == date) & 
                               ~raw_df['wind_speed'].isna() & 
                               ~raw_df['wind_direction'].isna()]
            if len(wind_data) >= 3:
                wind_found = True
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
                print(f"Created wind profile plot for {date}")
                break
    
    if not wind_found:
        print("Could not create wind profile plot due to insufficient wind data.")

    # Temperature Time-Height Cross Section
    if raw_ds is not None:
        temp_data = raw_ds['temperature']
        if not temp_data.isnull().all():
            plt.figure(figsize=(12, 8))
            temp_data.plot.contourf(x='datetime', y='pressure', levels=np.arange(-80, 40, 5), cmap='RdBu_r')
            plt.gca().invert_yaxis()
            plt.title('Temperature Time-Height Cross Section')
            plt.yscale('log')
            plt.savefig(os.path.join(output_dir, "temp_xarray_plot.png"))
            plt.close()
            print("Created temperature time-height cross section plot.")

    # Stability Indices Plot
    stability_cols = [col for col in ['cape', 'cin', 'lifted_index'] if col in derived_df.columns]
    if stability_cols:
        stability_df = derived_df.drop_duplicates(subset=['datetime'])[['datetime'] + stability_cols].dropna(subset=stability_cols, how='all')
        if not stability_df.empty:
            plt.figure(figsize=(12, 6))
            for col in stability_cols:
                plt.plot(stability_df['datetime'], stability_df[col], marker='o', label=col)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Stability Indices')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "stability_indices.png"))
            plt.close()
            print("Created stability indices time series plot.")

    print(f"\nAll output files saved in: {output_dir}")

if __name__ == "__main__":
    main()
