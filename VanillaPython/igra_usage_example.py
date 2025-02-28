import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from igra_loader import *  # Import the functions from the previous code

# Sample usage with your data
def analyze_igra_data(raw_data_text, derived_data_text=None):
    """
    Analyze IGRA data from text strings.
    """
    # Load the data
    raw_soundings = load_igra_data_from_text(raw_data_text, 'raw')
    raw_df = soundings_to_dataframe(raw_soundings)
    raw_ds = soundings_to_xarray(raw_soundings)
    
    if derived_data_text:
        derived_soundings = load_igra_data_from_text(derived_data_text, 'derived')
        derived_df = soundings_to_dataframe(derived_soundings)
        derived_ds = soundings_to_xarray(derived_soundings)
    else:
        derived_soundings = None
        derived_df = None
        derived_ds = None
    
    # Basic statistics
    print("=== Basic Statistics ===")
    print(f"Number of soundings: {len(raw_soundings)}")
    print(f"Date range: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")
    print(f"Pressure levels: Min {raw_df['pressure'].min():.1f} hPa, Max {raw_df['pressure'].max():.1f} hPa")
    
    # Temperature statistics
    print("\n=== Temperature Statistics ===")
    temp_stats = raw_df.groupby('pressure')['temperature'].agg(['mean', 'std', 'min', 'max']).reset_index()
    print(temp_stats.head())
    
    # Create visualizations
    
    # 1. Skew-T plot for a selected sounding
    fig, skew = plot_skewt(raw_soundings[0], "Sample Sounding")
    plt.savefig('skewt_plot.png')
    plt.close()
    
    # 2. Time-height plot of temperature
    if len(raw_soundings) > 1:
        # Create a pivot table for time-height plotting
        pivot_df = raw_df.pivot_table(
            index='datetime', 
            columns='pressure',
            values='temperature'
        )
        
        # Plot the time-height diagram
        plt.figure(figsize=(12, 8))
        plt.contourf(
            pivot_df.columns, 
            pivot_df.index, 
            pivot_df.values, 
            cmap='RdBu_r',
            levels=np.arange(-80, 40, 5)
        )
        plt.colorbar(label='Temperature (°C)')
        plt.yscale('log')
        plt.gca().invert_yaxis()  # Invert y-axis to have pressure decrease upward
        plt.ylabel('Pressure (hPa)')
        plt.xlabel('Date')
        plt.title('Temperature Time-Height Cross Section')
        plt.savefig('time_height_temp.png')
        plt.close()
    
    # 3. Wind profile plot
    if 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        # Get data for the first sounding
        sounding = raw_df[raw_df['datetime'] == raw_df['datetime'].iloc[0]]
        
        # Filter out rows with missing wind data
        wind_data = sounding.dropna(subset=['wind_speed', 'wind_direction'])
        
        if not wind_data.empty:
            # Convert wind speed and direction to u and v components
            wind_data['u'] = -wind_data['wind_speed'] * np.sin(np.radians(wind_data['wind_direction']))
            wind_data['v'] = -wind_data['wind_speed'] * np.cos(np.radians(wind_data['wind_direction']))
            
            # Plot the wind profile
            plt.figure(figsize=(10, 8))
            plt.barbs(
                np.zeros_like(wind_data['pressure']), 
                wind_data['pressure'],
                wind_data['u'].values,
                wind_data['v'].values
            )
            plt.ylim(1050, 100)
            plt.gca().invert_yaxis()
            plt.xlabel('Horizontal Position')
            plt.ylabel('Pressure (hPa)')
            plt.title(f'Wind Profile - {wind_data["datetime"].iloc[0]}')
            plt.grid(True)
            plt.savefig('wind_profile.png')
            plt.close()
    
    # Return the loaded data for further analysis
    return {
        'raw_soundings': raw_soundings,
        'raw_df': raw_df,
        'raw_ds': raw_ds,
        'derived_soundings': derived_soundings,
        'derived_df': derived_df,
        'derived_ds': derived_ds
    }

# Example code to process multiple files
def process_igra_files(data_dir, station_id=None):
    """
    Process multiple IGRA data files in a directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing IGRA data files
    station_id : str, optional
        Station ID to filter files
    """
    # Find all data files
    if station_id:
        raw_files = glob.glob(os.path.join(data_dir, f"{station_id}*-data.txt*"))
        derived_files = glob.glob(os.path.join(data_dir, f"{station_id}*-drvd.txt*"))
    else:
        raw_files = glob.glob(os.path.join(data_dir, "*-data.txt*"))
        derived_files = glob.glob(os.path.join(data_dir, "*-drvd.txt*"))
    
    # Create dictionaries to store data by station
    stations_raw_df = {}
    stations_derived_df = {}
    
    # Process each station's data
    for raw_file in raw_files:
        station_id = os.path.basename(raw_file).split('-')[0]
        print(f"Processing raw data for station {station_id}")
        
        # Load raw data
        raw_soundings = load_igra_file(raw_file, 'raw')
        raw_df = soundings_to_dataframe(raw_soundings)
        
        # Store in dictionary
        stations_raw_df[station_id] = raw_df
    
    # Process derived data if available
    for derived_file in derived_files:
        station_id = os.path.basename(derived_file).split('-')[0]
        print(f"Processing derived data for station {station_id}")
        
        # Load derived data
        derived_soundings = load_igra_file(derived_file, 'derived')
        derived_df = soundings_to_dataframe(derived_soundings)
        
        # Store in dictionary
        stations_derived_df[station_id] = derived_df
    
    return {
        'raw_data': stations_raw_df,
        'derived_data': stations_derived_df
    }

# Example of creating a map of station locations
def plot_station_locations(stations_df):
    """
    Plot the locations of IGRA stations on a map.
    
    Parameters:
    -----------
    stations_df : DataFrame
        DataFrame containing station information with latitude and longitude
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy is required for map plotting. Install with 'pip install cartopy'")
        return
    
    # Get unique stations
    unique_stations = stations_df[['station_id', 'latitude', 'longitude']].drop_duplicates()
    
    # Create the map
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Plot station locations
    ax.scatter(
        unique_stations['longitude'], 
        unique_stations['latitude'],
        color='red',
        marker='^',
        s=50,
        transform=ccrs.PlateCarree()
    )
    
    # Add station labels
    for _, station in unique_stations.iterrows():
        ax.text(
            station['longitude'] + 1, 
            station['latitude'] + 0.5, 
            station['station_id'],
            transform=ccrs.PlateCarree(),
            fontsize=8
        )
    
    plt.title('IGRA Station Locations')
    plt.savefig('station_locations.png')
    plt.close()

# Example of creating some more advanced plots
def create_advanced_plots(data):
    """
    Create some more advanced visualizations using the loaded data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing loaded data from analyze_igra_data()
    """
    raw_df = data['raw_df']
    
    # 1. Vertical profile comparison
    # Compare temperature profiles from different dates
    unique_dates = raw_df['datetime'].unique()
    
    if len(unique_dates) > 1:
        plt.figure(figsize=(10, 8))
        
        for i, date in enumerate(unique_dates[:3]):  # Limit to first 3 dates to avoid clutter
            profile = raw_df[raw_df['datetime'] == date]
            
            # Plot temperature profile
            plt.plot(
                profile['temperature'], 
                profile['pressure'],
                marker='o',
                label=str(date)
            )
        
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.ylim(1050, 100)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Pressure (hPa)')
        plt.legend()
        plt.title('Temperature Profiles Comparison')
        plt.grid(True)
        plt.savefig('temp_profile_comparison.png')
        plt.close()
    
    # 2. Time series of specific level
    # Extract data for 500 hPa level
    level_500 = raw_df[raw_df['pressure'].between(495, 505)]
    
    if not level_500.empty and len(level_500) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(
            level_500['datetime'],
            level_500['temperature'],
            marker='o',
            linestyle='-'
        )
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('500 hPa Temperature Time Series')
        plt.grid(True)
        plt.savefig('500hPa_temp_series.png')
        plt.close()
    
    # 3. Stability indices if derived data is available
    derived_df = data.get('derived_df')
    
    if derived_df is not None and not derived_df.empty:
        # Check if stability indices are available
        stability_cols = ['lifted_index', 'cape', 'cin']
        available_cols = [col for col in stability_cols if col in derived_df.columns]
        
        if available_cols:
            # Aggregate to get one value per sounding
            stability_df = derived_df.drop_duplicates(subset=['datetime'])[['datetime'] + available_cols]
            
            if not stability_df.empty and len(stability_df) > 1:
                plt.figure(figsize=(12, 6))
                
                for col in available_cols:
                    plt.plot(
                        stability_df['datetime'],
                        stability_df[col],
                        marker='o',
                        linestyle='-',
                        label=col
                    )
                
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title('Stability Indices Time Series')
                plt.legend()
                plt.grid(True)
                plt.savefig('stability_indices.png')
                plt.close()
    
    # 4. Create a hodograph for wind data if available
    if 'wind_speed' in raw_df.columns and 'wind_direction' in raw_df.columns:
        # Get data for the first sounding
        sounding = raw_df[raw_df['datetime'] == raw_df['datetime'].iloc[0]]
        
        # Filter out rows with missing wind data
        wind_data = sounding.dropna(subset=['wind_speed', 'wind_direction'])
        
        if not wind_data.empty:
            # Convert wind speed and direction to u and v components
            wind_data['u'] = -wind_data['wind_speed'] * np.sin(np.radians(wind_data['wind_direction']))
            wind_data['v'] = -wind_data['wind_speed'] * np.cos(np.radians(wind_data['wind_direction']))
            
            # Create hodograph
            plt.figure(figsize=(8, 8))
            
            # Plot circles for wind speed reference
            for r in [10, 20, 30, 40, 50]:
                circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--')
                plt.gca().add_artist(circle)
            
            # Plot the hodograph line
            plt.plot(wind_data['u'], wind_data['v'], 'r-')
            
            # Add markers for different pressure levels
            for level in [1000, 850, 700, 500, 300, 200]:
                level_data = wind_data[wind_data['pressure'].between(level-5, level+5)]
                if not level_data.empty:
                    plt.plot(
                        level_data['u'].values[0],
                        level_data['v'].values[0],
                        'bo',
                        markersize=8
                    )
                    plt.text(
                        level_data['u'].values[0] + 1,
                        level_data['v'].values[0] + 1,
                        str(level),
                        fontsize=10
                    )
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-50, 50)
            plt.ylim(-50, 50)
            plt.xlabel('U Wind Component (m/s)')
            plt.ylabel('V Wind Component (m/s)')
            plt.title(f'Hodograph - {wind_data["datetime"].iloc[0]}')
            plt.savefig('hodograph.png')
            plt.close()