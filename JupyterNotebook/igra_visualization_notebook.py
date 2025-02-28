# IGRA Weather Balloon Data Parser and Visualizer
# ===================================================
# This notebook helps beginners parse and visualize IGRA weather balloon (radiosonde) data
# Author: Claude
# Date: February 28, 2025

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set some visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Wider display

# ===================================================
# PART 1: Functions to Parse IGRA Data
# ===================================================

def parse_igra_header(header_line):
    """
    Parse a header line from IGRA data files.
    
    Parameters:
    -----------
    header_line : str
        A header line from an IGRA data file starting with '#'
        
    Returns:
    --------
    dict
        Dictionary with header information
    """
    # Ensure it's a header line
    if not header_line.startswith('#'):
        raise ValueError("Not a header line: doesn't start with #")
    
    # Parse according to format specification
    header = {
        'ID': header_line[1:12].strip(),
        'YEAR': int(header_line[13:17].strip()),
        'MONTH': int(header_line[18:20].strip()),
        'DAY': int(header_line[21:23].strip()),
        'HOUR': int(header_line[24:26].strip()),
    }
    
    # Add a datetime field for easier plotting
    try:
        if header['HOUR'] == 99:  # Missing hour
            header['datetime'] = datetime(header['YEAR'], header['MONTH'], header['DAY'])
        else:
            header['datetime'] = datetime(header['YEAR'], header['MONTH'], header['DAY'], header['HOUR'])
    except ValueError:
        # Handle potential invalid dates
        header['datetime'] = None
    
    return header

def parse_igra_data_line(data_line):
    """
    Parse a data line from IGRA data files.
    
    Parameters:
    -----------
    data_line : str
        A data line from an IGRA data file
        
    Returns:
    --------
    dict
        Dictionary with the data values
    """
    data = {
        'LVLTYP1': int(data_line[0:1].strip()),
        'LVLTYP2': int(data_line[1:2].strip()),
        'PRESS': int(data_line[9:15].strip()),
        'GPH': data_line[16:21].strip(),
        'TEMP': data_line[22:27].strip(),
        'RH': data_line[28:33].strip(),
        'WSPD': data_line[46:51].strip(),
        'WDIR': data_line[40:45].strip(),
    }
    
    # Convert the string values to proper numeric types, handling missing values
    for key in ['GPH', 'TEMP', 'RH', 'WSPD', 'WDIR']:
        try:
            if data[key].strip() == '-9999' or data[key].strip() == '-8888' or data[key].strip() == '':
                data[key] = np.nan
            else:
                data[key] = float(data[key])
                # Temperature is in tenths of degrees C
                if key == 'TEMP':
                    data[key] = data[key] / 10
                # Wind speed is in tenths of m/s
                if key == 'WSPD':
                    data[key] = data[key] / 10
        except ValueError:
            data[key] = np.nan
    
    # Convert pressure from Pa to hPa (mb)
    data['PRESS'] = data['PRESS'] / 100
    
    return data

def parse_igra_derived_line(data_line):
    """
    Parse a data line from IGRA derived parameters files.
    
    Parameters:
    -----------
    data_line : str
        A data line from an IGRA derived parameters file
        
    Returns:
    --------
    dict
        Dictionary with the derived data values
    """
    data = {
        'PRESS': int(data_line[0:7].strip()),
        'CALCGPH': data_line[16:23].strip(),
        'TEMP': data_line[24:31].strip(),
        'PTEMP': data_line[40:47].strip(),  # Potential temperature
        'CALCRH': data_line[96:103].strip(),  # Calculated RH
        'UWND': data_line[112:119].strip(),   # U wind component
        'VWND': data_line[128:135].strip(),   # V wind component
    }
    
    # Convert the string values to proper numeric types, handling missing values
    for key in ['CALCGPH', 'TEMP', 'PTEMP', 'CALCRH', 'UWND', 'VWND']:
        try:
            if data[key].strip() == '-99999' or data[key].strip() == '':
                data[key] = np.nan
            else:
                data[key] = float(data[key])
                # Temperature and potential temperature are in K*10
                if key in ['TEMP', 'PTEMP']:
                    data[key] = data[key] / 10
                # RH is in percent*10
                if key == 'CALCRH':
                    data[key] = data[key] / 10
                # Wind components are in (m/s)*10
                if key in ['UWND', 'VWND']:
                    data[key] = data[key] / 10
        except ValueError:
            data[key] = np.nan
    
    # Convert pressure from Pa to hPa (mb)
    data['PRESS'] = data['PRESS'] / 100
    
    return data

def parse_igra_file(file_path, is_derived=False):
    """
    Parse an entire IGRA data file into a list of soundings.
    
    Parameters:
    -----------
    file_path : str
        Path to the IGRA data file
    is_derived : bool
        Whether the file is a derived parameters file
        
    Returns:
    --------
    list
        List of dictionaries, each containing a sounding's header and data
    """
    soundings = []
    current_sounding = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # If this is a header line, start a new sounding
            if line.startswith('#'):
                if current_sounding is not None:
                    soundings.append(current_sounding)
                
                current_sounding = {
                    'header': parse_igra_header(line),
                    'data': []
                }
            # Otherwise, this is a data line
            elif current_sounding is not None:
                if is_derived:
                    data = parse_igra_derived_line(line)
                else:
                    data = parse_igra_data_line(line)
                current_sounding['data'].append(data)
    
    # Don't forget the last sounding
    if current_sounding is not None:
        soundings.append(current_sounding)
    
    return soundings

def convert_soundings_to_df(soundings):
    """
    Convert a list of soundings to a pandas DataFrame with a MultiIndex.
    
    Parameters:
    -----------
    soundings : list
        List of soundings as returned by parse_igra_file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with MultiIndex (datetime, pressure)
    """
    all_data = []
    
    for sounding in soundings:
        header = sounding['header']
        
        for data_point in sounding['data']:
            # Combine header and data
            combined = header.copy()
            combined.update(data_point)
            all_data.append(combined)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Set MultiIndex for easier analysis
    if 'datetime' in df.columns:
        df = df.set_index(['datetime', 'PRESS'])
    
    return df

# ===================================================
# PART 2: Example Usage - Load and Parse Data
# ===================================================

# Set the file paths - adjust these to your actual file locations
raw_data_file = 'USM0007479f-data-exampleportion.txt'
derived_data_file = 'USM0007479f-drvd-exampleportion.txt'

# Parse the raw data file
print("Parsing raw data file...")
raw_soundings = parse_igra_file(raw_data_file, is_derived=False)
print(f"Found {len(raw_soundings)} soundings in the raw data file")

# Parse the derived parameters file
print("Parsing derived parameters file...")
derived_soundings = parse_igra_file(derived_data_file, is_derived=True)
print(f"Found {len(derived_soundings)} soundings in the derived parameters file")

# Convert to DataFrames
raw_df = convert_soundings_to_df(raw_soundings)
derived_df = convert_soundings_to_df(derived_soundings)

# Display first few rows of each DataFrame
print("\nRaw Data Sample:")
display(raw_df.head())

print("\nDerived Data Sample:")
display(derived_df.head())

# ===================================================
# PART 3: Basic Visualizations
# ===================================================

# Example 1: Plot temperature profiles for all soundings
plt.figure(figsize=(10, 8))

# Reset index to use the columns in the plot
temp_data = raw_df.reset_index()

# Group by datetime and plot each sounding as a separate line
for date, group in temp_data.groupby('datetime'):
    # Skip groups with missing temperature data
    if group['TEMP'].isna().all():
        continue
    
    # Sort by pressure (descending) for correct vertical profile
    group = group.sort_values('PRESS', ascending=False)
    
    # Plot temperature vs pressure (note: y-axis is inverted for pressure)
    plt.plot(group['TEMP'], group['PRESS'], 
             marker='o', linewidth=1, markersize=4, 
             label=date.strftime('%Y-%m-%d %H:%M'))

# Customize the plot
plt.gca().invert_yaxis()  # Invert y-axis to show decreasing pressure with height
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')
plt.title('Temperature Profiles from Weather Balloon Data')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Example 2: Wind barbs for a single sounding
# Choose a single sounding for demonstration
if len(raw_soundings) > 0:
    sample_date = raw_soundings[0]['header']['datetime']
    sample_sounding = temp_data[temp_data['datetime'] == sample_date]
    
    # Only proceed if we have wind data
    if not (sample_sounding['WSPD'].isna().all() or sample_sounding['WDIR'].isna().all()):
        plt.figure(figsize=(8, 10))
        
        # Convert wind speed and direction to u, v components
        # Wind direction is in meteorological convention (0=N, 90=E)
        has_wind = ~(sample_sounding['WSPD'].isna() | sample_sounding['WDIR'].isna())
        wind_data = sample_sounding[has_wind].copy()
        
        # Convert degrees to radians for sin/cos
        wind_dir_rad = np.radians(270 - wind_data['WDIR'])  # Convert from meteorological to mathematical
        
        # Calculate u, v components
        wind_data['u'] = -wind_data['WSPD'] * np.cos(wind_dir_rad)  # U component (+ = eastward)
        wind_data['v'] = -wind_data['WSPD'] * np.sin(wind_dir_rad)  # V component (+ = northward)
        
        # Sort by pressure (descending) for correct vertical profile
        wind_data = wind_data.sort_values('PRESS', ascending=False)
        
        # Create the wind barb plot
        plt.barbs(np.zeros_like(wind_data['PRESS']), wind_data['PRESS'], 
                 wind_data['u'], wind_data['v'], 
                 length=6, pivot='middle', linewidth=1.5)
        
        # Customize the plot
        plt.gca().invert_yaxis()
        plt.xlim(-1, 1)  # Center the barbs
        plt.xlabel('')
        plt.ylabel('Pressure (hPa)')
        plt.title(f'Wind Profile for {sample_date.strftime("%Y-%m-%d %H:%M")}')
        plt.grid(True)
        plt.tight_layout()

# Example 3: Temperature vs height scatter plot
plt.figure(figsize=(10, 8))

# Plot all temperatures from all soundings
plt.scatter(derived_df['TEMP'], derived_df['CALCGPH'], 
           alpha=0.6, marker='o', s=20, c=derived_df['TEMP'], cmap='coolwarm')

plt.colorbar(label='Temperature (K)')
plt.xlabel('Temperature (K)')
plt.ylabel('Height (m)')
plt.title('Temperature vs Height')
plt.grid(True)
plt.tight_layout()

# ===================================================
# PART 4: Time Series Analysis
# ===================================================

# Example: Track temperature at 500 hPa over time
pressure_level = 50.0  # 500 hPa (50 hPa in derived data due to different units)

# Extract data for this pressure level
level_data = derived_df.reset_index()
level_data = level_data[abs(level_data['PRESS'] - pressure_level) < 0.1]

if not level_data.empty:
    plt.figure(figsize=(12, 6))
    
    # Sort by datetime
    level_data = level_data.sort_values('datetime')
    
    # Plot time series
    plt.plot(level_data['datetime'], level_data['TEMP'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    
    plt.xlabel('Date and Time')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature at {pressure_level*10} hPa Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

# ===================================================
# PART 5: Create custom visualizations
# ===================================================

# Function to create a skew-T diagram (simplified version)
def plot_skewt(sounding_data, title=None):
    """
    Create a simplified skew-T log-P diagram for a single sounding.
    
    Parameters:
    -----------
    sounding_data : DataFrame
        DataFrame containing a single sounding's data
    title : str, optional
        Title for the plot
    """
    # Ensure data is sorted by pressure
    sounding_data = sounding_data.sort_values('PRESS', ascending=False)
    
    # Check if we have temperature data
    if sounding_data['TEMP'].isna().all():
        print("No temperature data available for this sounding")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Calculate skew factor (simplified approach)
    skew = 30  # Degrees to skew the temperature lines
    
    # Plot temperature (with skew)
    temps = sounding_data['TEMP'].values
    pressures = sounding_data['PRESS'].values
    heights = np.log(1000/pressures) * skew  # Simple way to simulate skew
    
    ax.plot(temps + heights, pressures, 'r-', label='Temperature', linewidth=2)
    
    # If available, plot dewpoint temperature
    if 'DPDP' in sounding_data.columns and not sounding_data['DPDP'].isna().all():
        dewpoints = temps - sounding_data['DPDP'].values / 10  # Dewpoint depression to dewpoint
        ax.plot(dewpoints + heights, pressures, 'g-', label='Dewpoint', linewidth=2)
    
    # Add wind barbs if data is available
    if not (sounding_data['WSPD'].isna().all() or sounding_data['WDIR'].isna().all()):
        wind_data = sounding_data[~(sounding_data['WSPD'].isna() | sounding_data['WDIR'].isna())]
        
        # Convert wind direction to u, v components
        wind_dir_rad = np.radians(270 - wind_data['WDIR'])
        u = -wind_data['WSPD'] * np.cos(wind_dir_rad)
        v = -wind_data['WSPD'] * np.sin(wind_dir_rad)
        
        # Add wind barbs to right side of plot
        for i, p in enumerate(wind_data['PRESS']):
            if i % 2 == 0:  # Plot every other level to avoid crowding
                ax.barbs(temps.max() + heights.max() + 10, p, u.iloc[i], v.iloc[i], 
                        length=5, pivot='middle', linewidth=1.5)
    
    # Set up the axes
    ax.set_yscale('log')  # Log scale for pressure
    ax.invert_yaxis()  # Invert y-axis (pressure decreases with height)
    
    # Set y-axis limits and ticks
    ax.set_ylim(1050, 100)  # Typical tropospheric pressure range
    pressure_ticks = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]
    ax.set_yticks(pressure_ticks)
    ax.set_yticklabels([str(p) for p in pressure_ticks])
    
    # Add temperature grid lines (simplified)
    for temp in range(-80, 41, 10):
        x = np.linspace(temp, temp + heights.max(), 100)
        y = np.logspace(np.log10(100), np.log10(1050), 100)
        ax.plot(x, y, 'k-', alpha=0.2, linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (hPa)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Simplified Skew-T Diagram')
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Try to create a skew-T for the first sounding with sufficient data
for sounding in raw_soundings:
    sounding_data = pd.DataFrame(sounding['data'])
    
    # Check if this sounding has enough data
    if (not sounding_data['TEMP'].isna().all() and 
        len(sounding_data.dropna(subset=['TEMP'])) > 5):
        
        date_str = sounding['header']['datetime'].strftime('%Y-%m-%d %H:%M')
        plot_skewt(sounding_data, f'Skew-T for {date_str}')
        break

# ===================================================
# PART 6: Save the processed data for future use
# ===================================================

# Function to save DataFrames to CSV
def save_data_to_csv(df, filename):
    """
    Save a DataFrame to CSV with proper handling of MultiIndex.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    filename : str
        Output file name
    """
    # Reset index to have datetime as a column
    df_to_save = df.reset_index()
    
    # Convert datetime to string format for better CSV compatibility
    if 'datetime' in df_to_save.columns:
        df_to_save['datetime'] = df_to_save['datetime'].astype(str)
    
    # Save to CSV
    df_to_save.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Example of saving the processed data
# Uncomment to use:
# save_data_to_csv(raw_df, 'processed_raw_data.csv')
# save_data_to_csv(derived_df, 'processed_derived_data.csv')

print("\nNotebook execution complete!")
