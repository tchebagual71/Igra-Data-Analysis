# IGRA Data Analysis - Beginner's Tutorial

This tutorial will walk you through analyzing atmospheric sounding data from the Integrated Global Radiosonde Archive (IGRA) using Python in a Jupyter Notebook. No prior coding experience necessary!

## What You'll Learn
- How to save and organize IGRA data files
- How to set up a Jupyter Notebook for data analysis
- How to load and parse IGRA data
- How to create simple visualizations of atmospheric data

## Prerequisites
- Python (3.7 or newer)
- Jupyter Notebook
- Basic Python packages: pandas, matplotlib, numpy
- MetPy (for meteorological calculations and plots)

## Step 1: Setting Up Your Project Folder

1. Create a new folder on your computer for this project (e.g., "IGRA_Analysis")
2. Inside this folder, create a subfolder called "data"

## Step 2: Saving the Data Files

1. In the "data" folder, create two text files:
   - `raw_data.txt` - for the raw sounding data 
   - `derived_data.txt` - for the derived parameters data

2. Copy and paste the sample data from the examples into these files:
   - Copy the content from the "paste.txt" example into `raw_data.txt`
   - Copy the content from the "paste-2.txt" example into `derived_data.txt`

3. Make sure to save the files as plain text (.txt) files

## Step 3: Starting a Jupyter Notebook

1. Open a terminal or command prompt
2. Navigate to your project folder using the `cd` command:
   ```
   cd path/to/your/IGRA_Analysis
   ```
3. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. In the browser window that opens, click on "New" and select "Python 3" to create a new notebook
5. Rename the notebook by clicking on "Untitled" at the top and changing it to "IGRA_Data_Analysis"

## Step 4: Installing Required Packages

In the first cell of your notebook, type and run the following code to install required packages (if you don't have them already):

```python
# Run this if you don't have the packages installed
# Uncomment the lines below by removing the # at the beginning

# !pip install pandas matplotlib numpy metpy xarray
```

## Step 5: Importing Libraries

In the next cell, import the necessary libraries:

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# For meteorological plots
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
```

## Step 6: Reading the Data Files

Read the data files you saved earlier:

```python
# Read the data files
with open('data/raw_data.txt', 'r') as file:
    raw_data_text = file.read()
    
with open('data/derived_data.txt', 'r') as file:
    derived_data_text = file.read()

print(f"Raw data: {len(raw_data_text)} characters")
print(f"Derived data: {len(derived_data_text)} characters")
```

## Step 7: Parsing the IGRA Data - Helper Functions

We'll need some helper functions to parse the IGRA data format. Add the following code in a new cell:

```python
def parse_igra_header(header_line):
    """Parse the header line of IGRA raw sounding data."""
    header = {}
    try:
        header['station_id'] = header_line[1:12].strip()
        header['year'] = int(header_line[13:17])
        header['month'] = int(header_line[18:20])
        header['day'] = int(header_line[21:23])
        header['hour'] = int(header_line[24:26])
        header['reltime'] = header_line[27:31].strip()
        header['num_levels'] = int(header_line[32:36])
        
        # Try to parse datetime
        try:
            header['datetime'] = datetime(
                header['year'], header['month'], header['day'], 
                hour=0 if header['hour'] == 99 else header['hour']
            )
        except ValueError:
            header['datetime'] = None
    except Exception as e:
        print(f"Error parsing header: {e}")
        print(f"Header line: {header_line}")
        return None
    
    return header

def parse_igra_data_line(line, format_type='raw'):
    """Parse a data line from IGRA data.
    format_type can be 'raw' or 'derived'"""
    data = {}
    
    if format_type == 'raw':
        # Raw sounding data
        try:
            # Skip the first few columns that contain level type info
            press_str = line[9:15].strip()
            data['pressure'] = int(press_str) if press_str and press_str != '-9999' else np.nan
            
            gph_str = line[16:21].strip()
            data['height'] = int(gph_str) if gph_str and gph_str not in ['-8888', '-9999'] else np.nan
            
            temp_str = line[22:27].strip()
            data['temperature'] = int(temp_str) / 10 if temp_str and temp_str not in ['-8888', '-9999'] else np.nan
            
            rh_str = line[28:33].strip()
            data['relative_humidity'] = int(rh_str) / 10 if rh_str and rh_str not in ['-8888', '-9999'] else np.nan
            
            dpdp_str = line[34:39].strip()
            data['dewpoint_depression'] = int(dpdp_str) / 10 if dpdp_str and dpdp_str not in ['-8888', '-9999'] else np.nan
            
            wdir_str = line[40:45].strip()
            data['wind_direction'] = int(wdir_str) if wdir_str and wdir_str not in ['-8888', '-9999'] else np.nan
            
            wspd_str = line[46:51].strip()
            data['wind_speed'] = int(wspd_str) / 10 if wspd_str and wspd_str not in ['-8888', '-9999'] else np.nan
            
            # Calculate dewpoint from temperature and dewpoint depression if both are available
            if not np.isnan(data['temperature']) and not np.isnan(data['dewpoint_depression']):
                data['dewpoint'] = data['temperature'] - data['dewpoint_depression']
            else:
                data['dewpoint'] = np.nan
                
        except Exception as e:
            print(f"Error parsing raw data line: {e}")
            print(f"Line: {line}")
            return None
    else:
        # For derived data, we'll keep it simpler
        try:
            press_str = line[:7].strip()
            data['pressure'] = int(press_str) if press_str and press_str != '-99999' else np.nan
            
            temp_str = line[24:31].strip()
            data['temperature'] = int(temp_str) / 10 if temp_str and temp_str != '-99999' else np.nan
            
            # Get another relevant column that might be useful
            rh_str = line[96:103].strip() if len(line) > 100 else ''
            data['relative_humidity'] = int(rh_str) / 10 if rh_str and rh_str != '-99999' else np.nan
            
        except Exception as e:
            print(f"Error parsing derived data line: {e}")
            print(f"Line: {line}")
            return None
    
    return data

def load_igra_data_from_text(text_data, data_type='raw'):
    """Load IGRA data from a text string."""
    lines = text_data.strip().split('\n')
    
    soundings = []
    current_sounding = None
    current_header = None
    
    for line in lines:
        if not line.strip():
            continue
            
        if line.startswith('#'):
            # This is a header line
            if current_sounding is not None:
                soundings.append(current_sounding)
            
            # Parse header
            current_header = parse_igra_header(line)
                
            if current_header is None:
                continue
                
            current_sounding = {
                'header': current_header,
                'data': []
            }
        else:
            # This is a data line
            if current_sounding is None:
                continue
                
            data_line = parse_igra_data_line(line, data_type)
            if data_line is not None:
                current_sounding['data'].append(data_line)
    
    # Don't forget to add the last sounding
    if current_sounding is not None:
        soundings.append(current_sounding)
    
    return soundings

def soundings_to_dataframe(soundings):
    """Convert a list of parsed soundings to a pandas DataFrame."""
    all_data = []
    
    for sounding in soundings:
        header = sounding['header']
        
        for level in sounding['data']:
            row = {**header, **level}
            all_data.append(row)
    
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    
    # Convert pressure from Pa to hPa if needed
    if 'pressure' in df.columns and df['pressure'].max() > 110000:
        df['pressure'] = df['pressure'] / 100
    
    return df
```

## Step 8: Loading the Data

Now let's use our functions to load the data:

```python
# Parse the raw data
raw_soundings = load_igra_data_from_text(raw_data_text, 'raw')
print(f"Loaded {len(raw_soundings)} raw soundings")

# Convert to DataFrame
raw_df = soundings_to_dataframe(raw_soundings)

# Check the first few rows
print("\nRaw data preview:")
print(raw_df[['station_id', 'datetime', 'pressure', 'temperature', 'wind_speed']].head())

# Parse the derived data
derived_soundings = load_igra_data_from_text(derived_data_text, 'derived')
print(f"\nLoaded {len(derived_soundings)} derived soundings")

# Convert to DataFrame
derived_df = soundings_to_dataframe(derived_soundings)

# Check the first few rows
print("\nDerived data preview:")
print(derived_df[['station_id', 'datetime', 'pressure', 'temperature']].head())
```

## Step 9: Basic Data Exploration

Let's explore our data with some simple analyses:

```python
# Basic statistics for raw data
print("\nSummary of raw data:")
print(f"Date range: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")
print(f"Number of pressure levels: {raw_df['pressure'].nunique()}")
print(f"Temperature range: {raw_df['temperature'].min():.1f}°C to {raw_df['temperature'].max():.1f}°C")

# Count soundings by date
date_counts = raw_df.groupby('datetime').size()
print("\nNumber of levels per sounding:")
print(date_counts)
```

## Step 10: Creating Your First Plot - Temperature Profile

Let's create a simple temperature profile for the first sounding:

```python
# Get the first date
first_date = raw_df['datetime'].iloc[0]
first_sounding = raw_df[raw_df['datetime'] == first_date]

# Sort by pressure (high to low)
first_sounding = first_sounding.sort_values('pressure', ascending=False)

# Plot the temperature profile
plt.figure(figsize=(8, 10))
plt.plot(first_sounding['temperature'], first_sounding['pressure'], 'r-o', linewidth=2)

# Invert the y-axis (since pressure decreases with height)
plt.gca().invert_yaxis()

# Set y-axis to log scale (standard for atmospheric plots)
plt.yscale('log')
plt.ylim(1050, 100)  # from 1050 hPa to 100 hPa

# Add labels and title
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')
plt.title(f'Temperature Profile - {first_date}')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Step 11: Creating a Skew-T Log-P Diagram

For more advanced meteorological analysis, let's create a Skew-T plot using MetPy:

```python
# Get data for the first sounding
first_date = raw_df['datetime'].iloc[0]
sounding_data = raw_df[raw_df['datetime'] == first_date].copy().sort_values('pressure', ascending=False)

# Remove missing data
sounding_data = sounding_data.dropna(subset=['temperature', 'pressure'])

# Convert to units for MetPy
p = sounding_data['pressure'].values * units.hPa
T = sounding_data['temperature'].values * units.degC

# Create the Skew-T plot
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot data
skew.plot(p, T, 'r')

# Add dewpoint if available
if 'dewpoint' in sounding_data.columns and not sounding_data['dewpoint'].isna().all():
    Td = sounding_data['dewpoint'].values * units.degC
    skew.plot(p, Td, 'g')

# Add wind barbs if available
if 'wind_speed' in sounding_data.columns and 'wind_direction' in sounding_data.columns:
    wind_data = sounding_data.dropna(subset=['wind_speed', 'wind_direction'])
    if not wind_data.empty:
        # Calculate u and v components
        u = -wind_data['wind_speed'] * np.sin(np.radians(wind_data['wind_direction']))
        v = -wind_data['wind_speed'] * np.cos(np.radians(wind_data['wind_direction']))
        skew.plot_barbs(wind_data['pressure'].values * units.hPa, u, v)

# Add features
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Set limits
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 50)

# Add labels
plt.title(f'Skew-T Log-P Diagram - {first_date}', fontsize=14)
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')

plt.tight_layout()
plt.show()
```

## Step 12: Time-Height Plot of Temperature

Let's create a time-height cross-section to see how temperature changes with height and time:

```python
# Create a pivot table with time on the x-axis, pressure on the y-axis, and temperature as values
pivot = raw_df.pivot_table(
    index='pressure',
    columns='datetime',
    values='temperature'
)

# Create the plot
plt.figure(figsize=(12, 8))
plt.contourf(
    pivot.columns,
    pivot.index,
    pivot.values,
    levels=np.arange(-80, 40, 5),
    cmap='RdBu_r',
    extend='both'
)

# Add contour lines
plt.contour(
    pivot.columns,
    pivot.index,
    pivot.values,
    levels=np.arange(-80, 40, 10),
    colors='k',
    linewidths=0.5,
    alpha=0.5
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Temperature (°C)')

# Set y-axis to log scale and invert
plt.yscale('log')
plt.gca().invert_yaxis()
plt.ylim(1050, 100)

# Format the date axis
plt.gcf().autofmt_xdate()

# Add labels
plt.xlabel('Date')
plt.ylabel('Pressure (hPa)')
plt.title('Temperature Time-Height Cross Section')

plt.grid(False)
plt.tight_layout()
plt.show()
```

## Step 13: Wind Profile Plot

Let's create a wind profile for the first sounding:

```python
# Get data for the first sounding
first_date = raw_df['datetime'].iloc[0]
wind_data = raw_df[raw_df['datetime'] == first_date].copy()

# Filter out rows with missing wind data
wind_data = wind_data.dropna(subset=['wind_speed', 'wind_direction'])

# Sort by pressure
wind_data = wind_data.sort_values('pressure', ascending=False)

if not wind_data.empty:
    # Calculate u and v components
    u = -wind_data['wind_speed'] * np.sin(np.radians(wind_data['wind_direction']))
    v = -wind_data['wind_speed'] * np.cos(np.radians(wind_data['wind_direction']))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    
    # Plot wind speed
    ax1.plot(wind_data['wind_speed'], wind_data['pressure'], 'b-o', linewidth=2)
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title('Wind Speed Profile')
    ax1.grid(True)
    
    # Plot wind direction
    ax2.plot(wind_data['wind_direction'], wind_data['pressure'], 'r-o', linewidth=2)
    ax2.set_xlabel('Wind Direction (degrees)')
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
    ax2.set_title('Wind Direction Profile')
    ax2.grid(True)
    
    # Invert y-axis and set to log scale
    for ax in [ax1, ax2]:
        ax.invert_yaxis()
        ax.set_yscale('log')
        ax.set_ylim(1050, 100)
    
    plt.suptitle(f'Wind Profile - {first_date}', fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("No wind data available for the first sounding")
```

## Step 14: Comparing Temperature Profiles

Let's compare temperature profiles from different dates:

```python
# Get unique dates
unique_dates = raw_df['datetime'].unique()

# Plot temperature profiles for up to 3 dates
plt.figure(figsize=(10, 8))

for i, date in enumerate(unique_dates[:3]):  # Limit to first 3 dates
    sounding = raw_df[raw_df['datetime'] == date].sort_values('pressure', ascending=False)
    plt.plot(
        sounding['temperature'], 
        sounding['pressure'],
        marker='o',
        linewidth=2,
        label=date.strftime('%Y-%m-%d %H:%M')
    )

plt.gca().invert_yaxis()
plt.yscale('log')
plt.ylim(1050, 100)
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')
plt.title('Temperature Profile Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

## Step 15: Saving Your Plots

You can save your plots to files:

```python
# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Get the first date
first_date = raw_df['datetime'].iloc[0]
date_str = first_date.strftime('%Y-%m-%d_%H')

# Example of saving a temperature profile
plt.figure(figsize=(8, 10))
sounding = raw_df[raw_df['datetime'] == first_date].sort_values('pressure', ascending=False)
plt.plot(sounding['temperature'], sounding['pressure'], 'r-o', linewidth=2)

plt.gca().invert_yaxis()
plt.yscale('log')
plt.ylim(1050, 100)
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (hPa)')
plt.title(f'Temperature Profile - {first_date}')
plt.grid(True)

# Save the figure
plt.savefig(f'plots/temperature_profile_{date_str}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved as 'plots/temperature_profile_{date_str}.png'")
```

## Step 16: Analyzing the Data

Let's do some simple analysis of the temperature and wind data:

```python
# Calculate statistics for temperature at standard pressure levels
standard_levels = [1000, 850, 700, 500, 300, 200, 100]

# Find the nearest available pressure levels
level_stats = []

for level in standard_levels:
    # Find data points close to this level
    level_data = raw_df[(raw_df['pressure'] >= level-10) & (raw_df['pressure'] <= level+10)]
    
    if not level_data.empty:
        stats = {
            'pressure_level': level,
            'mean_temp': level_data['temperature'].mean(),
            'min_temp': level_data['temperature'].min(),
            'max_temp': level_data['temperature'].max(),
            'temp_range': level_data['temperature'].max() - level_data['temperature'].min()
        }
        level_stats.append(stats)

# Create a DataFrame with the statistics
stats_df = pd.DataFrame(level_stats)
print("Temperature statistics by pressure level:")
print(stats_df.round(1))

# Plot the temperature range by level
plt.figure(figsize=(10, 6))
plt.bar(stats_df['pressure_level'].astype(str), stats_df['temp_range'])
plt.xlabel('Pressure Level (hPa)')
plt.ylabel('Temperature Range (°C)')
plt.title('Temperature Variability by Pressure Level')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
```

## Step 17: Export Data to CSV

You can save your processed data to CSV files for future use or for sharing:

```python
# Create a directory for exported data if it doesn't exist
if not os.path.exists('exported_data'):
    os.makedirs('exported_data')

# Export the raw data DataFrame to CSV
raw_df.to_csv('exported_data/processed_raw_data.csv', index=False)

# Export the derived data DataFrame to CSV
derived_df.to_csv('exported_data/processed_derived_data.csv', index=False)

print("Data exported to CSV files in the 'exported_data' folder")
```

## Step 18: Extract Data for a Specific Time

If you want to analyze a specific sounding in detail:

```python
# Get all soundings for February 7, 1950 at 15Z
specific_date = datetime(1950, 2, 7, 15)
specific_sounding = raw_df[raw_df['datetime'] == specific_date]

if not specific_sounding.empty:
    # Sort by pressure (high to low)
    specific_sounding = specific_sounding.sort_values('pressure', ascending=False)
    
    print(f"Sounding data for {specific_date}:")
    print(specific_sounding[['pressure', 'temperature', 'wind_speed', 'wind_direction']].head(10))
    
    # Create a temperature and dewpoint profile
    plt.figure(figsize=(8, 10))
    
    plt.plot(specific_sounding['temperature'], specific_sounding['pressure'], 'r-o', label='Temperature')
    
    if 'dewpoint' in specific_sounding.columns:
        plt.plot(specific_sounding['dewpoint'], specific_sounding['pressure'], 'g-o', label='Dewpoint')
    
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.ylim(1050, 100)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Temperature and Dewpoint Profile - {specific_date}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"No data available for {specific_date}")
```

## Congratulations!

You've successfully:
1. Set up a project for IGRA data analysis
2. Parsed and loaded IGRA data into pandas DataFrames
3. Created various visualizations of atmospheric data
4. Performed basic data analysis
5. Exported your processed data

## Next Steps

Now that you understand the basics, you can:
1. Analyze other IGRA stations by downloading their data files
2. Create more advanced visualizations
3. Calculate meteorological parameters like CAPE, CIN, etc.
4. Compare data from different stations or time periods
5. Analyze seasonal trends in the atmospheric data

## Additional Resources

- IGRA Data Access: https://www.ncei.noaa.gov/products/integrated-global-radiosonde-archive
- MetPy Documentation: https://unidata.github.io/MetPy/latest/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Documentation: https://matplotlib.org/stable/users/index.html
