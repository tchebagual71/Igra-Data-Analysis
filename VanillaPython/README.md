# IGRA Data Analysis Toolkit

This toolkit provides a comprehensive set of functions to load, process, and visualize Integrated Global Radiosonde Archive (IGRA) data. It handles both raw sounding data and derived parameter data from the IGRA v2.2 dataset.

## Features

- Parse and load IGRA raw sounding data and derived parameter data
- Convert data to pandas DataFrames and xarray Datasets
- Create various visualizations including:
  - Skew-T log-P diagrams
  - Time-height cross sections
  - Wind profiles and hodographs
  - Stability parameter time series
  - Comprehensive sounding dashboards

## Prerequisites

```bash
pip install pandas numpy matplotlib xarray scipy metpy
```

For map visualizations (optional):
```bash
pip install cartopy
```

## Files in this Toolkit

1. **igra_loader.py** - Core functions for loading and parsing IGRA data
2. **igra_usage_example.py** - Examples of how to use the loading functions
3. **advanced_visualizations.py** - Functions for creating advanced meteorological visualizations
4. **complete_example.py** - Complete workflow for analyzing IGRA data

## Quick Start

```python
import pandas as pd
from igra_loader import load_igra_data_from_text, soundings_to_dataframe, plot_skewt

# Load data from file
with open("USM00074794-data.txt", "r") as f:
    raw_data = f.read()

# Parse the data
soundings = load_igra_data_from_text(raw_data, 'raw')

# Convert to DataFrame
df = soundings_to_dataframe(soundings)

# Create a Skew-T plot for the first sounding
fig, skew = plot_skewt(soundings[0], "Sample Sounding")
fig.savefig("skewt.png")
```

## Understanding IGRA Data Format

IGRA v2.2 data follows a specific format:

1. **Raw Sounding Data** (station-data.txt):
   - Each sounding begins with a header line starting with `#`
   - Header contains station ID, date, time, and number of levels
   - Following lines contain atmospheric measurements at each level
   - Variables include pressure, height, temperature, humidity, wind

2. **Derived Parameter Data** (station-drvd.txt):
   - Similar structure with header followed by level data
   - Header contains additional parameters like CAPE, CIN, LI
   - Level data includes gradients and calculated parameters

## Data Loading Functions

### Basic Loading

```python
# Load from text string
soundings = load_igra_data_from_text(text_data, data_type='raw')

# Load from file
soundings = load_igra_file(file_path, data_type='raw')

# Convert to DataFrame
df = soundings_to_dataframe(soundings)

# Convert to xarray
ds = soundings_to_xarray(soundings)
```

### Working with Multiple Stations

```python
import glob
import os

# Find all station files in a directory
raw_files = glob.glob(os.path.join(data_dir, "*-data.txt"))

# Process each station
for file_path in raw_files:
    station_id = os.path.basename(file_path).split('-')[0]
    soundings = load_igra_file(file_path, 'raw')
    df = soundings_to_dataframe(soundings)
    # Process data...
```

## Visualization Examples

### Basic Skew-T Plot

```python
from igra_loader import plot_skewt

# Create a Skew-T plot
fig, skew = plot_skewt(soundings[0])
fig.savefig("skewt.png")
```

### Time-Height Cross Sections

```python
from advanced_visualizations import plot_time_height_cross_section

# Create a time-height cross section of temperature
fig = plot_time_height_cross_section(df, variable='temperature')
fig.savefig("temp_cross_section.png")
```

### Wind Profiles

```python
from advanced_visualizations import plot_wind_cross_section

# Create a time-height cross section of wind
fig = plot_wind_cross_section(df)
fig.savefig("wind_cross_section.png")
```

### Comprehensive Dashboard

```python
from advanced_visualizations import create_custom_soundings_dashboard

# Create a comprehensive dashboard for a specific date
date = df['datetime'].iloc[0]
fig = create_custom_soundings_dashboard(df, date)
fig.savefig("dashboard.png")
```

### Complete Analysis Report

```python
from advanced_visualizations import create_igra_analysis_report

# Create a complete analysis report
results = create_igra_analysis_report(raw_df, derived_df, output_dir='output', station_id='USM00074794')
```

## Working with xarray for Advanced Analysis

```python
import xarray as xr

# Select a specific date
date = '1950-02-05'
temp_profile = ds.sel(datetime=date)['temperature']

# Calculate statistics
monthly_means = ds.groupby('datetime.month').mean()

# Plot using xarray's built-in plotting
ds['temperature'].plot(x='datetime', y='pressure')
```

## Tips for IGRA Data Analysis

1. **Handle Missing Values** - IGRA uses special values (-9999, -8888) for missing data
2. **Pressure Units** - Pressure may be in Pa or hPa depending on the source; check max values
3. **Wind Calculations** - Convert wind direction and speed to U/V components for analysis
4. **Interpolation** - Use interpolation to create regular grids for time-height sections
5. **Stability Analysis** - Derived files contain valuable stability parameters

## Additional Resources

- [IGRA v2.2 Documentation](https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt)
- [MetPy Documentation](https://unidata.github.io/MetPy/latest/)
- [xarray Documentation](https://docs.xarray.dev/en/stable/)
