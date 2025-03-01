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
with open("/home/tdieckman/Igra-Data-Analysis/USM0007479f-data-exampleportion.txt", "r") as f:
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


Let me complete the `run_igra_analysis.py` script:

Now, let me provide you with a step-by-step guide on how to run these scripts correctly in WSL with VS Code.

## Running the IGRA Analysis Scripts in VS Code with WSL

### Step 1: Set Up Your Environment

First, make sure you have the necessary Python packages installed:

```bash
pip3 install pandas numpy matplotlib xarray seaborn
```

### Step 2: Organize Your Files

1. Save all the Python scripts to your working directory:
   - `igra_loader.py` - The core data loading module
   - `igra_debug.py` - The debugging utility
   - `complete_example.py` - The main analysis script
   - `run_igra_analysis.py` - The command-line runner

2. Make the runner script executable:
   ```bash
   chmod +x run_igra_analysis.py
   ```

3. Create an output directory (optional, scripts will create it automatically):
   ```bash
   mkdir -p output
   ```

### Step 3: Testing the Installation

Let's first test if the `igra_loader.py` module works correctly:

```bash
cd /home/tdieckman/Igra-Data-Analysis/VanillaPython/
python3 -c "import igra_loader; print('igra_loader module imported successfully')"
```

If that works, you're ready to start using the scripts.

### Step 4: Running the Debug Utility

The debug utility is useful to examine your data files and diagnose any issues:

```bash
python3 igra_debug.py data/USM0007479f-data-exampleportion.txt raw
```

This will analyze the raw data file and provide detailed information about its structure, data quality, and compatibility with the plotting functions.

You can also run it on the derived data:

```bash
python3 igra_debug.py data/USM0007479f-drvd-exampleportion.txt derived
```

### Step 5: Running the Complete Analysis

Now you can run the full analysis script:

```bash
python3 complete_example.py
```

This will:
1. Load both raw and derived data files
2. Convert them to pandas DataFrames and xarray Datasets
3. Create several visualization plots
4. Save the processed data to CSV files
5. Attempt to save the data to NetCDF format (if successful)

All outputs will be saved to the `output` directory.

### Step 6: Using the Command-Line Runner (Alternative)

For a more flexible approach, you can use the command-line runner:

```bash
./run_igra_analysis.py analyze --raw data/USM0007479f-data-exampleportion.txt --derived data/USM0007479f-drvd-exampleportion.txt --output output
```

Or to debug a specific file:

```bash
./run_igra_analysis.py debug data/USM0007479f-data-exampleportion.txt --type raw
```

## Troubleshooting Common Issues

1. **Module not found errors**: Make sure all Python files are in the same directory or in a location Python can find.

2. **File not found errors**: Double-check your file paths. The scripts look for data in `/home/tdieckman/Igra-Data-Analysis/VanillaPython/data/` by default.

3. **Plotting errors**: The scripts are designed to handle missing data, but if you get errors related to plotting, it might be because there's not enough valid data for visualization. The debug utility can help identify these issues.

4. **Empty or incorrect output**: Check the console output for warnings or errors. The scripts will attempt to continue even if some operations fail, but they'll print messages explaining the issues.

Let me know if you encounter any specific errors, and I can provide more targeted troubleshooting advice.