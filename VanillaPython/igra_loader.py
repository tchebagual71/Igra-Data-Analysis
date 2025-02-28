import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import io
import os
import glob
from datetime import datetime

# Helper functions for data parsing
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
        
        # Handle latitude and longitude if present (may be in different positions)
        if len(header_line) > 70:
            try:
                lat_str = header_line[55:62].strip()
                lon_str = header_line[63:71].strip()
                header['latitude'] = int(lat_str) / 10000 if lat_str and lat_str != '-99999' else np.nan
                header['longitude'] = int(lon_str) / 10000 if lon_str and lon_str != '-99999' else np.nan
            except (ValueError, IndexError):
                header['latitude'] = np.nan
                header['longitude'] = np.nan
        else:
            header['latitude'] = np.nan
            header['longitude'] = np.nan
            
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
    """Parse a data line from IGRA raw sounding data.
    format_type can be 'raw' or 'derived'"""
    data = {}
    
    if format_type == 'raw':
        # Raw sounding data
        try:
            data['lvltyp1'] = int(line[0:1])
            data['lvltyp2'] = int(line[1:2])
            data['etime'] = line[3:8].strip()
            
            press_str = line[9:15].strip()
            data['pressure'] = int(press_str) if press_str and press_str != '-9999' else np.nan
            data['pflag'] = line[15:16]
            
            gph_str = line[16:21].strip()
            data['height'] = int(gph_str) if gph_str and gph_str not in ['-8888', '-9999'] else np.nan
            data['zflag'] = line[21:22]
            
            temp_str = line[22:27].strip()
            data['temperature'] = int(temp_str) / 10 if temp_str and temp_str not in ['-8888', '-9999'] else np.nan
            data['tflag'] = line[27:28]
            
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
            
    elif format_type == 'derived':
        # Derived parameters data
        try:
            press_str = line[:7].strip()
            data['pressure'] = int(press_str) if press_str and press_str != '-99999' else np.nan
            
            repgph_str = line[8:15].strip()
            data['reported_height'] = int(repgph_str) if repgph_str and repgph_str != '-99999' else np.nan
            
            calcgph_str = line[16:23].strip()
            data['calculated_height'] = int(calcgph_str) if calcgph_str and calcgph_str != '-99999' else np.nan
            
            temp_str = line[24:31].strip()
            data['temperature'] = int(temp_str) / 10 if temp_str and temp_str != '-99999' else np.nan
            
            tempgrad_str = line[32:39].strip()
            data['temp_gradient'] = int(tempgrad_str) / 10 if tempgrad_str and tempgrad_str != '-99999' else np.nan
            
            ptemp_str = line[40:47].strip()
            data['potential_temp'] = int(ptemp_str) / 10 if ptemp_str and ptemp_str != '-99999' else np.nan
            
            vappress_str = line[72:79].strip()
            data['vapor_pressure'] = int(vappress_str) / 1000 if vappress_str and vappress_str != '-99999' else np.nan
            
            rh_str = line[96:103].strip()
            data['relative_humidity'] = int(rh_str) / 10 if rh_str and rh_str != '-99999' else np.nan
            
            uwnd_str = line[112:119].strip()
            data['u_wind'] = int(uwnd_str) / 10 if uwnd_str and uwnd_str != '-99999' else np.nan
            
            vwnd_str = line[128:135].strip()
            data['v_wind'] = int(vwnd_str) / 10 if vwnd_str and vwnd_str != '-99999' else np.nan
            
            n_str = line[144:151].strip()
            data['refractive_index'] = int(n_str) if n_str and n_str != '-99999' else np.nan
            
        except Exception as e:
            print(f"Error parsing derived data line: {e}")
            print(f"Line: {line}")
            return None
    
    return data

def parse_derived_header(header_line):
    """Parse the header line of IGRA derived sounding data."""
    header = {}
    try:
        header['station_id'] = header_line[1:12].strip()
        header['year'] = int(header_line[13:17])
        header['month'] = int(header_line[18:20])
        header['day'] = int(header_line[21:23])
        header['hour'] = int(header_line[24:26])
        header['reltime'] = header_line[27:31].strip()
        header['num_levels'] = int(header_line[31:36])
        
        # Parse derived parameters
        pw_str = header_line[37:43].strip()
        header['precipitable_water'] = int(pw_str) / 100 if pw_str and pw_str != '-99999' else np.nan
        
        invpress_str = header_line[43:49].strip()
        header['inversion_pressure'] = int(invpress_str) if invpress_str and invpress_str != '-99999' else np.nan
        
        invhgt_str = header_line[49:55].strip()
        header['inversion_height'] = int(invhgt_str) if invhgt_str and invhgt_str != '-99999' else np.nan
        
        li_str = header_line[121:127].strip()
        header['lifted_index'] = int(li_str) if li_str and li_str != '-99999' else np.nan
        
        cape_str = header_line[145:151].strip()
        header['cape'] = int(cape_str) if cape_str and cape_str != '-99999' else np.nan
        
        cin_str = header_line[151:157].strip()
        header['cin'] = int(cin_str) if cin_str and cin_str != '-99999' else np.nan
        
        # Try to parse datetime
        try:
            header['datetime'] = datetime(
                header['year'], header['month'], header['day'], 
                hour=0 if header['hour'] == 99 else header['hour']
            )
        except ValueError:
            header['datetime'] = None
            
    except Exception as e:
        print(f"Error parsing derived header: {e}")
        print(f"Header line: {header_line}")
        return None
    
    return header

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
            
            # Parse header based on data type
            if data_type == 'raw':
                current_header = parse_igra_header(line)
            else:  # derived
                current_header = parse_derived_header(line)
                
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
    
    # Convert pressure from Pa to hPa (if needed)
    if 'pressure' in df.columns and df['pressure'].max() > 110000:
        df['pressure'] = df['pressure'] / 100
    
    return df

def soundings_to_xarray(soundings):
    """Convert a list of parsed soundings to an xarray Dataset."""
    # First convert to DataFrame
    df = soundings_to_dataframe(soundings)
    
    if df.empty:
        return None
    
    # Create a MultiIndex for pressure levels and time
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set up dimensions for xarray
    # We'll use datetime and pressure as dimensions
    ds = df.set_index(['datetime', 'pressure']).to_xarray()
    
    return ds

# Function to load and parse data from file
def load_igra_file(file_path, data_type='raw'):
    """Load IGRA data from a file."""
    with open(file_path, 'r') as f:
        text_data = f.read()
    
    return load_igra_data_from_text(text_data, data_type)

# Function to visualize a sounding as a Skew-T plot using MetPy
def plot_skewt(sounding_data, title=None):
    """Plot a sounding as a Skew-T log-P diagram using MetPy."""
    # Extract data from sounding
    pressure = []
    temperature = []
    dewpoint = []
    
    for level in sounding_data['data']:
        if 'pressure' in level and not np.isnan(level['pressure']):
            p = level['pressure']
            # Convert from Pa to hPa if needed
            if p > 110000:
                p = p / 100
            pressure.append(p)
            
            t = level.get('temperature', np.nan)
            temperature.append(t)
            
            # Get dewpoint either directly or calculate from temp and depression
            if 'dewpoint' in level and not np.isnan(level['dewpoint']):
                dewpoint.append(level['dewpoint'])
            elif 'dewpoint_depression' in level and not np.isnan(level['dewpoint_depression']) and not np.isnan(t):
                dewpoint.append(t - level['dewpoint_depression'])
            else:
                dewpoint.append(np.nan)
    
    # Skip if no valid data
    if not pressure:
        print("No valid pressure data found for Skew-T plot")
        return None
    
    # Convert lists to numpy arrays
    pressure = np.array(pressure) * units.hPa
    temperature = np.array(temperature) * units.degC
    dewpoint = np.array(dewpoint) * units.degC
    
    # Filter out NaN values
    mask = ~(np.isnan(pressure) | np.isnan(temperature))
    dew_mask = ~np.isnan(dewpoint)
    
    if not np.any(mask) or not np.any(dew_mask):
        print("No valid data for Skew-T plot after filtering NaNs")
        return None
    
    # Create figure and Skew-T plot
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45)
    
    # Plot data
    if np.any(mask):
        skew.plot(pressure[mask], temperature[mask], 'r')
    if np.any(dew_mask):
        skew.plot(pressure[dew_mask], dewpoint[dew_mask], 'g')
    
    # Add features
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    
    # Add title
    header = sounding_data['header']
    station_id = header['station_id']
    date_str = f"{header['year']}-{header['month']:02d}-{header['day']:02d} {header['hour']:02d}Z"
    
    if title:
        plt.title(f"{title}\n{station_id} - {date_str}")
    else:
        plt.title(f"Skew-T Log-P Diagram\n{station_id} - {date_str}")
    
    # Set limits and labels
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    
    return fig, skew

# Example usage function
def demo_with_sample_data(raw_text_data, derived_text_data=None):
    """Demo the functions with sample data."""
    # Load the raw data
    print("Loading raw sounding data...")
    raw_soundings = load_igra_data_from_text(raw_text_data, 'raw')
    print(f"Found {len(raw_soundings)} raw soundings")
    
    # Convert to DataFrame
    raw_df = soundings_to_dataframe(raw_soundings)
    print("\nRaw data DataFrame preview:")
    print(raw_df[['station_id', 'datetime', 'pressure', 'temperature', 'wind_speed']].head())
    
    # Convert to xarray
    raw_ds = soundings_to_xarray(raw_soundings)
    print("\nRaw data xarray Dataset info:")
    print(raw_ds)
    
    # Plot the first sounding
    if raw_soundings:
        print("\nCreating Skew-T plot for the first sounding...")
        fig, skew = plot_skewt(raw_soundings[0], "Raw Sounding Data")
        plt.tight_layout()
        print("Skew-T plot created. Display or save as needed.")
    
    # If derived data is provided, process it too
    if derived_text_data:
        print("\nLoading derived sounding data...")
        derived_soundings = load_igra_data_from_text(derived_text_data, 'derived')
        print(f"Found {len(derived_soundings)} derived soundings")
        
        # Convert to DataFrame
        derived_df = soundings_to_dataframe(derived_soundings)
        print("\nDerived data DataFrame preview:")
        if not derived_df.empty:
            print(derived_df[['station_id', 'datetime', 'pressure', 'temperature', 'potential_temp']].head())
        
        # Convert to xarray
        derived_ds = soundings_to_xarray(derived_soundings)
        print("\nDerived data xarray Dataset info:")
        print(derived_ds)
    
    return {
        'raw_soundings': raw_soundings,
        'raw_df': raw_df,
        'raw_ds': raw_ds,
        'derived_soundings': derived_soundings if derived_text_data else None,
        'derived_df': derived_df if derived_text_data else None,
        'derived_ds': derived_ds if derived_text_data else None
    }
