# Step 5: Importing Libraries
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

# Step 6: Reading the Data Files
# Read the data files
with open('/home/tdieckman/Igra-Data-Analysis/ZOLDSTUFF/data/raw_data.txt', 'r') as file:
    raw_data_text = file.read()
    
with open('/home/tdieckman/Igra-Data-Analysis/ZOLDSTUFF/data/derived_data.txt', 'r') as file:
    derived_data_text = file.read()

print(f"Raw data: {len(raw_data_text)} characters")
print(f"Derived data: {len(derived_data_text)} characters")

# Step 7: Parsing the IGRA Data - Helper Functions
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