import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import io

def safe_float(value, missing_values=('-99999', '-9999', '-8888', '')):
    """
    Convert a string to float, handling missing values.
    
    Parameters:
    -----------
    value : str
        String to convert
    missing_values : tuple
        Values to treat as missing
        
    Returns:
    --------
    float or np.nan
        Converted value or np.nan if missing
    """
    if value in missing_values:
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def parse_igra_header(header_line):
    """
    Parse a header line from a standard IGRA data file.
    
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
        'station_id': header_line[1:12].strip(),
        'year': int(header_line[13:17].strip()),
        'month': int(header_line[18:20].strip()),
        'day': int(header_line[21:23].strip()),
        'hour': int(header_line[24:26].strip()),
        'reltime': header_line[27:31].strip(),
        'num_levels': int(header_line[31:36].strip()),
    }
    
    # Add latitude and longitude if available in the header
    if len(header_line) > 55:
        try:
            header['latitude'] = float(header_line[55:62].strip()) / 1000.0
            header['longitude'] = float(header_line[63:71].strip()) / 1000.0
        except (ValueError, IndexError):
            header['latitude'] = np.nan
            header['longitude'] = np.nan
    
    # Add a datetime field for easier plotting
    try:
        if header['hour'] == 99:  # Missing hour
            header['datetime'] = datetime(header['year'], header['month'], header['day'])
        else:
            header['datetime'] = datetime(header['year'], header['month'], header['day'], header['hour'])
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
    # Ensure the line is long enough
    if len(data_line) < 46:  # Minimum length to get pressure and temp
        return {
            'lvltyp1': np.nan,
            'lvltyp2': np.nan,
            'pressure': np.nan,
            'temperature': np.nan
        }
    
    data = {
        'lvltyp1': int(data_line[0:1].strip() or 0),
        'lvltyp2': int(data_line[1:2].strip() or 0),
        'etime': safe_float(data_line[3:8].strip()),
        'pressure': safe_float(data_line[9:15].strip()),
        'pflag': data_line[15:16].strip(),
        'height': safe_float(data_line[16:21].strip()),
        'zflag': data_line[21:22].strip(),
        'temperature': np.nan,  # Initialize with nan, will update below
        'tflag': data_line[27:28].strip() if len(data_line) > 27 else '',
    }
    
    # Parse temperature correctly - it's in tenths of degrees C
    try:
        temp_str = data_line[22:27].strip()
        if temp_str in ('-9999', '-8888', ''):
            data['temperature'] = np.nan
        else:
            data['temperature'] = float(temp_str) / 10.0
    except (ValueError, IndexError):
        data['temperature'] = np.nan
    
    # Add relative humidity if available
    if len(data_line) > 28:
        try:
            rh_str = data_line[28:33].strip()
            data['relative_humidity'] = float(rh_str) / 10.0 if rh_str and rh_str not in ('-9999', '-8888') else np.nan
        except (ValueError, IndexError):
            data['relative_humidity'] = np.nan
    
    # Add dewpoint depression if available
    if len(data_line) > 34:
        try:
            dpdp_str = data_line[34:39].strip()
            data['dewpoint_depression'] = float(dpdp_str) / 10.0 if dpdp_str and dpdp_str not in ('-9999', '-8888') else np.nan
            
            # Calculate dewpoint if both temperature and dewpoint depression are available
            if not np.isnan(data['temperature']) and not np.isnan(data['dewpoint_depression']):
                data['dewpoint'] = data['temperature'] - data['dewpoint_depression']
            else:
                data['dewpoint'] = np.nan
        except (ValueError, IndexError):
            data['dewpoint_depression'] = np.nan
            data['dewpoint'] = np.nan
    
    # Add wind data if available
    if len(data_line) > 40:
        try:
            wdir_str = data_line[40:45].strip()
            data['wind_direction'] = float(wdir_str) if wdir_str and wdir_str not in ('-9999', '-8888') else np.nan
        except (ValueError, IndexError):
            data['wind_direction'] = np.nan
    
    if len(data_line) > 46:
        try:
            wspd_str = data_line[46:51].strip()
            data['wind_speed'] = float(wspd_str) / 10.0 if wspd_str and wspd_str not in ('-9999', '-8888') else np.nan
        except (ValueError, IndexError):
            data['wind_speed'] = np.nan
    
    # Convert pressure from Pa to hPa (mb)
    if 'pressure' in data and data['pressure'] is not None and not np.isnan(data['pressure']):
        data['pressure'] = data['pressure'] / 100.0
    
    return data

def parse_igra_derived_header(header_line):
    """
    Parse a header line from IGRA derived parameters file.
    
    Parameters:
    -----------
    header_line : str
        A header line from an IGRA derived parameters file starting with '#'
        
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
        'station_id': header_line[1:12].strip(),
        'year': int(header_line[13:17].strip()),
        'month': int(header_line[18:20].strip()),
        'day': int(header_line[21:23].strip()),
        'hour': int(header_line[24:26].strip()),
        'reltime': header_line[27:31].strip(),
        'num_levels': int(header_line[31:36].strip()),
    }
    
    # Extract derived parameters if available
    if len(header_line) > 37:
        header['pw'] = safe_float(header_line[37:43].strip())  # Precipitable water
    
    if len(header_line) > 43:
        header['invpress'] = safe_float(header_line[43:49].strip())  # Inversion pressure
    
    if len(header_line) > 49:
        header['invhgt'] = safe_float(header_line[49:55].strip())  # Inversion height
    
    if len(header_line) > 55:
        header['invtempdif'] = safe_float(header_line[55:61].strip())  # Inversion temp difference
    
    # Add more derived parameters if needed based on the format specification
    
    # Add a datetime field for easier plotting
    try:
        if header['hour'] == 99:  # Missing hour
            header['datetime'] = datetime(header['year'], header['month'], header['day'])
        else:
            header['datetime'] = datetime(header['year'], header['month'], header['day'], header['hour'])
    except ValueError:
        # Handle potential invalid dates
        header['datetime'] = None
    
    return header

def parse_igra_derived_data_line(data_line):
    """
    Parse a data line from IGRA derived parameters file.
    
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
        'pressure': safe_float(data_line[0:7].strip()),
        'repgph': safe_float(data_line[8:15].strip()),
        'calcgph': safe_float(data_line[16:23].strip()),
        'temp': safe_float(data_line[24:31].strip()),
        'tempgrad': safe_float(data_line[32:39].strip()),
        'ptemp': safe_float(data_line[40:47].strip()),
        'ptempgrad': safe_float(data_line[48:55].strip()),
        'vtemp': safe_float(data_line[56:63].strip()),
        'vptemp': safe_float(data_line[64:71].strip()),
        'vappress': safe_float(data_line[72:79].strip()),
        'satvap': safe_float(data_line[80:87].strip()),
        'reprh': safe_float(data_line[88:95].strip()),
        'calcrh': safe_float(data_line[96:103].strip()),
        'rhgrad': safe_float(data_line[104:111].strip()),
    }
    
    # Add wind components if available
    if len(data_line) > 111:
        data['uwnd'] = safe_float(data_line[112:119].strip())
        
    if len(data_line) > 119:
        data['uwndgrad'] = safe_float(data_line[120:127].strip())
        
    if len(data_line) > 127:
        data['vwnd'] = safe_float(data_line[128:135].strip())
        
    if len(data_line) > 135:
        data['vwndgrad'] = safe_float(data_line[136:143].strip())
        
    if len(data_line) > 143:
        data['n'] = safe_float(data_line[144:151].strip())
    
    # Convert values to proper units
    
    # Pressure from Pa to hPa
    if not np.isnan(data['pressure']):
        data['pressure'] = data['pressure'] / 100.0
    
    # Temperature from K*10 to K
    for key in ['temp', 'ptemp', 'vtemp', 'vptemp']:
        if key in data and not np.isnan(data[key]):
            data[key] = data[key] / 10.0
    
    # Relative humidity from %*10 to %
    for key in ['reprh', 'calcrh']:
        if key in data and not np.isnan(data[key]):
            data[key] = data[key] / 10.0
    
    # Wind components from (m/s)*10 to m/s
    for key in ['uwnd', 'vwnd']:
        if key in data and not np.isnan(data[key]):
            data[key] = data[key] / 10.0
    
    return data

def load_igra_data_from_text(data_text, data_type='raw'):
    """
    Load IGRA data from a text string.
    
    Parameters:
    -----------
    data_text : str
        Text content of an IGRA data file
    data_type : str
        Type of data: 'raw' or 'derived'
        
    Returns:
    --------
    list
        List of soundings
    """
    soundings = []
    current_sounding = None
    
    for line in data_text.strip().split('\n'):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Start a new sounding if we encounter a header line
        if line.startswith('#'):
            if current_sounding is not None:
                soundings.append(current_sounding)
            
            # Parse header based on data type
            if data_type == 'raw':
                header = parse_igra_header(line)
            else:  # derived
                header = parse_igra_derived_header(line)
            
            current_sounding = {
                'header': header,
                'data': []
            }
        # Otherwise, this is a data line for the current sounding
        elif current_sounding is not None:
            # Parse data based on data type
            if data_type == 'raw':
                data = parse_igra_data_line(line)
            else:  # derived
                data = parse_igra_derived_data_line(line)
            
            current_sounding['data'].append(data)
    
    # Don't forget to add the last sounding
    if current_sounding is not None:
        soundings.append(current_sounding)
    
    return soundings

def soundings_to_dataframe(soundings):
    """
    Convert a list of soundings to a pandas DataFrame.
    
    Parameters:
    -----------
    soundings : list
        List of soundings as returned by load_igra_data_from_text
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all sounding data
    """
    all_rows = []
    
    for sounding in soundings:
        header = sounding['header']
        
        for data_point in sounding['data']:
            # Combine header and data point
            row = {**header, **data_point}
            all_rows.append(row)
    
    # Convert to DataFrame
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    
    # Sort by datetime and pressure
    if 'datetime' in df.columns and 'pressure' in df.columns:
        df = df.sort_values(['datetime', 'pressure'], ascending=[True, False])
    
    return df

def soundings_to_xarray(soundings):
    """
    Convert a list of soundings to an xarray Dataset.
    
    Parameters:
    -----------
    soundings : list
        List of soundings as returned by load_igra_data_from_text
        
    Returns:
    --------
    xarray.Dataset or None
        Dataset containing all sounding data, or None if conversion fails
    """
    # First convert to DataFrame
    df = soundings_to_dataframe(soundings)
    
    if df.empty:
        return None
    
    try:
        # Get unique datetimes and pressure levels
        datetimes = df['datetime'].unique()
        pressures = sorted(df['pressure'].unique(), reverse=True)  # High to low
        
        # Determine variables to include in the dataset
        # Exclude metadata and index columns
        exclude_cols = ['datetime', 'pressure', 'station_id', 'year', 'month', 'day', 'hour', 
                        'reltime', 'num_levels', 'latitude', 'longitude']
        var_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create coordinates
        coords = {
            'datetime': datetimes,
            'pressure': pressures
        }
        
        # Create data variables
        data_vars = {}
        
        # Loop through each variable
        for var in var_cols:
            # Create a 2D array filled with NaNs
            data = np.full((len(datetimes), len(pressures)), np.nan)
            
            # Fill in the data
            for i, dt in enumerate(datetimes):
                for j, p in enumerate(pressures):
                    # Get value at this datetime and pressure level
                    val = df[(df['datetime'] == dt) & (df['pressure'] == p)][var].values
                    if len(val) > 0 and not pd.isna(val[0]):
                        data[i, j] = val[0]
            
            # Add to data variables
            data_vars[var] = (['datetime', 'pressure'], data)
        
        # Add station metadata
        station_id = df['station_id'].iloc[0]
        latitude = df['latitude'].iloc[0]
        longitude = df['longitude'].iloc[0]
        
        # Create the dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'station_id': station_id,
                'latitude': latitude,
                'longitude': longitude
            }
        )
        
        return ds
    
    except Exception as e:
        print(f"Error converting to xarray: {str(e)}")
        return None

def plot_skewt(sounding, title=None):
    """
    Create a simplified Skew-T log-P diagram for a single sounding.
    
    Parameters:
    -----------
    sounding : dict
        Dictionary containing a single sounding's data and header
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    tuple or None
        (fig, ax) if successful, None if insufficient data
    """
    # Extract data from sounding
    if 'data' not in sounding or len(sounding['data']) == 0:
        print("No data in sounding")
        return None
    
    # Extract pressure, temperature, and dewpoint data
    pressure = []
    temperature = []
    dewpoint = []
    wind_speed = []
    wind_dir = []
    
    for level in sounding['data']:
        p = level.get('pressure', np.nan)
        t = level.get('temperature', np.nan)
        td = level.get('dewpoint', np.nan)
        ws = level.get('wind_speed', np.nan)
        wd = level.get('wind_direction', np.nan)
        
        # Ensure values are floats and not strings or None
        try:
            p = float(p)
            t = float(t)
            if td is not None and not np.isnan(td):
                td = float(td)
            if ws is not None and not np.isnan(ws):
                ws = float(ws)
            if wd is not None and not np.isnan(wd):
                wd = float(wd)
        except (ValueError, TypeError):
            continue
        
        # Only add levels with valid pressure and temperature
        if not np.isnan(p) and not np.isnan(t):
            pressure.append(p)
            temperature.append(t)
            dewpoint.append(td)
            wind_speed.append(ws)
            wind_dir.append(wd)
    
    # Check if we have enough data to plot
    if len(pressure) < 3 or len(temperature) < 3:
        print("No valid data for Skew-T plot after filtering NaNs")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create skew factor (simplified approach)
    skew = 30  # Degrees to skew the temperature lines
    
    # Calculate x-coordinates with skew
    heights = np.log(1000/np.array(pressure)) * skew
    x_temp = np.array(temperature) + heights
    
    # Plot temperature profile
    ax.plot(x_temp, pressure, 'r-o', linewidth=2, markersize=4, label='Temperature')
    
    # Plot dewpoint profile if available
    valid_dewpoints = [dp for dp in dewpoint if dp is not None and not np.isnan(dp)]
    if len(valid_dewpoints) >= 3:
        # Filter to only include levels with valid dewpoints
        valid_indices = [i for i, dp in enumerate(dewpoint) if dp is not None and not np.isnan(dp)]
        x_dewp = np.array([temperature[i] - (temperature[i] - dewpoint[i]) for i in valid_indices]) + np.array([heights[i] for i in valid_indices])
        ax.plot(x_dewp, [pressure[i] for i in valid_indices], 'g-o', linewidth=2, markersize=4, label='Dewpoint')
    
    # Add wind barbs if we have wind data
    valid_wind = [i for i, (ws, wd) in enumerate(zip(wind_speed, wind_dir)) 
                 if ws is not None and wd is not None and not np.isnan(ws) and not np.isnan(wd)]
    
    if len(valid_wind) >= 3:
        # Only plot wind barbs for every other level to avoid crowding
        for i in valid_wind[::2]:
            # Convert to u, v components
            ws = wind_speed[i]
            wd = wind_dir[i]
            u = -ws * np.sin(np.radians(wd))
            v = -ws * np.cos(np.radians(wd))
            
            # Plot wind barb at far right of diagram
            ax.barbs(x_temp.max() + 5, pressure[i], u, v, length=6, pivot='middle')
    
    # Set up the axes
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # Set y-axis limits and ticks
    p_min = min(min(pressure), 100)  # Don't go below 100 hPa
    p_max = max(max(pressure), 1050)  # Don't go above 1050 hPa
    
    ax.set_ylim(p_max, p_min)
    
    # Add temperature grid lines (simplified)
    for temp in range(-80, 41, 10):
        x = np.linspace(temp, temp + heights.max(), 100)
        y = np.logspace(np.log10(p_min), np.log10(p_max), 100)
        ax.plot(x, y, 'k-', alpha=0.2, linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (hPa)')
    
    if title:
        ax.set_title(title)
    else:
        date_str = sounding['header']['datetime'].strftime('%Y-%m-%d %H:%M') if sounding['header']['datetime'] else 'Unknown Date'
        ax.set_title(f'Skew-T Log-P Diagram - {date_str}')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig, ax

def load_igra_data_from_file(file_path, data_type='raw'):
    """
    Load IGRA data from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the IGRA data file
    data_type : str
        Type of data: 'raw' or 'derived'
        
    Returns:
    --------
    list
        List of soundings
    """
    with open(file_path, 'r') as f:
        data_text = f.read()
    
    return load_igra_data_from_text(data_text, data_type)