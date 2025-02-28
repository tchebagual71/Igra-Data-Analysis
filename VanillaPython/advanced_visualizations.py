import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import datetime
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import scipy.interpolate as interpolate

def plot_interactive_skewt(data, date, pressure_min=100, pressure_max=1050):
    """
    Create an interactive Skew-T log-P diagram with multiple data panels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the sounding data
    date : datetime-like
        Date of the sounding to plot
    pressure_min : float
        Minimum pressure level to plot (hPa)
    pressure_max : float
        Maximum pressure level to plot (hPa)
    """
    # Filter data for the selected date
    sounding = data[data['datetime'] == date].copy()
    
    # Sort by pressure (decreasing)
    sounding = sounding.sort_values('pressure', ascending=False)
    
    # Check if we have temperature and pressure data
    if 'temperature' not in sounding.columns or 'pressure' not in sounding.columns:
        print("Missing required columns 'temperature' or 'pressure'")
        return None
    
    # Filter out rows with missing temperature or pressure
    sounding = sounding.dropna(subset=['temperature', 'pressure'])
    
    if sounding.empty:
        print("No valid data for this date")
        return None
    
    # Convert data to appropriate units and arrays
    p = sounding['pressure'].values * units.hPa
    T = sounding['temperature'].values * units.degC
    
    # Handle dewpoint - either use directly or calculate from RH or dewpoint depression
    if 'dewpoint' in sounding.columns:
        Td = sounding['dewpoint'].values * units.degC
    elif 'dewpoint_depression' in sounding.columns and not sounding['dewpoint_depression'].isna().all():
        Td = (sounding['temperature'] - sounding['dewpoint_depression']).values * units.degC
    elif 'relative_humidity' in sounding.columns and not sounding['relative_humidity'].isna().all():
        # Calculate dewpoint from RH
        e = mpcalc.vapor_pressure(p, T, sounding['relative_humidity'].values * units.percent)
        Td = mpcalc.dewpoint(e)
    else:
        Td = None
    
    # Process wind data if available
    has_wind = ('wind_speed' in sounding.columns and 'wind_direction' in sounding.columns and 
                not sounding['wind_speed'].isna().all() and not sounding['wind_direction'].isna().all())
    
    if has_wind:
        # Filter out missing wind data
        wind_data = sounding.dropna(subset=['wind_speed', 'wind_direction'])
        
        if not wind_data.empty:
            wind_p = wind_data['pressure'].values * units.hPa
            wind_speed = wind_data['wind_speed'].values * units.meter / units.second
            wind_dir = wind_data['wind_direction'].values * units.degree
            
            # Calculate u and v components
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
        else:
            has_wind = False
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(15, 10))
    
    # Define a layout with 2 rows and 3 columns
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1.2], width_ratios=[3, 1, 1])
    
    # Skew-T plot
    ax_skewt = plt.subplot(gs[0, 0])
    skew = SkewT(ax_skewt, rotation=45)
    
    # Plot data on Skew-T
    skew.plot(p, T, 'r')
    if Td is not None:
        skew.plot(p, Td, 'g')
    
    # Add wind barbs if available
    if has_wind:
        skew.plot_barbs(wind_p, u, v)
    
    # Plot thermodynamic parameters
    if Td is not None:
        # Calculate and plot the parcel profile
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, 'k--')
            
            # Calculate CAPE and CIN
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            
            # Add CAPE and CIN to the plot
            ax_skewt.text(0.85, 0.85, f'CAPE: {cape.magnitude:.0f} J/kg', transform=ax_skewt.transAxes)
            ax_skewt.text(0.85, 0.8, f'CIN: {cin.magnitude:.0f} J/kg', transform=ax_skewt.transAxes)
        except Exception as e:
            print(f"Error calculating parcel profile: {e}")
    
    # Add features to Skew-T
    skew.plot_dry_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_moist_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_mixing_lines(linewidth=0.5, alpha=0.5)
    
    # Set limits
    skew.ax.set_ylim(pressure_max, pressure_min)
    skew.ax.set_xlim(-40, 50)
    
    # Add labels
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_title(f'Skew-T Log-P - {date}')
    
    # Hodograph panel
    if has_wind:
        ax_hod = plt.subplot(gs[0, 1])
        h = Hodograph(ax_hod, component_range=80)
        h.plot(u, v)
        h.add_grid(increment=20)
        
        # Label specific pressure levels
        for level in [1000, 850, 700, 500, 300, 200]:
            level_data = wind_data[wind_data['pressure'].between(level-5, level+5)]
            if not level_data.empty:
                this_u = level_data['wind_speed'].values[0] * np.sin(np.radians(level_data['wind_direction'].values[0])) * -1
                this_v = level_data['wind_speed'].values[0] * np.cos(np.radians(level_data['wind_direction'].values[0])) * -1
                
                ax_hod.text(this_u + 2, this_v + 2, str(level), fontsize=8)
        
        ax_hod.set_title('Hodograph')
    
    # Additional panels for other parameters
    
    # Temperature and dewpoint profile
    ax_temp = plt.subplot(gs[0, 2])
    ax_temp.plot(T.magnitude, p.magnitude, 'r-', label='Temperature')
    if Td is not None:
        ax_temp.plot(Td.magnitude, p.magnitude, 'g-', label='Dewpoint')
    ax_temp.set_ylim(pressure_max, pressure_min)
    ax_temp.set_yscale('log')
    ax_temp.grid(True)
    ax_temp.set_xlabel('Temperature (°C)')
    ax_temp.set_title('Vertical Profile')
    ax_temp.legend()
    
    # Wind speed and direction panel
    if has_wind:
        ax_wind = plt.subplot(gs[1, 0])
        ax_wind.plot(wind_speed.magnitude, wind_p.magnitude, 'b-', label='Speed')
        ax_wind.set_xlabel('Wind Speed (m/s)')
        ax_wind.set_ylabel('Pressure (hPa)')
        ax_wind.set_yscale('log')
        ax_wind.set_ylim(pressure_max, pressure_min)
        ax_wind.grid(True)
        
        # Add wind direction on secondary axis
        ax_dir = ax_wind.twiny()
        ax_dir.plot(wind_dir.magnitude, wind_p.magnitude, 'r--', label='Direction')
        ax_dir.set_xlabel('Wind Direction (degrees)')
        ax_dir.set_xlim(0, 360)
        
        # Add legends
        lines, labels = ax_wind.get_legend_handles_labels()
        lines2, labels2 = ax_dir.get_legend_handles_labels()
        ax_wind.legend(lines + lines2, labels + labels2, loc='upper right')
        
        ax_wind.set_title('Wind Profile')
    
    # Potential temperature and equivalent potential temperature
    ax_theta = plt.subplot(gs[1, 1:])
    
    # Calculate potential temperature
    theta = mpcalc.potential_temperature(p, T)
    ax_theta.plot(theta.magnitude, p.magnitude, 'b-', label='θ')
    
    # Calculate equivalent potential temperature if dewpoint is available
    if Td is not None:
        theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
        ax_theta.plot(theta_e.magnitude, p.magnitude, 'r-', label='θe')
    
    ax_theta.set_ylim(pressure_max, pressure_min)
    ax_theta.set_yscale('log')
    ax_theta.set_xlabel('Potential Temperature (K)')
    ax_theta.set_title('Thermodynamic Parameters')
    ax_theta.grid(True)
    ax_theta.legend()
    
    plt.tight_layout()
    return fig

def plot_time_height_cross_section(data, variable='temperature', 
                                   pressure_levels=None, 
                                   cmap='RdBu_r',
                                   interpolate_data=True):
    """
    Create a time-height cross section plot of the specified variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the sounding data
    variable : str
        Column name of the variable to plot
    pressure_levels : list, optional
        List of pressure levels to interpolate to
    cmap : str
        Colormap to use
    interpolate_data : bool
        Whether to interpolate the data to a regular grid
    """
    if variable not in data.columns:
        print(f"Variable {variable} not found in data")
        return None
    
    # Create a pivot table
    pivot = data.pivot_table(
        index='datetime', 
        columns='pressure',
        values=variable
    )
    
    # Sort the pivot table columns (pressure levels)
    pivot = pivot.sort_index(axis=1, ascending=False)
    
    # Optional: Interpolate to regular pressure levels
    if interpolate_data and pressure_levels is not None:
        # Create a new DataFrame with the specified pressure levels
        new_pivot = pd.DataFrame(index=pivot.index, columns=pressure_levels)
        
        # Interpolate each row (time) to the new pressure levels
        for idx in pivot.index:
            row = pivot.loc[idx].dropna()
            if len(row) > 1:  # Need at least 2 points to interpolate
                f = interpolate.interp1d(
                    row.index, row.values, 
                    bounds_error=False, fill_value=np.nan
                )
                new_pivot.loc[idx] = f(pressure_levels)
        
        pivot = new_pivot
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the contourf plot
    if variable == 'temperature':
        levels = np.arange(-80, 40, 2)
        norm = plt.Normalize(-80, 40)
    elif variable == 'relative_humidity':
        levels = np.arange(0, 105, 5)
        norm = plt.Normalize(0, 100)
        cmap = 'Blues'
    elif variable == 'wind_speed':
        levels = np.arange(0, 100, 5)
        norm = plt.Normalize(0, 80)
        cmap = 'viridis'
    else:
        levels = 20
        norm = None
    
    # Make sure dates are formatted as datetime
    if pivot.index.dtype != 'datetime64[ns]':
        pivot.index = pd.to_datetime(pivot.index)
    
    # Plot the data
    cf = ax.contourf(
        pivot.index, pivot.columns, pivot.T.values, 
        levels=levels, cmap=cmap, norm=norm, extend='both'
    )
    
    # Add contour lines
    cs = ax.contour(
        pivot.index, pivot.columns, pivot.T.values, 
        levels=levels[::4], colors='k', linewidths=0.5, alpha=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax)
    if variable == 'temperature':
        cbar.set_label('Temperature (°C)')
    elif variable == 'relative_humidity':
        cbar.set_label('Relative Humidity (%)')
    elif variable == 'wind_speed':
        cbar.set_label('Wind Speed (m/s)')
    else:
        cbar.set_label(variable)
    
    # Set y-axis to log scale and invert
    ax.set_yscale('log')
    ax.set_ylim(1050, 100)
    
    # Format the date axis
    if len(pivot.index) > 1:
        date_range = (pivot.index.max() - pivot.index.min()).total_seconds()
        if date_range < 86400 * 3:  # Less than 3 days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        elif date_range < 86400 * 30:  # Less than a month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(f'Time-Height Cross Section - {variable}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_stability_timeseries(data, stability_params=None):
    """
    Plot a time series of stability parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the derived sounding data
    stability_params : list, optional
        List of stability parameters to plot
    """
    if stability_params is None:
        stability_params = ['cape', 'cin', 'lifted_index', 'precipitable_water']
    
    # Filter columns that are in the data
    available_params = [p for p in stability_params if p in data.columns]
    
    if not available_params:
        print("No stability parameters found in data")
        return None
    
    # Get one value per sounding (first non-null value for each datetime)
    df = data.sort_values(['datetime', 'pressure'])
    stability_df = df.groupby('datetime')[available_params].first().reset_index()
    
    # Create figure with subplots
    n_rows = len(available_params)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3*n_rows), sharex=True)
    
    # If only one parameter, make axes iterable
    if n_rows == 1:
        axes = [axes]
    
    # Plot each parameter
    for i, param in enumerate(available_params):
        ax = axes[i]
        
        # Format the parameter name for display
        param_name = ' '.join(w.capitalize() for w in param.split('_'))
        
        # Plot the data
        ax.plot(
            stability_df['datetime'], 
            stability_df[param], 
            marker='o', 
            linestyle='-'
        )
        
        # Fill under curve for CAPE and CIN
        if param == 'cape':
            ax.fill_between(
                stability_df['datetime'], 
                0, 
                stability_df[param], 
                alpha=0.3, 
                color='red'
            )
        elif param == 'cin':
            ax.fill_between(
                stability_df['datetime'], 
                0, 
                stability_df[param], 
                alpha=0.3, 
                color='blue'
            )
        
        # Add gridlines
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at zero for parameters where that's meaningful
        if param in ['lifted_index', 'cin']:
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Add labels
        ax.set_ylabel(param_name)
        
        # Add title to top subplot
        if i == 0:
            ax.set_title('Atmospheric Stability Parameters')
    
    # Set x-axis label on bottom subplot
    axes[-1].set_xlabel('Date')
    
    # Format the date axis
    if stability_df['datetime'].dtype != 'datetime64[ns]':
        stability_df['datetime'] = pd.to_datetime(stability_df['datetime'])
    
    date_range = (stability_df['datetime'].max() - stability_df['datetime'].min()).total_seconds()
    if date_range < 86400 * 3:  # Less than 3 days
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif date_range < 86400 * 30:  # Less than a month
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return fig

def plot_wind_cross_section(data):
    """
    Create a time-height cross section plot of wind speed and direction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the sounding data
    """
    if 'wind_speed' not in data.columns or 'wind_direction' not in data.columns:
        print("Wind data not found in DataFrame")
        return None
    
    # Create a pivot table for wind speed
    speed_pivot = data.pivot_table(
        index='datetime',
        columns='pressure',
        values='wind_speed'
    )
    
    # Create a pivot table for wind direction
    dir_pivot = data.pivot_table(
        index='datetime',
        columns='pressure',
        values='wind_direction'
    )
    
    # Sort by pressure (decreasing)
    speed_pivot = speed_pivot.sort_index(axis=1, ascending=False)
    dir_pivot = dir_pivot.sort_index(axis=1, ascending=False)
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot wind speed
    cf1 = ax1.contourf(
        speed_pivot.index, 
        speed_pivot.columns, 
        speed_pivot.T.values,
        levels=np.arange(0, 81, 5),
        cmap='viridis',
        extend='max'
    )
    
    # Add contour lines
    cs1 = ax1.contour(
        speed_pivot.index, 
        speed_pivot.columns, 
        speed_pivot.T.values,
        levels=np.arange(0, 81, 20),
        colors='k',
        linewidths=0.5,
        alpha=0.5
    )
    
    # Add colorbar
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label('Wind Speed (m/s)')
    
    # Plot wind direction
    # Use a circular colormap for wind direction
    cmap = plt.cm.hsv
    norm = plt.Normalize(0, 360)
    
    cf2 = ax2.contourf(
        dir_pivot.index, 
        dir_pivot.columns, 
        dir_pivot.T.values,
        levels=np.arange(0, 361, 15),
        cmap=cmap,
        norm=norm
    )
    
    # Add colorbar with custom ticks and labels
    cbar2 = plt.colorbar(cf2, ax=ax2, ticks=[0, 90, 180, 270, 360])
    cbar2.set_label('Wind Direction')
    cbar2.ax.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    
    # Set y-axis to log scale
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    
    # Set y-axis limits and invert
    ax1.set_ylim(1050, 100)
    ax2.set_ylim(1050, 100)
    
    # Add grids
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add plot labels and titles
    ax1.set_ylabel('Pressure (hPa)')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_xlabel('Date')
    
    ax1.set_title('Wind Speed Time-Height Cross Section')
    ax2.set_title('Wind Direction Time-Height Cross Section')
    
    # Format the date axis
    if dir_pivot.index.dtype != 'datetime64[ns]':
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        date_range = (dir_pivot.index.max() - dir_pivot.index.min()).total_seconds()
        if date_range < 86400 * 3:  # Less than 3 days
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        elif date_range < 86400 * 30:  # Less than a month
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        else:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return fig

def create_custom_soundings_dashboard(data, date=None):
    """
    Create a comprehensive dashboard of multiple sounding visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the sounding data
    date : datetime-like, optional
        Specific date to plot. If None, the first date in the data is used.
    """
    # If no specific date is provided, use the first date
    if date is None:
        date = data['datetime'].iloc[0]
    
    # Filter for the specific date
    sounding_data = data[data['datetime'] == date].copy()
    
    # Sort by pressure (decreasing)
    sounding_data = sounding_data.sort_values('pressure', ascending=False)
    
    # Check if we have enough data
    if sounding_data.empty or 'temperature' not in sounding_data.columns:
        print("Not enough data for analysis")
        return None
    
    # Create a complex figure with multiple panels
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    
    # ---- Skew-T Panel ----
    ax_skewt = plt.subplot(gs[0, :2])
    skew = SkewT(ax_skewt, rotation=45)
    
    # Convert data to appropriate units
    p = sounding_data['pressure'].values * units.hPa
    T = sounding_data['temperature'].values * units.degC
    
    # Add temperature profile
    skew.plot(p, T, 'r', linewidth=2)
    
    # Add dewpoint if available
    if 'dewpoint' in sounding_data.columns and not sounding_data['dewpoint'].isna().all():
        Td = sounding_data['dewpoint'].values * units.degC
        skew.plot(p, Td, 'g', linewidth=2)
    elif 'dewpoint_depression' in sounding_data.columns and not sounding_data['dewpoint_depression'].isna().all():
        Td = (sounding_data['temperature'] - sounding_data['dewpoint_depression']).values * units.degC
        skew.plot(p, Td, 'g', linewidth=2)
    elif 'relative_humidity' in sounding_data.columns and not sounding_data['relative_humidity'].isna().all():
        # Calculate dewpoint from RH (simplified)
        rh_values = np.where(sounding_data['relative_humidity'].isna(), 0, sounding_data['relative_humidity'])
        RH = rh_values * units.percent
        e = mpcalc.vapor_pressure(p, T, RH)
        Td = mpcalc.dewpoint(e)
        skew.plot(p, Td, 'g', linewidth=2)
    else:
        Td = None
    
    # Add wind barbs if available
    if 'wind_speed' in sounding_data.columns and 'wind_direction' in sounding_data.columns:
        wind_data = sounding_data.dropna(subset=['wind_speed', 'wind_direction'])
        if not wind_data.empty:
            p_wind = wind_data['pressure'].values * units.hPa
            speed = wind_data['wind_speed'].values * units.meter / units.second
            direction = wind_data['wind_direction'].values * units.degree
            u, v = mpcalc.wind_components(speed, direction)
            skew.plot_barbs(p_wind, u, v)
    
    # Add features
    skew.plot_dry_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_moist_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_mixing_lines(linewidth=0.5, alpha=0.5)
    
    # Calculate parcel profile and CAPE if dewpoint is available
    if Td is not None:
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, 'k--', linewidth=1.5)
            
            # Calculate CAPE and CIN
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            
            # Add values to the plot
            skew.ax.text(0.8, 0.95, f'CAPE: {cape.magnitude:.0f} J/kg',
                        transform=skew.ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            skew.ax.text(0.8, 0.90, f'CIN: {cin.magnitude:.0f} J/kg',
                        transform=skew.ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            
            # Shade CAPE and CIN areas
            skew.shade_cape(p, T, prof)
            skew.shade_cin(p, T, prof)
        except Exception as e:
            print(f"Error calculating parcel profile: {e}")
    
    # Set limits and labels
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 50)
    skew.ax.set_title(f'Skew-T Log-P Diagram - {date}', fontsize=14)
    
    # ---- Hodograph Panel ----
    ax_hodo = plt.subplot(gs[0, 2])
    
    if 'wind_speed' in sounding_data.columns and 'wind_direction' in sounding_data.columns:
        wind_data = sounding_data.dropna(subset=['wind_speed', 'wind_direction'])
        if not wind_data.empty:
            # Calculate u and v components
            speed = wind_data['wind_speed'].values * units.meter / units.second
            direction = wind_data['wind_direction'].values * units.degree
            u, v = mpcalc.wind_components(speed, direction)
            
            # Create hodograph
            h = Hodograph(ax_hodo, component_range=80)
            h.plot(u, v)
            h.add_grid(increment=20)
            
            # Add pressure labels
            for level in [1000, 850, 700, 500, 300, 200]:
                level_data = wind_data[wind_data['pressure'].between(level-5, level+5)]
                if not level_data.empty:
                    idx = level_data.index[0]
                    ax_hodo.text(
                        u[wind_data.index.get_indexer([idx])[0]].magnitude + 2,
                        v[wind_data.index.get_indexer([idx])[0]].magnitude + 2,
                        str(level),
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
    
    ax_hodo.set_title('Hodograph', fontsize=14)
    
    # ---- Wind Profile Panel ----
    ax_wind = plt.subplot(gs[1, 0])
    
    if 'wind_speed' in sounding_data.columns:
        wind_data = sounding_data.dropna(subset=['wind_speed'])
        if not wind_data.empty:
            # Plot wind speed
            ax_wind.plot(
                wind_data['wind_speed'],
                wind_data['pressure'],
                'b-',
                linewidth=2
            )
            
            # Add wind direction on a separate axis if available
            if 'wind_direction' in wind_data.columns and not wind_data['wind_direction'].isna().all():
                ax_dir = ax_wind.twiny()
                ax_dir.plot(
                    wind_data['wind_direction'],
                    wind_data['pressure'],
                    'r-.',
                    linewidth=1.5,
                    alpha=0.7
                )
                
                ax_dir.set_xlabel('Wind Direction (°)')
                ax_dir.set_xlim(0, 360)
                ax_dir.set_xticks([0, 90, 180, 270, 360])
                ax_dir.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
            
            ax_wind.set_yscale('log')
            ax_wind.set_ylim(1050, 100)
            ax_wind.set_xlabel('Wind Speed (m/s)')
            ax_wind.set_ylabel('Pressure (hPa)')
            ax_wind.grid(True, alpha=0.3)
            ax_wind.set_title('Wind Profile', fontsize=12)
    
    # ---- Temperature Profile Panel ----
    ax_temp = plt.subplot(gs[1, 1])
    
    # Plot temperature
    ax_temp.plot(sounding_data['temperature'], sounding_data['pressure'], 'r-', linewidth=2, label='Temperature')
    
    # Add dewpoint if available
    if 'dewpoint' in sounding_data.columns and not sounding_data['dewpoint'].isna().all():
        ax_temp.plot(sounding_data['dewpoint'], sounding_data['pressure'], 'g-', linewidth=2, label='Dewpoint')
    elif 'dewpoint_depression' in sounding_data.columns and not sounding_data['dewpoint_depression'].isna().all():
        dewpoint = sounding_data['temperature'] - sounding_data['dewpoint_depression']
        ax_temp.plot(dewpoint, sounding_data['pressure'], 'g-', linewidth=2, label='Dewpoint')
    
    ax_temp.set_yscale('log')
    ax_temp.set_ylim(1050, 100)
    ax_temp.set_xlabel('Temperature (°C)')
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_title('Temperature Profile', fontsize=12)
    ax_temp.legend(loc='upper right')
    
    # ---- Humidity Profile Panel ----
    ax_rh = plt.subplot(gs[1, 2])
    
    # Plot relative humidity if available
    if 'relative_humidity' in sounding_data.columns and not sounding_data['relative_humidity'].isna().all():
        rh_data = sounding_data.dropna(subset=['relative_humidity'])
        ax_rh.plot(
            rh_data['relative_humidity'],
            rh_data['pressure'],
            'b-',
            linewidth=2
        )
        
        ax_rh.set_yscale('log')
        ax_rh.set_ylim(1050, 100)
        ax_rh.set_xlim(0, 100)
        ax_rh.set_xlabel('Relative Humidity (%)')
        ax_rh.grid(True, alpha=0.3)
        ax_rh.set_title('Humidity Profile', fontsize=12)
    
    # ---- Stability Parameters Panel ----
    ax_stab = plt.subplot(gs[2, :])
    
    # Check if potential temperature is available or can be calculated
    if 'potential_temp' in sounding_data.columns:
        theta = sounding_data['potential_temp']
        ax_stab.plot(theta, sounding_data['pressure'], 'b-', linewidth=2, label='θ')
    elif 'temperature' in sounding_data.columns:
        # Calculate potential temperature
        p = sounding_data['pressure'].values * units.hPa
        T = sounding_data['temperature'].values * units.degC
        theta = mpcalc.potential_temperature(p, T)
        ax_stab.plot(theta.magnitude, p.magnitude, 'b-', linewidth=2, label='θ')
    
    # Calculate and plot equivalent potential temperature if possible
    if 'temperature' in sounding_data.columns and Td is not None:
        p = sounding_data['pressure'].values * units.hPa
        T = sounding_data['temperature'].values * units.degC
        theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
        ax_stab.plot(theta_e.magnitude, p.magnitude, 'r-', linewidth=2, label='θe')
    
    ax_stab.set_yscale('log')
    ax_stab.set_ylim(1050, 100)
    ax_stab.set_xlabel('Temperature (K)')
    ax_stab.set_ylabel('Pressure (hPa)')
    ax_stab.grid(True, alpha=0.3)
    ax_stab.set_title('Thermodynamic Parameters', fontsize=12)
    ax_stab.legend(loc='upper right')
    
    # Adjust the layout
    plt.tight_layout()
    
    return fig

# Example function to create a complete analysis of IGRA data
def create_igra_analysis_report(data, derived_data=None, output_dir='output', station_id=None):
    """
    Create a comprehensive analysis report of IGRA data with multiple visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the raw sounding data
    derived_data : pd.DataFrame, optional
        DataFrame containing the derived parameters
    output_dir : str
        Directory to save the output files
    station_id : str, optional
        Station ID for labeling purposes
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set station ID if not provided
    if station_id is None and 'station_id' in data.columns:
        station_id = data['station_id'].iloc[0]
    
    # Basic information
    print(f"Analyzing data for station {station_id}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Number of soundings: {data['datetime'].nunique()}")
    print(f"Number of levels: {len(data)}")
    
    # 1. Create a Skew-T plot for a selected sounding (e.g., the first one)
    first_date = data['datetime'].iloc[0]
    print(f"\nCreating Skew-T plot for {first_date}...")
    
    try:
        # Use the custom Skew-T function
        fig_skewt = plot_interactive_skewt(data, first_date)
        if fig_skewt:
            fig_skewt.savefig(os.path.join(output_dir, f"{station_id}_skewt.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_skewt)
    except Exception as e:
        print(f"Error creating Skew-T plot: {e}")
    
    # 2. Create a time-height cross section of temperature
    print("\nCreating time-height cross section of temperature...")
    
    try:
        fig_temp = plot_time_height_cross_section(data, variable='temperature')
        if fig_temp:
            fig_temp.savefig(os.path.join(output_dir, f"{station_id}_temp_cross_section.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_temp)
    except Exception as e:
        print(f"Error creating temperature cross section: {e}")
    
    # 3. Create a time-height cross section of relative humidity (if available)
    if 'relative_humidity' in data.columns and not data['relative_humidity'].isna().all():
        print("\nCreating time-height cross section of relative humidity...")
        
        try:
            fig_rh = plot_time_height_cross_section(data, variable='relative_humidity')
            if fig_rh:
                fig_rh.savefig(os.path.join(output_dir, f"{station_id}_rh_cross_section.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_rh)
        except Exception as e:
            print(f"Error creating humidity cross section: {e}")
    
    # 4. Create a wind cross section (if wind data is available)
    if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
        print("\nCreating wind cross section...")
        
        try:
            fig_wind = plot_wind_cross_section(data)
            if fig_wind:
                fig_wind.savefig(os.path.join(output_dir, f"{station_id}_wind_cross_section.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_wind)
        except Exception as e:
            print(f"Error creating wind cross section: {e}")
    
    # 5. Create a stability parameter time series (if derived data is available)
    if derived_data is not None and not derived_data.empty:
        print("\nCreating stability parameter time series...")
        
        try:
            fig_stab = plot_stability_timeseries(derived_data)
            if fig_stab:
                fig_stab.savefig(os.path.join(output_dir, f"{station_id}_stability_timeseries.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_stab)
        except Exception as e:
            print(f"Error creating stability time series: {e}")
    
    # 6. Create a comprehensive dashboard for a selected sounding
    print(f"\nCreating comprehensive dashboard for {first_date}...")
    
    try:
        fig_dash = create_custom_soundings_dashboard(data, first_date)
        if fig_dash:
            fig_dash.savefig(os.path.join(output_dir, f"{station_id}_dashboard.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_dash)
    except Exception as e:
        print(f"Error creating dashboard: {e}")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    
    return {
        'station_id': station_id,
        'output_dir': output_dir,
        'first_date': first_date,
        'num_soundings': data['datetime'].nunique(),
        'date_range': (data['datetime'].min(), data['datetime'].max())
    }
