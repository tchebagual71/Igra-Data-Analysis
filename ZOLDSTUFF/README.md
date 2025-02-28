# IGRA Data Analysis in Jupyter Notebook

This repository contains a Jupyter Notebook tutorial for analyzing atmospheric sounding data from the Integrated Global Radiosonde Archive (IGRA). This tutorial is designed for beginners with little to no coding experience.

## Overview

The Integrated Global Radiosonde Archive (IGRA) provides historical atmospheric sounding data from weather balloons worldwide. This tutorial shows you how to load, process, and visualize this data using Python and Jupyter Notebook.

## Getting Started

### Prerequisites

- Python 3.7 or newer
- Jupyter Notebook
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - metpy

### Installation

1. Clone or download this repository
2. Install required packages by running:
   ```
   pip install pandas numpy matplotlib metpy xarray
   ```
3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. Open `IGRA_Data_Analysis.ipynb`

## Repository Contents

- `IGRA_Data_Analysis.ipynb`: Main Jupyter Notebook with step-by-step tutorial
- `data/`: Folder containing example IGRA data files
  - `raw_data.txt`: Example raw sounding data
  - `derived_data.txt`: Example derived sounding parameters data
- `README.md`: This file

## Tutorial Structure

The notebook walks you through:

1. **Data Loading and Parsing**
   - Reading IGRA format text files
   - Parsing the specialized IGRA data structure
   - Converting to pandas DataFrames

2. **Data Exploration**
   - Basic statistics and structure analysis
   - Examining atmospheric pressure levels
   - Understanding data format and units

3. **Visualization**
   - Temperature profiles
   - Skew-T Log-P diagrams
   - Time-height cross sections
   - Wind profiles
   - Comparative analysis across times

4. **Analysis and Export**
   - Statistical analysis by pressure level
   - Exporting processed data to CSV
   - Saving high-quality plots

## Data Files

The sample data files included are from IGRA station USM00074794:

- `raw_data.txt`: Contains raw atmospheric measurements including pressure, temperature, humidity, and wind for multiple soundings
- `derived_data.txt`: Contains derived parameters calculated from the raw data

These files follow the IGRA v2.2 format as described in the [IGRA documentation](https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt).

## Output Files

When running the notebook, the following outputs will be generated:

- `plots/`: Folder containing saved visualization figures
- `exported_data/`: Folder containing processed CSV files

## Further Resources

- [IGRA Website](https://www.ncei.noaa.gov/products/integrated-global-radiosonde-archive)
- [MetPy Documentation](https://unidata.github.io/MetPy/latest/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Atmospheric Sounding Basics](https://www.weather.gov/jetstream/upperair_intro)

## Acknowledgments

- NOAA National Centers for Environmental Information for providing the IGRA dataset
- Unidata for the MetPy package

## License

This tutorial is available under the MIT License - feel free to use, modify, and share!
