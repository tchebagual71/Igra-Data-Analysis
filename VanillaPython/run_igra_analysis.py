#!/usr/bin/env python3
"""
IGRA Data Analysis Runner

This script provides a simple command-line interface to run 
the IGRA data analysis tools.
"""

import os
import sys
import argparse

def main():
    """Main function to parse arguments and run the appropriate analysis."""
    parser = argparse.ArgumentParser(description='IGRA Weather Balloon Data Analysis')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug IGRA data files')
    debug_parser.add_argument('file', help='Path to IGRA data file')
    debug_parser.add_argument('--type', choices=['raw', 'derived'], default='raw',
                            help='Type of data file (raw or derived)')
    
    # Full analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run full analysis')
    analyze_parser.add_argument('--raw', help='Path to raw data file')
    analyze_parser.add_argument('--derived', help='Path to derived data file')
    analyze_parser.add_argument('--output', default='output',
                              help='Output directory for results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command was specified, show help and exit
    if not args.command:
        parser.print_help()
        return
    
    # Handle debug command
    if args.command == 'debug':
        # Check if the file exists
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        
        try:
            # Import and run the debug module
            from igra_debug import analyze_file
            analyze_file(args.file, args.type)
        except ImportError:
            print("Error: Could not import igra_debug module.")
            print("Make sure igra_debug.py is in the current directory or Python path.")
            return 1
        except Exception as e:
            print(f"Error during debugging: {str(e)}")
            return 1
    
    # Handle analyze command
    elif args.command == 'analyze':
        # Set default file paths if not specified
        raw_path = args.raw or '/home/tdieckman/Igra-Data-Analysis/VanillaPython/data/USM0007479f-data-exampleportion.txt'
        derived_path = args.derived or '/home/tdieckman/Igra-Data-Analysis/VanillaPython/data/USM0007479f-drvd-exampleportion.txt'
        
        # Check if files exist
        if not os.path.exists(raw_path):
            print(f"Warning: Raw data file not found: {raw_path}")
            print("Will attempt to continue with derived data only.")
        
        if not os.path.exists(derived_path):
            print(f"Warning: Derived data file not found: {derived_path}")
            print("Will attempt to continue with raw data only.")
        
        if not os.path.exists(raw_path) and not os.path.exists(derived_path):
            print("Error: Both raw and derived data files not found.")
            return 1
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
            
            # Temporarily modify sys.path to use our current directory
            # This ensures Python can find our modules
            sys.path.insert(0, os.getcwd())
            
            # Set environment variables for file paths
            os.environ['IGRA_RAW_DATA'] = raw_path
            os.environ['IGRA_DERIVED_DATA'] = derived_path
            os.environ['IGRA_OUTPUT_DIR'] = args.output
            
            # Import and run the main analysis module
            from complete_example import main
            results = main()
            
            print(f"\nAnalysis completed successfully. Results saved to: {os.path.abspath(args.output)}")
        except ImportError as e:
            print(f"Error: Could not import required modules: {str(e)}")
            print("Make sure all required Python files are in the current directory.")
            return 1
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())