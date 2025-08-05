#!/usr/bin/env python3
"""
APACC Results Format Converter
==============================
Converts APACC validation results between different formats for analysis and sharing.
Supports HDF5, CSV, JSON, and Excel formats with optional merging capabilities.

Usage:
    python convert_results.py --input monte_carlo_results.hdf5 --format csv
    python convert_results.py --input results/20250128_143052 --merge --format excel
    python convert_results.py --input baseline_comparison.hdf5 --format json --pretty

Author: George Frangou
Institution: Cranfield University
DOI: https://doi.org/10.5281/zenodo.8475
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Try to import optional dependencies
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    warnings.warn("openpyxl not installed. Excel export will use basic xlsx writer.")

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


class ResultsConverter:
    """Convert APACC results between different formats."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.supported_formats = {
            'csv': self.to_csv,
            'json': self.to_json,
            'excel': self.to_excel,
            'xlsx': self.to_excel,  # Alias
            'parquet': self.to_parquet,
            'feather': self.to_feather,
            'hdf5': self.to_hdf5,
            'h5': self.to_hdf5,  # Alias
        }
        
        if ZARR_AVAILABLE:
            self.supported_formats['zarr'] = self.to_zarr
    
    def load_data(self, input_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """Load data from file or directory."""
        input_path = Path(input_path)
        data = {}
        
        if input_path.is_file():
            # Single file
            data = self._load_single_file(input_path)
        elif input_path.is_dir():
            # Directory - load all HDF5 files
            hdf5_files = list(input_path.glob("*.hdf5")) + list(input_path.glob("*.h5"))
            
            if not hdf5_files:
                raise ValueError(f"No HDF5 files found in {input_path}")
            
            for hdf5_file in hdf5_files:
                file_data = self._load_single_file(hdf5_file)
                data.update(file_data)
                
            if self.verbose:
                print(f"Loaded {len(data)} datasets from {len(hdf5_files)} files")
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        return merged
    
    def to_csv(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               output_path: Path, **kwargs):
        """Convert to CSV format."""
        if isinstance(data, dict):
            # Multiple datasets - save each separately
            for name, df in data.items():
                csv_path = output_path.parent / f"{output_path.stem}_{name}.csv"
                df.to_csv(csv_path, index=False, **kwargs)
                if self.verbose:
                    print(f"Saved {name} to {csv_path} ({len(df):,} rows)")
        else:
            # Single dataset
            data.to_csv(output_path, index=False, **kwargs)
            if self.verbose:
                print(f"Saved to {output_path} ({len(data):,} rows)")
    
    def to_json(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                output_path: Path, pretty: bool = False, **kwargs):
        """Convert to JSON format."""
        if pretty:
            kwargs['indent'] = 2
            kwargs['orient'] = kwargs.get('orient', 'records')
        
        if isinstance(data, dict):
            # Multiple datasets - save as nested JSON
            json_data = {}
            for name, df in data.items():
                # Convert DataFrame to JSON-serializable format
                json_data[name] = json.loads(df.to_json(orient='records'))
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, **kwargs)
                
            if self.verbose:
                total_records = sum(len(df) for df in data.values())
                print(f"Saved {len(data)} datasets to {output_path} "
                      f"({total_records:,} total records)")
        else:
            # Single dataset
            data.to_json(output_path, **kwargs)
            if self.verbose:
                print(f"Saved to {output_path} ({len(data):,} records)")
    
    def to_excel(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 output_path: Path, **kwargs):
        """Convert to Excel format."""
        if not output_path.suffix in ['.xlsx', '.xls']:
            output_path = output_path.with_suffix('.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl' if EXCEL_AVAILABLE else None) as writer:
            if isinstance(data, dict):
                # Multiple datasets - save as separate sheets
                for name, df in data.items():
                    # Excel sheet names have limitations
                    sheet_name = name[:31].replace('/', '_').replace('\\', '_')
                    df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)
                    
                    if self.verbose:
                        print(f"Added sheet '{sheet_name}' with {len(df):,} rows")
                        
                # Add summary sheet
                summary_data = {
                    'Dataset': list(data.keys()),
                    'Rows': [len(df) for df in data.values()],
                    'Columns': [len(df.columns) for df in data.values()],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='_Summary', index=False)
            else:
                # Single dataset
                data.to_excel(writer, sheet_name='Results', index=False, **kwargs)
        
        if self.verbose:
            print(f"Saved to {output_path}")
    
    def to_parquet(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                   output_path: Path, **kwargs):
        """Convert to Parquet format."""
        if isinstance(data, dict):
            # Multiple datasets - save each separately
            for name, df in data.items():
                parquet_path = output_path.parent / f"{output_path.stem}_{name}.parquet"
                df.to_parquet(parquet_path, index=False, **kwargs)
                if self.verbose:
                    print(f"Saved {name} to {parquet_path} ({len(df):,} rows)")
        else:
            # Single dataset
            data.to_parquet(output_path, index=False, **kwargs)
            if self.verbose:
                print(f"Saved to {output_path} ({len(data):,} rows)")
    
    def to_feather(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                   output_path: Path, **kwargs):
        """Convert to Feather format."""
        if isinstance(data, dict):
            # Multiple datasets - save each separately
            for name, df in data.items():
                feather_path = output_path.parent / f"{output_path.stem}_{name}.feather"
                df.to_feather(feather_path, **kwargs)
                if self.verbose:
                    print(f"Saved {name} to {feather_path} ({len(df):,} rows)")
        else:
            # Single dataset
            data.to_feather(output_path, **kwargs)
            if self.verbose:
                print(f"Saved to {output_path} ({len(data):,} rows)")
    
    def to_hdf5(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                output_path: Path, **kwargs):
        """Convert to HDF5 format."""
        if not output_path.suffix in ['.hdf5', '.h5']:
            output_path = output_path.with_suffix('.hdf5')
        
        with pd.HDFStore(output_path, 'w') as store:
            if isinstance(data, dict):
                # Multiple datasets
                for name, df in data.items():
                    # HDF5 keys must start with '/'
                    key = f"/{name}" if not name.startswith('/') else name
                    store.put(key, df, format='table', **kwargs)
                    
                    if self.verbose:
                        print(f"Added dataset '{key}' with {len(df):,} rows")
            else:
                # Single dataset
                store.put('/results', data, format='table', **kwargs)
        
        if self.verbose:
            print(f"Saved to {output_path}")
    
    def to_zarr(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                output_path: Path, **kwargs):
        """Convert to Zarr format."""
        if not ZARR_AVAILABLE:
            raise ImportError("zarr package not installed")
        
        if isinstance(data, dict):
            # Multiple datasets - create groups
            root = zarr.open_group(str(output_path), mode='w')
            
            for name, df in data.items():
                grp = root.create_group(name)
                grp.create_dataset('data', data=df.values, compression='gzip')
                grp.attrs['columns'] = list(df.columns)
                grp.attrs['shape'] = df.shape
                
                if self.verbose:
                    print(f"Added group '{name}' with {len(df):,} rows")
        else:
            # Single dataset
            z = zarr.open(str(output_path), mode='w')
            z.create_dataset('data', data=data.values, compression='gzip')
            z.attrs['columns'] = list(data.columns)
            z.attrs['shape'] = data.shape
        
        if self.verbose:
            print(f"Saved to {output_path}")
    
    def convert(self, input_path: Union[str, Path], output_format: str,
                output_path: Optional[Union[str, Path]] = None,
                merge: bool = False, **kwargs) -> Path:
        """Main conversion method."""
        input_path = Path(input_path)
        
        # Load data
        data = self.load_data(input_path)
        
        # Merge if requested
        if merge and isinstance(data, dict):
            if self.verbose:
                print("Merging datasets...")
            data = self.merge_datasets(data)
        
        # Determine output path
        if output_path is None:
            if input_path.is_file():
                output_path = input_path.with_suffix(f'.{output_format}')
            else:
                output_path = input_path / f"merged_results.{output_format}"
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported output format: {output_format}. "
                           f"Supported: {', '.join(self.supported_formats.keys())}")
        
        converter = self.supported_formats[output_format]
        converter(data, output_path, **kwargs)
        
        return output_path
    
    def print_data_info(self, data: Dict[str, pd.DataFrame]):
        """Print information about loaded datasets."""
        print("\nLoaded Datasets:")
        print("-" * 60)
        
        total_rows = 0
        total_size = 0
        
        for name, df in data.items():
            rows = len(df)
            cols = len(df.columns)
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            print(f"{name:30} | {rows:8,} rows | {cols:3} cols | {size_mb:6.1f} MB")
            
            total_rows += rows
            total_size += size_mb
        
        print("-" * 60)
        print(f"{'Total:':30} | {total_rows:8,} rows | {'':3}      | {total_size:6.1f} MB")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert APACC results between different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  - csv     : Comma-separated values
  - json    : JavaScript Object Notation
  - excel   : Microsoft Excel (xlsx)
  - parquet : Apache Parquet (efficient columnar storage)
  - feather : Feather format (fast read/write)
  - hdf5    : Hierarchical Data Format 5
  - zarr    : Zarr array storage (if installed)

Examples:
  # Convert single HDF5 file to CSV
  python convert_results.py --input results.hdf5 --format csv
  
  # Convert and merge all HDF5 files in directory to Excel
  python convert_results.py --input results/20250128_143052 --merge --format excel
  
  # Convert to pretty-printed JSON
  python convert_results.py --input data.hdf5 --format json --pretty
  
  # Convert with custom output path
  python convert_results.py --input results.hdf5 --format parquet --output /tmp/results.parquet
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input file or directory path')
    parser.add_argument('--format', '-f', type=str, required=True,
                      help='Output format')
    parser.add_argument('--output', '-o', type=str,
                      help='Output file path (default: auto-generated)')
    parser.add_argument('--merge', '-m', action='store_true',
                      help='Merge multiple datasets into one')
    parser.add_argument('--pretty', action='store_true',
                      help='Pretty-print JSON output')
    parser.add_argument('--info', action='store_true',
                      help='Show dataset information without converting')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create converter
    converter = ResultsConverter(verbose=args.verbose)
    
    try:
        if args.info:
            # Just show information
            data = converter.load_data(args.input)
            converter.print_data_info(data)
        else:
            # Perform conversion
            kwargs = {}
            if args.format == 'json' and args.pretty:
                kwargs['pretty'] = True
            
            output_path = converter.convert(
                args.input,
                args.format,
                args.output,
                merge=args.merge,
                **kwargs
            )
            
            print(f"\nConversion complete!")
            print(f"Output saved to: {output_path}")
            
            # Show file size
            if output_path.exists():
                size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"File size: {size_mb:.1f} MB")
                
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() data
    
    def _load_single_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Load data from a single file."""
        data = {}
        
        if file_path.suffix in ['.hdf5', '.h5']:
            # HDF5 file - may contain multiple datasets
            try:
                with pd.HDFStore(file_path, 'r') as store:
                    keys = store.keys()
                    if not keys:
                        # Try default key
                        df = pd.read_hdf(file_path, key='results')
                        data[file_path.stem] = df
                    else:
                        for key in keys:
                            dataset_name = f"{file_path.stem}{key.replace('/', '_')}"
                            data[dataset_name] = store[key]
                
                if self.verbose:
                    total_rows = sum(len(df) for df in data.values())
                    print(f"Loaded {len(data)} dataset(s) from {file_path.name} "
                          f"({total_rows:,} total rows)")
                          
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                raise
                
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            data[file_path.stem] = df
            
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
            data[file_path.stem] = df
            
        elif file_path.suffix in ['.xlsx', '.xls']:
            # Excel file - may contain multiple sheets
            xl_file = pd.ExcelFile(file_path)
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(xl_file, sheet_name=sheet_name)
                data[f"{file_path.stem}_{sheet_name}"] = df
                
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
            data[file_path.stem] = df
            
        elif file_path.suffix == '.feather':
            df = pd.read_feather(file_path)
            data[file_path.stem] = df
            
        elif ZARR_AVAILABLE and file_path.suffix == '.zarr':
            # Load from zarr
            z = zarr.open(str(file_path), mode='r')
            if 'data' in z:
                df = pd.DataFrame(z['data'][:])
                if 'columns' in z.attrs:
                    df.columns = z.attrs['columns']
                data[file_path.stem] = df
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return data
    
    def merge_datasets(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple datasets into a single DataFrame."""
        if len(data) == 1:
            return list(data.values())[0]
        
        merged_dfs = []
        
        for name, df in data.items():
            # Add source column to track origin
            df_copy = df.copy()
            df_copy['_source_dataset'] = name
            merged_dfs.append(df_copy)
        
        # Concatenate all dataframes
        merged = pd.concat(merged_dfs, ignore_index=True, sort=False)
        
        if self.verbose:
            print(f"Merged {len(data)} datasets into single DataFrame "
                  f"with {len(merged):,} rows and {len(merged.columns)} columns")
        
        return