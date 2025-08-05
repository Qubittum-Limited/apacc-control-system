#!/usr/bin/env python3
"""
convert_results_apaccsim.py

Converts APACC-Sim validation results between different formats
Supports CSV, JSON, Excel, Parquet, Feather, and more

Author: George Frangou
Institution: Cranfield University
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import datetime
import hashlib

# Core data handling
import numpy as np
import pandas as pd
import h5py

# Format-specific imports
try:
    import pyarrow.parquet as pq
    import pyarrow.feather as feather
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    warnings.warn("PyArrow not available. Parquet and Feather formats disabled.")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    warnings.warn("openpyxl not available. Excel writing may be limited.")

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. MATLAB .mat file support disabled.")

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    warnings.warn("msgpack not available. MessagePack format disabled.")


class ResultsConverter:
    """Converts APACC-Sim results between various formats"""
    
    # Supported formats
    FORMATS = {
        'csv': 'Comma-separated values',
        'json': 'JavaScript Object Notation',
        'excel': 'Microsoft Excel',
        'parquet': 'Apache Parquet (columnar)',
        'feather': 'Feather (fast serialization)',
        'hdf5': 'Hierarchical Data Format 5',
        'mat': 'MATLAB workspace',
        'msgpack': 'MessagePack binary format',
        'pickle': 'Python pickle format'
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.metadata = {
            'converter_version': '1.0',
            'conversion_timestamp': None,
            'source_format': None,
            'target_format': None,
            'data_shape': None,
            'compression': None
        }
        
    def convert(self, input_path: Union[str, Path], output_path: Union[str, Path],
                input_format: Optional[str] = None, output_format: Optional[str] = None,
                compression: Optional[str] = None, **kwargs):
        """
        Convert results from one format to another
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            input_format: Input format (auto-detected if None)
            output_format: Output format (auto-detected if None)
            compression: Compression method (format-specific)
            **kwargs: Additional format-specific options
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Auto-detect formats if not specified
        if input_format is None:
            input_format = self._detect_format(input_path)
        if output_format is None:
            output_format = self._detect_format(output_path)
            
        # Validate formats
        if input_format not in self.FORMATS:
            raise ValueError(f"Unsupported input format: {input_format}")
        if output_format not in self.FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Check format availability
        self._check_format_availability(input_format)
        self._check_format_availability(output_format)
        
        # Update metadata
        self.metadata.update({
            'conversion_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'source_format': input_format,
            'target_format': output_format,
            'compression': compression,
            'source_file': str(input_path),
            'target_file': str(output_path)
        })
        
        if self.verbose:
            print(f"Converting {input_path} ({input_format}) -> {output_path} ({output_format})")
            
        # Load data
        data = self._load_data(input_path, input_format, **kwargs)
        
        # Convert and save
        self._save_data(data, output_path, output_format, compression, **kwargs)
        
        if self.verbose:
            print(f"Conversion complete. Output saved to {output_path}")
            
    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format from extension"""
        ext = path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.mat': 'mat',
            '.msgpack': 'msgpack',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        
        if ext in format_map:
            return format_map[ext]
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")
            
    def _check_format_availability(self, format: str):
        """Check if required libraries for format are available"""
        if format == 'parquet' and not ARROW_AVAILABLE:
            raise ImportError("PyArrow required for Parquet format")
        if format == 'feather' and not ARROW_AVAILABLE:
            raise ImportError("PyArrow required for Feather format")
        if format == 'excel' and not EXCEL_AVAILABLE:
            warnings.warn("openpyxl not available, Excel support may be limited")
        if format == 'mat' and not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for MATLAB format")
        if format == 'msgpack' and not MSGPACK_AVAILABLE:
            raise ImportError("msgpack required for MessagePack format")
            
    def _load_data(self, path: Path, format: str, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load data from file"""
        if format == 'csv':
            return self._load_csv(path, **kwargs)
        elif format == 'json':
            return self._load_json(path, **kwargs)
        elif format == 'excel':
            return self._load_excel(path, **kwargs)
        elif format == 'parquet':
            return self._load_parquet(path, **kwargs)
        elif format == 'feather':
            return self._load_feather(path, **kwargs)
        elif format == 'hdf5':
            return self._load_hdf5(path, **kwargs)
        elif format == 'mat':
            return self._load_mat(path, **kwargs)
        elif format == 'msgpack':
            return self._load_msgpack(path, **kwargs)
        elif format == 'pickle':
            return self._load_pickle(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _save_data(self, data: Union[pd.DataFrame, Dict], path: Path, 
                   format: str, compression: Optional[str], **kwargs):
        """Save data to file"""
        # Create output directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._save_csv(data, path, compression, **kwargs)
        elif format == 'json':
            self._save_json(data, path, compression, **kwargs)
        elif format == 'excel':
            self._save_excel(data, path, **kwargs)
        elif format == 'parquet':
            self._save_parquet(data, path, compression, **kwargs)
        elif format == 'feather':
            self._save_feather(data, path, compression, **kwargs)
        elif format == 'hdf5':
            self._save_hdf5(data, path, compression, **kwargs)
        elif format == 'mat':
            self._save_mat(data, path, **kwargs)
        elif format == 'msgpack':
            self._save_msgpack(data, path, **kwargs)
        elif format == 'pickle':
            self._save_pickle(data, path, compression, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    # CSV methods
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        if self.verbose:
            print(f"Loading CSV from {path}")
            
        # Default CSV options
        options = {
            'index_col': kwargs.get('index_col', 0),
            'parse_dates': kwargs.get('parse_dates', True),
            'encoding': kwargs.get('encoding', 'utf-8')
        }
        
        df = pd.read_csv(path, **options)
        self.metadata['data_shape'] = df.shape
        
        return df
        
    def _save_csv(self, data: Union[pd.DataFrame, Dict], path: Path, 
                  compression: Optional[str], **kwargs):
        """Save to CSV file"""
        if isinstance(data, dict):
            # Convert dict to DataFrame
            data = pd.DataFrame(data)
            
        if compression:
            path = path.with_suffix(f'.csv.{compression}')
            
        if self.verbose:
            print(f"Saving CSV to {path} (compression: {compression or 'none'})")
            
        data.to_csv(path, compression=compression, **kwargs)
        
    # JSON methods
    def _load_json(self, path: Path, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load JSON file"""
        if self.verbose:
            print(f"Loading JSON from {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Try to convert to DataFrame if possible
        if kwargs.get('as_dataframe', True):
            try:
                return pd.DataFrame(data)
            except:
                return data
        
        return data
        
    def _save_json(self, data: Union[pd.DataFrame, Dict], path: Path,
                   compression: Optional[str], **kwargs):
        """Save to JSON file"""
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict
            orient = kwargs.get('orient', 'records')
            data = data.to_dict(orient=orient)
            
        if compression:
            import gzip
            path = path.with_suffix('.json.gz')
            
        if self.verbose:
            print(f"Saving JSON to {path}")
            
        if compression == 'gzip':
            import gzip
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
    # Excel methods
    def _load_excel(self, path: Path, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load Excel file"""
        if self.verbose:
            print(f"Loading Excel from {path}")
            
        sheet_name = kwargs.get('sheet_name', None)
        
        if sheet_name is None:
            # Load all sheets
            xl_file = pd.ExcelFile(path)
            data = {}
            for sheet in xl_file.sheet_names:
                data[sheet] = pd.read_excel(xl_file, sheet_name=sheet)
            return data
        else:
            return pd.read_excel(path, sheet_name=sheet_name)
            
    def _save_excel(self, data: Union[pd.DataFrame, Dict], path: Path, **kwargs):
        """Save to Excel file"""
        if self.verbose:
            print(f"Saving Excel to {path}")
            
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name='Sheet1', **kwargs)
            elif isinstance(data, dict):
                for sheet_name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=str(sheet_name)[:31], **kwargs)
                        
    # Parquet methods
    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        if self.verbose:
            print(f"Loading Parquet from {path}")
            
        return pd.read_parquet(path, **kwargs)
        
    def _save_parquet(self, data: Union[pd.DataFrame, Dict], path: Path,
                      compression: Optional[str], **kwargs):
        """Save to Parquet file"""
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        if self.verbose:
            print(f"Saving Parquet to {path} (compression: {compression or 'snappy'})")
            
        data.to_parquet(path, compression=compression or 'snappy', **kwargs)
        
    # Feather methods
    def _load_feather(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Feather file"""
        if self.verbose:
            print(f"Loading Feather from {path}")
            
        return pd.read_feather(path, **kwargs)
        
    def _save_feather(self, data: Union[pd.DataFrame, Dict], path: Path,
                      compression: Optional[str], **kwargs):
        """Save to Feather file"""
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        if self.verbose:
            print(f"Saving Feather to {path} (compression: {compression or 'lz4'})")
            
        data.to_feather(path, compression=compression or 'lz4', **kwargs)
        
    # HDF5 methods
    def _load_hdf5(self, path: Path, **kwargs) -> Dict:
        """Load HDF5 file"""
        if self.verbose:
            print(f"Loading HDF5 from {path}")
            
        data = {}
        with h5py.File(path, 'r') as f:
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[:]
                    
            f.visititems(extract_data)
            
        return data
        
    def _save_hdf5(self, data: Union[pd.DataFrame, Dict], path: Path,
                   compression: Optional[str], **kwargs):
        """Save to HDF5 file"""
        if self.verbose:
            print(f"Saving HDF5 to {path}")
            
        with h5py.File(path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                # Save DataFrame as group
                grp = f.create_group('dataframe')
                for col in data.columns:
                    grp.create_dataset(col, data=data[col].values,
                                     compression=compression)
                # Save index
                grp.create_dataset('_index', data=data.index.values)
                
                # Save metadata
                grp.attrs['shape'] = data.shape
                grp.attrs['columns'] = list(data.columns)
                
            elif isinstance(data, dict):
                # Save dict recursively
                def save_dict(group, d):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            subgroup = group.create_group(key)
                            save_dict(subgroup, value)
                        elif isinstance(value, (np.ndarray, list)):
                            group.create_dataset(key, data=value,
                                               compression=compression)
                        elif isinstance(value, pd.DataFrame):
                            subgroup = group.create_group(key)
                            for col in value.columns:
                                subgroup.create_dataset(col, data=value[col].values,
                                                      compression=compression)
                        else:
                            # Store as attribute for scalar values
                            group.attrs[key] = value
                            
                save_dict(f, data)
                
            # Add converter metadata
            f.attrs['converter_metadata'] = json.dumps(self.metadata)
            
    # MATLAB methods
    def _load_mat(self, path: Path, **kwargs) -> Dict:
        """Load MATLAB .mat file"""
        if self.verbose:
            print(f"Loading MATLAB from {path}")
            
        mat_data = sio.loadmat(path, **kwargs)
        
        # Clean up MATLAB-specific entries
        data = {}
        for key, value in mat_data.items():
            if not key.startswith('__'):
                data[key] = value
                
        return data
        
    def _save_mat(self, data: Union[pd.DataFrame, Dict], path: Path, **kwargs):
        """Save to MATLAB .mat file"""
        if self.verbose:
            print(f"Saving MATLAB to {path}")
            
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict of arrays
            mat_data = {col: data[col].values for col in data.columns}
            mat_data['_index'] = data.index.values
            mat_data['_columns'] = np.array(data.columns.tolist())
        else:
            mat_data = data
            
        sio.savemat(path, mat_data, **kwargs)
        
    # MessagePack methods
    def _load_msgpack(self, path: Path, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load MessagePack file"""
        if self.verbose:
            print(f"Loading MessagePack from {path}")
            
        with open(path, 'rb') as f:
            data = msgpack.unpack(f, raw=False)
            
        return data
        
    def _save_msgpack(self, data: Union[pd.DataFrame, Dict], path: Path, **kwargs):
        """Save to MessagePack file"""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
            
        if self.verbose:
            print(f"Saving MessagePack to {path}")
            
        with open(path, 'wb') as f:
            msgpack.pack(data, f, **kwargs)
            
    # Pickle methods
    def _load_pickle(self, path: Path, **kwargs) -> Any:
        """Load pickle file"""
        if self.verbose:
            print(f"Loading pickle from {path}")
            
        return pd.read_pickle(path, **kwargs)
        
    def _save_pickle(self, data: Any, path: Path, compression: Optional[str], **kwargs):
        """Save to pickle file"""
        if self.verbose:
            print(f"Saving pickle to {path}")
            
        pd.to_pickle(data, path, compression=compression, **kwargs)
        
    def batch_convert(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                     input_format: str, output_format: str, 
                     pattern: str = '*', **kwargs):
        """
        Batch convert multiple files
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            input_format: Input format
            output_format: Output format
            pattern: File pattern to match (default: *)
            **kwargs: Additional conversion options
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        if input_format in ['hdf5', 'h5']:
            files = list(input_dir.glob(f"{pattern}.h5")) + list(input_dir.glob(f"{pattern}.hdf5"))
        else:
            files = list(input_dir.glob(f"{pattern}.{input_format}"))
            
        if not files:
            print(f"No {input_format} files found matching pattern: {pattern}")
            return
            
        print(f"Found {len(files)} files to convert")
        
        # Convert each file
        success_count = 0
        for i, input_file in enumerate(files, 1):
            try:
                # Generate output filename
                output_name = input_file.stem
                output_ext = self._detect_format(Path(f"dummy.{output_format}"))
                output_file = output_dir / f"{output_name}.{output_format}"
                
                print(f"\n[{i}/{len(files)}] Converting {input_file.name}...")
                self.convert(input_file, output_file, input_format, output_format, **kwargs)
                success_count += 1
                
            except Exception as e:
                print(f"  ERROR: {e}")
                
        print(f"\nBatch conversion complete: {success_count}/{len(files)} successful")
        
    def validate_conversion(self, original_path: Path, converted_path: Path,
                          tolerance: float = 1e-6) -> bool:
        """
        Validate that conversion preserved data integrity
        
        Args:
            original_path: Path to original file
            converted_path: Path to converted file
            tolerance: Numerical tolerance for comparisons
            
        Returns:
            True if data matches within tolerance
        """
        print(f"Validating conversion...")
        
        # Load both files
        original_format = self._detect_format(original_path)
        converted_format = self._detect_format(converted_path)
        
        original_data = self._load_data(original_path, original_format)
        converted_data = self._load_data(converted_path, converted_format)
        
        # Convert to comparable format
        if isinstance(original_data, pd.DataFrame) and isinstance(converted_data, pd.DataFrame):
            # Compare DataFrames
            try:
                pd.testing.assert_frame_equal(original_data, converted_data,
                                            check_exact=False, atol=tolerance)
                print("✓ Data validation passed")
                return True
            except AssertionError as e:
                print(f"✗ Data validation failed: {e}")
                return False
                
        elif isinstance(original_data, dict) and isinstance(converted_data, dict):
            # Compare dictionaries
            if set(original_data.keys()) != set(converted_data.keys()):
                print(f"✗ Key mismatch: {set(original_data.keys()) ^ set(converted_data.keys())}")
                return False
                
            for key in original_data:
                if isinstance(original_data[key], np.ndarray):
                    if not np.allclose(original_data[key], converted_data[key], atol=tolerance):
                        print(f"✗ Array mismatch for key: {key}")
                        return False
                        
            print("✓ Data validation passed")
            return True
            
        else:
            print("✗ Cannot validate: incompatible data types")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert APACC-Sim results between different formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Supported formats:
  csv      - Comma-separated values
  json     - JavaScript Object Notation
  excel    - Microsoft Excel (.xlsx)
  parquet  - Apache Parquet (columnar)
  feather  - Feather (fast serialization)
  hdf5     - Hierarchical Data Format 5
  mat      - MATLAB workspace
  msgpack  - MessagePack binary format
  pickle   - Python pickle format

Examples:
  %(prog)s results.csv results.parquet                    # Convert CSV to Parquet
  %(prog)s data.h5 output.json -c gzip                   # Convert HDF5 to compressed JSON
  %(prog)s -b ./logs ./converted csv parquet             # Batch convert all CSVs
  %(prog)s results.xlsx results.csv --sheet-name Sheet2  # Convert specific Excel sheet
        '''
    )
    
    # Single file conversion
    parser.add_argument('input', nargs='?', help='Input file path')
    parser.add_argument('output', nargs='?', help='Output file path')
    
    # Batch conversion
    parser.add_argument('-b', '--batch', nargs=4, metavar=('IN_DIR', 'OUT_DIR', 'IN_FMT', 'OUT_FMT'),
                       help='Batch convert: input_dir output_dir input_format output_format')
    parser.add_argument('--pattern', default='*',
                       help='File pattern for batch conversion (default: *)')
    
    # Format options
    parser.add_argument('-i', '--input-format', help='Input format (auto-detect if not specified)')
    parser.add_argument('-o', '--output-format', help='Output format (auto-detect if not specified)')
    parser.add_argument('-c', '--compression', 
                       choices=['gzip', 'bz2', 'xz', 'zstd', 'snappy', 'lz4'],
                       help='Compression method (format-dependent)')
    
    # Validation
    parser.add_argument('--validate', action='store_true',
                       help='Validate conversion by comparing data')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Numerical tolerance for validation (default: 1e-6)')
    
    # Excel options
    parser.add_argument('--sheet-name', help='Excel sheet name to read/write')
    parser.add_argument('--index-col', type=int, default=0,
                       help='Column to use as index (default: 0)')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--list-formats', action='store_true',
                       help='List all supported formats')
    
    args = parser.parse_args()
    
    # Create converter
    converter = ResultsConverter(verbose=args.verbose)
    
    # List formats
    if args.list_formats:
        print("Supported formats:")
        for fmt, desc in converter.FORMATS.items():
            available = True
            if fmt == 'parquet' and not ARROW_AVAILABLE:
                available = False
            elif fmt == 'mat' and not SCIPY_AVAILABLE:
                available = False
            elif fmt == 'msgpack' and not MSGPACK_AVAILABLE:
                available = False
                
            status = "✓" if available else "✗"
            print(f"  {status} {fmt:<10} - {desc}")
        return
        
    # Batch conversion
    if args.batch:
        in_dir, out_dir, in_fmt, out_fmt = args.batch
        kwargs = {
            'compression': args.compression,
            'sheet_name': args.sheet_name,
            'index_col': args.index_col
        }
        converter.batch_convert(in_dir, out_dir, in_fmt, out_fmt,
                              pattern=args.pattern, **kwargs)
        return
        
    # Single file conversion
    if not args.input or not args.output:
        parser.error("Input and output files required (or use --batch)")
        
    # Convert
    kwargs = {
        'sheet_name': args.sheet_name,
        'index_col': args.index_col
    }
    
    converter.convert(
        args.input,
        args.output,
        input_format=args.input_format,
        output_format=args.output_format,
        compression=args.compression,
        **kwargs
    )
    
    # Validate if requested
    if args.validate:
        valid = converter.validate_conversion(
            Path(args.input),
            Path(args.output),
            tolerance=args.tolerance
        )
        sys.exit(0 if valid else 1)


if __name__ == '__main__':
    main()