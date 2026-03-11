
"""
data_preprocessing.py
=====================
Data preprocessing pipeline for the COVID-19 household secondary transmission study.

This module processes Swedish national register data (SWECOV) through four stages:

    1. Time-based filtering  – restrict raw CSV tables to the study period (2020).
    2. Feature selection     – retain only the columns required for downstream analysis.
    3. CSV-to-dict conversion – convert flat CSVs to person-keyed pickle dictionaries
                                for fast lookup during feature engineering.
    4. Population index generation – classify every household member as index case,
                                     secondary case, co-primary case, or uninfected,
                                     and apply death-date consistency filtering.

Key design choices
------------------
- Chunked I/O throughout to handle GB-scale files within available RAM.
- Automatic garbage collection and memory monitoring at configurable intervals.
- All output artefacts (filtered CSVs, pickle dicts, index files) are written to
  directories specified in the ``Config`` class constants.

Usage
-----
Run as a script to execute the full pipeline::

    python data_preprocessing.py

Individual steps can also be called programmatically; see each function's docstring.
"""

import os
import gc
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from tqdm import tqdm
import psutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class Config:
    """Configuration parameters for the preprocessing pipeline"""
    
    # Processing parameters
    CHUNK_SIZE = 100000000
    ENCODING = 'latin1'
    
    # Transmission parameters
    INCUBATION_DAYS = 14  # Days after first case to consider as secondary
    EXPOSURE_DAYS = 2     # Days after first case to consider as co-primary
    
    # Directory paths
    INPUT_DIR_RAW = '/path/to/raw_register_data'
    INPUT_DIR_FILTERED = 'Time_Filtered_Data'
    INPUT_DIR_FEATURES = 'Features_Selected_Data'
    OUTPUT_DIR_FILTERED = 'Time_Filtered_Data'
    OUTPUT_DIR_FEATURES = 'Features_Selected_Data'
    OUTPUT_DIR_INDEX = f'Index(InD={INCUBATION_DAYS},ExD={EXPOSURE_DAYS})'
    
    # Time filtering configurations for multiple tables
    TIME_FILTER_CONFIGS = {
        'FHM_SMINET.csv': {
            'output_name': 'FHM_SMINET_2020.csv',
            'date_column': 'Statistikdatum',
            'start_date': '2019-12-31',
            'end_date': '2020-12-31',
            'date_format': '%Y-%m-%d',
            'assume_sorted': False
        },
        'Inera_VPTU_Coronadata.csv': {
            'output_name': 'Inera_VPTU_Coronadata_2019_2020.csv',
            'date_column': 'documentCreatedTime',
            'start_date': '20191231000000',
            'end_date': '20201231000000',
            'date_format': '%Y%m%d%H%M%S',
            'assume_sorted': False
        },
        'SWECOV_SOS_LMED.csv': {
            'output_name': 'SWECOV_SOS_LMED_2019_2020.csv',
            'date_column': 'EDATUM',
            'start_date': '20191231',
            'end_date': '20201231',
            'date_format': '%Y%m%d',
            'assume_sorted': True
        },
        'SWECOV_SOS_DORS.csv': {
            'output_name': 'SWECOV_SOS_DORS_2020.csv',
            'date_column': 'DODSDAT',
            'start_date': '20191231',
            'end_date': '20201231',
            'date_format': '%Y%m%d',
            'assume_sorted': False
        },
        'SWECOV_SOS_SOL.csv': {
            'output_name': 'SWECOV_SOS_SOL_2018_2020.csv',
            'date_column': 'PERIOD',
            'start_date': '201901',
            'end_date': '202012',
            'date_format': '%Y%m',
            'assume_sorted': False
        }
    }


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in GB
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return {
        'process_rss_gb': mem_info.rss / (1024 ** 3),
        'process_vms_gb': mem_info.vms / (1024 ** 3),
        'system_used_gb': virtual_mem.used / (1024 ** 3),
        'system_available_gb': virtual_mem.available / (1024 ** 3),
        'system_percent': virtual_mem.percent
    }


def print_memory_usage(label: str = "") -> None:
    """
    Print current memory usage with optional label.
    
    Args:
        label: Descriptive label for the memory check
    """
    mem = get_memory_usage()
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Memory Usage:")
    print(f"  Process RSS: {mem['process_rss_gb']:.2f} GB")
    print(f"  System Used: {mem['system_used_gb']:.2f} GB / Available: {mem['system_available_gb']:.2f} GB")
    print(f"  System Usage: {mem['system_percent']:.1f}%")


def force_garbage_collection() -> None:
    """
    Force garbage collection to free memory.
    """
    gc.collect()
    

def clear_memory(*objects) -> None:
    """
    Clear memory by deleting objects and forcing garbage collection.
    
    Args:
        *objects: Variable number of objects to delete
    """
    for obj in objects:
        try:
            del obj
        except:
            pass
    force_garbage_collection()


def optimize_dataframe_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print optimization results
        
    Returns:
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Integer optimization
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            # Float optimization
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"  Memory optimization: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


# ============================================================================
# STEP 1: TIME-BASED FILTERING
# ============================================================================

def filter_data_by_time_range(
    input_path: str,
    output_path: str,
    date_column: str,
    start_date: str,
    end_date: str,
    date_format: str = '%Y-%m-%d',
    chunk_size: int = Config.CHUNK_SIZE,
    encoding: str = Config.ENCODING,
    assume_sorted: bool = False  # NEW parameter
) -> None:
    """
    Filter CSV data by date range using chunked processing for memory efficiency.
    Supports multiple date formats.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save filtered CSV file
        date_column: Name of the date column to filter on
        start_date: Start date as string
        end_date: End date as string
        date_format: Format of the date column (e.g., '%Y-%m-%d', '%Y%m%d')
        chunk_size: Number of rows to process per chunk
        encoding: File encoding
        assume_sorted: If True, assumes data is sorted by date and stops early when exceeding end_date
    """
    print(f"\n{'='*60}")
    print(f"Filtering {os.path.basename(input_path)}")
    print(f"Date column: {date_column}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Date format: {date_format}")
    if assume_sorted:
        print(f"⚡ OPTIMIZATION: Assuming sorted data (early termination enabled)")
    print(f"{'='*60}")
    print_memory_usage("Before filtering")
    
    filtered_chunks = []
    total_rows = 0
    filtered_rows = 0
    chunks_processed = 0
    found_start = False  # Track if we've found data in range
    
    # Convert start and end dates to datetime objects
    start_datetime = pd.to_datetime(start_date, format=date_format)
    end_datetime = pd.to_datetime(end_date, format=date_format)
    
    try:
        for i, chunk in enumerate(pd.read_csv(
            input_path,
            chunksize=chunk_size,
            encoding=encoding,
            low_memory=False
        )):
            chunks_processed = i + 1
            print(f'Processing chunk {chunks_processed} ({len(chunk):,} rows)...')
            total_rows += len(chunk)
            
            # Convert date column to datetime
            chunk[date_column] = pd.to_datetime(chunk[date_column], format=date_format, errors='coerce')
            
            # If assuming sorted data, check for early termination
            if assume_sorted and len(chunk) > 0:
                chunk_min = chunk[date_column].min()
                chunk_max = chunk[date_column].max()
                
                # Skip chunks entirely before start date
                if chunk_max < start_datetime:
                    print(f"  ⏭️  Skipping (all dates before {start_date})")
                    del chunk
                    continue
                
                # Stop processing if all dates are after end date
                if chunk_min > end_datetime:
                    if found_start:
                        print(f"  🛑 Early termination (all remaining dates after {end_date})")
                        del chunk
                        break
                    else:
                        print(f"  ⏭️  Skipping (all dates after {end_date})")
                        del chunk
                        continue
            
            # Filter by date range
            filtered_chunk = chunk[
                (chunk[date_column] >= start_datetime) & 
                (chunk[date_column] <= end_datetime)
            ]
            
            if not filtered_chunk.empty:
                found_start = True
                filtered_rows += len(filtered_chunk)
                filtered_chunks.append(filtered_chunk)
                print(f"  ✓ Retained {len(filtered_chunk):,} rows")
            else:
                print(f"  ⏭️  No data in range")
            
            # Clear chunk from memory
            del chunk, filtered_chunk
            
            # Force garbage collection every 5 chunks
            if chunks_processed % 5 == 0:
                force_garbage_collection()
                print_memory_usage(f"After chunk {chunks_processed}")
        
        # Combine all filtered chunks
        if filtered_chunks:
            print(f"\nCombining {len(filtered_chunks)} filtered chunks...")
            filtered_data = pd.concat(filtered_chunks, ignore_index=True)
            
            # Clear chunk list from memory
            del filtered_chunks
            force_garbage_collection()
            
            # Optimize memory before saving
            filtered_data = optimize_dataframe_memory(filtered_data, verbose=True)
            
            # Save filtered data
            print(f"Saving filtered data to {output_path}...")
            filtered_data.to_csv(output_path, index=False, encoding=encoding)
            
            print(f"\nFiltering complete:")
            print(f"  Total rows processed: {total_rows:,}")
            print(f"  Rows retained: {filtered_rows:,} ({filtered_rows/total_rows*100:.2f}%)")
            print(f"  Rows removed: {total_rows - filtered_rows:,}")
            print(f"  Saved to: {output_path}")
            
            # Clear filtered data from memory
            del filtered_data
        else:
            print("No data found in specified date range")
        
        # Final cleanup
        force_garbage_collection()
        print_memory_usage("After filtering (cleaned)")
            
    except Exception as e:
        print(f"Error filtering data: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        clear_memory(filtered_chunks)
        raise


def filter_all_tables_by_time() -> None:
    """
    Filter all tables defined in TIME_FILTER_CONFIGS by their respective time ranges.
    """
    print(f"\n{'='*80}")
    print("TIME-BASED FILTERING FOR ALL TABLES")
    print(f"{'='*80}")
    
    os.makedirs(Config.OUTPUT_DIR_FILTERED, exist_ok=True)
    
    for input_filename, config in Config.TIME_FILTER_CONFIGS.items():
        input_path = os.path.join(Config.INPUT_DIR_RAW, input_filename)
        output_path = os.path.join(Config.OUTPUT_DIR_FILTERED, config['output_name'])
        
        if not os.path.exists(input_path):
            print(f"\nWarning: File not found: {input_filename}")
            continue
        
        try:
            filter_data_by_time_range(
                input_path=input_path,
                output_path=output_path,
                date_column=config['date_column'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                date_format=config['date_format'],
                assume_sorted=config.get('assume_sorted', False)  # NEW parameter
            )
        except Exception as e:
            print(f"Error processing {input_filename}: {str(e)}")
            continue
        
        # Cleanup between tables
        force_garbage_collection()


# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================

class TableSchema:
    """Defines schema for different data tables"""
    
    SCHEMAS = {
        'FHM_SMINET_2020.csv': ['P1105_LopNr_PersonNr', 'Statistikdatum'],
        'Population_PersonNr_20221231.csv': ['P1105_LopNr_PersonNr', 'FodelseArMan', 'Kon'],
        'Fodelseuppg_20201231.csv': ['P1105_LopNr_PersonNr', 'UtlSvBakg', 'Fodelseland'],
        'HushallBoende_2019.csv': ['P1105_LopNr_PersonNr', 'Boarea_Person', 'Boendeform'],
        'HushallPerson_2019.csv': ['P1105_LopNr_PersonNr', 'P1105_LopNr_Hushallsid_2019', 'AntalBarnUnder18'],
        'Individ_2019.csv': ['P1105_LopNr_PersonNr', 'DispInk04', 'DispInkFam04'],
        'Inera_VPTU_Coronadata_2019_2020.csv': ['P1105_LopNr_PersonNr', 'documentCreatedTime', 'contactReason'],
        'SWECOV_SOS_LMED_2019_2020.csv': ['P1105_LopNr_PersonNr', 'ATC', 'EDATUM'],
        'SWECOV_SOS_OV.csv': ['P1105_LopNr_PersonNr', 'hdia', 'INDATUMA'],
        'SWECOV_SOS_SV.csv': ['P1105_LopNr_PersonNr', 'hdia', 'INDATUMA'],
        'SWECOV_SOS_DORS_2020.csv': ['P1105_LopNr_PersonNr', 'DODSDAT'],
        'SWECOV_SOS_SOL_2018_2020.csv': ['P1105_LopNr_PersonNr', 'PERIOD', 'TRYGG'],
    }


def select_features_from_table(
    filename: str,
    columns: List[str],
    input_dir: str = Config.INPUT_DIR_FILTERED,
    output_dir: str = Config.OUTPUT_DIR_FEATURES,
    chunk_size: int = Config.CHUNK_SIZE,
    encoding: str = Config.ENCODING
) -> None:
    """
    Select specific columns from a CSV file and save to output directory.
    Uses streaming to avoid loading entire file into memory.
    
    Args:
        filename: Name of the input CSV file
        columns: List of column names to select
        input_dir: Directory containing input files
        output_dir: Directory to save output files
        chunk_size: Number of rows to process per chunk
        encoding: File encoding
    """
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(input_path):
        print(f"File not found: {filename}")
        return
    
    print(f"\n{'='*60}")
    print(f"Selecting features from: {filename}")
    print(f"Columns: {columns}")
    print(f"{'='*60}")
    print_memory_usage("Before processing")
    
    try:
        # Verify columns exist in file
        header = pd.read_csv(input_path, encoding=encoding, nrows=0)
        missing_cols = [col for col in columns if col not in header.columns]
        
        if missing_cols:
            print(f"Error: Columns {missing_cols} not found in {filename}")
            return
        
        # Clear header from memory
        del header
        
        # Process file in chunks
        first_chunk = True
        chunk_count = 0
        total_rows = 0
        
        for chunk in pd.read_csv(
            input_path,
            encoding=encoding,
            usecols=columns,
            chunksize=chunk_size
        ):
            chunk.to_csv(
                output_path,
                mode='a',
                index=False,
                header=first_chunk,
                encoding=encoding
            )
            first_chunk = False
            chunk_count += 1
            total_rows += len(chunk)
            print(f"  Chunk {chunk_count}: {len(chunk):,} rows")
            
            # Clear chunk from memory
            del chunk
            
            # Force garbage collection every 5 chunks
            if chunk_count % 5 == 0:
                force_garbage_collection()
        
        print(f"\nCompleted: {filename}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Saved to: {output_path}")
        
        # Final cleanup
        force_garbage_collection()
        print_memory_usage("After processing (cleaned)")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise


def select_features_all_tables(
    schemas: Dict[str, List[str]] = TableSchema.SCHEMAS,
    **kwargs
) -> None:
    """
    Process all tables defined in schemas.
    
    Args:
        schemas: Dictionary mapping filenames to column lists
        **kwargs: Additional arguments passed to select_features_from_table
    """
    os.makedirs(Config.OUTPUT_DIR_FEATURES, exist_ok=True)
    
    for filename, columns in schemas.items():
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        select_features_from_table(filename, columns, **kwargs)
        
        # Cleanup between tables
        force_garbage_collection()


# ============================================================================
# STEP 3: CSV TO DICTIONARY CONVERSION
# ============================================================================

class DictionarySchema:
    """Defines value formatting for dictionary conversion"""
    
    SCHEMAS = {
        'FHM_SMINET_2020.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'Statistikdatum'],
            'value_format': lambda row: {'Statistikdatum': row['Statistikdatum']}
        },
        'Fodelseuppg_20201231.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'UtlSvBakg', 'Fodelseland'],
            'value_format': lambda row: {'UtlSvBakg': row['UtlSvBakg'], 'Fodelseland': row['Fodelseland']}
        },
        'Population_PersonNr_20221231.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'FodelseArMan', 'Kon'],
            'value_format': lambda row: {'FodelseArMan': row['FodelseArMan'], 'Kon': row['Kon']}
        },
        'HushallPerson_2019.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'P1105_LopNr_Hushallsid_2019', 'AntalBarnUnder18'],
            'value_format': lambda row: {
                'P1105_LopNr_Hushallsid_2019': row['P1105_LopNr_Hushallsid_2019'],
                'AntalBarnUnder18': row['AntalBarnUnder18']
            }
        },
        'HushallBoende_2019.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'Boarea_Person', 'Boendeform'],
            'value_format': lambda row: {'Boarea_Person': row['Boarea_Person'], 'Boendeform': row['Boendeform']}
        },
        'Individ_2019.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'DispInk04', 'DispInkFam04'],
            'value_format': lambda row: {'DispInk04': row['DispInk04'], 'DispInkFam04': row['DispInkFam04']}
        },
        'Inera_VPTU_Coronadata_2019_2020.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'documentCreatedTime', 'contactReason'],
            'value_format': lambda row: {
                'documentCreatedTime': row['documentCreatedTime'],
                'contactReason': row['contactReason']
            }
        },
        'SWECOV_SOS_SOL_2018_2020.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'PERIOD', 'TRYGG'],
            'value_format': lambda row: {'PERIOD': row['PERIOD'], 'TRYGG': row['TRYGG']}
        },
        'SWECOV_SOS_DORS_2020.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'DODSDAT'],
            'value_format': lambda row: {'DODSDAT': row['DODSDAT']}
        },
        'SWECOV_SOS_LMED_2019_2020.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'ATC', 'EDATUM'],
            'value_format': lambda row: {'CodeDate': row['EDATUM'], 'Code': row['ATC']}
        },
        'SWECOV_SOS_OV.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'hdia', 'INDATUMA'],
            'value_format': lambda row: {'CodeDate': row['INDATUMA'], 'Code': row['hdia']}
        },
        'SWECOV_SOS_SV.csv': {
            'columns': ['P1105_LopNr_PersonNr', 'hdia', 'INDATUMA'],
            'value_format': lambda row: {'CodeDate': row['INDATUMA'], 'Code': row['hdia']}
        },
    }


def convert_csv_to_dictionary(
    filename: str,
    columns: List[str],
    value_format: Callable,
    input_dir: str = Config.INPUT_DIR_FEATURES,
    output_dir: str = Config.OUTPUT_DIR_FEATURES,
    chunk_size: int = Config.CHUNK_SIZE,
    encoding: str = Config.ENCODING
) -> None:
    """
    Convert CSV file to dictionary format with person ID as key.
    Handles multiple records per person by storing them in a list.
    Uses chunked processing and periodic memory cleanup for GB-scale files.
    
    NOTE: This function does not return the dictionary to save memory.
    The dictionary is saved to disk and should be loaded when needed.
    
    Args:
        filename: Name of the input CSV file
        columns: List of columns to process
        value_format: Function to format row values
        input_dir: Directory containing input files
        output_dir: Directory to save output files
        chunk_size: Number of rows to process per chunk
        encoding: File encoding
    """
    input_path = os.path.join(input_dir, filename)
    output_filename = filename.replace('.csv', '_duplicates.pkl')
    output_path = os.path.join(output_dir, output_filename)
    
    if not os.path.exists(input_path):
        print(f"File not found: {filename}")
        return
    
    print(f"\n{'='*60}")
    print(f"Converting to dictionary: {filename}")
    print(f"{'='*60}")
    print_memory_usage("Before conversion")
    
    try:
        # Verify columns exist
        header = pd.read_csv(input_path, nrows=0, encoding=encoding)
        missing_cols = [col for col in columns if col not in header.columns]
        if missing_cols:
            print(f"Error: Columns {missing_cols} not found in {filename}")
            return
        
        # Clear header from memory
        del header
        
        # Initialize result dictionary
        result_dict = {}
        chunk_count = 0
        total_rows = 0
        
        # Process file in chunks
        for chunk in pd.read_csv(
            input_path,
            usecols=columns,
            chunksize=chunk_size,
            encoding=encoding,
            low_memory=False
        ):
            # Drop rows with missing person ID
            chunk = chunk.dropna(subset=['P1105_LopNr_PersonNr'])
            
            # Process each row
            for _, row in chunk.iterrows():
                person_id = row['P1105_LopNr_PersonNr']
                value = value_format(row)
                
                # Append to existing list or create new list
                if person_id in result_dict:
                    result_dict[person_id].append(value)
                else:
                    result_dict[person_id] = [value]
            
            chunk_count += 1
            total_rows += len(chunk)
            print(f"  Chunk {chunk_count}: {len(chunk):,} rows processed")
            
            # Clear chunk from memory
            del chunk
            
            # Force garbage collection and print memory every 5 chunks
            if chunk_count % 5 == 0:
                force_garbage_collection()
                print_memory_usage(f"After chunk {chunk_count}")
        
        # Save dictionary to pickle
        print(f"\nSaving dictionary to {output_filename}...")
        with open(output_path, 'wb') as f:
            pickle.dump(result_dict, f)
        
        # Report statistics
        total_ids = len(result_dict)
        duplicate_ids = sum(1 for values in result_dict.values() if len(values) > 1)
        
        print(f"\nConversion complete:")
        print(f"  Total rows processed: {total_rows:,}")
        print(f"  Unique person IDs: {total_ids:,}")
        print(f"  IDs with multiple records: {duplicate_ids:,} ({duplicate_ids/total_ids*100:.1f}%)")
        print(f"  Saved to: {output_path}")
        
        # Clear dictionary and force cleanup
        del result_dict
        force_garbage_collection()
        print_memory_usage("After conversion (cleaned)")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise


def convert_all_csv_to_dictionaries(
    schemas: Dict[str, Dict] = DictionarySchema.SCHEMAS,
    **kwargs
) -> None:
    """
    Convert all CSV files defined in schemas to dictionary format.
    
    Args:
        schemas: Dictionary mapping filenames to schema definitions
        **kwargs: Additional arguments passed to convert_csv_to_dictionary
    """
    for filename, schema in schemas.items():
        print(f"\n{'='*80}")
        print(f"CONVERTING: {filename}")
        print(f"{'='*80}")
        convert_csv_to_dictionary(
            filename,
            schema['columns'],
            schema['value_format'],
            **kwargs
        )
        
        # Cleanup between files
        force_garbage_collection()


# ============================================================================
# STEP 4: POPULATION INDEX GENERATION
# ============================================================================

def generate_population_index(
    population_df: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """
    Generate source population index by filtering invalid IDs.
    
    Filters out:
    - Reused IDs (AterAnv == 1)
    - Multiple-linked IDs (LopNrByte > 1)
    - Incorrect IDs (FelPersonNr == 1)
    
    Args:
        population_df: DataFrame from Population_PersonNr_20221231
        
    Returns:
        Tuple of (filtered_index, original_count, filtered_count)
    """
    print(f"\n{'='*60}")
    print("POPULATION INDEX GENERATION")
    print(f"{'='*60}")
    print_memory_usage("Before processing")
    
    required_cols = ['P1105_LopNr_PersonNr', 'AterAnv', 'LopNrByte', 'FelPersonNr']
    if not all(col in population_df.columns for col in required_cols):
        raise ValueError("Missing required columns in Population_PersonNr_20221231")
    
    original_count = len(population_df)
    
    # Filter out invalid IDs
    filtered_df = population_df[
        (population_df['AterAnv'] != 1) & 
        (population_df['LopNrByte'] <= 1) & 
        (population_df['FelPersonNr'] != 1)
    ].copy()
    
    filtered_count = len(filtered_df)
    
    # Create index
    population_index = filtered_df[['P1105_LopNr_PersonNr']].set_index('P1105_LopNr_PersonNr')
    
    # Clear filtered_df
    del filtered_df
    force_garbage_collection()
    
    print(f"\nPopulation index generation:")
    print(f"  Original count: {original_count:,}")
    print(f"  Filtered count: {filtered_count:,}")
    print(f"  Removed: {original_count - filtered_count:,} ({(original_count - filtered_count) / original_count * 100:.2f}%)")
    
    print_memory_usage("After generation (cleaned)")
    
    return population_index, original_count, filtered_count


def generate_2020_population_index(
    population_index: pd.DataFrame,
    rtb2020_df: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """
    Generate 2020 population index by filtering RTB2019 using the source population index.
    
    Args:
        population_index: Filtered population index from generate_population_index
        rtb2020_df: DataFrame from RTB2019
        
    Returns:
        Tuple of (filtered_2020_index, original_count, filtered_count)
    """
    print(f"\n{'='*60}")
    print("2020 POPULATION INDEX GENERATION")
    print(f"{'='*60}")
    print_memory_usage("Before processing")
    
    if 'P1105_LopNr_PersonNr' not in rtb2020_df.columns:
        raise ValueError("Missing 'P1105_LopNr_PersonNr' column in RTB2020")
    
    original_count = len(rtb2020_df)
    
    # Filter to only include valid personal IDs
    filtered_rtb2020 = rtb2020_df[
        rtb2020_df['P1105_LopNr_PersonNr'].isin(population_index.index)
    ].copy()
    
    filtered_count = len(filtered_rtb2020)
    
    # Create 2020 index
    rtb2020_index = filtered_rtb2020[['P1105_LopNr_PersonNr']].set_index('P1105_LopNr_PersonNr')
    
    # Clear filtered_rtb2020
    del filtered_rtb2020
    force_garbage_collection()
    
    print(f"\n2020 population index generation:")
    print(f"  Original count: {original_count:,}")
    print(f"  Filtered count: {filtered_count:,}")
    print(f"  Removed: {original_count - filtered_count:,}")
    
    print_memory_usage("After generation (cleaned)")
    
    return rtb2020_index, original_count, filtered_count


def classify_infected_healthy(
    population_index: pd.DataFrame,
    sminet_pkl_path: str,
    infected_save_path: str,
    healthy_save_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify population into infected and healthy based on FHM_SMINET data.
    Adds 'IndexDate' column with infection date for infected individuals.
    
    Args:
        population_index: Population index DataFrame with P1105_LopNr_PersonNr as index
        sminet_pkl_path: Path to FHM_SMINET_2020_duplicates.pkl
        infected_save_path: Path to save infected DataFrame
        healthy_save_path: Path to save healthy DataFrame
        
    Returns:
        Tuple of (infected_df, healthy_df)
    """
    print(f"\n{'='*60}")
    print("POPULATION CLASSIFICATION (INFECTED/HEALTHY)")
    print(f"{'='*60}")
    print_memory_usage("Before classification")
    
    # Load SmiNet dictionary
    with open(sminet_pkl_path, 'rb') as f:
        sminet_dict = pickle.load(f)
    
    print(f"SmiNet data loaded: {len(sminet_dict):,} entries")
    
    # Add IndexDate column
    population_index['IndexDate'] = None
    for person in population_index.index:
        if person in sminet_dict and sminet_dict[person]:
            first_entry = sminet_dict[person][0]
            if 'Statistikdatum' in first_entry:
                population_index.at[person, 'IndexDate'] = first_entry['Statistikdatum']
    
    # Clear sminet_dict
    del sminet_dict
    force_garbage_collection()
    
    # Classify into infected (with IndexDate) and healthy (without)
    infected_df = population_index[population_index['IndexDate'].notna()].copy()
    healthy_df = population_index[population_index['IndexDate'].isna()].copy()
    
    # Save DataFrames
    with open(infected_save_path, 'wb') as f:
        pickle.dump(infected_df, f)
    with open(healthy_save_path, 'wb') as f:
        pickle.dump(healthy_df, f)
    
    print(f"\nPopulation classification:")
    print(f"  Infected count: {len(infected_df):,}")
    print(f"  Healthy count: {len(healthy_df):,}")
    print(f"  Saved infected to: {infected_save_path}")
    print(f"  Saved healthy to: {healthy_save_path}")
    
    print_memory_usage("After classification (saved)")
    
    return infected_df, healthy_df


def generate_household_based_index(
    infected_df: pd.DataFrame,
    healthy_df: pd.DataFrame,
    hushall_pkl_path: str,
    new_index_path: str,
    remaining_healthy_path: str,
    test_limit: Optional[int] = None,
    incubation_days: int = Config.INCUBATION_DAYS,
    exposure_days: int = Config.EXPOSURE_DAYS
) -> None:
    """
    Generate household-based index classifying household members by transmission role.
    
    Classification logic:
    - Label 1 (Co-primary): Infections within exposure_days of first household case
    - Label 2 (Secondary): Infections after exposure_days but within incubation_days
    - Label 0 (Other): Healthy household members or infections after incubation_days
    
    Only considers the first transmission cluster in each household.
    
    Args:
        infected_df: Infected population with IndexDate column
        healthy_df: Healthy population (index only)
        hushall_pkl_path: Path to HushallPerson_2019_duplicates.pkl
        new_index_path: Path to save household-based index
        remaining_healthy_path: Path to save remaining healthy individuals
        test_limit: Limit processing to N households for testing (None = process all)
        incubation_days: Days after first case to consider as secondary
        exposure_days: Days after first case to consider as co-primary
    """
    print(f"\n{'='*60}")
    print("HOUSEHOLD-BASED INDEX GENERATION")
    print(f"Parameters: Incubation={incubation_days} days, Exposure={exposure_days} days")
    print(f"{'='*60}")
    print_memory_usage("Before processing")
    
    with open(hushall_pkl_path, 'rb') as f:
        hushall_dict = pickle.load(f)
    
    print(f'Household data loaded: {len(hushall_dict):,} entries')
    print_memory_usage("After loading household data")
    
    print('\nBuilding household to members mapping...')
    # Build household to members dictionary
    household_to_members: Dict[str, List[str]] = {}
    for person, entries in hushall_dict.items():
        if entries:
            hid = entries[0]['P1105_LopNr_Hushallsid_2019']
            if hid not in household_to_members:
                household_to_members[hid] = []
            household_to_members[hid].append(person)
    
    print(f'  Total households: {len(household_to_members):,}')
    
    print('\nBuilding infection date mapping...')
    # Convert infection dates to datetime
    infection_dates = {}
    for person, date_str in infected_df['IndexDate'].items():
        if pd.isna(date_str):
            continue
        if isinstance(date_str, str):
            try:
                infection_dates[person] = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
        else:
            infection_dates[person] = date_str
    
    print(f'  Infection dates processed: {len(infection_dates):,}')
    
    # Collect households with at least one infected person
    infected_households = set()
    for person in infection_dates.keys():
        if person in hushall_dict and hushall_dict[person]:
            hid = hushall_dict[person][0]['P1105_LopNr_Hushallsid_2019']
            infected_households.add(hid)
    
    print(f'  Infected households: {len(infected_households):,}')
    
    if test_limit:
        infected_households = set(list(infected_households)[:test_limit])
        print(f'  Limited to {test_limit} households for testing')
    
    # Clear infected_df to save memory (we only need infection_dates now)
    del infected_df
    force_garbage_collection()
    print_memory_usage("After building mappings")
    
    # Initialize sets for tracking
    healthy_set = set(healthy_df.index)
    del healthy_df  # No longer needed
    force_garbage_collection()
    
    added_persons = set()
    new_rows = []
    
    # Process each infected household
    print('\nProcessing households...')
    batch_size = 1000  # Process households in batches for memory management
    households_list = list(infected_households)
    
    for batch_start in range(0, len(households_list), batch_size):
        batch_end = min(batch_start + batch_size, len(households_list))
        batch_households = households_list[batch_start:batch_end]
        
        for hid in tqdm(batch_households, desc=f"Batch {batch_start//batch_size + 1}/{(len(households_list)-1)//batch_size + 1}"):
            members = household_to_members.get(hid, [])
            if not members:
                continue
            
            # Get infected members with dates
            infected_in_household = [
                (m, infection_dates[m]) 
                for m in members 
                if m in infection_dates
            ]
            
            if not infected_in_household:
                continue
            
            # Sort by infection date
            infected_in_household.sort(key=lambda x: x[1])
            first_date = infected_in_household[0][1]
            
            # Identify first cluster (within incubation period)
            first_cluster = [
                (m, d) for m, d in infected_in_household
                if (d - first_date).days <= incubation_days
            ]
            
            # Identify subsequent infections (after incubation period)
            subsequent_infected = [
                (m, d) for m, d in infected_in_household
                if (d - first_date).days > incubation_days
            ]
            
            # Split first cluster into co-primaries and secondaries
            co_primaries = []
            secondaries = []
            for m, d in first_cluster:
                if (d - first_date).days <= exposure_days:
                    co_primaries.append((m, d))
                else:
                    secondaries.append((m, d))
            
            # Add secondaries (label=2)
            for m, d in secondaries:
                if m not in added_persons:
                    new_rows.append([m, d, 2])
                    added_persons.add(m)
            
            # Add co-primaries (label=1)
            for m, d in co_primaries:
                if m not in added_persons:
                    new_rows.append([m, d, 1])
                    added_persons.add(m)
            
            # Calculate max date for first cluster
            first_cluster_max = max(d for _, d in first_cluster)
            
            # Add subsequent infected as label=0
            for m, d in subsequent_infected:
                if m not in added_persons:
                    new_rows.append([m, d, 0])
                    added_persons.add(m)
            
            # Add healthy household members (label=0, first cluster max date)
            household_healthy = [
                m for m in members 
                if m in healthy_set and m not in added_persons
            ]
            for h in household_healthy:
                new_rows.append([h, first_cluster_max, 0])
                added_persons.add(h)
                healthy_set.discard(h)
        
        # Periodic memory cleanup
        if batch_end % 5000 == 0:
            force_garbage_collection()
            print_memory_usage(f"After processing {batch_end} households")
    
    # Clear large dictionaries no longer needed
    clear_memory(household_to_members, infection_dates, hushall_dict, 
                 infected_households, households_list, added_persons)
    
    # Create new DataFrame
    print('\nCreating final dataframe...')
    new_df = pd.DataFrame(
        new_rows,
        columns=['P1105_LopNr_PersonNr', 'index_date', 'label']
    )
    new_df.set_index('P1105_LopNr_PersonNr', inplace=True)
    
    # Clear new_rows from memory
    del new_rows
    force_garbage_collection()
    
    # Optimize DataFrame memory
    new_df = optimize_dataframe_memory(new_df, verbose=True)
    
    # Save new index
    print(f'\nSaving household-based index...')
    with open(new_index_path, 'wb') as f:
        pickle.dump(new_df, f)
    
    csv_path = new_index_path.replace('.pkl', '.csv')
    new_df.to_csv(csv_path, index=True)
    
    # Create and save remaining healthy DataFrame
    remaining_healthy_df = pd.DataFrame(index=list(healthy_set))
    
    with open(remaining_healthy_path, 'wb') as f:
        pickle.dump(remaining_healthy_df, f)
    
    csv_path = remaining_healthy_path.replace('.pkl', '.csv')
    remaining_healthy_df.to_csv(csv_path, index=True)
    
    # Print summary statistics
    label_counts = new_df['label'].value_counts().sort_index()
    print(f"\nHousehold-based index generation complete:")
    print(f"  Total entries: {len(new_df):,}")
    print(f"  Label 1 (Co-primary): {label_counts.get(1, 0):,}")
    print(f"  Label 2 (Secondary): {label_counts.get(2, 0):,}")
    print(f"  Label 0 (Other): {label_counts.get(0, 0):,}")
    print(f"  Remaining healthy: {len(remaining_healthy_df):,}")
    print(f"  Saved to: {new_index_path}")
    
    # Final cleanup
    clear_memory(new_df, remaining_healthy_df, healthy_set)
    print_memory_usage("After generation (cleaned)")


def filter_index_by_death_date(
    index_path: str,
    dors_pkl_path: str,
    filtered_path: str
) -> None:
    """
    Filter generated index to remove individuals where index_date > death_date.
    
    This removes logically invalid cases where someone is recorded as infected
    after their death date.
    
    Args:
        index_path: Path to generated index pickle
        dors_pkl_path: Path to SWECOV_SOS_DORS_2020_duplicates.pkl
        filtered_path: Path to save filtered index pickle
    """
    print(f"\n{'='*60}")
    print("DEATH DATE FILTERING")
    print(f"{'='*60}")
    print_memory_usage("Before filtering")
    
    # Load household index
    with open(index_path, 'rb') as f:
        df = pickle.load(f)
    
    # Count before filtering
    label_counts_before = df['label'].value_counts().sort_index()
    
    print(f"Before filtering:")
    print(f"  Total entries: {len(df):,}")
    print(f"  Label 1 (Co-primary): {label_counts_before.get(1, 0):,}")
    print(f"  Label 2 (Secondary): {label_counts_before.get(2, 0):,}")
    print(f"  Label 0 (Other): {label_counts_before.get(0, 0):,}")
    
    # Load death dictionary
    with open(dors_pkl_path, 'rb') as f:
        dors_dict = pickle.load(f)
    
    print(f"\nDeath data loaded: {len(dors_dict):,} entries")
    
    def get_death_date(person):
        """Extract death date for a person"""
        if person in dors_dict and dors_dict[person]:
            dod = dors_dict[person][0]['DODSDAT']
            return pd.to_datetime(dod, errors='coerce', format='%Y%m%d')
        return None
    
    # Convert index_date to datetime
    df['index_date'] = pd.to_datetime(df['index_date'], errors='coerce')
    
    # Identify rows to remove
    print("\nChecking for invalid dates...")
    to_remove = []
    for person in tqdm(df.index, desc="Validating dates"):
        death_date = get_death_date(person)
        if death_date is not None:
            index_date = df.at[person, 'index_date']
            if pd.notna(index_date) and index_date > death_date:
                to_remove.append(person)
    
    # Clear death dictionary
    del dors_dict
    force_garbage_collection()
    
    # Filter the DataFrame
    filtered_df = df.drop(to_remove, errors='ignore')
    
    # Clear original df
    del df
    force_garbage_collection()
    
    # Optimize memory
    filtered_df = optimize_dataframe_memory(filtered_df, verbose=True)
    
    # Save filtered DataFrame
    print(f"\nSaving filtered index...")
    with open(filtered_path, 'wb') as f:
        pickle.dump(filtered_df, f)
    
    csv_path = filtered_path.replace('.pkl', '.csv')
    filtered_df.to_csv(csv_path, index=True)
    
    # Count after filtering
    label_counts_after = filtered_df['label'].value_counts().sort_index()
    
    print(f"\nAfter filtering:")
    print(f"  Removed: {len(to_remove):,} entries where index_date > death_date")
    print(f"  Total entries: {len(filtered_df):,}")
    print(f"  Label 1 (Co-primary): {label_counts_after.get(1, 0):,}")
    print(f"  Label 2 (Secondary): {label_counts_after.get(2, 0):,}")
    print(f"  Label 0 (Other): {label_counts_after.get(0, 0):,}")
    print(f"  Saved to: {filtered_path}")
    
    # Final cleanup
    clear_memory(filtered_df, to_remove)
    print_memory_usage("After filtering (cleaned)")



# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_complete_pipeline():
    """
    Execute the complete data preprocessing pipeline.
    
    Steps:
    1. Time-based filtering for all configured tables
    2. Feature selection
    3. CSV to dictionary conversion
    4. Population index generation
    5. Household transmission classification
    6. Death date filtering
    7. Cleanup old file versions
    """
    print("="*80)
    print("COVID-19 HOUSEHOLD TRANSMISSION DATA PREPROCESSING PIPELINE")
    print("Optimized for GB-scale data with memory management")
    print("="*80)
    print_memory_usage("Initial state")
    
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR_FILTERED, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR_FEATURES, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR_INDEX, exist_ok=True)
    
    # Step 1: Time filtering for all tables
    # print("\n" + "="*80)
    # print("STEP 1: TIME-BASED FILTERING")
    # print("="*80)
    # filter_all_tables_by_time()
    
    # Step 2: Feature selection
    # print("\n" + "="*80)
    # print("STEP 2: FEATURE SELECTION")
    # print("="*80)
    # select_features_all_tables()
    
    # Step 3: CSV to dictionary conversion
    # print("\n" + "="*80)
    # print("STEP 3: CSV TO DICTIONARY CONVERSION")
    # print("="*80)
    # convert_all_csv_to_dictionaries()
    
    
    # Step 4: Population index generation
    print("\n" + "="*80)
    print("STEP 4: POPULATION INDEX GENERATION")
    print("="*80)

    # Load full Population_PersonNr_20221231 (adjust path as needed)
    # print("\nLoading population data...")
    # population_csv_path = '/path/to/raw_register_data/Population_PersonNr_20221231.csv'
    # population_df = pd.read_csv(population_csv_path, encoding='latin1', low_memory=False)
    # print(f'Total population loaded: {len(population_df):,}')
    # print_memory_usage("After loading population")

    # Generate source population index
    # population_index, _, _ = generate_population_index(population_df)

    # Clear population_df from memory
    # del population_df
    # force_garbage_collection()

    # Load full RTB2019
    # print("\nLoading RTB2019 data...")
    # rtb2020_csv_path = '/path/to/raw_register_data/RTB2019.csv'
    # rtb2020_df = pd.read_csv(rtb2020_csv_path, encoding='latin1', low_memory=False)
    # print(f'RTB2019 population loaded: {len(rtb2020_df):,}')
    # print_memory_usage("After loading RTB2019")

    # Generate 2020 population index
    # rtb2020_index, _, _ = generate_2020_population_index(population_index, rtb2020_df)

    # Clear rtb2020_df and population_index from memory
    # del rtb2020_df, population_index
    # force_garbage_collection()

    # os.makedirs('PopIndex', exist_ok=True)

    # Classify infected and healthy (using 2020 index for relevant population)
    # print("\nClassifying infected and healthy populations...")
    # infected_df, healthy_df = classify_infected_healthy(
    #     rtb2020_index,
    #     sminet_pkl_path='Features_Selected_Data/FHM_SMINET_2020_duplicates.pkl',
    #     infected_save_path='PopIndex/infected_population.pkl',
    #     healthy_save_path='PopIndex/healthy_population.pkl'
    # )

    with open('PopIndex/infected_population.pkl','rb') as f:
        infected_df = pickle.load(f)

    with open('PopIndex/healthy_population.pkl','rb') as f:
        healthy_df = pickle.load(f)

    # Clear rtb2020_index from memory as we now have infected_df and healthy_df
    # del rtb2020_index
    # force_garbage_collection()
    # print_memory_usage("After population classification")

    # Generate household-based index
    new_index_path = os.path.join(
        Config.OUTPUT_DIR_INDEX,
        'Index_Secondary_Case_FirstCluster.pkl'
    )
    remaining_healthy_path = os.path.join(
        Config.OUTPUT_DIR_INDEX,
        'Index_Secondary_Healthy_FirstCluster.pkl'
    )
    
    generate_household_based_index(
        infected_df=infected_df,
        healthy_df=healthy_df,
        hushall_pkl_path='Features_Selected_Data/HushallPerson_2019_duplicates.pkl',
        new_index_path=new_index_path,
        remaining_healthy_path=remaining_healthy_path,
        test_limit=None,
        incubation_days=Config.INCUBATION_DAYS,
        exposure_days=Config.EXPOSURE_DAYS
    )
    
    # Step 5: Filter by death dates
    print("\n" + "="*80)
    print("STEP 5: DEATH DATE FILTERING")
    print("="*80)
    
    filtered_path = os.path.join(
        Config.OUTPUT_DIR_INDEX,
        'Filtered_Index_Secondary_Case_FirstCluster.pkl'
    )
    
    filter_index_by_death_date(
        index_path=new_index_path,
        dors_pkl_path='Features_Selected_Data/SWECOV_SOS_DORS_2020_duplicates.pkl',
        filtered_path=filtered_path
    )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print_memory_usage("Final state")


if __name__ == "__main__":
    run_complete_pipeline()