"""
feature_extraction.py
=====================
Individual-level feature extraction pipeline for the COVID-19 household
secondary transmission study.

Starting from the filtered population index produced by ``data_preprocessing.py``,
this module constructs a wide-format person-level feature table by:

    1. Loading the filtered population index (index cases, secondary cases, and
       uninfected household members with their reference dates).
    2. Extracting *static* demographic and socioeconomic features (age, sex,
       country of birth, housing, disposable income) from the register dictionaries.
    3. Computing *dynamic* medical-history features within configurable look-back
       windows before each person's index date:
           - Primary-care contact reasons (Inera VPTU, 1-year window)
           - Dispensed medications (ATC codes, 1-year window)
           - Outpatient diagnoses (ICD-10, all time)
           - Inpatient diagnoses (ICD-10, all time)
           - Elderly-care (TRYGG) indicators
    4. Writing the combined table in chunks to avoid holding the full dataset
       in memory simultaneously.

Output
------
``Feature_Tables/Raw_Feature_Secondary_Case.csv``
    Person-level feature table consumed by ``feature_aggregation.py``.

Usage
-----
    python feature_extraction.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import gc
import psutil
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for feature engineering"""
    
    # Input paths
    INDEX_PATH = 'Index(InD=14,ExD=2)/Filtered_Index_Secondary_Case_FirstCluster.pkl'
    FEATURES_DIR = 'Features_Selected_Data'
    
    # Output paths
    OUTPUT_DIR = 'Feature_Tables'
    OUTPUT_FILE = 'Raw_Feature_Secondary_Case.csv'
    
    # Processing parameters
    CHUNK_SIZE = 100000  # Number of persons to process per chunk
    ENCODING = 'latin1'
    
    # Time windows for dynamic features (days)
    INERA_WINDOW = 365  # 1 year before index date
    LMED_WINDOW = 365   # 1 year before index date
    
    # Feature extraction flags
    EXTRACT_STATIC = True
    EXTRACT_INERA = True
    EXTRACT_LMED = True
    EXTRACT_OV = True
    EXTRACT_SV = True
    EXTRACT_TRYGG = True
    
    # Testing mode
    TEST_MODE = False
    TEST_LIMIT = 1000  # Number of persons to process in test mode


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return {
        'process_rss_gb': mem_info.rss / (1024 ** 3),
        'system_used_gb': virtual_mem.used / (1024 ** 3),
        'system_available_gb': virtual_mem.available / (1024 ** 3),
        'system_percent': virtual_mem.percent
    }


def print_memory_usage(label: str = "") -> None:
    """Print current memory usage with optional label"""
    mem = get_memory_usage()
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Memory Usage:")
    print(f"  Process: {mem['process_rss_gb']:.2f} GB")
    print(f"  System: {mem['system_used_gb']:.2f} GB / {mem['system_available_gb']:.2f} GB available")
    print(f"  Usage: {mem['system_percent']:.1f}%")


def force_cleanup(*objects) -> None:
    """Force cleanup of objects and garbage collection"""
    for obj in objects:
        try:
            del obj
        except:
            pass
    gc.collect()


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_pickle_dict(path: str, normalize_keys: bool = True) -> Dict:
    """
    Load a pickle dictionary with optional key normalization.
    
    Args:
        path: Path to pickle file
        normalize_keys: If True, convert keys to string and remove trailing '.0'
        
    Returns:
        Dictionary loaded from pickle file
    """
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    
    if normalize_keys:
        data_dict = {str(k).rstrip('.0'): v for k, v in data_dict.items()}
    
    return data_dict


def load_unique_codes_from_csv(csv_path: str, column_name: str, encoding: str = Config.ENCODING) -> set:
    """
    Load unique codes from a CSV file.
    Memory-efficient: loads only required column.
    
    Args:
        csv_path: Path to CSV file
        column_name: Name of column containing codes
        encoding: File encoding
        
    Returns:
        Set of unique codes
    """
    print(f"  Loading unique codes from {os.path.basename(csv_path)}...")
    
    df = pd.read_csv(csv_path, usecols=[column_name], encoding=encoding)
    unique_codes = set(df[column_name].dropna().unique())
    
    del df
    gc.collect()
    
    print(f"  Found {len(unique_codes):,} unique codes")
    return unique_codes


# ============================================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================================

def parse_date(date_str: Any) -> pd.Timestamp:
    """
    Parse various date formats to pandas Timestamp.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Parsed datetime or NaT if parsing fails
    """
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Handle YYYYMMDD format (8 digits)
    if len(date_str) == 8 and date_str.isdigit():
        return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
    
    # Handle other formats
    return pd.to_datetime(date_str, errors='coerce')


def get_first_value(data_dict: Dict, key: str, field: str) -> Any:
    """
    Get first value from a dictionary of lists.
    
    Args:
        data_dict: Dictionary mapping keys to lists of dicts
        key: Key to look up
        field: Field to extract from first dict in list
        
    Returns:
        First value or None if not found
    """
    if key in data_dict and data_dict[key]:
        return data_dict[key][0].get(field, None)
    return None


def count_codes(
    person_id: str,
    code_dict: Dict,
    date_field: str,
    code_field: str,
    index_date: Optional[Any] = None,
    window_days: Optional[int] = None
) -> Dict[str, int]:
    """
    Count occurrences of codes for a person, optionally within a time window.
    
    Args:
        person_id: Person identifier
        code_dict: Dictionary mapping person IDs to lists of code records
        date_field: Name of date field in records
        code_field: Name of code field in records
        index_date: Index date for time window
        window_days: Number of days before index_date to count (None = all time)
        
    Returns:
        Dictionary mapping codes to counts
    """
    if person_id not in code_dict or not code_dict[person_id]:
        return defaultdict(int)
    
    counts = defaultdict(int)
    
    # Parse index date if time window is specified
    index_date_parsed = parse_date(index_date) if index_date else None
    start_date = (index_date_parsed - timedelta(days=window_days)) if (window_days and index_date_parsed) else None
    
    # Count codes
    for entry in code_dict[person_id]:
        entry_date_str = entry.get(date_field)
        if not entry_date_str:
            continue
        
        entry_date = parse_date(entry_date_str)
        if pd.isna(entry_date):
            continue
        
        code = entry.get(code_field)
        if code is None:
            continue
        
        # Apply time window filter if specified
        if window_days is not None and start_date is not None:
            if start_date <= entry_date <= index_date_parsed:
                counts[code] += 1
        else:
            counts[code] += 1
    
    return counts


def count_trygg(person_id: str, sol_dict: Dict) -> Tuple[int, int]:
    """
    Count TRYGG indicators for a person.
    
    Args:
        person_id: Person identifier
        sol_dict: Dictionary with TRYGG data
        
    Returns:
        Tuple of (trygg_1_count, total_count)
    """
    if person_id not in sol_dict or not sol_dict[person_id]:
        return 0, 0
    
    total = len(sol_dict[person_id])
    trygg_1 = sum(1 for entry in sol_dict[person_id] if entry.get('TRYGG') == 1)
    
    return trygg_1, total


# ============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================================

def load_main_index() -> pd.DataFrame:
    """
    Load and prepare the main population index.
    
    Returns:
        Prepared DataFrame with person_id and IndexDate
    """
    print(f"\n{'='*80}")
    print("LOADING POPULATION INDEX")
    print(f"{'='*80}")
    print_memory_usage("Before loading")
    
    # Load index
    with open(Config.INDEX_PATH, 'rb') as f:
        df = pickle.load(f)
    
    # Reset index and rename columns
    df = df.reset_index()
    df.rename(columns={
        'P1105_LopNr_PersonNr': 'person_id',
        'index_date': 'IndexDate'
    }, inplace=True)
    
    # Normalize person_id (remove trailing .0)
    df['person_id'] = df['person_id'].astype(str).str.rstrip('.0')
    
    # Test mode
    if Config.TEST_MODE:
        df = df.head(Config.TEST_LIMIT)
        print(f"⚠️  TEST MODE: Limited to {len(df)} persons")
    
    print(f"Loaded {len(df):,} persons")
    print_memory_usage("After loading")
    
    return df


def extract_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract static features (demographics, socioeconomic).
    
    Args:
        df: Main DataFrame with person_id
        
    Returns:
        DataFrame with static features added
    """
    if not Config.EXTRACT_STATIC:
        print("\n⏭️  Skipping static feature extraction")
        return df
    
    print(f"\n{'='*80}")
    print("EXTRACTING STATIC FEATURES")
    print(f"{'='*80}")
    print_memory_usage("Before extraction")
    
    # Load all necessary dictionaries
    print("Loading feature dictionaries...")
    sminet_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'FHM_SMINET_2020_duplicates.pkl'))
    fodelse_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'Fodelseuppg_20201231_duplicates.pkl'))
    population_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'Population_PersonNr_20221231_duplicates.pkl'))
    hushall_person_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'HushallPerson_2019_duplicates.pkl'))
    hushall_boende_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'HushallBoende_2019_duplicates.pkl'))
    individ_dict = load_pickle_dict(os.path.join(Config.FEATURES_DIR, 'Individ_2019_duplicates.pkl'))
    
    print("Extracting features...")
    
    # Initialize columns dictionary
    new_cols = {}
    
    # Extract features for each person
    for idx, person in enumerate(tqdm(df['person_id'], desc="Processing persons")):
        # SmiNet - Infection date
        new_cols.setdefault('Statistikdatum', []).append(
            get_first_value(sminet_dict, person, 'Statistikdatum')
        )
        
        # Birth background
        new_cols.setdefault('UtlSvBakg', []).append(
            get_first_value(fodelse_dict, person, 'UtlSvBakg')
        )
        new_cols.setdefault('Fodelseland', []).append(
            get_first_value(fodelse_dict, person, 'Fodelseland')
        )
        
        # Demographics
        new_cols.setdefault('FodelseArMan', []).append(
            get_first_value(population_dict, person, 'FodelseArMan')
        )
        new_cols.setdefault('Kon', []).append(
            get_first_value(population_dict, person, 'Kon')
        )
        
        # Household
        new_cols.setdefault('AntalBarnUnder18', []).append(
            get_first_value(hushall_person_dict, person, 'AntalBarnUnder18')
        )
        new_cols.setdefault('Boarea_Person', []).append(
            get_first_value(hushall_boende_dict, person, 'Boarea_Person')
        )
        new_cols.setdefault('Boendeform', []).append(
            get_first_value(hushall_boende_dict, person, 'Boendeform')
        )
        
        # Income
        new_cols.setdefault('DispInk04', []).append(
            get_first_value(individ_dict, person, 'DispInk04')
        )
        new_cols.setdefault('DispInkFam04', []).append(
            get_first_value(individ_dict, person, 'DispInkFam04')
        )
    
    # Add columns to DataFrame
    for col, values in new_cols.items():
        df[col] = values
    
    print(f"\nExtracted {len(new_cols)} static features")
    
    # Cleanup
    force_cleanup(new_cols, sminet_dict, fodelse_dict, population_dict,
                  hushall_person_dict, hushall_boende_dict, individ_dict)
    
    print_memory_usage("After extraction (cleaned)")
    
    return df


def load_unique_codes() -> Dict[str, set]:
    """
    Load unique codes from CSV files for dynamic features.
    
    Returns:
        Dictionary mapping feature names to sets of unique codes
    """
    print(f"\n{'='*80}")
    print("LOADING UNIQUE CODES")
    print(f"{'='*80}")
    
    unique_codes = {}
    
    if Config.EXTRACT_INERA:
        unique_codes['contact_reasons'] = load_unique_codes_from_csv(
            os.path.join(Config.FEATURES_DIR, 'Inera_VPTU_Coronadata_2019_2020.csv'),
            'contactReason'
        )
    
    if Config.EXTRACT_LMED:
        unique_codes['atc_codes'] = load_unique_codes_from_csv(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_LMED_2019_2020.csv'),
            'ATC'
        )
    
    if Config.EXTRACT_OV:
        unique_codes['ov_codes'] = load_unique_codes_from_csv(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_OV.csv'),
            'hdia'
        )
    
    if Config.EXTRACT_SV:
        unique_codes['sv_codes'] = load_unique_codes_from_csv(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_SV.csv'),
            'hdia'
        )
    
    return unique_codes


def extract_dynamic_features_chunk(
    chunk_df: pd.DataFrame,
    person_to_date: Dict[str, Any],
    feature_dicts: Dict[str, Dict],
    unique_codes: Dict[str, set]
) -> pd.DataFrame:
    """
    Extract dynamic features for a chunk of persons.
    
    Args:
        chunk_df: DataFrame chunk with person_id
        person_to_date: Mapping of person_id to index_date
        feature_dicts: Dictionary of feature dictionaries
        unique_codes: Dictionary of unique code sets
        
    Returns:
        DataFrame with dynamic features
    """
    chunk_persons = chunk_df['person_id'].tolist()
    count_data = []
    
    for person in tqdm(chunk_persons, desc="Processing persons in chunk"):
        row = {'person_id': person}
        index_date = person_to_date.get(person)
        
        # TRYGG features
        if Config.EXTRACT_TRYGG and 'sol' in feature_dicts:
            trygg_1, trygg_total = count_trygg(person, feature_dicts['sol'])
            row['TRYGG_1'] = trygg_1
            row['TRYGG_total'] = trygg_total
        
        # Inera features (medical contacts)
        if Config.EXTRACT_INERA and 'inera' in feature_dicts:
            inera_counts = count_codes(
                person, feature_dicts['inera'],
                'documentCreatedTime', 'contactReason',
                index_date, Config.INERA_WINDOW
            )
            for reason in unique_codes.get('contact_reasons', []):
                row[f'contact_{reason}'] = inera_counts.get(reason, 0)
        
        # LMED features (medications)
        if Config.EXTRACT_LMED and 'lmed' in feature_dicts:
            lmed_counts = count_codes(
                person, feature_dicts['lmed'],
                'CodeDate', 'Code',
                index_date, Config.LMED_WINDOW
            )
            for code in unique_codes.get('atc_codes', []):
                row[f'lmed_{code}'] = lmed_counts.get(code, 0)
        
        # OV features (outpatient diagnoses)
        if Config.EXTRACT_OV and 'ov' in feature_dicts:
            ov_counts = count_codes(
                person, feature_dicts['ov'],
                'CodeDate', 'Code'
            )
            for code in unique_codes.get('ov_codes', []):
                row[f'ov_{code}'] = ov_counts.get(code, 0)
        
        # SV features (inpatient diagnoses)
        if Config.EXTRACT_SV and 'sv' in feature_dicts:
            sv_counts = count_codes(
                person, feature_dicts['sv'],
                'CodeDate', 'Code'
            )
            for code in unique_codes.get('sv_codes', []):
                row[f'sv_{code}'] = sv_counts.get(code, 0)
        
        count_data.append(row)
    
    return pd.DataFrame(count_data)


def extract_dynamic_features(df: pd.DataFrame, unique_codes: Dict[str, set]) -> None:
    """
    Extract dynamic features (medical codes, medications) in chunks.
    Writes directly to CSV to manage memory.
    
    Args:
        df: Main DataFrame with person_id and static features
        unique_codes: Dictionary of unique code sets
    """
    print(f"\n{'='*80}")
    print("EXTRACTING DYNAMIC FEATURES (CHUNKED)")
    print(f"{'='*80}")
    print_memory_usage("Before extraction")
    
    # Load feature dictionaries
    print("Loading feature dictionaries...")
    feature_dicts = {}
    
    if Config.EXTRACT_INERA:
        feature_dicts['inera'] = load_pickle_dict(
            os.path.join(Config.FEATURES_DIR, 'Inera_VPTU_Coronadata_2019_2020_duplicates.pkl')
        )
    
    if Config.EXTRACT_LMED:
        feature_dicts['lmed'] = load_pickle_dict(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_LMED_2019_2020_duplicates.pkl')
        )
    
    if Config.EXTRACT_OV:
        feature_dicts['ov'] = load_pickle_dict(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_OV_duplicates.pkl')
        )
    
    if Config.EXTRACT_SV:
        feature_dicts['sv'] = load_pickle_dict(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_SV_duplicates.pkl')
        )
    
    if Config.EXTRACT_TRYGG:
        feature_dicts['sol'] = load_pickle_dict(
            os.path.join(Config.FEATURES_DIR, 'SWECOV_SOS_SOL_2018_2020_duplicates.pkl')
        )
    
    print_memory_usage("After loading dictionaries")
    
    # Create person_id to date mapping
    print("Creating person-to-date mapping...")
    person_to_date = dict(zip(df['person_id'], df['IndexDate']))
    
    # Setup output
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILE)
    
    # Process in chunks
    num_chunks = (len(df) + Config.CHUNK_SIZE - 1) // Config.CHUNK_SIZE
    print(f"\nProcessing {len(df):,} persons in {num_chunks} chunks...")
    
    first_chunk = True
    
    for chunk_num in range(num_chunks):
        start_idx = chunk_num * Config.CHUNK_SIZE
        end_idx = min(start_idx + Config.CHUNK_SIZE, len(df))
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"\n{'='*60}")
        print(f"Chunk {chunk_num + 1}/{num_chunks}")
        print(f"Processing persons {start_idx:,} to {end_idx:,}")
        print(f"{'='*60}")
        
        # Extract dynamic features for this chunk
        chunk_counts_df = extract_dynamic_features_chunk(
            chunk_df, person_to_date, feature_dicts, unique_codes
        )
        
        # Merge with static features
        merged_chunk = chunk_df.merge(chunk_counts_df, on='person_id', how='left')
        
        # Defragment
        merged_chunk = merged_chunk.copy()
        
        # Write to CSV
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        merged_chunk.to_csv(output_path, mode=mode, header=header, index=False, encoding=Config.ENCODING)
        first_chunk = False
        
        print(f"  Written {len(merged_chunk):,} rows to CSV")
        
        # Cleanup
        force_cleanup(chunk_df, chunk_counts_df, merged_chunk)
        
        if (chunk_num + 1) % 5 == 0:
            print_memory_usage(f"After chunk {chunk_num + 1}")
    
    # Final cleanup
    force_cleanup(feature_dicts, person_to_date)
    
    print(f"\n{'='*80}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"Output saved to: {output_path}")
    print(f"{'='*80}")
    print_memory_usage("Final state")


def run_feature_engineering_pipeline():
    """
    Execute the complete feature engineering pipeline.
    """
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("COVID-19 Household Transmission Study")
    print("="*80)
    print_memory_usage("Initial state")
    
    # Step 1: Load main index
    df = load_main_index()
    
    # Step 2: Extract static features
    df = extract_static_features(df)
    
    # Step 3: Load unique codes
    unique_codes = load_unique_codes()
    
    # Step 4: Extract dynamic features (writes to CSV)
    extract_dynamic_features(df, unique_codes)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_feature_engineering_pipeline()