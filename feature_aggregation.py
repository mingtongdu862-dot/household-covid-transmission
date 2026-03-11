"""
feature_aggregation.py
======================
Household-level feature aggregation and encoding pipeline for the COVID-19
household secondary transmission study.

Takes the person-level feature table produced by ``feature_extraction.py`` and
the household membership table from ``household_mapping.py``, then:

    1. Aggregates person-level features to the household level (age statistics,
       gender composition, income distribution, housing, medical-history sums, etc.).
    2. Encodes and transforms features: label-encoding, frequency-encoding, one-hot
       encoding, log-transforms, and median imputation for missing values.
    3. Selects medical-history binary features via chi-square test
       (p < 0.02, max-frequency < 0.95).
    4. Standardises continuous features using ``StandardScaler``.
    5. Splits the dataset into a held-out test set (10 %) and five stratified
       cross-validation folds, writing train/val/test CSVs for each fold.

Output
------
``Encoded_Household_Features_Full/train_fold_{k}.csv``
``Encoded_Household_Features_Full/val_fold_{k}.csv``
``Encoded_Household_Features_Full/test_fold_{k}.csv``  (k = 1 … 5)

These files are the direct inputs to all model-training scripts.

Usage
-----
    python feature_aggregation.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc
import psutil
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for household-level feature engineering"""
    
    # Input paths
    PERSON_TABLE_PATH = 'Feature_Tables/Raw_Feature_Secondary_Case.csv'
    HOUSEHOLD_MAP_PATH = 'Feature_Tables/household_member.csv'
    
    # Output paths
    OUTPUT_DIR_AGGREGATE = 'Household_Features'
    OUTPUT_DIR_ENCODED = 'Encoded_Household_Features_Full'
    HOUSEHOLD_TABLE_NAME = 'Household_Level_Feature_Table_Full.csv'
    
    # Processing parameters
    AGGREGATE_BATCH_SIZE = 100000  # Households per batch for aggregation
    PERSON_CHUNK_SIZE = 1000000    # Rows per chunk when reading person table
    ENCODING_BATCH_SIZE = 5000     # Features per batch for chi-square test
    
    # Encoding parameters
    ENCODING = 'latin1'
    
    # Feature selection parameters
    CHI2_P_VALUE_THRESHOLD = 0.02      # P-value threshold for chi-square test
    MAX_FREQ_THRESHOLD = 0.95          # Maximum frequency threshold for feature variance
    MISSING_RATE_THRESHOLD = 20        # Percentage threshold for dropping high-missing columns
    
    # Train/test split parameters
    TEST_SIZE = 0.1
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Testing mode
    TEST_MODE = False
    TEST_HOUSEHOLDS = 1000


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
    print(f"{prefix}Memory:")
    print(f"  Process: {mem['process_rss_gb']:.2f} GB")
    print(f"  System: {mem['system_used_gb']:.2f}/{mem['system_used_gb'] + mem['system_available_gb']:.2f} GB ({mem['system_percent']:.1f}%)")


def force_cleanup(*objects) -> None:
    """Force cleanup of objects and garbage collection"""
    for obj in objects:
        try:
            del obj
        except:
            pass
    gc.collect()


# ============================================================================
# STEP 1: HOUSEHOLD-LEVEL AGGREGATION
# ============================================================================

def get_count_columns(df: pd.DataFrame) -> List[str]:
    """
    Get all count columns that start with specified prefixes.
    
    Args:
        df: DataFrame to search
        
    Returns:
        List of column names
    """
    prefixes = ('contact_', 'lmed_', 'ov_', 'sv_')
    return [col for col in df.columns if col.startswith(prefixes)]


def load_person_table() -> pd.DataFrame:
    """
    Load person-level table in chunks and build indexed DataFrame.
    
    Returns:
        Indexed person-level DataFrame
    """
    print(f"\n{'='*80}")
    print("STEP 1: LOADING PERSON-LEVEL TABLE")
    print(f"{'='*80}")
    print_memory_usage("Before loading")
    
    # Define required columns
    person_dtypes = {
        'person_id': str,
        'label': 'float32',
        'IndexDate': str
    }
    
    base_columns = [
        'person_id', 'IndexDate', 'label', 'UtlSvBakg', 'Fodelseland',
        'FodelseArMan', 'Kon', 'AntalBarnUnder18', 'Boarea_Person',
        'Boendeform', 'DispInk04', 'DispInkFam04', 'TRYGG_1', 'TRYGG_total'
    ]
    
    # Detect count columns from sample
    print("Detecting count columns...")
    sample = pd.read_csv(
        Config.PERSON_TABLE_PATH,
        nrows=1000,
        encoding=Config.ENCODING,
        low_memory=False
    )
    count_cols = get_count_columns(sample)
    del sample
    gc.collect()
    
    print(f"Found {len(count_cols):,} count columns")
    
    # Read full table in chunks
    usecols = base_columns + count_cols
    person_chunks = pd.read_csv(
        Config.PERSON_TABLE_PATH,
        usecols=usecols,
        chunksize=Config.PERSON_CHUNK_SIZE,
        encoding=Config.ENCODING,
        low_memory=False,
        dtype=person_dtypes
    )
    
    person_indexed = None
    
    print("Building person index...")
    for chunk in tqdm(person_chunks, desc="Processing chunks"):
        # Normalize person_id
        chunk['person_id'] = chunk['person_id'].astype(str).str.rstrip('.0')
        chunk.set_index('person_id', inplace=True)
        
        # Convert types
        chunk['IndexDate'] = pd.to_datetime(chunk['IndexDate'], errors='coerce')
        chunk['label'] = pd.to_numeric(chunk['label'], errors='coerce')
        
        for col in count_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
        
        # Concatenate chunks
        if person_indexed is None:
            person_indexed = chunk.copy()
        else:
            person_indexed = pd.concat([person_indexed, chunk])
        
        del chunk
        gc.collect()
    
    # Remove duplicates
    person_indexed = person_indexed[~person_indexed.index.duplicated(keep='first')]
    
    print(f"Loaded {person_indexed.shape[0]:,} persons with {person_indexed.shape[1]:,} features")
    print_memory_usage("After loading")
    
    return person_indexed


def aggregate_household_features(
    household_row: pd.Series,
    person_indexed: pd.DataFrame,
    member_cols: List[str],
    count_cols: List[str]
) -> Optional[Dict]:
    """
    Aggregate features for a single household.
    
    Args:
        household_row: Row from household mapping table
        person_indexed: Indexed person-level DataFrame
        member_cols: List of member column names
        count_cols: List of count column names
        
    Returns:
        Dictionary of aggregated features or None if invalid
    """
    household_id = str(household_row.iloc[0])
    members = household_row[member_cols].dropna().astype(str).str.rstrip('.0').tolist()
    
    if not members:
        return None
    
    # Get valid members
    valid_members = [m for m in members if m in person_indexed.index]
    if not valid_members:
        return None
    
    sub = person_indexed.loc[valid_members]
    
    # Calculate core household variables
    household_size = len(sub)
    secondary_cases_cnt = int((sub['label'] == 2).sum())
    index_cases_cnt = int((sub['label'] == 1).sum())
    
    # Skip invalid households
    if secondary_cases_cnt == 0 and household_size == index_cases_cnt:
        return None
    if index_cases_cnt == 0:
        return None
    
    # Initialize feature dictionary
    features = {
        'household_id': household_id,
        'household_size': household_size,
        'IndexDate_household': sub['IndexDate'].max(),
        'secondary_cases_count': secondary_cases_cnt,
        'index_cases_count': index_cases_cnt
    }
    
    # Age aggregations
    birth_raw = pd.to_numeric(sub['FodelseArMan'], errors='coerce')
    birth_year = (birth_raw // 100).astype('Int64')
    age = 2020 - birth_year
    
    features.update({
        'mean_age_2020': age.mean(),
        'max_age_2020': age.max(),
        'min_age_2020': age.min(),
        'age_variance': age.var(),
        'age_IQR': age.quantile(0.75) - age.quantile(0.25),
        'age_range': age.max() - age.min(),
        'age_0_17_count': int((age <= 17).sum()),
        'age_18_64_count': int(((age >= 18) & (age <= 64)).sum()),
        'age_65plus_count': int((age >= 65).sum()),
        'has_member_75plus': int((age >= 75).any()),
        'proportion_children': (age < 18).sum() / household_size,
        'proportion_elderly': (age >= 65).sum() / household_size
    })
    
    # ========================================================================
    # IMMIGRATION BACKGROUND
    # ========================================================================
    utl_sv_bakg = sub['UtlSvBakg']
    mode_utl = utl_sv_bakg.mode()
    features.update({
        'prop_foreign_background': (utl_sv_bakg == 11).mean(),
        'has_any_foreign_background': int((utl_sv_bakg == 11).any()),
        'all_foreign_background': int((utl_sv_bakg == 11).all())
    })
    
    fodelseland = sub['Fodelseland']
    mode_fodel = fodelseland.mode()
    features.update({
        'Fodelseland_mode': mode_fodel[0] if len(mode_fodel) > 0 else np.nan,
        'Fodelseland_diversity': fodelseland.nunique(),
        'prop_born_sweden': (fodelseland == 'SVERIGE').mean() if 'SVERIGE' in fodelseland.values else 0.0
    })
    
    # Gender counts
    kon = pd.to_numeric(sub['Kon'], errors='coerce')
    male_count = int((kon == 1).sum())
    female_count = int((kon == 2).sum())
    
    features.update({
        'male_count': male_count,
        'female_count': female_count,
        'proportion_male': male_count / household_size,
        'proportion_female': female_count / household_size,
        'gender_diversity': 1 if (male_count > 0 and female_count > 0) else 0
    })
    
    # Family structure
    features.update({
        'has_child_under_6': int((age < 6).any()),
        'has_child_6_17': int(((age >= 6) & (age <= 17)).any()),
        'has_elderly_65plus': int((age >= 65).any()),
        'multigenerational': int((age.max() - age.min()) > 40),
        'three_generation': int((age < 18).any() and ((age >= 18) & (age <= 64)).any() and (age >= 65).any())
    })
    
    # Housing
    for col in ['AntalBarnUnder18', 'Boarea_Person']:
        val = sub[col].dropna()
        features[col] = val.iloc[0] if len(val) > 0 else np.nan
    
    mode_boende = sub['Boendeform'].mode()
    features['Boendeform_mode'] = mode_boende[0] if len(mode_boende) > 0 else np.nan
    
    # Derived housing features
    features['total_Boarea'] = features['Boarea_Person'] * household_size if pd.notna(features['Boarea_Person']) else np.nan
    features['crowding_index'] = household_size / features['Boarea_Person'] if pd.notna(features['Boarea_Person']) and features['Boarea_Person'] > 0 else np.nan
    
    # Crowding categories
    if pd.notna(features['crowding_index']):
        features['is_overcrowded'] = int(features['crowding_index'] > 1.5)
        features['is_spacious'] = int(features['crowding_index'] < 0.5)
    else:
        features['is_overcrowded'] = 0
        features['is_spacious'] = 0

    # Income aggregations
    for income_col in ['DispInk04', 'DispInkFam04']:
        income = pd.to_numeric(sub[income_col], errors='coerce')
        features.update({
            f'mean_{income_col}': income.mean(),
            f'max_{income_col}': income.max(),
            f'min_{income_col}': income.min(),
            f'sd_{income_col}': income.std(),
            f'median_{income_col}': income.median(),
            f'range_{income_col}': income.max() - income.min() if len(income) > 0 else np.nan
        })
    
    # TRYGG
    features.update({
        'TRYGG_1_sum': pd.to_numeric(sub['TRYGG_1'], errors='coerce').sum(),
        'TRYGG_total_sum': pd.to_numeric(sub['TRYGG_total'], errors='coerce').sum(),
        'any_TRYGG_1': int(pd.to_numeric(sub['TRYGG_1'], errors='coerce').sum() > 0),
        'any_TRYGG': int(pd.to_numeric(sub['TRYGG_total'], errors='coerce').sum() > 0),
        'proportion_with_TRYGG': (pd.to_numeric(sub['TRYGG_total'], errors='coerce') > 0).sum() / household_size
    })
    
    # TRYGG per elderly
    elderly_count = features['age_65plus_count']
    features['TRYGG_1_per_elderly'] = features['TRYGG_1_sum'] / elderly_count if elderly_count > 0 else 0
    features['TRYGG_total_per_capita'] = features['TRYGG_total_sum'] / household_size
    
    # Sum count features
    for col in count_cols:
        features[col] = sub[col].sum()
    
    return features


def aggregate_to_household_level() -> str:
    """
    Aggregate person-level features to household level.
    
    Returns:
        Path to output CSV file
    """
    print(f"\n{'='*80}")
    print("STEP 2: AGGREGATING TO HOUSEHOLD LEVEL")
    print(f"{'='*80}")
    
    # Load person table
    person_indexed = load_person_table()
    count_cols = get_count_columns(person_indexed)
    
    # Load household mapping
    print("\nLoading household mapping...")
    household_csv = pd.read_csv(Config.HOUSEHOLD_MAP_PATH, encoding=Config.ENCODING, low_memory=False)
    household_id_col = household_csv.columns[0]
    member_cols = [c for c in household_csv.columns if c.startswith('member_')]
    
    total_households = len(household_csv)
    print(f"Total households: {total_households:,}")
    
    # Test mode
    if Config.TEST_MODE:
        household_csv = household_csv.head(Config.TEST_HOUSEHOLDS)
        total_households = len(household_csv)
        print(f"⚠️  TEST MODE: Limited to {total_households:,} households")
    
    # Setup output
    os.makedirs(Config.OUTPUT_DIR_AGGREGATE, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR_AGGREGATE, Config.HOUSEHOLD_TABLE_NAME)
    
    # Process in batches
    batches = (total_households + Config.AGGREGATE_BATCH_SIZE - 1) // Config.AGGREGATE_BATCH_SIZE
    first_batch = True
    
    print(f"\nProcessing {total_households:,} households in {batches} batches...")
    
    for batch_idx in range(batches):
        start = batch_idx * Config.AGGREGATE_BATCH_SIZE
        end = min(start + Config.AGGREGATE_BATCH_SIZE, total_households)
        batch = household_csv.iloc[start:end]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{batches}: Households {start:,}-{end - 1:,}")
        print(f"{'='*60}")
        
        household_rows = []
        
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Processing batch {batch_idx + 1}"):
            features = aggregate_household_features(row, person_indexed, member_cols, count_cols)
            if features is not None:
                household_rows.append(features)
        
        # Create batch DataFrame
        batch_df = pd.DataFrame(household_rows)
        
        # Define column order
        base_cols = [
            'household_id', 'household_size', 'IndexDate_household', 'secondary_cases_count', 'index_cases_count', 

            'mean_age_2020', 'max_age_2020', 'min_age_2020', 'age_variance',
            'age_IQR', 'age_0_17_count', 'age_65plus_count', 'has_member_75plus',
            'proportion_children', 'proportion_elderly',

            'prop_foreign_background', 'has_any_foreign_background', 'all_foreign_background',

            'Fodelseland_diversity', 
            
            'male_count', 'female_count','proportion_male', 'proportion_female', 'gender_diversity',

            'has_child_under_6', 'has_child_6_17', 'has_elderly_65plus', 'multigenerational', 'three_generation',

            'AntalBarnUnder18', 'Boarea_Person', 'total_Boarea', 'crowding_index',

            'Boendeform_mode', 'is_overcrowded', 'is_spacious',

            'mean_DispInk04', 'max_DispInk04', 'min_DispInk04', 'sd_DispInk04', 'median_DispInk04', 'range_DispInk04',
            'mean_DispInkFam04', 'max_DispInkFam04', 'min_DispInkFam04', 'sd_DispInkFam04', 'median_DispInkFam04', 'range_DispInkFam04',
            
            'TRYGG_1_sum', 'TRYGG_total_sum', 'any_TRYGG_1', 'any_TRYGG', 'proportion_with_TRYGG', 'TRYGG_1_per_elderly', 'TRYGG_total_per_capita',
        ]
        other_cols = [c for c in batch_df.columns if c not in base_cols]
        batch_df = batch_df[base_cols + sorted(other_cols)]
        
        # Write to CSV
        mode = 'w' if first_batch else 'a'
        header = first_batch
        batch_df.to_csv(output_path, mode=mode, header=header, index=False, encoding=Config.ENCODING)
        
        print(f"Written {len(batch_df):,} households")
        
        first_batch = False
        force_cleanup(batch_df, household_rows, batch)
    
    force_cleanup(person_indexed, household_csv)
    
    print(f"\n{'='*80}")
    print(f"HOUSEHOLD AGGREGATION COMPLETE")
    print(f"Output: {output_path}")
    print(f"{'='*80}")
    print_memory_usage("After aggregation")
    
    return output_path


# ============================================================================
# STEP 2: FEATURE ENCODING AND TRANSFORMATION
# ============================================================================

def process_household_data(
    df: pd.DataFrame,
    mode: str = 'fit',
    thresholds: Optional[Dict] = None,
    keep_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[List[str]]]:
    """
    Process household data: handle missing values, encode features.
    
    Args:
        df: Input DataFrame
        mode: 'fit' or 'transform'
        thresholds: Dictionary of thresholds (required for transform mode)
        keep_cols: List of columns to keep (required for transform mode)
        
    Returns:
        Tuple of (processed_df, thresholds, keep_cols)
    """
    if mode not in ['fit', 'transform']:
        raise ValueError("Mode must be 'fit' or 'transform'")
    
    if mode == 'transform' and (thresholds is None or keep_cols is None):
        raise ValueError("Thresholds and keep_cols required for transform mode")
    
    if mode == 'fit':
        thresholds = {}
        keep_cols = []
    
    print(f"\n{'='*80}")
    print(f"PROCESSING DATA IN {mode.upper()} MODE")
    print(f"Shape: {df.shape}")
    print(f"{'='*80}")
    
    # Define column groups
    index_columns = ['household_id', 'IndexDate_household', 'label']
    
    general_columns = [
            'household_id', 'household_size', 'IndexDate_household', 'secondary_cases_count', 'index_cases_count', 

            'mean_age_2020', 'max_age_2020', 'min_age_2020', 'age_variance',
            'age_IQR', 'age_0_17_count', 'age_65plus_count', 'has_member_75plus',
            'proportion_children', 'proportion_elderly',

            'prop_foreign_background', 'has_any_foreign_background', 'all_foreign_background',

            'Fodelseland_diversity', 
            
            'male_count', 'female_count','proportion_male', 'proportion_female', 'gender_diversity',

            'has_child_under_6', 'has_child_6_17', 'has_elderly_65plus', 'multigenerational', 'three_generation',

            'AntalBarnUnder18', 'Boarea_Person', 'total_Boarea', 'crowding_index',

            'Boendeform_mode', 'is_overcrowded', 'is_spacious',

            'mean_DispInk04', 'max_DispInk04', 'min_DispInk04', 'sd_DispInk04', 'median_DispInk04', 'range_DispInk04',
            'mean_DispInkFam04', 'max_DispInkFam04', 'min_DispInkFam04', 'sd_DispInkFam04', 'median_DispInkFam04', 'range_DispInkFam04',
            
            'TRYGG_1_sum', 'TRYGG_total_sum', 'any_TRYGG_1', 'any_TRYGG', 'proportion_with_TRYGG', 'TRYGG_1_per_elderly', 'TRYGG_total_per_capita',
    ]
    
    count_prefixes = ['contact_', 'lmed_', 'ov_', 'sv_']
    count_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in count_prefixes)]
    
    print(f"General columns: {len(general_columns)}")
    print(f"Count columns: {len(count_columns)}")
    
    # Drop high-missing columns (fit mode only)
    if mode == 'fit':
        missing_rates = df[general_columns].isnull().mean() * 100
        columns_to_drop = missing_rates[missing_rates > Config.MISSING_RATE_THRESHOLD].index.tolist()
        thresholds['dropped_general_columns'] = columns_to_drop
        general_columns = [col for col in general_columns if col not in columns_to_drop]
        print(f"\nDropped {len(columns_to_drop)} columns with >{Config.MISSING_RATE_THRESHOLD}% missing")
    else:
        columns_to_drop = thresholds.get('dropped_general_columns', [])
        general_columns = [col for col in general_columns if col not in columns_to_drop]
    
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    gc.collect()
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    
    # UtlSvBakg_mode: Label Encoding
    if 'UtlSvBakg_mode' in df.columns:
        df['UtlSvBakg_mode'] = df['UtlSvBakg_mode'].fillna('unknown').astype(str)
        if mode == 'fit':
            le = LabelEncoder()
            df['UtlSvBakg_mode_enc'] = le.fit_transform(df['UtlSvBakg_mode'])
            thresholds['UtlSvBakg_mode_classes'] = list(le.classes_)
        else:
            classes = thresholds.get('UtlSvBakg_mode_classes', [])
            label_map = {cls: i for i, cls in enumerate(classes)}
            df['UtlSvBakg_mode_enc'] = df['UtlSvBakg_mode'].map(label_map).fillna(len(classes))
        df.drop(columns='UtlSvBakg_mode', errors='ignore', inplace=True)
        gc.collect()
    
    # Fodelseland_mode: Frequency Encoding
    if 'Fodelseland_mode' in df.columns:
        df['Fodelseland_mode'] = df['Fodelseland_mode'].fillna('unknown')
        if mode == 'fit':
            freq = df['Fodelseland_mode'].value_counts(normalize=True)
            thresholds['Fodelseland_mode_freq_map'] = freq.to_dict()
            df['Fodelseland_mode_freq'] = df['Fodelseland_mode'].map(freq).fillna(0)
        else:
            freq_map = thresholds.get('Fodelseland_mode_freq_map', {})
            df['Fodelseland_mode_freq'] = df['Fodelseland_mode'].map(freq_map).fillna(0)
        df.drop(columns='Fodelseland_mode', errors='ignore', inplace=True)
        gc.collect()
    
    # Boendeform_mode: One-Hot Encoding
    if 'Boendeform_mode' in df.columns:
        df['Boendeform_mode'] = df['Boendeform_mode'].fillna('unknown')
        if mode == 'fit':
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            boende_enc = ohe.fit_transform(df[['Boendeform_mode']])
            boende_cols = [f'Boendeform_mode_{cat}' for cat in ohe.categories_[0]]
            thresholds['Boendeform_mode_categories'] = list(ohe.categories_[0])
            df = pd.concat([df, pd.DataFrame(boende_enc, columns=boende_cols, index=df.index)], axis=1)
        else:
            categories = thresholds.get('Boendeform_mode_categories', [])
            ohe = OneHotEncoder(sparse_output=False, categories=[categories], handle_unknown='ignore')
            boende_enc = ohe.fit_transform(df[['Boendeform_mode']])
            boende_cols = [f'Boendeform_mode_{cat}' for cat in categories]
            df = pd.concat([df, pd.DataFrame(boende_enc, columns=boende_cols, index=df.index)], axis=1)
        df.drop('Boendeform_mode', axis=1, inplace=True)
        gc.collect()
    
    # Fill missing values for numerical columns
    print("Filling missing values...")
    
    # Age columns: median
    age_cols = ['mean_age_2020', 'max_age_2020', 'min_age_2020', 'age_variance', 'age_IQR']
    for col in age_cols:
        if col in df.columns:
            if mode == 'fit':
                median_val = df[col].median()
                thresholds[f'{col}_median'] = median_val
            else:
                median_val = thresholds.get(f'{col}_median', df[col].median())
            df[col] = df[col].fillna(median_val)
            gc.collect()
    
    # Binary/count columns: 0
    binary_count_cols = [
        col for col in general_columns
        if col.startswith(('multigenerational', 'three_generation', 'has_', 'all_', 'age_0_17_count', 'age_65plus_count', 'male_count', 'female_count','TRYGG_','any_'))
    ]
    for col in binary_count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
            gc.collect()
    
    # Proportion columns: 0
    prop_cols = [col for col in general_columns if col.startswith('prop_') or col.startswith('proportion_')]
    for col in prop_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            gc.collect()
    
    # Housing/income columns: median
    housing_income_cols = [
        'Boarea_Person', 'total_Boarea', 'crowding_index', 'Fodelseland_diversity', 'gender_diversity',
        'mean_DispInk04', 'max_DispInk04', 'min_DispInk04', 'sd_DispInk04', 'AntalBarnUnder18', 'median_DispInk04', 'range_DispInk04',
        'mean_DispInkFam04', 'max_DispInkFam04', 'min_DispInkFam04', 'sd_DispInkFam04', 'median_DispInkFam04', 'range_DispInkFam04',
    ]
    for col in housing_income_cols:
        if col in df.columns:
            if mode == 'fit':
                median_val = df[col].median()
                thresholds[f'{col}_median'] = median_val
            else:
                median_val = thresholds.get(f'{col}_median', df[col].median())
            df[col] = df[col].fillna(median_val)
            gc.collect()
    
    # Transform numerical features
    print("Transforming numerical features...")
    
    # Log transform for skewed features
    log_transform_cols = ['AntalBarnUnder18', 'Boarea_Person', 'total_Boarea', 
                          'mean_DispInk04', 'mean_DispInkFam04']
    for col in log_transform_cols:
        if col in df.columns:

            values = df[col].fillna(0).clip(lower=0, upper=1e10)
            df[f'{col}_log'] = np.log1p(values)

            df[f'{col}_log'] = df[f'{col}_log'].replace([np.inf, -np.inf], np.nan)
            df[f'{col}_log'] = df[f'{col}_log'].fillna(0)
    # Encode count columns using binning
    print("\nEncoding count columns...")
    
    if mode == 'fit':
        for col in tqdm(count_columns, desc="Encoding counts"):
            if col in df.columns:
                # Occurrence
                df[f'{col}_occurred'] = (df[col] > 0).astype(int)
                
                # Median threshold
                median_val = df[col].median()
                thresholds[f'{col}_median'] = median_val
                df[f'{col}_exceeds_median'] = (df[col] > median_val).astype(int)
                
                # 75th percentile threshold
                p75_val = df[col].quantile(0.75)
                thresholds[f'{col}_p75'] = p75_val
                df[f'{col}_exceeds_p75'] = (df[col] > p75_val).astype(int)
        
        # Drop original count columns
        df.drop(columns=count_columns, inplace=True)
        gc.collect()
    else:
        for col in tqdm(count_columns, desc="Encoding counts"):
            if col in df.columns:
                df[f'{col}_occurred'] = (df[col] > 0).astype(int)
                
                median_val = thresholds.get(f'{col}_median', df[col].median())
                df[f'{col}_exceeds_median'] = (df[col] > median_val).astype(int)
                
                p75_val = thresholds.get(f'{col}_p75', df[col].quantile(0.75))
                df[f'{col}_exceeds_p75'] = (df[col] > p75_val).astype(int)
        
        df.drop(columns=count_columns, inplace=True)
        gc.collect()
    
    # Feature selection using chi-square test (fit mode only)
    if mode == 'fit':
        print("\nPerforming feature selection (chi-square test)...")
        
        encoded_count_columns = [
            col for col in df.columns
            if col.endswith(('_occurred', '_exceeds_median', '_exceeds_p75'))
        ]
        
        print(f"Total encoded features: {len(encoded_count_columns)}")
        
        total_batches = (len(encoded_count_columns) + Config.ENCODING_BATCH_SIZE - 1) // Config.ENCODING_BATCH_SIZE
        
        for b in range(total_batches):
            start = b * Config.ENCODING_BATCH_SIZE
            end = min(start + Config.ENCODING_BATCH_SIZE, len(encoded_count_columns))
            batch_cols = encoded_count_columns[start:end]
            
            batch_df = df[batch_cols + ['label']].copy()
            batch_df[batch_cols] = batch_df[batch_cols].clip(lower=0)
            
            target = batch_df['label']
            
            # Chi-square test
            chi2_vals, p_vals = chi2(batch_df[batch_cols], target)
            
            # Frequency check
            freq_ok = []
            for col in batch_cols:
                vc = batch_df[col].value_counts(normalize=True)
                max_freq = vc.iloc[0] if not vc.empty else 0
                freq_ok.append(max_freq < Config.MAX_FREQ_THRESHOLD)
            
            # Keep features that pass both tests
            batch_keep = [
                col for col, p, ok in zip(batch_cols, p_vals, freq_ok)
                if p < Config.CHI2_P_VALUE_THRESHOLD and not np.isnan(p) and ok
            ]
            keep_cols.extend(batch_keep)
            
            print(f"  Batch {b + 1}/{total_batches}: {len(batch_keep)}/{len(batch_cols)} kept")
            
            del batch_df
            gc.collect()
        
        # Drop non-kept columns
        drop_cols = [col for col in encoded_count_columns if col not in keep_cols]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            print(f"\nDropped {len(drop_cols)} non-significant features")
        
        print(f"Final: Retained {len(keep_cols)} features")
    else:
        # Transform mode: drop columns not in keep_cols
        encoded_count_columns = [
            col for col in df.columns
            if col.endswith(('_occurred', '_exceeds_median', '_exceeds_p75'))
        ]
        drop_cols = [col for col in encoded_count_columns if col not in keep_cols]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            print(f"Dropped {len(drop_cols)} non-kept features")
        gc.collect()
    
    print(f"\nFinal shape: {df.shape}")
    print_memory_usage("After processing")
    
    if mode == 'fit':
        return df, thresholds, keep_cols
    else:
        return df, None, None


def standardize_household_data(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Standardize numerical features using StandardScaler.
    
    Args:
        train_df: Training DataFrame (used to fit scaler)
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        
    Returns:
        Tuple of (train_std, val_std, test_std)
    """
    print(f"\n{'='*80}")
    print("STANDARDIZING FEATURES")
    print(f"{'='*80}")
    
    # Identify columns to standardize
    cols_to_std = [
        col for col in train_df.columns
        if '_log' in col or '_freq' in col or col in [
            'mean_age_2020', 'max_age_2020', 'min_age_2020', 'age_variance', 'age_IQR',
            'prop_foreign_background', 'Fodelseland_diversity', 'Boarea_Person', 'total_Boarea',
            'crowding_index', 'mean_DispInk04', 'max_DispInk04', 'min_DispInk04', 'sd_DispInk04',
            'mean_DispInkFam04', 'max_DispInkFam04', 'min_DispInkFam04', 'sd_DispInkFam04',
            'TRYGG_1_per_elderly', 'index_cases_count'
        ]
    ]

    def clean_dataframe(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:

            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            if pd.notna(q01) and pd.notna(q99):
                df[col] = df[col].clip(lower=q01, upper=q99)
        return df
    
    print("Cleaning infinite and extreme values...")
    train_df = clean_dataframe(train_df)
    if val_df is not None:
        val_df = clean_dataframe(val_df)
    if test_df is not None:
        test_df = clean_dataframe(test_df)

    
    print(f"Standardizing {len(cols_to_std)} features")
    
    for col in tqdm(cols_to_std, desc="Standardizing"):
        if col in train_df.columns:
            scaler = StandardScaler()
            train_values = train_df[[col]].values
            scaler.fit(train_values)
            
            train_df[col + '_std'] = scaler.transform(train_values)
            train_df.drop(columns=[col], inplace=True)
            del train_values
            gc.collect()
            
            if val_df is not None and col in val_df.columns:
                val_values = val_df[[col]].values
                val_df[col + '_std'] = scaler.transform(val_values)
                val_df.drop(columns=[col], inplace=True)
                del val_values
                gc.collect()
            
            if test_df is not None and col in test_df.columns:
                test_values = test_df[[col]].values
                test_df[col + '_std'] = scaler.transform(test_values)
                test_df.drop(columns=[col], inplace=True)
                del test_values
                gc.collect()
    
    gc.collect()
    print_memory_usage("After standardization")
    
    return train_df, val_df, test_df


# ============================================================================
# STEP 3: K-FOLD SPLIT AND EXPORT
# ============================================================================

def create_kfold_datasets(household_table_path: str) -> None:
    """
    Create K-fold train/val/test datasets from household table.
    
    Args:
        household_table_path: Path to aggregated household table
    """
    print(f"\n{'='*80}")
    print("STEP 3: CREATING K-FOLD DATASETS")
    print(f"{'='*80}")
    
    # Load household table
    print(f"Loading household table from {household_table_path}...")
    df = pd.read_csv(household_table_path, encoding=Config.ENCODING)
    
    # Add label
    df['label'] = (df['secondary_cases_count'] > 0).astype(int)
    
    print(f"Loaded {len(df):,} households")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split into train_full and test
    print(f"\nSplitting into train ({100 - Config.TEST_SIZE * 100:.0f}%) and test ({Config.TEST_SIZE * 100:.0f}%)...")
    
    train_full_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=df['label']
    )
    
    del df
    gc.collect()
    
    print(f"Train full: {len(train_full_df):,} households")
    print(f"Test: {len(test_df):,} households")
    
    # Create K-folds
    print(f"\nCreating {Config.N_FOLDS}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    folds = list(skf.split(train_full_df, train_full_df['label']))
    
    # Setup output directory
    os.makedirs(Config.OUTPUT_DIR_ENCODED, exist_ok=True)
    
    # Process each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold_idx + 1}/{Config.N_FOLDS}")
        print(f"{'='*80}")
        
        train_fold_df = train_full_df.iloc[train_idx].copy()
        val_fold_df = train_full_df.iloc[val_idx].copy()
        
        print(f"Train: {len(train_fold_df):,} | Val: {len(val_fold_df):,}")
        
        # Process train fold (fit mode)
        thresholds = {}
        train_processed, thresholds, keep_cols = process_household_data(
            train_fold_df,
            mode='fit',
            thresholds=thresholds
        )
        
        del train_fold_df
        gc.collect()
        
        # Process validation fold (transform mode)
        val_processed, _, _ = process_household_data(
            val_fold_df,
            mode='transform',
            thresholds=thresholds,
            keep_cols=keep_cols
        )
        
        del val_fold_df
        gc.collect()
        
        # Process test fold (transform mode)
        test_copy = test_df.copy()
        test_processed, _, _ = process_household_data(
            test_copy,
            mode='transform',
            thresholds=thresholds,
            keep_cols=keep_cols
        )
        
        del test_copy
        gc.collect()
        
        # Standardize
        train_std, val_std, test_std = standardize_household_data(
            train_processed,
            val_processed,
            test_processed
        )
        
        force_cleanup(train_processed, val_processed, test_processed)
        
        # Save
        train_path = os.path.join(Config.OUTPUT_DIR_ENCODED, f'train_fold_{fold_idx + 1}.csv')
        val_path = os.path.join(Config.OUTPUT_DIR_ENCODED, f'val_fold_{fold_idx + 1}.csv')
        test_path = os.path.join(Config.OUTPUT_DIR_ENCODED, f'test_fold_{fold_idx + 1}.csv')
        
        train_std.to_csv(train_path, index=False, encoding=Config.ENCODING)
        val_std.to_csv(val_path, index=False, encoding=Config.ENCODING)
        test_std.to_csv(test_path, index=False, encoding=Config.ENCODING)
        
        print(f"\nSaved fold {fold_idx + 1} datasets:")
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")
        print(f"  Test: {test_path}")
        
        force_cleanup(train_std, val_std, test_std)
    
    # Save fold information
    folds_path = os.path.join(Config.OUTPUT_DIR_ENCODED, 'folds_info.txt')
    with open(folds_path, 'w') as f:
        f.write(f"K-Fold Cross Validation Information\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Number of folds: {Config.N_FOLDS}\n")
        f.write(f"Random state: {Config.RANDOM_STATE}\n")
        f.write(f"Test size: {Config.TEST_SIZE * 100:.0f}%\n\n")
        
        for i, (train_idx, val_idx) in enumerate(folds):
            f.write(f"Fold {i + 1}:\n")
            f.write(f"  Train indices: {len(train_idx):,} samples\n")
            f.write(f"  Val indices: {len(val_idx):,} samples\n\n")
    
    print(f"\n{'='*80}")
    print("K-FOLD DATASETS CREATION COMPLETE")
    print(f"Output directory: {Config.OUTPUT_DIR_ENCODED}")
    print(f"{'='*80}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_household_feature_engineering_pipeline():
    """
    Execute the complete household-level feature engineering pipeline.
    """
    print("="*80)
    print("HOUSEHOLD-LEVEL FEATURE ENGINEERING PIPELINE")
    print("COVID-19 Household Transmission Study")
    print("="*80)
    print_memory_usage("Initial state")
    
    # Step 1: Aggregate to household level
    household_table_path = aggregate_to_household_level()
    
    # Step 2 & 3: Encode and create K-fold datasets
    create_kfold_datasets(household_table_path)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print_memory_usage("Final state")


if __name__ == "__main__":
    run_household_feature_engineering_pipeline()