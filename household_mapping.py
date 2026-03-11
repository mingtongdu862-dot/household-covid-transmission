"""
Household Member Mapping Generator

Generates a household-to-member mapping table from a person-household
linkage file. The output is a wide-format CSV where each row represents
one household and each column contains the anonymised ID of one member.

Input
-----
HushallPerson_2019.csv  (person-household linkage table)

Output
------
household_member.csv    (wide-format mapping table)

    household_id | member_1 | member_2 | member_3 | ...
    -------------+----------+----------+----------+----
    H001         | P123     | P456     | P789     | ...

Pipeline position: Step 1 of 3 — run before feature_engineering.py.
"""

import os
import gc
import psutil
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for household member mapping generation."""

    # Input path
    INPUT_FILE = 'Features_Selected_Data/HushallPerson_2019.csv'

    # Output paths
    OUTPUT_DIR = 'Feature_Tables'
    OUTPUT_FILE = 'household_member.csv'
    INVALID_HOUSEHOLDS_FILE = 'invalid_households_report.csv'

    # Column names in input file
    HOUSEHOLD_ID_COL = 'P1105_LopNr_Hushallsid_2019'
    PERSON_ID_COL    = 'P1105_LopNr_PersonNr'

    # Processing parameters
    CHUNK_SIZE = 1_000_000   # Rows per chunk
    ENCODING   = 'latin1'

    # Validation parameters
    MAX_HOUSEHOLD_SIZE       = 200   # Maximum plausible household size
    MIN_HOUSEHOLD_SIZE       = 1     # Minimum household size
    REPORT_LARGE_HOUSEHOLDS  = True  # Write a report of filtered households


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def get_memory_usage() -> Dict[str, float]:
    """Return current process and system memory usage statistics."""
    process    = psutil.Process()
    mem_info   = process.memory_info()
    virtual    = psutil.virtual_memory()
    return {
        'process_rss_gb':      mem_info.rss / (1024 ** 3),
        'system_used_gb':      virtual.used / (1024 ** 3),
        'system_available_gb': virtual.available / (1024 ** 3),
        'system_percent':      virtual.percent,
    }


def print_memory_usage(label: str = '') -> None:
    """Print current memory usage with an optional descriptive label."""
    mem    = get_memory_usage()
    prefix = f'[{label}] ' if label else ''
    print(f'{prefix}Memory:')
    print(f'  Process: {mem["process_rss_gb"]:.2f} GB')
    print(f'  System:  {mem["system_used_gb"]:.2f} GB / '
          f'{mem["system_available_gb"]:.2f} GB available')
    print(f'  Usage:   {mem["system_percent"]:.1f}%')


def force_cleanup(*objects) -> None:
    """Delete the supplied objects and trigger garbage collection."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# ============================================================================
# HOUSEHOLD MEMBER MAPPING GENERATION
# ============================================================================

def save_invalid_households_report(invalid_households: Dict) -> None:
    """
    Write a CSV report of households that were excluded during validation.

    Args:
        invalid_households: Mapping of household_id → {'size': int, 'members': list}.
    """
    print(f"\n{'='*80}")
    print('SAVING INVALID HOUSEHOLDS REPORT')
    print(f"{'='*80}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(Config.OUTPUT_DIR, Config.INVALID_HOUSEHOLDS_FILE)

    report_rows = []
    for household_id, info in invalid_households.items():
        size    = info['size']
        members = info['members']

        if size < Config.MIN_HOUSEHOLD_SIZE:
            reason = f'Too small (< {Config.MIN_HOUSEHOLD_SIZE})'
        else:
            reason = f'Too large (> {Config.MAX_HOUSEHOLD_SIZE})'

        members_sample = members[:100]
        members_str    = ', '.join(members_sample)
        if len(members) > 100:
            members_str += f' ... ({len(members) - 100} more)'

        report_rows.append({
            'household_id':   household_id,
            'size':           size,
            'reason':         reason,
            'sample_members': members_str,
        })

    report_rows = sorted(report_rows, key=lambda x: x['size'], reverse=True)
    report_df   = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False, encoding=Config.ENCODING)

    print(f'Saved invalid households report to: {report_path}')
    print(f'Report contains {len(report_df):,} invalid households')

    print('\nTop 5 most problematic households:')
    for i, row in report_df.head(5).iterrows():
        print(f'  {i+1}. Household {row["household_id"]}: '
              f'{row["size"]:,} members — {row["reason"]}')


def build_household_member_dict() -> Dict[str, List[str]]:
    """
    Build a dictionary mapping each household ID to its list of member IDs.

    Reads the linkage file in chunks to keep memory usage manageable.
    Households outside [MIN_HOUSEHOLD_SIZE, MAX_HOUSEHOLD_SIZE] are
    excluded and optionally reported.

    Returns:
        dict: {household_id: [person_id, ...]} for all valid households.
    """
    print(f"\n{'='*80}")
    print('BUILDING HOUSEHOLD MEMBER DICTIONARY')
    print(f"{'='*80}")
    print(f'Reading from: {Config.INPUT_FILE}')
    print_memory_usage('Before reading')

    household_members = defaultdict(list)

    print(f'\nReading file in chunks of {Config.CHUNK_SIZE:,} rows...')
    chunk_reader = pd.read_csv(
        Config.INPUT_FILE,
        usecols=[Config.HOUSEHOLD_ID_COL, Config.PERSON_ID_COL],
        chunksize=Config.CHUNK_SIZE,
        encoding=Config.ENCODING,
        low_memory=False,
    )

    total_rows  = 0
    chunk_count = 0

    for chunk in tqdm(chunk_reader, desc='Processing chunks'):
        chunk_count += 1

        # Normalise IDs (strip trailing ".0" from integer-encoded floats)
        chunk[Config.HOUSEHOLD_ID_COL] = (
            chunk[Config.HOUSEHOLD_ID_COL].astype(str).str.rstrip('.0')
        )
        chunk[Config.PERSON_ID_COL] = (
            chunk[Config.PERSON_ID_COL].astype(str).str.rstrip('.0')
        )

        for _, row in chunk.iterrows():
            household_id = row[Config.HOUSEHOLD_ID_COL]
            person_id    = row[Config.PERSON_ID_COL]

            if pd.isna(household_id) or pd.isna(person_id):
                continue

            if person_id not in household_members[household_id]:
                household_members[household_id].append(person_id)

        total_rows += len(chunk)
        del chunk

        if chunk_count % 5 == 0:
            gc.collect()
            print_memory_usage(f'After chunk {chunk_count}')

    print(f'\nProcessed {total_rows:,} person-household records')
    print(f'Found {len(household_members):,} unique households')

    household_members = dict(household_members)

    # ── Statistics before filtering ──────────────────────────────────────────
    member_counts = [len(m) for m in household_members.values()]
    print('\nHousehold size statistics (BEFORE filtering):')
    print(f'  Total households: {len(household_members):,}')
    print(f'  Min members:      {min(member_counts)}')
    print(f'  Max members:      {max(member_counts):,}')
    print(f'  Mean members:     {np.mean(member_counts):.2f}')
    print(f'  Median members:   {np.median(member_counts):.0f}')

    # ── Filter invalid households ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('IDENTIFYING ABNORMAL HOUSEHOLDS')
    print(f"{'='*80}")
    print(f'Filter criteria:')
    print(f'  Min size: {Config.MIN_HOUSEHOLD_SIZE}')
    print(f'  Max size: {Config.MAX_HOUSEHOLD_SIZE}')

    valid_households   = {}
    invalid_households = {}

    for household_id, members in household_members.items():
        size = len(members)
        if Config.MIN_HOUSEHOLD_SIZE <= size <= Config.MAX_HOUSEHOLD_SIZE:
            valid_households[household_id] = members
        else:
            invalid_households[household_id] = {'size': size, 'members': members}

    print(f'\nFiltering results:')
    print(f'  Valid households:   {len(valid_households):,}')
    print(f'  Invalid households: {len(invalid_households):,}')

    if invalid_households:
        invalid_sizes = [info['size'] for info in invalid_households.values()]
        too_small = sum(1 for s in invalid_sizes if s < Config.MIN_HOUSEHOLD_SIZE)
        too_large = sum(1 for s in invalid_sizes if s > Config.MAX_HOUSEHOLD_SIZE)

        print(f'\nInvalid household breakdown:')
        print(f'  Too small (< {Config.MIN_HOUSEHOLD_SIZE}): {too_small:,}')
        print(f'  Too large (> {Config.MAX_HOUSEHOLD_SIZE}): {too_large:,}')

        if too_large > 0:
            large_sizes = sorted(
                [s for s in invalid_sizes if s > Config.MAX_HOUSEHOLD_SIZE],
                reverse=True,
            )
            print('\nTop 10 largest excluded households:')
            for i, size in enumerate(large_sizes[:10], 1):
                print(f'  {i}. {size:,} members')

        if Config.REPORT_LARGE_HOUSEHOLDS:
            save_invalid_households_report(invalid_households)

    # ── Statistics after filtering ────────────────────────────────────────────
    if valid_households:
        valid_counts = [len(m) for m in valid_households.values()]
        print('\nHousehold size statistics (AFTER filtering):')
        print(f'  Total households: {len(valid_households):,}')
        print(f'  Min members:      {min(valid_counts)}')
        print(f'  Max members:      {max(valid_counts)}')
        print(f'  Mean members:     {np.mean(valid_counts):.2f}')
        print(f'  Median members:   {np.median(valid_counts):.0f}')

    print_memory_usage('After filtering')
    return valid_households


def convert_dict_to_wide_format(household_members: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Convert the household member dictionary to a wide-format DataFrame.

    Each row represents one household; columns member_1, member_2, …
    contain the corresponding member IDs (NaN for absent positions).

    Args:
        household_members: {household_id: [person_id, ...]}

    Returns:
        pd.DataFrame: Wide-format table with shape
            (n_households, 1 + max_household_size).
    """
    print(f"\n{'='*80}")
    print('CONVERTING TO WIDE FORMAT')
    print(f"{'='*80}")
    print_memory_usage('Before conversion')

    max_members = max(len(m) for m in household_members.values())
    print(f'Maximum household size: {max_members} members')

    member_cols = [f'member_{i+1}' for i in range(max_members)]
    all_cols    = ['household_id'] + member_cols

    print(f'Creating DataFrame with {len(all_cols):,} columns...')

    rows = []
    for household_id, members in tqdm(household_members.items(), desc='Building rows'):
        row = [household_id] + members + [np.nan] * (max_members - len(members))
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_cols)

    print(f'\nDataFrame shape: {df.shape}')
    print(f'  Households:         {len(df):,}')
    print(f'  Max member columns: {max_members}')

    print('\nOptimising dtypes...')
    df['household_id'] = df['household_id'].astype(str)
    for col in member_cols:
        df[col] = df[col].astype(str).replace('nan', np.nan)

    print_memory_usage('After conversion')
    return df


def save_household_member_table(df: pd.DataFrame) -> str:
    """
    Write the household member DataFrame to CSV.

    Args:
        df: Wide-format household member DataFrame.

    Returns:
        str: Path to the saved CSV file.
    """
    print(f"\n{'='*80}")
    print('SAVING HOUSEHOLD MEMBER TABLE')
    print(f"{'='*80}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILE)

    print(f'Saving to: {output_path}')
    print(f'Estimated size: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB')

    df.to_csv(output_path, index=False, encoding=Config.ENCODING)

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f'File saved. Actual size: {file_size_mb:.2f} MB')
    print(f'\nFirst 5 rows:\n{df.head()}')
    print(f'\nLast 5 rows:\n{df.tail()}')

    return output_path


def generate_household_member_mapping() -> str:
    """
    Execute the full household member mapping pipeline.

    Steps:
        1. Build household → member dictionary from the linkage file.
        2. Convert to wide-format DataFrame.
        3. Save to CSV.

    Returns:
        str: Path to the output CSV file.
    """
    print('=' * 80)
    print('HOUSEHOLD MEMBER MAPPING GENERATOR')
    print('=' * 80)
    print_memory_usage('Initial state')

    household_members = build_household_member_dict()
    df = convert_dict_to_wide_format(household_members)

    del household_members
    gc.collect()

    output_path = save_household_member_table(df)

    del df
    gc.collect()

    print(f"\n{'='*80}")
    print('GENERATION COMPLETE')
    print(f"{'='*80}")
    print(f'Output file: {output_path}')
    print_memory_usage('Final state')

    return output_path


# ============================================================================
# OPTIONAL VALIDATION UTILITIES
# ============================================================================

def validate_household_member_table(output_path: str) -> None:
    """
    Run basic integrity checks on the generated household member table.

    Checks performed:
        - Duplicate household IDs
        - Household sizes within the configured valid range
        - Persons appearing in multiple households (within the sample)

    Args:
        output_path: Path to the CSV file to validate.
    """
    print(f"\n{'='*80}")
    print('VALIDATING OUTPUT TABLE')
    print(f"{'='*80}")

    print('Reading first 1,000 rows for validation...')
    sample = pd.read_csv(output_path, nrows=1000, encoding=Config.ENCODING)

    print(f'\nTable structure:')
    print(f'  Total columns:      {len(sample.columns)}')
    print(f'  First 10 columns:   {sample.columns.tolist()[:10]}')
    member_cols = [c for c in sample.columns if c.startswith('member_')]
    print(f'  Member columns:     {len(member_cols)}')

    print('\nChecking for duplicate household IDs...')
    full_df    = pd.read_csv(output_path, usecols=['household_id'],
                              encoding=Config.ENCODING)
    duplicates = full_df.duplicated().sum()
    if duplicates > 0:
        print(f'  WARNING: {duplicates:,} duplicate household IDs found!')
    else:
        print('  No duplicate household IDs found.')

    print('\nAnalysing household sizes in sample...')
    household_sizes = sample[member_cols].notna().sum(axis=1)
    print(f'  Sample statistics (first 1,000 households):')
    print(f'    Min:    {household_sizes.min()}')
    print(f'    Max:    {household_sizes.max()}')
    print(f'    Mean:   {household_sizes.mean():.2f}')
    print(f'    Median: {household_sizes.median():.0f}')

    invalid_in_sample = (
        (household_sizes < Config.MIN_HOUSEHOLD_SIZE) |
        (household_sizes > Config.MAX_HOUSEHOLD_SIZE)
    ).sum()
    if invalid_in_sample > 0:
        print(f'  WARNING: {invalid_in_sample} households outside valid range in sample!')
    else:
        print(f'  All sampled households within valid range '
              f'({Config.MIN_HOUSEHOLD_SIZE}–{Config.MAX_HOUSEHOLD_SIZE}).')

    print('\nChecking for persons in multiple households (sample only)...')
    all_members    = sample[member_cols].values.flatten()
    all_members    = [str(m) for m in all_members
                      if pd.notna(m) and str(m) != 'nan']
    unique_members = len(set(all_members))
    total_members  = len(all_members)

    if unique_members < total_members:
        n_dup = total_members - unique_members
        print(f'  WARNING: {n_dup:,} person IDs appear in multiple households '
              f'in the sample. This may indicate a data quality issue.')
    else:
        print('  All persons in sample belong to unique households.')

    print('\nValidation complete.')
    del sample, full_df
    gc.collect()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    output_path = generate_household_member_mapping()

    # Uncomment to run optional validation after generation:
    # validate_household_member_table(output_path)

    print(f'\nAll done. Output file: {output_path}')
    print('You can now use this file as input for feature_engineering.py.')
