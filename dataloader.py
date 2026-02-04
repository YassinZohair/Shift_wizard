#!/usr/bin/env python3
"""
==============================================================================
Shift Planning Decision Assistant - Data Loader & Cleaner
==============================================================================

SETUP:
1. Create a 'data' folder in your project
2. Put all CSV files in that folder
3. Run: pip install pandas numpy
4. Run: python dataloader.py
"""

import logging
import re
from pathlib import Path

# ==============================================================================
# INSTALL CHECK
# ==============================================================================
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("=" * 60)
    print("ERROR: Required packages not installed!")
    print("Run: pip install pandas numpy")
    print("=" * 60)
    exit(1)

# ==============================================================================
# CONFIGURATION - UPDATE THIS PATH IF NEEDED
# ==============================================================================

# For Codespace/Linux - put your CSVs in a 'data' folder:
DATA_DIR = Path("./data")

# For Windows (uncomment if running locally):
# DATA_DIR = Path(r"C:\Users\david\Downloads\Hackathon DATA")

# Output directory
OUTPUT_DIR = Path("./cleaned_data")

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# SENSITIVE COLUMNS TO REMOVE
# ==============================================================================
SENSITIVE_COLUMNS = {
    'password', 'passwords', 'api_key', 'api_keys', 'otp',
    'email', 'emails', 'email_temp',
    'mobile_phone', 'mobile_phone_temp', 'phone', 'customer_mobile_phone',
    'first_name', 'last_name', 'customer_name', 'contact_name',
    'date_of_birth', 'address', 'street_address', 'latitude', 'longitude',
    'store_address', 'picture', 'logo', 'image_1', 'image_2', 'image_3',
    'image_4', 'image_5', 'website', 'facebook', 'instagram',
    'takeaway_link', 'delivery_link', 'table_booking_link',
    'takeaway_qr_code_link', 'delivery_qr_code_link',
    'link', 'qr_code', 'table_stand', 'barcode_scanner_ids',
    'receipt_printer_ids', 'payment_terminal_ids',
    'external_id', 'nets_mid', 'realm', 'referral_id', 'barcode',
}

TIMESTAMP_COLUMNS = {
    'created', 'updated', 'activated', 'start_date_time', 'end_date_time',
    'pickup_time', 'promise_time', 'contract_start', 'invoicing_start_date',
    'termination_date', 'demo_end',
}

# ==============================================================================
# CLEANING FUNCTIONS
# ==============================================================================

def standardize_column_name(col_name: str) -> str:
    name = col_name.lower()
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def is_unix_timestamp(value) -> bool:
    try:
        return 0 <= float(value) <= 4102444800
    except (ValueError, TypeError):
        return False


def clean_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    logger.info(f"Cleaning {name} ({len(df)} rows, {len(df.columns)} columns)")

    # 1. Standardize column names
    df.columns = [standardize_column_name(col) for col in df.columns]

    # 2. Remove sensitive columns
    cols_to_remove = [col for col in df.columns if col in SENSITIVE_COLUMNS]
    if cols_to_remove:
        logger.info(f"  Removing: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)

    # 3. Clean strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['nan', 'None', '', 'NaN', 'null', 'NULL'], np.nan)

    # 4. Convert timestamps
    for col in df.columns:
        if col in TIMESTAMP_COLUMNS or col.endswith('_time') or col.endswith('_date_time'):
            try:
                numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_vals) > 0 and all(is_unix_timestamp(v) for v in numeric_vals.head(50)):
                    df[col] = pd.to_datetime(
                        pd.to_numeric(df[col], errors='coerce'),
                        unit='s', utc=True, errors='coerce'
                    )
                    logger.info(f"  Converted '{col}' to datetime")
            except Exception:
                pass

    # 5. Remove duplicates
    initial = len(df)
    df = df.drop_duplicates()
    if len(df) < initial:
        logger.info(f"  Removed {initial - len(df)} duplicates")

    # 6. ID uniqueness
    if 'id' in df.columns and 'updated' in df.columns:
        if df['id'].duplicated().any():
            df = df.sort_values('updated', ascending=False)
            df = df.drop_duplicates(subset=['id'], keep='first')

    logger.info(f"  Result: {len(df)} rows, {len(df.columns)} columns")
    return df


def generate_hourly_demand(orders_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating hourly demand...")

    if 'created' not in orders_df.columns or 'place_id' not in orders_df.columns:
        logger.error("Missing required columns")
        return pd.DataFrame()

    df = orders_df[orders_df['created'].notna()].copy()
    df['date'] = df['created'].dt.date
    df['hour'] = df['created'].dt.hour
    df['weekday'] = df['created'].dt.day_name()

    hourly = (
        df.groupby(['place_id', 'date', 'hour', 'weekday'])
        .size()
        .reset_index(name='orders')
    )
    hourly = hourly.sort_values(['place_id', 'date', 'hour']).reset_index(drop=True)
    hourly['date'] = hourly['date'].astype(str)

    logger.info(f"Generated: {len(hourly)} rows, {hourly['place_id'].nunique()} places")
    return hourly


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("SHIFT PLANNING - DATA LOADER & CLEANER")
    print("=" * 60)

    # Check data directory
    if not DATA_DIR.exists():
        print(f"\nERROR: Folder not found: {DATA_DIR.absolute()}")
        print("\nTO FIX:")
        print("  1. Create a 'data' folder in your project")
        print("  2. Put all your CSV files in that folder")
        print("  3. Run this script again")
        exit(1)

    # Files to load
    files = {
        "dim_campaigns": "dim_campaigns.csv",
        "dim_delivery_locations": "dim_delivery_locations.csv",
        "dim_places": "dim_places.csv",
        "dim_tables": "dim_tables.csv",
        "dim_taxonomy_terms": "dim_taxonomy_terms.csv",
        "dim_users": "dim_users.csv",
        "fct_bonus_codes": "fct_bonus_codes.csv",
        "fct_campaigns": "fct_campaigns.csv",
        "fct_order_items": "fct_order_items_sample_150k.csv",
        "fct_orders": "fct_orders_sample_150k.csv",
        "most_ordered": "most_ordered.csv",
    }

    # STEP 1: Load
    print(f"\n[1] Loading from: {DATA_DIR.absolute()}\n")
    raw_data = {}
    for name, filename in files.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            raw_data[name] = pd.read_csv(filepath, low_memory=False)
            print(f"  ✓ {filename}: {len(raw_data[name]):,} rows")
        else:
            print(f"  ✗ NOT FOUND: {filename}")

    if not raw_data:
        print("\nERROR: No CSV files found!")
        exit(1)

    # STEP 2: Clean
    print("\n" + "-" * 60)
    print("[2] CLEANING DATA")
    print("-" * 60 + "\n")

    cleaned_data = {}
    for name, df in raw_data.items():
        cleaned_data[name] = clean_dataframe(df, name)

    # STEP 3: Generate hourly demand
    print("\n" + "-" * 60)
    print("[3] GENERATING HOURLY DEMAND")
    print("-" * 60 + "\n")

    if "fct_orders" in cleaned_data:
        hourly_demand = generate_hourly_demand(cleaned_data["fct_orders"])
        if not hourly_demand.empty:
            cleaned_data["hourly_demand"] = hourly_demand

    # STEP 4: Save
    print("\n" + "-" * 60)
    print("[4] SAVING CLEANED DATA")
    print("-" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in cleaned_data.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
    print(f"Total: {len(cleaned_data)} files\n")

    for name, df in cleaned_data.items():
        print(f"  {name:25} {len(df):>7,} rows × {len(df.columns):>3} cols")

    return cleaned_data


if __name__ == "__main__":
    data = main()
