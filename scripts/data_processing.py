import pandas as pd
import os
import glob
from datetime import datetime

# Define file paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MERGED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "merged_order_history.csv")
CUSTOMER_SUMMARY_PATH = os.path.join(PROCESSED_DATA_DIR, "customer_summary.csv")

def merge_order_history():
    """
    Reads and merges multiple CSV files from the raw data directory.
    
    Returns:
    - df (pd.DataFrame): Merged order history dataset.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Ensure output folder exists

    # Find all CSV files in the directory
    all_filenames = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    if not all_filenames:
        print("❌ No CSV files found in raw data folder.")
        return None

    print(f"🔄 Found {len(all_filenames)} order history files. Merging...")

    # Concatenate all files at once (same as Jupyter Notebook approach)
    df = pd.concat([pd.read_csv(f, dtype={"customer_id": str, "order_id": str}) for f in all_filenames], ignore_index=True)

    # Save merged file for inspection
    df.to_csv(MERGED_FILE_PATH, index=False, encoding="utf-8-sig")
    
    print(f"✅ Merged data saved to: {MERGED_FILE_PATH} (Rows: {len(df)})")
    return df

def clean_and_convert_data(df):
    """
    Cleans the order history dataset:
    - Converts date columns
    - Ensures identifiers remain as strings
    - Handles missing values
    """
    print("🧹 Cleaning data...")

    # Convert date column (assumes 'day' is the date field)
    if 'day' in df.columns:
        df['day'] = pd.to_datetime(df['day'], errors='coerce')

    # Ensure identifier columns remain as strings
    id_cols = ['customer_id', 'order_id']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Convert categorical columns
    if 'customer_email' in df.columns:
        df['customer_email'] = df['customer_email'].astype(str)

    # Drop rows with missing essential data
    df.dropna(subset=['customer_id', 'customer_email', 'order_id'], inplace=True)

    print(f"✅ Data cleaned. Total rows after cleaning: {len(df)}")
    return df

def aggregate_customer_data(df, current_date=None):
    """
    Aggregates order history to the customer level:
    - Computes Recency (months since last purchase)
    - Computes Frequency (total orders)
    - Computes Monetary value (Net Sales, Discounts)

    Returns:
    - customer_summary (pd.DataFrame): Aggregated customer data.
    """
    print("📊 Aggregating customer data...")

    if current_date is None:
        current_date = datetime.today()

    customer_summary = df.groupby(['customer_id', 'customer_email'], observed=True).agg(
        frequency=('order_id', 'nunique'),  # Number of unique orders
        last_order_date=('day', 'max'),     # Most recent order date
        gross_sales=('gross_sales', 'sum'), # Total gross sales
        discounts=('discounts', 'sum'),     # Total discounts
        net_sales=('net_sales', 'sum')      # Total net sales
    ).reset_index()

    # Compute Recency in months
    customer_summary['recency'] = (
        (current_date.year - customer_summary['last_order_date'].dt.year) * 12 +
        (current_date.month - customer_summary['last_order_date'].dt.month)
    )

    # Drop customers who are missing emails AFTER aggregation
    customer_summary.dropna(subset=['customer_email'], inplace=True)

    # Ensure `customer_id` is stored as a string 
    customer_summary['customer_id'] = customer_summary['customer_id'].astype(str)

    # Save processed file with explicit formatting to avoid scientific notation in Excel
    customer_summary.to_csv(CUSTOMER_SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.2f")

    print(f"✅ Aggregation complete. Customer summary saved to: {CUSTOMER_SUMMARY_PATH} (Rows: {len(customer_summary)})")
    return customer_summary

def process_order_history():
    """
    Main function to:
    1. Merge order history files
    2. Clean & convert data
    3. Aggregate customer-level metrics
    """
    print("🚀 Starting order history processing...")

    df = merge_order_history()
    if df is None:
        print("❌ No data to process. Exiting.")
        return

    df = clean_and_convert_data(df)
    customer_data = aggregate_customer_data(df)

    print("🎯 Processing complete. Ready for stratified sampling.")
    return customer_data

# Run the script when executed directly
if __name__ == "__main__":
    process_order_history()
