import pandas as pd
import numpy as np
import os
import io

# Define default parameters
DEFAULT_RECENCY_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 13, 25, 37]  # Months
DEFAULT_FREQUENCY_BINS = [1, 2, 3, 4, 5, np.inf]  # Purchase counts

DEFAULT_SAMPLE_SIZES = {
    "Test": 7500,
    "Control": 7500,
    "Holdout": 15000
}

RANDOM_SEED = 42


def bin_customers(df, recency_bins=None, frequency_bins=None):
    """
    Categorizes customers into Recency and Frequency bins and prepares a downloadable CSV buffer.

    Args:
    - df (pd.DataFrame): Aggregated customer data with 'recency' and 'frequency' columns.
    - recency_bins (list): Custom bin edges for recency segmentation.
    - frequency_bins (list): Custom bin edges for frequency segmentation.

    Returns:
    - df (pd.DataFrame): Updated DataFrame with RF segment labels.
    - csv_buffer (io.StringIO): Buffer containing CSV data for download.
    """
    recency_bins = recency_bins or DEFAULT_RECENCY_BINS
    frequency_bins = frequency_bins or DEFAULT_FREQUENCY_BINS

    # Define labels for bins
    recency_labels = [f"R{i}" for i in range(1, len(recency_bins))]
    frequency_labels = [f"F{i}" for i in range(1, len(frequency_bins))]

    # Apply binning
    df["Recency_Bucket"] = pd.cut(df["recency"], bins=recency_bins, labels=recency_labels, right=False)
    df["Frequency_Bucket"] = pd.cut(df["frequency"], bins=frequency_bins, labels=frequency_labels, right=False)

    # Drop rows with NaN buckets (customers who don't fit into bins)
    df.dropna(subset=["Recency_Bucket", "Frequency_Bucket"], inplace=True)

    # Create combined RF segment
    df["Segment"] = df["Frequency_Bucket"].astype(str) + "_" + df["Recency_Bucket"].astype(str)

    # Create a CSV buffer for download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    return df, csv_data


def stratified_sample(df, sample_sizes=None, seed=RANDOM_SEED):
    """
    Performs stratified sampling within Recency-Frequency segments.

    Args:
    - df (pd.DataFrame): Binned customer dataset with 'Segment' column.
    - sample_sizes (dict): Desired sample sizes for each group (Test, Control, Holdout).
    - seed (int): Random seed for reproducibility.

    Returns:
    - combined_df (pd.DataFrame): Final dataset with assigned groups.
    """
    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES
    np.random.seed(seed)

    # Initialize empty DataFrames for each group
    test_group, control_group, holdout_group = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Calculate segment proportions
    segment_counts = df["Segment"].value_counts()
    segment_proportions = segment_counts / segment_counts.sum()

    # Stratified sampling within each segment
    for segment, proportion in segment_proportions.items():
        segment_data = df[df["Segment"] == segment]

        # Compute sample sizes per segment
        n_test = int(round(sample_sizes["Test"] * proportion))
        n_control = int(round(sample_sizes["Control"] * proportion))
        n_holdout = int(round(sample_sizes["Holdout"] * proportion))

        # Shuffle before sampling
        segment_data = segment_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Adjust sample sizes if segment is too small
        total_needed = n_test + n_control + n_holdout
        if len(segment_data) < total_needed:
            scaling_factor = len(segment_data) / total_needed
            n_test = int(round(n_test * scaling_factor))
            n_control = int(round(n_control * scaling_factor))
            n_holdout = len(segment_data) - n_test - n_control  # Ensure all customers are assigned

        # Assign groups
        test_group = pd.concat([test_group, segment_data.iloc[:n_test]], ignore_index=True)
        control_group = pd.concat([control_group, segment_data.iloc[n_test:n_test + n_control]], ignore_index=True)
        holdout_group = pd.concat([holdout_group, segment_data.iloc[n_test + n_control:n_test + n_control + n_holdout]], ignore_index=True)

    # Label groups
    test_group["Group"] = "Test"
    control_group["Group"] = "Control"
    holdout_group["Group"] = "Holdout"

    # Combine datasets
    combined_df = pd.concat([test_group, control_group, holdout_group], ignore_index=True)

    return combined_df


def save_segments(df, output_dir="data/processed", file_prefix=""):
    """
    Saves stratified sample groups as CSV files.

    Args:
    - df (pd.DataFrame): Dataset with assigned test/control/holdout groups.
    - output_dir (str): Directory to save files.
    - file_prefix (str): Optional prefix for filenames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each group separately
    df[df["Group"] == "Test"].to_csv(os.path.join(output_dir, f"{file_prefix}Test_Group.csv"), index=False)
    df[df["Group"] == "Control"].to_csv(os.path.join(output_dir, f"{file_prefix}Control_Group.csv"), index=False)
    df[df["Group"] == "Holdout"].to_csv(os.path.join(output_dir, f"{file_prefix}Holdout_Group.csv"), index=False)
    df.to_csv(os.path.join(output_dir, f"{file_prefix}Combined_Groups.csv"), index=False)

    print(f"✅ Files saved in {output_dir} with prefix '{file_prefix}'.")


# Run script if executed directly
if __name__ == "__main__":
    input_file = "data/processed/customer_summary.csv"

    if os.path.exists(input_file):
        print("📊 Loading customer summary file...")
        df = pd.read_csv(input_file)

        # Ensure data types are correct
        df["customer_id"] = df["customer_id"].astype(str)
        df["customer_email"] = df["customer_email"].astype(str)

        # Perform binning and sampling
        df = bin_customers(df)
        sampled_df = stratified_sample(df)

        # Save output
        save_segments(sampled_df, file_prefix="campaign_")
    else:
        print("❌ Customer summary file not found. Run data processing first.")
