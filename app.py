import streamlit as st
import pandas as pd
import io
from scripts.data_processing import clean_and_convert_data, aggregate_customer_data
from scripts.stratified_sampling import bin_customers, stratified_sample

# Configure the Streamlit page
st.set_page_config(page_title="Direct Mail Campaign Tool", layout="wide")
st.title("📬 Direct Mail Campaign Sampling Tool")

# Cache heavy processing functions to improve performance
@st.cache_data
def process_order_history_cached(df):
    cleaned_df = clean_and_convert_data(df)
    customer_summary = aggregate_customer_data(cleaned_df)
    return customer_summary

# Sidebar: File Upload and Reset Button
st.sidebar.header("📂 Upload Order History CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple CSV files", accept_multiple_files=True, type=["csv"]
)

if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()

merged_df = None
if uploaded_files:
    st.sidebar.success(f"Uploaded {len(uploaded_files)} files.")
    dataframes = []
    for file in uploaded_files:
        try:
            df_temp = pd.read_csv(file)
            dataframes.append(df_temp)
        except Exception as e:
            st.sidebar.error(f"Error reading file: {file.name}")
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        st.sidebar.info("Files merged successfully! Proceed to processing.")
    else:
        st.sidebar.error("No valid CSV files were uploaded.")

# Validate that the merged DataFrame contains the required columns
if merged_df is not None:
    required_columns = {"customer_id", "order_id", "day", "customer_email"}
    if not required_columns.issubset(merged_df.columns):
        st.error("Uploaded files are missing one or more required columns: customer_id, order_id, day, customer_email.")
        merged_df = None

# Process Order History Button
if merged_df is not None and st.sidebar.button("🚀 Process Order History"):
    with st.spinner("Processing order history..."):
        customer_summary = process_order_history_cached(merged_df)
        if customer_summary is not None:
            st.success("✅ Processing complete! Aggregated customer data is ready.")
            st.dataframe(customer_summary.head(20))
            st.session_state["customer_summary"] = customer_summary
        else:
            st.error("Error during processing order history.")

# Recency-Frequency Segmentation Section
if "customer_summary" in st.session_state:
    df = st.session_state["customer_summary"]

    st.subheader("📊 Recency-Frequency Segmentation")
    with st.form("segmentation_form"):
        recency_bins_input = st.text_input("Recency Bins (comma-separated)", "0,1,2,3,4,5,6,7,13,25,37")
        frequency_bins_input = st.text_input("Frequency Bins (comma-separated)", "1,2,3,4,5,inf")
        max_recency_threshold = st.number_input("Max Recency Threshold", value=36, step=1)
        segmentation_submitted = st.form_submit_button("🛠️ Apply Binning")

        if segmentation_submitted:
            try:
                # Convert inputs into bin boundaries
                recency_bins = [float(x.strip()) for x in recency_bins_input.split(",")]
                frequency_bins = [float(x.strip()) if x.strip().lower() != "inf" else float("inf")
                                  for x in frequency_bins_input.split(",")]

                # Filter out customers above the max recency threshold
                df_filtered = df[df["recency"] <= max_recency_threshold]
                if df_filtered.empty:
                    st.error("No customers have a recency value below the selected maximum threshold.")
                else:
                    df_segmented, segmented_csv = bin_customers(df_filtered.copy(), recency_bins, frequency_bins)
                    if df_segmented.empty:
                        st.error("Segmentation resulted in an empty DataFrame. Please check your bin ranges.")
                    else:
                        # Store the segmented data in session state
                        st.session_state["segmented_df"] = df_segmented
                        st.session_state["recency_bins"] = recency_bins
                        st.session_state["frequency_bins"] = frequency_bins
                        st.session_state["segmented_csv"] = segmented_csv
                        st.success("✅ Binning complete!")
                        st.dataframe(df_segmented.head(20))

            except Exception as e:
                st.error(f"Error processing binning: {e}")

    # Download Button for Segmented Data
    if "segmented_csv" in st.session_state:
        st.subheader("📥 Download Segmented Customer Data")
        st.download_button(
            label="📥 Download Segmented Customers",
            data=st.session_state["segmented_csv"],
            file_name="Segmented_Customers.csv",
            mime="text/csv"
        )

    # After segmentation: Overall RF Pivot Table
    if "segmented_df" in st.session_state:
        df_segmented = st.session_state["segmented_df"]
        st.subheader("📊 Overall RF Segment Proportions")
        pivot = pd.pivot_table(
            df_segmented,
            index="Frequency_Bucket",
            columns="Recency_Bucket",
            aggfunc="size",
            fill_value=0
        )
        # Sort Frequency and Recency buckets in descending order
        freq_order = sorted(pivot.index, key=lambda x: int(x[1:]), reverse=True)
        rec_order = sorted(pivot.columns, key=lambda x: int(x[1:]), reverse=True)
        pivot = pivot.reindex(index=freq_order, columns=rec_order)
        pivot_prop = pivot / pivot.values.sum()
        st.dataframe(pivot_prop.style.format("{:.2%}"))

    # Stratified Sampling Section
    st.subheader("🎯 Stratified Sampling")
    with st.form("sampling_form"):
        test_size = st.number_input("Test Group Size", value=7500, min_value=1000, step=100)
        control_size = st.number_input("Control Group Size", value=7500, min_value=1000, step=100)
        holdout_size = st.number_input("Holdout Group Size", value=15000, min_value=2000, step=100)
        sampling_submitted = st.form_submit_button("🔀 Run Sampling")
    
    if sampling_submitted:
        df_segmented = st.session_state.get("segmented_df")
        if df_segmented is None or "Segment" not in df_segmented.columns:
            st.error("Please apply segmentation before sampling.")
        else:
            sample_sizes = {"Test": test_size, "Control": control_size, "Holdout": holdout_size}
            with st.spinner("Performing stratified sampling..."):
                sampled_df = stratified_sample(df_segmented.copy(), sample_sizes)
                st.success("✅ Stratified sampling complete!")
                st.dataframe(sampled_df.head(20))
                st.session_state["sampled_df"] = sampled_df

    # After sampling: Create RF Pivot Matrices for each sample group
    if "sampled_df" in st.session_state:
        sampled_df = st.session_state["sampled_df"]
        for group in ["Test", "Control", "Holdout"]:
            st.subheader(f"📊 RF Segment Proportions for {group} Group")
            group_df = sampled_df[sampled_df["Group"] == group]
            pivot_group = pd.pivot_table(
                group_df,
                index="Frequency_Bucket",
                columns="Recency_Bucket",
                aggfunc="size",
                fill_value=0
            )
            # Sort descending
            freq_order_group = sorted(pivot_group.index, key=lambda x: int(x[1:]), reverse=True)
            rec_order_group = sorted(pivot_group.columns, key=lambda x: int(x[1:]), reverse=True)
            pivot_group = pivot_group.reindex(index=freq_order_group, columns=rec_order_group)
            pivot_group_prop = pivot_group / pivot_group.values.sum()
            st.dataframe(pivot_group_prop.style.format("{:.2%}"))

    # Download Buttons for Sample Groups
    if "sampled_df" in st.session_state:
        st.subheader("📥 Download Sampled Groups")
        sampled_df = st.session_state["sampled_df"]
        for group in ["Test", "Control", "Holdout"]:
            group_df = sampled_df[sampled_df["Group"] == group]
            csv_buffer = io.StringIO()
            group_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label=f"📥 Download {group} Group",
                data=csv_data,
                file_name=f"{group}_Group.csv",
                mime="text/csv"
            )
        combined_csv_buffer = io.StringIO()
        sampled_df.to_csv(combined_csv_buffer, index=False)
        combined_csv_data = combined_csv_buffer.getvalue()
        st.download_button(
            label="📥 Download Combined Sample Group",
            data=combined_csv_data,
            file_name="Combined_Sample_Group.csv",
            mime="text/csv"
        )
else:
    st.info("Please process the order history to begin segmentation and sampling.")
