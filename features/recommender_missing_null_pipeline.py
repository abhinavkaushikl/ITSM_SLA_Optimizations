import pandas as pd

def clean_itsm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and imputes ITSM SLA ticket dataset.

    Args:
        df (pd.DataFrame): Raw ITSM SLA dataset

    Returns:
        pd.DataFrame: Cleaned and imputed DataFrame
    """
    df = df.copy()

    # Parse datetime fields
    datetime_cols = ["Created At", "Responded At", "Resolved At"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fill missing categorical values with "Unknown"
    cat_fill = [
        "Technician Skill Level", "Customer Segment", "Priority",
        "Ticket Type", "Region", "Impact Level",
        "Technical Domain", "Primary Tool Used", "Certifications"
    ]
    for col in cat_fill:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill missing numeric values with median or safe defaults
    num_fill_median = [
        "Resolution Time (hours)", "MTTR (hours)", "MTBF (days)",
        "Customer Satisfaction Score (CSAT)"
    ]
    for col in num_fill_median:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Additional derived time-based features
    df["Response Delay (minutes)"] = (df["Responded At"] - df["Created At"]).dt.total_seconds() / 60.0
    df["Resolution Duration (hours)"] = (df["Resolved At"] - df["Created At"]).dt.total_seconds() / 3600.0

    df["Response Delay (minutes)"] = df["Response Delay (minutes)"].fillna(df["Response Delay (minutes)"].median())
    df["Resolution Duration (hours)"] = df["Resolution Duration (hours)"].fillna(df["Resolution Duration (hours)"].median())

    return df