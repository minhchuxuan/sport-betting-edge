"""
Data ingestion - load and clean football match data
"""

import pandas as pd
from pathlib import Path


def load_raw_data(data_dir="data/raw"):
    """
    Load all CSV files from raw data directory
    
    Returns:
        pd.DataFrame: Combined data from all seasons
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("E0_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, encoding='latin1')
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def clean_data(df):
    """
    Clean and standardize the dataset
    
    Steps:
    1. Parse dates (handle mixed formats)
    2. Sort chronologically
    3. Select core columns
    4. Handle missing values
    5. Validate data types
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    df = df.copy()
    
    # Parse dates - handle both dd/mm/yy and dd/mm/yyyy
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    if df['Date'].isna().any():
        # Try alternative format
        mask = df['Date'].isna()
        df.loc[mask, 'Date'] = pd.to_datetime(
            df.loc[mask, 'Date'], 
            format='%d/%m/%y', 
            errors='coerce'
        )
    
    # Sort chronologically (Arrow of Time)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Select core columns
    core_cols = [
        'Date', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR',  # Full time results
        'HS', 'AS', 'HST', 'AST',  # Shots
        'B365H', 'B365D', 'B365A'  # Betting odds
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in core_cols if col in df.columns]
    df = df[available_cols]
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    
    # Convert numeric columns
    numeric_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'B365H', 'B365D', 'B365A']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def save_processed_data(df, output_path="data/processed/matches_clean.parquet"):
    """Save cleaned data to parquet format"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} matches to {output_path}")


def run_pipeline():
    """Run the complete data ingestion pipeline"""
    print("Loading raw data...")
    df_raw = load_raw_data()
    print(f"Loaded {len(df_raw)} raw records")
    
    print("\nCleaning data...")
    df_clean = clean_data(df_raw)
    print(f"Cleaned to {len(df_clean)} valid records")
    
    print("\nSaving processed data...")
    save_processed_data(df_clean)
    
    return df_clean


if __name__ == "__main__":
    df = run_pipeline()
    print("\nPipeline complete!")
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

