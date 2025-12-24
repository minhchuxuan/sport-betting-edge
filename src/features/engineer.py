"""
Feature engineering pipeline
Combines Elo ratings and rolling statistics
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from elo import calculate_elo_ratings
from rolling import add_rolling_features, add_form_points


def engineer_features(df, elo_k=20, rolling_windows=[5]):
    """
    Complete feature engineering pipeline
    
    Steps:
    1. Calculate Elo ratings (pre-match values)
    2. Add rolling statistics (with shift to prevent leakage)
    3. Add form points
    4. Create difference features
    5. Drop rows with missing features (early matches)
    
    Args:
        df: Cleaned match data (sorted by Date)
        elo_k: Elo K-factor
        rolling_windows: Window sizes for rolling features
    
    Returns:
        DataFrame with engineered features
    """
    print("Engineering features...")
    
    # 1. Elo ratings
    print("  - Calculating Elo ratings...")
    df = calculate_elo_ratings(df, k_factor=elo_k)
    
    # 2. Rolling statistics
    print("  - Adding rolling statistics...")
    df = add_rolling_features(df, windows=rolling_windows)
    
    # 3. Form points
    print("  - Adding form points...")
    df = add_form_points(df, window=rolling_windows[0])
    
    # 4. Difference features
    print("  - Creating difference features...")
    for window in rolling_windows:
        df[f'Goals_Diff_L{window}'] = df[f'Home_Goals_L{window}'] - df[f'Away_Goals_L{window}']
        df[f'Form_Diff_L{window}'] = df[f'Home_Form_L{window}'] - df[f'Away_Form_L{window}']
    
    # 5. Drop rows with missing features (first few matches per team)
    print("  - Removing rows with missing features...")
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    print(f"    Dropped {initial_count - final_count} rows (early matches without history)")
    
    print(f"\nFeature engineering complete!")
    print(f"Final dataset: {len(df)} matches, {len(df.columns)} features")
    
    return df


def save_features(df, output_path='data/processed/features.parquet'):
    """Save feature-engineered dataset"""
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_path}")


def run_pipeline():
    """Run complete feature engineering pipeline"""
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_parquet('data/processed/matches_clean.parquet')
    print(f"Loaded {len(df)} matches\n")
    
    # Engineer features
    df = engineer_features(df, elo_k=20, rolling_windows=[5])
    
    # Save
    save_features(df)
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Total matches: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nFeature categories:")
    print(f"  - Elo: 3 features (Home_Elo, Away_Elo, Elo_Diff)")
    print(f"  - Rolling stats: 10 features (goals, conceded, shots, sot)")
    print(f"  - Form: 2 features (Home_Form_L5, Away_Form_L5)")
    print(f"  - Differences: 2 features (Goals_Diff_L5, Form_Diff_L5)")
    print(f"\nSample features:")
    feature_cols = [col for col in df.columns if any(x in col for x in ['Elo', '_L5', 'Diff'])]
    print(df[feature_cols].head())
    
    return df


if __name__ == "__main__":
    df = run_pipeline()

