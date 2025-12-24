"""
Test for data leakage prevention
Critical: Ensure no future data is used in features
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.elo import calculate_elo_ratings
from features.rolling import add_rolling_features, add_form_points


def test_elo_uses_prematch_values():
    """Elo ratings should use values BEFORE the match"""
    # Create simple test data
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'HomeTeam': ['A', 'B', 'A', 'B', 'A'],
        'AwayTeam': ['B', 'A', 'B', 'A', 'B'],
        'FTHG': [2, 0, 1, 3, 2],
        'FTAG': [0, 2, 1, 0, 1],
        'FTR': ['H', 'A', 'D', 'H', 'H'],
        'HS': [10, 8, 12, 15, 11],
        'AS': [5, 12, 10, 6, 8],
        'HST': [5, 3, 6, 8, 5],
        'AST': [2, 6, 5, 3, 4]
    })
    
    result = calculate_elo_ratings(df)
    
    # First match should have initial Elo (1500)
    assert result.iloc[0]['Home_Elo'] == 1500, "First match should have initial Elo"
    assert result.iloc[0]['Away_Elo'] == 1500, "First match should have initial Elo"
    
    # Second match should have updated Elo from first match
    assert result.iloc[1]['Home_Elo'] != 1500, "Elo should update after first match"
    
    print("[PASS] Elo test: Pre-match values used")


def test_rolling_features_shifted():
    """Rolling features should be None for first match (no history)"""
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'HomeTeam': ['A', 'B', 'A', 'B', 'A'],
        'AwayTeam': ['B', 'A', 'B', 'A', 'B'],
        'FTHG': [2, 0, 1, 3, 2],
        'FTAG': [0, 2, 1, 0, 1],
        'FTR': ['H', 'A', 'D', 'H', 'H'],
        'HS': [10, 8, 12, 15, 11],
        'AS': [5, 12, 10, 6, 8],
        'HST': [5, 3, 6, 8, 5],
        'AST': [2, 6, 5, 3, 4]
    })
    
    result = add_rolling_features(df, windows=[3])
    
    # First match should have None (no history)
    assert pd.isna(result.iloc[0]['Home_Goals_L3']), "First match should have no rolling features"
    assert pd.isna(result.iloc[0]['Away_Goals_L3']), "First match should have no rolling features"
    
    print("[PASS] Rolling features test: shift(1) applied correctly")


def test_form_points_shifted():
    """Form points should be None for first match"""
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5),
        'HomeTeam': ['A', 'B', 'A', 'B', 'A'],
        'AwayTeam': ['B', 'A', 'B', 'A', 'B'],
        'FTHG': [2, 0, 1, 3, 2],
        'FTAG': [0, 2, 1, 0, 1],
        'FTR': ['H', 'A', 'D', 'H', 'H'],
        'HS': [10, 8, 12, 15, 11],
        'AS': [5, 12, 10, 6, 8],
        'HST': [5, 3, 6, 8, 5],
        'AST': [2, 6, 5, 3, 4]
    })
    
    result = add_form_points(df, window=3)
    
    # First match should have None (no history)
    assert pd.isna(result.iloc[0]['Home_Form_L3']), "First match should have no form points"
    assert pd.isna(result.iloc[0]['Away_Form_L3']), "First match should have no form points"
    
    print("[PASS] Form points test: shift(1) applied correctly")


def test_chronological_order():
    """Features should respect chronological order"""
    df = pd.read_parquet('data/processed/features.parquet')
    
    # Check dates are sorted
    assert (df['Date'].diff().dropna() >= pd.Timedelta(0)).all(), "Data must be chronologically sorted"
    
    print("[PASS] Chronological order test")


def run_all_tests():
    """Run all leakage prevention tests"""
    print("="*60)
    print("LEAKAGE PREVENTION TESTS")
    print("="*60)
    print()
    
    try:
        test_elo_uses_prematch_values()
        test_rolling_features_shifted()
        test_form_points_shifted()
        test_chronological_order()
        
        print()
        print("="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nNo data leakage detected!")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

