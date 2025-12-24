"""
Data quality validation
"""

import pandas as pd


def validate_data(df):
    """
    Run data quality checks
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    # Completeness
    results['total_rows'] = len(df)
    results['missing_values'] = df.isnull().sum().to_dict()
    results['completeness'] = {
        col: f"{(1 - df[col].isnull().mean()) * 100:.1f}%"
        for col in df.columns
    }
    
    # Date range
    results['date_range'] = {
        'min': str(df['Date'].min()),
        'max': str(df['Date'].max()),
        'span_days': (df['Date'].max() - df['Date'].min()).days
    }
    
    # Chronological order check
    results['chronological'] = (df['Date'].diff().dropna() >= pd.Timedelta(0)).all()
    
    # Result consistency
    if all(col in df.columns for col in ['FTHG', 'FTAG', 'FTR']):
        df['result_check'] = df.apply(
            lambda row: (
                (row['FTR'] == 'H' and row['FTHG'] > row['FTAG']) or
                (row['FTR'] == 'D' and row['FTHG'] == row['FTAG']) or
                (row['FTR'] == 'A' and row['FTHG'] < row['FTAG'])
            ),
            axis=1
        )
        results['result_consistency'] = f"{df['result_check'].mean() * 100:.1f}%"
    
    # Odds validity (should be > 1.0)
    odds_cols = ['B365H', 'B365D', 'B365A']
    if all(col in df.columns for col in odds_cols):
        results['valid_odds'] = {
            col: f"{(df[col] > 1.0).mean() * 100:.1f}%"
            for col in odds_cols if col in df.columns
        }
    
    # Score validity (non-negative)
    score_cols = ['FTHG', 'FTAG']
    if all(col in df.columns for col in score_cols):
        results['valid_scores'] = {
            col: f"{(df[col] >= 0).mean() * 100:.1f}%"
            for col in score_cols
        }
    
    return results


def print_validation_report(results):
    """Print formatted validation report"""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    print(f"\nTotal Records: {results['total_rows']}")
    
    print(f"\nDate Range:")
    print(f"  From: {results['date_range']['min']}")
    print(f"  To: {results['date_range']['max']}")
    print(f"  Span: {results['date_range']['span_days']} days")
    
    print(f"\nChronological Order: {'PASS' if results['chronological'] else 'FAIL'}")
    
    if 'result_consistency' in results:
        print(f"\nResult Consistency: {results['result_consistency']}")
    
    if 'valid_odds' in results:
        print(f"\nOdds Validity (> 1.0):")
        for col, pct in results['valid_odds'].items():
            print(f"  {col}: {pct}")
    
    if 'valid_scores' in results:
        print(f"\nScore Validity (>= 0):")
        for col, pct in results['valid_scores'].items():
            print(f"  {col}: {pct}")
    
    print(f"\nCompleteness:")
    for col, pct in results['completeness'].items():
        if col not in ['result_check']:  # Skip temp columns
            print(f"  {col}: {pct}")
    
    print("\n" + "=" * 60)


def save_report(results, output_path="reports/data_quality.md"):
    """Save validation report to markdown"""
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"**Total Records:** {results['total_rows']}\n\n")
        
        f.write("## Date Range\n\n")
        f.write(f"- From: {results['date_range']['min']}\n")
        f.write(f"- To: {results['date_range']['max']}\n")
        f.write(f"- Span: {results['date_range']['span_days']} days\n\n")
        
        f.write("## Validation Checks\n\n")
        f.write(f"- Chronological Order: {'PASS' if results['chronological'] else 'FAIL'}\n")
        
        if 'result_consistency' in results:
            f.write(f"- Result Consistency: {results['result_consistency']}\n")
        
        if 'valid_odds' in results:
            f.write("\n## Odds Validity\n\n")
            for col, pct in results['valid_odds'].items():
                f.write(f"- {col}: {pct}\n")
        
        f.write("\n## Completeness\n\n")
        for col, pct in results['completeness'].items():
            if col not in ['result_check']:
                f.write(f"- {col}: {pct}\n")
    
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    from ingest import load_raw_data, clean_data
    
    df = load_raw_data()
    df = clean_data(df)
    
    results = validate_data(df)
    print_validation_report(results)
    save_report(results)

