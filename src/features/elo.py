"""
Elo rating calculation for team strength
"""

import pandas as pd


def calculate_elo_ratings(df, initial_elo=1500, k_factor=20):
    """
    Calculate Elo ratings for all teams over time
    
    Key principle: Use pre-match Elo values (no look-ahead bias)
    
    Args:
        df: DataFrame with Date, HomeTeam, AwayTeam, FTR columns (sorted by Date)
        initial_elo: Starting Elo rating for all teams
        k_factor: Elo update sensitivity (higher = more volatile)
    
    Returns:
        DataFrame with Home_Elo and Away_Elo columns added
    """
    df = df.copy()
    
    # Initialize Elo ratings for all teams
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: initial_elo for team in teams}
    
    # Store pre-match Elo values
    home_elos = []
    away_elos = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get current Elo (before this match)
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]
        
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if row['FTR'] == 'H':
            actual_home, actual_away = 1, 0
        elif row['FTR'] == 'D':
            actual_home, actual_away = 0.5, 0.5
        else:  # 'A'
            actual_home, actual_away = 0, 1
        
        # Update Elo ratings (after this match)
        elo_ratings[home_team] += k_factor * (actual_home - expected_home)
        elo_ratings[away_team] += k_factor * (actual_away - expected_away)
    
    df['Home_Elo'] = home_elos
    df['Away_Elo'] = away_elos
    df['Elo_Diff'] = df['Home_Elo'] - df['Away_Elo']
    
    return df


if __name__ == "__main__":
    # Test
    df = pd.read_parquet('data/processed/matches_clean.parquet')
    df = calculate_elo_ratings(df)
    
    print("Elo ratings calculated!")
    print(f"\nSample:")
    print(df[['Date', 'HomeTeam', 'AwayTeam', 'Home_Elo', 'Away_Elo', 'Elo_Diff']].head(10))
    
    print(f"\nElo range:")
    print(f"Min: {df['Home_Elo'].min():.1f}")
    print(f"Max: {df['Home_Elo'].max():.1f}")

