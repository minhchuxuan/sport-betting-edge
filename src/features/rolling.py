"""
Rolling statistics with leakage prevention
"""

import pandas as pd


def add_rolling_features(df, windows=[5]):
    """
    Add rolling statistics for team form
    
    Key principle: Use shift(1) to prevent look-ahead bias
    
    Args:
        df: DataFrame with team and match data (sorted by Date)
        windows: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    
    # Initialize feature columns
    for window in windows:
        for prefix in ['Home', 'Away']:
            df[f'{prefix}_Goals_L{window}'] = None
            df[f'{prefix}_Conceded_L{window}'] = None
            df[f'{prefix}_Shots_L{window}'] = None
            df[f'{prefix}_ShotsOnTarget_L{window}'] = None
    
    # Track team statistics
    team_history = {}
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Initialize team history if needed
        if home_team not in team_history:
            team_history[home_team] = {'goals': [], 'conceded': [], 'shots': [], 'sot': []}
        if away_team not in team_history:
            team_history[away_team] = {'goals': [], 'conceded': [], 'shots': [], 'sot': []}
        
        # Calculate rolling features using historical data (before this match)
        for window in windows:
            # Home team
            h_hist = team_history[home_team]
            df.loc[idx, f'Home_Goals_L{window}'] = (
                sum(h_hist['goals'][-window:]) / len(h_hist['goals'][-window:]) 
                if len(h_hist['goals']) > 0 else None
            )
            df.loc[idx, f'Home_Conceded_L{window}'] = (
                sum(h_hist['conceded'][-window:]) / len(h_hist['conceded'][-window:])
                if len(h_hist['conceded']) > 0 else None
            )
            df.loc[idx, f'Home_Shots_L{window}'] = (
                sum(h_hist['shots'][-window:]) / len(h_hist['shots'][-window:])
                if len(h_hist['shots']) > 0 else None
            )
            df.loc[idx, f'Home_ShotsOnTarget_L{window}'] = (
                sum(h_hist['sot'][-window:]) / len(h_hist['sot'][-window:])
                if len(h_hist['sot']) > 0 else None
            )
            
            # Away team
            a_hist = team_history[away_team]
            df.loc[idx, f'Away_Goals_L{window}'] = (
                sum(a_hist['goals'][-window:]) / len(a_hist['goals'][-window:])
                if len(a_hist['goals']) > 0 else None
            )
            df.loc[idx, f'Away_Conceded_L{window}'] = (
                sum(a_hist['conceded'][-window:]) / len(a_hist['conceded'][-window:])
                if len(a_hist['conceded']) > 0 else None
            )
            df.loc[idx, f'Away_Shots_L{window}'] = (
                sum(a_hist['shots'][-window:]) / len(a_hist['shots'][-window:])
                if len(a_hist['shots']) > 0 else None
            )
            df.loc[idx, f'Away_ShotsOnTarget_L{window}'] = (
                sum(a_hist['sot'][-window:]) / len(a_hist['sot'][-window:])
                if len(a_hist['sot']) > 0 else None
            )
        
        # Update team history (after this match)
        team_history[home_team]['goals'].append(row['FTHG'])
        team_history[home_team]['conceded'].append(row['FTAG'])
        team_history[home_team]['shots'].append(row['HS'])
        team_history[home_team]['sot'].append(row['HST'])
        
        team_history[away_team]['goals'].append(row['FTAG'])
        team_history[away_team]['conceded'].append(row['FTHG'])
        team_history[away_team]['shots'].append(row['AS'])
        team_history[away_team]['sot'].append(row['AST'])
    
    return df


def add_form_points(df, window=5):
    """
    Add form points (W=3, D=1, L=0) over last N games
    
    Args:
        df: DataFrame with FTR column
        window: Number of recent games to consider
    
    Returns:
        DataFrame with form points added
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    
    # Initialize columns
    df[f'Home_Form_L{window}'] = None
    df[f'Away_Form_L{window}'] = None
    
    # Track team form
    team_points = {}
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Initialize if needed
        if home_team not in team_points:
            team_points[home_team] = []
        if away_team not in team_points:
            team_points[away_team] = []
        
        # Calculate form using historical data (before this match)
        df.loc[idx, f'Home_Form_L{window}'] = (
            sum(team_points[home_team][-window:]) if len(team_points[home_team]) > 0 else None
        )
        df.loc[idx, f'Away_Form_L{window}'] = (
            sum(team_points[away_team][-window:]) if len(team_points[away_team]) > 0 else None
        )
        
        # Update team points (after this match)
        if row['FTR'] == 'H':
            team_points[home_team].append(3)
            team_points[away_team].append(0)
        elif row['FTR'] == 'D':
            team_points[home_team].append(1)
            team_points[away_team].append(1)
        else:  # 'A'
            team_points[home_team].append(0)
            team_points[away_team].append(3)
    
    return df


if __name__ == "__main__":
    # Test
    df = pd.read_parquet('data/processed/matches_clean.parquet')
    
    print("Adding rolling features...")
    df = add_rolling_features(df, windows=[5])
    df = add_form_points(df, window=5)
    
    print("Rolling features calculated!")
    print(f"\nSample (first 10 rows):")
    print(df[['Date', 'HomeTeam', 'Home_Goals_L5', 'Home_Conceded_L5', 'Home_Form_L5']].head(10))
    
    print(f"\nSample (rows 20-30, after teams have history):")
    print(df[['Date', 'HomeTeam', 'Home_Goals_L5', 'Home_Conceded_L5', 'Home_Form_L5']].iloc[20:30])
    
    print(f"\nFeature columns added:")
    rolling_cols = [col for col in df.columns if '_L5' in col]
    print(rolling_cols)
