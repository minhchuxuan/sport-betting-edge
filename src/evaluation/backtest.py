"""
Backtesting engine for betting strategies
Simulates betting performance with Kelly staking
"""

import pandas as pd
import numpy as np
from .metrics import calculate_ev, calculate_kelly_stake, calculate_roi, calculate_max_drawdown


class BettingBacktester:
    """
    Simulates betting performance over time using Kelly staking
    """
    
    def __init__(self, initial_bankroll=1000, kelly_fraction=0.25, ev_threshold=0.0):
        """
        Args:
            initial_bankroll: Starting capital
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            ev_threshold: Minimum EV to place bet (default 0.0)
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.ev_threshold = ev_threshold
        self.history = []
    
    def run(self, df, model_probs):
        """
        Execute backtest on test data
        
        Args:
            df: DataFrame with columns [Date, FTR, B365H, B365D, B365A]
            model_probs: np.array of shape (n_matches, 3) - [P_A, P_D, P_H]
        
        Returns:
            dict with performance metrics
        """
        bankroll = self.initial_bankroll
        n_bets = 0
        n_wins = 0
        total_staked = 0
        
        outcome_map = ['A', 'D', 'H']
        
        for idx, row in df.reset_index(drop=True).iterrows():
            probs = model_probs[idx]  # [P_A, P_D, P_H]
            odds = [row['B365A'], row['B365D'], row['B365H']]
            
            # Find best value bet
            best_ev = self.ev_threshold
            best_stake = 0
            best_outcome = None
            best_odds = 0
            
            for i, (p, o) in enumerate(zip(probs, odds)):
                ev = calculate_ev(p, o)
                if ev > best_ev:
                    best_ev = ev
                    kelly = calculate_kelly_stake(p, o, self.kelly_fraction)
                    best_stake = bankroll * kelly
                    best_outcome = outcome_map[i]
                    best_odds = o
            
            # Execute bet
            bet_placed = best_stake > 0 and best_outcome is not None
            won = False
            profit = 0
            
            if bet_placed:
                n_bets += 1
                total_staked += best_stake
                
                if row['FTR'] == best_outcome:
                    profit = best_stake * (best_odds - 1)
                    bankroll += profit
                    n_wins += 1
                    won = True
                else:
                    profit = -best_stake
                    bankroll -= best_stake
            
            # Record history
            self.history.append({
                'date': row['Date'] if 'Date' in row else idx,
                'bankroll': bankroll,
                'bet_placed': bet_placed,
                'stake': best_stake if bet_placed else 0,
                'outcome': best_outcome if bet_placed else None,
                'odds': best_odds if bet_placed else 0,
                'ev': best_ev if bet_placed else 0,
                'won': won,
                'profit': profit
            })
        
        # Calculate metrics
        history_df = pd.DataFrame(self.history)
        bankroll_values = history_df['bankroll'].values
        
        results = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll,
            'roi': calculate_roi(self.initial_bankroll, bankroll),
            'total_profit': bankroll - self.initial_bankroll,
            'total_bets': n_bets,
            'total_staked': total_staked,
            'win_rate': n_wins / n_bets if n_bets > 0 else 0,
            'max_drawdown': calculate_max_drawdown(bankroll_values),
            'history': history_df
        }
        
        return results
    
    def get_summary(self, results):
        """
        Print summary of backtest results
        """
        print("="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Bankroll:  ${results['initial_bankroll']:.2f}")
        print(f"Final Bankroll:    ${results['final_bankroll']:.2f}")
        print(f"Total Profit:      ${results['total_profit']:.2f}")
        print(f"ROI:               {results['roi']*100:.2f}%")
        print(f"Total Bets:        {results['total_bets']}")
        print(f"Total Staked:      ${results['total_staked']:.2f}")
        print(f"Win Rate:          {results['win_rate']*100:.1f}%")
        print(f"Max Drawdown:      {results['max_drawdown']*100:.1f}%")
        print("="*60)


def compare_strategies(df, model_probs, strategies):
    """
    Compare multiple betting strategies
    
    Args:
        df: Test data
        model_probs: Model predictions
        strategies: List of (name, kelly_fraction, ev_threshold) tuples
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, kelly_frac, ev_thresh in strategies:
        backtester = BettingBacktester(
            initial_bankroll=1000,
            kelly_fraction=kelly_frac,
            ev_threshold=ev_thresh
        )
        result = backtester.run(df, model_probs)
        
        results.append({
            'Strategy': name,
            'Kelly Fraction': kelly_frac,
            'EV Threshold': ev_thresh,
            'Final Bankroll': result['final_bankroll'],
            'ROI (%)': result['roi'] * 100,
            'Total Bets': result['total_bets'],
            'Win Rate (%)': result['win_rate'] * 100,
            'Max Drawdown (%)': result['max_drawdown'] * 100
        })
    
    return pd.DataFrame(results)

