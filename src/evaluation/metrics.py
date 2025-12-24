"""
Evaluation metrics for betting performance
Expected Value, Kelly Criterion, ROI, etc.
"""

import numpy as np
import pandas as pd


def calculate_ev(model_prob, odds):
    """
    Calculate Expected Value for a bet
    
    EV = (P_model * (Odds - 1)) - (1 - P_model)
    
    Args:
        model_prob: Model's probability estimate
        odds: Decimal odds offered
    
    Returns:
        float: Expected Value (positive = value bet)
    """
    return (model_prob * (odds - 1)) - (1 - model_prob)


def calculate_kelly_stake(model_prob, odds, fraction=0.25):
    """
    Calculate optimal Kelly Criterion stake
    
    f* = (bp - q) / b
    where b = odds - 1, p = prob of win, q = 1 - p
    
    Args:
        model_prob: Model's probability estimate
        odds: Decimal odds offered
        fraction: Kelly fraction (default 0.25 = quarter Kelly)
    
    Returns:
        float: Fraction of bankroll to stake (0 if negative EV)
    """
    b = odds - 1
    p = model_prob
    q = 1 - p
    
    if b <= 0:
        return 0
    
    f = (b * p - q) / b
    return max(0, f * fraction)


def calculate_implied_prob(odds):
    """
    Calculate bookmaker implied probability from odds
    
    Args:
        odds: Decimal odds
    
    Returns:
        float: Implied probability
    """
    return 1 / odds if odds > 0 else 0


def normalize_probabilities(probs):
    """
    Normalize probabilities to sum to 1
    (Removes bookmaker overround)
    
    Args:
        probs: Array of probabilities
    
    Returns:
        Array of normalized probabilities
    """
    return probs / probs.sum()


def calculate_roi(initial_bankroll, final_bankroll):
    """
    Calculate Return on Investment
    
    Args:
        initial_bankroll: Starting capital
        final_bankroll: Ending capital
    
    Returns:
        float: ROI as decimal (0.1 = 10%)
    """
    return (final_bankroll - initial_bankroll) / initial_bankroll


def calculate_max_drawdown(bankroll_history):
    """
    Calculate maximum drawdown from bankroll history
    
    Args:
        bankroll_history: List/array of bankroll values over time
    
    Returns:
        float: Maximum drawdown as decimal (0.2 = 20%)
    """
    if len(bankroll_history) == 0:
        return 0
    
    peak = bankroll_history[0]
    max_dd = 0
    
    for value in bankroll_history:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default 0)
    
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

