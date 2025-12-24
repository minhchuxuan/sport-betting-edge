"""
Sports Betting Edge Dashboard
Streamlit application for model predictions and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import calculate_ev, calculate_kelly_stake, calculate_implied_prob

# Figure save directory
FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def save_and_show_figure(fig, filename):
    """
    Save figure to reports/figures and display in Streamlit.
    
    Args:
        fig: matplotlib figure object
        filename: name of file (without path, with .png extension)
    """
    # Save figure
    save_path = FIGURES_DIR / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Close figure to free memory
    plt.close(fig)

# Page config
st.set_page_config(
    page_title="Sports Betting Edge Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Premium Modern Design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header with Gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    /* Warning Box - Enhanced */
    .warning-box {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe4cc 100%);
        border: 2px solid #ff9800;
        border-left: 6px solid #ff6b00;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.15);
        animation: slideInLeft 0.6s ease-out;
    }
    
    .warning-box h3 {
        color: #e65100;
        margin-top: 0;
        font-weight: 700;
    }
    
    .warning-box p {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }
    
    /* Streamlit Metric Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        border-color: #e0e7ff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: transparent !important;
    }
    
    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 2px solid #e0e7ff;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Info/Warning Boxes */
    .stSuccess {
        background: linear-gradient(135deg, #d4f4dd 0%, #c3f0cf 100%) !important;
        border-left: 5px solid #22c55e !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        animation: pulse 0.5s ease-out;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-left: 5px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        border-left: 5px solid #f59e0b !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.875rem;
        border-bottom: 3px solid #e0e7ff;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem;
        color: #475569;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Selectbox Enhancement */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e0e7ff;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider Enhancement */
    .stSlider > div > div > div {
        background-color: #667eea !important;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.9;
        }
    }
    
    /* Value Bet Badge */
    .value-bet-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
        animation: pulse 1s infinite;
    }
    
    /* No Value Badge */
    .no-value-badge {
        display: inline-block;
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Markdown Improvements */
    .stMarkdown {
        line-height: 1.7;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        baseline = joblib.load('models/logistic_baseline.joblib')
        xgb = joblib.load('models/xgb_calibrated.joblib')
        label_map = joblib.load('models/label_map.joblib')
        return baseline, xgb, label_map
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


@st.cache_data
def load_data():
    """Load feature data"""
    try:
        df = pd.read_parquet('data/processed/features.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def predict_match(model, features):
    """Get predictions for a match"""
    probs = model.predict_proba(features.reshape(1, -1))[0]
    return probs  # [Away, Draw, Home]


def main():
    # Header
    st.markdown('<div class="main-header">Sports Betting Edge Dashboard</div>', unsafe_allow_html=True)
    
    # Warning banner
    st.markdown("""
    <div class="warning-box">
        <h3>IMPORTANT DISCLAIMER</h3>
        <p><strong>This is an academic project only. DO NOT use for real gambling.</strong></p>
        <p>Backtesting shows <strong>negative ROI (-64% for XGBoost, -41% for Baseline)</strong>. 
        Models are well-calibrated but cannot beat bookmaker odds consistently.</p>
        <p>This dashboard demonstrates ML concepts, not profitable betting strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    baseline, xgb, label_map = load_models()
    df = load_data()
    
    if baseline is None or xgb is None or df is None:
        st.error("Failed to load models or data. Please check file paths.")
        return
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Info box explaining behavior
    st.sidebar.info("""
    **Quick Guide**
    
    These settings control your betting strategy, not the model predictions.
    
    - **Model**: Choose prediction algorithm
    - **Kelly Fraction**: Adjust bet size (risk management)
    - **EV Threshold**: Filter low-value bets
    
    Note: Probabilities and odds remain constant as they come from the trained model and historical data.
    """)
    
    st.sidebar.markdown("---")
    
    model_choice = st.sidebar.selectbox(
        "Model Selection",
        ["XGBoost (Calibrated)", "Baseline (Logistic)"],
        help="Choose which trained model to use for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Bankroll Management")
    
    bankroll = st.sidebar.number_input(
        "Starting Bankroll ($)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Total amount available for betting"
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more conservative (recommended: 0.25)"
    )
    
    st.sidebar.caption(f"Current setting: **{kelly_fraction:.0%}** (Quarter Kelly)" if kelly_fraction == 0.25 else f"Current setting: **{kelly_fraction:.0%}**")
    
    ev_threshold = st.sidebar.slider(
        "Min EV Threshold (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Minimum expected value to show bet"
    ) / 100
    
    st.sidebar.caption(f"Filtering bets below **{ev_threshold:.1%}** expected value")
    
    # Select model
    model = xgb if model_choice == "XGBoost (Calibrated)" else baseline
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Match Predictions", "Model Performance", "Visualizations & EDA", "About"])
    
    with tab1:
        st.header("Match Predictions")
        
        # Explain result codes
        st.info("""
        **Match Result Codes**: 
        - **H** = Home Win (home team wins)
        - **A** = Away Win (away team wins)
        - **D** = Draw (tied match)
        """)
        
        # Get test data
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
        
        # Select a match
        match_idx = st.selectbox(
            "Select Match",
            range(len(df_test)),
            format_func=lambda x: f"Match {x+1}: {df_test.iloc[x]['HomeTeam']} vs {df_test.iloc[x]['AwayTeam']} ({df_test.iloc[x]['Date']})"
        )
        
        match = df_test.iloc[match_idx]
        
        # Prepare features
        feature_cols = [
            'Home_Elo', 'Away_Elo', 'Elo_Diff',
            'Home_Goals_L5', 'Away_Goals_L5',
            'Home_Conceded_L5', 'Away_Conceded_L5',
            'Home_Shots_L5', 'Away_Shots_L5',
            'Home_ShotsOnTarget_L5', 'Away_ShotsOnTarget_L5',
            'Home_Form_L5', 'Away_Form_L5',
            'Goals_Diff_L5', 'Form_Diff_L5'
        ]
        
        features = match[feature_cols].values
        
        # Get predictions
        probs = predict_match(model, features)
        
        # Display match info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(match['HomeTeam'])
            st.metric("Elo Rating", f"{match['Home_Elo']:.0f}")
            st.metric("Recent Form", f"{match['Home_Form_L5']:.1f} pts")
        
        with col2:
            st.subheader("Match Info")
            st.write(f"**Date:** {match['Date']}")
            st.write(f"**Actual Result:** {match['FTR']}")
            st.write(f"**Score:** {match['FTHG']:.0f} - {match['FTAG']:.0f}")
        
        with col3:
            st.subheader(match['AwayTeam'])
            st.metric("Elo Rating", f"{match['Away_Elo']:.0f}")
            st.metric("Recent Form", f"{match['Away_Form_L5']:.1f} pts")
        
        st.markdown("---")
        
        # Predictions
        st.subheader("Model Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        outcomes = ['Away Win', 'Draw', 'Home Win']
        odds = [match['B365A'], match['B365D'], match['B365H']]
        
        for i, (col, outcome, prob, odd) in enumerate(zip([col1, col2, col3], outcomes, probs, odds)):
            with col:
                st.markdown(f"### {outcome}")
                st.metric("Model Probability", f"{prob*100:.1f}%")
                st.metric("Bookmaker Odds", f"{odd:.2f}")
                
                implied_prob = calculate_implied_prob(odd)
                st.metric("Implied Probability", f"{implied_prob*100:.1f}%")
                
                ev = calculate_ev(prob, odd)
                st.metric("Expected Value", f"{ev*100:+.2f}%")
                
                if ev > ev_threshold:
                    kelly = calculate_kelly_stake(prob, odd, kelly_fraction)
                    stake = bankroll * kelly
                    st.success(f"✅ VALUE BET")
                    st.write(f"**Suggested Stake:** ${stake:.2f}")
                else:
                    st.info("No value")
        
        # Probability chart
        st.markdown("---")
        st.subheader("Probability Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(3)
        width = 0.35
        
        implied_probs = [calculate_implied_prob(o) for o in odds]
        
        ax.bar(x - width/2, probs, width, label='Model', color='steelblue')
        ax.bar(x + width/2, implied_probs, width, label='Bookmaker', color='coral')
        
        ax.set_ylabel('Probability')
        ax.set_title('Model vs Bookmaker Probabilities')
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab2:
        st.header("Model Performance & Backtesting")
        
        # Get test data
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:].copy()
        
        # Ensure Date column is datetime
        if df_test['Date'].dtype != 'datetime64[ns]':
            df_test['Date'] = pd.to_datetime(df_test['Date'])
        
        # Backtesting controls section
        st.subheader("📅 Time Period Selection")
        st.info("""
        **Interactive Backtesting**: Select a custom date range to analyze model performance over specific time periods.
        This helps explore performance variance, seasonal effects, and sample size impact.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selector
            min_date = df_test['Date'].min().date()
            max_date = df_test['Date'].max().date()
            
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="First match date to include in backtest"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Last match date to include in backtest"
            )
        
        # Filter data by date range
        df_filtered = df_test[
            (df_test['Date'].dt.date >= start_date) & 
            (df_test['Date'].dt.date <= end_date)
        ].reset_index(drop=True)
        
        # Show period info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Matches in Period", len(df_filtered))
        with col2:
            st.metric("📅 Days Span", (end_date - start_date).days)
        with col3:
            period_pct = (len(df_filtered) / len(df_test)) * 100 if len(df_test) > 0 else 0
            st.metric("📈 % of Test Set", f"{period_pct:.1f}%")
        
        if len(df_filtered) == 0:
            st.error("No matches in selected date range. Please adjust dates.")
        else:
            st.markdown("---")
            
            # Run backtest on filtered data
            st.subheader("🎯 Backtesting Results")
            st.markdown(f"*Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}*")
            
            # Prepare features for filtered data
            feature_cols = [
                'Home_Elo', 'Away_Elo', 'Elo_Diff',
                'Home_Goals_L5', 'Away_Goals_L5',
                'Home_Conceded_L5', 'Away_Conceded_L5',
                'Home_Shots_L5', 'Away_Shots_L5',
                'Home_ShotsOnTarget_L5', 'Away_ShotsOnTarget_L5',
                'Home_Form_L5', 'Away_Form_L5',
                'Goals_Diff_L5', 'Form_Diff_L5'
            ]
            
            X_filtered = df_filtered[feature_cols]
            model_probs = model.predict_proba(X_filtered)
            
            # Run backtest using the backtester class
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from evaluation.backtest import BettingBacktester
            
            backtester = BettingBacktester(
                initial_bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                ev_threshold=ev_threshold
            )
            
            results = backtester.run(df_filtered, model_probs)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Bets", results['total_bets'], help="Number of value bets placed")
            with col2:
                roi_color = "normal" if results['roi'] >= 0 else "inverse"
                st.metric("ROI", f"{results['roi']*100:.2f}%", 
                         delta=f"{results['roi']*100:.2f}%", 
                         delta_color=roi_color, 
                         help="Return on Investment")
            with col3:
                st.metric("Win Rate", f"{results['win_rate']*100:.1f}%", 
                         help="Percentage of winning bets")
            with col4:
                st.metric("Max Drawdown", f"{results['max_drawdown']*100:.1f}%", 
                         delta=f"{results['max_drawdown']*100:.1f}%",
                         delta_color="inverse",
                         help="Largest peak-to-trough decline")
            
            # Show profit/loss
            profit_col1, profit_col2, profit_col3 = st.columns(3)
            with profit_col1:
                st.metric("Initial Bankroll", f"${results['initial_bankroll']:.2f}")
            with profit_col2:
                st.metric("Final Bankroll", f"${results['final_bankroll']:.2f}")
            with profit_col3:
                profit_color = "normal" if results['total_profit'] >= 0 else "inverse"
                st.metric("Total Profit", f"${results['total_profit']:.2f}",
                         delta=f"${results['total_profit']:.2f}",
                         delta_color=profit_color)
            
            # Interpretation
            if results['roi'] > 0:
                st.success(f"✅ **Profitable Period!** Model achieved {results['roi']*100:.2f}% ROI over {len(df_filtered)} matches.")
            elif results['roi'] > -0.10:
                st.warning(f"⚠️ **Slight Loss**: Model lost {abs(results['roi']*100):.2f}% over this period. Small sample variance may be a factor.")
            else:
                st.error(f"❌ **Significant Loss**: Model lost {abs(results['roi']*100):.2f}% over this period.")
            
            st.markdown("---")
            
            # Bankroll evolution chart
            st.subheader("💰 Bankroll Evolution Over Time")
            
            history_df = results['history']
            bet_history = history_df[history_df['bet_placed']].reset_index(drop=True)
            
            if len(bet_history) > 0:
                fig, ax = plt.subplots(figsize=(12, 5))
                
                bankroll_series = bet_history['bankroll'].values
                bet_numbers = range(len(bankroll_series))
                
                ax.plot(bet_numbers, bankroll_series, linewidth=2, color='steelblue', label='Bankroll')
                ax.axhline(y=bankroll, color='red', linestyle='--', alpha=0.5, label='Starting Bankroll')
                ax.fill_between(bet_numbers, bankroll_series, bankroll, alpha=0.3)
                
                ax.set_xlabel('Bet Number')
                ax.set_ylabel('Bankroll ($)')
                ax.set_title(f'Bankroll Evolution ({start_date} to {end_date})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Annotate peak and final if enough bets
                if len(bankroll_series) > 5:
                    peak_idx = np.argmax(bankroll_series)
                    peak_val = bankroll_series[peak_idx]
                    
                    # Calculate y-axis range for smart positioning
                    y_range = max(bankroll_series) - min(bankroll_series)
                    y_max = max(bankroll_series)
                    
                    # If peak is in top 20% of chart, annotate below; otherwise above
                    if peak_val > (y_max - 0.2 * y_range):
                        # Peak near top - annotate below
                        text_y = peak_val - bankroll*0.08
                    else:
                        # Peak has room above - annotate above
                        text_y = peak_val + bankroll*0.05
                    
                    ax.annotate(f'Peak: ${peak_val:.2f}', 
                               xy=(peak_idx, peak_val), 
                               xytext=(peak_idx + len(bankroll_series)*0.05, text_y),
                               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
                    
                    final_val = bankroll_series[-1]
                    ax.annotate(f'Final: ${final_val:.2f}\n(ROI: {results["roi"]*100:.1f}%)', 
                               xy=(len(bankroll_series)-1, final_val), 
                               xytext=(len(bankroll_series)*0.7, final_val - bankroll*0.1),
                               arrowprops=dict(arrowstyle='->', color='red' if results['roi'] < 0 else 'green', lw=1.5))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption(f"*Bankroll evolution over {len(bet_history)} bets with {kelly_fraction:.0%} Kelly staking*")
            else:
                st.info("No bets placed in this period. Try lowering the EV threshold or selecting a different date range.")
            
            st.markdown("---")
            
            # Sample size analysis
            st.subheader("📊 Sample Size & Variance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Period Statistics:**
                - Matches: {len(df_filtered)}
                - Bets Placed: {results['total_bets']}
                - Bet Frequency: {(results['total_bets']/len(df_filtered)*100):.1f}%
                - Avg Stake: ${results['total_staked']/results['total_bets'] if results['total_bets'] > 0 else 0:.2f}
                """)
            
            with col2:
                # Provide context on sample size
                if len(df_filtered) < 50:
                    st.warning("⚠️ **Small Sample**: Results may be heavily influenced by luck/variance.")
                elif len(df_filtered) < 150:
                    st.info("ℹ️ **Medium Sample**: Moderate confidence in results, but variance still significant.")
                else:
                    st.success("✅ **Large Sample**: Higher confidence in results due to larger sample size.")
            
            st.markdown("---")
            
            # Comparison with full test set
            st.subheader("📈 Comparison with Full Test Period")
            
            # Full period stats (hardcoded from evaluation report for comparison)
            full_test_stats = {
                'XGBoost (Calibrated)': {'roi': -0.636, 'total_bets': 357, 'win_rate': 0.272},
                'Baseline (Logistic)': {'roi': -0.411, 'total_bets': 331, 'win_rate': 0.335}
            }
            
            model_name = "XGBoost (Calibrated)" if model_choice == "XGBoost (Calibrated)" else "Baseline (Logistic)"
            full_stats = full_test_stats[model_name]
            
            comparison_data = {
                'Metric': ['ROI', 'Total Bets', 'Win Rate', 'Matches'],
                'Selected Period': [
                    f"{results['roi']*100:.2f}%",
                    results['total_bets'],
                    f"{results['win_rate']*100:.1f}%",
                    len(df_filtered)
                ],
                'Full Test Set': [
                    f"{full_stats['roi']*100:.2f}%",
                    full_stats['total_bets'],
                    f"{full_stats['win_rate']*100:.1f}%",
                    len(df_test)
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Insight
            roi_diff = results['roi'] - full_stats['roi']
            if abs(roi_diff) > 0.1:
                if roi_diff > 0:
                    st.success(f"✨ This period performed **{roi_diff*100:.1f}% better** than the full test set!")
                else:
                    st.warning(f"⬇️ This period performed **{abs(roi_diff)*100:.1f}% worse** than the full test set.")
            else:
                st.info(f"ℹ️ This period performed similarly to the full test set (±{abs(roi_diff)*100:.1f}% difference).")
        
        st.markdown("---")
        
        # Academic value
        st.info("""
        **🎓 Educational Value of Time Period Analysis:**
        
        1. **Sample Size Matters**: Small periods (50 matches) show high variance; large periods (300+) are more reliable
        2. **No Cherry-Picking**: Honest reporting includes both good and bad periods
        3. **Variance Reality**: Even with a 60% win rate model, individual periods can lose money
        4. **Market Efficiency**: Consistent profitability across all periods is extremely rare
        5. **Risk Management**: Kelly Criterion prevents complete bankroll loss even in bad periods
        
        Try different date ranges to see how results vary!
        """)
    
    with tab3:
        st.header("Visualizations & Exploratory Analysis")
        st.markdown("*Based on Course Modules: EDA, Visualization, Classification Evaluation*")
        
        # Get all data for visualizations
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        # Prepare features
        feature_cols = [
            'Home_Elo', 'Away_Elo', 'Elo_Diff',
            'Home_Goals_L5', 'Away_Goals_L5',
            'Home_Conceded_L5', 'Away_Conceded_L5',
            'Home_Shots_L5', 'Away_Shots_L5',
            'Home_ShotsOnTarget_L5', 'Away_ShotsOnTarget_L5',
            'Home_Form_L5', 'Away_Form_L5',
            'Goals_Diff_L5', 'Form_Diff_L5'
        ]
        
        X_test = df_test[feature_cols]
        y_test = df_test['FTR']
        
        # Get predictions
        model_probs = model.predict_proba(X_test)
        model_pred = model.predict(X_test)
        
        # Visualization selector
        viz_type = st.selectbox(
            "Select Visualization",
            [
                "1. Feature Distributions (Univariate)",
                "2. Feature Correlations (Heatmap)",
                "3. Elo Rating Over Time",
                "4. Confusion Matrix",
                "5. ROC Curves (One-vs-Rest)",
                "6. Calibration Curves (Reliability Diagram)",
                "7. Prediction Confidence Distribution",
                "8. Feature Importance (Box Plots)"
            ]
        )
        
        if "1. Feature Distributions" in viz_type:
            st.subheader("Feature Distributions (Univariate Analysis)")
            st.markdown("*Module 4: EDA - Understanding data distributions*")
            
            # Select feature
            feature = st.selectbox("Select Feature", feature_cols)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            axes[0].hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Distribution of {feature}')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot by outcome
            df_plot = df[[feature, 'FTR']].dropna()
            df_plot.boxplot(column=feature, by='FTR', ax=axes[1])
            axes[1].set_xlabel('Match Outcome')
            axes[1].set_ylabel(feature)
            axes[1].set_title(f'{feature} by Outcome')
            plt.suptitle('')  # Remove default title
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            st.markdown("**Descriptive Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df[feature].mean():.2f}")
            col2.metric("Median", f"{df[feature].median():.2f}")
            col3.metric("Std Dev", f"{df[feature].std():.2f}")
            col4.metric("Range", f"{df[feature].max() - df[feature].min():.2f}")
        
        elif "2. Feature Correlations" in viz_type:
            st.subheader("Feature Correlation Heatmap")
            st.markdown("*Module 4: EDA - Identifying relationships between features*")
            
            # Correlation matrix
            corr_matrix = df[feature_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Key correlations
            st.markdown("**Strongest Correlations:**")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(5)
            st.dataframe(corr_df, use_container_width=True)
        
        elif "3. Elo Rating Over Time" in viz_type:
            st.subheader("Elo Rating Evolution")
            st.markdown("*Module 3: Feature Engineering - Dynamic team strength over time*")
            
            # Select teams
            all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
            selected_teams = st.multiselect("Select Teams (max 5)", all_teams, default=all_teams[:3] if len(all_teams) >= 3 else all_teams)
            
            if selected_teams:
                fig, ax = plt.subplots(figsize=(14, 6))
                
                for team in selected_teams[:5]:  # Limit to 5 teams
                    # Get team's Elo over time
                    home_matches = df[df['HomeTeam'] == team][['Date', 'Home_Elo']].copy()
                    home_matches.columns = ['Date', 'Elo']
                    
                    away_matches = df[df['AwayTeam'] == team][['Date', 'Away_Elo']].copy()
                    away_matches.columns = ['Date', 'Elo']
                    
                    team_elo = pd.concat([home_matches, away_matches]).sort_values('Date')
                    
                    ax.plot(team_elo['Date'], team_elo['Elo'], label=team, linewidth=2, marker='o', markersize=3, alpha=0.7)
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Elo Rating', fontsize=12)
                ax.set_title('Team Strength Evolution (Elo Ratings)', fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Please select at least one team")
        
        elif "4. Confusion Matrix" in viz_type:
            st.subheader("Confusion Matrix")
            st.markdown("*Module 7: Classification Evaluation - Understanding prediction errors*")
            
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            # Handle label encoding for XGBoost
            if model_choice == "XGBoost (Calibrated)":
                label_map_local = joblib.load('models/label_map.joblib')
                reverse_map = {v: k for k, v in label_map_local.items()}
                pred_labels = [reverse_map[p] for p in model_pred]
            else:
                pred_labels = model_pred
            
            cm = confusion_matrix(y_test, pred_labels, labels=['A', 'D', 'H'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(cm, display_labels=['Away Win', 'Draw', 'Home Win'])
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title(f'Confusion Matrix - {model_choice}', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Metrics
            st.markdown("**Per-Class Metrics:**")
            col1, col2, col3 = st.columns(3)
            
            for i, (outcome, col) in enumerate(zip(['Away Win', 'Draw', 'Home Win'], [col1, col2, col3])):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                with col:
                    st.markdown(f"**{outcome}**")
                    st.write(f"Precision: {precision:.2%}")
                    st.write(f"Recall: {recall:.2%}")
                    st.write(f"F1-Score: {f1:.2%}")
        
        elif "5. ROC Curves" in viz_type:
            st.subheader("ROC Curves (One-vs-Rest)")
            st.markdown("*Module 7: Classification Evaluation - Model discrimination ability*")
            
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            # Binarize labels
            y_test_bin = label_binarize(y_test, classes=['A', 'D', 'H'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            outcomes = ['Away Win', 'Draw', 'Home Win']
            
            for i, (color, outcome) in enumerate(zip(colors, outcomes)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{outcome} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curves - {model_choice}', fontsize=14)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation:**
            - AUC > 0.5: Better than random
            - AUC > 0.7: Good discrimination
            - Closer to top-left corner = better performance
            """)
        
        elif "6. Calibration Curves" in viz_type:
            st.subheader("Calibration Curves (Reliability Diagrams)")
            st.markdown("*Module 7: Calibration - Do probabilities match reality?*")
            
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            outcomes = ['Away Win', 'Draw', 'Home Win']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (ax, outcome, color) in enumerate(zip(axes, outcomes, colors)):
                y_binary = (y_test == ['A', 'D', 'H'][i]).astype(int)
                
                frac_pos, mean_pred = calibration_curve(y_binary, model_probs[:, i], n_bins=10)
                brier = brier_score_loss(y_binary, model_probs[:, i])
                
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                ax.plot(mean_pred, frac_pos, 's-', color=color, label=f'Model (Brier={brier:.3f})')
                
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'{outcome}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Calibration Curves - {model_choice}', fontsize=14, y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            **Good Calibration:**
            - Points close to diagonal line
            - Brier Score < 0.25 (our models: ~0.18 ✅)
            - Probabilities reflect true frequencies
            """)
        
        elif "7. Prediction Confidence" in viz_type:
            st.subheader("Prediction Confidence Distribution")
            st.markdown("*Understanding model certainty across predictions*")
            
            # Get max probability for each prediction
            max_probs = model_probs.max(axis=1)
            predicted_outcomes = model_probs.argmax(axis=1)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Overall confidence distribution
            axes[0, 0].hist(max_probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 0].axvline(max_probs.mean(), color='red', linestyle='--', 
                              label=f'Mean: {max_probs.mean():.3f}')
            axes[0, 0].set_xlabel('Maximum Probability')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Overall Prediction Confidence')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Confidence by outcome
            outcomes = ['Away Win', 'Draw', 'Home Win']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (outcome, color) in enumerate(zip(outcomes, colors)):
                mask = predicted_outcomes == i
                if mask.sum() > 0:
                    axes[0, 1].hist(max_probs[mask], bins=20, alpha=0.5, 
                                   label=outcome, color=color, edgecolor='black')
            
            axes[0, 1].set_xlabel('Maximum Probability')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Confidence by Predicted Outcome')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Confidence vs Correctness
            if model_choice == "XGBoost (Calibrated)":
                reverse_map = {v: k for k, v in label_map.items()}
                pred_labels = np.array([reverse_map[p] for p in model_pred])
            else:
                pred_labels = model_pred
            
            correct = pred_labels == y_test.values
            
            axes[1, 0].hist([max_probs[correct], max_probs[~correct]], 
                           bins=20, label=['Correct', 'Incorrect'], 
                           color=['green', 'red'], alpha=0.6, edgecolor='black')
            axes[1, 0].set_xlabel('Maximum Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Confidence: Correct vs Incorrect Predictions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Confidence ranges
            confidence_ranges = ['Low (0.33-0.45)', 'Medium (0.45-0.60)', 'High (0.60+)']
            range_counts = [
                ((max_probs >= 0.33) & (max_probs < 0.45)).sum(),
                ((max_probs >= 0.45) & (max_probs < 0.60)).sum(),
                (max_probs >= 0.60).sum()
            ]
            
            axes[1, 1].bar(confidence_ranges, range_counts, color='steelblue', edgecolor='black')
            axes[1, 1].set_ylabel('Number of Predictions')
            axes[1, 1].set_title('Predictions by Confidence Range')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Confidence", f"{max_probs.mean():.3f}")
            col2.metric("Median Confidence", f"{np.median(max_probs):.3f}")
            col3.metric("High Confidence %", f"{(max_probs >= 0.60).sum() / len(max_probs):.1%}")
            col4.metric("Accuracy", f"{correct.sum() / len(correct):.1%}")
        
        elif "8. Feature Importance" in viz_type:
            st.subheader("Feature Importance (Box Plots by Outcome)")
            st.markdown("*Module 3: Feature Engineering - Which features matter most?*")
            
            # Select feature
            feature = st.selectbox("Select Feature to Analyze", feature_cols, key='feat_importance')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Box plot by actual outcome
            df_plot = df[[feature, 'FTR']].dropna()
            df_plot.boxplot(column=feature, by='FTR', ax=axes[0])
            axes[0].set_xlabel('Actual Outcome')
            axes[0].set_ylabel(feature)
            axes[0].set_title(f'{feature} by Actual Outcome')
            axes[0].set_xticklabels(['Away', 'Draw', 'Home'])
            plt.sca(axes[0])
            plt.xticks(rotation=0)
            
            # Violin plot
            for i, outcome in enumerate(['A', 'D', 'H']):
                data = df_plot[df_plot['FTR'] == outcome][feature]
                parts = axes[1].violinplot([data], positions=[i], widths=0.7,
                                          showmeans=True, showmedians=True)
            
            axes[1].set_xticks([0, 1, 2])
            axes[1].set_xticklabels(['Away', 'Draw', 'Home'])
            axes[1].set_xlabel('Actual Outcome')
            axes[1].set_ylabel(feature)
            axes[1].set_title(f'{feature} Distribution by Outcome (Violin Plot)')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics by outcome
            st.markdown("**Statistics by Outcome:**")
            stats_df = df_plot.groupby('FTR')[feature].agg(['mean', 'median', 'std']).round(2)
            stats_df.index = ['Away Win', 'Draw', 'Home Win']
            st.dataframe(stats_df, use_container_width=True)
    
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### Sports Betting Edge System
        
        **Academic Capstone Project - Business Analytics**
        
        This system demonstrates:
        - CRISP-DM methodology (Cross-Industry Standard Process for Data Mining)
        - Feature engineering (Elo ratings, rolling statistics)
        - Probability calibration (Isotonic Regression)
        - Time-series validation (preventing data leakage)
        - Kelly Criterion risk management
        - Backtesting framework
        
        ### Models
        
        1. **Baseline (Logistic Regression)**
           - Naturally well-calibrated
           - Simple and interpretable
           - Better backtesting performance
        
        2. **XGBoost (Calibrated)**
           - Captures non-linear patterns
           - Isotonic calibration applied
           - Higher accuracy but overconfident
        
        ### Features (15 total)
        
        - **Elo Ratings**: Dynamic team strength (3 features)
        - **Rolling Statistics**: Recent performance (10 features)
        - **Form Points**: Win/draw/loss record (2 features)
        
        ### Data
        
        - **Source**: Football-Data.co.uk
        - **League**: English Premier League
        - **Seasons**: 2019-20 to 2023-24
        - **Matches**: 1,884 total (377 test set)
        
        ### Ethical Disclaimer
        
        **DO NOT USE FOR REAL GAMBLING**
        
        - This is an academic project
        - Models show negative ROI
        - High risk of losses
        - Past performance ≠ future results
        
        ### Contact
        
        For academic purposes only.
        """)


if __name__ == "__main__":
    main()

