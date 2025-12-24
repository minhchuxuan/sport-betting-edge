"""
Sports Betting Edge Dashboard V2
Enhanced UI with interactive Plotly visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import calculate_ev, calculate_kelly_stake, calculate_implied_prob

# Page config
st.set_page_config(
    page_title="Sports Betting Edge Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe4cc 100%);
        border: 2px solid #ff9800;
        border-left: 6px solid #ff6b00;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.15);
    }
    
    .warning-box h3 {
        color: #e65100;
        margin-top: 0;
        font-weight: 700;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tabs */
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
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 2px solid #e0e7ff;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #1e293b;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Buttons */
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
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    h2 {
        font-size: 1.875rem;
        border-bottom: 3px solid #e0e7ff;
        padding-bottom: 0.5rem;
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
    st.markdown('<div class="main-header">⚽ Sports Betting Edge Dashboard</div>', unsafe_allow_html=True)
    
    # Warning banner
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ IMPORTANT DISCLAIMER</h3>
        <p><strong>This is an analytical tool only. DO NOT use for real gambling.</strong></p>
        <p>Historical backtesting shows <strong>negative ROI (-64% for XGBoost, -41% for Baseline)</strong>. 
        Models are well-calibrated but cannot beat bookmaker odds consistently.</p>
        <p>This dashboard demonstrates machine learning concepts, not profitable betting strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    baseline, xgb, label_map = load_models()
    df = load_data()
    
    if baseline is None or xgb is None or df is None:
        st.error("Failed to load models or data. Please check file paths.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.info("""
    **Quick Guide**
    
    Adjust your betting strategy parameters:
    - **Model**: Choose prediction algorithm
    - **Kelly Fraction**: Control bet sizing
    - **EV Threshold**: Filter low-value opportunities
    
    Note: Model probabilities are fixed from training.
    """)
    
    st.sidebar.markdown("---")
    
    model_choice = st.sidebar.selectbox(
        "Model Selection",
        ["XGBoost (Calibrated)", "Baseline (Logistic)"],
        help="Choose which trained model to use"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Bankroll Management")
    
    bankroll = st.sidebar.number_input(
        "Starting Bankroll ($)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Total capital available"
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more conservative (0.25 recommended)"
    )
    
    st.sidebar.caption(f"📊 Current: **{kelly_fraction:.0%}** Kelly")
    
    ev_threshold = st.sidebar.slider(
        "Min EV Threshold (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Minimum expected value to show bet"
    ) / 100
    
    st.sidebar.caption(f"🎯 Filtering bets below **{ev_threshold:.1%}** EV")
    
    # Select model
    model = xgb if model_choice == "XGBoost (Calibrated)" else baseline
    
    # Feature columns
    feature_cols = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'Home_Goals_L5', 'Away_Goals_L5',
        'Home_Conceded_L5', 'Away_Conceded_L5',
        'Home_Shots_L5', 'Away_Shots_L5',
        'Home_ShotsOnTarget_L5', 'Away_ShotsOnTarget_L5',
        'Home_Form_L5', 'Away_Form_L5',
        'Goals_Diff_L5', 'Form_Diff_L5'
    ]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Match Predictions", 
        "📊 Performance Analysis", 
        "📈 Visualizations", 
        "ℹ️ About"
    ])
    
    with tab1:
        render_predictions_tab(df, model, feature_cols, bankroll, kelly_fraction, ev_threshold, model_choice, label_map)
    
    with tab2:
        render_performance_tab(df, model, feature_cols, bankroll, kelly_fraction, ev_threshold, model_choice)
    
    with tab3:
        render_visualizations_tab(df, model, feature_cols, model_choice, label_map)
    
    with tab4:
        render_about_tab()


def render_predictions_tab(df, model, feature_cols, bankroll, kelly_fraction, ev_threshold, model_choice, label_map):
    """Render match predictions tab"""
    st.header("Match Predictions")
    
    st.info("""
    **Match Result Codes**: 
    - **H** = Home Win    - **A** = Away Win    - **D** = Draw
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
    features = match[feature_cols].values
    
    # Get predictions
    probs = predict_match(model, features)
    
    # Display match info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"🏠 {match['HomeTeam']}")
        st.metric("Elo Rating", f"{match['Home_Elo']:.0f}")
        st.metric("Recent Form", f"{match['Home_Form_L5']:.1f} pts")
    
    with col2:
        st.subheader("⚽ Match Info")
        st.write(f"**Date:** {match['Date']}")
        st.write(f"**Actual Result:** {match['FTR']}")
        st.write(f"**Score:** {match['FTHG']:.0f} - {match['FTAG']:.0f}")
    
    with col3:
        st.subheader(f"✈️ {match['AwayTeam']}")
        st.metric("Elo Rating", f"{match['Away_Elo']:.0f}")
        st.metric("Recent Form", f"{match['Away_Form_L5']:.1f} pts")
    
    st.markdown("---")
    
    # Predictions
    st.subheader("🎲 Model Predictions")
    
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
    
    # Interactive probability comparison chart
    st.markdown("---")
    st.subheader("📊 Probability Comparison")
    
    implied_probs = [calculate_implied_prob(o) for o in odds]
    
    #  Create interactive Plotly chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Model',
        x=outcomes,
        y=probs,
        marker_color='#4F46E5',
        text=[f'{p:.1%}' for p in probs],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Bookmaker',
        x=outcomes,
        y=implied_probs,
        marker_color='#F59E0B',
        text=[f'{p:.1%}' for p in implied_probs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Model vs Bookmaker Probabilities',
        xaxis_title='Outcome',
        yaxis_title='Probability',
        yaxis_tickformat='.0%',
        barmode='group',
        height=400,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_tab(df, model, feature_cols, bankroll, kelly_fraction, ev_threshold, model_choice):
    """Render performance analysis tab"""
    st.header("Performance Analysis & Backtesting")
    
    # Get test data
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    
    # Ensure Date column is datetime
    if df_test['Date'].dtype != 'datetime64[ns]':
        df_test['Date'] = pd.to_datetime(df_test['Date'])
    
    # Date range selection
    st.subheader("📅 Time Period Selection")
    st.info("""
    Select a custom date range to analyze model performance over specific time periods.
    Explore performance variance, seasonal effects, and sample size impact.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_date = df_test['Date'].min().date()
        max_date = df_test['Date'].max().date()
        
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Filter data
    df_filter = df_test[
        (df_test['Date'].dt.date >= start_date) & 
        (df_test['Date'].dt.date <= end_date)
    ].reset_index(drop=True)
    
    # Show period info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Matches", len(df_filter))
    with col2:
        st.metric("📅 Days Span", (end_date - start_date).days)
    with col3:
        period_pct = (len(df_filter) / len(df_test)) * 100 if len(df_test) > 0 else 0
        st.metric("📈 % of Test Set", f"{period_pct:.1f}%")
    
    if len(df_filter) == 0:
        st.error("No matches in selected date range. Please adjust dates.")
        return
    
    st.markdown("---")
    
    # Run backtest
    st.subheader("🎯 Backtesting Results")
    st.caption(f"*Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}*")
    
    X_filter = df_filter[feature_cols]
    model_probs = model.predict_proba(X_filter)
    
    # Import backtester
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from evaluation.backtest import BettingBacktester
    
    backtester = BettingBacktester(
        initial_bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        ev_threshold=ev_threshold
    )
    
    results = backtester.run(df_filter, model_probs)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Bets", results['total_bets'])
    with col2:
        roi_color = "normal" if results['roi'] >= 0 else "inverse"
        st.metric("ROI", f"{results['roi']*100:.2f}%", 
                 delta=f"{results['roi']*100:.2f}%", 
                 delta_color=roi_color)
    with col3:
        st.metric("Win Rate", f"{results['win_rate']*100:.1f}%")
    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']*100:.1f}%", 
                 delta=f"{results['max_drawdown']*100:.1f}%",
                 delta_color="inverse")
    
    # Profit metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Initial Bankroll", f"${results['initial_bankroll']:.2f}")
    with col2:
        st.metric("Final Bankroll", f"${results['final_bankroll']:.2f}")
    with col3:
        profit_color = "normal" if results['total_profit'] >= 0 else "inverse"
        st.metric("Total Profit", f"${results['total_profit']:.2f}",
                 delta=f"${results['total_profit']:.2f}",
                 delta_color=profit_color)
    
    # Interpretation
    if results['roi'] > 0:
        st.success(f"✅ **Profitable Period!** Model achieved {results['roi']*100:.2f}% ROI.")
    elif results['roi'] > -0.10:
        st.warning(f"⚠️ **Slight Loss**: {abs(results['roi']*100):.2f}% loss. Sample variance may be a factor.")
    else:
        st.error(f"❌ **Significant Loss**: {abs(results['roi']*100):.2f}% loss over this period.")
    
    st.markdown("---")
    
    # Interactive bankroll evolution chart
    st.subheader("💰 Bankroll Evolution")
    
    history_df = results['history']
    bet_history = history_df[history_df['bet_placed']].reset_index(drop=True)
    
    if len(bet_history) > 0:
        # Create Plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(bet_history))),
            y=bet_history['bankroll'],
            mode='lines',
            name='Bankroll',
            line=dict(color='#4F46E5', width=3),
            fill='tonexty',
            fillcolor='rgba(79, 70, 229, 0.1)'
        ))
        
        # Add starting bankroll line
        fig.add_hline(
            y=bankroll, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Starting Bankroll",
            annotation_position="right"
        )
        
        # Add peak annotation
        peak_idx = bet_history['bankroll'].argmax()
        peak_val = bet_history['bankroll'].max()
        
        fig.add_annotation(
            x=peak_idx,
            y=peak_val,
            text=f"Peak: ${peak_val:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            ax=30,
            ay=-40
        )
        
        # Add final annotation
        final_val = bet_history['bankroll'].iloc[-1]
        fig.add_annotation(
            x=len(bet_history)-1,
            y=final_val,
            text=f"Final: ${final_val:.2f}<br>ROI: {results['roi']*100:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red" if results['roi'] < 0 else "green",
            ax=-40,
            ay=-40
        )
        
        # Get first and last bet dates
        first_bet_date = df_filter.iloc[0]['Date'] if isinstance(df_filter.iloc[0]['Date'], str) else df_filter.iloc[0]['Date'].strftime('%Y-%m-%d')
        last_bet_date = df_filter.iloc[-1]['Date'] if isinstance(df_filter.iloc[-1]['Date'], str) else df_filter.iloc[-1]['Date'].strftime('%Y-%m-%d')
        
        fig.update_layout(
            title=f"Bankroll Evolution<br><sub>Period: {first_bet_date} to {last_bet_date}</sub>",
            xaxis_title="Bet Number",
            yaxis_title="Bankroll ($)",
            yaxis=dict(
                dtick=200,  # Set tick interval to $200
                tickformat='$,.0f'  # Format as currency
            ),
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"*Bankroll evolution over {len(bet_history)} bets with {kelly_fraction:.0%} Kelly staking*")
    else:
        st.info("No bets placed. Try lowering the EV threshold or selecting a different date range.")


def render_visualizations_tab(df, model, feature_cols, model_choice, label_map):
    """Render visualizations tab with interactive Plotly charts"""
    st.header("Data Visualizations")
    
    # Get test data
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:]
    
    X_test = df_test[feature_cols]
    y_test = df_test['FTR']
    
    # Get predictions
    model_probs = model.predict_proba(X_test)
    model_pred = model.predict(X_test)
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization",
        [
            "Feature Distributions",
            "Feature Correlations",
            "Elo Rating Over Time",
            "Confusion Matrix",
            "ROC Curves",
            "Calibration Curves",
            "Prediction Confidence"
        ]
    )
    
    if "Feature Distributions" in viz_type:
        render_feature_distributions(df, feature_cols)
    elif "Feature Correlations" in viz_type:
        render_feature_correlations(df, feature_cols)
    elif "Elo Rating" in viz_type:
        render_elo_evolution(df)
    elif "Confusion Matrix" in viz_type:
        render_confusion_matrix(y_test, model_pred, model_choice, label_map)
    elif "ROC Curves" in viz_type:
        render_roc_curves(y_test, model_probs, model_choice)
    elif "Calibration" in viz_type:
        render_calibration_curves(y_test, model_probs, model_choice)
    elif "Prediction Confidence" in viz_type:
        render_prediction_confidence(model_probs, model_pred, y_test, model_choice, label_map)


def render_feature_distributions(df, feature_cols):
    """Render feature distribution plots"""
    st.subheader("Feature Distributions")
    
    feature = st.selectbox("Select Feature", feature_cols)
    
    # Create subplot
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution", "By Outcome"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df[feature].dropna(),
            name='Distribution',
            marker_color='#4F46E5',
            nbinsx=30
        ),
        row=1, col=1
    )
    
    # Box plot by outcome
    for outcome, color in zip(['A', 'D', 'H'], ['#EF4444', '#F59E0B', '#22C55E']):
        outcome_name = {'A': 'Away', 'D': 'Draw', 'H': 'Home'}[outcome]
        fig.add_trace(
            go.Box(
                y=df[df['FTR'] == outcome][feature].dropna(),
                name=outcome_name,
                marker_color=color
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=f"{feature} Analysis",
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("**Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{df[feature].mean():.2f}")
    col2.metric("Median", f"{df[feature].median():.2f}")
    col3.metric("Std Dev", f"{df[feature].std():.2f}")
    col4.metric("Range", f"{df[feature].max() - df[feature].min():.2f}")


def render_feature_correlations(df, feature_cols):
    """Render correlation heatmap"""
    st.subheader("Feature Correlations")
    
    corr_matrix = df[feature_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_elo_evolution(df):
    """Render Elo rating evolution"""
    st.subheader("Elo Rating Evolution")
    
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    selected_teams = st.multiselect(
        "Select Teams (max 5)", 
        all_teams, 
        default=all_teams[:3] if len(all_teams) >= 3 else all_teams
    )
    
    if selected_teams:
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, team in enumerate(selected_teams[:5]):
            # Get team's Elo over time
            home_matches = df[df['HomeTeam'] == team][['Date', 'Home_Elo']].copy()
            home_matches.columns = ['Date', 'Elo']
            
            away_matches = df[df['AwayTeam'] == team][['Date', 'Away_Elo']].copy()
            away_matches.columns = ['Date', 'Elo']
            
            team_elo = pd.concat([home_matches, away_matches]).sort_values('Date')
            
            fig.add_trace(go.Scatter(
                x=team_elo['Date'],
                y=team_elo['Elo'],
                mode='lines+markers',
                name=team,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Team Strength Evolution (Elo Ratings)",
            xaxis_title="Date",
            yaxis_title="Elo Rating",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one team")


def render_confusion_matrix(y_test, model_pred, model_choice, label_map):
    """Render confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    st.subheader("Confusion Matrix")
    
    # Handle label encoding
    if model_choice == "XGBoost (Calibrated)":
        reverse_map = {v: k for k, v in label_map.items()}
        pred_labels = [reverse_map[p] for p in model_pred]
    else:
        pred_labels = model_pred
    
    cm = confusion_matrix(y_test, pred_labels, labels=['A', 'D', 'H'])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Away Win', 'Draw', 'Home Win'],
        y=['Away Win', 'Draw', 'Home Win'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_choice}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_roc_curves(y_test, model_probs, model_choice):
    """Render ROC curves"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    st.subheader("ROC Curves (One-vs-Rest)")
    
    y_test_bin = label_binarize(y_test, classes=['A', 'D', 'H'])
    
    fig = go.Figure()
    
    outcomes = ['Away Win', 'Draw', 'Home Win']
    colors = ['#EF4444', '#F59E0B', '#22C55E']
    
    for i, (outcome, color) in enumerate(zip(outcomes, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{outcome}',
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{outcome}</b><br>' +
                          'FPR: %{x:.3f}<br>' +
                          'TPR: %{y:.3f}<br>' +
                          f'AUC: {roc_auc:.3f}<br>' +
                          '<extra></extra>'
        ))
    
    # Add random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='black', dash='dash')
    ))
    
    # Calculate AUC scores first for title
    auc_scores = []
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
    
    fig.update_layout(
        title=f"ROC Curves - {model_choice}<br><sub>AUC: Away={auc_scores[0]:.3f}, Draw={auc_scores[1]:.3f}, Home={auc_scores[2]:.3f}</sub>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_calibration_curves(y_test, model_probs, model_choice):
    """Render calibration curves"""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    from plotly.subplots import make_subplots
    
    st.subheader("Calibration Curves")
    
    # Calculate Brier scores first
    brier_scores = []
    for i in range(3):
        y_binary = (y_test == ['A', 'D', 'H'][i]).astype(int)
        brier = brier_score_loss(y_binary, model_probs[:, i])
        brier_scores.append(brier)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"Away Win (Brier: {brier_scores[0]:.3f})", 
            f"Draw (Brier: {brier_scores[1]:.3f})", 
            f"Home Win (Brier: {brier_scores[2]:.3f})"
        )
    )
    
    for i, col_num in enumerate([1, 2, 3]):
        y_binary = (y_test == ['A', 'D', 'H'][i]).astype(int)
        
        frac_pos, mean_pred = calibration_curve(y_binary, model_probs[:, i], n_bins=10)
        brier = brier_score_loss(y_binary, model_probs[:, i])
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect',
                line=dict(color='black', dash='dash'),
                showlegend=(i == 0)
            ),
            row=1, col=col_num
        )
        
        # Model calibration
        fig.add_trace(
            go.Scatter(
                x=mean_pred,
                y=frac_pos,
                mode='lines+markers',
                name=f'Model',
                line=dict(width=2),
                marker=dict(size=8),
                showlegend=(i == 0),
                hovertemplate='<b>Predicted: %{x:.3f}</b><br>' +
                              'Actual: %{y:.3f}<br>' +
                              f'Brier Score: {brier:.3f}<br>' +
                              '<extra></extra>'
            ),
            row=1, col=col_num
        )
    
    fig.update_xaxes(title_text="Mean Predicted Probability")
    fig.update_yaxes(title_text="Fraction of Positives")
    
    fig.update_layout(
        title_text=f"Calibration Curves - {model_choice}",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Good Calibration:** Points close to diagonal line (Brier Score < 0.25)
    
    Our models achieve ~0.18 Brier score, indicating excellent calibration.
    """)


def render_prediction_confidence(model_probs, model_pred, y_test, model_choice, label_map):
    """Render prediction confidence analysis"""
    st.subheader("Prediction Confidence Distribution")
    
    max_probs = model_probs.max(axis=1)
    
    # Handle labels
    if model_choice == "XGBoost (Calibrated)":
        reverse_map = {v: k for k, v in label_map.items()}
        pred_labels = np.array([reverse_map[p] for p in model_pred])
    else:
        pred_labels = model_pred
    
    correct = pred_labels == y_test.values
    
    # Create subplot
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Overall Confidence", "Correct vs Incorrect")
    )
    
    # Overall distribution
    fig.add_trace(
        go.Histogram(
            x=max_probs,
            nbinsx=30,
            name='All Predictions',
            marker_color='#4F46E5'
        ),
        row=1, col=1
    )
    
    # Correct vs incorrect
    fig.add_trace(
        go.Histogram(
            x=max_probs[correct],
            nbinsx=20,
            name='Correct',
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(
            x=max_probs[~correct],
            nbinsx=20,
            name='Incorrect',
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Maximum Probability")
    fig.update_yaxes(title_text="Frequency")
    
    fig.update_layout(
        title_text="Prediction Confidence Analysis",
        height=400,
        template='plotly_white',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Confidence", f"{max_probs.mean():.3f}")
    col2.metric("Median Confidence", f"{np.median(max_probs):.3f}")
    col3.metric("High Confidence %", f"{(max_probs >= 0.60).sum() / len(max_probs):.1%}")
    col4.metric("Accuracy", f"{correct.sum() / len(correct):.1%}")


def render_about_tab():
    """Render about tab"""
    st.header("About This System")
    
    st.markdown("""
    ### Sports Betting Edge System
    
    A machine learning system for football match outcome prediction and value betting analysis.
    
    ---
    
    ### Models
    
    **1. Baseline (Logistic Regression)**
    - Naturally well-calibrated probabilities
    - Simple and interpretable
    - Less aggressive betting behavior
    
    **2. XGBoost (Calibrated)**
    - Captures non-linear patterns
    - Isotonic calibration applied
    - Higher accuracy but requires careful tuning
    
    ---
    
    ### Features (15 total)
    
    - **Elo Ratings** (3): Dynamic team strength metrics
    - **Rolling Statistics** (10): Recent performance indicators
    - **Form Points** (2): Win/draw/loss records
    
    All features use only historical data to prevent look-ahead bias.
    
    ---
    
    ### Data
    
    - **Source**: Football-Data.co.uk
    - **League**: English Premier League
    - **Seasons**: 2019-20 to 2023-24
    - **Total Matches**: 1,884 (377 in test set)
    
    ---
    
    ### Methodology
    
    1. **Feature Engineering**: Elo ratings + rolling statistics
    2. **Time-Series Validation**: Chronological train/test split
    3. **Probability Calibration**: Isotonic regression
    4. **Risk Management**: Kelly Criterion bet sizing
    5. **Backtesting**: Historical performance simulation
    
    ---
    
    ### Disclaimer
    
    **🚨 DO NOT USE FOR REAL GAMBLING**
    
    - Models show negative ROI in backtesting
    - Cannot beat professional bookmakers consistently
    - High risk of financial loss
    - For educational purposes only
    
    ---
    
    ### Performance Summary
    
    | Model | Accuracy | Brier Score | Test ROI |
    |-------|----------|-------------|----------|
    | Baseline | 57.6% | 0.183 | -41.1% |
    | XGBoost | 59.7% | 0.184 | -63.6% |
    
    Despite good calibration (Brier <  0.20), models cannot overcome bookmaker margins and information advantages.
    """)


if __name__ == "__main__":
    main()
