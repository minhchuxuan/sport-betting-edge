"""
Model training pipeline
Baseline (Logistic Regression) + Advanced (XGBoost with calibration)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from xgboost import XGBClassifier


def prepare_features(df):
    """
    Prepare feature matrix and target
    
    Returns:
        X: Feature matrix
        y: Target (FTR)
        feature_names: List of feature names
    """
    # Target
    y = df['FTR'].copy()
    
    # Features: Elo + Rolling + Form + Differences
    feature_cols = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'Home_Goals_L5', 'Away_Goals_L5',
        'Home_Conceded_L5', 'Away_Conceded_L5',
        'Home_Shots_L5', 'Away_Shots_L5',
        'Home_ShotsOnTarget_L5', 'Away_ShotsOnTarget_L5',
        'Home_Form_L5', 'Away_Form_L5',
        'Goals_Diff_L5', 'Form_Diff_L5'
    ]
    
    X = df[feature_cols].copy()
    
    return X, y, feature_cols


def train_baseline(X_train, y_train):
    """
    Train baseline Logistic Regression
    Naturally well-calibrated
    """
    print("Training baseline (Logistic Regression)...")
    
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  [OK] Baseline trained")
    
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier
    Better discrimination but needs calibration
    """
    print("Training XGBoost...")
    
    # Encode labels (XGBoost needs 0,1,2)
    label_map = {'A': 0, 'D': 1, 'H': 2}
    y_encoded = y_train.map(label_map)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_encoded)
    print("  [OK] XGBoost trained")
    
    return model, label_map


def calibrate_model(model, X_train, y_train, label_map=None):
    """
    Apply Isotonic calibration
    """
    print("Applying calibration (Isotonic Regression)...")
    
    # Encode if needed
    if label_map:
        y_encoded = y_train.map(label_map)
    else:
        y_encoded = y_train
    
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv=3
    )
    
    calibrated.fit(X_train, y_encoded)
    print("  [OK] Calibration complete")
    
    return calibrated


def evaluate_model(model, X, y, label_map=None, model_name="Model"):
    """
    Evaluate model performance
    """
    # Encode if needed
    if label_map:
        y_encoded = y.map(label_map)
        y_eval = y_encoded
    else:
        y_eval = y
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Metrics
    acc = accuracy_score(y_eval, y_pred)
    logloss = log_loss(y_eval, y_proba)
    
    # Brier score per class
    brier_scores = {}
    classes = ['A', 'D', 'H'] if not label_map else [0, 1, 2]
    for i, cls in enumerate(classes):
        y_binary = (y_eval == (i if label_map else cls)).astype(int)
        brier_scores[cls] = brier_score_loss(y_binary, y_proba[:, i])
    
    brier_avg = np.mean(list(brier_scores.values()))
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Log Loss:  {logloss:.3f}")
    print(f"  Brier (avg): {brier_avg:.3f}")
    print(f"  Brier (H): {brier_scores[classes[2]]:.3f}")
    print(f"  Brier (D): {brier_scores[classes[1]]:.3f}")
    print(f"  Brier (A): {brier_scores[classes[0]]:.3f}")
    
    return {
        'accuracy': acc,
        'log_loss': logloss,
        'brier_avg': brier_avg,
        'brier_scores': brier_scores
    }


def save_model(model, filename, models_dir='models'):
    """Save model to disk"""
    Path(models_dir).mkdir(exist_ok=True)
    filepath = Path(models_dir) / filename
    joblib.dump(model, filepath)
    print(f"  [OK] Saved to {filepath}")


def run_pipeline():
    """
    Complete training pipeline with time-series split
    """
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet('data/processed/features.parquet')
    print(f"   Loaded {len(df)} matches")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, feature_names = prepare_features(df)
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)}")
    
    # Time-series split (80/20)
    print("\n3. Splitting data (time-aware)...")
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)} matches")
    print(f"   Test:  {len(X_test)} matches")
    
    # Train baseline
    print("\n4. Training baseline model...")
    baseline = train_baseline(X_train, y_train)
    save_model(baseline, 'logistic_baseline.joblib')
    
    # Evaluate baseline
    baseline_metrics = evaluate_model(baseline, X_test, y_test, model_name="Baseline (Logistic)")
    
    # Train XGBoost
    print("\n5. Training XGBoost...")
    xgb, label_map = train_xgboost(X_train, y_train)
    
    # Evaluate XGBoost (uncalibrated)
    xgb_metrics = evaluate_model(xgb, X_test, y_test, label_map, model_name="XGBoost (Uncalibrated)")
    
    # Calibrate XGBoost
    print("\n6. Calibrating XGBoost...")
    xgb_calibrated = calibrate_model(xgb, X_train, y_train, label_map)
    save_model(xgb_calibrated, 'xgb_calibrated.joblib')
    save_model(label_map, 'label_map.joblib')
    
    # Evaluate calibrated
    calibrated_metrics = evaluate_model(xgb_calibrated, X_test, y_test, label_map, model_name="XGBoost (Calibrated)")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModel Comparison:")
    print(f"{'Model':<25} {'Accuracy':<10} {'Log Loss':<10} {'Brier':<10}")
    print("-"*60)
    print(f"{'Baseline (Logistic)':<25} {baseline_metrics['accuracy']:<10.3f} {baseline_metrics['log_loss']:<10.3f} {baseline_metrics['brier_avg']:<10.3f}")
    print(f"{'XGBoost (Uncalibrated)':<25} {xgb_metrics['accuracy']:<10.3f} {xgb_metrics['log_loss']:<10.3f} {xgb_metrics['brier_avg']:<10.3f}")
    print(f"{'XGBoost (Calibrated)':<25} {calibrated_metrics['accuracy']:<10.3f} {calibrated_metrics['log_loss']:<10.3f} {calibrated_metrics['brier_avg']:<10.3f}")
    
    return {
        'baseline': baseline_metrics,
        'xgb': xgb_metrics,
        'xgb_calibrated': calibrated_metrics
    }


if __name__ == "__main__":
    metrics = run_pipeline()

