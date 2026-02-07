#!/usr/bin/env python3
"""
==============================================================================
Shift Planning Decision Assistant - Forecasting Engine
==============================================================================

This module handles:
1. Loading cleaned hourly demand data
2. Training a Random Forest regression model for demand forecasting
3. Proper time-based train/test splitting (NOT random)
4. Computing regression metrics (MAE, RMSE, R²)
5. Generating professional evaluation visualizations
6. Predicting future demand and calculating staffing requirements

SETUP:
    pip install pandas numpy matplotlib scikit-learn

RUN:
    python shift_planner.py

OUTPUT:
    - shift_planning_output/weekly_schedule.csv
    - shift_planning_output/daily_summary.csv
    - shift_planning_output/model_evaluation_report.txt
    - shift_planning_output/predicted_vs_actual.png
    - shift_planning_output/residuals_vs_actual.png
    - shift_planning_output/residuals_histogram.png
    - shift_planning_output/feature_importance.png
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import json

# ==============================================================================
# INSTALL CHECK
# ==============================================================================
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    print("=" * 60)
    print("ERROR: Missing packages!")
    print("Run: pip install pandas numpy matplotlib scikit-learn")
    print(f"\nMissing: {e}")
    print("=" * 60)
    exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Directories
CLEANED_DATA_DIR = Path("./cleaned_data")
OUTPUT_DIR = Path("./shift_planning_output")

# Model parameters
TEST_SIZE = 0.20  # Use last 20% of data for testing (time-based)
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Staffing parameters (can be adjusted based on restaurant capacity)
ORDERS_PER_STAFF_PER_HOUR = 8   # How many orders one staff member can handle
MIN_STAFF_PER_SHIFT = 2         # Minimum staff always on duty
MAX_STAFF_PER_SHIFT = 15        # Maximum staff capacity

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---")


def get_shift_name(hour):
    """Convert hour to shift name."""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_hourly_demand():
    """
    Load the cleaned hourly demand data.
    
    Returns:
        pd.DataFrame: Hourly demand data with columns:
            - place_id: Restaurant identifier
            - date: Date of the observation
            - hour: Hour of the day (0-23)
            - weekday: Day of week (0=Monday, 6=Sunday)
            - orders: Number of orders (TARGET VARIABLE)
    """
    print_subheader("Loading Data")
    
    hourly_path = CLEANED_DATA_DIR / "hourly_demand.csv"
    
    if not hourly_path.exists():
        print(f"❌ ERROR: {hourly_path} not found!")
        print("   Please run dataloader.py first.")
        return None
    
    df = pd.read_csv(hourly_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df):,} hourly demand records")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Unique places: {df['place_id'].nunique()}")
    print(f"   Total orders: {df['orders'].sum():,}")
    
    return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def create_features(df):
    """
    Create features for the forecasting model.
    
    These features capture temporal patterns in restaurant demand:
    - Time-based: hour, day of week, month, week of year
    - Business patterns: weekend flag, lunch/dinner rush indicators
    
    Args:
        df: DataFrame with date and hour columns
        
    Returns:
        DataFrame with additional feature columns
    """
    print_subheader("Creating Features")
    
    df = df.copy()
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek      # 0=Monday, 6=Sunday
    df['day_of_month'] = df['date'].dt.day           # 1-31
    df['month'] = df['date'].dt.month                # 1-12
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)  # 1-52
    
    # Business pattern features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Sat/Sun = 1
    df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)   # 11am-2pm
    df['is_dinner'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)  # 6pm-9pm
    
    feature_cols = [
        'place_id',      # Restaurant identifier (different places have different patterns)
        'hour',          # Hour of day (captures daily demand curve)
        'day_of_week',   # Day of week (weekends vs weekdays)
        'day_of_month',  # Day of month (potential monthly patterns)
        'month',         # Month (seasonal patterns)
        'week_of_year',  # Week (finer seasonal granularity)
        'is_weekend',    # Weekend flag (typically higher demand)
        'is_lunch',      # Lunch rush indicator
        'is_dinner'      # Dinner rush indicator
    ]
    
    print(f"✅ Created {len(feature_cols)} features:")
    for f in feature_cols:
        print(f"   - {f}")
    
    return df, feature_cols


# ==============================================================================
# TIME-BASED TRAIN/TEST SPLIT
# ==============================================================================

def time_based_split(df, test_size=0.20):
    """
    Split data chronologically - train on past, test on recent data.
    
    This is CRITICAL for time series forecasting:
    - We cannot use random splitting because it would leak future information
    - The model must be trained only on past data to simulate real-world usage
    - We test on the most recent data to evaluate real predictive performance
    
    Args:
        df: DataFrame sorted by date
        test_size: Fraction of data to use for testing (default 20%)
        
    Returns:
        train_df, test_df: Split DataFrames
    """
    print_subheader("Time-Based Train/Test Split")
    
    # Sort by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Find the split point
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx]['date']
    
    # Split the data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"✅ Split data chronologically:")
    print(f"   Training set: {len(train_df):,} records ({100*(1-test_size):.0f}%)")
    print(f"      Date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"   Test set: {len(test_df):,} records ({100*test_size:.0f}%)")
    print(f"      Date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"\n   ⚠️  Note: Time-based split ensures no future data leakage")
    
    return train_df, test_df


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_model(train_df, feature_cols):
    """
    Train a Random Forest regression model.
    
    Random Forest is chosen because:
    - Handles non-linear relationships well
    - Robust to outliers
    - Provides feature importance for interpretability
    - Works well with mixed feature types (numeric, categorical)
    
    Args:
        train_df: Training DataFrame
        feature_cols: List of feature column names
        
    Returns:
        Trained model
    """
    print_subheader("Training Model")
    
    X_train = train_df[feature_cols]
    y_train = train_df['orders']
    
    # Initialize Random Forest with reasonable hyperparameters
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,  # Number of trees in the forest
        max_depth=MAX_DEPTH,        # Maximum depth of each tree (prevents overfitting)
        random_state=RANDOM_STATE,  # For reproducibility
        n_jobs=-1                   # Use all CPU cores for parallel training
    )
    
    print(f"   Training Random Forest with {N_ESTIMATORS} trees...")
    model.fit(X_train, y_train)
    
    print(f"✅ Model trained successfully")
    print(f"   Algorithm: Random Forest Regressor")
    print(f"   Trees: {N_ESTIMATORS}")
    print(f"   Max depth: {MAX_DEPTH}")
    print(f"   Training samples: {len(X_train):,}")
    
    return model


# ==============================================================================
# MODEL EVALUATION (REGRESSION METRICS)
# ==============================================================================

def evaluate_model(model, test_df, feature_cols):
    """
    Evaluate model performance using regression metrics.
    
    For REGRESSION tasks (predicting numeric values), we use:
    - MAE (Mean Absolute Error): Average prediction error in original units
    - RMSE (Root Mean Squared Error): Penalizes large errors more heavily
    - R² (Coefficient of Determination): Proportion of variance explained
    
    NOTE: We do NOT use classification metrics (precision, recall, F1, accuracy)
    because this is a REGRESSION problem, not classification.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        feature_cols: Feature column names
        
    Returns:
        Dictionary of metrics and predictions
    """
    print_subheader("Model Evaluation (Regression Metrics)")
    
    X_test = test_df[feature_cols]
    y_test = test_df['orders']
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are non-negative (can't have negative orders)
    y_pred = np.clip(y_pred, 0, None)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional statistics
    residuals = y_test - y_pred
    mean_actual = y_test.mean()
    mean_predicted = y_pred.mean()
    
    # Print results
    print(f"\n📊 REGRESSION METRICS (on held-out test set):")
    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  MAE  (Mean Absolute Error):    {mae:>8.2f}   │")
    print(f"   │  RMSE (Root Mean Squared Error): {rmse:>8.2f}   │")
    print(f"   │  R²   (Coefficient of Determination): {r2:>5.3f} │")
    print(f"   └─────────────────────────────────────────────┘")
    
    print(f"\n📈 INTERPRETATION:")
    print(f"   • MAE = {mae:.2f} means predictions are off by ~{mae:.1f} orders on average")
    print(f"   • RMSE = {rmse:.2f} penalizes large errors more heavily")
    print(f"   • R² = {r2:.3f} means the model explains {r2*100:.1f}% of demand variance")
    
    print(f"\n📉 PREDICTION STATISTICS:")
    print(f"   • Average actual orders:    {mean_actual:.2f}")
    print(f"   • Average predicted orders: {mean_predicted:.2f}")
    print(f"   • Residual mean:            {residuals.mean():.2f} (should be ~0)")
    print(f"   • Residual std:             {residuals.std():.2f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'residuals': residuals
    }


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_predicted_vs_actual(y_test, y_pred, output_dir):
    """
    Create a scatter plot of predicted vs actual values.
    
    A good model should have points clustered around the diagonal line.
    Points above the line = over-predictions
    Points below the line = under-predictions
    """
    print_subheader("Generating: Predicted vs Actual Plot")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.3, s=20, c='#3b82f6', edgecolors='none')
    
    # Perfect prediction line (diagonal)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Orders', fontsize=12)
    ax.set_ylabel('Predicted Orders', fontsize=12)
    ax.set_title('Predicted vs Actual Orders\n(Points on the diagonal line = perfect predictions)', fontsize=14)
    ax.legend(loc='upper left')
    
    # Add R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.annotate(f'R² = {r2:.3f}', xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Grid for readability
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Save
    filepath = output_dir / 'predicted_vs_actual.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filepath}")


def plot_residuals_vs_actual(y_test, residuals, output_dir):
    """
    Create a residuals vs actual values plot.
    
    This helps identify:
    - Heteroscedasticity: If residual spread changes with actual values
    - Bias: If residuals are systematically positive or negative
    - Good model: Residuals should be randomly scattered around 0
    """
    print_subheader("Generating: Residuals vs Actual Plot")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, residuals, alpha=0.3, s=20, c='#8b5cf6', edgecolors='none')
    
    # Zero line (perfect predictions have residual = 0)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
    
    # Labels and title
    ax.set_xlabel('Actual Orders', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Residuals vs Actual Orders\n(Random scatter around 0 = good model)', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add mean residual annotation
    mean_res = residuals.mean()
    ax.annotate(f'Mean Residual = {mean_res:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Save
    filepath = output_dir / 'residuals_vs_actual.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filepath}")


def plot_residuals_histogram(residuals, output_dir):
    """
    Create a histogram of residuals.
    
    A good model should have:
    - Residuals centered around 0 (unbiased)
    - Approximately normal distribution
    - No extreme outliers
    """
    print_subheader("Generating: Residuals Histogram")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Histogram
    n, bins, patches = ax.hist(residuals, bins=50, color='#10b981', edgecolor='white', alpha=0.7)
    
    # Vertical line at zero
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    # Vertical line at mean
    mean_res = residuals.mean()
    ax.axvline(x=mean_res, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean_res:.2f}')
    
    # Labels and title
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Errors (Residuals)\n(Centered around 0 = unbiased model)', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add statistics annotation
    stats_text = f'Mean: {mean_res:.2f}\nStd: {residuals.std():.2f}\nMin: {residuals.min():.2f}\nMax: {residuals.max():.2f}'
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Save
    filepath = output_dir / 'residuals_histogram.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filepath}")


def plot_feature_importance(model, feature_cols, output_dir):
    """
    Create a feature importance bar chart.
    
    This shows which features the model relies on most for predictions.
    Higher importance = feature has more influence on predictions.
    
    This is crucial for:
    - Model interpretability
    - Business insights (what drives demand?)
    - Feature selection for future improvements
    """
    print_subheader("Generating: Feature Importance Plot")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort descending
    
    # Create readable feature names
    feature_names = {
        'place_id': 'Restaurant ID',
        'hour': 'Hour of Day',
        'day_of_week': 'Day of Week',
        'day_of_month': 'Day of Month',
        'month': 'Month',
        'week_of_year': 'Week of Year',
        'is_weekend': 'Is Weekend',
        'is_lunch': 'Is Lunch Hour',
        'is_dinner': 'Is Dinner Hour'
    }
    
    # Sort features by importance
    sorted_features = [feature_names.get(feature_cols[i], feature_cols[i]) for i in indices]
    sorted_importances = importances[indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_features)))[::-1]
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor='white')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, sorted_importances)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.1%}', ha='left', va='center', fontsize=10)
    
    # Labels and title
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance for Demand Prediction\n(Higher = more influence on predictions)', fontsize=14)
    
    # Invert y-axis so most important is at top
    ax.invert_yaxis()
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Save
    filepath = output_dir / 'feature_importance.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filepath}")
    
    # Also print to console
    print(f"\n📊 FEATURE IMPORTANCE RANKING:")
    for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), 1):
        print(f"   {i}. {feat}: {imp:.1%}")


# ==============================================================================
# SAVE EVALUATION REPORT
# ==============================================================================

def save_evaluation_report(metrics, train_df, test_df, feature_cols, output_dir):
    """
    Save a comprehensive evaluation report as a text file.
    
    This report is designed to be:
    - Readable by non-technical stakeholders
    - Auditable for model governance
    - Defensible for consultant presentations
    """
    print_subheader("Saving Evaluation Report")
    
    report_path = output_dir / 'model_evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  SHIFT PLANNING MODEL - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("1. DATA SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total records:     {len(train_df) + len(test_df):,}\n")
        f.write(f"Training records:  {len(train_df):,} ({100*len(train_df)/(len(train_df)+len(test_df)):.0f}%)\n")
        f.write(f"Test records:      {len(test_df):,} ({100*len(test_df)/(len(train_df)+len(test_df)):.0f}%)\n")
        f.write(f"Training period:   {train_df['date'].min().date()} to {train_df['date'].max().date()}\n")
        f.write(f"Test period:       {test_df['date'].min().date()} to {test_df['date'].max().date()}\n")
        f.write(f"\nNote: Time-based split used (no data leakage from future)\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("2. MODEL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Algorithm:         Random Forest Regressor\n")
        f.write(f"Number of trees:   {N_ESTIMATORS}\n")
        f.write(f"Max tree depth:    {MAX_DEPTH}\n")
        f.write(f"Features used:     {len(feature_cols)}\n")
        for feat in feature_cols:
            f.write(f"  - {feat}\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("3. REGRESSION METRICS (Test Set Performance)\n")
        f.write("-" * 70 + "\n")
        f.write(f"\n")
        f.write(f"  MAE  (Mean Absolute Error):         {metrics['mae']:.4f}\n")
        f.write(f"  RMSE (Root Mean Squared Error):     {metrics['rmse']:.4f}\n")
        f.write(f"  R²   (Coefficient of Determination): {metrics['r2']:.4f}\n")
        f.write(f"\n")
        f.write(f"Interpretation:\n")
        f.write(f"  • MAE = {metrics['mae']:.2f}: On average, predictions are off by ~{metrics['mae']:.1f} orders\n")
        f.write(f"  • RMSE = {metrics['rmse']:.2f}: Accounts for larger errors more heavily\n")
        f.write(f"  • R² = {metrics['r2']:.3f}: Model explains {metrics['r2']*100:.1f}% of demand variance\n")
        f.write(f"\n")
        
        f.write("-" * 70 + "\n")
        f.write("4. RESIDUAL ANALYSIS\n")
        f.write("-" * 70 + "\n")
        residuals = metrics['residuals']
        f.write(f"Mean residual:     {residuals.mean():.4f} (ideal: 0)\n")
        f.write(f"Std residual:      {residuals.std():.4f}\n")
        f.write(f"Min residual:      {residuals.min():.4f}\n")
        f.write(f"Max residual:      {residuals.max():.4f}\n")
        f.write(f"\n")
        
        f.write("-" * 70 + "\n")
        f.write("5. OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"• predicted_vs_actual.png  - Scatter plot of predictions\n")
        f.write(f"• residuals_vs_actual.png  - Error analysis by actual value\n")
        f.write(f"• residuals_histogram.png  - Distribution of prediction errors\n")
        f.write(f"• feature_importance.png   - Which features matter most\n")
        f.write(f"\n")
        
        f.write("-" * 70 + "\n")
        f.write("6. NOTES FOR STAKEHOLDERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"• This is a REGRESSION model (predicts numeric values)\n")
        f.write(f"• Classification metrics (accuracy, precision, recall) do NOT apply\n")
        f.write(f"• Time-based split ensures realistic evaluation\n")
        f.write(f"• Feature importance shows what drives predictions\n")
        f.write(f"\n")
        
        f.write("=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Saved: {report_path}")


# ==============================================================================
# FORECASTING & STAFFING
# ==============================================================================

def generate_forecast(model, feature_cols, place_id, start_date, days=7):
    """
    Generate demand forecast for a specific place.
    
    Args:
        model: Trained model
        feature_cols: Feature column names
        place_id: Restaurant to forecast
        start_date: First day of forecast
        days: Number of days to forecast
        
    Returns:
        DataFrame with predictions
    """
    predictions = []
    
    for day_offset in range(days):
        date = start_date + timedelta(days=day_offset)
        for hour in range(6, 24):  # Operating hours 6am - midnight
            row = {
                'place_id': place_id,
                'date': date,
                'hour': hour,
                'day_of_week': date.weekday(),
                'day_of_month': date.day,
                'month': date.month,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_lunch': 1 if 11 <= hour <= 14 else 0,
                'is_dinner': 1 if 18 <= hour <= 21 else 0,
            }
            predictions.append(row)
    
    pred_df = pd.DataFrame(predictions)
    X_pred = pred_df[feature_cols]
    
    pred_df['predicted_orders'] = model.predict(X_pred).round().astype(int)
    pred_df['predicted_orders'] = np.clip(pred_df['predicted_orders'], 0, None)
    
    return pred_df


def calculate_staffing(forecast_df):
    """
    Calculate staffing requirements based on demand forecast.
    
    Args:
        forecast_df: DataFrame with predicted_orders column
        
    Returns:
        DataFrame with staffing requirements
    """
    df = forecast_df.copy()
    
    # Calculate raw staff needed
    df['staff_needed'] = (df['predicted_orders'] / ORDERS_PER_STAFF_PER_HOUR).apply(np.ceil).astype(int)
    
    # Apply min/max constraints
    df['staff_needed'] = np.clip(df['staff_needed'], MIN_STAFF_PER_SHIFT, MAX_STAFF_PER_SHIFT)
    
    # Add shift labels
    df['shift'] = df['hour'].apply(get_shift_name)
    df['day_name'] = df['date'].dt.day_name()
    
    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print_header("SHIFT PLANNING DECISION ASSISTANT")
    print_header("Model Training & Evaluation")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load data
    df = load_hourly_demand()
    if df is None:
        return
    
    # Step 2: Create features
    df, feature_cols = create_features(df)
    
    # Step 3: Time-based train/test split (CRITICAL for proper evaluation)
    train_df, test_df = time_based_split(df, test_size=TEST_SIZE)
    
    # Step 4: Train model
    model = train_model(train_df, feature_cols)
    
    # Step 5: Evaluate model (REGRESSION metrics only)
    metrics = evaluate_model(model, test_df, feature_cols)
    
    # Step 6: Generate visualizations
    print_header("Generating Evaluation Visualizations")
    plot_predicted_vs_actual(metrics['y_test'], metrics['y_pred'], OUTPUT_DIR)
    plot_residuals_vs_actual(metrics['y_test'], metrics['residuals'], OUTPUT_DIR)
    plot_residuals_histogram(metrics['residuals'], OUTPUT_DIR)
    plot_feature_importance(model, feature_cols, OUTPUT_DIR)
    
    # Step 7: Save evaluation report
    save_evaluation_report(metrics, train_df, test_df, feature_cols, OUTPUT_DIR)
    
    # Step 8: Generate sample forecast
    print_header("Sample Forecast Generation")
    
    # Get top place by order volume
    top_place = train_df.groupby('place_id')['orders'].sum().idxmax()
    print(f"   Generating 7-day forecast for Place {int(top_place)}...")
    
    # Generate forecast starting tomorrow
    start_date = df['date'].max() + timedelta(days=1)
    forecast = generate_forecast(model, feature_cols, top_place, start_date, days=7)
    schedule = calculate_staffing(forecast)
    
    # Save weekly schedule
    schedule_path = OUTPUT_DIR / 'weekly_schedule.csv'
    schedule.to_csv(schedule_path, index=False)
    print(f"✅ Saved: {schedule_path}")
    
    # Save daily summary
    daily_summary = schedule.groupby(['date', 'day_name']).agg({
        'predicted_orders': 'sum',
        'staff_needed': ['min', 'max', 'mean']
    }).round(1)
    daily_summary.columns = ['total_orders', 'min_staff', 'max_staff', 'avg_staff']
    daily_summary = daily_summary.reset_index()
    
    summary_path = OUTPUT_DIR / 'daily_summary.csv'
    daily_summary.to_csv(summary_path, index=False)
    print(f"✅ Saved: {summary_path}")
    
    # Final summary
    print_header("EXECUTION COMPLETE")
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")
    print(f"\n📊 Model Performance Summary:")
    print(f"   • MAE:  {metrics['mae']:.2f} orders")
    print(f"   • RMSE: {metrics['rmse']:.2f} orders")
    print(f"   • R²:   {metrics['r2']:.3f}")
    print(f"\n📈 Visualizations generated:")
    print(f"   • predicted_vs_actual.png")
    print(f"   • residuals_vs_actual.png")
    print(f"   • residuals_histogram.png")
    print(f"   • feature_importance.png")
    print(f"\n📝 Full report: model_evaluation_report.txt")


if __name__ == "__main__":
    main()
