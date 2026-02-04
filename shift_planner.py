#!/usr/bin/env python3
"""
==============================================================================
Shift Planning Decision Assistant - Full Pipeline
==============================================================================

This script provides:
1. Demand Forecasting - Predicts hourly orders using historical patterns
2. Staffing Calculator - Converts predicted demand into staff requirements
3. Dashboard - Generates visualizations for decision-making

SETUP:
    pip install pandas numpy matplotlib scikit-learn

USAGE:
    python shift_planner.py
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta

# ==============================================================================
# INSTALL CHECK
# ==============================================================================
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Input: cleaned data from dataloader.py
CLEANED_DATA_DIR = Path("./cleaned_data")

# Output: forecasts and reports
OUTPUT_DIR = Path("./shift_planning_output")

# Staffing parameters (adjust based on restaurant capacity)
ORDERS_PER_STAFF_PER_HOUR = 8  # How many orders one staff member can handle per hour
MIN_STAFF_PER_SHIFT = 2        # Minimum staff always on duty
MAX_STAFF_PER_SHIFT = 15       # Maximum staff capacity

# ==============================================================================
# 1. DEMAND FORECASTING MODEL
# ==============================================================================

class DemandForecaster:
    """Predicts hourly order demand using historical patterns."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for the model."""
        df = df.copy()
        
        # Convert date to datetime (handle any format)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Hour-based features
        df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_dinner'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        df['is_peak'] = (df['is_lunch'] | df['is_dinner']).astype(int)
        
        return df
    
    def train(self, hourly_demand: pd.DataFrame) -> dict:
        """Train the forecasting model."""
        print("Training demand forecasting model...")
        
        # Prepare features
        df = self.prepare_features(hourly_demand)
        
        # Feature columns
        feature_cols = [
            'place_id', 'hour', 'day_of_week', 'day_of_month', 
            'month', 'week_of_year', 'is_weekend', 'is_lunch', 
            'is_dinner', 'is_peak'
        ]
        
        X = df[feature_cols]
        y = df['orders']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_cols = feature_cols
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        print(f"  Model trained successfully!")
        print(f"  MAE: {mae:.2f} orders")
        print(f"  RMSE: {rmse:.2f} orders")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'feature_importance': importance
        }
    
    def predict(self, place_id: float, date: datetime, hours: list = None) -> pd.DataFrame:
        """Predict demand for a specific place and date."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if hours is None:
            hours = list(range(6, 24))  # 6 AM to 11 PM
        
        # Create prediction dataframe
        predictions = []
        for hour in hours:
            row = {
                'place_id': place_id,
                'date': date,
                'hour': hour,
            }
            predictions.append(row)
        
        pred_df = pd.DataFrame(predictions)
        pred_df = self.prepare_features(pred_df)
        
        # Predict
        X_pred = pred_df[self.feature_cols]
        pred_df['predicted_orders'] = self.model.predict(X_pred).round().astype(int)
        pred_df['predicted_orders'] = pred_df['predicted_orders'].clip(lower=0)
        
        return pred_df[['place_id', 'date', 'hour', 'predicted_orders']]
    
    def predict_week(self, place_id: float, start_date: datetime) -> pd.DataFrame:
        """Predict demand for a full week."""
        all_predictions = []
        
        for day_offset in range(7):
            date = start_date + timedelta(days=day_offset)
            daily_pred = self.predict(place_id, date)
            all_predictions.append(daily_pred)
        
        return pd.concat(all_predictions, ignore_index=True)


# ==============================================================================
# 2. STAFFING CALCULATOR
# ==============================================================================

class StaffingCalculator:
    """Converts demand predictions into staffing requirements."""
    
    def __init__(
        self,
        orders_per_staff: int = ORDERS_PER_STAFF_PER_HOUR,
        min_staff: int = MIN_STAFF_PER_SHIFT,
        max_staff: int = MAX_STAFF_PER_SHIFT
    ):
        self.orders_per_staff = orders_per_staff
        self.min_staff = min_staff
        self.max_staff = max_staff
    
    def calculate_staff_needed(self, predicted_orders: int) -> int:
        """Calculate staff needed for a given number of orders."""
        if predicted_orders <= 0:
            return self.min_staff
        
        staff_needed = int(np.ceil(predicted_orders / self.orders_per_staff))
        staff_needed = max(self.min_staff, staff_needed)
        staff_needed = min(self.max_staff, staff_needed)
        
        return staff_needed
    
    def generate_schedule(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Generate staffing schedule from predictions."""
        schedule = predictions.copy()
        
        # Calculate staff needed
        schedule['staff_needed'] = schedule['predicted_orders'].apply(
            self.calculate_staff_needed
        )
        
        # Add shift labels
        def get_shift(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        schedule['shift'] = schedule['hour'].apply(get_shift)
        
        # Add day name
        if 'date' in schedule.columns:
            schedule['day_name'] = pd.to_datetime(schedule['date']).dt.day_name()
        
        return schedule
    
    def get_daily_summary(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Summarize staffing by day."""
        summary = schedule.groupby(['date', 'day_name']).agg({
            'predicted_orders': 'sum',
            'staff_needed': ['min', 'max', 'mean']
        }).round(1)
        
        summary.columns = ['total_orders', 'min_staff', 'max_staff', 'avg_staff']
        summary = summary.reset_index()
        
        return summary
    
    def get_shift_summary(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Summarize staffing by shift."""
        summary = schedule.groupby(['date', 'day_name', 'shift']).agg({
            'predicted_orders': 'sum',
            'staff_needed': 'max'
        }).reset_index()
        
        summary.columns = ['date', 'day_name', 'shift', 'orders', 'staff_needed']
        
        return summary
    
    def detect_understaffing(
        self, 
        schedule: pd.DataFrame, 
        current_staff: dict = None
    ) -> pd.DataFrame:
        """Identify periods where more staff might be needed."""
        if current_staff is None:
            # Assume current staff is at minimum
            current_staff = {h: self.min_staff for h in range(24)}
        
        alerts = []
        for _, row in schedule.iterrows():
            hour = row['hour']
            needed = row['staff_needed']
            current = current_staff.get(hour, self.min_staff)
            
            if needed > current:
                alerts.append({
                    'date': row['date'],
                    'hour': hour,
                    'shift': row.get('shift', ''),
                    'predicted_orders': row['predicted_orders'],
                    'staff_needed': needed,
                    'current_staff': current,
                    'shortage': needed - current,
                    'alert_level': 'HIGH' if (needed - current) >= 3 else 'MEDIUM'
                })
        
        return pd.DataFrame(alerts)


# ==============================================================================
# 3. DASHBOARD / VISUALIZATION
# ==============================================================================

class ShiftDashboard:
    """Generates visualizations for shift planning."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    def plot_weekly_demand(
        self, 
        schedule: pd.DataFrame, 
        place_id: float,
        save: bool = True
    ) -> None:
        """Plot predicted demand for the week."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Pivot for heatmap-style view
        schedule['datetime'] = pd.to_datetime(schedule['date']) + \
                               pd.to_timedelta(schedule['hour'], unit='h')
        
        ax.fill_between(
            schedule['datetime'], 
            schedule['predicted_orders'],
            alpha=0.3, 
            color=self.colors[0]
        )
        ax.plot(
            schedule['datetime'], 
            schedule['predicted_orders'],
            color=self.colors[0], 
            linewidth=2,
            label='Predicted Orders'
        )
        
        ax.set_xlabel('Date & Time', fontsize=12)
        ax.set_ylabel('Predicted Orders', fontsize=12)
        ax.set_title(f'Weekly Demand Forecast - Place {int(place_id)}', fontsize=14)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'weekly_demand_{int(place_id)}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        plt.close()
    
    def plot_staffing_schedule(
        self, 
        schedule: pd.DataFrame, 
        place_id: float,
        save: bool = True
    ) -> None:
        """Plot staffing requirements alongside demand."""
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        schedule['datetime'] = pd.to_datetime(schedule['date']) + \
                               pd.to_timedelta(schedule['hour'], unit='h')
        
        # Plot orders
        ax1.fill_between(
            schedule['datetime'], 
            schedule['predicted_orders'],
            alpha=0.2, 
            color=self.colors[0]
        )
        ax1.plot(
            schedule['datetime'], 
            schedule['predicted_orders'],
            color=self.colors[0], 
            linewidth=2,
            label='Predicted Orders'
        )
        ax1.set_xlabel('Date & Time', fontsize=12)
        ax1.set_ylabel('Orders', color=self.colors[0], fontsize=12)
        ax1.tick_params(axis='y', labelcolor=self.colors[0])
        
        # Plot staff on secondary axis
        ax2 = ax1.twinx()
        ax2.step(
            schedule['datetime'], 
            schedule['staff_needed'],
            color=self.colors[1], 
            linewidth=2,
            where='mid',
            label='Staff Needed'
        )
        ax2.set_ylabel('Staff Needed', color=self.colors[1], fontsize=12)
        ax2.tick_params(axis='y', labelcolor=self.colors[1])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.set_title(f'Staffing Schedule - Place {int(place_id)}', fontsize=14)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'staffing_schedule_{int(place_id)}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        plt.close()
    
    def plot_daily_patterns(
        self, 
        hourly_demand: pd.DataFrame,
        place_id: float = None,
        save: bool = True
    ) -> None:
        """Plot average hourly patterns by day of week."""
        df = hourly_demand.copy()
        
        if place_id:
            df = df[df['place_id'] == place_id]
        
        # Convert date and get day of week
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        
        # Order days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, day in enumerate(day_order):
            day_data = df[df['day_of_week'] == day]
            hourly_avg = day_data.groupby('hour')['orders'].mean()
            
            ax.plot(
                hourly_avg.index, 
                hourly_avg.values,
                label=day,
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Orders', fontsize=12)
        title = f'Daily Order Patterns - Place {int(place_id)}' if place_id else 'Daily Order Patterns (All Places)'
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticks(range(0, 24))
        
        plt.tight_layout()
        
        if save:
            suffix = f'_{int(place_id)}' if place_id else '_all'
            filepath = self.output_dir / f'daily_patterns{suffix}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        plt.close()
    
    def plot_heatmap(
        self, 
        hourly_demand: pd.DataFrame,
        place_id: float = None,
        save: bool = True
    ) -> None:
        """Create a heatmap of demand by hour and day."""
        df = hourly_demand.copy()
        
        if place_id:
            df = df[df['place_id'] == place_id]
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create pivot table
        pivot = df.pivot_table(
            values='orders',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        ).fillna(0)
        
        # Rename index
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot.index = [day_names[i] for i in pivot.index]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{h}:00' for h in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        title = f'Demand Heatmap - Place {int(place_id)}' if place_id else 'Demand Heatmap (All Places)'
        ax.set_title(title, fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Orders', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            suffix = f'_{int(place_id)}' if place_id else '_all'
            filepath = self.output_dir / f'demand_heatmap{suffix}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        plt.close()
    
    def generate_report(
        self,
        schedule: pd.DataFrame,
        daily_summary: pd.DataFrame,
        shift_summary: pd.DataFrame,
        alerts: pd.DataFrame,
        place_id: float,
        metrics: dict
    ) -> str:
        """Generate a text report."""
        report = []
        report.append("=" * 70)
        report.append(f"SHIFT PLANNING REPORT - Place {int(place_id)}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 70)
        
        report.append("\n📊 MODEL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"  Mean Absolute Error: {metrics['mae']:.2f} orders")
        report.append(f"  RMSE: {metrics['rmse']:.2f} orders")
        
        report.append("\n📅 WEEKLY SUMMARY")
        report.append("-" * 40)
        for _, row in daily_summary.iterrows():
            report.append(
                f"  {row['day_name']:10} | Orders: {row['total_orders']:4.0f} | "
                f"Staff: {row['min_staff']:.0f}-{row['max_staff']:.0f} (avg: {row['avg_staff']:.1f})"
            )
        
        report.append("\n🚨 STAFFING ALERTS")
        report.append("-" * 40)
        if len(alerts) == 0:
            report.append("  No staffing alerts - schedule looks good!")
        else:
            for _, alert in alerts.head(10).iterrows():
                report.append(
                    f"  [{alert['alert_level']}] {alert['date']} @ {alert['hour']}:00 - "
                    f"Need {alert['staff_needed']} staff (short by {alert['shortage']})"
                )
            if len(alerts) > 10:
                report.append(f"  ... and {len(alerts) - 10} more alerts")
        
        report.append("\n📈 TOP FEATURES (Model Importance)")
        report.append("-" * 40)
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for feat, imp in sorted_features[:5]:
            report.append(f"  {feat:20} {imp:.1%}")
        
        report.append("\n" + "=" * 70)
        
        report_text = "\n".join(report)
        
        # Save report
        filepath = self.output_dir / f'report_{int(place_id)}.txt'
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"  Saved: {filepath}")
        
        return report_text


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("SHIFT PLANNING DECISION ASSISTANT")
    print("=" * 70)
    
    # Check for cleaned data
    hourly_demand_path = CLEANED_DATA_DIR / "hourly_demand.csv"
    if not hourly_demand_path.exists():
        print(f"\nERROR: Cleaned data not found at {hourly_demand_path}")
        print("Please run dataloader.py first!")
        exit(1)
    
    # Load data
    print("\n[1] LOADING DATA")
    print("-" * 40)
    hourly_demand = pd.read_csv(hourly_demand_path)
    print(f"  Loaded {len(hourly_demand):,} hourly demand records")
    print(f"  Places: {hourly_demand['place_id'].nunique()}")
    print(f"  Date range: {hourly_demand['date'].min()} to {hourly_demand['date'].max()}")
    
    # Initialize components
    forecaster = DemandForecaster()
    staffing = StaffingCalculator()
    dashboard = ShiftDashboard(OUTPUT_DIR)
    
    # Train model
    print("\n[2] TRAINING FORECASTING MODEL")
    print("-" * 40)
    metrics = forecaster.train(hourly_demand)
    
    # Select a place for demonstration (top place by orders)
    top_places = hourly_demand.groupby('place_id')['orders'].sum().nlargest(3)
    demo_place_id = top_places.index[0]
    print(f"\n  Using Place {int(demo_place_id)} for demonstration (highest volume)")
    
    # Generate forecast for next week
    print("\n[3] GENERATING WEEKLY FORECAST")
    print("-" * 40)
    
    # Use a recent date from the data as "today"
    last_date = pd.to_datetime(hourly_demand['date'].max())
    forecast_start = last_date + timedelta(days=1)
    print(f"  Forecasting week starting: {forecast_start.strftime('%Y-%m-%d')}")
    
    weekly_forecast = forecaster.predict_week(demo_place_id, forecast_start)
    print(f"  Generated {len(weekly_forecast)} hourly predictions")
    
    # Generate staffing schedule
    print("\n[4] CALCULATING STAFFING REQUIREMENTS")
    print("-" * 40)
    schedule = staffing.generate_schedule(weekly_forecast)
    daily_summary = staffing.get_daily_summary(schedule)
    shift_summary = staffing.get_shift_summary(schedule)
    alerts = staffing.detect_understaffing(schedule)
    
    print(f"  Schedule generated for {len(schedule)} hours")
    print(f"  Staffing alerts: {len(alerts)}")
    
    # Generate visualizations
    print("\n[5] GENERATING VISUALIZATIONS")
    print("-" * 40)
    dashboard.plot_weekly_demand(schedule, demo_place_id)
    dashboard.plot_staffing_schedule(schedule, demo_place_id)
    dashboard.plot_daily_patterns(hourly_demand, demo_place_id)
    dashboard.plot_heatmap(hourly_demand, demo_place_id)
    
    # Generate report
    print("\n[6] GENERATING REPORT")
    print("-" * 40)
    report = dashboard.generate_report(
        schedule, daily_summary, shift_summary, alerts, demo_place_id, metrics
    )
    
    # Save outputs
    print("\n[7] SAVING DATA FILES")
    print("-" * 40)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    schedule.to_csv(OUTPUT_DIR / 'weekly_schedule.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'weekly_schedule.csv'}")
    
    daily_summary.to_csv(OUTPUT_DIR / 'daily_summary.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'daily_summary.csv'}")
    
    shift_summary.to_csv(OUTPUT_DIR / 'shift_summary.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'shift_summary.csv'}")
    
    if len(alerts) > 0:
        alerts.to_csv(OUTPUT_DIR / 'staffing_alerts.csv', index=False)
        print(f"  Saved: {OUTPUT_DIR / 'staffing_alerts.csv'}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nFiles generated:")
    print("  📊 weekly_schedule.csv     - Hourly staffing schedule")
    print("  📊 daily_summary.csv       - Daily staffing summary")
    print("  📊 shift_summary.csv       - Shift-level summary")
    print("  📊 staffing_alerts.csv     - Understaffing warnings")
    print("  📈 weekly_demand_*.png     - Demand forecast chart")
    print("  📈 staffing_schedule_*.png - Staffing chart")
    print("  📈 daily_patterns_*.png    - Daily patterns chart")
    print("  📈 demand_heatmap_*.png    - Demand heatmap")
    print("  📝 report_*.txt            - Full text report")
    
    # Print the report
    print("\n")
    print(report)
    
    return {
        'schedule': schedule,
        'daily_summary': daily_summary,
        'shift_summary': shift_summary,
        'alerts': alerts,
        'metrics': metrics
    }


if __name__ == "__main__":
    results = main()
