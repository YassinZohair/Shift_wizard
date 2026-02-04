#!/usr/bin/env python3
"""
==============================================================================
Shift Planning Decision Assistant - Web Dashboard
==============================================================================

Interactive web dashboard for shift planning and demand forecasting.

SETUP:
    pip install streamlit plotly pandas numpy scikit-learn

RUN:
    streamlit run dashboard.py
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
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    print("=" * 60)
    print("ERROR: Missing packages!")
    print("Run: pip install streamlit plotly pandas numpy scikit-learn")
    print(f"\nMissing: {e}")
    print("=" * 60)
    exit(1)

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Shift Planning Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CLEANED_DATA_DIR = Path("./cleaned_data")
DEFAULT_ORDERS_PER_STAFF = 8
DEFAULT_MIN_STAFF = 2
DEFAULT_MAX_STAFF = 15

# ==============================================================================
# CACHING & DATA LOADING
# ==============================================================================

@st.cache_data
def load_data():
    """Load cleaned data files."""
    hourly_demand_path = CLEANED_DATA_DIR / "hourly_demand.csv"
    
    if not hourly_demand_path.exists():
        return None, None
    
    hourly_demand = pd.read_csv(hourly_demand_path)
    hourly_demand['date'] = pd.to_datetime(hourly_demand['date'])
    
    # Load places if available
    places_path = CLEANED_DATA_DIR / "dim_places.csv"
    places = None
    if places_path.exists():
        places = pd.read_csv(places_path)
    
    return hourly_demand, places


@st.cache_resource
def train_model(hourly_demand: pd.DataFrame):
    """Train the forecasting model."""
    df = hourly_demand.copy()
    
    # Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_dinner'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    
    feature_cols = [
        'place_id', 'hour', 'day_of_week', 'day_of_month',
        'month', 'week_of_year', 'is_weekend', 'is_lunch', 'is_dinner'
    ]
    
    X = df[feature_cols]
    y = df['orders']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, feature_cols


def predict_demand(model, feature_cols, place_id, start_date, days=7):
    """Generate demand predictions."""
    predictions = []
    
    for day_offset in range(days):
        date = start_date + timedelta(days=day_offset)
        for hour in range(6, 24):
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
    pred_df['predicted_orders'] = model.predict(X_pred).round().astype(int); pred_df['predicted_orders'] = np.clip(pred_df['predicted_orders'], 0, None)
    
    return pred_df


def calculate_staffing(predictions, orders_per_staff, min_staff, max_staff):
    """Calculate staffing requirements."""
    df = predictions.copy()
    
    df['staff_needed'] = (df['predicted_orders'] / orders_per_staff).apply(np.ceil).astype(int)
    df['staff_needed'] = np.clip(df['staff_needed'], min_staff, max_staff)
    
    def get_shift(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    df['shift'] = df['hour'].apply(get_shift)
    df['day_name'] = df['date'].dt.day_name()
    
    return df


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar(hourly_demand, places):
    """Render the sidebar with controls."""
    st.sidebar.title("⚙️ Settings")
    
    # Place selection
    place_ids = sorted(hourly_demand['place_id'].unique())
    
    # Get top places by volume
    top_places = hourly_demand.groupby('place_id')['orders'].sum().nlargest(10).index.tolist()
    
    selected_place = st.sidebar.selectbox(
        "Select Restaurant",
        options=place_ids,
        index=place_ids.index(top_places[0]) if top_places[0] in place_ids else 0,
        format_func=lambda x: f"Place {int(x)}"
    )
    
    st.sidebar.markdown("---")
    
    # Date selection
    st.sidebar.subheader("📅 Forecast Period")
    max_date = hourly_demand['date'].max()
    default_start = max_date + timedelta(days=1)
    
    forecast_start = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    forecast_days = st.sidebar.slider("Days to Forecast", 1, 14, 7)
    
    st.sidebar.markdown("---")
    
    # Staffing parameters
    st.sidebar.subheader("👥 Staffing Parameters")
    
    orders_per_staff = st.sidebar.slider(
        "Orders per Staff per Hour",
        min_value=1,
        max_value=20,
        value=DEFAULT_ORDERS_PER_STAFF
    )
    
    min_staff = st.sidebar.slider(
        "Minimum Staff",
        min_value=1,
        max_value=10,
        value=DEFAULT_MIN_STAFF
    )
    
    max_staff = st.sidebar.slider(
        "Maximum Staff",
        min_value=5,
        max_value=30,
        value=DEFAULT_MAX_STAFF
    )
    
    return {
        'place_id': selected_place,
        'forecast_start': datetime.combine(forecast_start, datetime.min.time()),
        'forecast_days': forecast_days,
        'orders_per_staff': orders_per_staff,
        'min_staff': min_staff,
        'max_staff': max_staff
    }


# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

def render_kpis(schedule, historical_data, place_id):
    """Render KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_predicted = schedule['predicted_orders'].sum()
    avg_staff = schedule['staff_needed'].mean()
    peak_staff = schedule['staff_needed'].max()
    
    # Historical comparison
    place_history = historical_data[historical_data['place_id'] == place_id]
    avg_historical = place_history['orders'].mean() if len(place_history) > 0 else 0
    
    with col1:
        st.metric(
            label="📦 Total Predicted Orders",
            value=f"{total_predicted:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="👥 Average Staff Needed",
            value=f"{avg_staff:.1f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="📈 Peak Staff Required",
            value=f"{peak_staff}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="📊 Avg Historical Orders/Hour",
            value=f"{avg_historical:.1f}",
            delta=None
        )


def render_demand_chart(schedule):
    """Render demand forecast chart."""
    st.subheader("📈 Demand Forecast")
    
    df = schedule.copy()
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Demand line
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['predicted_orders'],
            name="Predicted Orders",
            fill='tozeroy',
            line=dict(color='#2E86AB', width=2)
        ),
        secondary_y=False
    )
    
    # Staff line
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['staff_needed'],
            name="Staff Needed",
            line=dict(color='#A23B72', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date & Time")
    fig.update_yaxes(title_text="Orders", secondary_y=False)
    fig.update_yaxes(title_text="Staff", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(schedule):
    """Render demand heatmap."""
    st.subheader("🗓️ Weekly Demand Heatmap")
    
    df = schedule.copy()
    df['day_num'] = df['date'].dt.dayofweek
    
    pivot = df.pivot_table(
        values='predicted_orders',
        index='day_num',
        columns='hour',
        aggfunc='mean'
    ).fillna(0)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot.index = [day_names[i] for i in pivot.index]
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Day", color="Orders"),
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    
    fig.update_layout(height=300)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_daily_breakdown(schedule):
    """Render daily breakdown table."""
    st.subheader("📋 Daily Breakdown")
    
    daily = schedule.groupby(['date', 'day_name']).agg({
        'predicted_orders': 'sum',
        'staff_needed': ['min', 'max', 'mean']
    }).round(1)
    
    daily.columns = ['Total Orders', 'Min Staff', 'Max Staff', 'Avg Staff']
    daily = daily.reset_index()
    daily['date'] = daily['date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        daily,
        use_container_width=True,
        hide_index=True
    )


def render_shift_breakdown(schedule):
    """Render shift breakdown."""
    st.subheader("⏰ Shift Breakdown")
    
    shift_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    schedule['shift_order'] = schedule['shift'].map(shift_order)
    
    shift_summary = schedule.groupby(['day_name', 'shift', 'shift_order']).agg({
        'predicted_orders': 'sum',
        'staff_needed': 'max'
    }).reset_index().sort_values(['shift_order'])
    
    fig = px.bar(
        shift_summary,
        x='day_name',
        y='staff_needed',
        color='shift',
        barmode='group',
        category_orders={
            'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'shift': ['Morning', 'Afternoon', 'Evening', 'Night']
        },
        color_discrete_sequence=['#F4D35E', '#F18F01', '#C73E1D', '#3B1F2B']
    )
    
    fig.update_layout(
        height=350,
        xaxis_title="Day",
        yaxis_title="Staff Needed",
        legend_title="Shift"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_schedule_table(schedule):
    """Render detailed schedule table."""
    st.subheader("📅 Detailed Schedule")
    
    display_df = schedule[['date', 'day_name', 'hour', 'shift', 'predicted_orders', 'staff_needed']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['hour'] = display_df['hour'].apply(lambda x: f"{x:02d}:00")
    display_df.columns = ['Date', 'Day', 'Hour', 'Shift', 'Predicted Orders', 'Staff Needed']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )


def render_download_section(schedule):
    """Render download buttons."""
    st.subheader("💾 Download Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = schedule.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Schedule (CSV)",
            data=csv,
            file_name="shift_schedule.csv",
            mime="text/csv"
        )
    
    with col2:
        # Daily summary
        daily = schedule.groupby(['date', 'day_name']).agg({
            'predicted_orders': 'sum',
            'staff_needed': ['min', 'max']
        }).reset_index()
        daily.columns = ['Date', 'Day', 'Total Orders', 'Min Staff', 'Max Staff']
        
        csv_daily = daily.to_csv(index=False)
        st.download_button(
            label="📥 Download Daily Summary (CSV)",
            data=csv_daily,
            file_name="daily_summary.csv",
            mime="text/csv"
        )


# ==============================================================================
# RE-PLANNING SECTION
# ==============================================================================

def render_replanning_section(schedule, settings):
    """Render the real-time re-planning section."""
    st.markdown("---")
    st.header("🔄 Real-Time Re-Planning")
    st.markdown("*Simulate disruptions like call-offs and adjust the schedule.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Report a Disruption")
        
        disruption_type = st.selectbox(
            "Disruption Type",
            ["Staff Call-Off", "Unexpected Demand Spike", "Event/Promotion"]
        )
        
        disruption_date = st.date_input(
            "Date",
            value=schedule['date'].min()
        )
        
        if disruption_type == "Staff Call-Off":
            num_calloffs = st.number_input("Number of Staff Calling Off", 1, 5, 1)
            affected_shift = st.selectbox("Affected Shift", ["Morning", "Afternoon", "Evening", "All Day"])
        
        elif disruption_type == "Unexpected Demand Spike":
            demand_increase = st.slider("Expected Demand Increase (%)", 10, 100, 30)
            affected_shift = st.selectbox("Affected Period", ["Lunch (11-14)", "Dinner (18-21)", "All Day"])
        
        else:  # Event
            demand_increase = st.slider("Expected Demand Increase (%)", 20, 200, 50)
            affected_shift = st.selectbox("Event Period", ["Morning", "Afternoon", "Evening", "All Day"])
        
        if st.button("🔄 Recalculate Schedule", type="primary"):
            st.session_state['disruption'] = {
                'type': disruption_type,
                'date': disruption_date,
                'details': {
                    'num_calloffs': num_calloffs if disruption_type == "Staff Call-Off" else 0,
                    'demand_increase': demand_increase if disruption_type != "Staff Call-Off" else 0,
                    'affected_shift': affected_shift
                }
            }
    
    with col2:
        st.subheader("📊 Adjusted Schedule")
        
        if 'disruption' in st.session_state:
            disruption = st.session_state['disruption']
            adjusted = schedule.copy()
            
            # Filter to affected date
            date_mask = adjusted['date'].dt.date == disruption['date']
            
            # Apply shift mask
            shift = disruption['details']['affected_shift']
            if shift == "All Day":
                shift_mask = date_mask
            elif shift == "Lunch (11-14)":
                shift_mask = date_mask & (adjusted['hour'] >= 11) & (adjusted['hour'] <= 14)
            elif shift == "Dinner (18-21)":
                shift_mask = date_mask & (adjusted['hour'] >= 18) & (adjusted['hour'] <= 21)
            else:
                shift_mask = date_mask & (adjusted['shift'] == shift)
            
            if disruption['type'] == "Staff Call-Off":
                # Increase staff needed to cover call-offs
                adjusted.loc[shift_mask, 'staff_needed'] += disruption['details']['num_calloffs']
                adjusted['staff_needed'] = np.clip(adjusted['staff_needed'], None, settings['max_staff'])
                
                st.warning(f"⚠️ {disruption['details']['num_calloffs']} staff called off on {disruption['date']}")
                st.info(f"💡 Recommendation: Call in {disruption['details']['num_calloffs']} additional staff for {shift}")
            
            else:
                # Increase demand prediction and recalculate staff
                increase_pct = disruption['details']['demand_increase'] / 100
                adjusted.loc[shift_mask, 'predicted_orders'] = (
                    adjusted.loc[shift_mask, 'predicted_orders'] * (1 + increase_pct)
                ).round().astype(int)
                
                # Recalculate staff
                staff_calc = (
                    adjusted.loc[shift_mask, 'predicted_orders'] / settings['orders_per_staff']
                ).apply(np.ceil).astype(int)
                adjusted.loc[shift_mask, 'staff_needed'] = np.clip(staff_calc, settings['min_staff'], settings['max_staff'])
                
                st.warning(f"📈 Expected {disruption['details']['demand_increase']}% demand increase on {disruption['date']}")
                new_staff = adjusted.loc[shift_mask, 'staff_needed'].max()
                st.info(f"💡 Recommendation: Schedule up to {new_staff} staff for {shift}")
            
            # Show comparison
            comparison = adjusted[date_mask][['hour', 'shift', 'predicted_orders', 'staff_needed']].copy()
            original = schedule[date_mask][['hour', 'staff_needed']].copy()
            comparison['original_staff'] = original['staff_needed'].values
            comparison['change'] = comparison['staff_needed'] - comparison['original_staff']
            comparison = comparison[comparison['change'] != 0]
            
            if len(comparison) > 0:
                st.dataframe(comparison, use_container_width=True, hide_index=True)
            else:
                st.success("No changes needed for the selected period.")
        
        else:
            st.info("👆 Report a disruption to see adjusted recommendations")


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.title("📊 Shift Planning Decision Assistant")
    st.markdown("*AI-powered demand forecasting and staffing optimization for restaurants*")
    
    # Load data
    hourly_demand, places = load_data()
    
    if hourly_demand is None:
        st.error("❌ Data not found! Please run `dataloader.py` first.")
        st.code("python dataloader.py", language="bash")
        return
    
    # Sidebar settings
    settings = render_sidebar(hourly_demand, places)
    
    # Train model
    with st.spinner("Training forecasting model..."):
        model, feature_cols = train_model(hourly_demand)
    
    # Generate predictions
    predictions = predict_demand(
        model, 
        feature_cols, 
        settings['place_id'], 
        settings['forecast_start'],
        settings['forecast_days']
    )
    
    # Calculate staffing
    schedule = calculate_staffing(
        predictions,
        settings['orders_per_staff'],
        settings['min_staff'],
        settings['max_staff']
    )
    
    # Render dashboard
    st.markdown("---")
    
    # KPIs
    render_kpis(schedule, hourly_demand, settings['place_id'])
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_demand_chart(schedule)
    
    with col2:
        render_heatmap(schedule)
    
    # Breakdowns
    col1, col2 = st.columns(2)
    
    with col1:
        render_daily_breakdown(schedule)
    
    with col2:
        render_shift_breakdown(schedule)
    
    # Detailed schedule
    render_schedule_table(schedule)
    
    # Downloads
    render_download_section(schedule)
    
    # Re-planning section
    render_replanning_section(schedule, settings)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built for Hackathon 2024 | Shift Planning Decision Assistant"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
