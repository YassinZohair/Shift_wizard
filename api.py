#!/usr/bin/env python3
"""
==============================================================================
Shift Planning Decision Assistant - FastAPI Backend
==============================================================================

REST API for the shift planning system.

SETUP:
    pip install fastapi uvicorn pandas numpy scikit-learn

RUN:
    uvicorn api:app --reload --port 8000

API DOCS:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# APP SETUP
# ==============================================================================

app = FastAPI(
    title="Shift Planning API",
    description="AI-powered demand forecasting and staffing optimization",
    version="1.0.0"
)

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CLEANED_DATA_DIR = Path("./cleaned_data")
DEFAULT_ORDERS_PER_STAFF = 8
DEFAULT_MIN_STAFF = 2
DEFAULT_MAX_STAFF = 15

# ==============================================================================
# DATA MODELS (Pydantic)
# ==============================================================================

class ForecastRequest(BaseModel):
    place_id: float
    start_date: str  # YYYY-MM-DD
    days: int = 7
    orders_per_staff: int = 8
    min_staff: int = 2
    max_staff: int = 15

class DisruptionRequest(BaseModel):
    place_id: float
    date: str  # YYYY-MM-DD
    disruption_type: str  # "call_off" or "demand_spike"
    affected_shift: str  # "Morning", "Afternoon", "Evening", "All Day"
    num_calloffs: int = 0
    demand_increase_pct: int = 0
    orders_per_staff: int = 8
    min_staff: int = 2
    max_staff: int = 15

class HourlyForecast(BaseModel):
    date: str
    hour: int
    day_name: str
    shift: str
    predicted_orders: int
    staff_needed: int

class DailySummary(BaseModel):
    date: str
    day_name: str
    total_orders: int
    min_staff: int
    max_staff: int
    avg_staff: float

class PlaceInfo(BaseModel):
    place_id: float
    total_orders: int
    avg_hourly_orders: float

# ==============================================================================
# GLOBAL STATE (loaded once at startup)
# ==============================================================================

class AppState:
    def __init__(self):
        self.hourly_demand = None
        self.model = None
        self.feature_cols = None
        self.places = []
        
state = AppState()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_shift(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def train_model(hourly_demand: pd.DataFrame):
    """Train the forecasting model."""
    df = hourly_demand.copy()
    df['date'] = pd.to_datetime(df['date'])
    
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

def predict_demand(place_id: float, start_date: datetime, days: int) -> pd.DataFrame:
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
    X_pred = pred_df[state.feature_cols]
    pred_df['predicted_orders'] = state.model.predict(X_pred).round().astype(int)
    pred_df['predicted_orders'] = np.clip(pred_df['predicted_orders'], 0, None)
    
    return pred_df

def calculate_staffing(predictions: pd.DataFrame, orders_per_staff: int, min_staff: int, max_staff: int) -> pd.DataFrame:
    """Calculate staffing requirements."""
    df = predictions.copy()
    
    df['staff_needed'] = (df['predicted_orders'] / orders_per_staff).apply(np.ceil).astype(int)
    df['staff_needed'] = np.clip(df['staff_needed'], min_staff, max_staff)
    df['shift'] = df['hour'].apply(get_shift)
    df['day_name'] = df['date'].dt.day_name()
    
    return df

# ==============================================================================
# STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Load data and train model on startup."""
    print("🚀 Starting Shift Planning API...")
    
    # Load hourly demand
    hourly_demand_path = CLEANED_DATA_DIR / "hourly_demand.csv"
    if not hourly_demand_path.exists():
        print("❌ ERROR: cleaned_data/hourly_demand.csv not found!")
        print("   Run 'python dataloader.py' first.")
        return
    
    state.hourly_demand = pd.read_csv(hourly_demand_path)
    state.hourly_demand['date'] = pd.to_datetime(state.hourly_demand['date'])
    print(f"✅ Loaded {len(state.hourly_demand):,} demand records")
    
    # Get places
    place_stats = state.hourly_demand.groupby('place_id').agg({
        'orders': ['sum', 'mean']
    }).reset_index()
    place_stats.columns = ['place_id', 'total_orders', 'avg_hourly_orders']
    place_stats = place_stats.sort_values('total_orders', ascending=False)
    state.places = place_stats.to_dict('records')
    print(f"✅ Found {len(state.places)} places")
    
    # Train model
    print("🤖 Training forecasting model...")
    state.model, state.feature_cols = train_model(state.hourly_demand)
    print("✅ Model trained successfully!")
    
    print("🎉 API ready!")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "message": "Shift Planning API is running",
        "docs": "/docs"
    }

@app.get("/api/places", response_model=List[PlaceInfo])
async def get_places():
    """Get list of all places with stats."""
    if not state.places:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    return state.places[:50]  # Return top 50 places

@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    if state.hourly_demand is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    return {
        "total_records": len(state.hourly_demand),
        "total_places": len(state.places),
        "date_range": {
            "start": state.hourly_demand['date'].min().strftime('%Y-%m-%d'),
            "end": state.hourly_demand['date'].max().strftime('%Y-%m-%d')
        },
        "total_orders": int(state.hourly_demand['orders'].sum())
    }

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    """Generate demand forecast and staffing schedule."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Generate predictions
    predictions = predict_demand(request.place_id, start_date, request.days)
    
    # Calculate staffing
    schedule = calculate_staffing(
        predictions, 
        request.orders_per_staff, 
        request.min_staff, 
        request.max_staff
    )
    
    # Format hourly data
    hourly_data = []
    for _, row in schedule.iterrows():
        hourly_data.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "hour": int(row['hour']),
            "day_name": row['day_name'],
            "shift": row['shift'],
            "predicted_orders": int(row['predicted_orders']),
            "staff_needed": int(row['staff_needed'])
        })
    
    # Calculate daily summary
    daily_summary = []
    for date in schedule['date'].unique():
        day_data = schedule[schedule['date'] == date]
        daily_summary.append({
            "date": date.strftime('%Y-%m-%d'),
            "day_name": day_data['day_name'].iloc[0],
            "total_orders": int(day_data['predicted_orders'].sum()),
            "min_staff": int(day_data['staff_needed'].min()),
            "max_staff": int(day_data['staff_needed'].max()),
            "avg_staff": round(day_data['staff_needed'].mean(), 1)
        })
    
    # Calculate shift summary
    shift_summary = schedule.groupby(['date', 'day_name', 'shift']).agg({
        'predicted_orders': 'sum',
        'staff_needed': 'max'
    }).reset_index()
    
    shift_data = []
    for _, row in shift_summary.iterrows():
        shift_data.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "day_name": row['day_name'],
            "shift": row['shift'],
            "orders": int(row['predicted_orders']),
            "staff_needed": int(row['staff_needed'])
        })
    
    # Calculate totals
    total_orders = int(schedule['predicted_orders'].sum())
    avg_staff = round(schedule['staff_needed'].mean(), 1)
    peak_staff = int(schedule['staff_needed'].max())
    
    return {
        "summary": {
            "total_orders": total_orders,
            "avg_staff": avg_staff,
            "peak_staff": peak_staff,
            "days": request.days
        },
        "hourly": hourly_data,
        "daily": daily_summary,
        "shifts": shift_data
    }

@app.post("/api/replan")
async def replan(request: DisruptionRequest):
    """Handle disruption and recalculate schedule."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        date = datetime.strptime(request.date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Generate base predictions for the day
    predictions = predict_demand(request.place_id, date, 1)
    schedule = calculate_staffing(
        predictions,
        request.orders_per_staff,
        request.min_staff,
        request.max_staff
    )
    
    # Apply disruption
    shift = request.affected_shift
    if shift == "All Day":
        mask = pd.Series([True] * len(schedule))
    elif shift == "Morning":
        mask = (schedule['hour'] >= 6) & (schedule['hour'] < 12)
    elif shift == "Afternoon":
        mask = (schedule['hour'] >= 12) & (schedule['hour'] < 17)
    elif shift == "Evening":
        mask = (schedule['hour'] >= 17) & (schedule['hour'] < 22)
    else:
        mask = pd.Series([True] * len(schedule))
    
    original_staff = schedule.loc[mask, 'staff_needed'].copy()
    
    if request.disruption_type == "call_off":
        # Need more staff to cover call-offs
        schedule.loc[mask, 'staff_needed'] += request.num_calloffs
        schedule['staff_needed'] = np.clip(schedule['staff_needed'], request.min_staff, request.max_staff)
        recommendation = f"Call in {request.num_calloffs} additional staff for {shift}"
    else:
        # Demand spike - recalculate
        increase = request.demand_increase_pct / 100
        schedule.loc[mask, 'predicted_orders'] = (
            schedule.loc[mask, 'predicted_orders'] * (1 + increase)
        ).round().astype(int)
        
        new_staff = (schedule.loc[mask, 'predicted_orders'] / request.orders_per_staff).apply(np.ceil).astype(int)
        schedule.loc[mask, 'staff_needed'] = np.clip(new_staff, request.min_staff, request.max_staff)
        recommendation = f"Increase staffing to {schedule.loc[mask, 'staff_needed'].max()} for {shift}"
    
    # Format response
    adjusted_hours = []
    for idx in schedule[mask].index:
        row = schedule.loc[idx]
        orig = original_staff.loc[idx] if idx in original_staff.index else row['staff_needed']
        if row['staff_needed'] != orig:
            adjusted_hours.append({
                "hour": int(row['hour']),
                "shift": row['shift'],
                "original_staff": int(orig),
                "new_staff": int(row['staff_needed']),
                "change": int(row['staff_needed'] - orig)
            })
    
    return {
        "recommendation": recommendation,
        "disruption_type": request.disruption_type,
        "affected_shift": shift,
        "adjusted_hours": adjusted_hours,
        "total_additional_staff_hours": sum(h['change'] for h in adjusted_hours)
    }

@app.post("/api/smart-alerts")
async def get_smart_alerts(request: ForecastRequest):
    """
    AI-powered smart alerts with actionable recommendations.
    Analyzes the forecast and provides specific staffing suggestions.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Generate predictions
    predictions = predict_demand(request.place_id, start_date, request.days)
    schedule = calculate_staffing(
        predictions,
        request.orders_per_staff,
        request.min_staff,
        request.max_staff
    )
    
    alerts = []
    
    # Analyze each day for potential issues
    for date in schedule['date'].unique():
        day_data = schedule[schedule['date'] == date]
        day_name = day_data['day_name'].iloc[0]
        
        # Find peak hours
        peak_hour_data = day_data.loc[day_data['predicted_orders'].idxmax()]
        peak_hour = int(peak_hour_data['hour'])
        peak_orders = int(peak_hour_data['predicted_orders'])
        peak_staff = int(peak_hour_data['staff_needed'])
        
        # Find low demand periods
        low_hour_data = day_data.loc[day_data['predicted_orders'].idxmin()]
        low_hour = int(low_hour_data['hour'])
        low_orders = int(low_hour_data['predicted_orders'])
        low_staff = int(low_hour_data['staff_needed'])
        
        # Calculate daily metrics
        total_orders = int(day_data['predicted_orders'].sum())
        avg_orders = day_data['predicted_orders'].mean()
        max_staff_needed = int(day_data['staff_needed'].max())
        min_staff_needed = int(day_data['staff_needed'].min())
        
        # ALERT TYPE 1: Demand Spike Detection
        # If peak is significantly higher than average
        if peak_orders > avg_orders * 1.5 and peak_orders > 5:
            alerts.append({
                "id": f"spike_{date.strftime('%Y%m%d')}_{peak_hour}",
                "type": "demand_spike",
                "severity": "high" if peak_orders > avg_orders * 2 else "medium",
                "date": date.strftime('%Y-%m-%d'),
                "day_name": day_name,
                "hour": peak_hour,
                "shift": get_shift(peak_hour),
                "title": f"Demand Spike Expected",
                "description": f"Peak demand of {peak_orders} orders expected at {peak_hour}:00",
                "current_metric": int(avg_orders),
                "predicted_metric": peak_orders,
                "recommendation": {
                    "action": "increase_staff",
                    "staff_needed": peak_staff,
                    "additional_staff": max(0, peak_staff - request.min_staff),
                    "message": f"Schedule {peak_staff} staff members for the {get_shift(peak_hour)} shift ({peak_hour}:00-{peak_hour+1}:00). This is {max(0, peak_staff - request.min_staff)} more than minimum staffing.",
                    "cost_impact": f"Additional labor cost: ~{max(0, peak_staff - request.min_staff) * 15}$/hr",
                    "risk_if_ignored": "Customer wait times may increase by 50-100%, potential loss of customers"
                }
            })
        
        # ALERT TYPE 2: Low Demand Period (Overstaffing Risk)
        if low_orders < avg_orders * 0.5 and request.min_staff > 1:
            potential_savings = request.min_staff - max(1, low_staff)
            if potential_savings > 0:
                alerts.append({
                    "id": f"low_{date.strftime('%Y%m%d')}_{low_hour}",
                    "type": "low_demand",
                    "severity": "low",
                    "date": date.strftime('%Y-%m-%d'),
                    "day_name": day_name,
                    "hour": low_hour,
                    "shift": get_shift(low_hour),
                    "title": f"Low Demand Period",
                    "description": f"Only {low_orders} orders expected at {low_hour}:00",
                    "current_metric": request.min_staff,
                    "predicted_metric": low_orders,
                    "recommendation": {
                        "action": "reduce_staff",
                        "staff_needed": max(1, low_staff),
                        "reduction": potential_savings,
                        "message": f"Consider reducing staff to {max(1, low_staff)} during {low_hour}:00-{low_hour+1}:00. Potential to reduce by {potential_savings} staff member(s).",
                        "cost_impact": f"Potential savings: ~{potential_savings * 15}$/hr",
                        "risk_if_ignored": "Overstaffing leads to unnecessary labor costs"
                    }
                })
        
        # ALERT TYPE 3: Weekend Rush
        if day_name in ['Saturday', 'Sunday'] and total_orders > 50:
            alerts.append({
                "id": f"weekend_{date.strftime('%Y%m%d')}",
                "type": "weekend_rush",
                "severity": "medium",
                "date": date.strftime('%Y-%m-%d'),
                "day_name": day_name,
                "hour": None,
                "shift": "All Day",
                "title": f"Weekend Rush - {day_name}",
                "description": f"Higher demand expected: {total_orders} total orders",
                "current_metric": None,
                "predicted_metric": total_orders,
                "recommendation": {
                    "action": "prepare_weekend",
                    "staff_needed": max_staff_needed,
                    "message": f"Ensure {max_staff_needed} staff available during peak hours. Consider having 1-2 on-call staff ready.",
                    "cost_impact": "Plan for 20-30% higher labor costs than weekdays",
                    "risk_if_ignored": "Long wait times, negative customer reviews"
                }
            })
        
        # ALERT TYPE 4: Lunch Rush
        lunch_data = day_data[(day_data['hour'] >= 11) & (day_data['hour'] <= 14)]
        if len(lunch_data) > 0:
            lunch_orders = int(lunch_data['predicted_orders'].sum())
            lunch_peak = int(lunch_data['predicted_orders'].max())
            if lunch_peak > avg_orders * 1.3:
                alerts.append({
                    "id": f"lunch_{date.strftime('%Y%m%d')}",
                    "type": "lunch_rush",
                    "severity": "medium",
                    "date": date.strftime('%Y-%m-%d'),
                    "day_name": day_name,
                    "hour": 12,
                    "shift": "Afternoon",
                    "title": "Lunch Rush Expected",
                    "description": f"Peak lunch demand: {lunch_peak} orders/hour",
                    "current_metric": int(avg_orders),
                    "predicted_metric": lunch_peak,
                    "recommendation": {
                        "action": "prepare_lunch",
                        "staff_needed": int(lunch_data['staff_needed'].max()),
                        "message": f"Schedule {int(lunch_data['staff_needed'].max())} staff for lunch shift (11:00-14:00). Prep ingredients before 11:00.",
                        "cost_impact": None,
                        "risk_if_ignored": "Lunch customers are time-sensitive, may leave if wait is too long"
                    }
                })
        
        # ALERT TYPE 5: Dinner Rush
        dinner_data = day_data[(day_data['hour'] >= 18) & (day_data['hour'] <= 21)]
        if len(dinner_data) > 0:
            dinner_peak = int(dinner_data['predicted_orders'].max())
            if dinner_peak > avg_orders * 1.3:
                alerts.append({
                    "id": f"dinner_{date.strftime('%Y%m%d')}",
                    "type": "dinner_rush",
                    "severity": "medium",
                    "date": date.strftime('%Y-%m-%d'),
                    "day_name": day_name,
                    "hour": 19,
                    "shift": "Evening",
                    "title": "Dinner Rush Expected",
                    "description": f"Peak dinner demand: {dinner_peak} orders/hour",
                    "current_metric": int(avg_orders),
                    "predicted_metric": dinner_peak,
                    "recommendation": {
                        "action": "prepare_dinner",
                        "staff_needed": int(dinner_data['staff_needed'].max()),
                        "message": f"Schedule {int(dinner_data['staff_needed'].max())} staff for dinner shift (18:00-21:00).",
                        "cost_impact": None,
                        "risk_if_ignored": "Dinner is typically highest revenue period"
                    }
                })
    
    # Sort alerts by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["date"]))
    
    # Summary stats
    summary = {
        "total_alerts": len(alerts),
        "high_severity": len([a for a in alerts if a["severity"] == "high"]),
        "medium_severity": len([a for a in alerts if a["severity"] == "medium"]),
        "low_severity": len([a for a in alerts if a["severity"] == "low"]),
        "total_predicted_orders": int(schedule['predicted_orders'].sum()),
        "avg_daily_orders": int(schedule.groupby('date')['predicted_orders'].sum().mean()),
        "peak_staff_needed": int(schedule['staff_needed'].max()),
        "estimated_labor_hours": int(schedule['staff_needed'].sum())
    }
    
    return {
        "summary": summary,
        "alerts": alerts
    }


@app.get("/api/historical/{place_id}")
async def get_historical(place_id: float):
    """Get historical demand patterns for a place."""
    if state.hourly_demand is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    place_data = state.hourly_demand[state.hourly_demand['place_id'] == place_id]
    
    if len(place_data) == 0:
        raise HTTPException(status_code=404, detail=f"Place {place_id} not found")
    
    # Hourly pattern (average by hour)
    hourly_pattern = place_data.groupby('hour')['orders'].mean().round(1).to_dict()
    
    # Daily pattern (average by weekday)
    place_data['day_of_week'] = place_data['date'].dt.dayofweek
    daily_pattern = place_data.groupby('day_of_week')['orders'].mean().round(1).to_dict()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_pattern = {day_names[k]: v for k, v in daily_pattern.items()}
    
    return {
        "place_id": place_id,
        "total_records": len(place_data),
        "avg_orders_per_hour": round(place_data['orders'].mean(), 2),
        "max_orders_per_hour": int(place_data['orders'].max()),
        "hourly_pattern": hourly_pattern,
        "daily_pattern": daily_pattern
    }

# ==============================================================================
# RUN SERVER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
