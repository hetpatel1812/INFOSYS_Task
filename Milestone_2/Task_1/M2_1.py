"""
Heart Rate Forecasting Dashboard using Streamlit and Prophet
Task 1: Simple Heart Rate Forecasting

Installation:
pip install streamlit prophet pandas matplotlib plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Heart Rate Forecasting",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Rate Forecasting Dashboard")
st.markdown("### Prophet-based Time Series Forecasting | Task 1")
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Model Settings")
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=30, value=14)
    generate_new = st.button("🔄 Generate New Data")
    
    st.divider()
    st.header("📚 About")
    st.info("""
    This dashboard demonstrates heart rate forecasting using Facebook Prophet.
    
    **Features:**
    - 60 days of training data
    - Customizable forecast period
    - Confidence intervals
    - Model metrics (MAE)
    """)

# Function to generate sample heart rate data
@st.cache_data
def generate_heart_rate_data(seed=42):
    """Generate 60 days of realistic heart rate data"""
    np.random.seed(seed)
    
    # Base heart rate around 72 bpm
    base_heart_rate = 72
    days = 60
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(days):
        # Add trend (slight sine wave)
        trend = np.sin(i / 10) * 3
        
        # Add weekly seasonality
        weekly_pattern = np.sin((i % 7) / 7 * np.pi * 2) * 2
        
        # Add random noise
        noise = np.random.normal(0, 2)
        
        # Calculate heart rate
        heart_rate = base_heart_rate + trend + weekly_pattern + noise
        
        data.append({
            'ds': start_date + timedelta(days=i),
            'y': round(heart_rate, 1)
        })
    
    return pd.DataFrame(data)

# Function to fit Prophet model and forecast
@st.cache_data
def fit_and_forecast(df, periods):
    """Fit Prophet model and generate forecast"""
    
    # Initialize Prophet model
    model = Prophet(
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    
    # Fit model
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

# Generate or load data
if 'data_seed' not in st.session_state or generate_new:
    st.session_state.data_seed = np.random.randint(0, 1000)

df = generate_heart_rate_data(st.session_state.data_seed)

# Display data preview
with st.expander("📊 View Training Data"):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.metric("Total Days", len(df))
        st.metric("Avg Heart Rate", f"{df['y'].mean():.1f} bpm")
        st.metric("Std Deviation", f"{df['y'].std():.1f} bpm")

# Fit model and generate forecast
with st.spinner("🔮 Training Prophet model..."):
    model, forecast = fit_and_forecast(df, forecast_days)

# Calculate MAE on training data
y_true = df['y'].values
y_pred = forecast.iloc[:len(df)]['yhat'].values
mae = np.mean(np.abs(y_true - y_pred))

# Get Day 67 forecast (7 days after training period)
day_67_idx = 60 + 6  # Index 66 is day 67 (0-indexed)
day_67_forecast = forecast.iloc[day_67_idx]

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Day 67 Forecast",
        f"{day_67_forecast['yhat']:.1f} bpm",
        delta=f"±{(day_67_forecast['yhat_upper'] - day_67_forecast['yhat']):.1f}"
    )

with col2:
    st.metric(
        "📊 Mean Absolute Error",
        f"{mae:.2f} bpm"
    )

with col3:
    st.metric(
        "📅 Training Days",
        "60 days"
    )

with col4:
    st.metric(
        "🔮 Forecast Period",
        f"{forecast_days} days"
    )

st.divider()

# Create interactive Plotly visualization
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='lines+markers',
    name='Actual Heart Rate',
    line=dict(color='#EF4444', width=2),
    marker=dict(size=6)
))

# Add forecast
forecast_future = forecast.iloc[len(df):]
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat'],
    mode='lines+markers',
    name='Forecasted Heart Rate',
    line=dict(color='#6366F1', width=2, dash='dash'),
    marker=dict(size=6)
))

# Add confidence interval
fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_upper'],
    mode='lines',
    name='Upper Confidence',
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast_future['ds'],
    y=forecast_future['yhat_lower'],
    mode='lines',
    name='Confidence Interval',
    fill='tonexty',
    fillcolor='rgba(99, 102, 241, 0.2)',
    line=dict(width=0)
))

# Update layout
fig.update_layout(
    title="Heart Rate Forecast with 95% Confidence Intervals",
    xaxis_title="Date",
    yaxis_title="Heart Rate (bpm)",
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Analysis section
st.divider()
st.header("📈 Analysis & Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Day 67 Forecast")
    st.write(f"""
    **Predicted Value:** {day_67_forecast['yhat']:.1f} bpm
    
    **Confidence Interval:**
    - Lower Bound: {day_67_forecast['yhat_lower']:.1f} bpm
    - Upper Bound: {day_67_forecast['yhat_upper']:.1f} bpm
    - Range: ±{(day_67_forecast['yhat_upper'] - day_67_forecast['yhat']):.1f} bpm
    
    This means we are 95% confident that the actual heart rate on Day 67 
    will fall between {day_67_forecast['yhat_lower']:.1f} and 
    {day_67_forecast['yhat_upper']:.1f} bpm.
    """)

with col2:
    st.subheader("🔍 Confidence Interval Explanation")
    st.write("""
    **What does it tell you?**
    
    1. **Uncertainty Quantification:** The confidence interval shows the range 
    where we expect the true value to fall with 95% probability.
    
    2. **Increasing Uncertainty:** Notice how the interval widens as we forecast 
    further into the future - this reflects growing uncertainty.
    
    3. **Model Confidence:** Narrower intervals indicate higher confidence, 
    while wider intervals suggest more variability in predictions.
    
    4. **Decision Making:** Use the intervals to assess risk and make informed 
    decisions based on the worst-case or best-case scenarios.
    """)

# Components visualization
st.divider()
st.header("🔬 Model Components")

tab1, tab2, tab3 = st.tabs(["📊 Trend", "📅 Weekly Seasonality", "📈 All Components"])

with tab1:
    fig_trend = model.plot_components(forecast)
    st.pyplot(fig_trend)

with tab2:
    st.write("The weekly seasonality component shows recurring patterns in heart rate across days of the week.")

with tab3:
    fig_all = model.plot(forecast)
    st.pyplot(fig_all)

# Model Performance
st.divider()
st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)', 
                   'Training Period', 'Forecast Period'],
        'Value': [f'{mae:.2f} bpm', 
                  f'{np.sqrt(np.mean((y_true - y_pred)**2)):.2f} bpm',
                  '60 days', 
                  f'{forecast_days} days']
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Model Quality")
    if mae < 2:
        st.success("✅ Excellent forecast accuracy (MAE < 2 bpm)")
    elif mae < 3:
        st.info("ℹ️ Good forecast accuracy (MAE < 3 bpm)")
    else:
        st.warning("⚠️ Moderate forecast accuracy (MAE ≥ 3 bpm)")
    
    st.write(f"""
    The model achieves an MAE of {mae:.2f} bpm, meaning predictions 
    deviate by an average of {mae:.2f} beats per minute from actual values.
    """)

# Download forecast data
st.divider()
st.header("💾 Export Results")

csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(
    label="📥 Download Forecast Data (CSV)",
    data=csv,
    file_name=f"heart_rate_forecast_{forecast_days}days.csv",
    mime="text/csv"
)

# Footer
st.divider()
st.caption("Built with Streamlit and Prophet | Heart Rate Forecasting Dashboard")