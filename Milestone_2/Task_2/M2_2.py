"""
Sleep Pattern Analysis Dashboard using Streamlit and Prophet
Task 2: Sleep Pattern Forecasting with Weekly Seasonality

Installation:
pip install streamlit prophet pandas matplotlib plotly seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Sleep Pattern Analysis",
    page_icon="😴",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("😴 Sleep Pattern Analysis Dashboard")
st.markdown("### Prophet-based Sleep Duration Forecasting with Weekly Seasonality | Task 2")
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    st.subheader("📊 Data Settings")
    data_days = st.slider("Training Days", min_value=60, max_value=180, value=90, step=30)
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=30, value=7)
    
    st.subheader("🔧 Prophet Parameters")
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    changepoint_prior = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001, 
                                   help="Higher values = more flexible trend")
    
    generate_new = st.button("🔄 Generate New Data")
    
    st.divider()
    st.header("📚 About")
    st.info("""
    This dashboard analyzes sleep patterns using Facebook Prophet.
    
    **Key Features:**
    - Weekly seasonality detection
    - Trend analysis
    - Best/worst sleep days
    - 7-day forecast
    """)

# Function to generate realistic sleep data
@st.cache_data
def generate_sleep_data(days=90, seed=42):
    """Generate realistic sleep duration data with patterns"""
    np.random.seed(seed)
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    # Base sleep duration (7.5 hours)
    base_sleep = 7.5
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        
        # Weekly pattern: More sleep on weekends
        if day_of_week == 4:  # Friday
            weekly_effect = 0.3
        elif day_of_week == 5:  # Saturday
            weekly_effect = 1.2
        elif day_of_week == 6:  # Sunday
            weekly_effect = 1.0
        elif day_of_week == 0:  # Monday (sleep debt)
            weekly_effect = -0.5
        elif day_of_week == 3:  # Thursday (mid-week fatigue)
            weekly_effect = -0.3
        else:
            weekly_effect = 0
        
        # Long-term trend (slightly decreasing sleep over time - stress accumulation)
        trend = -0.005 * i
        
        # Monthly cycle (stress/work cycles)
        monthly_cycle = 0.3 * np.sin(2 * np.pi * i / 30)
        
        # Random variation (daily noise)
        noise = np.random.normal(0, 0.3)
        
        # Occasional bad nights (10% chance)
        if np.random.random() < 0.1:
            noise -= np.random.uniform(0.5, 1.5)
        
        # Calculate total sleep duration
        sleep_hours = base_sleep + weekly_effect + trend + monthly_cycle + noise
        sleep_hours = max(4.5, min(10, sleep_hours))  # Realistic bounds
        
        data.append({
            'ds': current_date,
            'y': round(sleep_hours, 2),
            'day_name': current_date.strftime('%A')
        })
    
    return pd.DataFrame(data)

# Function to fit Prophet model
@st.cache_data
def fit_prophet_model(df, weekly_seas, cp_prior, periods=7):
    """Fit Prophet model with configurable parameters"""
    
    # Prepare data (Prophet needs only ds and y)
    df_prophet = df[['ds', 'y']].copy()
    
    # Initialize Prophet with parameters
    model = Prophet(
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=weekly_seas,
        yearly_seasonality=False,
        changepoint_prior_scale=cp_prior
    )
    
    # Fit model
    with st.spinner("🔮 Training Prophet model..."):
        model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

# Generate or load data
if 'sleep_data_seed' not in st.session_state or generate_new:
    st.session_state.sleep_data_seed = np.random.randint(0, 1000)

df = generate_sleep_data(days=data_days, seed=st.session_state.sleep_data_seed)

# Display data overview
st.header("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📅 Total Days", len(df))
with col2:
    st.metric("😴 Avg Sleep", f"{df['y'].mean():.2f} hrs")
with col3:
    st.metric("📈 Max Sleep", f"{df['y'].max():.2f} hrs")
with col4:
    st.metric("📉 Min Sleep", f"{df['y'].min():.2f} hrs")

# Data preview
with st.expander("👀 View Raw Data"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# Fit Prophet model
model, forecast = fit_prophet_model(df, weekly_seasonality, changepoint_prior, forecast_days)

# Calculate MAE
y_true = df['y'].values
y_pred = forecast.iloc[:len(df)]['yhat'].values
mae = np.mean(np.abs(y_true - y_pred))

# ==================== ANALYSIS SECTION ====================

st.header("🔍 Pattern Analysis")

# Extract weekly seasonality component
weekly_component = forecast[['ds']].copy()
day_of_week = forecast['ds'].dt.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Calculate average weekly effect for each day
weekly_seasonality_data = []
for day_idx, day_name in enumerate(day_names):
    mask = day_of_week == day_idx
    if 'weekly' in forecast.columns:
        avg_effect = forecast.loc[mask, 'weekly'].mean()
    else:
        # Approximate from component seasonality
        avg_effect = forecast.loc[mask, 'yhat'].mean() - forecast['trend'].mean()
    
    weekly_seasonality_data.append({
        'Day': day_name,
        'Day_Idx': day_idx,
        'Effect': avg_effect,
        'Avg_Sleep': df[df['ds'].dt.dayofweek == day_idx]['y'].mean() if len(df[df['ds'].dt.dayofweek == day_idx]) > 0 else 0
    })

weekly_df = pd.DataFrame(weekly_seasonality_data)

# Find best and worst sleep days
best_day = weekly_df.loc[weekly_df['Avg_Sleep'].idxmax()]
worst_day = weekly_df.loc[weekly_df['Avg_Sleep'].idxmin()]

# Trend analysis
trend_start = forecast.iloc[0]['trend']
trend_end = forecast.iloc[len(df)-1]['trend']
trend_change = trend_end - trend_start
trend_direction = "📈 Increasing" if trend_change > 0 else "📉 Decreasing"

# Display key insights
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌟 Best Sleep Day")
    st.markdown(f"""
    <div class="insight-box">
        <h2 class="big-metric">{best_day['Day']}</h2>
        <p style="font-size: 1.2rem;">Average: <strong>{best_day['Avg_Sleep']:.2f} hours</strong></p>
        <p>You sleep best on {best_day['Day']}s, likely due to weekend relaxation or reduced work stress.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 😰 Worst Sleep Day")
    st.markdown(f"""
    <div class="insight-box" style="background-color: #fff0f0; border-left-color: #ef4444;">
        <h2 class="big-metric">{worst_day['Day']}</h2>
        <p style="font-size: 1.2rem;">Average: <strong>{worst_day['Avg_Sleep']:.2f} hours</strong></p>
        <p>You sleep least on {worst_day['Day']}s, potentially due to work stress or sleep debt accumulation.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 📊 Trend Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Overall Trend", trend_direction, 
              f"{abs(trend_change):.2f} hrs over {data_days} days")
with col2:
    st.metric("Model Accuracy (MAE)", f"{mae:.2f} hrs")
with col3:
    st.metric("Sleep Variability", f"{df['y'].std():.2f} hrs")

if trend_change < -0.1:
    st.warning("⚠️ **Sleep duration is decreasing over time.** Consider reviewing your sleep hygiene and stress management strategies.")
elif trend_change > 0.1:
    st.success("✅ **Sleep duration is increasing over time.** Keep up the good habits!")
else:
    st.info("ℹ️ **Sleep duration remains relatively stable.** No significant trend detected.")

st.divider()

# ==================== VISUALIZATION SECTION ====================

st.header("📈 Visualizations")

# Tab layout for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📅 Weekly Pattern", "🔬 Components", "🔮 Forecast", "📊 Statistics"])

with tab1:
    st.subheader("Weekly Sleep Pattern")
    
    # Bar chart of average sleep by day
    fig_weekly = px.bar(
        weekly_df.sort_values('Day_Idx'),
        x='Day',
        y='Avg_Sleep',
        title="Average Sleep Duration by Day of Week",
        labels={'Avg_Sleep': 'Average Sleep (hours)', 'Day': 'Day of Week'},
        color='Avg_Sleep',
        color_continuous_scale='blues',
        text='Avg_Sleep'
    )
    
    fig_weekly.update_traces(texttemplate='%{text:.2f} hrs', textposition='outside')
    fig_weekly.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Box plot showing distribution
    st.subheader("Sleep Distribution by Day")
    df_with_day = df.copy()
    df_with_day['Day_of_Week'] = df_with_day['ds'].dt.day_name()
    df_with_day['Day_Idx'] = df_with_day['ds'].dt.dayofweek
    df_with_day = df_with_day.sort_values('Day_Idx')
    
    fig_box = px.box(
        df_with_day,
        x='Day_of_Week',
        y='y',
        title="Sleep Duration Distribution by Day",
        labels={'y': 'Sleep Duration (hours)', 'Day_of_Week': 'Day of Week'},
        color='Day_of_Week'
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("Prophet Model Components")
    
    # Plot Prophet components
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
    
    st.markdown("""
    **Component Interpretation:**
    - **Trend:** Shows the long-term direction of sleep duration (increasing/decreasing)
    - **Weekly:** Reveals which days of the week have higher/lower sleep duration
    """)

with tab3:
    st.subheader("7-Day Sleep Forecast")
    
    # Create interactive forecast plot
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines+markers',
        name='Actual Sleep',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    forecast_future = forecast.iloc[len(df):]
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines+markers',
        name='Forecasted Sleep',
        line=dict(color='#8b5cf6', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(139, 92, 246, 0.2)',
        line=dict(width=0),
        name='95% Confidence Interval'
    ))
    
    fig_forecast.update_layout(
        title="Sleep Duration: Historical Data and 7-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Sleep Duration (hours)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Display forecast table
    st.subheader("Forecast Details")
    forecast_display = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%A, %B %d, %Y')
    forecast_display.columns = ['Date', 'Predicted Sleep (hrs)', 'Lower Bound', 'Upper Bound']
    forecast_display['Predicted Sleep (hrs)'] = forecast_display['Predicted Sleep (hrs)'].round(2)
    forecast_display['Lower Bound'] = forecast_display['Lower Bound'].round(2)
    forecast_display['Upper Bound'] = forecast_display['Upper Bound'].round(2)
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Overall Statistics**")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"{df['y'].mean():.2f} hrs",
                f"{df['y'].median():.2f} hrs",
                f"{df['y'].std():.2f} hrs",
                f"{df['y'].min():.2f} hrs",
                f"{df['y'].max():.2f} hrs",
                f"{df['y'].max() - df['y'].min():.2f} hrs"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Weekly Statistics**")
        st.dataframe(
            weekly_df[['Day', 'Avg_Sleep']].rename(columns={'Avg_Sleep': 'Average Sleep (hrs)'}),
            use_container_width=True,
            hide_index=True
        )
    
    # Histogram
    st.subheader("Sleep Duration Distribution")
    fig_hist = px.histogram(
        df,
        x='y',
        nbins=30,
        title="Distribution of Sleep Duration",
        labels={'y': 'Sleep Duration (hours)', 'count': 'Frequency'},
        color_discrete_sequence=['#3b82f6']
    )
    fig_hist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ==================== ANSWERS SECTION ====================

st.header("💡 Key Findings & Answers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Question 1: Best & Worst Sleep Days")
    st.markdown(f"""
    **Question:** On which day of the week do you sleep most? Least?
    
    **Answer:**
    - 🌟 **Most Sleep:** {best_day['Day']} ({best_day['Avg_Sleep']:.2f} hours on average)
    - 😰 **Least Sleep:** {worst_day['Day']} ({worst_day['Avg_Sleep']:.2f} hours on average)
    - **Difference:** {best_day['Avg_Sleep'] - worst_day['Avg_Sleep']:.2f} hours
    
    This pattern likely reflects weekend relaxation versus weekday work stress.
    """)

with col2:
    st.markdown("### 📋 Question 2: Sleep Trend Direction")
    trend_answer = "increasing" if trend_change > 0 else "decreasing" if trend_change < -0.01 else "stable"
    st.markdown(f"""
    **Question:** Is sleep duration increasing or decreasing over time?
    
    **Answer:**
    Sleep duration is **{trend_answer}** over the {data_days}-day period.
    
    - **Trend Change:** {trend_change:+.2f} hours
    - **Rate:** {(trend_change/data_days)*7:.2f} hours per week
    
    {"⚠️ Consider implementing better sleep habits to reverse this trend." if trend_change < -0.1 else 
     "✅ Your sleep habits are improving over time!" if trend_change > 0.1 else 
     "ℹ️ Your sleep remains consistent, which is good for maintaining circadian rhythm."}
    """)

st.divider()

# ==================== DOWNLOAD SECTION ====================

st.header("💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Download forecast
    csv_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Data",
        data=csv_forecast,
        file_name=f"sleep_forecast_{forecast_days}days.csv",
        mime="text/csv"
    )

with col2:
    # Download weekly analysis
    csv_weekly = weekly_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Weekly Analysis",
        data=csv_weekly,
        file_name="sleep_weekly_pattern.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.caption("Built with Streamlit and Prophet | Sleep Pattern Analysis Dashboard")
st.caption(f"Model Accuracy (MAE): {mae:.2f} hours | Training Days: {data_days} | Forecast Period: {forecast_days} days")