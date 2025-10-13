# 🕒 FitPulse Timestamp Normalization

A Streamlit dashboard for handling multi-timezone fitness data with automatic timezone detection and UTC normalization.

## Features
- **Multi-timezone processing**: Auto-detect and normalize timestamps to UTC
- **Smart detection**: Identifies timezone from user location or data patterns
- **Edge case handling**: DST transitions, travel patterns, corrupted timestamps
- **Validation dashboard**: Before/after comparison and 24-hour activity heatmaps

## Installation
```bash
pip install streamlit pandas numpy pytz plotly matplotlib seaborn
streamlit run app.py
```

## Usage

### Core Function
```python
def detect_and_normalize_timestamps(df: pd.DataFrame, user_location: str = None) -> pd.DataFrame:
    """Automatically detect timezone and normalize to UTC"""
```

### Dashboard Features
- **Data source**: Generate sample data or upload files
- **Location selection**: New York, London, Los Angeles, Tokyo, etc.
- **Timeline comparison**: Before/after normalization visualization
- **Activity heatmaps**: 24-hour patterns in UTC
- **Edge case testing**: DST, travel, corrupted data scenarios

## Supported Locations
- **New York**: America/New_York (with DST handling)
- **London**: Europe/London (DST transitions)
- **Los Angeles**: America/Los_Angeles 
- **Tokyo**: Asia/Tokyo
- **Paris**: Europe/Paris
- **Sydney**: Australia/Sydney

## Sample Scenarios

### Travel Data (NY to Tokyo)
```python
# Simulates user traveling across timezones
ny_dates = pd.date_range('2024-01-01 08:00:00', periods=24, freq='1H')
tokyo_dates = pd.date_range('2024-01-02 21:00:00', periods=24, freq='1H')
```

### DST Transition
```python
# Spring forward scenario
dst_dates = pd.date_range('2024-03-10 01:00:00', '2024-03-10 04:00:00', freq='30min')
```

## Edge Cases Handled
- **Mixed timestamp formats**: Multiple date/time formats in same dataset
- **Daylight Saving Time**: Automatic spring forward/fall back detection
- **Travel patterns**: Timezone changes during data collection
- **Corrupted data**: Invalid dates, malformed timestamps
- **Missing timezone info**: Defaults to UTC with pattern analysis

## Output Data
Normalized dataframe includes:
- `timestamp`: UTC-normalized timestamps
- `hour_utc`: Hour in UTC (0-23)
- `day_of_week`: Day name (Monday, Tuesday, etc.)
- `source_timezone`: Detected/specified timezone
- `timestamp_local`: Original local time (if available)

## File Structure
```
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md          # This file
```
