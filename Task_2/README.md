# 📊 FitPulse Data Format Mastery

A Streamlit dashboard for loading and analyzing multiple fitness data formats with robust error handling.

## Features
- **Multi-format loader**: CSV, JSON, Excel with auto-detection
- **Error handling**: Corrupted files, encoding issues, missing data
- **Format analysis**: Performance comparison and recommendations
- **Interactive dashboard**: Real-time loading metrics and visualizations

## Installation
```bash
pip install streamlit pandas numpy plotly openpyxl
streamlit run app.py
```

## Usage

### Core Function
```python
def load_fitness_data(file_path: str, file_type: str = 'auto') -> pd.DataFrame:
    """Load fitness data from CSV, JSON, or Excel files"""
```

### Dashboard Tabs
1. **Data Loading**: Upload files or generate sample data
2. **Format Analysis**: Compare file sizes and loading times
3. **Error Testing**: Test edge cases and error handling
4. **Visualizations**: View heart rate, steps, and sleep data

## Supported Data Columns
- **Timestamp**: `timestamp`, `time`, `datetime`, `date`
- **Heart Rate**: `heart_rate`, `hr`, `bpm`, `pulse` 
- **Steps**: `step_count`, `steps`, `step`
- **Sleep**: `sleep_stage`, `sleep`, `sleep_state`

## Sample Data Format

**CSV:**
```csv
timestamp,heart_rate,step_count,sleep_stage
2024-01-01 08:00:00,75,45,awake
```

**JSON:**
```json
{
  "fitness_data": [
    {"timestamp": "2024-01-01 08:00:00", "heart_rate": 75}
  ]
}
```

## Error Handling
- Auto-detects file formats and encodings
- Handles empty files and corrupted data
- Maps different column naming conventions
- Validates data ranges (heart rate: 20-250 BPM)

## File Structure
```
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── sample_data/        # Generated test files
```
