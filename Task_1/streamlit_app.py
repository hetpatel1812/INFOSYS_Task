import streamlit as st
import pandas as pd
from src.data_processing import load_data
from src.utils import dataset_info


st.sidebar.image("https://image.shutterstock.com/image-vector/geometric-running-man-stock-illustration-250nw-2314462233.jpg", use_container_width=True)
st.sidebar.title("📊 Weekend Project")
st.sidebar.write("Health/Fitness Data Exploration Dashboard")
st.sidebar.info("Upload CSV/JSON to get started!")


st.title("📂 Data Upload & Preview App")


uploaded_file = st.file_uploader("Upload your CSV or JSON file", type=["csv", "json"])


show_info = st.checkbox("Show Dataset Info", value=True) 
num_rows = st.slider("Select number of rows to preview:", min_value=5, max_value=50, value=10, step=5,)

if uploaded_file is not None:
    df = load_data(uploaded_file)

   
    if show_info:
        st.subheader("📑 Dataset Information")
        info = dataset_info(df)
        st.json(info)

   
    st.subheader(f"👀 First {num_rows} Rows")
    st.dataframe(df.head(num_rows))


st.markdown("""
    <style>
    /* Main app background and text */
    .stApp {
        background-color: #FFFFFF;  /* White background */
        color: #000000;  /* Black text */
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar background and text */
    section[data-testid="stSidebar"] {
        background-color: #1a1e24;  /* Dark gray/blue background */
        color: #FFFFFF;  /* White text */
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* Headings in main content */
    h1, h2, h3 {
        color: black !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #2E86C1;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #1B4F72;
        transform: scale(1.03);
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2E86C1;
        border-radius: 10px;
        padding: 12px;
        background-color: #F4F6F7;
        color: white !important;
    }

    /* Fix labels (like slider, checkboxes, uploader text) */
    div[data-testid="stSlider"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stCheckbox"] label span {
        color: #000000 !important;
        font-size: 16px;
        font-weight: 600;
    }
        
    </style>
""", unsafe_allow_html=True)

# python -m streamlit run streamlit_app.py