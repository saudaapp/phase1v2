import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image
import io
import yfinance as yf
import random
import json
from typing import Dict, List, Optional

# Set page configuration
st.set_page_config(
    page_title="Sauda Food Insights LLC",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme based on logo
PRIMARY_COLOR = "#2a5d4c"  # Dark green
SECONDARY_COLOR = "#8bc34a"  # Light green
ACCENT_COLOR = "#4fc3f7"  # Light blue
BG_COLOR = "#f9f8e8"  # Light cream

# Custom CSS for branding and theme
st.markdown(
    f"""
    <style>
        .main {{
            background-color: {BG_COLOR if st.session_state.get('theme', 'light') == 'light' else '#333333'};
            color: {'#000000' if st.session_state.get('theme', 'light') == 'light' else '#ffffff'};
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_COLOR};
        }}
        .stButton>button {{
            background-color: {SECONDARY_COLOR};
            color: white;
        }}
        .stSpinner>div {{
            border-color: {ACCENT_COLOR};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add theme toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
theme_button = st.sidebar.button("Toggle Theme", key="theme_toggle")
if theme_button:
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.experimental_rerun()

# Load and display logo
logo_path = "IMG_3036.png"
try:
    logo = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with