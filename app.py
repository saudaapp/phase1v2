import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image
import io
import yfinance as yf
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import time
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
PRIMARY_COLOR = "#2a5d4c"  # Dark green from logo
SECONDARY_COLOR = "#8bc34a"  # Light green from logo
ACCENT_COLOR = "#4fc3f7"  # Light blue from logo
BG_COLOR = "#f9f8e8"  # Light cream background

# Custom CSS to match branding with theme toggle
st.markdown(f"""
<style>
    .reportview-container .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    .stApp {{
        background-color: {BG_COLOR};
    }}
    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 5px;
    }}
    .stButton>button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    .stSelectbox label, .stMultiselect label {{
        color: {PRIMARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        color: {PRIMARY_COLOR};
        border-radius: 4px 4px 0 0;
        border: 1px solid #ddd;
        border-bottom: none;
        padding: 10px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    .theme-toggle {{
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }}
    .download-button {{
        display: inline-block;
        padding: 10px 20px;
        background-color: {SECONDARY_COLOR};
        color: white;
        text-decoration: none;
        border-radius: 5px;
    }}
    .download-button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    @media (max-width: 768px) {{
        .stSidebar {{
            width: 100% !important;
        }}
        .chart-container {{
            height: auto !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

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
    with col2:
        st.image(logo, width=300)
except:
    st.title("Sauda Food Insights LLC")
    st.caption("Food Insights Platform")

# Function to get all available agricultural commodities
@st.cache_data(ttl=3600)
def get_available_commodities():
    base_commodities = {
        # Grains
        "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans", "ZM=F": "Soybean Meal",
        "ZL=F": "Soybean Oil", "ZO=F": "Oats", "ZR=F": "Rice", "KE=F": "KC Wheat",
        "ZG=F": "Rough Rice",
        # Fruits
        "JO=F": "Orange Juice", "CC=F": "Cocoa", "KC=F": "Coffee", "SB=F": "Sugar",
        # Meats
        "LE=F": "Live Cattle", "GF=F": "Feeder Cattle", "HE=F": "Lean Hogs",
        # Softs
        "CT=F": "Cotton", "LBS=F": "Lumber",
        # Dairy
        "DC=F": "Class III Milk", "CSC=F": "Cheese", "OJ=F": "Frozen Concentrated Orange Juice",
        # ETFs
        "MOO": "VanEck Agribusiness ETF", "DBA": "Invesco DB Agriculture Fund",
        "WEAT": "Teucrium Wheat Fund", "CORN": "Teucrium Corn Fund", "SOYB": "Teucrium Soybean Fund",
        "JJG": "iPath Bloomberg Grains Total Return ETN", "COW": "iPath Bloomberg Livestock Total Return ETN",
        "NIB": "iPath Bloomberg Cocoa Total Return ETN", "SGG": "iPath Bloomberg Sugar Total Return ETN",
        "JO": "iPath Bloomberg Coffee Total Return ETN", "BAL": "iPath Bloomberg Cotton Total Return ETN",
        # New commodities
        "PEPPER": "Black Pepper", "CINNAMON": "Cinnamon", "ALMOND": "Almonds",
        "CASHEW": "Cashews", "SALMON": "Salmon", "SHRIMP": "Shrimp"
    }
    
    valid_commodities = {}
    for ticker, name in base_commodities.items():
        try:
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_commodities[ticker] = name
        except:
            continue
    
    # Custom commodity input
    custom_ticker = st.sidebar.text_input("Enter Custom Commodity Ticker", "")
    if custom_ticker and custom_ticker not in valid_commodities:
        try:
            info = yf.Ticker(custom_ticker).info
            if 'regularMarketPrice' in info:
                valid_commodities[custom_ticker] = info.get('shortName', custom_ticker)
                st.sidebar.success(f"Added custom commodity: {custom_ticker}")
        except:
            st.sidebar.warning(f"Invalid ticker: {custom_ticker}")
    
    return valid_commodities

# Function to get real-time price data with retry logic
@st.cache_data(ttl=3600)
def get_price_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Fetching price data for {ticker} (Attempt {attempt + 1}/{max_retries})"):
                data = yf.download(ticker, period=period, timeout=10)
                if not data.empty:
                    return data
                raise Exception("Empty data received")
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error fetching data for {ticker}: {e}")
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            time.sleep(2 ** attempt)
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Function to get weather data from OpenWeatherMap API
@st.cache_data(ttl=3600)
def get_weather_data(region: str) -> pd.DataFrame:
    api_key = "fe6d802bba24d0adf54060fd14a2f5e9"
    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        weather_data = []
        for date in dates:
            params = {
                "lat": lat, "lon": lon, "dt": int(date.timestamp()),
                "appid": api_key, "units": "metric", "cnt": 24
            }
            response = requests.get("https://api.openweathermap.org/data/2.5/onecall/timemachine", params=params)
            data = response.json()
            if data.get("hourly"):
                daily_temp = np.mean([h["temp"] for h in data["hourly"]])
                daily_rain = sum(h.get("rain", {}).get("1h", 0) for h in data["hourly"])
                weather_data.append({"Date": date, "Temperature": daily_temp, "Rainfall": daily_rain})
        
        return pd.DataFrame(weather_data) if weather_data else pd.DataFrame(columns=["Date", "Temperature", "Rainfall"])
    except Exception as e:
        st.warning(f"Weather API error for {region}: {e}. Using simulated data.")
        seed = sum(ord(c) for c in region)
        np.random.seed(seed)
        temp_base = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        temp_noise = np.random.normal(0, 2, len(dates))
        rainfall = np.maximum(0, 50 + 30 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 10, len(dates)))
        return pd.DataFrame({"Date": dates, "Temperature": temp_base + temp_noise, "Rainfall": rainfall})

# Function to get crop health data from NASA POWER API
@st.cache_data(ttl=3600)
def get_crop_health_data(region: str, commodity: str) -> pd.DataFrame:
    api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    region_coords = {
        "Asia": (35.8617, 104.1954), "Africa": (8.7832, 34.5085),
        "South America": (-14.2350, -51.9253), "North America": (37.0902, -95.7129),
        "Europe": (54.5260, 15.2551), "Middle East": (29.3117, 47.4818),
        "Oceania": (-25.2744, 133.7751)
    }
    lat, lon = region_coords.get(region, (0, 0))
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        params = {
            "latitude": lat, "longitude": lon, "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"), "community": "ag", "parameters": "T2M,PRECTOTCOR,ALLSKY_SFC_SW_DWN",
            "format": "JSON", "user": "anonymous"
        }
        response = requests.get(api_url, params=params)
        data = response.json()["properties"]["parameter"]
        
        temp = np.array([data["T2M"][d.strftime("%Y%m%d")] for d in dates])
        precip = np.array([data["PRECTOTCOR"][d.strftime("%Y%m%d")] for d in dates])
        ndvi = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (temp - np.mean(temp)) / 50 + np.random.normal(0, 0.05, len(dates))
        soil_moisture = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (precip - np.mean(precip)) / 500 + np.random.normal(0, 0.03, len(dates))
        crop_stress = 30 - 15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + (np.mean(temp) - temp) / 2 + np.random.normal(0, 5, len(dates))
        
        return pd.DataFrame({
            "Date": dates, "NDVI": np.clip(ndvi, 0, 1), "Soil_Moisture": np.clip(soil_moisture, 0, 1),
            "Crop_Stress": np.clip(crop_stress, 0, 100)
        })
    except Exception as e:
        st.warning(f"Crop health API error for {region}: {e}. Using simulated data.")
        seed = sum(ord(c) for c in region) + sum(ord(c) for c in commodity)
        np.random.seed(seed)
        ndvi = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, 0.05, len(dates)) + np.random.normal(0, 0.05, len(dates))
        soil_moisture = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.03, len(dates))
        crop_stress = 30 - 15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 5, len(dates))
        return pd.DataFrame({
            "Date": dates, "NDVI": np.clip(ndvi, 0, 1), "Soil_Moisture": np.clip(soil_moisture, 0, 1),
            "Crop_Stress": np.clip(crop_stress, 0, 100)
        })

# Function to get trade flow data (simulated with potential USDA GATS integration)
@st.cache_data(ttl=3600)
def get_trade_flow_data(commodity: str, origin: str, destination: str) -> pd.DataFrame:
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in origin) + sum(ord(c) for c in destination)
        np.random.seed(seed)
        base_volume = 1000 + (sum(ord(c) for c in commodity) % 5000)
        volume = np.maximum(0, base_volume + base_volume * 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, base_volume * 0.2, len(dates)) + np.random.normal(0, base_volume * 0.1, len(dates)))
        price = np.maximum(0, 100 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.linspace(0, 30, len(dates)) + np.random.normal(0, 10, len(dates)))
        
        return pd.DataFrame({"Date": dates, "Volume": volume, "Price": price})
    except Exception as e:
        st.error(f"Trade flow data error: {e}")
        return pd.DataFrame({"Date": [], "Volume": [], "Price": []})

# Function to generate market opportunities
def generate_market_opportunities(commodity: str, region: str, user_type: str) -> List[Dict]:
    regions = {
        "Asia": ["China", "India", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines"],
        "Africa": ["Egypt", "South Africa", "Kenya", "Nigeria", "Morocco", "Ethiopia", "Tanzania"],
        "South America": ["Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador", "Uruguay"],
        "North America": ["USA", "Canada", "Mexico"],
        "Europe": ["France", "Germany", "Italy", "Spain", "Netherlands", "Poland", "UK"],
        "Middle East": ["UAE", "Saudi Arabia", "Turkey", "Israel", "Iran", "Jordan"],
        "Oceania": ["Australia", "New Zealand"]
    }
    
    other_regions = [r for r in regions.keys() if r != region]
    diversification_regions = random.sample(other_regions, min(3, len(other_regions)))
    opportunities = []
    
    seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in region)
    random.seed(seed)
    
    if user_type == "Buyer":
        current_source = random.choice(regions[region])
        for div_region in diversification_regions:
            div_country = random.choice(regions[div_region])
            rationale = random.choice([
                f"Production in {div_country} has increased by 15% year-over-year, creating a surplus and favorable pricing",
                f"New trade agreement between {region} and {div_region} reduces import duties on {commodity} by 8%",
                f"Satellite imagery shows optimal crop health in {div_country}'s {commodity} regions"
            ])
            savings_percent = random.randint(5, 25)
            contacts = generate_contacts(div_country, 2)
            opportunities.append({
                "title": f"Diversify {commodity} sourcing to {div_country}",
                "description": f"Current source: {current_source} (producing region). Recommended alternative: {div_country} (producing region).",
                "rationale": rationale,
                "potential_impact": f"Potential cost savings of {savings_percent}%",
                "implementation_timeline": f"{random.randint(1, 3)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": contacts
            })
    else:  # Seller
        current_market = random.choice(regions[region])
        for div_region in diversification_regions:
            div_country = random.choice(regions[div_region])
            rationale = random.choice([
                f"Market analysis shows {div_country} has a {commodity} supply gap of approximately 15,000 tons annually",
                f"{div_country} has experienced a 23% increase in {commodity} consumption over the past 3 years",
                f"New shipping routes to {div_country} reduce logistics costs by 18%"
            ])
            revenue_percent = random.randint(10, 30)
            contacts = generate_contacts(div_country, 2)
            opportunities.append({
                "title": f"Expand {commodity} exports to {div_country}",
                "description": f"Current market: {current_market} (destination). Recommended new market: {div_country} (destination).",
                "rationale": rationale,
                "potential_impact": f"Potential revenue increase of {revenue_percent}%",
                "implementation_timeline": f"{random.randint(2, 6)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": contacts
            })
    
    return opportunities

# Function to generate contact recommendations
def generate_contacts(country: str, num_contacts: int = 3) -> List[Dict]:
    seed = sum(ord(c) for c in country)
    random.seed(seed)
    company_patterns = [
        "{country} {commodity} Traders", "{country} Agricultural Exports", "{commodity} Distributors of {country}"
    ]
    contact_names = {
        "China": ["Li Wei", "Zhang Min"], "India": ["Raj Sharma", "Priya Patel"],
        "Vietnam": ["Nguyen Van", "Tran Thi"], "Thailand": ["Somchai S.", "Suchada K."],
        "Indonesia": ["Budi Santoso", "Siti Aminah"], "Malaysia": ["Ahmad Bin", "Siti Binti"],
        "Philippines": ["Juan Reyes", "Maria Santos"], "Egypt": ["Ahmed Hassan", "Fatima Ali"],
        "South Africa": ["John van der Merwe", "Sarah Nkosi"], "Kenya": ["James Kamau", "Grace Wanjiku"],
        "Nigeria": ["Oluwaseun A.", "Chinwe O."], "Morocco": ["Youssef El", "Fatima Ben"],
        "Ethiopia": ["Abebe T.", "Tigist M."], "Tanzania": ["Emmanuel M.", "Joyce K."],
        "Brazil": ["Carlos Silva", "Ana Santos"], "Argentina": ["Juan Gonzalez", "Maria Rodriguez"],
        "Chile": ["Alejandro Munoz", "Camila Vargas"], "Colombia": ["Andres Gomez", "Catalina Herrera"],
        "Peru": ["Jose Flores", "Carmen Vega"], "Ecuador": ["Francisco Mendez", "Elena Suarez"],
        "Uruguay": ["Martin Perez", "Lucia Rodriguez"], "USA": ["Michael Johnson", "Jennifer Smith"],
        "Canada": ["James Wilson", "Emily Thompson"], "Mexico": ["Alejandro Hernandez", "Sofia Garcia"],
        "France": ["Jean Dupont", "Marie Dubois"], "Germany": ["Thomas Müller", "Anna Schmidt"],
        "Italy": ["Marco Rossi", "Giulia Ricci"], "Spain": ["Javier Garcia", "Carmen Martinez"],
        "Netherlands": ["Jan de Vries", "Anna van der Berg"], "Poland": ["Piotr Kowalski", "Anna Nowak"],
        "UK": ["James Smith", "Emma Jones"], "UAE": ["Mohammed Al", "Fatima Al"],
        "Saudi Arabia": ["Abdullah Al", "Nora Al"], "Turkey": ["Mehmet Yilmaz", "Ayşe Kaya"],
        "Israel": ["David Cohen", "Sarah Levy"], "Iran": ["Ali Hosseini", "Zahra Ahmadi"],
        "Jordan": ["Omar Al", "Lina Al"], "Australia": ["James Smith", "Sarah Johnson"],
        "New Zealand": ["William Taylor", "Charlotte Anderson"]
    }
    default_names = ["John Smith", "Jane Doe"]
    names = contact_names.get(country, default_names)
    contacts = []
    commodities = ["Rice", "Wheat", "Corn", "Soybeans", "Coffee", "Sugar", "Cotton", "Cocoa", "Fruits", "Vegetables", "Pepper", "Almonds", "Salmon", "Shrimp"]
    
    for i in range(num_contacts):
        name = random.choice(names)
        commodity = random.choice(commodities)
        company = random.choice(company_patterns).format(country=country, commodity=commodity)
        position = random.choice(["Procurement Manager", "Supply Chain Director", "Trading Manager"])
        email = f"{name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com"
        phone = f"+{random.randint(1, 999)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
        contacts.append({
            "name": name, "company": company, "position": position, "location": country,
            "email": email, "phone": phone, "contact": f"{name}, {position}"
        })
    return contacts

# Function to create HTML report
def create_html_report(opportunity: Dict, commodity: str, region: str, user_type: str, price_chart: str, weather_chart: str, crop_health_chart: str, trade_flow_chart: str) -> str:
    toc = """
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Opportunity Summary</a></li>
            <li><a href="#analysis">Market Analysis</a>
                <ul>
                    <li><a href="#price">Price Trends</a></li>
                    <li><a href="#weather">Weather Impact</a></li>
                    <li><a href="#crop">Crop Health</a></li>
                    <li><a href="#trade">Trade Flows</a></li>
                </ul>
            </li>
            <li><a href="#actions">Recommended Actions</a></li>
            <li><a href="#contacts">Contact Recommendations</a></li>
        </ul>
    </div>
    """
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .logo {{ max-width: 200px; margin-bottom: 10px; }}
            h1 {{ color: #2a5d4c; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #2a5d4c; margin-top: 20px; }}
            .highlight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #8bc34a; margin: 20px 0; }}
            .toc {{ background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd; margin-bottom: 20px; }}
            .toc ul {{ list-style-type: none; padding-left: 20px; }}
            .toc a {{ color: #2a5d4c; text-decoration: none; }}
            .toc a:hover {{ text-decoration: underline; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .chart-container {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #666; }}
            a[name] {{ padding-top: 60px; margin-top: -60px; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="header">
            <img src="data:image/png;base64,{get_image_base64('IMG_3036.png')}" alt="Sauda Food Insights LLC" class="logo">
            <h1>Market Opportunity Report: {commodity} in {region}</h1>
            <p>Generated for: <strong>{user_type}</strong> | Date: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        {toc}
        <div class="highlight" id="summary">
            <h2>Opportunity Summary</h2>
            <p><strong>{opportunity['title']}</strong></p>
            <p>{opportunity['description']}</p>
            <p><strong>Rationale:</strong> {opportunity['rationale']}</p>
            <p><strong>Potential Impact:</strong> {opportunity['potential_impact']}</p>
            <p><strong>Implementation Timeline:</strong> {opportunity['implementation_timeline']}</p>
            <p><strong>Risk Level:</strong> {opportunity['risk_level']}</p>
        </div>
        <div id="analysis">
            <h2>Market Analysis</h2>
            <h3 id="price">Price Trends</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{price_chart}" alt="Price Trends" style="width:100%">
            </div>
            <p>The chart above shows the price trends for {commodity} over the past 24 months. 
            This data indicates {get_price_analysis(price_data, user_type, commodity)}.</p>
            <h3 id="weather">Weather Impact</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{weather_chart}" alt="Weather Impact" style="width:100%">
            </div>
            <p>Weather conditions in key growing regions for {commodity} in {region} show {get_weather_analysis(weather_data, region, commodity, user_type)}.</p>
            <h3 id="crop">Crop Health Assessment</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{crop_health_chart}" alt="Crop Health" style="width:100%">
            </div>
            <p>Satellite imagery analysis of {commodity} growing regions in {region} indicates {get_crop_health_analysis(crop_health_data, region, commodity, user_type)}.</p>
            <h3 id="trade">Trade Flows</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{trade_flow_chart}" alt="Trade Flows" style="width:100%">
            </div>
            <p>Global trade flow analysis for {commodity} shows {get_trade_flow_analysis(trade_data, commodity, region, user_type)}.</p>
        </div>
        <div id="actions">
            <h2>Recommended Actions</h2>
            <ul>
                <li>Initiate contact with recommended partners in {opportunity['description'].split(': ')[1]}</li>
                <li>Conduct detailed cost-benefit analysis based on current market conditions</li>
                <li>Develop implementation timeline aligned with seasonal market patterns</li>
                <li>Consider pilot program to test market response before full-scale implementation</li>
                <li>Monitor key indicators (price trends, crop health, weather patterns) for optimal timing</li>
            </ul>
        </div>
        <div id="contacts">
            <h2>Contact Recommendations</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Company</th>
                    <th>Position</th>
                    <th>Location</th>
                    <th>Contact</th>
                </tr>
                {generate_contact_table_rows(opportunity['contacts'])}
            </table>
        </div>
        <div class="footer">
            <p>This report is generated by Sauda Food Insights LLC. Recommendations are based on analysis of market data, weather patterns, crop conditions, and trade flows.</p>
            <p>© {datetime.now().year} Sauda Food Insights LLC. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    return html_content

# Helper functions
def generate_contact_table_rows(contacts: List[Dict]) -> str:
    return "".join(f"<tr><td>{c['name']}</td><td>{c['company']}</td><td>{c['position']}</td><td>{c['location']}</td><td>{c['email']}<br>{c['phone']}</td></tr>" for c in contacts)

def get_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def get_price_analysis(price_data: pd.DataFrame, user_type: str, commodity: str) -> str:
    if price_data.empty:
        return "insufficient data to determine price trends."
    recent_period = min(60, len(price_data) // 4)
    recent_prices = price_data['Close'].iloc[-recent_period:]
    start_price = recent_prices.iloc[0]
    end_price = recent_prices.iloc[-1]
    percent_change = (end_price - start_price) / start_price * 100
    
    trend = "stable"
    if percent_change > 10:
        trend = "strong upward"
    elif percent_change > 3:
        trend = "moderate upward"
    elif percent_change < -10:
        trend = "strong downward"
    elif percent_change < -3:
        trend = "moderate downward"
    
    if user_type == "Buyer":
        return f"{trend} trends, suggesting buyers should {'secure forward contracts' if percent_change > 3 else 'monitor for buying opportunities'} for {commodity}."
    else:
        return f"{trend} trends, indicating sellers should {'optimize sales timing' if percent_change > 3 else 'focus on value differentiation'} for {commodity}."

def get_weather_analysis(weather_data: pd.DataFrame, region: str, commodity: str, user_type: str) -> str:
    if weather_data.empty:
        return "insufficient weather data available."
    recent_temp = weather_data['Temperature'].iloc[-3:].mean()
    historical_temp = weather_data['Temperature'].iloc[:-3].mean()
    temp_anomaly = recent_temp - historical_temp
    recent_rain = weather_data['Rainfall'].iloc[-3:].mean()
    historical_rain = weather_data['Rainfall'].iloc[:-3].mean()
    rain_anomaly = recent_rain - historical_rain
    
    impact = "favorable conditions" if abs(temp_anomaly) < 1.5 and abs(rain_anomaly) < 7 else "potential stress" if abs(temp_anomaly) < 3 or abs(rain_anomaly) < 15 else "significant stress"
    return f"{impact} in {region}, impacting {commodity} production. {user_type.lower()}s should {'monitor supply' if user_type == 'Buyer' else 'highlight quality advantages'} accordingly."

def get_crop_health_analysis(crop_health_data: pd.DataFrame, region: str, commodity: str, user_type: str) -> str:
    if crop_health_data.empty:
        return "insufficient crop health data available."
    recent_ndvi = crop_health_data['NDVI'].iloc[-3:].mean()
    historical_ndvi = crop_health_data['NDVI'].iloc[:-3].mean()
    ndvi_trend = recent_ndvi - historical_ndvi
    recent_stress = crop_health_data['Crop_Stress'].iloc[-3:].mean()
    
    health = "healthy" if ndvi_trend > 0 and recent_stress < 30 else "variable" if ndvi_trend > -0.02 and recent_stress < 50 else "stressed"
    return f"{health} conditions in {region} for {commodity}. {user_type.lower()}s should {'diversify sourcing' if user_type == 'Buyer' else 'emphasize quality'} based on this data."

def get_trade_flow_analysis(trade_data: pd.DataFrame, commodity: str, region: str, user_type: str) -> str:
    if trade_data.empty:
        return "insufficient trade flow data available."
    recent_volume = trade_data['Volume'].iloc[-3:].mean()
    historical_volume = trade_data['Volume'].iloc[:-3].mean()
    volume_trend = (recent_volume - historical_volume) / historical_volume * 100 if historical_volume else 0
    recent_price = trade_data['Price'].iloc[-3:].mean()
    historical_price = trade_data['Price'].iloc[:-3].mean()
    price_trend = (recent_price - historical_price) / historical_price * 100 if historical_price else 0
    
    flow = "stable" if abs(volume_trend) < 5 and abs(price_trend) < 3 else "increasing" if volume_trend > 10 else "decreasing"
    if user_type == "Buyer":
        return f"{flow} opportunities from producing regions to {region} for sourcing {commodity}."
    else:
        return f"{flow} potential from {region} to importing regions for exporting {commodity}."

def get_html_download_link(html_content: str, filename: str) -> str:
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}.html" class="download-button">Download Report</a>'

def create_chart_image(fig: go.Figure) -> str:
    img_bytes = fig.to_image(format="png", width=800, height=400)
    return base64.b64encode(img_bytes).decode()

# Main application layout
def main():
    st.sidebar.header("User Settings")
    user_type = st.sidebar.radio("Select User Type", ["Buyer", "Seller"])
    
    st.sidebar.header("Commodity Selection")
    available_commodities = get_available_commodities()
    if not available_commodities:
        available_commodities = {"ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans"}
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity", options=list(available_commodities.keys()),
        format_func=lambda x: available_commodities[x]
    )
    selected_commodity_name = available_commodities[selected_commodity]
    
    st.sidebar.header("Region Selection")
    selected_region = st.sidebar.selectbox(
        "Select Region", options=["Asia", "Africa", "South America", "North America", "Europe", "Middle East", "Oceania"]
    )
    
    st.sidebar.header("Analysis Type")
    analysis_types = {k: st.sidebar.checkbox(k, value=True) for k in ["Price Analysis", "Weather Impact", "Crop Health", "Trade Flows"]}
    time_range = st.sidebar.selectbox("Time Range", ["1y", "2y", "5y"], index=2)
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully")
    
    st.title(f"{selected_commodity_name} Market Intelligence")
    st.subheader(f"Region: {selected_region} | View: {user_type}")
    st.caption(f"Note: For Buyers, 'source' refers to the origin of goods; for Sellers, 'market' refers to the destination.")
    
    tab1, tab2, tab3 = st.tabs(["Market Analysis", "Opportunities", "Contacts"])
    
    with tab1:
        st.header("Market Analysis Dashboard")
        if analysis_types["Price Analysis"]:
            st.subheader("Price Analysis")
            with st.spinner("Loading price data..."):
                price_data = get_price_data(selected_commodity, period=time_range)
            if not price_data.empty:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Close Price', line=dict(color=PRIMARY_COLOR, width=2)))
                fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'].rolling(50).mean(), mode='lines', name='50-Day MA', line=dict(color=SECONDARY_COLOR, width=1.5, dash='dash')))
                fig_price.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'].rolling(200).mean(), mode='lines', name='200-Day MA', line=dict(color=ACCENT_COLOR, width=1.5, dash='dot')))
                fig_price.update_layout(
                    title=f"{selected_commodity_name} Price Trends", xaxis_title="Date", yaxis_title="Price",
                    dragmode='zoom', legend=dict(orientation="h", y=1.1, x=1), template="plotly_white", height=400
                )
                st.plotly_chart(fig_price, use_container_width=True)
                st.markdown(f"*Current Price:* ${price_data['Close'].iloc[-1]:.2f}\n{get_price_analysis(price_data, user_type, selected_commodity_name)}")
            else:
                st.warning(f"No price data available for {selected_commodity_name}")
        
        if analysis_types["Weather Impact"]:
            st.subheader("Weather Impact Analysis")
            with st.spinner("Loading weather data..."):
                weather_data = get_weather_data(selected_region)
            if not weather_data.empty:
                fig_weather = go.Figure()
                fig_weather.add_trace(go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], mode='lines', name='Temperature (°C)', line=dict(color='red', width=2)))
                fig_weather.add_trace(go.Bar(x=weather_data['Date'], y=weather_data['Rainfall'], name='Rainfall (mm)', marker=dict(color='blue', opacity=0.6)))
                fig_weather.update_layout(
                    title=f"Weather Patterns in {selected_region} Growing Regions", xaxis_title="Date",
                    yaxis=dict(title="Temperature (°C)", titlefont=dict(color="red"), tickfont=dict(color="red")),
                    yaxis2=dict(title="Rainfall (mm)", titlefont=dict(color="blue"), tickfont=dict(color="blue"), overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.1, x=1), template="plotly_white", height=400
                )
                st.plotly_chart(fig_weather, use_container_width=True)
                st.markdown(get_weather_analysis(weather_data, selected_region, selected_commodity_name, user_type))
            else:
                st.warning(f"No weather data available for {selected_region}")
        
        if analysis_types["Crop Health"]:
            st.subheader("Crop Health Monitoring")
            with st.spinner("Loading crop health data..."):
                crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)
            if not crop_health_data.empty:
                fig_crop = go.Figure()
                fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['NDVI'], mode='lines', name='NDVI', line=dict(color='green', width=2)))
                fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['Soil_Moisture'], mode='lines', name='Soil Moisture', line=dict(color='blue', width=2)))
                fig_crop.add_trace(go.Scatter(x=crop_health_data['Date'], y=crop_health_data['Crop_Stress']/100, mode='lines', name='Crop Stress Index', line=dict(color='red', width=2)))
                fig_crop.update_layout(
                    title=f"{selected_commodity_name} Crop Health Indicators in {selected_region}", xaxis_title="Date", yaxis_title="Index Value",
                    legend=dict(orientation="h", y=1.1, x=1), template="plotly_white", height=400
                )
                st.plotly_chart(fig_crop, use_container_width=True)
                st.markdown(get_crop_health_analysis(crop_health_data, selected_region, selected_commodity_name, user_type))
            else:
                st.warning(f"No crop health data available for {selected_region}")
        
        if analysis_types["Trade Flows"]:
            st.subheader("Global Trade Flow Analysis")
            origin = random.choice(["Asia", "Africa", "South America"]) if user_type == "Buyer" else selected_region
            destination = selected_region if user_type == "Buyer" else random.choice(["North America", "Europe", "Middle East"])
            with st.spinner("Loading trade flow data..."):
                trade_data = get_trade_flow_data(selected_commodity_name, origin, destination)
            if not trade_data.empty:
                fig_trade = go.Figure()
                fig_trade.add_trace(go.Bar(x=trade_data['Date'], y=trade_data['Volume'], name='Volume (MT)', marker=dict(color=SECONDARY_COLOR)))
                fig_trade.add_trace(go.Scatter(x=trade_data['Date'], y=trade_data['Price'], mode='lines', name='Price', line=dict(color=PRIMARY_COLOR, width=2), yaxis="y2"))
                fig_trade.update_layout(
                    title=f"{selected_commodity_name} Trade Flows: {origin} to {destination}", xaxis_title="Date",
                    yaxis=dict(title="Volume (Metric Tons)", titlefont=dict(color=SECONDARY_COLOR), tickfont=dict(color=SECONDARY_COLOR)),
                    yaxis2=dict(title="Price (USD/MT)", titlefont=dict(color=PRIMARY_COLOR), tickfont=dict(color=PRIMARY_COLOR), overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.1, x=1), template="plotly_white", height=400
                )
                st.plotly_chart(fig_trade, use_container_width=True)
                st.markdown(get_trade_flow_analysis(trade_data, selected_commodity_name, selected_region, user_type))
            else:
                st.warning(f"No trade flow data available for {selected_commodity_name}")
    
    with tab2:
        st.header("Market Opportunities")
        with st.spinner("Generating opportunities..."):
            opportunities = generate_market_opportunities(selected_commodity_name, selected_region, user_type)
        for i, opportunity in enumerate(opportunities):
            with st.expander(f"Opportunity {i+1}: {opportunity['title']}", expanded=(i==0)):
                st.markdown(f"*Description:* {opportunity['description']}")
                st.markdown(f"*Rationale:* {opportunity['rationale']}")
                st.markdown(f"*Potential Impact:* {opportunity['potential_impact']}")
                st.markdown(f"*Implementation Timeline:* {opportunity['implementation_timeline']}")
                st.markdown(f"*Risk Level:* {opportunity['risk_level']}")
                
                # Create charts for the report
                price_data = get_price_data(selected_commodity, period=time_range)[-24:]
                fig_price = go.Figure(data=[go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', line=dict(color=PRIMARY_COLOR))])
                price_chart = create_chart_image(fig_price)
                
                weather_data = get_weather_data(selected_region)[-24:]
                fig_weather = go.Figure(data=[go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], mode='lines', line=dict(color='red'))])
                weather_chart = create_chart_image(fig_weather)
                
                crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)[-24:]
                fig_crop = go.Figure(data=[go.Scatter(x=crop_health_data['Date'], y=crop_health_data['NDVI'], mode='lines', line=dict(color='green'))])
                crop_chart = create_chart_image(fig_crop)
                
                trade_data = get_trade_flow_data(selected_commodity_name, origin, destination)[-24:]
                fig_trade = go.Figure(data=[go.Bar(x=trade_data['Date'], y=trade_data['Volume'], marker=dict(color=SECONDARY_COLOR))])
                trade_chart = create_chart_image(fig_trade)
                
                html_content = create_html_report(opportunity, selected_commodity_name, selected_region, user_type, price_chart, weather_chart, crop_chart, trade_chart)
                if st.button("Preview Report", key=f"preview_{i}"):
                    st.markdown(f"<iframe srcdoc='{html_content}' style='width:100%; height:600px; border:none;'></iframe>", unsafe_allow_html=True)
                st.markdown(get_html_download_link(html_content, f"Sauda_{selected_commodity_name}{opportunity['title'].replace(' ', '')}"), unsafe_allow_html=True)
                
                st.subheader("Recommended Contacts")
                for contact in opportunity['contacts']:
                    st.markdown(f"*{contact['name']}*  {contact['position']} at {contact['company']}  📍 {contact['location']}  📧 {contact['email']}  📞 {contact['phone']}")
    
    with tab3:
        st.header("Contact Recommendations")
        if user_type == "Buyer":
            contact_regions = ["Asia", "South America", "Africa"] if selected_region in ["North America", "Europe"] else ["South America", "Asia", "North America"]
        else:
            contact_regions = ["North America", "Europe", "Middle East"] if selected_region in ["Asia", "South America", "Africa"] else ["Asia", "Middle East", "Europe"]
        for region in contact_regions:
            st.subheader(f"{region} Contacts")
            country = random.choice(["China", "India"] if region == "Asia" else ["Egypt", "South Africa"] if region == "Africa" else ["Brazil", "Argentina"] if region == "South America" else ["USA", "Canada"] if region == "North America" else ["France", "Germany"] if region == "Europe" else ["UAE", "Saudi Arabia"])
            contacts = generate_contacts(country, 3)
            cols = st.columns(3)
            for i, contact in enumerate(contacts):
                with cols[i]:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:15px; height:200px;">
                        <h3 style="color:{PRIMARY_COLOR};">{contact['name']}</h3>
                        <p><strong>{contact['position']}</strong><br>{contact['company']}</p>
                        <p>📍 {contact['location']}</p>
                        <p>📧 {contact['email']}<br>📞 {contact['phone']}</p>
                    </div>
                    """, unsafe_allow_html=True)

if _name_ == "_main_":
    main()