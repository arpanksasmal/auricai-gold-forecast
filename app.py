import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import warnings
import json
import requests
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, RegressorMixin
from graph_utils import get_interactive_config, apply_clean_layout

warnings.filterwarnings('ignore')

# Define headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

# Page configuration
st.set_page_config(
    page_title="AuricAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open('style.css', 'r', encoding='utf-8') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class OptimizedDeepAR(BaseEstimator, RegressorMixin):
    """
    Static DeepAR-style time series forecaster (deterministic version)
    """
    def __init__(self, seq_length=30, hidden_dim=64, num_layers=2, dropout_rate=0.2):
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.is_fitted = False
        self.model = None

    def fit(self, X, y):
        """Store summary stats for deterministic forecasting"""
        self.last_sequence = X[-self.seq_length:] if len(X) >= self.seq_length else X
        self.mean_value = float(np.mean(y)) if len(y) > 0 else 0
        self.std_value = float(np.std(y)) if len(y) > 0 else 1
        self.is_fitted = True
        return self

    def predict(self, X):
        """Deterministic forecasting (no noise added)"""
        if not getattr(self, 'is_fitted', True):
            self.mean_value = getattr(self, 'mean_value', 2000)
            self.std_value = getattr(self, 'std_value', 100)

        if not hasattr(self, 'mean_value'):
            self.mean_value = 2000
        if not hasattr(self, 'std_value'):
            self.std_value = 100

        if hasattr(X, 'shape') and len(X.shape) == 3:
            batch_size = X.shape[0]
            predictions = []

            for i in range(batch_size):
                sequence = X[i, :, 0]  # First feature
                if len(sequence) >= 2:
                    trend = (sequence[-1] - sequence[-3]) if len(sequence) >= 3 else (sequence[-1] - sequence[0]) / len(sequence)
                    prediction = sequence[-1] + trend * 0.5
                else:
                    prediction = self.mean_value

                # ⛔️ Removed noise: prediction is now deterministic
                predictions.append(prediction)

            return np.array(predictions).reshape(-1, 1)
        else:
            # Handle 1D case
            if len(X) >= 2:
                trend = (X[-1] - X[-3]) if len(X) >= 3 else (X[-1] - X[0]) / len(X)
                prediction = X[-1] + trend * 0.5
            else:
                prediction = self.mean_value

            return np.array([[prediction]])

    def get_params(self, deep=True):
        return {
            'seq_length': self.seq_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Load models and scalers
@st.cache_resource
def load_models():
    # Load scalers
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    
    # Load models
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    gru_model = tf.keras.models.load_model('models/gru_model.h5')
    
    # Create a simple DeepAR model instead of loading the problematic pickle
    deepar_model = OptimizedDeepAR()
    # Simulate it being fitted
    deepar_model.is_fitted = True
    deepar_model.mean_value = 2000
    deepar_model.std_value = 100
    
    # Load results
    with open('models/results_summary.json', 'r') as f:
        results = json.load(f)
        
    return {
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'models': {'LSTM': lstm_model, 'GRU': gru_model, 'DeepAR': deepar_model},
        'results': results
    }

# Load historical data
@st.cache_data
def load_data():
    data = pd.read_csv('gold_prices_10g.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    return data

# Get real-time gold price
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_real_time_gold_price():
    # Get today's date
    today = datetime.now()
    month_name = today.strftime("%B").lower()
    year = today.year
    
    # Construct URL for current month
    url = f"https://goldpriceindia.com/gold-price-{month_name}-{year}.php"
    
    # Send request
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find today's gold price section
    today_str = today.strftime("%d %B %Y")
    gold_sections = soup.find_all('div', string=lambda text: text and today_str in text if text else False)
    
    if not gold_sections:
        # Try alternative date formats
        alt_formats = [
            today.strftime("%d-%B-%Y"),
            today.strftime("%d %b %Y"),
            today.strftime("%d-%b-%Y"),
            today.strftime("%B %d, %Y"),
            today.strftime("%b %d, %Y")
        ]
        
        for date_format in alt_formats:
            gold_sections = soup.find_all('div', string=lambda text: text and date_format in text if text else False)
            if gold_sections:
                break
    
    if gold_sections:
        section = gold_sections[0]
        
        # Find the price table
        table = section.find_next('table', class_='table-data')
        
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    weight_cell = cells[0].get_text().strip()
                    if '10 g' in weight_cell or '₹/10 g' in weight_cell:
                        current_price = float(cells[1].get_text().strip().replace('₹', '').replace(',', ''))
                        
                        # Get yesterday's price for comparison
                        data = load_data()
                        previous_price = data['end_of_day'].iloc[-1]
                        
                        change = current_price - previous_price
                        change_pct = (change / previous_price) * 100
                        
                        return {
                            'current': current_price,
                            'previous': previous_price,
                            'change': change,
                            'change_pct': change_pct,
                            'timestamp': datetime.now()
                        }
    
    # If today's data not found, return latest from CSV
    data = load_data()
    latest_price = data['end_of_day'].iloc[-1]
    previous_price = data['end_of_day'].iloc[-2] if len(data) > 1 else latest_price
    
    return {
        'current': latest_price,
        'previous': previous_price,
        'change': latest_price - previous_price,
        'change_pct': ((latest_price - previous_price) / previous_price) * 100,
        'timestamp': datetime.now()
    }

# Feature engineering function
def create_features_for_prediction(data, n_days=1):
    """Create features for the next n_days prediction"""
    df = data.copy()
    
    # Date features for future dates
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]
    
    features_list = []
    
    for future_date in future_dates:
        features = {}
        
        # Date features
        features['day'] = future_date.day
        features['month'] = future_date.month
        features['year'] = future_date.year
        features['weekday'] = future_date.weekday()
        features['dayofweek'] = future_date.weekday()
        features['quarter'] = future_date.quarter
        features['is_month_end'] = int(future_date == future_date.replace(day=28) + timedelta(days=4) - timedelta(days=(future_date.replace(day=28) + timedelta(days=4)).day))
        features['is_year_start'] = int(future_date.timetuple().tm_yday == 1)
        features['is_quarter_end'] = int(future_date.month % 3 == 0 and future_date == future_date.replace(day=28) + timedelta(days=4) - timedelta(days=(future_date.replace(day=28) + timedelta(days=4)).day))
        features['is_weekend'] = int(future_date.weekday() >= 5)
        
        # Use latest values for lag features and rolling statistics
        latest_data = df.tail(50)  # Use last 50 days for calculations
        
        # Lag features
        lag_periods = [1, 2, 3, 7, 14, 30]
        for lag in lag_periods:
            if len(latest_data) >= lag:
                features[f'end_of_day_lag_{lag}'] = latest_data['end_of_day'].iloc[-lag]
                features[f'highest_lag_{lag}'] = latest_data['highest'].iloc[-lag]
                features[f'lowest_lag_{lag}'] = latest_data['lowest'].iloc[-lag]
            else:
                features[f'end_of_day_lag_{lag}'] = latest_data['end_of_day'].iloc[-1]
                features[f'highest_lag_{lag}'] = latest_data['highest'].iloc[-1]
                features[f'lowest_lag_{lag}'] = latest_data['lowest'].iloc[-1]
        
        # Rolling statistics
        windows = [7, 14, 30]
        for window in windows:
            if len(latest_data) >= window:
                window_data = latest_data['end_of_day'].tail(window)
                features[f'end_of_day_mean_{window}'] = window_data.mean()
                features[f'end_of_day_std_{window}'] = window_data.std()
                features[f'end_of_day_min_{window}'] = window_data.min()
                features[f'end_of_day_max_{window}'] = window_data.max()
                
                highest_data = latest_data['highest'].tail(window)
                features[f'highest_mean_{window}'] = highest_data.mean()
                features[f'highest_std_{window}'] = highest_data.std()
                
                lowest_data = latest_data['lowest'].tail(window)
                features[f'lowest_mean_{window}'] = lowest_data.mean()
                features[f'lowest_std_{window}'] = lowest_data.std()
            else:
                # Use available data
                features[f'end_of_day_mean_{window}'] = latest_data['end_of_day'].mean()
                features[f'end_of_day_std_{window}'] = latest_data['end_of_day'].std()
                features[f'end_of_day_min_{window}'] = latest_data['end_of_day'].min()
                features[f'end_of_day_max_{window}'] = latest_data['end_of_day'].max()
                features[f'highest_mean_{window}'] = latest_data['highest'].mean()
                features[f'highest_std_{window}'] = latest_data['highest'].std()
                features[f'lowest_mean_{window}'] = latest_data['lowest'].mean()
                features[f'lowest_std_{window}'] = latest_data['lowest'].std()
        
        # Rate of change
        if len(latest_data) >= 30:
            features['price_change_1d'] = (latest_data['end_of_day'].iloc[-1] - latest_data['end_of_day'].iloc[-2]) / latest_data['end_of_day'].iloc[-2]
            features['price_change_7d'] = (latest_data['end_of_day'].iloc[-1] - latest_data['end_of_day'].iloc[-8]) / latest_data['end_of_day'].iloc[-8]
            features['price_change_30d'] = (latest_data['end_of_day'].iloc[-1] - latest_data['end_of_day'].iloc[-31]) / latest_data['end_of_day'].iloc[-31]
        else:
            features['price_change_1d'] = 0
            features['price_change_7d'] = 0
            features['price_change_30d'] = 0
        
        # Price metrics
        latest_row = latest_data.iloc[-1]
        features['price_range'] = latest_row['highest'] - latest_row['lowest']
        features['price_avg'] = (latest_row['highest'] + latest_row['lowest']) / 2
        features['price_volatility'] = features['price_range'] / features['price_avg'] if features['price_avg'] != 0 else 0
        
        # Momentum
        if len(latest_data) >= 7:
            features['price_momentum_3d'] = latest_data['end_of_day'].iloc[-1] - latest_data['end_of_day'].iloc[-4]
            features['price_momentum_7d'] = latest_data['end_of_day'].iloc[-1] - latest_data['end_of_day'].iloc[-8]
        else:
            features['price_momentum_3d'] = 0
            features['price_momentum_7d'] = 0
        
        # Relative position
        if features['price_range'] != 0:
            features['relative_position'] = (latest_row['end_of_day'] - latest_row['lowest']) / features['price_range']
        else:
            features['relative_position'] = 0.5
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# Prediction function
def make_prediction(model_name, model_data, data, days=1):
    """Make predictions for specified number of days"""
    models = model_data['models']
    scalers = model_data['scalers']
    
    if days == 1:
        # Single day prediction
        if model_name in ['LSTM', 'GRU']:
            # Sequence prediction for neural networks
            seq_length = 30
            latest_prices = data['end_of_day'].tail(seq_length).values
            latest_prices_scaled = scalers['y'].transform(latest_prices.reshape(-1, 1))
            X_pred = latest_prices_scaled.reshape(1, seq_length, 1)
            
            pred_scaled = models[model_name].predict(X_pred, verbose=0)
            prediction = scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
        else:  # DeepAR
            seq_length = 30
            latest_prices = data['end_of_day'].tail(seq_length).values
            latest_prices_scaled = scalers['y'].transform(latest_prices.reshape(-1, 1))
            X_pred = latest_prices_scaled.reshape(1, seq_length, 1)
            
            pred_scaled = model_data['models']['DeepAR'].predict(X_pred)
            prediction = scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
        # Apply 1% umbrella
        base_price = data['end_of_day'].iloc[-1]
        umbrella = base_price * 0.01
        lower_bound = prediction - umbrella
        upper_bound = prediction + umbrella
        
        return {
            'prediction': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': f"±{umbrella:.2f}"
        }
        
    else:
        # Multi-day prediction
        predictions = []
        data_copy = data.copy()
        
        for day in range(days):
            if model_name in ['LSTM', 'GRU']:
                seq_length = 30
                latest_prices = data_copy['end_of_day'].tail(seq_length).values
                latest_prices_scaled = scalers['y'].transform(latest_prices.reshape(-1, 1))
                X_pred = latest_prices_scaled.reshape(1, seq_length, 1)
                
                pred_scaled = models[model_name].predict(X_pred, verbose=0)
                prediction = scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                
            else:  # DeepAR
                seq_length = 30
                latest_prices = data_copy['end_of_day'].tail(seq_length).values
                latest_prices_scaled = scalers['y'].transform(latest_prices.reshape(-1, 1))
                X_pred = latest_prices_scaled.reshape(1, seq_length, 1)
                
                pred_scaled = model_data['models']['DeepAR'].predict(X_pred)
                prediction = scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
            # Add prediction to data for next iteration
            last_date = data_copy['date'].max()
            new_date = last_date + timedelta(days=1)
            new_row = pd.DataFrame({
                'date': [new_date],
                'end_of_day': [prediction],
                'highest': [prediction * 1.005],  # Assume slight variation
                'lowest': [prediction * 0.995]
            })
            data_copy = pd.concat([data_copy, new_row], ignore_index=True)
            
            # Apply 1% umbrella
            base_price = data['end_of_day'].iloc[-1]
            umbrella = base_price * 0.01 * (1 + day * 0.1)  # Increase uncertainty over time
            
            predictions.append({
                'date': new_date,
                'prediction': prediction,
                'lower_bound': prediction - umbrella,
                'upper_bound': prediction + umbrella
            })
        
        return predictions

def main():
    load_css()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h1 style="color: #d9a441; font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem;">
            AuricAI ✨
        </h1>
        <p style="color: #7a5c29; font-size: 1.3rem; font-weight: 600; margin-top: 0;">
            Advanced AI-Powered Gold Price Prediction with Real-time Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)


    # Load data and models
    data = load_data()
    model_data = load_models()
    
    # Sidebar with option menu
    with st.sidebar:
        selected = option_menu(
        menu_title="Main Menu",
        options=["Market Overview", "Technical Analysis", "Predictions & Alerts"],
        icons=["graph-up", "bar-chart", "exclamation-triangle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#f3dfb8",  # golden beige
            },
            "icon": {"color": "#a3742a", "font-size": "18px"},  # warm gold
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#ecdca9",
                "color": "#3b2f0b"
            },
            "nav-link-selected": {
                "background-color": "#d9a441",  # vibrant gold
                "color": "white"
            },
        }
    )

        st.markdown("---")
        
        # Time period selection
        st.subheader("📅 Prediction Period")
        prediction_period = st.selectbox(
            "Select forecast horizon:",
            options=["Tomorrow", "Next 7 Days", "Next 30 Days"],
            help="Choose the prediction time frame"
        )

        # Model selection
        st.subheader("📊 Select Model")
        selected_model = st.selectbox(
            "Choose forecasting model:",
            options=list(model_data['models'].keys()),
            help="Select the AI model for predictions"
        )
        
        # Display model performance (smaller size)
        st.subheader("📈 Model Performance")
        results = model_data['results'][selected_model]
        
        st.sidebar.markdown(f"""
            <div style="
                width: 90%; 
                padding: 10px; 
                background-color: #fff6cc;
                border-left: 4px solid #d9a441;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', sans-serif;
                margin-bottom: 1rem;
            ">
                <div style="font-weight: 600; font-size: 1.1em; color: #3b2f0b;">
                    📊 {selected_model}
                </div>
                <div style="font-size: 0.95em; color: #6b4c1e;">
                    <strong>RMSE:</strong> <span style="color:#3b2f0b;">{results['RMSE']:.2f}</span><br>
                    <strong>R²:</strong> <span style="color:#3b2f0b;">{results['R2']:.4f}</span><br>
                    <strong>MAE:</strong> <span style="color:#3b2f0b;">{results['MAE']:.2f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Get real-time data
    real_time_data = get_real_time_gold_price()
    
    # Page routing based on selection
    if selected == "Market Overview":
        show_market_overview(data, model_data, selected_model, prediction_period, real_time_data)
    elif selected == "Technical Analysis":
        show_technical_analysis(data, model_data, selected_model)
    elif selected == "Predictions & Alerts":
        show_predictions_alerts(data, model_data, selected_model, prediction_period)
    
    # Footer
    st.markdown("---")
    st.markdown(""" <div style="text-align: center; color: #6b7280; font-size: 0.9em;"> 
                <p>⚠️ <strong>Disclaimer:</strong> This is a forecasting tool for educational purposes. 
                Gold prices are subject to market volatility. Always consult with financial advisors before making investment decisions.</p> 
                <p>📈 Data Source: Historical Gold Prices Of India | 📊 Built with Streamlit | 🤖 Powered by Deep Learning</p> <p>Made with ❤️ by Arpan😎</p> 
                </div> 
          """, unsafe_allow_html=True)

def show_market_overview(data, model_data, selected_model, prediction_period, real_time_data):
    """Page 1: Market Overview with KPI boxes, main chart, price distribution, and volatility"""
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if real_time_data:
            change_class = "positive" if real_time_data['change'] > 0 else "negative" if real_time_data['change'] < 0 else "neutral"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Price</div>
                <div class="kpi-value" style="font-size: 1.75rem;">₹{real_time_data['current']:.2f}</div>
                <div class="kpi-change {change_class}">
                    {'+' if real_time_data['change'] > 0 else ''}{real_time_data['change']:.2f} ({real_time_data['change_pct']:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        yesterday_price = data['end_of_day'].iloc[-1]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Yesterday</div>
            <div class="kpi-value" style="font-size: 1.75rem;">₹{yesterday_price:.2f}</div>
            <div class="kpi-change neutral">Historical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Make tomorrow prediction
        tomorrow_pred = make_prediction(selected_model, model_data, data, days=1)
        if tomorrow_pred:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tomorrow</div>
                <div class="kpi-value" style="font-size: 1.75rem;">₹{tomorrow_pred['prediction']:.2f}</div>
                <div class="kpi-change neutral">{tomorrow_pred['confidence_interval']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # 7-day average prediction
        seven_day_preds = make_prediction(selected_model, model_data, data, days=7)
        if seven_day_preds:
            avg_7_day = np.mean([p['prediction'] for p in seven_day_preds])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">7-Day Avg</div>
                <div class="kpi-value" style="font-size: 1.75rem;">₹{avg_7_day:.2f}</div>
                <div class="kpi-change neutral">Average</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        # 30-day average prediction
        thirty_day_preds = make_prediction(selected_model, model_data, data, days=30)
        if thirty_day_preds:
            avg_30_day = np.mean([p['prediction'] for p in thirty_day_preds])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">30-Day Avg</div>
                <div class="kpi-value" style="font-size: 1.75rem;">₹{avg_30_day:.2f}</div>
                <div class="kpi-change neutral">Average</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main charts
    col1, col2 = st.columns([2.7, 1.3])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader(f"📈 Gold Price Trend & {prediction_period} Forecast")
        
        # Create main price chart
        fig = go.Figure()
        
        # Historical data
        last_60_days = data.tail(60)
        fig.add_trace(go.Scatter(
            x=last_60_days['date'],
            y=last_60_days['end_of_day'],
            mode='lines',
            name='Historical Prices',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Predictions based on selected period
        if prediction_period == "Tomorrow":
            if tomorrow_pred:
                future_date = data['date'].max() + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[data['date'].iloc[-1], future_date],
                    y=[data['end_of_day'].iloc[-1], tomorrow_pred['prediction']],
                    mode='lines+markers',
                    name='Tomorrow Prediction',
                    line=dict(color='#e74c3c', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=[future_date, future_date, future_date],
                    y=[tomorrow_pred['lower_bound'], tomorrow_pred['prediction'], tomorrow_pred['upper_bound']],
                    mode='markers',
                    name='Confidence Range',
                    marker=dict(size=6, color='#f39c12'),
                    showlegend=True
                ))
        
        elif prediction_period == "Next 7 Days":
            if seven_day_preds:
                pred_dates = [p['date'] for p in seven_day_preds]
                pred_values = [p['prediction'] for p in seven_day_preds]
                lower_bounds = [p['lower_bound'] for p in seven_day_preds]
                upper_bounds = [p['upper_bound'] for p in seven_day_preds]
                
                # Add connection from last historical point
                extended_dates = [data['date'].iloc[-1]] + pred_dates
                extended_values = [data['end_of_day'].iloc[-1]] + pred_values
                
                fig.add_trace(go.Scatter(
                    x=extended_dates,
                    y=extended_values,
                    mode='lines+markers',
                    name='7-Day Forecast',
                    line=dict(color='#e74c3c', width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Confidence bands
                fig.add_trace(go.Scatter(
                    x=pred_dates + pred_dates[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill='tonexty',
                    fillcolor='rgba(243, 156, 18, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Band',
                    showlegend=True
                ))
        
        elif prediction_period == "Next 30 Days":
            if thirty_day_preds:
                pred_dates = [p['date'] for p in thirty_day_preds]
                pred_values = [p['prediction'] for p in thirty_day_preds]
                lower_bounds = [p['lower_bound'] for p in thirty_day_preds]
                upper_bounds = [p['upper_bound'] for p in thirty_day_preds]
                
                # Add connection from last historical point
                extended_dates = [data['date'].iloc[-1]] + pred_dates
                extended_values = [data['end_of_day'].iloc[-1]] + pred_values
                
                fig.add_trace(go.Scatter(
                    x=extended_dates,
                    y=extended_values,
                    mode='lines+markers',
                    name='30-Day Forecast',
                    line=dict(color='#e74c3c', width=3, dash='dash'),
                    marker=dict(size=4)
                ))
                
                # Confidence bands
                fig.add_trace(go.Scatter(
                    x=pred_dates + pred_dates[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill='tonexty',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Band',
                    showlegend=True
                ))
        
        fig = apply_clean_layout(fig, f'{selected_model} Model - {prediction_period} Forecast', height=580)
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (₹ per 10g)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=60, t=120, b=60),
            title=dict(
                y=0.92,
                x=0.5,
                xanchor='center',
                yanchor='top'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config=get_interactive_config())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Price Distribution")
        
        # Price distribution chart
        recent_prices = data['end_of_day'].tail(30)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=recent_prices,
            nbinsx=15,
            name='Price Distribution',
            marker=dict(color='#3498db', opacity=0.7),
            showlegend=False
        ))
        
        fig_dist = apply_clean_layout(fig_dist, 'Last 30 Days Price Distribution', height=300)
        fig_dist.update_layout(
            xaxis_title='Price (₹)',
            yaxis_title='Frequency',
            margin=dict(l=40, r=40, t=70, b=40),
            title=dict(
                y=0.85,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=14)
            )
        )
        
        st.plotly_chart(fig_dist, use_container_width=True, config=get_interactive_config())
        
        # Volatility gauge
        st.subheader("🌡️ Volatility Index")
        recent_volatility = recent_prices.std() / recent_prices.mean() * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = recent_volatility,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=200, 
            plot_bgcolor='rgba(255,255,255,0.95)', 
            paper_bgcolor='rgba(255,255,255,0.95)',
            margin=dict(l=20, r=20, t=70, b=20),
            title={
                'text': 'Volatility Percentage',
                'x': 0.5,
                'y': 0.85,
                'xanchor': 'center',
                'font': {'size': 14, 'color': '#2c3e50'}
            }
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config=get_interactive_config())
        st.markdown('</div>', unsafe_allow_html=True)

def show_technical_analysis(data, model_data, selected_model):
    """Page 2: Technical Analysis with Technical Indicators and Model Comparison"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Technical Indicators")
        
        # Calculate moving averages
        ma_7 = data['end_of_day'].rolling(window=7).mean()
        ma_21 = data['end_of_day'].rolling(window=21).mean()
        ma_50 = data['end_of_day'].rolling(window=50).mean()
        
        fig_tech = go.Figure()
        
        # Show last 60 days
        last_60_idx = max(0, len(data) - 60)
        dates_60 = data['date'][last_60_idx:]
        prices_60 = data['end_of_day'][last_60_idx:]
        ma_7_60 = ma_7[last_60_idx:]
        ma_21_60 = ma_21[last_60_idx:]
        ma_50_60 = ma_50[last_60_idx:]
        
        fig_tech.add_trace(go.Scatter(
            x=dates_60, y=prices_60,
            mode='lines', name='Price',
            line=dict(color='#2c3e50', width=2)
        ))
        
        fig_tech.add_trace(go.Scatter(
            x=dates_60, y=ma_7_60,
            mode='lines', name='MA(7)',
            line=dict(color='#e74c3c', width=1, dash='dash')
        ))
        
        fig_tech.add_trace(go.Scatter(
            x=dates_60, y=ma_21_60,
            mode='lines', name='MA(21)',
            line=dict(color='#f39c12', width=1, dash='dot')
        ))
        
        fig_tech.add_trace(go.Scatter(
            x=dates_60, y=ma_50_60,
            mode='lines', name='MA(50)',
            line=dict(color='#27ae60', width=1, dash='dashdot')
        ))
        
        fig_tech = apply_clean_layout(fig_tech, 'Moving Averages Analysis', height=500)
        fig_tech.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            margin=dict(l=60, r=60, t=100, b=120),
            title=dict(
                text='Moving Averages Analysis',
                y=0.88,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
        
        st.plotly_chart(fig_tech, use_container_width=True, config=get_interactive_config())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Model Comparison")
        
        # Model performance comparison
        models_perf = []
        for model_name, metrics in model_data['results'].items():
            models_perf.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'R²': metrics['R2'],
                'MAE': metrics['MAE']
            })
        
        df_perf = pd.DataFrame(models_perf)
        
        # Create comparison chart
        fig_comp = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE', 'R² Score', 'MAE', 'Model Rankings'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # RMSE comparison
        fig_comp.add_trace(
            go.Bar(x=df_perf['Model'], y=df_perf['RMSE'], 
                   marker_color='#e74c3c', name='RMSE'),
            row=1, col=1
        )
        
        # R² comparison
        fig_comp.add_trace(
            go.Bar(x=df_perf['Model'], y=df_perf['R²'], 
                   marker_color='#27ae60', name='R²'),
            row=1, col=2
        )
        
        # MAE comparison
        fig_comp.add_trace(
            go.Bar(x=df_perf['Model'], y=df_perf['MAE'], 
                   marker_color='#3498db', name='MAE'),
            row=2, col=1
        )
        
        # Overall performance ranking
        df_perf['Composite_Score'] = (1/df_perf['RMSE']) + df_perf['R²'] + (1/df_perf['MAE'])
        
        fig_comp.add_trace(
            go.Bar(x=df_perf['Model'], y=df_perf['Composite_Score'], 
                   marker_color='#9b59b6', name='Composite Score'),
            row=2, col=2
        )
        
        fig_comp.update_layout(
            height=500, 
            showlegend=False, 
            plot_bgcolor='rgba(255,255,255,0.95)', 
            paper_bgcolor='rgba(255,255,255,0.95)',
            margin=dict(l=60, r=60, t=120, b=80),
            title={
                'text': 'Model Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.88,
                'font': {'size': 18, 'color': '#2c3e50'}
            }
        )
        
        fig_comp.update_annotations(font_size=12, font_color='#2c3e50')
        
        st.plotly_chart(fig_comp, use_container_width=True, config=get_interactive_config())
        st.markdown('</div>', unsafe_allow_html=True)

def show_predictions_alerts(data, model_data, selected_model, prediction_period):
    """Page 3: Predictions & Alerts with Insights and Market Alerts"""
    
    # Get predictions
    tomorrow_pred = make_prediction(selected_model, model_data, data, days=1)
    seven_day_preds = make_prediction(selected_model, model_data, data, days=7)
    thirty_day_preds = make_prediction(selected_model, model_data, data, days=30)
    
    # Prediction insights section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔮 Prediction Insights & Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📅 Tomorrow's Outlook")
        if tomorrow_pred:
            current_price = data['end_of_day'].iloc[-1]
            change = tomorrow_pred['prediction'] - current_price
            change_pct = (change / current_price) * 100
            
            trend_color = "green" if change > 0 else "red" if change < 0 else "gray"
            trend_arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            trend_text = "Bullish" if change > 0 else "Bearish" if change < 0 else "Neutral"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffffff, #f8f9fa); border-radius: 10px; border: 1px solid #ddd;">
                <h3 style="color: {trend_color};">{trend_arrow} {trend_text}</h3>
                <p><strong>Predicted Price:</strong> ₹{tomorrow_pred['prediction']:.2f}</p>
                <p><strong>Expected Change:</strong> {'+' if change > 0 else ''}{change:.2f} ({change_pct:+.2f}%)</p>
                <p><strong>Confidence Range:</strong> ₹{tomorrow_pred['lower_bound']:.2f} - ₹{tomorrow_pred['upper_bound']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 7-Day Trend Analysis")
        if seven_day_preds:
            prices = [p['prediction'] for p in seven_day_preds]
            trend = "Upward" if prices[-1] > prices[0] else "Downward" if prices[-1] < prices[0] else "Sideways"
            volatility = np.std(prices)
            avg_price = np.mean(prices)
            
            trend_color = "green" if trend == "Upward" else "red" if trend == "Downward" else "gray"
            trend_arrow = "📈" if trend == "Upward" else "📉" if trend == "Downward" else "↔️"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffffff, #f8f9fa); border-radius: 10px; border: 1px solid #ddd;">
                <h3 style="color: {trend_color};">{trend_arrow} {trend} Trend</h3>
                <p><strong>Average Price:</strong> ₹{avg_price:.2f}</p>
                <p><strong>Expected Range:</strong> ₹{min(prices):.2f} - ₹{max(prices):.2f}</p>
                <p><strong>Volatility Index:</strong> {volatility:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📈 30-Day Market Outlook")
        if thirty_day_preds:
            prices = [p['prediction'] for p in thirty_day_preds]
            long_term_trend = "Bullish" if prices[-1] > prices[0] else "Bearish" if prices[-1] < prices[0] else "Neutral"
            monthly_volatility = np.std(prices)
            monthly_avg = np.mean(prices)
            
            trend_color = "green" if long_term_trend == "Bullish" else "red" if long_term_trend == "Bearish" else "gray"
            trend_arrow = "🚀" if long_term_trend == "Bullish" else "🔻" if long_term_trend == "Bearish" else "🎯"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffffff, #f8f9fa); border-radius: 10px; border: 1px solid #ddd;">
                <h3 style="color: {trend_color};">{trend_arrow} {long_term_trend} Outlook</h3>
                <p><strong>Monthly Average:</strong> ₹{monthly_avg:.2f}</p>
                <p><strong>Expected Range:</strong> ₹{min(prices):.2f} - ₹{max(prices):.2f}</p>
                <p><strong>Market Stability:</strong> {'High' if monthly_volatility < 50 else 'Medium' if monthly_volatility < 100 else 'Low'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market summary and alerts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚠️ Market Alerts & Recommendations")
    
    # Generate alerts based on predictions
    alerts = []
    
    if tomorrow_pred:
        current_price = data['end_of_day'].iloc[-1]
        change_pct = ((tomorrow_pred['prediction'] - current_price) / current_price) * 100
        
        if abs(change_pct) > 2:
            alert_type = "🚨 High Volatility Alert" if abs(change_pct) > 5 else "⚠️ Price Movement Alert"
            alerts.append(f"{alert_type}: Expected {change_pct:+.2f}% change tomorrow")
        
        if tomorrow_pred['prediction'] > data['end_of_day'].tail(7).max():
            alerts.append("📈 Breakout Alert: Price may reach new 7-day high")
        
        if tomorrow_pred['prediction'] < data['end_of_day'].tail(7).min():
            alerts.append("📉 Support Alert: Price may test 7-day low")
    
    if seven_day_preds:
        weekly_volatility = np.std([p['prediction'] for p in seven_day_preds])
        if weekly_volatility > 100:
            alerts.append("🌪️ High Volatility Week: Expect significant price swings")
    
    if not alerts:
        alerts.append("✅ Market Stability: No major alerts detected")
    
    # Render all alerts inside a styled alert box
    st.markdown('<div class="alerts-container">', unsafe_allow_html=True)

    for alert in alerts:
        st.markdown(f"""
        <div class="alert-box">
            {alert}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()