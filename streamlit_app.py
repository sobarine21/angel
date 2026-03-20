# app.py
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="OHLC Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load credentials from Streamlit secrets
def load_config():
    """Load API configuration from Streamlit secrets"""
    try:
        # Check if secrets are available
        if not hasattr(st, 'secrets'):
            st.error("Streamlit secrets not available")
            return None
        
        # Try to access the api section
        if 'api' not in st.secrets:
            st.error("'api' section not found in secrets.toml")
            st.info("""
            Please ensure your .streamlit/secrets.toml file has this structure:
```toml
            [api]
            base_url = "https://YOUR_PROJECT.supabase.co/functions/v1"
            anon_key = "YOUR_ANON_KEY"
```
            """)
            return None
        
        config = {
            'base_url': st.secrets["api"]["base_url"],
            'anon_key': st.secrets["api"]["anon_key"]
        }
        
        # Validate the configuration
        if not config['base_url'] or not config['anon_key']:
            st.error("API credentials are empty in secrets.toml")
            return None
        
        if "YOUR_" in config['base_url'] or "YOUR_" in config['anon_key']:
            st.warning("⚠️ Please update the placeholder values in secrets.toml with your actual credentials")
            return None
        
        return config
        
    except KeyError as e:
        st.error(f"Missing key in secrets.toml: {e}")
        st.info("""
        Required structure for .streamlit/secrets.toml:
```toml
        [api]
        base_url = "https://your-project.supabase.co/functions/v1"
        anon_key = "your-actual-anon-key"
```
        """)
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        st.info("Please check your .streamlit/secrets.toml file")
        return None

# API Client Class
class OHLCAPIClient:
    def __init__(self, base_url, anon_key):
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.headers = {
            'Authorization': f'Bearer {anon_key}',
            'Content-Type': 'application/json'
        }
    
    def get_ohlc_data(self, symbol=None, start_date=None, end_date=None, limit=1000, offset=0):
        """Get OHLC data with optional filters"""
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if symbol:
            params['symbol'] = symbol
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        try:
            url = f"{self.base_url}/ohlc-data"
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
    
    def get_ohlc_by_id(self, record_id):
        """Get specific OHLC record by ID"""
        try:
            url = f"{self.base_url}/ohlc-data/{record_id}"
            response = requests.get(
                url,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
    
    def get_symbol_data(self, symbol, limit=100):
        """Get data for specific symbol"""
        try:
            url = f"{self.base_url}/ohlc-data/symbol/{symbol}"
            response = requests.get(
                url,
                headers=self.headers,
                params={'limit': limit},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None
    
    def get_latest_ohlc(self, symbol):
        """Get latest OHLC for a symbol"""
        try:
            url = f"{self.base_url}/ohlc-data/latest/{symbol}"
            response = requests.get(
                url,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")
            return None

# Data Processing Functions
def process_ohlc_dataframe(data):
    """Convert API response to pandas DataFrame"""
    if not data or 'data' not in data:
        return None
    
    df = pd.DataFrame(data['data'])
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    return df

def create_candlestick_chart(df, symbol):
    """Create interactive candlestick chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Volume bar chart
    colors = ['red' if close < open else 'green' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} OHLC Chart',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='x unified'
    )
    
    return fig

def calculate_technical_indicators(df):
    """Calculate basic technical indicators"""
    if df is None or df.empty:
        return df
    
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # Price change
    df['price_change'] = df['close'].diff()
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # Volume change
    df['volume_change_pct'] = df['volume'].pct_change() * 100
    
    return df

# Main Application
def main():
    st.markdown('<p class="main-header">📈 OHLC Data Dashboard</p>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    if not config:
        st.stop()
    
    # Display connection status
    with st.sidebar:
        st.success("✅ Configuration Loaded")
        with st.expander("Connection Details"):
            st.code(f"Base URL: {config['base_url']}")
            st.code(f"Auth: Bearer {config['anon_key'][:20]}...")
    
    # Initialize API client
    api_client = OHLCAPIClient(config['base_url'], config['anon_key'])
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Symbol Analysis", "Data Explorer", "API Testing", "Bulk Data Download"]
    )
    
    st.sidebar.markdown("---")
    
    # MODE 1: Symbol Analysis
    if mode == "Symbol Analysis":
        st.header("Symbol Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input("Enter Symbol", value="AAPL", help="Enter stock symbol")
        
        with col2:
            limit = st.number_input("Number of records", min_value=10, max_value=1000, value=100)
        
        if st.button("Fetch Data", type="primary"):
            with st.spinner(f"Fetching data for {symbol}..."):
                result = api_client.get_symbol_data(symbol, limit)
                
                if result:
                    df = process_ohlc_dataframe(result)
                    
                    if df is not None and not df.empty:
                        # Calculate indicators
                        df = calculate_technical_indicators(df)
                        
                        # Store in session state
                        st.session_state['current_df'] = df
                        st.session_state['current_symbol'] = symbol
                        
                        # Display latest data
                        latest = df.iloc[-1]
                        
                        st.subheader("Latest Data")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Close", f"${latest['close']:.2f}", 
                                     f"{latest['price_change_pct']:.2f}%")
                        with col2:
                            st.metric("Open", f"${latest['open']:.2f}")
                        with col3:
                            st.metric("High", f"${latest['high']:.2f}")
                        with col4:
                            st.metric("Low", f"${latest['low']:.2f}")
                        with col5:
                            st.metric("Volume", f"{latest['volume']:,.0f}")
                        
                        # Candlestick chart
                        st.plotly_chart(create_candlestick_chart(df, symbol), use_container_width=True)
                        
                        # Technical indicators
                        st.subheader("Technical Indicators")
                        fig_indicators = go.Figure()
                        
                        fig_indicators.add_trace(go.Scatter(
                            x=df['date'], y=df['close'],
                            name='Close Price', line=dict(color='blue', width=2)
                        ))
                        fig_indicators.add_trace(go.Scatter(
                            x=df['date'], y=df['SMA_20'],
                            name='SMA 20', line=dict(color='orange', width=1, dash='dash')
                        ))
                        fig_indicators.add_trace(go.Scatter(
                            x=df['date'], y=df['SMA_50'],
                            name='SMA 50', line=dict(color='red', width=1, dash='dash')
                        ))
                        
                        fig_indicators.update_layout(
                            title='Price with Moving Averages',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_indicators, use_container_width=True)
                        
                        # Data table
                        st.subheader("Raw Data")
                        st.dataframe(
                            df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(20),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name=f"{symbol}_ohlc_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No data found for symbol: {symbol}")
    
    # MODE 2: Data Explorer
    elif mode == "Data Explorer":
        st.header("Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol_filter = st.text_input("Symbol (optional)", placeholder="e.g., AAPL")
        
        with col2:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        
        with col3:
            end_date = st.date_input("End Date", value=datetime.now())
        
        col4, col5 = st.columns(2)
        with col4:
            limit = st.number_input("Limit", min_value=10, max_value=5000, value=500)
        with col5:
            offset = st.number_input("Offset", min_value=0, value=0)
        
        if st.button("Search", type="primary"):
            with st.spinner("Searching database..."):
                result = api_client.get_ohlc_data(
                    symbol=symbol_filter if symbol_filter else None,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    limit=limit,
                    offset=offset
                )
                
                if result:
                    df = process_ohlc_dataframe(result)
                    
                    if df is not None and not df.empty:
                        st.success(f"Found {len(df)} records")
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Unique Symbols", df['symbol'].nunique())
                        with col3:
                            st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
                        
                        # Symbol distribution
                        st.subheader("Symbol Distribution")
                        symbol_counts = df['symbol'].value_counts().head(10)
                        fig_dist = go.Figure(data=[
                            go.Bar(x=symbol_counts.index, y=symbol_counts.values)
                        ])
                        fig_dist.update_layout(
                            title='Top 10 Symbols by Record Count',
                            xaxis_title='Symbol',
                            yaxis_title='Count',
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Data preview
                        st.subheader("Data Preview")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"ohlc_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data found matching your criteria")
    
    # MODE 3: API Testing
    elif mode == "API Testing":
        st.header("API Testing")
        
        endpoint = st.selectbox(
            "Select Endpoint",
            [
                "GET /ohlc-data (All Data)",
                "GET /ohlc-data/:id (By ID)",
                "GET /ohlc-data/symbol/:symbol (By Symbol)",
                "GET /ohlc-data/latest/:symbol (Latest)"
            ]
        )
        
        if "All Data" in endpoint:
            st.subheader("Test: Get All OHLC Data")
            
            col1, col2 = st.columns(2)
            with col1:
                test_symbol = st.text_input("Symbol (optional)")
                test_limit = st.number_input("Limit", value=10, min_value=1, max_value=100)
            with col2:
                test_start = st.date_input("Start Date (optional)", value=None)
                test_end = st.date_input("End Date (optional)", value=None)
            
            if st.button("Test API Call"):
                with st.spinner("Making API call..."):
                    result = api_client.get_ohlc_data(
                        symbol=test_symbol if test_symbol else None,
                        start_date=test_start.isoformat() if test_start else None,
                        end_date=test_end.isoformat() if test_end else None,
                        limit=test_limit
                    )
                    
                    st.subheader("Response")
                    st.json(result)
        
        elif "By ID" in endpoint:
            st.subheader("Test: Get OHLC by ID")
            
            record_id = st.number_input("Record ID", min_value=1, value=1)
            
            if st.button("Test API Call"):
                with st.spinner("Making API call..."):
                    result = api_client.get_ohlc_by_id(record_id)
                    
                    st.subheader("Response")
                    st.json(result)
        
        elif "By Symbol" in endpoint:
            st.subheader("Test: Get Symbol Data")
            
            col1, col2 = st.columns(2)
            with col1:
                test_symbol = st.text_input("Symbol", value="AAPL")
            with col2:
                test_limit = st.number_input("Limit", value=50, min_value=1, max_value=1000)
            
            if st.button("Test API Call"):
                with st.spinner("Making API call..."):
                    result = api_client.get_symbol_data(test_symbol, test_limit)
                    
                    st.subheader("Response")
                    st.json(result)
        
        else:  # Latest
            st.subheader("Test: Get Latest OHLC")
            
            test_symbol = st.text_input("Symbol", value="AAPL")
            
            if st.button("Test API Call"):
                with st.spinner("Making API call..."):
                    result = api_client.get_latest_ohlc(test_symbol)
                    
                    st.subheader("Response")
                    st.json(result)
    
    # MODE 4: Bulk Data Download
    elif mode == "Bulk Data Download":
        st.header("Bulk Data Download")
        
        st.info("Download large datasets in batches")
        
        col1, col2 = st.columns(2)
        with col1:
            bulk_symbol = st.text_input("Symbol (leave empty for all)", placeholder="Optional")
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col2:
            total_records = st.number_input("Total Records to Download", min_value=100, max_value=50000, value=1000)
            batch_size = st.number_input("Batch Size", min_value=100, max_value=1000, value=500)
        
        end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("Start Download", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_data = []
            offset = 0
            
            while offset < total_records:
                current_batch = min(batch_size, total_records - offset)
                
                status_text.text(f"Downloading records {offset} to {offset + current_batch}...")
                
                result = api_client.get_ohlc_data(
                    symbol=bulk_symbol if bulk_symbol else None,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    limit=current_batch,
                    offset=offset
                )
                
                if result and 'data' in result and result['data']:
                    all_data.extend(result['data'])
                    offset += len(result['data'])
                    
                    progress = min(offset / total_records, 1.0)
                    progress_bar.progress(progress)
                    
                    if len(result['data']) < current_batch:
                        # No more data available
                        break
                else:
                    break
            
            if all_data:
                df = pd.DataFrame(all_data)
                st.success(f"Successfully downloaded {len(df)} records!")
                
                # Preview
                st.dataframe(df.head(50), use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Complete Dataset as CSV",
                    data=csv,
                    file_name=f"ohlc_bulk_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data downloaded")

if __name__ == "__main__":
    main()
