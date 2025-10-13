import streamlit as st
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta  # Technical Analysis library

# Supabase imports
from supabase import create_client, Client

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"

# Initialize session state variables if they don't exist
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "saved_indexes" not in st.session_state:
    st.session_state["saved_indexes"] = []
if "current_calculated_index_data" not in st.session_state: # To store current CSV's index data
    st.session_state["current_calculated_index_data"] = None
if "current_calculated_index_history" not in st.session_state: # To store historical index values for plotting
    st.session_state["current_calculated_index_history"] = pd.DataFrame()
if "kt_ticker" not in st.session_state:
    st.session_state["kt_ticker"] = None
if "kt_thread" not in st.session_state:
    st.session_state["kt_thread"] = None
if "kt_running" not in st.session_state:
    st.session_state["kt_running"] = False
if "kt_ticks" not in st.session_state:
    st.session_state["kt_ticks"] = []
if "kt_live_prices" not in st.session_state:
    st.session_state["kt_live_prices"] = pd.DataFrame(columns=['timestamp', 'last_price', 'instrument_token'])
if "kt_status_message" not in st.session_state:
    st.session_state["kt_status_message"] = "Not started"
if "_rerun_ws" not in st.session_state: # Flag for WebSocket UI updates
    st.session_state["_rerun_ws"] = False
if "broker_data_fetched_and_saved" not in st.session_state: # Flag for auto redirect
    st.session_state["broker_data_fetched_and_saved"] = False

# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})
    auto_redirect_conf = secrets.get("auto_redirect", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not supabase_conf.get("url") or not supabase_conf.get("anon_key"):
        errors.append("Supabase credentials (url, anon_key)")
    if not auto_redirect_conf.get("url"):
        errors.append("Auto redirect URL (url in [auto_redirect])")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n\n[supabase]\nurl=\"YOUR_SUPABASE_URL\"\nanon_key=\"YOUR_SUPABASE_ANON_KEY\"\n\n[auto_redirect]\nurl=\"YOUR_REDIRECT_URL\"\n```")
        st.stop()
    return kite_conf, supabase_conf, auto_redirect_conf["url"]

KITE_CREDENTIALS, SUPABASE_CREDENTIALS, AUTO_REDIRECT_URL = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600)
def init_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

# --- KiteConnect Client Initialization (Unauthenticated for login URL) ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

def safe_get_numeric(data, key, default_value=None): # Changed default to None
    """Safely retrieve a numerical value. Returns default_value if None, non-numeric, or not present."""
    value = data.get(key)
    if value is None:
        return default_value # Return None if original value is None
    try:
        # Attempt to convert to float first for broader compatibility
        float_val = float(value)
        # If it's a whole number, convert to int. Otherwise, keep as float.
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except (ValueError, TypeError):
        return default_value # Return None if conversion fails


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    request_token_param = st.query_params.get("request_token")

    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.pop("request_token", None)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", key="kite_logout_btn"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.sidebar.success("Logged out from Kite. Please login again.")
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")


# --- Sidebar: Supabase Authentication ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    
    def _refresh_supabase_session():
        try:
            session_data = supabase.auth.get_session()
            if session_data and session_data.user:
                st.session_state["user_session"] = session_data
                st.session_state["user_id"] = session_data.user.id
            else:
                st.session_state["user_session"] = None
                st.session_state["user_id"] = None
        except Exception:
            st.session_state["user_session"] = None
            st.session_state["user_id"] = None

    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session()
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        with st.form("supabase_auth_form"):
            st.markdown("##### Email/Password Login/Sign Up")
            email = st.text_input("Email", key="supabase_email_input", help="Your email for Supabase authentication.")
            password = st.text_input("Password", type="password", key="supabase_password_input", help="Your password for Supabase authentication.")
            
            col_auth1, col_auth2 = st.columns(2)
            with col_auth1:
                login_submitted = st.form_submit_button("Login")
            with col_auth2:
                signup_submitted = st.form_submit_button("Sign Up")

            if login_submitted:
                if email and password:
                    try:
                        with st.spinner("Logging in..."):
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Login successful! Welcome.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.warning("Please enter both email and password for login.")
            
            if signup_submitted:
                if email and password:
                    try:
                        with st.spinner("Signing up..."):
                            response = supabase.auth.sign_up({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Sign up successful! Please check your email to confirm your account.")
                        st.info("After confirming your email, you can log in.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
                else:
                    st.warning("Please enter both email and password for sign up.")

    st.markdown("---")
    st.markdown("### 3. Broker Data Fetching & Auto Redirect")
    if st.session_state["kite_access_token"] and st.session_state["user_session"]:
        st.success("Kite and Supabase Authenticated. Fetching Broker Data...")
        
        if st.button("Fetch & Save Profile Data", key="fetch_save_broker_data_btn"):
            try:
                kite_client_authenticated = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
                if not kite_client_authenticated:
                    st.error("Kite client not authenticated. Please re-login.")
                    st.stop()
                
                profile_data = kite_client_authenticated.profile()
                margins_data = kite_client_authenticated.margins()
                
                user_profile_info = {
                    "user_id": st.session_state["user_id"],
                    "email": profile_data.get("email"),
                    "user_name": profile_data.get("user_name"),
                    "broker": "KiteConnect",
                    # Use safe_get_numeric with a default of 0 for margins
                    "funds_available_equity_live": safe_get_numeric(margins_data.get("equity", {}).get("available", {}), "live_balance", default_value=0),
                    "funds_utilized_equity": safe_get_numeric(margins_data.get("equity", {}).get("utilised", {}), "overall", default_value=0),
                    "funds_available_commodity_live": safe_get_numeric(margins_data.get("commodity", {}).get("available", {}), "live_balance", default_value=0),
                    "funds_utilized_commodity": safe_get_numeric(margins_data.get("commodity", {}).get("utilised", {}), "overall", default_value=0),
                    "fetched_at": datetime.now().isoformat()
                }
                
                # Save to Supabase
                existing_profile_query = supabase.table("user_profiles").select("id").eq("user_id", st.session_state["user_id"]).execute()
                
                if existing_profile_query.data:
                    profile_id_to_update = existing_profile_query.data[0]['id']
                    update_response = supabase.table("user_profiles").update(user_profile_info).eq("id", profile_id_to_update).execute()
                    if update_response.data is not None and update_response.count > 0:
                        st.success("User profile and fund data updated successfully in Supabase!")
                    else:
                        st.error(f"Failed to update user profile. Response: {update_response.data}")
                else:
                    insert_response = supabase.table("user_profiles").insert(user_profile_info).execute()
                    if insert_response.data is not None and insert_response.count > 0:
                        st.success("User profile and fund data saved successfully to Supabase!")
                    else:
                        st.error(f"Failed to save user profile. Response: {insert_response.data}")

                # Fetch and save order history
                try:
                    order_history = kite_client_authenticated.orders()
                    trades_history = kite_client_authenticated.trades() 

                    if order_history:
                        for order in order_history:
                            order_data = {
                                "user_id": st.session_state["user_id"],
                                "order_id": order["order_id"],
                                "parent_order_id": order.get("parent_order_id"),
                                "status": order["status"],
                                "symbol": order.get("tradingsymbol"),
                                "exchange": order["exchange"],
                                "order_type": order["order_type"],
                                "variety": order["variety"],
                                "validity": order["validity"],
                                "transaction_type": order["transaction_type"],
                                # Use safe_get_numeric with a default of 0 for these fields
                                "quantity": safe_get_numeric(order, "quantity", default_value=0),
                                "price": safe_get_numeric(order, "price", default_value=0.0), # Default to float 0.0
                                "trigger_price": safe_get_numeric(order, "trigger_price", default_value=0.0), # Default to float 0.0
                                "placed_at": order["order_timestamp"],
                                "product": order["product"],
                                "created_at": datetime.now().isoformat()
                            }
                            existing_order_query = supabase.table("order_history").select("id").eq("order_id", order_data["order_id"]).execute()
                            if existing_order_query.data:
                                update_order_response = supabase.table("order_history").update(order_data).eq("order_id", order_data["order_id"]).execute()
                                if update_order_response.data is None or update_order_response.count == 0:
                                    st.warning(f"Could not update order {order_data['order_id']}.")
                            else:
                                insert_order_response = supabase.table("order_history").insert(order_data).execute()
                                if insert_order_response.data is None or insert_order_response.count == 0:
                                    st.warning(f"Could not insert order {order_data['order_id']}.")
                        st.success(f"Processed {len(order_history)} orders. Check Supabase for details.")
                    
                    if trades_history:
                        for trade in trades_history:
                            trade_data = {
                                "user_id": st.session_state["user_id"],
                                "trade_id": trade["trade_id"],
                                "order_id": trade["order_id"],
                                "symbol": trade.get("tradingsymbol"),
                                "exchange": trade["exchange"],
                                "transaction_type": trade["transaction_type"],
                                # Use safe_get_numeric with a default of 0
                                "quantity": safe_get_numeric(trade, "quantity", default_value=0),
                                "price": safe_get_numeric(trade, "price", default_value=0.0), # Default to float 0.0
                                "executed_at": trade["execution_time"],
                                "product": trade["product"],
                                "created_at": datetime.now().isoformat()
                            }
                            existing_trade_query = supabase.table("trade_history").select("id").eq("trade_id", trade_data["trade_id"]).execute()
                            if existing_trade_query.data:
                                update_trade_response = supabase.table("trade_history").update(trade_data).eq("trade_id", trade_data["trade_id"]).execute()
                                if update_trade_response.data is None or update_trade_response.count == 0:
                                    st.warning(f"Could not update trade {trade_data['trade_id']}.")
                            else:
                                insert_trade_response = supabase.table("trade_history").insert(trade_data).execute()
                                if insert_trade_response.data is None or insert_trade_response.count == 0:
                                    st.warning(f"Could not insert trade {trade_data['trade_id']}.")
                        st.success(f"Processed {len(trades_history)} trades. Check Supabase for details.")

                    st.session_state["broker_data_fetched_and_saved"] = True

                except Exception as e:
                    st.error(f"Error fetching or saving order/trade history: {e}")

            except Exception as e:
                st.error(f"An error occurred during data fetching or saving: {e}")
    else:
        st.info("Please login to Kite and Supabase to enable Broker Data Fetching.")

    st.markdown("---")
    st.markdown("### 4. Auto Redirect")
    st.caption(f"Configured redirect URL: **{AUTO_REDIRECT_URL}**")
    
    if st.session_state.get("broker_data_fetched_and_saved"):
        st.info("Data fetched and saved. Redirecting in 5 seconds...")
        st.components.v1.html(f"<script>setTimeout(() => {{window.location.href = '{AUTO_REDIRECT_URL}';}}, 5000);</script>", height=0)
    else:
        st.info("Auto-redirect will occur after successful data fetch and save.")

# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main UI - Tabs for modules ---
tabs_list = ["Dashboard"]
if k and st.session_state["user_session"]:
    tabs_list.extend([
        "Portfolio", "Orders", "Market & Historical", "Machine Learning Analysis",
        "Risk & Stress Testing", "Performance Analysis", "Multi-Asset Analysis",
        "Custom Index", "Websocket (stream)", "Instruments Utils"
    ])

tabs = st.tabs(tabs_list)

tab_dashboard = tabs[0]
tab_portfolio = tabs[1] if len(tabs) > 1 else None
tab_orders = tabs[2] if len(tabs) > 2 else None
tab_market = tabs[3] if len(tabs) > 3 else None
tab_ml = tabs[4] if len(tabs) > 4 else None
tab_risk = tabs[5] if len(tabs) > 5 else None
tab_performance = tabs[6] if len(tabs) > 6 else None
tab_multi_asset = tabs[7] if len(tabs) > 7 else None
tab_custom_index = tabs[8] if len(tabs) > 8 else None
tab_ws = tabs[9] if len(tabs) > 9 else None
tab_inst = tabs[10] if len(tabs) > 10 else None


# --- Tab Logic Functions ---
def render_dashboard_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Dashboard")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to view the dashboard.")
        return
    
    st.success("Welcome! You are logged in.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Broker Account Summary")
        try:
            profile = kite_client.profile()
            margins = kite_client.margins()
            st.write(f"**User Name:** {profile.get('user_name', 'N/A')}")
            st.write(f"**Email:** {profile.get('email', 'N/A')}")
            # Using safe_get_numeric here too for consistency
            st.write(f"**Equity Available Margin:** ₹{safe_get_numeric(margins.get('equity', {}).get('available', {}), 'live_balance', 0):,.2f}")
            st.write(f"**Commodity Available Margin:** ₹{safe_get_numeric(margins.get('commodity', {}).get('available', {}), 'live_balance', 0):,.2f}")
        except Exception as e:
            st.error(f"Could not fetch broker details: {e}")
            
    with col2:
        st.subheader("Supabase Account Info")
        st.write(f"**Supabase User ID:** {st.session_state['user_id']}")
        st.write(f"**Supabase Email:** {st.session_state['user_session'].user.email}")

def render_portfolio_tab(kite_client: KiteConnect | None):
    st.header("Portfolio")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to view Portfolio.")
        return
    st.write("Portfolio details would be displayed here.")

def render_orders_tab(kite_client: KiteConnect | None):
    st.header("Orders")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to manage Orders.")
        return
    st.write("Order management functionalities would be here.")

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Market & Historical")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to access Market Data.")
        return
    st.write("Market data and historical charting would be here.")

def render_ml_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Machine Learning Analysis")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase for ML Analysis.")
        return
    st.write("ML-driven analysis tools would be here.")

def render_risk_stress_testing_tab(kite_client: KiteConnect | None):
    st.header("Risk & Stress Testing")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase for Risk Analysis.")
        return
    st.write("Risk assessment and stress testing tools would be here.")

def render_performance_analysis_tab(kite_client: KiteConnect | None):
    st.header("Performance Analysis")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase for Performance Analysis.")
        return
    st.write("Performance metrics and benchmarking would be here.")

def render_multi_asset_analysis_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Multi-Asset Analysis")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase for Multi-Asset Analysis.")
        return
    st.write("Correlation and diversification analysis would be here.")

def render_custom_index_tab(kite_client: KiteConnect | None, supabase_client: Client, api_key: str | None, access_token: str | None):
    st.header("Custom Index")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to create and manage Custom Indexes.")
        return
    st.write("Custom index creation and management features would be here.")

def render_websocket_tab(kite_client: KiteConnect | None):
    st.header("Websocket (stream)")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase to use WebSockets.")
        return
    st.write("Live data streaming using WebSockets would be here.")

def render_instruments_utils_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Instruments Utils")
    if not kite_client or not st.session_state["user_session"]:
        st.info("Please login to Kite and Supabase for Instrument Utilities.")
        return
    st.write("Instrument lookup and utility functions would be here.")


# --- Main Application Logic (Tab Rendering) ---
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

if tab_dashboard:
    with tab_dashboard: render_dashboard_tab(k, api_key, access_token)

if tab_portfolio:
    with tab_portfolio: render_portfolio_tab(k)
if tab_orders:
    with tab_orders: render_orders_tab(k)
if tab_market:
    with tab_market: render_market_historical_tab(k, api_key, access_token)
if tab_ml:
    with tab_ml: render_ml_analysis_tab(k, api_key, access_token)
if tab_risk:
    with tab_risk: render_risk_stress_testing_tab(k)
if tab_performance:
    with tab_performance: render_performance_analysis_tab(k)
if tab_multi_asset:
    with tab_multi_asset: render_multi_asset_analysis_tab(k, api_key, access_token)
if tab_custom_index:
    with tab_custom_index: render_custom_index_tab(k, supabase, api_key, access_token)
if tab_ws:
    with tab_ws: render_websocket_tab(k)
if tab_inst:
    with tab_inst: render_instruments_utils_tab(k, api_key, access_token)
```

### Key Changes and Reasoning:

1.  **`safe_get_numeric` Default Value**:
    *   I changed the `default_value` in `safe_get_numeric` from `0` to `None`.
    *   **Reasoning**: When Supabase inserts data, it's often better to send `NULL` for fields that are genuinely missing or `None` rather than `0`. This aligns better with the database schema and avoids potential issues if `0` has a specific meaning that is different from "missing data". Your Supabase tables should ideally allow `NULL` for numeric columns that might not always have a value (e.g., `trigger_price` if it's not set). If your tables are defined as `NOT NULL` for these columns, then `0` might be a safer default, but the error suggests `NoneType` is the problem. Let's try `None` first.

2.  **Specific Default Values for Fields**:
    *   For `order_data` and `trade_data`, when calling `safe_get_numeric`:
        *   `quantity`: `default_value=0` (Quantities are usually integers, and `0` is a sensible default if missing).
        *   `price`, `trigger_price`: `default_value=0.0` (Prices can be floats, so `0.0` is a reasonable default. If your DB column is `numeric` or `decimal`, this is fine. If it's `integer`, you'd use `0`).
    *   For `user_profile_info` (margins): `default_value=0`.
    *   **Reasoning**: This allows you to be more precise about the default for each type of numeric field.

3.  **Where the Comparison Might Happen**:
    The error `'>' not supported between instances of 'NoneType' and 'int'` strongly suggests that somewhere in the code, a comparison like `some_numeric_variable > some_integer` is happening, and `some_numeric_variable` is `None`.

    *   **My Best Guess**: It's highly likely that within the Kite API response for `orders()` or `trades()`, a field like `price`, `quantity`, or `trigger_price` is sometimes `None`. The `safe_get_numeric` function, with `default_value=None`, will correctly return `None` in these cases.
    *   The issue might arise if *after* `safe_get_numeric` returns `None`, there's some other logic that implicitly assumes it's a number and tries to compare it. For example, if you had a check like `if order['quantity'] > 0: ...` and `order['quantity']` was `None`.
    *   However, in the provided code, the direct assignment to `order_data` uses `safe_get_numeric`. This means `order_data['quantity']` will be `0` (if `default_value=0` was used). If it's `None` (if `default_value=None` was used), and the database column is `NOT NULL`, Supabase might throw an error *during insert*.

**Troubleshooting Steps if the error persists:**

1.  **Check your Supabase Table Schema:**
    *   Go to your Supabase project -> **Database** -> **Table Editor**.
    *   Inspect `public.order_history` and `public.trade_history`.
    *   Are `quantity`, `price`, and `trigger_price` columns set to `NOT NULL`?
    *   If they are `NOT NULL`, then `safe_get_numeric` *must* return a number (like `0` or `0.0`), not `None`. In this case, you should explicitly set `default_value=0` or `default_value=0.0` in your `safe_get_numeric` calls for these fields.
    *   If they *can* be `NULL`, then returning `None` from `safe_get_numeric` is fine for the Python side, and Supabase will correctly store them as `NULL`.

2.  **Look for Implicit Comparisons in Your Code**:
    Although the `order_data` and `trade_data` assignments are now safer, there might be other parts of your application (perhaps in the other tabs you haven't fully implemented yet, or in logic that runs *after* this data fetching) that perform comparisons. If the error persists, you'll need to debug line by line or add more print statements to see *exactly* when the `NoneType` and `int` are involved in a comparison.

3.  **Consider the "Data fetching" part of the error**:
    This implies the error might not be in the *saving* to Supabase, but in the *processing* of the Kite data itself. For example, if there's a function that takes the raw Kite order dictionary and tries to compute something based on `order['price'] > 100` and `order['price']` is `None`.

**Actionable Advice:**

*   **If your Supabase numeric columns are `NOT NULL`**: Revert `safe_get_numeric`'s default to `0` or `0.0` where appropriate, and ensure you are using those defaults in the code.
*   **If your Supabase numeric columns allow `NULL`**: The current code with `default_value=None` should be fine on the Python side. The error might be in a very specific comparison you've made elsewhere.

Given the error message, I'd recommend trying the version with `default_value=0` or `default_value=0.0` for `quantity`, `price`, and `trigger_price` as the most direct fix if your database columns are `NOT NULL`.
26.7s
