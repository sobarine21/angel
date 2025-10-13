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
if "current_calculated_index_data" not in st.session_state:
    st.session_state["current_calculated_index_data"] = None
if "current_calculated_index_history" not in st.session_state:
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
if "_rerun_ws" not in st.session_state:
    st.session_state["_rerun_ws"] = False
if "broker_data_fetched_and_saved" not in st.session_state:
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

def safe_get_numeric(data, key, default_value_if_none_or_error=0.0):
    """
    Safely retrieve a numerical value.
    Returns default_value_if_none_or_error if the key is not present, the value is None,
    or conversion to float fails.
    If successful, converts to int if the float is an integer, otherwise returns float.
    """
    value = data.get(key)
    if value is None:
        return default_value_if_none_or_error

    try:
        float_val = float(value)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except (ValueError, TypeError):
        return default_value_if_none_or_error

def check_supabase_response_success(response):
    """
    Check if a Supabase response was successful.
    Returns True if data exists and is not empty, False otherwise.
    """
    if response is None:
        return False
    if hasattr(response, 'data') and response.data:
        return True
    return False


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
        if st.button("Logout from Kite", key="kite_logout_btn"):
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
            if not supabase:
                st.session_state["user_session"] = None
                st.session_state["user_id"] = None
                return
                
            session_data = supabase.auth.get_session()
            if session_data and session_data.user:
                st.session_state["user_session"] = session_data
                st.session_state["user_id"] = session_data.user.id
            else:
                st.session_state["user_session"] = None
                st.session_state["user_id"] = None
        except Exception as e:
            st.session_state["user_session"] = None
            st.session_state["user_id"] = None

    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        if st.button("Logout from Supabase", key="supabase_logout_btn"):
            try:
                if supabase:
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
                            if supabase:
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
                            if supabase:
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
        st.success("Kite and Supabase Authenticated. Ready to fetch Broker Data.")
        
        if st.button("Fetch & Save Profile Data", key="fetch_save_broker_data_btn"):
            try:
                kite_client_authenticated = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
                if not kite_client_authenticated:
                    st.error("Kite client not authenticated. Please re-login.")
                    st.stop()
                
                with st.spinner("Fetching profile and margins..."):
                    profile_data = kite_client_authenticated.profile()
                    margins_data = kite_client_authenticated.margins()
                
                user_profile_info = {
                    "user_id": st.session_state["user_id"],
                    "email": profile_data.get("email"),
                    "user_name": profile_data.get("user_name"),
                    "broker": "KiteConnect",
                    "funds_available_equity_live": safe_get_numeric(margins_data.get("equity", {}).get("available", {}), "live_balance", default_value_if_none_or_error=0.0),
                    "funds_utilized_equity": safe_get_numeric(margins_data.get("equity", {}).get("utilised", {}), "overall", default_value_if_none_or_error=0.0),
                    "funds_available_commodity_live": safe_get_numeric(margins_data.get("commodity", {}).get("available", {}), "live_balance", default_value_if_none_or_error=0.0),
                    "funds_utilized_commodity": safe_get_numeric(margins_data.get("commodity", {}).get("utilised", {}), "overall", default_value_if_none_or_error=0.0),
                    "fetched_at": datetime.now().isoformat()
                }
                
                # Save to Supabase
                with st.spinner("Saving profile to Supabase..."):
                    existing_profile_query = supabase.table("user_profiles").select("id").eq("user_id", st.session_state["user_id"]).execute()
                    
                    if existing_profile_query.data:
                        profile_id_to_update = existing_profile_query.data[0]['id']
                        update_response = supabase.table("user_profiles").update(user_profile_info).eq("id", profile_id_to_update).execute()
                        if check_supabase_response_success(update_response):
                            st.success("User profile and fund data updated successfully in Supabase!")
                        else:
                            st.error(f"Failed to update user profile. Response: {update_response}")
                    else:
                        insert_response = supabase.table("user_profiles").insert(user_profile_info).execute()
                        if check_supabase_response_success(insert_response):
                            st.success("User profile and fund data saved successfully to Supabase!")
                        else:
                            st.error(f"Failed to save user profile. Response: {insert_response}")

                # Fetch and save order history
                try:
                    with st.spinner("Fetching order and trade history..."):
                        order_history = kite_client_authenticated.orders()
                        trades_history = kite_client_authenticated.trades()

                    orders_processed = 0
                    orders_failed = 0
                    
                    if order_history:
                        with st.spinner(f"Processing {len(order_history)} orders..."):
                            for order in order_history:
                                try:
                                    order_quantity = safe_get_numeric(order, "quantity", default_value_if_none_or_error=0)
                                    if not isinstance(order_quantity, int):
                                        order_quantity = int(order_quantity) if order_quantity is not None else 0

                                    order_price_raw = order.get("price")
                                    order_price = safe_get_numeric(order, "price", default_value_if_none_or_error=0.0)
                                    
                                    order_trigger_price_raw = order.get("trigger_price")
                                    order_trigger_price = safe_get_numeric(order, "trigger_price", default_value_if_none_or_error=0.0)

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
                                        "quantity": order_quantity,
                                        "price": order_price if order_price_raw is not None else None,
                                        "trigger_price": order_trigger_price if order_trigger_price_raw is not None else None,
                                        "placed_at": order["order_timestamp"],
                                        "product": order["product"],
                                        "created_at": datetime.now().isoformat()
                                    }
                                    
                                    existing_order_query = supabase.table("order_history").select("id").eq("order_id", order_data["order_id"]).execute()
                                    if existing_order_query.data:
                                        update_order_response = supabase.table("order_history").update(order_data).eq("order_id", order_data["order_id"]).execute()
                                        if check_supabase_response_success(update_order_response):
                                            orders_processed += 1
                                        else:
                                            orders_failed += 1
                                    else:
                                        insert_order_response = supabase.table("order_history").insert(order_data).execute()
                                        if check_supabase_response_success(insert_order_response):
                                            orders_processed += 1
                                        else:
                                            orders_failed += 1
                                except Exception as order_error:
                                    orders_failed += 1
                                    st.warning(f"Error processing order {order.get('order_id', 'unknown')}: {order_error}")
                        
                        if orders_processed > 0:
                            st.success(f"Successfully processed {orders_processed} orders.")
                        if orders_failed > 0:
                            st.warning(f"Failed to process {orders_failed} orders.")
                    
                    trades_processed = 0
                    trades_failed = 0
                    
                    if trades_history:
                        with st.spinner(f"Processing {len(trades_history)} trades..."):
                            for trade in trades_history:
                                try:
                                    trade_quantity = safe_get_numeric(trade, "quantity", default_value_if_none_or_error=0)
                                    if not isinstance(trade_quantity, int):
                                        trade_quantity = int(trade_quantity) if trade_quantity is not None else 0

                                    trade_price_raw = trade.get("price")
                                    trade_price = safe_get_numeric(trade, "price", default_value_if_none_or_error=0.0)

                                    trade_data = {
                                        "user_id": st.session_state["user_id"],
                                        "trade_id": trade["trade_id"],
                                        "order_id": trade["order_id"],
                                        "symbol": trade.get("tradingsymbol"),
                                        "exchange": trade["exchange"],
                                        "transaction_type": trade["transaction_type"],
                                        "quantity": trade_quantity,
                                        "price": trade_price if trade_price_raw is not None else None,
                                        "executed_at": trade["execution_time"],
                                        "product": trade["product"],
                                        "created_at": datetime.now().isoformat()
                                    }
                                    
                                    existing_trade_query = supabase.table("trade_history").select("id").eq("trade_id", trade_data["trade_id"]).execute()
                                    if existing_trade_query.data:
                                        update_trade_response = supabase.table("trade_history").update(trade_data).eq("trade_id", trade_data["trade_id"]).execute()
                                        if check_supabase_response_success(update_trade_response):
                                            trades_processed += 1
                                        else:
                                            trades_failed += 1
                                    else:
                                        insert_trade_response = supabase.table("trade_history").insert(trade_data).execute()
                                        if check_supabase_response_success(insert_trade_response):
                                            trades_processed += 1
                                        else:
                                            trades_failed += 1
                                except Exception as trade_error:
                                    trades_failed += 1
                                    st.warning(f"Error processing trade {trade.get('trade_id', 'unknown')}: {trade_error}")
                        
                        if trades_processed > 0:
                            st.success(f"Successfully processed {trades_processed} trades.")
                        if trades_failed > 0:
                            st.warning(f"Failed to process {trades_failed} trades.")

                    st.session_state["broker_data_fetched_and_saved"] = True
                    st.success("✅ All broker data fetched and saved successfully!")

                except Exception as e:
                    st.error(f"Error fetching or saving order/trade history: {e}")

            except Exception as e:
                st.error(f"An error occurred during data fetching or saving: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    else:
        st.info("Please login to Kite and Supabase to enable Broker Data Fetching.")

    st.markdown("---")
    st.markdown("### 4. Auto Redirect")
    st.caption(f"Configured redirect URL: **{AUTO_REDIRECT_URL}**")
    
    if st.session_state.get("broker_data_fetched_and_saved"):
        st.info("Data fetched and saved. Redirecting in 5 seconds...")
        st.markdown(f'<script>setTimeout(() => {{window.location.href = "{AUTO_REDIRECT_URL}";}}, 5000);</script>', unsafe_allow_html=True)
    else:
        st.info("Auto-redirect will occur after successful data fetch and save.")


# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Tab Definitions ---
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


# --- Tab Rendering Functions ---

def render_dashboard_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None, supabase_client: Client | None):
    st.header("Dashboard")
    if not kite_client or not st.session_state["user_session"] or not supabase_client:
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
            st.write(f"**Equity Available Margin:** ₹{safe_get_numeric(margins.get('equity', {}).get('available', {}), 'live_balance', default_value_if_none_or_error=0.0):,.2f}")
            st.write(f"**Commodity Available Margin:** ₹{safe_get_numeric(margins.get('commodity', {}).get('available', {}), 'live_balance', default_value_if_none_or_error=0.0):,.2f}")
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

def render_custom_index_tab(kite_client: KiteConnect | None, supabase_client: Client | None, api_key: str | None, access_token: str | None):
    st.header("Custom Index")
    if not kite_client or not st.session_state["user_session"] or not supabase_client:
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
    with tab_dashboard:
        render_dashboard_tab(k, api_key, access
