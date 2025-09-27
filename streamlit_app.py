import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import concurrent.futures
# Matplotlib is imported but Plotly is preferred for Streamlit interactive charts
# import matplotlib.pyplot as plt 
import plotly.express as px

# --- Configuration ---
API_ENDPOINTS = {
    "sebi-files": "https://chat.api.navigateaif.co.in/sebi-files/",
    "create_chat": "https://chat.api.navigateaif.co.in/create_chat",
    "get_user_chat_history": "https://chat.api.navigateaif.co.in/get_user_chat_history",
}

# --- Helper Functions for API Calls ---

def call_sebi_files(url):
    """Calls the sebi-files endpoint."""
    headers = {"accept": "application/json"}
    start_time = time.perf_counter()
    try:
        response = requests.get(url, headers=headers, timeout=10)
        end_time = time.perf_counter()
        return {
            "status_code": response.status_code,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": response.ok,
            "error_message": None if response.ok else response.text,
            "endpoint": "sebi-files"
        }
    except requests.exceptions.RequestException as e:
        end_time = time.perf_counter()
        return {
            "status_code": None,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": False,
            "error_message": str(e),
            "endpoint": "sebi-files"
        }

def call_create_chat(url):
    """Calls the create_chat endpoint with dummy data."""
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    # Dummy data for demonstration. Adjust as per actual API requirements.
    payload = {
        "user_id": "test_user_123",
        "user_query": "What is the latest SEBI circular?",
        "chat_history": []
    }
    start_time = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        end_time = time.perf_counter()
        return {
            "status_code": response.status_code,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": response.ok,
            "error_message": None if response.ok else response.text,
            "endpoint": "create_chat"
        }
    except requests.exceptions.RequestException as e:
        end_time = time.perf_counter()
        return {
            "status_code": None,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": False,
            "error_message": str(e),
            "endpoint": "create_chat"
        }

def call_get_user_chat_history(url):
    """Calls the get_user_chat_history endpoint with dummy data."""
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    # Dummy data. Adjust as per actual API requirements.
    # Note: This endpoint often expects a user_id or similar in the body/query params.
    # Assuming a POST request with a user_id for demonstration.
    payload = {"user_id": "test_user_123"}
    start_time = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        end_time = time.perf_counter()
        return {
            "status_code": response.status_code,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": response.ok,
            "error_message": None if response.ok else response.text,
            "endpoint": "get_user_chat_history"
        }
    except requests.exceptions.RequestException as e:
        end_time = time.perf_counter()
        return {
            "status_code": None,
            "response_time_ms": (end_time - start_time) * 1000,
            "success": False,
            "error_message": str(e),
            "endpoint": "get_user_chat_history"
        }

# Map endpoint names to their respective calling functions
ENDPOINT_CALL_FUNCS = {
    "sebi-files": call_sebi_files,
    "create_chat": call_create_chat,
    "get_user_chat_history": call_get_user_chat_history,
}


def run_load_test(selected_endpoint_name, num_requests, concurrency, delay_per_request, timeout_per_request):
    """
    Executes the load test.
    """
    st.info(f"Starting load test for '{selected_endpoint_name}' with {num_requests} requests, {concurrency} concurrent users, and {delay_per_request}s delay.")
    
    endpoint_url = API_ENDPOINTS[selected_endpoint_name]
    call_func = ENDPOINT_CALL_FUNCS[selected_endpoint_name]
    
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_test_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            futures.append(executor.submit(call_func, endpoint_url))
            time.sleep(delay_per_request) # Introduce a delay between starting requests
            
            # Update progress
            progress = (i + 1) / num_requests
            progress_bar.progress(progress)
            status_text.text(f"Requests initiated: {i+1}/{num_requests}")

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                st.error(f'Request generated an exception: {exc}')
            
            # This second progress bar update isn't strictly necessary visually
            # if the first one already tracks initiation well, but can be useful
            # to show completion progress for long-running individual requests.
            # However, for simplicity and clearer UI, often one progress is enough.
            # Keeping it commented out for now to avoid rapid updates potentially
            # making the UI jumpy.
            # progress_bar.progress((i + 1) / num_requests) 
            # status_text.text(f"Requests completed: {i+1}/{num_requests}")

    end_test_time = time.perf_counter()
    total_test_duration = end_test_time - start_test_time
    
    st.success("Load test completed!")
    return results, total_test_duration

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="API Load Testing Tool")

st.title("🚀 API Load Testing Tool")
st.markdown("Test the performance and resilience of your API endpoints.")

# Sidebar for configuration
st.sidebar.header("Test Configuration")
selected_endpoint_name = st.sidebar.selectbox(
    "Select API Endpoint",
    list(API_ENDPOINTS.keys())
)
st.sidebar.markdown(f"**URL:** `{API_ENDPOINTS[selected_endpoint_name]}`")

# Modified: Set max_value to 1,000,000 as requested
num_requests = st.sidebar.number_input(
    "Number of Requests to Send",
    min_value=1,
    max_value=1_000_000, # Capped at 1 Million requests
    value=1000000,
    step=10
)

concurrency = st.sidebar.slider(
    "Concurrent Users (Threads)",
    min_value=1,
    max_value=50, # Increased max for more stress
    value=5,
    step=1
)

delay_per_request = st.sidebar.number_input(
    "Delay Between Request Initiations (seconds)",
    min_value=0.0,
    value=0.01, # Small delay to avoid hammering the client machine
    step=0.001,
    format="%.3f",
    help="Delay between *initiating* each request. Total number of active requests will depend on concurrency."
)

timeout_per_request = st.sidebar.number_input(
    "Request Timeout (seconds)",
    min_value=1,
    value=10,
    step=1,
    help="Maximum time to wait for a single request to complete before timing out."
)

st.sidebar.markdown("---")
if st.sidebar.button("Start Load Test", type="primary"):
    if selected_endpoint_name and num_requests > 0 and concurrency > 0:
        # Clear previous results if a new test starts
        if 'test_results' in st.session_state:
            del st.session_state['test_results']
            del st.session_state['total_test_duration']
        
        st.session_state.running_test = True
        st.session_state.test_results, st.session_state.total_test_duration = run_load_test(
            selected_endpoint_name,
            num_requests,
            concurrency,
            delay_per_request,
            timeout_per_request
        )
        st.session_state.running_test = False
    else:
        st.sidebar.error("Please configure the test parameters correctly (Number of Requests, Concurrent Users must be > 0).")

# --- Display Results ---
if 'test_results' in st.session_state and st.session_state.test_results:
    st.header("Test Results")

    df = pd.DataFrame(st.session_state.test_results)
    
    st.subheader("Summary Statistics")
    total_requests = len(df)
    successful_requests = df['success'].sum()
    failed_requests = total_requests - successful_requests
    
    avg_response_time = df['response_time_ms'].mean()
    median_response_time = df['response_time_ms'].median()
    p90_response_time = df['response_time_ms'].quantile(0.90)
    p95_response_time = df['response_time_ms'].quantile(0.95)
    
    error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
    
    # Calculate throughput
    total_test_duration_sec = st.session_state.total_test_duration
    throughput_rps = total_requests / total_test_duration_sec if total_test_duration_sec > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", total_requests)
    with col2:
        st.metric("Successful Requests", successful_requests)
    with col3:
        st.metric("Failed Requests", failed_requests)
    with col4:
        st.metric("Error Rate", f"{error_rate:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Response Time", f"{avg_response_time:.2f} ms")
    with col6:
        st.metric("Median Response Time", f"{median_response_time:.2f} ms")
    with col7:
        st.metric("90th Percentile RT", f"{p90_response_time:.2f} ms")
    with col8:
        st.metric("95th Percentile RT", f"{p95_response_time:.2f} ms")
        
    st.metric("Total Test Duration", f"{total_test_duration_sec:.2f} seconds")
    st.metric("Throughput (Requests/Sec)", f"{throughput_rps:.2f}")

    st.subheader("Response Time Distribution")
    fig_hist = px.histogram(df, x="response_time_ms", nbins=50, 
                            title="Distribution of Response Times",
                            labels={"response_time_ms": "Response Time (ms)"},
                            height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Response Time Over Time (First 100 Requests)")
    # It's better to plot actual request completion time if possible, but for simplicity
    # we'll just plot by index. For real-world, timestamp each request.
    df_plot = df.head(100).reset_index()
    fig_scatter = px.line(df_plot, x="index", y="response_time_ms",)
  
