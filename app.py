# app.py
import streamlit as st
from modules import config # Import your config module

# --- Page Configuration ---
st.set_page_config(
    page_title="Keyword Search & Analysis Tool",
    page_icon="🔎",
    layout="wide"
)

# --- Load Configuration ---
# This should be one of the first things in your app
cfg = config.load_config()

if not cfg:
    st.stop() # Halts execution if config loading failed critically

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []
if 'results_data' not in st.session_state:
    st.session_state.results_data = [] # To store structured results for display/download
if 'last_keywords' not in st.session_state: # To repopulate input
    st.session_state.last_keywords = ""
if 'last_extract_query' not in st.session_state:
    st.session_state.last_extract_query = ""


# --- UI Layout ---
st.title("Keyword Search & Analysis Tool 🔎📝")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Search Parameters")
    keywords_input = st.text_area(
        "Keywords (one per line or comma-separated):",
        value=st.session_state.last_keywords,
        height=150,
        key="keywords_text_area" # Unique key
    )
    num_results_input = st.slider(
        "Number of results per keyword:",
        min_value=1, max_value=10,
        value=cfg.num_results_per_keyword_default,
        key="num_results_slider"
    )

    st.subheader("LLM Processing (Optional)")
    enable_llm_summary = st.checkbox(
        "Generate LLM Summary?",
        value=True,
        key="llm_summary_checkbox",
        disabled=not cfg.openai.api_key # Disable if OpenAI key not configured
    )
    llm_extract_query_input = st.text_input(
        "Specific info to extract with LLM:",
        value=st.session_state.last_extract_query,
        placeholder="e.g., Key technologies mentioned",
        key="llm_extract_text_input",
        disabled=not cfg.openai.api_key
    )
    if not cfg.openai.api_key:
        st.caption("OpenAI API Key not configured in secrets. LLM features disabled.")


    start_button = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = []
    st.session_state.last_keywords = keywords_input # Save for next run
    st.session_state.last_extract_query = llm_extract_query_input

    keywords_list = [k.strip() for k in keywords_input.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    with results_container:
        st.info(f"Processing for keywords: {', '.join(keywords_list)}. Please wait...")
        # Placeholder for actual processing logic
        # For now, just show the inputs
        st.write("Keywords to process:", keywords_list)
        st.write("Number of results per keyword:", num_results_input)
        st.write("Enable LLM Summary:", enable_llm_summary)
        st.write("LLM Extraction Query:", llm_extract_query_input)

    # Simulate some work
    import time
    progress_bar = st.progress(0, text="Initializing...")
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1, text=f"Simulating work step {i+1}/100")
    progress_bar.empty() # Remove progress bar when done
    st.success("Mock processing complete!")


# --- Display Stored Results (Example) ---
if st.session_state.results_data:
    with results_container:
        st.subheader("📊 Processed Results")
        # For now, just display raw data
        # Later, this will be a more structured display (e.g., expanders, dataframes)
        for item in st.session_state.results_data:
            st.write(item) # Replace with better formatting

# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            for entry in st.session_state.processing_log:
                st.text(entry)
