# app.py
import streamlit as st
from modules import config
from modules import search_engine # <-- IMPORT THE NEW MODULE
import time # Keep for mock progress for now

# --- Page Configuration ---
st.set_page_config(
    page_title="Keyword Search & Analysis Tool",
    page_icon="🔎",
    layout="wide"
)

# --- Load Configuration ---
cfg = config.load_config()
if not cfg:
    st.stop()

# --- Session State Initialization ---
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []
if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'last_keywords' not in st.session_state:
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
        value=st.session_state.last_keywords, height=150, key="keywords_text_area"
    )
    num_results_input = st.slider(
        "Number of results per keyword:", min_value=1, max_value=10,
        value=cfg.num_results_per_keyword_default, key="num_results_slider"
    )
    st.subheader("LLM Processing (Optional)")
    enable_llm_summary = st.checkbox(
        "Generate LLM Summary?", value=True, key="llm_summary_checkbox",
        disabled=not cfg.openai.api_key
    )
    llm_extract_query_input = st.text_input(
        "Specific info to extract with LLM:", value=st.session_state.last_extract_query,
        placeholder="e.g., Key technologies mentioned", key="llm_extract_text_input",
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
    st.session_state.results_data = [] # Clear previous actual results
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_query = llm_extract_query_input

    keywords_list = [k.strip() for k in keywords_input.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    # --- Actual Processing Logic ---
    total_keywords_to_process = len(keywords_list)
    progress_bar = st.progress(0, text="Initializing search...")
    current_keyword_index = 0

    all_search_results_for_display = [] # Temporary list to hold raw search results

    for keyword in keywords_list:
        current_keyword_index += 1
        progress_text = f"Searching for: {keyword} ({current_keyword_index}/{total_keywords_to_process})"
        st.session_state.processing_log.append(f"\n🔎 Searching for: {keyword}")
        progress_bar.progress(current_keyword_index / total_keywords_to_process, text=progress_text)

        # --- CALL THE SEARCH ENGINE MODULE ---
        if cfg.google.api_key and cfg.google.cse_id: # Ensure keys are loaded
            search_results = search_engine.perform_search(
                query=keyword,
                api_key=cfg.google.api_key,
                cse_id=cfg.google.cse_id,
                num_results=num_results_input
            )
            st.session_state.processing_log.append(f"Found {len(search_results)} results for '{keyword}'.")

            if search_results:
                for res_item in search_results:
                    # Store what you need for the next steps (scraping) and display
                    item_data = {
                        "keyword_searched": keyword,
                        "url": res_item.get('link'),
                        "title": res_item.get('title'),
                        "snippet": res_item.get('snippet'),
                        # Add placeholders for data from future modules
                        "scraped_main_text": None,
                        "llm_summary": None,
                        "llm_extracted_info": None,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.results_data.append(item_data) # Add to main results list
                    all_search_results_for_display.append(item_data) # Also add to temp display list
            else:
                st.session_state.processing_log.append(f"No results from Google or error for '{keyword}'.")
        else:
            st.session_state.processing_log.append(f"Skipping search for '{keyword}' due to missing Google API key/CSE ID.")
            st.error("Google API Key or CSE ID not configured. Cannot perform search.")
            st.stop() # Stop if essential for search

        time.sleep(0.1) # Small delay, purely illustrative

    progress_bar.empty() # Remove progress bar when done with all keywords

    if all_search_results_for_display:
        st.success("Search phase complete!")
    else:
        st.warning("Search phase complete, but no URLs were found across all keywords.")


# --- Display Stored Results ---
if st.session_state.results_data:
    with results_container:
        st.subheader("📊 Initial Search Results (URLs to be processed)")
        for item in st.session_state.results_data:
            with st.expander(f"{item['keyword_searched']} - {item.get('title', 'No Title')} ({item.get('url')})"):
                st.markdown(f"**URL:** [{item.get('url')}]({item.get('url')})")
                st.markdown(f"**Snippet:** {item.get('snippet', 'N/A')}")
                # We'll add more details here as other modules are built

# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.text_area("Log:", value=log_text, height=200, disabled=True, key="log_display_area")
