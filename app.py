# app.py
import streamlit as st
from modules import config
from modules import search_engine
from modules import scraper # <-- IMPORT THE NEW MODULE
import time

# ... (Page Config, Load Config, Session State Initialization - KEEP AS IS) ...
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

# --- UI Layout (KEEP AS IS) ---
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
    st.session_state.results_data = []
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_query = llm_extract_query_input

    keywords_list = [k.strip() for k in keywords_input.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    # --- Processing Logic ---
    # Calculate total steps for progress bar: num_keywords * num_results_per_keyword
    total_urls_to_process = len(keywords_list) * num_results_input
    processed_urls_count = 0
    progress_bar = st.progress(0, text="Initializing...")

    for keyword in keywords_list:
        st.session_state.processing_log.append(f"\n🔎 Searching for: {keyword}")
        progress_bar.progress(
            processed_urls_count / total_urls_to_process if total_urls_to_process > 0 else 0,
            text=f"Searching for: {keyword}..."
        )

        if not (cfg.google.api_key and cfg.google.cse_id):
            st.error("Google API Key or CSE ID not configured. Cannot perform search.")
            st.stop()

        search_results_items = search_engine.perform_search(
            query=keyword, api_key=cfg.google.api_key, cse_id=cfg.google.cse_id,
            num_results=num_results_input
        )
        st.session_state.processing_log.append(f"Found {len(search_results_items)} Google result(s) for '{keyword}'.")

        if not search_results_items:
             # If no search results for this keyword, update progress for the expected number of URLs
            processed_urls_count += num_results_input
            progress_bar.progress(
                processed_urls_count / total_urls_to_process if total_urls_to_process > 0 else 0,
                text=f"No results for {keyword}. Moving on..."
            )
            continue # Move to the next keyword

        for search_item in search_results_items:
            url_to_scrape = search_item.get('link')
            if not url_to_scrape:
                st.session_state.processing_log.append(f"  - Skipping item with no URL for '{keyword}'.")
                processed_urls_count +=1
                continue

            progress_text = f"Scraping: {url_to_scrape[:70]}... ({processed_urls_count + 1}/{total_urls_to_process})"
            progress_bar.progress(
                (processed_urls_count + 1) / total_urls_to_process if total_urls_to_process > 0 else 0,
                text=progress_text
            )
            st.session_state.processing_log.append(f"  ➔ Scraping: {url_to_scrape}")

            # --- CALL THE SCRAPER MODULE ---
            scraped_content = scraper.fetch_and_extract_content(url_to_scrape)
            processed_urls_count += 1

            if scraped_content.get('error'):
                st.session_state.processing_log.append(f"    ❌ Error scraping {url_to_scrape}: {scraped_content['error']}")
            else:
                st.session_state.processing_log.append(f"    ✔️ Successfully scraped: {url_to_scrape}")

            # Combine search result data with scraped data
            # Initialize with basic search info in case scraping fails partially
            item_data = {
                "keyword_searched": keyword,
                "url": url_to_scrape,
                "search_title": search_item.get('title'),
                "search_snippet": search_item.get('snippet'),
                "scraped_title": scraped_content.get('title'),
                "scraped_meta_description": scraped_content.get('meta_description'),
                "scraped_og_title": scraped_content.get('og_title'),
                "scraped_og_description": scraped_content.get('og_description'),
                "scraped_main_text": scraped_content.get('main_text'),
                "scraping_error": scraped_content.get('error'),
                # Placeholders for LLM and Sheets
                "llm_summary": None,
                "llm_extracted_info": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.results_data.append(item_data)
            time.sleep(0.2) # Small polite delay between scraping URLs

    progress_bar.empty()
    if st.session_state.results_data:
        st.success("Processing complete! Found and scraped content.")
    else:
        st.warning("Processing complete, but no content was successfully scraped.")


# --- Display Stored Results ---
if st.session_state.results_data:
    with results_container:
        st.subheader("📊 Processed Content")
        for i, item in enumerate(st.session_state.results_data):
            expander_title = f"{item['keyword_searched']} | {item.get('scraped_title') or item.get('search_title', 'No Title')} ({item.get('url')})"
            with st.expander(expander_title):
                st.markdown(f"**URL:** [{item.get('url')}]({item.get('url')})")
                if item.get('scraping_error'):
                    st.error(f"Scraping Error: {item['scraping_error']}")
                st.markdown(f"**Title (Scraped):** {item.get('scraped_title', 'N/A')}")
                st.markdown(f"**Meta Desc (Scraped):** {item.get('scraped_meta_description', 'N/A')}")

                if item.get('scraped_main_text') and not item.get('scraping_error'):
                    with st.popover("View Main Text"): # New popover for text
                        st.text_area(
                            f"Main Text for {item.get('url')}",
                            value=item['scraped_main_text'],
                            height=300,
                            key=f"main_text_{i}", # Unique key
                            disabled=True
                        )
                elif not item.get('scraping_error'):
                    st.caption("No main text extracted.")


# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            # Using st.code for better scrollability and fixed-width font
            st.code(log_text, language=None)
