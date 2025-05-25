# app.py
# Version 1.2: Implemented oversampling for search results to improve chances of getting desired number of scrapes.

import streamlit as st
from modules import config
from modules import search_engine
from modules import scraper
import time

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
    st.error("Critical configuration failed to load. Application cannot proceed.")
    st.stop() # Halts execution if config loading failed critically

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []
if 'results_data' not in st.session_state:
    st.session_state.results_data = [] # To store structured results for display/download
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
    keywords_input_val = st.text_area(
        "Keywords (one per line or comma-separated):",
        value=st.session_state.last_keywords,
        height=150,
        key="keywords_text_area"
    )
    num_results_wanted_per_keyword = st.slider(
        "Number of successfully scraped results per keyword:",
        min_value=1, max_value=10, # User selects desired *successful* scrapes
        value=cfg.num_results_per_keyword_default,
        key="num_results_slider"
    )

    st.subheader("LLM Processing (Optional)")
    enable_llm_summary_val = st.checkbox(
        "Generate LLM Summary?",
        value=True,
        key="llm_summary_checkbox",
        disabled=not cfg.openai.api_key
    )
    llm_extract_query_input_val = st.text_input(
        "Specific info to extract with LLM:",
        value=st.session_state.last_extract_query,
        placeholder="e.g., Key technologies mentioned",
        key="llm_extract_text_input",
        disabled=not cfg.openai.api_key
    )
    if not cfg.openai.api_key:
        st.caption("OpenAI API Key not configured in secrets. LLM features disabled.")


    start_button_val = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = []
    st.session_state.last_keywords = keywords_input_val
    st.session_state.last_extract_query = llm_extract_query_input_val

    keywords_list_val = [k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list_val:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    # --- Processing Logic with Oversampling ---
    oversample_factor = 2.0 # Fetch twice as many URLs as requested for successful scrapes
    max_google_fetch_per_keyword = 10 # Google Custom Search API practical limit per call

    # Calculate total number of scrape ATTEMPTS for progress bar
    # This is based on the oversampled amount we'll try to fetch from Google
    est_urls_to_fetch_per_keyword = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword: # Ensure we fetch at least what's needed if possible
        est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword

    total_scrape_attempts_for_progress = len(keywords_list_val) * est_urls_to_fetch_per_keyword
    current_scrape_attempt_count = 0

    progress_bar = st.progress(0, text="Initializing...")

    for keyword_val in keywords_list_val:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        progress_bar.progress(
            current_scrape_attempt_count / total_scrape_attempts_for_progress if total_scrape_attempts_for_progress > 0 else 0,
            text=f"Starting keyword: {keyword_val}..."
        )

        if not (cfg.google.api_key and cfg.google.cse_id):
            st.error("Google API Key or CSE ID not configured. Cannot perform search.")
            st.session_state.processing_log.append(f"  ❌ ERROR: Google API Key or CSE ID missing. Halting for '{keyword_val}'.")
            st.stop() # Stop if essential for search

        urls_to_fetch_from_google = est_urls_to_fetch_per_keyword # Use pre-calculated value

        st.session_state.processing_log.append(f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val = search_engine.perform_search(
            query=keyword_val, api_key=cfg.google.api_key, cse_id=cfg.google.cse_id,
            num_results=urls_to_fetch_from_google
        )
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")

        successfully_scraped_for_this_keyword = 0

        if not search_results_items_val:
            st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'. Moving on.")
            current_scrape_attempt_count += urls_to_fetch_from_google # Assume these attempts were "made" for progress
            continue

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} successful scrapes for '{keyword_val}'. Skipping remaining {len(search_results_items_val) - search_item_idx} Google result(s).")
                # Add the count of skipped URLs to current_scrape_attempt_count for progress bar accuracy
                current_scrape_attempt_count += (len(search_results_items_val) - search_item_idx)
                break # Move to the next keyword

            current_scrape_attempt_count += 1 # Increment for each URL we attempt to scrape
            url_to_scrape_val = search_item_val.get('link')

            if not url_to_scrape_val:
                st.session_state.processing_log.append(f"  - Google item has no URL for '{keyword_val}'. Skipping.")
                continue

            progress_text = f"Scraping {keyword_val} ({successfully_scraped_for_this_keyword+1}/{num_results_wanted_per_keyword}): {url_to_scrape_val[:50]}... ({current_scrape_attempt_count}/{total_scrape_attempts_for_progress})"
            progress_bar.progress(
                 current_scrape_attempt_count / total_scrape_attempts_for_progress if total_scrape_attempts_for_progress > 0 else 0,
                 text=progress_text
            )

            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val = scraper.fetch_and_extract_content(url_to_scrape_val)

            # Prepare item_data regardless of scrape success for logging/partial info
            item_data_val = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val,
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "scraped_title": scraped_content_val.get('title'),
                "scraped_meta_description": scraped_content_val.get('meta_description'),
                "scraped_og_title": scraped_content_val.get('og_title'),
                "scraped_og_description": scraped_content_val.get('og_description'),
                "scraped_main_text": scraped_content_val.get('main_text'),
                "scraping_error": scraped_content_val.get('error'),
                "llm_summary": None, "llm_extracted_info": None, # Placeholders
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if scraped_content_val.get('error'):
                st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
                # Optionally store items with errors if you want to see them in results
                # st.session_state.results_data.append(item_data_val)
            else:
                # Define "good scrape": no error AND has main_text that's not an extraction failure message
                is_good_scrape = (
                    scraped_content_val.get('main_text') and
                    "could not extract main content" not in scraped_content_val.get('main_text', "").lower() and
                    "not processed for main text" not in scraped_content_val.get('main_text', "").lower()
                )
                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Successfully scraped with main text.")
                    successfully_scraped_for_this_keyword += 1
                    st.session_state.results_data.append(item_data_val) # Add to main results
                else:
                    st.session_state.processing_log.append(f"    ⚠️ Scraped, but no usable main text extracted (or content type not suitable).")
                    # Optionally store these partial successes too, if desired for review
                    # st.session_state.results_data.append(item_data_val)

            time.sleep(0.2) # Polite delay

        # After iterating all fetched Google results for a keyword (or breaking early)
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword:
            st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired successful scrapes after checking {len(search_results_items_val)} Google results.")
        # Ensure progress bar reflects attempts for this keyword if loop was exited early or had no Google results
        # If we didn't process est_urls_to_fetch_per_keyword for this keyword (e.g. break early, or no google results),
        # we need to "catch up" current_scrape_attempt_count for the progress bar.
        # The loop structure with current_scrape_attempt_count +=1 on each attempt should handle this.
        # If no Google results, earlier 'current_scrape_attempt_count += urls_to_fetch_from_google' handles it.
        # If we broke early, 'current_scrape_attempt_count += (len(search_results_items_val) - search_item_idx)' handles it.


    progress_bar.empty() # Remove progress bar
    if st.session_state.results_data:
        st.success("Processing complete!")
    else:
        st.warning("Processing complete, but no content was successfully scraped and met criteria across all keywords.")


# --- Display Stored Results ---
if st.session_state.results_data:
    with results_container:
        st.subheader(f"📊 Processed Content ({len(st.session_state.results_data)} item(s) with main text)")
        for i, item_val in enumerate(st.session_state.results_data):
            # Using a more robust way to get a display title
            display_title = item_val.get('scraped_title') or \
                            item_val.get('scraped_og_title') or \
                            item_val.get('search_title') or \
                            "Untitled"
            expander_title = f"{item_val['keyword_searched']} | {display_title} ({item_val.get('url')})"

            with st.expander(expander_title):
                st.markdown(f"**URL:** [{item_val.get('url')}]({item_val.get('url')})")

                if item_val.get('scraping_error'):
                    st.error(f"Scraping Error: {item_val['scraping_error']}")
                else:
                    st.markdown(f"**Scraped Title:** {item_val.get('scraped_title', 'N/A')}")
                    st.markdown(f"**Scraped Meta Desc:** {item_val.get('scraped_meta_description', 'N/A')}")
                    st.markdown(f"**Scraped OG Title:** {item_val.get('scraped_og_title', 'N/A')}")
                    st.markdown(f"**Scraped OG Desc:** {item_val.get('scraped_og_description', 'N/A')}")

                    if item_val.get('scraped_main_text'):
                        with st.popover("View Main Text", use_container_width=True):
                            st.text_area(
                                f"Main Text for {item_val.get('url')}",
                                value=item_val['scraped_main_text'],
                                height=400, # Increased height
                                key=f"main_text_popover_{i}", # Unique key for popover content
                                disabled=True
                            )
                    else:
                        st.caption("No main text extracted or content type not suitable.")

# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None) # Using st.code for better scrollability

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.2")

# end of app.py
