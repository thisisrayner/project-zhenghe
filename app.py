# app.py
# Version 1.3.1: Ensured correct LLM model from config is passed.

import streamlit as st
from modules import config
from modules import search_engine
from modules import scraper
from modules import llm_processor
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Keyword Search & Analysis Tool",
    page_icon="🔎",
    layout="wide"
)

# --- Load Configuration ---
cfg = config.load_config()
if not cfg:
    st.error("Critical configuration failed to load. Application cannot proceed.")
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
    keywords_input_val = st.text_area(
        "Keywords (one per line or comma-separated):",
        value=st.session_state.last_keywords, height=150, key="keywords_text_area"
    )
    num_results_wanted_per_keyword = st.slider(
        "Number of successfully scraped results per keyword:",
        min_value=1, max_value=10,
        value=cfg.num_results_per_keyword_default, key="num_results_slider"
    )
    st.subheader(f"LLM Processing (Optional) - Provider: {cfg.llm.provider.upper()}")
    llm_key_available = False
    if cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key:
        llm_key_available = True
        st.caption(f"Using Gemini Model: {cfg.llm.google_gemini_model}") # Display configured model
    elif cfg.llm.provider == "openai" and cfg.llm.openai_api_key:
        llm_key_available = True
        st.caption(f"Using OpenAI Model (Summarize): {cfg.llm.openai_model_summarize}")

    enable_llm_summary_val = st.checkbox(
        "Generate LLM Summary?", value=True, key="llm_summary_checkbox", disabled=not llm_key_available
    )
    llm_extract_query_input_val = st.text_input(
        "Specific info to extract with LLM:", value=st.session_state.last_extract_query,
        placeholder="e.g., Key technologies mentioned", key="llm_extract_text_input", disabled=not llm_key_available
    )
    if not llm_key_available:
        st.caption(f"API Key for {cfg.llm.provider.upper()} not configured. LLM features disabled.")

    start_button_val = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button_val:
    # ... (Keep the existing processing logic from app.py v1.3 here) ...
    # The important part is how llm_model_name is determined:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = [] # Clear previous results
    st.session_state.last_keywords = keywords_input_val
    st.session_state.last_extract_query = llm_extract_query_input_val

    keywords_list_val = [k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list_val:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    oversample_factor = 2.0
    max_google_fetch_per_keyword = 10
    est_urls_to_fetch_per_keyword = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword:
        est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    total_llm_tasks_per_good_scrape = 0
    if llm_key_available:
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    
    total_major_steps_for_progress = (len(keywords_list_val) * est_urls_to_fetch_per_keyword) + \
                                   (len(keywords_list_val) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    current_major_step_count = 0
    progress_bar = st.progress(0, text="Initializing...")

    for keyword_val in keywords_list_val:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        progress_bar.progress(
            current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
            text=f"Starting keyword: {keyword_val}..."
        )
        if not (cfg.google_search.api_key and cfg.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured.")
            st.session_state.processing_log.append(f"  ❌ ERROR: Google Search API Key or CSE ID missing. Halting for '{keyword_val}'.")
            st.stop()

        urls_to_fetch_from_google = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val = search_engine.perform_search(
            query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id,
            num_results=urls_to_fetch_from_google
        )
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword = 0

        if not search_results_items_val:
            st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'. Moving on.")
            current_major_step_count += urls_to_fetch_from_google
            continue

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} successful scrapes for '{keyword_val}'. Skipping remaining Google result(s).")
                current_major_step_count += (len(search_results_items_val) - search_item_idx)
                break
            current_major_step_count += 1
            url_to_scrape_val = search_item_val.get('link')
            if not url_to_scrape_val:
                st.session_state.processing_log.append(f"  - Google item has no URL for '{keyword_val}'. Skipping.")
                continue
            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            progress_bar.progress(
                 current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
                 text=progress_text_scrape
            )
            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val = scraper.fetch_and_extract_content(url_to_scrape_val)
            item_data_val = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val,
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "scraped_title": scraped_content_val.get('title'),
                "scraped_meta_description": scraped_content_val.get('meta_description'),
                "scraped_og_title": scraped_content_val.get('og_title'),
                "scraped_og_description": scraped_content_val.get('og_description'),
                "scraped_main_text": scraped_content_val.get('main_text'),
                "scraping_error": scraped_content_val.get('error'),
                "llm_summary": None, "llm_extracted_info": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if scraped_content_val.get('error'):
                st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                is_good_scrape = (
                    scraped_content_val.get('main_text') and
                    "could not extract main content" not in scraped_content_val.get('main_text', "").lower() and
                    "not processed for main text" not in scraped_content_val.get('main_text', "").lower()
                )
                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Successfully scraped with main text.")
                    successfully_scraped_for_this_keyword += 1
                    if llm_key_available:
                        llm_api_key_to_use = None
                        llm_model_to_use = None
                        if cfg.llm.provider == "google":
                            llm_api_key_to_use = cfg.llm.google_gemini_api_key
                            llm_model_to_use = cfg.llm.google_gemini_model # This now comes from config
                        elif cfg.llm.provider == "openai":
                            llm_api_key_to_use = cfg.llm.openai_api_key
                            llm_model_to_use = cfg.llm.openai_model_summarize # Or make specific for extract too

                        if enable_llm_summary_val:
                            current_major_step_count +=1
                            progress_text_llm = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            progress_bar.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"       Generating LLM summary ({cfg.llm.provider})...")
                            summary = llm_processor.generate_summary(
                                item_data_val["scraped_main_text"],
                                api_key=llm_api_key_to_use,
                                model_name=llm_model_to_use,
                                max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_summary"] = summary
                            st.session_state.processing_log.append(f"        Summary: {summary[:100] if summary else 'Failed/Empty'}...")
                            time.sleep(0.1)
                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1
                            progress_text_llm = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            progress_bar.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'...")
                            # For OpenAI, you might want a different model for extraction if configured
                            # model_for_extraction = cfg.llm.openai_model_extract if cfg.llm.provider == "openai" else llm_model_to_use
                            extracted_info = llm_processor.extract_specific_information(
                                item_data_val["scraped_main_text"],
                                extraction_query=llm_extract_query_input_val,
                                api_key=llm_api_key_to_use,
                                model_name=llm_model_to_use, # Or model_for_extraction
                                max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_extracted_info"] = extracted_info
                            st.session_state.processing_log.append(f"        Extracted: {extracted_info[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1)
                    st.session_state.results_data.append(item_data_val)
                else:
                    st.session_state.processing_log.append(f"    ⚠️ Scraped, but no usable main text extracted.")
            time.sleep(0.2)
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword:
            st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired successful scrapes.")
            remaining_llm_tasks_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword
    progress_bar.empty()
    if st.session_state.results_data:
        st.success("Processing complete!")
    else:
        st.warning("Processing complete, but no content was successfully scraped and met criteria.")

# --- Display Stored Results (Keep as is from v1.3) ---
if st.session_state.results_data:
    with results_container:
        st.subheader(f"📊 Processed Content ({len(st.session_state.results_data)} item(s))")
        for i, item_val in enumerate(st.session_state.results_data):
            display_title = item_val.get('scraped_title') or item_val.get('scraped_og_title') or item_val.get('search_title') or "Untitled"
            expander_title = f"{item_val['keyword_searched']} | {display_title} ({item_val.get('url')})"
            with st.expander(expander_title):
                st.markdown(f"**URL:** [{item_val.get('url')}]({item_val.get('url')})")
                if item_val.get('scraping_error'):
                    st.error(f"Scraping Error: {item_val['scraping_error']}")
                with st.container(border=True):
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val.get('scraped_title', 'N/A')}")
                    st.markdown(f"  - **Meta Desc:** {item_val.get('scraped_meta_description', 'N/A')}")
                    st.markdown(f"  - **OG Title:** {item_val.get('scraped_og_title', 'N/A')}")
                    st.markdown(f"  - **OG Desc:** {item_val.get('scraped_og_description', 'N/A')}")
                if item_val.get('scraped_main_text'):
                    with st.popover("View Main Text", use_container_width=True):
                        st.text_area(f"Main Text", value=item_val['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                else:
                    st.caption("No main text was extracted or deemed usable.")
                if item_val.get("llm_summary") or item_val.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val.get("llm_summary"):
                        with st.container(border=True):
                             st.markdown(f"**Summary ({cfg.llm.provider.upper()}):**")
                             st.markdown(item_val["llm_summary"])
                    if item_val.get("llm_extracted_info"):
                        with st.container(border=True):
                            st.markdown(f"**Extracted Info ({cfg.llm.provider.upper()}) for '{st.session_state.last_extract_query}':**")
                            st.markdown(item_val["llm_extracted_info"])
                st.caption(f"Timestamp: {item_val.get('timestamp')}")

# --- Display Processing Log (Keep as is from v1.3) ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None)

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.3.1")

# end of app.py
