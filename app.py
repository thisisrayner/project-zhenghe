# app.py
# Version 1.3: Integrated Google Gemini for LLM processing.

import streamlit as st
from modules import config
from modules import search_engine
from modules import scraper
from modules import llm_processor # Import the LLM module
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
        value=st.session_state.last_keywords,
        height=150,
        key="keywords_text_area"
    )
    num_results_wanted_per_keyword = st.slider(
        "Number of successfully scraped results per keyword:",
        min_value=1, max_value=10,
        value=cfg.num_results_per_keyword_default,
        key="num_results_slider"
    )

    st.subheader(f"LLM Processing (Optional) - Provider: {cfg.llm.provider.upper()}")
    # Determine if LLM features should be enabled based on selected provider's key
    llm_key_available = False
    if cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key:
        llm_key_available = True
    elif cfg.llm.provider == "openai" and cfg.llm.openai_api_key:
        llm_key_available = True

    enable_llm_summary_val = st.checkbox(
        "Generate LLM Summary?",
        value=True,
        key="llm_summary_checkbox",
        disabled=not llm_key_available
    )
    llm_extract_query_input_val = st.text_input(
        "Specific info to extract with LLM:",
        value=st.session_state.last_extract_query,
        placeholder="e.g., Key technologies mentioned",
        key="llm_extract_text_input",
        disabled=not llm_key_available
    )
    if not llm_key_available:
        st.caption(f"{cfg.llm.provider.upper()} API Key not configured in secrets. LLM features disabled.")


    start_button_val = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = [] # Clear previous results
    st.session_state.last_keywords = keywords_input_val
    st.session_state.last_extract_query = llm_extract_query_input_val

    keywords_list_val = [k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list_val:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop()

    # --- Processing Logic ---
    oversample_factor = 2.0
    max_google_fetch_per_keyword = 10
    est_urls_to_fetch_per_keyword = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword:
        est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    # total_steps includes search, scrape, and potential LLM calls for each target successful scrape
    # For simplicity in progress, we'll focus on scrape attempts + LLM processing attempts
    total_llm_tasks_per_good_scrape = 0
    if llm_key_available:
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    
    # Max attempts = (keywords * oversampled_urls_per_keyword_for_scrape) + (keywords * wanted_good_scrapes * llm_tasks_per_good_scrape)
    total_major_steps_for_progress = (len(keywords_list_val) * est_urls_to_fetch_per_keyword) + \
                                   (len(keywords_list_val) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    current_major_step_count = 0

    progress_bar = st.progress(0, text="Initializing...")

    for keyword_val in keywords_list_val:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        # Update progress for starting a new keyword (rough estimate)
        progress_bar.progress(
            current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
            text=f"Starting keyword: {keyword_val}..."
        )

        if not (cfg.google_search.api_key and cfg.google_search.cse_id): # Corrected config access
            st.error("Google Search API Key or CSE ID not configured.")
            st.session_state.processing_log.append(f"  ❌ ERROR: Google Search API Key or CSE ID missing. Halting for '{keyword_val}'.")
            st.stop()

        urls_to_fetch_from_google = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val = search_engine.perform_search(
            query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id, # Corrected config access
            num_results=urls_to_fetch_from_google
        )
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")

        successfully_scraped_for_this_keyword = 0

        if not search_results_items_val:
            st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'. Moving on.")
            current_major_step_count += urls_to_fetch_from_google # Count these as "scrape attempts" made for progress
            continue

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} successful scrapes for '{keyword_val}'. Skipping remaining Google result(s).")
                current_major_step_count += (len(search_results_items_val) - search_item_idx) # Add skipped scrape attempts to progress
                break

            current_major_step_count += 1 # Increment for each scrape attempt
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
                    
                    # --- LLM Processing for good scrapes ---
                    if llm_key_available:
                        llm_api_key = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                        llm_model_name = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize # Default to summarize model for OpenAI if extract model not distinct

                        if enable_llm_summary_val:
                            current_major_step_count +=1 # Increment for LLM task
                            progress_text_llm = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            progress_bar.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"       Generating LLM summary ({cfg.llm.provider})...")
                            summary = llm_processor.generate_summary(
                                item_data_val["scraped_main_text"],
                                api_key=llm_api_key,
                                model_name=llm_model_name, # Pass appropriate model
                                max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_summary"] = summary
                            st.session_state.processing_log.append(f"        Summary: {summary[:100] if summary else 'Failed/Empty'}...")
                            time.sleep(0.1) # Small delay between LLM calls

                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1 # Increment for LLM task
                            progress_text_llm = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            progress_bar.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'...")
                            extracted_info = llm_processor.extract_specific_information(
                                item_data_val["scraped_main_text"],
                                extraction_query=llm_extract_query_input_val,
                                api_key=llm_api_key,
                                model_name=llm_model_name, # Pass appropriate model
                                max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_extracted_info"] = extracted_info
                            st.session_state.processing_log.append(f"        Extracted: {extracted_info[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1) # Small delay
                    
                    st.session_state.results_data.append(item_data_val) # Add to main results
                else:
                    st.session_state.processing_log.append(f"    ⚠️ Scraped, but no usable main text extracted.")
            time.sleep(0.2) # Polite delay after each scrape attempt

        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword:
            st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired successful scrapes.")
            # Account for any "promised" LLM tasks that didn't happen due to insufficient good scrapes
            remaining_llm_tasks_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword


    progress_bar.empty()
    if st.session_state.results_data:
        st.success("Processing complete!")
    else:
        st.warning("Processing complete, but no content was successfully scraped and met criteria across all keywords.")

# --- Display Stored Results ---
if st.session_state.results_data:
    with results_container:
        st.subheader(f"📊 Processed Content ({len(st.session_state.results_data)} item(s))") # Total items that passed good scrape criteria
        for i, item_val in enumerate(st.session_state.results_data):
            display_title = item_val.get('scraped_title') or item_val.get('scraped_og_title') or item_val.get('search_title') or "Untitled"
            expander_title = f"{item_val['keyword_searched']} | {display_title} ({item_val.get('url')})"

            with st.expander(expander_title):
                st.markdown(f"**URL:** [{item_val.get('url')}]({item_val.get('url')})")
                if item_val.get('scraping_error'): # Should not happen if only good scrapes are in results_data
                    st.error(f"Scraping Error: {item_val['scraping_error']}")
                
                # Display Scraped Metadata
                with st.container(border=True):
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val.get('scraped_title', 'N/A')}")
                    st.markdown(f"  - **Meta Desc:** {item_val.get('scraped_meta_description', 'N/A')}")
                    st.markdown(f"  - **OG Title:** {item_val.get('scraped_og_title', 'N/A')}")
                    st.markdown(f"  - **OG Desc:** {item_val.get('scraped_og_description', 'N/A')}")

                # Display Main Text (if available)
                if item_val.get('scraped_main_text'):
                    with st.popover("View Main Text", use_container_width=True):
                        st.text_area(f"Main Text", value=item_val['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                else:
                    st.caption("No main text was extracted or deemed usable.")

                # Display LLM Outputs (if available)
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


# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None)

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.3")

# end of app.py
