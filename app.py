# app.py
# Version 1.5: Automatic consolidated summary generation and Google Sheets integration.

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage # Added data_storage
import time

# --- Page Configuration ---
st.set_page_config(page_title="Keyword Search & Analysis Tool", page_icon="🔎", layout="wide")

# --- Load Configuration ---
cfg = config.load_config()
if not cfg: st.error("Critical configuration failed. Application cannot proceed."); st.stop()

# --- Session State Initialization ---
default_session_state = {
    'processing_log': [], 'results_data': [], 'last_keywords': "",
    'last_extract_query': "", 'consolidated_summary': None,
    'gs_worksheet': None, # For caching the worksheet object
    'sheet_writing_enabled': False # Control if writing to sheet is attempted
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Google Sheets Setup (Attempt once per session or if config changes) ---
# Determine if Sheets writing should be attempted based on config
if cfg.gsheets.service_account_info and cfg.gsheets.spreadsheet_name:
    st.session_state.sheet_writing_enabled = True
    if st.session_state.gs_worksheet is None: # Only connect if not already connected
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
            cfg.gsheets.service_account_info,
            cfg.gsheets.spreadsheet_name,
            cfg.gsheets.worksheet_name
        )
        if st.session_state.gs_worksheet:
            data_storage.ensure_header(st.session_state.gs_worksheet) # Ensure header on connect
        else:
            st.session_state.sheet_writing_enabled = False # Disable if connection failed
else:
    st.session_state.sheet_writing_enabled = False


# --- UI Layout ---
st.title("Keyword Search & Analysis Tool 🔎📝")
# ... (rest of UI layout from v1.4.1 - no major changes here, but one caption added) ...
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Search Parameters")
    keywords_input_val = st.text_area("Keywords (one per line or comma-separated):", value=st.session_state.last_keywords, height=150, key="keywords_text_area")
    num_results_wanted_per_keyword = st.slider("Number of successfully scraped results per keyword:", 1, 10, cfg.num_results_per_keyword_default, key="num_results_slider")
    
    st.subheader(f"LLM Processing (Optional) - Provider: {cfg.llm.provider.upper()}")
    llm_key_available = (cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key) or \
                        (cfg.llm.provider == "openai" and cfg.llm.openai_api_key)
    if llm_key_available:
        model_display_name = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
        st.caption(f"Using Model: {model_display_name}")
    else:
        st.caption(f"API Key for {cfg.llm.provider.upper()} not configured. LLM features disabled.")

    enable_llm_summary_val = st.checkbox("Generate LLM Summary?", value=True, key="llm_summary_checkbox", disabled=not llm_key_available)
    llm_extract_query_input_val = st.text_input("Specific info to extract with LLM:", value=st.session_state.last_extract_query, placeholder="e.g., Key technologies mentioned", key="llm_extract_text_input", disabled=not llm_key_available)
    
    if not st.session_state.sheet_writing_enabled:
        st.sidebar.caption("⚠️ Google Sheets integration not configured or failed. Results will not be saved to a sheet.")

    start_button_val = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)


# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = []
    st.session_state.consolidated_summary = None
    st.session_state.last_keywords = keywords_input_val
    st.session_state.last_extract_query = llm_extract_query_input_val
    keywords_list_val_runtime = [k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list_val_runtime: st.sidebar.error("Please enter at least one keyword."); st.stop()

    # ... (Main processing loop from v1.4.1 for search, scrape, individual LLM calls)
    # The loop populates st.session_state.results_data
    # Key change: After this loop, we'll call data_storage.write_data_to_sheet
    # and then automatically generate the consolidated summary.
    # For brevity, the detailed loop is omitted here, but it's the one from v1.4.1 that populates results_data
    oversample_factor = 2.0; max_google_fetch_per_keyword = 10
    est_urls_to_fetch_per_keyword = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword: est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    total_llm_tasks_per_good_scrape = 0
    if llm_key_available:
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    total_major_steps_for_progress = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    current_major_step_count = 0; progress_bar_placeholder = st.empty()
    for keyword_val in keywords_list_val_runtime:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}") # ... (rest of loop from 1.4.1)
        with progress_bar_placeholder: st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=f"Starting keyword: {keyword_val}...")
        if not (cfg.google_search.api_key and cfg.google_search.cse_id): st.error("Google Search API Key or CSE ID not configured."); st.session_state.processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'."); st.stop()
        urls_to_fetch_from_google = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val = search_engine.perform_search(query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id, num_results=urls_to_fetch_from_google)
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword = 0
        if not search_results_items_val: st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'. Moving on."); current_major_step_count += urls_to_fetch_from_google; continue
        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} successful scrapes for '{keyword_val}'. Skipping remaining Google result(s)."); current_major_step_count += (len(search_results_items_val) - search_item_idx); break
            current_major_step_count += 1; url_to_scrape_val = search_item_val.get('link')
            if not url_to_scrape_val: st.session_state.processing_log.append(f"  - Google item has no URL for '{keyword_val}'. Skipping."); continue
            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder: st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_scrape)
            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val = scraper.fetch_and_extract_content(url_to_scrape_val)
            item_data_val = {"keyword_searched": keyword_val, "url": url_to_scrape_val, "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'), "scraped_title": scraped_content_val.get('title'), "scraped_meta_description": scraped_content_val.get('meta_description'), "scraped_og_title": scraped_content_val.get('og_title'), "scraped_og_description": scraped_content_val.get('og_description'), "scraped_main_text": scraped_content_val.get('main_text'), "scraping_error": scraped_content_val.get('error'), "llm_summary": None, "llm_extracted_info": None, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            if scraped_content_val.get('error'): st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                is_good_scrape = (scraped_content_val.get('main_text') and "could not extract main content" not in scraped_content_val.get('main_text', "").lower() and "not processed for main text" not in scraped_content_val.get('main_text', "").lower())
                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Successfully scraped with main text.")
                    successfully_scraped_for_this_keyword += 1
                    if llm_key_available:
                        llm_api_key_to_use = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                        llm_model_to_use = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                        if enable_llm_summary_val:
                            current_major_step_count +=1; progress_text_llm = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."; 
                            with progress_bar_placeholder: st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"       Generating LLM summary ({cfg.llm.provider})...")
                            summary = llm_processor.generate_summary(item_data_val["scraped_main_text"], api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars)
                            item_data_val["llm_summary"] = summary; st.session_state.processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}..."); time.sleep(0.1) # Ensure summary is str for slicing
                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1; progress_text_llm = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder: st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'...")
                            extracted_info = llm_processor.extract_specific_information(item_data_val["scraped_main_text"], extraction_query=llm_extract_query_input_val, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars)
                            item_data_val["llm_extracted_info"] = extracted_info; st.session_state.processing_log.append(f"        Extracted: {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}..."); time.sleep(0.1) # Ensure extracted_info is str
                    st.session_state.results_data.append(item_data_val)
                else: st.session_state.processing_log.append(f"    ⚠️ Scraped, but no usable main text extracted.")
            time.sleep(0.2)
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} successful scrapes."); remaining_llm_tasks_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape; current_major_step_count += remaining_llm_tasks_for_keyword
    
    with progress_bar_placeholder: st.empty() # Clear progress bar

    # --- Write to Google Sheets (after all individual processing) ---
    if st.session_state.results_data and st.session_state.sheet_writing_enabled and st.session_state.gs_worksheet:
        st.session_state.processing_log.append(f"\n💾 Attempting to write {len(st.session_state.results_data)} items to Google Sheets...")
        # Pass the actual extraction query text if it was used
        extraction_query_for_sheet = st.session_state.last_extract_query if llm_extract_query_input_val.strip() else None
        rows_written = data_storage.write_data_to_sheet(
            st.session_state.gs_worksheet,
            st.session_state.results_data,
            extraction_query_text=extraction_query_for_sheet
        )
        st.session_state.processing_log.append(f"  {rows_written} row(s) written to Google Sheets.")
    elif st.session_state.results_data and not st.session_state.sheet_writing_enabled:
        st.session_state.processing_log.append("\n⚠️ Google Sheets writing is disabled or not configured. Data not saved to sheet.")

    # --- Automatic Consolidated Summary Generation ---
    if st.session_state.results_data and llm_key_available and \
       (enable_llm_summary_val or llm_extract_query_input_val.strip()): # Only if any LLM task was enabled
        st.session_state.processing_log.append(f"\n✨ Attempting to generate consolidated overview...")
        with st.spinner("Generating consolidated overview... This may take a moment."):
            # Use keywords from the current run
            if not keywords_list_val_runtime: topic_for_consolidation = "the searched topics"
            elif len(keywords_list_val_runtime) == 1: topic_for_consolidation = keywords_list_val_runtime[0]
            else: topic_for_consolidation = f"topics: {', '.join(keywords_list_val_runtime[:3])}{'...' if len(keywords_list_val_runtime) > 3 else ''}"

            all_individual_summaries = [item.get("llm_summary") for item in st.session_state.results_data if item.get("llm_summary") and not str(item.get("llm_summary", "")).startswith("LLM Error")]
            
            if not all_individual_summaries and llm_extract_query_input_val.strip(): # Fallback to using extracted info if no summaries
                st.session_state.processing_log.append("  No valid summaries for consolidation, attempting with extracted info.")
                all_individual_summaries = [item.get("llm_extracted_info") for item in st.session_state.results_data if item.get("llm_extracted_info") and not str(item.get("llm_extracted_info", "")).startswith("LLM Error")]

            if not all_individual_summaries:
                st.warning("No valid LLM outputs (summaries or extractions) available to consolidate.")
                st.session_state.consolidated_summary = "Error: No valid LLM outputs were available for consolidation."
                st.session_state.processing_log.append("  ❌ No valid LLM outputs for consolidation.")
            else:
                llm_api_key_to_use = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                llm_model_to_use = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                consolidated_summary_text = llm_processor.generate_consolidated_summary(
                    all_individual_summaries, topic_context=topic_for_consolidation,
                    api_key=llm_api_key_to_use, model_name=llm_model_to_use,
                    max_input_chars=cfg.llm.max_input_chars
                )
                st.session_state.consolidated_summary = consolidated_summary_text
                st.session_state.processing_log.append(f"  Consolidated Overview: {str(consolidated_summary_text)[:150] if consolidated_summary_text else 'Failed/Empty'}...")
    
    if st.session_state.results_data: st.success("All processing complete!")
    else: st.warning("Processing complete, but no content met criteria.")
    # No st.rerun() here, let Streamlit naturally update with new session state values for display.

# --- Display Consolidated Summary (if it exists in session state) ---
if st.session_state.get('consolidated_summary'):
    with results_container: # This ensures it's defined before use
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        with st.container(border=True):
            st.markdown(st.session_state.consolidated_summary)

# --- Display Individual Stored Results (remains the same as v1.4.1) ---
if st.session_state.results_data:
    with results_container:
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        # ... (rest of the individual results display loop from app.py v1.4.1) ...
        for i, item_val in enumerate(st.session_state.results_data):
            display_title = item_val.get('scraped_title') or item_val.get('scraped_og_title') or item_val.get('search_title') or "Untitled"; expander_title = f"{item_val['keyword_searched']} | {display_title} ({item_val.get('url')})"
            with st.expander(expander_title):
                st.markdown(f"**URL:** [{item_val.get('url')}]({item_val.get('url')})");
                if item_val.get('scraping_error'): st.error(f"Scraping Error: {item_val['scraping_error']}")
                with st.container(border=True): st.markdown("**Scraped Metadata:**"); st.markdown(f"  - **Title:** {item_val.get('scraped_title', 'N/A')}"); st.markdown(f"  - **Meta Desc:** {item_val.get('scraped_meta_description', 'N/A')}"); st.markdown(f"  - **OG Title:** {item_val.get('scraped_og_title', 'N/A')}"); st.markdown(f"  - **OG Desc:** {item_val.get('scraped_og_description', 'N/A')}")
                if item_val.get('scraped_main_text'):
                    with st.popover("View Main Text", use_container_width=True): st.text_area(f"Main Text", value=item_val['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                else: st.caption("No main text was extracted or deemed usable.")
                if item_val.get("llm_summary") or item_val.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val.get("llm_summary"): with st.container(border=True): st.markdown(f"**Summary ({cfg.llm.provider.upper()}):**"); st.markdown(item_val["llm_summary"])
                    if item_val.get("llm_extracted_info"): with st.container(border=True): st.markdown(f"**Extracted Info ({cfg.llm.provider.upper()}) for '{st.session_state.last_extract_query}':**"); st.markdown(item_val["llm_extracted_info"])
                st.caption(f"Timestamp: {item_val.get('timestamp')}")


# --- Display Processing Log (remains the same) ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None)

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.5")

# end of app.py
