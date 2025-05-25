# app.py
# Version 1.6.3: Commented out most DEBUG print/write statements for cleaner UI.

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage
import time

# --- Page Configuration ---
st.set_page_config(page_title="Keyword Search & Analysis Tool", page_icon="🔎", layout="wide")

# --- Load Configuration ---
cfg = config.load_config()
if not cfg: 
    st.error("CRITICAL: Configuration failed to load. Application cannot proceed.")
    st.stop()
# else:
    # st.success("DEBUG: Configuration loaded successfully.") # Optional: remove

# --- Session State Initialization ---
default_session_state = {
    'processing_log': [], 'results_data': [], 'last_keywords': "",
    'last_extract_query': "", 'consolidated_summary_text': None,
    'gs_worksheet': None, 
    'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: 
        st.session_state[key] = default_value

# --- Google Sheets Setup ---
if not st.session_state.sheet_connection_attempted_this_session and \
   (cfg.gsheets.service_account_info and (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name)):
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = True
    # st.sidebar.info("Attempting to connect to Google Sheets...") # Less verbose
    st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
        cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id,
        cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name
    )
    if st.session_state.gs_worksheet:
        data_storage.ensure_master_header(st.session_state.gs_worksheet)
        # st.sidebar.success("Google Sheets connection successful.") # Less verbose
    else:
        st.session_state.sheet_writing_enabled = False
        st.sidebar.error("Google Sheets connection failed. Check logs/secrets if writing is desired.")
elif not (cfg.gsheets.service_account_info and (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name)):
    st.session_state.sheet_writing_enabled = False

# --- UI Layout ---
st.title("Keyword Search & Analysis Tool 🔎📝")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

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
    
    if not st.session_state.sheet_writing_enabled and \
       (cfg.gsheets.service_account_info or cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name):
        st.sidebar.warning("⚠️ Google Sheets connection failed or not configured correctly. Results will not be saved to sheet.")
    elif not st.session_state.sheet_writing_enabled:
         st.sidebar.caption("Google Sheets integration not configured (no secrets found).")

    start_button_val = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# --- Main Area for Results & Log ---
results_container = st.container()
log_container = st.container()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None 
    st.session_state.last_keywords = keywords_input_val
    st.session_state.last_extract_query = llm_extract_query_input_val
    keywords_list_val_runtime = [k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()]

    if not keywords_list_val_runtime: st.sidebar.error("Please enter at least one keyword."); st.stop()

    # ... (Main processing loop from v1.6.1 - this populates st.session_state.results_data) ...
    # For brevity, the detailed loop is omitted here but it should be identical to app.py v1.6.1.
    # The key part is that st.session_state.results_data is populated correctly.
    oversample_factor = 2.0; max_google_fetch_per_keyword = 10
    est_urls_to_fetch_per_keyword = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword: est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    total_llm_tasks_per_good_scrape = 0
    if llm_key_available:
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    total_major_steps_for_progress = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    current_major_step_count = 0; progress_bar_placeholder = st.empty()
    for keyword_val in keywords_list_val_runtime: # Main Processing Loop (Simplified for this paste)
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
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
                            item_data_val["llm_summary"] = summary; st.session_state.processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}..."); time.sleep(0.1)
                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1; progress_text_llm = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder: st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'...")
                            extracted_info = llm_processor.extract_specific_information(item_data_val["scraped_main_text"], extraction_query=llm_extract_query_input_val, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars)
                            item_data_val["llm_extracted_info"] = extracted_info; st.session_state.processing_log.append(f"        Extracted: {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}..."); time.sleep(0.1)
                    st.session_state.results_data.append(item_data_val)
                else: st.session_state.processing_log.append(f"    ⚠️ Scraped, but no usable main text extracted.")
            time.sleep(0.2)
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} successful scrapes."); remaining_llm_tasks_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape; current_major_step_count += remaining_llm_tasks_for_keyword
    
    with progress_bar_placeholder: st.empty()

    consolidated_summary_text_for_batch = None
    topic_for_consolidation_for_batch = "Multiple Topics / Not Specified"
    if st.session_state.results_data and llm_key_available and (enable_llm_summary_val or llm_extract_query_input_val.strip()):
        st.session_state.processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            if not keywords_list_val_runtime: topic_for_consolidation_for_batch = "the searched topics"
            elif len(keywords_list_val_runtime) == 1: topic_for_consolidation_for_batch = keywords_list_val_runtime[0]
            else: topic_for_consolidation_for_batch = f"topics: {', '.join(keywords_list_val_runtime[:3])}{'...' if len(keywords_list_val_runtime) > 3 else ''}"
            all_valid_llm_outputs = [item.get("llm_summary") or item.get("llm_extracted_info") for item in st.session_state.results_data if (item.get("llm_summary") and not str(item.get("llm_summary", "")).startswith("LLM Error")) or (item.get("llm_extracted_info") and not str(item.get("llm_extracted_info", "")).startswith("LLM Error"))]
            if not all_valid_llm_outputs:
                st.warning("No valid LLM outputs to consolidate."); consolidated_summary_text_for_batch = "Error: No valid LLM outputs for consolidation."; st.session_state.processing_log.append("  ❌ No valid LLM outputs for consolidation.")
            else:
                llm_api_key_to_use = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                llm_model_to_use = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(all_valid_llm_outputs, topic_context=topic_for_consolidation_for_batch, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars)
                st.session_state.processing_log.append(f"  Consolidated Overview: {str(consolidated_summary_text_for_batch)[:150] if consolidated_summary_text_for_batch else 'Failed/Empty'}...")
        st.session_state.consolidated_summary_text = consolidated_summary_text_for_batch

    if st.session_state.sheet_writing_enabled and st.session_state.gs_worksheet:
        if st.session_state.results_data or st.session_state.consolidated_summary_text: # Check if any data to write
            batch_process_timestamp_for_sheet = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing_log.append(f"\n💾 Writing batch data (started {batch_process_timestamp_for_sheet}) to Google Sheets...")
            extraction_query_for_sheet = st.session_state.last_extract_query if llm_extract_query_input_val.strip() else None
            write_successful = data_storage.write_batch_summary_and_items_to_sheet(
                worksheet=st.session_state.gs_worksheet, batch_timestamp=batch_process_timestamp_for_sheet,
                consolidated_summary=st.session_state.consolidated_summary_text,
                topic_context=topic_for_consolidation_for_batch,
                item_data_list=st.session_state.results_data,
                extraction_query_text=extraction_query_for_sheet
            )
            if write_successful: st.session_state.processing_log.append(f"  Batch data written to Google Sheets.")
            else: st.session_state.processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    elif st.session_state.results_data: # Only log warning if there was data but sheets wasn't enabled
        st.session_state.processing_log.append("\n⚠️ Google Sheets writing is disabled or not configured. Data not saved to sheet.")

    if st.session_state.results_data or st.session_state.consolidated_summary_text : st.success("All processing complete!")
    else: st.warning("Processing complete, but no data was generated.")

# --- Display Consolidated Summary ---
if st.session_state.get('consolidated_summary_text'):
    with results_container:
        st.markdown("---"); st.subheader("✨ Consolidated Overview Result")
        with st.container(border=True): st.markdown(st.session_state.consolidated_summary_text)

# --- Display Individual Stored Results ---
if st.session_state.results_data:
    with results_container:
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
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
                    if item_val.get("llm_summary"):
                        with st.container(border=True): st.markdown(f"**Summary (LLM):**"); st.markdown(item_val["llm_summary"])
                    if item_val.get("llm_extracted_info"):
                        with st.container(border=True): st.markdown(f"**Extracted Info (LLM) for '{st.session_state.last_extract_query}':**"); st.markdown(item_val["llm_extracted_info"])
                st.caption(f"Timestamp: {item_val.get('timestamp')}")

# --- Display Processing Log ---
if st.session_state.processing_log:
    with log_container:
        with st.expander("📜 View Processing Log", expanded=False):
            log_text = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None)

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.6.3") # Updated version

# end of app.py
