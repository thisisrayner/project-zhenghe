# app.py
# Version 1.9.0: Added LLM-powered search query generation.
# Integrates with data_storage v1.5 for unified header and batch writing.

"""
Streamlit Web Application for Keyword Search, Web Scraping, LLM Analysis, and Data Recording.
(Rest of docstring as in v1.8.1)
"""

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage
import time
import pandas as pd 
from io import BytesIO 
from typing import List, Dict, Any, Optional 
import math # Added for math.floor

# --- Page Configuration ---
# (Same as v1.8.1)
st.set_page_config(page_title="Keyword Search & Analysis Tool", page_icon="🔮", layout="wide")

# --- Load Application Configuration ---
# (Same as v1.8.1)
cfg: Optional[config.AppConfig] = config.load_config()
if not cfg: st.error("CRITICAL: Application configuration failed to load. Check secrets.toml."); st.stop()

# --- Session State Initialization ---
# (Same as v1.8.1)
default_session_state: Dict[str, Any] = {'processing_log': [], 'results_data': [], 'last_keywords': "", 'last_extract_query': "", 'consolidated_summary_text': None, 'gs_worksheet': None, 'sheet_writing_enabled': False, 'sheet_connection_attempted_this_session': False }
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Google Sheets Setup ---
# (Same as v1.8.1)
if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True 
    if cfg.gsheets.service_account_info and (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name):
        st.session_state.sheet_writing_enabled = True 
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id, cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name)
        if st.session_state.gs_worksheet: data_storage.ensure_master_header(st.session_state.gs_worksheet) 
        else: st.session_state.sheet_writing_enabled = False 
    else: st.session_state.sheet_writing_enabled = False

# --- UI Layout Definition ---
# (Same as v1.8.1)
st.title("Keyword Search & Analysis Tool 🔮")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Search Parameters")
    keywords_input_val: str = st.text_area("Keywords (one per line or comma-separated):", value=st.session_state.last_keywords, height=150, key="keywords_text_area", help="Enter each keyword or phrase on a new line, or separate them with commas.")
    num_results_wanted_per_keyword: int = st.slider("Number of successfully scraped results per keyword:", min_value=1, max_value=10, value=cfg.num_results_per_keyword_default, key="num_results_slider", help="The tool will attempt to get this many usable web pages for each keyword.")
    enable_llm_query_generation_val: bool = st.checkbox( # NEW CHECKBOX
        "✨ Enhance with LLM-generated search queries?", 
        value=True, key="llm_query_gen_checkbox", 
        help="Let the LLM generate additional search queries based on your input to broaden the search."
    )
    st.subheader(f"LLM Processing (Optional) - Provider: {cfg.llm.provider.upper()}")
    llm_key_available: bool = (cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key) or (cfg.llm.provider == "openai" and cfg.llm.openai_api_key)
    if llm_key_available: model_display_name: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize; st.caption(f"Using Model: {model_display_name}")
    else: st.caption(f"API Key for {cfg.llm.provider.upper()} not configured in secrets. LLM features disabled.")
    enable_llm_summary_val: bool = st.checkbox("Generate LLM Summary?", value=True, key="llm_summary_checkbox", disabled=not llm_key_available)
    llm_extract_query_input_val: str = st.text_input("Specific info to extract with LLM (also guides focused consolidated summary):", value=st.session_state.last_extract_query, placeholder="e.g., Key products, contact emails", key="llm_extract_text_input", disabled=not llm_key_available)
    if not st.session_state.sheet_writing_enabled:
        if cfg.gsheets.service_account_info or cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name: st.sidebar.warning("⚠️ Google Sheets: Connection failed or sheet/worksheet not found. Results will not be saved.")
        else: st.sidebar.caption("Google Sheets integration not configured (no secrets found).")
    else: st.sidebar.success(f"Google Sheets: Connected to '{st.session_state.gs_worksheet.spreadsheet.title if st.session_state.gs_worksheet else 'N/A'}' -> '{st.session_state.gs_worksheet.title if st.session_state.gs_worksheet else 'N/A'}'.")
    start_button_val: bool = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

results_container = st.container()
log_container = st.container()

def to_excel(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes:
    # (Same as v1.8.1)
    output = BytesIO();
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details') 
        if df_consolidated_summary is not None and not df_consolidated_summary.empty: df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = [] 
    st.session_state.consolidated_summary_text = None 
    st.session_state.last_keywords = keywords_input_val 
    st.session_state.last_extract_query = llm_extract_query_input_val
    
    initial_keywords_list: List[str] = [
        k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()
    ]

    if not initial_keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop() 

    keywords_list_val_runtime: List[str] = list(initial_keywords_list) # Start with user's keywords

    # --- LLM Search Query Generation ---
    if enable_llm_query_generation_val and llm_key_available and initial_keywords_list:
        st.session_state.processing_log.append("\n🧠 Attempting to generate additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list)
        num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)

        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
            llm_model_for_query_gen: str = cfg.llm.google_gemini_model # Or a specific model for this task
            
            with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."):
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                    original_keywords=initial_keywords_list,
                    specific_info_query=llm_extract_query_input_val if llm_extract_query_input_val.strip() else None,
                    num_queries_to_generate=num_llm_terms_to_generate,
                    api_key=llm_api_key_to_use,
                    model_name=llm_model_for_query_gen # Make sure this model is appropriate
                )
            
            if generated_queries:
                st.session_state.processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}")
                # Add unique generated queries to the list
                existing_keywords_set = set(k.lower() for k in keywords_list_val_runtime)
                for gq in generated_queries:
                    if gq.lower() not in existing_keywords_set:
                        keywords_list_val_runtime.append(gq)
                        existing_keywords_set.add(gq.lower())
                st.session_state.processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else:
                st.session_state.processing_log.append("  ⚠️ LLM did not generate new search queries or an error occurred. Proceeding with original keywords.")
        else:
            st.session_state.processing_log.append("  ℹ️ No additional LLM queries requested based on calculation.")
    
    # --- Recalculate Progress Bar based on potentially augmented keyword list ---
    oversample_factor: float = 2.0
    max_google_fetch_per_keyword: int = 10 
    est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword: est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: 
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    
    total_major_steps_for_progress: int = \
        (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
        (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    # If LLM query generation was enabled, add a step for it.
    if enable_llm_query_generation_val and llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        total_major_steps_for_progress +=1 
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty() 

    # Increment progress if LLM query generation ran
    if enable_llm_query_generation_val and llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0 :
        current_major_step_count +=1
        with progress_bar_placeholder.container():
             st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
                        text="LLM Query Generation Complete...")


    # --- Main Loop through keywords (original + LLM generated) ---
    # (Rest of the main processing loop, including individual item processing, 
    #  consolidated summary generation, GSheets writing, and result display,
    #  remains structurally the same as v1.8.1. Only `keywords_list_val_runtime` has changed.)
    # ...
    for keyword_val in keywords_list_val_runtime:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        with progress_bar_placeholder.container(): 
             st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
                        text=f"Starting keyword: {keyword_val}...")
        if not (cfg.google_search.api_key and cfg.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured. Cannot proceed."); st.session_state.processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'."); st.stop()
        urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id, num_results=urls_to_fetch_from_google)
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword: int = 0
        if not search_results_items_val:
            st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'."); current_major_step_count += urls_to_fetch_from_google; continue 
        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} for '{keyword_val}'. Skipping {len(search_results_items_val) - search_item_idx} Google result(s).")
                current_major_step_count += (len(search_results_items_val) - search_item_idx) ; break 
            current_major_step_count += 1 
            url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: st.session_state.processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping."); continue
            progress_text_scrape: str = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder.container(): st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_scrape)
            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
            item_data_val: Dict[str, Any] = {"keyword_searched": keyword_val, "url": url_to_scrape_val, "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'), "scraped_title": scraped_content_val.get('title'), "scraped_meta_description": scraped_content_val.get('meta_description'), "scraped_og_title": scraped_content_val.get('og_title'), "scraped_og_description": scraped_content_val.get('og_description'), "scraped_main_text": scraped_content_val.get('main_text'), "scraping_error": scraped_content_val.get('error'), "llm_summary": None, "llm_extracted_info": None, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") }
            if scraped_content_val.get('error'): st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                min_main_text_length: int = 200; current_main_text: str = scraped_content_val.get('main_text', ''); is_good_scrape: bool = (current_main_text and len(current_main_text.strip()) >= min_main_text_length and "could not extract main content" not in current_main_text.lower() and "not processed for main text" not in current_main_text.lower())
                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)})."); successfully_scraped_for_this_keyword += 1; main_text_for_llm: str = current_main_text
                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key; llm_model_to_use: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                        if enable_llm_summary_val:
                            current_major_step_count +=1 ; progress_text_llm: str = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."; with progress_bar_placeholder.container(): st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"       Generating LLM summary ({cfg.llm.provider})..."); summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars); item_data_val["llm_summary"] = summary; st.session_state.processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}..."); time.sleep(0.1) 
                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1 ; progress_text_llm: str = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."; with progress_bar_placeholder.container(): st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'..."); extracted_info: Optional[str] = llm_processor.extract_specific_information(main_text_for_llm, extraction_query=llm_extract_query_input_val, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars); item_data_val["llm_extracted_info"] = extracted_info; st.session_state.processing_log.append(f"        Extracted: {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}..."); time.sleep(0.1) 
                    st.session_state.results_data.append(item_data_val) 
                else: st.session_state.processing_log.append(f"    ⚠️ Scraped, but main text insufficient (len={len(current_main_text.strip())}). LLM processing skipped.")
            time.sleep(0.2) 
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes."); remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape; current_major_step_count += remaining_llm_tasks_for_keyword
    with progress_bar_placeholder.container(): st.empty() 
    consolidated_summary_text_for_batch: Optional[str] = None; topic_for_consolidation_for_batch: str = "Multiple Topics / Not Specified" 
    if st.session_state.results_data and llm_key_available and (enable_llm_summary_val or llm_extract_query_input_val.strip()): 
        st.session_state.processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics" # Use initial for context
            elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
            else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"
            all_valid_llm_outputs: List[str] = []; is_focused_consolidation_intended = bool(st.session_state.last_extract_query and st.session_state.last_extract_query.strip())
            for item in st.session_state.results_data:
                summary_text = item.get("llm_summary"); extraction_text = item.get("llm_extracted_info")
                is_summary_valid = summary_text and not str(summary_text).lower().startswith("llm error") and not str(summary_text).lower().startswith("no text content") and not str(summary_text).lower().startswith("llm_processor:")
                is_extraction_valid = extraction_text and not str(extraction_text).lower().startswith("llm error") and not str(extraction_text).lower().startswith("no text content") and not str(extraction_text).lower().startswith("llm_processor:")
                chosen_text_for_consolidation = None
                if is_focused_consolidation_intended:
                    if is_extraction_valid: chosen_text_for_consolidation = extraction_text
                    elif is_summary_valid: chosen_text_for_consolidation = summary_text
                else: 
                    if is_summary_valid: chosen_text_for_consolidation = summary_text
                    elif is_extraction_valid: chosen_text_for_consolidation = extraction_text
                if chosen_text_for_consolidation: all_valid_llm_outputs.append(chosen_text_for_consolidation)
            if not all_valid_llm_outputs: st.warning("No valid individual LLM outputs available for consolidated overview."); consolidated_summary_text_for_batch = "Error: No valid LLM outputs for consolidation."; st.session_state.processing_log.append("  ❌ No valid LLM outputs found.")
            else:
                llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key; llm_model_to_use: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize 
                extraction_query_context_for_consol: Optional[str] = None
                if st.session_state.last_extract_query and st.session_state.last_extract_query.strip(): extraction_query_context_for_consol = st.session_state.last_extract_query
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(all_valid_llm_outputs, topic_context=topic_for_consolidation_for_batch, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars, extraction_query_for_consolidation=extraction_query_context_for_consol )
                st.session_state.processing_log.append(f"  Consolidated Overview (first 150 chars): {str(consolidated_summary_text_for_batch)[:150] if consolidated_summary_text_for_batch else 'Failed/Empty'}...")
        st.session_state.consolidated_summary_text = consolidated_summary_text_for_batch 
    if st.session_state.sheet_writing_enabled and st.session_state.gs_worksheet:
        if st.session_state.results_data or st.session_state.consolidated_summary_text:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S"); st.session_state.processing_log.append(f"\n💾 Writing batch data to Google Sheets...")
            extraction_query_for_sheet: Optional[str] = st.session_state.last_extract_query if llm_extract_query_input_val.strip() else None
            write_successful: bool = data_storage.write_batch_summary_and_items_to_sheet(worksheet=st.session_state.gs_worksheet, batch_timestamp=batch_process_timestamp_for_sheet, consolidated_summary=st.session_state.consolidated_summary_text, topic_context=topic_for_consolidation_for_batch, item_data_list=st.session_state.results_data, extraction_query_text=extraction_query_for_sheet)
            if write_successful: st.session_state.processing_log.append(f"  Batch data written to Google Sheets.")
            else: st.session_state.processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    elif st.session_state.results_data: st.session_state.processing_log.append("\n⚠️ Google Sheets writing disabled. Data not saved to sheet.")
    if st.session_state.results_data or st.session_state.consolidated_summary_text: st.success("All processing complete!")
    else: st.warning("Processing complete, but no data was generated.")

# (Display Sections: Same as v1.8.1)
# ...
with results_container:
    if st.session_state.results_data: 
        st.markdown("---") 
        item_details_for_excel: List[Dict[str,Any]] = []
        excel_item_headers: List[str] = [ "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL", "Search Result Title", "Search Result Snippet", "Scraped Page Title", "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description", "LLM Summary (Individual)", "LLM Extracted Info (Query)", "LLM Extraction Query", "Scraping Error", "Main Text (Truncated)" ]
        for item_val_excel in st.session_state.results_data:
            row_data_excel: Dict[str, Any] = { "Batch Timestamp": item_val_excel.get("timestamp"), "Item Timestamp": item_val_excel.get("timestamp"), "Keyword Searched": item_val_excel.get("keyword_searched"), "URL": item_val_excel.get("url"), "Search Result Title": item_val_excel.get("search_title"), "Search Result Snippet": item_val_excel.get("search_snippet"), "Scraped Page Title": item_val_excel.get("scraped_title"), "Scraped Meta Description": item_val_excel.get("scraped_meta_description"), "Scraped OG Title": item_val_excel.get("scraped_og_title"), "Scraped OG Description": item_val_excel.get("scraped_og_description"), "LLM Summary (Individual)": item_val_excel.get("llm_summary"), "LLM Extracted Info (Query)": item_val_excel.get("llm_extracted_info"), "LLM Extraction Query": st.session_state.last_extract_query if item_val_excel.get("llm_extracted_info") else "", "Scraping Error": item_val_excel.get("scraping_error"), "Main Text (Truncated)": (str(item_val_excel.get("scraped_main_text", ""))[:10000] + "...") if item_val_excel.get("scraped_main_text") and len(str(item_val_excel.get("scraped_main_text", ""))) > 10000 else str(item_val_excel.get("scraped_main_text", "")) }
            item_details_for_excel.append({header: row_data_excel.get(header, "") for header in excel_item_headers})
        df_item_details = pd.DataFrame(item_details_for_excel, columns=excel_item_headers) 
        df_consolidated_summary_excel: Optional[pd.DataFrame] = None
        if st.session_state.get('consolidated_summary_text') and not str(st.session_state.consolidated_summary_text).lower().startswith("error:"):
            last_run_keywords_excel_display: List[str] = [k.strip() for k in st.session_state.last_keywords.replace(',', '\n').split('\n') if k.strip()] # Based on original input
            topic_display_excel: str = last_run_keywords_excel_display[0] if len(last_run_keywords_excel_display) == 1 else (f"Topics: {', '.join(last_run_keywords_excel_display[:3])}{'...' if len(last_run_keywords_excel_display) > 3 else ''}" if last_run_keywords_excel_display else "General Batch")
            excel_consolidation_note = "General Overview"
            if st.session_state.last_extract_query and st.session_state.last_extract_query.strip(): excel_consolidation_note = f"Focused Overview on: '{st.session_state.last_extract_query}'"
            consolidated_data_excel: Dict[str, List[Any]] = {"Batch Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")], "Topic/Keywords": [topic_display_excel], "Consolidated Summary": [st.session_state.consolidated_summary_text], "Source Items Count": [len(st.session_state.results_data)], "Consolidation Note": [excel_consolidation_note] }
            df_consolidated_summary_excel = pd.DataFrame(consolidated_data_excel)
        excel_file_bytes: bytes = to_excel(df_item_details, df_consolidated_summary_excel)
        st.download_button(label="📥 Download Results as Excel", data=excel_file_bytes, file_name=f"keyword_analysis_results_{time.strftime('%Y%m%d-%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="download_excel_button")
    if st.session_state.get('consolidated_summary_text'):
        st.markdown("---"); st.subheader("✨ Consolidated Overview Result")
        if st.session_state.last_extract_query and st.session_state.last_extract_query.strip() and not str(st.session_state.consolidated_summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"): st.caption(f"Overview focused on: '{st.session_state.last_extract_query}'.")
        elif str(st.session_state.consolidated_summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"): st.warning(f"Could not generate focused overview for '{st.session_state.last_extract_query}'.")
        with st.container(border=True): st.markdown(st.session_state.consolidated_summary_text)
    if st.session_state.results_data:
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui: str = item_val_display.get('scraped_title') or item_val_display.get('scraped_og_title') or item_val_display.get('search_title') or "Untitled"
            expander_title_ui: str = f"{item_val_display['keyword_searched']} | {display_title_ui} ({item_val_display.get('url')})"
            with st.expander(expander_title_ui):
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                if item_val_display.get('scraping_error'): st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                with st.container(border=True): st.markdown("**Scraped Metadata:**"); st.markdown(f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}\n  - **Meta Desc:** {item_val_display.get('scraped_meta_description', 'N/A')}\n  - **OG Title:** {item_val_display.get('scraped_og_title', 'N/A')}\n  - **OG Desc:** {item_val_display.get('scraped_og_description', 'N/A')}")
                if item_val_display.get('scraped_main_text'):
                    with st.popover("View Main Text", use_container_width=True): st.text_area(f"Main Text", value=item_val_display['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                else: st.caption("No main text usable for LLM processing.")
                if item_val_display.get("llm_summary") or item_val_display.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val_display.get("llm_summary"): with st.container(border=True): st.markdown(f"**Summary (LLM):**"); st.markdown(item_val_display["llm_summary"])
                    if item_val_display.get("llm_extracted_info"): with st.container(border=True): st.markdown(f"**Extracted Info (LLM) for '{st.session_state.last_extract_query}':**"); st.text(item_val_display["llm_extracted_info"]) 
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")
with log_container: 
    if st.session_state.processing_log: 
        with st.expander("📜 View Processing Log", expanded=False): st.code("\n".join(st.session_state.processing_log), language=None)
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.9.0")

# end of app.py
