# app.py
# Version 2.0.2: Final syntax fix for progress text assignment.
# Includes all UI refinements from v2.0.0 (Sidebar changes, dynamic button).
# Features: PDF text extraction, consistent keyword input, LLM query gen, tuple passing for cache, relevancy emojis.

"""
Streamlit Web Application for Keyword Search & Analysis Tool (KSAT).

This application allows users to:
1. Input keywords for searching via Google Custom Search.
2. Scrape metadata and main content from the search result URLs (supports HTML & PDF text).
3. Utilize a Large Language Model (LLM, e.g., Google Gemini) to:
    a. Generate individual summaries for each scraped page/document (always on).
    b. Extract specific user-defined information from each page/document (with relevancy score).
    c. Create a consolidated overview summary from all processed items in a batch,
       potentially focused by the user's extraction query and item relevancy.
    d. Generate alternative search queries to diversify search scope (always on).
4. Display the processed information and LLM insights (with relevancy emojis) in an interactive UI.
5. Store the detailed results (including a batch summary row and individual item rows)
   into a specified Google Sheet (if configured).
6. Download the results (item details and consolidated summary) as an Excel file.

The application is structured modularly, with separate Python files in the 'modules'
directory handling configuration, search, scraping, LLM processing, and data storage.
API keys and sensitive settings are managed via Streamlit Secrets (`.streamlit/secrets.toml`).
"""

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage
import time
import pandas as pd 
from io import BytesIO 
from typing import List, Dict, Any, Optional, Set 
import math 

# --- Helper function for Display Logic ---
def get_display_prefix_for_item(item_data: Dict[str, Any], llm_generated_keywords: Set[str]) -> str:
    prefix = "" 
    llm_extracted_info = item_data.get("llm_extracted_info")
    score: Optional[int] = None
    if llm_extracted_info and llm_extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = llm_extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError): score = None 
    if score is not None:
        is_llm_keyword = item_data.get('keyword_searched', '').lower() in {k.lower() for k in llm_generated_keywords}
        if score == 5: prefix = "5️⃣ "
        elif score == 4: prefix = "4️⃣ "
        elif score == 3: prefix = "✨3️⃣ " if is_llm_keyword else "3️⃣ "
    return prefix

# --- Page Configuration ---
st.set_page_config(page_title="Keyword Search & Analysis Tool (KSAT)", page_icon="🔮", layout="wide") 

# --- Load Application Configuration ---
cfg: Optional[config.AppConfig] = config.load_config()
if not cfg: st.error("CRITICAL: Application configuration failed to load. Check secrets.toml."); st.stop()

# --- Session State Initialization ---
default_session_state: Dict[str, Any] = { 
    'processing_log': [], 'results_data': [], 
    'last_keywords': "", 'last_extract_query': "", 
    'consolidated_summary_text': None, 'gs_worksheet': None, 
    'sheet_writing_enabled': False, 
    'sheet_connection_attempted_this_session': False,
    'gsheets_error_message': None, 
    'initial_keywords_for_display': set(), 
    'llm_generated_keywords_set_for_display': set()
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Google Sheets Setup ---
gsheets_secrets_present = cfg.gsheets.service_account_info and (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name)
if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True 
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id, cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name )
        if st.session_state.gs_worksheet: data_storage.ensure_master_header(st.session_state.gs_worksheet); st.session_state.sheet_writing_enabled = True 
        else: st.session_state.gsheets_error_message = "Google Sheets connection failed. Check Sheet ID/Name & sharing."
    else: st.session_state.gsheets_error_message = "Google Sheets not configured in secrets.toml. Search disabled."

# --- UI Layout Definition ---
st.title("Keyword Search & Analysis Tool (KSAT) 🔮") 
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

with st.sidebar:
    st.subheader("Search Parameters")
    keywords_input_val: str = st.text_input("Keywords (comma-separated):", value=st.session_state.last_keywords, key="keywords_text_input_main", help="Enter comma-separated keywords. Press Enter to apply.")
    num_results_wanted_per_keyword: int = st.slider("Number of successfully scraped results per keyword:", 1, 10, cfg.num_results_per_keyword_default, key="num_results_slider")
    
    st.subheader(f"LLM Processing - Provider: {cfg.llm.provider.upper()}")
    llm_key_available: bool = (cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key) or (cfg.llm.provider == "openai" and cfg.llm.openai_api_key)
    if llm_key_available: model_display_name: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize; st.caption(f"Using Model: {model_display_name}")
    else: st.caption(f"API Key for {cfg.llm.provider.upper()} not configured. LLM features disabled.")
    
    llm_extract_query_input_val: str = st.text_input("Specific info to extract with LLM (guides focused summary):", value=st.session_state.last_extract_query, placeholder="e.g., Key products, contact emails", key="llm_extract_text_input", help="Comma-separated. Press Enter to apply.") 
    
    st.markdown("---") 

    button_streamlit_type = "secondary" 
    button_disabled = True
    button_help_text = st.session_state.gsheets_error_message or "Google Sheets connection status undetermined." 

    if st.session_state.sheet_writing_enabled:
        button_streamlit_type = "primary" 
        button_disabled = False
        button_help_text = "Google Sheets connected. Click to start processing."
    elif st.session_state.gsheets_error_message: 
        st.error(st.session_state.gsheets_error_message) 
    
    start_button_val: bool = st.button(
        "🚀 Start Search & Analysis", 
        type=button_streamlit_type, 
        use_container_width=True, 
        disabled=button_disabled,
        help=button_help_text
    )

    st.markdown("---") 
    st.caption("✨ LLM-generated search queries will be automatically used (if LLM is available).")
    st.caption("📄 LLM Summaries for items will be automatically generated (if LLM is available).")

# --- Custom CSS for Green Button (Applied globally if type is primary and not disabled) ---
green_button_css = """
<style>
div[data-testid="stButton"] > button:not(:disabled)[kind="primary"] {
    background-color: #4CAF50; color: white; border: 1px solid #4CAF50;
}
div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:hover {
    background-color: #45a049; color: white; border: 1px solid #45a049;
}
div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:active {
    background-color: #3e8e41; color: white; border: 1px solid #3e8e41;
}
/* Ensure disabled secondary button is clearly grey */
div[data-testid="stButton"] > button:disabled[kind="secondary"] {
    background-color: #f0f2f6; color: rgba(38, 39, 48, 0.4); border: 1px solid rgba(38, 39, 48, 0.2);
}
</style>
"""
st.markdown(green_button_css, unsafe_allow_html=True)

results_container = st.container()
log_container = st.container()

def to_excel(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes:
    output = BytesIO();
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details') 
        if df_consolidated_summary is not None and not df_consolidated_summary.empty: df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

if start_button_val:
    st.session_state.processing_log = ["Processing started..."]; st.session_state.results_data = []; st.session_state.consolidated_summary_text = None 
    st.session_state.last_keywords = keywords_input_val; st.session_state.last_extract_query = llm_extract_query_input_val
    initial_keywords_list: List[str] = [ k.strip() for k in keywords_input_val.split(',') if k.strip() ]
    st.session_state.initial_keywords_for_display = set(k.lower() for k in initial_keywords_list) 
    st.session_state.llm_generated_keywords_set_for_display = set() 
    if not initial_keywords_list: st.sidebar.error("Please enter at least one keyword."); st.stop() 
    keywords_list_val_runtime: List[str] = list(initial_keywords_list) 
    enable_llm_query_generation_val_runtime = True 
    if enable_llm_query_generation_val_runtime and llm_key_available and initial_keywords_list:
        st.session_state.processing_log.append("\n🧠 Generating additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key; llm_model_for_query_gen: str = cfg.llm.google_gemini_model 
            with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."):
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(original_keywords=tuple(initial_keywords_list), specific_info_query=llm_extract_query_input_val if llm_extract_query_input_val.strip() else None, num_queries_to_generate=num_llm_terms_to_generate, api_key=llm_api_key_to_use, model_name=llm_model_for_query_gen )
            if generated_queries:
                st.session_state.processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}")
                current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                for gq in generated_queries:
                    if gq.lower() not in current_runtime_keywords_lower: keywords_list_val_runtime.append(gq); current_runtime_keywords_lower.add(gq.lower()); temp_llm_generated_set.add(gq.lower()) 
                st.session_state.llm_generated_keywords_set_for_display = temp_llm_generated_set
                st.session_state.processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else: st.session_state.processing_log.append("  ⚠️ LLM did not generate new queries.")
        else: st.session_state.processing_log.append("  ℹ️ No additional LLM queries requested.")
    oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10 ; est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword: est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    enable_llm_summary_val_runtime = True 
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: 
        if enable_llm_summary_val_runtime: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if enable_llm_query_generation_val_runtime and llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: total_major_steps_for_progress +=1 
    current_major_step_count: int = 0; progress_bar_placeholder = st.empty() 
    if enable_llm_query_generation_val_runtime and llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0 :
        current_major_step_count +=1
        with progress_bar_placeholder.container(): st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text="LLM Query Generation Complete...")
    
    for keyword_val in keywords_list_val_runtime:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        with progress_bar_placeholder.container(): 
             st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=f"Starting keyword: {keyword_val}...")
        if not (cfg.google_search.api_key and cfg.google_search.cse_id): st.error("Google Search API Key or CSE ID not configured."); st.session_state.processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'."); st.stop()
        urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(f"  Attempting to fetch {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id, num_results=urls_to_fetch_from_google)
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword: int = 0
        if not search_results_items_val: st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'."); current_major_step_count += urls_to_fetch_from_google; continue 
        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} for '{keyword_val}'. Skipping {len(search_results_items_val) - search_item_idx} Google result(s)."); current_major_step_count += (len(search_results_items_val) - search_item_idx) ; break 
            current_major_step_count += 1 ; url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: st.session_state.processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping."); continue
            
            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder.container(): 
                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_scrape)
            
            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
            item_data_val: Dict[str, Any] = {"keyword_searched": keyword_val, "url": url_to_scrape_val, "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'), "scraped_title": scraped_content_val.get('scraped_title'), "meta_description": scraped_content_val.get('meta_description'), "og_title": scraped_content_val.get('og_title'), "og_description": scraped_content_val.get('og_description'), "scraped_main_text": scraped_content_val.get('main_text'), "scraping_error": scraped_content_val.get('error'), "content_type": scraped_content_val.get('content_type'), "llm_summary": None, "llm_extracted_info": None, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") }
            if scraped_content_val.get('error'): st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                min_main_text_length: int = 200; current_main_text: str = scraped_content_val.get('main_text', ''); is_good_scrape: bool = (current_main_text and len(current_main_text.strip()) >= min_main_text_length and "could not extract main content" not in current_main_text.lower() and "not processed for main text" not in current_main_text.lower() and not str(current_main_text).startswith("SCRAPER_INFO:"))
                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {item_data_val.get('content_type')})."); successfully_scraped_for_this_keyword += 1; main_text_for_llm: str = current_main_text
                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key; llm_model_to_use: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                        if enable_llm_summary_val_runtime: 
                            current_major_step_count +=1
                            progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container():
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_summary)
                            st.session_state.processing_log.append(f"       Generating LLM summary..."); summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars); item_data_val["llm_summary"] = summary; st.session_state.processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}..."); time.sleep(0.1) 
                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1
                            progress_text_llm_extract = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container():
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_extract)
                            st.session_state.processing_log.append(f"      Extracting info: '{llm_extract_query_input_val}'..."); extracted_info: Optional[str] = llm_processor.extract_specific_information(main_text_for_llm, extraction_query=llm_extract_query_input_val, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars); item_data_val["llm_extracted_info"] = extracted_info; st.session_state.processing_log.append(f"        Extracted: {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}..."); time.sleep(0.1) 
                    st.session_state.results_data.append(item_data_val) 
                else: st.session_state.processing_log.append(f"    ⚠️ Scraped, but main text insufficient (len={len(current_main_text.strip())}, type: {item_data_val.get('content_type')}). LLM processing skipped.")
            time.sleep(0.2) 
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: st.session_state.processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes."); remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape; current_major_step_count += remaining_llm_tasks_for_keyword
    with progress_bar_placeholder.container(): st.empty() 
    consolidated_summary_text_for_batch: Optional[str] = None; topic_for_consolidation_for_batch: str = "Multiple Topics / Not Specified" 
    if st.session_state.results_data and llm_key_available and (enable_llm_summary_val_runtime or llm_extract_query_input_val.strip()): 
        st.session_state.processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics" 
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
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(summaries=tuple(all_valid_llm_outputs), topic_context=topic_for_consolidation_for_batch, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars, extraction_query_for_consolidation=extraction_query_context_for_consol )
                st.session_state.processing_log.append(f"  Consolidated Overview (first 150 chars): {str(consolidated_summary_text_for_batch)[:150] if consolidated_summary_text_for_batch else 'Failed/Empty'}...")
        st.session_state.consolidated_summary_text = consolidated_summary_text_for_batch 
    if st.session_state.sheet_writing_enabled : 
        if st.session_state.results_data or st.session_state.consolidated_summary_text:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S"); st.session_state.processing_log.append(f"\n💾 Writing batch data to Google Sheets...")
            extraction_query_for_sheet: Optional[str] = st.session_state.last_extract_query if llm_extract_query_input_val.strip() else None
            write_successful: bool = data_storage.write_batch_summary_and_items_to_sheet(worksheet=st.session_state.gs_worksheet, batch_timestamp=batch_process_timestamp_for_sheet, consolidated_summary=st.session_state.consolidated_summary_text, topic_context=topic_for_consolidation_for_batch, item_data_list=st.session_state.results_data, extraction_query_text=extraction_query_for_sheet)
            if write_successful: st.session_state.processing_log.append(f"  Batch data written to Google Sheets.")
            else: st.session_state.processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    elif gsheets_secrets_present and not st.session_state.sheet_writing_enabled : 
        st.session_state.processing_log.append("\n⚠️ Google Sheets connection failed earlier. Data not saved to sheet.")
    elif not gsheets_secrets_present: 
        st.session_state.processing_log.append("\nℹ️ Google Sheets integration not configured. Data not saved to sheet.")
    if st.session_state.results_data or st.session_state.consolidated_summary_text: st.success("All processing complete!")
    else: st.warning("Processing complete, but no data was generated.")
with results_container:
    if st.session_state.results_data: 
        st.markdown("---") 
        item_details_for_excel: List[Dict[str,Any]] = []
        excel_item_headers: List[str] = [ "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL", "Search Result Title", "Search Result Snippet", "Scraped Page Title", "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description", "Content Type", "LLM Summary (Individual)", "LLM Extracted Info (Query)", "LLM Extraction Query", "Scraping Error", "Main Text (Truncated)" ] 
        for item_val_excel in st.session_state.results_data:
            row_data_excel: Dict[str, Any] = { "Batch Timestamp": item_val_excel.get("timestamp"), "Item Timestamp": item_val_excel.get("timestamp"), "Keyword Searched": item_val_excel.get("keyword_searched"), "URL": item_val_excel.get("url"), "Search Result Title": item_val_excel.get("search_title"), "Search Result Snippet": item_val_excel.get("search_snippet"), "Scraped Page Title": item_val_excel.get("scraped_title"), "Scraped Meta Description": item_val_excel.get("meta_description"), "Scraped OG Title": item_val_excel.get("og_title"), "Scraped OG Description": item_val_excel.get("og_description"), "Content Type": item_val_excel.get("content_type"), "LLM Summary (Individual)": item_val_excel.get("llm_summary"), "LLM Extracted Info (Query)": item_val_excel.get("llm_extracted_info"), "LLM Extraction Query": st.session_state.last_extract_query if item_val_excel.get("llm_extracted_info") else "", "Scraping Error": item_val_excel.get("scraping_error"), "Main Text (Truncated)": (str(item_val_excel.get("scraped_main_text", ""))[:10000] + "...") if item_val_excel.get("scraped_main_text") and len(str(item_val_excel.get("scraped_main_text", ""))) > 10000 else str(item_val_excel.get("scraped_main_text", "")) }
            item_details_for_excel.append({header: row_data_excel.get(header, "") for header in excel_item_headers})
        df_item_details = pd.DataFrame(item_details_for_excel, columns=excel_item_headers) 
        df_consolidated_summary_excel: Optional[pd.DataFrame] = None
        if st.session_state.get('consolidated_summary_text') and not str(st.session_state.consolidated_summary_text).lower().startswith("error:"):
            last_run_keywords_excel_display: List[str] = [k.strip() for k in st.session_state.last_keywords.split(',') if k.strip()] 
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
        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui: str = item_val_display.get('scraped_title') or item_val_display.get('og_title') or item_val_display.get('search_title') or "Untitled"
            display_prefix = get_display_prefix_for_item(item_val_display, llm_gen_kws_for_display)
            content_type_marker = "📄" if 'pdf' in item_val_display.get('content_type', '') else "" 
            expander_title_ui = f"{display_prefix}{content_type_marker}{item_val_display['keyword_searched']} | {display_title_ui} ({item_val_display.get('url')})"
            with st.expander(expander_title_ui):
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                st.caption(f"Content Type: {item_val_display.get('content_type', 'N/A')}") 
                if item_val_display.get('scraping_error'): st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                with st.container(border=True): 
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}\n  - **Meta Desc:** {item_val_display.get('meta_description', 'N/A')}\n  - **OG Title:** {item_val_display.get('og_title', 'N/A')}\n  - **OG Desc:** {item_val_display.get('og_description', 'N/A')}")
                if item_val_display.get('scraped_main_text') and not str(item_val_display.get('scraped_main_text','')).startswith("SCRAPER_INFO:"):
                    with st.popover("View Main Text", use_container_width=True): 
                        st.text_area(f"Main Text ({item_val_display.get('content_type')})", value=item_val_display['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                elif str(item_val_display.get('scraped_main_text','')).startswith("SCRAPER_INFO:"): st.caption(item_val_display['scraped_main_text'])
                else: st.caption("No main text extracted or usable for LLM processing.")
                if item_val_display.get("llm_summary") or item_val_display.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val_display.get("llm_summary"):
                        with st.container(border=True): st.markdown(f"**Summary (LLM):**"); st.markdown(item_val_display["llm_summary"])
                    if item_val_display.get("llm_extracted_info"): 
                        with st.container(border=True): st.markdown(f"**Extracted Info (LLM) for '{st.session_state.last_extract_query}':**"); st.text(item_val_display["llm_extracted_info"]) 
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")
with log_container: 
    if st.session_state.processing_log: 
        with st.expander("📜 View Processing Log", expanded=False): st.code("\n".join(st.session_state.processing_log), language=None)
st.markdown("---")
st.caption("Keyword Search & Analysis Tool (KSAT) v2.0.2")

# end of app.py
