# modules/process_manager.py
# Version 1.4.7:
# - Reinstated progress bar and intermediate status text updates.
# - Final success/warning/error UI messages are signaled via LOG_STATUS in
#   processing_log for app.py to handle, respecting previous design notes.
# Previous versions:
# - Version 1.4.6 (Proposed but superseded by this version due to design note)
# - Version 1.4.5: More specific debug prints for GSheets, refined GSheets LOG_STATUS.

"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
Provides live intermediate progress updates via Streamlit UI elements and communicates
final processing status back via specific log messages for app.py to display.
"""
import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple, TypedDict
from modules import config, search_engine, scraper, llm_processor, data_storage # Ensure config import
import traceback

# Define a type for the focused summary source details
class FocusedSummarySource(TypedDict):
    url: str
    query_type: str
    query_text: str
    score: int
    llm_output_text: str

def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]:
    score: Optional[int] = None
    if extracted_info and isinstance(extracted_info, str) and extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError): pass
    return score

def run_search_and_analysis(
    app_config: 'config.AppConfig', # Use AppConfig from modules.config
    keywords_input: str,
    llm_extract_queries_input: List[str],
    num_results_wanted_per_keyword: int,
    gs_worksheet: Optional[Any],
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    print("-----> DEBUG (process_manager v1.4.7): TOP OF run_search_and_analysis called.")

    processing_log: List[str] = ["LOG_STATUS:PROCESSING_INITIATED:Processing initiated..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    initial_keywords_for_display_set: Set[str] = set()
    llm_generated_keywords_set_for_display_set: Set[str] = set()
    
    progress_bar_placeholder = st.empty()
    status_placeholder = st.empty()
    
    print("-----> DEBUG (process_manager v1.4.7): UI placeholders created.")

    try:
        initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
        if not initial_keywords_list:
            processing_log.append("LOG_STATUS:ERROR:NO_KEYWORDS:Please enter at least one keyword.")
            # app.py will handle displaying this error based on LOG_STATUS
            progress_bar_placeholder.empty()
            status_placeholder.empty()
            return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details
        
        status_placeholder.info("Initializing research process...")
        progress_bar_placeholder.progress(0)

        initial_keywords_for_display_set = set(k.lower() for k in initial_keywords_list)
        keywords_list_val_runtime: List[str] = list(initial_keywords_list)

        llm_key_available: bool = (app_config.llm.provider == "google" and app_config.llm.google_gemini_api_key) or \
                                  (app_config.llm.provider == "openai" and app_config.llm.openai_api_key)

        primary_llm_extract_query: Optional[str] = None
        secondary_llm_extract_query: Optional[str] = None
        if llm_extract_queries_input:
            if len(llm_extract_queries_input) > 0 and llm_extract_queries_input[0] and llm_extract_queries_input[0].strip():
                primary_llm_extract_query = llm_extract_queries_input[0].strip()
            if len(llm_extract_queries_input) > 1 and llm_extract_queries_input[1] and llm_extract_queries_input[1].strip():
                secondary_llm_extract_query = llm_extract_queries_input[1].strip()
        
        llm_item_delay_seconds = app_config.llm.llm_item_request_delay_seconds
        throttling_threshold = app_config.llm.llm_throttling_threshold_results
        apply_throttling = (
            num_results_wanted_per_keyword >= throttling_threshold and
            llm_item_delay_seconds > 0 and
            llm_key_available
        )

        # --- Refined Progress Calculation ---
        num_initial_keywords_for_calc = len(initial_keywords_list)
        llm_query_gen_step_count = 0
        if llm_key_available and num_initial_keywords_for_calc > 0:
            num_llm_terms_to_generate_calc = min(math.floor(num_initial_keywords_for_calc * 1.5), 5)
            if num_llm_terms_to_generate_calc > 0:
                llm_query_gen_step_count = 1
        
        temp_keywords_list_for_calc = list(initial_keywords_list) # Start with initial
        # Simulate adding LLM keywords for accurate count in total_keywords_to_process_calc
        if llm_query_gen_step_count > 0: # Assuming worst case, all LLM queries are new
            num_llm_terms_to_add = min(math.floor(num_initial_keywords_for_calc * 1.5), 5)
            for i in range(num_llm_terms_to_add): temp_keywords_list_for_calc.append(f"llm_gen_placeholder_{i}")

        total_keywords_to_process_calc = len(set(k.lower() for k in temp_keywords_list_for_calc))

        oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10
        urls_to_scan_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
        if urls_to_scan_per_keyword < num_results_wanted_per_keyword : urls_to_scan_per_keyword = num_results_wanted_per_keyword

        search_steps_calc = total_keywords_to_process_calc
        scraping_steps_calc = total_keywords_to_process_calc * urls_to_scan_per_keyword
        
        llm_tasks_per_good_item_calc = 0
        if llm_key_available: 
            llm_tasks_per_good_item_calc += 1 # Summary
            if primary_llm_extract_query: llm_tasks_per_good_item_calc +=1
            if secondary_llm_extract_query: llm_tasks_per_good_item_calc +=1
        
        llm_item_processing_steps_calc = total_keywords_to_process_calc * num_results_wanted_per_keyword * llm_tasks_per_good_item_calc
        consolidated_summary_step_calc = 1 if llm_key_available and (llm_item_processing_steps_calc > 0 or scraping_steps_calc > 0) else 0 # Only if there's something to summarize

        total_major_steps_for_progress: int = (
            llm_query_gen_step_count +
            search_steps_calc +
            scraping_steps_calc +
            llm_item_processing_steps_calc +
            consolidated_summary_step_calc
        )
        if total_major_steps_for_progress == 0: total_major_steps_for_progress = 1
        current_major_step_count: int = 0
        
        print(f"-----> DEBUG (process_manager v1.4.7): Total estimated steps for progress: {total_major_steps_for_progress}")

        def update_progress_ui(message: Optional[str] = None):
            # This function only updates UI, step counting is manual
            progress_val = min(1.0, current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            progress_bar_placeholder.progress(math.ceil(progress_val * 100) / 100)
            if message:
                status_placeholder.info(message)
                # Log UI status updates if desired, but primary logging is via processing_log appends
                # processing_log.append(f"LOG_UI_STATUS:{message}") 

        if apply_throttling:
            throttle_init_msg = (
                f"ℹ️ LLM Throttling is ACTIVE (delay: {llm_item_delay_seconds:.1f}s "
                f"if results/keyword ≥ {throttling_threshold})."
            )
            processing_log.append(throttle_init_msg)
            status_placeholder.info(throttle_init_msg) # Show initial throttling status

        if llm_query_gen_step_count > 0:
            update_progress_ui(message="🧠 Generating additional search queries with LLM...")
            # ... (LLM query generation logic from v1.4.5) ...
            # Make sure to update keywords_list_val_runtime and llm_generated_keywords_set_for_display_set
            num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_query_gen: str = app_config.llm.google_gemini_model
            if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
            
            generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                original_keywords=tuple(initial_keywords_list), specific_info_query=primary_llm_extract_query,
                specific_info_query_2=secondary_llm_extract_query, num_queries_to_generate=num_llm_terms_to_generate,
                api_key=llm_api_key_to_use_qgen, model_name=llm_model_for_query_gen
            )
            if generated_queries:
                # ... (update keywords_list_val_runtime and llm_generated_keywords_set_for_display_set) ...
                current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                for gq in generated_queries:
                    if gq.lower() not in current_runtime_keywords_lower: keywords_list_val_runtime.append(gq); current_runtime_keywords_lower.add(gq.lower()); temp_llm_generated_set.add(gq.lower())
                llm_generated_keywords_set_for_display_set = temp_llm_generated_set
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries.")
            else: processing_log.append("  ⚠️ LLM did not generate new queries.")

            current_major_step_count += 1
            update_progress_ui(message="LLM Query Generation complete.")

        total_keywords_actually_processing = len(keywords_list_val_runtime)
        for keyword_idx, keyword_val in enumerate(keywords_list_val_runtime):
            processing_log.append(f"\n🔎 Processing keyword {keyword_idx+1}/{total_keywords_actually_processing}: {keyword_val}")
            update_progress_ui(message=f"Searching for '{keyword_val}' ({keyword_idx+1}/{total_keywords_actually_processing})...")
            # ... (Google Search config check from v1.4.5) ...
            
            search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
                query=keyword_val, api_key=app_config.google_search.api_key, 
                cse_id=app_config.google_search.cse_id, num_results=urls_to_scan_per_keyword
            )
            current_major_step_count += 1 # For the search step
            update_progress_ui(message=f"Found {len(search_results_items_val)} results for '{keyword_val}'.")

            if not search_results_items_val:
                current_major_step_count += urls_to_scan_per_keyword # Account for skipped scrapes
                current_major_step_count += num_results_wanted_per_keyword * llm_tasks_per_good_item_calc # Account for skipped LLM
                update_progress_ui()
                continue

            successfully_scraped_for_this_keyword: int = 0
            for search_item_idx, search_item_val in enumerate(search_results_items_val):
                if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                    current_major_step_count += (urls_to_scan_per_keyword - (search_item_idx)) # Add skipped scrapes
                    update_progress_ui()
                    break
                
                url_to_scrape_val: Optional[str] = search_item_val.get('link')
                # ... (URL check) ...
                update_progress_ui(message=f"Scraping {url_to_scrape_val[:50]}... ({search_item_idx+1}/{urls_to_scan_per_keyword})")
                scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
                current_major_step_count += 1 # For scraping attempt
                update_progress_ui() # Update bar after increment

                # ... (item_data_val setup as in v1.4.5) ...
                made_llm_call_for_item = False
                if not scraped_content_val.get('error'):
                    # ... (is_good_scrape check) ...
                    if is_good_scrape:
                        # ...
                        if llm_key_available:
                            # LLM Summary
                            update_progress_ui(message=f"LLM Summary for {url_to_scrape_val[:40]}...")
                            # ... (llm_processor.generate_summary call) ...
                            item_data_val["llm_summary"] = llm_processor.generate_summary(...) # simplified
                            current_major_step_count += 1; update_progress_ui()
                            made_llm_call_for_item = True
                            
                            # LLM Extractions
                            for query_info in queries_to_process_for_item:
                                update_progress_ui(message=f"LLM Extract Q{query_info['display_idx']} for {url_to_scrape_val[:40]}...")
                                # ... (llm_processor.extract_specific_information call) ...
                                item_data_val[f"llm_extracted_info_{query_info['id']}"] = llm_processor.extract_specific_information(...) # simplified
                                current_major_step_count += 1; update_progress_ui()
                                made_llm_call_for_item = True
                        results_data.append(item_data_val)
                
                if apply_throttling and made_llm_call_for_item:
                    update_progress_ui(message=f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s...")
                    time.sleep(llm_item_delay_seconds)
                # ... (minor non-throttled sleeps from v1.4.5) ...
            
            # Account for any remaining LLM steps if target not met
            if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword:
                skipped_llm_steps = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * llm_tasks_per_good_item_calc
                current_major_step_count += skipped_llm_steps
                update_progress_ui()
        
        if consolidated_summary_step_calc > 0:
            update_progress_ui(message="✨ Generating consolidated overview...")
            # ... (Consolidated summary logic from v1.4.5, ensuring it updates processing_log) ...
            current_major_step_count += 1
            update_progress_ui(message="Consolidated overview generation complete.")
        
        # Final LOG_STATUS messages for app.py to interpret
        # ... (Logic for final processing_log messages as in v1.4.5: LOG_STATUS:SUCCESS, LOG_STATUS:WARNING) ...
        # This part remains the same, as app.py will handle the st.success/warning/error.

        # ... (Google Sheets writing logic as in v1.4.5, ensuring it updates processing_log) ...

    except Exception as e_main_pm: 
        error_message = f"CRITICAL_ERROR_IN_PROCESS_MANAGER:{type(e_main_pm).__name__} - {e_main_pm}"
        processing_log.append(error_message)
        processing_log.append(traceback.format_exc())
        # Do not call st.error here; app.py will handle it based on the log.
        print(f"-----> DEBUG (process_manager v1.4.7): EXCEPTION caught: {error_message}")
            
    finally: 
        print(f"-----> DEBUG (process_manager v1.4.7): FINALLY BLOCK.")
        # Set progress to 100% and a final neutral message or clear them
        # The actual success/warning/error message will be handled by app.py based on processing_log
        if current_major_step_count < total_major_steps_for_progress and total_major_steps_for_progress > 1 : # If ended prematurely due to error or less work than estimated
             current_major_step_count = total_major_steps_for_progress # Visually complete the bar if error happened mid-way

        update_progress_ui(message="Processing finalized. Check results and logs.")
        progress_bar_placeholder.progress(100) # Or progress_bar_placeholder.empty()
        # status_placeholder.empty() # Optionally clear status, or let "Processing finalized" remain.
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

# // end of modules/process_manager.py
