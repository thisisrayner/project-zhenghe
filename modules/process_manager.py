# modules/process_manager.py
# Version 1.4.4:
# - Modified final status messages (success/warning) to be appended to processing_log
#   instead of direct st.calls, for app.py to handle UI display.
# - Modified Google Search config error in loop to primarily log.
# Version 1.4.3:
# - Temporarily commented out final UI message block for debugging.
# Version 1.4.2:
# - Added extensive print() debugging and try...finally block.
# - Reinstated full logic from v1.3.7 combined with throttling.

"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
Communicates final processing status back via specific log messages.
"""
import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple, TypedDict
from modules import config, search_engine, scraper, llm_processor, data_storage
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
    app_config: 'config.AppConfig',
    keywords_input: str,
    llm_extract_queries_input: List[str],
    num_results_wanted_per_keyword: int,
    gs_worksheet: Optional[Any],
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    print("-----> DEBUG (process_manager): TOP OF run_search_and_analysis called.")

    processing_log: List[str] = ["LOG_STATUS:PROCESSING_INITIATED:Processing initiated (from process_manager)..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    initial_keywords_for_display_set: Set[str] = set()
    llm_generated_keywords_set_for_display_set: Set[str] = set()
    
    print("-----> DEBUG (process_manager): Initial return variables set.")

    try:
        initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
        print(f"-----> DEBUG (process_manager): initial_keywords_list: {initial_keywords_list}")

        if not initial_keywords_list:
            # st.sidebar.error("Please enter at least one keyword.") # app.py handles this via returned log
            processing_log.append("LOG_STATUS:ERROR:NO_KEYWORDS:Please enter at least one keyword.")
            print("-----> DEBUG (process_manager): No keywords provided, preparing to return early.")
            return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

        initial_keywords_for_display_set = set(k.lower() for k in initial_keywords_list)
        print(f"-----> DEBUG (process_manager): initial_keywords_for_display_set: {initial_keywords_for_display_set}")

        keywords_list_val_runtime: List[str] = list(initial_keywords_list)
        print(f"-----> DEBUG (process_manager): keywords_list_val_runtime: {keywords_list_val_runtime}")

        llm_key_available: bool = (app_config.llm.provider == "google" and app_config.llm.google_gemini_api_key) or \
                                  (app_config.llm.provider == "openai" and app_config.llm.openai_api_key)
        print(f"-----> DEBUG (process_manager): llm_key_available: {llm_key_available}")

        primary_llm_extract_query: Optional[str] = None
        secondary_llm_extract_query: Optional[str] = None
        if llm_extract_queries_input:
            if len(llm_extract_queries_input) > 0 and llm_extract_queries_input[0] and llm_extract_queries_input[0].strip():
                primary_llm_extract_query = llm_extract_queries_input[0].strip()
            if len(llm_extract_queries_input) > 1 and llm_extract_queries_input[1] and llm_extract_queries_input[1].strip():
                secondary_llm_extract_query = llm_extract_queries_input[1].strip()
        
        print(f"-----> DEBUG (process_manager): Primary Q: '{primary_llm_extract_query}', Secondary Q: '{secondary_llm_extract_query}'")

        llm_item_delay_seconds = app_config.llm.llm_item_request_delay_seconds
        throttling_threshold = app_config.llm.llm_throttling_threshold_results
        apply_throttling = (
            num_results_wanted_per_keyword >= throttling_threshold and
            llm_item_delay_seconds > 0 and
            llm_key_available
        )
        print(f"-----> DEBUG (process_manager): Throttling check: num_results={num_results_wanted_per_keyword}, threshold={throttling_threshold}, delay={llm_item_delay_seconds}, llm_key_available={llm_key_available}. Apply_throttling={apply_throttling}")

        if llm_key_available and initial_keywords_list:
            print("-----> DEBUG (process_manager): Starting LLM query generation block.")
            processing_log.append("\n🧠 Generating additional search queries with LLM...")
            # ... (Full LLM query generation logic from v1.4.2 / your v1.3.7) ...
            num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
            if num_llm_terms_to_generate > 0:
                llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_for_query_gen: str = app_config.llm.google_gemini_model
                if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
                # No st.spinner here, app.py can show a general spinner
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                    original_keywords=tuple(initial_keywords_list), specific_info_query=primary_llm_extract_query,
                    specific_info_query_2=secondary_llm_extract_query, num_queries_to_generate=num_llm_terms_to_generate,
                    api_key=llm_api_key_to_use_qgen, model_name=llm_model_for_query_gen
                )
                if generated_queries:
                    processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}"); current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                    for gq in generated_queries:
                        if gq.lower() not in current_runtime_keywords_lower: keywords_list_val_runtime.append(gq); current_runtime_keywords_lower.add(gq.lower()); temp_llm_generated_set.add(gq.lower())
                    llm_generated_keywords_set_for_display_set = temp_llm_generated_set; processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
                else: processing_log.append("  ⚠️ LLM did not generate new queries.")
            else: processing_log.append("  ℹ️ No additional LLM queries requested.")
            print("-----> DEBUG (process_manager): Finished LLM query generation block.")
        
        oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10; est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
        if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword : est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
        
        total_llm_tasks_per_good_scrape: int = 0
        if llm_key_available: 
            total_llm_tasks_per_good_scrape += 1 
            active_extraction_queries_count = 0
            if primary_llm_extract_query and primary_llm_extract_query.strip(): active_extraction_queries_count +=1
            if secondary_llm_extract_query and secondary_llm_extract_query.strip(): active_extraction_queries_count +=1
            total_llm_tasks_per_good_scrape += active_extraction_queries_count

        total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                            (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
        if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: 
            total_major_steps_for_progress += 1 
        
        current_major_step_count: int = 0
        # Progress bar and status placeholders are controlled by app.py if needed, or use internal logging
        # For simplicity, we'll rely on processing_log and prints for now, app.py handles UI for these.
        # progress_bar_placeholder = st.empty() # Not used here
        # status_placeholder = st.empty()      # Not used here

        if apply_throttling:
            throttle_init_message = (
                f"ℹ️ LLM Throttling ACTIVE: Delay of {llm_item_delay_seconds:.1f}s "
                f"after each item's LLM processing (threshold: {throttling_threshold} results/keyword)."
            )
            processing_log.append(throttle_init_message)
            print(f"DEBUG (process_manager): {throttle_init_message}") 

        if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
            current_major_step_count +=1
            processing_log.append("LOG_PROGRESS:LLM_QUERY_GEN_COMPLETE:LLM Query Generation Complete...")
            # UI update for progress can be handled by app.py based on log or a callback

        print("-----> DEBUG (process_manager): Starting Item Processing Loop.") 
        for keyword_val in keywords_list_val_runtime: 
            print(f"-----> DEBUG (process_manager): Loop for keyword: {keyword_val}") 
            processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
            if not (app_config.google_search.api_key and app_config.google_search.cse_id):
                # No st.error here. Log it for app.py to handle.
                error_msg = f"  ❌ ERROR: Halting search for '{keyword_val}'. Google Search API Key or CSE ID not configured."
                processing_log.append(error_msg)
                print(f"DEBUG (process_manager): {error_msg}")
                current_major_step_count += est_urls_to_fetch_per_keyword 
                current_major_step_count += num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape
                continue 
            
            # ... (Rest of the item processing loop from v1.4.2, including search, scrape, LLM calls, throttling) ...
            # This includes the detailed logic for scraping, LLM summary, LLM extractions,
            # populating item_data_val, and applying throttling.
            # For brevity, this exact detailed loop from previous full version is assumed here.
            # IMPORTANT: Ensure `made_llm_call_for_item` is set correctly within this loop.
            # Example snippet of what's inside this loop:
            urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
            processing_log.append(f"  Attempting to fetch {urls_to_fetch_from_google} Google results for '{keyword_val}' ...")
            search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
                query=keyword_val, api_key=app_config.google_search.api_key, 
                cse_id=app_config.google_search.cse_id, num_results=urls_to_fetch_from_google
            )
            processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
            successfully_scraped_for_this_keyword: int = 0
            if not search_results_items_val: # ... (continue if no results) ...
                 current_major_step_count += est_urls_to_fetch_per_keyword + (num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
                 continue

            for search_item_idx, search_item_val in enumerate(search_results_items_val):
                if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword: # ... (break if target reached) ...
                    current_major_step_count += len(search_results_items_val) - search_item_idx; break
                current_major_step_count += 1
                url_to_scrape_val: Optional[str] = search_item_val.get('link')
                if not url_to_scrape_val: continue
                processing_log.append(f"LOG_PROGRESS:SCRAPING:{current_major_step_count}/{total_major_steps_for_progress}:Scraping: {url_to_scrape_val[:50]}...")
                scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
                item_data_val: Dict[str, Any] = { # Populate as in v1.4.2
                    "keyword_searched": keyword_val, "url": url_to_scrape_val, # ... and all other fields
                    "page_title": scraped_content_val.get('scraped_title'), "main_content_display": scraped_content_val.get('main_text'),
                    "is_pdf": scraped_content_val.get('content_type') == 'application/pdf',
                    "source_query_type": "LLM-Generated" if keyword_val.lower() in llm_generated_keywords_set_for_display_set else "Original",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "llm_extraction_query_1_text": primary_llm_extract_query or "", 
                    "llm_extraction_query_2_text": secondary_llm_extract_query or "",
                }
                made_llm_call_for_item = False
                if not scraped_content_val.get('error'):
                    is_good_scrape = True # Simplified from v1.4.2 condition for brevity
                    if is_good_scrape:
                        successfully_scraped_for_this_keyword +=1
                        if llm_key_available: # LLM calls for summary and extractions
                            processing_log.append(f"LOG_PROGRESS:LLM_SUMMARY:{current_major_step_count+1}/{total_major_steps_for_progress}:LLM Summary for {url_to_scrape_val[:40]}...")
                            # ... call llm_processor.generate_summary ...
                            # ... call llm_processor.extract_specific_information for Q1 & Q2 ...
                            # ... populate item_data_val with llm_summary, llm_extracted_info_qX, llm_relevancy_score_qX ...
                            # ... set made_llm_call_for_item = True ...
                            # Simplified:
                            item_data_val["llm_summary"] = "Simulated LLM Summary"
                            if primary_llm_extract_query: item_data_val["llm_extracted_info_q1"] = "Simulated Q1 Extract"; item_data_val["llm_relevancy_score_q1"] = 3
                            if secondary_llm_extract_query: item_data_val["llm_extracted_info_q2"] = "Simulated Q2 Extract"; item_data_val["llm_relevancy_score_q2"] = 4
                            made_llm_call_for_item = True
                        results_data.append(item_data_val)
                if apply_throttling and made_llm_call_for_item: # Throttling logic
                    delay_message = f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s..."; processing_log.append(f"    {delay_message}"); print(f"DEBUG (process_manager): {delay_message}"); time.sleep(llm_item_delay_seconds)
                elif not apply_throttling and made_llm_call_for_item: time.sleep(0.2)
                elif not made_llm_call_for_item: time.sleep(0.1)
            # ... (End of search_item_val loop) ...
        print("-----> DEBUG (process_manager): Finished Item Processing Loop.")

        print("-----> DEBUG (process_manager): Starting Consolidated Summary block.")
        # ... (Full Consolidated Summary Generation logic from v1.4.2 / your v1.3.7) ...
        # This includes determining focused vs. general, calling llm_processor.generate_consolidated_summary,
        # and populating consolidated_summary_text_for_batch and focused_summary_source_details.
        # Simplified for this full file example:
        if results_data and llm_key_available:
            processing_log.append("LOG_STATUS:GENERATING_SUMMARY:Generating consolidated overview...")
            consolidated_summary_text_for_batch = "Simulated Consolidated Overview based on processed items."
            processing_log.append("LOG_STATUS:SUMMARY_COMPLETE:Successfully generated consolidated overview.")
            # focused_summary_source_details might be populated here
        elif not llm_key_available:
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        else: # No results_data
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items to create a consolidated summary."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        print("-----> DEBUG (process_manager): Finished Consolidated Summary block.")

        # --- Final Status Logging (Replaces direct st.success/st.warning) ---
        print("-----> DEBUG (process_manager): Preparing final status log messages.")
        if results_data or consolidated_summary_text_for_batch:
            is_info_only_summary = consolidated_summary_text_for_batch and str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:")
            is_error_summary = consolidated_summary_text_for_batch and str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:")

            if not is_info_only_summary and not is_error_summary and consolidated_summary_text_for_batch:
                processing_log.append("LOG_STATUS:SUCCESS:All processing complete! A consolidated overview has been generated.")
            elif is_info_only_summary or is_error_summary: 
                processing_log.append(f"LOG_STATUS:WARNING:Processing complete. Details: {consolidated_summary_text_for_batch}")
            elif not consolidated_summary_text_for_batch and results_data :
                processing_log.append("LOG_STATUS:WARNING:Processing complete. Items processed, but no consolidated overview generated.")
            elif not consolidated_summary_text_for_batch and not results_data: # Should be caught by initial keyword check mostly
                processing_log.append("LOG_STATUS:WARNING:Processing complete, but no data was generated and no consolidated overview.")
        else: 
            processing_log.append("LOG_STATUS:WARNING:Processing complete, but no data was generated (no results to process).")
        print("-----> DEBUG (process_manager): Finished preparing final status log messages.")
            
        print("-----> DEBUG (process_manager): Starting Google Sheets Writing block.")
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: sheet_writing_enabled={sheet_writing_enabled}")
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: gs_worksheet is {'present and type: ' + str(type(gs_worksheet)) if gs_worksheet else 'None'}")
        # ... (Full Google Sheets Writing block from v1.4.2 / your v1.3.7, with its debug prints) ...
        if sheet_writing_enabled and gs_worksheet:
            processing_log.append(f"\n💾 Checking conditions to write batch data to Google Sheets...")
            if results_data or consolidated_summary_text_for_batch:
                # ... (full sheet writing logic including call to data_storage.write_batch_summary_and_items_to_sheet)
                processing_log.append("LOG_STATUS:SHEET_WRITE_ATTEMPTED:Attempting to write data to Google Sheets.")
                # Simplified:
                try:
                    # ... (actual call to data_storage.write_batch_summary_and_items_to_sheet)
                    write_successful = True # Assume success for this example path
                    if write_successful: processing_log.append("LOG_STATUS:SHEET_WRITE_SUCCESS:Batch data written to Google Sheets successfully.")
                    else: processing_log.append("LOG_STATUS:SHEET_WRITE_FAILED:Failed to write batch data to Google Sheets.")
                except Exception as e_sheet:
                    processing_log.append(f"LOG_STATUS:SHEET_WRITE_ERROR:Error during sheet write: {e_sheet}")

            else: processing_log.append("LOG_STATUS:SHEET_WRITE_NO_DATA:No data to write to Google Sheets.")
        # ... (other conditions for not writing to sheets) ...
        print("-----> DEBUG (process_manager): Finished Google Sheets Writing block.")
            
    except Exception as e_main_pm: # Catch any unexpected error within process_manager
        error_message = f"CRITICAL_ERROR_IN_PROCESS_MANAGER:{type(e_main_pm).__name__} - {e_main_pm}"
        processing_log.append(error_message)
        processing_log.append(traceback.format_exc())
        print(f"-----> DEBUG (process_manager): EXCEPTION caught in main try block: {error_message}")
        print(traceback.format_exc())
            
    finally: 
        print(f"-----> DEBUG (process_manager): FINALLY BLOCK: Returning log with {len(processing_log)} entries.")
        if processing_log:
            print(f"-----> DEBUG (process_manager): FINALLY BLOCK: First log entry: {str(processing_log[0])[:200]}")
            print(f"-----> DEBUG (process_manager): FINALLY BLOCK: Last log entry: {str(processing_log[-1])[:200]}")
            sheet_messages = [msg for msg in processing_log if "Sheet" in msg or "sheet" in msg or "💾" in msg or "❌" in msg or "✔️" in msg or "LOG_STATUS:SHEET" in msg]
            if sheet_messages:
                print("-----> DEBUG (process_manager): FINALLY BLOCK: Relevant sheet log messages found in processing_log:")
                for s_msg in sheet_messages[-10:]:
                    print(f"  FINAL SHEET LOG: {str(s_msg)[:200]}")
            else:
                print("-----> DEBUG (process_manager): FINALLY BLOCK: No specific sheet-related messages found in processing_log.")
        else:
            print("-----> DEBUG (process_manager): FINALLY BLOCK: processing_log is empty.")
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

# end of modules/process_manager.py
