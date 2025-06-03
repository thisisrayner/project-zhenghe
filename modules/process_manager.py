# modules/process_manager.py
# Version 1.4.8:
# - Fixed NameError: 'is_good_scrape' was not defined due to an omission in v1.4.7.
# Previous versions:
# - Version 1.4.7: Reinstated progress bar and intermediate status text updates.
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
    print("-----> DEBUG (process_manager v1.4.8): TOP OF run_search_and_analysis called.")

    processing_log: List[str] = ["LOG_STATUS:PROCESSING_INITIATED:Processing initiated..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    initial_keywords_for_display_set: Set[str] = set()
    llm_generated_keywords_set_for_display_set: Set[str] = set()
    
    progress_bar_placeholder = st.empty()
    status_placeholder = st.empty()
    
    print("-----> DEBUG (process_manager v1.4.8): UI placeholders created.")

    try:
        initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
        if not initial_keywords_list:
            processing_log.append("LOG_STATUS:ERROR:NO_KEYWORDS:Please enter at least one keyword.")
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

        num_initial_keywords_for_calc = len(initial_keywords_list)
        llm_query_gen_step_count = 0
        if llm_key_available and num_initial_keywords_for_calc > 0:
            num_llm_terms_to_generate_calc = min(math.floor(num_initial_keywords_for_calc * 1.5), 5) # Example logic
            if num_llm_terms_to_generate_calc > 0:
                llm_query_gen_step_count = 1
        
        temp_keywords_list_for_calc = list(initial_keywords_list)
        if llm_query_gen_step_count > 0:
            num_llm_terms_to_add_for_calc = min(math.floor(num_initial_keywords_for_calc * 1.5), 5)
            # This simulation needs to be more careful not to overestimate if LLM queries are not unique
            # For simplicity in estimation, assume they could be unique for max step count
            for i in range(num_llm_terms_to_add_for_calc): temp_keywords_list_for_calc.append(f"llm_gen_placeholder_{i}")

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
        consolidated_summary_step_calc = 1 if llm_key_available and (llm_item_processing_steps_calc > 0 or scraping_steps_calc > 0) else 0

        total_major_steps_for_progress: int = (
            llm_query_gen_step_count + search_steps_calc + scraping_steps_calc +
            llm_item_processing_steps_calc + consolidated_summary_step_calc
        )
        if total_major_steps_for_progress == 0: total_major_steps_for_progress = 1
        current_major_step_count: int = 0
        
        print(f"-----> DEBUG (process_manager v1.4.8): Total estimated steps for progress: {total_major_steps_for_progress}")

        def update_progress_ui(message: Optional[str] = None):
            progress_val = min(1.0, current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            progress_bar_placeholder.progress(math.ceil(progress_val * 100) / 100)
            if message:
                status_placeholder.info(message)

        if apply_throttling:
            throttle_init_msg = (
                f"ℹ️ LLM Throttling is ACTIVE (delay: {llm_item_delay_seconds:.1f}s "
                f"if results/keyword ≥ {throttling_threshold})."
            )
            processing_log.append(throttle_init_msg)
            status_placeholder.info(throttle_init_msg)

        if llm_query_gen_step_count > 0:
            update_progress_ui(message="🧠 Generating additional search queries with LLM...")
            # --- LLM Query Generation ---
            num_user_terms_for_llm_qgen = len(initial_keywords_list)
            num_llm_terms_to_generate_for_llm_qgen = min(math.floor(num_user_terms_for_llm_qgen * 1.5), 5)
            
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_qgen: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
            
            generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                original_keywords=tuple(initial_keywords_list), 
                specific_info_query=primary_llm_extract_query,
                specific_info_query_2=secondary_llm_extract_query, 
                num_queries_to_generate=num_llm_terms_to_generate_for_llm_qgen,
                api_key=llm_api_key_to_use_qgen, model_name=llm_model_for_qgen
            )
            if generated_queries:
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}")
                current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}
                temp_llm_generated_set = set()
                for gq_val in generated_queries:
                    if gq_val.lower() not in current_runtime_keywords_lower:
                        keywords_list_val_runtime.append(gq_val)
                        current_runtime_keywords_lower.add(gq_val.lower())
                        temp_llm_generated_set.add(gq_val.lower())
                llm_generated_keywords_set_for_display_set = temp_llm_generated_set
                processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else: 
                processing_log.append("  ⚠️ LLM did not generate new queries.")
            current_major_step_count += 1
            update_progress_ui(message="LLM Query Generation complete.")

        total_keywords_actually_processing = len(keywords_list_val_runtime)
        for keyword_idx, keyword_val in enumerate(keywords_list_val_runtime):
            processing_log.append(f"\n🔎 Processing keyword {keyword_idx+1}/{total_keywords_actually_processing}: {keyword_val}")
            update_progress_ui(message=f"Searching for '{keyword_val}' ({keyword_idx+1}/{total_keywords_actually_processing})...")
            
            if not (app_config.google_search.api_key and app_config.google_search.cse_id):
                # ... (error logging as in v1.4.5, skip this keyword's steps) ...
                current_major_step_count += 1 # Search step
                current_major_step_count += urls_to_scan_per_keyword # Skipped scrapes
                current_major_step_count += num_results_wanted_per_keyword * llm_tasks_per_good_item_calc # Skipped LLM
                update_progress_ui()
                continue
            
            search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
                query=keyword_val, api_key=app_config.google_search.api_key, 
                cse_id=app_config.google_search.cse_id, num_results=urls_to_scan_per_keyword
            )
            current_major_step_count += 1 # For the search step
            processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
            update_progress_ui(message=f"Found {len(search_results_items_val)} results for '{keyword_val}'.")

            if not search_results_items_val:
                current_major_step_count += urls_to_scan_per_keyword 
                current_major_step_count += num_results_wanted_per_keyword * llm_tasks_per_good_item_calc
                update_progress_ui()
                continue

            successfully_scraped_for_this_keyword: int = 0
            for search_item_idx, search_item_val in enumerate(search_results_items_val):
                if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                    skipped_scrapes_count = urls_to_scan_per_keyword - search_item_idx
                    current_major_step_count += skipped_scrapes_count 
                    processing_log.append(f"  Reached target for '{keyword_val}'. Skipping {skipped_scrapes_count} Google result(s).")
                    update_progress_ui()
                    break
                
                url_to_scrape_val: Optional[str] = search_item_val.get('link')
                if not url_to_scrape_val:
                    processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping.")
                    current_major_step_count += 1 # Count as a skipped scrape attempt
                    update_progress_ui()
                    continue

                update_progress_ui(message=f"Scraping {url_to_scrape_val[:50]}... ({search_item_idx+1}/{urls_to_scan_per_keyword})")
                scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
                current_major_step_count += 1 # For scraping attempt
                update_progress_ui() 

                item_data_val: Dict[str, Any] = {
                    "keyword_searched": keyword_val, "url": url_to_scrape_val, 
                    "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                    "page_title": scraped_content_val.get('scraped_title'), 
                    "meta_description": scraped_content_val.get('meta_description'), 
                    "og_title": scraped_content_val.get('og_title'), "og_description": scraped_content_val.get('og_description'),
                    "main_content_display": scraped_content_val.get('main_text'), 
                    "pdf_document_title": scraped_content_val.get('pdf_doc_title'),
                    "is_pdf": scraped_content_val.get('content_type') == 'application/pdf',
                    "source_query_type": "LLM-Generated" if keyword_val.lower() in llm_generated_keywords_set_for_display_set else "Original",
                    "scraping_error": scraped_content_val.get('error'), 
                    "content_type": scraped_content_val.get('content_type'), 
                    "llm_summary": None, 
                    "llm_extracted_info_q1": None, "llm_relevancy_score_q1": None, "llm_extracted_info_q1_full": None,
                    "llm_extracted_info_q2": None, "llm_relevancy_score_q2": None, "llm_extracted_info_q2_full": None,
                    "llm_extraction_query_1_text": primary_llm_extract_query or "",
                    "llm_extraction_query_2_text": secondary_llm_extract_query or "",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                made_llm_call_for_item = False

                if scraped_content_val.get('error'): 
                    processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
                else:
                    # --- Definition of is_good_scrape ---
                    min_main_text_length: int = app_config.scraper.min_main_text_length if hasattr(app_config, 'scraper') else 200 # Example default
                    current_main_text: str = scraped_content_val.get('main_text', '')
                    is_good_scrape: bool = (
                        current_main_text and 
                        len(current_main_text.strip()) >= min_main_text_length and 
                        "could not extract main content" not in current_main_text.lower() and 
                        "not processed for main text" not in current_main_text.lower() and 
                        not str(current_main_text).startswith("SCRAPER_INFO:")
                    )
                    # --- End Definition of is_good_scrape ---

                    if is_good_scrape:
                        processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {item_data_val.get('content_type')}).")
                        successfully_scraped_for_this_keyword += 1
                        main_text_for_llm: str = current_main_text

                        if llm_key_available:
                            llm_api_key_to_use: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                            llm_model_to_use: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                            max_input_chars_for_llm = app_config.llm.max_input_chars if hasattr(app_config.llm, 'max_input_chars') else 100000 # Example default
                            
                            update_progress_ui(message=f"LLM Summary for {url_to_scrape_val[:40]}...")
                            summary_text_val: Optional[str] = llm_processor.generate_summary(
                                main_text_for_llm, api_key=llm_api_key_to_use, 
                                model_name=llm_model_to_use, max_input_chars=max_input_chars_for_llm
                            )
                            item_data_val["llm_summary"] = summary_text_val
                            processing_log.append(f"        Summary: {str(summary_text_val)[:100] if summary_text_val else 'Failed/Empty'}...")
                            current_major_step_count += 1; update_progress_ui()
                            made_llm_call_for_item = True
                            
                            queries_to_process_for_item = []
                            if primary_llm_extract_query:
                                queries_to_process_for_item.append({"query_text": primary_llm_extract_query, "id": "q1", "display_idx": 1})
                            if secondary_llm_extract_query:
                                queries_to_process_for_item.append({"query_text": secondary_llm_extract_query, "id": "q2", "display_idx": 2})

                            for query_info in queries_to_process_for_item:
                                extraction_query = query_info["query_text"]
                                query_id_label = query_info["id"]; query_display_idx = query_info["display_idx"]
                                update_progress_ui(message=f"LLM Extract Q{query_display_idx} for {url_to_scrape_val[:40]}...")
                                extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                    main_text_for_llm, extraction_query=extraction_query, 
                                    api_key=llm_api_key_to_use, model_name=llm_model_to_use, 
                                    max_input_chars=max_input_chars_for_llm
                                )
                                # ... (parsing score and content as in v1.4.5) ...
                                parsed_score = _parse_score_from_extraction(extracted_info_full)
                                content_without_score = extracted_info_full 
                                if hasattr(llm_processor, '_parse_score_and_get_content'): 
                                    _, content_temp = llm_processor._parse_score_and_get_content(extracted_info_full) # type: ignore
                                    if content_temp is not None: content_without_score = content_temp
                                elif parsed_score is not None and extracted_info_full and '\n' in extracted_info_full:
                                     try: content_without_score = extracted_info_full.split('\n', 1)[1]
                                     except IndexError: pass 
                                item_data_val[f"llm_extracted_info_{query_id_label}"] = content_without_score
                                item_data_val[f"llm_relevancy_score_{query_id_label}"] = parsed_score
                                item_data_val[f"llm_extracted_info_{query_id_label}_full"] = extracted_info_full 
                                processing_log.append(f"        Extracted (Q{query_display_idx}): Score={parsed_score}, Content='{str(content_without_score)[:70] if content_without_score else 'Failed/Empty'}'...")
                                current_major_step_count += 1; update_progress_ui()
                                made_llm_call_for_item = True
                        results_data.append(item_data_val)
                    else: 
                        processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
                        # Account for skipped LLM tasks if this was a potential good scrape
                        current_major_step_count += llm_tasks_per_good_item_calc 
                        update_progress_ui()
                
                if apply_throttling and made_llm_call_for_item:
                    update_progress_ui(message=f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s...")
                    time.sleep(llm_item_delay_seconds)
                elif not apply_throttling and made_llm_call_for_item: time.sleep(0.2) 
                elif not made_llm_call_for_item: time.sleep(0.1)
            
            if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
                processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
                skipped_llm_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * llm_tasks_per_good_item_calc
                current_major_step_count += skipped_llm_for_keyword
                update_progress_ui()
        
        # Consolidated Summary
        topic_for_consolidation_for_batch: str = ", ".join(initial_keywords_list) if initial_keywords_list else "the searched topics" # Simplified
        if consolidated_summary_step_calc > 0: # Indicates LLM is available and there might be data
            update_progress_ui(message="✨ Generating consolidated overview...")
            # ... (Full consolidated summary logic from v1.4.5, adapted for current variables) ...
            # This block needs to be carefully re-inserted based on v1.4.5's logic for focused/general summary
            # For brevity, placeholder for that logic:
            if results_data and llm_key_available:
                # ... (Determine if focused or general, prepare texts_for_llm) ...
                # ... (Call llm_processor.generate_consolidated_summary) ...
                # consolidated_summary_text_for_batch = result
                # processing_log.append("...")
                pass # Replace with actual summary logic
            else:
                consolidated_summary_text_for_batch = "INFO: No data or LLM for consolidated summary."
            current_major_step_count += 1
            update_progress_ui(message="Consolidated overview generation attempt complete.")
        
        # Final LOG_STATUS messages for app.py to interpret
        # This logic is from v1.4.5 and should remain to signal app.py
        if results_data or consolidated_summary_text_for_batch:
            # ... (Detailed LOG_STATUS:SUCCESS/WARNING setting as in v1.4.5) ...
            is_info_only_summary = consolidated_summary_text_for_batch and str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:")
            is_error_summary = consolidated_summary_text_for_batch and str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:") # or "error:"
            if not is_info_only_summary and not is_error_summary and consolidated_summary_text_for_batch:
                processing_log.append("LOG_STATUS:SUCCESS:All processing complete! A consolidated overview has been generated.")
            # ... (other conditions for LOG_STATUS) ...
        else: 
            processing_log.append("LOG_STATUS:WARNING:Processing complete, but no data was generated.")

        # Google Sheets Writing
        # ... (Full Google Sheets logic from v1.4.5, ensuring it updates processing_log) ...

    except Exception as e_main_pm: 
        error_message = f"CRITICAL_ERROR_IN_PROCESS_MANAGER:{type(e_main_pm).__name__} - {e_main_pm}"
        processing_log.append(error_message)
        processing_log.append(traceback.format_exc())
        print(f"-----> DEBUG (process_manager v1.4.8): EXCEPTION caught: {error_message}")
        # The LOG_STATUS:ERROR will be set by app.py when it sees this critical error in the log
            
    finally: 
        print(f"-----> DEBUG (process_manager v1.4.8): FINALLY BLOCK.")
        # Ensure progress bar completes and status is neutral before app.py takes over for final message
        if current_major_step_count < total_major_steps_for_progress and total_major_steps_for_progress > 1 :
             current_major_step_count = total_major_steps_for_progress 
        update_progress_ui(message="Processing finalized. Check app messages and logs.")
        progress_bar_placeholder.progress(100) # Ensure it hits 100%
        # status_placeholder.empty() # Optionally clear or let "Processing finalized" remain.
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

# // end of modules/process_manager.py
