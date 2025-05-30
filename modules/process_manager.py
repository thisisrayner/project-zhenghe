# modules/process_manager.py
# Version 1.2.0: Integrated support for multiple LLM extraction queries.
# Progress bar adjusted for multiple LLM calls per item.
# Consolidated summary focus remains on the first extraction query.
"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
"""

import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple

from modules import config, search_engine, scraper, llm_processor, data_storage

def run_search_and_analysis(
    app_config: config.AppConfig,
    keywords_input: str,
    llm_extract_queries_input: List[str], # MODIFIED: Now a list of queries
    num_results_wanted_per_keyword: int,
    gs_worksheet: Optional[Any], 
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str]]:
    """
    Executes the full search, scrape, LLM process, and GSheet storage pipeline.
    """
    processing_log: List[str] = ["Processing started..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    
    initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
    initial_keywords_for_display: Set[str] = set(k.lower() for k in initial_keywords_list)
    llm_generated_keywords_set_for_display: Set[str] = set()

    if not initial_keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        return ["ERROR: No keywords provided."], [], None, initial_keywords_for_display, llm_generated_keywords_set_for_display

    keywords_list_val_runtime: List[str] = list(initial_keywords_list)
    
    llm_key_available: bool = (app_config.llm.provider == "google" and app_config.llm.google_gemini_api_key) or \
                              (app_config.llm.provider == "openai" and app_config.llm.openai_api_key)

    # Determine the primary extraction query for context in query generation and consolidation
    primary_llm_extract_query: Optional[str] = llm_extract_queries_input[0] if llm_extract_queries_input else None

    # LLM Query Generation
    if llm_key_available and initial_keywords_list:
        processing_log.append("\n🧠 Generating additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list)
        num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)

        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_query_gen: str = app_config.llm.google_gemini_model
            if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize

            with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."):
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                    original_keywords=tuple(initial_keywords_list),
                    specific_info_query=primary_llm_extract_query, # Use primary query for context
                    num_queries_to_generate=num_llm_terms_to_generate,
                    api_key=llm_api_key_to_use_qgen,
                    model_name=llm_model_for_query_gen
                )
            # ... (rest of LLM query generation logic remains similar) ...
            if generated_queries:
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}")
                current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}
                temp_llm_generated_set = set()
                for gq in generated_queries:
                    if gq.lower() not in current_runtime_keywords_lower:
                        keywords_list_val_runtime.append(gq)
                        current_runtime_keywords_lower.add(gq.lower())
                        temp_llm_generated_set.add(gq.lower())
                llm_generated_keywords_set_for_display = temp_llm_generated_set
                processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else:
                processing_log.append("  ⚠️ LLM did not generate new queries.")
        else:
            processing_log.append("  ℹ️ No additional LLM queries requested (or needed based on input).")


    # Progress Bar Setup
    oversample_factor: float = 2.0
    max_google_fetch_per_keyword: int = 10
    est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword : est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword

    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available:
        total_llm_tasks_per_good_scrape += 1 # For summary
        total_llm_tasks_per_good_scrape += len(llm_extract_queries_input) # For each active extraction query

    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                        (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        total_major_steps_for_progress += 1
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty()

    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        current_major_step_count +=1
        # ... (progress update for query gen) ...
        progress_text = "LLM Query Generation Complete..."
        with progress_bar_placeholder.container():
            st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text)


    # Main Processing Loop
    for keyword_val in keywords_list_val_runtime:
        # ... (keyword processing setup, search_engine.perform_search - remains similar) ...
        processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        # ... (Google Search API check and search_results_items_val fetching logic) ...
        if not (app_config.google_search.api_key and app_config.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured.")
            processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'. Google Search not configured.")
            current_major_step_count += est_urls_to_fetch_per_keyword 
            current_major_step_count += num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape
            continue 

        urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
        processing_log.append(f"  Attempting to fetch {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
            query=keyword_val, api_key=app_config.google_search.api_key, cse_id=app_config.google_search.cse_id, num_results=urls_to_fetch_from_google
        )
        processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword: int = 0
        if not search_results_items_val:
            processing_log.append(f"  No Google results for '{keyword_val}'."); current_major_step_count += est_urls_to_fetch_per_keyword; continue


        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            # ... (logic for skipping if target reached, URL check, scraper.fetch_and_extract_content - remains similar) ...
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                skipped_google_results = len(search_results_items_val) - search_item_idx
                processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} for '{keyword_val}'. Skipping {skipped_google_results} Google result(s).")
                current_major_step_count += skipped_google_results 
                break 
            current_major_step_count += 1 
            url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping."); continue
            
            # ... (progress update for scraping) ...
            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder.container(): 
                 st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_scrape)
            
            processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
            
            # Initialize item_data_val with placeholders for multiple extractions
            item_data_val: Dict[str, Any] = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val,
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "scraped_title": scraped_content_val.get('scraped_title'),
                "meta_description": scraped_content_val.get('meta_description'),
                "og_title": scraped_content_val.get('og_title'),
                "og_description": scraped_content_val.get('og_description'),
                "scraped_main_text": scraped_content_val.get('main_text'),
                "scraping_error": scraped_content_val.get('error'),
                "content_type": scraped_content_val.get('content_type'),
                "llm_summary": None,
                # Placeholders for multiple extractions (up to 2 for now)
                "llm_extracted_info_q1": None, 
                "llm_extracted_info_q2": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if scraped_content_val.get('error'):
                processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                min_main_text_length: int = 200 # ... (is_good_scrape logic remains similar)
                current_main_text: str = scraped_content_val.get('main_text', '')
                is_good_scrape: bool = ( current_main_text and len(current_main_text.strip()) >= min_main_text_length and "could not extract main content" not in current_main_text.lower() and "not processed for main text" not in current_main_text.lower() and not str(current_main_text).startswith("SCRAPER_INFO:") )

                if is_good_scrape:
                    processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {item_data_val.get('content_type')}).")
                    successfully_scraped_for_this_keyword += 1
                    main_text_for_llm: str = current_main_text

                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                        llm_model_to_use: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                        
                        # LLM Summary
                        current_major_step_count += 1
                        # ... (progress update for LLM Summary) ...
                        progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                        with progress_bar_placeholder.container():
                            st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_summary)
                        
                        processing_log.append(f"       Generating LLM summary...")
                        summary: Optional[str] = llm_processor.generate_summary(
                            main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars
                        )
                        item_data_val["llm_summary"] = summary
                        processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                        time.sleep(0.1)

                        # LLM Extractions for each query
                        for q_idx, extraction_query in enumerate(llm_extract_queries_input):
                            if not extraction_query.strip(): continue # Skip empty queries

                            current_major_step_count += 1
                            # ... (progress update for LLM Extraction q_idx+1) ...
                            progress_text_llm_extract = f"LLM Extract Q{q_idx+1} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container():
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_extract)
                            
                            processing_log.append(f"      Extracting info for Q{q_idx+1}: '{extraction_query}'...")
                            extracted_info: Optional[str] = llm_processor.extract_specific_information(
                                main_text_for_llm, extraction_query=extraction_query,
                                api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars
                            )
                            item_data_val[f"llm_extracted_info_q{q_idx+1}"] = extracted_info
                            processing_log.append(f"        Extracted (Q{q_idx+1}): {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1)
                    results_data.append(item_data_val)
                else:
                    processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
            time.sleep(0.2)
        # ... (logic for handling insufficient scrapes for keyword - remains similar) ...
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
            processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes."); 
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape; 
            current_major_step_count += remaining_llm_tasks_for_keyword

    with progress_bar_placeholder.container(): st.empty()

    # Consolidated Summary
    topic_for_consolidation_for_batch: str # ... (remains similar)
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics" 
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"


    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            all_valid_llm_outputs_for_consolidation: List[str] = []
            # Focus consolidation on the first extraction query if it was provided
            is_focused_consolidation_intended = bool(primary_llm_extract_query and primary_llm_extract_query.strip())

            for item in results_data:
                summary_text = item.get("llm_summary")
                # For consolidation, primarily use the output of the FIRST extraction query if focused
                extraction_text_q1 = item.get("llm_extracted_info_q1")

                is_summary_valid = summary_text and not str(summary_text).lower().startswith(("llm error", "no text content", "llm_processor:"))
                is_extraction_q1_valid = extraction_text_q1 and not str(extraction_text_q1).lower().startswith(("llm error", "no text content", "llm_processor:"))
                
                item_relevancy_score_q1: Optional[int] = None
                if is_extraction_q1_valid and isinstance(extraction_text_q1, str) and extraction_text_q1.startswith("Relevancy Score: "):
                    try:
                        score_str = extraction_text_q1.split("Relevancy Score: ")[1].split("/")[0]
                        item_relevancy_score_q1 = int(score_str)
                    except (IndexError, ValueError): pass

                chosen_text_for_consolidation = None
                if is_focused_consolidation_intended:
                    if is_extraction_q1_valid and item_relevancy_score_q1 is not None and item_relevancy_score_q1 >= 3:
                        chosen_text_for_consolidation = extraction_text_q1
                else: # General consolidation
                    if is_summary_valid: chosen_text_for_consolidation = summary_text
                    elif is_extraction_q1_valid: chosen_text_for_consolidation = extraction_text_q1 # Fallback to q1 if no summary
                
                if chosen_text_for_consolidation:
                    all_valid_llm_outputs_for_consolidation.append(chosen_text_for_consolidation)

            if not all_valid_llm_outputs_for_consolidation:
                warning_msg = "No valid individual LLM outputs available for consolidated overview."
                if is_focused_consolidation_intended: warning_msg = f"LLM_PROCESSOR: No individual items met the minimum relevancy (from Q1) for '{primary_llm_extract_query}'. Cannot generate focused consolidated overview."
                st.warning(warning_msg); consolidated_summary_text_for_batch = warning_msg; processing_log.append(f"  ❌ {warning_msg}")
            else:
                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(
                    summaries=tuple(all_valid_llm_outputs_for_consolidation),
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                    max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=primary_llm_extract_query if is_focused_consolidation_intended else None
                )
                processing_log.append(f"  Consolidated Overview (first 150 chars): {str(consolidated_summary_text_for_batch)[:150] if consolidated_summary_text_for_batch else 'Failed/Empty'}...")
    
    # Google Sheets Writing
    if sheet_writing_enabled and gs_worksheet:
        if results_data or consolidated_summary_text_for_batch:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
            processing_log.append(f"\n💾 Writing batch data to Google Sheets...")
            
            # Pass the list of actual queries used for column headers in GSheet
            # GSheet function will need to be aware of these for header generation
            actual_extraction_queries_for_sheet = llm_extract_queries_input # Or filter out empty ones if GSheet func expects only active

            write_successful: bool = data_storage.write_batch_summary_and_items_to_sheet(
                worksheet=gs_worksheet, batch_timestamp=batch_process_timestamp_for_sheet,
                consolidated_summary=consolidated_summary_text_for_batch, topic_context=topic_for_consolidation_for_batch,
                item_data_list=results_data,
                extraction_queries_list=actual_extraction_queries_for_sheet # MODIFIED: Pass the list of queries
            )
            if write_successful: processing_log.append(f"  Batch data written to Google Sheets.")
            else: processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    # ... (rest of logging for GSheets status) ...
    elif gsheets_secrets_present and not sheet_writing_enabled : 
        processing_log.append("\n⚠️ Google Sheets connection failed earlier. Data not saved to sheet.")
    elif not gsheets_secrets_present: 
        processing_log.append("\nℹ️ Google Sheets integration not configured. Data not saved to sheet.")

    if results_data or consolidated_summary_text_for_batch :
        if not (consolidated_summary_text_for_batch and consolidated_summary_text_for_batch.startswith("LLM_PROCESSOR: No individual items met")):
            st.success("All processing complete!")
    else: st.warning("Processing complete, but no data was generated.")
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display

# end of modules/process_manager.py
