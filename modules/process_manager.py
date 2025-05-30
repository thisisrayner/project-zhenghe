# modules/process_manager.py
# Version 1.2.5:
# - Implemented a more robust fallback for consolidated summary: if a focused
#   summary is attempted with >=3 score items, but the LLM still returns a
#   message like "No items met score >=3...", it now explicitly falls back
#   to attempting a general summary from all item summaries.
"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
"""
import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from modules import config, search_engine, scraper, llm_processor, data_storage

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
    app_config: config.AppConfig, keywords_input: str, llm_extract_queries_input: List[str], 
    num_results_wanted_per_keyword: int, gs_worksheet: Optional[Any], 
    sheet_writing_enabled: bool, gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str]]:
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

    primary_llm_extract_query: Optional[str] = None
    if llm_extract_queries_input and llm_extract_queries_input[0] and llm_extract_queries_input[0].strip():
        primary_llm_extract_query = llm_extract_queries_input[0].strip()

    if llm_key_available and initial_keywords_list: 
        processing_log.append("\n🧠 Generating additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_query_gen: str = app_config.llm.google_gemini_model
            if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
            with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."):
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(original_keywords=tuple(initial_keywords_list), specific_info_query=primary_llm_extract_query, num_queries_to_generate=num_llm_terms_to_generate, api_key=llm_api_key_to_use_qgen, model_name=llm_model_for_query_gen)
            if generated_queries:
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}"); current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                for gq in generated_queries:
                    if gq.lower() not in current_runtime_keywords_lower: keywords_list_val_runtime.append(gq); current_runtime_keywords_lower.add(gq.lower()); temp_llm_generated_set.add(gq.lower())
                llm_generated_keywords_set_for_display = temp_llm_generated_set; processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else: processing_log.append("  ⚠️ LLM did not generate new queries.")
        else: processing_log.append("  ℹ️ No additional LLM queries requested (or needed based on input).")

    oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10; est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword : est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: 
        total_llm_tasks_per_good_scrape += 1 # For summary
        total_llm_tasks_per_good_scrape += len([q for q in llm_extract_queries_input if q.strip()]) # For extractions

    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                          (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: 
        total_major_steps_for_progress += 1 # For LLM query gen step
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty()

    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        current_major_step_count +=1
        progress_text = "LLM Query Generation Complete..."
        with progress_bar_placeholder.container(): 
            st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text)

    for keyword_val in keywords_list_val_runtime: 
        processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        if not (app_config.google_search.api_key and app_config.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured.")
            processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'. Google Search not configured.")
            current_major_step_count += est_urls_to_fetch_per_keyword + (num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
            continue 
        
        urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
        processing_log.append(f"  Attempting to fetch {urls_to_fetch_from_google} Google results for '{keyword_val}' to get {num_results_wanted_per_keyword} good scrapes.")
        search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
            query=keyword_val, 
            api_key=app_config.google_search.api_key, 
            cse_id=app_config.google_search.cse_id, 
            num_results=urls_to_fetch_from_google
        )
        processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        successfully_scraped_for_this_keyword: int = 0

        if not search_results_items_val: 
            processing_log.append(f"  No Google results for '{keyword_val}'.")
            current_major_step_count += est_urls_to_fetch_per_keyword # Account for skipped fetches
            # No LLM tasks if no scrapes
            continue

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                skipped_google_results = len(search_results_items_val) - search_item_idx
                processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} for '{keyword_val}'. Skipping {skipped_google_results} Google result(s).")
                current_major_step_count += skipped_google_results # Account for skipped fetches
                # LLM tasks for these skipped fetches are implicitly skipped
                break 
            
            current_major_step_count += 1 # Increment for the fetch/scrape attempt
            url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: 
                processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping.")
                continue

            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder.container(): 
                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_scrape)
            
            processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
            
            item_data_val: Dict[str, Any] = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val, 
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "scraped_title": scraped_content_val.get('scraped_title'), 
                "meta_description": scraped_content_val.get('meta_description'), 
                "og_title": scraped_content_val.get('og_title'), "og_description": scraped_content_val.get('og_description'),
                "scraped_main_text": scraped_content_val.get('main_text'), 
                "scraping_error": scraped_content_val.get('error'), 
                "content_type": scraped_content_val.get('content_type'), 
                "llm_summary": None, "llm_extracted_info_q1": None, "llm_extracted_info_q2": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if scraped_content_val.get('error'): 
                processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                min_main_text_length: int = 200
                current_main_text: str = scraped_content_val.get('main_text', '')
                is_good_scrape: bool = (
                    current_main_text and 
                    len(current_main_text.strip()) >= min_main_text_length and 
                    "could not extract main content" not in current_main_text.lower() and 
                    "not processed for main text" not in current_main_text.lower() and 
                    not str(current_main_text).startswith("SCRAPER_INFO:")
                )

                if is_good_scrape:
                    processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {item_data_val.get('content_type')}).")
                    successfully_scraped_for_this_keyword += 1
                    main_text_for_llm: str = current_main_text

                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                        llm_model_to_use: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                        
                        # LLM Summary
                        current_major_step_count += 1
                        progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                        with progress_bar_placeholder.container(): 
                            st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_summary)
                        processing_log.append(f"       Generating LLM summary...")
                        summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                        item_data_val["llm_summary"] = summary
                        processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                        time.sleep(0.1) 

                        # LLM Extractions
                        for q_idx, extraction_query in enumerate(llm_extract_queries_input):
                            if not extraction_query.strip(): continue
                            current_major_step_count += 1
                            progress_text_llm_extract = f"LLM Extract Q{q_idx+1} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container(): 
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm_extract)
                            processing_log.append(f"      Extracting info for Q{q_idx+1}: '{extraction_query}'...")
                            extracted_info: Optional[str] = llm_processor.extract_specific_information(main_text_for_llm, extraction_query=extraction_query, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                            item_data_val[f"llm_extracted_info_q{q_idx+1}"] = extracted_info
                            processing_log.append(f"        Extracted (Q{q_idx+1}): {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1) 
                    results_data.append(item_data_val)
                else: 
                    processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
                    # If it's not a good scrape, we don't do LLM tasks, so we need to account for these skipped LLM steps in progress
                    current_major_step_count += total_llm_tasks_per_good_scrape
            time.sleep(0.2) 
        
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
            processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword # Account for these LLM tasks that won't happen

    with progress_bar_placeholder.container(): st.empty()

    topic_for_consolidation_for_batch: str
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics" 
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

    # --- CONSOLIDATED SUMMARY LOGIC REFINED ---
    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            focused_texts_for_consolidation: List[str] = []
            info_message_for_ui: str = ""
            llm_call_made_for_focused: bool = False
            focused_summary_llm_output: Optional[str] = None

            # 1. Attempt to collect items for Focused Consolidation
            if primary_llm_extract_query:
                processing_log.append(f"  Attempting to collect items for focused consolidation based on Main Query 1: '{primary_llm_extract_query}'")
                for item in results_data:
                    extraction_text_q1 = item.get("llm_extracted_info_q1")
                    if not extraction_text_q1: continue

                    item_relevancy_score_q1 = _parse_score_from_extraction(extraction_text_q1)
                    if item_relevancy_score_q1 is not None and item_relevancy_score_q1 >= 3:
                        content_after_score = ""
                        if extraction_text_q1.startswith("Relevancy Score:"):
                            parts = extraction_text_q1.split('\n', 1)
                            if len(parts) > 1:
                                content_after_score = parts[1].strip()
                        # If _parse_score_from_extraction got a score, it must start with "Relevancy Score: "
                        
                        is_content_valid = content_after_score and \
                                           not str(content_after_score).lower().startswith(("llm error", "no text content", "llm_processor:"))
                        
                        if is_content_valid:
                            focused_texts_for_consolidation.append(content_after_score)
                
                if focused_texts_for_consolidation:
                    processing_log.append(f"  Found {len(focused_texts_for_consolidation)} items meeting Main Query 1 (score >=3) criteria for focused summary.")
                    # Attempt to generate focused summary
                    llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                    llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                    
                    focused_summary_llm_output = llm_processor.generate_consolidated_summary(
                        summaries=tuple(focused_texts_for_consolidation),
                        topic_context=topic_for_consolidation_for_batch,
                        api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                        max_input_chars=app_config.llm.max_input_chars,
                        extraction_query_for_consolidation=primary_llm_extract_query
                    )
                    llm_call_made_for_focused = True

                    # Check if the LLM failed to use the focused items it was GIVEN
                    problematic_llm_responses = [
                        "llm_processor: no items met score >=3 for query", # example from user
                        "llm_processor_error: no items met score >=3",
                        "could not generate summary from the provided texts", # a generic one
                        "no suitable content provided for focused summary" 
                    ]
                    is_problematic_response = False
                    if focused_summary_llm_output:
                        for resp_text in problematic_llm_responses:
                            if resp_text in focused_summary_llm_output.lower():
                                is_problematic_response = True
                                break
                    
                    if is_problematic_response:
                        processing_log.append(f"  ⚠️ LLM indicated no suitable items for focused summary (returned: '{focused_summary_llm_output[:100]}...') despite being provided with {len(focused_texts_for_consolidation)} high-scoring items. Will fall back to general summary.")
                        info_message_for_ui = (
                            f"LLM_PROCESSOR_INFO: LLM could not generate a focused summary from items matching '{primary_llm_extract_query}' (score >=3). "
                            "Attempting general overview instead."
                        )
                        consolidated_summary_text_for_batch = None # Ensure fallback
                    elif focused_summary_llm_output and not focused_summary_llm_output.lower().startswith("llm_processor"):
                        consolidated_summary_text_for_batch = focused_summary_llm_output # Successful focused summary
                    else: # LLM call failed for other reasons or returned empty
                        processing_log.append(f"  Focused summary LLM call failed or returned empty. Will attempt general summary. LLM output: {focused_summary_llm_output}")
                        info_message_for_ui = (
                            f"LLM_PROCESSOR_INFO: Focused summary generation for '{primary_llm_extract_query}' failed. Attempting general overview."
                        )
                        consolidated_summary_text_for_batch = None # Ensure fallback
                else: # No items met >=3 criteria initially
                    processing_log.append(f"  ⚠️ No items met >=3 relevancy with valid content for Main Query 1 ('{primary_llm_extract_query}'). Will attempt general consolidation.")
                    info_message_for_ui = (
                        f"LLM_PROCESSOR_INFO: No items met score >=3 with valid content for Main Query 1 ('{primary_llm_extract_query}'). "
                        "Attempting general overview."
                    )
                    consolidated_summary_text_for_batch = None # Ensure fallback
            else: # No primary_llm_extract_query, so go directly to general
                processing_log.append(f"  No Main Query 1 provided. Attempting general consolidation using item summaries.")
                consolidated_summary_text_for_batch = None # Ensure general summary path

            # 2. Attempt General Consolidation if no successful focused summary yet
            if consolidated_summary_text_for_batch is None:
                general_texts_for_consolidation: List[str] = []
                for item in results_data:
                    summary_text = item.get("llm_summary")
                    is_summary_valid = summary_text and not str(summary_text).lower().startswith(("llm error", "no text content", "llm_processor:"))
                    if is_summary_valid and summary_text.strip():
                        general_texts_for_consolidation.append(summary_text)
                
                if general_texts_for_consolidation:
                    processing_log.append(f"  Found {len(general_texts_for_consolidation)} items with general summaries for fallback consolidation.")
                    llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                    llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                    
                    general_overview_from_llm = llm_processor.generate_consolidated_summary(
                        summaries=tuple(general_texts_for_consolidation),
                        topic_context=topic_for_consolidation_for_batch,
                        api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                        max_input_chars=app_config.llm.max_input_chars,
                        extraction_query_for_consolidation=None # General summary
                    )
                    if info_message_for_ui: # Prepend if there was a focused attempt that led to this fallback
                        consolidated_summary_text_for_batch = info_message_for_ui + "\n\n--- General Overview ---\n" + (general_overview_from_llm or "LLM could not generate general overview from available summaries.")
                    else: # This was a direct attempt at general summary
                        consolidated_summary_text_for_batch = general_overview_from_llm if general_overview_from_llm else "LLM_PROCESSOR_ERROR: Could not generate a general consolidated overview from available item summaries."
                else: # No general summaries found either
                    if info_message_for_ui : # Focused attempt failed, and general also has no content
                         consolidated_summary_text_for_batch = info_message_for_ui + " Additionally, no general summaries were available for fallback."
                    else: # No primary query AND no general summaries
                        consolidated_summary_text_for_batch = "LLM_PROCESSOR_ERROR: No valid content (neither focused extractions nor general summaries) found to generate a consolidated overview."
            
            if consolidated_summary_text_for_batch:
                 processing_log.append(f"  Final Consolidated Overview (first 150 chars): {str(consolidated_summary_text_for_batch)[:150]}...")
            else: # Should ideally be caught by above, but as a final safeguard
                 consolidated_summary_text_for_batch = "LLM_PROCESSOR_ERROR: Failed to generate any consolidated overview."
                 processing_log.append(f"  ❌ {consolidated_summary_text_for_batch}")
    
    # Google Sheets Writing
    if sheet_writing_enabled and gs_worksheet: 
        if results_data or consolidated_summary_text_for_batch:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
            processing_log.append(f"\n💾 Writing batch data to Google Sheets...")
            active_extraction_queries_for_sheet = [q for q in llm_extract_queries_input if q.strip()]
            write_successful: bool = data_storage.write_batch_summary_and_items_to_sheet(
                worksheet=gs_worksheet, 
                batch_timestamp=batch_process_timestamp_for_sheet, 
                consolidated_summary=consolidated_summary_text_for_batch, 
                topic_context=topic_for_consolidation_for_batch, 
                item_data_list=results_data, 
                extraction_queries_list=active_extraction_queries_for_sheet
            )
            if write_successful: 
                processing_log.append(f"  Batch data written to Google Sheets.")
            else: 
                processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    elif gsheets_secrets_present and not sheet_writing_enabled : 
        processing_log.append("\n⚠️ Google Sheets connection failed earlier. Data not saved to sheet.")
    elif not gsheets_secrets_present: 
        processing_log.append("\nℹ️ Google Sheets integration not configured. Data not saved to sheet.")

    # Final messages
    if results_data or consolidated_summary_text_for_batch :
        is_info_or_error_summary = consolidated_summary_text_for_batch and \
                                   (consolidated_summary_text_for_batch.lower().startswith("llm_processor_info:") or \
                                    consolidated_summary_text_for_batch.lower().startswith("llm_processor_error:"))
        if not is_info_or_error_summary and consolidated_summary_text_for_batch: # Added check for actual summary content
            st.success("All processing complete! Consolidated overview generated.")
        elif consolidated_summary_text_for_batch: # It's an info/error message
            st.warning(f"Processing complete. Note on consolidated overview: {consolidated_summary_text_for_batch}")
        else: # No summary text at all
             st.warning("Processing complete, but no consolidated overview was generated and no data found for it.")
    else: 
        st.warning("Processing complete, but no data was generated.")
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display

# end of modules/process_manager.py
