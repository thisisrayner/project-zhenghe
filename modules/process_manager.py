# modules/process_manager.py
# Version 1.4.0:
# - Implemented conditional LLM request throttling based on 'num_results_wanted_per_keyword'
#   and new configurations in AppConfig (llm_item_request_delay_seconds,
#   llm_throttling_threshold_results).
# Version 1.3.7:
# - Passed secondary_llm_extract_query (Q2) to llm_processor.generate_consolidated_summary
#   for the new `secondary_query_for_enrichment` parameter, enabling Q2 to enrich
#   the Q1-focused consolidated summary.
# Version 1.3.6:
# - Enhanced logging for focused consolidated summary, returns source details.
# - Corrected NameError for `config.AppConfig` type hint.
# Version 1.3.5:
# - Updated call to `llm_processor.generate_search_queries` to include Q2.

"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
"""
import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple, TypedDict
from modules import config, search_engine, scraper, llm_processor, data_storage

# Define a type for the focused summary source details
class FocusedSummarySource(TypedDict):
    url: str
    query_type: str # "Q1" or "Q2"
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
    num_results_wanted_per_keyword: int, # Value from the UI slider
    gs_worksheet: Optional[Any],
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    processing_log: List[str] = ["Processing started..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
    initial_keywords_for_display: Set[str] = set(k.lower() for k in initial_keywords_list)
    llm_generated_keywords_set_for_display: Set[str] = set()

    if not initial_keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        # Ensure the return tuple matches the expected structure even on error
        return ["ERROR: No keywords provided."], [], None, initial_keywords_for_display, llm_generated_keywords_set_for_display, []

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

    # --- Throttling Configuration ---
    llm_item_delay_seconds = app_config.llm.llm_item_request_delay_seconds
    throttling_threshold = app_config.llm.llm_throttling_threshold_results

    apply_throttling = (
        num_results_wanted_per_keyword >= throttling_threshold and
        llm_item_delay_seconds > 0 and
        llm_key_available # Only throttle if LLM is being used
    )
    # --- End Throttling Configuration ---

    # LLM Query Generation (logic from your v1.3.7)
    if llm_key_available and initial_keywords_list:
        processing_log.append("\n🧠 Generating additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_query_gen: str = app_config.llm.google_gemini_model
            if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize # or specific model for query gen if different
            with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."):
                generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                    original_keywords=tuple(initial_keywords_list),
                    specific_info_query=primary_llm_extract_query,
                    specific_info_query_2=secondary_llm_extract_query,
                    num_queries_to_generate=num_llm_terms_to_generate,
                    api_key=llm_api_key_to_use_qgen,
                    model_name=llm_model_for_query_gen
                )
            if generated_queries:
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}"); current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                for gq in generated_queries:
                    if gq.lower() not in current_runtime_keywords_lower: keywords_list_val_runtime.append(gq); current_runtime_keywords_lower.add(gq.lower()); temp_llm_generated_set.add(gq.lower())
                llm_generated_keywords_set_for_display = temp_llm_generated_set; processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else: processing_log.append("  ⚠️ LLM did not generate new queries.")
        else: processing_log.append("  ℹ️ No additional LLM queries requested (or needed based on input).")

    # Progress Bar Setup (logic from your v1.3.7)
    oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10; est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword : est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: 
        total_llm_tasks_per_good_scrape += 1 # For summary
        total_llm_tasks_per_good_scrape += len([q for q in [primary_llm_extract_query, secondary_llm_extract_query] if q and q.strip()]) 

    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                          (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: 
        total_major_steps_for_progress += 1 # For LLM query generation step
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty() # For progress bar
    status_placeholder = st.empty()      # For status messages like throttling

    if apply_throttling: # Inform user if throttling is active
        throttle_init_message = (
            f"ℹ️ LLM Throttling ACTIVE: A delay of {llm_item_delay_seconds:.1f}s "
            f"will be applied after LLM processing for each item "
            f"(threshold: {throttling_threshold} results/keyword)."
        )
        processing_log.append(throttle_init_message)
        # Display prominently if desired, or just log it.
        # For now, it will be in the processing log displayed at the end.
        # To show immediately: st.info(throttle_init_message) - but might be too much if run often.

    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        current_major_step_count +=1
        progress_text = "LLM Query Generation Complete..."
        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
        with progress_bar_placeholder.container(): 
            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text)

    # --- Item Processing Loop (from your v1.3.7, with throttling integrated) ---
    for keyword_val in keywords_list_val_runtime: 
        processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        if not (app_config.google_search.api_key and app_config.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured.") # This is good.
            processing_log.append(f"  ❌ ERROR: Halting for '{keyword_val}'. Google Search not configured.")
            current_major_step_count += est_urls_to_fetch_per_keyword 
            current_major_step_count += num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape
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
            current_major_step_count += est_urls_to_fetch_per_keyword 
            current_major_step_count += num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape
            continue

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                skipped_google_results = len(search_results_items_val) - search_item_idx
                processing_log.append(f"  Reached target of {num_results_wanted_per_keyword} for '{keyword_val}'. Skipping {skipped_google_results} Google result(s).")
                current_major_step_count += skipped_google_results 
                break 
            
            current_major_step_count += 1 
            url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: 
                processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping.")
                continue

            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            status_placeholder.text(progress_text_scrape) # Update status
            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            with progress_bar_placeholder.container(): 
                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_scrape)

            processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
            
            item_data_val: Dict[str, Any] = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val, 
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "page_title": scraped_content_val.get('scraped_title'), # Using 'page_title' as a more generic key
                "meta_description": scraped_content_val.get('meta_description'), 
                "og_title": scraped_content_val.get('og_title'), "og_description": scraped_content_val.get('og_description'),
                "main_content_display": scraped_content_val.get('main_text'), # Key for Excel export and potentially display
                "pdf_document_title": scraped_content_val.get('pdf_doc_title'), # Assuming scraper might provide this
                "is_pdf": scraped_content_val.get('content_type') == 'application/pdf',
                "source_query_type": "LLM-Generated" if keyword_val.lower() in llm_generated_keywords_set_for_display else "Original",
                "scraping_error": scraped_content_val.get('error'), 
                "llm_summary": None, "llm_extracted_info_q1": None, "llm_relevancy_score_q1": None, 
                "llm_extracted_info_q2": None, "llm_relevancy_score_q2": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            made_llm_call_for_item = False # Reset for each item
            if scraped_content_val.get('error'): 
                processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
            else:
                min_main_text_length: int = 200 # From your v1.3.7
                current_main_text: str = scraped_content_val.get('main_text', '')
                is_good_scrape: bool = ( # Logic from your v1.3.7
                    current_main_text and 
                    len(current_main_text.strip()) >= min_main_text_length and 
                    "could not extract main content" not in current_main_text.lower() and 
                    "not processed for main text" not in current_main_text.lower() and 
                    not str(current_main_text).startswith("SCRAPER_INFO:")
                )

                if is_good_scrape:
                    processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {scraped_content_val.get('content_type')}).")
                    successfully_scraped_for_this_keyword += 1
                    main_text_for_llm: str = current_main_text

                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                        llm_model_to_use: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                        
                        # LLM Summary
                        current_major_step_count += 1
                        progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                        status_placeholder.text(progress_text_llm_summary) # Update status
                        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                        with progress_bar_placeholder.container(): 
                            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_summary)
                        processing_log.append(f"       Generating LLM summary...")
                        summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                        item_data_val["llm_summary"] = summary
                        processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                        made_llm_call_for_item = True
                        # Removed time.sleep(0.1) from here, main throttle is at the end of item LLM calls

                        # LLM Extractions (Q1 and Q2)
                        current_extraction_queries = []
                        if primary_llm_extract_query: current_extraction_queries.append(primary_llm_extract_query)
                        if secondary_llm_extract_query: current_extraction_queries.append(secondary_llm_extract_query)

                        for q_idx, extraction_query in enumerate(current_extraction_queries):
                            if not extraction_query.strip(): continue # Should not happen if list is pre-filtered
                            query_label = f"q{q_idx+1}" # q1 or q2

                            current_major_step_count += 1
                            progress_text_llm_extract = f"LLM Extract Q{q_idx+1} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            status_placeholder.text(progress_text_llm_extract) # Update status
                            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                            with progress_bar_placeholder.container(): 
                                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_extract)
                            
                            processing_log.append(f"      Extracting info for Q{q_idx+1}: '{extraction_query}'...")
                            extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                main_text_for_llm, 
                                extraction_query=extraction_query, 
                                api_key=llm_api_key_to_use, 
                                model_name=llm_model_to_use, # Consider specific model for extraction if different
                                max_input_chars=app_config.llm.max_input_chars
                            )
                            
                            # Parse score and get content separately
                            score, content_without_score = llm_processor._parse_score_and_get_content(extracted_info_full)

                            item_data_val[f"llm_extracted_info_{query_label}"] = content_without_score
                            item_data_val[f"llm_relevancy_score_{query_label}"] = score
                            processing_log.append(f"        Extracted (Q{q_idx+1}): Score={score}, Content='{str(content_without_score)[:70] if content_without_score else 'Failed/Empty'}'...")
                            made_llm_call_for_item = True
                            # Removed time.sleep(0.1) from here

                    results_data.append(item_data_val)
                else: 
                    processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
            
            # --- Apply Conditional Throttling Delay ---
            if apply_throttling and made_llm_call_for_item:
                delay_message = f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s..."
                processing_log.append(f"    {delay_message}") # Add to log for this item
                status_placeholder.text(delay_message) # Show in UI
                time.sleep(llm_item_delay_seconds)
                # Restore general status after delay, or clear it
                status_placeholder.text(f"Continuing processing...") # Or more specific next step
            elif not apply_throttling and made_llm_call_for_item:
                 time.sleep(0.2) # Retain the small existing delay if not full throttling
            elif not made_llm_call_for_item: # If no LLM call was made, just a very short pause
                 time.sleep(0.1)


        # End of loop for search_results_items_val
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
            processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword 
    # --- End of Item Processing Loop (keywords_list_val_runtime) ---

    # Progress bar final update and clear status
    final_progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 1.0
    final_progress_text = "All item processing complete. Generating final report..."
    with progress_bar_placeholder.container():
        st.progress(min(max(final_progress_value, 0.0), 1.0), text=final_progress_text)
    status_placeholder.text(final_progress_text) # Keep final status

    if abs(final_progress_value - 1.0) > 0.01 and total_major_steps_for_progress > 0 and final_progress_value <=1.0 :
            processing_log.append(f"  DEBUG: Final progress: current_steps={current_major_step_count}, total_steps={total_major_steps_for_progress}, value={final_progress_value}")

    # Consolidated Summary Generation (logic from your v1.3.7)
    topic_for_consolidation_for_batch: str
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics"
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        status_placeholder.text("Generating consolidated overview...") # Update status
        with st.spinner("Generating consolidated overview..."): # This spinner is good
            temp_focused_texts_for_llm: List[str] = []
            processed_item_texts_for_focused = set() # To store unique text snippets

            # focused_summary_source_details is initialized as an empty list at the start
            if primary_llm_extract_query or secondary_llm_extract_query:
                for item in results_data:
                    item_url = item.get("url", "Unknown URL")
                    # Q1 processing for focused summary
                    if primary_llm_extract_query:
                        extraction_text_q1 = item.get("llm_extracted_info_q1")
                        score_q1 = item.get("llm_relevancy_score_q1") # Use pre-parsed score
                        if extraction_text_q1 and score_q1 is not None and score_q1 >= 3:
                            source_entry_q1: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q1",
                                "query_text": primary_llm_extract_query, "score": score_q1,
                                "llm_output_text": extraction_text_q1
                            }
                            # Avoid duplicate source entries if item contributed to both Q1 & Q2 focus
                            if not any(d['url'] == item_url and d['query_type'] == 'Q1' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q1)
                            # Add text to LLM input if not already added from this item (e.g. from Q2)
                            if extraction_text_q1 not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q1)
                                processed_item_texts_for_focused.add(extraction_text_q1)
                    
                    # Q2 processing for focused summary
                    if secondary_llm_extract_query:
                        extraction_text_q2 = item.get("llm_extracted_info_q2")
                        score_q2 = item.get("llm_relevancy_score_q2") # Use pre-parsed score
                        if extraction_text_q2 and score_q2 is not None and score_q2 >= 3:
                            source_entry_q2: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q2",
                                "query_text": secondary_llm_extract_query, "score": score_q2,
                                "llm_output_text": extraction_text_q2
                            }
                            if not any(d['url'] == item_url and d['query_type'] == 'Q2' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q2)
                            if extraction_text_q2 not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q2)
                                processed_item_texts_for_focused.add(extraction_text_q2)
            
            final_texts_for_llm = temp_focused_texts_for_llm

            if final_texts_for_llm: # If there are texts for focused summary
                # ... (Focused summary generation logic from your v1.3.7, ensure it uses primary_llm_extract_query and secondary_llm_extract_query correctly) ...
                llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                if not primary_llm_extract_query and secondary_llm_extract_query: # If Q1 is blank but Q2 exists
                    llm_context_for_focused_summary = secondary_llm_extract_query

                processing_log.append(f"\n📋 Preparing inputs for FOCUSED consolidated summary:")
                # ... (rest of logging for focused summary details, as in your v1.3.7) ...

                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(
                    summaries=tuple(final_texts_for_llm),
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                    max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=llm_context_for_focused_summary,
                    secondary_query_for_enrichment=secondary_llm_extract_query if secondary_llm_extract_query and secondary_llm_extract_query.strip() else None
                )
                # ... (Error handling for focused summary from your v1.3.7) ...
                is_llm_call_problematic = False # Copied from your logic
                # ... (rest of your logic for checking problematic summary and setting consolidated_summary_text_for_batch)

            else: # Fallback to General Summary (logic from your v1.3.7)
                # ... (General summary generation logic from your v1.3.7) ...
                # Ensure this path is correctly handled and consolidated_summary_text_for_batch is set.
                general_overview_info_prefix = "LLM_PROCESSOR_INFO: General overview as follows."
                # ... (rest of your general summary logic) ...

        # Clear spinner area after consolidated summary
        status_placeholder.text("Report generation complete.") 
    
    # ... (Rest of the function: Sheet writing, final UI messages from your v1.3.7) ...
    # Ensure the return tuple matches the annotation exactly.
    # The original annotation was:
    # Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]
    # My earlier suggestion included llm_augmented_insights_text, which is not in your v1.3.7 signature.
    # Sticking to your v1.3.7 signature:
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display, focused_summary_source_details

# end of modules/process_manager.py
