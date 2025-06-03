# modules/process_manager.py
# Version 1.4.1: (as per my previous suggestion for this file)
# - Added extensive print() debugging for log creation and sheet writing conditions.
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
from modules import config, search_engine, scraper, llm_processor, data_storage # Ensure data_storage is imported
import traceback # For detailed error printing

# Define a type for the focused summary source details
class FocusedSummarySource(TypedDict):
    url: str
    query_type: str # "Q1" or "Q2"
    query_text: str
    score: int
    llm_output_text: str # The text from which score was parsed, and used for LLM input


def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]: # This is from your v1.3.7
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
    llm_extract_queries_input: List[str], # This is active_llm_extract_queries from app.py
    num_results_wanted_per_keyword: int, # Value from the UI slider
    gs_worksheet: Optional[Any], # gspread.Worksheet or None
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    print("DEBUG (process_manager): run_search_and_analysis called.") # DEBUG
    processing_log: List[str] = ["Processing started..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = [] # For sources UI
    initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
    initial_keywords_for_display: Set[str] = set(k.lower() for k in initial_keywords_list)
    llm_generated_keywords_set_for_display: Set[str] = set()

    if not initial_keywords_list:
        st.sidebar.error("Please enter at least one keyword.")
        processing_log.append("ERROR: No keywords provided.")
        print("DEBUG (process_manager): No keywords provided, returning early.") # DEBUG
        return processing_log, [], None, initial_keywords_for_display, llm_generated_keywords_set_for_display, []

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
    
    print(f"DEBUG (process_manager): Primary Q: '{primary_llm_extract_query}', Secondary Q: '{secondary_llm_extract_query}'") # DEBUG

    # --- Throttling Configuration ---
    llm_item_delay_seconds = app_config.llm.llm_item_request_delay_seconds
    throttling_threshold = app_config.llm.llm_throttling_threshold_results

    apply_throttling = (
        num_results_wanted_per_keyword >= throttling_threshold and
        llm_item_delay_seconds > 0 and
        llm_key_available
    )
    print(f"DEBUG (process_manager): Throttling check: num_results={num_results_wanted_per_keyword}, threshold={throttling_threshold}, delay={llm_item_delay_seconds}, llm_key={llm_key_available}. Apply_throttling={apply_throttling}") # DEBUG
    # --- End Throttling Configuration ---

    if llm_key_available and initial_keywords_list:
        processing_log.append("\n🧠 Generating additional search queries with LLM...")
        num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
        if num_llm_terms_to_generate > 0:
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_query_gen: str = app_config.llm.google_gemini_model
            if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
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

    oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 10; est_urls_to_fetch_per_keyword: int = min(max_google_fetch_per_keyword, int(num_results_wanted_per_keyword * oversample_factor))
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword : est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: 
        total_llm_tasks_per_good_scrape += 1 # For summary
        # Count active extraction queries
        active_extraction_queries_count = 0
        if primary_llm_extract_query and primary_llm_extract_query.strip(): active_extraction_queries_count +=1
        if secondary_llm_extract_query and secondary_llm_extract_query.strip(): active_extraction_queries_count +=1
        total_llm_tasks_per_good_scrape += active_extraction_queries_count

    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                          (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: 
        total_major_steps_for_progress += 1 
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty()
    status_placeholder = st.empty()      # For status messages like throttling

    if apply_throttling:
        throttle_init_message = (
            f"ℹ️ LLM Throttling ACTIVE: Delay of {llm_item_delay_seconds:.1f}s "
            f"after each item's LLM processing (threshold: {throttling_threshold} results/keyword)."
        )
        processing_log.append(throttle_init_message)
        print(f"DEBUG (process_manager): {throttle_init_message}") # DEBUG
        # You might want to display this in the UI more prominently if desired, e.g., st.info()
        # For now, it will be in the processing_log.

    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: # Check if LLM query gen step exists
        current_major_step_count +=1
        progress_text = "LLM Query Generation Complete..."
        status_placeholder.text(progress_text) # Update status
        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
        with progress_bar_placeholder.container(): 
            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text)

    # --- Item Processing Loop ---
    for keyword_val in keywords_list_val_runtime: 
        processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        if not (app_config.google_search.api_key and app_config.google_search.cse_id):
            # This st.error will appear in the main app area.
            st.error("Google Search API Key or CSE ID not configured. Cannot perform searches.")
            processing_log.append(f"  ❌ ERROR: Halting search for '{keyword_val}'. Google Search not configured.")
            # Update progress for skipped steps for this keyword
            current_major_step_count += est_urls_to_fetch_per_keyword # Skipped fetching
            current_major_step_count += num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape # Skipped LLM for these
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
                current_major_step_count += skipped_google_results # Account for skipped fetches
                # No LLM tasks for these skipped ones as they weren't even scraped.
                break 
            
            current_major_step_count += 1 # Increment for the scraping attempt
            url_to_scrape_val: Optional[str] = search_item_val.get('link')
            if not url_to_scrape_val: 
                processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping.")
                continue

            progress_text_scrape = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            status_placeholder.text(progress_text_scrape)
            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            with progress_bar_placeholder.container(): 
                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_scrape)

            processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
            
            # Prepare item_data_val dictionary
            item_data_val: Dict[str, Any] = {
                "keyword_searched": keyword_val, 
                "url": url_to_scrape_val, 
                "search_title": search_item_val.get('title'), 
                "search_snippet": search_item_val.get('snippet'),
                "page_title": scraped_content_val.get('scraped_title'), # scraper returns 'scraped_title'
                "meta_description": scraped_content_val.get('meta_description'), 
                "og_title": scraped_content_val.get('og_title'), 
                "og_description": scraped_content_val.get('og_description'),
                "main_content_display": scraped_content_val.get('main_text'), # For Excel/display
                "pdf_document_title": scraped_content_val.get('pdf_doc_title'), # From scraper if PDF
                "is_pdf": scraped_content_val.get('content_type') == 'application/pdf',
                "source_query_type": "LLM-Generated" if keyword_val.lower() in llm_generated_keywords_set_for_display else "Original",
                "scraping_error": scraped_content_val.get('error'), 
                "content_type": scraped_content_val.get('content_type'), # From scraper
                "llm_summary": None, 
                "llm_extracted_info_q1": None, "llm_relevancy_score_q1": None, 
                "llm_extracted_info_q2": None, "llm_relevancy_score_q2": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            made_llm_call_for_item = False # Reset for each item

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
                        
                        current_major_step_count += 1
                        progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                        status_placeholder.text(progress_text_llm_summary)
                        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                        with progress_bar_placeholder.container(): 
                            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_summary)
                        processing_log.append(f"       Generating LLM summary...")
                        summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                        item_data_val["llm_summary"] = summary
                        processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                        made_llm_call_for_item = True
                        
                        # Iterating through Q1 and Q2 if they exist
                        queries_to_process = []
                        if primary_llm_extract_query and primary_llm_extract_query.strip():
                            queries_to_process.append({"query_text": primary_llm_extract_query, "id": "q1"})
                        if secondary_llm_extract_query and secondary_llm_extract_query.strip():
                            queries_to_process.append({"query_text": secondary_llm_extract_query, "id": "q2"})

                        for q_idx_enum, query_info in enumerate(queries_to_process):
                            extraction_query = query_info["query_text"]
                            query_id_label = query_info["id"] # "q1" or "q2"
                            query_display_idx = q_idx_enum + 1 # 1 or 2 for display

                            current_major_step_count += 1
                            progress_text_llm_extract = f"LLM Extract Q{query_display_idx} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            status_placeholder.text(progress_text_llm_extract)
                            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                            with progress_bar_placeholder.container(): 
                                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_extract)
                            
                            processing_log.append(f"      Extracting info for Q{query_display_idx}: '{extraction_query}'...")
                            # Assume llm_processor.extract_specific_information returns the full string with score
                            extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                main_text_for_llm, 
                                extraction_query=extraction_query, 
                                api_key=llm_api_key_to_use, 
                                model_name=llm_model_to_use, # Or a specific extraction model
                                max_input_chars=app_config.llm.max_input_chars
                            )
                            
                            # Use the local _parse_score_from_extraction to get score
                            # And then use llm_processor's version to get content (if it exists, otherwise implement similar logic)
                            parsed_score = _parse_score_from_extraction(extracted_info_full)
                            # Assuming llm_processor has a way to get content without score or you implement it:
                            # For now, let's assume extracted_info_full is what we store if no separate content getter
                            content_without_score = extracted_info_full 
                            if hasattr(llm_processor, '_parse_score_and_get_content'):
                                _, content_without_score_temp = llm_processor._parse_score_and_get_content(extracted_info_full)
                                if content_without_score_temp is not None: # Check if parsing was successful
                                    content_without_score = content_without_score_temp
                            elif parsed_score is not None and extracted_info_full: # Manual stripping if needed
                                 try: content_without_score = extracted_info_full.split('\n', 1)[1] if '\n' in extracted_info_full else extracted_info_full
                                 except IndexError: pass # Keep full string if split fails weirdly

                            item_data_val[f"llm_extracted_info_{query_id_label}"] = content_without_score
                            item_data_val[f"llm_relevancy_score_{query_id_label}"] = parsed_score
                            processing_log.append(f"        Extracted (Q{query_display_idx}): Score={parsed_score}, Content='{str(content_without_score)[:70] if content_without_score else 'Failed/Empty'}'...")
                            made_llm_call_for_item = True
                    results_data.append(item_data_val)
                else: 
                    processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
            
            # --- Apply Conditional Throttling Delay ---
            if apply_throttling and made_llm_call_for_item:
                delay_message = f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s..."
                processing_log.append(f"    {delay_message}")
                status_placeholder.text(delay_message)
                print(f"DEBUG (process_manager): {delay_message}") # DEBUG
                time.sleep(llm_item_delay_seconds)
                status_placeholder.text(f"Continuing processing...")
            elif not apply_throttling and made_llm_call_for_item: # Original small delay if not full throttling
                 time.sleep(0.2) 
            elif not made_llm_call_for_item: # Smallest delay if no LLM call
                 time.sleep(0.1)
        
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
            processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword 
    # --- End of Item Processing Loop ---

    final_progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 1.0
    final_progress_text = "All item processing complete. Generating final report..."
    with progress_bar_placeholder.container():
        st.progress(min(max(final_progress_value, 0.0), 1.0), text=final_progress_text)
    status_placeholder.text(final_progress_text)

    if abs(final_progress_value - 1.0) > 0.01 and total_major_steps_for_progress > 0 and final_progress_value <=1.0 : # Check if progress didn't hit 100%
         processing_log.append(f"  DEBUG: Final progress: current_steps={current_major_step_count}, total_steps={total_major_steps_for_progress}, value={final_progress_value}")

    topic_for_consolidation_for_batch: str
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics"
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        status_placeholder.text("Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            temp_focused_texts_for_llm: List[str] = []
            processed_item_texts_for_focused = set() # To store unique text snippets

            if primary_llm_extract_query or secondary_llm_extract_query:
                for item in results_data:
                    item_url = item.get("url", "Unknown URL")
                    if primary_llm_extract_query:
                        extraction_text_q1 = item.get("llm_extracted_info_q1") # This should be content only
                        score_q1 = item.get("llm_relevancy_score_q1") # This is the parsed score
                        if extraction_text_q1 and score_q1 is not None and score_q1 >= 3:
                            # llm_output_text should be the original full text from LLM if different
                            # For now, assume extraction_text_q1 is what we use.
                            source_entry_q1: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q1",
                                "query_text": primary_llm_extract_query, "score": score_q1,
                                "llm_output_text": item.get("llm_extracted_info_q1_full", extraction_text_q1) # Prefer full if available
                            }
                            if not any(d['url'] == item_url and d['query_type'] == 'Q1' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q1)
                            if extraction_text_q1 not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q1)
                                processed_item_texts_for_focused.add(extraction_text_q1)
                    if secondary_llm_extract_query:
                        extraction_text_q2 = item.get("llm_extracted_info_q2") # Content only
                        score_q2 = item.get("llm_relevancy_score_q2") # Parsed score
                        if extraction_text_q2 and score_q2 is not None and score_q2 >= 3:
                            source_entry_q2: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q2",
                                "query_text": secondary_llm_extract_query, "score": score_q2,
                                "llm_output_text": item.get("llm_extracted_info_q2_full", extraction_text_q2)
                            }
                            if not any(d['url'] == item_url and d['query_type'] == 'Q2' for d in focused_summary_source_details):
                                 focused_summary_source_details.append(source_entry_q2)
                            if extraction_text_q2 not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q2)
                                processed_item_texts_for_focused.add(extraction_text_q2)
            
            final_texts_for_llm = temp_focused_texts_for_llm

            if final_texts_for_llm:
                llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                if not primary_llm_extract_query and secondary_llm_extract_query:
                    llm_context_for_focused_summary = secondary_llm_extract_query

                processing_log.append(f"\n📋 Preparing inputs for FOCUSED consolidated summary:")
                processing_log.append(f"  Central Query Theme (typically based on Q1): '{llm_context_for_focused_summary}'")
                if secondary_llm_extract_query and (any(details['query_type'] == 'Q2' and details['query_text'] == secondary_llm_extract_query for details in focused_summary_source_details)):
                    processing_log.append(f"  Secondary Query for Enrichment (Q2): '{secondary_llm_extract_query}' will be used as relevant snippets are included.")
                
                if focused_summary_source_details: # This list now holds the details
                    processing_log.append(f"  Found {len(focused_summary_source_details)} source snippet(s) (from Q1/Q2 extractions scoring >=3/5) that contributed text for the LLM:")
                    sorted_source_details = sorted(focused_summary_source_details, key=lambda x: (x['url'], x['query_type']))
                    for source_item in sorted_source_details:
                        log_url = source_item['url']
                        log_q_type = source_item['query_type']
                        log_q_text_short = source_item['query_text'][:30] + "..." if len(source_item['query_text']) > 30 else source_item['query_text']
                        log_score = source_item['score']
                        processing_log.append(f"    - From: {log_q_type} ('{log_q_text_short}') on URL: {log_url} (Score: {log_score}/5)")
                else: # This else should ideally not be hit if final_texts_for_llm is populated from focused_summary_source_details logic
                     processing_log.append(f"  Found {len(final_texts_for_llm)} unique text snippet(s) for LLM input (details not itemized in this pass - this might be an issue).")

                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                
                generated_focused_summary = llm_processor.generate_consolidated_summary(
                    summaries=tuple(final_texts_for_llm), # These should be content-only texts
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                    max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=llm_context_for_focused_summary,
                    secondary_query_for_enrichment=secondary_llm_extract_query if secondary_llm_extract_query and secondary_llm_extract_query.strip() else None
                )

                is_llm_call_problematic = False
                if not generated_focused_summary: is_llm_call_problematic = True
                else:
                    problematic_substrings = ["llm_processor", "could not generate", "no items met score", "no suitable content", "error:"]
                    for sub in problematic_substrings:
                        if sub in str(generated_focused_summary).lower():
                            is_llm_call_problematic = True
                            break

                if is_llm_call_problematic:
                    processing_log.append(f"  ❌ LLM failed to generate a FOCUSED summary from the {len(final_texts_for_llm)} provided high-scoring Q1/Q2 item(s). LLM Output (first 100 chars): {str(generated_focused_summary)[:100]}")
                    error_query_context_msg = ""
                    if primary_llm_extract_query: error_query_context_msg += f" Q1 ('{primary_llm_extract_query}')"
                    if secondary_llm_extract_query: error_query_context_msg += f"{' or ' if primary_llm_extract_query else ''}Q2 ('{secondary_llm_extract_query}')"
                    consolidated_summary_text_for_batch = (
                        f"LLM_PROCESSOR_ERROR: The LLM failed to generate a focused consolidated summary based on "
                        f"{len(final_texts_for_llm)} item(s) that met score >=3 criteria for specific queries{error_query_context_msg}. "
                        f"The LLM processor reported: \"{str(generated_focused_summary)[:100]}...\""
                    )
                else:
                    consolidated_summary_text_for_batch = generated_focused_summary
                    processing_log.append(f"  ✔️ Successfully generated FOCUSED consolidated summary.")
            else: # General summary fallback
                general_overview_info_prefix = "LLM_PROCESSOR_INFO: General overview as follows."
                log_message_reason = ""
                if primary_llm_extract_query or secondary_llm_extract_query:
                    log_message_reason = f" (No items met score >=3 criteria for "
                    if primary_llm_extract_query: log_message_reason += f"Q1:'{primary_llm_extract_query}'"
                    if secondary_llm_extract_query: log_message_reason += f"{' or ' if primary_llm_extract_query else ''}Q2:'{secondary_llm_extract_query}'"
                    log_message_reason += ".)"
                else:
                    log_message_reason = " (No specific queries were provided for focused summary.)"
                processing_log.append(f"  {general_overview_info_prefix}{log_message_reason} Attempting general overview from item summaries.")

                general_texts_for_consolidation: List[str] = []
                for item in results_data:
                    summary_text = item.get("llm_summary")
                    is_summary_valid = summary_text and not str(summary_text).lower().startswith(("llm error", "no text content", "llm_processor:"))
                    if is_summary_valid and summary_text.strip():
                        general_texts_for_consolidation.append(summary_text)

                if general_texts_for_consolidation:
                    processing_log.append(f"  Attempting GENERAL consolidated overview using {len(general_texts_for_consolidation)} item summaries.")
                    llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                    llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                    
                    generated_general_overview = llm_processor.generate_consolidated_summary(
                        summaries=tuple(general_texts_for_consolidation),
                        topic_context=topic_for_consolidation_for_batch,
                        api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                        max_input_chars=app_config.llm.max_input_chars,
                        extraction_query_for_consolidation=None, # No primary query for general
                        secondary_query_for_enrichment=None # No secondary query for general
                    )
                    
                    if generated_general_overview and not str(generated_general_overview).lower().startswith("llm_processor_error:"): # Check for specific error prefix
                        consolidated_summary_text_for_batch = generated_general_overview
                        processing_log.append("  ✔️ Successfully generated GENERAL consolidated overview.")
                    else:
                        processing_log.append(f"  ❌ LLM failed to generate a GENERAL overview from item summaries. LLM Output: {str(generated_general_overview)[:150]}")
                        consolidated_summary_text_for_batch = (general_overview_info_prefix +
                                                              "\n\n--- General Overview ---\nLLM_PROCESSOR_ERROR: The LLM failed to generate a general consolidated overview from the available item summaries.")
                else: 
                    no_summaries_message_suffix = " Additionally, no valid general item summaries were found to generate a general overview."
                    log_message_text = general_overview_info_prefix
                    if log_message_reason: log_message_text = log_message_text.replace("." , "") + log_message_reason # Avoid double period
                    processing_log.append(f"  ❌ {log_message_text}{no_summaries_message_suffix}")
                    consolidated_summary_text_for_batch = general_overview_info_prefix + " However, no valid item summaries were available to generate it."
        status_placeholder.text("Report generation complete.") # Clear spinner after consolidated summary attempt

    elif not results_data and llm_key_available: # No results data, but LLM was available
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items were successfully scraped and processed to create a consolidated summary."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
    elif not llm_key_available: # LLM not available
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")

    # --- Google Sheets Writing ---
    print(f"DEBUG (process_manager) PRE-SHEET WRITE: sheet_writing_enabled={sheet_writing_enabled}") # DEBUG
    print(f"DEBUG (process_manager) PRE-SHEET WRITE: gs_worksheet is {'present and type: ' + str(type(gs_worksheet)) if gs_worksheet else 'None'}") # DEBUG
    print(f"DEBUG (process_manager) PRE-SHEET WRITE: results_data length={len(results_data)}") # DEBUG
    print(f"DEBUG (process_manager) PRE-SHEET WRITE: consolidated_summary_text_for_batch is {'present' if consolidated_summary_text_for_batch else 'None'}") # DEBUG
    
    if sheet_writing_enabled and gs_worksheet:
        processing_log.append(f"\n💾 Checking conditions to write batch data to Google Sheets...") # DEBUG
        if results_data or consolidated_summary_text_for_batch:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
            processing_log.append(f"  Attempting to write batch data to Google Sheets at {batch_process_timestamp_for_sheet}...")
            print(f"DEBUG (process_manager): Calling data_storage.write_batch_summary_and_items_to_sheet") # DEBUG
            active_extraction_queries_for_sheet = [q for q in llm_extract_queries_input if q and q.strip()]
            
            write_successful: bool = False
            try:
                write_successful = data_storage.write_batch_summary_and_items_to_sheet(
                    worksheet=gs_worksheet,
                    batch_timestamp=batch_process_timestamp_for_sheet,
                    consolidated_summary=consolidated_summary_text_for_batch,
                    topic_context=topic_for_consolidation_for_batch, # Use calculated topic context
                    item_data_list=results_data,
                    extraction_queries_list=active_extraction_queries_for_sheet
                )
            except Exception as e_write_sheet:
                error_msg = f"  ❌ CRITICAL ERROR calling write_batch_summary_and_items_to_sheet: {e_write_sheet}"
                processing_log.append(error_msg)
                print(f"DEBUG (process_manager): {error_msg}") # DEBUG
                print(traceback.format_exc()) # DEBUG
                
            if write_successful:
                processing_log.append(f"  ✔️ Batch data written to Google Sheets successfully.")
                print(f"DEBUG (process_manager): Batch data write reported successful.") # DEBUG
            else:
                processing_log.append(f"  ❌ Failed to write batch data to Google Sheets (write_successful is False from data_storage or due to caught exception).")
                print(f"DEBUG (process_manager): Batch data write reported as FAILED (write_successful=False).") # DEBUG
        else:
            processing_log.append(f"  ℹ️ No data (results or summary) to write to Google Sheets for this batch.")
            print(f"DEBUG (process_manager): No data (results or summary) to write to Google Sheets.") # DEBUG
    elif gsheets_secrets_present and not sheet_writing_enabled :
        processing_log.append("\n⚠️ Google Sheets connection failed earlier or sheet object is invalid. Data not saved to sheet.")
        print(f"DEBUG (process_manager): GSheets secrets present, but writing not enabled (gs_worksheet type: {type(gs_worksheet)}).") # DEBUG
    elif not gsheets_secrets_present:
        processing_log.append("\nℹ️ Google Sheets integration not configured. Data not saved to sheet.")
        print(f"DEBUG (process_manager): GSheets secrets not present.") # DEBUG
    else: # Catch all for sheet writing not happening
        processing_log.append("\nℹ️ Google Sheets writing skipped (general conditions not met - e.g., sheet_writing_enabled or gs_worksheet missing).")
        print(f"DEBUG (process_manager): Google Sheets writing skipped (general conditions not met). sheet_writing_enabled={sheet_writing_enabled}, gs_worksheet_present={bool(gs_worksheet)}")


    if results_data or consolidated_summary_text_for_batch:
        is_info_or_error_summary = consolidated_summary_text_for_batch and \
                                   (str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:") or \
                                    str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:"))

        if not is_info_or_error_summary and consolidated_summary_text_for_batch:
            st.success("All processing complete! A consolidated overview has been generated. See above.")
        elif is_info_or_error_summary: 
            st.warning("Processing complete. Please check the consolidated overview section for details on the summary generation process.")
        elif not consolidated_summary_text_for_batch and results_data : # Results but no summary
             st.warning("Processing complete. Items were processed, but no consolidated overview was generated (e.g. LLM issue or no suitable content).")
        elif not consolidated_summary_text_for_batch and not results_data: # No results and no summary
             st.warning("Processing complete, but no data was generated and no consolidated overview.")
    else: 
        st.warning("Processing complete, but no data was generated (no results to process).")
        
    # --- DEBUG: Final log state before returning ---
    print(f"DEBUG (process_manager): FINAL: Returning log with {len(processing_log)} entries.") # DEBUG
    if processing_log:
        print(f"DEBUG (process_manager): FINAL: First log entry: {str(processing_log[0])[:200]}") # DEBUG
        print(f"DEBUG (process_manager): FINAL: Last log entry: {str(processing_log[-1])[:200]}") # DEBUG
        sheet_messages = [msg for msg in processing_log if "Sheet" in msg or "sheet" in msg or "💾" in msg or "❌" in msg or "✔️" in msg] # DEBUG
        if sheet_messages: # DEBUG
            print("DEBUG (process_manager): FINAL: Relevant sheet log messages found in processing_log:") # DEBUG
            for s_msg in sheet_messages[-10:]: # Last 10 sheet related messages # DEBUG
                print(f"  FINAL SHEET LOG: {str(s_msg)[:200]}") # DEBUG
        else: # DEBUG
            print("DEBUG (process_manager): FINAL: No specific sheet-related messages found in processing_log.") # DEBUG
    else: # DEBUG
        print("DEBUG (process_manager): FINAL: processing_log is empty.") # DEBUG
    # --- END DEBUG ---

    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display, focused_summary_source_details

# end of modules/process_manager.py
