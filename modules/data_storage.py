# modules/process_manager.py
# Version 1.4.2:
# - Added extensive print() debugging at the very start of run_search_and_analysis
#   and a try...finally block to ensure final log state is printed.
# - Reinstated full logic from v1.3.7 combined with throttling from v1.4.0/1.4.1.
# Version 1.4.1: (Internal debug version, merged into 1.4.2)
# Version 1.4.0:
# - Implemented conditional LLM request throttling.
# Version 1.3.7: (Base functionality)
# - Passed Q2 to llm_processor.generate_consolidated_summary for enrichment.

"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
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
    llm_extract_queries_input: List[str], # This is active_llm_extract_queries from app.py
    num_results_wanted_per_keyword: int,
    gs_worksheet: Optional[Any], # gspread.Worksheet or None
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    print("-----> DEBUG (process_manager): TOP OF run_search_and_analysis called.") # NEW DEBUG

    # Initialize all variables that will be returned
    processing_log: List[str] = ["Processing initiated (from process_manager)..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    # Use distinct names for the sets to be returned to avoid conflict if keywords_input becomes a set
    initial_keywords_for_display_set: Set[str] = set()
    llm_generated_keywords_set_for_display_set: Set[str] = set()
    
    print("-----> DEBUG (process_manager): Initial return variables set.") # NEW DEBUG

    try:
        initial_keywords_list: List[str] = [k.strip() for k in keywords_input.split(',') if k.strip()]
        print(f"-----> DEBUG (process_manager): initial_keywords_list: {initial_keywords_list}") # NEW DEBUG

        if not initial_keywords_list:
            # This st.sidebar.error will appear in the UI.
            st.sidebar.error("Please enter at least one keyword.")
            processing_log.append("ERROR: No keywords provided by user.")
            print("-----> DEBUG (process_manager): No keywords provided, preparing to return early.") # NEW DEBUG
            return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

        initial_keywords_for_display_set = set(k.lower() for k in initial_keywords_list)
        print(f"-----> DEBUG (process_manager): initial_keywords_for_display_set: {initial_keywords_for_display_set}") # NEW DEBUG

        keywords_list_val_runtime: List[str] = list(initial_keywords_list)
        print(f"-----> DEBUG (process_manager): keywords_list_val_runtime: {keywords_list_val_runtime}") # NEW DEBUG

        llm_key_available: bool = (app_config.llm.provider == "google" and app_config.llm.google_gemini_api_key) or \
                                  (app_config.llm.provider == "openai" and app_config.llm.openai_api_key)
        print(f"-----> DEBUG (process_manager): llm_key_available: {llm_key_available}") # NEW DEBUG

        primary_llm_extract_query: Optional[str] = None
        secondary_llm_extract_query: Optional[str] = None
        if llm_extract_queries_input:
            if len(llm_extract_queries_input) > 0 and llm_extract_queries_input[0] and llm_extract_queries_input[0].strip():
                primary_llm_extract_query = llm_extract_queries_input[0].strip()
            if len(llm_extract_queries_input) > 1 and llm_extract_queries_input[1] and llm_extract_queries_input[1].strip():
                secondary_llm_extract_query = llm_extract_queries_input[1].strip()
        
        print(f"-----> DEBUG (process_manager): Primary Q: '{primary_llm_extract_query}', Secondary Q: '{secondary_llm_extract_query}'") # NEW DEBUG

        # --- Throttling Configuration ---
        llm_item_delay_seconds = app_config.llm.llm_item_request_delay_seconds
        throttling_threshold = app_config.llm.llm_throttling_threshold_results
        apply_throttling = (
            num_results_wanted_per_keyword >= throttling_threshold and
            llm_item_delay_seconds > 0 and
            llm_key_available
        )
        print(f"-----> DEBUG (process_manager): Throttling check: num_results={num_results_wanted_per_keyword}, threshold={throttling_threshold}, delay={llm_item_delay_seconds}, llm_key_available={llm_key_available}. Apply_throttling={apply_throttling}") # NEW DEBUG
        # --- End Throttling Configuration ---

        # LLM Query Generation (from v1.3.7)
        if llm_key_available and initial_keywords_list:
            print("-----> DEBUG (process_manager): Starting LLM query generation block.") # NEW DEBUG
            processing_log.append("\n🧠 Generating additional search queries with LLM...")
            num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
            if num_llm_terms_to_generate > 0:
                llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_for_query_gen: str = app_config.llm.google_gemini_model
                if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
                with st.spinner(f"LLM generating {num_llm_terms_to_generate} additional search queries..."): # This spinner is UI, won't show in pure prints
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
                    llm_generated_keywords_set_for_display_set = temp_llm_generated_set; processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
                else: processing_log.append("  ⚠️ LLM did not generate new queries.")
            else: processing_log.append("  ℹ️ No additional LLM queries requested (or needed based on input).")
            print("-----> DEBUG (process_manager): Finished LLM query generation block.") # NEW DEBUG
        
        # Progress Bar Setup (from v1.3.7)
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
        progress_bar_placeholder = st.empty()
        status_placeholder = st.empty()

        if apply_throttling:
            throttle_init_message = (
                f"ℹ️ LLM Throttling ACTIVE: Delay of {llm_item_delay_seconds:.1f}s "
                f"after each item's LLM processing (threshold: {throttling_threshold} results/keyword)."
            )
            processing_log.append(throttle_init_message)
            print(f"DEBUG (process_manager): {throttle_init_message}") 

        if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
            current_major_step_count +=1
            progress_text = "LLM Query Generation Complete..."
            status_placeholder.text(progress_text)
            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            with progress_bar_placeholder.container(): 
                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text)

        # --- Item Processing Loop (from v1.3.7 with throttling integrated) ---
        print("-----> DEBUG (process_manager): Starting Item Processing Loop.") # NEW DEBUG
        for keyword_val in keywords_list_val_runtime: 
            print(f"-----> DEBUG (process_manager): Loop for keyword: {keyword_val}") # NEW DEBUG
            processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
            if not (app_config.google_search.api_key and app_config.google_search.cse_id):
                st.error("Google Search API Key or CSE ID not configured.")
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
                status_placeholder.text(progress_text_scrape)
                progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                with progress_bar_placeholder.container(): 
                    st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_scrape)

                processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
                scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val) 
                
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
                    "llm_extracted_info_q1": None, "llm_relevancy_score_q1": None, 
                    "llm_extracted_info_q2": None, "llm_relevancy_score_q2": None,
                    "llm_extraction_query_1_text": primary_llm_extract_query if primary_llm_extract_query else "", # For excel_handler
                    "llm_extraction_query_2_text": secondary_llm_extract_query if secondary_llm_extract_query else "", # For excel_handler
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                made_llm_call_for_item = False

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
                            summary_text_val: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                            item_data_val["llm_summary"] = summary_text_val
                            processing_log.append(f"        Summary: {str(summary_text_val)[:100] if summary_text_val else 'Failed/Empty'}...")
                            made_llm_call_for_item = True
                            
                            queries_to_process_for_item = []
                            if primary_llm_extract_query and primary_llm_extract_query.strip():
                                queries_to_process_for_item.append({"query_text": primary_llm_extract_query, "id": "q1", "display_idx": 1})
                            if secondary_llm_extract_query and secondary_llm_extract_query.strip():
                                queries_to_process_for_item.append({"query_text": secondary_llm_extract_query, "id": "q2", "display_idx": 2})

                            for query_info in queries_to_process_for_item:
                                extraction_query = query_info["query_text"]
                                query_id_label = query_info["id"] 
                                query_display_idx = query_info["display_idx"]

                                current_major_step_count += 1
                                progress_text_llm_extract = f"LLM Extract Q{query_display_idx} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                                status_placeholder.text(progress_text_llm_extract)
                                progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                                with progress_bar_placeholder.container(): 
                                    st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_extract)
                                
                                processing_log.append(f"      Extracting info for Q{query_display_idx}: '{extraction_query}'...")
                                extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                    main_text_for_llm, extraction_query=extraction_query, 
                                    api_key=llm_api_key_to_use, model_name=llm_model_to_use, 
                                    max_input_chars=app_config.llm.max_input_chars
                                )
                                
                                parsed_score = _parse_score_from_extraction(extracted_info_full)
                                content_without_score = extracted_info_full 
                                if hasattr(llm_processor, '_parse_score_and_get_content'): # Check if llm_processor has this helper
                                    _, content_without_score_temp = llm_processor._parse_score_and_get_content(extracted_info_full)
                                    if content_without_score_temp is not None: 
                                        content_without_score = content_without_score_temp
                                elif parsed_score is not None and extracted_info_full and '\n' in extracted_info_full:
                                     try: content_without_score = extracted_info_full.split('\n', 1)[1]
                                     except IndexError: pass # Keep full string if split fails

                                item_data_val[f"llm_extracted_info_{query_id_label}"] = content_without_score
                                item_data_val[f"llm_relevancy_score_{query_id_label}"] = parsed_score
                                processing_log.append(f"        Extracted (Q{query_display_idx}): Score={parsed_score}, Content='{str(content_without_score)[:70] if content_without_score else 'Failed/Empty'}'...")
                                made_llm_call_for_item = True
                        results_data.append(item_data_val)
                    else: 
                        processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
                
                if apply_throttling and made_llm_call_for_item:
                    delay_message = f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s..."
                    processing_log.append(f"    {delay_message}")
                    status_placeholder.text(delay_message)
                    print(f"DEBUG (process_manager): {delay_message}") 
                    time.sleep(llm_item_delay_seconds)
                    status_placeholder.text(f"Continuing processing...")
                elif not apply_throttling and made_llm_call_for_item:
                     time.sleep(0.2) 
                elif not made_llm_call_for_item:
                     time.sleep(0.1)
            
            if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
                processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
                remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
                current_major_step_count += remaining_llm_tasks_for_keyword 
        print(f"-----> DEBUG (process_manager): Finished Item Processing Loop.") # NEW DEBUG

        # Consolidated Summary Generation (from v1.3.7)
        print("-----> DEBUG (process_manager): Starting Consolidated Summary block.") # NEW DEBUG
        topic_for_consolidation_for_batch: str
        if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics"
        elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
        else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

        if results_data and llm_key_available:
            processing_log.append(f"\n✨ Generating consolidated overview...")
            status_placeholder.text("Generating consolidated overview...")
            with st.spinner("Generating consolidated overview..."):
                temp_focused_texts_for_llm: List[str] = []
                processed_item_texts_for_focused = set()

                if primary_llm_extract_query or secondary_llm_extract_query:
                    for item in results_data:
                        item_url = item.get("url", "Unknown URL")
                        # Q1 processing for focused summary
                        if primary_llm_extract_query:
                            extraction_text_q1_content = item.get("llm_extracted_info_q1") # Content only
                            score_q1 = item.get("llm_relevancy_score_q1") # Parsed score
                            # Store original full text if different from content_only for llm_output_text
                            # Assuming process_manager now stores this if llm_processor.py returns it.
                            # For now, use content if full isn't explicitly stored.
                            full_q1_output = item.get("llm_extracted_info_q1_full", extraction_text_q1_content) 

                            if extraction_text_q1_content and score_q1 is not None and score_q1 >= 3:
                                source_entry_q1: FocusedSummarySource = {
                                    "url": item_url, "query_type": "Q1",
                                    "query_text": primary_llm_extract_query, "score": score_q1,
                                    "llm_output_text": full_q1_output
                                }
                                if not any(d['url'] == item_url and d['query_type'] == 'Q1' for d in focused_summary_source_details):
                                    focused_summary_source_details.append(source_entry_q1)
                                if extraction_text_q1_content not in processed_item_texts_for_focused: # Use content for LLM input
                                    temp_focused_texts_for_llm.append(extraction_text_q1_content)
                                    processed_item_texts_for_focused.add(extraction_text_q1_content)
                        # Q2 processing for focused summary
                        if secondary_llm_extract_query:
                            extraction_text_q2_content = item.get("llm_extracted_info_q2") # Content only
                            score_q2 = item.get("llm_relevancy_score_q2") # Parsed score
                            full_q2_output = item.get("llm_extracted_info_q2_full", extraction_text_q2_content)

                            if extraction_text_q2_content and score_q2 is not None and score_q2 >= 3:
                                source_entry_q2: FocusedSummarySource = {
                                    "url": item_url, "query_type": "Q2",
                                    "query_text": secondary_llm_extract_query, "score": score_q2,
                                    "llm_output_text": full_q2_output
                                }
                                if not any(d['url'] == item_url and d['query_type'] == 'Q2' for d in focused_summary_source_details):
                                    focused_summary_source_details.append(source_entry_q2)
                                if extraction_text_q2_content not in processed_item_texts_for_focused:
                                    temp_focused_texts_for_llm.append(extraction_text_q2_content)
                                    processed_item_texts_for_focused.add(extraction_text_q2_content)
                
                final_texts_for_llm = temp_focused_texts_for_llm

                if final_texts_for_llm:
                    llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                    if not primary_llm_extract_query and secondary_llm_extract_query:
                        llm_context_for_focused_summary = secondary_llm_extract_query

                    processing_log.append(f"\n📋 Preparing inputs for FOCUSED consolidated summary:")
                    processing_log.append(f"  Central Query Theme (typically based on Q1): '{llm_context_for_focused_summary}'")
                    q2_contributed = any(details['query_type'] == 'Q2' and details['query_text'] == secondary_llm_extract_query for details in focused_summary_source_details)
                    if secondary_llm_extract_query and q2_contributed:
                        processing_log.append(f"  Secondary Query for Enrichment (Q2): '{secondary_llm_extract_query}' will be used as relevant snippets are included.")
                    
                    if focused_summary_source_details:
                        processing_log.append(f"  Found {len(focused_summary_source_details)} source snippet(s) (from Q1/Q2 extractions scoring >=3/5) that contributed text for the LLM:")
                        sorted_source_details = sorted(focused_summary_source_details, key=lambda x: (x['url'], x['query_type']))
                        for source_item in sorted_source_details:
                            log_url, log_q_type, log_q_text, log_score = source_item['url'], source_item['query_type'], source_item['query_text'], source_item['score']
                            log_q_text_short = log_q_text[:30] + "..." if len(log_q_text) > 30 else log_q_text
                            processing_log.append(f"    - From: {log_q_type} ('{log_q_text_short}') on URL: {log_url} (Score: {log_score}/5)")
                    else:
                        processing_log.append(f"  Found {len(final_texts_for_llm)} unique text snippet(s) for LLM input (details not itemized - check logic if this appears with focused sources).")

                    llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                    llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize

                    generated_focused_summary = llm_processor.generate_consolidated_summary(
                        summaries=tuple(final_texts_for_llm), topic_context=topic_for_consolidation_for_batch,
                        api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                        max_input_chars=app_config.llm.max_input_chars,
                        extraction_query_for_consolidation=llm_context_for_focused_summary,
                        secondary_query_for_enrichment=secondary_llm_extract_query if secondary_llm_extract_query and secondary_llm_extract_query.strip() and q2_contributed else None
                    )

                    is_llm_call_problematic = False # Logic from v1.3.7
                    if not generated_focused_summary: is_llm_call_problematic = True
                    else:
                        problematic_substrings = ["llm_processor", "could not generate", "no items met score", "no suitable content", "error:"]
                        for sub in problematic_substrings:
                            if sub in str(generated_focused_summary).lower(): is_llm_call_problematic = True; break
                    if is_llm_call_problematic:
                        processing_log.append(f"  ❌ LLM failed to generate FOCUSED summary. Output: {str(generated_focused_summary)[:100]}")
                        # ... (Error message construction from v1.3.7) ...
                        consolidated_summary_text_for_batch = f"LLM_PROCESSOR_ERROR: Failed focused summary. LLM: {str(generated_focused_summary)[:100]}"
                    else:
                        consolidated_summary_text_for_batch = generated_focused_summary
                        processing_log.append(f"  ✔️ Successfully generated FOCUSED consolidated summary.")
                else: # General summary fallback logic from v1.3.7
                    # ... (Full general summary logic from your v1.3.7 needs to be here) ...
                    processing_log.append("  Attempting GENERAL consolidated overview...")
                    general_texts_for_consolidation: List[str] = [item.get("llm_summary","") for item in results_data if item.get("llm_summary") and not str(item.get("llm_summary")).lower().startswith(("llm error", "no text content", "llm_processor:"))]
                    if general_texts_for_consolidation:
                         consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(summaries=tuple(general_texts_for_consolidation), topic_context=topic_for_consolidation_for_batch, api_key=app_config.llm.google_gemini_api_key, model_name=app_config.llm.google_gemini_model)
                         processing_log.append(f"  ✔️ General summary attempted. Result: {str(consolidated_summary_text_for_batch)[:100]}")
                    else:
                         processing_log.append("  ❌ No valid item summaries for general overview.")
                         consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No valid item summaries for general overview."

            status_placeholder.text("Report generation complete.")
        elif not results_data and llm_key_available:
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items were successfully scraped and processed to create a consolidated summary."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        elif not llm_key_available:
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        print("-----> DEBUG (process_manager): Finished Consolidated Summary block.") # NEW DEBUG

        # Google Sheets Writing (from v1.3.7 with debugs)
        print("-----> DEBUG (process_manager): Starting Google Sheets Writing block.") # NEW DEBUG
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: sheet_writing_enabled={sheet_writing_enabled}")
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: gs_worksheet is {'present and type: ' + str(type(gs_worksheet)) if gs_worksheet else 'None'}")
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: results_data length={len(results_data)}")
        print(f"DEBUG (process_manager) PRE-SHEET WRITE: consolidated_summary_text_for_batch is {'present' if consolidated_summary_text_for_batch else 'None'}")
        
        if sheet_writing_enabled and gs_worksheet:
            processing_log.append(f"\n💾 Checking conditions to write batch data to Google Sheets...")
            if results_data or consolidated_summary_text_for_batch:
                batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
                processing_log.append(f"  Attempting to write batch data to Google Sheets at {batch_process_timestamp_for_sheet}...")
                print(f"DEBUG (process_manager): Calling data_storage.write_batch_summary_and_items_to_sheet")
                active_extraction_queries_for_sheet = [q for q in llm_extract_queries_input if q and q.strip()]
                
                write_successful: bool = False
                try:
                    write_successful = data_storage.write_batch_summary_and_items_to_sheet(
                        worksheet=gs_worksheet, batch_timestamp=batch_process_timestamp_for_sheet,
                        consolidated_summary=consolidated_summary_text_for_batch,
                        topic_context=topic_for_consolidation_for_batch,
                        item_data_list=results_data,
                        extraction_queries_list=active_extraction_queries_for_sheet
                    )
                except Exception as e_write_sheet:
                    error_msg = f"  ❌ CRITICAL ERROR calling write_batch_summary_and_items_to_sheet: {e_write_sheet}"
                    processing_log.append(error_msg)
                    print(f"DEBUG (process_manager): {error_msg}")
                    print(traceback.format_exc())
                    
                if write_successful:
                    processing_log.append(f"  ✔️ Batch data written to Google Sheets successfully.")
                    print(f"DEBUG (process_manager): Batch data write reported successful.")
                else:
                    processing_log.append(f"  ❌ Failed to write batch data to Google Sheets (write_successful is False or due to caught exception).")
                    print(f"DEBUG (process_manager): Batch data write reported as FAILED (write_successful=False).")
            else:
                processing_log.append(f"  ℹ️ No data (results or summary) to write to Google Sheets for this batch.")
                print(f"DEBUG (process_manager): No data (results or summary) to write to Google Sheets.")
        elif gsheets_secrets_present and not sheet_writing_enabled :
            processing_log.append("\n⚠️ Google Sheets connection failed earlier or sheet object is invalid. Data not saved to sheet.")
            print(f"DEBUG (process_manager): GSheets secrets present, but writing not enabled (gs_worksheet type: {type(gs_worksheet)}).")
        elif not gsheets_secrets_present:
            processing_log.append("\nℹ️ Google Sheets integration not configured. Data not saved to sheet.")
            print(f"DEBUG (process_manager): GSheets secrets not present.")
        else:
            processing_log.append("\nℹ️ Google Sheets writing skipped (general conditions not met - e.g., sheet_writing_enabled or gs_worksheet missing).")
            print(f"DEBUG (process_manager): Google Sheets writing skipped (general conditions not met). sheet_writing_enabled={sheet_writing_enabled}, gs_worksheet_present={bool(gs_worksheet)}")
        print("-----> DEBUG (process_manager): Finished Google Sheets Writing block.") # NEW DEBUG

        # Final UI messages (from v1.3.7)
        if results_data or consolidated_summary_text_for_batch:
            is_info_or_error_summary = consolidated_summary_text_for_batch and \
                                    (str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:") or \
                                        str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:"))
            if not is_info_or_error_summary and consolidated_summary_text_for_batch:
                st.success("All processing complete! A consolidated overview has been generated. See above.")
            elif is_info_or_error_summary: 
                st.warning("Processing complete. Please check the consolidated overview section for details on the summary generation process.")
            elif not consolidated_summary_text_for_batch and results_data :
                st.warning("Processing complete. Items were processed, but no consolidated overview was generated (e.g. LLM issue or no suitable content).")
            elif not consolidated_summary_text_for_batch and not results_data:
                st.warning("Processing complete, but no data was generated and no consolidated overview.")
        else: 
            st.warning("Processing complete, but no data was generated (no results to process).")
            
    finally: # NEW FINALLY BLOCK
        print(f"-----> DEBUG (process_manager): FINALLY BLOCK: Returning log with {len(processing_log)} entries.")
        if processing_log:
            print(f"-----> DEBUG (process_manager): FINALLY BLOCK: First log entry: {str(processing_log[0])[:200]}")
            print(f"-----> DEBUG (process_manager): FINALLY BLOCK: Last log entry: {str(processing_log[-1])[:200]}")
            sheet_messages = [msg for msg in processing_log if "Sheet" in msg or "sheet" in msg or "💾" in msg or "❌" in msg or "✔️" in msg]
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
