# modules/process_manager.py
# Version 1.4.5:
# - Added more specific debug prints before the conditional GSheets write.
# - Ensured LOG_STATUS for GSheets write is more accurately reflecting if data was present.
# Version 1.4.4:
# - Modified final status messages (success/warning) to be appended to processing_log
#   instead of direct st.calls, for app.py to handle UI display.
# - Modified Google Search config error in loop to primarily log.
# Version 1.4.3: (Internal debug, merged)
# Version 1.4.2: (Internal debug, merged)

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
    llm_output_text: str # The raw text from LLM that was scored and used


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
        if llm_extract_queries_input: # This is active_llm_extract_queries from app.py
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
            num_user_terms = len(initial_keywords_list); num_llm_terms_to_generate = min(math.floor(num_user_terms * 1.5), 5)
            if num_llm_terms_to_generate > 0:
                llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_for_query_gen: str = app_config.llm.google_gemini_model
                if app_config.llm.provider == "openai": llm_model_for_query_gen = app_config.llm.openai_model_summarize
                
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
            else: processing_log.append("  ℹ️ No additional LLM queries requested (or needed based on input).")
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
        # app.py uses a general st.spinner, so these placeholders are not strictly needed here anymore for that.
        # However, they can be used if granular progress text updates directly from PM are desired in future.
        # progress_bar_placeholder = st.empty() 
        # status_placeholder = st.empty()      

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
        
        print("-----> DEBUG (process_manager): Starting Item Processing Loop.") 
        for keyword_val in keywords_list_val_runtime: 
            print(f"-----> DEBUG (process_manager): Loop for keyword: {keyword_val}") 
            processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
            if not (app_config.google_search.api_key and app_config.google_search.cse_id):
                error_msg = f"  LOG_STATUS:ERROR:NO_SEARCH_CONFIG:Halting search for '{keyword_val}'. Google Search API Key or CSE ID not configured."
                processing_log.append(error_msg)
                print(f"DEBUG (process_manager): {error_msg}")
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

                processing_log.append(f"LOG_PROGRESS:SCRAPING:{current_major_step_count}/{total_major_steps_for_progress}:Scraping: {url_to_scrape_val[:50]}...")
                
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
                            current_major_step_count += 1 # Increment for this LLM task
                            processing_log.append(f"LOG_PROGRESS:LLM_SUMMARY:{current_major_step_count}/{total_major_steps_for_progress}:LLM Summary for {url_to_scrape_val[:40]}...")
                            summary_text_val: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                            item_data_val["llm_summary"] = summary_text_val
                            processing_log.append(f"        Summary: {str(summary_text_val)[:100] if summary_text_val else 'Failed/Empty'}...")
                            made_llm_call_for_item = True
                            
                            # LLM Extractions
                            queries_to_process_for_item = []
                            if primary_llm_extract_query and primary_llm_extract_query.strip():
                                queries_to_process_for_item.append({"query_text": primary_llm_extract_query, "id": "q1", "display_idx": 1})
                            if secondary_llm_extract_query and secondary_llm_extract_query.strip():
                                queries_to_process_for_item.append({"query_text": secondary_llm_extract_query, "id": "q2", "display_idx": 2})

                            for query_info in queries_to_process_for_item:
                                extraction_query = query_info["query_text"]
                                query_id_label = query_info["id"] 
                                query_display_idx = query_info["display_idx"]

                                current_major_step_count += 1 # Increment for this LLM task
                                processing_log.append(f"LOG_PROGRESS:LLM_EXTRACT_Q{query_display_idx}:{current_major_step_count}/{total_major_steps_for_progress}:LLM Extract Q{query_display_idx} for {url_to_scrape_val[:40]}...")
                                
                                processing_log.append(f"      Extracting info for Q{query_display_idx}: '{extraction_query}'...")
                                extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                    main_text_for_llm, extraction_query=extraction_query, 
                                    api_key=llm_api_key_to_use, model_name=llm_model_to_use, 
                                    max_input_chars=app_config.llm.max_input_chars
                                )
                                
                                parsed_score = _parse_score_from_extraction(extracted_info_full)
                                content_without_score = extracted_info_full 
                                if hasattr(llm_processor, '_parse_score_and_get_content'): 
                                    _, content_temp = llm_processor._parse_score_and_get_content(extracted_info_full)
                                    if content_temp is not None: content_without_score = content_temp
                                elif parsed_score is not None and extracted_info_full and '\n' in extracted_info_full:
                                     try: content_without_score = extracted_info_full.split('\n', 1)[1]
                                     except IndexError: pass 

                                item_data_val[f"llm_extracted_info_{query_id_label}"] = content_without_score
                                item_data_val[f"llm_relevancy_score_{query_id_label}"] = parsed_score
                                item_data_val[f"llm_extracted_info_{query_id_label}_full"] = extracted_info_full 
                                processing_log.append(f"        Extracted (Q{query_display_idx}): Score={parsed_score}, Content='{str(content_without_score)[:70] if content_without_score else 'Failed/Empty'}'...")
                                made_llm_call_for_item = True
                        results_data.append(item_data_val)
                    else: 
                        processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
                
                if apply_throttling and made_llm_call_for_item:
                    delay_message = f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s..."
                    processing_log.append(f"    {delay_message}")
                    # status_placeholder.text(delay_message) # app.py handles general spinner
                    print(f"DEBUG (process_manager): {delay_message}") 
                    time.sleep(llm_item_delay_seconds)
                    # status_placeholder.text(f"Continuing processing...")
                elif not apply_throttling and made_llm_call_for_item:
                     time.sleep(0.2) 
                elif not made_llm_call_for_item:
                     time.sleep(0.1)
            
            if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
                processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
                remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
                current_major_step_count += remaining_llm_tasks_for_keyword 
        print("-----> DEBUG (process_manager): Finished Item Processing Loop.") 

        print("-----> DEBUG (process_manager): Starting Consolidated Summary block.") 
        topic_for_consolidation_for_batch: str
        if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics"
        elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
        else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

        if results_data and llm_key_available:
            processing_log.append("LOG_STATUS:GENERATING_SUMMARY:\n✨ Generating consolidated overview...")
            # status_placeholder.text("Generating consolidated overview...") # app.py handles general spinner
            # with st.spinner("Generating consolidated overview..."): # app.py handles general spinner
            temp_focused_texts_for_llm: List[str] = []
            processed_item_texts_for_focused = set()

            if primary_llm_extract_query or secondary_llm_extract_query:
                for item in results_data:
                    item_url = item.get("url", "Unknown URL")
                    if primary_llm_extract_query:
                        extraction_text_q1_content = item.get("llm_extracted_info_q1") 
                        score_q1 = item.get("llm_relevancy_score_q1") 
                        full_q1_output_for_source = item.get("llm_extracted_info_q1_full", extraction_text_q1_content) 
                        if extraction_text_q1_content and score_q1 is not None and score_q1 >= 3:
                            source_entry_q1: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q1", "query_text": primary_llm_extract_query, 
                                "score": score_q1, "llm_output_text": full_q1_output_for_source 
                            }
                            if not any(d['url'] == item_url and d['query_type'] == 'Q1' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q1)
                            if extraction_text_q1_content not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q1_content)
                                processed_item_texts_for_focused.add(extraction_text_q1_content)
                    if secondary_llm_extract_query:
                        extraction_text_q2_content = item.get("llm_extracted_info_q2") 
                        score_q2 = item.get("llm_relevancy_score_q2") 
                        full_q2_output_for_source = item.get("llm_extracted_info_q2_full", extraction_text_q2_content)
                        if extraction_text_q2_content and score_q2 is not None and score_q2 >= 3:
                            source_entry_q2: FocusedSummarySource = {
                                "url": item_url, "query_type": "Q2", "query_text": secondary_llm_extract_query, 
                                "score": score_q2, "llm_output_text": full_q2_output_for_source
                            }
                            if not any(d['url'] == item_url and d['query_type'] == 'Q2' for d in focused_summary_source_details):
                                 focused_summary_source_details.append(source_entry_q2)
                            if extraction_text_q2_content not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q2_content)
                                processed_item_texts_for_focused.add(extraction_text_q2_content)
            final_texts_for_llm = temp_focused_texts_for_llm
            if final_texts_for_llm:
                # ... (Focused summary logic from v1.4.2) ...
                llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                if not primary_llm_extract_query and secondary_llm_extract_query: llm_context_for_focused_summary = secondary_llm_extract_query
                processing_log.append(f"\n📋 Preparing inputs for FOCUSED consolidated summary...")
                q2_contributed = any(details['query_type'] == 'Q2' and details['query_text'] == secondary_llm_extract_query for details in focused_summary_source_details)
                # ... (logging for focused summary sources from v1.4.2) ...
                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                generated_focused_summary = llm_processor.generate_consolidated_summary(
                    summaries=tuple(final_texts_for_llm), topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol, max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=llm_context_for_focused_summary,
                    secondary_query_for_enrichment=secondary_llm_extract_query if secondary_llm_extract_query and secondary_llm_extract_query.strip() and q2_contributed else None
                )
                is_llm_call_problematic = not generated_focused_summary or any(sub in str(generated_focused_summary).lower() for sub in ["llm_processor", "could not generate", "no items met score", "no suitable content", "error:"])
                if is_llm_call_problematic: # ... (set error summary from v1.4.2) ...
                     consolidated_summary_text_for_batch = f"LLM_PROCESSOR_ERROR: Failed focused summary. LLM: {str(generated_focused_summary)[:100]}"
                     processing_log.append(f"  ❌ LLM failed FOCUSED summary. Output: {str(generated_focused_summary)[:100]}")
                else: consolidated_summary_text_for_batch = generated_focused_summary; processing_log.append(f"  ✔️ Successfully generated FOCUSED consolidated summary.")
            else: # General summary fallback
                # ... (Full general summary logic from v1.4.2) ...
                processing_log.append("  Attempting GENERAL consolidated overview...")
                general_texts_for_consolidation: List[str] = [item.get("llm_summary","") for item in results_data if item.get("llm_summary") and not str(item.get("llm_summary")).lower().startswith(("llm error", "no text content", "llm_processor:"))]
                if general_texts_for_consolidation:
                     consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(summaries=tuple(general_texts_for_consolidation), topic_context=topic_for_consolidation_for_batch, api_key=app_config.llm.google_gemini_api_key, model_name=app_config.llm.google_gemini_model) # Simplified call
                     processing_log.append(f"  ✔️ General summary attempted. Result: {str(consolidated_summary_text_for_batch)[:100]}")
                else:
                     processing_log.append("  ❌ No valid item summaries for general overview."); consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No valid item summaries for general overview."
            # status_placeholder.text("Report generation complete.") # Handled by app.py spinner
        elif not results_data and llm_key_available:
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items were successfully scraped and processed to create a consolidated summary."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        elif not llm_key_available:
            consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
            processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
        print("-----> DEBUG (process_manager): Finished Consolidated Summary block.")

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
            elif not consolidated_summary_text_for_batch and not results_data: 
                processing_log.append("LOG_STATUS:WARNING:Processing complete, but no data was generated and no consolidated overview.")
        else: 
            processing_log.append("LOG_STATUS:WARNING:Processing complete, but no data was generated (no results to process).")
        print("-----> DEBUG (process_manager): Finished preparing final status log messages.")
            
        print("-----> DEBUG (process_manager): Starting Google Sheets Writing block.")
        print(f"DEBUG (process_manager) PRE-SHEET BLOCK: sheet_writing_enabled={sheet_writing_enabled}")
        print(f"DEBUG (process_manager) PRE-SHEET BLOCK: gs_worksheet is {'present and type: ' + str(type(gs_worksheet)) if gs_worksheet else 'None'}")
        print(f"DEBUG (process_manager) PRE-SHEET BLOCK: len(results_data) = {len(results_data) if results_data is not None else 'None'}")
        print(f"DEBUG (process_manager) PRE-SHEET BLOCK: consolidated_summary_text_for_batch is {'NOT None/empty' if consolidated_summary_text_for_batch else 'None/empty'}")

        if sheet_writing_enabled and gs_worksheet:
            processing_log.append(f"\n💾 Checking conditions to write batch data to Google Sheets...")
            print(f"DEBUG (process_manager) INSIDE IF_ENABLED_AND_WORKSHEET: results_data length={len(results_data) if results_data is not None else 'None'}") 
            print(f"DEBUG (process_manager) INSIDE IF_ENABLED_AND_WORKSHEET: consolidated_summary_text_for_batch is {'present' if consolidated_summary_text_for_batch else 'None'}")

            if results_data or consolidated_summary_text_for_batch:
                batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
                processing_log.append(f"LOG_STATUS:SHEET_WRITE_ATTEMPTED:Attempting to write batch data to Google Sheets at {batch_process_timestamp_for_sheet}...") 
                print(f"DEBUG (process_manager): Condition (results_data or summary) is TRUE. Calling data_storage.write_batch_summary_and_items_to_sheet")
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
                    error_msg = f"  ❌ CRITICAL ERROR calling data_storage.write_batch_summary_and_items_to_sheet: {e_write_sheet}"
                    processing_log.append(f"LOG_STATUS:SHEET_WRITE_ERROR:{error_msg}")
                    print(f"DEBUG (process_manager): {error_msg}")
                    print(traceback.format_exc())
                    
                if write_successful:
                    processing_log.append("LOG_STATUS:SHEET_WRITE_SUCCESS:Batch data written to Google Sheets successfully.")
                    print(f"DEBUG (process_manager): data_storage.write_batch_summary_and_items_to_sheet reported SUCCESS (True).")
                else:
                    processing_log.append("LOG_STATUS:SHEET_WRITE_FAILED:Failed to write batch data to Google Sheets (data_storage returned False or an error occurred during call).")
                    print(f"DEBUG (process_manager): data_storage.write_batch_summary_and_items_to_sheet reported FAILED (False) or error during call.")
            else: 
                processing_log.append(f"LOG_STATUS:SHEET_WRITE_NO_DATA:No data (results or summary) to write to Google Sheets for this batch.")
                print(f"DEBUG (process_manager): Condition (results_data or summary) is FALSE. No data to write.")
        elif gsheets_secrets_present and not sheet_writing_enabled :
            processing_log.append("LOG_STATUS:SHEET_WRITE_SKIPPED:Google Sheets connection failed earlier or sheet object is invalid. Data not saved to sheet.")
            print(f"DEBUG (process_manager): GSheets secrets present, but writing not enabled (gs_worksheet type: {type(gs_worksheet)}).")
        elif not gsheets_secrets_present:
            processing_log.append("LOG_STATUS:SHEET_WRITE_SKIPPED:Google Sheets integration not configured. Data not saved to sheet.")
            print(f"DEBUG (process_manager): GSheets secrets not present.")
        else: 
            processing_log.append("LOG_STATUS:SHEET_WRITE_SKIPPED:Google Sheets writing skipped (general conditions not met - e.g., sheet_writing_enabled or gs_worksheet missing).")
            print(f"DEBUG (process_manager): Google Sheets writing skipped (general conditions not met). sheet_writing_enabled={sheet_writing_enabled}, gs_worksheet_present={bool(gs_worksheet)}")
        print("-----> DEBUG (process_manager): Finished Google Sheets Writing block.")
            
    except Exception as e_main_pm: 
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
