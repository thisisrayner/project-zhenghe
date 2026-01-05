# modules/process_manager.py
# Version 1.4.12: Increased max_google_fetch_per_keyword to 30 to support pagination.
# Version 1.4.11: Added sorting logic to deprioritise wikipedia.org results.
# Version 1.4.10: Updated to use specific Google Gemini model for consolidation if configured.
# - CRITICAL FIX: Ensured that the score parsed by llm_processor._parse_score_and_get_content
#   is correctly assigned and used, rather than being overridden by the local parser's result.
# Previous versions:
# - Version 1.4.8: Fixed NameError: 'is_good_scrape' was not defined.
# - Version 1.4.7: Reinstated progress bar and intermediate status text updates.

"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
Provides live intermediate progress updates via Streamlit UI elements and communicates
final processing status back via specific log messages for app.py to display.
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
    """
    A local, simpler parser for relevancy score as a fallback.
    Expects "Relevancy Score: X/5" at the beginning of the string.
    """
    score: Optional[int] = None
    if extracted_info and isinstance(extracted_info, str) and extracted_info.strip().startswith("Relevancy Score: "):
        try:
            score_line = extracted_info.strip().split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError): 
            pass # Silently fail, score remains None
    return score

def run_search_and_analysis(
    app_config: config.AppConfig, # Explicitly use AppConfig from modules.config
    keywords_input: str,
    llm_extract_queries_input: List[str],
    num_results_wanted_per_keyword: int,
    gs_worksheet: Optional[Any],
    sheet_writing_enabled: bool,
    gsheets_secrets_present: bool
) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]:
    print("-----> DEBUG (process_manager v1.4.9): TOP OF run_search_and_analysis called.")

    processing_log: List[str] = ["LOG_STATUS:PROCESSING_INITIATED:Processing initiated..."]
    results_data: List[Dict[str, Any]] = []
    consolidated_summary_text_for_batch: Optional[str] = None
    focused_summary_source_details: List[FocusedSummarySource] = []
    initial_keywords_for_display_set: Set[str] = set()
    llm_generated_keywords_set_for_display_set: Set[str] = set()
    
    progress_bar_placeholder = st.empty()
    status_placeholder = st.empty()
    
    print("-----> DEBUG (process_manager v1.4.9): UI placeholders created.")

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

        # --- Progress Calculation Setup (Copied from v1.4.8, review for accuracy if needed) ---
        num_initial_keywords_for_calc = len(initial_keywords_list)
        llm_query_gen_step_count = 0
        if llm_key_available and num_initial_keywords_for_calc > 0:
            num_llm_terms_to_generate_calc = min(math.floor(num_initial_keywords_for_calc * 1.5), 5)
            if num_llm_terms_to_generate_calc > 0: llm_query_gen_step_count = 1
        
        temp_keywords_list_for_calc = list(initial_keywords_list)
        if llm_query_gen_step_count > 0:
            num_llm_terms_to_add_for_calc = min(math.floor(num_initial_keywords_for_calc * 1.5), 5)
            for i in range(num_llm_terms_to_add_for_calc): temp_keywords_list_for_calc.append(f"llm_gen_placeholder_{i}")
        total_keywords_to_process_calc = len(set(k.lower() for k in temp_keywords_list_for_calc))
        oversample_factor: float = 2.0; max_google_fetch_per_keyword: int = 30
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
        total_major_steps_for_progress: int = (llm_query_gen_step_count + search_steps_calc + scraping_steps_calc + llm_item_processing_steps_calc + consolidated_summary_step_calc)
        if total_major_steps_for_progress == 0: total_major_steps_for_progress = 1
        current_major_step_count: int = 0
        print(f"-----> DEBUG (process_manager v1.4.9): Total estimated steps for progress: {total_major_steps_for_progress}")
        # --- End Progress Calculation Setup ---

        def update_progress_ui(message: Optional[str] = None):
            progress_val = min(1.0, current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            progress_bar_placeholder.progress(math.ceil(progress_val * 100) / 100)
            if message: status_placeholder.info(message)

        if apply_throttling:
            throttle_init_msg = (f"ℹ️ LLM Throttling is ACTIVE (delay: {llm_item_delay_seconds:.1f}s if results/keyword ≥ {throttling_threshold}).")
            processing_log.append(throttle_init_msg); status_placeholder.info(throttle_init_msg)

        if llm_query_gen_step_count > 0:
            update_progress_ui(message="🧠 Generating additional search queries with LLM...")
            # ... (LLM Query Generation logic as in v1.4.8) ...
            num_user_terms_for_llm_qgen = len(initial_keywords_list)
            num_llm_terms_to_generate_for_llm_qgen = min(math.floor(num_user_terms_for_llm_qgen * 1.5), 5)
            llm_api_key_to_use_qgen: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            llm_model_for_qgen: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
            generated_queries: Optional[List[str]] = llm_processor.generate_search_queries(
                original_keywords=tuple(initial_keywords_list), specific_info_query=primary_llm_extract_query,
                specific_info_query_2=secondary_llm_extract_query, num_queries_to_generate=num_llm_terms_to_generate_for_llm_qgen,
                api_key=llm_api_key_to_use_qgen, model_name=llm_model_for_qgen)
            if generated_queries:
                processing_log.append(f"  ✨ LLM generated {len(generated_queries)} new queries: {', '.join(generated_queries)}")
                current_runtime_keywords_lower = {k.lower() for k in keywords_list_val_runtime}; temp_llm_generated_set = set()
                for gq_val in generated_queries:
                    if gq_val.lower() not in current_runtime_keywords_lower:
                        keywords_list_val_runtime.append(gq_val); current_runtime_keywords_lower.add(gq_val.lower()); temp_llm_generated_set.add(gq_val.lower())
                llm_generated_keywords_set_for_display_set = temp_llm_generated_set
                processing_log.append(f"  🔍 Total unique keywords to search: {len(keywords_list_val_runtime)}")
            else: processing_log.append("  ⚠️ LLM did not generate new queries.")
            current_major_step_count += 1
            update_progress_ui(message="LLM Query Generation complete.")

        total_keywords_actually_processing = len(keywords_list_val_runtime)
        for keyword_idx, keyword_val in enumerate(keywords_list_val_runtime):
            processing_log.append(f"\n🔎 Processing keyword {keyword_idx+1}/{total_keywords_actually_processing}: {keyword_val}")
            update_progress_ui(message=f"Searching for '{keyword_val}' ({keyword_idx+1}/{total_keywords_actually_processing})...")
            
            if not (app_config.google_search.api_key and app_config.google_search.cse_id):
                processing_log.append(f"LOG_STATUS:ERROR:NO_SEARCH_CONFIG: Search for '{keyword_val}' skipped.") # Simplified
                current_major_step_count += 1 + urls_to_scan_per_keyword + (num_results_wanted_per_keyword * llm_tasks_per_good_item_calc)
                update_progress_ui(); continue
            
            search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
                query=keyword_val, api_key=app_config.google_search.api_key, 
                cse_id=app_config.google_search.cse_id, num_results=urls_to_scan_per_keyword)
            
            # Deprioritise Wikipedia: Move wikipedia.org links to the end
            search_results_items_val.sort(key=lambda x: 1 if 'wikipedia.org' in x.get('link', '').lower() else 0)

            current_major_step_count += 1
            processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
            update_progress_ui(message=f"Found {len(search_results_items_val)} results for '{keyword_val}'.")

            if not search_results_items_val:
                current_major_step_count += urls_to_scan_per_keyword + (num_results_wanted_per_keyword * llm_tasks_per_good_item_calc)
                update_progress_ui(); continue

            successfully_scraped_for_this_keyword: int = 0
            for search_item_idx, search_item_val in enumerate(search_results_items_val):
                if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                    current_major_step_count += (urls_to_scan_per_keyword - search_item_idx)
                    update_progress_ui(); break
                
                url_to_scrape_val: Optional[str] = search_item_val.get('link')
                if not url_to_scrape_val:
                    current_major_step_count += 1; update_progress_ui(); continue

                update_progress_ui(message=f"Scraping {url_to_scrape_val[:50]}... ({search_item_idx+1}/{urls_to_scan_per_keyword})")
                scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
                current_major_step_count += 1; update_progress_ui()

                item_data_val: Dict[str, Any] = {
                    "keyword_searched": keyword_val, "url": url_to_scrape_val, 
                    "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                    "page_title": scraped_content_val.get('scraped_title'), "meta_description": scraped_content_val.get('meta_description'), 
                    "og_title": scraped_content_val.get('og_title'), "og_description": scraped_content_val.get('og_description'),
                    "main_content_display": scraped_content_val.get('main_text'), "pdf_document_title": scraped_content_val.get('pdf_doc_title'),
                    "is_pdf": scraped_content_val.get('content_type') == 'application/pdf',
                    "source_query_type": "LLM-Generated" if keyword_val.lower() in llm_generated_keywords_set_for_display_set else "Original",
                    "scraping_error": scraped_content_val.get('error'), "content_type": scraped_content_val.get('content_type'), 
                    "llm_summary": None, "llm_extracted_info_q1": None, "llm_relevancy_score_q1": None, "llm_extracted_info_q1_full": None,
                    "llm_extracted_info_q2": None, "llm_relevancy_score_q2": None, "llm_extracted_info_q2_full": None,
                    "llm_extraction_query_1_text": primary_llm_extract_query or "", "llm_extraction_query_2_text": secondary_llm_extract_query or "",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                made_llm_call_for_item = False

                if scraped_content_val.get('error'): 
                    processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
                else:
                    min_main_text_length = getattr(app_config.scraper, 'min_main_text_length', 200) if hasattr(app_config, 'scraper') else 200
                    current_main_text: str = scraped_content_val.get('main_text', '')
                    is_good_scrape: bool = (current_main_text and len(current_main_text.strip()) >= min_main_text_length and "could not extract main content" not in current_main_text.lower() and "not processed for main text" not in current_main_text.lower() and not str(current_main_text).startswith("SCRAPER_INFO:"))

                    if is_good_scrape:
                        processing_log.append(f"    ✔️ Scraped with sufficient text (len={len(current_main_text)}, type: {item_data_val.get('content_type')}).")
                        successfully_scraped_for_this_keyword += 1
                        main_text_for_llm: str = current_main_text

                        if llm_key_available:
                            llm_api_key_to_use: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                            llm_model_to_use: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                            max_input_chars_for_llm = getattr(getattr(app_config, 'llm', object()), 'max_input_chars', 100000) # Safe access

                            update_progress_ui(message=f"LLM Summary for {url_to_scrape_val[:40]}...")
                            summary_text_val: Optional[str] = llm_processor.generate_summary(
                                main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=max_input_chars_for_llm)
                            item_data_val["llm_summary"] = summary_text_val
                            processing_log.append(f"        Summary: {str(summary_text_val)[:100] if summary_text_val else 'Failed/Empty'}...")
                            current_major_step_count += 1; update_progress_ui()
                            made_llm_call_for_item = True
                            
                            queries_to_process_for_item = []
                            if primary_llm_extract_query: queries_to_process_for_item.append({"query_text": primary_llm_extract_query, "id": "q1", "display_idx": 1})
                            if secondary_llm_extract_query: queries_to_process_for_item.append({"query_text": secondary_llm_extract_query, "id": "q2", "display_idx": 2})

                            for query_info in queries_to_process_for_item:
                                extraction_query = query_info["query_text"]
                                query_id_label = query_info["id"]; query_display_idx = query_info["display_idx"]
                                update_progress_ui(message=f"LLM Extract Q{query_display_idx} for {url_to_scrape_val[:40]}...")
                                
                                extracted_info_full: Optional[str] = llm_processor.extract_specific_information(
                                    main_text_for_llm, extraction_query=extraction_query, api_key=llm_api_key_to_use, 
                                    model_name=llm_model_to_use, max_input_chars=max_input_chars_for_llm)
                                
                                # --- START: CRITICAL SCORE PARSING FIX ---
                                raw_output_debug_msg = f"-----> PM DEBUG LLM RAW OUTPUT for Q{query_display_idx} ('{extraction_query[:50]}...'):\nSTART_OF_OUTPUT\n'{extracted_info_full}'\nEND_OF_OUTPUT\n-----"
                                print(raw_output_debug_msg)
                                processing_log.append(f"DEBUG_LLM_RAW_Q{query_display_idx}: '{str(extracted_info_full)[:300]}...'")

                                parsed_score_from_module: Optional[int] = None
                                content_from_module: str = extracted_info_full if extracted_info_full else ""

                                if hasattr(llm_processor, '_parse_score_and_get_content'):
                                    parsed_score_from_module, content_from_module = llm_processor._parse_score_and_get_content(extracted_info_full)
                                    print(f"-----> PM DEBUG: Score from llm_processor._parse_score_and_get_content: {parsed_score_from_module}")
                                else: # Fallback if the function doesn't exist in llm_processor for some reason
                                    parsed_score_from_module = _parse_score_from_extraction(extracted_info_full)
                                    if parsed_score_from_module is not None and extracted_info_full and '\n' in extracted_info_full:
                                        try: content_from_module = extracted_info_full.split('\n', 1)[1].strip()
                                        except IndexError: content_from_module = ""
                                    elif extracted_info_full: # No newline, content is whatever remains or full text if no score
                                        content_from_module = extracted_info_full.replace(f"Relevancy Score: {parsed_score_from_module}/5", "").strip() if parsed_score_from_module is not None else extracted_info_full
                                    print(f"-----> PM DEBUG: Using local _parse_score_from_extraction. Score: {parsed_score_from_module}")
                                
                                item_data_val[f"llm_extracted_info_{query_id_label}"] = content_from_module
                                item_data_val[f"llm_relevancy_score_{query_id_label}"] = parsed_score_from_module # Use score from module
                                item_data_val[f"llm_extracted_info_{query_id_label}_full"] = extracted_info_full 
                                # --- END: CRITICAL SCORE PARSING FIX ---
                                
                                processing_log.append(f"        Extracted (Q{query_display_idx}): Score={parsed_score_from_module}, Content='{str(content_from_module)[:70] if content_from_module else 'Failed/Empty'}'...")
                                current_major_step_count += 1; update_progress_ui()
                                made_llm_call_for_item = True
                        results_data.append(item_data_val)
                    else: 
                        processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
                        current_major_step_count += llm_tasks_per_good_item_calc; update_progress_ui()
                
                if apply_throttling and made_llm_call_for_item:
                    update_progress_ui(message=f"⏳ Throttling: Pausing for {llm_item_delay_seconds:.1f}s...")
                    time.sleep(llm_item_delay_seconds)
                elif not apply_throttling and made_llm_call_for_item: time.sleep(0.2) 
                elif not made_llm_call_for_item: time.sleep(0.1)
            
            if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
                skipped_llm_for_keyword = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * llm_tasks_per_good_item_calc
                current_major_step_count += skipped_llm_for_keyword; update_progress_ui()
        
        topic_for_consolidation_for_batch: str = ", ".join(initial_keywords_list) if initial_keywords_list else "the searched topics"
        if consolidated_summary_step_calc > 0:
            update_progress_ui(message="✨ Generating consolidated overview...")
            # --- START: CONSOLIDATED SUMMARY LOGIC (Adapted from v1.4.5 / earlier versions) ---
            temp_focused_texts_for_llm: List[str] = []
            processed_item_texts_for_focused = set() # To avoid duplicate texts if Q1/Q2 extract same snippet

            # Populate focused_summary_source_details and temp_focused_texts_for_llm
            if primary_llm_extract_query or secondary_llm_extract_query:
                for item_cs in results_data:
                    item_url_cs = item_cs.get("url", "Unknown URL")
                    if primary_llm_extract_query:
                        extraction_text_q1_cs = item_cs.get("llm_extracted_info_q1")
                        score_q1_cs = item_cs.get("llm_relevancy_score_q1")
                        full_q1_output_cs = item_cs.get("llm_extracted_info_q1_full", extraction_text_q1_cs)
                        if extraction_text_q1_cs and score_q1_cs is not None and score_q1_cs >= 3:
                            source_entry_q1: FocusedSummarySource = {"url": item_url_cs, "query_type": "Q1", "query_text": primary_llm_extract_query, "score": score_q1_cs, "llm_output_text": full_q1_output_cs or ""}
                            if not any(d['url'] == item_url_cs and d['query_type'] == 'Q1' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q1)
                            if extraction_text_q1_cs not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q1_cs); processed_item_texts_for_focused.add(extraction_text_q1_cs)
                    
                    if secondary_llm_extract_query:
                        extraction_text_q2_cs = item_cs.get("llm_extracted_info_q2")
                        score_q2_cs = item_cs.get("llm_relevancy_score_q2")
                        full_q2_output_cs = item_cs.get("llm_extracted_info_q2_full", extraction_text_q2_cs)
                        if extraction_text_q2_cs and score_q2_cs is not None and score_q2_cs >= 3:
                            source_entry_q2: FocusedSummarySource = {"url": item_url_cs, "query_type": "Q2", "query_text": secondary_llm_extract_query, "score": score_q2_cs, "llm_output_text": full_q2_output_cs or ""}
                            if not any(d['url'] == item_url_cs and d['query_type'] == 'Q2' for d in focused_summary_source_details):
                                focused_summary_source_details.append(source_entry_q2)
                            if extraction_text_q2_cs not in processed_item_texts_for_focused:
                                temp_focused_texts_for_llm.append(extraction_text_q2_cs); processed_item_texts_for_focused.add(extraction_text_q2_cs)
            
            final_texts_for_llm_consolidation = temp_focused_texts_for_llm
            llm_api_key_consol = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
            
            # Select model for consolidation (use specific consolidation model for Google if available)
            if app_config.llm.provider == "google":
                llm_model_consol = app_config.llm.google_gemini_model_consolidation if app_config.llm.google_gemini_model_consolidation else app_config.llm.google_gemini_model
            else:
                 llm_model_consol = app_config.llm.openai_model_summarize
            max_input_chars_consol = getattr(getattr(app_config, 'llm', object()), 'max_input_chars_consolidation', 150000) # Example

            if final_texts_for_llm_consolidation: # Attempt Focused Summary
                llm_context_q1_focused = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                if not primary_llm_extract_query and secondary_llm_extract_query: llm_context_q1_focused = secondary_llm_extract_query
                
                q2_contributed_to_focused = any(d['query_type'] == 'Q2' and d['score'] >=3 for d in focused_summary_source_details) if secondary_llm_extract_query else False
                enrichment_q2_for_focused = secondary_llm_extract_query if q2_contributed_to_focused else None

                processing_log.append(f"\n📋 Preparing FOCUSED consolidated summary based on {len(final_texts_for_llm_consolidation)} high-score items.")
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(
                    summaries=tuple(final_texts_for_llm_consolidation), topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_consol, model_name=llm_model_consol, max_input_chars=max_input_chars_consol,
                    extraction_query_for_consolidation=llm_context_q1_focused,
                    secondary_query_for_enrichment=enrichment_q2_for_focused)
                if not consolidated_summary_text_for_batch or "error" in consolidated_summary_text_for_batch.lower() or "failed" in consolidated_summary_text_for_batch.lower():
                     processing_log.append(f"  ❌ LLM FAILED FOCUSED summary. Output: {str(consolidated_summary_text_for_batch)[:100]}")
                else: processing_log.append(f"  ✔️ Successfully generated FOCUSED consolidated summary.")
            
            else: # Fallback to General Summary
                processing_log.append("  ℹ️ No high-scoring items for focused summary. Attempting GENERAL consolidated overview...")
                general_texts_for_consolidation: List[str] = [item.get("llm_summary","") for item in results_data if item.get("llm_summary") and not str(item.get("llm_summary")).lower().startswith(("llm error", "no text content", "llm_processor:"))]
                if general_texts_for_consolidation:
                     consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(
                         summaries=tuple(general_texts_for_consolidation), topic_context=topic_for_consolidation_for_batch, 
                         api_key=llm_api_key_consol, model_name=llm_model_consol, max_input_chars=max_input_chars_consol)
                     processing_log.append(f"  ✔️ General summary attempted. Result: {str(consolidated_summary_text_for_batch)[:100]}")
                else:
                     processing_log.append("  ❌ No valid item summaries for general overview."); consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No valid item summaries for general overview."
            # --- END: CONSOLIDATED SUMMARY LOGIC ---
            current_major_step_count += 1
            update_progress_ui(message="Consolidated overview generation attempt complete.")
        
        # Final LOG_STATUS messages
        # ... (Logic from v1.4.8 for setting LOG_STATUS:SUCCESS/WARNING/ERROR in processing_log) ...
        if results_data or consolidated_summary_text_for_batch:
            is_info_only_summary = consolidated_summary_text_for_batch and str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:")
            is_error_summary = consolidated_summary_text_for_batch and (str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:") or str(consolidated_summary_text_for_batch).lower().startswith("llm error"))
            if not is_info_only_summary and not is_error_summary and consolidated_summary_text_for_batch and not "error" in consolidated_summary_text_for_batch.lower() and not "failed" in consolidated_summary_text_for_batch.lower():
                processing_log.append("LOG_STATUS:SUCCESS:All processing complete! A consolidated overview has been generated.")
            elif is_error_summary :
                 processing_log.append(f"LOG_STATUS:ERROR:Processing completed with errors in LLM summary generation. Details: {consolidated_summary_text_for_batch}")
            elif is_info_only_summary :
                processing_log.append(f"LOG_STATUS:WARNING:Processing complete. Note on summary: {consolidated_summary_text_for_batch}")
            elif not consolidated_summary_text_for_batch and results_data :
                processing_log.append("LOG_STATUS:WARNING:Processing complete. Items processed, but no consolidated overview generated (LLM issue or no suitable content).")
            else: # Catch-all warning if no specific success/error/info for summary but data might exist
                processing_log.append("LOG_STATUS:WARNING:Processing complete. Check details; overview may be missing or sub-optimal.")
        else: 
            processing_log.append("LOG_STATUS:WARNING:Processing complete, but no result data was generated.")


        # Google Sheets Writing
        # ... (Logic from v1.4.8 for Google Sheets, ensuring it updates processing_log) ...
        print("-----> DEBUG (process_manager v1.4.9): Starting Google Sheets Writing block.")
        if sheet_writing_enabled and gs_worksheet:
            processing_log.append(f"\n💾 Checking conditions to write batch data to Google Sheets...")
            if results_data or (consolidated_summary_text_for_batch and not str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:")): # Ensure summary has content
                batch_ts_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
                processing_log.append(f"LOG_STATUS:SHEET_WRITE_ATTEMPTED:Attempting to write to Google Sheets at {batch_ts_sheet}...") 
                active_ext_queries_sheet = [q for q in llm_extract_queries_input if q and q.strip()]
                write_ok: bool = False
                try:
                    write_ok = data_storage.write_batch_summary_and_items_to_sheet(
                        worksheet=gs_worksheet, batch_timestamp=batch_ts_sheet, consolidated_summary=consolidated_summary_text_for_batch,
                        topic_context=topic_for_consolidation_for_batch, item_data_list=results_data, extraction_queries_list=active_ext_queries_sheet)
                except Exception as e_write: processing_log.append(f"LOG_STATUS:SHEET_WRITE_ERROR: CRITICAL ERROR during sheet write: {e_write}")
                if write_ok: processing_log.append("LOG_STATUS:SHEET_WRITE_SUCCESS:Batch data written to Google Sheets.")
                else: processing_log.append("LOG_STATUS:SHEET_WRITE_FAILED:Failed to write to Google Sheets (data_storage returned False or error).")
            else: processing_log.append(f"LOG_STATUS:SHEET_WRITE_NO_DATA:No substantial data (results or valid summary) to write to Google Sheets.")
        # ... (other sheet logging conditions from v1.4.8) ...


    except Exception as e_main_pm: 
        error_message = f"CRITICAL_ERROR_IN_PROCESS_MANAGER:{type(e_main_pm).__name__} - {e_main_pm}"
        processing_log.append(error_message)
        processing_log.append(traceback.format_exc())
        print(f"-----> DEBUG (process_manager v1.4.9): EXCEPTION caught: {error_message}")
            
    finally: 
        print(f"-----> DEBUG (process_manager v1.4.9): FINALLY BLOCK.")
        if current_major_step_count < total_major_steps_for_progress and total_major_steps_for_progress > 1 :
             current_major_step_count = total_major_steps_for_progress 
        update_progress_ui(message="Processing finalized. Check app messages and logs.")
        progress_bar_placeholder.progress(100)
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display_set, llm_generated_keywords_set_for_display_set, focused_summary_source_details

# // end of modules/process_manager.py
