# modules/process_manager.py
# Version 1.3.6:
# - Enhanced logging for focused consolidated summary: now logs detailed source
#   information (URL, query type, score) for each item contributing to the summary.
# - `run_search_and_analysis` now returns a list of these source details for potential UI display.
# - Corrected NameError for `config.AppConfig` type hint by using string literal.
# Version 1.3.5:
# - Updated call to `llm_processor.generate_search_queries` to include Q2.
"""
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
"""
import streamlit as st
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple, TypedDict
from modules import config, search_engine, scraper, llm_processor, data_storage # Ensure config is imported

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
    app_config: 'config.AppConfig', # MODIFIED: Used string literal for forward reference
    keywords_input: str,
    llm_extract_queries_input: List[str],
    num_results_wanted_per_keyword: int,
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
        total_llm_tasks_per_good_scrape += 1 
        total_llm_tasks_per_good_scrape += len([q for q in llm_extract_queries_input if q.strip()]) 

    total_major_steps_for_progress: int = (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
                                          (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0: 
        total_major_steps_for_progress += 1 
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty()

    if llm_key_available and initial_keywords_list and min(math.floor(len(initial_keywords_list) * 1.5), 5) > 0:
        current_major_step_count +=1
        progress_text = "LLM Query Generation Complete..."
        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
        with progress_bar_placeholder.container(): 
            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text)

    # --- Item Processing Loop ---
    for keyword_val in keywords_list_val_runtime: 
        # ... (rest of the item processing loop is unchanged)
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
            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
            with progress_bar_placeholder.container(): 
                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_scrape)

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
                        
                        current_major_step_count += 1
                        progress_text_llm_summary = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                        progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                        with progress_bar_placeholder.container(): 
                            st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_summary)
                        processing_log.append(f"       Generating LLM summary...")
                        summary: Optional[str] = llm_processor.generate_summary(main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                        item_data_val["llm_summary"] = summary
                        processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                        time.sleep(0.1) 

                        for q_idx, extraction_query in enumerate(llm_extract_queries_input):
                            if not extraction_query.strip(): continue
                            current_major_step_count += 1
                            progress_text_llm_extract = f"LLM Extract Q{q_idx+1} ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 0
                            with progress_bar_placeholder.container(): 
                                st.progress(min(max(progress_value, 0.0), 1.0), text=progress_text_llm_extract)
                            processing_log.append(f"      Extracting info for Q{q_idx+1}: '{extraction_query}'...")
                            extracted_info: Optional[str] = llm_processor.extract_specific_information(main_text_for_llm, extraction_query=extraction_query, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=app_config.llm.max_input_chars)
                            item_data_val[f"llm_extracted_info_q{q_idx+1}"] = extracted_info
                            processing_log.append(f"        Extracted (Q{q_idx+1}): {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1) 
                    results_data.append(item_data_val)
                else: 
                    processing_log.append(f"    ⚠️ Scraped, but main text insufficient. LLM processing skipped.")
            time.sleep(0.2) 
        
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword: 
            processing_log.append(f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} desired scrapes.")
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword 
    # --- End of Item Processing Loop ---

    final_progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 1.0
    with progress_bar_placeholder.container():
        st.progress(min(max(final_progress_value, 0.0), 1.0), text="Processing complete. Generating final report...")
        if abs(final_progress_value - 1.0) > 0.01 and total_major_steps_for_progress > 0 and final_progress_value <=1.0 :
             processing_log.append(f"  DEBUG: Final progress: current_steps={current_major_step_count}, total_steps={total_major_steps_for_progress}, value={final_progress_value}")

    topic_for_consolidation_for_batch: str
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics"
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            temp_focused_texts_for_llm: List[str] = [] 

            if primary_llm_extract_query or secondary_llm_extract_query:
                # focused_summary_source_details is already initialized as an empty list
                # We will populate it here.
                processed_item_texts_for_focused = set() # To avoid adding the exact same text snippet multiple times to LLM

                for item in results_data:
                    item_url = item.get("url", "Unknown URL")
                    
                    # Check Q1
                    if primary_llm_extract_query:
                        extraction_text_q1 = item.get("llm_extracted_info_q1")
                        if extraction_text_q1:
                            score_q1 = _parse_score_from_extraction(extraction_text_q1)
                            if score_q1 is not None and score_q1 >= 3:
                                source_entry_q1: FocusedSummarySource = {
                                    "url": item_url, "query_type": "Q1",
                                    "query_text": primary_llm_extract_query, "score": score_q1,
                                    "llm_output_text": extraction_text_q1
                                }
                                # Add to details if not an exact duplicate entry (e.g. same URL, Q-type, text)
                                if not any(d == source_entry_q1 for d in focused_summary_source_details):
                                    focused_summary_source_details.append(source_entry_q1)
                                
                                if extraction_text_q1 not in processed_item_texts_for_focused:
                                    temp_focused_texts_for_llm.append(extraction_text_q1)
                                    processed_item_texts_for_focused.add(extraction_text_q1)
                    
                    # Check Q2
                    if secondary_llm_extract_query:
                        extraction_text_q2 = item.get("llm_extracted_info_q2")
                        if extraction_text_q2:
                            score_q2 = _parse_score_from_extraction(extraction_text_q2)
                            if score_q2 is not None and score_q2 >= 3:
                                source_entry_q2: FocusedSummarySource = {
                                    "url": item_url, "query_type": "Q2",
                                    "query_text": secondary_llm_extract_query, "score": score_q2,
                                    "llm_output_text": extraction_text_q2
                                }
                                if not any(d == source_entry_q2 for d in focused_summary_source_details):
                                     focused_summary_source_details.append(source_entry_q2)

                                if extraction_text_q2 not in processed_item_texts_for_focused:
                                    temp_focused_texts_for_llm.append(extraction_text_q2)
                                    processed_item_texts_for_focused.add(extraction_text_q2)
            
            final_texts_for_llm = temp_focused_texts_for_llm # Already unique due to `processed_item_texts_for_focused` set

            if final_texts_for_llm:
                llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else "specific extracted information"
                if not primary_llm_extract_query and secondary_llm_extract_query:
                    llm_context_for_focused_summary = secondary_llm_extract_query

                processing_log.append(f"\n📋 Preparing inputs for FOCUSED consolidated summary:")
                processing_log.append(f"  Central Query Theme (typically based on Q1): '{llm_context_for_focused_summary}'")
                if focused_summary_source_details: # Check if the detailed list has items
                    processing_log.append(f"  Found {len(focused_summary_source_details)} source snippet(s) (from Q1/Q2 extractions scoring >=3/5) that contributed text for the LLM:")
                    # Sort for consistent logging, e.g., by URL then query_type
                    sorted_source_details = sorted(focused_summary_source_details, key=lambda x: (x['url'], x['query_type']))
                    for source_item in sorted_source_details:
                        log_url = source_item['url']
                        log_q_type = source_item['query_type']
                        log_q_text_short = source_item['query_text'][:30] + "..." if len(source_item['query_text']) > 30 else source_item['query_text']
                        log_score = source_item['score']
                        processing_log.append(f"    - From: {log_q_type} ('{log_q_text_short}') on URL: {log_url} (Score: {log_score}/5)")
                else:
                     processing_log.append(f"  Found {len(final_texts_for_llm)} unique text snippet(s) for LLM input (details not itemized in this pass).")


                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize

                generated_focused_summary = llm_processor.generate_consolidated_summary(
                    summaries=tuple(final_texts_for_llm), 
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                    max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=llm_context_for_focused_summary
                )

                is_llm_call_problematic = False
                if not generated_focused_summary: is_llm_call_problematic = True
                else:
                    problematic_substrings = ["llm_processor", "could not generate", "no items met score", "no suitable content"]
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

            else: 
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
                        extraction_query_for_consolidation=None
                    )
                    
                    if generated_general_overview and not str(generated_general_overview).lower().startswith("llm_processor"):
                        consolidated_summary_text_for_batch = generated_general_overview
                        processing_log.append("  ✔️ Successfully generated GENERAL consolidated overview.")
                    else:
                        processing_log.append(f"  ❌ LLM failed to generate a GENERAL overview from item summaries. LLM Output: {str(generated_general_overview)[:150]}")
                        consolidated_summary_text_for_batch = (general_overview_info_prefix +
                                                              "\n\n--- General Overview ---\nLLM_PROCESSOR_ERROR: The LLM failed to generate a general consolidated overview from the available item summaries.")
                else: 
                    no_summaries_message_suffix = " Additionally, no valid general item summaries were found to generate a general overview."
                    log_message_text = general_overview_info_prefix
                    if log_message_reason: log_message_text = log_message_text.replace("." , "") + log_message_reason
                    processing_log.append(f"  ❌ {log_message_text}{no_summaries_message_suffix}")
                    consolidated_summary_text_for_batch = general_overview_info_prefix + " However, no valid item summaries were available to generate it."

        with progress_bar_placeholder.container(): st.empty() 

    elif not results_data and llm_key_available: 
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items were successfully scraped and processed to create a consolidated summary."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
    elif not llm_key_available: 
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")

    # ... (Google Sheets writing logic is unchanged)
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

    # ... (Final UI messages are unchanged)
    if results_data or consolidated_summary_text_for_batch:
        is_info_or_error_summary = consolidated_summary_text_for_batch and \
                                   (str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_info:") or \
                                    str(consolidated_summary_text_for_batch).lower().startswith("llm_processor_error:"))

        if not is_info_or_error_summary and consolidated_summary_text_for_batch:
            st.success("All processing complete! A consolidated overview has been generated. See above.")
        elif is_info_or_error_summary: 
            st.warning("Processing complete. Please check the consolidated overview section for details on the summary generation process.")
        elif not consolidated_summary_text_for_batch: 
             st.warning("Processing complete, but no consolidated overview was generated as no suitable content was found.")
    else: 
        st.warning("Processing complete, but no data was generated (no results to process).")
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display, focused_summary_source_details

# end of modules/process_manager.py
