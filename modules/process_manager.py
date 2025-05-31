# modules/process_manager.py
# Version 1.3.3:
# - Modified consolidated summary logic to include high-scoring Q2 items:
#   - If any Q1 or Q2 item scores >=3, collect their respective
#     `llm_extracted_info_q1` or `llm_extracted_info_q2` text.
#   - Attempt a FOCUSED summary with all such collected Q1/Q2 texts.
#     The `primary_llm_extract_query` is used as the context for this focused summary if provided.
#   - If this focused LLM call fails or returns an LLM-internal error/problematic message,
#     report that failure directly. NO fallback to general summaries.
#   - Only if NO Q1 items AND NO Q2 items score >=3 (or no specific queries were provided),
#     then attempt a GENERAL overview using item `llm_summary` fields.
# - Retains progress bar fix and simplified general overview prefix from previous versions.
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
    secondary_llm_extract_query: Optional[str] = None
    if llm_extract_queries_input:
        if len(llm_extract_queries_input) > 0 and llm_extract_queries_input[0] and llm_extract_queries_input[0].strip():
            primary_llm_extract_query = llm_extract_queries_input[0].strip()
        if len(llm_extract_queries_input) > 1 and llm_extract_queries_input[1] and llm_extract_queries_input[1].strip():
            secondary_llm_extract_query = llm_extract_queries_input[1].strip()

    # ... (LLM query generation and item processing loop as in v1.3.2 - no changes needed there) ...
    # The item processing loop already populates item_data_val["llm_extracted_info_q1"] and item_data_val["llm_extracted_info_q2"]
    # --- Start of existing item processing loop (condensed for brevity, no changes from v1.3.2) ---
    if llm_key_available and initial_keywords_list: 
        processing_log.append("\n🧠 Generating additional search queries with LLM...")
        # ... (LLM query generation logic as before) ...
    # ... (Progress bar setup as before) ...
    for keyword_val in keywords_list_val_runtime: 
        processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        # ... (Search, scrape, individual LLM summary & extraction logic as before) ...
        # This loop populates `results_data` with items, each potentially having 
        # 'llm_extracted_info_q1' and 'llm_extracted_info_q2' fields
    # --- End of existing item processing loop ---

    final_progress_value = (current_major_step_count / total_major_steps_for_progress) if total_major_steps_for_progress > 0 else 1.0 # Placeholder for actual calc
    with progress_bar_placeholder.container(): # Placeholder for actual calc
        st.progress(min(max(final_progress_value, 0.0), 1.0), text="Processing complete. Generating final report...") # Placeholder for actual calc

    topic_for_consolidation_for_batch: str
    if not initial_keywords_list: topic_for_consolidation_for_batch = "the searched topics" 
    elif len(initial_keywords_list) == 1: topic_for_consolidation_for_batch = initial_keywords_list[0]
    else: topic_for_consolidation_for_batch = f"topics: {', '.join(initial_keywords_list[:3])}{'...' if len(initial_keywords_list) > 3 else ''}"

    # --- CONSOLIDATED SUMMARY LOGIC (Version 1.3.3) ---
    if results_data and llm_key_available:
        processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview..."):
            texts_for_focused_summary: List[str] = []
            
            # 1. Check for any Q1 or Q2 scores >= 3
            # Collect entire llm_extracted_info_qX field if score is >= 3
            if primary_llm_extract_query or secondary_llm_extract_query: # Only try to collect if a query was specified
                for item in results_data:
                    # Check Q1
                    if primary_llm_extract_query:
                        extraction_text_q1 = item.get("llm_extracted_info_q1")
                        if extraction_text_q1:
                            score_q1 = _parse_score_from_extraction(extraction_text_q1)
                            if score_q1 is not None and score_q1 >= 3:
                                texts_for_focused_summary.append(extraction_text_q1)
                    
                    # Check Q2
                    if secondary_llm_extract_query:
                        extraction_text_q2 = item.get("llm_extracted_info_q2")
                        if extraction_text_q2:
                            score_q2 = _parse_score_from_extraction(extraction_text_q2)
                            if score_q2 is not None and score_q2 >= 3:
                                texts_for_focused_summary.append(extraction_text_q2)
            
            # Remove duplicates if an item's Q1 and Q2 were identical and both scored high (edge case)
            if texts_for_focused_summary:
                texts_for_focused_summary = sorted(list(set(texts_for_focused_summary)))


            # 2. If ANY Q1 or Q2 items meet criteria (score >= 3), attempt ONLY FOCUSED summary
            if texts_for_focused_summary:
                # Determine context for the LLM based on which queries yielded results
                focused_query_context = "specific extracted information" # Default
                if primary_llm_extract_query and any(_parse_score_from_extraction(txt) is not None and _parse_score_from_extraction(txt) >=3 for txt in texts_for_focused_summary if primary_llm_extract_query in item.get("llm_extraction_query_q1","")): # Rough check
                     focused_query_context = primary_llm_extract_query
                elif secondary_llm_extract_query and any(_parse_score_from_extraction(txt) is not None and _parse_score_from_extraction(txt) >=3 for txt in texts_for_focused_summary if secondary_llm_extract_query in item.get("llm_extraction_query_q2","")):
                     focused_query_context = secondary_llm_extract_query
                # If both primary and secondary queries are present and contributed, primary takes precedence for context label.
                # More robust way: check if any of the texts_for_focused_summary originated from Q1 specifically, etc.
                # For now, primary_llm_extract_query (if it exists) is a good general context if any specific info was found.
                # If primary_llm_extract_query is None but secondary exists & items were found, that could be context.
                # Simplified: if primary_llm_extract_query exists, use it as the main topic for LLM.
                
                llm_context_for_focused_summary = primary_llm_extract_query if primary_llm_extract_query else focused_query_context

                processing_log.append(f"  Attempting FOCUSED consolidated summary using {len(texts_for_focused_summary)} high-scoring Q1/Q2 item(s). Context: '{llm_context_for_focused_summary}'.")
                llm_api_key_to_use_consol: Optional[str] = app_config.llm.google_gemini_api_key if app_config.llm.provider == "google" else app_config.llm.openai_api_key
                llm_model_to_use_consol: str = app_config.llm.google_gemini_model if app_config.llm.provider == "google" else app_config.llm.openai_model_summarize
                
                generated_focused_summary = llm_processor.generate_consolidated_summary(
                    summaries=tuple(texts_for_focused_summary),
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use_consol, model_name=llm_model_to_use_consol,
                    max_input_chars=app_config.llm.max_input_chars,
                    extraction_query_for_consolidation=llm_context_for_focused_summary # Use the determined context
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
                    processing_log.append(f"  ❌ LLM failed to generate a FOCUSED summary from the {len(texts_for_focused_summary)} provided high-scoring Q1/Q2 item(s). LLM Output (first 100 chars): {str(generated_focused_summary)[:100]}")
                    error_query_context_msg = ""
                    if primary_llm_extract_query: error_query_context_msg += f" Q1 ('{primary_llm_extract_query}')"
                    if secondary_llm_extract_query: error_query_context_msg += f"{' or ' if primary_llm_extract_query else ''}Q2 ('{secondary_llm_extract_query}')"
                    
                    consolidated_summary_text_for_batch = (
                        f"LLM_PROCESSOR_ERROR: The LLM failed to generate a focused consolidated summary based on "
                        f"{len(texts_for_focused_summary)} item(s) that met score >=3 criteria for specific queries{error_query_context_msg}. "
                        f"The LLM processor reported: \"{str(generated_focused_summary)[:100]}...\""
                    )
                else:
                    consolidated_summary_text_for_batch = generated_focused_summary
                    processing_log.append(f"  ✔️ Successfully generated FOCUSED consolidated summary from Q1/Q2 items.")

            # 3. If NO Q1 or Q2 items meet criteria (score >= 3) (or no specific queries provided), attempt GENERAL overview
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
                        consolidated_summary_text_for_batch = general_overview_info_prefix + "\n\n--- General Overview ---\n" + generated_general_overview
                        processing_log.append("  ✔️ Successfully generated GENERAL consolidated overview.")
                    else:
                        processing_log.append(f"  ❌ LLM failed to generate a GENERAL overview from item summaries. LLM Output: {str(generated_general_overview)[:150]}")
                        consolidated_summary_text_for_batch = general_overview_info_prefix + "\n\n--- General Overview ---\nLLM_PROCESSOR_ERROR: The LLM failed to generate a general consolidated overview from the available item summaries."
                else: 
                    processing_log.append(f"  ❌ No valid general item summaries found to generate a general overview.{log_message_reason}")
                    consolidated_summary_text_for_batch = general_overview_info_prefix + " However, no valid item summaries were available to generate it."
        
        with progress_bar_placeholder.container(): st.empty() 
    
    elif not results_data and llm_key_available: 
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: No result items were successfully scraped and processed to create a consolidated summary."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")
    elif not llm_key_available: 
        consolidated_summary_text_for_batch = "LLM_PROCESSOR_INFO: LLM processing is not configured. Consolidated summary cannot be generated."
        processing_log.append(f"\nℹ️ {consolidated_summary_text_for_batch}")

    # ... (Google Sheets Writing and final messages as in v1.3.2) ...
    if sheet_writing_enabled and gs_worksheet: 
        # ...
        pass # Code as before
    # ...

    if results_data or consolidated_summary_text_for_batch:
        # ...
        pass # Code as before
    # ...
        
    return processing_log, results_data, consolidated_summary_text_for_batch, initial_keywords_for_display, llm_generated_keywords_set_for_display

# end of modules/process_manager.py
