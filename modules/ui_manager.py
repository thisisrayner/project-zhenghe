# modules/ui_manager.py
# Version 1.1.11:
# - Reads 'llm_globally_enabled' from st.session_state for LLM availability check.
# - Removes recalculation of llm_key_available from app_config within display_individual_results.
# Previous versions:
# - Version 1.1.10: Corrected logic for displaying "LLM processing disabled" captions.

"""
Manages the Streamlit User Interface elements, layout, and user inputs for D.O.R.A.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
# from modules import config # No longer need to import config for AppConfig type hint if not passed
import re
import html 

# ... (sanitize_text_for_markdown, _parse_score_from_extraction, get_display_prefix_for_item - unchanged) ...
# ... (apply_custom_css, display_consolidated_summary_and_sources - unchanged) ...

def render_sidebar(cfg: 'config.AppConfig', current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]:
    # This function might still need app_config (cfg) for other details like default slider values,
    # or provider name. If st.session_state.llm_globally_enabled is used, the direct
    # calculation of llm_key_available_sidebar can be replaced.
    with st.sidebar:
        st.subheader("Search Parameters")
        # ... (keywords_input_val, num_results_wanted_per_keyword) ...
        keywords_input_val: str = st.text_input("Keywords (comma-separated):", value=st.session_state.get('last_keywords', ""), key="keywords_text_input_main", help="Enter comma-separated keywords. Press Enter to apply.")
        default_slider_val = getattr(cfg, 'num_results_per_keyword_default', 3) if cfg else 3
        num_results_wanted_per_keyword: int = st.slider("Number of successfully scraped results per keyword:", 1, 10, default_slider_val, key="num_results_slider")


        # Use the globally set LLM status from session state
        llm_is_enabled_globally = st.session_state.get('llm_globally_enabled', False)
        
        llm_provider_display = "N/A"
        model_display_name: str = "N/A"
        if cfg and hasattr(cfg, 'llm'): # Still need cfg for provider and model name display
            llm_provider_display = cfg.llm.provider.upper()
            if llm_is_enabled_globally:
                model_display_name = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                st.subheader(f"LLM Processing - Provider: {llm_provider_display}")
                st.caption(f"Using Model: {model_display_name}")
            else:
                st.subheader(f"LLM Processing - Provider: {llm_provider_display}")
                st.caption(f"API Key for {cfg.llm.provider.upper()} not configured or LLM disabled.")
        else:
            st.subheader(f"LLM Processing - Provider: {llm_provider_display}")
            st.caption("LLM Configuration not loaded. LLM features disabled.")

        # ... (rest of sidebar - unchanged) ...
        st.markdown("**Specific Info to Extract (LLM):**")
        last_extract_queries_val = st.session_state.get('last_extract_queries', ["", ""])
        if not isinstance(last_extract_queries_val, list) or len(last_extract_queries_val) < 2: last_extract_queries_val = ["", ""]
        tooltip_text_q1 = "Main Query 1 drives primary relevancy and focused summary."
        tooltip_text_q2 = "Optional: secondary question or keywords."
        llm_extract_query_1_input_val: str = st.text_input("Main Query 1:", value=last_extract_queries_val[0], placeholder="e.g., Key methodologies", key="llm_extract_q1_input_sidebar", help=tooltip_text_q1) # Changed key
        llm_extract_query_2_input_val: str = st.text_input("Additional Query 2:", value=last_extract_queries_val[1], placeholder="e.g., Mentioned limitations", key="llm_extract_q2_input_sidebar", help=tooltip_text_q2) # Changed key
        returned_queries = [llm_extract_query_1_input_val, llm_extract_query_2_input_val]
        st.markdown("---")
        button_streamlit_type = "secondary"; button_disabled = True
        button_help_text = current_gsheets_error or "Google Sheets status undetermined."
        if sheet_writing_enabled: button_streamlit_type = "primary"; button_disabled = False; button_help_text = "Google Sheets connected. Ready to process."
        start_button_val: bool = st.button("🚀 Start Search & Analysis", type=button_streamlit_type, use_container_width=True, disabled=button_disabled, help=button_help_text)
        st.markdown("---")
        st.caption("✨ LLM-generated search queries will be used if LLM available.")
        st.caption("📄 LLM Summaries for items will be generated if LLM available.")
    return keywords_input_val, num_results_wanted_per_keyword, returned_queries, start_button_val


def display_individual_results():
    if st.session_state.get('results_data'):
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        
        # Get the global LLM availability status from session state
        llm_globally_enabled = st.session_state.get('llm_globally_enabled', False)
        print(f"UI_MANAGER DEBUG: llm_globally_enabled from session_state: {llm_globally_enabled}")

        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
        last_extract_queries_for_display = st.session_state.get('last_extract_queries', ["", ""]) 
        
        for i, item_val_display in enumerate(st.session_state.results_data):
            # ... (expander title setup and item details - unchanged from v1.1.10) ...
            display_title_ui = item_val_display.get('page_title') or item_val_display.get('og_title') or \
                               item_val_display.get('search_title') or item_val_display.get('pdf_document_title') or "Untitled"
            score_emoji_prefix = get_display_prefix_for_item(item_val_display)
            content_type_marker = "📄 " if item_val_display.get('is_pdf') else ""
            is_llm_keyword_source = item_val_display.get('keyword_searched','').lower() in llm_gen_kws_for_display
            llm_query_marker = "🤖" if is_llm_keyword_source else ""
            full_prefix_parts = [part for part in [score_emoji_prefix, llm_query_marker, content_type_marker] if part]
            full_prefix = " ".join(full_prefix_parts) + " " if full_prefix_parts else ""
            expander_title_ui = (f"{full_prefix}"
                                 f"{item_val_display.get('keyword_searched','Unknown Keyword')} | "
                                 f"{display_title_ui} ({item_val_display.get('url', 'No URL')})").replace("  ", " ").strip()

            with st.expander(expander_title_ui):
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                st.caption(f"Content Type: {item_val_display.get('content_type', 'N/A')}")
                if item_val_display.get('scraping_error'): st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                with st.container(border=True): 
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {sanitize_text_for_markdown(item_val_display.get('page_title', 'N/A'))}\n" 
                                f"  - **Meta Desc:** {sanitize_text_for_markdown(item_val_display.get('meta_description', 'N/A'))}\n" 
                                f"  - **OG Title:** {sanitize_text_for_markdown(item_val_display.get('og_title', 'N/A'))}\n"         
                                f"  - **OG Desc:** {sanitize_text_for_markdown(item_val_display.get('og_description', 'N/A'))}")    
                scraped_main_text = item_val_display.get('main_content_display', '') 
                if scraped_main_text and not str(scraped_main_text).startswith("SCRAPER_INFO:"):
                    with st.popover("View Main Text", use_container_width=True): 
                        st.text_area(f"Main Text ({item_val_display.get('content_type')})", value=scraped_main_text, height=400, key=f"main_text_popover_{i}", disabled=True)
                elif str(scraped_main_text).startswith("SCRAPER_INFO:"): st.caption(scraped_main_text)
                else: st.caption("No main text extracted or usable for LLM processing.")
                
                has_llm_insights = False # This flag is local to each item
                insights_container = st.container(border=True)
                raw_llm_summary = item_val_display.get("llm_summary")
                if raw_llm_summary and not str(raw_llm_summary).lower().startswith(("llm error", "llm_processor", "no text content")): 
                    has_llm_insights = True
                    insights_container.markdown("**Summary (LLM):**")
                    insights_container.markdown(raw_llm_summary)
                
                for q_idx in range(2): 
                    # ... (logic to display Q1/Q2 info and score - unchanged from v1.1.10) ...
                    query_key_text = f"llm_extraction_query_{q_idx+1}_text"
                    score_key = f"llm_relevancy_score_q{q_idx+1}"
                    extracted_info_key = f"llm_extracted_info_q{q_idx+1}"
                    query_text_from_item = item_val_display.get(query_key_text)
                    query_text_for_label = query_text_from_item if query_text_from_item else \
                                          (last_extract_queries_for_display[q_idx] if q_idx < len(last_extract_queries_for_display) else "")
                    raw_extracted_content = item_val_display.get(extracted_info_key)
                    score_value = item_val_display.get(score_key)
                    query_label_prefix = f"Main Query 1" if q_idx == 0 else f"Additional Query 2"
                    if query_text_for_label and query_text_for_label.strip() and raw_extracted_content:
                        has_llm_insights = True
                        insights_container.markdown(f"**Extracted Info for {query_label_prefix} ('{query_text_for_label}'):**")
                        if score_value is not None:
                            insights_container.markdown(f"    *Relevancy Score for this item: {score_value}/5*")
                        insights_container.markdown(raw_extracted_content)

                # --- CORRECTED CAPTION LOGIC using llm_globally_enabled ---
                if not llm_globally_enabled:
                    insights_container.caption("LLM processing disabled (no API key configured); no LLM insights could be generated.")
                elif not has_llm_insights: # LLM is globally enabled, but this item has no insights
                    insights_container.caption("No specific LLM insights (summary/extractions) generated or available for this item.")
                # --- END CORRECTED CAPTION LOGIC ---
                
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

# ... (display_processing_log - unchanged) ...
def display_processing_log():
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            log_content = "\n".join(st.session_state.processing_log)
            filtered_log_lines = [line for line in log_content.splitlines() if not line.startswith("LOG_UI_STATUS:")]
            st.text_area("Processing Log (filtered):", value="\n".join(filtered_log_lines), height=300, key="log_display_text_area_ui", disabled=True) # Changed key

# // end of modules/ui_manager.py
