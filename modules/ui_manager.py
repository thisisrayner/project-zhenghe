# modules/ui_manager.py
# Version 1.1.4:
# 1. Added sanitize_text_for_markdown helper to prevent LLM output rendering issues.
# 2. Applied sanitization to consolidated summary, individual summaries, and extracted info.
# (Retains label and emoji logic from v1.1.3)
"""
Manages the Streamlit User Interface elements, layout, and user inputs.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
from modules import config # For AppConfig type hint
import re # For sanitization regex

# --- Sanitization Helper ---
def sanitize_text_for_markdown(text: Optional[str]) -> str:
    """
    Sanitizes text to prevent common markdown rendering issues, especially from LLM output.
    Escapes markdown special characters.
    """
    if text is None:
        return ""
    
    # Escape backslashes first as they are used in escape sequences
    escaped_text = text.replace('\\', '\\\\')
    
    # Characters that have special meaning in Markdown
    # Note: `.` and `!` are common and usually don't need escaping unless part of ordered lists or image links.
    # We'll be more conservative for now.
    # Not escaping '>' and '<' here as it can make text less readable if not actual HTML.
    # If HTML-like tags become an issue, they can be added or a proper HTML sanitizer used.
    markdown_chars_to_escape = r"([`*_#{}\[\]()+.!-])" # Removed hyphen from general escape, handle lists carefully
                                                      # Added hyphen to the list as it can create horizontal rules or list items unexpectedly.
    
    escaped_text = re.sub(markdown_chars_to_escape, r"\\\1", escaped_text)
    
    # Handle specific cases like three or more hyphens which create horizontal rules
    escaped_text = re.sub(r"---", r"\-\-\-", escaped_text) # Escape at least three hyphens
    escaped_text = re.sub(r"\*\*\*", r"\*\*\*", escaped_text) # Escape at least three asterisks
    escaped_text = re.sub(r"___", r"\_\_\_", escaped_text) # Escape at least three underscores


    return escaped_text

# _parse_score_from_extraction and get_display_prefix_for_item remain the same as v1.1.3
def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]:
    score: Optional[int] = None
    if extracted_info and isinstance(extracted_info, str) and extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError): pass 
    return score

def get_display_prefix_for_item(item_data: Dict[str, Any]) -> str:
    prefix = ""
    score_q1 = _parse_score_from_extraction(item_data.get("llm_extracted_info_q1"))
    score_q2 = _parse_score_from_extraction(item_data.get("llm_extracted_info_q2"))
    highest_score: Optional[int] = None
    if score_q1 is not None and score_q2 is not None: highest_score = max(score_q1, score_q2)
    elif score_q1 is not None: highest_score = score_q1
    elif score_q2 is not None: highest_score = score_q2
    if highest_score is not None:
        if highest_score == 5: prefix = "5️⃣"
        elif highest_score == 4: prefix = "4️⃣"
        elif highest_score == 3: prefix = "3️⃣"
    return prefix

# render_sidebar remains the same as v1.1.3
def render_sidebar(cfg: config.AppConfig, current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]:
    with st.sidebar:
        st.subheader("Search Parameters")
        keywords_input_val: str = st.text_input(
            "Keywords (comma-separated):", value=st.session_state.get('last_keywords', ""),
            key="keywords_text_input_main", help="Enter comma-separated keywords. Press Enter to apply."
        )
        num_results_wanted_per_keyword: int = st.slider(
            "Number of successfully scraped results per keyword:", 1, 10, cfg.num_results_per_keyword_default,
            key="num_results_slider"
        )
        st.subheader(f"LLM Processing - Provider: {cfg.llm.provider.upper()}")
        llm_key_available: bool = (cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key) or \
                                  (cfg.llm.provider == "openai" and cfg.llm.openai_api_key)
        if llm_key_available:
            model_display_name: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
            st.caption(f"Using Model: {model_display_name}")
        else:
            st.caption(f"API Key for {cfg.llm.provider.upper()} not configured. LLM features disabled.")
        st.markdown("**Specific Info to Extract (LLM):**")
        last_extract_queries = st.session_state.get('last_extract_queries', ["", ""])
        if not isinstance(last_extract_queries, list) or len(last_extract_queries) < 2:
            last_extract_queries = ["", ""]
        tooltip_text_q1 = "You can type in keywords to focus the analysis or ask a question. Main Query 1 drives the primary relevancy score and focused consolidated summary."
        tooltip_text_q2 = "Optional: ask a different question or look for other specific keywords."
        llm_extract_query_1_input_val: str = st.text_input(
            "Main Query 1:", value=last_extract_queries[0],
            placeholder="e.g., Key methodologies, primary conclusions", key="llm_extract_q1_input", help=tooltip_text_q1 
        )
        llm_extract_query_2_input_val: str = st.text_input(
            "Additional Query 2:", value=last_extract_queries[1],
            placeholder="e.g., Mentioned limitations, future work", key="llm_extract_q2_input", help=tooltip_text_q2
        )
        returned_queries = [llm_extract_query_1_input_val, llm_extract_query_2_input_val]
        st.markdown("---")
        button_streamlit_type = "secondary"; button_disabled = True
        button_help_text = current_gsheets_error or "Google Sheets connection status undetermined."
        if sheet_writing_enabled:
            button_streamlit_type = "primary"; button_disabled = False
            button_help_text = "Google Sheets connected. Click to start processing."
        elif current_gsheets_error: st.error(current_gsheets_error)
        start_button_val: bool = st.button(
            "🚀 Start Search & Analysis", type=button_streamlit_type, use_container_width=True,
            disabled=button_disabled, help=button_help_text
        )
        st.markdown("---")
        st.caption("✨ LLM-generated search queries will be automatically used (if LLM is available).")
        st.caption("📄 LLM Summaries for items will be automatically generated (if LLM is available).")
    return keywords_input_val, num_results_wanted_per_keyword, returned_queries, start_button_val

# apply_custom_css remains the same
def apply_custom_css():
    green_button_css = """<style>div[data-testid="stButton"] > button:not(:disabled)[kind="primary"] {background-color: #4CAF50; color: white; border: 1px solid #4CAF50;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:hover {background-color: #45a049; color: white; border: 1px solid #45a049;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:active {background-color: #3e8e41; color: white; border: 1px solid #3e8e41;} div[data-testid="stButton"] > button:disabled[kind="secondary"] {background-color: #f0f2f6; color: rgba(38, 39, 48, 0.4); border: 1px solid rgba(38, 39, 48, 0.2);}</style>"""
    st.markdown(green_button_css, unsafe_allow_html=True)

def display_consolidated_summary():
    if st.session_state.get('consolidated_summary_text'):
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        raw_summary_text = st.session_state.consolidated_summary_text
        
        # Check for specific LLM messages using raw_summary_text BEFORE sanitization
        is_error_message = str(raw_summary_text).lower().startswith("llm_processor: no individual items met")
        
        sanitized_summary_text = sanitize_text_for_markdown(raw_summary_text) # MODIFIED: Sanitize
        
        last_extract_queries = st.session_state.get('last_extract_queries', [""])
        primary_extract_query = last_extract_queries[0] if last_extract_queries else "" 
        
        if primary_extract_query and primary_extract_query.strip() and not is_error_message:
            st.caption(f"Overview focused on insights related to Main Query 1: '{primary_extract_query}'.")
        elif is_error_message: # Display the specific warning/error from LLM Processor
            if primary_extract_query and primary_extract_query.strip():
                 st.warning(f"Could not generate focused overview for Main Query 1 ('{primary_extract_query}'). Reason: {raw_summary_text}")
            else: 
                 st.warning(f"Could not generate overview. Reason: {raw_summary_text}")
        
        with st.container(border=True):
            st.markdown(sanitized_summary_text, unsafe_allow_html=False) # Use sanitized text

def display_individual_results():
    if st.session_state.get('results_data'):
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
        last_extract_queries_for_display = st.session_state.get('last_extract_queries', ["", ""]) 
        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui = item_val_display.get('scraped_title') or item_val_display.get('og_title') or \
                               item_val_display.get('search_title') or "Untitled"
            score_emoji_prefix = get_display_prefix_for_item(item_val_display)
            content_type_marker = "📄" if 'pdf' in item_val_display.get('content_type', '').lower() else ""
            is_llm_keyword_source = item_val_display.get('keyword_searched','').lower() in llm_gen_kws_for_display
            llm_query_marker = "🤖" if is_llm_keyword_source else ""
            full_prefix = score_emoji_prefix
            if score_emoji_prefix and llm_query_marker: full_prefix += " " + llm_query_marker
            elif llm_query_marker: full_prefix = llm_query_marker
            if full_prefix: full_prefix += " "
            expander_title_ui = (f"{full_prefix}{content_type_marker}"
                                 f"{item_val_display.get('keyword_searched','Unknown Keyword')} | "
                                 f"{display_title_ui} ({item_val_display.get('url', 'No URL')})").replace("  ", " ").strip()
            with st.expander(expander_title_ui):
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                st.caption(f"Content Type: {item_val_display.get('content_type', 'N/A')}")
                if item_val_display.get('scraping_error'):
                    st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                with st.container(border=True): # Metadata
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}\n"
                                f"  - **Meta Desc:** {sanitize_text_for_markdown(item_val_display.get('meta_description', 'N/A'))}\n" # Sanitize
                                f"  - **OG Title:** {sanitize_text_for_markdown(item_val_display.get('og_title', 'N/A'))}\n"         # Sanitize
                                f"  - **OG Desc:** {sanitize_text_for_markdown(item_val_display.get('og_description', 'N/A'))}")    # Sanitize
                
                scraped_main_text = item_val_display.get('scraped_main_text', '')
                if scraped_main_text and not str(scraped_main_text).startswith("SCRAPER_INFO:"):
                    with st.popover("View Main Text", use_container_width=True): # Main text is in text_area, less likely to break rendering
                        st.text_area(f"Main Text ({item_val_display.get('content_type')})", value=scraped_main_text,
                                     height=400, key=f"main_text_popover_{i}", disabled=True)
                elif str(scraped_main_text).startswith("SCRAPER_INFO:"): st.caption(scraped_main_text)
                else: st.caption("No main text extracted or usable for LLM processing.")
                
                has_llm_insights = False
                insights_html_parts = ["<div><p><strong>LLM Insights:</strong></p>"] # Build as list then join
                
                raw_llm_summary = item_val_display.get("llm_summary")
                if raw_llm_summary:
                    has_llm_insights = True
                    sanitized_llm_summary = sanitize_text_for_markdown(raw_llm_summary)
                    insights_html_parts.append(f"<div style='border:1px solid #e6e6e6; padding:10px; margin-bottom:10px;'><strong>Summary (LLM):</strong><br>{sanitized_llm_summary}</div>")
                
                for q_idx in range(2): 
                    query_text = last_extract_queries_for_display[q_idx] if q_idx < len(last_extract_queries_for_display) else ""
                    extracted_info_key = f"llm_extracted_info_q{q_idx+1}"
                    raw_extracted_content = item_val_display.get(extracted_info_key)
                    query_label = f"Main Query 1 ('{query_text}')" if q_idx == 0 else f"Additional Query 2 ('{query_text}')"
                    
                    if query_text and query_text.strip() and raw_extracted_content:
                        has_llm_insights = True
                        # For extracted info, which includes "Relevancy Score:", we want to preserve newlines.
                        # The <pre> tag helps with this. Escaping markdown within <pre> might be too aggressive.
                        # Let's try minimal escaping for HTML safety within the <pre> block.
                        html_safe_extracted_content = raw_extracted_content.replace('&', '&').replace('<', '<').replace('>', '>')
                        insights_html_parts.append(f"<div style='border:1px solid #e6e6e6; padding:10px; margin-bottom:10px;'><strong>Extracted Info for {query_label}:</strong><br><pre style='white-space: pre-wrap; word-wrap: break-word;'>{html_safe_extracted_content}</pre></div>")
                
                insights_html_parts.append("</div>")
                if has_llm_insights:
                    st.markdown("".join(insights_html_parts), unsafe_allow_html=True)
                
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

# display_processing_log remains the same
def display_processing_log():
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            st.code("\n".join(st.session_state.processing_log), language=None)

# end of modules/ui_manager.py
