# modules/ui_manager.py
# Version 1.1.6:
# - Removed sanitize_text_for_markdown from the main consolidated summary display
#   to allow TL;DR list formatting (newlines and dashes) from llm_processor to render correctly.
# - Sanitization is still applied to pure info/error messages from the LLM.
# Version 1.1.5:
# - Renamed display_consolidated_summary to display_consolidated_summary_and_sources.
# - Added display of sources for focused consolidated summaries if available.
# - Function now accepts summary_text, focused_sources, and last_extract_queries as args.
"""
Manages the Streamlit User Interface elements, layout, and user inputs.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
from modules import config
import re

# --- Sanitization Helper ---
def sanitize_text_for_markdown(text: Optional[str]) -> str:
    """
    Sanitizes text to prevent common markdown rendering issues, especially from LLM output.
    Escapes markdown special characters. Intended for text where no markdown is desired.
    """
    if text is None:
        return ""
    escaped_text = text.replace('\\', '\\\\')
    # Characters that have special meaning in Markdown
    markdown_chars_to_escape = r"([`*_#{}\[\]()+.!-])"
    escaped_text = re.sub(markdown_chars_to_escape, r"\\\1", escaped_text)
    # Handle specific cases like three or more hyphens/asterisks/underscores
    escaped_text = re.sub(r"---", r"\-\-\-", escaped_text)
    escaped_text = re.sub(r"\*\*\*", r"\*\*\*", escaped_text)
    escaped_text = re.sub(r"___", r"\_\_\_", escaped_text)
    return escaped_text

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

def render_sidebar(cfg: 'config.AppConfig', current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]:
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
        last_extract_queries_val = st.session_state.get('last_extract_queries', ["", ""])
        if not isinstance(last_extract_queries_val, list) or len(last_extract_queries_val) < 2:
            last_extract_queries_val = ["", ""]
        tooltip_text_q1 = "You can type in keywords to focus the analysis or ask a question. Main Query 1 drives the primary relevancy score and focused consolidated summary."
        tooltip_text_q2 = "Optional: ask a different question or look for other specific keywords."
        llm_extract_query_1_input_val: str = st.text_input(
            "Main Query 1:", value=last_extract_queries_val[0],
            placeholder="e.g., Key methodologies, primary conclusions", key="llm_extract_q1_input", help=tooltip_text_q1
        )
        llm_extract_query_2_input_val: str = st.text_input(
            "Additional Query 2:", value=last_extract_queries_val[1],
            placeholder="e.g., Mentioned limitations, future work", key="llm_extract_q2_input", help=tooltip_text_q2
        )
        returned_queries = [llm_extract_query_1_input_val, llm_extract_query_2_input_val]
        st.markdown("---")
        button_streamlit_type = "secondary"; button_disabled = True
        button_help_text = current_gsheets_error or "Google Sheets connection status undetermined."
        if sheet_writing_enabled:
            button_streamlit_type = "primary"; button_disabled = False
            button_help_text = "Google Sheets connected. Click to start processing."
        start_button_val: bool = st.button(
            "🚀 Start Search & Analysis", type=button_streamlit_type, use_container_width=True,
            disabled=button_disabled, help=button_help_text
        )
        st.markdown("---")
        st.caption("✨ LLM-generated search queries will be automatically used (if LLM is available).")
        st.caption("📄 LLM Summaries for items will be automatically generated (if LLM is available).")
    return keywords_input_val, num_results_wanted_per_keyword, returned_queries, start_button_val

def apply_custom_css():
    green_button_css = """<style>div[data-testid="stButton"] > button:not(:disabled)[kind="primary"] {background-color: #4CAF50; color: white; border: 1px solid #4CAF50;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:hover {background-color: #45a049; color: white; border: 1px solid #45a049;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:active {background-color: #3e8e41; color: white; border: 1px solid #3e8e41;} div[data-testid="stButton"] > button:disabled[kind="secondary"] {background-color: #f0f2f6; color: rgba(38, 39, 48, 0.4); border: 1px solid rgba(38, 39, 48, 0.2);}</style>"""
    st.markdown(green_button_css, unsafe_allow_html=True)

def display_consolidated_summary_and_sources(
    summary_text: Optional[str],
    focused_sources: Optional[List[Dict[str, Any]]],
    last_extract_queries: List[str]
) -> None:
    """
    Displays the consolidated summary. If focused, also shows sources.
    Allows intended Markdown (like TLDR lists) from summary_text to render.
    """
    if summary_text:
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        
        is_error_message = "LLM_PROCESSOR_ERROR:" in summary_text
        is_info_only_summary = "LLM_PROCESSOR_INFO:" in summary_text and not is_error_message
        
        primary_extract_query = last_extract_queries[0] if last_extract_queries and len(last_extract_queries) > 0 else ""
        was_focused_attempt = bool(any(q.strip() for q in last_extract_queries))

        # Determine the text to display and if it should be sanitized
        text_to_display = summary_text
        
        if is_error_message:
            st.error(f"Consolidated Overview Generation Error: {sanitize_text_for_markdown(summary_text)}")
            return 
        elif is_info_only_summary:
            st.info(sanitize_text_for_markdown(summary_text)) # Sanitize simple info messages
            if "General overview as follows" in summary_text:
                was_focused_attempt = False # Don't show "sources" for general overview
        else: # This is a successfully generated narrative + TLDR
            if was_focused_attempt: # and not error/info
                if primary_extract_query and primary_extract_query.strip():
                    st.caption(f"This overview is focused on insights related to Main Query 1: '{primary_extract_query}'.")
            # For successfully generated content (narrative + TLDR), display WITHOUT SANITIZATION
            # to allow Markdown lists from TLDR to render.
            # text_to_display is already summary_text

        # Display the main content (potentially with Markdown if not error/info)
        with st.container(border=True):
            st.markdown(text_to_display, unsafe_allow_html=False) # Key change: use text_to_display

        # Display sources if it was a focused attempt that succeeded and produced sources
        if was_focused_attempt and focused_sources and not is_error_message and not is_info_only_summary:
            with st.expander("ℹ️ View Sources for Focused Consolidated Overview", expanded=False):
                st.markdown(f"This focused overview was synthesized from **{len(focused_sources)}** high-scoring (>=3/5) extractions:")
                for i, source in enumerate(focused_sources):
                    url = source.get('url', 'N/A')
                    query_type = source.get('query_type', 'N/A')
                    query_text_short = source.get('query_text', 'N/A')
                    if len(query_text_short) > 40: query_text_short = query_text_short[:37] + "..."
                    score = source.get('score', 'N/A')
                    st.markdown(
                        f"  {i+1}. **URL:** [{url}]({url})\n"
                        f"     - **Source Query:** {query_type} (\"{query_text_short}\")\n"
                        f"     - **Relevancy Score for this item:** {score}/5"
                    )
                st.caption("The LLM was instructed to synthesize these snippets with a primary focus on insights related to Main Query 1 (if provided).")
    
    elif 'last_keywords' in st.session_state and st.session_state.last_keywords:
         st.info("Consolidated overview is not available for this run (e.g., no items processed or LLM disabled).")

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
                with st.container(border=True): 
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}\n"
                                f"  - **Meta Desc:** {sanitize_text_for_markdown(item_val_display.get('meta_description', 'N/A'))}\n" 
                                f"  - **OG Title:** {sanitize_text_for_markdown(item_val_display.get('og_title', 'N/A'))}\n"         
                                f"  - **OG Desc:** {sanitize_text_for_markdown(item_val_display.get('og_description', 'N/A'))}")    
                
                scraped_main_text = item_val_display.get('scraped_main_text', '')
                if scraped_main_text and not str(scraped_main_text).startswith("SCRAPER_INFO:"):
                    with st.popover("View Main Text", use_container_width=True): 
                        st.text_area(f"Main Text ({item_val_display.get('content_type')})", value=scraped_main_text,
                                     height=400, key=f"main_text_popover_{i}", disabled=True)
                elif str(scraped_main_text).startswith("SCRAPER_INFO:"): st.caption(scraped_main_text)
                else: st.caption("No main text extracted or usable for LLM processing.")
                
                has_llm_insights = False
                insights_html_parts = ["<div><p><strong>LLM Insights:</strong></p>"] 
                
                raw_llm_summary = item_val_display.get("llm_summary")
                if raw_llm_summary: # Individual summaries are always sanitized
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
                        # Extracted info (score + text) is complex. Score line should be as-is. Content part is plain text.
                        # Using <pre> handles newlines in the plain text content part well.
                        html_safe_extracted_content = raw_extracted_content.replace('&', '&').replace('<', '<').replace('>', '>')
                        insights_html_parts.append(f"<div style='border:1px solid #e6e6e6; padding:10px; margin-bottom:10px;'><strong>Extracted Info for {query_label}:</strong><br><pre style='white-space: pre-wrap; word-wrap: break-word;'>{html_safe_extracted_content}</pre></div>")
                
                insights_html_parts.append("</div>")
                if has_llm_insights:
                    st.markdown("".join(insights_html_parts), unsafe_allow_html=True)
                
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

def display_processing_log():
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            st.code("\n".join(st.session_state.processing_log), language=None)

# end of modules/ui_manager.py
