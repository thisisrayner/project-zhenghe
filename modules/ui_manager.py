# modules/ui_manager.py
# Version 1.1.2:
# 1. Number emoji reflects highest score from Q1 or Q2.
# 2. Removed ✨ emoji; 🤖 solely indicates LLM-generated keyword.
# 3. Ensured 🤖 appears after the score emoji (e.g., 5️⃣ 🤖).
"""
Manages the Streamlit User Interface elements, layout, and user inputs.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
from modules import config # For AppConfig type hint

def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]:
    """Helper function to parse relevancy score from an LLM extraction string."""
    score: Optional[int] = None
    if extracted_info and isinstance(extracted_info, str) and extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError):
            pass # Score remains None
    return score

# --- Helper function for Display Logic ---
def get_display_prefix_for_item(item_data: Dict[str, Any]) -> str: # Removed llm_generated_keywords_set_for_display as arg
    """
    Determines the number emoji prefix for an item based on the HIGHEST relevancy score
    from either Extraction Query 1 or Extraction Query 2.
    Returns only the number emoji (e.g., "5️⃣ ") or an empty string.
    """
    prefix = ""
    
    score_q1 = _parse_score_from_extraction(item_data.get("llm_extracted_info_q1"))
    score_q2 = _parse_score_from_extraction(item_data.get("llm_extracted_info_q2"))

    highest_score: Optional[int] = None
    if score_q1 is not None and score_q2 is not None:
        highest_score = max(score_q1, score_q2)
    elif score_q1 is not None:
        highest_score = score_q1
    elif score_q2 is not None:
        highest_score = score_q2

    if highest_score is not None:
        if highest_score == 5: prefix = "5️⃣" # No trailing space here, will be added later if needed
        elif highest_score == 4: prefix = "4️⃣"
        elif highest_score == 3: prefix = "3️⃣"
        # Scores below 3 do not get a prefix
            
    return prefix


def render_sidebar(cfg: config.AppConfig, current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]:
    # ... (render_sidebar remains the same as v1.1.1) ...
    with st.sidebar:
        st.subheader("Search Parameters")
        keywords_input_val: str = st.text_input(
            "Keywords (comma-separated):",
            value=st.session_state.get('last_keywords', ""),
            key="keywords_text_input_main",
            help="Enter comma-separated keywords. Press Enter to apply."
        )
        num_results_wanted_per_keyword: int = st.slider(
            "Number of successfully scraped results per keyword:",
            1, 10, cfg.num_results_per_keyword_default,
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
        tooltip_text = "You can type in keywords to focus the analysis or ask a question."

        llm_extract_query_1_input_val: str = st.text_input(
            "Extraction Query 1 (drives focused summary if its score is high):", # Clarified label
            value=last_extract_queries[0],
            placeholder="e.g., Key methodologies, primary conclusions",
            key="llm_extract_q1_input",
            help=tooltip_text
        )
        llm_extract_query_2_input_val: str = st.text_input(
            "Extraction Query 2 (optional additional extraction):",
            value=last_extract_queries[1],
            placeholder="e.g., Mentioned limitations, future work",
            key="llm_extract_q2_input",
            help=tooltip_text
        )
        
        llm_extract_queries_input_vals: List[str] = [] # This is not directly used for return but for internal logic if needed
        if llm_extract_query_1_input_val.strip():
            llm_extract_queries_input_vals.append(llm_extract_query_1_input_val.strip())
        if llm_extract_query_2_input_val.strip():
            llm_extract_queries_input_vals.append(llm_extract_query_2_input_val.strip())
        returned_queries = [llm_extract_query_1_input_val, llm_extract_query_2_input_val] # Return typed values for session state
        
        st.markdown("---")
        button_streamlit_type = "secondary"
        button_disabled = True
        button_help_text = current_gsheets_error or "Google Sheets connection status undetermined."
        if sheet_writing_enabled:
            button_streamlit_type = "primary"
            button_disabled = False
            button_help_text = "Google Sheets connected. Click to start processing."
        elif current_gsheets_error:
            st.error(current_gsheets_error)
        start_button_val: bool = st.button(
            "🚀 Start Search & Analysis",
            type=button_streamlit_type,
            use_container_width=True,
            disabled=button_disabled,
            help=button_help_text
        )
        st.markdown("---")
        st.caption("✨ LLM-generated search queries will be automatically used (if LLM is available).")
        st.caption("📄 LLM Summaries for items will be automatically generated (if LLM is available).")

    return keywords_input_val, num_results_wanted_per_keyword, returned_queries, start_button_val


def apply_custom_css():
    # ... (remains the same as v1.1.1) ...
    green_button_css = """
    <style>
    div[data-testid="stButton"] > button:not(:disabled)[kind="primary"] {
        background-color: #4CAF50; color: white; border: 1px solid #4CAF50;
    }
    div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:hover {
        background-color: #45a049; color: white; border: 1px solid #45a049;
    }
    div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:active {
        background-color: #3e8e41; color: white; border: 1px solid #3e8e41;
    }
    div[data-testid="stButton"] > button:disabled[kind="secondary"] {
        background-color: #f0f2f6; color: rgba(38, 39, 48, 0.4); border: 1px solid rgba(38, 39, 48, 0.2);
    }
    </style>
    """
    st.markdown(green_button_css, unsafe_allow_html=True)


def display_consolidated_summary():
    # ... (remains the same as v1.1.1 - primary query (q1) still drives focus note) ...
    if st.session_state.get('consolidated_summary_text'):
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        summary_text = st.session_state.consolidated_summary_text
        last_extract_queries = st.session_state.get('last_extract_queries', [""])
        primary_extract_query = last_extract_queries[0] if last_extract_queries else "" # Q1 is the primary for focus
        if primary_extract_query and primary_extract_query.strip() and \
           not str(summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
            st.caption(f"Overview focused on insights related to Q1: '{primary_extract_query}'.")
        elif str(summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
            if primary_extract_query and primary_extract_query.strip():
                 st.warning(f"Could not generate focused overview for Q1 ('{primary_extract_query}'). No items met minimum relevancy for Q1.")
            else: 
                 st.warning(f"Could not generate focused overview. No items met minimum relevancy (Q1).")
        with st.container(border=True):
            st.markdown(summary_text)

def display_individual_results():
    """Displays individual processed items in expanders if available in session state."""
    if st.session_state.get('results_data'):
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        
        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
        last_extract_queries_for_display = st.session_state.get('last_extract_queries', ["", ""])

        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui: str = item_val_display.get('scraped_title') or \
                                    item_val_display.get('og_title') or \
                                    item_val_display.get('search_title') or \
                                    "Untitled"
            
            score_emoji_prefix = get_display_prefix_for_item(item_val_display) # Gets highest score emoji (e.g., "5️⃣")
            
            content_type_marker = "📄" if 'pdf' in item_val_display.get('content_type', '').lower() else ""
            
            is_llm_keyword_source = item_val_display.get('keyword_searched','').lower() in llm_gen_kws_for_display
            llm_query_marker = "🤖" if is_llm_keyword_source else ""

            # Construct the full prefix: Score [space if score exists] Robot [space if robot exists and score does not]
            full_prefix = score_emoji_prefix
            if score_emoji_prefix and llm_query_marker:
                full_prefix += " " + llm_query_marker # e.g., "5️⃣ 🤖"
            elif llm_query_marker: # Only robot, no score emoji
                full_prefix = llm_query_marker # e.g., "🤖"
            # If only score_emoji_prefix, it's already set. If neither, full_prefix is "".

            # Add a space after the emojis if there are any, before content_type_marker
            if full_prefix:
                full_prefix += " "


            expander_title_ui = (
                f"{full_prefix}{content_type_marker}"
                f"{item_val_display.get('keyword_searched','Unknown Keyword')} | "
                f"{display_title_ui} ({item_val_display.get('url', 'No URL')})"
            )
            # Remove potential double spaces if content_type_marker is empty but full_prefix had a trailing space
            expander_title_ui = expander_title_ui.replace("  ", " ").strip()


            with st.expander(expander_title_ui):
                # ... (rest of individual item display logic remains same as v1.1.1) ...
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                st.caption(f"Content Type: {item_val_display.get('content_type', 'N/A')}")
                if item_val_display.get('scraping_error'):
                    st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                with st.container(border=True):
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(
                        f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}\n"
                        f"  - **Meta Desc:** {item_val_display.get('meta_description', 'N/A')}\n"
                        f"  - **OG Title:** {item_val_display.get('og_title', 'N/A')}\n"
                        f"  - **OG Desc:** {item_val_display.get('og_description', 'N/A')}"
                    )
                scraped_main_text = item_val_display.get('scraped_main_text', '')
                if scraped_main_text and not str(scraped_main_text).startswith("SCRAPER_INFO:"):
                    with st.popover("View Main Text", use_container_width=True):
                        st.text_area(
                            f"Main Text ({item_val_display.get('content_type')})",
                            value=scraped_main_text,
                            height=400,
                            key=f"main_text_popover_{i}",
                            disabled=True
                        )
                elif str(scraped_main_text).startswith("SCRAPER_INFO:"):
                    st.caption(scraped_main_text)
                else:
                    st.caption("No main text extracted or usable for LLM processing.")
                has_llm_insights = False
                insights_html = "<div><p><strong>LLM Insights:</strong></p>"
                if item_val_display.get("llm_summary"):
                    has_llm_insights = True
                    insights_html += f"<div style='border:1px solid #e6e6e6; padding:10px; margin-bottom:10px;'><strong>Summary (LLM):</strong><br>{item_val_display['llm_summary']}</div>"
                for q_idx in range(2): 
                    query_text = last_extract_queries_for_display[q_idx] if q_idx < len(last_extract_queries_for_display) else ""
                    extracted_info_key = f"llm_extracted_info_q{q_idx+1}"
                    if query_text and query_text.strip() and item_val_display.get(extracted_info_key):
                        has_llm_insights = True
                        extracted_content = item_val_display[extracted_info_key]
                        insights_html += f"<div style='border:1px solid #e6e6e6; padding:10px; margin-bottom:10px;'><strong>Extracted Info for Q{q_idx+1} ('{query_text}'):</strong><br><pre style='white-space: pre-wrap; word-wrap: break-word;'>{extracted_content}</pre></div>"
                insights_html += "</div>"
                if has_llm_insights:
                    st.markdown(insights_html, unsafe_allow_html=True)
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")


def display_processing_log():
    # ... (remains the same as v1.1.1) ...
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            st.code("\n".join(st.session_state.processing_log), language=None)

# end of modules/ui_manager.py
