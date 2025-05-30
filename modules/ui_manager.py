# modules/ui_manager.py
# Version 1.0.0: Initial module for UI rendering and interaction logic.
# Version 1.0.1: Corrected `get_display_prefix_for_item` to handle `llm_generated_keywords` correctly.
"""
Manages the Streamlit User Interface elements, layout, and user inputs.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
from modules import config # For AppConfig type hint

# --- Helper function for Display Logic (moved from app.py) ---
def get_display_prefix_for_item(item_data: Dict[str, Any], llm_generated_keywords_set_for_display: Set[str]) -> str:
    """
    Determines the emoji prefix for an item based on its relevancy score and
    whether it was found using an LLM-generated keyword.

    Args:
        item_data: Dictionary containing data for a single processed item.
        llm_generated_keywords_set_for_display: A set of lowercased LLM-generated keywords.

    Returns:
        A string prefix (e.g., "5️⃣ ", "✨3️⃣ ").
    """
    prefix = ""
    llm_extracted_info = item_data.get("llm_extracted_info")
    score: Optional[int] = None

    if llm_extracted_info and isinstance(llm_extracted_info, str) and llm_extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = llm_extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError):
            score = None # Reset score if parsing fails

    if score is not None:
        # Ensure keyword_searched is present and is a string before lowercasing
        keyword_searched_lower = item_data.get('keyword_searched', '').lower() if isinstance(item_data.get('keyword_searched'), str) else ''

        is_from_llm_keyword = keyword_searched_lower in llm_generated_keywords_set_for_display

        if score == 5:
            prefix = "5️⃣ "
        elif score == 4:
            prefix = "4️⃣ "
        elif score == 3:
            if is_from_llm_keyword :
                 prefix = "✨3️⃣ " # Special marker for LLM keyword and score 3
            else:
                 prefix = "3️⃣ "
        # Scores below 3 do not get a prefix in this version
    return prefix


def render_sidebar(cfg: config.AppConfig, current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, str, bool]:
    """
    Renders the sidebar UI elements and returns user inputs.

    Args:
        cfg: The application configuration.
        current_gsheets_error: Any current error message related to Google Sheets.
        sheet_writing_enabled: Boolean indicating if Google Sheets is connected and writable.

    Returns:
        A tuple containing:
        - keywords_input_val (str): Comma-separated keywords from user.
        - num_results_wanted_per_keyword (int): Number of results desired.
        - llm_extract_query_input_val (str): Specific info extraction query.
        - start_button_val (bool): True if the start button was pressed.
    """
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

        llm_extract_query_input_val: str = st.text_input(
            "Specific info to extract with LLM (guides focused summary):",
            value=st.session_state.get('last_extract_query', ""),
            placeholder="e.g., Key products, contact emails",
            key="llm_extract_text_input",
            help="Enter comma-separated. Press Enter to apply."
        )
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

    return keywords_input_val, num_results_wanted_per_keyword, llm_extract_query_input_val, start_button_val

def apply_custom_css():
    """Applies custom CSS for styling elements like the primary button."""
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
    """Displays the consolidated summary if available in session state."""
    if st.session_state.get('consolidated_summary_text'):
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        summary_text = st.session_state.consolidated_summary_text
        last_extract_query = st.session_state.get('last_extract_query', "")

        if last_extract_query and last_extract_query.strip() and \
           not str(summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
            st.caption(f"Overview focused on: '{last_extract_query}'.")
        elif str(summary_text).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
            st.warning(f"Could not generate focused overview for '{last_extract_query}'. No items met minimum relevancy.")
        
        with st.container(border=True):
            st.markdown(summary_text)

def display_individual_results():
    """Displays individual processed items in expanders if available in session state."""
    if st.session_state.get('results_data'):
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        
        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())

        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui: str = item_val_display.get('scraped_title') or \
                                    item_val_display.get('og_title') or \
                                    item_val_display.get('search_title') or \
                                    "Untitled"
            
            display_prefix = get_display_prefix_for_item(item_val_display, llm_gen_kws_for_display)
            content_type_marker = "📄" if 'pdf' in item_val_display.get('content_type', '').lower() else ""
            llm_query_marker = "🤖" if item_val_display.get('keyword_searched','').lower() in llm_gen_kws_for_display else ""


            expander_title_ui = (
                f"{display_prefix}{content_type_marker}{llm_query_marker}"
                f"{item_val_display.get('keyword_searched','Unknown Keyword')} | "
                f"{display_title_ui} ({item_val_display.get('url', 'No URL')})"
            )

            with st.expander(expander_title_ui):
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

                if item_val_display.get("llm_summary") or item_val_display.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val_display.get("llm_summary"):
                        with st.container(border=True):
                            st.markdown(f"**Summary (LLM):**")
                            st.markdown(item_val_display["llm_summary"])
                    if item_val_display.get("llm_extracted_info"):
                        with st.container(border=True):
                            st.markdown(f"**Extracted Info (LLM) for '{st.session_state.get('last_extract_query','Not specified')}':**")
                            st.text(item_val_display["llm_extracted_info"])
                
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

def display_processing_log():
    """Displays the processing log in an expander if available in session state."""
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            st.code("\n".join(st.session_state.processing_log), language=None)

# end of modules/ui_manager.py
