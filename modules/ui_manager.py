# modules/ui_manager.py
# Version 1.0.0: Initial module for UI rendering and management.
"""
Manages the Streamlit User Interface components, including sidebar,
results display, and logging areas.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Set, Tuple
from modules import config # For AppConfig type hint

# --- Helper function for Display Logic (Moved from app.py) ---
def get_display_prefix_for_item(item_data: Dict[str, Any], llm_generated_keywords: Set[str]) -> str:
    """
    Determines the emoji prefix for an item based on its relevancy score
    and whether it was from an LLM-generated keyword.

    Args:
        item_data: Dictionary containing data for the item.
        llm_generated_keywords: A set of lowercased LLM-generated keywords.

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
        is_llm_keyword_result = item_data.get('keyword_searched', '').lower() in llm_generated_keywords
        if score == 5:
            prefix = "5️⃣ "
        elif score == 4:
            prefix = "4️⃣ "
        elif score == 3:
            # Add sparkle if it's from an LLM keyword *and* score is 3
            # prefix = "✨3️⃣ " if is_llm_keyword_result else "3️⃣ "
            # Simpler: always sparkle for LLM keyword if score is 3 or more for now
            # To match original logic, it should only sparkle if score is 3 AND it's LLM.
            # The original logic was:
            # if score == 3: prefix = "✨3️⃣ " if is_llm_keyword else "3️⃣ "
            # Let's refine to ensure is_llm_keyword is clearly identified and used.
            # For now, sticking to simpler prefixes based purely on score,
            # and the LLM keyword origin can be marked separately if needed.
            # The provided code for app.py used:
            # if score == 3: prefix = "✨3️⃣ " if is_llm_keyword else "3️⃣ "
            # So we should try to replicate that.
            # The `is_llm_keyword_result` correctly identifies this.
            if is_llm_keyword_result:
                prefix = f"✨{score}️⃣ " if score >=3 else f"{score}️⃣ " # Sparkle any LLM result if >=3
            else:
                prefix = f"{score}️⃣ "
            # Specific original logic for score 3:
            if score == 3:
                 prefix = "✨3️⃣ " if is_llm_keyword_result else "3️⃣ "


    # If it's an LLM generated keyword result and didn't get a score or a score-based prefix, mark it.
    # This can be done by adding a general marker in render_results_area if needed.
    # For now, the prefix is primarily score-driven.
    return prefix


def render_sidebar(cfg: config.AppConfig) -> Tuple[str, int, str, bool]:
    """
    Renders the sidebar UI components and returns their current values.

    Args:
        cfg: The application configuration object.

    Returns:
        A tuple containing:
            - keywords_input (str): Comma-separated keywords from user.
            - num_results_per_keyword (int): Number of results desired per keyword.
            - llm_extract_query (str): Specific information to extract with LLM.
            - start_button_pressed (bool): True if the start button was pressed.
    """
    with st.sidebar:
        st.subheader("Search Parameters")
        keywords_input: str = st.text_input(
            "Keywords (comma-separated):",
            value=st.session_state.get('last_keywords', ""),
            key="keywords_text_input_main",
            help="Enter comma-separated keywords. Press Enter to apply."
        )
        num_results_per_keyword: int = st.slider(
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

        llm_extract_query: str = st.text_input(
            "Specific info to extract with LLM (guides focused summary):",
            value=st.session_state.get('last_extract_query', ""),
            placeholder="e.g., Key products, contact emails",
            key="llm_extract_text_input",
            help="Enter comma-separated query. Press Enter to apply."
        )

        st.markdown("---")

        # Button logic based on GSheets connection status from session_state
        button_streamlit_type = "secondary"
        button_disabled = True
        button_help_text = st.session_state.get('gsheets_error_message', "Google Sheets connection status undetermined.")

        if st.session_state.get('sheet_writing_enabled', False):
            button_streamlit_type = "primary"
            button_disabled = False
            button_help_text = "Google Sheets connected. Click to start processing."
        elif st.session_state.get('gsheets_error_message'):
            st.error(st.session_state.gsheets_error_message) # Show error if connection failed

        start_button_pressed: bool = st.button(
            "🚀 Start Search & Analysis",
            type=button_streamlit_type,
            use_container_width=True,
            disabled=button_disabled,
            help=button_help_text
        )

        st.markdown("---")
        st.caption("✨ LLM-generated search queries will be automatically used (if LLM is available).")
        st.caption("📄 LLM Summaries for items will be automatically generated (if LLM is available).")

    return keywords_input, num_results_per_keyword, llm_extract_query, start_button_pressed

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

def display_main_content(results_container, log_container):
    """
    Displays the main content area including results and logs.
    This function assumes data is fetched from st.session_state.
    """
    with results_container:
        if st.session_state.get('results_data'): # Check if there's data to show download button
            # Download button is handled by excel_handler and app.py logic directly
            pass # Placeholder if any specific container logic for results overview needed

        # Display Consolidated Summary
        if st.session_state.get('consolidated_summary_text'):
            st.markdown("---")
            st.subheader("✨ Consolidated Overview Result")
            last_extract_query = st.session_state.get('last_extract_query', "")
            consolidated_summary = st.session_state.consolidated_summary_text

            if last_extract_query and last_extract_query.strip() and \
               not str(consolidated_summary).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
                st.caption(f"Overview focused on: '{last_extract_query}'.")
            elif str(consolidated_summary).lower().startswith("llm_processor: no individual items met the minimum relevancy score"):
                st.warning(f"Could not generate focused overview for '{last_extract_query}'.")
            
            with st.container(border=True):
                st.markdown(consolidated_summary)

        # Display Individual Item Results
        if st.session_state.get('results_data'):
            st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
            llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
            initial_kws_for_display = st.session_state.get('initial_keywords_for_display', set())

            for i, item_data in enumerate(st.session_state.results_data):
                display_title_ui: str = item_data.get('scraped_title') or \
                                        item_data.get('og_title') or \
                                        item_data.get('search_title') or \
                                        "Untitled"
                
                # Determine if the keyword for this item was LLM-generated
                keyword_searched_lower = item_data.get('keyword_searched', '').lower()
                is_llm_generated_keyword_item = keyword_searched_lower in llm_gen_kws_for_display and \
                                                keyword_searched_lower not in initial_kws_for_display


                display_prefix = get_display_prefix_for_item(item_data, llm_gen_kws_for_display)
                content_type_marker = "📄" if 'pdf' in item_data.get('content_type', '').lower() else ""
                llm_keyword_marker = "🤖" if is_llm_generated_keyword_item else ""


                expander_title_ui = (
                    f"{display_prefix}{content_type_marker}{llm_keyword_marker}"
                    f"{item_data.get('keyword_searched','Unknown Keyword')} | "
                    f"{display_title_ui} ({item_data.get('url')})"
                )

                with st.expander(expander_title_ui):
                    st.markdown(f"**URL:** [{item_data.get('url')}]({item_data.get('url')})")
                    st.caption(f"Content Type: {item_data.get('content_type', 'N/A')}")
                    if item_data.get('scraping_error'):
                        st.error(f"Scraping Error: {item_data['scraping_error']}")

                    with st.container(border=True):
                        st.markdown("**Scraped Metadata:**")
                        st.markdown(
                            f"  - **Title:** {item_data.get('scraped_title', 'N/A')}\n"
                            f"  - **Meta Desc:** {item_data.get('meta_description', 'N/A')}\n"
                            f"  - **OG Title:** {item_data.get('og_title', 'N/A')}\n"
                            f"  - **OG Desc:** {item_data.get('og_description', 'N/A')}"
                        )

                    main_text = item_data.get('scraped_main_text', '')
                    if main_text and not str(main_text).startswith("SCRAPER_INFO:"):
                        with st.popover("View Main Text", use_container_width=True):
                            st.text_area(
                                f"Main Text ({item_data.get('content_type')})",
                                value=main_text,
                                height=400,
                                key=f"main_text_popover_{i}",
                                disabled=True
                            )
                    elif str(main_text).startswith("SCRAPER_INFO:"):
                        st.caption(main_text)
                    else:
                        st.caption("No main text extracted or usable for LLM processing.")

                    if item_data.get("llm_summary") or item_data.get("llm_extracted_info"):
                        st.markdown("**LLM Insights:**")
                        if item_data.get("llm_summary"):
                            with st.container(border=True):
                                st.markdown(f"**Summary (LLM):**")
                                st.markdown(item_data["llm_summary"])
                        if item_data.get("llm_extracted_info"):
                            with st.container(border=True):
                                st.markdown(f"**Extracted Info (LLM) for '{st.session_state.get('last_extract_query', 'N/A')}':**")
                                st.text(item_data["llm_extracted_info"]) # Using st.text for preformatted relevancy score
                    
                    st.caption(f"Item Timestamp: {item_data.get('timestamp')}")

    with log_container:
        if st.session_state.get('processing_log'):
            with st.expander("📜 View Processing Log", expanded=False):
                st.code("\n".join(st.session_state.processing_log), language=None)

# end of modules/ui_manager.py
