# modules/ui_manager.py
# Version 1.1.12:
# - Added _PROCESSING_SPINNER_MESSAGES list.
# - Added get_random_spinner_message() function to select a random message.
# Previous versions:
# - Version 1.1.11: Reads 'llm_globally_enabled' from st.session_state.

"""
Manages the Streamlit User Interface elements, layout, and user inputs for D.O.R.A.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Set
import re
import html 
import random # Import the random module

# --- Constants ---
_PROCESSING_SPINNER_MESSAGES: Tuple[str, ...] = (
    "Perfect time to update that ‘Methodology’ slide you always leave until the last minute. Yes, we see you.",
    "Analyst hack: If you reorder your pivot table columns right now, you’ll still hate them tomorrow—but at least you tried.",
    "Peer review tip: Triple-checking your citations while you wait guarantees your reviewer will find the one you missed.",
    "Fun fact: 87% of whitepapers are written during progress bars. The other 13% are written during coffee breaks.",
    "While D.O.R.A. loads, you could draft a ‘Limitations’ section—don’t worry, nobody reads it anyway.",
    "Industry secret: True experts always Google their own insights while waiting for results (we won’t tell).",
    "If you’re waiting, might as well start a new ‘Version History’ file. Because if you don’t, your boss will ask for one anyway.",
    "Consultant’s Law: For every minute you wait, someone, somewhere, is adding a new “Key Takeaway” to your deck.",
    "Procrastination paradox: The more you refresh this page, the less time you spend on the executive summary—and that’s where all the glory goes.",
    "Here’s a stat: A survey found that people who check their posture every hour report a 19% drop in neck pain.",
    "It’s proven: Taking 5 deep breaths while your results load can boost decision-making accuracy by up to 11%.",
    "Stanford scientists say listening to upbeat music while waiting can enhance creativity by 13%. Queue your favorite playlist!",
    "Engineers agree: Reviewing your to-do list now can help reduce end-of-day stress by nearly 15%.",
    "Fun insight: Users who doodle on sticky notes during wait times recall information 16% better, says a Cambridge study.",
    "Research indicates: Sending a thank-you message to a colleague while waiting can improve workplace happiness scores by 9%.",
    "Power user move: Set your PowerPoint animations to ‘Fade Fast’ while you wait. The client won’t notice, but you’ll sleep better.",
    "Quick fact: The average person blinks 15 times a minute, but staring at screens can drop this to 5. Give your eyes a blink boost!",
    "Consulting legend: The limitations section is always longer when you write it while waiting for data.",
    "According to MIT, sipping water during tasks like this can improve cognitive function by 8%. Hydrate while you wait!",
    "Try this: Typing out one sentence with your non-dominant hand can sharpen neural connections, according to neuroscientists."
)

# --- New Function ---
def get_random_spinner_message() -> str:
    """
    Returns a randomly selected processing message for the st.spinner.
    """
    return random.choice(_PROCESSING_SPINNER_MESSAGES)

# --- Existing Functions (Unchanged from v1.1.11) ---
def sanitize_text_for_markdown(text: Optional[str]) -> str:
    # ... (as in v1.1.11) ...
    if text is None: return ""
    escaped_text = text.replace('\\', '\\\\')
    markdown_chars_to_escape = r"([`*_#{}\[\]()+.!-])" 
    escaped_text = re.sub(markdown_chars_to_escape, r"\\\1", escaped_text)
    escaped_text = re.sub(r"---", r"\-\-\-", escaped_text) 
    escaped_text = re.sub(r"\*\*\*", r"\*\*\*", escaped_text) 
    escaped_text = re.sub(r"___", r"\_\_\_", escaped_text) 
    return escaped_text

def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]:
    # ... (as in v1.1.11) ...
    score: Optional[int] = None
    if extracted_info and isinstance(extracted_info, str) and extracted_info.startswith("Relevancy Score: "):
        try:
            score_line = extracted_info.split('\n', 1)[0]
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
        except (IndexError, ValueError): pass
    return score

def get_display_prefix_for_item(item_data: Dict[str, Any]) -> str:
    # ... (as in v1.1.11) ...
    prefix = ""
    score_q1 = item_data.get("llm_relevancy_score_q1") 
    score_q2 = item_data.get("llm_relevancy_score_q2")
    highest_score: Optional[int] = None
    if isinstance(score_q1, (int, float)): score_q1 = int(score_q1)
    else: score_q1 = None
    if isinstance(score_q2, (int, float)): score_q2 = int(score_q2)
    else: score_q2 = None
    if score_q1 is not None and score_q2 is not None: highest_score = max(score_q1, score_q2)
    elif score_q1 is not None: highest_score = score_q1
    elif score_q2 is not None: highest_score = score_q2
    if highest_score is not None:
        if highest_score >= 5: prefix = "5️⃣"
        elif highest_score == 4: prefix = "4️⃣"
        elif highest_score == 3: prefix = "3️⃣"
    return prefix

def render_sidebar(cfg: 'config.AppConfig', current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]:
    # ... (as in v1.1.11 - ensure 'config.AppConfig' type hint is valid) ...
    with st.sidebar:
        st.subheader("Search Parameters")
        keywords_input_val: str = st.text_input("Keywords (comma-separated):", value=st.session_state.get('last_keywords', ""), key="keywords_text_input_main_sidebar_v1112", help="Enter comma-separated keywords. Press Enter to apply.")
        default_slider_val = getattr(cfg, 'num_results_per_keyword_default', 3) if cfg else 3
        num_results_wanted_per_keyword: int = st.slider("Number of successfully scraped results per keyword:", 1, 10, default_slider_val, key="num_results_slider_sidebar_v1112")
        llm_is_enabled_globally = st.session_state.get('llm_globally_enabled', False)
        llm_provider_display = "N/A"; model_display_name: str = "N/A"
        if cfg and hasattr(cfg, 'llm'):
            llm_provider_display = cfg.llm.provider.upper()
            st.subheader(f"LLM Processing - Provider: {llm_provider_display}")
            if llm_is_enabled_globally:
                model_display_name = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                st.caption(f"Using Model: {model_display_name}")
            else: st.caption(f"API Key for {cfg.llm.provider.upper()} not configured or LLM disabled globally.")
        else: 
            st.subheader(f"LLM Processing - Provider: {llm_provider_display}")
            st.caption("LLM Configuration not fully loaded. LLM features likely disabled.")
        st.markdown("**Specific Info to Extract (LLM):**")
        last_extract_queries_val = st.session_state.get('last_extract_queries', ["", ""])
        if not isinstance(last_extract_queries_val, list) or len(last_extract_queries_val) < 2: last_extract_queries_val = ["", ""]
        tooltip_text_q1 = "Main Query 1 drives primary relevancy and focused summary."
        tooltip_text_q2 = "Optional: secondary question or keywords."
        llm_extract_query_1_input_val: str = st.text_input("Main Query 1:", value=last_extract_queries_val[0], placeholder="e.g., Key methodologies", key="llm_extract_q1_input_sidebar_v1112", help=tooltip_text_q1)
        llm_extract_query_2_input_val: str = st.text_input("Additional Query 2:", value=last_extract_queries_val[1], placeholder="e.g., Mentioned limitations", key="llm_extract_q2_input_sidebar_v1112", help=tooltip_text_q2)
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

def apply_custom_css():
    # ... (as in v1.1.11) ...
    green_button_css = """<style>div[data-testid="stButton"] > button:not(:disabled)[kind="primary"] {background-color: #4CAF50; color: white; border: 1px solid #4CAF50;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:hover {background-color: #45a049; color: white; border: 1px solid #45a049;} div[data-testid="stButton"] > button:not(:disabled)[kind="primary"]:active {background-color: #3e8e41; color: white; border: 1px solid #3e8e41;} div[data-testid="stButton"] > button:disabled[kind="secondary"] {background-color: #f0f2f6; color: rgba(38, 39, 48, 0.4); border: 1px solid rgba(38, 39, 48, 0.2);}</style>"""
    st.markdown(green_button_css, unsafe_allow_html=True)

def display_consolidated_summary_and_sources(summary_text: Optional[str], focused_sources: Optional[List[Dict[str, Any]]], last_extract_queries: List[str]) -> None:
    # ... (as in v1.1.11) ...
    if summary_text:
        st.markdown("---"); st.subheader("✨ Consolidated Overview Result")
        is_error_message = "LLM_PROCESSOR_ERROR:" in summary_text
        is_info_only_summary = "LLM_PROCESSOR_INFO:" in summary_text and not is_error_message
        primary_extract_query = last_extract_queries[0] if last_extract_queries and len(last_extract_queries) > 0 else ""
        was_focused_attempt = bool(any(q.strip() for q in last_extract_queries))
        if is_error_message: st.error(summary_text); return 
        elif is_info_only_summary: st.info(summary_text)
        else: 
            if was_focused_attempt and primary_extract_query and primary_extract_query.strip(): st.caption(f"This overview is focused on Main Query 1: '{primary_extract_query}'.")
            with st.container(border=True): st.markdown(summary_text, unsafe_allow_html=False)
        if was_focused_attempt and focused_sources and not is_error_message and not is_info_only_summary:
            with st.expander("ℹ️ View Sources for Focused Consolidated Overview", expanded=False):
                st.markdown(f"This focused overview was synthesized from **{len(focused_sources)}** high-scoring (>=3/5) extractions:")
                for i, source in enumerate(focused_sources):
                    url = source.get('url', 'N/A'); query_type = source.get('query_type', 'N/A')
                    query_text_short = source.get('query_text', 'N/A'); score_val = source.get('score', 'N/A')
                    if len(query_text_short) > 40: query_text_short = query_text_short[:37] + "..."
                    st.markdown(f"  {i+1}. **URL:** [{url}]({url})\n     - **Source Query:** {query_type} (\"{query_text_short}\")\n     - **Relevancy Score for this item:** {score_val}/5")
                st.caption("The LLM was instructed to synthesize these snippets with a primary focus on Main Query 1 (if provided).")
    elif 'last_keywords' in st.session_state and st.session_state.last_keywords:
         st.info("Consolidated overview is not available for this run.")


def display_individual_results():
    # ... (as in v1.1.11, including the corrected caption logic) ...
    if st.session_state.get('results_data'):
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        llm_globally_enabled = st.session_state.get('llm_globally_enabled', False)
        llm_gen_kws_for_display = st.session_state.get('llm_generated_keywords_set_for_display', set())
        last_extract_queries_for_display = st.session_state.get('last_extract_queries', ["", ""]) 
        for i, item_val_display in enumerate(st.session_state.results_data):
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
                        st.text_area(f"Main Text ({item_val_display.get('content_type')})", value=scraped_main_text, height=400, key=f"main_text_popover_{i}_v1112", disabled=True) # Unique key
                elif str(scraped_main_text).startswith("SCRAPER_INFO:"): st.caption(scraped_main_text)
                else: st.caption("No main text extracted or usable for LLM processing.")
                has_llm_insights = False 
                insights_container = st.container(border=True)
                raw_llm_summary = item_val_display.get("llm_summary")
                if raw_llm_summary and not str(raw_llm_summary).lower().startswith(("llm error", "llm_processor", "no text content")): 
                    has_llm_insights = True; insights_container.markdown("**Summary (LLM):**"); insights_container.markdown(raw_llm_summary)
                for q_idx in range(2): 
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
                        if score_value is not None: insights_container.markdown(f"    *Relevancy Score for this item: {score_value}/5*")
                        insights_container.markdown(raw_extracted_content)
                if not llm_globally_enabled: insights_container.caption("LLM processing disabled (no API key configured); no LLM insights could be generated.")
                elif not has_llm_insights: insights_container.caption("No specific LLM insights (summary/extractions) generated or available for this item.")
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

def display_processing_log():
    # ... (as in v1.1.11) ...
    if st.session_state.get('processing_log'):
        with st.expander("📜 View Processing Log", expanded=False):
            log_content = "\n".join(st.session_state.processing_log)
            filtered_log_lines = [line for line in log_content.splitlines() if not line.startswith("LOG_UI_STATUS:")]
            st.text_area("Processing Log (filtered for UI display):", value="\n".join(filtered_log_lines), height=300, key="log_display_text_area_main_v1112", disabled=True) # Unique key

# // end of modules/ui_manager.py
