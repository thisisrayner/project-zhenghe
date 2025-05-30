# app.py
# Version 3.1.0: Integrated support for two distinct LLM extraction queries.
# Main app orchestrates calls to new modules: ui_manager, process_manager, excel_handler.
# Retains session state management, config loading, and GSheets setup.
"""
Streamlit Web Application for Keyword Search & Analysis Tool (KSAT).
Orchestrates UI, data processing, and export functionalities by calling
dedicated modules.
"""

import streamlit as st
from modules import config, data_storage, ui_manager, process_manager, excel_handler
import time
from typing import Dict, Any, Optional, List # Added List

# --- Page Configuration ---
st.set_page_config(page_title="Keyword Search & Analysis Tool (KSAT)", page_icon="🔮", layout="wide")

# --- Load Application Configuration ---
cfg: Optional[config.AppConfig] = config.load_config()
if not cfg:
    st.error("CRITICAL: Application configuration failed to load. Check secrets.toml.")
    st.stop()

# --- Session State Initialization ---
default_session_state: Dict[str, Any] = {
    'processing_log': [],
    'results_data': [],
    'last_keywords': "",
    'last_extract_queries': ["", ""], # MODIFIED: Store as list for multiple queries
    'consolidated_summary_text': None,
    'gs_worksheet': None,
    'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False,
    'gsheets_error_message': None,
    'initial_keywords_for_display': set(),
    'llm_generated_keywords_set_for_display': set()
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Google Sheets Setup ---
# ... (GSheets setup remains the same)
gsheets_secrets_present = bool(cfg.gsheets.service_account_info and \
                           (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name)) 

if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None  
    
    if gsheets_secrets_present:
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
            cfg.gsheets.service_account_info,
            cfg.gsheets.spreadsheet_id,
            cfg.gsheets.spreadsheet_name, 
            cfg.gsheets.worksheet_name
        )
        if st.session_state.gs_worksheet:
            data_storage.ensure_master_header(st.session_state.gs_worksheet)
            st.session_state.sheet_writing_enabled = True
        else:
            st.session_state.gsheets_error_message = "Google Sheets connection failed. Check Sheet ID/Name & sharing with service account."
    else:
        st.session_state.gsheets_error_message = "Google Sheets partially or not configured in secrets.toml. Data storage to Sheets disabled."


# --- UI Rendering ---
st.title("Keyword Search & Analysis Tool (KSAT) 🔮")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

# Render Sidebar and get inputs
keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar( # MODIFIED
    cfg,
    st.session_state.gsheets_error_message,
    st.session_state.sheet_writing_enabled
)

ui_manager.apply_custom_css()
results_container = st.container()
log_container = st.container()

# --- Main Processing Logic ---
if start_button:
    st.session_state.processing_log = ["Processing initiated..."]
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()

    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list # MODIFIED: Store list

    # Filter out empty strings from the list of queries to pass to process_manager
    active_llm_extract_queries = [q for q in llm_extract_queries_list if q.strip()]


    log, data, summary, initial_kws_display, llm_kws_display = process_manager.run_search_and_analysis(
        app_config=cfg,
        keywords_input=keywords_input,
        llm_extract_queries_input=active_llm_extract_queries, # MODIFIED: Pass the list
        num_results_wanted_per_keyword=num_results,
        gs_worksheet=st.session_state.gs_worksheet,
        sheet_writing_enabled=st.session_state.sheet_writing_enabled,
        gsheets_secrets_present=gsheets_secrets_present
    )

    st.session_state.processing_log = log
    st.session_state.results_data = data
    st.session_state.consolidated_summary_text = summary
    st.session_state.initial_keywords_for_display = initial_kws_display
    st.session_state.llm_generated_keywords_set_for_display = llm_kws_display

# --- Display Results and Logs ---
with results_container:
    if st.session_state.results_data:
        st.markdown("---")
        df_item_details = excel_handler.prepare_item_details_df(
            st.session_state.results_data,
            st.session_state.last_extract_queries # MODIFIED: Pass list of queries
        )
        
        batch_excel_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        df_consolidated_summary_excel = None
        if st.session_state.consolidated_summary_text:
             df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                st.session_state.consolidated_summary_text,
                len(st.session_state.results_data),
                st.session_state.last_keywords,
                st.session_state.last_extract_queries[0] if st.session_state.last_extract_queries and st.session_state.last_extract_queries[0] else None, # Use primary for note
                batch_excel_timestamp
            )

        excel_file_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_file_bytes,
            file_name=f"ksat_results_{time.strftime('%Y%m%d-%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_excel_button"
        )

    ui_manager.display_consolidated_summary()
    ui_manager.display_individual_results()

with log_container:
    ui_manager.display_processing_log()

st.markdown("---")
st.caption("Keyword Search & Analysis Tool (KSAT) v3.1.0 (Multi-Query Extraction)")

# end of app.py
