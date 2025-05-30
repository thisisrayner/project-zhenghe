# app.py
# Version 3.0.0: Modularized UI, processing, and Excel logic.
# Main app orchestrates calls to new modules: ui_manager, process_manager, excel_handler.
# Retains session state management, config loading, and GSheets setup.
# KSAT v2.0.2 featureset maintained.
# Version 3.0.1: Pass gsheets_secrets_present to process_manager.
# Version 3.0.2: Corrected initial_keywords_for_display and llm_generated_keywords_set_for_display update from process_manager.
# Version 3.0.3: Ensured batch_timestamp for Excel export is generated when needed.
"""
Streamlit Web Application for Keyword Search & Analysis Tool (KSAT).

Orchestrates UI, data processing, and export functionalities by calling
dedicated modules.
"""

import streamlit as st
from modules import config, data_storage, ui_manager, process_manager, excel_handler
import time
from typing import Dict, Any, Optional

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
    'last_keywords': "",  # For repopulating input field
    'last_extract_query': "", # For repopulating input field
    'consolidated_summary_text': None,
    'gs_worksheet': None,
    'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False,
    'gsheets_error_message': None,
    'initial_keywords_for_display': set(), # For UI logic (e.g. distinguishing LLM queries)
    'llm_generated_keywords_set_for_display': set() # For UI logic
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Google Sheets Setup ---
# This needs to run once per session ideally, or be robust to reruns
gsheets_secrets_present = bool(cfg.gsheets.service_account_info and \
                           (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name)) # Ensure boolean

if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False # Reset on new attempt
    st.session_state.gsheets_error_message = None  # Reset on new attempt
    
    if gsheets_secrets_present:
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
            cfg.gsheets.service_account_info,
            cfg.gsheets.spreadsheet_id,
            cfg.gsheets.spreadsheet_name, # Pass name as well
            cfg.gsheets.worksheet_name
        )
        if st.session_state.gs_worksheet:
            data_storage.ensure_master_header(st.session_state.gs_worksheet)
            st.session_state.sheet_writing_enabled = True
        else:
            st.session_state.gsheets_error_message = "Google Sheets connection failed. Check Sheet ID/Name & sharing with service account."
    else:
        st.session_state.gsheets_error_message = "Google Sheets partially or not configured in secrets.toml. Data storage to Sheets disabled."
        # No st.stop() here, app can still run without GSheets

# --- UI Rendering ---
st.title("Keyword Search & Analysis Tool (KSAT) 🔮")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

# Render Sidebar and get inputs
keywords_input, num_results, llm_extract_query, start_button = ui_manager.render_sidebar(
    cfg,
    st.session_state.gsheets_error_message,
    st.session_state.sheet_writing_enabled
)

# Apply Custom CSS
ui_manager.apply_custom_css()

# Define containers for results and logs (can be filled after processing)
results_container = st.container()
log_container = st.container()


# --- Main Processing Logic ---
if start_button:
    # Reset state for new run
    st.session_state.processing_log = ["Processing initiated..."]
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()


    # Store last used inputs for UI persistence
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_query = llm_extract_query

    # Call the processing manager
    log, data, summary, initial_kws_display, llm_kws_display = process_manager.run_search_and_analysis(
        app_config=cfg,
        keywords_input=keywords_input,
        llm_extract_query_input=llm_extract_query,
        num_results_wanted_per_keyword=num_results,
        gs_worksheet=st.session_state.gs_worksheet,
        sheet_writing_enabled=st.session_state.sheet_writing_enabled,
        gsheets_secrets_present=gsheets_secrets_present
    )

    # Update session state with results from processing
    st.session_state.processing_log = log
    st.session_state.results_data = data
    st.session_state.consolidated_summary_text = summary
    st.session_state.initial_keywords_for_display = initial_kws_display
    st.session_state.llm_generated_keywords_set_for_display = llm_kws_display


# --- Display Results and Logs (runs on every interaction if data exists) ---
with results_container:
    if st.session_state.results_data: # Check if there are item details to download
        st.markdown("---")
        # Prepare data for Excel export
        df_item_details = excel_handler.prepare_item_details_df(
            st.session_state.results_data,
            st.session_state.last_extract_query
        )
        
        # Generate batch timestamp for Excel if not already present from a successful GSheet write
        # For simplicity, we'll always generate one here for Excel context if results exist
        batch_excel_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        df_consolidated_summary_excel = None
        if st.session_state.consolidated_summary_text:
             df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                st.session_state.consolidated_summary_text,
                len(st.session_state.results_data),
                st.session_state.last_keywords,
                st.session_state.last_extract_query,
                batch_excel_timestamp # Use consistent timestamp for this export
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

    # Display consolidated summary (if any)
    ui_manager.display_consolidated_summary() # Reads from session_state

    # Display individual results (if any)
    ui_manager.display_individual_results() # Reads from session_state

with log_container:
    # Display processing log (if any)
    ui_manager.display_processing_log() # Reads from session_state

st.markdown("---")
st.caption("Keyword Search & Analysis Tool (KSAT) v3.0.3 (Modularized)")

# end of app.py
