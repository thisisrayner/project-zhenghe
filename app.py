# app.py
# Version 3.1.3:
# - Corrected arguments passed to excel_handler.prepare_consolidated_summary_df
#   to match its updated signature (v1.2.0 of excel_handler).
# Version 3.1.2:
# - Rebranded UI to "D.O.R.A".
# - Updated page title, main title, subtitle, caption, and download filenames per new branding.
# - Final subtitle capitalization and bolding corrections.
# Version 3.1.1:
# - Displays sources used for focused consolidated summaries.
# - Handles new return value from process_manager.
"""
Streamlit Web Application for D.O.R.A - The Research Agent.
"""

import streamlit as st
from modules import config, data_storage, ui_manager, process_manager, excel_handler
import time
from typing import Dict, Any, Optional, List

# --- Page Configuration ---
st.set_page_config(
    page_title="D.O.R.A - The Research Agent",
    page_icon="🔮",
    layout="wide"
)

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
    'last_extract_queries': ["", ""], # Stores [Q1_text, Q2_text]
    'consolidated_summary_text': None,
    'focused_summary_sources': [], # List of FocusedSummarySource TypedDicts
    'gs_worksheet': None,
    'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False,
    'gsheets_error_message': None,
    'initial_keywords_for_display': set(),
    'llm_generated_keywords_set_for_display': set(),
    'batch_timestamp_for_excel': None # To store a consistent timestamp for the batch
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Google Sheets Setup ---
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
st.title("D.O.R.A 🔮")
st.markdown("The **Research** **Agent** For **Domain**-Wide **Overview** and Insights.")

keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
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
    st.session_state.focused_summary_sources = []
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S') # Set batch timestamp here

    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list # This now holds [Q1, Q2] from UI

    active_llm_extract_queries = [q for q in llm_extract_queries_list if q.strip()]

    # Call process_manager.run_search_and_analysis
    # The signature of run_search_and_analysis in process_manager v1.4.0 (with throttling)
    # returns: processing_log, results_data, consolidated_summary_text,
    #          initial_keywords_for_display, llm_generated_keywords_set_for_display,
    #          focused_summary_source_details
    # This matches the unpacking here.
    log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
        app_config=cfg,
        keywords_input=keywords_input,
        llm_extract_queries_input=active_llm_extract_queries,
        num_results_wanted_per_keyword=num_results,
        gs_worksheet=st.session_state.gs_worksheet,
        sheet_writing_enabled=st.session_state.sheet_writing_enabled,
        gsheets_secrets_present=gsheets_secrets_present
    )

    st.session_state.processing_log = log
    st.session_state.results_data = data
    st.session_state.consolidated_summary_text = summary
    st.session_state.focused_summary_sources = focused_sources # List of FocusedSummarySource dicts
    st.session_state.initial_keywords_for_display = initial_kws_display
    st.session_state.llm_generated_keywords_set_for_display = llm_kws_display

# --- Display Results and Logs ---
with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"): # Check if there's anything to show/download
        st.markdown("---")
        
        # Prepare df_item_details (assuming results_data might be empty but summary exists)
        df_item_details = excel_handler.prepare_item_details_df(
            st.session_state.get("results_data", []), # Pass empty list if no results_data
            st.session_state.last_extract_queries # Contains [Q1_text, Q2_text]
        )

        df_consolidated_summary_excel = None
        if st.session_state.consolidated_summary_text:
            q1_text_for_excel = st.session_state.last_extract_queries[0] if st.session_state.last_extract_queries and st.session_state.last_extract_queries[0] else None
            q2_text_for_excel = st.session_state.last_extract_queries[1] if st.session_state.last_extract_queries and len(st.session_state.last_extract_queries) > 1 and st.session_state.last_extract_queries[1] else None
            
            # Determine focused_summary_source_count
            focused_count_for_excel = None
            if st.session_state.focused_summary_sources is not None: # Check if it's None or an empty list
                focused_count_for_excel = len(st.session_state.focused_summary_sources)

            df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                consolidated_summary_text=st.session_state.consolidated_summary_text,
                results_data_count=len(st.session_state.get("results_data", [])),
                last_keywords=st.session_state.last_keywords,
                primary_llm_extract_query=q1_text_for_excel,
                secondary_llm_extract_query=q2_text_for_excel, # ADDED
                batch_timestamp=st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y-%m-%d %H:%M:%S')),
                focused_summary_source_count=focused_count_for_excel # ADDED
            )

        excel_file_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
        
        # Use a consistent timestamp for the filename
        filename_timestamp = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d-%H%M%S')).replace(":", "").replace("-", "")
        
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_file_bytes,
            file_name=f"dora_results_{filename_timestamp}.xlsx", # Updated filename
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_excel_button"
        )

    # Display consolidated summary and sources (if any)
    ui_manager.display_consolidated_summary_and_sources(
        st.session_state.consolidated_summary_text,
        st.session_state.focused_summary_sources, # This is the list of dicts
        st.session_state.last_extract_queries # Pass Q1, Q2 texts
    )
    # Display individual results (if any)
    ui_manager.display_individual_results() # This function needs access to st.session_state.results_data

with log_container:
    ui_manager.display_processing_log() # This function needs access to st.session_state.processing_log

st.markdown("---")
st.caption(f"D.O.R.A v{config.APP_VERSION}")

# end of app.py
