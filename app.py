# app.py
# Version 3.1.8:
# - Corrected AttributeError: Used 'cfg.sheets' instead of 'cfg.gsheets' to access
#   Google Sheets configuration, aligning with AppConfig dataclass.
# Previous versions:
# - Version 3.1.7: Stores loaded app config in session_state, initializes llm_globally_enabled.

"""
Streamlit Web Application for D.O.R.A - The Research Agent.
"""

import streamlit as st
from modules import config, data_storage, ui_manager, process_manager, excel_handler
import time
from typing import Dict, Any, Optional, List
import traceback 

# --- Page Configuration ---
st.set_page_config(page_title="D.O.R.A - The Research Agent", page_icon="🔮", layout="wide")
print("DEBUG (app.py): app.py execution started/re-run.") 

# --- Load Application Configuration AND STORE IT IN SESSION STATE ---
if 'app_config' not in st.session_state or st.session_state.app_config is None:
    print("DEBUG (app.py): Attempting to load config and store in session_state.")
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config:
        # Assuming APP_VERSION is a constant in config.py, not an attribute of AppConfig instance
        print(f"DEBUG (app.py): Config loaded into session_state. App Version from config module: {config.APP_VERSION}")
    else:
        st.error("CRITICAL: Application configuration failed to load. Check secrets.toml and logs.")
        print("CRITICAL ERROR (app.py): config.load_config() returned None. Stopping execution.")
        st.stop()

cfg: Optional[config.AppConfig] = st.session_state.app_config
if not cfg: # Should not happen if above logic is sound
    st.error("CRITICAL: AppConfig is None after session_state retrieval. Halting.")
    print("CRITICAL ERROR (app.py): cfg became None unexpectedly. Stopping execution.")
    st.stop()

print(f"DEBUG (app.py): Using AppConfig. LLM Throttling Threshold: {cfg.llm.llm_throttling_threshold_results}, Delay: {cfg.llm.llm_item_request_delay_seconds}") 

# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state:
    if cfg.llm: # cfg and cfg.llm should be valid here
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = (
            (llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or
            (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key)
        )
        print(f"DEBUG (app.py): llm_globally_enabled set in session_state: {st.session_state.llm_globally_enabled}")
    else: # Should not happen if config loaded correctly
        st.session_state.llm_globally_enabled = False
        print(f"DEBUG (app.py): llm_globally_enabled set to False (cfg.llm missing) in session_state.")

# --- Session State Initialization (Defaults for other keys) ---
default_session_state_keys: Dict[str, Any] = {
    'processing_log': [], 'results_data': [], 'last_keywords': "",
    'last_extract_queries': ["", ""], 'consolidated_summary_text': None,
    'focused_summary_sources': [], 'gs_worksheet': None, 'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False, 'gsheets_error_message': None,
    'initial_keywords_for_display': set(), 'llm_generated_keywords_set_for_display': set(),
    'batch_timestamp_for_excel': None, 'run_complete_status_message': None 
}
for key, default_value in default_session_state_keys.items():
    if key not in st.session_state: st.session_state[key] = default_value
print("DEBUG (app.py): Other session state keys initialized/verified.") 

# --- Google Sheets Setup ---
# CORRECTED: Use cfg.sheets instead of cfg.gsheets
gsheets_secrets_present = bool(cfg.sheets.service_account_info and \
                           (cfg.sheets.spreadsheet_id or cfg.sheets.spreadsheet_name))
print(f"DEBUG (app.py): gsheets_secrets_present = {gsheets_secrets_present}") 

if not st.session_state.sheet_connection_attempted_this_session:
    print("DEBUG (app.py): Attempting Google Sheets connection for the first time this session.") 
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            # CORRECTED: Use cfg.sheets
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( 
                cfg.sheets.service_account_info, cfg.sheets.spreadsheet_id,
                cfg.sheets.spreadsheet_name, cfg.sheets.worksheet_name
            )
            if st.session_state.gs_worksheet:
                print(f"DEBUG (app.py): Successfully got worksheet object: {st.session_state.gs_worksheet.title if hasattr(st.session_state.gs_worksheet, 'title') else type(st.session_state.gs_worksheet)}") 
                data_storage.ensure_master_header(st.session_state.gs_worksheet) 
                st.session_state.sheet_writing_enabled = True
                print("DEBUG (app.py): sheet_writing_enabled set to True.") 
            else:
                st.session_state.gsheets_error_message = st.session_state.get("gsheets_error_message", "Google Sheets connection failed (worksheet object is None).")
                print(f"DEBUG (app.py): gs_worksheet is None. Error: {st.session_state.gsheets_error_message}") 
        except Exception as e_gs_setup:
            st.session_state.gsheets_error_message = f"Error during Google Sheets setup: {e_gs_setup}"
            print(f"DEBUG (app.py): EXCEPTION during Google Sheets setup: {e_gs_setup}") 
            print(traceback.format_exc())
    else:
        st.session_state.gsheets_error_message = "Google Sheets not fully configured. Data storage to Sheets disabled."
        print(f"DEBUG (app.py): {st.session_state.gsheets_error_message}") 
else:
    print("DEBUG (app.py): Google Sheets connection already attempted.") 
    # print(f"DEBUG (app.py): Current sheet_writing_enabled: {st.session_state.get('sheet_writing_enabled', 'N/A')}, Worksheet type: {type(st.session_state.get('gs_worksheet'))}, Error: {st.session_state.get('gsheets_error_message', 'None')}") 

# --- UI Rendering ---
st.title("D.O.R.A 🔮")
st.markdown("The **Research** **Agent** For **Domain**-Wide **Overview** and Insights.")
print("DEBUG (app.py): Main UI title rendered.") 

keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, # Pass the cfg object loaded from session_state
    st.session_state.gsheets_error_message, 
    st.session_state.sheet_writing_enabled
)
# print(f"DEBUG (app.py): Sidebar rendered. Start button state: {start_button}, Keywords: '{keywords_input}', Num_results: {num_results}, Q1: '{llm_extract_queries_list[0] if llm_extract_queries_list else ''}', Q2: '{llm_extract_queries_list[1] if len(llm_extract_queries_list) > 1 else ''}'") 

ui_manager.apply_custom_css()
status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()

# --- Main Processing Logic ---
if start_button:
    print("DEBUG (app.py): 'Start Search & Analysis' button pressed.") 
    st.session_state.processing_log = ["Processing initiated... (app.py)"] 
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None
    st.session_state.focused_summary_sources = []
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.run_complete_status_message = None 
    print(f"DEBUG (app.py): Session state for run initialized. Batch timestamp: {st.session_state.batch_timestamp_for_excel}") 

    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list 

    active_llm_extract_queries = [q for q in llm_extract_queries_list if q.strip()]
    print(f"DEBUG (app.py): Calling process_manager.run_search_and_analysis. Active Queries: {active_llm_extract_queries}, Num Results per Keyword: {num_results}") 
    # print(f"DEBUG (app.py): Sheet writing enabled for PM: {st.session_state.sheet_writing_enabled}, Worksheet for PM: {type(st.session_state.gs_worksheet)}") 

    with st.spinner("D.O.R.A. is thinking... please wait for all processing to complete."): 
        try:
            # Ensure cfg is passed here
            log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
                app_config=cfg, keywords_input=keywords_input,
                llm_extract_queries_input=active_llm_extract_queries,
                num_results_wanted_per_keyword=num_results,
                gs_worksheet=st.session_state.gs_worksheet,
                sheet_writing_enabled=st.session_state.sheet_writing_enabled,
                gsheets_secrets_present=gsheets_secrets_present
            )
            print(f"DEBUG (app.py): Returned from process_manager.run_search_and_analysis.") 
            st.session_state.processing_log = log
            st.session_state.results_data = data
            st.session_state.consolidated_summary_text = summary
            st.session_state.focused_summary_sources = focused_sources
            st.session_state.initial_keywords_for_display = initial_kws_display
            st.session_state.llm_generated_keywords_set_for_display = llm_kws_display
            print("DEBUG (app.py): Session state updated with results from process_manager.") 

            final_status_log_entry = next((item for item in reversed(log) if isinstance(item, str) and item.startswith("LOG_STATUS:")), None)
            if final_status_log_entry:
                st.session_state.run_complete_status_message = final_status_log_entry
            else:
                st.session_state.run_complete_status_message = "LOG_STATUS:WARNING:Processing finished, but final status unclear from logs."
        except Exception as e_process_mgr:
            status_message_placeholder.error(f"A critical error occurred during processing: {e_process_mgr}")
            print(f"CRITICAL ERROR (app.py): Exception in process_manager.run_search_and_analysis call: {e_process_mgr}") 
            current_log_val = st.session_state.get('processing_log', [])
            if not isinstance(current_log_val, list): current_log_val = [str(current_log_val)] 
            err_msg_for_log = f"APP_PY_ERROR: Main process failed: {e_process_mgr}\n{traceback.format_exc()}"
            current_log_val.append(err_msg_for_log)
            st.session_state.processing_log = current_log_val
            st.session_state.run_complete_status_message = f"LOG_STATUS:APP_PY_CRITICAL_ERROR:{err_msg_for_log}" 

if st.session_state.get('run_complete_status_message'):
    status_parts = st.session_state.run_complete_status_message.split(':', 2)
    status_type = "INFO" 
    status_text = st.session_state.run_complete_status_message
    if len(status_parts) > 1: status_type = status_parts[1]
    if len(status_parts) > 2: status_text = status_parts[2]
    
    print(f"DEBUG (app.py): Displaying final status. Type: {status_type}, Text: {status_text[:100]}")
    if status_type == "SUCCESS": status_message_placeholder.success(status_text)
    elif status_type == "WARNING": status_message_placeholder.warning(status_text)
    elif status_type == "ERROR" or "CRITICAL_ERROR" in status_type: status_message_placeholder.error(status_text)
    else: status_message_placeholder.info(status_text)
    st.session_state.run_complete_status_message = None 

print("DEBUG (app.py): Entering Display Results and Logs section.") 
# print(f"DEBUG (app.py): results_data available: {bool(st.session_state.get('results_data'))}, summary_text available: {bool(st.session_state.get('consolidated_summary_text'))}") 

with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"):
        st.markdown("---")
        # print("DEBUG (app.py): Preparing Excel data for download button.") 
        try:
            df_item_details = excel_handler.prepare_item_details_df(st.session_state.get("results_data", []), st.session_state.last_extract_queries )
            df_consolidated_summary_excel = None
            if st.session_state.consolidated_summary_text:
                q1_text_for_excel = st.session_state.last_extract_queries[0] if st.session_state.last_extract_queries and st.session_state.last_extract_queries[0] else None
                q2_text_for_excel = st.session_state.last_extract_queries[1] if st.session_state.last_extract_queries and len(st.session_state.last_extract_queries) > 1 and st.session_state.last_extract_queries[1] else None
                focused_count_for_excel = None
                if isinstance(st.session_state.focused_summary_sources, list): focused_count_for_excel = len(st.session_state.focused_summary_sources)
                df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                    consolidated_summary_text=st.session_state.consolidated_summary_text,
                    results_data_count=len(st.session_state.get("results_data", [])),
                    last_keywords=st.session_state.last_keywords,
                    primary_llm_extract_query=q1_text_for_excel, secondary_llm_extract_query=q2_text_for_excel, 
                    batch_timestamp=st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y-%m-%d %H:%M:%S')),
                    focused_summary_source_count=focused_count_for_excel)
            excel_file_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
            filename_timestamp = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d%H%M%S')).replace(":", "").replace("-", "").replace(" ", "_") 
            st.download_button(label="📥 Download Results as Excel", data=excel_file_bytes, file_name=f"dora_results_{filename_timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="download_excel_button")
            # print("DEBUG (app.py): Excel download button rendered.") 
        except Exception as e_excel:
            st.error(f"Error preparing Excel download: {e_excel}")
            print(f"ERROR (app.py): Exception during Excel preparation: {e_excel}") 

    ui_manager.display_consolidated_summary_and_sources(st.session_state.consolidated_summary_text, st.session_state.focused_summary_sources, st.session_state.last_extract_queries)
    ui_manager.display_individual_results() # This will use st.session_state.llm_globally_enabled
    # print("DEBUG (app.py): Summary and individual results display methods called from ui_manager.") 

# --- DEBUG for UI Log Display in Sidebar (from v3.1.6) ---
# (This section can remain for debugging purposes if needed)
st.sidebar.subheader("Log Debug (app.py - UI section)")
# ... (rest of this debug block from v3.1.6) ...

with log_container:
    # print("DEBUG (app.py): Calling ui_manager.display_processing_log().") 
    ui_manager.display_processing_log() 

st.markdown("---")
st.caption(f"D.O.R.A v{config.APP_VERSION}") 
print(f"DEBUG (app.py): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
