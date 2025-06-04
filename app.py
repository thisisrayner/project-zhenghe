# app.py
# Version 3.1.16:
# - Helper text is now hidden immediately on the same re-run when the start button is clicked.
# Previous versions:
# - Version 3.1.15: Helper text hidden after start button click (on next re-run).

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
print(f"DEBUG (app.py V{config.APP_VERSION}): app.py execution started/re-run.") 

# --- Load Application Configuration AND STORE IT IN SESSION STATE ---
# ... (Config loading as in v3.1.15) ...
if 'app_config' not in st.session_state or st.session_state.app_config is None:
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config is None: st.error("CRITICAL FAILURE: config.load_config() returned None."); st.stop()
cfg: Optional[config.AppConfig] = st.session_state.app_config
if cfg is None: st.error("CRITICAL ERROR: cfg is None."); st.stop()
if not hasattr(cfg, 'gsheets') or cfg.gsheets is None: st.error("CRITICAL ERROR: cfg.gsheets missing/None."); st.stop()

# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state:
    if hasattr(cfg, 'llm') and cfg.llm:
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = ((llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or \
                                                (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key))
    else: st.session_state.llm_globally_enabled = False

# --- Session State Initialization (Defaults for other keys) ---
default_session_state_keys: Dict[str, Any] = {
    'processing_log': [], 'results_data': [], 'last_keywords': "", 'last_extract_queries': ["", ""], 
    'consolidated_summary_text': None, 'focused_summary_sources': [], 'gs_worksheet': None, 
    'sheet_writing_enabled': False, 'sheet_connection_attempted_this_session': False, 
    'gsheets_error_message': None, 'initial_keywords_for_display': set(), 
    'llm_generated_keywords_set_for_display': set(), 'batch_timestamp_for_excel': None, 
    'run_complete_status_message': None,
    'show_initial_helper_text': True 
}
for key, default_value in default_session_state_keys.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- Google Sheets Setup ---
# ... (as in v3.1.15) ...
gsheets_secrets_present = bool(cfg.gsheets.service_account_info and (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name))
if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False; st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id, cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name)
            if st.session_state.gs_worksheet: data_storage.ensure_master_header(st.session_state.gs_worksheet); st.session_state.sheet_writing_enabled = True
            else: st.session_state.gsheets_error_message = st.session_state.get("gsheets_error_message", "GSheets connection failed.")
        except Exception as e_gs_setup: st.session_state.gsheets_error_message = f"Error GSheets setup: {e_gs_setup}"
    else: st.session_state.gsheets_error_message = "GSheets not configured."


# --- UI Rendering ---
st.title("D.O.R.A 🔮")
st.markdown("The **Domain**-wide **Overview** For **Research** **Agent**") 

# Sidebar is rendered first to get the start_button state
keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, 
    st.session_state.gsheets_error_message, 
    st.session_state.sheet_writing_enabled
)
ui_manager.apply_custom_css() 

# --- MODIFIED HELPER TEXT SECTION ---
# Helper text is shown if the flag is True AND the start button was NOT just pressed
# on this current script run.
if st.session_state.get('show_initial_helper_text', True) and not start_button:
    st.write("") 
    st.write("Welcome to D.O.R.A.! To begin your research, please enter keywords and any specific queries in the sidebar on the left.")
st.markdown("---") 
# --- END MODIFIED HELPER TEXT SECTION ---

print(f"DEBUG (app.py V{config.APP_VERSION}): Main UI title rendered. Start button state: {start_button}, Show helper: {st.session_state.get('show_initial_helper_text', True) and not start_button}") 

status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()

# --- Main Processing Logic ---
if start_button:
    print(f"DEBUG (app.py V{config.APP_VERSION}): 'Start Search & Analysis' button pressed.") 
    st.session_state.show_initial_helper_text = False # Hide helper text for future re-runs
    
    # ... (Reset session state variables as in v3.1.15) ...
    st.session_state.processing_log = [f"Processing initiated at {time.strftime('%Y-%m-%d %H:%M:%S')}... (app.py V{config.APP_VERSION})"] 
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None
    st.session_state.focused_summary_sources = []
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.run_complete_status_message = None 
    
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list 
    active_llm_extract_queries = [q for q in llm_extract_queries_list if q.strip()]
    spinner_message = ui_manager.get_random_spinner_message()

    with st.spinner(spinner_message): 
        # ... (try-except block for process_manager.run_search_and_analysis as in v3.1.15) ...
        try:
            log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
                app_config=cfg, keywords_input=keywords_input,
                llm_extract_queries_input=active_llm_extract_queries, num_results_wanted_per_keyword=num_results,
                gs_worksheet=st.session_state.gs_worksheet, sheet_writing_enabled=st.session_state.sheet_writing_enabled,
                gsheets_secrets_present=gsheets_secrets_present)
            st.session_state.processing_log = log; st.session_state.results_data = data
            st.session_state.consolidated_summary_text = summary; st.session_state.focused_summary_sources = focused_sources
            st.session_state.initial_keywords_for_display = initial_kws_display; st.session_state.llm_generated_keywords_set_for_display = llm_kws_display
            final_status_log_entry = next((item for item in reversed(log) if isinstance(item, str) and item.startswith("LOG_STATUS:")), None)
            if final_status_log_entry: st.session_state.run_complete_status_message = final_status_log_entry
            else: st.session_state.run_complete_status_message = "LOG_STATUS:WARNING:Processing finished, status unclear."
        except Exception as e_process_mgr:
            detailed_error = f"APP_PY_ERROR: Main process failed: {type(e_process_mgr).__name__} - {e_process_mgr}\n{traceback.format_exc()}"
            status_message_placeholder.error(f"A critical error occurred in processing: {e_process_mgr}")
            current_log_val = st.session_state.get('processing_log', [])
            if not isinstance(current_log_val, list): current_log_val = [str(current_log_val)] 
            current_log_val.append(detailed_error)
            st.session_state.processing_log = current_log_val
            st.session_state.run_complete_status_message = f"LOG_STATUS:APP_PY_CRITICAL_ERROR:{detailed_error}"


if st.session_state.get('run_complete_status_message'):
    # ... (Display final status message logic as in v3.1.15) ...
    status_parts = st.session_state.run_complete_status_message.split(':', 2)
    status_type = "INFO"; status_text = st.session_state.run_complete_status_message
    if len(status_parts) > 1: status_type = status_parts[1]
    if len(status_parts) > 2: status_text = status_parts[2]
    if status_type == "SUCCESS": status_message_placeholder.success(status_text)
    elif status_type == "WARNING": status_message_placeholder.warning(status_text)
    elif "ERROR" in status_type: status_message_placeholder.error(status_text) 
    else: status_message_placeholder.info(status_text)
    st.session_state.run_complete_status_message = None 

# --- Results and Logs Display ---
# ... (as in v3.1.15, including the conditional st.markdown("---") before results) ...
with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"):
        st.markdown("---") 
        try:
            df_item_details = excel_handler.prepare_item_details_df(st.session_state.get("results_data", []), st.session_state.get('last_extract_queries', ["", ""]))
            df_consolidated_summary_excel = None
            if st.session_state.consolidated_summary_text:
                last_queries_for_excel = st.session_state.get('last_extract_queries', ["", ""])
                q1_text_for_excel = last_queries_for_excel[0] if last_queries_for_excel and last_queries_for_excel[0] else None
                q2_text_for_excel = last_queries_for_excel[1] if last_queries_for_excel and len(last_queries_for_excel) > 1 and last_queries_for_excel[1] else None
                focused_count_for_excel = len(st.session_state.focused_summary_sources) if isinstance(st.session_state.focused_summary_sources, list) else None
                df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                    consolidated_summary_text=st.session_state.consolidated_summary_text, results_data_count=len(st.session_state.get("results_data", [])),
                    last_keywords=st.session_state.get('last_keywords', ""), primary_llm_extract_query=q1_text_for_excel, secondary_llm_extract_query=q2_text_for_excel, 
                    batch_timestamp=st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y-%m-%d %H:%M:%S')), focused_summary_source_count=focused_count_for_excel)
            excel_file_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
            filename_timestamp = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d%H%M%S')).replace(":", "").replace("-", "").replace(" ", "_") 
            st.download_button(label="📥 Download Results as Excel", data=excel_file_bytes, file_name=f"dora_results_{filename_timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="download_excel_button_main_v3116") # Unique key
        except Exception as e_excel: st.error(f"Error preparing Excel download: {e_excel}")

    ui_manager.display_consolidated_summary_and_sources(st.session_state.consolidated_summary_text, st.session_state.focused_summary_sources, st.session_state.get('last_extract_queries', ["", ""]))
    ui_manager.display_individual_results()

with log_container: ui_manager.display_processing_log() 
st.markdown("---") 
st.caption(f"D.O.R.A v{config.APP_VERSION}") 
print(f"DEBUG (app.py V{config.APP_VERSION}): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
