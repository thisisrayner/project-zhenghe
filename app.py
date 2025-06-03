# app.py
# Version 3.1.10:
# - Corrected attribute access: Uses 'cfg.gsheets' to access Google Sheets
#   configuration, aligning with AppConfig dataclass in config.py (v1.5.0).
# Previous versions:
# - Version 3.1.9: Added more robust checks for AppConfig loading.
# - Version 3.1.8: (Attempted to correct gsheets to sheets, but was incorrect based on config.py)

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
print("DEBUG (app.py V3.1.10): app.py execution started/re-run.") 

# --- Load Application Configuration AND STORE IT IN SESSION STATE ---
if 'app_config' not in st.session_state or st.session_state.app_config is None:
    print("DEBUG (app.py V3.1.10): 'app_config' not in session_state or is None. Calling config.load_config().")
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config is None:
        st.error("CRITICAL FAILURE: config.load_config() returned None. Application cannot start. Check secrets.toml and logs.")
        print("CRITICAL ERROR (app.py V3.1.10): config.load_config() returned None. Stopping execution.")
        st.stop()
    else:
        print(f"DEBUG (app.py V3.1.10): config.load_config() successful. AppConfig type: {type(st.session_state.app_config)}")
        # Check for gsheets specifically after load from config.py
        if hasattr(st.session_state.app_config, 'gsheets'): # Use 'gsheets'
            print(f"DEBUG (app.py V3.1.10): Newly loaded app_config.gsheets type: {type(st.session_state.app_config.gsheets)}")
        else:
            print("ERROR (app.py V3.1.10): Newly loaded app_config does NOT have 'gsheets' attribute immediately after load!")


cfg: Optional[config.AppConfig] = st.session_state.app_config

if cfg is None:
    st.error("CRITICAL ERROR: Application configuration (cfg) is None. Halting.")
    print("CRITICAL ERROR (app.py V3.1.10): cfg is None. Halting.")
    st.stop()

# Use 'gsheets' as defined in config.py AppConfig
if not hasattr(cfg, 'gsheets') or cfg.gsheets is None:
    st.error("CRITICAL ERROR: Configuration 'cfg' missing 'gsheets' attribute or 'cfg.gsheets' is None. Check config.py. Halting.")
    print(f"CRITICAL ERROR (app.py V3.1.10): cfg.gsheets missing/None. cfg type: {type(cfg)}. hasattr: {hasattr(cfg, 'gsheets')}. Halting.")
    st.stop()
print(f"DEBUG (app.py V3.1.10): cfg and cfg.gsheets valid. cfg.gsheets type: {type(cfg.gsheets)}. LLM Thresh: {cfg.llm.llm_throttling_threshold_results}")


# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state:
    if hasattr(cfg, 'llm') and cfg.llm:
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = (
            (llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or
            (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key)
        )
    else: st.session_state.llm_globally_enabled = False
    print(f"DEBUG (app.py V3.1.10): llm_globally_enabled set in session_state: {st.session_state.llm_globally_enabled}")

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
print("DEBUG (app.py V3.1.10): Other session state keys initialized/verified.") 

# --- Google Sheets Setup ---
# CORRECTED TO USE cfg.gsheets
gsheets_secrets_present = bool(cfg.gsheets.service_account_info and \
                           (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name))
print(f"DEBUG (app.py V3.1.10): gsheets_secrets_present = {gsheets_secrets_present}") 

if not st.session_state.sheet_connection_attempted_this_session:
    print("DEBUG (app.py V3.1.10): Attempting Google Sheets connection for the first time this session.") 
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            # CORRECTED TO USE cfg.gsheets
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( 
                cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id,
                cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name
            )
            if st.session_state.gs_worksheet:
                print(f"DEBUG (app.py V3.1.10): Successfully got worksheet: {st.session_state.gs_worksheet.title if hasattr(st.session_state.gs_worksheet, 'title') else type(st.session_state.gs_worksheet)}") 
                data_storage.ensure_master_header(st.session_state.gs_worksheet) 
                st.session_state.sheet_writing_enabled = True
            else:
                st.session_state.gsheets_error_message = st.session_state.get("gsheets_error_message", "Google Sheets connection failed (worksheet object is None).")
        except Exception as e_gs_setup:
            st.session_state.gsheets_error_message = f"Error during Google Sheets setup: {e_gs_setup}"
            print(f"DEBUG (app.py V3.1.10): EXCEPTION during Google Sheets setup: {e_gs_setup}") 
    else:
        st.session_state.gsheets_error_message = "Google Sheets not fully configured. Data storage to Sheets disabled."
else:
    print("DEBUG (app.py V3.1.10): Google Sheets connection already attempted.") 


# --- UI Rendering ---
# ... (Unchanged from v3.1.9 - Sidebar call, Main Processing Logic, Results Display) ...
# ... (The rest of the app.py file content from v3.1.9 should follow here) ...
st.title("D.O.R.A 🔮")
st.markdown("The **Research** **Agent** For **Domain**-Wide **Overview** and Insights.")
keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, st.session_state.gsheets_error_message, st.session_state.sheet_writing_enabled)
ui_manager.apply_custom_css()
status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()
if start_button:
    st.session_state.processing_log = ["Processing initiated... (app.py V3.1.10)"] 
    st.session_state.results_data = [] # etc.
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list 
    with st.spinner("D.O.R.A. is thinking... please wait for all processing to complete."): 
        try:
            log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
                app_config=cfg, keywords_input=keywords_input,
                llm_extract_queries_input=[q for q in llm_extract_queries_list if q.strip()],
                num_results_wanted_per_keyword=num_results,
                gs_worksheet=st.session_state.gs_worksheet,
                sheet_writing_enabled=st.session_state.sheet_writing_enabled,
                gsheets_secrets_present=gsheets_secrets_present)
            st.session_state.processing_log = log; st.session_state.results_data = data
            st.session_state.consolidated_summary_text = summary; st.session_state.focused_summary_sources = focused_sources
            st.session_state.initial_keywords_for_display = initial_kws_display; st.session_state.llm_generated_keywords_set_for_display = llm_kws_display
            final_status_log_entry = next((item for item in reversed(log) if isinstance(item, str) and item.startswith("LOG_STATUS:")), None)
            if final_status_log_entry: st.session_state.run_complete_status_message = final_status_log_entry
            else: st.session_state.run_complete_status_message = "LOG_STATUS:WARNING:Processing finished, status unclear."
        except Exception as e_pm:
            st.session_state.run_complete_status_message = f"LOG_STATUS:APP_PY_CRITICAL_ERROR:Process manager failed: {e_pm}\n{traceback.format_exc()}"
if st.session_state.get('run_complete_status_message'):
    status_parts = st.session_state.run_complete_status_message.split(':', 2)
    status_type = status_parts[1] if len(status_parts) > 1 else "INFO"
    status_text = status_parts[2] if len(status_parts) > 2 else st.session_state.run_complete_status_message
    if status_type == "SUCCESS": status_message_placeholder.success(status_text)
    elif status_type == "WARNING": status_message_placeholder.warning(status_text)
    else: status_message_placeholder.error(status_text)
    st.session_state.run_complete_status_message = None 
with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"):
        st.markdown("---")
        try:
            df_item_details = excel_handler.prepare_item_details_df(st.session_state.get("results_data", []), st.session_state.last_extract_queries )
            df_consolidated_summary_excel = None
            if st.session_state.consolidated_summary_text:
                q1_excel = st.session_state.last_extract_queries[0] if st.session_state.last_extract_queries else None
                q2_excel = st.session_state.last_extract_queries[1] if len(st.session_state.last_extract_queries) > 1 else None
                focused_src_count = len(st.session_state.focused_summary_sources) if isinstance(st.session_state.focused_summary_sources, list) else None
                df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                    consolidated_summary_text=st.session_state.consolidated_summary_text, results_data_count=len(st.session_state.get("results_data", [])),
                    last_keywords=st.session_state.last_keywords, primary_llm_extract_query=q1_excel, secondary_llm_extract_query=q2_excel, 
                    batch_timestamp=st.session_state.get("batch_timestamp_for_excel", ""), focused_summary_source_count=focused_src_count )
            excel_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
            dl_ts = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d%H%M%S')).replace(":", "").replace("-", "").replace(" ", "_")
            st.download_button(label="📥 Download Results as Excel", data=excel_bytes, file_name=f"dora_results_{dl_ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        except Exception as e_excel_main: st.error(f"Error preparing Excel: {e_excel_main}")
    ui_manager.display_consolidated_summary_and_sources(st.session_state.consolidated_summary_text, st.session_state.focused_summary_sources, st.session_state.last_extract_queries)
    ui_manager.display_individual_results()
with log_container: ui_manager.display_processing_log() 
st.markdown("---")
st.caption(f"D.O.R.A v{config.APP_VERSION}") 
print(f"DEBUG (app.py V3.1.10): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
