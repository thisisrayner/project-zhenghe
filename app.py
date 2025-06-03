# app.py
# Version 3.1.9:
# - Added more robust checks and debug prints around AppConfig loading and
#   specifically for cfg.sheets before its use.
# Previous versions:
# - Version 3.1.8: Corrected cfg.gsheets to cfg.sheets.

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
print("DEBUG (app.py V3.1.9): app.py execution started/re-run.") 

# --- Load Application Configuration AND STORE IT IN SESSION STATE ---
if 'app_config' not in st.session_state or st.session_state.app_config is None:
    print("DEBUG (app.py V3.1.9): 'app_config' not in session_state or is None. Calling config.load_config().")
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config is None:
        # config.load_config() should ideally call st.error itself.
        # This is a fallback if it returns None silently.
        st.error("CRITICAL FAILURE: config.load_config() returned None. Application cannot start. Check secrets.toml and logs.")
        print("CRITICAL ERROR (app.py V3.1.9): config.load_config() returned None. Stopping execution.")
        st.stop()
    else:
        print(f"DEBUG (app.py V3.1.9): config.load_config() successful. AppConfig type: {type(st.session_state.app_config)}")
        if hasattr(st.session_state.app_config, 'sheets'):
            print(f"DEBUG (app.py V3.1.9): Newly loaded app_config.sheets type: {type(st.session_state.app_config.sheets)}")
        else:
            print("ERROR (app.py V3.1.9): Newly loaded app_config does NOT have 'sheets' attribute immediately after load!")


cfg: Optional[config.AppConfig] = st.session_state.app_config

# --- CRITICAL CHECK FOR CFG AND CFG.SHEETS ---
if cfg is None:
    st.error("CRITICAL ERROR: Application configuration (cfg) is None. This should have been caught earlier. Stopping.")
    print("CRITICAL ERROR (app.py V3.1.9): cfg is None after retrieving from session_state. This indicates a serious issue. Stopping.")
    st.stop()

if not hasattr(cfg, 'sheets') or cfg.sheets is None:
    st.error("CRITICAL ERROR: Configuration object 'cfg' is missing the 'sheets' attribute or 'cfg.sheets' is None. Check config.py and config.load_config(). Stopping.")
    print(f"CRITICAL ERROR (app.py V3.1.9): cfg.sheets is missing or None. cfg type: {type(cfg)}. hasattr(cfg, 'sheets'): {hasattr(cfg, 'sheets')}. Stopping.")
    if hasattr(cfg, 'sheets'): print(f"DEBUG: cfg.sheets object is: {cfg.sheets}")
    st.stop()
# --- END CRITICAL CHECK ---

print(f"DEBUG (app.py V3.1.9): cfg and cfg.sheets seem valid. cfg.sheets type: {type(cfg.sheets)}. LLM Thresh: {cfg.llm.llm_throttling_threshold_results}")

# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state:
    # ... (logic from v3.1.8 for llm_globally_enabled - ensure cfg.llm is accessed safely) ...
    if hasattr(cfg, 'llm') and cfg.llm:
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = (
            (llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or
            (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key)
        )
    else: # cfg.llm is missing or None
        st.session_state.llm_globally_enabled = False
    print(f"DEBUG (app.py V3.1.9): llm_globally_enabled set in session_state: {st.session_state.llm_globally_enabled}")


# --- Session State Initialization (Defaults for other keys) ---
# ... (Unchanged from v3.1.8) ...
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
print("DEBUG (app.py V3.1.9): Other session state keys initialized/verified.") 


# --- Google Sheets Setup ---
# Now cfg and cfg.sheets have been validated
gsheets_secrets_present = bool(cfg.sheets.service_account_info and \
                           (cfg.sheets.spreadsheet_id or cfg.sheets.spreadsheet_name))
print(f"DEBUG (app.py V3.1.9): gsheets_secrets_present = {gsheets_secrets_present}") 

# ... (Rest of Google Sheets setup - unchanged from v3.1.8, it should now work) ...
if not st.session_state.sheet_connection_attempted_this_session:
    print("DEBUG (app.py V3.1.9): Attempting Google Sheets connection for the first time this session.") 
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( 
                cfg.sheets.service_account_info, cfg.sheets.spreadsheet_id,
                cfg.sheets.spreadsheet_name, cfg.sheets.worksheet_name)
            if st.session_state.gs_worksheet:
                print(f"DEBUG (app.py V3.1.9): Successfully got worksheet object: {st.session_state.gs_worksheet.title if hasattr(st.session_state.gs_worksheet, 'title') else type(st.session_state.gs_worksheet)}") 
                data_storage.ensure_master_header(st.session_state.gs_worksheet) 
                st.session_state.sheet_writing_enabled = True
            else:
                st.session_state.gsheets_error_message = st.session_state.get("gsheets_error_message", "Google Sheets connection failed (worksheet object is None).")
        except Exception as e_gs_setup:
            st.session_state.gsheets_error_message = f"Error during Google Sheets setup: {e_gs_setup}"
            print(f"DEBUG (app.py V3.1.9): EXCEPTION during Google Sheets setup: {e_gs_setup}") 
    else:
        st.session_state.gsheets_error_message = "Google Sheets not fully configured. Data storage to Sheets disabled."
else:
    print("DEBUG (app.py V3.1.9): Google Sheets connection already attempted.") 


# --- UI Rendering ---
# ... (Unchanged from v3.1.8 - Sidebar call, Main Processing Logic, Results Display) ...
st.title("D.O.R.A 🔮")
st.markdown("The **Research** **Agent** For **Domain**-Wide **Overview** and Insights.")
keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, st.session_state.gsheets_error_message, st.session_state.sheet_writing_enabled)
ui_manager.apply_custom_css()
status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()
if start_button:
    st.session_state.processing_log = ["Processing initiated... (app.py V3.1.9)"] 
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
print(f"DEBUG (app.py V3.1.9): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
