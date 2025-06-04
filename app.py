# app.py
# Version 3.1.12:
# - Added an instructional helper text on the main page for new users.
# Previous versions:
# - Version 3.1.11: Integrated dynamic spinner messages by calling ui_manager.get_random_spinner_message().

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
if 'app_config' not in st.session_state or st.session_state.app_config is None:
    print(f"DEBUG (app.py V{config.APP_VERSION}): 'app_config' not in session_state or is None. Calling config.load_config().")
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config is None:
        st.error("CRITICAL FAILURE: config.load_config() returned None. Application cannot start. Check secrets.toml and logs.")
        print(f"CRITICAL ERROR (app.py V{config.APP_VERSION}): config.load_config() returned None. Stopping execution.")
        st.stop()
    else:
        print(f"DEBUG (app.py V{config.APP_VERSION}): config.load_config() successful. AppConfig type: {type(st.session_state.app_config)}")
        if hasattr(st.session_state.app_config, 'gsheets'):
            print(f"DEBUG (app.py V{config.APP_VERSION}): Newly loaded app_config.gsheets type: {type(st.session_state.app_config.gsheets)}")
        else:
            print(f"ERROR (app.py V{config.APP_VERSION}): Newly loaded app_config does NOT have 'gsheets' attribute immediately after load!")


cfg: Optional[config.AppConfig] = st.session_state.app_config

if cfg is None:
    st.error("CRITICAL ERROR: Application configuration (cfg) is None. Halting.")
    print(f"CRITICAL ERROR (app.py V{config.APP_VERSION}): cfg is None. Halting.")
    st.stop()

if not hasattr(cfg, 'gsheets') or cfg.gsheets is None: 
    st.error("CRITICAL ERROR: Configuration 'cfg' missing 'gsheets' attribute or 'cfg.gsheets' is None. Check config.py. Halting.")
    print(f"CRITICAL ERROR (app.py V{config.APP_VERSION}): cfg.gsheets missing/None. cfg type: {type(cfg)}. hasattr: {hasattr(cfg, 'gsheets')}. Halting.")
    st.stop()
print(f"DEBUG (app.py V{config.APP_VERSION}): cfg and cfg.gsheets valid. LLM Thresh: {cfg.llm.llm_throttling_threshold_results}")


# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state:
    if hasattr(cfg, 'llm') and cfg.llm:
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = (
            (llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or
            (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key)
        )
    else: st.session_state.llm_globally_enabled = False
    print(f"DEBUG (app.py V{config.APP_VERSION}): llm_globally_enabled set in session_state: {st.session_state.llm_globally_enabled}")

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
print(f"DEBUG (app.py V{config.APP_VERSION}): Other session state keys initialized/verified.") 

# --- Google Sheets Setup ---
gsheets_secrets_present = bool(cfg.gsheets.service_account_info and \
                           (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name))
print(f"DEBUG (app.py V{config.APP_VERSION}): gsheets_secrets_present = {gsheets_secrets_present}") 

if not st.session_state.sheet_connection_attempted_this_session:
    print(f"DEBUG (app.py V{config.APP_VERSION}): Attempting Google Sheets connection for the first time this session.") 
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( 
                cfg.gsheets.service_account_info, cfg.gsheets.spreadsheet_id,
                cfg.gsheets.spreadsheet_name, cfg.gsheets.worksheet_name
            )
            if st.session_state.gs_worksheet:
                print(f"DEBUG (app.py V{config.APP_VERSION}): Successfully got worksheet: {st.session_state.gs_worksheet.title if hasattr(st.session_state.gs_worksheet, 'title') else type(st.session_state.gs_worksheet)}") 
                data_storage.ensure_master_header(st.session_state.gs_worksheet) 
                st.session_state.sheet_writing_enabled = True
            else:
                st.session_state.gsheets_error_message = st.session_state.get("gsheets_error_message", "Google Sheets connection failed (worksheet object is None).")
        except Exception as e_gs_setup:
            st.session_state.gsheets_error_message = f"Error during Google Sheets setup: {e_gs_setup}"
            print(f"DEBUG (app.py V{config.APP_VERSION}): EXCEPTION during Google Sheets setup: {e_gs_setup}") 
            print(traceback.format_exc())
    else:
        st.session_state.gsheets_error_message = "Google Sheets not fully configured. Data storage to Sheets disabled."
else:
    print(f"DEBUG (app.py V{config.APP_VERSION}): Google Sheets connection already attempted.") 

# --- UI Rendering ---
st.title("D.O.R.A 🔮")
st.markdown("The **Domain**-wide **Overview** For **Research** **Agent**") # Your existing subtitle

# --- START: NEW HELPER TEXT ---
# Display this only if no results or summary are currently in session state,
# and no processing is currently indicated by a run_complete_status_message
if not st.session_state.get("results_data") and \
   not st.session_state.get("consolidated_summary_text") and \
   not st.session_state.get('run_complete_status_message'): # Avoid showing after a run that yielded no results
    st.markdown("> Welcome to D.O.R.A.! To begin your research, please enter keywords and any specific queries in the sidebar on the left.")
st.markdown("---") # Visual separator
# --- END: NEW HELPER TEXT ---

print(f"DEBUG (app.py V{config.APP_VERSION}): Main UI title and helper text rendered.") 

keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, 
    st.session_state.gsheets_error_message, 
    st.session_state.sheet_writing_enabled
)
ui_manager.apply_custom_css() 
status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()

# --- Main Processing Logic ---
if start_button:
    print(f"DEBUG (app.py V{config.APP_VERSION}): 'Start Search & Analysis' button pressed.") 
    st.session_state.processing_log = [f"Processing initiated at {time.strftime('%Y-%m-%d %H:%M:%S')}... (app.py V{config.APP_VERSION})"] 
    st.session_state.results_data = []
    st.session_state.consolidated_summary_text = None
    st.session_state.focused_summary_sources = []
    st.session_state.initial_keywords_for_display = set()
    st.session_state.llm_generated_keywords_set_for_display = set()
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.run_complete_status_message = None 
    
    print(f"DEBUG (app.py V{config.APP_VERSION}): Session state for run initialized. Batch timestamp: {st.session_state.batch_timestamp_for_excel}") 

    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list 

    active_llm_extract_queries = [q for q in llm_extract_queries_list if q.strip()]
    print(f"DEBUG (app.py V{config.APP_VERSION}): Calling process_manager.run_search_and_analysis. Active Queries: {active_llm_extract_queries}, Num Results: {num_results}") 
    
    spinner_message = ui_manager.get_random_spinner_message()
    print(f"DEBUG (app.py V{config.APP_VERSION}): Using spinner message: '{spinner_message}'")

    with st.spinner(spinner_message): 
        try:
            log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
                app_config=cfg, 
                keywords_input=keywords_input,
                llm_extract_queries_input=active_llm_extract_queries,
                num_results_wanted_per_keyword=num_results,
                gs_worksheet=st.session_state.gs_worksheet,
                sheet_writing_enabled=st.session_state.sheet_writing_enabled,
                gsheets_secrets_present=gsheets_secrets_present
            )
            print(f"DEBUG (app.py V{config.APP_VERSION}): Returned from process_manager.run_search_and_analysis.") 
            st.session_state.processing_log = log
            st.session_state.results_data = data
            st.session_state.consolidated_summary_text = summary
            st.session_state.focused_summary_sources = focused_sources
            st.session_state.initial_keywords_for_display = initial_kws_display
            st.session_state.llm_generated_keywords_set_for_display = llm_kws_display
            print(f"DEBUG (app.py V{config.APP_VERSION}): Session state updated with results from process_manager.") 

            final_status_log_entry = next((item for item in reversed(log) if isinstance(item, str) and item.startswith("LOG_STATUS:")), None)
            if final_status_log_entry:
                st.session_state.run_complete_status_message = final_status_log_entry
            else:
                st.session_state.run_complete_status_message = "LOG_STATUS:WARNING:Processing finished, but final status unclear from logs."
                print(f"DEBUG (app.py V{config.APP_VERSION}): No LOG_STATUS found in log. Defaulting status.")
        except Exception as e_process_mgr:
            detailed_error = f"APP_PY_ERROR: Main process failed: {type(e_process_mgr).__name__} - {e_process_mgr}\n{traceback.format_exc()}"
            status_message_placeholder.error(f"A critical error occurred in processing: {e_process_mgr}") 
            print(f"CRITICAL ERROR (app.py V{config.APP_VERSION}): Exception in process_manager call: {detailed_error}") 
            current_log_val = st.session_state.get('processing_log', [])
            if not isinstance(current_log_val, list): current_log_val = [str(current_log_val)] 
            current_log_val.append(detailed_error)
            st.session_state.processing_log = current_log_val
            st.session_state.run_complete_status_message = f"LOG_STATUS:APP_PY_CRITICAL_ERROR:{detailed_error}"

# Display final status message from processing run (if any)
if st.session_state.get('run_complete_status_message'):
    status_parts = st.session_state.run_complete_status_message.split(':', 2)
    status_type = "INFO"; status_text = st.session_state.run_complete_status_message
    if len(status_parts) > 1: status_type = status_parts[1]
    if len(status_parts) > 2: status_text = status_parts[2]
    
    print(f"DEBUG (app.py V{config.APP_VERSION}): Displaying final status. Type: {status_type}, Text: {status_text[:100]}")
    if status_type == "SUCCESS": status_message_placeholder.success(status_text)
    elif status_type == "WARNING": status_message_placeholder.warning(status_text)
    elif "ERROR" in status_type: status_message_placeholder.error(status_text) 
    else: status_message_placeholder.info(status_text)
    st.session_state.run_complete_status_message = None 

print(f"DEBUG (app.py V{config.APP_VERSION}): Entering Display Results and Logs section.") 

with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"):
        st.markdown("---")
        try:
            df_item_details = excel_handler.prepare_item_details_df(
                st.session_state.get("results_data", []), 
                st.session_state.get('last_extract_queries', ["", ""]) 
            )
            df_consolidated_summary_excel = None
            if st.session_state.consolidated_summary_text:
                last_queries_for_excel = st.session_state.get('last_extract_queries', ["", ""])
                q1_text_for_excel = last_queries_for_excel[0] if last_queries_for_excel and last_queries_for_excel[0] else None
                q2_text_for_excel = last_queries_for_excel[1] if last_queries_for_excel and len(last_queries_for_excel) > 1 and last_queries_for_excel[1] else None
                focused_count_for_excel = None
                if isinstance(st.session_state.focused_summary_sources, list): 
                    focused_count_for_excel = len(st.session_state.focused_summary_sources)
                
                df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                    consolidated_summary_text=st.session_state.consolidated_summary_text,
                    results_data_count=len(st.session_state.get("results_data", [])),
                    last_keywords=st.session_state.get('last_keywords', ""), 
                    primary_llm_extract_query=q1_text_for_excel,
                    secondary_llm_extract_query=q2_text_for_excel, 
                    batch_timestamp=st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y-%m-%d %H:%M:%S')),
                    focused_summary_source_count=focused_count_for_excel
                )
            excel_file_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
            filename_timestamp = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d%H%M%S')).replace(":", "").replace("-", "").replace(" ", "_") 
            st.download_button(
                label="📥 Download Results as Excel", data=excel_file_bytes,
                file_name=f"dora_results_{filename_timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="download_excel_button" 
            )
        except Exception as e_excel:
            st.error(f"Error preparing Excel download: {e_excel}")
            print(f"ERROR (app.py V{config.APP_VERSION}): Exception during Excel preparation: {e_excel}") 
            print(traceback.format_exc())

    ui_manager.display_consolidated_summary_and_sources(
        st.session_state.consolidated_summary_text,
        st.session_state.focused_summary_sources,
        st.session_state.get('last_extract_queries', ["", ""]) 
    )
    ui_manager.display_individual_results()

with log_container:
    ui_manager.display_processing_log() 

st.markdown("---")
st.caption(f"D.O.R.A v{config.APP_VERSION}") 
print(f"DEBUG (app.py V{config.APP_VERSION}): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
