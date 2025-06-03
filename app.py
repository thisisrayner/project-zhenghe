# app.py
# Version 3.1.7:
# - Stores the loaded app configuration (cfg) into st.session_state.app_config.
# - Initializes st.session_state.llm_globally_enabled based on this config.
# Previous versions:
# - Version 3.1.6: Fixed ValueError in sidebar debug log display.

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
if 'app_config' not in st.session_state or st.session_state.app_config is None: # Load only if not already loaded
    print("DEBUG (app.py): Attempting to load config and store in session_state.")
    st.session_state.app_config = config.load_config() 
    if st.session_state.app_config:
        print(f"DEBUG (app.py): Config loaded into session_state. App Version: {st.session_state.app_config.APP_VERSION if hasattr(st.session_state.app_config, 'APP_VERSION') else config.APP_VERSION}")
    else:
        # config.load_config() should ideally handle its own st.error if it fails critically
        # This is a fallback if it returns None without explicit error to user.
        st.error("CRITICAL: Application configuration failed to load from config.load_config(). Check secrets.toml and logs.")
        print("CRITICAL ERROR (app.py): config.load_config() returned None during session state init. Stopping execution.")
        st.stop()

cfg: Optional[config.AppConfig] = st.session_state.app_config
# Double check cfg is not None after trying to load from session_state, though load_config() itself should stop if critical.
if not cfg:
    st.error("CRITICAL: Application configuration is None after attempting to load from session_state. This should not happen. Check logs.")
    print("CRITICAL ERROR (app.py): cfg is None after session_state assignment. Stopping execution.")
    st.stop()

print(f"DEBUG (app.py): Using AppConfig from session_state. LLM Throttling Threshold: {cfg.llm.llm_throttling_threshold_results}, Delay: {cfg.llm.llm_item_request_delay_seconds}") 

# --- Determine and store global LLM availability status IN SESSION STATE ---
if 'llm_globally_enabled' not in st.session_state: # Calculate only once per session start effectively
    if cfg and cfg.llm: # cfg should always be valid here due to checks above
        llm_cfg_init = cfg.llm
        st.session_state.llm_globally_enabled = (
            (llm_cfg_init.provider == "google" and llm_cfg_init.google_gemini_api_key) or
            (llm_cfg_init.provider == "openai" and llm_cfg_init.openai_api_key)
        )
        print(f"DEBUG (app.py): llm_globally_enabled set in session_state: {st.session_state.llm_globally_enabled}")
    else:
        st.session_state.llm_globally_enabled = False
        print(f"DEBUG (app.py): llm_globally_enabled set to False (no cfg.llm) in session_state.")


# --- Session State Initialization (Defaults for other keys) ---
default_session_state_keys: Dict[str, Any] = { # Renamed to avoid conflict
    'processing_log': [], 'results_data': [], 'last_keywords': "",
    'last_extract_queries': ["", ""], 'consolidated_summary_text': None,
    'focused_summary_sources': [], 'gs_worksheet': None, 'sheet_writing_enabled': False,
    'sheet_connection_attempted_this_session': False, 'gsheets_error_message': None,
    'initial_keywords_for_display': set(), 'llm_generated_keywords_set_for_display': set(),
    'batch_timestamp_for_excel': None, 'run_complete_status_message': None 
}
for key, default_value in default_session_state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
print("DEBUG (app.py): Other session state keys initialized/verified.") 

# ... (Rest of your app.py code from v3.1.6 for Google Sheets, UI Rendering, Main Processing Logic) ...
# Ensure that ui_manager.render_sidebar(cfg, ...) still receives the cfg object
# as it uses it for things other than just llm_key_available (e.g., provider name, model name display).

# --- Google Sheets Setup ---
# (Ensure this logic uses cfg from st.session_state.app_config)
gsheets_secrets_present = bool(cfg.sheets.service_account_info and \
                           (cfg.sheets.spreadsheet_id or cfg.sheets.spreadsheet_name)) # Corrected to cfg.sheets
print(f"DEBUG (app.py): gsheets_secrets_present = {gsheets_secrets_present}") 

if not st.session_state.sheet_connection_attempted_this_session:
    print("DEBUG (app.py): Attempting Google Sheets connection for the first time this session.") 
    st.session_state.sheet_connection_attempted_this_session = True
    st.session_state.sheet_writing_enabled = False 
    st.session_state.gsheets_error_message = None 
    if gsheets_secrets_present:
        try:
            st.session_state.gs_worksheet = data_storage.get_gspread_worksheet( 
                cfg.sheets.service_account_info, cfg.sheets.spreadsheet_id, # Corrected to cfg.sheets
                cfg.sheets.spreadsheet_name, cfg.sheets.worksheet_name # Corrected to cfg.sheets
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
    else:
        st.session_state.gsheets_error_message = "Google Sheets not fully configured. Data storage to Sheets disabled."
        print(f"DEBUG (app.py): {st.session_state.gsheets_error_message}") 
else:
    print("DEBUG (app.py): Google Sheets connection already attempted.") 

# --- UI Rendering ---
st.title("D.O.R.A 🔮")
st.markdown("The **Research** **Agent** For **Domain**-Wide **Overview** and Insights.")
print("DEBUG (app.py): Main UI title rendered.") 

keywords_input, num_results, llm_extract_queries_list, start_button = ui_manager.render_sidebar(
    cfg, # Pass the cfg object loaded from session_state
    st.session_state.gsheets_error_message, 
    st.session_state.sheet_writing_enabled
)
# ... (rest of app.py including main processing loop, results display, etc. from v3.1.6) ...
# The important part is that ui_manager functions will now use st.session_state.llm_globally_enabled
# and st.session_state.app_config.

# (Ensure the parts of your app.py for Start Button logic, process_manager call,
# Excel download, and final UI updates remain the same as your v3.1.6,
# as those were not the focus of this specific fix)

# Example of how the results display section might look now (simplified)
status_message_placeholder = st.empty() 
results_container = st.container()
log_container = st.container()

if start_button:
    # ... (reset session state for new run) ...
    st.session_state.processing_log = ["Processing initiated... (app.py)"] 
    st.session_state.results_data = [] # etc.
    st.session_state.batch_timestamp_for_excel = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.last_keywords = keywords_input
    st.session_state.last_extract_queries = llm_extract_queries_list 
    
    with st.spinner("D.O.R.A. is thinking... please wait for all processing to complete."): 
        try:
            log, data, summary, initial_kws_display, llm_kws_display, focused_sources = process_manager.run_search_and_analysis(
                app_config=cfg, # Pass the cfg object
                # ... other args
                keywords_input=keywords_input,
                llm_extract_queries_input=[q for q in llm_extract_queries_list if q.strip()],
                num_results_wanted_per_keyword=num_results,
                gs_worksheet=st.session_state.gs_worksheet,
                sheet_writing_enabled=st.session_state.sheet_writing_enabled,
                gsheets_secrets_present=gsheets_secrets_present
            )
            st.session_state.processing_log = log
            st.session_state.results_data = data
            # ... (update other session state vars) ...
            st.session_state.focused_summary_sources = focused_sources # Ensure this is assigned

            # Logic to set st.session_state.run_complete_status_message from log
            final_status_log_entry = next((item for item in reversed(log) if isinstance(item, str) and item.startswith("LOG_STATUS:")), None)
            if final_status_log_entry: st.session_state.run_complete_status_message = final_status_log_entry
            else: st.session_state.run_complete_status_message = "LOG_STATUS:WARNING:Processing finished, status unclear."
        except Exception as e_pm:
            # ... (error handling for process_manager failure) ...
            st.session_state.run_complete_status_message = f"LOG_STATUS:APP_PY_CRITICAL_ERROR:Process manager failed: {e_pm}"


if st.session_state.get('run_complete_status_message'):
    # ... (display final status using status_message_placeholder) ...
    status_parts = st.session_state.run_complete_status_message.split(':', 2)
    status_type = status_parts[1] if len(status_parts) > 1 else "INFO"
    status_text = status_parts[2] if len(status_parts) > 2 else st.session_state.run_complete_status_message
    if status_type == "SUCCESS": status_message_placeholder.success(status_text)
    elif status_type == "WARNING": status_message_placeholder.warning(status_text)
    else: status_message_placeholder.error(status_text) # Includes ERROR and CRITICAL_ERROR
    st.session_state.run_complete_status_message = None 


with results_container:
    if st.session_state.get("results_data") or st.session_state.get("consolidated_summary_text"):
        # ... (Excel download button logic from v3.1.6) ...
        # Ensure this part uses st.session_state.app_config (cfg) if needed for any defaults,
        # though it mostly uses other session_state variables.
        try:
            df_item_details = excel_handler.prepare_item_details_df(st.session_state.get("results_data", []), st.session_state.last_extract_queries)
            df_consolidated_summary_excel = None
            if st.session_state.consolidated_summary_text:
                q1_excel = st.session_state.last_extract_queries[0] if st.session_state.last_extract_queries else None
                q2_excel = st.session_state.last_extract_queries[1] if len(st.session_state.last_extract_queries) > 1 else None
                focused_src_count = len(st.session_state.focused_summary_sources) if isinstance(st.session_state.focused_summary_sources, list) else None
                df_consolidated_summary_excel = excel_handler.prepare_consolidated_summary_df(
                    consolidated_summary_text=st.session_state.consolidated_summary_text,
                    results_data_count=len(st.session_state.get("results_data", [])),
                    last_keywords=st.session_state.last_keywords,
                    primary_llm_extract_query=q1_excel, secondary_llm_extract_query=q2_excel, 
                    batch_timestamp=st.session_state.get("batch_timestamp_for_excel", ""),
                    focused_summary_source_count=focused_src_count )
            excel_bytes = excel_handler.to_excel_bytes(df_item_details, df_consolidated_summary_excel)
            dl_ts = st.session_state.get("batch_timestamp_for_excel", time.strftime('%Y%m%d%H%M%S')).replace(":", "").replace("-", "").replace(" ", "_")
            st.download_button(label="📥 Download Results as Excel", data=excel_bytes, file_name=f"dora_results_{dl_ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        except Exception as e_excel_main: st.error(f"Error preparing Excel: {e_excel_main}")

    ui_manager.display_consolidated_summary_and_sources(
        st.session_state.consolidated_summary_text,
        st.session_state.focused_summary_sources, # Make sure this is correctly populated by PM
        st.session_state.last_extract_queries
    )
    ui_manager.display_individual_results() # This will now use st.session_state.llm_globally_enabled

with log_container:
    ui_manager.display_processing_log() 

st.markdown("---")
# Use APP_VERSION from the config module directly, or from cfg if AppConfig is extended to hold it.
# For now, assuming config.APP_VERSION is the single source of truth.
st.caption(f"D.O.R.A v{config.APP_VERSION}") 
print(f"DEBUG (app.py): Reached end of app.py script execution. D.O.R.A v{config.APP_VERSION}") 

# // end of app.py
