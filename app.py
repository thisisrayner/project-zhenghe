# app.py (Temporary Debugging Modifications near the top)
# Version 1.5.2d (d for debug)

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage
import time

# --- Page Configuration ---
st.set_page_config(page_title="Keyword Search & Analysis Tool", page_icon="🔎", layout="wide")

st.subheader("DEBUG: App Rerun / Script Start") # To see when script reruns

# --- Load Configuration ---
cfg = config.load_config()
if not cfg:
    st.error("CRITICAL: Configuration failed to load. Application cannot proceed.")
    st.stop()
else:
    st.success("DEBUG: Configuration loaded successfully.")
    # print("DEBUG cfg.gsheets.service_account_info:", cfg.gsheets.service_account_info) # Be careful printing secrets to console
    # print("DEBUG cfg.gsheets.spreadsheet_name:", cfg.gsheets.spreadsheet_name)
    # print("DEBUG cfg.gsheets.worksheet_name:", cfg.gsheets.worksheet_name)


# --- Session State Initialization ---
default_session_state = {
    'processing_log': [], 'results_data': [], 'last_keywords': "",
    'last_extract_query': "", 'consolidated_summary': None,
    'gs_worksheet': None,
    'sheet_writing_enabled': False,
    'sheet_connection_attempted': False # New flag for debug
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

st.subheader("DEBUG: Attempting Google Sheets Connection...")
# --- Google Sheets Setup (Attempt once per session or if config changes) ---
if cfg.gsheets.service_account_info and cfg.gsheets.spreadsheet_name:
    st.write("DEBUG: Service account info and spreadsheet name ARE present in config.")
    st.session_state.sheet_writing_enabled = True # Tentatively enable
    
    # Force connection attempt for debugging if not already attempted this session
    if not st.session_state.get('sheet_connection_attempted_this_run'): # Prevents multiple attempts in one rerun
        st.write("DEBUG: `gs_worksheet` is None or connection not attempted. Attempting connection now...")
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
            cfg.gsheets.service_account_info,
            cfg.gsheets.spreadsheet_name,
            cfg.gsheets.worksheet_name
        )
        st.session_state.sheet_connection_attempted_this_run = True # Mark as attempted for this specific rerun

        if st.session_state.gs_worksheet:
            st.success("DEBUG: `get_gspread_worksheet` returned a worksheet object.")
            st.write(f"DEBUG: Connected to Sheet: {st.session_state.gs_worksheet.spreadsheet.title}, Worksheet: {st.session_state.gs_worksheet.title}")
            # Try ensuring header immediately after successful connection for debug
            try:
                st.write("DEBUG: Attempting to ensure header...")
                data_storage.ensure_header(st.session_state.gs_worksheet)
                st.write("DEBUG: `ensure_header` call completed.")
            except Exception as e_header:
                st.error(f"DEBUG: Error during `ensure_header`: {e_header}")
                st.session_state.sheet_writing_enabled = False
        else:
            st.error("DEBUG: `get_gspread_worksheet` returned None. Connection failed.")
            st.session_state.sheet_writing_enabled = False # Definitely disable if connection failed
else:
    st.warning("DEBUG: Service account info OR spreadsheet name missing in config. Sheets integration will be disabled.")
    st.session_state.sheet_writing_enabled = False

# Reset the attempt flag for the *next* full script rerun (not for widget interaction reruns)
# This is tricky; for now, we just let it attempt once per "fresh" load or major action.
# A more robust way might involve a button "Test Sheet Connection".

st.write(f"DEBUG: `sheet_writing_enabled` is now: {st.session_state.sheet_writing_enabled}")
st.write(f"DEBUG: `gs_worksheet` object is: {'Exists' if st.session_state.gs_worksheet else 'None'}")
st.markdown("---")


# --- UI Layout ---
st.title("Keyword Search & Analysis Tool 🔎📝")
# ... (rest of your app.py code from v1.5.1) ...
# ... ensure the sidebar caption for sheets warning uses st.session_state.sheet_writing_enabled ...
with st.sidebar:
    # ...
    if not st.session_state.sheet_writing_enabled:
        st.sidebar.warning("⚠️ Google Sheets integration not configured or connection failed.") # Changed to warning for visibility
    # ...

# (The rest of your app.py Version 1.5.1)
# ...
# end of app.py
