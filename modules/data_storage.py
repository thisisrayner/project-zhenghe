# modules/data_storage.py
# Version 1.1: Prioritize opening Google Sheet by ID if available.

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time # For timestamp in test block

# --- Google Sheets Client Initialization (Cached) ---
@st.cache_resource
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],       # Prioritized
    spreadsheet_name: Optional[str],   # Fallback
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    if not service_account_info:
        st.error("Google Sheets: Service account information not provided in secrets.")
        return None
    if not spreadsheet_id and not spreadsheet_name: # Check if at least one identifier is present
        st.error("Google Sheets: Neither Spreadsheet ID nor Name provided in secrets.")
        return None

    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = None

        if spreadsheet_id:
            st.info(f"Attempting to open spreadsheet by ID: {spreadsheet_id}")
            try:
                spreadsheet = client.open_by_key(spreadsheet_id) # open_by_key uses the ID
            except gspread.exceptions.APIError as e_id: # More specific exception for API errors
                st.error(f"Google Sheets: APIError opening by ID '{spreadsheet_id}': {e_id}. Check ID and sharing.")
                # If open by ID fails, optionally try by name if name is also provided
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed, attempting to open by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None # No name to fallback to
            except Exception as e_id_other: # Catch other potential exceptions
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed, attempting to open by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None

        if not spreadsheet and spreadsheet_name: # Fallback to opening by name
            st.info(f"Attempting to open spreadsheet by name: {spreadsheet_name}")
            try:
                spreadsheet = client.open(spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Spreadsheet '{spreadsheet_name}' not found by name. Please verify name or use SPREADSHEET_ID.")
                return None
            except Exception as e_name:
                st.error(f"Google Sheets: Error opening by name '{spreadsheet_name}': {e_name}")
                return None
        
        if not spreadsheet: # If still no spreadsheet after trying ID and/or name
            st.error("Google Sheets: Could not open spreadsheet using provided ID or Name.")
            return None

        # Try to get the worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            if worksheet_name == "Sheet1":
                st.info(f"Worksheet '{worksheet_name}' not found in '{spreadsheet.title}'. Using the first sheet.")
                worksheet = spreadsheet.sheet1
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in '{spreadsheet.title}'.")
                return None
        
        st.success(f"Successfully connected to Google Sheet: '{spreadsheet.title}' -> Worksheet: '{worksheet.title}'")
        return worksheet

    except Exception as e:
        st.error(f"Google Sheets connection/setup error: {e}")
        return None

# ... (EXPECTED_HEADER, ensure_header, write_data_to_sheet functions remain the same as data_storage.py v1.0) ...
EXPECTED_HEADER = [
    "Timestamp", "Keyword Searched", "URL", "Search Result Title", "Search Result Snippet",
    "Scraped Page Title", "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description",
    "LLM Summary", "LLM Extracted Info (Query)", "LLM Extraction Query", "Scraping Error",
]
def ensure_header(worksheet: gspread.Worksheet) -> None:
    try:
        current_header = worksheet.row_values(1)
        if not current_header or all(cell == '' for cell in current_header):
            worksheet.append_row(EXPECTED_HEADER, value_input_option='USER_ENTERED')
        elif current_header != EXPECTED_HEADER:
            st.warning(f"Worksheet '{worksheet.title}' header mismatch.")
    except gspread.exceptions.APIError as e:
        if 'exceeds grid limits' in str(e).lower():
            worksheet.append_row(EXPECTED_HEADER, value_input_option='USER_ENTERED')
        else: st.error(f"Google Sheets API error checking/writing header: {e}")
    except Exception as e: st.error(f"Unexpected error checking/writing header: {e}")

def write_data_to_sheet(worksheet: gspread.Worksheet, item_data_list: List[Dict[str, Any]], extraction_query_text: Optional[str] = None) -> int:
    if not worksheet: st.error("Google Sheets: No valid worksheet provided."); return 0
    rows_to_append = []
    for item in item_data_list:
        row = [
            item.get("timestamp", ""), item.get("keyword_searched", ""), item.get("url", ""),
            item.get("search_title", ""), item.get("search_snippet", ""),
            item.get("scraped_title", ""), item.get("scraped_meta_description", ""),
            item.get("scraped_og_title", ""), item.get("scraped_og_description", ""),
            item.get("llm_summary", ""), item.get("llm_extracted_info", ""),
            extraction_query_text if item.get("llm_extracted_info") else "",
            item.get("scraping_error", ""),
        ]
        if len(row) != len(EXPECTED_HEADER):
            padded_row = ["" for _ in EXPECTED_HEADER];
            for i, val in enumerate(row):
                if i < len(padded_row): padded_row[i] = val
            row = padded_row[:len(EXPECTED_HEADER)]
        rows_to_append.append(row)
    if not rows_to_append: st.info("Google Sheets: No data to write."); return 0
    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        st.success(f"Successfully wrote {len(rows_to_append)} row(s) to Google Sheet: '{worksheet.spreadsheet.title}' -> '{worksheet.title}'.")
        return len(rows_to_append)
    except Exception as e: st.error(f"Google Sheets: Error writing data: {e}"); return 0

# ... (if __name__ == '__main__' test block - update to pass spreadsheet_id)
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.1)")
    try:
        # Assuming config.py is in parent directory, adjust path if needed for standalone test
        # from .. import config # Would be typical if modules is a proper package installed
        from config import load_config # Simpler for direct run if config.py is findable
        cfg_test = load_config()
        if cfg_test and cfg_test.gsheets.service_account_info and \
           (cfg_test.gsheets.spreadsheet_id or cfg_test.gsheets.spreadsheet_name):
            st.info("Attempting to connect to Google Sheets with loaded config...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.gsheets.service_account_info,
                cfg_test.gsheets.spreadsheet_id, # Pass ID
                cfg_test.gsheets.spreadsheet_name, # Pass Name as fallback
                cfg_test.gsheets.worksheet_name
            )
            if worksheet_test:
                ensure_header(worksheet_test)
                st.subheader("Test Data Writing")
                if st.button("Write Sample Data"):
                    sample_items = [
                        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "test id 1", "url": "http://example.com/id1", "scraped_title": "ID Test 1"},
                        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "test id 2", "url": "http://example.com/id2", "scraping_error": "ID Test Error"}
                    ]
                    rows_written = write_data_to_sheet(worksheet_test, sample_items)
                    st.write(f"{rows_written} sample rows written.")
        else:
            st.warning("Google Sheets config (service account, ID/Name) missing or failed to load.")
    except ImportError: st.error("Could not import 'config' module for testing.")
    except Exception as e: st.error(f"An error occurred during test setup: {e}")
# end of modules/data_storage.py
