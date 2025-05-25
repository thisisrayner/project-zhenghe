# modules/data_storage.py
# Version 1.0: Initial implementation for Google Sheets integration.

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st # For caching
from typing import Dict, List, Optional, Any

# --- Google Sheets Client Initialization (Cached) ---
@st.cache_resource # Cache the gspread client and worksheet object
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_name: Optional[str],
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    """
    Authorizes gspread client with service account and returns the specified worksheet.
    Caches the worksheet object.
    """
    if not service_account_info:
        st.error("Google Sheets: Service account information not provided in secrets.")
        return None
    if not spreadsheet_name:
        st.error("Google Sheets: Spreadsheet name not provided in secrets.")
        return None

    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file' # Required for opening by name/creating if not exists
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)

        # Try to open the spreadsheet by name
        try:
            spreadsheet = client.open(spreadsheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            st.warning(f"Spreadsheet '{spreadsheet_name}' not found. It will NOT be created automatically by this script.")
            # If you want to create it:
            # st.info(f"Creating spreadsheet '{spreadsheet_name}'...")
            # spreadsheet = client.create(spreadsheet_name)
            # spreadsheet.share(service_account_info["client_email"], perm_type="user", role="writer") # Share back
            return None # For now, we require the sheet to exist and be shared.

        # Try to get the worksheet by name, or the first one if specific name not found/defaulted
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            if worksheet_name == "Sheet1": # Common default
                st.info(f"Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'. Using the first sheet instead.")
                worksheet = spreadsheet.sheet1
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'. Data will not be written.")
                return None
        
        st.success(f"Successfully connected to Google Sheet: '{spreadsheet_name}' -> Worksheet: '{worksheet.title}'")
        return worksheet

    except Exception as e:
        st.error(f"Google Sheets connection error: {e}")
        return None

# --- Header Row ---
# Define your desired header row structure
# This should match the keys you plan to extract from item_data_val in app.py
EXPECTED_HEADER = [
    "Timestamp", "Keyword Searched", "URL",
    "Search Result Title", "Search Result Snippet",
    "Scraped Page Title", "Scraped Meta Description",
    "Scraped OG Title", "Scraped OG Description",
    "LLM Summary", "LLM Extracted Info (Query)", "LLM Extraction Query",
    "Scraping Error",
    # "Full Main Text" # Optional: usually too long for a sheet cell
]

def ensure_header(worksheet: gspread.Worksheet) -> None:
    """
    Checks if the header row exists and matches EXPECTED_HEADER. If not, appends it.
    """
    try:
        # Fetch current header (first row)
        current_header = worksheet.row_values(1)
        if not current_header or all(cell == '' for cell in current_header): # If sheet is empty or first row blank
            worksheet.append_row(EXPECTED_HEADER, value_input_option='USER_ENTERED')
            print(f"Appended header row to worksheet '{worksheet.title}'.")
        elif current_header != EXPECTED_HEADER:
            # Basic check; for more robustness, compare sets of column names
            # or allow for different order if necessary.
            st.warning(f"Worksheet '{worksheet.title}' header mismatch. Expected: {EXPECTED_HEADER}, Found: {current_header}. Data will still be appended.")
            # Optionally, you could decide to not append or try to map columns.
            # For simplicity, we'll just append based on our expected order.
    except gspread.exceptions.APIError as e:
        if 'exceeds grid limits' in str(e).lower(): # Typically for an empty sheet
            worksheet.append_row(EXPECTED_HEADER, value_input_option='USER_ENTERED')
            print(f"Appended header row to empty worksheet '{worksheet.title}'.")
        else:
            st.error(f"Google Sheets API error checking/writing header: {e}")
    except Exception as e:
        st.error(f"Unexpected error checking/writing header in Google Sheets: {e}")


# --- Write Data Function ---
def write_data_to_sheet(
    worksheet: gspread.Worksheet,
    item_data_list: List[Dict[str, Any]], # Expecting a list of processed items
    extraction_query_text: Optional[str] = None # To include the query if specific info was extracted
) -> int:
    """
    Writes a list of processed data items to the Google Sheet.
    Each item in item_data_list should be a dictionary.
    Returns the number of rows successfully written.
    """
    if not worksheet:
        st.error("Google Sheets: No valid worksheet provided. Cannot write data.")
        return 0

    rows_to_append = []
    for item in item_data_list:
        # Map item_data keys to the EXPECTED_HEADER order
        # Handle missing keys gracefully by using .get() with a default
        row = [
            item.get("timestamp", ""),
            item.get("keyword_searched", ""),
            item.get("url", ""),
            item.get("search_title", ""),
            item.get("search_snippet", ""),
            item.get("scraped_title", ""),
            item.get("scraped_meta_description", ""),
            item.get("scraped_og_title", ""),
            item.get("scraped_og_description", ""),
            item.get("llm_summary", ""),
            item.get("llm_extracted_info", ""),
            extraction_query_text if item.get("llm_extracted_info") else "", # Only add query if info was extracted
            item.get("scraping_error", ""),
            # item.get("scraped_main_text", "")[:500] + "..." if item.get("scraped_main_text") else "" # Example if including partial main text
        ]
        # Ensure the row has the same number of columns as the header
        if len(row) != len(EXPECTED_HEADER):
            # This can happen if item_data is missing expected keys and no defaults are provided,
            # or if EXPECTED_HEADER structure changes without updating this mapping.
            # For now, we'll pad or truncate, but ideally, the mapping should be robust.
            # This is a simplistic fix; robust solution would involve careful mapping.
            padded_row = ["" for _ in EXPECTED_HEADER]
            for i, val in enumerate(row):
                if i < len(padded_row):
                    padded_row[i] = val
            row = padded_row[:len(EXPECTED_HEADER)]

        rows_to_append.append(row)

    if not rows_to_append:
        st.info("Google Sheets: No data to write.")
        return 0

    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        st.success(f"Successfully wrote {len(rows_to_append)} row(s) to Google Sheet: '{worksheet.spreadsheet.title}' -> '{worksheet.title}'.")
        return len(rows_to_append)
    except Exception as e:
        st.error(f"Google Sheets: Error writing data: {e}")
        return 0

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets)")

    # This test requires Google Sheets secrets to be configured in .streamlit/secrets.toml
    # And the sheet to be shared with the service account email.
    try:
        from config import load_config # Assuming config.py is in the parent directory or Python path
        cfg_test = load_config()
        if cfg_test and cfg_test.gsheets.service_account_info and cfg_test.gsheets.spreadsheet_name:
            st.info("Attempting to connect to Google Sheets with loaded config...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.gsheets.service_account_info,
                cfg_test.gsheets.spreadsheet_name,
                cfg_test.gsheets.worksheet_name
            )
            if worksheet_test:
                ensure_header(worksheet_test) # Test header function
                
                st.subheader("Test Data Writing")
                if st.button("Write Sample Data"):
                    sample_items = [
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "test keyword 1",
                            "url": "http://example.com/1", "search_title": "Example 1",
                            "scraped_title": "Scraped Example 1", "llm_summary": "This is a test summary 1.",
                            "scraping_error": ""
                        },
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "test keyword 2",
                            "url": "http://example.com/2", "search_title": "Example 2",
                            "scraped_title": "Scraped Example 2", "llm_summary": "This is a test summary 2.",
                            "llm_extracted_info": "Found: Test value", "scraping_error": "Test scrape error"
                        }
                    ]
                    rows_written = write_data_to_sheet(worksheet_test, sample_items, extraction_query_text="What is test value?")
                    st.write(f"{rows_written} sample rows written.")
        else:
            st.warning("Google Sheets configuration missing in secrets or failed to load. Cannot run full test.")
    except ImportError:
        st.error("Could not import 'config' module. Place it appropriately or adjust import path for testing.")
    except Exception as e:
        st.error(f"An error occurred during test setup: {e}")

# end of modules/data_storage.py
