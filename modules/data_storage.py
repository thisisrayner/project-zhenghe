# modules/data_storage.py
# Version 1.5.1: Corrected keys for retrieving meta/OG data for GSheet export.
# Implements unified header and writes batch summary row before itemized data.

"""
Handles data storage operations, primarily focused on Google Sheets integration.

This module provides functions to:
- Connect to a specified Google Sheet using service account credentials.
- Ensure a master header row exists in the worksheet.
- Write batch summary information and detailed item data in a structured format.
It uses the `gspread` library for Google Sheets API interactions.
"""

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st # For @st.cache_resource and UI elements in direct test
from typing import Dict, List, Optional, Any
import time # For timestamp in test block

# --- Google Sheets Client Initialization (Cached) ---
@st.cache_resource
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],       # Primary identifier for the sheet
    spreadsheet_name: Optional[str],   # Fallback or for display purposes
    worksheet_name: str = "Sheet1"     # Default worksheet name if not specified
) -> Optional[gspread.Worksheet]:
    """
    Authorizes gspread client with service account info and returns the specified worksheet.

    Prioritizes opening the spreadsheet by its ID if provided; otherwise,
    falls back to opening by name. Caches the worksheet object for the session
    to avoid repeated authentications.

    Args:
        service_account_info: A dictionary containing the service account credentials.
        spreadsheet_id: The unique ID of the Google Spreadsheet.
        spreadsheet_name: The name of the Google Spreadsheet.
        worksheet_name: The name of the specific worksheet (tab) to open.

    Returns:
        A gspread.Worksheet object if connection and worksheet retrieval are successful,
        otherwise None. Errors are displayed using st.error/st.warning.
    """
    if not service_account_info:
        st.error("Google Sheets Error: Service account information not provided in secrets.")
        return None
    if not spreadsheet_id and not spreadsheet_name:
        st.error("Google Sheets Error: Neither Spreadsheet ID nor Spreadsheet Name provided in secrets.")
        return None

    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file'
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet: Optional[gspread.Spreadsheet] = None

        if spreadsheet_id:
            try:
                spreadsheet = client.open_by_key(spreadsheet_id)
            except gspread.exceptions.APIError as e_id_api:
                st.error(f"Google Sheets APIError opening by ID '{spreadsheet_id}': {e_id_api}. "
                           "Check ID, sharing permissions, and that Drive & Sheets APIs are enabled.")
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed for '{spreadsheet_id}'. Attempting by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None
            except Exception as e_id_other:
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None

        if not spreadsheet and spreadsheet_name:
            try:
                spreadsheet = client.open(spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Google Sheets Error: Spreadsheet '{spreadsheet_name}' not found by name. "
                           "Please verify the name or ensure SPREADSHEET_ID is correctly set in secrets.")
                return None
            except Exception as e_name:
                st.error(f"Google Sheets Error: Error opening by name '{spreadsheet_name}': {e_name}")
                return None
        
        if not spreadsheet:
            st.error("Google Sheets Error: Could not open spreadsheet using provided ID or Name.")
            return None

        worksheet_obj: Optional[gspread.Worksheet] = None # Renamed to avoid conflict with worksheet_name param
        try:
            worksheet_obj = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            if worksheet_name == "Sheet1" and spreadsheet.sheet1: # Check if sheet1 attribute exists
                st.info(f"Worksheet '{worksheet_name}' not found in '{spreadsheet.title}'. Using the first available sheet ('{spreadsheet.sheet1.title}').")
                worksheet_obj = spreadsheet.sheet1
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Data cannot be written to this tab.")
                return None
        
        return worksheet_obj

    except Exception as e:
        st.error(f"Google Sheets connection/setup main error: {e}")
        return None

# --- UNIFIED MASTER HEADER for the Google Sheet ---
MASTER_HEADER: List[str] = [
    "Record Type",
    "Batch Timestamp",
    "Batch Consolidated Summary",
    "Batch Topic/Keywords",
    "Items in Batch",
    "Item Timestamp",
    "Keyword Searched",
    "URL",
    "Search Result Title",
    "Search Result Snippet",
    "Scraped Page Title",
    "Scraped Meta Description",
    "Scraped OG Title",
    "Scraped OG Description",
    "Content Type", # Added Content Type as it's useful and likely scraped
    "LLM Summary (Individual)",
    "LLM Extracted Info (Query)",
    "LLM Extraction Query",
    "Scraping Error",
    "Main Text (Truncated)"
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    """
    Ensures the MASTER_HEADER is present in Row 1 of the worksheet.
    """
    try:
        current_row1_values = worksheet.row_values(1)
        if not current_row1_values or all(cell == '' for cell in current_row1_values):
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED')
            # st.info(f"Initialized worksheet '{worksheet.title}' with MASTER_HEADER in Row 1.") # Can be verbose
        elif current_row1_values != MASTER_HEADER:
            st.warning(
                f"Worksheet '{worksheet.title}' Row 1 header does not match the expected MASTER_HEADER. "
                "Data will be appended according to the new structure, which might misalign with existing data. "
                "Consider clearing the sheet or manually adjusting the header if this is unintended."
            )
    except gspread.exceptions.APIError as e:
        if 'exceeds grid limits' in str(e).lower() or isinstance(e, (gspread.exceptions.CellNotFound, IndexError)):
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED')
            # st.info(f"Initialized empty worksheet '{worksheet.title}' with MASTER_HEADER (APIError/IndexError catch).") # Can be verbose
        else:
            st.error(f"Google Sheets API error while ensuring header: {e}")
    except Exception as e:
        st.error(f"Unexpected error while ensuring header in Google Sheets: {e}")


# --- Write Data Functions ---
def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_data_list: List[Dict[str, Any]],
    extraction_query_text: Optional[str] = None,
    main_text_truncate_limit: int = 10000
) -> bool:
    """
    Writes a batch summary row followed by individual item detail rows to the Google Sheet.
    """
    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        return False

    rows_to_append: List[List[Any]] = []

    batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
    batch_summary_row_dict["Record Type"] = "Batch Summary"
    batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
    batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else "N/A or Error"
    batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
    batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
    
    rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    for item_detail in item_data_list:
        main_text = item_detail.get("scraped_main_text", "")
        truncated_main_text = (main_text[:main_text_truncate_limit] + "..." if (main_text and len(main_text) > main_text_truncate_limit) else main_text)
        
        item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
        item_row_dict["Record Type"] = "Item Detail"
        item_row_dict["Batch Timestamp"] = batch_timestamp
        
        item_row_dict["Item Timestamp"] = item_detail.get("timestamp", batch_timestamp)
        item_row_dict["Keyword Searched"] = item_detail.get("keyword_searched", "")
        item_row_dict["URL"] = item_detail.get("url", "")
        item_row_dict["Search Result Title"] = item_detail.get("search_title", "")
        item_row_dict["Search Result Snippet"] = item_detail.get("search_snippet", "")
        item_row_dict["Scraped Page Title"] = item_detail.get("scraped_title", "")
        
        # --- THE FIX ---
        item_row_dict["Scraped Meta Description"] = item_detail.get("meta_description", "") # Use 'meta_description'
        item_row_dict["Scraped OG Title"] = item_detail.get("og_title", "")                 # Use 'og_title'
        item_row_dict["Scraped OG Description"] = item_detail.get("og_description", "")     # Use 'og_description'
        # --- END OF FIX ---

        item_row_dict["Content Type"] = item_detail.get("content_type", "") # Added Content Type
        item_row_dict["LLM Summary (Individual)"] = item_detail.get("llm_summary", "")
        item_row_dict["LLM Extracted Info (Query)"] = item_detail.get("llm_extracted_info", "")
        item_row_dict["LLM Extraction Query"] = extraction_query_text if item_detail.get("llm_extracted_info") else ""
        item_row_dict["Scraping Error"] = item_detail.get("scraping_error", "")
        item_row_dict["Main Text (Truncated)"] = truncated_main_text
        
        rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    if not rows_to_append:
        # st.info("Google Sheets: No data (batch summary or items) formatted for writing.") # Can be verbose
        return False
        
    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Google Sheets Error: Failed to write batch data rows: {e}")
        return False

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.1 - Key Fix)")
    try:
        from config import load_config # Assumes config.py is in PYTHONPATH or same dir for test
        cfg_test = load_config()
        
        if cfg_test and cfg_test.gsheets.service_account_info and \
           (cfg_test.gsheets.spreadsheet_id or cfg_test.gsheets.spreadsheet_name):
            st.info("Attempting to connect to Google Sheets using loaded configuration...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.gsheets.service_account_info,
                cfg_test.gsheets.spreadsheet_id,
                cfg_test.gsheets.spreadsheet_name,
                cfg_test.gsheets.worksheet_name
            )
            if worksheet_test:
                st.success(f"Successfully connected to: {worksheet_test.spreadsheet.title} -> {worksheet_test.title}")
                ensure_master_header(worksheet_test)
                
                st.subheader("Test Data Writing")
                if st.button("Write Sample Batch Data"):
                    test_batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    test_consolidated_summary = "Overall summary for 'sample testing' batch."
                    test_topic_context = "Sample Testing Keywords"
                    test_item_list = [
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "sample testing",
                            "url": "http://example.com/test1", "search_title": "Test Page 1",
                            "scraped_title": "Scraped Test Page 1", 
                            "meta_description": "This is a sample meta description for test1.", # Test data
                            "og_title": "OG Title for Test1", # Test data
                            "og_description": "OG Description for Test1.", # Test data
                            "content_type": "html",
                            "llm_summary": "Individual summary for test page 1.",
                            "scraped_main_text": "Main content of test page 1..." * 10
                        },
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "sample testing",
                            "url": "http://example.com/test2", "search_title": "Test Page 2",
                            "scraped_title": "Scraped Test Page 2",
                            "meta_description": "Another meta for test2.", # Test data
                            "og_title": "Test2 OpenGraph Title", # Test data
                            "og_description": "Test2 OpenGraph Description is quite nice.", # Test data
                            "content_type": "pdf",
                            "llm_extracted_info": "Specific info: datum A, datum B.",
                            "scraping_error": "Minor scrape warning here.",
                            "scraped_main_text": "This is another main content..." * 5
                        }
                    ]
                    
                    batch_write_success = write_batch_summary_and_items_to_sheet(
                        worksheet=worksheet_test,
                        batch_timestamp=test_batch_timestamp,
                        consolidated_summary=test_consolidated_summary,
                        topic_context=test_topic_context,
                        item_data_list=test_item_list,
                        extraction_query_text="Find datum A and B"
                    )
                    
                    if batch_write_success:
                        st.success(f"Batch summary row and {len(test_item_list)} item detail rows written successfully.")
                    else:
                        st.error("Failed to write sample batch data to the sheet.")
        else:
            st.warning("Google Sheets configuration missing/incomplete. Full test cannot run.")
    except ImportError:
        st.error("Could not import 'config' module. Ensure it's in Python path for standalone testing.")
    except Exception as e:
        st.error(f"An error occurred during the test setup or execution: {e}")

# end of modules/data_storage.py
