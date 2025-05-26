# modules/data_storage.py
# Version 1.5: Enhanced docstrings, type hinting, and comments.
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
        # Define the scopes required for gspread to function correctly.
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file' # Needed for opening by name/ID and discovery
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet: Optional[gspread.Spreadsheet] = None # Type hint for clarity

        # Prioritize opening by Spreadsheet ID for robustness
        if spreadsheet_id:
            # st.info(f"Attempting to open spreadsheet by ID: {spreadsheet_id}") # Can be verbose
            try:
                spreadsheet = client.open_by_key(spreadsheet_id)
            except gspread.exceptions.APIError as e_id_api: # Specific API errors
                st.error(f"Google Sheets APIError opening by ID '{spreadsheet_id}': {e_id_api}. "
                           "Check ID, sharing permissions, and that Drive & Sheets APIs are enabled.")
                # If opening by ID fails, and a name is available, try falling back.
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed for '{spreadsheet_id}'. Attempting by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None # No fallback possible
            except Exception as e_id_other: # Catch other potential exceptions
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                if spreadsheet_name:
                    st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                else:
                    return None

        # Fallback to opening by name if ID failed or was not provided
        if not spreadsheet and spreadsheet_name:
            # st.info(f"Attempting to open spreadsheet by name: {spreadsheet_name}") # Can be verbose
            try:
                spreadsheet = client.open(spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Google Sheets Error: Spreadsheet '{spreadsheet_name}' not found by name. "
                           "Please verify the name or ensure SPREADSHEET_ID is correctly set in secrets.")
                return None
            except Exception as e_name: # Other errors during open by name
                st.error(f"Google Sheets Error: Error opening by name '{spreadsheet_name}': {e_name}")
                return None
        
        if not spreadsheet: # If still no spreadsheet after all attempts
            st.error("Google Sheets Error: Could not open spreadsheet using provided ID or Name.")
            return None

        # Try to get the specific worksheet (tab)
        worksheet: Optional[gspread.Worksheet] = None
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # If the target worksheet is the common default "Sheet1" and it's not found,
            # try using the first sheet available as a fallback.
            if worksheet_name == "Sheet1" and spreadsheet.sheet1:
                st.info(f"Worksheet '{worksheet_name}' not found in '{spreadsheet.title}'. Using the first available sheet ('{spreadsheet.sheet1.title}').")
                worksheet = spreadsheet.sheet1
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Data cannot be written to this tab.")
                return None # Worksheet specifically not found
        
        # st.success(f"Successfully connected to Google Sheet: '{spreadsheet.title}' -> Worksheet: '{worksheet.title}'") # Can be verbose
        return worksheet

    except Exception as e: # Catch-all for other unexpected errors during client setup/auth
        st.error(f"Google Sheets connection/setup main error: {e}")
        return None

# --- UNIFIED MASTER HEADER for the Google Sheet ---
MASTER_HEADER: List[str] = [
    "Record Type",              # E.g., "Batch Summary" or "Item Detail"
    "Batch Timestamp",          # Timestamp for the entire processing batch
    "Batch Consolidated Summary", # LLM-generated summary for the whole batch
    "Batch Topic/Keywords",     # Keywords or topic context for the batch summary
    "Items in Batch",           # Count of items processed in this batch
    # Item Specific Headers (these will be blank for "Batch Summary" type rows)
    "Item Timestamp",           # Timestamp for when this specific item was processed
    "Keyword Searched",         # The keyword that led to this item
    "URL",                      # The URL of the item
    "Search Result Title",      # Title from Google Search result
    "Search Result Snippet",    # Snippet from Google Search result
    "Scraped Page Title",       # <title> tag from the scraped page
    "Scraped Meta Description", # Meta description from the scraped page
    "Scraped OG Title",         # OpenGraph title from the scraped page
    "Scraped OG Description",   # OpenGraph description from the scraped page
    "LLM Summary (Individual)", # LLM summary for this specific item
    "LLM Extracted Info (Query)", # LLM extracted info for this item
    "LLM Extraction Query",     # The query used for LLM extraction for this item
    "Scraping Error",           # Any error message from the scraping process
    "Main Text (Truncated)"     # Truncated main text content of the page
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    """
    Ensures the MASTER_HEADER is present in Row 1 of the worksheet.

    If the worksheet is completely empty, it writes the MASTER_HEADER.
    If Row 1 exists but does not match the MASTER_HEADER, a warning is logged.
    More sophisticated header migration logic is not implemented here.

    Args:
        worksheet: The gspread.Worksheet object to check/update.
    """
    try:
        current_row1_values = worksheet.row_values(1) # gspread is 1-indexed for rows/cols
        if not current_row1_values or all(cell == '' for cell in current_row1_values):
            # Sheet is empty or first row is entirely blank, so write the header.
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED')
            st.info(f"Initialized worksheet '{worksheet.title}' with MASTER_HEADER in Row 1.")
        elif current_row1_values != MASTER_HEADER:
            # Header exists but doesn't match. This could cause data misalignment.
            st.warning(
                f"Worksheet '{worksheet.title}' Row 1 header does not match the expected MASTER_HEADER. "
                "Data will be appended according to the new structure, which might misalign with existing data. "
                "Consider clearing the sheet or manually adjusting the header if this is unintended."
            )
            # For a production tool, you might want to:
            # 1. Stop and alert the user.
            # 2. Try to intelligently map old columns to new ones (very complex).
            # 3. Create a new versioned worksheet.
    except gspread.exceptions.APIError as e:
        # This can happen if the sheet is truly empty and row_values(1) fails in a way that
        # isn't just returning an empty list (e.g., older gspread versions or specific API states).
        if 'exceeds grid limits' in str(e).lower() or isinstance(e, (gspread.exceptions.CellNotFound, IndexError)): # More specific
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED')
            st.info(f"Initialized empty worksheet '{worksheet.title}' with MASTER_HEADER (APIError/IndexError catch).")
        else:
            st.error(f"Google Sheets API error while ensuring header: {e}")
    except Exception as e: # Catch any other unexpected issues
        st.error(f"Unexpected error while ensuring header in Google Sheets: {e}")


# --- Write Data Functions ---
def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str, # Keywords or general topic for the batch summary
    item_data_list: List[Dict[str, Any]], # List of dictionaries, one per scraped item
    extraction_query_text: Optional[str] = None, # The query used for LLM extraction, if any
    main_text_truncate_limit: int = 10000 # Character limit for storing main text
) -> bool:
    """
    Writes a batch summary row followed by individual item detail rows to the Google Sheet.

    All data is written in alignment with the MASTER_HEADER.

    Args:
        worksheet: The gspread.Worksheet object to write to.
        batch_timestamp: Timestamp for when this batch of processing started.
        consolidated_summary: The LLM-generated consolidated summary for the batch.
        topic_context: The keyword(s) or topic the consolidated summary pertains to.
        item_data_list: A list of dictionaries, each representing a processed web page/item.
        extraction_query_text: The text of the query used for LLM extraction, if any.
        main_text_truncate_limit: Max characters of main_text to store in the sheet.

    Returns:
        True if data was successfully appended, False otherwise.
    """
    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        return False

    rows_to_append: List[List[Any]] = [] # List of lists, where each inner list is a row

    # 1. Prepare and add the Batch Summary Row
    batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER} # Init with blanks
    batch_summary_row_dict["Record Type"] = "Batch Summary"
    batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
    batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else "N/A or Error"
    batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
    batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
    
    # Convert dictionary to list in MASTER_HEADER order for gspread
    rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    # 2. Prepare and add Item Detail Rows
    for item_detail in item_data_list:
        main_text = item_detail.get("scraped_main_text", "")
        # Truncate main_text if it's not None and exceeds the limit
        truncated_main_text = (main_text[:main_text_truncate_limit] + "..." if (main_text and len(main_text) > main_text_truncate_limit) else main_text)
        
        item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER} # Init with blanks
        item_row_dict["Record Type"] = "Item Detail"
        item_row_dict["Batch Timestamp"] = batch_timestamp # Link item to its batch
        
        # Populate item-specific fields
        item_row_dict["Item Timestamp"] = item_detail.get("timestamp", batch_timestamp) # Use item's own timestamp if available
        item_row_dict["Keyword Searched"] = item_detail.get("keyword_searched", "")
        item_row_dict["URL"] = item_detail.get("url", "")
        item_row_dict["Search Result Title"] = item_detail.get("search_title", "")
        item_row_dict["Search Result Snippet"] = item_detail.get("search_snippet", "")
        item_row_dict["Scraped Page Title"] = item_detail.get("scraped_title", "")
        item_row_dict["Scraped Meta Description"] = item_detail.get("scraped_meta_description", "")
        item_row_dict["Scraped OG Title"] = item_detail.get("scraped_og_title", "")
        item_row_dict["Scraped OG Description"] = item_detail.get("scraped_og_description", "")
        item_row_dict["LLM Summary (Individual)"] = item_detail.get("llm_summary", "")
        item_row_dict["LLM Extracted Info (Query)"] = item_detail.get("llm_extracted_info", "")
        item_row_dict["LLM Extraction Query"] = extraction_query_text if item_detail.get("llm_extracted_info") else ""
        item_row_dict["Scraping Error"] = item_detail.get("scraping_error", "")
        item_row_dict["Main Text (Truncated)"] = truncated_main_text
        
        rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    if not rows_to_append: # Should not happen if called, as summary row is always prepared
        st.info("Google Sheets: No data (batch summary or items) formatted for writing.")
        return False
        
    try:
        # Append all prepared rows (summary + items) in one API call
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        # Success message will be handled by app.py for the overall batch
        return True
    except Exception as e:
        st.error(f"Google Sheets Error: Failed to write batch data rows: {e}")
        return False

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5 - Unified Header & Docs)")
    try:
        # For direct testing, ensure config.py is accessible.
        # This might require adjusting Python path or running from project root.
        from config import load_config
        cfg_test = load_config() # Load config to get sheet credentials and identifiers
        
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
                ensure_master_header(worksheet_test) # Test the header ensuring logic
                
                st.subheader("Test Data Writing with Unified Header")
                if st.button("Write Sample Batch Data (Unified)"):
                    test_batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    test_consolidated_summary = "This is an overall summary for the test batch of items related to 'sample testing'."
                    test_topic_context = "Sample Testing Keywords"
                    test_item_list = [
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "sample testing",
                            "url": "http://example.com/test1", "search_title": "Test Page 1",
                            "scraped_title": "Scraped Test Page 1", 
                            "llm_summary": "Individual summary for test page 1.",
                            "scraped_main_text": "Main content of test page 1..." * 10
                        },
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "sample testing",
                            "url": "http://example.com/test2", "search_title": "Test Page 2",
                            "scraped_title": "Scraped Test Page 2",
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
                        extraction_query_text="Find datum A and B" # Example query text
                    )
                    
                    if batch_write_success:
                        st.write(f"Batch summary row and {len(test_item_list)} item detail rows written successfully.")
                    else:
                        st.error("Failed to write sample batch data to the sheet.")
        else:
            st.warning("Google Sheets configuration (service account details, Spreadsheet ID/Name) "
                       "is missing or could not be loaded from secrets. Full test cannot run.")
    except ImportError:
        st.error("Could not import 'config' module. Ensure it's in Python path for standalone testing.")
    except Exception as e:
        st.error(f"An error occurred during the test setup or execution: {e}")

# end of modules/data_storage.py
