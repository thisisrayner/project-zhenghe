# modules/data_storage.py
# Version 1.5.6:
# - Added robust handling for `main_text` being None before len() or slicing.
# - Added more specific debug prints within item processing loop.
# Version 1.5.5:
# - Added extensive print() debugging for sheet writing process and connection.
# Version 1.5.4: Added support for two distinct LLM extraction queries in GSheet output.

"""
Handles data storage operations, primarily focused on Google Sheets integration.
"""
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time 
import traceback

@st.cache_resource 
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],
    spreadsheet_name: Optional[str], 
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    """
    Connects to Google Sheets using service account credentials and returns a specific worksheet.

    It tries to open the spreadsheet first by ID (if provided), then by name as a fallback.
    It then attempts to get the specified worksheet by its name, with a fallback to the
    first sheet if `worksheet_name` is 'Sheet1' and it's not found.

    Args:
        service_account_info: Dictionary containing Google Service Account credentials.
        spreadsheet_id: The ID of the Google Sheet (from its URL).
        spreadsheet_name: The name of the Google Spreadsheet file.
        worksheet_name: The name of the tab/worksheet within the spreadsheet.

    Returns:
        A `gspread.Worksheet` object if successful, otherwise `None`.
        Errors are logged to `st.error` or `st.warning`.
    """
    print(f"DEBUG (data_storage): get_gspread_worksheet called. SID: '{spreadsheet_id}', SName: '{spreadsheet_name}', WSName: '{worksheet_name}'") 
    if not service_account_info:
        st.error("Google Sheets Error: Service account information not provided in secrets.")
        print("DEBUG (data_storage): No service_account_info provided to get_gspread_worksheet.") 
        return None
    if not spreadsheet_id and not spreadsheet_name:
        st.error("Google Sheets Error: Neither Spreadsheet ID nor Spreadsheet Name provided in secrets.")
        print("DEBUG (data_storage): No spreadsheet_id and no spreadsheet_name provided.") 
        return None
        
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file'
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        print("DEBUG (data_storage): gspread client authorized successfully.") 
        
        spreadsheet: Optional[gspread.Spreadsheet] = None
        
        if spreadsheet_id:
            try:
                print(f"DEBUG (data_storage): Attempting to open spreadsheet by ID: '{spreadsheet_id}'") 
                spreadsheet = client.open_by_key(spreadsheet_id)
                print(f"DEBUG (data_storage): Successfully opened spreadsheet by ID: '{spreadsheet.title}'") 
            except gspread.exceptions.APIError as e_id_api:
                st.error(f"Google Sheets APIError opening by ID '{spreadsheet_id}': {e_id_api}. Check ID, ensure sheet is shared with service account email ({service_account_info.get('client_email')}), and verify Drive & Sheets APIs are enabled in GCP.")
                print(f"ERROR (data_storage): APIError opening by ID '{spreadsheet_id}': {e_id_api}") 
                print(traceback.format_exc()) 
                if spreadsheet_name: 
                    st.warning(f"Opening by ID ('{spreadsheet_id}') failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                    print(f"DEBUG (data_storage): Opening by ID failed, falling back to name '{spreadsheet_name}'.") 
                else: 
                    return None 
            except Exception as e_id_other: 
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                print(f"ERROR (data_storage): Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}") 
                print(traceback.format_exc()) 
                if spreadsheet_name: 
                    st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                    print(f"DEBUG (data_storage): Opening by ID failed (unexpected error), falling back to name '{spreadsheet_name}'.") 
                else: 
                    return None
        
        if not spreadsheet and spreadsheet_name:
            try:
                print(f"DEBUG (data_storage): Attempting to open spreadsheet by name: '{spreadsheet_name}'") 
                spreadsheet = client.open(spreadsheet_name)
                print(f"DEBUG (data_storage): Successfully opened spreadsheet by name: '{spreadsheet.title}'") 
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Google Sheets Error: Spreadsheet '{spreadsheet_name}' not found by name. Please verify the name and ensure it's shared with the service account email ({service_account_info.get('client_email')}). If using SPREADSHEET_ID, ensure it's correct.")
                print(f"ERROR (data_storage): Spreadsheet '{spreadsheet_name}' not found by name.") 
                return None
            except Exception as e_name: 
                st.error(f"Google Sheets Error: Error opening by name '{spreadsheet_name}': {e_name}")
                print(f"ERROR (data_storage): Error opening by name '{spreadsheet_name}': {e_name}") 
                print(traceback.format_exc()) 
                return None
        
        if not spreadsheet: 
            st.error("Google Sheets Error: Could not open spreadsheet using provided ID or Name. Please check secrets.toml and sharing settings.")
            print("ERROR (data_storage): Could not open spreadsheet after all attempts.") 
            return None
            
        worksheet_obj: Optional[gspread.Worksheet] = None
        try:
            print(f"DEBUG (data_storage): Attempting to get worksheet by name: '{worksheet_name}' from SSheet: '{spreadsheet.title}'") 
            worksheet_obj = spreadsheet.worksheet(worksheet_name)
            print(f"DEBUG (data_storage): Successfully got worksheet: '{worksheet_obj.title}'") 
        except gspread.exceptions.WorksheetNotFound:
            print(f"DEBUG (data_storage): Worksheet '{worksheet_name}' not found. Trying fallback to first sheet (often 'Sheet1').") 
            if worksheet_name.lower() == "sheet1" and hasattr(spreadsheet, 'sheet1') and spreadsheet.sheet1 is not None:
                st.info(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Using the first available sheet (likely named '{spreadsheet.sheet1.title}').")
                worksheet_obj = spreadsheet.sheet1 
                print(f"DEBUG (data_storage): Fallback to first sheet successful: '{worksheet_obj.title}'") 
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Data cannot be written to this specific tab. Please create it or check the name.")
                print(f"ERROR (data_storage): Worksheet '{worksheet_name}' not found, and fallback condition not met or failed.") 
                return None 
        return worksheet_obj

    except Exception as e: 
        st.error(f"Google Sheets: Main connection/setup error: {e}. Please check your service account credentials, API enablement, and sheet sharing.")
        print(f"ERROR (data_storage): Main exception in get_gspread_worksheet: {e}") 
        print(traceback.format_exc()) 
        return None

MASTER_HEADER: List[str] = [
    "Record Type", "Batch Timestamp", "Batch Consolidated Summary", "Batch Topic/Keywords",
    "Items in Batch", "Item Timestamp", "Keyword Searched", "URL", "Search Result Title",
    "Search Result Snippet", "Scraped Page Title", "Scraped Meta Description",
    "Scraped OG Title", "Scraped OG Description", "Content Type", "LLM Summary (Individual)",
    "LLM Extraction Query 1", "LLM Extracted Info (Q1)", "LLM Extraction Query 2",
    "LLM Extracted Info (Q2)", "Scraping Error", "Main Text (Truncated)"
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    """
    Ensures that the first row of the given worksheet matches the MASTER_HEADER.
    If it doesn't match, or if the row is empty/blank, it overwrites Row 1
    with the MASTER_HEADER. This is an assertive function.

    Args:
        worksheet: The `gspread.Worksheet` object to check and update.
    """
    print(f"DEBUG (data_storage): ensure_master_header called for worksheet: '{worksheet.title}' in spreadsheet '{worksheet.spreadsheet.title}'") 
    header_action_needed = False
    action_reason = ""
    try:
        current_row1_values = worksheet.row_values(1) 
        if not current_row1_values: 
            header_action_needed = True
            action_reason = "Row 1 is empty (no values returned from worksheet.row_values(1))."
        elif all(cell == '' for cell in current_row1_values): 
            header_action_needed = True
            action_reason = "Row 1 is blank (all cells are empty strings)."
        elif current_row1_values != MASTER_HEADER:
            header_action_needed = True
            action_reason = (f"Row 1 header (len {len(current_row1_values)}) does not match expected MASTER_HEADER (len {len(MASTER_HEADER)}). "
                             f"Current (first 5): {str(current_row1_values[:5]) if current_row1_values else 'N/A'}... "
                             f"Expected (first 5): {MASTER_HEADER[:5]}...")
            print(f"DEBUG (data_storage): Header mismatch. Current: {current_row1_values}, Expected: {MASTER_HEADER}") 
        else:
            print("DEBUG (data_storage): Header matches MASTER_HEADER. No action needed.") 
    except (gspread.exceptions.APIError, IndexError, gspread.exceptions.CellNotFound) as e:
        header_action_needed = True
        action_reason = f"Could not read Row 1 (typical for new/empty sheet or API issue: {type(e).__name__} - {e})."
        print(f"DEBUG (data_storage): Exception reading Row 1: {action_reason}") 
    except Exception as e_check_header: 
        st.error(f"Google Sheets: Unexpected error while checking header: {e_check_header}. Attempting to write header as fallback.")
        header_action_needed = True
        action_reason = f"Unexpected error during header check: {e_check_header}."
        print(f"ERROR (data_storage): Unexpected error checking header: {e_check_header}") 
        print(traceback.format_exc()) 

    if header_action_needed:
        st.info(f"Google Sheets Header Info: {action_reason} Writing/Overwriting MASTER_HEADER to worksheet '{worksheet.title}'.")
        print(f"DEBUG (data_storage): Action needed for header: {action_reason}. Attempting to write MASTER_HEADER.") 
        try:
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED') 
            st.success(f"Google Sheets: MASTER_HEADER written/updated successfully in Row 1 of worksheet '{worksheet.title}'.")
            print(f"DEBUG (data_storage): Successfully wrote MASTER_HEADER to '{worksheet.title}'") 
        except Exception as e_write_header:
            st.error(f"Google Sheets ERROR: Failed to write/update MASTER_HEADER to Row 1 of '{worksheet.title}': {e_write_header}")
            print(f"ERROR (data_storage): Failed to write MASTER_HEADER: {e_write_header}") 
            print(traceback.format_exc()) 


def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_data_list: List[Dict[str, Any]],
    extraction_queries_list: List[str], 
    main_text_truncate_limit: int = 10000
) -> bool:
    """
    Writes a batch summary row and multiple item detail rows to the specified Google Sheet.

    The data is structured according to the MASTER_HEADER.
    It first prepares a batch summary row, then a row for each item in `item_data_list`.
    All rows are then appended to the sheet in a single API call if possible.

    Args:
        worksheet: The `gspread.Worksheet` object to write to.
        batch_timestamp: Timestamp string for the overall batch.
        consolidated_summary: The consolidated LLM summary text for the batch.
        topic_context: Keywords or topic context for the batch.
        item_data_list: A list of dictionaries, each representing a processed item's data.
                        Expected keys align with MASTER_HEADER columns for "Item Detail".
        extraction_queries_list: A list of the extraction query strings used [Q1_text, Q2_text].
        main_text_truncate_limit: Character limit for truncating main text content.

    Returns:
        `True` if data was successfully written (or if only a batch summary was written
        when no items were present), `False` otherwise.
        Errors are logged via `st.error`.
    """
    print(f"DEBUG (data_storage): write_batch_summary_and_items_to_sheet called.") 
    print(f"  Worksheet: '{worksheet.title if worksheet else 'None'}' in SSheet '{worksheet.spreadsheet.title if worksheet else 'N/A'}'") 
    print(f"  Batch TS: {batch_timestamp}, Topic: '{topic_context}'") 
    print(f"  Num items: {len(item_data_list) if item_data_list is not None else 'N/A'}, Num extraction queries: {len(extraction_queries_list) if extraction_queries_list is not None else 'N/A'}") 
    if extraction_queries_list: 
        print(f"  Extraction Query 1: '{extraction_queries_list[0] if len(extraction_queries_list) > 0 else 'N/A'}'") 
        print(f"  Extraction Query 2: '{extraction_queries_list[1] if len(extraction_queries_list) > 1 else 'N/A'}'") 


    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        print("ERROR (data_storage): No valid worksheet provided to write_batch_summary_and_items_to_sheet.") 
        return False

    rows_to_append: List[List[Any]] = []

    # Prepare Batch Summary Row
    batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
    batch_summary_row_dict["Record Type"] = "Batch Summary"
    batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
    batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else "N/A or Error"
    batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
    batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
    rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
    print(f"DEBUG (data_storage): Prepared batch summary row: {str(rows_to_append[0][:5])[:200]}...") 


    # Prepare Item Detail Rows
    query1_text_for_sheet = extraction_queries_list[0] if len(extraction_queries_list) > 0 and extraction_queries_list[0] else ""
    query2_text_for_sheet = extraction_queries_list[1] if len(extraction_queries_list) > 1 and extraction_queries_list[1] else ""

    if item_data_list: 
        for idx, item_detail in enumerate(item_data_list):
            print(f"DEBUG (data_storage): Processing item {idx + 1}/{len(item_data_list)} for sheet: {item_detail.get('url', 'N/A')}") # ADDED
            
            main_text_raw = item_detail.get("main_content_display", item_detail.get("scraped_main_text")) 
            
            main_text_for_sheet = "" # Default to empty string
            if isinstance(main_text_raw, str): # Ensure it's a string before len() or slicing
                main_text_for_sheet = main_text_raw
            
            print(f"DEBUG (data_storage): item {idx + 1} main_text type: {type(main_text_for_sheet)}, len: {len(main_text_for_sheet) if isinstance(main_text_for_sheet, str) else 'N/A'}") # ADDED

            truncated_main_text = (main_text_for_sheet[:main_text_truncate_limit] + "..." if (len(main_text_for_sheet) > main_text_truncate_limit) else main_text_for_sheet)
            
            item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
            item_row_dict["Record Type"] = "Item Detail"
            item_row_dict["Batch Timestamp"] = batch_timestamp 
            item_row_dict["Item Timestamp"] = item_detail.get("timestamp", batch_timestamp) 
            item_row_dict["Keyword Searched"] = item_detail.get("keyword_searched", "")
            item_row_dict["URL"] = item_detail.get("url", "")
            item_row_dict["Search Result Title"] = item_detail.get("search_title", "") 
            item_row_dict["Search Result Snippet"] = item_detail.get("search_snippet", "") 
            item_row_dict["Scraped Page Title"] = item_detail.get("page_title", item_detail.get("scraped_title", "")) 
            item_row_dict["Scraped Meta Description"] = item_detail.get("meta_description", "") 
            item_row_dict["Scraped OG Title"] = item_detail.get("og_title", "") 
            item_row_dict["Scraped OG Description"] = item_detail.get("og_description", "") 
            item_row_dict["Content Type"] = item_detail.get("content_type", "pdf" if item_detail.get("is_pdf") else "html") 
            item_row_dict["LLM Summary (Individual)"] = item_detail.get("llm_summary", "") 
            
            item_row_dict["LLM Extraction Query 1"] = query1_text_for_sheet if item_detail.get("llm_extracted_info_q1") or item_detail.get("llm_relevancy_score_q1") is not None else ""
            item_row_dict["LLM Extracted Info (Q1)"] = item_detail.get("llm_extracted_info_q1", "") 
            
            item_row_dict["LLM Extraction Query 2"] = query2_text_for_sheet if item_detail.get("llm_extracted_info_q2") or item_detail.get("llm_relevancy_score_q2") is not None else ""
            item_row_dict["LLM Extracted Info (Q2)"] = item_detail.get("llm_extracted_info_q2", "") 
            
            item_row_dict["Scraping Error"] = item_detail.get("scraping_error", item_detail.get("error_message", "")) 
            item_row_dict["Main Text (Truncated)"] = truncated_main_text
            
            rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
            if idx < 2: 
                 print(f"DEBUG (data_storage): Prepared item row {idx+1}: {str(rows_to_append[-1][:5])[:200]}...") 
    else:
        print("DEBUG (data_storage): item_data_list is empty or None. No item detail rows will be prepared.") 

    if not rows_to_append: 
        print("DEBUG (data_storage): No rows to append at all (not even batch summary). This is unexpected. Returning False.") 
        return False
    
    if len(rows_to_append) == 1 and rows_to_append[0][0] == "Batch Summary" and not item_data_list:
        print(f"DEBUG (data_storage): Only batch summary row to append (no items).") 
    
    print(f"DEBUG (data_storage): Total rows to append: {len(rows_to_append)}") 
        
    try:
        print(f"DEBUG (data_storage): Attempting to append {len(rows_to_append)} total rows to worksheet '{worksheet.title}'.") 
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"DEBUG (data_storage): Successfully appended rows to worksheet '{worksheet.title}'.") 
        return True
    except Exception as e_append:
        st.error(f"Google Sheets Error: Failed to write {len(rows_to_append)} batch data rows to '{worksheet.title}': {e_append}")
        print(f"ERROR in write_batch_summary_and_items_to_sheet (appending rows): {e_append}") 
        print(traceback.format_exc()) 
        return False

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.6 - Main Text Fix & Debugs)")
    try:
        # This import assumes config.py is in a 'modules' subdirectory relative to this test script's location
        # when run directly. Adjust if your test setup is different.
        # For example, if running from project root: from modules.config import load_config
        try:
            from modules.config import load_config
        except ImportError:
            # Fallback if running data_storage.py directly from within modules folder
            import sys
            sys.path.insert(0, '..') # Add parent directory (project root) to path
            from modules.config import load_config

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
                
                if st.button("TEST: Ensure Master Header (v1.5.6)"):
                    ensure_master_header(worksheet_test)
                    st.write("Ensure Master Header call completed. Check sheet and terminal logs.")

                st.subheader("Test Data Writing (Robust Main Text)")
                if st.button("Write Sample Batch Data (Main Text Fix Test)"):
                    test_batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    test_consolidated_summary = "Test summary with main_text handling. v1.5.6"
                    test_topic_context = "Main Text Robustness Test"
                    test_extraction_queries = ["Is main_text handled?", "Any None values?"]
                    test_item_list = [
                        { # Item with string main_text
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "text test",
                            "url": "http://example.com/text1", "page_title": "Text Page 1", 
                            "main_content_display": "This is a normal string for main text.",
                            "llm_extracted_info_q1": "Yes, handled.", "llm_relevancy_score_q1": 5,
                        },
                        { # Item with None main_text
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "none test",
                            "url": "http://example.com/none1", "page_title": "None Text Page",
                            "main_content_display": None, # Test case for None
                            "llm_extracted_info_q2": "None values are now handled.", "llm_relevancy_score_q2": 5
                        },
                        { # Item with empty string main_text
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "empty test",
                            "url": "http://example.com/empty1", "page_title": "Empty Text Page",
                            "main_content_display": "", # Test case for empty string
                        }
                    ]
                    batch_write_success = write_batch_summary_and_items_to_sheet(
                        worksheet=worksheet_test, batch_timestamp=test_batch_timestamp,
                        consolidated_summary=test_consolidated_summary, topic_context=test_topic_context,
                        item_data_list=test_item_list,
                        extraction_queries_list=test_extraction_queries
                    )
                    if batch_write_success: st.success(f"Batch data written successfully.")
                    else: st.error("Failed to write sample batch data. Check terminal logs.")
            else:
                st.warning("Could not connect to worksheet. Check GSheets setup details printed in terminal/UI.")
        else: st.warning("Google Sheets configuration missing or incomplete in secrets.toml. Full test cannot run.")
    except ImportError as e_imp: 
        st.error(f"Could not import 'modules.config' module. Ensure it's in the Python path or adjust import. Error: {e_imp}")
        print(traceback.format_exc())
    except Exception as e_main_test: 
        st.error(f"An error occurred during the test setup: {e_main_test}")
        print(traceback.format_exc())

# end of modules/data_storage.py
