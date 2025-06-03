# modules/data_storage.py
# Version 1.5.5: (as per my previous suggestion for this file)
# - Added extensive print() debugging for sheet writing process and connection.
# Version 1.5.4: Added support for two distinct LLM extraction queries in GSheet output.
# Assertive ensure_master_header and previous key fixes maintained.

"""
Handles data storage operations, primarily focused on Google Sheets integration.
"""
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time # Retained from your version, though not directly used in this snippet
import traceback # For detailed error printing

@st.cache_resource # Cache the gspread client and worksheet object
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],
    spreadsheet_name: Optional[str], # Name of the spreadsheet file
    worksheet_name: str = "Sheet1"  # Name of the tab/worksheet within the spreadsheet
) -> Optional[gspread.Worksheet]:
    print(f"DEBUG (data_storage): get_gspread_worksheet called. SID: '{spreadsheet_id}', SName: '{spreadsheet_name}', WSName: '{worksheet_name}'") # DEBUG
    if not service_account_info:
        st.error("Google Sheets Error: Service account information not provided in secrets.")
        print("DEBUG (data_storage): No service_account_info provided to get_gspread_worksheet.") # DEBUG
        return None
    if not spreadsheet_id and not spreadsheet_name:
        st.error("Google Sheets Error: Neither Spreadsheet ID nor Spreadsheet Name provided in secrets.")
        print("DEBUG (data_storage): No spreadsheet_id and no spreadsheet_name provided.") # DEBUG
        return None
        
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file' # Often needed to discover/open files by name
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        print("DEBUG (data_storage): gspread client authorized successfully.") # DEBUG
        
        spreadsheet: Optional[gspread.Spreadsheet] = None
        
        # Try opening by ID first if provided
        if spreadsheet_id:
            try:
                print(f"DEBUG (data_storage): Attempting to open spreadsheet by ID: '{spreadsheet_id}'") # DEBUG
                spreadsheet = client.open_by_key(spreadsheet_id)
                print(f"DEBUG (data_storage): Successfully opened spreadsheet by ID: '{spreadsheet.title}'") # DEBUG
            except gspread.exceptions.APIError as e_id_api:
                # This error often means the ID is wrong, or the sheet isn't shared, or APIs aren't enabled.
                st.error(f"Google Sheets APIError opening by ID '{spreadsheet_id}': {e_id_api}. Check ID, ensure sheet is shared with service account email ({service_account_info.get('client_email')}), and verify Drive & Sheets APIs are enabled in GCP.")
                print(f"ERROR (data_storage): APIError opening by ID '{spreadsheet_id}': {e_id_api}") # DEBUG
                print(traceback.format_exc()) # DEBUG
                if spreadsheet_name: 
                    st.warning(f"Opening by ID ('{spreadsheet_id}') failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                    print(f"DEBUG (data_storage): Opening by ID failed, falling back to name '{spreadsheet_name}'.") # DEBUG
                else: 
                    return None # No name to fall back to
            except Exception as e_id_other: # Catch any other unexpected errors
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                print(f"ERROR (data_storage): Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}") # DEBUG
                print(traceback.format_exc()) # DEBUG
                if spreadsheet_name: 
                    st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                    print(f"DEBUG (data_storage): Opening by ID failed (unexpected error), falling back to name '{spreadsheet_name}'.") # DEBUG
                else: 
                    return None
        
        # If opening by ID failed (and name was provided) or ID was not provided, try by name
        if not spreadsheet and spreadsheet_name:
            try:
                print(f"DEBUG (data_storage): Attempting to open spreadsheet by name: '{spreadsheet_name}'") # DEBUG
                spreadsheet = client.open(spreadsheet_name)
                print(f"DEBUG (data_storage): Successfully opened spreadsheet by name: '{spreadsheet.title}'") # DEBUG
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Google Sheets Error: Spreadsheet '{spreadsheet_name}' not found by name. Please verify the name and ensure it's shared with the service account email ({service_account_info.get('client_email')}). If using SPREADSHEET_ID, ensure it's correct.")
                print(f"ERROR (data_storage): Spreadsheet '{spreadsheet_name}' not found by name.") # DEBUG
                return None
            except Exception as e_name: # Catch other errors opening by name
                st.error(f"Google Sheets Error: Error opening by name '{spreadsheet_name}': {e_name}")
                print(f"ERROR (data_storage): Error opening by name '{spreadsheet_name}': {e_name}") # DEBUG
                print(traceback.format_exc()) # DEBUG
                return None
        
        if not spreadsheet: # If still no spreadsheet after trying ID and/or name
            st.error("Google Sheets Error: Could not open spreadsheet using provided ID or Name. Please check secrets.toml and sharing settings.")
            print("ERROR (data_storage): Could not open spreadsheet after all attempts.") # DEBUG
            return None
            
        # Now try to get the specific worksheet (tab)
        worksheet_obj: Optional[gspread.Worksheet] = None
        try:
            print(f"DEBUG (data_storage): Attempting to get worksheet by name: '{worksheet_name}' from SSheet: '{spreadsheet.title}'") # DEBUG
            worksheet_obj = spreadsheet.worksheet(worksheet_name)
            print(f"DEBUG (data_storage): Successfully got worksheet: '{worksheet_obj.title}'") # DEBUG
        except gspread.exceptions.WorksheetNotFound:
            print(f"DEBUG (data_storage): Worksheet '{worksheet_name}' not found. Trying fallback to first sheet (often 'Sheet1').") # DEBUG
            # Fallback to the first sheet if the specified one isn't found and the target was 'Sheet1' (common default)
            if worksheet_name.lower() == "sheet1" and hasattr(spreadsheet, 'sheet1') and spreadsheet.sheet1 is not None:
                st.info(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Using the first available sheet (likely named '{spreadsheet.sheet1.title}').")
                worksheet_obj = spreadsheet.sheet1 # Get the first sheet object
                print(f"DEBUG (data_storage): Fallback to first sheet successful: '{worksheet_obj.title}'") # DEBUG
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Data cannot be written to this specific tab. Please create it or check the name.")
                print(f"ERROR (data_storage): Worksheet '{worksheet_name}' not found, and fallback condition not met or failed.") # DEBUG
                return None # Cannot proceed without a valid worksheet
        return worksheet_obj

    except Exception as e: # Catch-all for any other gspread or auth errors
        st.error(f"Google Sheets: Main connection/setup error: {e}. Please check your service account credentials, API enablement, and sheet sharing.")
        print(f"ERROR (data_storage): Main exception in get_gspread_worksheet: {e}") # DEBUG
        print(traceback.format_exc()) # DEBUG
        return None


MASTER_HEADER: List[str] = [
    "Record Type", "Batch Timestamp", "Batch Consolidated Summary", "Batch Topic/Keywords",
    "Items in Batch", "Item Timestamp", "Keyword Searched", "URL", "Search Result Title",
    "Search Result Snippet", "Scraped Page Title", "Scraped Meta Description",
    "Scraped OG Title", "Scraped OG Description", "Content Type", "LLM Summary (Individual)",
    "LLM Extraction Query 1", "LLM Extracted Info (Q1)", "LLM Extraction Query 2",
    "LLM Extracted Info (Q2)", "Scraping Error", "Main Text (Truncated)"
] # This is from your v1.5.4

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    print(f"DEBUG (data_storage): ensure_master_header called for worksheet: '{worksheet.title}' in spreadsheet '{worksheet.spreadsheet.title}'") # DEBUG
    header_action_needed = False
    action_reason = ""
    try:
        current_row1_values = worksheet.row_values(1) # Fetches all values in the first row
        if not current_row1_values: # Row exists but is completely empty
            header_action_needed = True
            action_reason = "Row 1 is empty (no values returned from worksheet.row_values(1))."
        elif all(cell == '' for cell in current_row1_values): # Row exists, cells exist, but all are blank strings
            header_action_needed = True
            action_reason = "Row 1 is blank (all cells are empty strings)."
        elif current_row1_values != MASTER_HEADER:
            header_action_needed = True
            action_reason = (f"Row 1 header (len {len(current_row1_values)}) does not match expected MASTER_HEADER (len {len(MASTER_HEADER)}). "
                             f"Current (first 5): {str(current_row1_values[:5]) if current_row1_values else 'N/A'}... "
                             f"Expected (first 5): {MASTER_HEADER[:5]}...")
            print(f"DEBUG (data_storage): Header mismatch. Current: {current_row1_values}, Expected: {MASTER_HEADER}") # DEBUG
        else:
            print("DEBUG (data_storage): Header matches MASTER_HEADER. No action needed.") # DEBUG
    except (gspread.exceptions.APIError, IndexError, gspread.exceptions.CellNotFound) as e:
        # CellNotFound can occur if the sheet is truly empty (no cells at all in row 1)
        # APIError if there's a problem fetching, IndexError if row_values returns something unexpected (less likely)
        header_action_needed = True
        action_reason = f"Could not read Row 1 (typical for new/empty sheet or API issue: {type(e).__name__} - {e})."
        print(f"DEBUG (data_storage): Exception reading Row 1: {action_reason}") # DEBUG
    except Exception as e_check_header: # Catch any other unexpected error
        st.error(f"Google Sheets: Unexpected error while checking header: {e_check_header}. Attempting to write header as fallback.")
        header_action_needed = True
        action_reason = f"Unexpected error during header check: {e_check_header}."
        print(f"ERROR (data_storage): Unexpected error checking header: {e_check_header}") # DEBUG
        print(traceback.format_exc()) # DEBUG

    if header_action_needed:
        st.info(f"Google Sheets Header Info: {action_reason} Writing/Overwriting MASTER_HEADER to worksheet '{worksheet.title}'.")
        print(f"DEBUG (data_storage): Action needed for header: {action_reason}. Attempting to write MASTER_HEADER.") # DEBUG
        try:
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED') # Update expects list of lists
            st.success(f"Google Sheets: MASTER_HEADER written/updated successfully in Row 1 of worksheet '{worksheet.title}'.")
            print(f"DEBUG (data_storage): Successfully wrote MASTER_HEADER to '{worksheet.title}'") # DEBUG
        except Exception as e_write_header:
            st.error(f"Google Sheets ERROR: Failed to write/update MASTER_HEADER to Row 1 of '{worksheet.title}': {e_write_header}")
            print(f"ERROR (data_storage): Failed to write MASTER_HEADER: {e_write_header}") # DEBUG
            print(traceback.format_exc()) # DEBUG


def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_data_list: List[Dict[str, Any]],
    extraction_queries_list: List[str], # List of query strings: [Q1_text, Q2_text]
    main_text_truncate_limit: int = 10000
) -> bool:
    print(f"DEBUG (data_storage): write_batch_summary_and_items_to_sheet called.") # DEBUG
    print(f"  Worksheet: '{worksheet.title if worksheet else 'None'}' in SSheet '{worksheet.spreadsheet.title if worksheet else 'N/A'}'") # DEBUG
    print(f"  Batch TS: {batch_timestamp}, Topic: '{topic_context}'") # DEBUG
    print(f"  Num items: {len(item_data_list) if item_data_list is not None else 'N/A'}, Num extraction queries: {len(extraction_queries_list) if extraction_queries_list is not None else 'N/A'}") # DEBUG
    if extraction_queries_list: #DEBUG
        print(f"  Extraction Query 1: '{extraction_queries_list[0] if len(extraction_queries_list) > 0 else 'N/A'}'") #DEBUG
        print(f"  Extraction Query 2: '{extraction_queries_list[1] if len(extraction_queries_list) > 1 else 'N/A'}'") #DEBUG


    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        print("ERROR (data_storage): No valid worksheet provided to write_batch_summary_and_items_to_sheet.") # DEBUG
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
    print(f"DEBUG (data_storage): Prepared batch summary row: {rows_to_append[0][:5]}...") # DEBUG


    # Prepare Item Detail Rows
    query1_text_for_sheet = extraction_queries_list[0] if len(extraction_queries_list) > 0 and extraction_queries_list[0] else ""
    query2_text_for_sheet = extraction_queries_list[1] if len(extraction_queries_list) > 1 and extraction_queries_list[1] else ""

    if item_data_list: # Only proceed if there are items
        for idx, item_detail in enumerate(item_data_list):
            # Using item_detail.get("main_content_display", "") as it's a key from process_manager for excel handler
            # Fallback to "scraped_main_text" if that's what scraper provides and process_manager passes through
            main_text = item_detail.get("main_content_display", item_detail.get("scraped_main_text", ""))
            truncated_main_text = (main_text[:main_text_truncate_limit] + "..." if (main_text and len(main_text) > main_text_truncate_limit) else main_text)
            
            item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
            item_row_dict["Record Type"] = "Item Detail"
            item_row_dict["Batch Timestamp"] = batch_timestamp # Link item to batch
            
            item_row_dict["Item Timestamp"] = item_detail.get("timestamp", batch_timestamp) # Specific item timestamp
            item_row_dict["Keyword Searched"] = item_detail.get("keyword_searched", "")
            item_row_dict["URL"] = item_detail.get("url", "")
            item_row_dict["Search Result Title"] = item_detail.get("search_title", "") # From search_engine
            item_row_dict["Search Result Snippet"] = item_detail.get("search_snippet", "") # From search_engine
            item_row_dict["Scraped Page Title"] = item_detail.get("page_title", item_detail.get("scraped_title", "")) # From scraper
            item_row_dict["Scraped Meta Description"] = item_detail.get("meta_description", "") # From scraper
            item_row_dict["Scraped OG Title"] = item_detail.get("og_title", "") # From scraper
            item_row_dict["Scraped OG Description"] = item_detail.get("og_description", "") # From scraper
            item_row_dict["Content Type"] = item_detail.get("content_type", "pdf" if item_detail.get("is_pdf") else "html") # From scraper or inferred
            item_row_dict["LLM Summary (Individual)"] = item_detail.get("llm_summary", "") # From llm_processor
            
            # Populate extraction query fields
            # Only show query text if there's corresponding extracted info or a score
            item_row_dict["LLM Extraction Query 1"] = query1_text_for_sheet if item_detail.get("llm_extracted_info_q1") or item_detail.get("llm_relevancy_score_q1") is not None else ""
            item_row_dict["LLM Extracted Info (Q1)"] = item_detail.get("llm_extracted_info_q1", "") # Content only
            
            item_row_dict["LLM Extraction Query 2"] = query2_text_for_sheet if item_detail.get("llm_extracted_info_q2") or item_detail.get("llm_relevancy_score_q2") is not None else ""
            item_row_dict["LLM Extracted Info (Q2)"] = item_detail.get("llm_extracted_info_q2", "") # Content only
            
            item_row_dict["Scraping Error"] = item_detail.get("scraping_error", item_detail.get("error_message", "")) # From scraper or process_manager
            item_row_dict["Main Text (Truncated)"] = truncated_main_text
            
            rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
            if idx < 2: # Print first couple of item rows for debug
                 print(f"DEBUG (data_storage): Prepared item row {idx+1}: {rows_to_append[-1][:5]}...") # DEBUG
    else:
        print("DEBUG (data_storage): item_data_list is empty or None. No item detail rows will be prepared.") # DEBUG


    if not rows_to_append: # Should not happen if batch summary is always added
        print("DEBUG (data_storage): No rows to append at all (not even batch summary). This is unexpected. Returning False.") # DEBUG
        return False
    
    # If only the batch summary row is present and item_data_list was empty, still write it.
    if len(rows_to_append) == 1 and rows_to_append[0][0] == "Batch Summary" and not item_data_list:
        print(f"DEBUG (data_storage): Only batch summary row to append (no items).") #DEBUG
        # Fall through to the main append_rows call

    print(f"DEBUG (data_storage): Total rows to append: {len(rows_to_append)}") #DEBUG
        
    try:
        print(f"DEBUG (data_storage): Attempting to append {len(rows_to_append)} total rows to worksheet '{worksheet.title}'.") # DEBUG
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"DEBUG (data_storage): Successfully appended rows to worksheet '{worksheet.title}'.") # DEBUG
        return True
    except Exception as e_append:
        st.error(f"Google Sheets Error: Failed to write {len(rows_to_append)} batch data rows to '{worksheet.title}': {e_append}")
        print(f"ERROR in write_batch_summary_and_items_to_sheet (appending rows): {e_append}") # DEBUG
        print(traceback.format_exc()) # DEBUG
        return False

# --- if __name__ == '__main__': block for testing ---
# (Your test block from v1.5.4 remains here, unchanged for this response, but ensure it uses
# keys consistent with what process_manager.py would produce for item_data_list if you run it.)
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.5 - Debug Prints)")
    try:
        # To run this test, ensure config.py is in a place Python can find it,
        # or adjust the import path. For example, if config.py is one level up:
        # import sys
        # sys.path.append('..')
        from modules.config import load_config # Assuming it's in modules folder relative to a potential test runner
        
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
                
                if st.button("TEST: Ensure Master Header (v1.5.5 - Check Sheet First)"):
                    ensure_master_header(worksheet_test)
                    st.write("Ensure Master Header call completed. Check sheet and terminal logs.")

                st.subheader("Test Data Writing (Multi-Query)")
                if st.button("Write Sample Batch Data (Multi-Query)"):
                    # ensure_master_header(worksheet_test) # Call this first if sheet might be new/changed
                    test_batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    test_consolidated_summary = "Overall summary for 'multi-query testing' batch. This is a test with debugs."
                    test_topic_context = "Multi-Query Testing with Debugs v1.5.5"
                    test_extraction_queries = ["What is the main color?", "What is the shape of the object?"]
                    test_item_list = [
                        { # Item 1 data - keys should match what process_manager produces for results_data
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "blue circle",
                            "url": "http://example.com/blue-circle", "page_title": "Blue Circles Info", 
                            "meta_description": "All about blue circles.", "og_title": "Blue Circles OG", "is_pdf": False,
                            "llm_summary": "This page discusses blue circles and their properties.", 
                            "source_query_type": "Original", "content_type": "text/html",
                            "main_content_display": "The main content about blue circles...", # or "scraped_main_text"
                            "llm_extracted_info_q1": "The main color is blue.", "llm_relevancy_score_q1": 5,
                            "llm_extracted_info_q2": "The shape of the object is a circle.", "llm_relevancy_score_q2": 5,
                            "search_title": "Search Result: Blue Circles", "search_snippet": "Find all info on blue circles here."
                        },
                        { # Item 2 data
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "red square",
                            "url": "http://example.com/red-square", "page_title": "Red Squares Info", "is_pdf": True,
                            "meta_description": "All about red squares.", "og_description": "Red Squares OG desc", 
                            "source_query_type": "LLM-Generated", "content_type": "application/pdf",
                            "main_content_display": "The main content about red squares from a PDF...",
                            "llm_summary": "This document is about red squares.",
                            "llm_extracted_info_q1": "The main color is red.", "llm_relevancy_score_q1": 5,
                            "llm_extracted_info_q2": "The shape of the object is a square.", "llm_relevancy_score_q2": 5,
                            "search_title": "Search Result: Red Squares", "search_snippet": "Find all info on red squares here."
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
