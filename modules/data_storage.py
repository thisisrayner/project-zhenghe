# modules/data_storage.py
# Version 1.4: Unified header for cleaner sheet structure with batch summaries.

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time

# --- Google Sheets Client Initialization (Cached) ---
@st.cache_resource
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],
    spreadsheet_name: Optional[str],
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    # ... (get_gspread_worksheet function from v1.3 remains the same, no changes needed here) ...
    if not service_account_info: st.error("Google Sheets: Service account info not provided."); return None
    if not spreadsheet_id and not spreadsheet_name: st.error("Google Sheets: Neither Spreadsheet ID nor Name provided."); return None
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = None
        if spreadsheet_id:
            try: spreadsheet = client.open_by_key(spreadsheet_id)
            except Exception as e_id:
                st.error(f"GS: Error opening by ID '{spreadsheet_id}': {e_id}.")
                if spreadsheet_name: st.warning(f"Attempting by name: '{spreadsheet_name}' as fallback...")
                else: return None
        if not spreadsheet and spreadsheet_name:
            try: spreadsheet = client.open(spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound: st.error(f"Spreadsheet '{spreadsheet_name}' not found by name."); return None
            except Exception as e_name: st.error(f"GS: Error opening by name '{spreadsheet_name}': {e_name}"); return None
        if not spreadsheet: st.error("GS: Could not open spreadsheet."); return None
        try: worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            if worksheet_name == "Sheet1": worksheet = spreadsheet.sheet1
            else: st.warning(f"Worksheet '{worksheet_name}' not found."); return None
        return worksheet
    except Exception as e: st.error(f"Google Sheets connection/setup error: {e}"); return None

# --- UNIFIED MASTER HEADER ---
MASTER_HEADER = [
    "Record Type",              # New: "Batch Summary" or "Item Detail"
    "Batch Timestamp",          # For both batch summary and to group items
    "Batch Consolidated Summary", # Only for Batch Summary rows
    "Batch Topic/Keywords",     # Only for Batch Summary rows
    "Items in Batch",           # Only for Batch Summary rows
    # Item Specific Headers (will be blank for Batch Summary rows)
    "Item Timestamp",           # Specific timestamp of the item processing
    "Keyword Searched",
    "URL",
    "Search Result Title",
    "Search Result Snippet",
    "Scraped Page Title",
    "Scraped Meta Description",
    "Scraped OG Title",
    "Scraped OG Description",
    "LLM Summary (Individual)",
    "LLM Extracted Info (Query)",
    "LLM Extraction Query",
    "Scraping Error",
    "Main Text (Truncated)"
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    """
    Ensures the MASTER_HEADER is present in Row 1 of the worksheet.
    If the sheet is empty, it appends the header.
    If Row 1 exists but doesn't match, it logs a warning (more advanced handling could be added).
    """
    try:
        current_row1_values = worksheet.row_values(1) # Check if anything is in row 1
        if not current_row1_values or all(cell == '' for cell in current_row1_values): # If sheet is empty or first row blank
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED') # Write header to A1
            st.info(f"Initialized worksheet '{worksheet.title}' with MASTER_HEADER in Row 1.")
        elif current_row1_values != MASTER_HEADER:
            st.warning(f"Worksheet '{worksheet.title}' Row 1 header does not match expected MASTER_HEADER. Data appending might be misaligned if structure is different.")
            # For robustness, you might check column count or specific key columns.
            # If critical mismatch, could raise error or try to insert new header (complex).
    except gspread.exceptions.APIError as e: # Catch API errors, e.g. if sheet is truly empty and row_values(1) fails
        if 'exceeds grid limits' in str(e).lower() or isinstance(e, gspread.exceptions.CellNotFound): # More specific for empty sheet
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED')
            st.info(f"Initialized empty worksheet '{worksheet.title}' with MASTER_HEADER in Row 1 (APIError catch).")
        else:
            st.error(f"Google Sheets API error ensuring header: {e}")
    except Exception as e:
        st.error(f"Unexpected error ensuring header in Google Sheets: {e}")


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
    if not worksheet:
        st.error("Google Sheets: No valid worksheet provided for batch writing.")
        return False

    rows_to_append = []

    # 1. Prepare Batch Summary Row
    batch_summary_row_dict = {header: "" for header in MASTER_HEADER} # Initialize with blanks
    batch_summary_row_dict["Record Type"] = "Batch Summary"
    batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
    batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else "N/A or Error"
    batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
    batch_summary_row_dict["Items in Batch"] = len(item_data_list)
    rows_to_append.append([batch_summary_row_dict.get(col, "") for col in MASTER_HEADER])

    # 2. Prepare Item Detail Rows
    for item in item_data_list:
        main_text = item.get("scraped_main_text", "")
        truncated_main_text = (main_text[:main_text_truncate_limit] + "...") if main_text and len(main_text) > main_text_truncate_limit else main_text
        
        item_row_dict = {header: "" for header in MASTER_HEADER} # Initialize with blanks
        item_row_dict["Record Type"] = "Item Detail"
        item_row_dict["Batch Timestamp"] = batch_timestamp # Repeat batch timestamp for grouping
        # Batch specific fields will be blank for item rows
        item_row_dict["Item Timestamp"] = item.get("timestamp", "")
        item_row_dict["Keyword Searched"] = item.get("keyword_searched", "")
        item_row_dict["URL"] = item.get("url", "")
        item_row_dict["Search Result Title"] = item.get("search_title", "")
        item_row_dict["Search Result Snippet"] = item.get("search_snippet", "")
        item_row_dict["Scraped Page Title"] = item.get("scraped_title", "")
        item_row_dict["Scraped Meta Description"] = item.get("scraped_meta_description", "")
        item_row_dict["Scraped OG Title"] = item.get("scraped_og_title", "")
        item_row_dict["Scraped OG Description"] = item.get("scraped_og_description", "")
        item_row_dict["LLM Summary (Individual)"] = item.get("llm_summary", "")
        item_row_dict["LLM Extracted Info (Query)"] = item.get("llm_extracted_info", "")
        item_row_dict["LLM Extraction Query"] = extraction_query_text if item.get("llm_extracted_info") else "",
        item_row_dict["Scraping Error"] = item.get("scraping_error", "")
        item_row_dict["Main Text (Truncated)"] = truncated_main_text
        rows_to_append.append([item_row_dict.get(col, "") for col in MASTER_HEADER])

    if not rows_to_append:
        st.info("Google Sheets: No data (batch summary or items) to write.")
        return False
    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        # Success message handled by app.py
        return True
    except Exception as e:
        st.error(f"Google Sheets: Error writing batch data: {e}")
        return False

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.4 - Unified Header)")
    try:
        from config import load_config
        cfg_test = load_config()
        if cfg_test and cfg_test.gsheets.service_account_info and \
           (cfg_test.gsheets.spreadsheet_id or cfg_test.gsheets.spreadsheet_name):
            st.info("Attempting to connect to Google Sheets...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.gsheets.service_account_info,
                cfg_test.gsheets.spreadsheet_id,
                cfg_test.gsheets.spreadsheet_name,
                cfg_test.gsheets.worksheet_name
            )
            if worksheet_test:
                st.success(f"Connected to {worksheet_test.spreadsheet.title} -> {worksheet_test.title}")
                ensure_master_header(worksheet_test) # Test new header function
                
                st.subheader("Test Data Writing")
                if st.button("Write Sample Batch Data"):
                    batch_ts_test = time.strftime("%Y-%m-%d %H:%M:%S")
                    sample_items_test = [
                        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "unified item 1", "url": "http://example.com/unified1", "scraped_title": "Unified Item 1 Title", "llm_summary": "Summary for unified item 1."},
                        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "unified item 2", "url": "http://example.com/unified2", "scraping_error": "Error for unified item 2"}
                    ]
                    batch_written = write_batch_summary_and_items_to_sheet(
                        worksheet_test,
                        batch_timestamp=batch_ts_test,
                        consolidated_summary="This is a test consolidated batch summary for unified header.",
                        topic_context="Unified Test Topic",
                        item_data_list=sample_items_test,
                        extraction_query_text="Test query for items"
                    )
                    if batch_written:
                        st.write(f"Batch summary and {len(sample_items_test)} item rows written.")
                    else:
                        st.error("Failed to write batch data.")
        else: st.warning("GS config missing for full test.")
    except ImportError: st.error("Could not import 'config' module for testing.")
    except Exception as e: st.error(f"Test setup error: {e}")

# end of modules/data_storage.py
