# modules/data_storage.py
# Version 1.5.4: Added support for two distinct LLM extraction queries in GSheet output.
# Assertive ensure_master_header and previous key fixes maintained.

"""
Handles data storage operations, primarily focused on Google Sheets integration.
"""
# ... (imports and get_gspread_worksheet remain the same as v1.5.3) ...
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time

@st.cache_resource
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],
    spreadsheet_name: Optional[str],
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    # ... (implementation from v1.5.3) ...
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
                st.error(f"Google Sheets APIError opening by ID '{spreadsheet_id}': {e_id_api}. Check ID, sharing permissions, and that Drive & Sheets APIs are enabled.")
                if spreadsheet_name: st.warning(f"Opening by ID failed for '{spreadsheet_id}'. Attempting by name: '{spreadsheet_name}' as fallback...")
                else: return None
            except Exception as e_id_other:
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                if spreadsheet_name: st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback...")
                else: return None
        if not spreadsheet and spreadsheet_name:
            try:
                spreadsheet = client.open(spreadsheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                st.error(f"Google Sheets Error: Spreadsheet '{spreadsheet_name}' not found by name. Please verify the name or ensure SPREADSHEET_ID is correctly set in secrets.")
                return None
            except Exception as e_name:
                st.error(f"Google Sheets Error: Error opening by name '{spreadsheet_name}': {e_name}")
                return None
        if not spreadsheet:
            st.error("Google Sheets Error: Could not open spreadsheet using provided ID or Name.")
            return None
        worksheet_obj: Optional[gspread.Worksheet] = None
        try:
            worksheet_obj = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            if worksheet_name == "Sheet1" and hasattr(spreadsheet, 'sheet1') and spreadsheet.sheet1:
                st.info(f"Worksheet '{worksheet_name}' not found in '{spreadsheet.title}'. Using the first available sheet ('{spreadsheet.sheet1.title}').")
                worksheet_obj = spreadsheet.sheet1
            else:
                st.warning(f"Worksheet '{worksheet_name}' not found in spreadsheet '{spreadsheet.title}'. Data cannot be written to this tab.")
                return None
        return worksheet_obj
    except Exception as e:
        st.error(f"Google Sheets connection/setup main error: {e}")
        return None


MASTER_HEADER: List[str] = [
    "Record Type",              # A
    "Batch Timestamp",          # B
    "Batch Consolidated Summary", # C
    "Batch Topic/Keywords",     # D
    "Items in Batch",           # E
    "Item Timestamp",           # F
    "Keyword Searched",         # G
    "URL",                      # H
    "Search Result Title",      # I
    "Search Result Snippet",    # J
    "Scraped Page Title",       # K
    "Scraped Meta Description", # L
    "Scraped OG Title",         # M
    "Scraped OG Description",   # N
    "Content Type",             # O
    "LLM Summary (Individual)", # P
    "LLM Extraction Query 1",   # Q (NEW)
    "LLM Extracted Info (Q1)",  # R (NEW - was "LLM Extracted Info (Query)")
    "LLM Extraction Query 2",   # S (NEW)
    "LLM Extracted Info (Q2)",  # T (NEW)
    "Scraping Error",           # U
    "Main Text (Truncated)"     # V
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    # ... (implementation from v1.5.3 - assertive header writing - remains the same) ...
    header_action_needed = False; action_reason = ""
    try:
        current_row1_values = worksheet.row_values(1)
        if not current_row1_values: header_action_needed = True; action_reason = "Row 1 is empty (no values returned)."
        elif all(cell == '' for cell in current_row1_values): header_action_needed = True; action_reason = "Row 1 is blank (all cells are empty strings)."
        elif current_row1_values != MASTER_HEADER:
            header_action_needed = True
            action_reason = (f"Row 1 header does not match expected MASTER_HEADER. "
                             f"Old: {str(current_row1_values[:5]) if current_row1_values else 'N/A'}... ({len(current_row1_values)} cols). "
                             f"New: {MASTER_HEADER[:5]}... ({len(MASTER_HEADER)} cols).")
    except (gspread.exceptions.APIError, IndexError, gspread.exceptions.CellNotFound) as e:
        header_action_needed = True; action_reason = f"Could not read Row 1 (typical for new/empty sheet: {type(e).__name__})."
    except Exception as e:
        st.error(f"ERROR: Unexpected error while checking header: {e}. Attempting to write header as fallback."); header_action_needed = True; action_reason = f"Unexpected error during header check: {e}."
    if header_action_needed:
        st.info(f"GSheets Header Info: {action_reason} Writing/Overwriting MASTER_HEADER.")
        try:
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED'); st.success(f"SUCCESS: MASTER_HEADER written/updated in Row 1 of worksheet '{worksheet.title}'.")
        except Exception as e_write: st.error(f"ERROR: Failed to write/update MASTER_HEADER to Row 1: {e_write}")


def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_data_list: List[Dict[str, Any]],
    extraction_queries_list: List[str], # MODIFIED: Expecting list of query strings
    main_text_truncate_limit: int = 10000
) -> bool:
    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        return False

    rows_to_append: List[List[Any]] = []

    # Prepare Batch Summary Row
    batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
    # ... (batch summary row population remains same)
    batch_summary_row_dict["Record Type"] = "Batch Summary"
    batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
    batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else "N/A or Error"
    batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
    batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
    rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])


    # Prepare Item Detail Rows
    query1_text_for_sheet = extraction_queries_list[0] if len(extraction_queries_list) > 0 and extraction_queries_list[0] else ""
    query2_text_for_sheet = extraction_queries_list[1] if len(extraction_queries_list) > 1 and extraction_queries_list[1] else ""

    for item_detail in item_data_list:
        main_text = item_detail.get("scraped_main_text", "")
        truncated_main_text = (main_text[:main_text_truncate_limit] + "..." if (main_text and len(main_text) > main_text_truncate_limit) else main_text)
        
        item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
        item_row_dict["Record Type"] = "Item Detail"
        item_row_dict["Batch Timestamp"] = batch_timestamp
        
        # ... (standard item fields populate as before) ...
        item_row_dict["Item Timestamp"] = item_detail.get("timestamp", batch_timestamp)
        item_row_dict["Keyword Searched"] = item_detail.get("keyword_searched", "")
        item_row_dict["URL"] = item_detail.get("url", "")
        item_row_dict["Search Result Title"] = item_detail.get("search_title", "")
        item_row_dict["Search Result Snippet"] = item_detail.get("search_snippet", "")
        item_row_dict["Scraped Page Title"] = item_detail.get("scraped_title", "")
        item_row_dict["Scraped Meta Description"] = item_detail.get("meta_description", "")
        item_row_dict["Scraped OG Title"] = item_detail.get("og_title", "")
        item_row_dict["Scraped OG Description"] = item_detail.get("og_description", "")
        item_row_dict["Content Type"] = item_detail.get("content_type", "")
        item_row_dict["LLM Summary (Individual)"] = item_detail.get("llm_summary", "")
        
        # Populate extraction query fields
        item_row_dict["LLM Extraction Query 1"] = query1_text_for_sheet if item_detail.get("llm_extracted_info_q1") else ""
        item_row_dict["LLM Extracted Info (Q1)"] = item_detail.get("llm_extracted_info_q1", "")
        
        item_row_dict["LLM Extraction Query 2"] = query2_text_for_sheet if item_detail.get("llm_extracted_info_q2") else ""
        item_row_dict["LLM Extracted Info (Q2)"] = item_detail.get("llm_extracted_info_q2", "")
        
        item_row_dict["Scraping Error"] = item_detail.get("scraping_error", "")
        item_row_dict["Main Text (Truncated)"] = truncated_main_text
        
        rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    if not rows_to_append: return False
        
    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Google Sheets Error: Failed to write batch data rows: {e}")
        return False

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    # ... (Test block can be updated to reflect sending a list of queries and checking new columns) ...
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.4 - Multi-Query Support)")
    try:
        from config import load_config
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
                
                if st.button("TEST: Ensure Master Header (v1.5.4 - Clear Sheet First)"):
                    ensure_master_header(worksheet_test)
                    st.write("Ensure Master Header call completed. Check sheet and logs.")

                st.subheader("Test Data Writing (Multi-Query)")
                if st.button("Write Sample Batch Data (Multi-Query)"):
                    ensure_master_header(worksheet_test) # Ensure header is correct before write
                    test_batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    test_consolidated_summary = "Overall summary for 'multi-query testing' batch."
                    test_topic_context = "Multi-Query Testing"
                    test_extraction_queries = ["What is the main color?", "What is the shape?"]
                    test_item_list = [
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "color test",
                            "url": "http://example.com/color1", "scraped_title": "Color Page 1", 
                            "meta_description": "All about colors.", "og_title": "Colors OG",
                            "llm_summary": "This page is about colors.",
                            "llm_extracted_info_q1": "Relevancy Score: 5/5\nThe main color is blue.",
                            "llm_extracted_info_q2": "Relevancy Score: 4/5\nThe shape is round."
                        },
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "keyword_searched": "shape test",
                            "url": "http://example.com/shape1", "scraped_title": "Shape Page 1",
                            "meta_description": "All about shapes.", "og_description": "Shapes OG desc",
                            "llm_summary": "This page is about shapes.",
                            "llm_extracted_info_q1": "Relevancy Score: 3/5\nThe main color is not mentioned.", # Q1 might fail
                            "llm_extracted_info_q2": "Relevancy Score: 5/5\nThe shape is square."
                        }
                    ]
                    batch_write_success = write_batch_summary_and_items_to_sheet(
                        worksheet=worksheet_test, batch_timestamp=test_batch_timestamp,
                        consolidated_summary=test_consolidated_summary, topic_context=test_topic_context,
                        item_data_list=test_item_list,
                        extraction_queries_list=test_extraction_queries # Pass the list of queries
                    )
                    if batch_write_success: st.success(f"Batch data written successfully.")
                    else: st.error("Failed to write sample batch data.")
        else: st.warning("Google Sheets configuration missing. Full test cannot run.")
    except ImportError: st.error("Could not import 'config' module.")
    except Exception as e: st.error(f"An error occurred: {e}")

# end of modules/data_storage.py
