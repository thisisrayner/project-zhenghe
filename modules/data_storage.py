# modules/data_storage.py
# Version 1.5.8:
# - Added LLM Relevancy Score (Q1) and (Q2) to MASTER_HEADER and sheet writing logic.
# Previous versions:
# - Version 1.5.7: Refined logic and logging for when item_data_list is empty.

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
    # ... (Exact same code as in v1.5.7) ...
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
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
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
                if spreadsheet_name: st.warning(f"Opening by ID ('{spreadsheet_id}') failed. Attempting by name: '{spreadsheet_name}' as fallback..."); print(f"DEBUG (data_storage): Opening by ID failed, falling back to name '{spreadsheet_name}'.") 
                else: return None 
            except Exception as e_id_other: 
                st.error(f"Google Sheets: Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}")
                print(f"ERROR (data_storage): Unexpected error opening by ID '{spreadsheet_id}': {e_id_other}") 
                if spreadsheet_name: st.warning(f"Opening by ID failed. Attempting by name: '{spreadsheet_name}' as fallback..."); print(f"DEBUG (data_storage): Opening by ID failed (unexpected error), falling back to name '{spreadsheet_name}'.") 
                else: return None
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
    "LLM Extraction Query 1", "LLM Extracted Info (Q1)", "LLM Relevancy Score (Q1)", # ADDED SCORE Q1
    "LLM Extraction Query 2", "LLM Extracted Info (Q2)", "LLM Relevancy Score (Q2)", # ADDED SCORE Q2
    "Scraping Error", "Main Text (Truncated)"
]

def ensure_master_header(worksheet: gspread.Worksheet) -> None:
    # ... (Exact same code as in v1.5.7) ...
    print(f"DEBUG (data_storage): ensure_master_header called for worksheet: '{worksheet.title}' in spreadsheet '{worksheet.spreadsheet.title}'") 
    header_action_needed = False; action_reason = ""
    try:
        current_row1_values = worksheet.row_values(1) 
        if not current_row1_values: header_action_needed = True; action_reason = "Row 1 is empty."
        elif all(cell == '' for cell in current_row1_values): header_action_needed = True; action_reason = "Row 1 is blank."
        elif current_row1_values != MASTER_HEADER:
            header_action_needed = True; action_reason = (f"Header mismatch. Current (len {len(current_row1_values)}): {str(current_row1_values[:5]) if current_row1_values else 'N/A'}..., Expected (len {len(MASTER_HEADER)}): {MASTER_HEADER[:5]}...")
            print(f"DEBUG (data_storage): Header mismatch. Current: {current_row1_values}, Expected: {MASTER_HEADER}") 
        else: print("DEBUG (data_storage): Header matches MASTER_HEADER.") 
    except (gspread.exceptions.APIError, IndexError, gspread.exceptions.CellNotFound) as e: header_action_needed = True; action_reason = f"Could not read Row 1 ({type(e).__name__}: {e})."
    except Exception as e_check_header: st.error(f"Google Sheets: Unexpected error checking header: {e_check_header}."); header_action_needed = True; action_reason = f"Unexpected error: {e_check_header}."
    if header_action_needed:
        st.info(f"Google Sheets Header Info: {action_reason} Writing/Overwriting MASTER_HEADER to '{worksheet.title}'.")
        print(f"DEBUG (data_storage): Action for header: {action_reason}. Writing MASTER_HEADER.") 
        try:
            worksheet.update('A1', [MASTER_HEADER], value_input_option='USER_ENTERED') 
            st.success(f"Google Sheets: MASTER_HEADER written/updated in '{worksheet.title}'.")
            print(f"DEBUG (data_storage): Wrote MASTER_HEADER to '{worksheet.title}'") 
        except Exception as e_write_header: st.error(f"Google Sheets ERROR: Failed to write/update MASTER_HEADER to '{worksheet.title}': {e_write_header}"); print(f"ERROR (data_storage): Failed to write MASTER_HEADER: {e_write_header}") 

def write_batch_summary_and_items_to_sheet(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_data_list: List[Dict[str, Any]],
    extraction_queries_list: List[str], 
    main_text_truncate_limit: int = 10000
) -> bool:
    # ... (Initial debug prints as in v1.5.7) ...
    print(f"DEBUG (data_storage v1.5.8): write_batch_summary_and_items_to_sheet called.")
    if not worksheet:
        st.error("Google Sheets Error: No valid worksheet provided for writing batch data.")
        return False

    rows_to_append: List[List[Any]] = []
    items_successfully_prepared = 0

    if consolidated_summary or item_data_list:
        batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
        batch_summary_row_dict["Record Type"] = "Batch Summary"
        batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
        batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else ("N/A" if item_data_list else "No data processed.")
        batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
        batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
        rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
    else:
        print("DEBUG (data_storage): No consolidated summary AND no item data. Not preparing batch summary row.")

    query1_text_for_sheet = extraction_queries_list[0] if len(extraction_queries_list) > 0 and extraction_queries_list[0] else ""
    query2_text_for_sheet = extraction_queries_list[1] if len(extraction_queries_list) > 1 and extraction_queries_list[1] else ""

    if item_data_list: 
        for idx, item_detail in enumerate(item_data_list):
            try:
                main_text_raw = item_detail.get("main_content_display", item_detail.get("scraped_main_text")) 
                main_text_for_sheet = str(main_text_raw) if main_text_raw is not None else ""
                truncated_main_text = (main_text_for_sheet[:main_text_truncate_limit] + "..." if (len(main_text_for_sheet) > main_text_truncate_limit) else main_text_for_sheet)
                
                item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
                item_row_dict["Record Type"] = "Item Detail"; item_row_dict["Batch Timestamp"] = batch_timestamp
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
                
                # Populate Q1 info including score
                item_row_dict["LLM Extraction Query 1"] = query1_text_for_sheet if item_detail.get("llm_extracted_info_q1") or item_detail.get("llm_relevancy_score_q1") is not None else ""
                item_row_dict["LLM Extracted Info (Q1)"] = item_detail.get("llm_extracted_info_q1", "")
                item_row_dict["LLM Relevancy Score (Q1)"] = item_detail.get("llm_relevancy_score_q1", "") # ADDED

                # Populate Q2 info including score
                item_row_dict["LLM Extraction Query 2"] = query2_text_for_sheet if item_detail.get("llm_extracted_info_q2") or item_detail.get("llm_relevancy_score_q2") is not None else ""
                item_row_dict["LLM Extracted Info (Q2)"] = item_detail.get("llm_extracted_info_q2", "")
                item_row_dict["LLM Relevancy Score (Q2)"] = item_detail.get("llm_relevancy_score_q2", "") # ADDED
                
                item_row_dict["Scraping Error"] = item_detail.get("scraping_error", item_detail.get("error_message", ""))
                item_row_dict["Main Text (Truncated)"] = truncated_main_text
                
                rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
                items_successfully_prepared += 1
            except Exception as e_item_prep:
                print(f"ERROR (data_storage): Failed to prepare item {idx+1} ({item_detail.get('url', 'N/A')}) for sheet: {e_item_prep}")
                error_row_dict = {header: "" for header in MASTER_HEADER}
                error_row_dict["Record Type"] = "Item Error"; error_row_dict["URL"] = item_detail.get('url', 'N/A')
                error_row_dict["Scraping Error"] = f"Failed to prepare for GSheet: {e_item_prep}"
                rows_to_append.append([error_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
    else: 
        print("DEBUG (data_storage): item_data_list is empty or None. No item detail rows will be prepared.") 

    if not rows_to_append: 
        print("DEBUG (data_storage): No rows prepared. Nothing to write.") 
        return False
        
    try:
        print(f"DEBUG (data_storage): Appending {len(rows_to_append)} rows to '{worksheet.title}'.") 
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"DEBUG (data_storage): Appended rows to '{worksheet.title}'.") 
        return True
    except Exception as e_append:
        st.error(f"Google Sheets Error: Failed to write {len(rows_to_append)} rows to '{worksheet.title}': {e_append}")
        print(f"ERROR in write_batch_summary_and_items_to_sheet (appending): {e_append}") 
        print(traceback.format_exc()) 
        return False

if __name__ == '__main__':
    # ... (Test block from v1.5.7 can remain here) ...
    # Make sure to update the test_items in the __main__ block if you want to test the new score fields.
    # Example:
    # test_items = [
    #     {"main_content_display": "Item 1 text", "url":"url1", "llm_relevancy_score_q1": 5, "llm_extracted_info_q1": "Info for Q1"}, 
    # ]
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.8)")
    try:
        try: from modules.config import load_config
        except ImportError: import sys; sys.path.insert(0, '..'); from modules.config import load_config
        cfg_test = load_config()
        if cfg_test and cfg_test.sheets.service_account_info and \
           (cfg_test.sheets.spreadsheet_id or cfg_test.sheets.spreadsheet_name): # Corrected attribute
            st.info("Attempting to connect...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.sheets.service_account_info, cfg_test.sheets.spreadsheet_id, # Corrected attribute
                cfg_test.sheets.spreadsheet_name, cfg_test.sheets.worksheet_name # Corrected attribute
            )
            if worksheet_test:
                st.success(f"Connected to: {worksheet_test.spreadsheet.title} -> {worksheet_test.title}")
                if st.button("TEST: Ensure Master Header (v1.5.8)"):
                    ensure_master_header(worksheet_test); st.write("Header check done.")
                if st.button("Write Sample Batch Data (v1.5.8)"):
                    test_items_data = [
                        {"main_content_display": "Item 1 text", "url":"url1", "timestamp": "ts1", "keyword_searched": "kw1", 
                         "llm_extracted_info_q1": "Q1 info for item 1", "llm_relevancy_score_q1": 5,
                         "llm_extracted_info_q2": "Q2 info for item 1", "llm_relevancy_score_q2": 4}, 
                        {"main_content_display": None, "url":"url2", "timestamp": "ts2", "keyword_searched": "kw2", "llm_relevancy_score_q1": 1}
                    ]
                    success = write_batch_summary_and_items_to_sheet(worksheet_test, time.strftime("%Y%m%d-%H%M%S"), "Test Summary from v1.5.8", "Test Topic", test_items_data, ["Q1: What is X?", "Q2: Tell me about Y?"])
                    if success: st.success("Sample data written.")
                    else: st.error("Sample data write FAILED.")
            else: st.warning("Could not connect to GSheet.")
        else: st.warning("GSheet config missing or incomplete in secrets (service_account_info, spreadsheet_id/name).")
    except Exception as e: st.error(f"Test Error: {e}"); print(traceback.format_exc())


# // end of modules/data_storage.py
