# modules/data_storage.py
# Version 1.5.7:
# - Refined logic and logging for when item_data_list is empty.
# - Added try-except around individual item processing in sheet prep for robustness.
# Version 1.5.6:
# - Added robust handling for `main_text` being None before len() or slicing.
# - Added more specific debug prints within item processing loop.

"""
Handles data storage operations, primarily focused on Google Sheets integration.
"""
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import Dict, List, Optional, Any
import time 
import traceback 

# get_gspread_worksheet and ensure_master_header remain IDENTICAL to v1.5.6
# For brevity, I will not repeat them here but assume they are present from the previous full code block.
# Make sure to copy them from the v1.5.6 full code I provided.

@st.cache_resource 
def get_gspread_worksheet(
    service_account_info: Optional[Dict[str, Any]],
    spreadsheet_id: Optional[str],
    spreadsheet_name: Optional[str], 
    worksheet_name: str = "Sheet1"
) -> Optional[gspread.Worksheet]:
    # ... (Exact same code as in v1.5.6) ...
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
    # ... (Exact same code as in v1.5.6) ...
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
    items_successfully_prepared = 0

    # Always prepare Batch Summary Row if a summary or items exist
    if consolidated_summary or item_data_list:
        batch_summary_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
        batch_summary_row_dict["Record Type"] = "Batch Summary"
        batch_summary_row_dict["Batch Timestamp"] = batch_timestamp
        batch_summary_row_dict["Batch Consolidated Summary"] = consolidated_summary if consolidated_summary else ("N/A" if item_data_list else "No data processed for this batch.")
        batch_summary_row_dict["Batch Topic/Keywords"] = topic_context
        batch_summary_row_dict["Items in Batch"] = len(item_data_list) if item_data_list else 0
        rows_to_append.append([batch_summary_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
        print(f"DEBUG (data_storage): Prepared batch summary row: {str(rows_to_append[0][:5])[:200]}...") 
    else:
        print("DEBUG (data_storage): No consolidated summary AND no item data. Not preparing batch summary row.")

    query1_text_for_sheet = extraction_queries_list[0] if len(extraction_queries_list) > 0 and extraction_queries_list[0] else ""
    query2_text_for_sheet = extraction_queries_list[1] if len(extraction_queries_list) > 1 and extraction_queries_list[1] else ""

    if item_data_list: 
        for idx, item_detail in enumerate(item_data_list):
            try: # Add try-except around each item's processing
                print(f"DEBUG (data_storage): Processing item {idx + 1}/{len(item_data_list)} for sheet: {item_detail.get('url', 'N/A')}") 
                
                main_text_raw = item_detail.get("main_content_display", item_detail.get("scraped_main_text")) 
                main_text_for_sheet = "" 
                if isinstance(main_text_raw, str): 
                    main_text_for_sheet = main_text_raw
                
                print(f"DEBUG (data_storage): item {idx + 1} main_text type: {type(main_text_for_sheet)}, len: {len(main_text_for_sheet)}. Truncate limit: {main_text_truncate_limit}")

                truncated_main_text = (main_text_for_sheet[:main_text_truncate_limit] + "..." if (len(main_text_for_sheet) > main_text_truncate_limit) else main_text_for_sheet)
                
                item_row_dict: Dict[str, Any] = {header: "" for header in MASTER_HEADER}
                # ... (All item_row_dict population from v1.5.6)
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
                item_row_dict["LLM Extraction Query 1"] = query1_text_for_sheet if item_detail.get("llm_extracted_info_q1") or item_detail.get("llm_relevancy_score_q1") is not None else ""
                item_row_dict["LLM Extracted Info (Q1)"] = item_detail.get("llm_extracted_info_q1", "")
                item_row_dict["LLM Extraction Query 2"] = query2_text_for_sheet if item_detail.get("llm_extracted_info_q2") or item_detail.get("llm_relevancy_score_q2") is not None else ""
                item_row_dict["LLM Extracted Info (Q2)"] = item_detail.get("llm_extracted_info_q2", "")
                item_row_dict["Scraping Error"] = item_detail.get("scraping_error", item_detail.get("error_message", ""))
                item_row_dict["Main Text (Truncated)"] = truncated_main_text
                
                rows_to_append.append([item_row_dict.get(col_name, "") for col_name in MASTER_HEADER])
                items_successfully_prepared += 1
                if idx < 2: 
                     print(f"DEBUG (data_storage): Prepared item row {idx+1}: {str(rows_to_append[-1][:5])[:200]}...") 
            except Exception as e_item_prep:
                print(f"ERROR (data_storage): Failed to prepare item {idx+1} ({item_detail.get('url', 'N/A')}) for sheet: {e_item_prep}")
                print(traceback.format_exc())
                # Optionally, append a row indicating this item failed to process for sheets
                error_row_dict = {header: "" for header in MASTER_HEADER}
                error_row_dict["Record Type"] = "Item Error"; error_row_dict["URL"] = item_detail.get('url', 'N/A')
                error_row_dict["Scraping Error"] = f"Failed to prepare for GSheet: {e_item_prep}"
                rows_to_append.append([error_row_dict.get(col_name, "") for col_name in MASTER_HEADER])

    else: # item_data_list is None or empty
        print("DEBUG (data_storage): item_data_list is empty or None. No item detail rows will be prepared.") 

    if not rows_to_append: 
        print("DEBUG (data_storage): No rows (not even batch summary) were prepared to append. Nothing to write.") 
        return False # Nothing to write
    
    print(f"DEBUG (data_storage): Total rows prepared to append: {len(rows_to_append)}. Items successfully prepared: {items_successfully_prepared}")
        
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

# --- if __name__ == '__main__': block for testing (from v1.5.6) ---
if __name__ == '__main__':
    # ... (Test block from v1.5.6 can remain here) ...
    # Ensure it's updated if MASTER_HEADER or item_detail keys change significantly.
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.5.7)")
    try:
        try: from modules.config import load_config
        except ImportError:
            import sys; sys.path.insert(0, '..'); from modules.config import load_config
        cfg_test = load_config()
        if cfg_test and cfg_test.gsheets.service_account_info and \
           (cfg_test.gsheets.spreadsheet_id or cfg_test.gsheets.spreadsheet_name):
            st.info("Attempting to connect...")
            worksheet_test = get_gspread_worksheet(
                cfg_test.gsheets.service_account_info, cfg_test.gsheets.spreadsheet_id,
                cfg_test.gsheets.spreadsheet_name, cfg_test.gsheets.worksheet_name
            )
            if worksheet_test:
                st.success(f"Connected to: {worksheet_test.spreadsheet.title} -> {worksheet_test.title}")
                if st.button("TEST: Ensure Master Header (v1.5.7)"):
                    ensure_master_header(worksheet_test); st.write("Header check done.")
                if st.button("Write Sample Batch Data (v1.5.7)"):
                    # ... (Sample data from v1.5.6 test block, ensure keys are consistent) ...
                    test_items = [
                        {"main_content_display": "Item 1 text", "url":"url1"}, 
                        {"main_content_display": None, "url":"url2"},
                        {"main_content_display": "", "url":"url3"}
                    ]
                    success = write_batch_summary_and_items_to_sheet(worksheet_test, time.strftime("%Y%m%d-%H%M%S"), "Test Summary", "Test Topic", test_items, ["Q1?", "Q2?"])
                    if success: st.success("Sample data written.")
                    else: st.error("Sample data write FAILED.")
            else: st.warning("Could not connect to GSheet.")
        else: st.warning("GSheet config missing.")
    except Exception as e: st.error(f"Test Error: {e}"); print(traceback.format_exc())

# end of modules/data_storage.py
