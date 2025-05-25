# modules/data_storage.py
# Version 1.3: Writes a batch consolidated summary as a distinct row before itemized data.

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
    # ... (get_gspread_worksheet function from v1.1 remains the same) ...
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


# --- Header Row Definitions ---
# Header for the BATCH SUMMARY row
BATCH_SUMMARY_HEADER = [
    "Batch Timestamp", "Batch Consolidated Summary", "Topic/Keywords Context", "Total Items in Batch"
    # Add more batch-level info if needed
]

# Header for the INDIVIDUAL ITEM rows
ITEM_DATA_HEADER = [
    "Item Timestamp", "Keyword Searched", "URL",
    "Search Result Title", "Search Result Snippet",
    "Scraped Page Title", "Scraped Meta Description",
    "Scraped OG Title", "Scraped OG Description",
    "LLM Summary (Individual)", "LLM Extracted Info (Query)", "LLM Extraction Query",
    "Scraping Error", "Main Text (Truncated)"
]

def ensure_sheet_structure(worksheet: gspread.Worksheet) -> None:
    """
    Checks if the sheet is empty. If so, adds a general note or a combined header.
    For this version, we assume the user knows the structure or the sheet is new.
    A more robust solution would check for specific headers for batch vs. item.
    For now, if empty, we'll add the ITEM_DATA_HEADER as it's more frequent.
    """
    try:
        all_values = worksheet.get_all_values()
        if not all_values: # Sheet is completely empty
            worksheet.append_row(["Note: This sheet contains Batch Summaries and Item Details. Item Detail header below."], value_input_option='USER_ENTERED')
            worksheet.append_row(ITEM_DATA_HEADER, value_input_option='USER_ENTERED')
            st.info(f"Initialized empty worksheet '{worksheet.title}' with headers.")
        # If not empty, we assume user has set it up or understands the mixed content.
        # A more complex check could verify if ITEM_DATA_HEADER exists somewhere.
    except Exception as e:
        st.error(f"Error ensuring sheet structure: {e}")

# --- Write Data Functions ---
def write_batch_summary_row(
    worksheet: gspread.Worksheet,
    batch_timestamp: str,
    consolidated_summary: Optional[str],
    topic_context: str,
    item_count: int
) -> bool:
    if not worksheet: return False
    try:
        # Prepare row according to BATCH_SUMMARY_HEADER (or a simplified version for this row type)
        summary_row_data = [
            batch_timestamp,
            consolidated_summary if consolidated_summary else "N/A or Error",
            topic_context,
            item_count
            # Pad with blanks if BATCH_SUMMARY_HEADER is longer and you want alignment
        ]
        # To make it visually distinct, you could leave other cells blank or add a type
        # Example: ["BATCH SUMMARY", batch_timestamp, consolidated_summary, topic_context, item_count, "", "", ...]
        # For now, a simpler distinct row:
        row_to_write = ["" for _ in ITEM_DATA_HEADER] # Create a blank row of item data length
        row_to_write[0] = batch_timestamp # A - Batch Timestamp
        row_to_write[1] = consolidated_summary if consolidated_summary else "N/A or Error" # B - Summary
        row_to_write[2] = f"CONTEXT: {topic_context}" # C - Topic
        row_to_write[3] = f"ITEMS IN BATCH: {item_count}" # D - Item count
        # The rest of the cells in this row will be blank according to ITEM_DATA_HEADER length

        worksheet.append_row(row_to_write, value_input_option='USER_ENTERED')
        # st.info("Batch consolidated summary row written to sheet.")
        return True
    except Exception as e:
        st.error(f"Google Sheets: Error writing batch summary row: {e}")
        return False


def write_item_data_to_sheet( # Renamed from write_data_to_sheet for clarity
    worksheet: gspread.Worksheet,
    item_data_list: List[Dict[str, Any]],
    extraction_query_text: Optional[str] = None,
    main_text_truncate_limit: int = 10000
) -> int:
    if not worksheet: st.error("Google Sheets: No valid worksheet for item data."); return 0
    
    rows_to_append = []
    for item in item_data_list:
        main_text = item.get("scraped_main_text", "")
        truncated_main_text = (main_text[:main_text_truncate_limit] + "...") if main_text and len(main_text) > main_text_truncate_limit else main_text
        
        # Map item_data keys to the ITEM_DATA_HEADER order
        row_dict = {
            "Item Timestamp": item.get("timestamp", ""), # Changed from "Timestamp"
            "Keyword Searched": item.get("keyword_searched", ""),
            "URL": item.get("url", ""),
            "Search Result Title": item.get("search_title", ""),
            "Search Result Snippet": item.get("search_snippet", ""),
            "Scraped Page Title": item.get("scraped_title", ""),
            "Scraped Meta Description": item.get("scraped_meta_description", ""),
            "Scraped OG Title": item.get("scraped_og_title", ""),
            "Scraped OG Description": item.get("scraped_og_description", ""),
            "LLM Summary (Individual)": item.get("llm_summary", ""), # Changed from "LLM Summary"
            "LLM Extracted Info (Query)": item.get("llm_extracted_info", ""),
            "LLM Extraction Query": extraction_query_text if item.get("llm_extracted_info") else "",
            "Scraping Error": item.get("scraping_error", ""),
            "Main Text (Truncated)": truncated_main_text
        }
        current_row = [row_dict.get(header_col, "") for header_col in ITEM_DATA_HEADER]
        rows_to_append.append(current_row)

    if not rows_to_append: st.info("Google Sheets: No item data to write."); return 0
    try:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        return len(rows_to_append)
    except Exception as e: st.error(f"Google Sheets: Error writing item data: {e}"); return 0

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Data Storage Module Test (Google Sheets v1.3 - Batch Summary)")
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
                ensure_sheet_structure(worksheet_test) # Test header function
                
                st.subheader("Test Data Writing")
                if st.button("Write Sample Batch Summary and Items"):
                    batch_ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    summary_written = write_batch_summary_row(
                        worksheet_test, batch_ts, "This is the overall batch summary.", "Test Topic", 2
                    )
                    if summary_written: st.write("Batch summary row written.")

                    sample_items = [
                        {"timestamp": batch_ts, "keyword_searched": "item 1", "url": "http://example.com/item1", "scraped_title": "Item 1 Title", "llm_summary": "Summary for item 1."},
                        {"timestamp": batch_ts, "keyword_searched": "item 2", "url": "http://example.com/item2", "scraping_error": "Error for item 2"}
                    ]
                    items_written = write_item_data_to_sheet(worksheet_test, sample_items)
                    st.write(f"{items_written} sample item rows written.")
        else: st.warning("GS config missing.")
    except ImportError: st.error("Could not import 'config'.")
    except Exception as e: st.error(f"Test setup error: {e}")

# end of modules/data_storage.py
