# modules/excel_handler.py
# Version 1.1.1: Comprehensive docstring updates for all functions and module.
# Version 1.1.0: Added support for two distinct LLM extraction queries in Excel output.
"""
Handles the creation and formatting of Excel files for exporting application results.

This module provides functions to convert processed item data and consolidated
summaries into structured Pandas DataFrames and then into a downloadable
Excel file format (.xlsx) with separate sheets for detailed item results and
the overall summary.
"""

import pandas as pd
from io import BytesIO
from typing import List, Dict, Any, Optional

def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_queries: List[str]) -> pd.DataFrame:
    """
    Prepares a Pandas DataFrame for the 'Item_Details' sheet in the Excel export.

    This function transforms a list of dictionaries, where each dictionary represents
    a processed item (e.g., a scraped URL with its metadata, LLM summary, and
    extracted information), into a Pandas DataFrame. It includes columns for
    both LLM extraction queries and their corresponding extracted information.

    Args:
        results_data: List[Dict[str, Any]]: A list of dictionaries, where each
            dictionary contains the data for a single processed item. Expected keys
            include 'timestamp', 'keyword_searched', 'url', 'llm_summary',
            'llm_extracted_info_q1', 'llm_extracted_info_q2', etc.
        last_extract_queries: List[str]: A list containing the text of the
            extraction queries used. The first element is Q1, the second is Q2.
            Used to populate the 'LLM Extraction Query' columns.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an item and columns
            represent various details of the item, ready for Excel export.
    """
    item_details_for_excel: List[Dict[str, Any]] = []
    excel_item_headers: List[str] = [
        "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL",
        "Search Result Title", "Search Result Snippet", "Scraped Page Title",
        "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description",
        "Content Type", "LLM Summary (Individual)",
        "LLM Extraction Query 1", "LLM Extracted Info (Q1)",
        "LLM Extraction Query 2", "LLM Extracted Info (Q2)",
        "Scraping Error", "Main Text (Truncated)"
    ]

    query1_text = last_extract_queries[0] if len(last_extract_queries) > 0 and last_extract_queries[0] else ""
    query2_text = last_extract_queries[1] if len(last_extract_queries) > 1 and last_extract_queries[1] else ""

    for item_val_excel in results_data:
        row_data_excel: Dict[str, Any] = {
            "Batch Timestamp": item_val_excel.get("timestamp"), # Assuming item timestamp can serve as batch for individual rows
            "Item Timestamp": item_val_excel.get("timestamp"),
            "Keyword Searched": item_val_excel.get("keyword_searched"),
            "URL": item_val_excel.get("url"),
            "Search Result Title": item_val_excel.get("search_title"),
            "Search Result Snippet": item_val_excel.get("search_snippet"),
            "Scraped Page Title": item_val_excel.get("scraped_title"),
            "Scraped Meta Description": item_val_excel.get("meta_description"),
            "Scraped OG Title": item_val_excel.get("og_title"),
            "Scraped OG Description": item_val_excel.get("og_description"),
            "Content Type": item_val_excel.get("content_type"),
            "LLM Summary (Individual)": item_val_excel.get("llm_summary"),

            "LLM Extraction Query 1": query1_text if item_val_excel.get("llm_extracted_info_q1") else "",
            "LLM Extracted Info (Q1)": item_val_excel.get("llm_extracted_info_q1"),

            "LLM Extraction Query 2": query2_text if item_val_excel.get("llm_extracted_info_q2") else "",
            "LLM Extracted Info (Q2)": item_val_excel.get("llm_extracted_info_q2"),

            "Scraping Error": item_val_excel.get("scraping_error"),
            "Main Text (Truncated)": (str(item_val_excel.get("scraped_main_text", ""))[:10000] + "...")
                                      if item_val_excel.get("scraped_main_text") and len(str(item_val_excel.get("scraped_main_text", ""))) > 10000
                                      else str(item_val_excel.get("scraped_main_text", ""))
        }
        # Ensure all headers are present, filling with empty string if data is missing for a key
        item_details_for_excel.append({header: row_data_excel.get(header, "") for header in excel_item_headers})

    return pd.DataFrame(item_details_for_excel, columns=excel_item_headers)

def prepare_consolidated_summary_df(
    consolidated_summary_text: Optional[str],
    results_data_count: int,
    last_keywords: str,
    primary_last_extract_query: Optional[str],
    batch_timestamp: str
) -> Optional[pd.DataFrame]:
    """
    Prepares a Pandas DataFrame for the 'Consolidated_Summary' sheet.

    This function creates a DataFrame containing the consolidated LLM summary,
    details about the batch (timestamp, keywords), the number of source items,
    and a note indicating if the summary was focused on a specific query.

    Args:
        consolidated_summary_text: Optional[str]: The text of the consolidated
            summary. If None, or starts with "error:", no DataFrame is created.
        results_data_count: int: The number of individual items that were
            considered for the consolidated summary.
        last_keywords: str: A comma-separated string of keywords used for the batch.
        primary_last_extract_query: Optional[str]: The text of the primary
            extraction query (Q1), if provided. Used to note if the summary
            was focused.
        batch_timestamp: str: The timestamp for the batch processing run.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the consolidated summary details,
            or None if the consolidated_summary_text is empty, None, or an error message.
    """
    if not consolidated_summary_text or str(consolidated_summary_text).lower().startswith("error:"):
        return None

    last_run_keywords_excel_display: List[str] = [k.strip() for k in last_keywords.split(',') if k.strip()]
    topic_display_excel: str = last_run_keywords_excel_display[0] if len(last_run_keywords_excel_display) == 1 else \
                               (f"Topics: {', '.join(last_run_keywords_excel_display[:3])}{'...' if len(last_run_keywords_excel_display) > 3 else ''}"
                                if last_run_keywords_excel_display else "General Batch")

    excel_consolidation_note = "General Overview"
    if primary_last_extract_query and primary_last_extract_query.strip():
        excel_consolidation_note = f"Focused Overview on insights from Q1: '{primary_last_extract_query}'"

    consolidated_data_excel: Dict[str, List[Any]] = {
        "Batch Timestamp": [batch_timestamp],
        "Topic/Keywords": [topic_display_excel],
        "Consolidated Summary": [consolidated_summary_text],
        "Source Items Count": [results_data_count],
        "Consolidation Note": [excel_consolidation_note]
    }
    return pd.DataFrame(consolidated_data_excel)


def to_excel_bytes(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes:
    """
    Converts DataFrames for item details and an optional consolidated summary into Excel bytes.

    This function takes one mandatory DataFrame (item details) and an optional
    DataFrame (consolidated summary). It writes them to separate sheets
    ('Item_Details' and 'Consolidated_Summary') in an Excel file and returns
    the file content as a byte string, suitable for downloading.

    Args:
        df_item_details: pd.DataFrame: The DataFrame containing detailed information
            for each processed item.
        df_consolidated_summary: Optional[pd.DataFrame]: An optional DataFrame
            containing the consolidated summary. If None or empty, this sheet
            will not be added to the Excel file.

    Returns:
        bytes: A byte string representing the content of the generated .xlsx Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details')
        if df_consolidated_summary is not None and not df_consolidated_summary.empty:
            df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

# end of modules/excel_handler.py
