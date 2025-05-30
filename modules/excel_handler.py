# modules/excel_handler.py
# Version 1.0.0: Initial module for Excel export functionalities.
"""
Handles the creation and formatting of Excel files for exporting results.
"""

import pandas as pd
from io import BytesIO
from typing import List, Dict, Any, Optional

def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_query: str) -> pd.DataFrame:
    """
    Prepares a Pandas DataFrame for the item details sheet in the Excel export.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      the processed data for an individual item.
        last_extract_query: The specific information extraction query used, if any.

    Returns:
        A Pandas DataFrame containing the item details, ready for Excel export.
    """
    item_details_for_excel: List[Dict[str, Any]] = []
    excel_item_headers: List[str] = [
        "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL",
        "Search Result Title", "Search Result Snippet", "Scraped Page Title",
        "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description",
        "Content Type", "LLM Summary (Individual)", "LLM Extracted Info (Query)",
        "LLM Extraction Query", "Scraping Error", "Main Text (Truncated)"
    ]

    for item_val_excel in results_data:
        row_data_excel: Dict[str, Any] = {
            "Batch Timestamp": item_val_excel.get("timestamp"), # Assuming batch timestamp is the same as item for now
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
            "LLM Extracted Info (Query)": item_val_excel.get("llm_extracted_info"),
            "LLM Extraction Query": last_extract_query if item_val_excel.get("llm_extracted_info") else "",
            "Scraping Error": item_val_excel.get("scraping_error"),
            "Main Text (Truncated)": (str(item_val_excel.get("scraped_main_text", ""))[:10000] + "...")
                                       if item_val_excel.get("scraped_main_text") and \
                                          len(str(item_val_excel.get("scraped_main_text", ""))) > 10000
                                       else str(item_val_excel.get("scraped_main_text", ""))
        }
        item_details_for_excel.append({header: row_data_excel.get(header, "") for header in excel_item_headers})

    return pd.DataFrame(item_details_for_excel, columns=excel_item_headers)

def prepare_consolidated_summary_df(
    consolidated_summary_text: Optional[str],
    results_data_count: int,
    last_keywords: str,
    last_extract_query: Optional[str],
    batch_timestamp: str
) -> Optional[pd.DataFrame]:
    """
    Prepares a Pandas DataFrame for the consolidated summary sheet.

    Args:
        consolidated_summary_text: The text of the consolidated summary.
        results_data_count: The number of items that contributed to the summary.
        last_keywords: The keywords used for the batch.
        last_extract_query: The specific information extraction query used, if any.
        batch_timestamp: The timestamp for the batch processing.

    Returns:
        A Pandas DataFrame with the consolidated summary, or None if no summary.
    """
    if consolidated_summary_text and not str(consolidated_summary_text).lower().startswith("error:"):
        last_run_keywords_excel_display: List[str] = [k.strip() for k in last_keywords.split(',') if k.strip()]
        topic_display_excel: str = last_run_keywords_excel_display[0] if len(last_run_keywords_excel_display) == 1 else \
                                   (f"Topics: {', '.join(last_run_keywords_excel_display[:3])}{'...' if len(last_run_keywords_excel_display) > 3 else ''}"
                                    if last_run_keywords_excel_display else "General Batch")
        excel_consolidation_note = "General Overview"
        if last_extract_query and last_extract_query.strip():
            excel_consolidation_note = f"Focused Overview on: '{last_extract_query}'"

        consolidated_data_excel: Dict[str, List[Any]] = {
            "Batch Timestamp": [batch_timestamp],
            "Topic/Keywords": [topic_display_excel],
            "Consolidated Summary": [consolidated_summary_text],
            "Source Items Count": [results_data_count],
            "Consolidation Note": [excel_consolidation_note]
        }
        return pd.DataFrame(consolidated_data_excel)
    return None

def to_excel_bytes(
    df_item_details: pd.DataFrame,
    df_consolidated_summary: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Converts DataFrames into an Excel file (bytes).

    Args:
        df_item_details: DataFrame with detailed item results.
        df_consolidated_summary: Optional DataFrame with the consolidated summary.

    Returns:
        Bytes representing the Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details')
        if df_consolidated_summary is not None and not df_consolidated_summary.empty:
            df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

# end of modules/excel_handler.py
