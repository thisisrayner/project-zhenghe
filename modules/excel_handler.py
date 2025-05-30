# modules/excel_handler.py
# Version 1.1.0: Added support for two distinct LLM extraction queries in Excel output.
"""
Handles the creation and formatting of Excel files for exporting KSAT results.
"""

import pandas as pd
from io import BytesIO
from typing import List, Dict, Any, Optional

def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_queries: List[str]) -> pd.DataFrame: # MODIFIED
    """
    Prepares a Pandas DataFrame for the item details sheet in the Excel export.
    """
    item_details_for_excel: List[Dict[str, Any]] = []
    excel_item_headers: List[str] = [
        "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL",
        "Search Result Title", "Search Result Snippet", "Scraped Page Title",
        "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description",
        "Content Type", "LLM Summary (Individual)", 
        "LLM Extraction Query 1", "LLM Extracted Info (Q1)", # MODIFIED
        "LLM Extraction Query 2", "LLM Extracted Info (Q2)", # NEW
        "Scraping Error", "Main Text (Truncated)"
    ]
    
    query1_text = last_extract_queries[0] if len(last_extract_queries) > 0 and last_extract_queries[0] else ""
    query2_text = last_extract_queries[1] if len(last_extract_queries) > 1 and last_extract_queries[1] else ""


    for item_val_excel in results_data:
        row_data_excel: Dict[str, Any] = {
            "Batch Timestamp": item_val_excel.get("timestamp"),
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
        item_details_for_excel.append({header: row_data_excel.get(header, "") for header in excel_item_headers})
    
    return pd.DataFrame(item_details_for_excel, columns=excel_item_headers)

def prepare_consolidated_summary_df(
    consolidated_summary_text: Optional[str],
    results_data_count: int,
    last_keywords: str,
    primary_last_extract_query: Optional[str], # MODIFIED: Now just the primary for context
    batch_timestamp: str
) -> Optional[pd.DataFrame]:
    # ... (remains largely the same, uses primary_last_extract_query for the note) ...
    if not consolidated_summary_text or str(consolidated_summary_text).lower().startswith("error:"):
        return None

    last_run_keywords_excel_display: List[str] = [k.strip() for k in last_keywords.split(',') if k.strip()]
    topic_display_excel: str = last_run_keywords_excel_display[0] if len(last_run_keywords_excel_display) == 1 else \
                               (f"Topics: {', '.join(last_run_keywords_excel_display[:3])}{'...' if len(last_run_keywords_excel_display) > 3 else ''}"
                                if last_run_keywords_excel_display else "General Batch")

    excel_consolidation_note = "General Overview"
    if primary_last_extract_query and primary_last_extract_query.strip(): # Use the primary for this note
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
    # ... (remains the same) ...
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details')
        if df_consolidated_summary is not None and not df_consolidated_summary.empty:
            df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

# end of modules/excel_handler.py
