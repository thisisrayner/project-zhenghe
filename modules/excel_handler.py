# modules/excel_handler.py
# Version: 1.2.0 (Incorporated character cleaning for Excel export and comprehensive docstrings)

"""
Handles the creation and formatting of Excel files for exporting application results.

This module provides functions to:
1. Prepare Pandas DataFrames from processed item data and consolidated summaries.
2. Clean string data within these DataFrames to remove illegal XML characters
   that would prevent successful export to .xlsx format.
3. Convert the cleaned DataFrames into a downloadable Excel file (.xlsx)
   with separate sheets for detailed item results and the overall summary.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import io
import re

# Define the regex for illegal XML characters.
# This regex targets control characters from \x00-\x1F, excluding
# tab (\x09), newline (\x0A), and carriage return (\x0D).
ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def _clean_string_for_excel(text: Any) -> Any:
    """
    Removes or replaces illegal XML characters from a string.

    These characters can cause errors when writing to .xlsx files with openpyxl.
    The function specifically targets control characters in the ASCII range
    0-31, excluding tab, newline, and carriage return, which are generally
    permissible.

    Args:
        text: The input value. If it's a string, it will be cleaned by
              removing illegal characters. Otherwise, it will be returned as is.

    Returns:
        The cleaned string with illegal characters removed, or the original
        value if it was not a string.
    """
    if isinstance(text, str):
        return ILLEGAL_CHARACTERS_RE.sub('', text) # Replace illegal characters with an empty string
    return text

def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_queries: List[str]) -> pd.DataFrame:
    """
    Prepares a Pandas DataFrame for the 'Item_Details' sheet in the Excel export.

    This function transforms a list of dictionaries, where each dictionary represents
    a processed item (e.g., a scraped URL with its metadata, LLM summary, and
    extracted information), into a Pandas DataFrame. It defines a standard set
    of columns and their order, populates extraction query text, and renames
    columns for better readability in the exported Excel file.

    The content of 'main_content_display' is assumed to be handled (e.g., truncated)
    before being passed into this function if necessary, or it will export the full content.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
            the data for a single processed item. Expected keys include
            'timestamp', 'keyword_searched', 'url', 'page_title', 
            'llm_summary', 'llm_extracted_info_q1', 'llm_relevancy_score_q1',
            'llm_extracted_info_q2', 'llm_relevancy_score_q2', 'main_content_display', etc.
        last_extract_queries: A list containing the text of the extraction
            queries used. The first element is assumed to be Q1, the second is Q2.
            These are used to populate the 'LLM Extraction Query' columns.

    Returns:
        A Pandas DataFrame where each row corresponds to a processed item and
        columns represent various details of that item, ready for Excel export.
    """
    df = pd.DataFrame(results_data)

    # Define a comprehensive list of desired columns and their order
    # These keys should align with what process_manager.py produces for each item
    columns_ordered = [
        'timestamp', 'keyword_searched', 'source_query_type', 'url', 'is_pdf',
        'page_title', 'meta_description', 'og_title', 'og_description',
        'pdf_document_title',
        'main_content_display', # This column should contain the text to be exported.
                                # Truncation should happen before this function if needed,
                                # or it's assumed the full text is desired here.
        'llm_summary',
        'llm_extraction_query_1_text', 'llm_extracted_info_q1', 'llm_relevancy_score_q1',
        'llm_extraction_query_2_text', 'llm_extracted_info_q2', 'llm_relevancy_score_q2',
        'error_message'
    ]

    # Ensure all desired columns exist in the DataFrame, adding them with None if missing
    for col in columns_ordered:
        if col not in df.columns:
            df[col] = None

    # Populate query text columns directly using last_extract_queries
    # The column names 'llm_extraction_query_1_text' and 'llm_extraction_query_2_text'
    # are used here to store the actual query texts for the Excel sheet.
    # The individual item data in results_data might already have specific query texts
    # if different queries were used per item, but this ensures a general column for the batch queries.
    if 'llm_extraction_query_1_text' in df.columns: # Check if column exists to avoid adding it if already handled
        df['llm_extraction_query_1_text'] = last_extract_queries[0] if len(last_extract_queries) > 0 and last_extract_queries[0] else None
    
    if 'llm_extraction_query_2_text' in df.columns:
        df['llm_extraction_query_2_text'] = last_extract_queries[1] if len(last_extract_queries) > 1 and last_extract_queries[1] else None


    # Select and reorder columns based on columns_ordered
    # Ensure only existing columns are selected to avoid KeyError
    existing_columns_to_order = [col for col in columns_ordered if col in df.columns]
    df = df[existing_columns_to_order]

    # Rename columns for better readability in Excel
    df.rename(columns={
        'timestamp': 'Timestamp',
        'keyword_searched': 'Keyword Searched',
        'source_query_type': 'Search Query Type', # e.g., 'Original', 'LLM-Generated'
        'url': 'URL',
        'is_pdf': 'Is PDF?',
        'page_title': 'Page Title (Scraped/PDF)', # Consolidate title source
        'meta_description': 'Meta Description (HTML)',
        'og_title': 'OpenGraph Title (HTML)',
        'og_description': 'OpenGraph Description (HTML)',
        'pdf_document_title': 'PDF Document Title (Metadata)', # Specific for PDFs if available
        'main_content_display': 'Main Content', # Text content used for analysis
        'llm_summary': 'LLM Item Summary',
        'llm_extraction_query_1_text': 'LLM Extraction Query 1',
        'llm_extracted_info_q1': 'LLM Extracted Info (Q1)',
        'llm_relevancy_score_q1': 'LLM Relevancy Score (Q1)',
        'llm_extraction_query_2_text': 'LLM Extraction Query 2',
        'llm_extracted_info_q2': 'LLM Extracted Info (Q2)',
        'llm_relevancy_score_q2': 'LLM Relevancy Score (Q2)',
        'error_message': 'Processing Error'
    }, inplace=True, errors='ignore') # errors='ignore' prevents error if a key to rename is not found

    return df


def prepare_consolidated_summary_df(
    consolidated_summary_text: Optional[str],
    results_data_count: int, # Total items processed in batch
    last_keywords: str,
    primary_llm_extract_query: Optional[str], # Q1 text
    secondary_llm_extract_query: Optional[str], # Q2 text
    batch_timestamp: str,
    focused_summary_source_count: Optional[int] = None # Number of items contributing to focused summary
) -> Optional[pd.DataFrame]:
    """
    Prepares a Pandas DataFrame for the 'Consolidated_Summary' sheet.

    This function creates a DataFrame containing the consolidated LLM summary,
    details about the batch (timestamp, keywords, extraction queries), the number
    of source items, and a note indicating whether the summary was general or
    focused (and on which queries).

    Args:
        consolidated_summary_text: The text of the consolidated summary.
            If None, or starts with "error:", no DataFrame is created.
        results_data_count: The total number of individual items processed in the batch.
        last_keywords: A comma-separated string of keywords used for the batch.
        primary_llm_extract_query: The text of the primary extraction query (Q1), if provided.
        secondary_llm_extract_query: The text of the secondary extraction query (Q2), if provided.
        batch_timestamp: The timestamp for the batch processing run.
        focused_summary_source_count: If the summary was focused, this is the count
            of items that met the criteria and were used as input. If None or 0,
            the summary is considered general or a fallback.

    Returns:
        A Pandas DataFrame with the consolidated summary details, or None if the
        consolidated_summary_text is empty, None, or an error message.
    """
    if not consolidated_summary_text or str(consolidated_summary_text).lower().startswith("error:"):
        return None

    summary_type_note = "General summary based on available item summaries."
    if focused_summary_source_count is not None and focused_summary_source_count > 0:
        query_parts = []
        if primary_llm_extract_query:
            query_parts.append(f"Q1 ('{primary_llm_extract_query}')")
        if secondary_llm_extract_query:
            query_parts.append(f"Q2 ('{secondary_llm_extract_query}')")
        
        if query_parts:
            summary_type_note = (
                f"Focused summary from {focused_summary_source_count} item(s) highly relevant to "
                f"{' and '.join(query_parts)}."
            )
        else: # Should not happen if focused_summary_source_count > 0, but as a fallback
            summary_type_note = (
                f"Focused summary from {focused_summary_source_count} item(s) "
                "based on high relevancy scores (specific queries not identified for this note)."
            )
    elif primary_llm_extract_query or secondary_llm_extract_query: # Queries were provided, but no items met focused criteria
        q_info_parts = []
        if primary_llm_extract_query: q_info_parts.append(f"Q1 ('{primary_llm_extract_query}')")
        if secondary_llm_extract_query: q_info_parts.append(f"Q2 ('{secondary_llm_extract_query}')")
        summary_type_note = (
            "General summary (fallback as no items met focused criteria for "
            f"{' or '.join(q_info_parts)})."
        )
    
    # Sanitize query texts for display if they are very long (optional, but good for Excel width)
    q1_display = (primary_llm_extract_query[:100] + '...') if primary_llm_extract_query and len(primary_llm_extract_query) > 100 else primary_llm_extract_query
    q2_display = (secondary_llm_extract_query[:100] + '...') if secondary_llm_extract_query and len(secondary_llm_extract_query) > 100 else secondary_llm_extract_query

    summary_data = {
        "Batch Timestamp": [batch_timestamp],
        "Keywords Searched": [last_keywords],
        "Primary Extraction Query (Q1)": [q1_display if q1_display else "N/A"],
        "Secondary Extraction Query (Q2)": [q2_display if q2_display else "N/A"],
        "Total Items Processed in Batch": [results_data_count],
        "Summary Type Note": [summary_type_note],
        "Consolidated Summary": [consolidated_summary_text], # This will be cleaned in to_excel_bytes
    }
    df = pd.DataFrame(summary_data)
    return df


def to_excel_bytes(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes:
    """
    Converts DataFrames for item details and an optional consolidated summary into Excel bytes.

    This function cleans string data in the DataFrames to remove illegal XML characters
    before writing them to separate sheets ('Item_Details' and 'Consolidated_Summary')
    in an Excel file. It returns the file content as a byte string, suitable for downloading.

    Args:
        df_item_details: A Pandas DataFrame containing detailed information
            for each processed item.
        df_consolidated_summary: An optional Pandas DataFrame containing the
            consolidated summary. If None or empty, this sheet will not be
            added to the Excel file.

    Returns:
        A byte string representing the content of the generated .xlsx Excel file.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Create copies of the DataFrames to avoid modifying the originals in place
        # and to ensure cleaning is applied to the data being written.
        df_item_details_cleaned = df_item_details.copy()
        
        # Apply cleaning to all object-type columns (which may contain strings)
        for col in df_item_details_cleaned.columns:
            if df_item_details_cleaned[col].dtype == 'object': # Check if column's data type is object
                 df_item_details_cleaned[col] = df_item_details_cleaned[col].apply(_clean_string_for_excel)
        
        df_item_details_cleaned.to_excel(writer, index=False, sheet_name='Item_Details')

        if df_consolidated_summary is not None and not df_consolidated_summary.empty:
            df_consolidated_summary_cleaned = df_consolidated_summary.copy()
            for col in df_consolidated_summary_cleaned.columns:
                 if df_consolidated_summary_cleaned[col].dtype == 'object':
                    df_consolidated_summary_cleaned[col] = df_consolidated_summary_cleaned[col].apply(_clean_string_for_excel)
            df_consolidated_summary_cleaned.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
        
    return output.getvalue()

# end of modules/excel_handler.py
