# app.py
# Version 1.7.0: Finalized docstrings, type hints, comments.
# Integrates with data_storage v1.5 for unified header and batch writing.
# Removed extensive DEBUG print statements.


"""
Streamlit Web Application for Keyword Search, Web Scraping, LLM Analysis, and Data Recording.

This application allows users to:
1. Input keywords for searching via Google Custom Search.
2. Scrape metadata and main content from the search result URLs.
3. Utilize a Large Language Model (LLM, e.g., Google Gemini) to:
    a. Generate individual summaries for each scraped page.
    b. Extract specific user-defined information from each page.
    c. Create a consolidated overview summary from all processed items in a batch.
4. Display the processed information and LLM insights in an interactive UI.
5. Store the detailed results (including a batch summary row and individual item rows)
   into a specified Google Sheet.
6. Download the results (item details and consolidated summary) as an Excel file.

The application is structured modularly, with separate Python files in the 'modules'
directory handling configuration, search, scraping, LLM processing, and data storage.
API keys and sensitive settings are managed via Streamlit Secrets (`.streamlit/secrets.toml`).
"""

import streamlit as st
from modules import config, search_engine, scraper, llm_processor, data_storage
import time
import pandas as pd # For DataFrame and Excel export
from io import BytesIO # For in-memory Excel file
from typing import List, Dict, Any, Optional # For type hinting

# --- Page Configuration ---
st.set_page_config(
    page_title="Keyword Search & Analysis Tool",
    page_icon="🔮",
    layout="wide"
)

# --- Load Application Configuration ---
# This is one of the first things done to ensure app has necessary settings.
cfg: Optional[config.AppConfig] = config.load_config()
if not cfg:
    # If essential configurations (like API keys) are missing,
    # load_config() returns None and displays an error. App should halt.
    st.error("CRITICAL: Application configuration failed to load. "
             "Please check secrets.toml and error messages above. Application cannot proceed.")
    st.stop()

# --- Session State Initialization ---
# Initialize session state variables if they don't already exist.
# This preserves state across Streamlit script reruns (e.g., after widget interactions).
default_session_state: Dict[str, Any] = {
    'processing_log': [],                   # Stores log messages during processing
    'results_data': [],                     # Stores dictionaries of processed item data
    'last_keywords': "",                    # Remembers the last keywords input
    'last_extract_query': "",               # Remembers the last LLM extraction query
    'consolidated_summary_text': None,      # Stores the batch's consolidated summary
    'gs_worksheet': None,                   # Caches the gspread.Worksheet object
    'sheet_writing_enabled': False,         # Flag indicating if GSheets writing is configured & working
    'sheet_connection_attempted_this_session': False # Ensures GSheets connection is tried only once per session load
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Google Sheets Setup (Attempt once per session on first load if configured) ---
# This block tries to connect to Google Sheets if configured and hasn't been attempted yet in this session.
if not st.session_state.sheet_connection_attempted_this_session:
    st.session_state.sheet_connection_attempted_this_session = True # Mark as attempted for this session
    if cfg.gsheets.service_account_info and \
       (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name):
        
        st.session_state.sheet_writing_enabled = True # Assume true, will be set to false on actual failure
        # Display a message in the sidebar while attempting connection
        # with st.sidebar: # This can cause issues if sidebar isn't fully rendered yet on first load
            # sheets_status_placeholder = st.info("Attempting to connect to Google Sheets...")
        
        st.session_state.gs_worksheet = data_storage.get_gspread_worksheet(
            cfg.gsheets.service_account_info,
            cfg.gsheets.spreadsheet_id,     # data_storage prioritizes ID
            cfg.gsheets.spreadsheet_name,   # Name as fallback
            cfg.gsheets.worksheet_name
        )
        if st.session_state.gs_worksheet:
            data_storage.ensure_master_header(st.session_state.gs_worksheet) # Ensure header is present
            # if 'sheets_status_placeholder' in locals(): sheets_status_placeholder.success("Google Sheets connection successful.")
        else:
            st.session_state.sheet_writing_enabled = False # Connection failed
            # if 'sheets_status_placeholder' in locals(): sheets_status_placeholder.error("Google Sheets connection failed.")
    else:
        # No service account info or sheet identifier in config, so disable writing.
        st.session_state.sheet_writing_enabled = False

# --- UI Layout Definition ---
st.title("Keyword Search & Analysis Tool 🔎📝")
st.markdown("Enter keywords, configure options, and let the tool gather insights for you.")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Search Parameters")
    keywords_input_val: str = st.text_area(
        "Keywords (one per line or comma-separated):",
        value=st.session_state.last_keywords,
        height=150,
        key="keywords_text_area",
        help="Enter each keyword or phrase on a new line, or separate them with commas."
    )
    num_results_wanted_per_keyword: int = st.slider(
        "Number of successfully scraped results per keyword:",
        min_value=1, max_value=10,
        value=cfg.num_results_per_keyword_default,
        key="num_results_slider",
        help="The tool will attempt to get this many usable web pages for each keyword."
    )
    
    st.subheader(f"LLM Processing (Optional) - Provider: {cfg.llm.provider.upper()}")
    # Check if the API key for the configured LLM provider is available
    llm_key_available: bool = (cfg.llm.provider == "google" and cfg.llm.google_gemini_api_key) or \
                              (cfg.llm.provider == "openai" and cfg.llm.openai_api_key)
    if llm_key_available:
        model_display_name: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
        st.caption(f"Using Model: {model_display_name}")
    else:
        st.caption(f"API Key for {cfg.llm.provider.upper()} not configured in secrets. LLM features disabled.")

    enable_llm_summary_val: bool = st.checkbox(
        "Generate LLM Summary?", value=True, key="llm_summary_checkbox", disabled=not llm_key_available
    )
    llm_extract_query_input_val: str = st.text_input(
        "Specific info to extract with LLM:",
        value=st.session_state.last_extract_query,
        placeholder="e.g., Key products, contact emails",
        key="llm_extract_text_input",
        disabled=not llm_key_available
    )
    
    # Display Google Sheets status
    if not st.session_state.sheet_writing_enabled:
        if cfg.gsheets.service_account_info or cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name:
             # Config was present, but connection failed
            st.sidebar.warning("⚠️ Google Sheets: Connection failed or sheet/worksheet not found. Results will not be saved to sheet.")
        else: # No config for sheets was found in secrets
             st.sidebar.caption("Google Sheets integration not configured (no secrets found).")
    else:
        st.sidebar.success(f"Google Sheets: Connected to '{st.session_state.gs_worksheet.spreadsheet.title if st.session_state.gs_worksheet else 'N/A'}' -> '{st.session_state.gs_worksheet.title if st.session_state.gs_worksheet else 'N/A'}'.")


    start_button_val: bool = st.button("🚀 Start Search & Analysis", type="primary", use_container_width=True)

# Define containers for organizing output on the main page
results_container = st.container()
log_container = st.container()

# --- Helper function for Excel Download ---
def to_excel(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes:
    """
    Converts one or two Pandas DataFrames into an Excel file in memory.

    Args:
        df_item_details: DataFrame containing the detailed item results.
        df_consolidated_summary: Optional DataFrame for the consolidated summary.

    Returns:
        Bytes representing the Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_item_details.to_excel(writer, index=False, sheet_name='Item_Details') # Sheet names Excel-friendly
        if df_consolidated_summary is not None and not df_consolidated_summary.empty:
            df_consolidated_summary.to_excel(writer, index=False, sheet_name='Consolidated_Summary')
    return output.getvalue()

# --- Main Processing Logic (Triggered by Start Button) ---
if start_button_val:
    # Initialize/clear session state for this run
    st.session_state.processing_log = ["Processing started..."]
    st.session_state.results_data = [] # This will hold list of Dict[str, Any]
    st.session_state.consolidated_summary_text = None 
    st.session_state.last_keywords = keywords_input_val # Save current inputs for next run
    st.session_state.last_extract_query = llm_extract_query_input_val
    
    keywords_list_val_runtime: List[str] = [
        k.strip() for k in keywords_input_val.replace(',', '\n').split('\n') if k.strip()
    ]

    if not keywords_list_val_runtime:
        st.sidebar.error("Please enter at least one keyword.")
        st.stop() # Halt execution if no keywords

    # --- Configuration for processing loop ---
    oversample_factor: float = 2.0
    max_google_fetch_per_keyword: int = 10 # Practical limit for Google Search API 'num' parameter
    
    # Calculate how many URLs to try fetching from Google per keyword based on oversampling
    est_urls_to_fetch_per_keyword: int = min(
        max_google_fetch_per_keyword,
        int(num_results_wanted_per_keyword * oversample_factor)
    )
    # Ensure we fetch at least what the user wants, up to the API limit
    if est_urls_to_fetch_per_keyword < num_results_wanted_per_keyword:
        est_urls_to_fetch_per_keyword = num_results_wanted_per_keyword
    
    # Calculate total steps for the progress bar
    total_llm_tasks_per_good_scrape: int = 0
    if llm_key_available: # Only count LLM tasks if an API key is available
        if enable_llm_summary_val: total_llm_tasks_per_good_scrape += 1
        if llm_extract_query_input_val.strip(): total_llm_tasks_per_good_scrape +=1
    
    total_major_steps_for_progress: int = \
        (len(keywords_list_val_runtime) * est_urls_to_fetch_per_keyword) + \
        (len(keywords_list_val_runtime) * num_results_wanted_per_keyword * total_llm_tasks_per_good_scrape)
    
    current_major_step_count: int = 0
    progress_bar_placeholder = st.empty() # Placeholder for dynamically updating the progress bar

    # --- Main Loop: Iterate through keywords, then search results ---
    for keyword_val in keywords_list_val_runtime:
        st.session_state.processing_log.append(f"\n🔎 Processing keyword: {keyword_val}")
        with progress_bar_placeholder.container(): # Use .container() to update progress bar in place
             st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
                        text=f"Starting keyword: {keyword_val}...")
        
        # Ensure Google Search API config is present
        if not (cfg.google_search.api_key and cfg.google_search.cse_id):
            st.error("Google Search API Key or CSE ID not configured. Cannot proceed.")
            st.session_state.processing_log.append(f"  ❌ ERROR: Google Search config missing. Halting for '{keyword_val}'.")
            st.stop()
        
        urls_to_fetch_from_google: int = est_urls_to_fetch_per_keyword
        st.session_state.processing_log.append(
            f"  Attempting to fetch up to {urls_to_fetch_from_google} Google results for '{keyword_val}' "
            f"to get {num_results_wanted_per_keyword} good scrapes."
        )
        search_results_items_val: List[Dict[str, Any]] = search_engine.perform_search(
            query=keyword_val, api_key=cfg.google_search.api_key, cse_id=cfg.google_search.cse_id,
            num_results=urls_to_fetch_from_google
        )
        st.session_state.processing_log.append(f"  Found {len(search_results_items_val)} Google result(s) for '{keyword_val}'.")
        
        successfully_scraped_for_this_keyword: int = 0
        if not search_results_items_val:
            st.session_state.processing_log.append(f"  No Google results for '{keyword_val}'. Moving to next keyword.")
            # Account for these "attempts" in progress calculation
            current_major_step_count += urls_to_fetch_from_google 
            continue # Skip to the next keyword

        for search_item_idx, search_item_val in enumerate(search_results_items_val):
            # Stop processing URLs for this keyword if desired number of good scrapes is met
            if successfully_scraped_for_this_keyword >= num_results_wanted_per_keyword:
                st.session_state.processing_log.append(
                    f"  Reached target of {num_results_wanted_per_keyword} successful scrapes for '{keyword_val}'. "
                    f"Skipping remaining {len(search_results_items_val) - search_item_idx} Google result(s)."
                )
                current_major_step_count += (len(search_results_items_val) - search_item_idx) # Add skipped attempts to progress
                break 
            
            current_major_step_count += 1 # Increment for each scrape attempt
            url_to_scrape_val: Optional[str] = search_item_val.get('link')

            if not url_to_scrape_val:
                st.session_state.processing_log.append(f"  - Item {search_item_idx+1} for '{keyword_val}' has no URL. Skipping.")
                continue
            
            progress_text_scrape: str = f"Scraping ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:50]}..."
            with progress_bar_placeholder.container():
                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0,
                            text=progress_text_scrape)
            
            st.session_state.processing_log.append(f"  ➔ Attempting to scrape ({search_item_idx+1}/{len(search_results_items_val)}): {url_to_scrape_val}")
            scraped_content_val: scraper.ScrapedData = scraper.fetch_and_extract_content(url_to_scrape_val)
            
            # Prepare data structure for this item's results
            item_data_val: Dict[str, Any] = {
                "keyword_searched": keyword_val, "url": url_to_scrape_val,
                "search_title": search_item_val.get('title'), "search_snippet": search_item_val.get('snippet'),
                "scraped_title": scraped_content_val.get('title'),
                "scraped_meta_description": scraped_content_val.get('meta_description'),
                "scraped_og_title": scraped_content_val.get('og_title'),
                "scraped_og_description": scraped_content_val.get('og_description'),
                "scraped_main_text": scraped_content_val.get('main_text'),
                "scraping_error": scraped_content_val.get('error'),
                "llm_summary": None, "llm_extracted_info": None, # Initialize LLM fields
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") # Timestamp for this specific item
            }
            
            if scraped_content_val.get('error'):
                st.session_state.processing_log.append(f"    ❌ Error scraping: {scraped_content_val['error']}")
                # Decide if you want to store items with scraping errors in results_data and Sheets
                # For now, only "good scrapes" proceed to LLM and are fully stored.
            else:
                # Check if the scrape yielded usable main text
                min_main_text_length: int = 200 # Characters; adjust as needed
                current_main_text: str = scraped_content_val.get('main_text', '')
                is_good_scrape: bool = (
                    current_main_text and
                    len(current_main_text.strip()) >= min_main_text_length and
                    "could not extract main content" not in current_main_text.lower() and
                    "not processed for main text" not in current_main_text.lower()
                )

                if is_good_scrape:
                    st.session_state.processing_log.append(f"    ✔️ Successfully scraped with sufficient main text (len={len(current_main_text)}).")
                    successfully_scraped_for_this_keyword += 1
                    main_text_for_llm: str = current_main_text
                    
                    # --- LLM Processing for good scrapes ---
                    if llm_key_available:
                        llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                        llm_model_to_use: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize
                        
                        if enable_llm_summary_val:
                            current_major_step_count +=1 # Increment for this LLM task
                            progress_text_llm: str = f"LLM Summary ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container():
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"       Generating LLM summary ({cfg.llm.provider})...")
                            summary: Optional[str] = llm_processor.generate_summary(
                                main_text_for_llm, api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_summary"] = summary
                            st.session_state.processing_log.append(f"        Summary: {str(summary)[:100] if summary else 'Failed/Empty'}...")
                            time.sleep(0.1) # Small delay between API calls

                        if llm_extract_query_input_val.strip():
                            current_major_step_count +=1 # Increment for this LLM task
                            progress_text_llm: str = f"LLM Extract ({current_major_step_count}/{total_major_steps_for_progress}): {url_to_scrape_val[:40]}..."
                            with progress_bar_placeholder.container():
                                st.progress(current_major_step_count / total_major_steps_for_progress if total_major_steps_for_progress > 0 else 0, text=progress_text_llm)
                            st.session_state.processing_log.append(f"      Extracting info ({cfg.llm.provider}): '{llm_extract_query_input_val}'...")
                            extracted_info: Optional[str] = llm_processor.extract_specific_information(
                                main_text_for_llm, extraction_query=llm_extract_query_input_val,
                                api_key=llm_api_key_to_use, model_name=llm_model_to_use, max_input_chars=cfg.llm.max_input_chars
                            )
                            item_data_val["llm_extracted_info"] = extracted_info
                            st.session_state.processing_log.append(f"        Extracted: {str(extracted_info)[:100] if extracted_info else 'Failed/Empty'}...")
                            time.sleep(0.1) # Small delay
                    
                    st.session_state.results_data.append(item_data_val) # Add item if scrape was good and LLM processed
                else:
                    st.session_state.processing_log.append(f"    ⚠️ Scraped, but main text was insufficient (len={len(current_main_text.strip())}) or not usable. LLM processing skipped for this item.")
                    # If you want to store items even if LLM is skipped due to bad scrape, add item_data_val here:
                    # item_data_val["llm_summary"] = "N/A - Insufficient text"
                    # st.session_state.results_data.append(item_data_val)

            time.sleep(0.2) # Polite delay after each scrape attempt
        
        # After processing all URLs for a keyword
        if successfully_scraped_for_this_keyword < num_results_wanted_per_keyword:
            st.session_state.processing_log.append(
                f"  ⚠️ For '{keyword_val}', only got {successfully_scraped_for_this_keyword}/{num_results_wanted_per_keyword} "
                f"desired successful scrapes (with sufficient text)."
            )
            # Account for LLM tasks that didn't run due to insufficient good scrapes for this keyword
            remaining_llm_tasks_for_keyword: int = (num_results_wanted_per_keyword - successfully_scraped_for_this_keyword) * total_llm_tasks_per_good_scrape
            current_major_step_count += remaining_llm_tasks_for_keyword
    
    # --- End of all keyword processing ---
    with progress_bar_placeholder.container():
        st.empty() # Clear the progress bar

    # --- Automatic Consolidated Summary Generation ---
    consolidated_summary_text_for_batch: Optional[str] = None
    topic_for_consolidation_for_batch: str = "Multiple Topics / Not Specified" # Default
    if st.session_state.results_data and llm_key_available and \
       (enable_llm_summary_val or llm_extract_query_input_val.strip()): # Only if LLM tasks were enabled
        st.session_state.processing_log.append(f"\n✨ Generating consolidated overview...")
        with st.spinner("Generating consolidated overview... (This can take a moment)"):
            # Determine topic context for the consolidated summary
            if not keywords_list_val_runtime: topic_for_consolidation_for_batch = "the searched topics"
            elif len(keywords_list_val_runtime) == 1: topic_for_consolidation_for_batch = keywords_list_val_runtime[0]
            else: topic_for_consolidation_for_batch = f"topics: {', '.join(keywords_list_val_runtime[:3])}{'...' if len(keywords_list_val_runtime) > 3 else ''}"
            
            # Collect all valid LLM outputs (summaries or extractions) from the processed items
            all_valid_llm_outputs: List[str] = [
                item.get("llm_summary") or item.get("llm_extracted_info") # Prioritize summary
                for item in st.session_state.results_data
                if (item.get("llm_summary") and not str(item.get("llm_summary", "")).lower().startswith("llm error") and not str(item.get("llm_summary", "")).lower().startswith("no text content")) or \
                   (item.get("llm_extracted_info") and not str(item.get("llm_extracted_info", "")).lower().startswith("llm error") and not str(item.get("llm_extracted_info", "")).lower().startswith("no text content"))
            ]
            
            if not all_valid_llm_outputs:
                st.warning("No valid individual LLM outputs (summaries or extractions) were available to create a consolidated overview.");
                consolidated_summary_text_for_batch = "Error: No valid LLM outputs were available for consolidation."
                st.session_state.processing_log.append("  ❌ No valid LLM outputs found for consolidation.")
            else:
                llm_api_key_to_use: Optional[str] = cfg.llm.google_gemini_api_key if cfg.llm.provider == "google" else cfg.llm.openai_api_key
                llm_model_to_use: str = cfg.llm.google_gemini_model if cfg.llm.provider == "google" else cfg.llm.openai_model_summarize # Or a specific model for consolidation
                
                consolidated_summary_text_for_batch = llm_processor.generate_consolidated_summary(
                    all_valid_llm_outputs,
                    topic_context=topic_for_consolidation_for_batch,
                    api_key=llm_api_key_to_use,
                    model_name=llm_model_to_use, # Consider a more powerful model for this if available
                    max_input_chars=cfg.llm.max_input_chars # Use a generous input limit
                )
                st.session_state.processing_log.append(f"  Consolidated Overview Raw (first 150 chars): {str(consolidated_summary_text_for_batch)[:150] if consolidated_summary_text_for_batch else 'Failed/Empty'}...")
        st.session_state.consolidated_summary_text = consolidated_summary_text_for_batch # Store in session state for display

    # --- Write to Google Sheets (after all individual processing & consolidated summary) ---
    if st.session_state.sheet_writing_enabled and st.session_state.gs_worksheet:
        # Only write if there's something to write (either item data or a consolidated summary)
        if st.session_state.results_data or st.session_state.consolidated_summary_text:
            batch_process_timestamp_for_sheet: str = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing_log.append(f"\n💾 Writing batch data (started {batch_process_timestamp_for_sheet}) to Google Sheets...")
            
            extraction_query_for_sheet: Optional[str] = st.session_state.last_extract_query if llm_extract_query_input_val.strip() else None
            
            write_successful: bool = data_storage.write_batch_summary_and_items_to_sheet(
                worksheet=st.session_state.gs_worksheet,
                batch_timestamp=batch_process_timestamp_for_sheet,
                consolidated_summary=st.session_state.consolidated_summary_text, # Pass the generated text
                topic_context=topic_for_consolidation_for_batch, # Use the topic determined for this batch
                item_data_list=st.session_state.results_data, # Pass all collected item data
                extraction_query_text=extraction_query_for_sheet
            )
            if write_successful:
                st.session_state.processing_log.append(f"  Batch data (summary row + {len(st.session_state.results_data)} item rows) written to Google Sheets.")
            else:
                st.session_state.processing_log.append(f"  ❌ Failed to write batch data to Google Sheets.")
    elif st.session_state.results_data: # If there was data but sheets wasn't enabled/working
        st.session_state.processing_log.append("\n⚠️ Google Sheets writing is disabled or connection failed. Data not saved to sheet.")

    # Final status message
    if st.session_state.results_data or st.session_state.consolidated_summary_text:
        st.success("All processing complete!")
    else:
        st.warning("Processing complete, but no data was generated (no items scraped or LLM outputs produced).")

# --- Display Sections (Download button, Consolidated Summary, Individual Results, Log) ---
with results_container:
    # --- Download Button ---
    if st.session_state.results_data: # Show download only if there are item results
        st.markdown("---") # Separator
        
        item_details_for_excel: List[Dict[str,Any]] = []
        # Define headers for Excel item details sheet, can be a subset or reordered from MASTER_HEADER
        excel_item_headers: List[str] = [
            "Batch Timestamp", "Item Timestamp", "Keyword Searched", "URL", 
            "Search Result Title", "Search Result Snippet", "Scraped Page Title", 
            "Scraped Meta Description", "Scraped OG Title", "Scraped OG Description", 
            "LLM Summary (Individual)", "LLM Extracted Info (Query)", 
            "LLM Extraction Query", "Scraping Error", "Main Text (Truncated)"
        ]
        for item_val_excel in st.session_state.results_data:
            row_data_excel: Dict[str, Any] = {
                "Batch Timestamp": item_val_excel.get("timestamp"), # Using item timestamp as a proxy for batch here
                "Item Timestamp": item_val_excel.get("timestamp"),
                "Keyword Searched": item_val_excel.get("keyword_searched"),
                "URL": item_val_excel.get("url"),
                "Search Result Title": item_val_excel.get("search_title"),
                "Search Result Snippet": item_val_excel.get("search_snippet"),
                "Scraped Page Title": item_val_excel.get("scraped_title"),
                "Scraped Meta Description": item_val_excel.get("scraped_meta_description"),
                "Scraped OG Title": item_val_excel.get("scraped_og_title"),
                "Scraped OG Description": item_val_excel.get("scraped_og_description"),
                "LLM Summary (Individual)": item_val_excel.get("llm_summary"),
                "LLM Extracted Info (Query)": item_val_excel.get("llm_extracted_info"),
                "LLM Extraction Query": st.session_state.last_extract_query if item_val_excel.get("llm_extracted_info") else "",
                "Scraping Error": item_val_excel.get("scraping_error"),
                "Main Text (Truncated)": (str(item_val_excel.get("scraped_main_text", ""))[:10000] + "...") if item_val_excel.get("scraped_main_text") and len(str(item_val_excel.get("scraped_main_text", ""))) > 10000 else str(item_val_excel.get("scraped_main_text", ""))
            }
            # Ensure only defined headers are included, in order
            ordered_row_excel = {header: row_data_excel.get(header, "") for header in excel_item_headers}
            item_details_for_excel.append(ordered_row_excel)

        df_item_details = pd.DataFrame(item_details_for_excel, columns=excel_item_headers) # Ensure column order
        
        df_consolidated_summary_excel: Optional[pd.DataFrame] = None
        if st.session_state.get('consolidated_summary_text') and \
           not str(st.session_state.consolidated_summary_text).lower().startswith("error:"):
            
            last_run_keywords_excel: List[str] = [k.strip() for k in st.session_state.last_keywords.replace(',', '\n').split('\n') if k.strip()]
            topic_display_excel: str
            if not last_run_keywords_excel: topic_display_excel = "General Batch"
            elif len(last_run_keywords_excel) == 1: topic_display_excel = last_run_keywords_excel[0]
            else: topic_display_excel = f"Topics: {', '.join(last_run_keywords_excel[:3])}{'...' if len(last_run_keywords_excel) > 3 else ''}"

            consolidated_data_excel: Dict[str, List[Any]] = {
                "Batch Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
                "Topic/Keywords": [topic_display_excel],
                "Consolidated Summary": [st.session_state.consolidated_summary_text],
                "Source Items Count": [len(st.session_state.results_data)] # Number of items that fed into this summary
            }
            df_consolidated_summary_excel = pd.DataFrame(consolidated_data_excel)

        excel_file_bytes: bytes = to_excel(df_item_details, df_consolidated_summary_excel)
        
        current_time_str_excel: str = time.strftime("%Y%m%d-%H%M%S")
        download_filename_excel: str = f"keyword_analysis_results_{current_time_str_excel}.xlsx"

        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_file_bytes,
            file_name=download_filename_excel,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_excel_button"
        )

    # Display Consolidated Summary on page
    if st.session_state.get('consolidated_summary_text'):
        st.markdown("---")
        st.subheader("✨ Consolidated Overview Result")
        with st.container(border=True):
            st.markdown(st.session_state.consolidated_summary_text)

    # Display Individual Stored Results on page
    if st.session_state.results_data:
        st.subheader(f"📊 Individually Processed Content ({len(st.session_state.results_data)} item(s))")
        for i, item_val_display in enumerate(st.session_state.results_data):
            display_title_ui: str = item_val_display.get('scraped_title') or \
                                 item_val_display.get('scraped_og_title') or \
                                 item_val_display.get('search_title') or \
                                 "Untitled"
            expander_title_ui: str = f"{item_val_display['keyword_searched']} | {display_title_ui} ({item_val_display.get('url')})"
            
            with st.expander(expander_title_ui):
                st.markdown(f"**URL:** [{item_val_display.get('url')}]({item_val_display.get('url')})")
                if item_val_display.get('scraping_error'):
                    st.error(f"Scraping Error: {item_val_display['scraping_error']}")
                
                with st.container(border=True): # Scraped Metadata container
                    st.markdown("**Scraped Metadata:**")
                    st.markdown(f"  - **Title:** {item_val_display.get('scraped_title', 'N/A')}")
                    st.markdown(f"  - **Meta Desc:** {item_val_display.get('scraped_meta_description', 'N/A')}")
                    st.markdown(f"  - **OG Title:** {item_val_display.get('scraped_og_title', 'N/A')}")
                    st.markdown(f"  - **OG Desc:** {item_val_display.get('scraped_og_description', 'N/A')}")
                
                if item_val_display.get('scraped_main_text'):
                    with st.popover("View Main Text", use_container_width=True):
                        st.text_area(f"Main Text Content", value=item_val_display['scraped_main_text'], height=400, key=f"main_text_popover_{i}", disabled=True)
                else:
                    st.caption("No main text was extracted or deemed usable for LLM processing for this item.")
                
                # LLM Insights display
                if item_val_display.get("llm_summary") or item_val_display.get("llm_extracted_info"):
                    st.markdown("**LLM Insights:**")
                    if item_val_display.get("llm_summary"):
                        with st.container(border=True):
                            st.markdown(f"**Summary (LLM):**")
                            st.markdown(item_val_display["llm_summary"])
                    if item_val_display.get("llm_extracted_info"):
                        with st.container(border=True):
                            st.markdown(f"**Extracted Info (LLM) for '{st.session_state.last_extract_query}':**")
                            st.markdown(item_val_display["llm_extracted_info"])
                st.caption(f"Item Timestamp: {item_val_display.get('timestamp')}")

# --- Display Processing Log ---
with log_container: # log_container defined globally
    if st.session_state.processing_log: # Only show if there's a log
        with st.expander("📜 View Processing Log", expanded=False):
            log_text: str = "\n".join(st.session_state.processing_log)
            st.code(log_text, language=None)

# --- Footer for Version ---
st.markdown("---")
st.caption("Keyword Search & Analysis Tool v1.7.0")

# end of app.py
