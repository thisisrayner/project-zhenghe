# Codebase Summary (v1.3)

## File: app.py
Module Docstring:
```text
Streamlit Web Application for D.O.R.A - The Research Agent.
```

---

## File: modules/excel_handler.py
Module Docstring:
```text
Handles the creation and formatting of Excel files for exporting application results.

This module provides functions to convert processed item data and consolidated
summaries into structured Pandas DataFrames and then into a downloadable
Excel file format (.xlsx) with separate sheets for detailed item results and
the overall summary.
```

### def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_queries: List[str]) -> pd.DataFrame
Docstring:
```text
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
```

### def prepare_consolidated_summary_df(consolidated_summary_text: Optional[str], results_data_count: int, last_keywords: str, primary_last_extract_query: Optional[str], batch_timestamp: str) -> Optional[pd.DataFrame]
Docstring:
```text
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
```

### def to_excel_bytes(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes
Docstring:
```text
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
```

---

## File: modules/data_storage.py
Module Docstring:
```text
Handles data storage operations, primarily focused on Google Sheets integration.
```

### def get_gspread_worksheet(service_account_info: Optional[Dict[str, Any]], spreadsheet_id: Optional[str], spreadsheet_name: Optional[str], worksheet_name: str = 'Sheet1') -> Optional[gspread.Worksheet]
Docstring:
[No docstring provided]

### def ensure_master_header(worksheet: gspread.Worksheet) -> None
Docstring:
[No docstring provided]

### def write_batch_summary_and_items_to_sheet(worksheet: gspread.Worksheet, batch_timestamp: str, consolidated_summary: Optional[str], topic_context: str, item_data_list: List[Dict[str, Any]], extraction_queries_list: List[str], main_text_truncate_limit: int = 10000) -> bool
Docstring:
[No docstring provided]

---

## File: modules/search_engine.py
Module Docstring:
```text
Handles interactions with the Google Custom Search API.

This module provides functionality to perform searches using specified
keywords, API key, and Custom Search Engine (CSE) ID.
It uses the google-api-python-client library and includes
a retry mechanism for API calls to handle transient errors and rate limits.
```

### def perform_search(query: str, api_key: str, cse_id: str, num_results: int = 5, max_retries: int = DEFAULT_MAX_RETRIES, initial_backoff: float = DEFAULT_INITIAL_BACKOFF, max_backoff: float = DEFAULT_MAX_BACKOFF, **kwargs: Any) -> List[Dict[str, Any]]
Docstring:
```text
Performs a Google Custom Search for the given query with retry logic.

Args:
    query: The search term(s).
    api_key: The Google API key authorized for Custom Search API.
    cse_id: The ID of the Custom Search Engine to use.
    num_results: The number of search results to return (max 10 per API call).
    max_retries: Maximum number of retries for API calls.
    initial_backoff: Initial delay in seconds for the first retry.
    max_backoff: Maximum delay in seconds for a single retry.
    **kwargs: Additional parameters to pass to the CSE list method,
              e.g., siteSearch, exactTerms, etc. Refer to Google CSE API docs.

Returns:
    A list of search result item dictionaries as returned by the API.
    Each item typically contains 'title', 'link', 'snippet', etc.
    Returns an empty list if an error occurs after all retries or no results are found.
```

---

## File: modules/config.py
Module Docstring:
```text
Configuration management for the Streamlit Keyword Search & Analysis Tool.

This module defines dataclasses for structuring configuration parameters and
provides a function to load these configurations primarily from Streamlit
secrets (`secrets.toml`). It handles settings for Google Search, LLM providers
(Google Gemini, OpenAI), and Google Sheets integration. It also defines
the application's version.
```

### class GoogleSearchConfig
Docstring:
```text
Configuration specific to Google Custom Search API.
```

### class LLMConfig
Docstring:
```text
Configuration for Large Language Model (LLM) interactions.
```

### class GoogleSheetsConfig
Docstring:
```text
Configuration for Google Sheets integration.
```

### class AppConfig
Docstring:
```text
Main application configuration, aggregating other configs.
```

### def load_config() -> Optional[AppConfig]
Docstring:
```text
Loads application configurations from Streamlit secrets.

Reads API keys, model names, sheet identifiers, and other settings
from `.streamlit/secrets.toml`. Provides sensible defaults if some
optional settings are not found.

Returns:
    Optional[AppConfig]: An AppConfig object populated with settings,
                         or None if essential configurations (like
                         Google Search API keys) are missing.
```

---

## File: modules/llm_processor.py
Module Docstring:
```text
Handles interactions with Large Language Models (LLMs), specifically Google Gemini.

This module is responsible for configuring the LLM client, making API calls,
and providing functionalities such as:
- Generating summaries of text content.
- Extracting specific information based on user queries and providing relevancy scores.
- Generating consolidated overviews from multiple text snippets. The overview consists
  of a narrative part (plain text) followed by a "TLDR:" section with dash-bulleted key points.
  This can be a general overview or focused on a specific query (Q1) with potential
  enrichment from a secondary query (Q2).
- Generating alternative search queries based on initial keywords and user goals (Q1 and Q2).

It incorporates caching for LLM responses to optimize performance and reduce API costs,
and includes retry mechanisms for API calls.
```

### def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool
Docstring:
```text
Configures the Google Gemini API client and lists available models.
Args:
    api_key: Optional[str]: The Google Gemini API key.
    force_recheck_models: bool: If True, forces a re-fetch of available models.
Returns:
    bool: True if configuration was successful, False otherwise.
```

### def _call_gemini_api(model_name: str, prompt_parts: List[str], generation_config_args: Optional[Dict[str, Any]] = None, safety_settings_args: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3, initial_backoff_seconds: float = 5.0, max_backoff_seconds: float = 60.0) -> Optional[str]
Docstring:
```text
Calls the Google Gemini API with specified parameters and handles retries.
Args:
    model_name: Name of the Gemini model.
    prompt_parts: List of strings for the prompt.
    generation_config_args: Optional generation config.
    safety_settings_args: Optional safety settings.
    max_retries: Max retries for rate limits.
    initial_backoff_seconds: Initial retry delay.
    max_backoff_seconds: Max retry delay.
Returns:
    LLM text response or an error message string.
```

### def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str
Docstring:
```text
Truncates text to a maximum character limit.
```

### def generate_summary(text_content: Optional[str], api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring:
```text
Generates a plain text summary for given text content.
```

### def extract_specific_information(text_content: Optional[str], extraction_query: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring:
```text
Extracts specific info and scores relevancy. Extracted info part is plain text.
```

### def _parse_score_and_get_content(text_with_potential_score: str) -> Tuple[Optional[int], str]
Docstring:
```text
Parses score and extracts content.
```

### def generate_consolidated_summary(summaries: Tuple[Optional[str], ...], topic_context: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 150000, extraction_query_for_consolidation: Optional[str] = None, secondary_query_for_enrichment: Optional[str] = None) -> Optional[str]
Docstring:
```text
Generates a consolidated summary with a narrative part and a TLDR section.
The narrative is plain text. The TLDR section uses dash-prefixed key points.
```

### def generate_search_queries(original_keywords: Tuple[str, ...], specific_info_query: Optional[str], specific_info_query_2: Optional[str], num_queries_to_generate: int, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 2500) -> Optional[List[str]]
Docstring:
```text
Generates new search queries.
```

---

## File: modules/ui_manager.py
Module Docstring:
```text
Manages the Streamlit User Interface elements, layout, and user inputs.
```

### def sanitize_text_for_markdown(text: Optional[str]) -> str
Docstring:
```text
Sanitizes text to prevent common markdown rendering issues, especially from LLM output.
Escapes markdown special characters.
```

### def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]
Docstring:
[No docstring provided]

### def get_display_prefix_for_item(item_data: Dict[str, Any]) -> str
Docstring:
[No docstring provided]

### def render_sidebar(cfg: 'config.AppConfig', current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]
Docstring:
[No docstring provided]

### def apply_custom_css()
Docstring:
[No docstring provided]

### def display_consolidated_summary_and_sources(summary_text: Optional[str], focused_sources: Optional[List[Dict[str, Any]]], last_extract_queries: List[str]) -> None
Docstring:
```text
Displays the consolidated summary and, if applicable, the sources used for a focused summary.
```

### def display_individual_results()
Docstring:
[No docstring provided]

### def display_processing_log()
Docstring:
[No docstring provided]

---

## File: modules/_init_.py
Module Docstring:
[No docstring provided]

---

## File: modules/scraper.py
Module Docstring:
```text
Web scraping module for fetching and extracting content from URLs.

This module uses 'requests' to fetch web page content, 'BeautifulSoup'
for parsing HTML and extracting metadata (like title, description, OpenGraph tags),
'trafilatura' for extracting the main textual content of an HTML article,
and 'PyMuPDF' (fitz) for extracting text from PDF documents.
```

### class ScrapedData(TypedDict)
Docstring:
```text
A dictionary structure for storing data scraped from a web page or PDF.
`total=False` means keys are optional and might not always be present.
```

### def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[Optional[str], Optional[str]]
Docstring:
```text
Extracts text and title from PDF bytes using PyMuPDF (fitz).

Args:
    pdf_bytes: The byte content of the PDF file.

Returns:
    A tuple (full_text, document_title).
    - full_text: Concatenated text from all pages.
    - document_title: Title from PDF metadata, if available.
    Returns (None, None) if a significant error occurs during PDF processing.
```

### def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData
Docstring:
```text
Fetches content from the given URL and extracts metadata and main text.
Supports HTML and PDF documents.

Args:
    url: The URL of the web page/document to scrape.
    timeout: The timeout in seconds for the HTTP GET request.

Returns:
    A ScrapedData dictionary containing the extracted information.
    If an error occurs, the 'error' key in the dictionary will be populated.
```

---

## File: modules/process_manager.py
Module Docstring:
```text
Handles the main workflow of searching, scraping, LLM processing, and data aggregation.
```

### class FocusedSummarySource(TypedDict)
Docstring:
[No docstring provided]

### def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]
Docstring:
[No docstring provided]

### def run_search_and_analysis(app_config: 'config.AppConfig', keywords_input: str, llm_extract_queries_input: List[str], num_results_wanted_per_keyword: int, gs_worksheet: Optional[Any], sheet_writing_enabled: bool, gsheets_secrets_present: bool) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str], List[FocusedSummarySource]]
Docstring:
[No docstring provided]

---

