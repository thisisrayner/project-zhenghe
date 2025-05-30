# Codebase Summary (v1.1)

## File: app.py
Module Docstring: Streamlit Web Application for Keyword Search, Web Scraping, LLM Analysis, and Data Recording.

### def to_excel(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes
Docstring: Converts DataFrames to Excel. (Implementation as provided previously) 

---

## File: modules/config.py
Module Docstring: Configuration management for the Streamlit Keyword Search & Analysis Tool.

### class GoogleSearchConfig
Docstring: Configuration specific to Google Custom Search API.

### class LLMConfig
Docstring: Configuration for Large Language Model (LLM) interactions.

### class GoogleSheetsConfig
Docstring: Configuration for Google Sheets integration.

### class AppConfig
Docstring: Main application configuration, aggregating other configs.

### def load_config() -> Optional[AppConfig]
Docstring: Loads application configurations from Streamlit secrets.

---

## File: modules/scraper.py
Module Docstring: Web scraping module for fetching and extracting content from URLs.

### class ScrapedData(TypedDict)
Docstring: A dictionary structure for storing data scraped from a web page.

### def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData
Docstring: Fetches content from the given URL and extracts metadata and main text.

---

Error parsing /home/runner/work/project-zhenghe/project-zhenghe/modules/llm_processor.py: invalid syntax (llm_processor.py, line 523)
## File: modules/_init_.py
Module Docstring: None

---

## File: modules/search_engine.py
Module Docstring: Handles interactions with the Google Custom Search API.

### def perform_search(query: str, api_key: str, cse_id: str, num_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]
Docstring: Performs a Google Custom Search for the given query.

---

## File: modules/data_storage.py
Module Docstring: Handles data storage operations, primarily focused on Google Sheets integration.

### def get_gspread_worksheet(service_account_info: Optional[Dict[str, Any]], spreadsheet_id: Optional[str], spreadsheet_name: Optional[str], worksheet_name: str = 'Sheet1') -> Optional[gspread.Worksheet]
Docstring: Authorizes gspread client with service account info and returns the specified worksheet.

### def ensure_master_header(worksheet: gspread.Worksheet) -> None
Docstring: Ensures the MASTER_HEADER is present in Row 1 of the worksheet.

### def write_batch_summary_and_items_to_sheet(worksheet: gspread.Worksheet, batch_timestamp: str, consolidated_summary: Optional[str], topic_context: str, item_data_list: List[Dict[str, Any]], extraction_query_text: Optional[str] = None, main_text_truncate_limit: int = 10000) -> bool
Docstring: Writes a batch summary row followed by individual item detail rows to the Google Sheet.

---

