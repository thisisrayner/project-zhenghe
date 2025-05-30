# Codebase Summary (v1.1)

## File: app.py
Module Docstring: Streamlit Web Application for Keyword Search, Web Scraping, LLM Analysis, and Data Recording.

### def get_display_prefix_for_item(item_data: Dict[str, Any], llm_generated_keywords: Set[str]) -> str
Docstring: Determines an emoji prefix for an item based on its LLM relevancy score

### def to_excel(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes
Docstring: None

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
Docstring: A dictionary structure for storing data scraped from a web page or PDF.

### def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[Optional[str], Optional[str]]
Docstring: Extracts text and title from PDF bytes using PyMuPDF (fitz).

### def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData
Docstring: Fetches content from the given URL and extracts metadata and main text.

---

## File: modules/llm_processor.py
Module Docstring: Handles interactions with Large Language Models (LLMs) for text processing.

### def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool
Docstring: Configures the Google Generative AI client with the provided API key.

### def _call_gemini_api(model_name: str, prompt_parts: List[str], generation_config_args: Optional[Dict[str, Any]] = None, safety_settings_args: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3, initial_backoff_seconds: float = 5.0, max_backoff_seconds: float = 60.0) -> Optional[str]
Docstring: Internal helper function to make a call to the Google Gemini API (generate_content).

### def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str
Docstring: Truncates text to a specified maximum number of characters.

### def generate_summary(text_content: Optional[str], api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring: Generates a narrative summary for the given text content using Gemini.

### def extract_specific_information(text_content: Optional[str], extraction_query: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring:     Extracts specific information based on a user's query from the text content using Gemini

### def _parse_score_and_get_content(text_with_potential_score: str) -> tuple[Optional[int], str]
Docstring:     Parses a relevancy score from the beginning of a string (if present)

### def generate_consolidated_summary(summaries: tuple[Optional[str], ...], topic_context: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 150000, extraction_query_for_consolidation: Optional[str] = None) -> Optional[str]
Docstring:     Generates a consolidated overview from a list of individual LLM outputs.

### def generate_search_queries(original_keywords: tuple[str, ...], specific_info_query: Optional[str], num_queries_to_generate: int, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 2000) -> Optional[List[str]]
Docstring: Generates a list of new search queries based on original keywords and a specific info query.

---

## File: modules/_init_.py
Module Docstring: None

---

## File: modules/search_engine.py
Module Docstring: Handles interactions with the Google Custom Search API.

### def perform_search(query: str, api_key: str, cse_id: str, num_results: int = 5, max_retries: int = DEFAULT_MAX_RETRIES, initial_backoff: float = DEFAULT_INITIAL_BACKOFF, max_backoff: float = DEFAULT_MAX_BACKOFF, **kwargs: Any) -> List[Dict[str, Any]]
Docstring: Performs a Google Custom Search for the given query with retry logic.

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

