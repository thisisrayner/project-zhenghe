# Codebase Summary (v1.2)

## File: app.py
Module Docstring: Streamlit Web Application for Keyword Search & Analysis Tool (KSAT).

---

## File: modules/excel_handler.py
Module Docstring: Handles the creation and formatting of Excel files for exporting KSAT results.

### def prepare_item_details_df(results_data: List[Dict[str, Any]], last_extract_queries: List[str]) -> pd.DataFrame
Docstring: Prepares a Pandas DataFrame for the item details sheet in the Excel export.

### def prepare_consolidated_summary_df(consolidated_summary_text: Optional[str], results_data_count: int, last_keywords: str, primary_last_extract_query: Optional[str], batch_timestamp: str) -> Optional[pd.DataFrame]
Docstring: [No docstring provided]

### def to_excel_bytes(df_item_details: pd.DataFrame, df_consolidated_summary: Optional[pd.DataFrame] = None) -> bytes
Docstring: [No docstring provided]

---

## File: modules/data_storage.py
Module Docstring: Handles data storage operations, primarily focused on Google Sheets integration.

### def get_gspread_worksheet(service_account_info: Optional[Dict[str, Any]], spreadsheet_id: Optional[str], spreadsheet_name: Optional[str], worksheet_name: str = 'Sheet1') -> Optional[gspread.Worksheet]
Docstring: [No docstring provided]

### def ensure_master_header(worksheet: gspread.Worksheet) -> None
Docstring: [No docstring provided]

### def write_batch_summary_and_items_to_sheet(worksheet: gspread.Worksheet, batch_timestamp: str, consolidated_summary: Optional[str], topic_context: str, item_data_list: List[Dict[str, Any]], extraction_queries_list: List[str], main_text_truncate_limit: int = 10000) -> bool
Docstring: [No docstring provided]

---

## File: modules/search_engine.py
Module Docstring: Handles interactions with the Google Custom Search API.

### def perform_search(query: str, api_key: str, cse_id: str, num_results: int = 5, max_retries: int = DEFAULT_MAX_RETRIES, initial_backoff: float = DEFAULT_INITIAL_BACKOFF, max_backoff: float = DEFAULT_MAX_BACKOFF, **kwargs: Any) -> List[Dict[str, Any]]
Docstring: Performs a Google Custom Search for the given query with retry logic.

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

## File: modules/llm_processor.py
Module Docstring: Handles interactions with Large Language Models (LLMs) for text processing.

### def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool
Docstring: [No docstring provided]

### def _call_gemini_api(model_name: str, prompt_parts: List[str], generation_config_args: Optional[Dict[str, Any]] = None, safety_settings_args: Optional[List[Dict[str, Any]]] = None, max_retries: int = 3, initial_backoff_seconds: float = 5.0, max_backoff_seconds: float = 60.0) -> Optional[str]
Docstring: [No docstring provided]

### def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str
Docstring: [No docstring provided]

### def generate_summary(text_content: Optional[str], api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring: [No docstring provided]

### def extract_specific_information(text_content: Optional[str], extraction_query: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 100000) -> Optional[str]
Docstring: [No docstring provided]

### def _parse_score_and_get_content(text_with_potential_score: str) -> tuple[Optional[int], str]
Docstring: [No docstring provided]

### def generate_consolidated_summary(summaries: tuple[Optional[str], ...], topic_context: str, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 150000, extraction_query_for_consolidation: Optional[str] = None) -> Optional[str]
Docstring: [No docstring provided]

### def generate_search_queries(original_keywords: tuple[str, ...], specific_info_query: Optional[str], num_queries_to_generate: int, api_key: Optional[str], model_name: str = 'models/gemini-1.5-flash-latest', max_input_chars: int = 2000) -> Optional[List[str]]
Docstring: [No docstring provided]

---

## File: modules/ui_manager.py
Module Docstring: Manages the Streamlit User Interface elements, layout, and user inputs.

### def sanitize_text_for_markdown(text: Optional[str]) -> str
Docstring: Sanitizes text to prevent common markdown rendering issues, especially from LLM output.

### def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]
Docstring: [No docstring provided]

### def get_display_prefix_for_item(item_data: Dict[str, Any]) -> str
Docstring: [No docstring provided]

### def render_sidebar(cfg: config.AppConfig, current_gsheets_error: Optional[str], sheet_writing_enabled: bool) -> Tuple[str, int, List[str], bool]
Docstring: [No docstring provided]

### def apply_custom_css()
Docstring: [No docstring provided]

### def display_consolidated_summary()
Docstring: [No docstring provided]

### def display_individual_results()
Docstring: [No docstring provided]

### def display_processing_log()
Docstring: [No docstring provided]

---

## File: modules/_init_.py
Module Docstring: [No docstring provided]

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

## File: modules/process_manager.py
Module Docstring: Handles the main workflow of searching, scraping, LLM processing, and data aggregation.

### def _parse_score_from_extraction(extracted_info: Optional[str]) -> Optional[int]
Docstring: [No docstring provided]

### def run_search_and_analysis(app_config: config.AppConfig, keywords_input: str, llm_extract_queries_input: List[str], num_results_wanted_per_keyword: int, gs_worksheet: Optional[Any], sheet_writing_enabled: bool, gsheets_secrets_present: bool) -> Tuple[List[str], List[Dict[str, Any]], Optional[str], Set[str], Set[str]]
Docstring: [No docstring provided]

---

