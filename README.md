      
# Streamlit Keyword Search & Analysis Tool

This Streamlit application empowers users to input keywords, perform Google searches (automatically enhanced by LLM-generated queries if an LLM is configured), extract metadata and main content from resulting URLs (supporting HTML and PDF text), and leverage a Large Language Model (LLM) – currently configured for **Google Gemini** – to summarize content, extract specific information with relevancy scoring, and generate a consolidated overview. Results are displayed interactively with visual cues for relevance, can be downloaded as an Excel file, and are also recorded to a Google Sheet.

The project is designed with a high degree of modularity, with `app.py` acting as an orchestrator for specialized modules. This facilitates future enhancements, including the potential addition of a separate API layer for programmatic access.

## Features

*   **Keyword Search:**
    *   Perform Google searches for multiple user-defined keywords (comma-separated, "Enter" to commit input).
    *   **LLM-Enhanced Queries:** If an LLM is configured, it automatically generates additional, related search queries based on the user's initial input and specific information goals, aiming to broaden the search scope effectively.
*   **Configurable Search Depth:** Specify the number of desired successfully scraped results per keyword. Oversampling is used to improve success rates.
*   **Content Extraction:**
    *   **HTML:** Fetches URL, page title, meta description, and OpenGraph tags. Uses `trafilatura` for main text extraction.
    *   **PDF:** Extracts document title (from metadata or filename) and full text content from PDF documents using `PyMuPDF`.
*   **LLM Integration (Google Gemini):**
    *   **Individual Summaries:** If an LLM is configured, it automatically generates summaries of each successfully scraped web page's/document's content.
    *   **Specific Information Extraction & Relevancy Scoring:** Extracts user-defined information from page/document content. The LLM also assigns a relevancy score (1/5 to 5/5) indicating how well the content matches the extraction query.
    *   **Consolidated Overview:** Automatically generates a synthesized overview.
        *   If a specific information query was provided by the user for extraction, the consolidated summary will focus on that query, using only individual item outputs that achieved a relevancy score of 3/5 or higher.
        *   If no specific information query was used, a general consolidated summary is created from all valid individual LLM outputs.
*   **Interactive UI & Results Display:**
    *   Built with Streamlit for easy input, configuration, and viewing of results and processing logs.
    *   **Visual Relevancy Cues:** Individually processed items in the results list are prefixed with visual markers:
        *   Relevancy score emojis (e.g., 5️⃣, 4️⃣, 3️⃣).
        *   A special ✨3️⃣ prefix if an item from an LLM-generated query also has a relevancy score of 3/5.
        *   A 🤖 marker for any item originating from an LLM-generated search query.
        *   A 📄 marker for PDF documents.
*   **Google Sheets Integration:**
    *   Stores detailed results, including a batch summary row and individual item rows, in a structured Google Sheet.
*   **Download Results:** Option to download all processed item details and the consolidated summary into an Excel (`.xlsx`) file.
*   **Safeguards & Performance:**
    *   **Retry Mechanisms:** Implemented for Google Custom Search API calls and LLM API calls to handle transient errors and rate limits using exponential backoff.
    *   **LLM Caching:** Caches results from LLM functions (`@st.cache_data`) to improve performance on repeated identical requests and reduce API calls.
*   **Modular Design & Configuration:**
    *   Code is separated into functional modules (`ui_manager.py`, `process_manager.py`, `excel_handler.py`, `search_engine.py`, `scraper.py`, `llm_processor.py`, `data_storage.py`, `config.py`). `app.py` serves as the main orchestrator, leading to clearer separation of concerns.
    *   API keys and settings are managed via Streamlit Secrets (`secrets.toml`).

## Project Structure


streamlit-search-tool/
├── app.py # Main Streamlit application orchestrator
├── modules/
│ ├── init.py
│ ├── config.py # Handles loading secrets & configurations
│ ├── search_engine.py # Google Search API interactions
│ ├── scraper.py # Web fetching & content/metadata extraction (HTML & PDF)
│ ├── llm_processor.py # LLM interactions (summaries, extractions, query gen, caching)
│ ├── data_storage.py # Google Sheets interactions
│ ├── ui_manager.py # Streamlit UI rendering and input handling
│ ├── process_manager.py # Core search, scrape, and analysis workflow orchestration
│ └── excel_handler.py # Excel file generation and formatting
├── .streamlit/
│ └── secrets.toml # Storing API keys & sensitive info
├── requirements.txt # Python dependencies
└── README.md # This file

      
## Prerequisites

Before you begin, ensure you have met the following requirements:

*   **Python 3.8+**
*   **Google Account**
*   **Google Cloud Platform (GCP) Project:**
    *   "Google Custom Search API" enabled.
    *   "Google Sheets API" enabled.
    *   "Google Drive API" enabled (required by `gspread`).
    *   "Generative Language API" (or "Vertex AI API") enabled for LLM features.
*   **Google Custom Search Engine (CSE):**
    *   Create one at [Programmable Search Engine](https://programmablesearchengine.google.com/). Note your **Search engine ID (CX)**.
*   **Google API Key (for Custom Search).**
*   **Google Gemini API Key (for LLM features).** (Or other LLM provider API key if support is extended).
*   **Google Service Account for Google Sheets:** (JSON key file, share Sheet with service account email).

## Setup & Installation

1.  **Clone the repository (if applicable).**
2.  **Create a virtual environment (recommended).**
3.  **Install dependencies:**
    Ensure your `requirements.txt` includes `google-generativeai` (or your chosen LLM's SDK), `pandas`, `openpyxl`, `streamlit`, `trafilatura`, `PyMuPDF`, `google-api-python-client`, `gspread`, `oauth2client`, `beautifulsoup4`, `requests`, and `tenacity`.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Secrets (`.streamlit/secrets.toml`):**
    (Example structure as provided previously, ensure all necessary keys are present).

    **Example `secrets.toml` structure:**
    ```toml
    # Google Custom Search
    GOOGLE_API_KEY = "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY"
    CSE_ID = "YOUR_CUSTOM_SEARCH_ENGINE_ID"

    # LLM Configuration (using Google Gemini by default)
    LLM_PROVIDER = "google"
    GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FROM_AI_STUDIO"
    GOOGLE_GEMINI_MODEL = "models/gemini-1.5-flash-latest" # Or your preferred compatible model

    # Google Sheets Integration
    SPREADSHEET_ID = "YOUR_GOOGLE_SHEET_ID_FROM_ITS_URL"
    WORKSHEET_NAME = "Sheet1" # Or your target sheet name

    # Google Cloud Platform Service Account details for Google Sheets & Drive API
    # Copy these values directly from your service account JSON key file
    [gcp_service_account]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "your-private-key-id"
    private_key = """-----BEGIN PRIVATE KEY-----\nYOUR_MULTI_LINE_PRIVATE_KEY_CONTENT_HERE\n-----END PRIVATE KEY-----\n"""
    client_email = "your-service-account-email@your-gcp-project-id.iam.gserviceaccount.com"
    client_id = "your-client-id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-gcp-project-id.iam.gserviceaccount.com"
    # Ensure the 'private_key' is correctly formatted as a multi-line string in TOML.
    # Ensure the 'client_x509_cert_url' correctly reflects your service account email.
    ```

## Running the Application

```bash
streamlit run app.py


Access at http://localhost:8501.
Usage

    Navigate to the application URL.

    Enter initial search keywords in the "Keywords (comma-separated)" field. Press Enter after each keyword or after pasting a comma-separated list.

    Configure "Number of successfully scraped results per keyword".

    Optionally, provide a query in "Specific info to extract with LLM...". This influences relevancy scoring, the context for LLM-generated search queries, and the focus of the consolidated summary. (If left blank, general summaries will be generated, and the consolidated overview will be broader).

    Click "🚀 Start Search & Analysis". (Button will be enabled if Google Sheets is correctly configured and connected, or if GSheets integration is not fully configured in secrets - in which case it will run without GSheet writing).

    View progress and results. Expanders for individual items will show:

        Emoji prefixes (e.g., 5️⃣, 4️⃣, 3️⃣) for relevancy scores if specific info was extracted.

        A special ✨3️⃣ prefix if an item from an LLM-generated query has a relevancy score of 3/5.

        A 🤖 marker for any item originating from an LLM-generated search query.

        A 📄 marker for PDF documents.

    Expand the "📜 View Processing Log" for detailed step-by-step information.

    If Google Sheets integration is correctly set up and enabled, data will be automatically written to the specified sheet.

    Download results via the "📥 Download Results as Excel" button.

Note: LLM-enhanced query generation and individual item summarization are automatic features that run if a valid LLM API key is provided in the secrets.toml file. They are not separate toggles in the UI.
Important Note for Contributing Agents

When modifying the codebase (app.py, modules/*.py):

    Always provide the FULL and COMPLETE code for the modified file(s). Do not provide only snippets or omit sections like docstrings or existing function implementations unless explicitly asked to do so for brevity in a specific, isolated discussion. The primary user will be copy-pasting the entire file content.

    Ensure all docstrings (module, function, class) are updated to reflect any changes in parameters, behavior, or new functionalities.

    Maintain versioning comments at the top of each modified file. Increment the version appropriately (e.g., v3.0.0 to v3.0.1) and briefly note the key change in the version comment.

    Every Python code file provided MUST end with the comment // end of [filename] as the absolute last line. For example, app.py should end with // end of app.py.

Adherence to these guidelines is crucial for maintaining clarity and ensuring the primary user can integrate changes smoothly.
Future API Layer Plan

The modular design supports future API development (e.g., using FastAPI or Flask) for programmatic access and integration with other tools. This could involve creating endpoints that accept keywords and parameters, returning structured JSON results.
Contributing

[Details on how to contribute to the project, if applicable. This could include coding standards, branch strategy, pull request process, etc.]
License

[Specify the license for your project, e.g., MIT License, Apache 2.0 License.]
