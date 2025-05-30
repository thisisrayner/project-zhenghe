      
# Streamlit Keyword Search & Analysis Tool 

This Streamlit application empowers users to input keywords, perform Google searches (enhanced by optional LLM-generated queries), extract metadata and main content from resulting URLs (supporting HTML and PDF text), and leverage a Large Language Model (LLM) – currently configured for **Google Gemini** – to summarize content, extract specific information with relevancy scoring, and generate a consolidated overview. Results are displayed interactively with visual cues for relevance, can be downloaded as an Excel file, and are also recorded to a Google Sheet.

The project is designed with modularity to facilitate future enhancements, including the potential addition of a separate API layer for programmatic access.

## Features

*   **Keyword Search:**
    *   Perform Google searches for multiple user-defined keywords (comma-separated, "Enter" to commit input).
    *   **LLM-Enhanced Queries:** Optionally uses an LLM to generate additional, related search queries based on the user's initial input and specific information goals, aiming to broaden the search scope effectively.
*   **Configurable Search Depth:** Specify the number of desired successfully scraped results per keyword. Oversampling is used to improve success rates.
*   **Content Extraction:**
    *   **HTML:** Fetches URL, page title, meta description, and OpenGraph tags. Uses `trafilatura` for main text extraction.
    *   **PDF:** Extracts document title (from metadata or filename) and full text content from PDF documents using `PyMuPDF`.
*   **LLM Integration (Google Gemini):**
    *   **Individual Summaries:** Generates summaries of each successfully scraped web page's/document's content.
    *   **Specific Information Extraction & Relevancy Scoring:** Extracts user-defined information from page/document content. The LLM also assigns a relevancy score (1/5 to 5/5) indicating how well the content matches the extraction query.
    *   **Consolidated Overview:** Automatically generates a synthesized overview. 
        *   If a specific information query was provided by the user for extraction, the consolidated summary will focus on that query, using only individual item outputs that achieved a relevancy score of 3/5 or higher.
        *   If no specific information query was used, a general consolidated summary is created from all valid individual LLM outputs.
*   **Interactive UI & Results Display:**
    *   Built with Streamlit for easy input, configuration, and viewing of results and processing logs.
    *   **Visual Relevancy Cues:** Individually processed items are prefixed with emojis (e.g., 5️⃣, 4️⃣, ✨3️⃣, 3️⃣) in the results list to quickly identify items with relevancy scores of 3/5 or higher. Results from LLM-generated queries are also marked.
*   **Google Sheets Integration:**
    *   Stores detailed results, including a batch summary row and individual item rows, in a structured Google Sheet.
*   **Download Results:** Option to download all processed item details and the consolidated summary into an Excel (`.xlsx`) file.
*   **Safeguards & Performance:**
    *   **Retry Mechanisms:** Implemented for Google Custom Search API calls and LLM API calls to handle transient errors and rate limits using exponential backoff.
    *   **LLM Caching:** Caches results from LLM functions (`@st.cache_data`) to improve performance on repeated identical requests and reduce API calls.
*   **Modular Design & Configuration:**
    *   Code is separated into functional modules.
    *   API keys and settings are managed via Streamlit Secrets (`secrets.toml`).

## Project Structure

    

IGNORE_WHEN_COPYING_START
Use code with caution. Markdown
IGNORE_WHEN_COPYING_END

streamlit-search-tool/
├── app.py # Main Streamlit application file (UI logic, orchestration)
├── modules/
│ ├── init.py
│ ├── config.py # Handles loading secrets & configurations
│ ├── search_engine.py # Google Search API interactions (with retries)
│ ├── scraper.py # Web fetching & content/metadata extraction (HTML & PDF)
│ ├── llm_processor.py # LLM interactions (Gemini: summaries, extractions, query gen, caching)
│ └── data_storage.py # Google Sheets interactions
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
*   **Google Gemini API Key (for LLM features).**
*   **Google Service Account for Google Sheets:** (JSON key file, share Sheet with service account email).

## Setup & Installation

1.  **Clone the repository (if applicable).**
2.  **Create a virtual environment (recommended).**
3.  **Install dependencies:**
    Ensure your `requirements.txt` includes `google-generativeai`, `pandas`, `openpyxl`, `streamlit`, `trafilatura`, `PyMuPDF`, `google-api-python-client`, and `beautifulsoup4`.
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
    GOOGLE_GEMINI_MODEL = "models/gemini-1.5-flash-latest" 

    # Google Sheets Integration
    SPREADSHEET_ID = "YOUR_GOOGLE_SHEET_ID_FROM_ITS_URL" 
    WORKSHEET_NAME = "Sheet1" 

    [gcp_service_account]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "your-private-key-id"
    private_key = """-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_CONTENT\n-----END PRIVATE KEY-----\n"""
    client_email = "your-service-account-email@your-gcp-project-id.iam.gserviceaccount.com"
    client_id = "your-client-id"
    # ... (other service account fields as in your JSON key)
    ```

## Running the Application

```bash
streamlit run app.py

    

IGNORE_WHEN_COPYING_START
Use code with caution.
IGNORE_WHEN_COPYING_END

Access at http://localhost:8501.
Usage

    Navigate to the application URL.

    Enter initial search keywords in the "Keywords (comma-separated)" field. Press Enter.

    Configure "Number of successfully scraped results per keyword".

    Optionally, enable "✨ Enhance with LLM-generated search queries?".

    Optionally, enable "Generate LLM Summary?".

    Optionally, provide a query in "Specific info to extract with LLM...". This influences extraction, relevancy scoring, LLM query generation context, and the focus of the consolidated summary.

    Click "🚀 Start Search & Analysis".

    View progress and results. Expanders for individual items will show emoji prefixes (5️⃣, 4️⃣, ✨3️⃣, 3️⃣) for relevancy scores >= 3 and markers for results from LLM-generated queries or PDFs (📄).

    Expand the processing log for details.

    If Google Sheets is set up, data is written there.

    Download results via the Excel download button.

Important Note for Contributing Agents

When modifying the codebase (app.py, modules/*.py):

    Always provide the FULL and COMPLETE code for the modified file(s). Do not provide only snippets or omit sections like docstrings or existing function implementations unless explicitly asked to do so for brevity in a specific, isolated discussion. The primary user will be copy-pasting the entire file content.

    Ensure all docstrings (module, function, class) are updated to reflect any changes in parameters, behavior, or new functionalities.

    Maintain versioning comments at the top of each modified file. Increment the version appropriately (e.g., v1.9.2 to v1.9.3) and briefly note the key change in the version comment.

    Every Python code file provided MUST end with the comment // end of [filename] as the absolute last line. For example, app.py should end with // end of app.py.

Adherence to these guidelines is crucial for maintaining clarity and ensuring the primary user can integrate changes smoothly.
Future API Layer Plan

The modular design supports future API development (e.g., FastAPI/Flask) for programmatic access and integration with other tools.
Contributing

[Details on how to contribute to the project, if applicable.]
License

[Specify the license for your project, e.g., MIT License.]
