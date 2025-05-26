      
# Streamlit Keyword Search & Analysis Tool 

This Streamlit application allows users to input keywords, perform Google searches, extract metadata and main content from the resulting URLs, and leverage a Large Language Model (LLM) – currently configured for **Google Gemini** – to summarize content, extract specific information, and generate a consolidated overview. Results are displayed in the app, can be downloaded as an Excel file, and are also recorded to a Google Sheet.

The project is designed with modularity to facilitate future enhancements, including the potential addition of a separate API layer for programmatic access.

## Features

*   **Keyword Search:** Perform Google searches for multiple keywords.
*   **Configurable Search Depth:** Specify the number of desired successfully scraped results per keyword. Oversampling is used to improve success rates.
*   **Metadata Extraction:** Fetches URL, page title, meta description, and OpenGraph tags.
*   **Main Content Extraction:** Uses `trafilatura` to extract the primary text content from web pages.
*   **LLM Integration (Google Gemini):**
    *   **Individual Summaries:** Generates summaries of each successfully scraped web page's content.
    *   **Specific Information Extraction:** Extracts user-defined information from page content.
    *   **Consolidated Overview:** Automatically generates a synthesized overview from all valid individual LLM outputs of the batch.
*   **Google Sheets Integration:**
    *   Stores detailed results, including a batch summary row and individual item rows, in a structured Google Sheet.
    *   Uses a master header for clarity.
*   **Download Results:** Option to download all processed item details and the consolidated summary into an Excel (`.xlsx`) file with separate sheets.
*   **Interactive UI:** Built with Streamlit for easy input, configuration, and viewing of results and processing logs.
*   **Modular Design:** Code is separated into functional modules for better maintainability and future expansion.
*   **Configuration Management:** API keys and settings are managed via Streamlit Secrets (`secrets.toml`).

## Project Structure


streamlit-search-tool/
├── app.py # Main Streamlit application file (UI logic)
├── modules/
│ ├── init.py
│ ├── config.py # Handles loading secrets & configurations
│ ├── search_engine.py # Google Search API interactions
│ ├── scraper.py # Web fetching & content/metadata extraction
│ ├── llm_processor.py # LLM interactions (now primarily Gemini)
│ ├── data_storage.py # Google Sheets interactions
│ └── utils.py # (Currently empty, for future helper functions)
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
    *   "Google Drive API" enabled (required by `gspread` for full sheet access functionality).
    *   "Generative Language API" (or "Vertex AI API" if using Gemini through Vertex) enabled for LLM features.
*   **Google Custom Search Engine (CSE):**
    *   Create one at [Programmable Search Engine](https://programmablesearchengine.google.com/).
    *   Note your **Search engine ID (CX)**.
*   **Google API Key (for Custom Search):**
    *   Create an API key in your GCP project with access to the Custom Search API.
*   **Google Gemini API Key (for LLM features):**
    *   Generate an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
*   **Google Service Account for Google Sheets:**
    *   Create a service account in GCP.
    *   Download its JSON key file.
    *   Share your target Google Sheet (and ensure it exists) with the service account's email address (grant "Editor" permissions).

## Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd streamlit-search-tool
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Ensure your `requirements.txt` includes `google-generativeai`, `pandas`, and `openpyxl` among others.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Secrets:**
    *   Create a file named `.streamlit/secrets.toml`.
    *   Add your API keys and other sensitive information.

    **Example `secrets.toml` structure:**
    ```toml
    # Google Custom Search
    GOOGLE_API_KEY = "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY"
    CSE_ID = "YOUR_CUSTOM_SEARCH_ENGINE_ID"

    # LLM Configuration (using Google Gemini by default)
    LLM_PROVIDER = "google" # Can be "google" or "openai" (if OpenAI logic is retained/added)
    GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FROM_AI_STUDIO"
    GOOGLE_GEMINI_MODEL = "models/gemini-1.5-flash-latest" # Or another model like "models/gemini-pro"

    # Optional: For OpenAI if you switch LLM_PROVIDER
    # OPENAI_API_KEY = "sk-YOUR_OPENAI_API_KEY"
    # OPENAI_MODEL_SUMMARIZE = "gpt-3.5-turbo"

    # Google Sheets Integration
    SPREADSHEET_ID = "YOUR_GOOGLE_SHEET_ID_FROM_ITS_URL" # More robust
    # SPREADSHEET_NAME = "Your Target Google Sheet Name" # Fallback if ID is not used
    WORKSHEET_NAME = "Sheet1" # Or your target worksheet tab name

    [gcp_service_account]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "your-private-key-id"
    private_key = """-----BEGIN PRIVATE KEY-----
    YOUR_VERY_LONG_PRIVATE_KEY_CONTENT_WITH_NEWLINES_PRESERVED
    -----END PRIVATE KEY-----"""
    client_email = "your-service-account-email@your-gcp-project-id.iam.gserviceaccount.com"
    client_id = "your-client-id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "PASTE_client_x509_cert_url_HERE"
    # universe_domain = "googleapis.com" # If present in your service account JSON
    ```

## Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py

our default web browser should open to the application (usually http://localhost:8501).
Usage

    Navigate to the application URL.

    Enter keywords in the sidebar.

    Configure the "Number of successfully scraped results per keyword".

    Optionally, enable LLM summarization and/or provide a query for specific information extraction using the configured LLM provider (default is Google Gemini).

    Click the "🚀 Start Search & Analysis" button.

    View the progress, individual item results (including metadata and LLM insights), and the automatically generated consolidated overview in the main panel.

    The processing log can be expanded to see detailed steps.

    If Google Sheets integration is configured successfully, a batch summary row followed by individual item data rows will be written to your specified sheet.

    A "📥 Download Results as Excel" button will appear below the results, allowing you to download an Excel file with "Item Details" and "Consolidated Summary" sheets.

Future API Layer Plan

The modular design of this project (modules/ directory) is intended to support the future development of a separate API layer (e.g., using FastAPI or Flask). This API would:

    Reuse the core logic functions from the modules/ directory.

    Provide HTTP endpoints for programmatic access to the tool's capabilities.

    Allow integration with tools like n8n, Make.com, or custom scripts.

Contributing

[Details on how to contribute to the project, if applicable.]
License

[Specify the license for your project, e.g., MIT License.]
    
