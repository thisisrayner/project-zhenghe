      
# Streamlit Keyword Search & Analysis Tool 🔎📝

This Streamlit application allows users to input keywords, perform Google searches, extract metadata and main content from the resulting URLs, and optionally use an LLM (e.g., OpenAI's GPT) to summarize content or extract specific information. Results are then displayed and can be (optionally) recorded to a Google Sheet.

The project is designed with modularity to facilitate future enhancements, including the potential addition of a separate API layer for programmatic access.

## Features

*   **Keyword Search:** Perform Google searches for multiple keywords.
*   **Configurable Search Depth:** Specify the number of search results per keyword.
*   **Metadata Extraction:** Fetches URL, page title, meta description, and OpenGraph tags.
*   **Main Content Extraction:** Uses `trafilatura` to extract the primary text content from web pages.
*   **LLM Integration (Optional):**
    *   **Summarization:** Generate concise summaries of web page content.
    *   **Specific Information Extraction:** Query the LLM to find particular pieces of information from the text.
*   **Google Sheets Integration (Optional):** Store search results and LLM outputs in a Google Sheet.
*   **Interactive UI:** Built with Streamlit for easy use.

## Project Structure

    

IGNORE_WHEN_COPYING_START
Use code with caution. Markdown
IGNORE_WHEN_COPYING_END

streamlit-search-tool/
├── app.py # Main Streamlit application file (UI logic)
├── modules/
│ ├── init.py # Makes 'modules' a Python package
│ ├── config.py # Handles loading secrets & configurations
│ ├── search_engine.py # Google Search API interactions
│ ├── scraper.py # Web fetching & content/metadata extraction
│ ├── llm_processor.py # LLM interactions
│ ├── data_storage.py # Google Sheets (or other storage) interactions
│ └── utils.py # Helper functions (logging, error handling, etc.)
├── .streamlit/
│ └── secrets.toml # Storing API keys & sensitive info (for Streamlit Cloud)
├── requirements.txt # Python dependencies
└── README.md # This file

      
## Prerequisites

Before you begin, ensure you have met the following requirements:

*   **Python 3.8+**
*   **Google Account**
*   **Google Cloud Platform (GCP) Project:**
    *   "Google Custom Search API" enabled.
    *   "Google Sheets API" enabled.
*   **Google Custom Search Engine (CSE):**
    *   Create one at [Programmable Search Engine](https://programmablesearchengine.google.com/).
    *   Note your **Search engine ID (CX)**.
*   **Google API Key:**
    *   Create an API key in your GCP project with access to the Custom Search API.
*   **OpenAI API Key (Optional):**
    *   If using LLM features, get an API key from [OpenAI Platform](https://platform.openai.com/api-keys).
*   **Google Service Account for Google Sheets (Optional):**
    *   Create a service account in GCP.
    *   Download its JSON key file.
    *   Share your target Google Sheet with the service account's email address (grant "Editor" permissions).

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
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Secrets:**
    *   Create a file named `.streamlit/secrets.toml`.
    *   Add your API keys and other sensitive information to this file. **Do not commit `secrets.toml` if it contains real secrets and your repository is public.** For local development outside Streamlit Cloud, you might need to adjust `modules/config.py` to read from environment variables or a `.env` file if `st.secrets` isn't directly available or suitable.

    **Example `secrets.toml` structure:**

    ```toml
    # .streamlit/secrets.toml

    GOOGLE_API_KEY = "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY"
    CSE_ID = "YOUR_CUSTOM_SEARCH_ENGINE_ID"

    # Optional: For LLM features
    OPENAI_API_KEY = "sk-YOUR_OPENAI_API_KEY"

    # Optional: For Google Sheets integration
    # Store the entire content of your GCP service account JSON key here
    # You can copy-paste the JSON content directly or use a tool to format it for TOML
    # Example for storing the JSON content directly as a multi-line string (ensure proper TOML escaping if needed)
    # Or, more commonly, store individual fields if preferred by your gspread setup.
    # For gspread from_service_account_info, you'd typically pass a dictionary.
    # A simple way is to store the JSON string and parse it in config.py,
    # or store key fields if your config.py expects them.
    # Example (if your config.py expects a dictionary from this structure):
    [gcp_service_account]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "your-private-key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_CONTENT\n-----END PRIVATE KEY-----\n"
    client_email = "your-service-account-email@your-gcp-project-id.iam.gserviceaccount.com"
    client_id = "your-client-id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-gcp-project-id.iam.gserviceaccount.com"
    # Add any other fields from your service account JSON

    # Configuration for Google Sheets
    SPREADSHEET_NAME = "Your Target Google Sheet Name"
    WORKSHEET_NAME = "Sheet1" # Or your target worksheet name
    ```

## Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py

    # project-zhenghe
