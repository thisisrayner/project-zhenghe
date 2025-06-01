      
# D.O.R.A - The Research Agent
https://github.com/thisisrayner/project-zhenghe

The **Research** **Agent** For **Domain**-Wide **Overview** and **Insights**.

D.O.R.A (Domain Overview & Research Agent) is a Streamlit application that empowers users to input keywords, perform Google searches (automatically enhanced by LLM-generated queries if an LLM is configured), extract metadata and main content from resulting URLs (supporting HTML and PDF text), and leverage a Large Language Model (LLM) – currently configured for **Google Gemini** – to summarize content, extract specific information with relevancy scoring, and generate a consolidated overview. Results are displayed interactively with visual cues for relevance, can be downloaded as an Excel file, and are also recorded to a Google Sheet.

The project is designed with a high degree of modularity, with `app.py` acting as an orchestrator for specialized modules. This facilitates future enhancements, including the potential addition of a separate API layer for programmatic access.

## Features

*   **Keyword Search:**
    *   Perform Google searches for multiple user-defined keywords (comma-separated, "Enter" to commit input).
    *   **LLM-Enhanced Queries:** If an LLM is configured, it automatically generates additional, related search queries. This generation is based on the user's initial input keywords and their specific information goals (Main Query 1 and, if provided, Additional Query 2), aiming to broaden the search scope effectively to find the most relevant content.
*   **Configurable Search Depth:** Specify the number of desired successfully scraped results per keyword. Oversampling is used to improve success rates.
*   **Content Extraction:**
    *   **HTML:** Fetches URL, page title, meta description, and OpenGraph tags. Uses `trafilatura` for main text extraction.
    *   **PDF:** Extracts document title (from metadata or filename) and full text content from PDF documents using `PyMuPDF`.
*   **LLM Integration (Google Gemini):**
    *   **Individual Summaries:** If an LLM is configured, it automatically generates summaries of each successfully scraped web page's/document's content.
    *   **Specific Information Extraction & Relevancy Scoring:** Extracts user-defined information (for up to two specific queries, Q1 and Q2) from page/document content. The LLM also assigns a relevancy score (1/5 to 5/5) indicating how well the content matches each extraction query. LLM prompts for this feature instruct the LLM to output only plain text.
    *   **Consolidated Overview:** Automatically generates a synthesized overview.
        *   If specific information query/queries (Q1 and/or Q2) were provided and items achieve a relevancy score of 3/5 or higher for *either* query, the consolidated summary will be **focused**. It will use the full text of these high-scoring Q1/Q2 extractions. The LLM is prompted to provide a more detailed and potentially longer overview (aiming for up to 2x the length of a general summary) based on these specific, relevant snippets. The "Main Query 1" serves as the primary contextual theme, and "Additional Query 2" (if provided and relevant snippets are found) is used by the LLM to enrich the overview with more nuanced details.
        *   If no specific information queries were used, or if no items achieved a relevancy score of 3/5 or higher for any provided specific query, a **general consolidated summary** is created from all valid individual LLM-generated item summaries.
        *   All consolidated overviews are instructed to be generated in plain text to avoid unwanted formatting.
*   **Interactive UI & Results Display:**
    *   Built with Streamlit for easy input, configuration, and viewing of results and processing logs.
    *   **Visual Relevancy Cues:** Individually processed items in the results list are prefixed with visual markers based on relevancy scores for Q1 and Q2.
    *   A 🤖 marker for any item originating from an LLM-generated search query.
    *   A 📄 marker for PDF documents.
    *   An expander below a focused consolidated summary lists the specific source items (URL, query type, score) that contributed to its generation.
*   **Google Sheets Integration:**
    *   Stores detailed results, including a batch summary row and individual item rows, in a structured Google Sheet.
*   **Download Results:** Option to download all processed item details and the consolidated summary into an Excel (`.xlsx`) file.
*   **Safeguards & Performance:**
    *   **Retry Mechanisms:** Implemented for Google Custom Search API calls and LLM API calls to handle transient errors and rate limits using exponential backoff.
    *   **LLM Caching:** Caches results from LLM functions (`@st.cache_data`) to improve performance on repeated identical requests and reduce API calls.
*   **Modular Design & Configuration:**
    *   Code is separated into functional modules. `app.py` serves as the main orchestrator.
    *   API keys and settings are managed via Streamlit Secrets (`secrets.toml`).

## Project Structure

    dora-research-agent/ # Assuming project root, can be updated if actual folder name changes
    ├── app.py # Main Streamlit application orchestrator for D.O.R.A
    ├── modules/
    │   ├── __init__.py
    │   ├── config.py # Handles loading secrets & configurations
    │   ├── search_engine.py # Google Search API interactions
    │   ├── scraper.py # Web fetching & content/metadata extraction (HTML & PDF)
    │   ├── llm_processor.py # LLM interactions (summaries, extractions, query gen, caching)
    │   ├── data_storage.py # Google Sheets interactions
    │   ├── ui_manager.py # Streamlit UI rendering and input handling
    │   ├── process_manager.py # Core search, scrape, and analysis workflow orchestration
    │   └── excel_handler.py # Excel file generation and formatting
    ├── .streamlit/
    │   └── secrets.toml # Storing API keys & sensitive info
    ├── requirements.txt # Python dependencies
    └── README.md # This file

## Prerequisites & Configuration

To run this application, you will need the codebase and the following:
*   Python 3.8+
*   All dependencies listed in `requirements.txt` installed.
*   Google Account & Google Cloud Platform (GCP) Project with necessary APIs enabled (Custom Search, Sheets, Drive, Generative Language/Vertex AI).
*   Google Custom Search Engine (CSE) ID.
*   Google API Key (for Custom Search).
*   Google Gemini API Key (or other LLM provider API key if support is extended).
*   Google Service Account for Google Sheets (JSON key file).
*   Configure Secrets (`.streamlit/secrets.toml`) as per the example structure provided below.

    **Example `secrets.toml` structure:**
    ```toml
    # Google Custom Search
    GOOGLE_API_KEY = "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY"
    CSE_ID = "YOUR_CUSTOM_SEARCH_ENGINE_ID"

    # LLM Configuration (using Google Gemini by default)
    LLM_PROVIDER = "google" # or "openai"
    GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FROM_AI_STUDIO"
    GOOGLE_GEMINI_MODEL = "models/gemini-1.5-flash-latest" 
    # OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    # OPENAI_MODEL_SUMMARIZE = "gpt-3.5-turbo" 
    # OPENAI_MODEL_EXTRACT = "gpt-3.5-turbo"

    # Google Sheets Integration
    SPREADSHEET_ID = "YOUR_GOOGLE_SHEET_ID_FROM_ITS_URL"
    WORKSHEET_NAME = "Sheet1" 

    [gcp_service_account]
    type = "service_account"
    # ... (rest of service account details) ...
    ```

## Running the Application

From the root directory of the project:
```bash
streamlit run app.py


Access D.O.R.A in your browser, typically at http://localhost:8501.
Usage

    Navigate to the application URL.

    Enter initial search keywords.

    Configure the number of results per keyword.

    Optionally, provide "Main Query 1" and "Additional Query 2" for specific information extraction and to guide LLM-generated searches and focused consolidated overviews.

    Click "🚀 Start Search & Analysis".

    View progress, results, and the consolidated overview. If the overview was focused, an expander will show the source items used.

    Expand the "📜 View Processing Log" for details.

    Download results via the "📥 Download Results as Excel" button.

Note: LLM-enhanced query generation and individual item summarization/extraction are automatic features that run if a valid LLM API key is provided.
Important Note for Contributing Agents

When modifying the codebase (app.py, modules/*.py):

    Provide Options First: When asked for input on code design, solutions to problems, or ideas for implementation, first present a set of options or a discussion of possibilities. Do not proceed directly to writing or modifying code based on a potential solution until explicitly instructed to do so after the options have been reviewed.

    Always provide the FULL and COMPLETE code for the modified file(s). Do not provide only snippets or omit sections like docstrings or existing function implementations unless explicitly asked to do so for brevity in a specific, isolated discussion. The primary user will be copy-pasting the entire file content.

    Comprehensive Docstring Updates are CRUCIAL:

        Ensure all docstrings (module-level, function-level, and class-level) are meticulously updated to reflect any changes in parameters, return types, behavior, or new functionalities. The automated codebase_summary.md relies heavily on these docstrings for accuracy.

        Module Docstrings: Should provide a concise overview of the module's purpose and its primary responsibilities within the application. Aim for 1-3 clear sentences.

        Function/Method Docstrings:

            Start with a concise one-line summary of what the function/method does.

            Clearly document all Args:.

            Clearly document Returns:.

            If the function can Raises: specific exceptions, document these.

        Class Docstrings: Describe the purpose of the class.

        Avoid "None" or Placeholder Docstrings.

    Maintain versioning comments at the top of each modified file. Increment the version appropriately and briefly note the key change(s).

    Every Python code file provided MUST end with the comment // end of [filename] as the absolute last line.

Adherence to these guidelines is crucial for maintaining clarity, ensuring the primary user can integrate changes smoothly, and keeping the automated codebase_summary.md accurate and useful.
Future API Layer Plan

The modular design supports future API development (e.g., using FastAPI or Flask) for programmatic access and integration with other tools. This could involve creating endpoints that accept keywords and parameters, returning structured JSON results for D.O.R.A's analyses.
Contributing

[Details on how to contribute to the project, if applicable.]
License

[Specify the license for your project, e.g., MIT License, Apache 2.0 License.]
