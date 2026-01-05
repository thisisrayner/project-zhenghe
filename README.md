# D.O.R.A - The Research Agent

The **Research** **Agent** For **Domain**-Wide **Overview** and **Insights**.

D.O.R.A. (Domain-wide Overview Research Agent) is a powerful Streamlit application that automates deep, multi-source research. It leverages Google Search, advanced web scraping, and Large Language Models (LLMs) to transform keywords into comprehensive, consolidated reports.

## Latest Features

*   **Multi-Model Gemini Workflow:**
    *   Optimized for cost and quality: Uses faster models (e.g., `gemini-2.0-flash`) for search query generation, summarization, and extraction, while utilizing a more reasoning-heavy model (e.g., `gemini-3-pro-preview`) for the final consolidated summary.
*   **Research Voices (Bias Control):**
    *   **Ground Voice**: Biases 60% of search results towards community forums like Reddit for "authentic" consumer sentiment.
    *   **Tech Voice**: Biases results towards technical domains (GitHub, StackOverflow, etc.).
    *   **Market Voice**: Biases results towards market intelligence and consulting sites (Bloomberg, McKinsey, etc.).
    *   **General**: Standard balanced search.
*   **Search Pagination:** Now supports fetching up to **30 successfully scraped results per keyword** (automatically handling multiple API pagination requests).
*   **Daily Usage Tracker:** A real-time UI indicator tracks your Google Search API calls, persisting counts to prevent exceeding the 100-query daily free limit.
*   **Intelligent Source Filtering:**
    *   **Wikipedia Deprioritization**: Automatically moves Wikipedia results to the end of the queue to ensure fresher, more diverse primary sources are processed first.
*   **Content Extraction:**
    *   **HTML & PDF Support**: Robust scraping of standard web pages and full-text extraction from PDF files via `PyMuPDF`.
*   **Relevancy Scoring (1/5 to 5/5):** Every piece of extracted information is scored for relevance against user queries, with visual markers in the UI.

## Project Structure

    /project-zhenghe
    ├── app.py                     # Main Streamlit application orchestrator
    ├── modules/
    │   ├── usage_tracker.py       # NEW: Daily API quota persistence logic
    │   ├── config.py              # Handles loading secrets & configurations
    │   ├── search_engine.py       # Google Search API interactions (v1.2.3 with pagination)
    │   ├── scraper.py             # Web fetching & content extraction (HTML & PDF)
    │   ├── llm_processor.py       # LLM interactions & caching
    │   ├── data_storage.py        # Google Sheets interactions
    │   ├── ui_manager.py          # Streamlit UI & Research Voice selection
    │   ├── process_manager.py     # Core workflow (60/40 voice split logic)
    │   └── excel_handler.py       # Excel reports (.xlsx)
    ├── usage_stats.json           # Auto-generated daily usage persistence
    ├── .streamlit/
    │   └── secrets.toml           # API keys & model configuration
    └── requirements.txt           # Python dependencies

## Prerequisites & Configuration

1.  **GCP Setup**: Enable Custom Search API and create a Custom Search Engine (CSE).
2.  **LLM Keys**: Obtain a Google Gemini API key from AI Studio.
3.  **Secrets (`.streamlit/secrets.toml`):**

```toml
# Google Custom Search
GOOGLE_API_KEY = "..."
CSE_ID = "..."

# LLM Configuration
LLM_PROVIDER = "google"
GOOGLE_GEMINI_API_KEY = "..."

# Main model for indexing/summarizing (Step 1-3)
GOOGLE_GEMINI_MODEL = "models/gemini-2.0-flash" 

# Stronger model for final consolidated report (Step 4)
GOOGLE_GEMINI_MODEL_CONSOLIDATION = "models/gemini-3-pro-preview"

# Google Sheets (Optional)
SPREADSHEET_ID = "..."
WORKSHEET_NAME = "Sheet1"

[gcp_service_account]
# ... (service account JSON fields) ...
```

## Running the Application

```bash
streamlit run app.py
```

## Important Note for Contributing Agents

*   **Logic Isolation**: Long-running processes in `process_manager.py` must return status via logs rather than updating UI directly to avoid race conditions.
*   **Docstring Integrity**: The `codebase_summary.md` is automatically generated from docstrings. Meticulous documentation of `Args` and `Returns` is mandatory.
*   **Versioning**: Increment file version headers and strictly follow the `# // end of [filename]` footer rule.

---
*D.O.R.A - Built for deep insights.*
