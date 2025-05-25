# modules/config.py
# Version 1.3: Added SPREADSHEET_ID for more robust sheet access.

import streamlit as st
from dataclasses import dataclass, field
import json

# --- Configuration Classes ---
@dataclass
class GoogleSearchConfig:
    api_key: str | None = None
    cse_id: str | None = None

@dataclass
class LLMConfig:
    provider: str = "google"
    openai_api_key: str | None = None
    openai_model_summarize: str = "gpt-3.5-turbo"
    openai_model_extract: str = "gpt-3.5-turbo"
    google_gemini_api_key: str | None = None
    google_gemini_model: str = "models/gemini-1.5-flash-latest"
    max_input_chars: int = 100000

@dataclass
class GoogleSheetsConfig:
    service_account_info: dict | None = field(default_factory=dict)
    spreadsheet_name: str | None = None # Keep for potential fallback or display
    spreadsheet_id: str | None = None   # NEW: For opening by ID
    worksheet_name: str = "Sheet1"

@dataclass
class AppConfig:
    google_search: GoogleSearchConfig = field(default_factory=GoogleSearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gsheets: GoogleSheetsConfig = field(default_factory=GoogleSheetsConfig)
    num_results_per_keyword_default: int = 3

# --- Main Config Loading Function ---
@st.cache_resource
def load_config() -> AppConfig | None:
    cfg = AppConfig()
    essential_secrets_loaded = True

    # --- Google Search Config ---
    try:
        cfg.google_search.api_key = st.secrets["GOOGLE_API_KEY"]
        cfg.google_search.cse_id = st.secrets["CSE_ID"]
    except KeyError as e:
        st.error(f"Missing Google Search secret: {e}. Application cannot start for search.")
        essential_secrets_loaded = False

    # --- LLM Configuration ---
    cfg.llm.provider = st.secrets.get("LLM_PROVIDER", "google").lower()
    if cfg.llm.provider == "openai":
        try:
            cfg.llm.openai_api_key = st.secrets.get("OPENAI_API_KEY")
            cfg.llm.openai_model_summarize = st.secrets.get("OPENAI_MODEL_SUMMARIZE", cfg.llm.openai_model_summarize)
            cfg.llm.openai_model_extract = st.secrets.get("OPENAI_MODEL_EXTRACT", cfg.llm.openai_model_extract)
            if not cfg.llm.openai_api_key: st.caption("OpenAI selected, but API Key missing.")
        except Exception as e: st.warning(f"Could not load OpenAI config: {e}")
    elif cfg.llm.provider == "google":
        try:
            cfg.llm.google_gemini_api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
            cfg.llm.google_gemini_model = st.secrets.get("GOOGLE_GEMINI_MODEL", cfg.llm.google_gemini_model)
            if not cfg.llm.google_gemini_api_key: st.caption("Google Gemini selected, but API Key missing.")
        except Exception as e: st.error(f"Could not load Google Gemini config: {e}")
    else: st.warning(f"Unsupported LLM_PROVIDER: '{cfg.llm.provider}'.")
    cfg.llm.max_input_chars = int(st.secrets.get("LLM_MAX_INPUT_CHARS", cfg.llm.max_input_chars))

    # --- Google Sheets Config (Optional) ---
    try:
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        
        # Load SPREADSHEET_ID first, then SPREADSHEET_NAME as a fallback
        cfg.gsheets.spreadsheet_id = st.secrets.get("SPREADSHEET_ID") # NEW
        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME") # Keep for reference
        
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", cfg.gsheets.worksheet_name)

        # Check if at least one identifier is present for the sheet
        if not cfg.gsheets.spreadsheet_id and not cfg.gsheets.spreadsheet_name:
            st.caption("Google Sheets: Neither SPREADSHEET_ID nor SPREADSHEET_NAME found in secrets.")
            # This will cause sheet_writing_enabled to be false in app.py

    except Exception as e:
        st.warning(f"Could not load Google Sheets config (optional): {e}")

    if not essential_secrets_loaded: return None
    return cfg

# ... (if __name__ == '__main__' block from config.py v1.2 - update to show spreadsheet_id)
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Config Loader Test (v1.3)")
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.json(vars(loaded_cfg.google_search))
        st.subheader("LLM Config:")
        st.json(vars(loaded_cfg.llm))
        st.subheader("Google Sheets Config:") # Updated for clarity
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")
    else:
        st.error("Failed to load critical configuration.")
# end of modules/config.py
