# modules/config.py
# Version 1.4.1: Added APP_VERSION constant.
# Version 1.4: Enhanced docstrings, type hinting, and added comments.

"""
Configuration management for the Streamlit Keyword Search & Analysis Tool.

This module defines dataclasses for structuring configuration parameters and
provides a function to load these configurations primarily from Streamlit
secrets (`secrets.toml`). It handles settings for Google Search, LLM providers
(Google Gemini, OpenAI), and Google Sheets integration. It also defines
the application's version.
"""

import streamlit as st
from dataclasses import dataclass, field
import json
from typing import Optional, Dict, Any

# --- Application Version ---
APP_VERSION = "3.1.1" # Or align with the app.py version you are setting

# --- Configuration Classes ---
@dataclass
class GoogleSearchConfig:
    """Configuration specific to Google Custom Search API."""
    api_key: Optional[str] = None
    cse_id: Optional[str] = None

@dataclass
class LLMConfig:
    """Configuration for Large Language Model (LLM) interactions."""
    provider: str = "google"  # "google" or "openai"
    # OpenAI specific
    openai_api_key: Optional[str] = None
    openai_model_summarize: str = "gpt-3.5-turbo"
    openai_model_extract: str = "gpt-3.5-turbo"
    # Google Gemini specific
    google_gemini_api_key: Optional[str] = None
    google_gemini_model: str = "models/gemini-1.5-flash-latest"

    max_input_chars: int = 100000  # Max characters to send to LLM (practical limit)

@dataclass
class GoogleSheetsConfig:
    """Configuration for Google Sheets integration."""
    service_account_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    spreadsheet_name: Optional[str] = None # For display or fallback
    spreadsheet_id: Optional[str] = None   # Primary identifier for the sheet
    worksheet_name: str = "Sheet1"

@dataclass
class AppConfig:
    """Main application configuration, aggregating other configs."""
    google_search: GoogleSearchConfig = field(default_factory=GoogleSearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gsheets: GoogleSheetsConfig = field(default_factory=GoogleSheetsConfig)
    num_results_per_keyword_default: int = 3
    # Add other general app settings here, e.g.,
    # default_oversample_factor: float = 2.0

# --- Main Config Loading Function ---
@st.cache_resource # Cache the loaded configuration object for the session
def load_config() -> Optional[AppConfig]:
    """
    Loads application configurations from Streamlit secrets.

    Reads API keys, model names, sheet identifiers, and other settings
    from `.streamlit/secrets.toml`. Provides sensible defaults if some
    optional settings are not found.

    Returns:
        Optional[AppConfig]: An AppConfig object populated with settings,
                             or None if essential configurations (like
                             Google Search API keys) are missing.
    """
    cfg = AppConfig()
    essential_secrets_loaded = True # Flag to track if critical secrets are present

    # --- Google Search Config ---
    try:
        cfg.google_search.api_key = st.secrets["GOOGLE_API_KEY"]
        cfg.google_search.cse_id = st.secrets["CSE_ID"]
    except KeyError as e:
        st.error(f"Missing Google Search secret: {e}. Search functionality will fail.")
        essential_secrets_loaded = False

    # --- LLM Configuration ---
    # Determine LLM provider from secrets, default to "google"
    cfg.llm.provider = st.secrets.get("LLM_PROVIDER", "google").lower()

    if cfg.llm.provider == "openai":
        try:
            cfg.llm.openai_api_key = st.secrets.get("OPENAI_API_KEY")
            cfg.llm.openai_model_summarize = st.secrets.get("OPENAI_MODEL_SUMMARIZE", cfg.llm.openai_model_summarize)
            cfg.llm.openai_model_extract = st.secrets.get("OPENAI_MODEL_EXTRACT", cfg.llm.openai_model_extract)
            if not cfg.llm.openai_api_key:
                st.caption("LLM Provider is OpenAI, but OPENAI_API_KEY is missing. LLM features will be disabled.")
        except Exception as e: # Catch any unexpected issues during optional OpenAI config load
            st.warning(f"Could not load OpenAI specific config: {e}")
    elif cfg.llm.provider == "google":
        try:
            cfg.llm.google_gemini_api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
            cfg.llm.google_gemini_model = st.secrets.get("GOOGLE_GEMINI_MODEL", cfg.llm.google_gemini_model)
            if not cfg.llm.google_gemini_api_key:
                st.caption("LLM Provider is Google, but GOOGLE_GEMINI_API_KEY is missing. LLM features will be disabled.")
        except Exception as e:
            st.warning(f"Could not load Google Gemini specific config: {e}") # Changed to warning
    else:
        st.warning(f"Unsupported LLM_PROVIDER: '{cfg.llm.provider}'. LLM features will be disabled.")

    # Load general LLM settings
    cfg.llm.max_input_chars = int(st.secrets.get("LLM_MAX_INPUT_CHARS", cfg.llm.max_input_chars))

    # --- Google Sheets Config (Optional) ---
    try:
        # Load the service account info if the [gcp_service_account] table exists in secrets
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        
        cfg.gsheets.spreadsheet_id = st.secrets.get("SPREADSHEET_ID")
        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME") # Still load name for potential display/fallback
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", cfg.gsheets.worksheet_name)

        # Check if critical sheet identifiers are missing if service account info IS present
        if cfg.gsheets.service_account_info and not (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name):
            st.caption("Google Sheets: Service account configured, but SPREADSHEET_ID (recommended) or SPREADSHEET_NAME is missing.")
    except json.JSONDecodeError as e: # Should not happen if TOML is correct
        st.warning(f"Error parsing Google Sheets service account JSON from secrets (unexpected): {e}")
    except Exception as e:
        st.warning(f"Could not fully load Google Sheets config (optional section): {e}")

    if not essential_secrets_loaded:
        st.error("Essential configurations are missing. Please check secrets.toml. App may not function correctly.")
        return None # Critical failure

    return cfg

if __name__ == '__main__':
    # Test block for verifying config loading
    st.set_page_config(layout="wide")
    st.title("Config Loader Test (v1.4.1)") # Updated test title
    st.write(f"App Version from config: {APP_VERSION}") # Test the new constant
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.subheader("Google Search Config:")
        st.json(vars(loaded_cfg.google_search))
        st.subheader("LLM Config:")
        st.json(vars(loaded_cfg.llm))
        st.subheader("Google Sheets Config:")
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")
    else:
        st.error("Failed to load critical configuration parts.")
# end of modules/config.py
