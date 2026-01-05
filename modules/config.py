# modules/config.py
# Version 1.5.1: Added separate configuration for Google Gemini Consolidation model.
# Version 1.5.0: Added LLM request delay and throttling threshold configurations.
# Version 1.4.1: Added APP_VERSION constant.
# Version 1.4: Enhanced docstrings, type hinting, and added comments.

"""
Configuration management for D.O.R.A. - The Research Agent.

This module defines dataclasses for structuring configuration parameters and
provides a function to load these configurations primarily from Streamlit
secrets (`secrets.toml`). It handles settings for Google Search, LLM providers
(Google Gemini, OpenAI), Google Sheets integration, application-specific
behaviors like LLM request throttling, and defines the application's version.
"""

import streamlit as st
from dataclasses import dataclass, field
import json # Retained from your version
from typing import Optional, Dict, Any

# --- Application Version ---
APP_VERSION = "3.1.1" # As per your provided file, or align with app.py

# --- Configuration Classes ---
@dataclass
class GoogleSearchConfig:
    """Configuration specific to Google Custom Search API."""
    api_key: Optional[str] = None
    cse_id: Optional[str] = None

@dataclass
class LLMConfig:
    """
    Configuration for Large Language Model (LLM) interactions.

    Attributes:
        provider: The LLM provider to use (e.g., "google", "openai").
        openai_api_key: API key for OpenAI.
        openai_model_summarize: The OpenAI model for summarization tasks.
        openai_model_extract: The OpenAI model for extraction tasks.
        google_gemini_api_key: API key for Google Gemini.
        google_gemini_model: The specific Google Gemini model to use.
        max_input_chars: Max characters to send to LLM (practical limit).
        llm_item_request_delay_seconds: Delay in seconds to apply after each item's
            LLM processing, if throttling is active.
        llm_throttling_threshold_results: The number of results per keyword at or
            above which LLM request throttling is activated.
    """
    provider: str = "google"
    # OpenAI specific
    openai_api_key: Optional[str] = None
    openai_model_summarize: str = "gpt-3.5-turbo"
    openai_model_extract: str = "gpt-3.5-turbo"
    # Google Gemini specific
    google_gemini_api_key: Optional[str] = None
    google_gemini_model: str = "models/gemini-1.5-flash-latest"
    google_gemini_model_consolidation: Optional[str] = None # Specific model for consolidation step

    max_input_chars: int = 100000
    # New throttling parameters
    llm_item_request_delay_seconds: float = 0.0  # Default: no delay
    llm_throttling_threshold_results: int = 999 # Default: throttling effectively off

@dataclass
class GoogleSheetsConfig:
    """Configuration for Google Sheets integration."""
    service_account_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    spreadsheet_name: Optional[str] = None
    spreadsheet_id: Optional[str] = None
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

    Reads API keys, model names, sheet identifiers, throttling parameters, and other
    settings from `.streamlit/secrets.toml`. Provides sensible defaults if some
    optional settings are not found.

    Returns:
        Optional[AppConfig]: An AppConfig object populated with settings,
                             or None if essential configurations (like
                             Google Search API keys) are missing.
    """
    cfg = AppConfig()
    essential_secrets_loaded = True

    # --- Google Search Config ---
    try:
        cfg.google_search.api_key = st.secrets["GOOGLE_API_KEY"]
        cfg.google_search.cse_id = st.secrets["CSE_ID"]
    except KeyError as e:
        st.error(f"Missing Google Search secret: {e}. Search functionality will fail.")
        essential_secrets_loaded = False

    # --- LLM Configuration ---
    cfg.llm.provider = st.secrets.get("LLM_PROVIDER", "google").lower()

    if cfg.llm.provider == "openai":
        try:
            cfg.llm.openai_api_key = st.secrets.get("OPENAI_API_KEY")
            cfg.llm.openai_model_summarize = st.secrets.get("OPENAI_MODEL_SUMMARIZE", cfg.llm.openai_model_summarize)
            cfg.llm.openai_model_extract = st.secrets.get("OPENAI_MODEL_EXTRACT", cfg.llm.openai_model_extract)
            if not cfg.llm.openai_api_key:
                st.caption("LLM Provider is OpenAI, but OPENAI_API_KEY is missing. LLM features will be disabled.")
        except Exception as e:
            st.warning(f"Could not load OpenAI specific config: {e}")
    elif cfg.llm.provider == "google":
        try:
            cfg.llm.google_gemini_api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
            cfg.llm.google_gemini_model = st.secrets.get("GOOGLE_GEMINI_MODEL", cfg.llm.google_gemini_model)
            # Load consolidation model, default to main model if not set
            cfg.llm.google_gemini_model_consolidation = st.secrets.get("GOOGLE_GEMINI_MODEL_CONSOLIDATION", cfg.llm.google_gemini_model)
            
            if not cfg.llm.google_gemini_api_key:
                st.caption("LLM Provider is Google, but GOOGLE_GEMINI_API_KEY is missing. LLM features will be disabled.")
        except Exception as e:
            st.warning(f"Could not load Google Gemini specific config: {e}")
    else:
        st.warning(f"Unsupported LLM_PROVIDER: '{cfg.llm.provider}'. LLM features will be disabled.")

    # Load general LLM settings
    cfg.llm.max_input_chars = int(st.secrets.get("LLM_MAX_INPUT_CHARS", cfg.llm.max_input_chars))

    # Load new LLM throttling settings
    cfg.llm.llm_item_request_delay_seconds = float(st.secrets.get("LLM_ITEM_REQUEST_DELAY_SECONDS", cfg.llm.llm_item_request_delay_seconds))
    cfg.llm.llm_throttling_threshold_results = int(st.secrets.get("LLM_THROTTLING_THRESHOLD_RESULTS", cfg.llm.llm_throttling_threshold_results))


    # --- Google Sheets Config (Optional) ---
    try:
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        
        cfg.gsheets.spreadsheet_id = st.secrets.get("SPREADSHEET_ID")
        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME")
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", cfg.gsheets.worksheet_name)

        if cfg.gsheets.service_account_info and not (cfg.gsheets.spreadsheet_id or cfg.gsheets.spreadsheet_name):
            st.caption("Google Sheets: Service account configured, but SPREADSHEET_ID (recommended) or SPREADSHEET_NAME is missing.")
    except json.JSONDecodeError as e:
        st.warning(f"Error parsing Google Sheets service account JSON from secrets (unexpected): {e}")
    except Exception as e:
        st.warning(f"Could not fully load Google Sheets config (optional section): {e}")

    if not essential_secrets_loaded:
        st.error("Essential configurations are missing. Please check secrets.toml. App may not function correctly.")
        return None

    return cfg

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Config Loader Test (v1.5.0)") # Updated test title for new version
    st.write(f"App Version from config: {APP_VERSION}")
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.subheader("Google Search Config:")
        st.json(vars(loaded_cfg.google_search))
        st.subheader("LLM Config:")
        st.json(vars(loaded_cfg.llm)) # This will now include throttling params
        st.subheader("Google Sheets Config:")
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")
    else:
        st.error("Failed to load critical configuration parts.")

# end of modules/config.py
