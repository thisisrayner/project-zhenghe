# modules/config.py
# Version 1.1: Added Google Gemini configuration.

import streamlit as st
from dataclasses import dataclass, field
import json

# --- Configuration Classes ---
@dataclass
class GoogleSearchConfig: # Renamed for clarity
    api_key: str | None = None
    cse_id: str | None = None

@dataclass
class LLMConfig: # Generic LLM Config, can hold specific provider configs
    provider: str = "openai" # Default or can be set from secrets
    # OpenAI specific (can be None if not using OpenAI)
    openai_api_key: str | None = None
    openai_model_summarize: str = "gpt-3.5-turbo"
    openai_model_extract: str = "gpt-3.5-turbo"
    # Google Gemini specific (can be None if not using Gemini)
    google_gemini_api_key: str | None = None
    google_gemini_model: str = "gemini-1.0-pro" # Or "gemini-1.5-pro-latest" etc.

    max_input_chars: int = 12000 # Increased default for potentially larger Gemini context

@dataclass
class GoogleSheetsConfig:
    service_account_info: dict | None = field(default_factory=dict)
    spreadsheet_name: str | None = None
    worksheet_name: str = "Sheet1"

@dataclass
class AppConfig:
    google_search: GoogleSearchConfig = field(default_factory=GoogleSearchConfig) # Updated name
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
        st.error(f"Missing Google Search secret: {e}. Please check .streamlit/secrets.toml")
        essential_secrets_loaded = False

    # --- LLM Configuration ---
    # Determine LLM provider (default to openai if not specified)
    # You could add a secret like LLM_PROVIDER="google" or LLM_PROVIDER="openai"
    cfg.llm.provider = st.secrets.get("LLM_PROVIDER", "google").lower() # Default to google now

    if cfg.llm.provider == "openai":
        try:
            cfg.llm.openai_api_key = st.secrets.get("OPENAI_API_KEY")
            if not cfg.llm.openai_api_key:
                st.warning("OpenAI selected as LLM provider, but OPENAI_API_KEY is missing. LLM features might fail.")
            # Load OpenAI model names from secrets if desired
            cfg.llm.openai_model_summarize = st.secrets.get("OPENAI_MODEL_SUMMARIZE", "gpt-3.5-turbo")
            cfg.llm.openai_model_extract = st.secrets.get("OPENAI_MODEL_EXTRACT", "gpt-3.5-turbo")
        except Exception as e:
            st.warning(f"Could not load OpenAI config: {e}")
    elif cfg.llm.provider == "google":
        try:
            cfg.llm.google_gemini_api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
            if not cfg.llm.google_gemini_api_key:
                st.error("Google Gemini selected/defaulted as LLM provider, but GOOGLE_GEMINI_API_KEY is missing. LLM features will be disabled.")
                # No LLM functionality if key is missing for selected provider
            cfg.llm.google_gemini_model = st.secrets.get("GOOGLE_GEMINI_MODEL", "gemini-1.0-pro") # Or gemini-pro
        except Exception as e:
            st.error(f"Could not load Google Gemini config: {e}")
    else:
        st.warning(f"Unsupported LLM_PROVIDER: '{cfg.llm.provider}'. LLM features disabled.")


    # Max input chars for LLM (can be general or provider-specific if needed)
    cfg.llm.max_input_chars = int(st.secrets.get("LLM_MAX_INPUT_CHARS", "12000"))


    # --- Google Sheets Config (Optional) ---
    try:
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME")
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", "Sheet1")
    except Exception as e:
        st.warning(f"Could not load Google Sheets config (optional): {e}")


    if not essential_secrets_loaded:
        # st.error("Essential secrets for core functionality are missing. Application might not work correctly.")
        return None # Or handle this more gracefully in app.py

    return cfg

# ... (if __name__ == '__main__': block for testing - update to show new LLMConfig) ...
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Config Loader Test (v1.1)")
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.json(vars(loaded_cfg.google_search))
        st.subheader("LLM Config:")
        st.json(vars(loaded_cfg.llm)) # Display the whole LLM config object
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")
    else:
        st.error("Failed to load configuration.")
# end of modules/config.py
