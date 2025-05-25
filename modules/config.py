# modules/config.py
# Version 1.2: Updated default Gemini model and refined LLM config loading.

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
    # OpenAI specific
    openai_api_key: str | None = None
    openai_model_summarize: str = "gpt-3.5-turbo"
    openai_model_extract: str = "gpt-3.5-turbo"
    # Google Gemini specific
    google_gemini_api_key: str | None = None
    google_gemini_model: str = "models/gemini-1.5-flash-latest" # Updated default

    max_input_chars: int = 750000 # Generous default for Gemini 1.5 Pro

@dataclass
class GoogleSheetsConfig:
    service_account_info: dict | None = field(default_factory=dict)
    spreadsheet_name: str | None = None
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
    cfg.llm.provider = st.secrets.get("LLM_PROVIDER", "google").lower() # Default to google

    if cfg.llm.provider == "openai":
        try:
            cfg.llm.openai_api_key = st.secrets.get("OPENAI_API_KEY")
            cfg.llm.openai_model_summarize = st.secrets.get("OPENAI_MODEL_SUMMARIZE", cfg.llm.openai_model_summarize)
            cfg.llm.openai_model_extract = st.secrets.get("OPENAI_MODEL_EXTRACT", cfg.llm.openai_model_extract)
            if not cfg.llm.openai_api_key:
                st.caption("OpenAI selected, but API Key missing. LLM features disabled.")
        except Exception as e:
            st.warning(f"Could not load OpenAI config: {e}")
    elif cfg.llm.provider == "google":
        try:
            cfg.llm.google_gemini_api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
            # Load the model name from secrets, fallback to the class default if not in secrets
            cfg.llm.google_gemini_model = st.secrets.get("GOOGLE_GEMINI_MODEL", cfg.llm.google_gemini_model)
            if not cfg.llm.google_gemini_api_key:
                st.caption("Google Gemini selected, but API Key missing. LLM features disabled.")
        except Exception as e:
            st.error(f"Could not load Google Gemini config: {e}")
    else:
        st.warning(f"Unsupported LLM_PROVIDER: '{cfg.llm.provider}'. LLM features disabled.")

    cfg.llm.max_input_chars = int(st.secrets.get("LLM_MAX_INPUT_CHARS", cfg.llm.max_input_chars))

    # --- Google Sheets Config (Optional) ---
    try:
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME")
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", cfg.gsheets.worksheet_name)
    except Exception as e:
        st.warning(f"Could not load Google Sheets config (optional): {e}")

    if not essential_secrets_loaded:
        return None
    return cfg

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Config Loader Test (v1.2)")
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.json(vars(loaded_cfg.google_search))
        st.subheader("LLM Config:")
        st.json(vars(loaded_cfg.llm))
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")
    else:
        st.error("Failed to load critical configuration.")
# end of modules/config.py
