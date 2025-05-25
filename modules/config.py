# modules/config.py

import streamlit as st
from dataclasses import dataclass, field
import json # For parsing JSON string from secrets if needed

# --- Configuration Classes ---
@dataclass
class GoogleConfig:
    api_key: str | None = None
    cse_id: str | None = None

@dataclass
class OpenAIConfig:
    api_key: str | None = None
    default_model_summarize: str = "gpt-3.5-turbo"
    default_model_extract: str = "gpt-3.5-turbo"
    max_tokens_for_llm: int = 3000 # Rough character limit before token counting

@dataclass
class GoogleSheetsConfig:
    # Option 1: Store entire service account JSON as a string in secrets.toml
    # service_account_json_str: str | None = None

    # Option 2: Store individual fields from the service account JSON
    # (as shown in the README.md secrets.toml example)
    service_account_info: dict | None = field(default_factory=dict)
    spreadsheet_name: str | None = None
    worksheet_name: str = "Sheet1" # Default worksheet name

@dataclass
class AppConfig:
    google: GoogleConfig = field(default_factory=GoogleConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    gsheets: GoogleSheetsConfig = field(default_factory=GoogleSheetsConfig)
    num_results_per_keyword_default: int = 3
    # Add other general app settings here

# --- Main Config Loading Function ---
@st.cache_resource # Cache the loaded configuration object
def load_config() -> AppConfig | None:
    """
    Loads configuration from Streamlit secrets.
    Returns an AppConfig object or None if critical secrets are missing.
    """
    cfg = AppConfig()
    secrets_loaded = True

    # --- Google Search Config ---
    try:
        cfg.google.api_key = st.secrets["GOOGLE_API_KEY"]
        cfg.google.cse_id = st.secrets["CSE_ID"]
    except KeyError as e:
        st.error(f"Missing Google Search secret: {e}. Please check .streamlit/secrets.toml")
        secrets_loaded = False # Mark as partially failed, but continue to load others

    # --- OpenAI Config (Optional) ---
    try:
        cfg.openai.api_key = st.secrets.get("OPENAI_API_KEY") # .get() makes it optional
        # You could also load model names from secrets if you want them configurable
    except Exception as e: # Catch broader exceptions if any during optional loading
        st.warning(f"Could not load OpenAI config (optional): {e}")


    # --- Google Sheets Config (Optional) ---
    try:
        # This assumes you've structured your secrets.toml for 'gcp_service_account' as a table
        # matching the keys of a service account JSON.
        if "gcp_service_account" in st.secrets:
            cfg.gsheets.service_account_info = dict(st.secrets["gcp_service_account"])
        else:
            # If you stored the whole JSON as a string under a key like "GCP_SERVICE_ACCOUNT_JSON_STR"
            # json_str = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON_STR")
            # if json_str:
            #     cfg.gsheets.service_account_info = json.loads(json_str)
            pass # It's optional, so no error if not found

        cfg.gsheets.spreadsheet_name = st.secrets.get("SPREADSHEET_NAME")
        cfg.gsheets.worksheet_name = st.secrets.get("WORKSHEET_NAME", "Sheet1") # Default to "Sheet1"
    except json.JSONDecodeError as e:
        st.warning(f"Error parsing Google Sheets service account JSON from secrets: {e}")
    except KeyError as e:
        st.warning(f"Missing Google Sheets configuration key in secrets (optional): {e}")
    except Exception as e:
        st.warning(f"Could not load Google Sheets config (optional): {e}")


    # Return None if essential secrets (like Google Search) are missing,
    # or handle this more gracefully in app.py
    if not cfg.google.api_key or not cfg.google.cse_id:
        st.error("Essential Google Search API Key or CSE ID is missing. Application cannot proceed.")
        return None # Or raise an exception

    return cfg

if __name__ == '__main__':
    # For testing this module directly (streamlit run modules/config.py)
    # Note: st.secrets won't work this way unless you deploy a dummy app
    # or mock st.secrets
    st.set_page_config(layout="wide")
    st.title("Config Loader Test")
    loaded_cfg = load_config()
    if loaded_cfg:
        st.success("Configuration loaded successfully!")
        st.json(vars(loaded_cfg.google))
        st.json(vars(loaded_cfg.openai))
        st.json(vars(loaded_cfg.gsheets))
        st.write(f"Default results per keyword: {loaded_cfg.num_results_per_keyword_default}")

        # Test individual secret access for debugging
        st.write("Attempting to access GOOGLE_API_KEY directly from secrets:")
        try:
            st.write(st.secrets["GOOGLE_API_KEY"])
        except Exception as e:
            st.error(f"Failed: {e}")

    else:
        st.error("Failed to load configuration.")
