# modules/llm_processor.py
# Version 1.1: Adapted to use Google Gemini API.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any

# --- Google Gemini Client Initialization (Implicit) ---
# The genai library configures itself when you set the API key.
# We don't need a specific "get_client" like with OpenAI's library,
# but we do need to configure the API key.

_GEMINI_CONFIGURED = False # Module-level flag to check if configured

def configure_gemini(api_key: Optional[str]) -> bool:
    """
    Configures the Google Generative AI client with the API key.
    Returns True if configuration was successful or already done, False otherwise.
    """
    global _GEMINI_CONFIGURED
    if _GEMINI_CONFIGURED:
        return True
    if not api_key:
        # st.error("Google Gemini API Key not provided for configuration.") # Handled in app.py
        return False
    try:
        genai.configure(api_key=api_key)
        # You can test configuration by trying to list models, but it might incur a small check.
        # For now, we assume configure() is enough or errors will occur on first model use.
        # models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # if not models:
        #     st.error("No Gemini models found supporting 'generateContent'. Check API key and permissions.")
        #     return False
        _GEMINI_CONFIGURED = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Google Gemini client: {e}")
        _GEMINI_CONFIGURED = False
        return False

# --- Core LLM Interaction Function for Gemini ---
def _call_gemini_api(
    model_name: str,
    prompt_parts: list, # Gemini prefers a list of parts for the prompt
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Helper function to make a call to the Google Gemini API.
    """
    if not _GEMINI_CONFIGURED:
        return "Gemini client not configured. API key might be missing or invalid."

    model = genai.GenerativeModel(model_name)

    # Default generation config if none provided
    generation_config = genai.types.GenerationConfig(**(generation_config_args or {
        "temperature": 0.3,
        "max_output_tokens": 512, # Adjust as needed for summary vs extraction
        # "top_p": 1.0,
        # "top_k": 1
    }))

    # Default safety settings (adjust as needed - be aware of blocking)
    # For general web content, some might be too restrictive.
    safety_settings = safety_settings_args or [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    # To be less restrictive (use with caution):
    # safety_settings = [
    #     {"category": c, "threshold": "BLOCK_NONE"} for c in [
    #         "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    #         "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
    #     ]
    # ]


    try:
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False # Set to True if you want to stream tokens
        )
        # Handle potential blocks or empty responses
        if not response.candidates:
            finish_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            reason_message = f"LLM response blocked or empty. Reason: {finish_reason}."
            if response.prompt_feedback and response.prompt_feedback.block_reason_message:
                reason_message += f" Message: {response.prompt_feedback.block_reason_message}"

            # Check safety ratings for more details if blocked
            if response.prompt_feedback and response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings:
                    if rating.blocked:
                        reason_message += f" Blocked by safety category: {rating.category} (Severity: {rating.severity})"

            st.warning(reason_message)
            return reason_message

        return response.text # Accessing response.text directly is common
    except Exception as e:
        # More specific error handling can be added for google.api_core.exceptions
        st.error(f"An error occurred with Google Gemini API: {e} (Type: {e.__class__.__name__})")
        return f"LLM Error: ({e.__class__.__name__})"

# --- Text Truncation (Basic) ---
def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    """
    Basic truncation. Gemini has large context windows, but extremely long text
    can still be an issue for cost or processing time.
    Consider model-specific limits if known (e.g. gemini-1.0-pro is ~30k tokens, 1.5-pro is 1M).
    """
    # Rough character limit, can be more generous for Gemini
    # Actual token limits are much higher for models like gemini-1.5-pro.
    # This max_input_chars is more of a practical limit for this app's current design.
    if len(text) > max_input_chars:
        # st.caption(f"LLM Input: Text truncated from {len(text)} to {max_input_chars} characters for {model_name}.")
        return text[:max_input_chars]
    return text

# --- Public Functions for App ---
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str], # Gemini API key
    model_name: str = "gemini-1.0-pro", # Default Gemini model
    max_input_chars: int = 30000 # Gemini can handle more, e.g. ~7.5k tokens for 1.0 Pro
) -> Optional[str]:
    if not text_content:
        return "No text content provided for summary."
    if not configure_gemini(api_key):
        return "Gemini LLM not configured for summary (API key issue)."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are a helpful assistant designed to provide concise summaries of web page content.\n"
        "Please provide a summary of around 2-5 sentences for the following text. "
        "Focus on the main topics and key takeaways. Avoid introductory phrases like 'This text is about...'.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        "Concise Summary:"
    )
    # Gemini API call (adjust max_output_tokens if needed for summary length)
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 256})


def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str], # Gemini API key
    model_name: str = "gemini-1.0-pro",
    max_input_chars: int = 30000
) -> Optional[str]:
    if not text_content:
        return "No text content provided for extraction."
    if not extraction_query:
        return "No extraction query provided."
    if not configure_gemini(api_key):
        return "Gemini LLM not configured for extraction (API key issue)."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are an intelligent assistant skilled at finding specific information within a given text.\n"
        f"Please analyze the following web page content and extract information related to: '{extraction_query}'.\n"
        "Present your findings clearly and concisely. If the information cannot be found, explicitly state 'Information not found' for that query or specific part.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        f"Extracted information regarding '{extraction_query}':"
    )
    # Gemini API call (adjust max_output_tokens based on expected length of extracted info)
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 512})


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.1)")

    try:
        from config import load_config # Ensure this is modules.config if running from root
        cfg_test = load_config()
        GEMINI_API_KEY_TEST = cfg_test.llm.google_gemini_api_key if cfg_test and cfg_test.llm.provider == "google" else None
        GEMINI_MODEL_TEST = cfg_test.llm.google_gemini_model if cfg_test and cfg_test.llm.provider == "google" else "gemini-1.0-pro"

        if GEMINI_API_KEY_TEST:
            st.success(f"Google Gemini API Key loaded. Model: {GEMINI_MODEL_TEST}")
            if not configure_gemini(GEMINI_API_KEY_TEST):
                st.error("Failed to configure Gemini for testing.")
                GEMINI_API_KEY_TEST = None # Prevent further tests
        else:
            st.error("Could not load Google Gemini API Key from config for testing. Ensure LLM_PROVIDER is 'google' and GOOGLE_GEMINI_API_KEY is in secrets.toml.")
    except Exception as e:
        st.error(f"Error loading/configuring for test: {e}")
        GEMINI_API_KEY_TEST = None


    sample_text = st.text_area("Sample Text for LLM:", """
    The Gemini family of models by Google AI represents a significant leap in multimodal AI capabilities.
    Announced in late 2023, Gemini comes in three sizes: Ultra for highly complex tasks, Pro for a balance
    of performance and scalability, and Nano for on-device efficiency. Gemini 1.0 Pro is widely available
    via the Gemini API and Google AI Studio. A key highlight is its native multimodality, meaning it's
    trained from the ground up to understand and combine different types of information like text, code,
    images, and video. For developer support regarding the API, one might check the official Google Cloud
    documentation or community forums. The development was led by teams at Google DeepMind and Google Research.
    """, height=250)

    if GEMINI_API_KEY_TEST and _GEMINI_CONFIGURED:
        st.subheader("Test Summary Generation (Gemini)")
        if st.button("Generate Summary (Gemini)"):
            with st.spinner("Generating summary with Gemini..."):
                summary = generate_summary(sample_text, GEMINI_API_KEY_TEST, model_name=GEMINI_MODEL_TEST)
            st.markdown("**Summary:**")
            st.write(summary)

        st.subheader("Test Specific Information Extraction (Gemini)")
        query = st.text_input("Extraction Query (Gemini):", "Model sizes and who led development")
        if st.button("Extract Information (Gemini)"):
            if query:
                with st.spinner("Extracting information with Gemini..."):
                    extracted_info = extract_specific_information(sample_text, query, GEMINI_API_KEY_TEST, model_name=GEMINI_MODEL_TEST)
                st.markdown(f"**Extracted Info for '{query}':**")
                st.write(extracted_info)
            else:
                st.warning("Please enter an extraction query.")
    elif not GEMINI_API_KEY_TEST:
        st.warning("Google Gemini API Key not available. Cannot run LLM tests.")
    elif not _GEMINI_CONFIGURED:
        st.warning("Gemini client not configured. Cannot run LLM tests.")

# end of modules/llm_processor.py
