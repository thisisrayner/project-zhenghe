# modules/llm_processor.py
# Version 1.4: Implemented exponential backoff and retries for API calls.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List
import time # Added for sleep in retries
import random # Optional: for adding jitter to backoff

# --- Module-level flag to check if configured ---
_GEMINI_CONFIGURED = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True
    if not api_key:
        return False # Error displayed by app.py
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True

        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            # st.info("Attempting to list available Gemini models...") # Less verbose now
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    st.warning("No Gemini models found supporting 'generateContent'. Check API key/project/services.")
                    return False
            except Exception as e_list_models:
                st.error(f"Error listing Gemini models: {e_list_models}. API key/service issue likely.")
                _AVAILABLE_MODELS_CACHE = []
                _GEMINI_CONFIGURED = False
                return False
        return True
    except Exception as e_configure:
        st.error(f"Failed to configure Google Gemini client with API key: {e_configure}")
        _GEMINI_CONFIGURED = False
        return False

# --- Core LLM Interaction Function for Gemini with Retries ---
def _call_gemini_api(
    model_name: str,
    prompt_parts: list,
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    initial_backoff_seconds: float = 2.0, # Start with 2 seconds
    max_backoff_seconds: float = 60.0 # Cap backoff to 1 minute
) -> Optional[str]:

    if not _GEMINI_CONFIGURED:
        return "Gemini client not configured. API key might be missing, invalid, or model listing failed."

    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE and f"models/{model_name}" not in _AVAILABLE_MODELS_CACHE:
            if not model_name.startswith("models/"):
                prefixed_attempt = f"models/{model_name}"
                if prefixed_attempt in _AVAILABLE_MODELS_CACHE:
                    validated_model_name = prefixed_attempt
                else:
                    # st.error(f"Model '{model_name}' (or '{prefixed_attempt}') not found in available models.")
                    return f"LLM Error: Model '{model_name}' not available."
            elif model_name not in _AVAILABLE_MODELS_CACHE:
                 # st.error(f"Model '{model_name}' not found in available models.")
                 return f"LLM Error: Model '{model_name}' not available."

    try:
        model = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init:
        # st.error(f"Failed to initialize Gemini model '{validated_model_name}': {e_model_init}")
        return f"LLM Error: Could not initialize model '{validated_model_name}': {e_model_init}"

    generation_config = genai.types.GenerationConfig(**(generation_config_args or {
        "temperature": 0.3, "max_output_tokens": 1024,
    }))
    safety_settings = safety_settings_args or [
        {"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]

    current_retry = 0
    current_backoff = initial_backoff_seconds

    while current_retry <= max_retries:
        try:
            response = model.generate_content(
                prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False
            )
            if not response.candidates:
                reason_message = "LLM response was blocked or empty."
                # ... (construct detailed reason_message as before) ...
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message:
                        reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked:
                                reason_message += f" Blocked by safety category: {rating.category}."
                # Content blocks are usually not retried as they are deterministic based on content/safety.
                return reason_message
            return response.text

        except Exception as e:
            error_str = str(e).lower()
            # More specific check for google.api_core.exceptions.ResourceExhausted or similar gRPC errors
            is_rate_limit_error = (
                "429" in error_str or
                "resourceexhausted" in error_str.replace(" ", "") or
                "resource exhausted" in error_str or
                "rate" in error_str and "limit" in error_str or
                (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or # For some gRPC errors
                (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) # gRPC status code 8 is RESOURCE_EXHAUSTED
            )

            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1
                # Log this to the Streamlit UI log via app.py if possible, or print for now
                # For now, we'll return a string that app.py can log or handle
                retry_log_message = f"LLM Rate limit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries})."
                print(retry_log_message) # For console debugging
                # We cannot directly use st.caption here as this module should be UI agnostic
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2 + random.uniform(0, 0.5), max_backoff_seconds) # Exponential backoff with jitter, capped
            else: # Not a retriable error or max retries reached
                return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {e}"
    
    return f"LLM Error: Max retries ({max_retries}) reached for '{validated_model_name}' due to persistent rate limiting."

# --- Text Truncation (Basic) ---
def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    effective_max_chars = max_input_chars
    if len(text) > effective_max_chars:
        return text[:effective_max_chars]
    return text

# --- Public Functions for App ---
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-pro-latest",
    max_input_chars: int = 750000
) -> Optional[str]:
    if not text_content:
        return "No text content provided for summary."
    if not configure_gemini(api_key):
        return "Gemini LLM not configured for summary (API key or model issue)."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are an expert assistant tasked with creating concise, informative summaries of web page content.\n"
        "Analyze the following text and provide a neutral, factual summary of 3-5 sentences. "
        "Focus on the core message, key arguments, and any significant conclusions. "
        "Avoid personal opinions or introductory phrases like 'The text discusses...'.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        "Concise Summary:"
    )
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 300})

def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-pro-latest",
    max_input_chars: int = 750000
) -> Optional[str]:
    if not text_content:
        return "No text content provided for extraction."
    if not extraction_query:
        return "No extraction query provided."
    if not configure_gemini(api_key):
        return "Gemini LLM not configured for extraction (API key or model issue)."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are a highly skilled information extraction assistant.\n"
        f"Carefully review the following web page content and extract information specifically related to: '{extraction_query}'.\n"
        "Present your findings clearly and directly. If the requested information or any part of it cannot be found in the text, "
        "explicitly state 'Information not found for [specific part of query]' or 'The requested information was not found in the provided text'.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        f"Extracted Information for '{extraction_query}':"
    )
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 600})

# --- if __name__ == '__main__': block for testing (kept similar to v1.3) ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.4 - Retries)")
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-pro-latest")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True):
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE:
                 st.write("Available Models (first 5):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True
        else:
            st.error("Failed to configure Gemini for testing.")
    else:
        st.info("Enter Gemini API Key to enable tests.")
    sample_text_content = st.text_area("Sample Text for LLM:", """
    The Alpha Centauri system is the closest star system to our Solar System, located at a distance of 4.37 light-years.
    It consists of three stars: Alpha Centauri A (Rigil Kentaurus), Alpha Centauri B (Toliman), and the closest star,
    Proxima Centauri. Proxima Centauri is a red dwarf and is known to host at least one exoplanet, Proxima Centauri b,
    discovered in 2016. This exoplanet orbits within the habitable zone of Proxima Centauri. Alpha Centauri A and B
    are Sun-like stars orbiting each other closely.
    """, height=200)
    if configured_for_test:
        st.subheader("Test Summary Generation (Gemini)")
        if st.button("Generate Summary (Gemini)"):
            with st.spinner(f"Generating summary with {MODEL_NAME_TEST}..."):
                summary = generate_summary(sample_text_content, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            st.markdown("**Summary:**")
            st.write(summary)
        st.subheader("Test Specific Information Extraction (Gemini)")
        extraction_q = st.text_input("Extraction Query (Gemini):", "Distance to Alpha Centauri and name of Proxima Centauri's exoplanet")
        if st.button("Extract Information (Gemini)"):
            if extraction_q:
                with st.spinner(f"Extracting information with {MODEL_NAME_TEST}..."):
                    extracted_info = extract_specific_information(sample_text_content, extraction_q, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                st.markdown(f"**Extracted Info for '{extraction_q}':**")
                st.write(extracted_info)
            else:
                st.warning("Please enter an extraction query.")
    else:
        st.warning("Gemini not configured for testing.")

# end of modules/llm_processor.py
