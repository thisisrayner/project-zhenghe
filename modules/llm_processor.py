# modules/llm_processor.py
# Version 1.5: Changed default model to models/gemini-1.5-flash-latest for better free tier compatibility.
# Retains exponential backoff and retries for API calls.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List
import time
import random

# --- Module-level flag to check if configured ---
_GEMINI_CONFIGURED = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True
    if not api_key:
        # Error will be displayed by app.py if key is missing for the selected provider
        return False
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True # Mark as configured for API key

        # List and cache available models
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            # st.info("Attempting to list available Gemini models...") # Can be verbose
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    st.warning("No Gemini models found supporting 'generateContent'. Check API key, project, and enabled services.")
                    # _GEMINI_CONFIGURED = False # Keep True if API key is set, model selection is separate
                    return False # Return False if no usable models
                # else:
                    # st.caption(f"Found {len(_AVAILABLE_MODELS_CACHE)} usable Gemini models.")
            except Exception as e_list_models:
                st.error(f"Error listing Gemini models: {e_list_models}. API key might be invalid or service not enabled.")
                _AVAILABLE_MODELS_CACHE = []
                _GEMINI_CONFIGURED = False # If listing fails, consider full config failed
                return False
        return True # API key configured, and models (if checked) are cached
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
    initial_backoff_seconds: float = 5.0, # Increased initial backoff for stricter free tier limits
    max_backoff_seconds: float = 60.0
) -> Optional[str]:

    if not _GEMINI_CONFIGURED:
        return "Gemini client not configured. API key might be missing, invalid, or model listing failed."

    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None: # Only validate if cache is populated
        if model_name not in _AVAILABLE_MODELS_CACHE and f"models/{model_name}" not in _AVAILABLE_MODELS_CACHE:
            if not model_name.startswith("models/"): # If no "models/" prefix, try adding it
                prefixed_attempt = f"models/{model_name}"
                if prefixed_attempt in _AVAILABLE_MODELS_CACHE:
                    validated_model_name = prefixed_attempt
                else:
                    return f"LLM Error: Model '{model_name}' (or '{prefixed_attempt}') not in available: {_AVAILABLE_MODELS_CACHE}"
            elif model_name not in _AVAILABLE_MODELS_CACHE: # Already prefixed, but still not found
                 return f"LLM Error: Model '{model_name}' not in available: {_AVAILABLE_MODELS_CACHE}"
    # If _AVAILABLE_MODELS_CACHE is None, proceed with model_name, hoping it's correct.

    try:
        model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init:
        return f"LLM Error: Could not initialize model '{validated_model_name}': {e_model_init}"

    generation_config = genai.types.GenerationConfig(**(generation_config_args or {
        "temperature": 0.3,
        "max_output_tokens": 1024, # Max tokens for the LLM's response
    }))
    safety_settings = safety_settings_args or [
        {"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in [ # More permissive than MEDIUM
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]

    current_retry = 0
    current_backoff = initial_backoff_seconds

    while current_retry <= max_retries:
        try:
            response = model_obj.generate_content(
                prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False
            )
            if not response.candidates: # Check if response was blocked or empty
                reason_message = "LLM response was blocked or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message:
                         reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    # Check specific safety ratings if blocked
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: # This attribute is present if this category caused a block
                                reason_message += f" Blocked by safety category: {rating.category}."
                return reason_message # Content blocks are usually not retried
            return response.text # Success

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = (
                "429" in error_str or # HTTP Too Many Requests
                "resourceexhausted" in error_str.replace(" ", "") or
                "resource exhausted" in error_str or
                ("rate" in error_str and "limit" in error_str) or
                (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or
                (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) # gRPC status code 8 for RESOURCE_EXHAUSTED
            )

            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1
                retry_log_message = (f"LLM Rate limit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s "
                                     f"(Attempt {current_retry}/{max_retries}). Error: {str(e)[:100]}...")
                print(retry_log_message) # Console log for retries
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds) # Exponential backoff with jitter
            else: # Not a retriable error or max retries reached
                return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}" # Truncate long error messages
    
    return f"LLM Error: Max retries ({max_retries}) reached for '{validated_model_name}' due to persistent rate limiting."

# --- Text Truncation (Basic) ---
def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    # Practical limit for app, actual model context window is much larger for 1.5 series.
    effective_max_chars = max_input_chars
    if len(text) > effective_max_chars:
        # st.caption(f"LLM Input: Text truncated from {len(text)} to {effective_max_chars} chars for {model_name}.") # Can be noisy
        return text[:effective_max_chars]
    return text

# --- Public Functions for App ---
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", # Updated default
    max_input_chars: int = 100000 # Generous for Flash models, but less than 1.5 Pro full potential
) -> Optional[str]:
    if not text_content:
        return "No text content provided for summary."
    if not configure_gemini(api_key): # Ensures API key is set and models were checked
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
    # Adjust max_output_tokens if summaries need to be longer/shorter
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 350})


def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", # Updated default
    max_input_chars: int = 100000
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
    # Adjust max_output_tokens based on expected length of extracted info
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 700})


# --- if __name__ == '__main__': block for testing (remains similar) ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.5 - Flash Default)")
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest") # Updated default for test
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): # Force recheck for direct test
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE:
                 st.write("Available Models (first 5):", _AVAILABLE_MODELS_CACHE[:5]) # Show some listed models
            configured_for_test = True
        else:
            st.error("Failed to configure Gemini for testing. Check API key and console for errors from model listing.")
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
        st.warning("Gemini not configured for testing. Please provide API key.")

# end of modules/llm_processor.py
