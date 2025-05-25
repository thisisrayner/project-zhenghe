# modules/llm_processor.py
# Version 1.2: Targeting Gemini 1.5 Pro, includes model listing debug.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List

# --- Module-level flag to check if configured ---
_GEMINI_CONFIGURED = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None # Cache for available models

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    """
    Configures the Google Generative AI client with the API key.
    Lists available models supporting 'generateContent' if not already cached or forced.
    Returns True if configuration was successful, False otherwise.
    """
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True # Already configured and models checked (unless forced)
    if not api_key:
        st.error("Google Gemini API Key not provided for configuration.")
        return False
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True # Mark as configured even before model check for basic API key validation

        # --- List and Cache Available Models ---
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            st.info("Attempting to list available Gemini models supporting 'generateContent'...")
            print("Attempting to list available Gemini models supporting 'generateContent'...")
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                        print(f"DEBUG: Found model: {m.name}")
                        st.caption(f"DEBUG: Found model: {m.name}") # Show in UI for debugging
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    st.warning("No Gemini models found supporting 'generateContent'. Check API key, project, and enabled services.")
                    # _GEMINI_CONFIGURED = False # Optionally mark as not truly configured if no usable models
                    return False # Cannot proceed if no models
                else:
                    st.success(f"Successfully listed {len(_AVAILABLE_MODELS_CACHE)} usable Gemini model(s).")

            except Exception as e_list_models:
                st.error(f"Error listing Gemini models: {e_list_models}. API key might be invalid or service not enabled.")
                _AVAILABLE_MODELS_CACHE = [] # Ensure it's not None if an error occurred
                _GEMINI_CONFIGURED = False # Configuration failed if models can't be listed
                return False
        return True # Configured (API key set) and models checked/cached

    except Exception as e_configure:
        st.error(f"Failed to configure Google Gemini client with API key: {e_configure}")
        _GEMINI_CONFIGURED = False
        return False

# --- Core LLM Interaction Function for Gemini ---
def _call_gemini_api(
    model_name: str,
    prompt_parts: list,
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not _GEMINI_CONFIGURED:
        return "Gemini client not configured. API key might be missing, invalid, or model listing failed."
    
    # Validate if the requested model_name is in the list of known available models
    if _AVAILABLE_MODELS_CACHE is not None and model_name not in _AVAILABLE_MODELS_CACHE and f"models/{model_name}" not in _AVAILABLE_MODELS_CACHE:
        alt_model_name = f"models/{model_name}" # Sometimes the prefix is needed
        if alt_model_name not in _AVAILABLE_MODELS_CACHE:
            st.error(f"Model '{model_name}' (or '{alt_model_name}') not found in the list of available models for your API key. Available: {_AVAILABLE_MODELS_CACHE}")
            return f"LLM Error: Model '{model_name}' not available."
        model_name = alt_model_name # Use the prefixed name if that's what was found

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e_model_init:
        st.error(f"Failed to initialize Gemini model '{model_name}': {e_model_init}")
        return f"LLM Error: Could not initialize model '{model_name}'."

    generation_config = genai.types.GenerationConfig(**(generation_config_args or {
        "temperature": 0.3, # Lower for more factual, higher for creative
        "max_output_tokens": 1024, # Gemini 1.5 can output more; adjust as needed
    }))

    # Adjust safety settings - for web content, you might need to be less restrictive
    # Default is often BLOCK_MEDIUM_AND_ABOVE for all.
    # BLOCK_ONLY_HIGH might be a better starting point for general web summarization.
    safety_settings = safety_settings_args or [
        {"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]
    # To be very permissive (USE WITH EXTREME CAUTION AND UNDERSTAND RISKS):
    # safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in [...] ]

    try:
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False
        )

        if not response.candidates:
            reason_message = "LLM response was blocked or empty."
            if response.prompt_feedback:
                reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                if response.prompt_feedback.block_reason_message:
                     reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                if response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        if rating.blocked: # This attribute tells if this specific category caused a block
                            reason_message += f" Blocked by safety category: {rating.category}."
            st.warning(reason_message)
            return reason_message

        return response.text
    except Exception as e:
        st.error(f"Error during Gemini API call for model '{model_name}': {e} (Type: {e.__class__.__name__})")
        return f"LLM Error during API call ({e.__class__.__name__})"

# --- Text Truncation (Basic) ---
def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    # Gemini 1.5 Pro has a 1M token context. This char limit is a practical app limit.
    # 1M tokens is roughly 4M characters.
    # A more robust solution would use client-side token counting if available for Gemini.
    effective_max_chars = max_input_chars
    # if "1.5-pro" in model_name: # Example: be more generous for 1.5 pro
    #     effective_max_chars = min(max_input_chars, 800000) # Cap at ~200k tokens worth of chars

    if len(text) > effective_max_chars:
        st.caption(f"LLM Input: Text truncated from {len(text)} to {effective_max_chars} characters for {model_name}.")
        return text[:effective_max_chars]
    return text

# --- Public Functions for App ---
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "gemini-1.5-pro-latest", # Default to Gemini 1.5 Pro
    max_input_chars: int = 750000 # Generous limit for Gemini 1.5 Pro (approx <200k tokens)
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
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 300})


def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "gemini-1.5-pro-latest", # Default to Gemini 1.5 Pro
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


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.2 - 1.5 Pro Focus)")

    # This test part assumes your config.py is in the same directory or Python path
    # For modular structure, it's better to run app.py and test through it.
    # However, for isolated module testing:
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing (or ensure it's in secrets if config load works):", type="password")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "gemini-1.5-pro-latest") # Or "gemini-pro"

    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): # Force model check for direct test
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}. Available models (check console/UI for DEBUG): {_AVAILABLE_MODELS_CACHE}")
            configured_for_test = True
        else:
            st.error("Failed to configure Gemini for testing. Check API key and console for errors.")
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
