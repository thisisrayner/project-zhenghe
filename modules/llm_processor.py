# modules/llm_processor.py
# Version 1.7: Enhanced docstrings, type hinting, and comments.
# Default model remains gemini-1.5-flash-latest.

"""
Handles interactions with Large Language Models (LLMs) for text processing.

Currently configured to primarily use Google's Gemini models via the
`google-generativeai` library. It includes functionalities for:
- Configuring the LLM client.
- Generating summaries of text content.
- Extracting specific information based on user queries.
- Generating a consolidated overview from multiple text summaries.
- Implements exponential backoff and retries for API calls to handle rate limits.
"""

import google.generativeai as genai
import streamlit as st # Used for @st.cache_resource and error display in direct test
from typing import Optional, Dict, Any, List
import time
import random

# --- Module-level state for Gemini configuration ---
_GEMINI_CONFIGURED: bool = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    """
    Configures the Google Generative AI client with the provided API key.

    It also lists available models that support 'generateContent' and caches them.
    This function should be called before any Gemini API interactions.

    Args:
        api_key: The API key for Google Gemini.
        force_recheck_models: If True, forces a re-fetch of available models
                              even if they were previously cached.

    Returns:
        True if configuration was successful (API key set and usable models found),
        False otherwise.
    """
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    
    # If already configured and models are cached (and not forced to recheck), return True.
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True
    
    if not api_key:
        # Error reporting for missing API key is handled by the calling function (e.g., in app.py)
        return False
    
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True # Mark API key as configured

        # List and cache available models if not already done or if forced
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                _AVAILABLE_MODELS_CACHE = current_available_models
                
                if not _AVAILABLE_MODELS_CACHE:
                    st.warning("LLM_PROCESSOR: No Gemini models found supporting 'generateContent'. "
                               "Check API key permissions, GCP project, and enabled GenAI/VertexAI services.")
                    # If no models, true configuration isn't complete for practical use.
                    # However, the API key itself might be valid. For now, let's return False if no models.
                    return False 
            except Exception as e_list_models:
                st.error(f"LLM_PROCESSOR: Error listing Gemini models: {e_list_models}. "
                           "API key might be invalid or service/project not correctly set up.")
                _AVAILABLE_MODELS_CACHE = [] # Reset cache on error
                _GEMINI_CONFIGURED = False # Full configuration failed
                return False
        return True # Successfully configured API key and model list (if checked)
    except Exception as e_configure:
        st.error(f"LLM_PROCESSOR: Failed to configure Google Gemini client with API key: {e_configure}")
        _GEMINI_CONFIGURED = False
        return False

def _call_gemini_api(
    model_name: str,
    prompt_parts: List[str], # Gemini API prefers prompts as a list of strings or Parts
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[List[Dict[str, Any]]] = None, # Safety settings list of dicts
    max_retries: int = 3,
    initial_backoff_seconds: float = 5.0,
    max_backoff_seconds: float = 60.0
) -> Optional[str]:
    """
    Internal helper function to make a call to the Google Gemini API (generate_content).
    Includes model name validation and exponential backoff retry logic for rate limits.

    Args:
        model_name: The name of the Gemini model to use (e.g., "models/gemini-1.5-flash-latest").
        prompt_parts: A list containing the prompt content. For simple text prompts,
                      this will be a list with a single string element.
        generation_config_args: Dictionary of arguments for genai.types.GenerationConfig.
        safety_settings_args: List of safety setting dictionaries.
        max_retries: Maximum number of times to retry on rate limit errors.
        initial_backoff_seconds: Initial delay in seconds before the first retry.
        max_backoff_seconds: Maximum delay in seconds for a single retry.

    Returns:
        The text response from the LLM, an error message string if an error occurred,
        or a message indicating a content block. Returns None if client is not configured.
    """
    if not _GEMINI_CONFIGURED:
        return "LLM_PROCESSOR Error: Gemini client not configured (API key missing or invalid)."

    validated_model_name = model_name
    # Validate model_name against cached list if available
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE:
            prefixed_attempt = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            if prefixed_attempt in _AVAILABLE_MODELS_CACHE:
                validated_model_name = prefixed_attempt
            else:
                return f"LLM_PROCESSOR Error: Model '{model_name}' (or '{prefixed_attempt}') not found in available models for your API key. Available: {_AVAILABLE_MODELS_CACHE}"
    
    try:
        # Initialize the generative model object.
        model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init:
        return f"LLM_PROCESSOR Error: Could not initialize model '{validated_model_name}': {e_model_init}"

    # Prepare generation configuration
    effective_gen_config_params = {"temperature": 0.3, "max_output_tokens": 1024} # Sensible defaults
    if generation_config_args:
        effective_gen_config_params.update(generation_config_args)
    generation_config = genai.types.GenerationConfig(**effective_gen_config_params)

    # Prepare safety settings
    # BLOCK_ONLY_HIGH is generally safer for diverse web content than BLOCK_MEDIUM_AND_ABOVE.
    # Adjust thresholds based on content sensitivity and desired filtering.
    safety_settings = safety_settings_args or [
        {"category": cat, "threshold": "BLOCK_ONLY_HIGH"} for cat in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]

    current_retry = 0
    current_backoff = initial_backoff_seconds

    while current_retry <= max_retries:
        try:
            # Make the API call to generate content.
            response = model_obj.generate_content(
                prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False # Keep stream False for simpler response handling
            )

            # Check for content blocks or empty responses from the API.
            if not response.candidates:
                reason_message = "LLM_PROCESSOR: Response was blocked by API or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message:
                         reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    # Log specific safety categories that caused a block
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: # `blocked` attribute indicates if this category caused the block
                                reason_message += f" Blocked by safety category: {rating.category} (Severity: {rating.probability.name})." # Use probability name for more info
                print(reason_message) # Log to console for server-side debugging
                return reason_message # Return the block message to the caller
            
            # If successful, return the generated text.
            return response.text.strip() if response.text else "LLM_PROCESSOR: Received an empty text response."

        except Exception as e:
            error_str = str(e).lower()
            # Check for common rate limit / resource exhaustion errors.
            is_rate_limit_error = (
                "429" in error_str or # HTTP status code for Too Many Requests
                "resourceexhausted" in error_str.replace(" ", "") or
                "resource exhausted" in error_str or
                ("rate" in error_str and "limit" in error_str) or
                (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or # Some gRPC errors
                (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) # gRPC status code 8 is RESOURCE_EXHAUSTED
            )

            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1
                retry_log_message = (f"LLM_PROCESSOR: Rate limit hit for '{validated_model_name}'. "
                                     f"Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). "
                                     f"Error: {str(e)[:100]}...") # Log snippet of error
                print(retry_log_message) # Log retries to console
                time.sleep(current_backoff)
                # Exponential backoff with jitter
                current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: # Not a retriable error, or max retries have been reached.
                print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}")
                return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}"
    
    # If loop finishes due to max_retries
    return f"LLM_PROCESSOR Error: Max retries ({max_retries}) reached for '{validated_model_name}' due to persistent rate limiting."


def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    """
    Truncates text to a specified maximum number of characters.
    This is a practical limit for the application, not necessarily the model's absolute token limit.

    Args:
        text: The input text string.
        model_name: The name of the model (for potential future model-specific truncation logic).
        max_input_chars: The maximum number of characters to allow.

    Returns:
        The truncated text, or the original text if it's within the limit.
    """
    # For Gemini 1.5 models with 1M token context, this char limit is very conservative.
    # It's more about managing processing time and cost for this specific app.
    if len(text) > max_input_chars:
        # Consider logging this truncation event if it's important for debugging.
        # print(f"DEBUG (_truncate_text_for_gemini): Text truncated from {len(text)} to {max_input_chars} chars for {model_name}.")
        return text[:max_input_chars]
    return text

def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000 # Approx. 25k tokens, well within Flash's context
) -> Optional[str]:
    """
    Generates a narrative summary for the given text content using Gemini.

    Args:
        text_content: The main textual content extracted from a web page.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use.
        max_input_chars: Maximum characters of `text_content` to send to the LLM.

    Returns:
        A string containing the generated summary, or an error/status message string.
    """
    if not text_content:
        return "LLM_PROCESSOR: No text content provided for summary."
    if not configure_gemini(api_key): # Ensures API key is set and client is configured
        return "LLM_PROCESSOR Error: Gemini LLM not configured for summary (API key or model issue)."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    
    # Refined prompt for better, more direct summaries
    prompt = (
        "You are an expert assistant specializing in creating detailed and insightful summaries of web page content.\n"
        "Analyze the following text and provide a comprehensive summary of approximately 4-6 substantial sentences (or 2-3 short paragraphs if the content is rich). "
        "Your summary should capture the core message, key arguments, supporting details, and any significant conclusions or implications. "
        "Maintain a neutral and factual tone. Avoid introductory phrases like 'This text discusses...' or 'The summary of the text is...'. Go directly into the summary content.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        "Detailed Summary:"
    )
    return _call_gemini_api(
        model_name,
        [prompt], # Gemini API expects a list for prompt_parts
        generation_config_args={"max_output_tokens": 512} # Allow for a reasonably detailed summary
    )

def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000
) -> Optional[str]:
    """
    Extracts specific information based on a user's query from the text content using Gemini.

    Args:
        text_content: The main textual content.
        extraction_query: The user's question or description of information to extract.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use.
        max_input_chars: Maximum characters of `text_content` to send.

    Returns:
        A string containing the extracted information, or an error/status message.
    """
    if not text_content:
        return "LLM_PROCESSOR: No text content provided for extraction."
    if not extraction_query: # Should be validated by caller (app.py) too
        return "LLM_PROCESSOR: No extraction query provided."
    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini LLM not configured for extraction."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are a highly skilled information extraction assistant.\n"
        f"Carefully review the following web page content. Your task is to extract information specifically related to: '{extraction_query}'.\n"
        "Present your findings comprehensively and clearly. If the requested information or any part of it cannot be found in the text, "
        "explicitly state 'Information not found for [specific part of query]' or 'The requested information was not found in the provided text'. "
        "If multiple pieces of information are requested, address each one directly.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        f"Comprehensive Extracted Information regarding '{extraction_query}':"
    )
    return _call_gemini_api(
        model_name,
        [prompt],
        generation_config_args={"max_output_tokens": 700} # Allow more tokens for potentially detailed extractions
    )

def generate_consolidated_summary(
    summaries: List[Optional[str]], # List of individual summaries or relevant text snippets
    topic_context: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", # Flash for cost, Pro for potentially better synthesis
    max_input_chars: int = 150000 # Max chars for the combined input of summaries
) -> Optional[str]:
    """
    Generates a consolidated overview from a list of individual summaries or text snippets.

    Args:
        summaries: A list of strings, where each string is an individual summary or relevant text.
        topic_context: A string describing the overall topic or keywords these summaries relate to.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use for consolidation.
        max_input_chars: Maximum characters for the combined input of all summaries.

    Returns:
        A string containing the consolidated overview, or an error/status message.
    """
    if not summaries:
        return "LLM_PROCESSOR: No individual summaries provided for consolidation."
    
    # Filter out None, empty, or known error/placeholder strings from individual LLM outputs
    valid_texts_for_consolidation = [
        s for s in summaries if s and
        not s.lower().startswith("llm error") and
        not s.lower().startswith("no text content") and
        not s.lower().startswith("please provide the web page content") # Filter out placeholders
    ]
    if not valid_texts_for_consolidation:
        return "LLM_PROCESSOR: No valid individual LLM outputs available to consolidate."

    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini LLM not configured for consolidated summary."

    # Format the input for the consolidation prompt
    summary_entries = []
    for i, s_text in enumerate(valid_texts_for_consolidation):
        summary_entries.append(f"Source Document {i+1} Content:\n{s_text}")
    combined_texts = "\n\n---\n\n".join(summary_entries)

    truncated_combined_text = _truncate_text_for_gemini(combined_texts, model_name, max_input_chars)
    
    prompt = (
        f"You are an expert analyst tasked with synthesizing information from multiple text sources related to '{topic_context}'.\n"
        "The following are several pieces of content (summaries or extractions) from different web pages. Your objective is to create a single, "
        "coherent consolidated overview. This overview should:\n"
        "1. Identify and clearly state the main recurring themes or central topics present across the provided texts.\n"
        "2. Synthesize key pieces of information and arguments into a cohesive narrative that reflects the collective understanding from these sources.\n"
        "3. If applicable, highlight any notable patterns, unique insights, supporting evidence, or significant discrepancies/contradictions found across the different sources.\n"
        "4. The final output should be a well-structured and comprehensive overview, not merely a list of points. Aim for a few informative paragraphs.\n\n"
        "--- PROVIDED TEXTS START ---\n"
        f"{truncated_combined_text}\n"
        "--- PROVIDED TEXTS END ---\n\n"
        f"Consolidated Overview regarding '{topic_context}':"
    )
    return _call_gemini_api(
        model_name,
        [prompt],
        generation_config_args={"max_output_tokens": 800} # Allow more tokens for a detailed consolidated summary
    )

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.7 - Docs & Hints)")
    # ... (Test block from v1.6.1 or v1.5.1, can be kept or simplified) ...
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): # Force recheck for direct test
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE:
                 st.write("Available Models (first 5 from cache):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True
        else:
            st.error("Failed to configure Gemini for testing. Check API key and console for errors from model listing.")
    else:
        st.info("Enter Gemini API Key to enable tests.")

    sample_text_content_1 = st.text_area("Sample Text 1 for LLM:", "The sky is blue due to Rayleigh scattering. This scattering affects electromagnetic radiation whose wavelength is longer than the scattering particles.", height=100)
    sample_text_content_2 = st.text_area("Sample Text 2 for LLM:", "Oceans appear blue because water absorbs red light more than blue. Blue light penetrates deeper.", height=100)
    
    if configured_for_test:
        st.subheader("Test Individual Summary")
        if st.button("Test Summary for Text 1"):
             with st.spinner(f"Generating summary with {MODEL_NAME_TEST}..."):
                summary1 = generate_summary(sample_text_content_1, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
             st.markdown("**Summary 1:**"); st.write(summary1)

        st.subheader("Test Consolidation")
        if st.button("Test Consolidation of Summaries"):
            # For a proper test, generate summaries first
            s1 = generate_summary(sample_text_content_1, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            s2 = generate_summary(sample_text_content_2, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            st.write("Individual Summary 1 for consolidation:", s1)
            st.write("Individual Summary 2 for consolidation:", s2)
            if s1 and s2 and not s1.startswith("LLM") and not s2.startswith("LLM"): # Check they are actual summaries
                with st.spinner("Generating consolidated summary..."):
                    consolidated = generate_consolidated_summary(
                        [s1, s2], 
                        "Color of Sky and Ocean", 
                        GEMINI_API_KEY_TEST, 
                        model_name=MODEL_NAME_TEST
                    )
                st.markdown("**Consolidated Summary:**"); st.write(consolidated)
            else:
                st.warning("Could not generate valid individual summaries to test consolidation.")
    else:
        st.warning("Gemini not configured for testing. Please provide API key.")
# end of modules/llm_processor.py
