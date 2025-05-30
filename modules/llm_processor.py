# modules/llm_processor.py
# Version 1.7.1: Added relevancy scoring to extract_specific_information.
# Default model remains gemini-1.5-flash-latest.

"""
Handles interactions with Large Language Models (LLMs) for text processing.

Currently configured to primarily use Google's Gemini models via the
`google-generativeai` library. It includes functionalities for:
- Configuring the LLM client.
- Generating summaries of text content.
- Extracting specific information based on user queries, including a relevancy score.
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

            if not response.candidates:
                reason_message = "LLM_PROCESSOR: Response was blocked by API or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message:
                         reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: 
                                reason_message += f" Blocked by safety category: {rating.category} (Severity: {rating.probability.name})."
                print(reason_message) 
                return reason_message 
            
            return response.text.strip() if response.text else "LLM_PROCESSOR: Received an empty text response."

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = (
                "429" in error_str or 
                "resourceexhausted" in error_str.replace(" ", "") or
                "resource exhausted" in error_str or
                ("rate" in error_str and "limit" in error_str) or
                (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or 
                (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) 
            )

            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1
                retry_log_message = (f"LLM_PROCESSOR: Rate limit hit for '{validated_model_name}'. "
                                     f"Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). "
                                     f"Error: {str(e)[:100]}...") 
                print(retry_log_message) 
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: 
                print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}")
                return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}"
    
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
    if len(text) > max_input_chars:
        return text[:max_input_chars]
    return text

def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000 
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
    if not configure_gemini(api_key): 
        return "LLM_PROCESSOR Error: Gemini LLM not configured for summary (API key or model issue)."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    
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
        [prompt], 
        generation_config_args={"max_output_tokens": 512} 
    )

def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000
) -> Optional[str]:
    """
    Extracts specific information based on a user's query from the text content using Gemini
    and includes a relevancy score as the first line of the output.

    The relevancy score is determined based on the number of distinct pieces of information
    found related to the extraction_query:
    - 5/5: 5 or more pieces of information.
    - 4/5: Exactly 4 pieces of information.
    - 3/5: 1, 2, or 3 pieces of information.
    - 1/5: No information found.

    Args:
        text_content: The main textual content.
        extraction_query: The user's question or description of information to extract.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use.
        max_input_chars: Maximum characters of `text_content` to send.

    Returns:
        A string starting with "Relevancy Score: X/5\n" followed by the extracted
        information, or an error/status message.
    """
    if not text_content:
        return "LLM_PROCESSOR: No text content provided for extraction."
    if not extraction_query: 
        return "LLM_PROCESSOR: No extraction query provided."
    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini LLM not configured for extraction."

    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    
    prompt = (
        "You are a highly skilled information extraction and relevancy scoring assistant.\n"
        f"Your primary task is to analyze the following web page content based on the user's query: '{extraction_query}'.\n\n"
        "First, you MUST determine a relevancy score by counting how many distinct pieces of information you find that are directly related to the query '{extraction_query}'. Follow these scoring guidelines precisely:\n"
        "- **Relevancy Score: 5/5** - Awarded if you find 5 or more distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 4/5** - Awarded if you find exactly 4 distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 3/5** - Awarded if you find 1, 2, or 3 distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 1/5** - Awarded if you find no distinct pieces of information directly related to '{extraction_query}'.\n\n"
        "The VERY FIRST line of your response MUST be the relevancy score in the format 'Relevancy Score: X/5'.\n\n"
        "After the relevancy score, starting on a new line, present the extracted information comprehensively and clearly. "
        "List or detail each piece of information you found that contributed to the score. "
        "If, after thorough review, no relevant information is found (leading to a 1/5 score), "
        "after stating 'Relevancy Score: 1/5', on the next line you should explicitly state 'No distinct pieces of information related to \"{extraction_query}\" were found in the provided text'.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        f"Response (starting with Relevancy Score) regarding '{extraction_query}':"
    )
    
    return _call_gemini_api(
        model_name,
        [prompt], 
        generation_config_args={"max_output_tokens": 750} 
    )

def generate_consolidated_summary(
    summaries: List[Optional[str]], 
    topic_context: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", 
    max_input_chars: int = 150000 
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
    
    valid_texts_for_consolidation = [
        s for s in summaries if s and
        not s.lower().startswith("llm error") and
        not s.lower().startswith("no text content") and
        not s.lower().startswith("please provide the web page content") 
    ]
    if not valid_texts_for_consolidation:
        return "LLM_PROCESSOR: No valid individual LLM outputs available to consolidate."

    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini LLM not configured for consolidated summary."

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
        generation_config_args={"max_output_tokens": 800} 
    )

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.7.1 - Relevancy Score)") # Updated title
    
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password", key="api_key_test")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest", key="model_name_test")
    
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): 
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE:
                 st.write("Available Models (first 5 from cache):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True
        else:
            st.error("Failed to configure Gemini for testing. Check API key and console for errors from model listing.")
    else:
        st.info("Enter Gemini API Key to enable tests.")

    st.subheader("Test Individual Summary")
    sample_text_summary = st.text_area("Sample Text for Summary:", 
                                       "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
                                       "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
                                       "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.", 
                                       height=150, key="sample_summary_text")
    if configured_for_test:
        if st.button("Test Summary Generation", key="test_summary_button"):
             with st.spinner(f"Generating summary with {MODEL_NAME_TEST}..."):
                summary_output = generate_summary(sample_text_summary, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
             st.markdown("**Generated Summary:**"); st.write(summary_output)

    st.subheader("Test Specific Information Extraction (with Relevancy Score)")
    sample_text_extraction = st.text_area("Sample Text for Extraction:", 
                                          "Quantum computing harnesses the principles of quantum mechanics to solve complex problems beyond the reach of classical computers. Key concepts include qubits, superposition, and entanglement. Qubits, unlike classical bits which are 0 or 1, can exist in multiple states simultaneously due to superposition. Entanglement links qubits in such a way that their fates are intertwined, regardless of the distance separating them. Potential applications include drug discovery, materials science, financial modeling, and cryptography. Major challenges involve qubit stability (decoherence) and error correction. Several companies like Google, IBM, and Microsoft are actively researching and building quantum processors.", 
                                          height=200, key="sample_extraction_text")
    extraction_query_test = st.text_input("Extraction Query (e.g., 'key concepts and challenges'):", "key concepts, challenges, and major companies", key="extraction_query_input")

    if configured_for_test:
        if st.button("Test Extraction & Relevancy Scoring", key="test_extraction_button"):
            if not extraction_query_test:
                st.warning("Please enter an extraction query.")
            else:
                with st.spinner(f"Extracting info with {MODEL_NAME_TEST}..."):
                    extraction_output = extract_specific_information(
                        sample_text_extraction, 
                        extraction_query_test, 
                        GEMINI_API_KEY_TEST, 
                        model_name=MODEL_NAME_TEST
                    )
                st.markdown("**Extraction Output (with Relevancy Score):**"); st.text(extraction_output) # Using st.text to preserve newlines accurately

    st.subheader("Test Consolidation")
    consolidate_text_1 = st.text_area("Text 1 for Consolidation:", 
                                      "Summary: Project Alpha aims to reduce carbon emissions by 20% using new solar panel technology. Key finding: efficiency increased by 5% in desert conditions.", 
                                      height=100, key="consolidate_text1")
    consolidate_text_2 = st.text_area("Text 2 for Consolidation:", 
                                      "Extraction: Project Alpha focuses on solar energy. Challenges include scaling production of the new panels and initial investment costs. No mention of wind power.", 
                                      height=100, key="consolidate_text2")
    consolidation_topic = st.text_input("Topic for Consolidation:", "Project Alpha Overview", key="consolidation_topic_input")
    
    if configured_for_test:
        if st.button("Test Consolidation of Texts", key="test_consolidation_button"):
            if not consolidation_topic:
                st.warning("Please enter a topic for consolidation.")
            else:
                texts_to_consolidate = []
                if consolidate_text_1: texts_to_consolidate.append(consolidate_text_1)
                if consolidate_text_2: texts_to_consolidate.append(consolidate_text_2)

                if not texts_to_consolidate:
                    st.warning("Please provide at least one text for consolidation.")
                else:
                    with st.spinner("Generating consolidated summary..."):
                        consolidated_output = generate_consolidated_summary(
                            texts_to_consolidate, 
                            consolidation_topic, 
                            GEMINI_API_KEY_TEST, 
                            model_name=MODEL_NAME_TEST
                        )
                    st.markdown("**Consolidated Output:**"); st.write(consolidated_output)
    else:
        st.warning("Gemini not configured. Please provide API key to run tests.")

# end of modules/llm_processor.py
