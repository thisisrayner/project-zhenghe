# modules/llm_processor.py
# Version 1.9.1: Added @st.cache_data to LLM processing functions.
# Includes generate_search_queries function from v1.9.0.

"""
Handles interactions with Large Language Models (LLMs) for text processing.

Currently configured to primarily use Google's Gemini models via the
`google-generativeai` library. It includes functionalities for:
- Configuring the LLM client.
- Generating summaries of text content.
- Extracting specific information based on user queries, including a relevancy score.
- Generating a consolidated overview from multiple text summaries, with options
  for filtering based on relevancy scores and focusing on specific extraction queries.
- Generating alternative search queries based on initial user input.
- Implements exponential backoff and retries for API calls to handle rate limits.
"""

import google.generativeai as genai
import streamlit as st 
from typing import Optional, Dict, Any, List
import time
import random
import re 

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
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None: return True
    if not api_key: return False
    try:
        genai.configure(api_key=api_key); _GEMINI_CONFIGURED = True 
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods: current_available_models.append(m.name)
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    try: st.warning("LLM_PROCESSOR: No Gemini models found supporting 'generateContent'.")
                    except Exception: print("LLM_PROCESSOR_WARNING: No Gemini models found supporting 'generateContent'.")
                    return False 
            except Exception as e_list_models:
                try: st.error(f"LLM_PROCESSOR: Error listing Gemini models: {e_list_models}.")
                except Exception: print(f"LLM_PROCESSOR_ERROR: Error listing Gemini models: {e_list_models}.")
                _AVAILABLE_MODELS_CACHE = [] ; _GEMINI_CONFIGURED = False ; return False
        return True 
    except Exception as e_configure:
        try: st.error(f"LLM_PROCESSOR: Failed to configure Google Gemini client: {e_configure}")
        except Exception: print(f"LLM_PROCESSOR_ERROR: Failed to configure Google Gemini client: {e_configure}")
        _GEMINI_CONFIGURED = False; return False

def _call_gemini_api(
    model_name: str,
    prompt_parts: List[str], 
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[List[Dict[str, Any]]] = None, 
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
        or a message indicating a content block. Returns "LLM_PROCESSOR Error: ..." if client is not configured.
    """
    if not _GEMINI_CONFIGURED: return "LLM_PROCESSOR Error: Gemini client not configured."
    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE:
            prefixed_attempt = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            if prefixed_attempt in _AVAILABLE_MODELS_CACHE: validated_model_name = prefixed_attempt
            else: return f"LLM_PROCESSOR Error: Model '{model_name}' not found. Available: {_AVAILABLE_MODELS_CACHE}"
    try: model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init: return f"LLM_PROCESSOR Error: Could not initialize model '{validated_model_name}': {e_model_init}"
    
    effective_gen_config_params = {"max_output_tokens": 1024} 
    if generation_config_args: effective_gen_config_params.update(generation_config_args)
    if "temperature" not in effective_gen_config_params:
        effective_gen_config_params["temperature"] = 0.4 

    generation_config = genai.types.GenerationConfig(**effective_gen_config_params)
    safety_settings = safety_settings_args or [{"category": cat, "threshold": "BLOCK_ONLY_HIGH"} for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    current_retry = 0; current_backoff = initial_backoff_seconds
    while current_retry <= max_retries:
        try:
            response = model_obj.generate_content(prompt_parts, generation_config=generation_config, safety_settings=safety_settings, stream=False)
            if not response.candidates:
                reason_message = "LLM_PROCESSOR: Response blocked or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message: reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: reason_message += f" Blocked by safety category: {rating.category}."
                print(reason_message); return reason_message 
            return response.text.strip() if response.text else "LLM_PROCESSOR: Received empty text response."
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or "resourceexhausted" in error_str.replace(" ", "") or "resource exhausted" in error_str or ("rate" in error_str and "limit" in error_str) or (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) ) 
            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1; print(f"LLM_PROCESSOR: Rate limit hit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). Error: {str(e)[:150]}...") 
                time.sleep(current_backoff); current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}"); return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}"
    return f"LLM_PROCESSOR Error: Max retries ({max_retries}) reached for '{validated_model_name}'."

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

@st.cache_data(max_entries=50, ttl=3600, show_spinner=False)
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
        return "LLM_PROCESSOR: No text content for summary."
    if not configure_gemini(api_key): 
        return "LLM_PROCESSOR Error: Gemini not configured for summary."
    
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
        generation_config_args={"max_output_tokens": 512, "temperature": 0.3}
    )

@st.cache_data(max_entries=100, ttl=3600, show_spinner=False)
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
        return "LLM_PROCESSOR: No text content for extraction."
    if not extraction_query: 
        return "LLM_PROCESSOR: No extraction query."
    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini not configured for extraction."

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
        generation_config_args={"max_output_tokens": 750, "temperature": 0.3}
    )

def _parse_score_and_get_content(text_with_potential_score: str) -> tuple[Optional[int], str]:
    """
    Parses a relevancy score from the beginning of a string (if present)
    and returns the score and the remaining content.

    Args:
        text_with_potential_score: The text string, which might start with "Relevancy Score: X/5\n".

    Returns:
        A tuple: (score, content_text).
        - score: An integer (1-5) if successfully parsed, otherwise None.
        - content_text: The text after the score line, or the original text if no score line.
    """
    score = None
    content_text = text_with_potential_score  

    if text_with_potential_score.startswith("Relevancy Score: "):
        parts = text_with_potential_score.split('\n', 1)
        score_line = parts[0]
        
        try:
            score_str = score_line.split("Relevancy Score: ")[1].split('/')[0]
            score = int(score_str)
            if len(parts) > 1: 
                content_text = parts[1].strip() 
            else:  
                content_text = "" 
        except (IndexError, ValueError):
            score = None 
            content_text = text_with_potential_score             
    return score, content_text.strip() 

@st.cache_data(max_entries=20, ttl=1800, show_spinner=False)
def generate_consolidated_summary(
    summaries: tuple[Optional[str], ...], # Changed to tuple for hashability
    topic_context: str, 
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", 
    max_input_chars: int = 150000,
    extraction_query_for_consolidation: Optional[str] = None 
) -> Optional[str]:
    """
    Generates a consolidated overview from a list of individual LLM outputs.

    If `extraction_query_for_consolidation` is provided, it filters the inputs to use only those
    with a parsed "Relevancy Score: X/5" of 3 or higher, and focuses the summary on that query.
    Otherwise, it generates a general summary from all valid provided texts.

    Args:
        summaries: A tuple of strings, where each is an LLM output (summary or extraction).
                   Extractions are expected to start with "Relevancy Score: X/5\n".
        topic_context: General topic/keywords of the search batch.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use.
        max_input_chars: Maximum characters for the combined input of all texts.
        extraction_query_for_consolidation: Optional user-defined query for specific info.
                                            If provided, triggers filtering and focused summary.

    Returns:
        A string containing the consolidated overview, or an error/status message.
    """
    if not summaries: 
        return "LLM_PROCESSOR: No individual LLM outputs provided for consolidation."
    if not configure_gemini(api_key): 
        return "LLM_PROCESSOR Error: Gemini not configured for consolidated summary."
    
    texts_for_llm_input: List[str] = []
    query_focused_consolidation_active = bool(extraction_query_for_consolidation and extraction_query_for_consolidation.strip())

    for item_text in summaries: 
        if not item_text: continue
        lower_item_text = item_text.lower()
        if lower_item_text.startswith("llm error") or lower_item_text.startswith("no text content") or lower_item_text.startswith("llm_processor:"): continue
        
        score, content = _parse_score_and_get_content(item_text)
        
        if query_focused_consolidation_active:
            if score is not None and score >= 3:
                if content: texts_for_llm_input.append(content)
        else:
            if content: texts_for_llm_input.append(content)

    if not texts_for_llm_input:
        if query_focused_consolidation_active: return (f"LLM_PROCESSOR: No items met score >=3 for query: '{extraction_query_for_consolidation}'.")
        else: return "LLM_PROCESSOR: No valid LLM outputs to consolidate."
    
    combined_texts = "\n\n---\n\n".join([f"Source Document {i+1} Content:\n{text_entry}" for i, text_entry in enumerate(texts_for_llm_input)])
    truncated_combined_text = _truncate_text_for_gemini(combined_texts, model_name, max_input_chars)
    prompt_instruction: str
    final_topic_context_for_prompt: str

    if query_focused_consolidation_active:
        final_topic_context_for_prompt = extraction_query_for_consolidation 
        prompt_instruction = (f"You are an expert analyst. Synthesize info from texts pre-selected for relevance to: '{final_topic_context_for_prompt}'.\nCreate a consolidated overview addressing this query. Focus exclusively on pertinent info. Synthesize key findings into a cohesive narrative. Highlight patterns or discrepancies concerning '{final_topic_context_for_prompt}'. Output well-structured paragraphs. Do NOT include irrelevant info.\n\n")
    else:
        final_topic_context_for_prompt = topic_context 
        prompt_instruction = (f"You are an expert analyst synthesizing info from multiple texts broadly related to '{final_topic_context_for_prompt}'.\nCreate a single, coherent consolidated overview. Identify main themes. Synthesize key info into a cohesive narrative. Highlight patterns or discrepancies. Output well-structured paragraphs.\n\n")
    
    prompt = (f"{prompt_instruction}--- PROVIDED TEXTS START ---\n{truncated_combined_text}\n--- PROVIDED TEXTS END ---\n\nConsolidated Overview regarding '{final_topic_context_for_prompt}':")
    return _call_gemini_api( model_name, [prompt], generation_config_args={"max_output_tokens": 800, "temperature": 0.3})

@st.cache_data(max_entries=50, ttl=3600, show_spinner=False)
def generate_search_queries(
    original_keywords: tuple[str, ...], 
    specific_info_query: Optional[str],
    num_queries_to_generate: int,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest", 
    max_input_chars: int = 2000 
) -> Optional[List[str]]:
    """
    Generates a list of new search queries based on original keywords and a specific info query.

    Args:
        original_keywords: A tuple of keywords/queries input by the user (must be hashable for caching).
        specific_info_query: The user's query for specific information to extract (can be None).
        num_queries_to_generate: The desired number of new search queries.
        api_key: The Google Gemini API key.
        model_name: The specific Gemini model to use.
        max_input_chars: Maximum characters of combined input to send to the LLM.

    Returns:
        A list of new search query strings, or None if an error occurs or no queries are generated.
    """
    if not original_keywords: return None 
    if num_queries_to_generate <= 0: return [] 
    if not configure_gemini(api_key): print("LLM_PROCESSOR Warning: Gemini not configured for generating search queries."); return None

    original_keywords_str = ", ".join(original_keywords)
    context_for_llm = f"Original user search keywords: \"{original_keywords_str}\"."
    if specific_info_query and specific_info_query.strip(): context_for_llm += f"\nUser's specific information goal: \"{specific_info_query.strip()}\"."
    else: context_for_llm += "\nUser has not provided a specific information goal beyond initial keywords."
    truncated_context = _truncate_text_for_gemini(context_for_llm, model_name, max_input_chars)
    prompt = ( "You are an expert Search Query Assistant.\nBased on user input, generate exactly {num_queries_to_generate} new, distinct search queries to find highly relevant web pages. Consider synonyms, related concepts, long-tail variations, and different angles. New queries should differ from original keywords and each other.\n\nUSER INPUT CONTEXT:\n---\n{context}\n---\n\nProvide your {num_queries_to_generate} new search queries below, each on a new line. No numbering, bullets, or other formatting. Just raw search queries."
    ).format(num_queries_to_generate=num_queries_to_generate, context=truncated_context)
    response_text = _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": num_queries_to_generate * 40, "temperature": 0.5} )
    if response_text and not response_text.startswith("LLM Error") and not response_text.startswith("LLM_PROCESSOR Error"):
        generated_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        generated_queries = [re.sub(r'^(["\'])(.*)\1$', r'\2', q) for q in generated_queries] 
        return generated_queries[:num_queries_to_generate] 
    print(f"LLM_PROCESSOR Warning: Failed to generate search queries or error: {response_text}"); return None

# --- if __name__ == '__main__': block for testing ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.9.1 - Caching)") 
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password", key="main_api_key_test_llm_v191") 
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest", key="main_model_name_test_llm_v191") 
    configured_for_test = False 
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): 
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE: st.write("Available Models (first 5):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True 
        else: st.error("Failed to configure Gemini for testing.")
    else: st.info("Enter Gemini API Key to enable tests.")

    if configured_for_test:
        with st.expander("Test Generate Search Queries (Cached)", expanded=False):
            test_original_keywords_str_gsq = st.text_input("Original Keywords (comma-separated):", "home office setup, ergonomic chair", key="test_orig_kw_gsq_v191")
            test_specific_info_query_gsq = st.text_input("Specific Info Goal (Optional):", "best chair for back pain under $300", key="test_spec_info_gsq_v191")
            test_num_queries_to_gen_gsq = st.number_input("Number of New Queries to Generate:", min_value=1, max_value=10, value=2, key="test_num_q_gen_gsq_v191")
            if st.button("Test Query Generation (Cached)", key="test_query_gen_button_gsq_v191"):
                test_original_keywords_tuple = tuple(k.strip() for k in test_original_keywords_str_gsq.split(',') if k.strip()) 
                if not test_original_keywords_tuple: st.warning("Please enter original keywords.")
                else:
                    with st.spinner("Generating new search queries..."):
                        new_queries = generate_search_queries(original_keywords=test_original_keywords_tuple, specific_info_query=test_specific_info_query_gsq if test_specific_info_query_gsq.strip() else None, num_queries_to_generate=test_num_queries_to_gen_gsq, api_key=GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                    st.markdown("**Generated New Search Queries:**"); 
                    if new_queries: st.json(new_queries)
                    else: st.write("No new queries generated or error.")
        
        with st.expander("Test Individual Summary (Cached)", expanded=False):
            sample_text_summary_cached = st.text_area("Sample Text for Summary:", "The Eiffel Tower is a famous landmark in Paris, France.", height=100, key="test_sample_summary_text_v191_cached")
            if st.button("Test Summary Generation (Cached)", key="test_summary_gen_button_v191_cached"):
                 with st.spinner(f"Generating summary..."): 
                    summary_output = generate_summary(sample_text_summary_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                 st.markdown("**Generated Summary:**"); st.write(summary_output)

        with st.expander("Test Specific Information Extraction (Cached)", expanded=False):
            sample_text_extraction_cached = st.text_area("Sample Text for Extraction:", "The main product is X and it costs $50. Contact sales@example.com for more.", height=100, key="test_sample_extraction_text_v191_cached")
            extraction_query_test_cached = st.text_input("Extraction Query:", "product name and contact email", key="test_extraction_query_input_v191_cached")
            if st.button("Test Extraction & Relevancy Scoring (Cached)", key="test_extraction_score_button_v191_cached"):
                if not extraction_query_test_cached: st.warning("Please enter an extraction query.")
                else:
                    with st.spinner(f"Extracting info..."): 
                        extraction_output = extract_specific_information(sample_text_extraction_cached, extraction_query_test_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                    st.markdown("**Extraction Output (with Relevancy Score):**"); st.text(extraction_output)

        with st.expander("Test Consolidation (Cached)", expanded=False): 
            st.write("Provide texts for consolidation.")
            consolidate_text_1_cached = st.text_area("Text 1 for Consolidation:", "Relevancy Score: 4/5\nAlpha project is about solar.", height=100, key="test_consolidate_text1_v191_cached")
            consolidate_text_2_cached = st.text_area("Text 2 for Consolidation:", "Summary: Beta project reduces emissions.", height=100, key="test_consolidate_text2_v191_cached")
            general_topic_context_cached = st.text_input("General Topic Context for Consolidation:", "Energy Projects", key="test_general_topic_input_v191_cached")
            focused_extraction_query_cached = st.text_input("Focused Extraction Query for Consolidation (Optional):", "Alpha project", key="test_focused_query_input_v191_cached")
            if st.button("Test Consolidation (Cached)", key="test_consolidation_button_main_v191_cached"):
                if not general_topic_context_cached: st.warning("Please enter a general topic context.")
                else:
                    texts_to_consolidate_tuple = tuple(t for t in [consolidate_text_1_cached, consolidate_text_2_cached] if t) # Use tuple
                    if not texts_to_consolidate_tuple: st.warning("Please provide at least one text.")
                    else:
                        consolidation_query_to_use_cached = focused_extraction_query_cached if focused_extraction_query_cached.strip() else None
                        with st.spinner("Generating consolidated summary..."): 
                            consolidated_output = generate_consolidated_summary(summaries=texts_to_consolidate_tuple, topic_context=general_topic_context_cached, api_key=GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST, extraction_query_for_consolidation=consolidation_query_to_use_cached)
                        st.markdown("**Consolidated Output:**"); st.write(consolidated_output)
    else: 
        st.warning("Gemini not configured. Provide API key for tests.")

# end of modules/llm_processor.py
