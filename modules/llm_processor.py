# modules/llm_processor.py
# Version 1.9.4:
# - Enhanced `generate_search_queries` to include 'Main Query 2' (Q2) in the LLM context.
# - Updated prompt for `generate_search_queries` to explicitly guide LLM on using
#   original keywords, Q1, and Q2 for more comprehensive query generation.
# - Updated relevant docstrings and test block.
# Version 1.9.3:
# - Enhanced the focused consolidated summary prompt for depth and plain text output.
# - Comprehensive docstring review and updates.

"""
Handles interactions with Large Language Models (LLMs), specifically Google Gemini.

This module is responsible for configuring the LLM client, making API calls,
and providing functionalities such as:
- Generating summaries of text content.
- Extracting specific information based on user queries and providing relevancy scores.
- Generating consolidated overviews from multiple text snippets (either general or
  focused on a specific query from high-relevance items).
- Generating alternative search queries based on initial keywords and user goals (Q1 and Q2).

It incorporates caching for LLM responses to optimize performance and reduce API costs,
and includes retry mechanisms for API calls. All LLM outputs requiring direct display
in the UI are generally formatted or instructed to be in plain text.
"""

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import time
import random
import re

_GEMINI_CONFIGURED: bool = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    """
    Configures the Google Gemini API client and lists available models.

    Sets up the `genai` client with the provided API key. If successful, it attempts
    to list models supporting 'generateContent' and caches them. This function is
    called before any LLM operation. Configuration status and model list are cached
    at the module level to avoid redundant calls.

    Args:
        api_key: Optional[str]: The Google Gemini API key. If None or empty,
            configuration will fail.
        force_recheck_models: bool: If True, forces a re-fetch of available models
            even if they were previously cached. Defaults to False.

    Returns:
        bool: True if configuration was successful and at least one suitable model
            is available, False otherwise. Outputs warnings/errors to Streamlit UI
            or console on failure.
    """
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True
    if not api_key:
        _GEMINI_CONFIGURED = False
        return False
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    try: st.warning("LLM_PROCESSOR: No Gemini models found supporting 'generateContent'. LLM features disabled.")
                    except Exception: print("LLM_PROCESSOR_WARNING: No Gemini models found supporting 'generateContent'. LLM features disabled.")
                    _GEMINI_CONFIGURED = False
                    return False
            except Exception as e_list_models:
                try: st.error(f"LLM_PROCESSOR: Error listing Gemini models: {e_list_models}. LLM features may be impaired.")
                except Exception: print(f"LLM_PROCESSOR_ERROR: Error listing Gemini models: {e_list_models}. LLM features may be impaired.")
                _AVAILABLE_MODELS_CACHE = []
                _GEMINI_CONFIGURED = False # If model listing fails, consider it not fully configured
                return False
        return True
    except Exception as e_configure:
        try: st.error(f"LLM_PROCESSOR: Failed to configure Google Gemini client: {e_configure}. LLM features disabled.")
        except Exception: print(f"LLM_PROCESSOR_ERROR: Failed to configure Google Gemini client: {e_configure}. LLM features disabled.")
        _GEMINI_CONFIGURED = False
        _AVAILABLE_MODELS_CACHE = None
        return False

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
    Calls the Google Gemini API with specified parameters and handles retries.

    This private helper function manages the direct interaction with the Gemini
    LLM. It validates the model name against a cache of available models,
    initializes the model object, and sends the request with generation
    configuration and safety settings. It implements an exponential backoff
    retry mechanism for rate limit errors.

    Args:
        model_name: str: The name of the Gemini model to use (e.g., "models/gemini-1.5-flash-latest").
        prompt_parts: List[str]: A list of strings forming the prompt content.
        generation_config_args: Optional[Dict[str, Any]]: Additional arguments for
            `genai.types.GenerationConfig` (e.g., max_output_tokens, temperature).
            Defaults are applied if not provided.
        safety_settings_args: Optional[List[Dict[str, Any]]]: Safety settings for the
            API call. Defaults to blocking only high-harm categories.
        max_retries: int: Maximum number of retries for rate limit errors.
        initial_backoff_seconds: float: Initial delay for retries.
        max_backoff_seconds: float: Maximum delay for retries.

    Returns:
        Optional[str]: The text response from the LLM, or an error message string
            prefixed with "LLM_PROCESSOR Error:", "LLM Error:", or "LLM_PROCESSOR:"
            if an error occurs, the response is blocked, or retries are exhausted.
            Returns None if Gemini is not configured.
    """
    if not _GEMINI_CONFIGURED:
        return "LLM_PROCESSOR Error: Gemini client not configured."

    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE:
            prefixed_attempt = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            if prefixed_attempt in _AVAILABLE_MODELS_CACHE:
                validated_model_name = prefixed_attempt
            else:
                return f"LLM_PROCESSOR Error: Model '{model_name}' not found or not supported. Available: {_AVAILABLE_MODELS_CACHE}"
    try:
        model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init:
        return f"LLM_PROCESSOR Error: Could not initialize model '{validated_model_name}': {e_model_init}"

    effective_gen_config_params = {"max_output_tokens": 1024}
    if generation_config_args:
        effective_gen_config_params.update(generation_config_args)
    if "temperature" not in effective_gen_config_params:
        effective_gen_config_params["temperature"] = 0.4

    generation_config = genai.types.GenerationConfig(**effective_gen_config_params)
    safety_settings = safety_settings_args or [
        {"category": cat, "threshold": "BLOCK_ONLY_HIGH"}
        for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
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
            if not response.candidates:
                reason_message = "LLM_PROCESSOR: Response blocked or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message:
                        reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked:
                                reason_message += f" Blocked by safety category: {rating.category}."
                print(reason_message)
                return reason_message
            return response.text.strip() if response.text else "LLM_PROCESSOR: Received empty text response."
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
                print(f"LLM_PROCESSOR: Rate limit hit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). Error: {str(e)[:150]}...")
                time.sleep(current_backoff)
                current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else:
                print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}")
                return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}"
    return f"LLM_PROCESSOR Error: Max retries ({max_retries}) reached for '{validated_model_name}'."

def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    """
    Truncates text to ensure it's within a specified character limit for Gemini.

    Args:
        text: str: The input text string.
        model_name: str: The name of the model (currently unused in logic but
            kept for potential future model-specific limits).
        max_input_chars: int: The maximum allowed number of characters for the input text.

    Returns:
        str: The original text if within limits, or the truncated text.
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
    Generates a summary for the given text content using the LLM.

    The LLM is prompted to create a detailed summary of 4-6 substantial sentences
    or 2-3 short paragraphs, capturing core messages and key details.
    The function handles API key configuration and text truncation.
    Results are cached.

    Args:
        text_content: Optional[str]: The text to be summarized.
        api_key: Optional[str]: The Google Gemini API key.
        model_name: str: The Gemini model to use for summarization.
        max_input_chars: int: Maximum characters of text_content to pass to the LLM.

    Returns:
        Optional[str]: The generated summary as a string, or an informational/error
            message if summarization fails (e.g., "LLM_PROCESSOR: No text content...",
            "LLM_PROCESSOR Error: Gemini not configured...").
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
    Extracts specific information from text content based on a query and scores its relevancy.

    The LLM is prompted to first provide a relevancy score (1/5 to 5/5) on the first line,
    formatted as "Relevancy Score: X/5". Subsequent lines should contain the extracted
    information. If no information is found, it should state so after a "Relevancy Score: 1/5".
    Results are cached.

    Args:
        text_content: Optional[str]: The text from which to extract information.
        extraction_query: str: The user's query guiding the information extraction.
        api_key: Optional[str]: The Google Gemini API key.
        model_name: str: The Gemini model to use.
        max_input_chars: int: Maximum characters of text_content to pass to the LLM.

    Returns:
        Optional[str]: A string starting with "Relevancy Score: X/5" followed by
            the extracted information, or an informational/error message if extraction fails.
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

def _parse_score_and_get_content(text_with_potential_score: str) -> Tuple[Optional[int], str]:
    """
    Parses a relevancy score and extracts content from a string.

    Assumes the input string might start with "Relevancy Score: X/5".
    If so, it extracts the integer score (X) and the text content that follows.
    If the score line is not present or malformed, score is None and original text is returned.

    Args:
        text_with_potential_score: str: The input string, potentially containing a
            score line.

    Returns:
        Tuple[Optional[int], str]: A tuple containing the extracted integer score
            (or None if not found/parsable) and the remaining content text (stripped).
    """
    score = None
    content_text = text_with_potential_score
    if text_with_potential_score and text_with_potential_score.startswith("Relevancy Score: "):
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
    summaries: Tuple[Optional[str], ...],
    topic_context: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 150000,
    extraction_query_for_consolidation: Optional[str] = None
) -> Optional[str]:
    """
    Generates a consolidated summary from a collection of text snippets.

    This function synthesizes an overview from provided summaries or extraction results.
    It supports two modes:
    1.  **Focused Consolidation:** If `extraction_query_for_consolidation` is given,
        it filters input `summaries` for items with a relevancy score >= 3 (parsed
        from the item text). The LLM is then prompted to create a detailed and
        comprehensive overview deeply exploring insights related to this specific query,
        drawing extensively from these high-scoring snippets. This mode aims for a
        longer, more elaborate output.
    2.  **General Consolidation:** If no `extraction_query_for_consolidation` is provided,
        or no items meet the relevancy criteria for a focused summary, it generates
        a general overview based on all valid input `summaries` and the `topic_context`.

    All LLM-generated consolidated summaries are explicitly instructed to be in PLAIN TEXT ONLY.
    Filters out error messages or irrelevant content from input summaries before processing.
    Results are cached.

    Args:
        summaries: Tuple[Optional[str], ...]: A tuple of strings, where each string is
            an individual LLM output (summary or extraction result). These may include
            relevancy scores.
        topic_context: str: General topic context for the batch, used if a general
            summary is generated.
        api_key: Optional[str]: The Google Gemini API key.
        model_name: str: The Gemini model to use.
        max_input_chars: int: Maximum combined characters of input texts to pass to the LLM.
        extraction_query_for_consolidation: Optional[str]: If provided, this query
            guides a focused consolidation using only high-relevancy items related to it.

    Returns:
        Optional[str]: The consolidated summary as a plain text string, or an
            informational/error message (e.g., "LLM_PROCESSOR: No individual LLM outputs...",
            "LLM_PROCESSOR_INFO: No suitable content found...").
    """
    if not summaries:
        return "LLM_PROCESSOR: No individual LLM outputs provided for consolidation."
    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini not configured for consolidated summary."

    texts_for_llm_input: List[str] = []
    is_focused_consolidation_active = bool(extraction_query_for_consolidation and extraction_query_for_consolidation.strip())

    for item_text in summaries:
        if not item_text: continue
        lower_item_text = item_text.lower()
        if lower_item_text.startswith("llm error") or \
           lower_item_text.startswith("no text content") or \
           lower_item_text.startswith("llm_processor: no text content") or \
           lower_item_text.startswith("llm_processor error:") or \
           lower_item_text.startswith("llm_processor: no extraction query") or \
           lower_item_text.startswith("llm_processor: response blocked"):
            continue

        score, content = _parse_score_and_get_content(item_text)

        if is_focused_consolidation_active:
            if score is not None and score >= 3:
                if content:
                    texts_for_llm_input.append(item_text)
        else:
            if content: texts_for_llm_input.append(content)
            elif item_text: texts_for_llm_input.append(item_text)

    if not texts_for_llm_input:
        if is_focused_consolidation_active:
            return (f"LLM_PROCESSOR_INFO: No suitable content found from items matching relevancy criteria (score >= 3) for query: '{extraction_query_for_consolidation}'. Cannot generate focused overview.")
        else:
            return "LLM_PROCESSOR_INFO: No valid LLM outputs found to consolidate into a general overview."

    combined_texts = "\n\n---\n\n".join([f"Source Document Content Snippet {i+1}:\n{text_entry}" for i, text_entry in enumerate(texts_for_llm_input)])
    truncated_combined_text = _truncate_text_for_gemini(combined_texts, model_name, max_input_chars)

    prompt_instruction: str
    final_topic_context_for_prompt: str
    max_tokens_for_call = 800

    plain_text_instruction = "IMPORTANT: Your entire response MUST be in PLAIN TEXT only. Do not use any markdown formatting (e.g., no bolding, italics, headers, bullet points using '*', '-', or numbers, etc.). Paragraphs should be separated by a single newline character."

    if is_focused_consolidation_active:
        final_topic_context_for_prompt = extraction_query_for_consolidation
        prompt_instruction = (
            f"You are an expert research analyst. You have been provided with several text snippets, each pre-selected for its high relevance to the central query: '{final_topic_context_for_prompt}'. "
            "These snippets are extraction results and include a 'Relevancy Score: X/5' line followed by the extracted text.\n"
            "Your task is to synthesize a **detailed and comprehensive consolidated overview that deeply explores the insights related to this central query**, drawing extensively from the provided text snippets (specifically, the extracted text part).\n"
            "Construct your overview by:\n"
            "1. Identifying the key findings, arguments, data points, and examples within the extracted text portions of the snippets that directly address the query: '{final_topic_context_for_prompt}'.\n"
            "2. Weaving these specific pieces of information into a cohesive and insightful narrative that thoroughly explains what was found concerning the query.\n"
            "3. Highlighting any significant patterns, corroborating evidence, or notable discrepancies among the snippets regarding the query.\n"
            "4. Elaborating on the implications or main takeaways from the collective information related to this specific query.\n"
            "A longer, more elaborate output (e.g., 3-5 substantial paragraphs, or more if the content from the snippets supports it) is **strongly preferred** for this focused task, providing a thorough analysis. "
            "The overview must be structured in well-organized paragraphs and **focus predominantly on synthesizing the details from the provided snippets relevant to the central query.** "
            "Do NOT include information clearly irrelevant to this central query, even if peripherally mentioned in the source texts. "
            f"Ignore the 'Relevancy Score' lines themselves when forming the summary; focus on the content that follows them. "
            f"Avoid generic introductory phrases (e.g., 'This summary will discuss...') or concluding phrases about the act of summarizing. Begin directly with the synthesis.\n{plain_text_instruction}\n\n"
        )
        max_tokens_for_call = 1600
    else:
        final_topic_context_for_prompt = topic_context
        prompt_instruction = (
            f"You are an expert analyst tasked with synthesizing information from multiple text sources broadly related to the topic: '{final_topic_context_for_prompt}'. "
            "These sources are summaries or general extractions from documents.\n"
            "Your goal is to create a single, coherent consolidated overview. Identify the main themes, arguments, and key pieces of information present across the texts. "
            "Synthesize these into a cohesive narrative. If there are notable patterns, supporting evidence, or discrepancies across the sources, please highlight them. "
            f"Present your overview in well-structured paragraphs. Aim for a comprehensive yet reasonably concise summary (e.g., 2-4 substantial paragraphs).\n{plain_text_instruction}\n\n"
        )

    prompt = (f"{prompt_instruction}--- PROVIDED TEXTS START ---\n{truncated_combined_text}\n--- PROVIDED TEXTS END ---\n\nConsolidated Overview (focused on '{final_topic_context_for_prompt}' if applicable):")

    result_prefix = ""
    if not is_focused_consolidation_active:
        result_prefix = "LLM_PROCESSOR_INFO: General overview as follows.\n"

    llm_response = _call_gemini_api(
        model_name,
        [prompt],
        generation_config_args={"max_output_tokens": max_tokens_for_call, "temperature": 0.35}
    )
    if llm_response and not llm_response.lower().startswith("llm error") and not llm_response.lower().startswith("llm_processor error"):
        return result_prefix + llm_response
    return llm_response


@st.cache_data(max_entries=50, ttl=3600, show_spinner=False)
def generate_search_queries(
    original_keywords: Tuple[str, ...],
    specific_info_query: Optional[str], # Represents Main Query 1 (Q1)
    specific_info_query_2: Optional[str], # Represents Main Query 2 (Q2)
    num_queries_to_generate: int,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 2500 # Increased slightly for Q2
) -> Optional[List[str]]:
    """
    Generates new search queries based on original keywords and specific information goals (Q1 and Q2).

    The LLM is prompted to create diverse search queries by considering the original
    keywords, a primary information goal (Q1), and a secondary information goal (Q2).
    It aims to generate queries that holistically address the user's intent,
    exploring intersections, synonyms, and related concepts.
    Results are cached.

    Args:
        original_keywords: Tuple[str, ...]: Initial keywords from the user.
        specific_info_query: Optional[str]: The primary specific information goal (Main Query 1).
        specific_info_query_2: Optional[str]: The secondary specific information goal (Main Query 2).
        num_queries_to_generate: int: The exact number of new search queries to generate.
        api_key: Optional[str]: The Google Gemini API key.
        model_name: str: The Gemini model to use.
        max_input_chars: int: Max characters of the context (keywords + Q1 + Q2) for the LLM.

    Returns:
        Optional[List[str]]: A list of generated search query strings. Returns None
            if generation fails (e.g., Gemini not configured, LLM error). Returns an
            empty list if `num_queries_to_generate` is zero or less. Queries are
            stripped of leading/trailing quotes.
    """
    if not original_keywords:
        return None
    if num_queries_to_generate <= 0:
        return []
    if not configure_gemini(api_key):
        print("LLM_PROCESSOR Warning: Gemini not configured for generating search queries.")
        return None

    original_keywords_str = ", ".join(original_keywords)
    context_lines = [f"Original user search keywords: \"{original_keywords_str}\"."]

    has_q1 = specific_info_query and specific_info_query.strip()
    has_q2 = specific_info_query_2 and specific_info_query_2.strip()

    if has_q1:
        context_lines.append(f"User's primary information goal (Query 1): \"{specific_info_query.strip()}\".")
    if has_q2:
        context_lines.append(f"User's secondary information goal (Query 2): \"{specific_info_query_2.strip()}\".")

    if not has_q1 and not has_q2:
        context_lines.append("User has not provided any specific information goals beyond initial keywords.")
    # Note: If only Q1 or only Q2 is present, the labels "primary" or "secondary" might seem odd in isolation,
    # but they accurately reflect which input field they came from. The LLM should be able to handle this.

    context_for_llm = "\n".join(context_lines)
    truncated_context = _truncate_text_for_gemini(context_for_llm, model_name, max_input_chars)

    prompt = (
        "You are an expert Search Query Assistant.\n"
        "Your task is to generate exactly {num_queries_to_generate} new, distinct search queries to find web pages that comprehensively address the user's overall search intent. "
        "The user's intent is defined by their original keywords and up to two specific information goals (Query 1 and Query 2) provided in the context below.\n\n"
        "USER INPUT CONTEXT:\n---\n{context}\n---\n\n"
        "When generating your new search queries, carefully consider:\n"
        "- How Query 1 and Query 2 (if both provided) relate to each other and to the original keywords. Are they complementary, refining, or distinct aspects of a broader topic?\n"
        "- If only one Query (1 or 2) is provided, how does it refine or specify the original keywords?\n"
        "- Generating queries that might address the intersection of these goals, or provide comprehensive information covering multiple aspects if appropriate.\n"
        "- Synonyms, related concepts, long-tail variations, and different phrasings for all elements of the user's input (original keywords, Query 1, and Query 2).\n"
        "The new queries should be effective for web searching, differ from the original keywords, and be distinct from each other. "
        "Provide your {num_queries_to_generate} new search queries below, each on a new line. No numbering, bullets, or other formatting. Just raw search queries."
    ).format(num_queries_to_generate=num_queries_to_generate, context=truncated_context)

    response_text = _call_gemini_api(
        model_name,
        [prompt],
        generation_config_args={"max_output_tokens": num_queries_to_generate * 50, "temperature": 0.55} # Slightly more tokens, adjusted temp
    )

    if response_text and not response_text.startswith("LLM Error") and not response_text.startswith("LLM_PROCESSOR Error"):
        generated_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        generated_queries = [re.sub(r'^(["\'])(.*)\1$', r'\2', q) for q in generated_queries]
        return generated_queries[:num_queries_to_generate]

    print(f"LLM_PROCESSOR Warning: Failed to generate search queries or error occurred. Response: {response_text}")
    return None


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.9.4 - Q2 in Search Gen)")
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password", key="main_api_key_test_llm_v194")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest", key="main_model_name_test_llm_v194")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True):
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE: st.write("Available Models (first 5):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True
        else: st.error("Failed to configure Gemini for testing.")
    else: st.info("Enter Gemini API Key to enable tests.")

    if configured_for_test:
        with st.expander("Test Generate Search Queries (Cached)", expanded=True): # Expanded for current focus
            test_original_keywords_str_gsq = st.text_input("Original Keywords (comma-separated):", "sustainable energy solutions, home impact", key="test_orig_kw_gsq_v194")
            test_specific_info_query_gsq = st.text_input("Specific Info Goal (Q1 - Optional):", "solar panel cost and efficiency 2024", key="test_spec_info_q1_gsq_v194")
            test_specific_info_query_2_gsq = st.text_input("Specific Info Goal (Q2 - Optional):", "geothermal heating pros and cons for residential", key="test_spec_info_q2_gsq_v194") # New input for Q2
            test_num_queries_to_gen_gsq = st.number_input("Number of New Queries to Generate:", min_value=1, max_value=10, value=3, key="test_num_q_gen_gsq_v194")
            if st.button("Test Query Generation (Cached)", key="test_query_gen_button_gsq_v194"):
                test_original_keywords_tuple = tuple(k.strip() for k in test_original_keywords_str_gsq.split(',') if k.strip())
                if not test_original_keywords_tuple: st.warning("Please enter original keywords.")
                else:
                    with st.spinner("Generating new search queries..."):
                        new_queries = generate_search_queries(
                            original_keywords=test_original_keywords_tuple,
                            specific_info_query=test_specific_info_query_gsq if test_specific_info_query_gsq.strip() else None,
                            specific_info_query_2=test_specific_info_query_2_gsq if test_specific_info_query_2_gsq.strip() else None, # Pass Q2
                            num_queries_to_generate=test_num_queries_to_gen_gsq,
                            api_key=GEMINI_API_KEY_TEST,
                            model_name=MODEL_NAME_TEST
                        )
                    st.markdown("**Generated New Search Queries:**");
                    if new_queries is not None: st.json(new_queries)
                    else: st.write("No new queries generated or error.")

        with st.expander("Test Individual Summary (Cached)", expanded=False):
            sample_text_summary_cached = st.text_area("Sample Text for Summary:", "The Eiffel Tower is a famous landmark in Paris, France.", height=100, key="test_sample_summary_text_v194_cached")
            if st.button("Test Summary Generation (Cached)", key="test_summary_gen_button_v194_cached"):
                 with st.spinner(f"Generating summary..."):
                    summary_output = generate_summary(sample_text_summary_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                 st.markdown("**Generated Summary:**"); st.write(summary_output)

        with st.expander("Test Specific Information Extraction (Cached)", expanded=False):
            sample_text_extraction_cached = st.text_area("Sample Text for Extraction:", "The main product is X and it costs $50. Contact sales@example.com for more.", height=100, key="test_sample_extraction_text_v194_cached")
            extraction_query_test_cached = st.text_input("Extraction Query:", "product name and contact email", key="test_extraction_query_input_v194_cached")
            if st.button("Test Extraction & Relevancy Scoring (Cached)", key="test_extraction_score_button_v194_cached"):
                if not extraction_query_test_cached: st.warning("Please enter an extraction query.")
                else:
                    with st.spinner(f"Extracting info..."):
                        extraction_output = extract_specific_information(sample_text_extraction_cached, extraction_query_test_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                    st.markdown("**Extraction Output (with Relevancy Score):**"); st.text(extraction_output)

        with st.expander("Test Consolidation (Cached)", expanded=False):
            st.write("Provide texts for consolidation. For focused test, ensure Text 1 has a relevant score (e.g., 'Relevancy Score: 3/5' or higher).")
            consolidate_text_1_cached = st.text_area("Text 1 for Consolidation (LLM output format):", "Relevancy Score: 4/5\nThe Alpha project focuses on advanced solar panel technology and energy storage solutions.", height=100, key="test_consolidate_text1_v194_cached")
            consolidate_text_2_cached = st.text_area("Text 2 for Consolidation (LLM output format):", "LLM Summary: The Beta project is dedicated to reducing carbon emissions through innovative industrial processes. It has achieved a 20% reduction so far.", height=100, key="test_consolidate_text2_v194_cached")
            consolidate_text_3_cached = st.text_area("Text 3 for Consolidation (Irrelevant/Low Score):", "Relevancy Score: 1/5\nThis document discusses unrelated marketing strategies.", height=100, key="test_consolidate_text3_v194_cached")

            general_topic_context_cached = st.text_input("General Topic Context (for general summary):", "Renewable Energy Projects", key="test_general_topic_input_v194_cached")
            focused_extraction_query_cached = st.text_input("Focused Extraction Query (for focused summary, e.g., 'Alpha project details'):", "Alpha project details", key="test_focused_query_input_v194_cached")

            if st.button("Test Consolidation (Cached)", key="test_consolidation_button_main_v194_cached"):
                if not general_topic_context_cached: st.warning("Please enter a general topic context.")
                else:
                    texts_to_consolidate_tuple = tuple(t for t in [consolidate_text_1_cached, consolidate_text_2_cached, consolidate_text_3_cached] if t and t.strip())
                    if not texts_to_consolidate_tuple: st.warning("Please provide at least one text for consolidation.")
                    else:
                        consolidation_query_to_use_cached = focused_extraction_query_cached if focused_extraction_query_cached.strip() else None
                        with st.spinner("Generating consolidated summary..."):
                            consolidated_output = generate_consolidated_summary(
                                summaries=texts_to_consolidate_tuple,
                                topic_context=general_topic_context_cached,
                                api_key=GEMINI_API_KEY_TEST,
                                model_name=MODEL_NAME_TEST,
                                extraction_query_for_consolidation=consolidation_query_to_use_cached
                            )
                        st.markdown(f"**Consolidated Output (Mode: {'Focused on Q1' if consolidation_query_to_use_cached else 'General'}):**")
                        st.text(consolidated_output)
    else:
        st.warning("Gemini not configured. Provide API key for tests.")

# end of modules/llm_processor.py
