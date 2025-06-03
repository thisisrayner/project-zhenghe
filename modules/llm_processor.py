# modules/llm_processor.py
# Version 1.9.10:
# - Updated _parse_score_and_get_content with more robust regex-based parsing
#   for relevancy scores and added debug prints.
# Previous versions:
# - Version 1.9.9: Corrected narrative paragraph instruction, simplified TLDR post-processing.

"""
Handles interactions with Large Language Models (LLMs), specifically Google Gemini.

This module is responsible for configuring the LLM client, making API calls,
and providing functionalities such as:
- Generating summaries of text content.
- Extracting specific information based on user queries and providing relevancy scores.
- Generating consolidated overviews from multiple text snippets. The overview consists
  of a narrative part (plain text with proper paragraph separation) followed by a
  "TLDR:" section with dash-bulleted key points.
  This can be a general overview or focused on a specific query (Q1) with potential
  enrichment from a secondary query (Q2).
- Generating alternative search queries based on initial keywords and user goals (Q1 and Q2).

It incorporates caching for LLM responses to optimize performance and reduce API costs,
and includes retry mechanisms for API calls.
"""

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import time
import random
import re # Ensure re is imported for regex operations

_GEMINI_CONFIGURED: bool = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    """
    Configures the Google Gemini API client and lists available models.
    Args:
        api_key: Optional[str]: The Google Gemini API key.
        force_recheck_models: bool: If True, forces a re-fetch of available models.
    Returns:
        bool: True if configuration was successful, False otherwise.
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
                _GEMINI_CONFIGURED = False 
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
    Args:
        model_name: Name of the Gemini model.
        prompt_parts: List of strings for the prompt.
        generation_config_args: Optional generation config.
        safety_settings_args: Optional safety settings.
        max_retries: Max retries for rate limits.
        initial_backoff_seconds: Initial retry delay.
        max_backoff_seconds: Max retry delay.
    Returns:
        LLM text response or an error message string.
    """
    if not _GEMINI_CONFIGURED:
        return "LLM_PROCESSOR Error: Gemini client not configured."
    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None: # Check if cache is populated
        if model_name not in _AVAILABLE_MODELS_CACHE:
            prefixed_attempt = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            if prefixed_attempt in _AVAILABLE_MODELS_CACHE: 
                validated_model_name = prefixed_attempt
            else: 
                print(f"LLM_PROCESSOR Warning: Model '{model_name}' not in available list. Prefixed='{prefixed_attempt}'. Available: {_AVAILABLE_MODELS_CACHE}")
                # Proceed with original model_name if not found, let genai.GenerativeModel handle it
                # This avoids erroring out if _AVAILABLE_MODELS_CACHE is somehow incomplete or a new model is used.
    try: 
        model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init: 
        return f"LLM_PROCESSOR Error: Could not initialize model '{validated_model_name}': {e_model_init}"
    
    effective_gen_config_params = {"max_output_tokens": 1024} 
    if generation_config_args: effective_gen_config_params.update(generation_config_args)
    if "temperature" not in effective_gen_config_params: effective_gen_config_params["temperature"] = 0.4 

    generation_config = genai.types.GenerationConfig(**effective_gen_config_params)
    safety_settings = safety_settings_args or [{"category": cat, "threshold": "BLOCK_ONLY_HIGH"} for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    current_retry = 0; current_backoff = initial_backoff_seconds
    while current_retry <= max_retries:
        try:
            response = model_obj.generate_content(prompt_parts, generation_config=generation_config, safety_settings=safety_settings, stream=False)
            if not response.candidates:
                reason_message = "LLM_PROCESSOR: Response blocked or empty."
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback: # Check existence
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message: reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: reason_message += f" Blocked by safety category: {rating.category}."
                print(reason_message); return reason_message 
            return response.text.strip() if hasattr(response, 'text') and response.text else "LLM_PROCESSOR: Received empty text response or response has no text attribute."
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or "resourceexhausted" in error_str.replace(" ", "") or "resource exhausted" in error_str or ("rate" in error_str and "limit" in error_str) or (hasattr(e, 'details') and callable(getattr(e, 'details')) and "Quota" in e.details()) or (hasattr(e, 'code') and callable(getattr(e, 'code')) and e.code().value == 8) ) 
            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1; print(f"LLM_PROCESSOR: Rate limit hit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). Error: {str(e)[:150]}...") 
                time.sleep(current_backoff); current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}"); return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:300]}"
    return f"LLM_PROCESSOR Error: Max retries ({max_retries}) reached for '{validated_model_name}'."


def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    """Truncates text to a maximum character limit."""
    if len(text) > max_input_chars:
        # st.warning(f"LLM_PROCESSOR: Truncating input text for model {model_name} from {len(text)} to {max_input_chars} characters.")
        return text[:max_input_chars]
    return text

@st.cache_data(max_entries=50, ttl=3600, show_spinner=False)
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000
) -> Optional[str]:
    """Generates a plain text summary for given text content."""
    if not text_content: return "LLM_PROCESSOR: No text content for summary."
    if not configure_gemini(api_key): return "LLM_PROCESSOR Error: Gemini not configured for summary."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (
        "You are an expert assistant specializing in creating detailed and insightful summaries of web page content.\n"
        "Analyze the following text and provide a comprehensive summary of approximately 4-6 substantial sentences (or 2-3 short paragraphs if the content is rich). "
        "Your summary should capture the core message, key arguments, supporting details, and any significant conclusions or implications. "
        "Maintain a neutral and factual tone. Your entire response for this summary MUST be in PLAIN TEXT only. Do not use any markdown formatting (e.g., no bolding, italics, headers, or lists). "
        "Paragraphs should be separated by a blank line (two newline characters). "
        "Avoid introductory phrases like 'This text discusses...' or 'The summary of the text is...'. Go directly into the summary content.\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        "Detailed Summary (PLAIN TEXT ONLY, paragraphs separated by a blank line):"
    )
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 512, "temperature": 0.3})

@st.cache_data(max_entries=100, ttl=3600, show_spinner=False)
def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 100000
) -> Optional[str]:
    """Extracts specific info and scores relevancy. Extracted info part is plain text."""
    if not text_content: return "LLM_PROCESSOR: No text content for extraction."
    if not extraction_query: return "LLM_PROCESSOR: No extraction query."
    if not configure_gemini(api_key): return "LLM_PROCESSOR Error: Gemini not configured for extraction."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    plain_text_instruction_for_extraction = (
        "For the extracted information that follows the 'Relevancy Score:' line, "
        "present it in PLAIN TEXT. If multiple pieces of information are found, "
        "separate them clearly, typically by starting each distinct piece on a new line. "
        "Do not use markdown like bolding, italics, or lists for this extracted content part."
    )
    # The prompt needs to be robust. Using a more detailed structure for the score.
    prompt = (
        "You are a highly skilled information extraction and relevancy scoring assistant.\n"
        f"Your primary task is to analyze the following web page content based on the user's query: '{extraction_query}'.\n\n"
        "CRITICAL OUTPUT FORMAT REQUIREMENT:\n"
        "1. The VERY FIRST line of your response MUST be the relevancy score in the exact format: 'Relevancy Score: X/5' (where X is a digit from 1 to 5).\n"
        "2. After the 'Relevancy Score:' line, starting on a NEW LINE, present ALL extracted information that directly answers or relates to the query '{extraction_query}'. This extracted information MUST be in PLAIN TEXT.\n"
        "3. If no relevant information is found, your output must be:\n"
        "   Relevancy Score: 1/5\n"
        "   No distinct pieces of information related to \"{extraction_query}\" were found in the provided text.\n\n"
        "GUIDELINES FOR RELEVANCY SCORE (X/5):\n"
        "- **Relevancy Score: 5/5** - Awarded if you find 5 or more distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 4/5** - Awarded if you find exactly 4 distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 3/5** - Awarded if you find 1, 2, or 3 distinct pieces of information directly related to '{extraction_query}'.\n"
        "- **Relevancy Score: 1/5** - Awarded if you find NO distinct pieces of information directly related to '{extraction_query}'.\n\n"
        f"Remember the plain text instruction for the extracted content: {plain_text_instruction_for_extraction}\n\n"
        "--- WEB PAGE CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- WEB PAGE CONTENT END ---\n\n"
        f"Response (strictly following format: 'Relevancy Score: X/5' on first line, then extracted info) regarding '{extraction_query}':"
    )
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 750, "temperature": 0.3})

def _parse_score_and_get_content(text_with_potential_score: Optional[str]) -> Tuple[Optional[int], str]:
    """
    Parses relevancy score and extracts content from LLM output.
    Expects score on the first line like "Relevancy Score: X/5".
    Uses regex for more robust parsing.

    Args:
        text_with_potential_score: The string output from the LLM, potentially
                                   containing the score and extracted information.

    Returns:
        A tuple containing:
            - Optional[int]: The parsed relevancy score (1-5), or None if not found/parsed.
            - str: The extracted content part of the text. If no score is parsed,
                   this will be the original input string. If a score is parsed,
                   this will be the text after the score line.
    """
    score: Optional[int] = None
    content_text: str = text_with_potential_score if text_with_potential_score else ""

    if not text_with_potential_score:
        print("LLM_PROCESSOR_DEBUG (parse_score): Input text is None or empty.")
        return score, "" # Return empty string for content if input is None

    # Regex to find "Relevancy Score: D/5" possibly with whitespace, and capture D and the rest.
    # Handles score at the beginning of the string, possibly after some leading/trailing whitespace on the score line itself.
    # re.DOTALL makes . match newlines for capturing the rest of the content.
    match = re.match(r"^\s*Relevancy Score:\s*(\d)\s*/\s*5\s*\n?(.*)", text_with_potential_score, re.DOTALL)

    if match:
        try:
            score = int(match.group(1))
            content_text = match.group(2).strip()
            print(f"LLM_PROCESSOR_DEBUG (parse_score): Regex Matched! Score: {score}, Content Preview: '{content_text[:70]}...'")
        except (IndexError, ValueError) as e_parse:
            score = None 
            content_text = text_with_potential_score # Fallback to original if parsing group fails
            print(f"LLM_PROCESSOR_DEBUG (parse_score): Regex matched but group parsing failed. Error: {e_parse}. Original text: '{text_with_potential_score[:100]}...'")
    else:
        # Fallback: Try a simpler check if regex fails (e.g. LLM provides score but no newline after it)
        # This is less robust but can catch some edge cases missed by a strict regex.
        print(f"LLM_PROCESSOR_DEBUG (parse_score): Primary regex FAILED. Text: '{text_with_potential_score[:100]}...'")
        if text_with_potential_score.strip().startswith("Relevancy Score:"):
             # Attempt to parse score from the first line if it looks like a score line
            first_line = text_with_potential_score.strip().split('\n', 1)[0]
            score_match_simple = re.search(r"Relevancy Score:\s*(\d)\s*/\s*5", first_line)
            if score_match_simple:
                try:
                    score = int(score_match_simple.group(1))
                    # Try to get content after the first line
                    if '\n' in text_with_potential_score.strip():
                        content_text = text_with_potential_score.strip().split('\n', 1)[1].strip()
                    else: # Score might be the only thing on the line, or LLM missed newline
                        content_text = "" # Or, text_with_potential_score.replace(first_line, "").strip()
                    print(f"LLM_PROCESSOR_DEBUG (parse_score): Fallback startswith + regex search matched! Score: {score}")
                except (IndexError, ValueError):
                    score = None # Reset if fallback parsing fails
                    content_text = text_with_potential_score # Revert to full text
                    print(f"LLM_PROCESSOR_DEBUG (parse_score): Fallback parsing failed after initial match.")
            # else:
                # If even the simple startswith fails to find a parsable score, score remains None
                # and content_text remains the original text_with_potential_score
    
    return score, content_text.strip()


@st.cache_data(max_entries=20, ttl=1800, show_spinner=False)
def generate_consolidated_summary(
    # ... (parameters as in v1.9.9) ...
    summaries: Tuple[Optional[str], ...],
    topic_context: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 150000,
    extraction_query_for_consolidation: Optional[str] = None,
    secondary_query_for_enrichment: Optional[str] = None
) -> Optional[str]:
    """
    Generates a consolidated summary with a narrative part and a TLDR section.
    Narrative is plain text with paragraphs separated by blank lines.
    TLDR section uses dash-prefixed key points, each on a new line.
    """
    # --- Function content from v1.9.9 remains unchanged ---
    if not summaries: return "LLM_PROCESSOR: No individual LLM outputs provided for consolidation."
    if not configure_gemini(api_key): return "LLM_PROCESSOR Error: Gemini not configured for consolidated summary."

    texts_for_llm_input: List[str] = []
    is_primary_focused_consolidation_active = bool(extraction_query_for_consolidation and extraction_query_for_consolidation.strip())
    is_secondary_enrichment_active = bool(secondary_query_for_enrichment and secondary_query_for_enrichment.strip())

    for item_text in summaries:
        if not item_text: continue
        lower_item_text = item_text.lower()
        # More robust check for error/info prefixes from LLM calls
        if any(lower_item_text.startswith(prefix) for prefix in [
            "llm error", "no text content", "llm_processor: no text content",
            "llm_processor error:", "llm_processor: no extraction query",
            "llm_processor: response blocked", "llm_processor info:" 
        ]):
            continue
        texts_for_llm_input.append(item_text)

    if not texts_for_llm_input:
        if is_primary_focused_consolidation_active:
            return (f"LLM_PROCESSOR_INFO: No suitable content found for query: '{extraction_query_for_consolidation}'. Cannot generate focused overview.")
        else:
            return "LLM_PROCESSOR_INFO: No valid LLM outputs found to consolidate."

    combined_texts = "\n\n---\n\n".join([f"Source Snippet {i+1}:\n{text_entry}" for i, text_entry in enumerate(texts_for_llm_input)])
    truncated_combined_text = _truncate_text_for_gemini(combined_texts, model_name, max_input_chars)

    prompt_instruction: str
    max_tokens_for_call = 800

    narrative_plain_text_instruction = (
        "For the main narrative overview that you generate, it MUST be in PLAIN TEXT. "
        "This means no markdown formatting like bolding, italics, headers, or lists using '*', '#', etc. "
        "Paragraphs should be separated by a blank line (i.e., two newline characters)."
    )
    tldr_specific_instruction = (
        "\n\nAfter completing the comprehensive narrative overview, create a distinct section titled 'TLDR:'. "
        "Under this 'TLDR:' title, list the 3-5 most critical key points from your narrative summary. "
        "Each key point MUST start on a new line and be prefixed with a dash and a single space (e.g., '- This is a key point.'). "
        "Ensure there is a clear visual separation (like one or two blank lines, or '---' on its own line) before the 'TLDR:' title."
    )
    final_tldr_emphasis = "\n\nIMPORTANT FINAL STEP: You absolutely MUST include the 'TLDR:' section as described, with dash-prefixed key points, after the main narrative."

    if is_primary_focused_consolidation_active:
        central_query_text = extraction_query_for_consolidation
        enrichment_instruction_text = ""
        if is_secondary_enrichment_active:
            enrichment_instruction_text = (
                f"\nFurthermore, consider insights related to a secondary query: '{secondary_query_for_enrichment}'. "
                f"Where relevant, integrate these secondary insights to enrich the primary overview on '{central_query_text}', "
                f"adding nuance and depth, especially where they complement or expand upon the primary findings."
            )
        prompt_instruction = (
            f"You are an expert research analyst. Based on the provided text snippets (LLM extraction results, ignore 'Relevancy Score' lines for summarization), "
            f"your primary task is to synthesize a detailed and comprehensive consolidated **narrative overview**. This narrative should deeply explore insights related to the central query: '{central_query_text}'.{enrichment_instruction_text}\n"
            f"{narrative_plain_text_instruction}\n"
            "Construct this narrative overview by:\n"
            "1. Identifying key findings, arguments, and data points relevant to the central query.\n"
            "2. Weaving this information into a cohesive narrative.\n"
            "3. Highlighting significant patterns or discrepancies.\n"
            "4. Elaborating on implications concerning the central query.\n"
            "A longer, more elaborate narrative (3-5 substantial paragraphs or more) is strongly preferred. "
            "Focus predominantly on the central query. Avoid generic phrases. "
            f"Begin the narrative directly.{tldr_specific_instruction}{final_tldr_emphasis}\n\n"
        )
        max_tokens_for_call = 2048 
        final_topic_context_for_prompt = central_query_text
    else: # General consolidation
        final_topic_context_for_prompt = topic_context
        prompt_instruction = (
            f"You are an expert analyst. Based on the provided text snippets (general summaries or extractions), "
            f"your task is to first create a single, coherent consolidated **narrative overview** broadly related to the topic: '{final_topic_context_for_prompt}'.\n"
            f"{narrative_plain_text_instruction}\n"
            "Identify main themes and key information. Synthesize into a cohesive narrative. "
            "Highlight notable patterns. Present in well-structured paragraphs (2-4 substantial paragraphs)."
            f"{tldr_specific_instruction}{final_tldr_emphasis}\n\n"
        )
        max_tokens_for_call = 1024

    prompt = (f"{prompt_instruction}--- PROVIDED TEXTS START ---\n{truncated_combined_text}\n--- PROVIDED TEXTS END ---\n\nConsolidated Overview and TLDR (focused on '{final_topic_context_for_prompt}' if applicable):")
    
    # Removed result_prefix, app.py should handle context for display if needed.
    # result_prefix = ""
    # if not is_primary_focused_consolidation_active: result_prefix = "LLM_PROCESSOR_INFO: General overview as follows.\n"

    llm_response = _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": max_tokens_for_call, "temperature": 0.35})

    if llm_response and not llm_response.lower().startswith("llm error") and not llm_response.lower().startswith("llm_processor error"):
        if "TLDR:" in llm_response:
            parts = llm_response.split("TLDR:", 1)
            narrative_part = parts[0].rstrip() 
            tldr_content_raw = ""
            if len(parts) == 2:
                tldr_content_raw = parts[1].strip()

            if tldr_content_raw:
                llm_response = narrative_part + "\n\nTLDR:\n" + tldr_content_raw
            else: 
                 llm_response = narrative_part + "\n\nTLDR:\n(No key points provided by LLM)"
        else:
            print("LLM_PROCESSOR_WARNING: Consolidated summary: 'TLDR:' section title missing from LLM output.")
        
        return llm_response # Return without the prefix.
    return llm_response


@st.cache_data(max_entries=50, ttl=3600, show_spinner=False)
def generate_search_queries(
    # ... (parameters as in v1.9.9) ...
    original_keywords: Tuple[str, ...],
    specific_info_query: Optional[str],
    specific_info_query_2: Optional[str],
    num_queries_to_generate: int,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 2500
) -> Optional[List[str]]:
    """Generates new search queries."""
    # --- Function content from v1.9.9 remains unchanged ---
    if not original_keywords: return None
    if num_queries_to_generate <= 0: return []
    if not configure_gemini(api_key): print("LLM_PROCESSOR Warning: Gemini not configured for generating search queries."); return None
    original_keywords_str = ", ".join(original_keywords)
    context_lines = [f"Original user search keywords: \"{original_keywords_str}\"."]
    has_q1 = specific_info_query and specific_info_query.strip()
    has_q2 = specific_info_query_2 and specific_info_query_2.strip()
    if has_q1: context_lines.append(f"User's primary information goal (Query 1): \"{specific_info_query.strip()}\".")
    if has_q2: context_lines.append(f"User's secondary information goal (Query 2): \"{specific_info_query_2.strip()}\".")
    if not has_q1 and not has_q2: context_lines.append("User has not provided any specific information goals beyond initial keywords.")
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
        "Provide your {num_queries_to_generate} new search queries below, each on a new line. "
        "Output only the raw search query strings, one per line. Do NOT use any numbering, bullets, quotation marks around the queries, or any other formatting."
    ).format(num_queries_to_generate=num_queries_to_generate, context=truncated_context)
    response_text = _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": num_queries_to_generate * 50, "temperature": 0.55} )
    if response_text and not response_text.startswith("LLM Error") and not response_text.startswith("LLM_PROCESSOR Error"):
        generated_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        generated_queries = [re.sub(r'^(["\'])(.*)\1$', r'\2', q) for q in generated_queries] # Remove surrounding quotes
        return generated_queries[:num_queries_to_generate]
    print(f"LLM_PROCESSOR Warning: Failed to generate search queries or error occurred. Response: {response_text}"); return None


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test (Google Gemini v1.9.10 - Score Parsing Test)") # Updated title
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password", key="main_api_key_test_llm_v1910")
    MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest", key="main_model_name_test_llm_v1910")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True):
            st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}.")
            if _AVAILABLE_MODELS_CACHE: st.write("Available Models (first 5):", _AVAILABLE_MODELS_CACHE[:5])
            configured_for_test = True
        else: st.error("Failed to configure Gemini for testing.")
    else: st.info("Enter Gemini API Key to enable tests.")

    if configured_for_test:
        # Test Generate Search Queries (Cached) - code unchanged
        with st.expander("Test Generate Search Queries (Cached)", expanded=False):
            # ... (content from v1.9.9) ...
            test_original_keywords_str_gsq = st.text_input("Original Keywords (comma-separated):", "home office setup, ergonomic chair", key="test_orig_kw_gsq_v1910")
            test_specific_info_query_gsq = st.text_input("Specific Info Goal (Q1 - Optional):", "best chair for back pain under $300", key="test_spec_info_q1_gsq_v1910")
            test_specific_info_query_2_gsq = st.text_input("Specific Info Goal (Q2 - Optional):", "desk height for standing", key="test_spec_info_q2_gsq_v1910")
            test_num_queries_to_gen_gsq = st.number_input("Number of New Queries to Generate:", min_value=1, max_value=10, value=2, key="test_num_q_gen_gsq_v1910")
            if st.button("Test Query Generation (Cached)", key="test_query_gen_button_gsq_v1910"):
                test_original_keywords_tuple = tuple(k.strip() for k in test_original_keywords_str_gsq.split(',') if k.strip())
                if not test_original_keywords_tuple: st.warning("Please enter original keywords.")
                else:
                    with st.spinner("Generating new search queries..."):
                        new_queries = generate_search_queries(
                            original_keywords=test_original_keywords_tuple,
                            specific_info_query=test_specific_info_query_gsq if test_specific_info_query_gsq.strip() else None,
                            specific_info_query_2=test_specific_info_query_2_gsq if test_specific_info_query_2_gsq.strip() else None,
                            num_queries_to_generate=test_num_queries_to_gen_gsq,
                            api_key=GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST
                        )
                    st.markdown("**Generated New Search Queries:**");
                    if new_queries is not None: st.json(new_queries)
                    else: st.write("No new queries generated or error.")


        # Test Individual Summary (Cached) - code unchanged
        with st.expander("Test Individual Summary (Cached)", expanded=False):
            # ... (content from v1.9.9) ...
            sample_text_summary_cached = st.text_area("Sample Text for Summary:", "The Eiffel Tower is a famous landmark in Paris, France.", height=100, key="test_sample_summary_text_v1910_cached")
            if st.button("Test Summary Generation (Cached)", key="test_summary_gen_button_v1910_cached"):
                 with st.spinner(f"Generating summary..."):
                    summary_output = generate_summary(sample_text_summary_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                 st.markdown("**Generated Summary (Plain Text Expected):**"); st.text(summary_output)


        # Test Specific Information Extraction (Cached) - Focus for scoring debug
        with st.expander("Test Specific Information Extraction (Cached & Score Parsing)", expanded=True): # Expanded for focus
            sample_text_extraction_cached = st.text_area("Sample Text for Extraction:", "The main product is WidgetPro and it costs $50. Contact sales@example.com for more details. Another feature is its durability. It is manufactured in Germany.", height=100, key="test_sample_extraction_text_v1910_cached")
            extraction_query_test_cached = st.text_input("Extraction Query:", "product name and contact email", key="test_extraction_query_input_v1910_cached")
            if st.button("Test Extraction & Score Parsing (Cached)", key="test_extraction_score_button_v1910_cached"):
                if not extraction_query_test_cached: st.warning("Please enter an extraction query.")
                else:
                    st.markdown("---")
                    st.write("1. Calling `extract_specific_information` (LLM Call):")
                    with st.spinner(f"Extracting info from LLM..."):
                        llm_raw_output = extract_specific_information(sample_text_extraction_cached, extraction_query_test_cached, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                    st.markdown("**LLM Raw Output:**"); st.text(llm_raw_output)
                    st.markdown("---")
                    st.write("2. Parsing with `_parse_score_and_get_content`:")
                    if llm_raw_output:
                        parsed_score, parsed_content = _parse_score_and_get_content(llm_raw_output)
                        st.markdown(f"**Parsed Score:** `{parsed_score}`")
                        st.markdown("**Parsed Content (Plain Text Expected):**"); st.text(parsed_content)
                    else:
                        st.warning("LLM Raw Output was None or empty, cannot parse.")
            st.markdown("---")
            st.subheader("Manual Test of `_parse_score_and_get_content`")
            manual_test_text_score = st.text_area("Text to manually test parsing:", "Relevancy Score: 3/5\nThis is some content.", key="manual_test_score_parse_v1910")
            if st.button("Manually Test Parser", key="manual_parse_button_v1910"):
                m_score, m_content = _parse_score_and_get_content(manual_test_text_score)
                st.write(f"Manual Parse -> Score: {m_score}, Content: '{m_content}'")


        # Test Consolidation (Narrative + TLDR) - code unchanged
        with st.expander("Test Consolidation (Narrative + TLDR)", expanded=False): # Collapsed by default now
            # ... (content from v1.9.9, ensure keys are unique if needed e.g. _v1910) ...
            st.write("Input texts. LLM should produce plain text narrative with paragraphs separated by blank lines, then 'TLDR:' with dash-prefixed points each on a new line.")
            cs_text1_v1910 = st.text_area("Text 1:", "Relevancy Score: 4/5\nSolar panel efficiency research focuses on perovskite materials. Polysilicon panels degrade at 0.5% annually.", height=80, key="cs_text1_v1910")
            cs_text2_v1910 = st.text_area("Text 2:", "Relevancy Score: 3/5\nInverters last 10-15 years. Maintenance is a cost factor.", height=80, key="cs_text2_v1910")
            cs_text3_v1910 = st.text_area("Text 3:", "Summary: Wind energy is another option with different challenges.", height=80, key="cs_text3_v1910")

            cs_topic_v1910 = st.text_input("General Topic Context:", "Renewable Energy Tech", key="cs_topic_v1910")
            cs_q1_v1910 = st.text_input("Q1 for Consolidation (Primary Focus):", "solar panel tech and performance", key="cs_q1_v1910")
            cs_q2_v1910 = st.text_input("Q2 for Enrichment (Secondary Context):", "solar system components and maintenance", key="cs_q2_v1910")

            if st.button("Test Focused Consolidation (Narrative + TLDR)", key="focused_cs_button_v1910"):
                if not cs_q1_v1910: st.warning("Please enter Q1 for primary focus.")
                else:
                    texts_cs = tuple(t for t in [cs_text1_v1910, cs_text2_v1910, cs_text3_v1910] if t and t.strip())
                    if not texts_cs: st.warning("Please provide at least one text.")
                    else:
                        with st.spinner("Generating focused summary..."):
                            output = generate_consolidated_summary(
                                summaries=texts_cs, topic_context=cs_topic_v1910,
                                api_key=GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST,
                                extraction_query_for_consolidation=cs_q1_v1910.strip(),
                                secondary_query_for_enrichment=cs_q2_v1910.strip() if cs_q2_v1910.strip() else None )
                        st.markdown("**Consolidated Output (Focused Narrative + TLDR):**"); st.markdown(output)

            if st.button("Test General Consolidation (Narrative + TLDR)", key="general_cs_button_v1910"):
                texts_cs = tuple(t for t in [cs_text1_v1910, cs_text2_v1910, cs_text3_v1910] if t and t.strip())
                if not texts_cs: st.warning("Please provide at least one text.")
                else:
                    with st.spinner("Generating general summary..."):
                         output = generate_consolidated_summary(
                            summaries=texts_cs, topic_context=cs_topic_v1910,
                            api_key=GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST,
                            extraction_query_for_consolidation=None, secondary_query_for_enrichment=None)
                    st.markdown("**Consolidated Output (General Narrative + TLDR):**"); st.markdown(output)

    else:
        st.warning("Gemini not configured. Provide API key for tests.")

# // end of modules/llm_processor.py
