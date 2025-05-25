# modules/llm_processor.py
# Version 1.5.1: Commented out model listing debug prints. Default model remains gemini-1.5-flash-latest.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List
import time
import random

_GEMINI_CONFIGURED = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None

def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
    global _GEMINI_CONFIGURED, _AVAILABLE_MODELS_CACHE
    if _GEMINI_CONFIGURED and not force_recheck_models and _AVAILABLE_MODELS_CACHE is not None:
        return True
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        _GEMINI_CONFIGURED = True
        if _AVAILABLE_MODELS_CACHE is None or force_recheck_models:
            # st.info("Attempting to list available Gemini models supporting 'generateContent'...") # COMMENTED
            # print("Attempting to list available Gemini models supporting 'generateContent'...") # COMMENTED
            current_available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        current_available_models.append(m.name)
                        # print(f"DEBUG: Found model: {m.name}") # COMMENTED
                        # st.caption(f"DEBUG: Found model: {m.name}") # COMMENTED
                _AVAILABLE_MODELS_CACHE = current_available_models
                if not _AVAILABLE_MODELS_CACHE:
                    st.warning("No Gemini models found supporting 'generateContent'. Check API key, project, and enabled services.")
                    return False
                # else:
                    # st.caption(f"Found {len(_AVAILABLE_MODELS_CACHE)} usable Gemini models.") # COMMENTED
            except Exception as e_list_models:
                st.error(f"Error listing Gemini models: {e_list_models}. API key might be invalid or service not enabled.")
                _AVAILABLE_MODELS_CACHE = []
                _GEMINI_CONFIGURED = False
                return False
        return True
    except Exception as e_configure:
        st.error(f"Failed to configure Google Gemini client with API key: {e_configure}")
        _GEMINI_CONFIGURED = False
        return False

# ... (Rest of _call_gemini_api, _truncate_text_for_gemini, generate_summary,
#      extract_specific_information, generate_consolidated_summary, and if __name__ == '__main__'
#      remain IDENTICAL to modules/llm_processor.py Version 1.5) ...
def _call_gemini_api(
    model_name: str, prompt_parts: list, generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[Dict[str, Any]] = None, max_retries: int = 3,
    initial_backoff_seconds: float = 5.0, max_backoff_seconds: float = 60.0
) -> Optional[str]:
    if not _GEMINI_CONFIGURED: return "Gemini client not configured."
    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE and f"models/{model_name}" not in _AVAILABLE_MODELS_CACHE:
            if not model_name.startswith("models/"):
                prefixed_attempt = f"models/{model_name}"
                if prefixed_attempt in _AVAILABLE_MODELS_CACHE: validated_model_name = prefixed_attempt
                else: return f"LLM Error: Model '{model_name}' (or '{prefixed_attempt}') not in available: {_AVAILABLE_MODELS_CACHE}"
            elif model_name not in _AVAILABLE_MODELS_CACHE: return f"LLM Error: Model '{model_name}' not in available: {_AVAILABLE_MODELS_CACHE}"
    try: model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init: return f"LLM Error: Could not initialize model '{validated_model_name}': {e_model_init}"
    effective_generation_config = {"temperature": 0.3, "max_output_tokens": 1024};
    if generation_config_args: effective_generation_config.update(generation_config_args)
    generation_config = genai.types.GenerationConfig(**effective_generation_config)
    safety_settings = safety_settings_args or [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    current_retry = 0; current_backoff = initial_backoff_seconds
    while current_retry <= max_retries:
        try:
            response = model_obj.generate_content(prompt_parts, generation_config=generation_config, safety_settings=safety_settings, stream=False)
            if not response.candidates:
                reason_message = "LLM response was blocked or empty."
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message: reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: reason_message += f" Blocked by safety category: {rating.category}."
                return reason_message
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or "resourceexhausted" in error_str.replace(" ", "") or "resource exhausted" in error_str or ("rate" in error_str and "limit" in error_str) or (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) )
            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1; print(f"LLM Rate limit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). Error: {str(e)[:100]}..."); time.sleep(current_backoff); current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}"
    return f"LLM Error: Max retries ({max_retries}) reached for '{validated_model_name}' due to persistent rate limiting."

def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    if len(text) > max_input_chars: return text[:max_input_chars]
    return text

def generate_summary(text_content: Optional[str], api_key: Optional[str], model_name: str = "models/gemini-1.5-flash-latest", max_input_chars: int = 100000) -> Optional[str]:
    if not text_content: return "No text content provided for summary."
    if not configure_gemini(api_key): return "Gemini LLM not configured for summary."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = ("You are an expert assistant specializing in creating detailed and insightful summaries of web page content.\nAnalyze the following text and provide a comprehensive summary of approximately 4-6 substantial sentences (or 2-3 short paragraphs if the content is rich). Your summary should capture the core message, key arguments, supporting details, and any significant conclusions or implications. Maintain a neutral and factual tone. Avoid introductory phrases like 'This text discusses...'. Go directly into the summary.\n\n--- WEB PAGE CONTENT START ---\n{truncated_text}\n--- WEB PAGE CONTENT END ---\n\nDetailed Summary:")
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 512})

def extract_specific_information(text_content: Optional[str], extraction_query: str, api_key: Optional[str], model_name: str = "models/gemini-1.5-flash-latest", max_input_chars: int = 100000) -> Optional[str]:
    if not text_content: return "No text content provided for extraction."
    if not extraction_query: return "No extraction query provided."
    if not configure_gemini(api_key): return "Gemini LLM not configured for extraction."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (f"You are a highly skilled information extraction assistant.\nCarefully review the following web page content. Your task is to extract information specifically related to: '{extraction_query}'.\nPresent your findings comprehensively and clearly. If the information or any part of it cannot be found in the text, explicitly state 'Information not found for [specific part of query]' or 'The requested information was not found in the provided text'. If multiple pieces of information are requested, address each one.\n\n--- WEB PAGE CONTENT START ---\n{truncated_text}\n--- WEB PAGE CONTENT END ---\n\nComprehensive Extracted Information regarding '{extraction_query}':")
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 700})

def generate_consolidated_summary(summaries: List[Optional[str]], topic_context: str, api_key: Optional[str], model_name: str = "models/gemini-1.5-flash-latest", max_input_chars: int = 150000) -> Optional[str]:
    if not summaries: return "No individual summaries provided for consolidation."
    valid_summaries = [s for s in summaries if s and not s.startswith("LLM Error") and not s.startswith("No text content")]
    if not valid_summaries: return "No valid individual summaries available to consolidate."
    if not configure_gemini(api_key): return "Gemini LLM not configured for consolidated summary."
    summary_entries = [f"Source Summary {i+1}:\n{s}" for i, s in enumerate(valid_summaries)]
    combined_summaries_text = "\n\n---\n\n".join(summary_entries)
    truncated_combined_text = _truncate_text_for_gemini(combined_summaries_text, model_name, max_input_chars)
    prompt = (f"You are an expert analyst synthesizing information from multiple sources related to '{topic_context}'.\nThe following are several summaries extracted from different web pages. Your task is to create a single, coherent consolidated overview. This overview should:\n1. Identify the main recurring themes or topics across the summaries.\n2. Synthesize key pieces of information, presenting a cohesive narrative.\n3. Highlight any notable patterns, unique insights, or significant discrepancies if they exist.\n4. The final output should be a well-structured overview, not just a list of points. Aim for a few comprehensive paragraphs.\n\n--- INDIVIDUAL SUMMARIES START ---\n{truncated_combined_text}\n--- INDIVIDUAL SUMMARIES END ---\n\nConsolidated Overview for '{topic_context}':")
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 800})

if __name__ == '__main__':
    st.set_page_config(layout="wide"); st.title("LLM Processor Module Test (Google Gemini v1.5.1 - Debug Commented)")
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password"); MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}."); configured_for_test = True
        else: st.error("Failed to configure Gemini for testing.")
    else: st.info("Enter Gemini API Key to enable tests.")
    sample_text_content_1 = st.text_area("Sample Text 1 for LLM:", "The sky is blue due to Rayleigh scattering.", height=100)
    sample_text_content_2 = st.text_area("Sample Text 2 for LLM:", "Oceans appear blue because water absorbs red light more than blue.", height=100)
    if configured_for_test:
        if st.button("Test Individual Summaries & Consolidation"):
            s1 = generate_summary(sample_text_content_1, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            s2 = generate_summary(sample_text_content_2, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            st.markdown("**Summary 1:**"); st.write(s1); st.markdown("**Summary 2:**"); st.write(s2)
            if s1 and s2:
                with st.spinner("Generating consolidated summary..."): consolidated = generate_consolidated_summary([s1, s2], "Color of Sky and Ocean", GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
                st.markdown("**Consolidated Summary:**"); st.write(consolidated)
# end of modules/llm_processor.py
