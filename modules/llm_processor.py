# modules/llm_processor.py
# Version 1.6.1: Added aggressive debugging for the exact prompt sent to Gemini.

import google.generativeai as genai
import streamlit as st
from typing import Optional, Dict, Any, List
import time
import random

# ... (configure_gemini function remains the same as v1.5.1 - with debug prints commented) ...
_GEMINI_CONFIGURED = False
_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None
def configure_gemini(api_key: Optional[str], force_recheck_models: bool = False) -> bool:
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
                if not _AVAILABLE_MODELS_CACHE: st.warning("No Gemini models found supporting 'generateContent'."); return False
            except Exception as e_list_models: st.error(f"Error listing Gemini models: {e_list_models}."); _AVAILABLE_MODELS_CACHE = []; _GEMINI_CONFIGURED = False; return False
        return True
    except Exception as e_configure: st.error(f"Failed to configure Google Gemini client: {e_configure}"); _GEMINI_CONFIGURED = False; return False

def _call_gemini_api(
    model_name: str,
    prompt_parts: list, # This should be a list containing the full prompt string(s)
    generation_config_args: Optional[Dict[str, Any]] = None,
    safety_settings_args: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    initial_backoff_seconds: float = 5.0,
    max_backoff_seconds: float = 60.0
) -> Optional[str]:

    if not _GEMINI_CONFIGURED: return "Gemini client not configured."
    # ... (validated_model_name logic from v1.5.1) ...
    validated_model_name = model_name
    if _AVAILABLE_MODELS_CACHE is not None:
        if model_name not in _AVAILABLE_MODELS_CACHE and f"models/{model_name}" not in _AVAILABLE_MODELS_CACHE:
            if not model_name.startswith("models/"): prefixed_attempt = f"models/{model_name}"
            else: prefixed_attempt = model_name # Should not happen if already prefixed and not found
            if prefixed_attempt in _AVAILABLE_MODELS_CACHE: validated_model_name = prefixed_attempt
            else: return f"LLM Error: Model '{model_name}' (or '{prefixed_attempt}') not in available: {_AVAILABLE_MODELS_CACHE}"
        elif model_name not in _AVAILABLE_MODELS_CACHE and model_name.startswith("models/"): # Already prefixed, but still not found
                return f"LLM Error: Model '{model_name}' not in available: {_AVAILABLE_MODELS_CACHE}"

    try: model_obj = genai.GenerativeModel(validated_model_name)
    except Exception as e_model_init: return f"LLM Error: Could not initialize model '{validated_model_name}': {e_model_init}"
    
    effective_generation_config = {"temperature": 0.3, "max_output_tokens": 1024};
    if generation_config_args: effective_generation_config.update(generation_config_args)
    generation_config = genai.types.GenerationConfig(**effective_generation_config)
    safety_settings = safety_settings_args or [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    
    current_retry = 0; current_backoff = initial_backoff_seconds

    # --- AGGRESSIVE DEBUG FOR PROMPT PARTS ---
    print(f"\n--- DEBUG: Calling _call_gemini_api for model: {validated_model_name} ---")
    if isinstance(prompt_parts, list) and len(prompt_parts) > 0:
        full_prompt_for_debug = str(prompt_parts[0]) # Assuming the main prompt is the first part
        print(f"PROMPT BEING SENT (first 1000 chars of first part):\n{full_prompt_for_debug[:1000]}\n[...]")
        if len(full_prompt_for_debug) > 1000:
            print(f"PROMPT BEING SENT (last 500 chars of first part):\n[...]\n{full_prompt_for_debug[-500:]}")
        print(f"TOTAL LENGTH of first prompt part: {len(full_prompt_for_debug)}")
    else:
        print(f"PROMPT_PARTS is not a valid list or is empty: {prompt_parts}")
    print(f"Generation Config: {generation_config}")
    print(f"Safety Settings: {safety_settings}")
    # --- END AGGRESSIVE DEBUG ---

    while current_retry <= max_retries:
        try:
            response = model_obj.generate_content(
                prompt_parts, # This is what's sent
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False
            )
            if not response.candidates:
                reason_message = "LLM response was blocked or empty."
                # ... (detailed reason_message construction) ...
                if response.prompt_feedback:
                    reason_message += f" Reason: {response.prompt_feedback.block_reason}."
                    if response.prompt_feedback.block_reason_message: reason_message += f" Message: {response.prompt_feedback.block_reason_message}"
                    if response.prompt_feedback.safety_ratings:
                        for rating in response.prompt_feedback.safety_ratings:
                            if rating.blocked: reason_message += f" Blocked by safety category: {rating.category}."
                print(f"LLM_PROCESSOR: Response blocked. Reason: {reason_message}") # CONSOLE PRINT
                return reason_message
            
            print(f"LLM_PROCESSOR: Received response text (first 100): {response.text[:100] if response.text else 'EMPTY_RESPONSE_TEXT'}") # CONSOLE PRINT
            return response.text 

        except Exception as e:
            # ... (rate limit check and retry logic from v1.5) ...
            error_str = str(e).lower()
            is_rate_limit_error = ("429" in error_str or "resourceexhausted" in error_str.replace(" ", "") or "resource exhausted" in error_str or ("rate" in error_str and "limit" in error_str) or (hasattr(e, 'details') and callable(e.details) and "Quota" in e.details()) or (hasattr(e, 'code') and callable(e.code) and e.code().value == 8) )
            if is_rate_limit_error and current_retry < max_retries:
                current_retry += 1; print(f"LLM Rate limit for '{validated_model_name}'. Retrying in {current_backoff:.1f}s (Attempt {current_retry}/{max_retries}). Error: {str(e)[:100]}..."); time.sleep(current_backoff); current_backoff = min(current_backoff * 2 + random.uniform(0, 1.0), max_backoff_seconds)
            else: print(f"LLM_PROCESSOR: Unrecoverable error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}"); return f"LLM Error for '{validated_model_name}': {e.__class__.__name__} - {str(e)[:500]}"
    
    return f"LLM Error: Max retries ({max_retries}) reached for '{validated_model_name}' due to persistent rate limiting."


# ... (_truncate_text_for_gemini, generate_summary, extract_specific_information, generate_consolidated_summary
#      and if __name__ == '__main__' block remain IDENTICAL to modules/llm_processor.py Version 1.6) ...
# --- Text Truncation (Basic) ---
def _truncate_text_for_gemini(text: str, model_name: str, max_input_chars: int) -> str:
    if len(text) > max_input_chars: return text[:max_input_chars]
    return text

# --- Public Functions for App ---
def generate_summary(text_content: Optional[str], api_key: Optional[str], model_name: str = "models/gemini-1.5-flash-latest", max_input_chars: int = 100000) -> Optional[str]:
    if not text_content: return "No text content provided for summary."
    if not configure_gemini(api_key): return "Gemini LLM not configured for summary."
    truncated_text = _truncate_text_for_gemini(text_content, model_name, max_input_chars)
    prompt = (f"You are an expert assistant specializing in creating detailed and insightful summaries of web page content.\nAnalyze the following text and provide a comprehensive summary of approximately 4-6 substantial sentences (or 2-3 short paragraphs if the content is rich). Your summary should capture the core message, key arguments, supporting details, and any significant conclusions or implications. Maintain a neutral and factual tone. Avoid introductory phrases like 'This text discusses...'. Go directly into the summary.\n\n--- WEB PAGE CONTENT START ---\n{truncated_text}\n--- WEB PAGE CONTENT END ---\n\nDetailed Summary:")
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
    # --- TEMPORARY DEBUG from previous step - can be removed if the one in _call_gemini_api is sufficient ---
    # print(f"DEBUG LLM_PROCESSOR (generate_consolidated_summary): Attempting consolidated summary for topic: {topic_context}")
    # print(f"DEBUG LLM_PROCESSOR (generate_consolidated_summary): Number of valid summaries received: {len(valid_summaries)}")
    # print(f"DEBUG LLM_PROCESSOR (generate_consolidated_summary): Combined text BEFORE truncation (first 500 chars): {combined_summaries_text[:500]}")
    # print(f"DEBUG LLM_PROCESSOR (generate_consolidated_summary): Final truncated_combined_text being sent to LLM (first 500 chars):\n{truncated_combined_text[:500]}")
    # print("--- END DEBUG LLM_PROCESSOR (generate_consolidated_summary) ---")
    # --- END TEMPORARY DEBUG ---
    prompt = (f"You are an expert analyst. The following are several summaries of web pages related to the keyword '{topic_context}'.\nPlease synthesize these summaries into a single, coherent consolidated overview. Identify the main themes, key pieces of information, and any notable patterns or discrepancies across the sources. The consolidated overview should be comprehensive yet concise.\n\n--- INDIVIDUAL SUMMARIES START ---\n{truncated_combined_text}\n--- INDIVIDUAL SUMMARIES END ---\n\nConsolidated Overview for '{topic_context}':")
    return _call_gemini_api(model_name, [prompt], generation_config_args={"max_output_tokens": 800})

if __name__ == '__main__':
    st.set_page_config(layout="wide"); st.title("LLM Processor Module Test (Google Gemini v1.6.1 - Deep Debug)")
    GEMINI_API_KEY_TEST = st.text_input("Enter Gemini API Key for testing:", type="password"); MODEL_NAME_TEST = st.text_input("Gemini Model Name for testing:", "models/gemini-1.5-flash-latest")
    configured_for_test = False
    if GEMINI_API_KEY_TEST:
        if configure_gemini(GEMINI_API_KEY_TEST, force_recheck_models=True): st.success(f"Gemini configured for testing with model: {MODEL_NAME_TEST}."); configured_for_test = True
        else: st.error("Failed to configure Gemini for testing.")
    else: st.info("Enter Gemini API Key to enable tests.")
    sample_text_content_1 = st.text_area("Sample Text 1 for LLM:", "The sky is blue due to Rayleigh scattering. This scattering affects electromagnetic radiation whose wavelength is longer than the scattering particles.", height=100)
    if configured_for_test:
        if st.button("Test Individual Summary (Gemini)"):
            with st.spinner(f"Generating summary with {MODEL_NAME_TEST}..."): summary = generate_summary(sample_text_content_1, GEMINI_API_KEY_TEST, model_name=MODEL_NAME_TEST)
            st.markdown("**Summary 1:**"); st.write(summary)
# end of modules/llm_processor.py
