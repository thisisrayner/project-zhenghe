# modules/llm_processor.py (relevant function with Option A applied)
# Version 1.9.6a (or just update 1.9.6 if preferred)

# ... (imports and other functions remain the same) ...

@st.cache_data(max_entries=20, ttl=1800, show_spinner=False)
def generate_consolidated_summary(
    summaries: Tuple[Optional[str], ...],
    topic_context: str,
    api_key: Optional[str],
    model_name: str = "models/gemini-1.5-flash-latest",
    max_input_chars: int = 150000,
    extraction_query_for_consolidation: Optional[str] = None,
    secondary_query_for_enrichment: Optional[str] = None
) -> Optional[str]:
    # ... (initial checks and text processing same as before) ...
    if not summaries:
        return "LLM_PROCESSOR: No individual LLM outputs provided for consolidation."
    if not configure_gemini(api_key):
        return "LLM_PROCESSOR Error: Gemini not configured for consolidated summary."

    texts_for_llm_input: List[str] = []
    is_primary_focused_consolidation_active = bool(extraction_query_for_consolidation and extraction_query_for_consolidation.strip())
    is_secondary_enrichment_active = bool(secondary_query_for_enrichment and secondary_query_for_enrichment.strip())

    for item_text in summaries: 
        if not item_text: continue
        lower_item_text = item_text.lower()
        if lower_item_text.startswith(("llm error", "no text content", "llm_processor: no text content",
                                       "llm_processor error:", "llm_processor: no extraction query",
                                       "llm_processor: response blocked")):
            continue
        texts_for_llm_input.append(item_text)

    if not texts_for_llm_input:
        if is_primary_focused_consolidation_active:
            return (f"LLM_PROCESSOR_INFO: No suitable content found from items matching relevancy criteria for query: '{extraction_query_for_consolidation}'. Cannot generate focused overview.")
        else:
            return "LLM_PROCESSOR_INFO: No valid LLM outputs found to consolidate into a general overview."

    combined_texts = "\n\n---\n\n".join([f"Source Document Content Snippet {i+1}:\n{text_entry}" for i, text_entry in enumerate(texts_for_llm_input)])
    truncated_combined_text = _truncate_text_for_gemini(combined_texts, model_name, max_input_chars)

    prompt_instruction: str
    max_tokens_for_call = 800

    narrative_plain_text_instruction = (
        "For the main narrative overview that you generate, it MUST be in PLAIN TEXT. "
        "This means no markdown formatting like bolding, italics, headers, or lists using '*', '#', etc. "
        "Paragraphs should be separated by a single newline character."
        # Removed: "However, for the 'TLDR:' section that follows, you will use dashes."
        # Let the tldr_specific_instruction handle its own format details.
    )

    tldr_specific_instruction = (
        "\n\nAfter completing the comprehensive narrative overview, create a distinct section titled 'TLDR:'. "
        "Under this 'TLDR:' title, list the 3-5 most critical key points from your narrative summary. "
        "Each key point MUST start on a new line and be prefixed with a dash and a single space (e.g., '- This is a key point.'). "
        "Ensure there is a clear visual separation (like one or two blank lines, or '---' on its own line) before the 'TLDR:' title."
    )
    
    # Option A: Emphasize TLDR requirement again at the end of main instructions.
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
            f"You are an expert research analyst. Based on the provided text snippets (which are LLM extraction results, possibly including 'Relevancy Score' lines that you should ignore for summarization), "
            f"your primary task is to synthesize a detailed and comprehensive consolidated **narrative overview**. This narrative should deeply explore insights related to the central query: '{central_query_text}'.{enrichment_instruction_text}\n"
            f"{narrative_plain_text_instruction}\n"
            "Construct this narrative overview by:\n"
            "1. Identifying key findings, arguments, and data points in the snippets relevant to the central query.\n"
            "2. Weaving this information into a cohesive narrative that thoroughly explains what was found regarding the query.\n"
            "3. Highlighting significant patterns, corroborating evidence, or discrepancies.\n"
            "4. Elaborating on implications or main takeaways concerning the central query.\n"
            "A longer, more elaborate narrative (e.g., 3-5 substantial paragraphs) is strongly preferred. "
            "Focus predominantly on the central query. Avoid generic phrases. "
            f"Begin the narrative directly.{tldr_specific_instruction}{final_tldr_emphasis}\n\n" # Added emphasis
        )
        max_tokens_for_call = 2048
        final_topic_context_for_prompt = central_query_text
    else: # General consolidation
        final_topic_context_for_prompt = topic_context
        prompt_instruction = (
            f"You are an expert analyst. Based on the provided text snippets (which are general summaries or extractions), "
            f"your task is to first create a single, coherent consolidated **narrative overview** broadly related to the topic: '{final_topic_context_for_prompt}'.\n"
            f"{narrative_plain_text_instruction}\n"
            "Identify main themes, arguments, and key information across the texts. Synthesize these into a cohesive narrative. "
            "Highlight notable patterns or discrepancies. Present in well-structured paragraphs (e.g., 2-4 substantial paragraphs)."
            f"{tldr_specific_instruction}{final_tldr_emphasis}\n\n" # Added emphasis
        )
        max_tokens_for_call = 1024

    prompt = (f"{prompt_instruction}--- PROVIDED TEXTS START ---\n{truncated_combined_text}\n--- PROVIDED TEXTS END ---\n\nConsolidated Overview and TLDR (focused on '{final_topic_context_for_prompt}' if applicable):")

    result_prefix = ""
    if not is_primary_focused_consolidation_active:
        result_prefix = "LLM_PROCESSOR_INFO: General overview as follows.\n"

    llm_response = _call_gemini_api(
        model_name,
        [prompt],
        generation_config_args={"max_output_tokens": max_tokens_for_call, "temperature": 0.35}
    )
    if llm_response and not llm_response.lower().startswith("llm error") and not llm_response.lower().startswith("llm_processor error"):
        # Simple check to see if TLDR was likely generated before returning
        if "TLDR:" not in llm_response:
            print("LLM_PROCESSOR_WARNING: Consolidated summary generated, but 'TLDR:' section seems to be missing from LLM output.")
            # Decide if you want to append a warning to the output or just log it.
            # For now, just log and return the LLM's output as is.
        return result_prefix + llm_response
    return llm_response

# ... (rest of the file, including the test block which you should use to verify this)
