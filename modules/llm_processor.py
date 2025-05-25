# modules/llm_processor.py
# Version 1.0: Initial implementation for OpenAI summarization and specific info extraction.

import openai
import streamlit as st # For caching the client and displaying errors if run directly
from typing import Optional, Dict, Any # For type hinting

# --- OpenAI Client Initialization (Cached) ---
@st.cache_resource # Cache the OpenAI client resource
def get_openai_client(api_key: Optional[str]) -> Optional[openai.OpenAI]:
    """
    Initializes and returns an OpenAI client if the API key is provided.
    Caches the client instance for reuse.
    """
    if not api_key:
        # st.warning("OpenAI API Key not provided. LLM features will be disabled.") # Handled in app.py
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        # Test the client with a very small request (optional, but good for early error detection)
        # client.models.list() # This call can verify authentication
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

# --- Core LLM Interaction Function ---
def _call_openai_api(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens_response: int = 300, # Max tokens for the LLM's response
    temperature: float = 0.3
) -> Optional[str]:
    """
    Helper function to make a call to the OpenAI Chat Completions API.
    """
    if not client:
        return "OpenAI client not available."

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens_response,
            temperature=temperature,
            # top_p=1.0,
            # frequency_penalty=0.0,
            # presence_penalty=0.0
        )
        response_text = completion.choices[0].message.content
        if response_text:
            return response_text.strip()
        return "LLM returned an empty response."
    except openai.APIConnectionError as e:
        st.error(f"OpenAI APIConnectionError: {e}")
        return f"LLM Error: API Connection Problem ({e.__class__.__name__})"
    except openai.RateLimitError as e:
        st.error(f"OpenAI RateLimitError: {e}. You might be exceeding your quota or rate limits.")
        return f"LLM Error: Rate Limit Exceeded ({e.__class__.__name__})"
    except openai.AuthenticationError as e:
        st.error(f"OpenAI AuthenticationError: {e}. Check your API key.")
        return f"LLM Error: Authentication Failed ({e.__class__.__name__})"
    except openai.APIStatusError as e: # Catches other API errors e.g. 400, 500
        st.error(f"OpenAI APIStatusError: Status {e.status_code} - {e.message}")
        return f"LLM Error: API Status {e.status_code} ({e.__class__.__name__})"
    except Exception as e:
        st.error(f"An unexpected error occurred with OpenAI API: {e}")
        return f"LLM Error: Unexpected ({e.__class__.__name__})"


# --- Text Truncation (Basic) ---
def _truncate_text(text: str, max_chars: int) -> str:
    """
    Basic truncation of text to a maximum number of characters.
    A proper solution would use tiktoken to count tokens.
    """
    if len(text) > max_chars:
        # st.caption(f"LLM Input: Text truncated from {len(text)} to {max_chars} characters.")
        return text[:max_chars]
    return text

# --- Public Functions for App ---
def generate_summary(
    text_content: Optional[str],
    api_key: Optional[str],
    model: str = "gpt-3.5-turbo",
    max_input_chars: int = 8000 # Approx 2000 tokens for gpt-3.5-turbo (4 chars/token)
) -> Optional[str]:
    """
    Generates a summary for the given text content using OpenAI.
    """
    if not text_content:
        return "No text content provided for summary."
    if not api_key:
        return "OpenAI API Key not configured for summary."

    client = get_openai_client(api_key)
    if not client:
        return "OpenAI client initialization failed for summary."

    truncated_text = _truncate_text(text_content, max_input_chars)

    system_prompt = "You are a helpful assistant designed to provide concise summaries."
    user_prompt = (
        "Please provide a concise summary (around 2-4 sentences) of the following web page content. "
        "Focus on the main topics and key takeaways.\n\n"
        "--- CONTENT START ---\n"
        f"{truncated_text}\n"
        "--- CONTENT END ---\n\n"
        "Summary:"
    )
    return _call_openai_api(client, system_prompt, user_prompt, model, max_tokens_response=150)


def extract_specific_information(
    text_content: Optional[str],
    extraction_query: str,
    api_key: Optional[str],
    model: str = "gpt-3.5-turbo",
    max_input_chars: int = 8000
) -> Optional[str]:
    """
    Extracts specific information based on a query from the text content using OpenAI.
    """
    if not text_content:
        return "No text content provided for extraction."
    if not extraction_query:
        return "No extraction query provided."
    if not api_key:
        return "OpenAI API Key not configured for extraction."

    client = get_openai_client(api_key)
    if not client:
        return "OpenAI client initialization failed for extraction."

    truncated_text = _truncate_text(text_content, max_input_chars)

    system_prompt = (
        "You are an intelligent assistant skilled at finding specific information within a given text. "
        "If the requested information is not found, clearly state 'Information not found'."
    )
    user_prompt = (
        f"From the following text, please extract information related to: '{extraction_query}'.\n"
        "Present the findings clearly. If specific pieces of data are requested (e.g., names, dates, numbers), list them. "
        "If the information is not present, state 'Information not found' for that specific part or overall.\n\n"
        "--- TEXT START ---\n"
        f"{truncated_text}\n"
        "--- TEXT END ---\n\n"
        f"Extracted information regarding '{extraction_query}':"
    )
    return _call_openai_api(client, system_prompt, user_prompt, model, max_tokens_response=250)


if __name__ == '__main__':
    # --- Simple Test for this module ---
    st.set_page_config(layout="wide")
    st.title("LLM Processor Module Test")

    # This requires OPENAI_API_KEY in your .streamlit/secrets.toml
    # Attempt to load config to get API key for testing
    try:
        from config import load_config
        cfg = load_config()
        if cfg and cfg.openai.api_key:
            OPENAI_API_KEY_TEST = cfg.openai.api_key
            st.success("OpenAI API Key loaded from config.")
        else:
            st.error("Could not load OpenAI API Key from config for testing. Please ensure it's in secrets.toml.")
            OPENAI_API_KEY_TEST = None
    except ImportError:
        st.error("Could not import config module for testing.")
        OPENAI_API_KEY_TEST = None
    except Exception as e:
        st.error(f"Error loading config: {e}")
        OPENAI_API_KEY_TEST = None


    sample_text = st.text_area("Sample Text for LLM:", """
    Streamlit is an open-source Python library that makes it easy to create and share beautiful,
    custom web apps for machine learning and data science. In just a few minutes you can build
    and deploy powerful data apps. Version 1.0 was released in 2019. The current CEO is Adrien Treuille.
    Key features include interactive widgets, easy deployment, and a vibrant community.
    The company is based in San Francisco. For support, email support@streamlit.io.
    """, height=200)

    if OPENAI_API_KEY_TEST:
        st.subheader("Test Summary Generation")
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(sample_text, OPENAI_API_KEY_TEST)
            st.markdown("**Summary:**")
            st.write(summary)

        st.subheader("Test Specific Information Extraction")
        query = st.text_input("Extraction Query:", "CEO name and support email")
        if st.button("Extract Information"):
            if query:
                with st.spinner("Extracting information..."):
                    extracted_info = extract_specific_information(sample_text, query, OPENAI_API_KEY_TEST)
                st.markdown(f"**Extracted Info for '{query}':**")
                st.write(extracted_info)
            else:
                st.warning("Please enter an extraction query.")
    else:
        st.warning("OpenAI API Key not available. Cannot run LLM tests.")

# end of modules/llm_processor.py
