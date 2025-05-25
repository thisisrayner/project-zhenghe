# modules/search_engine.py

from googleapiclient.discovery import build
import streamlit as st # For potential caching, though core logic is separate

# No direct st.secrets access here; API keys are passed in.

def perform_search(query: str, api_key: str, cse_id: str, num_results: int = 5, **kwargs) -> list[dict]:
    """
    Performs a Google Custom Search.

    Args:
        query (str): The search query.
        api_key (str): Your Google API key.
        cse_id (str): Your Custom Search Engine ID.
        num_results (int): Number of results to retrieve (max 10 per call for free tier).
        **kwargs: Additional parameters for the CSE API (e.g., siteSearch).

    Returns:
        list[dict]: A list of search result items (dictionaries),
                    or an empty list if an error occurs or no results.
                    Each item typically contains 'title', 'link', 'snippet'.
    """
    if not api_key or not cse_id:
        # This check should ideally be handled before calling,
        # e.g., by config loader ensuring keys exist.
        st.error("Google API Key or CSE ID is missing in search_engine.perform_search.")
        return []
    if num_results > 10:
        num_results = 10 # API allows max 10 for standard usage in one request

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results,
            # Add other parameters from kwargs if needed
            **kwargs
        ).execute()
        return result.get('items', []) # result['items'] can be missing if no results
    except Exception as e:
        # In a real app, log this error more robustly
        st.error(f"Google Search API error for query '{query}': {e}")
        return []

if __name__ == '__main__':
    # --- Simple Test for this module ---
    # To run this test, you would need to set your secrets.toml and run:
    # streamlit run modules/search_engine.py
    # This is a basic test; real testing should happen within the app.py context
    # or with dedicated unit tests.

    st.title("Search Engine Module Test")

    # Attempt to load config to get API keys for testing
    # This requires modules/config.py to be importable and working
    try:
        from config import load_config
        cfg = load_config()
        if cfg and cfg.google.api_key and cfg.google.cse_id:
            GOOGLE_API_KEY_TEST = cfg.google.api_key
            CSE_ID_TEST = cfg.google.cse_id
        else:
            st.error("Could not load valid Google API Key/CSE ID from config for testing.")
            GOOGLE_API_KEY_TEST = None
            CSE_ID_TEST = None
    except ImportError:
        st.error("Could not import config module for testing.")
        GOOGLE_API_KEY_TEST = None
        CSE_ID_TEST = None


    test_query = st.text_input("Enter test query:", "streamlit python")
    if st.button("Test Search"):
        if GOOGLE_API_KEY_TEST and CSE_ID_TEST:
            st.write(f"Performing search for: '{test_query}'")
            results = perform_search(test_query, GOOGLE_API_KEY_TEST, CSE_ID_TEST, num_results=3)
            if results:
                st.success(f"Found {len(results)} results:")
                for res in results:
                    st.markdown(f"- **[{res.get('title')}]({res.get('link')})**\n  {res.get('snippet')}")
            else:
                st.warning("No results found or an error occurred.")
        else:
            st.error("API Key or CSE ID not available for testing. Please check secrets.toml and config.py")
