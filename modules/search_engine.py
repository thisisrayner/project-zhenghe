# modules/search_engine.py
# Version 1.1: Enhanced docstrings and type hinting.

"""
Handles interactions with the Google Custom Search API.

This module provides functionality to perform searches using specified
keywords, API key, and Custom Search Engine (CSE) ID.
It uses the google-api-python-client library.
"""

from googleapiclient.discovery import build
# Removed st from here as it's not strictly needed for core logic, errors can be raised/returned
# import streamlit as st
from typing import List, Dict, Any

def perform_search(
    query: str,
    api_key: str,
    cse_id: str,
    num_results: int = 5,
    **kwargs: Any  # Allows passing additional CSE API parameters
) -> List[Dict[str, Any]]:
    """
    Performs a Google Custom Search for the given query.

    Args:
        query: The search term(s).
        api_key: The Google API key authorized for Custom Search API.
        cse_id: The ID of the Custom Search Engine to use.
        num_results: The number of search results to return (max 10 per API call).
        **kwargs: Additional parameters to pass to the CSE list method,
                  e.g., siteSearch, exactTerms, etc. Refer to Google CSE API docs.

    Returns:
        A list of search result item dictionaries as returned by the API.
        Each item typically contains 'title', 'link', 'snippet', etc.
        Returns an empty list if an error occurs or no results are found.
    """
    if not api_key or not cse_id:
        # This situation should ideally be caught by config validation in app.py
        print("ERROR (search_engine): Google API Key or CSE ID is missing.")
        return []
    
    # The API typically allows a maximum of 10 results per request.
    if num_results > 10:
        num_results = 10
    if num_results < 1:
        num_results = 1

    try:
        # Build the service object for the Custom Search API.
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search query.
        result: Dict[str, Any] = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results,
            **kwargs # Pass through any additional API parameters
        ).execute()
        
        # Extract and return the 'items' list from the result, or an empty list.
        return result.get('items', [])
    except Exception as e:
        # Log the error. In a larger application, use a proper logging framework.
        print(f"ERROR (search_engine): Google Search API call failed for query '{query}'. Error: {e}")
        # Consider raising a custom exception or returning a more specific error indicator
        # if app.py needs to differentiate error types. For now, empty list on error.
        return []

if __name__ == '__main__':
    # This test requires secrets to be available, ideally run through app.py or with mocked config.
    # For direct testing, you'd need to manually provide API_KEY and CSE_ID.
    print("Search Engine Module - Direct Test Mode")
    # Example (replace with actual keys for a real test, or use a mock)
    # TEST_API_KEY = "YOUR_API_KEY"
    # TEST_CSE_ID = "YOUR_CSE_ID"
    # if TEST_API_KEY != "YOUR_API_KEY":
    #     test_query = input("Enter test search query: ")
    #     results = perform_search(test_query, TEST_API_KEY, TEST_CSE_ID, num_results=3)
    #     if results:
    #         print(f"\nFound {len(results)} results for '{test_query}':")
    #         for i, res in enumerate(results):
    #             print(f"  {i+1}. {res.get('title')} - {res.get('link')}")
    #     else:
    #         print("No results found or an error occurred.")
    # else:
    #     print("Test API key and CSE ID not set. Skipping direct test.")
    print("To test this module, integrate it with app.py which handles configuration loading.")
# end of modules/search_engine.py
