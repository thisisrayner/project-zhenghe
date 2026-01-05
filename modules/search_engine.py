# modules/search_engine.py
# Version 1.2.2: Integrated daily usage tracker.
# Version 1.2.1: Implement pagination to support up to 30 results (3 API calls).
# Version 1.2.0: Added retry mechanism with exponential backoff for API calls.
# Handles common Google Search API rate limit errors.

"""
Handles interactions with the Google Custom Search API.

This module provides functionality to perform searches using specified
keywords, API key, and Custom Search Engine (CSE) ID.
It uses the google-api-python-client library and includes
a retry mechanism for API calls to handle transient errors and rate limits.
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Dict, Any
import time
import random
from modules import usage_tracker

# Default retry parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 2.0  # seconds
DEFAULT_MAX_BACKOFF = 30.0    # seconds

def perform_search(
    query: str,
    api_key: str,
    cse_id: str,
    num_results: int = 5,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Performs a Google Custom Search for the given query with retry logic.

    Args:
        query: The search term(s).
        api_key: The Google API key authorized for Custom Search API.
        cse_id: The ID of the Custom Search Engine to use.
        num_results: The number of search results to return (max 10 per API call).
        max_retries: Maximum number of retries for API calls.
        initial_backoff: Initial delay in seconds for the first retry.
        max_backoff: Maximum delay in seconds for a single retry.
        **kwargs: Additional parameters to pass to the CSE list method,
                  e.g., siteSearch, exactTerms, etc. Refer to Google CSE API docs.

    Returns:
        A list of search result item dictionaries as returned by the API.
        Each item typically contains 'title', 'link', 'snippet', etc.
        Returns an empty list if an error occurs after all retries or no results are found.
    """
    if not api_key or not cse_id:
        print("ERROR (search_engine): Google API Key or CSE ID is missing.")
        return []
    
    if num_results > 30: num_results = 30 # Hard cap at 30 to prevent excessive API usage (approx 3 calls)
    if num_results < 1: num_results = 1

    all_results = []
    results_fetched = 0
    start_index = 1
    
    # Calculate how many calls are needed. Max 10 per call.
    # We loop until we have enough results or we hit the limit.
    
    while results_fetched < num_results:
        # Determine how many to fetch in this batch (max 10)
        num_to_fetch_this_batch = min(10, num_results - results_fetched)
        
        current_retry = 0
        current_backoff_delay = initial_backoff
        batch_success = False
        
        while current_retry <= max_retries:
            try:
                service = build("customsearch", "v1", developerKey=api_key)
                result: Dict[str, Any] = service.cse().list(
                    q=query,
                    cx=cse_id,
                    num=num_to_fetch_this_batch,
                    start=start_index,
                    **kwargs 
                ).execute()
                
                # Track usage
                usage_tracker.increment_usage(1)
                
                items = result.get('items', [])
                all_results.extend(items)
                results_fetched += len(items)
                start_index += len(items)
                batch_success = True
                
                # If we got fewer items than requested, it means no more results are available
                if len(items) < num_to_fetch_this_batch:
                    return all_results
                
                break # Exit retry loop on success
            
            except HttpError as e:
                # ... (Logic for retries same as before) ...
                if e.resp.status in [429, 403]:
                    error_content = e.content.decode('utf-8').lower()
                    is_rate_limit_error = (
                        "rate limit" in error_content or
                        "quota exceeded" in error_content or
                        "userrate" in error_content or 
                        "qpd" in error_content 
                    )
                    if is_rate_limit_error and current_retry < max_retries:
                        current_retry += 1
                        print(f"WARNING (search_engine): Google Search API rate limit hit for query '{query}'. "
                              f"Status: {e.resp.status}. Retrying in {current_backoff_delay:.1f}s "
                              f"(Attempt {current_retry}/{max_retries}). Error: {str(e)[:200]}")
                        time.sleep(current_backoff_delay)
                        current_backoff_delay = min(current_backoff_delay * 2 + random.uniform(0, 1.0), max_backoff)
                        continue 
                    else: 
                        print(f"ERROR (search_engine): Google Search API call failed for query '{query}' after {current_retry} retries or non-retriable HTTP error. Status: {e.resp.status}. Error: {e}")
                        return all_results # Return what we have so far
                else: 
                    print(f"ERROR (search_engine): Google Search API call failed for query '{query}' with non-retriable HTTP error. Status: {e.resp.status}. Error: {e}")
                    return all_results # Return what we have so far
            
            except Exception as e: 
                if current_retry < max_retries:
                    current_retry += 1
                    print(f"WARNING (search_engine): Google Search API call failed for query '{query}' due to a non-HTTP error. "
                          f"Retrying in {current_backoff_delay:.1f}s (Attempt {current_retry}/{max_retries}). Error: {e}")
                    time.sleep(current_backoff_delay)
                    current_backoff_delay = min(current_backoff_delay * 2 + random.uniform(0, 1.0), max_backoff)
                    continue
                else:
                    print(f"ERROR (search_engine): Google Search API call failed for query '{query}' after {current_retry} retries due to a non-HTTP error. Error: {e}")
                    return all_results # Failed after retries

        if not batch_success:
             return all_results # Stop if a batch fails completely

    return all_results


if __name__ == '__main__':
    print("Search Engine Module - Direct Test Mode (v1.2.0 with retries)")
    print("To test this module, integrate it with app.py which handles configuration loading,")
    print("or manually provide API_KEY and CSE_ID below and uncomment test code.")
    
    # Example (replace with actual keys for a real test, or use a mock)
    # TEST_API_KEY = "YOUR_API_KEY_HERE" 
    # TEST_CSE_ID = "YOUR_CSE_ID_HERE"

    # if TEST_API_KEY != "YOUR_API_KEY_HERE" and TEST_CSE_ID != "YOUR_CSE_ID_HERE":
    #     test_query = input("Enter test search query (e.g., 'streamlit features'): ")
    #     if test_query:
    #         print(f"\nPerforming search for: '{test_query}'")
    #         results = perform_search(test_query, TEST_API_KEY, TEST_CSE_ID, num_results=3)
    #         if results:
    #             print(f"\nFound {len(results)} results for '{test_query}':")
    #             for i, res in enumerate(results):
    #                 print(f"  {i+1}. {res.get('title')} - {res.get('link')}")
    #         else:
    #             print("No results found or an error occurred after retries.")
    # else:
    #     print("Test API key and CSE ID not set. Skipping direct test call.")

# end of modules/search_engine.py
