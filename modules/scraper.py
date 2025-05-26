# modules/scraper.py
# Version 1.1: Enhanced docstrings, type hinting, and added comments.

"""
Web scraping module for fetching and extracting content from URLs.

This module uses 'requests' to fetch web page content, 'BeautifulSoup'
for parsing HTML and extracting metadata (like title, description, OpenGraph tags),
and 'trafilatura' for extracting the main textual content of an article.
"""

import requests
from bs4 import BeautifulSoup
import trafilatura
from typing import TypedDict, Optional, Dict # For type hinting

# --- Type Definition for Scraped Data ---
class ScrapedData(TypedDict, total=False):
    """
    A dictionary structure for storing data scraped from a web page.
    `total=False` means keys are optional and might not always be present.
    """
    url: str                     # The URL that was scraped
    raw_html: Optional[str]      # Full raw HTML content (optional to store)
    title: Optional[str]         # HTML <title> tag content
    meta_description: Optional[str] # Content of <meta name="description">
    og_title: Optional[str]      # OpenGraph title (og:title)
    og_description: Optional[str]# OpenGraph description (og:description)
    main_text: Optional[str]     # Main article text extracted by trafilatura
    error: Optional[str]         # Error message if scraping failed for this URL

# --- Main Scraping Function ---
def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData:
    """
    Fetches content from the given URL and extracts metadata and main text.

    Args:
        url: The URL of the web page to scrape.
        timeout: The timeout in seconds for the HTTP GET request.

    Returns:
        A ScrapedData dictionary containing the extracted information.
        If an error occurs, the 'error' key in the dictionary will be populated.
    """
    scraped_data: ScrapedData = {'url': url} # Initialize with the URL
    
    # Using a common browser user-agent can help avoid simple bot blocks.
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36" # A common Chrome user agent
    )
    headers = {'User-Agent': user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)

        # Optional: Store raw HTML if needed for deeper analysis or debugging later.
        # scraped_data['raw_html'] = response.text

        # --- Metadata Extraction with BeautifulSoup ---
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract HTML Title
        if soup.title and soup.title.string:
            scraped_data['title'] = soup.title.string.strip()
        else: # Fallback for title: try to get OG title if HTML title is missing/empty
            og_title_tag_for_title_fallback = soup.find('meta', property='og:title')
            if og_title_tag_for_title_fallback and og_title_tag_for_title_fallback.get('content'):
                 scraped_data['title'] = og_title_tag_for_title_fallback.get('content').strip()

        # Extract Meta Description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            scraped_data['meta_description'] = desc_tag.get('content').strip()

        # Extract OpenGraph Title (often preferred for social sharing)
        og_title_tag = soup.find('meta', property='og:title')
        if og_title_tag and og_title_tag.get('content'):
            scraped_data['og_title'] = og_title_tag.get('content').strip()
        elif scraped_data.get('title'): # Fallback to HTML title if OG title not found
            scraped_data['og_title'] = scraped_data['title']

        # Extract OpenGraph Description
        og_desc_tag = soup.find('meta', property='og:description')
        if og_desc_tag and og_desc_tag.get('content'):
            scraped_data['og_description'] = og_desc_tag.get('content').strip()
        elif scraped_data.get('meta_description'): # Fallback to meta description if OG description not found
            scraped_data['og_description'] = scraped_data['meta_description']

        # --- Main Content Extraction with Trafilatura ---
        # Check content type to ensure we're trying to parse HTML.
        # Trafilatura works best on HTML content.
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # `trafilatura.extract` attempts to get the main body of text.
            # It's generally good but can struggle with complex JS-heavy sites or non-article layouts.
            main_text_extracted = trafilatura.extract(
                response.text, # Use response.text (decoded string) for trafilatura
                include_comments=False,
                include_tables=False, # Set to True if table data is important
                no_fallback=True    # If True, doesn't use BeautifulSoup as a basic fallback
                                    # Set to False if you want it to try harder with a basic extractor.
            )
            if main_text_extracted:
                scraped_data['main_text'] = main_text_extracted.strip()
            else:
                scraped_data['main_text'] = "Trafilatura could not extract main content from this HTML page."
        else:
            scraped_data['main_text'] = f"Content type '{content_type}' not processed for main text extraction."

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        scraped_data['error'] = f"Request timed out after {timeout} seconds."
    except requests.exceptions.HTTPError as e:
        scraped_data['error'] = f"HTTP error: {e.response.status_code} {e.response.reason}."
    except requests.exceptions.RequestException as e:
        # Catches other requests-related errors (e.g., DNS failure, connection error)
        scraped_data['error'] = f"Web request error: {e}."
    except Exception as e:
        # Catch-all for any other unexpected errors during scraping or parsing
        scraped_data['error'] = f"An unexpected error occurred during scraping/parsing: {e}."
        # In a production app, log the full traceback here for debugging.
        # import traceback
        # print(traceback.format_exc())

    return scraped_data

if __name__ == '__main__':
    # This block is for direct testing of the scraper module.
    # It requires Streamlit to be installed to use st.* functions.
    # To run: streamlit run modules/scraper.py
    import streamlit as st_test # Alias to avoid confusion with module-level st if any
    st_test.set_page_config(layout="wide")
    st_test.title("Scraper Module Test (v1.1)")
    
    test_url_input = st_test.text_input("Enter URL to scrape:", "https://blog.streamlit.io/how-to-master-llm-hallucinations/")

    if st_test.button("Scrape URL"):
        if test_url_input:
            st_test.write(f"Scraping: {test_url_input}")
            with st_test.spinner("Fetching and extracting content..."):
                data_output = fetch_and_extract_content(test_url_input)

            if data_output.get('error'):
                st_test.error(f"Scraping Error: {data_output['error']}")

            st_test.subheader("Extracted Data (JSON):")
            st_test.json(data_output) # Display all data as JSON for easy inspection

            if data_output.get('main_text') and not data_output.get('error'):
                st_test.subheader("Main Text Preview (first 500 chars):")
                st_test.text_area("Main Text:", value=data_output['main_text'][:500]+"...", height=200, disabled=True)
        else:
            st_test.warning("Please enter a URL to test scraping.")
# end of modules/scraper.py
