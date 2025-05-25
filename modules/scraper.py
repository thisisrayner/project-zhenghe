# modules/scraper.py

import requests
from bs4 import BeautifulSoup
import trafilatura # For main content extraction
import streamlit as st # For potential error display or logging if run directly
from typing import Dict, Optional, TypedDict # For type hinting

# --- Type Hint for Scraper Output ---
class ScrapedData(TypedDict, total=False): # total=False means keys are optional
    url: str
    raw_html: Optional[str] # Optional: if you want to store it
    title: Optional[str]
    meta_description: Optional[str]
    og_title: Optional[str]
    og_description: Optional[str]
    main_text: Optional[str]
    error: Optional[str] # To store any scraping errors

# --- Main Scraping Function ---
def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData:
    """
    Fetches a URL and extracts metadata and main content.

    Args:
        url (str): The URL to scrape.
        timeout (int): Request timeout in seconds.

    Returns:
        ScrapedData: A dictionary containing the scraped information or an error message.
    """
    scraped_data: ScrapedData = {'url': url}
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
    headers = {'User-Agent': user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)

        # Store raw HTML if needed for debugging or advanced parsing later
        # scraped_data['raw_html'] = response.text

        # --- Metadata Extraction with BeautifulSoup ---
        soup = BeautifulSoup(response.content, 'html.parser')

        # Title
        if soup.title and soup.title.string:
            scraped_data['title'] = soup.title.string.strip()
        else: # Fallback for title if soup.title.string is None or empty
            og_title_tag = soup.find('meta', property='og:title')
            if og_title_tag and og_title_tag.get('content'):
                 scraped_data['title'] = og_title_tag.get('content').strip()

        # Meta Description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            scraped_data['meta_description'] = desc_tag.get('content').strip()

        # OpenGraph Title (often more share-friendly)
        og_title_tag = soup.find('meta', property='og:title')
        if og_title_tag and og_title_tag.get('content'):
            scraped_data['og_title'] = og_title_tag.get('content').strip()
        elif scraped_data.get('title'): # Use main title if OG title not found
            scraped_data['og_title'] = scraped_data['title']


        # OpenGraph Description
        og_desc_tag = soup.find('meta', property='og:description')
        if og_desc_tag and og_desc_tag.get('content'):
            scraped_data['og_description'] = og_desc_tag.get('content').strip()
        elif scraped_data.get('meta_description'): # Use meta desc if OG desc not found
            scraped_data['og_description'] = scraped_data['meta_description']


        # --- Main Content Extraction with Trafilatura ---
        # Trafilatura works best with the raw HTML string
        # You can also pass response.content, but response.text is often preferred by trafilatura
        # For PDFs or non-HTML, trafilatura might return None or raise an error
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            main_text = trafilatura.extract(
                response.text, # Pass the decoded text
                include_comments=False,
                include_tables=False, # Set to True if you need table content
                no_fallback=True # If True, doesn't fall back to basic extractors
            )
            if main_text:
                scraped_data['main_text'] = main_text.strip()
            else:
                # Fallback if trafilatura returns nothing but it was HTML
                # Basic text extraction (less clean)
                # scraped_data['main_text'] = soup.get_text(separator=' ', strip=True)
                scraped_data['main_text'] = "Trafilatura could not extract main content."
        else:
            scraped_data['main_text'] = f"Content type '{response.headers.get('Content-Type')}' not processed for main text."


    except requests.exceptions.Timeout:
        scraped_data['error'] = f"Timeout after {timeout} seconds."
    except requests.exceptions.HTTPError as e:
        scraped_data['error'] = f"HTTP error: {e.response.status_code} {e.response.reason}."
    except requests.exceptions.RequestException as e:
        scraped_data['error'] = f"Request error: {e}."
    except Exception as e:
        # Catch any other unexpected errors during parsing
        scraped_data['error'] = f"An unexpected error occurred during scraping/parsing: {e}."

    return scraped_data

if __name__ == '__main__':
    # --- Simple Test for this module ---
    st.title("Scraper Module Test")
    test_url = st.text_input("Enter URL to scrape:", "https://streamlit.io/blog")

    if st.button("Scrape URL"):
        if test_url:
            st.write(f"Scraping: {test_url}")
            with st.spinner("Fetching and extracting..."):
                data = fetch_and_extract_content(test_url)

            if data.get('error'):
                st.error(f"Error: {data['error']}")

            st.subheader("Extracted Data:")
            st.json(data) # Display all data as JSON for easy inspection

            if data.get('main_text'):
                st.subheader("Main Text Preview (first 500 chars):")
                st.text_area("Main Text:", value=data['main_text'][:500]+"...", height=200, disabled=True)
        else:
            st.warning("Please enter a URL.")
