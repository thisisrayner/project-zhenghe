# modules/scraper.py
# Version 1.2.0: Added PDF text extraction using PyMuPDF. Based on original v1.1.
# Handles HTML metadata/text and PDF text/metadata title.

"""
Web scraping module for fetching and extracting content from URLs.

This module uses 'requests' to fetch web page content, 'BeautifulSoup'
for parsing HTML and extracting metadata (like title, description, OpenGraph tags),
'trafilatura' for extracting the main textual content of an HTML article,
and 'PyMuPDF' (fitz) for extracting text from PDF documents.
"""

import requests
from bs4 import BeautifulSoup
import trafilatura
from typing import TypedDict, Optional, Dict, List # Added List for PDF text
import fitz # PyMuPDF

# --- Type Definition for Scraped Data ---
class ScrapedData(TypedDict, total=False):
    """
    A dictionary structure for storing data scraped from a web page or PDF.
    `total=False` means keys are optional and might not always be present.
    """
    url: str                     # The URL that was scraped
    # raw_html: Optional[str]      # Full raw HTML content (optional to store from v1.1) - kept commented
    scraped_title: Optional[str]         # HTML <title> tag content or PDF metadata title
    meta_description: Optional[str] # Content of <meta name="description"> (N/A for PDF)
    og_title: Optional[str]      # OpenGraph title (og:title) (N/A for PDF)
    og_description: Optional[str]# OpenGraph description (og:description) (N/A for PDF)
    main_text: Optional[str]     # Main article text extracted by trafilatura or PyMuPDF
    error: Optional[str]         # Error message if scraping failed for this URL
    content_type: Optional[str]  # Detected content type (e.g., 'html', 'pdf')

DEFAULT_USER_AGENT: str = ( # Moved to global scope for consistency
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36" 
)

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    """
    Extracts text and title from PDF bytes using PyMuPDF (fitz).

    Args:
        pdf_bytes: The byte content of the PDF file.

    Returns:
        A tuple (full_text, document_title).
        - full_text: Concatenated text from all pages.
        - document_title: Title from PDF metadata, if available.
        Returns (None, None) if a significant error occurs during PDF processing.
    """
    full_text_list: List[str] = []
    document_title: Optional[str] = None
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.metadata:
                document_title = doc.metadata.get('title')
                if document_title and document_title.lower().endswith(".pdf"):
                    document_title = document_title[:-4].strip()
                if not document_title: 
                    document_title = None
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    full_text_list.append(page_text.strip())
        return "\n\n".join(full_text_list) if full_text_list else None, document_title
    except Exception as e:
        print(f"SCRAPER_PDF_ERROR: Error extracting text/title from PDF: {e}")
        return None, None

# --- Main Scraping Function ---
def fetch_and_extract_content(url: str, timeout: int = 15) -> ScrapedData: # timeout from v1.1
    """
    Fetches content from the given URL and extracts metadata and main text.
    Supports HTML and PDF documents.

    Args:
        url: The URL of the web page/document to scrape.
        timeout: The timeout in seconds for the HTTP GET request.

    Returns:
        A ScrapedData dictionary containing the extracted information.
        If an error occurs, the 'error' key in the dictionary will be populated.
    """
    scraped_data: ScrapedData = {
        'url': url,
        "scraped_title": None, # Changed from 'title' to 'scraped_title' for consistency
        "meta_description": None,
        "og_title": None,
        "og_description": None,
        "main_text": None,
        "error": None,
        "content_type": None
    }
    
    headers = {'User-Agent': DEFAULT_USER_AGENT} # Use global DEFAULT_USER_AGENT

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status() 

        # Optional: Store raw HTML - from v1.1, kept commented
        # scraped_data['raw_html'] = response.text 

        current_content_type: str = response.headers.get('Content-Type', '').lower()
        scraped_data['content_type'] = current_content_type.split(';')[0] 

        if 'application/pdf' in current_content_type:
            scraped_data['content_type'] = 'pdf' 
            pdf_bytes = response.content
            extracted_text, pdf_doc_title = _extract_text_from_pdf_bytes(pdf_bytes)
            
            if extracted_text:
                scraped_data['main_text'] = extracted_text
            else:
                scraped_data['main_text'] = "SCRAPER_INFO: Could not extract text content from PDF or PDF was empty."

            if pdf_doc_title:
                scraped_data['scraped_title'] = pdf_doc_title
            else: 
                try:
                    url_filename = url.split('/')[-1]
                    if url_filename.lower().endswith(".pdf"): url_filename = url_filename[:-4]
                    url_filename = url_filename.replace('-', ' ').replace('_', ' ').strip()
                    scraped_data['scraped_title'] = url_filename if url_filename else "Untitled PDF Document"
                except Exception:
                    scraped_data['scraped_title'] = "Untitled PDF Document"
            
            scraped_data['meta_description'] = "N/A for PDF"
            scraped_data['og_title'] = "N/A for PDF"
            scraped_data['og_description'] = "N/A for PDF"

        elif 'text/html' in current_content_type or not current_content_type or 'text/plain' in current_content_type:
            if not current_content_type: scraped_data['content_type'] = 'html (assumed)'
            else: scraped_data['content_type'] = 'html'

            soup = BeautifulSoup(response.content, 'html.parser') # Use response.content for BS consistency

            # Extract HTML Title
            if soup.title and soup.title.string:
                scraped_data['scraped_title'] = soup.title.string.strip()
            else: 
                og_title_tag_for_title_fallback = soup.find('meta', property='og:title')
                if og_title_tag_for_title_fallback and og_title_tag_for_title_fallback.get('content'):
                     scraped_data['scraped_title'] = og_title_tag_for_title_fallback.get('content').strip()

            # Extract Meta Description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag and desc_tag.get('content'):
                scraped_data['meta_description'] = desc_tag.get('content').strip()

            # Extract OpenGraph Title
            og_title_tag = soup.find('meta', property='og:title')
            if og_title_tag and og_title_tag.get('content'):
                scraped_data['og_title'] = og_title_tag.get('content').strip()
            elif scraped_data.get('scraped_title'): 
                scraped_data['og_title'] = scraped_data['scraped_title']

            # Extract OpenGraph Description
            og_desc_tag = soup.find('meta', property='og:description')
            if og_desc_tag and og_desc_tag.get('content'):
                scraped_data['og_description'] = og_desc_tag.get('content').strip()
            elif scraped_data.get('meta_description'): 
                scraped_data['og_description'] = scraped_data['meta_description']
            
            # Fallback for scraped_title (already done, but ensure it's set if only OG title was found initially)
            if not scraped_data['scraped_title'] and scraped_data['og_title']:
                 scraped_data['scraped_title'] = scraped_data['og_title']


            # Main Content Extraction with Trafilatura (from v1.1 logic)
            main_text_extracted = trafilatura.extract(
                response.text, 
                include_comments=False,
                include_tables=False, # As per v1.1
                no_fallback=True    # As per v1.1
            )
            if main_text_extracted:
                scraped_data['main_text'] = main_text_extracted.strip()
            else:
                scraped_data['main_text'] = "Trafilatura could not extract main content from this HTML page."
        
        else: 
            scraped_data['error'] = f"Unsupported content type for detailed scraping: {current_content_type.split(';')[0]}"
            scraped_data['main_text'] = f"SCRAPER_INFO: Content type '{current_content_type.split(';')[0]}' not processed for main text."
            try:
                soup_fallback = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
                if soup_fallback.title and soup_fallback.title.string:
                    scraped_data['scraped_title'] = soup_fallback.title.string.strip()
                else:
                    scraped_data['scraped_title'] = url.split('/')[-1] if '/' in url else url
            except:
                 scraped_data['scraped_title'] = url.split('/')[-1] if '/' in url else url

    except requests.exceptions.Timeout:
        scraped_data['error'] = f"Request timed out after {timeout} seconds."
    except requests.exceptions.HTTPError as e:
        scraped_data['error'] = f"HTTP error: {e.response.status_code} {e.response.reason}."
    except requests.exceptions.RequestException as e:
        scraped_data['error'] = f"Web request error: {e}."
    except Exception as e:
        scraped_data['error'] = f"An unexpected error occurred during scraping/parsing: {e}."

    if scraped_data.get('main_text') is None and not scraped_data.get('error'):
        scraped_data['main_text'] = "SCRAPER_INFO: No main text could be extracted."
    if scraped_data.get('scraped_title') is None and not scraped_data.get('error'): # Ensure title key consistency
        scraped_data['scraped_title'] = "Untitled Document"

    return scraped_data

if __name__ == '__main__':
    import streamlit as st_test 
    st_test.set_page_config(layout="wide")
    st_test.title("Scraper Module Test (v1.2.0 - PDF Support Added)") # Updated title
    
    test_url_input = st_test.text_input("Enter URL to scrape (HTML or PDF):", "https://arxiv.org/pdf/1706.03762") # Example PDF

    if st_test.button("Scrape URL"):
        if test_url_input:
            st_test.write(f"Scraping: {test_url_input}")
            with st_test.spinner("Fetching and extracting content..."):
                data_output = fetch_and_extract_content(test_url_input)

            if data_output.get('error'):
                st_test.error(f"Scraping Error: {data_output['error']}")

            st_test.subheader("Extracted Data (JSON):")
            st_test.json(data_output) 

            if data_output.get('scraped_title'): # Check for consistent title key
                st_test.write(f"**Detected Title:** {data_output['scraped_title']}")
            if data_output.get('content_type'):
                st_test.write(f"**Detected Content Type:** {data_output['content_type']}")


            if data_output.get('main_text') and not data_output.get('error'):
                st_test.subheader("Main Text Preview (first 1000 chars):")
                st_test.text_area("Main Text:", value=data_output['main_text'][:1000]+"...", height=300, disabled=True)
        else:
            st_test.warning("Please enter a URL to test scraping.")

# end of modules/scraper.py
