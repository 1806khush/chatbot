import requests
from bs4 import BeautifulSoup
import re

def fetch_wikipedia_content(url):
    """
    Fetch content from a Wikipedia page
    
    Args:
        url (str): URL of the Wikipedia page
        
    Returns:
        str: Extracted text content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for error status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for unwanted in soup.select('table, .mw-editsection, .reference, .mw-jump-link, .mw-headline-anchor'):
            unwanted.decompose()
            
        # Get the main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return "Failed to find main content area on the page."
        
        # Extract paragraphs and headers
        text_parts = []
        for element in content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text().strip()
            if text:
                # Add a newline before headers for better spacing
                if element.name.startswith('h'):
                    text_parts.append(f"\n\n{text}\n")
                else:
                    text_parts.append(text)
        
        # Join all text parts
        full_text = '\n'.join(text_parts)
        
        # Clean up the text (remove multiple consecutive newlines, citation markers, etc.)
        full_text = re.sub(r'\[\d+\]', '', full_text)  # Remove citation markers like [1]
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Replace 3+ newlines with 2
        
        return full_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Wikipedia content: {e}")
        return f"Error fetching content: {str(e)}"