"""
Web page tool for Terminal Agent.

This tool allows the agent to crawl web pages and extract content.
"""

import logging
import re
import json
from typing import Dict, Any, Optional

from ...utils.crawler import Crawler

logger = logging.getLogger(__name__)

WEB_PAGE_DESCRIPTION = """
        Fetch and extract readable content from web url provided by web search tool with intelligent processing.
        
        Features:
        - HTML content extraction and cleaning
        - Article and main content detection
        - JavaScript-free text extraction
        - Metadata preservation (title, author, date)
        - Link and image extraction
        - Character encoding handling
        - HTTP error handling
        
        Content extraction:
        - Main article text extraction
        - Navigation and advertisement removal
        - Code block preservation
        - Table and list formatting
        - Link URL collection
        
        Use this tool for:
        - Documentation reading
        - Article content analysis
        - API endpoint documentation
        - Tutorial and guide extraction
        - Research paper access
        """ 

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_page",
        "description": WEB_PAGE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to crawl"
                },
                "format": {
                    "type": "string",
                    "description": "Output format (markdown or json)",
                    "default": "markdown"
                }
            },
            "required": ["url"]
        }
    }
}

def web_page_tool(query: str) -> str:
    """
    Crawl a web page and extract its content.
    
    Args:
        query: Either a URL string, a JSON string, or a dictionary containing the URL to crawl 
               and optional parameters. Format: {"url": "https://example.com", "format": "markdown"}
               
    Returns:
        The extracted content in the specified format (markdown by default).
    """
    try:
        # Handle case where query is already a dictionary
        if isinstance(query, dict):
            params = query
        # Parse the query as JSON if it's a string
        elif isinstance(query, str):
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # If the query is not valid JSON, assume it's a URL
                params = {"url": query}
        else:
            return "Error: Invalid input type. Expected URL string, JSON string, or dictionary."
        
        # Extract parameters
        url = str(params.get("url", "")).strip()
        output_format = str(params.get("format", "markdown")).lower()
        
        if not url:
            return "Error: No URL provided. Please provide a URL to crawl."
        
        # Validate URL format
        if not re.match(r'^https?://', url):
            return f"Error: Invalid URL format: {url}. URL must start with http:// or https://"
        
        # Crawl the web page
        import asyncio
        crawler = Crawler()
        article = asyncio.run(crawler.crawl(url))
        
        # Return the content in the specified format
        if output_format.lower() == "markdown":
            return article.to_markdown()
        elif output_format.lower() == "json":
            return json.dumps(article.to_message())
        else:
            return article.to_markdown()
            
    except Exception as e:
        logger.error(f"Error in web_page_tool: {str(e)}")
        return f"Error crawling web page: {str(e)}"
