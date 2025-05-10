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


def web_page_tool(query: str) -> str:
    """
    Crawl a web page and extract its content.
    
    Args:
        query: A JSON string containing the URL to crawl and optional parameters.
               Format: {"url": "https://example.com", "format": "markdown"}
               
    Returns:
        The extracted content in the specified format (markdown by default).
    """
    try:
        # Parse the query as JSON
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            # If the query is not valid JSON, assume it's a URL
            params = {"url": query}
        
        # Extract parameters
        url = params.get("url", "")
        output_format = params.get("format", "markdown")
        
        if not url:
            return "Error: No URL provided. Please provide a URL to crawl."
        
        # Validate URL format
        if not re.match(r'^https?://', url):
            return f"Error: Invalid URL format: {url}. URL must start with http:// or https://"
        
        # Crawl the web page
        crawler = Crawler()
        article = crawler.crawl(url)
        
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
