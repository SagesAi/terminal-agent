"""
Jina client for web crawling.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)


class JinaClient:
    """
    Client for Jina's web crawling API.
    """
    
    def crawl(self, url: str, return_format: str = "html") -> str:
        """
        Crawl a web page using Jina's API.
        
        Args:
            url: The URL to crawl
            return_format: The format to return (html or markdown)
            
        Returns:
            The crawled content in the specified format
        """
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        
        # Add API key if available
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. "
                "See https://jina.ai/reader for more information."
            )
            
        data = {"url": url}
        response = requests.post("https://r.jina.ai/", headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"Failed to crawl {url}. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise Exception(f"Failed to crawl {url}. Status code: {response.status_code}")
            
        return response.text
