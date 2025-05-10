"""
Web crawler implementation for Terminal Agent.
"""

import logging
import sys
from typing import Optional

from .article import Article
from .jina_client import JinaClient
from .readability_extractor import ReadabilityExtractor

logger = logging.getLogger(__name__)


class Crawler:
    """
    Crawler for extracting clean, readable content from web pages.
    """
    
    def __init__(self):
        """Initialize the crawler."""
        self.jina_client = JinaClient()
        self.extractor = ReadabilityExtractor()
    
    def crawl(self, url: str) -> Article:
        """
        Crawl a web page and extract its content.
        
        Args:
            url: The URL to crawl
            
        Returns:
            An Article object containing the extracted content
        """
        logger.info(f"Crawling URL: {url}")
        
        try:
            # Get HTML content using Jina
            html = self.jina_client.crawl(url, return_format="html")
            
            # Extract article content using readability
            article = self.extractor.extract_article(html)
            
            # Set the article URL
            article.url = url
            
            logger.info(f"Successfully crawled {url}")
            return article
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            # Return an empty article in case of error
            article = Article("Error", f"<p>Failed to crawl {url}: {str(e)}</p>")
            article.url = url
            return article


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get URL from command line or use default
    if len(sys.argv) == 2:
        url = sys.argv[1]
    else:
        url = "https://www.example.com"
    
    # Crawl the URL and print the result
    crawler = Crawler()
    article = crawler.crawl(url)
    print(article.to_markdown())
