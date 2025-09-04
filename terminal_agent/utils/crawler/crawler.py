"""
Web crawler implementation for Terminal Agent using crawl4ai.
"""

import logging
import sys
from typing import Optional

from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig

from .article import Article

logger = logging.getLogger(__name__)


class Crawler:
    """
    Crawler for extracting clean, readable content from web pages using crawl4ai.
    """
    
    def __init__(self):
        """Initialize the crawler with crawl4ai."""
        self.config = CrawlerRunConfig(
            exclude_external_links=True,
            exclude_all_images=True,
            table_extraction=None,
            exclude_domains=None,
            wait_until='domcontentloaded',
            page_timeout=60000
        )
        self.crawler = AsyncWebCrawler()
    
    async def crawl(self, url: str) -> Article:
        """
        Crawl a web page and extract its content using crawl4ai.
        
        Args:
            url: The URL to crawl
            
        Returns:
            An Article object containing the extracted content
        """
        logger.info(f"Crawling URL with crawl4ai: {url}")
        try:
            # Execute the crawl
            result = await self.crawler.arun(url, config=self.config)
            
            # Create an Article object from the result
            # Extract title from HTML if available, otherwise use URL
            title = "Untitled"
            if result.html:
                # Try to extract title from HTML
                import re
                title_match = re.search(r'<title>(.*?)</title>', result.html, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
            
            article = Article(
                title=title,
                html_content=result.html or result.markdown or ""
            )
            return article
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            # Return an empty article with error information
            return Article(
                title="Error",
                html_content=f"Failed to crawl {url}: {str(e)}"
            )


if __name__ == "__main__":
    import asyncio
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get URL from command line or use default
    if len(sys.argv) == 2:
        url = sys.argv[1]
    else:
        url = "https://www.example.com"
    
    # Create and run the async function
    async def main():
        crawler = Crawler()
        article = await crawler.crawl(url)
        print(article.to_markdown())
    
    # Run the async function
    asyncio.run(main())
