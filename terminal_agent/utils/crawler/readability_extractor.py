"""
Readability extractor for web content.
"""

from readabilipy import simple_json_from_html_string

from .article import Article


class ReadabilityExtractor:
    """
    Extracts clean article content from HTML using readability algorithms.
    """
    
    def extract_article(self, html: str) -> Article:
        """
        Extract article content from HTML.
        
        Args:
            html: The HTML content to extract from
            
        Returns:
            An Article object containing the extracted content
        """
        article = simple_json_from_html_string(html, use_readability=True)
        
        return Article(
            title=article.get("title", ""),
            html_content=article.get("content", ""),
        )
