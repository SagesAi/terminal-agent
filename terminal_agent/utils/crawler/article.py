"""
Article class for representing and formatting web page content.
"""

import re
from urllib.parse import urljoin

from markdownify import markdownify as md


class Article:
    """
    Represents a web article with methods to convert to different formats.
    """
    url: str = ""

    def __init__(self, title: str, html_content: str):
        """
        Initialize an Article object.
        
        Args:
            title: The title of the article
            html_content: The HTML content of the article
        """
        self.title = title
        self.html_content = html_content

    def to_markdown(self, including_title: bool = True) -> str:
        """
        Convert the article to markdown format.
        
        Args:
            including_title: Whether to include the title in the markdown
            
        Returns:
            The article content in markdown format
        """
        markdown = ""
        if including_title and self.title:
            markdown += f"# {self.title}\n\n"
        markdown += md(self.html_content)
        return markdown

    def to_message(self) -> list[dict]:
        """
        Convert the article to a list of message parts (text and images).
        
        Returns:
            A list of dictionaries representing the article content
        """
        image_pattern = r"!\[.*?\]\((.*?)\)"

        content: list[dict[str, str]] = []
        parts = re.split(image_pattern, self.to_markdown())

        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is an image URL
                image_url = urljoin(self.url, part.strip())
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:  # This is text content
                if part.strip():  # Only add non-empty parts
                    content.append({"type": "text", "text": part.strip()})

        return content
