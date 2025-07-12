"""
Web search tool for Terminal Agent.

This tool allows the agent to perform web searches using DuckDuckGo.
"""

import logging
import json
import time
from typing import Dict, Any
from ddgs import DDGS

logger = logging.getLogger(__name__)


def web_search_tool(query) -> str:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: Either a JSON string or a dictionary containing search parameters.
               Format: {"query": "search terms", "max_results": 5}
               Or simple string with search terms.

    Returns:
        Search results in JSON format.
    """
    try:
        # Handle different input types
        if isinstance(query, dict):
            params = query
        else:
            # Try to parse as JSON if it's a string
            try:
                params = json.loads(query)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON or not a string, assume it's a search term
                params = {"query": query}

        # Extract parameters with defaults
        search_query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not search_query:
            return json.dumps({"error": "No search query provided"})

        # Validate max_results
        try:
            max_results = int(max_results)
            if max_results <= 0:
                return json.dumps({"error": "max_results must be positive"})
        except (TypeError, ValueError):
            return json.dumps({"error": "Invalid max_results value"})

        # Perform the search
        start_time = time.time()
        
        # Initialize DDGS and perform search
        search_results = list(DDGS().text(search_query, max_results=max_results))
        search_time = time.time() - start_time
        
        # Process results
        results = []
        for result in search_results:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "description": result.get("body", "")
            })
            
        # Add metadata about the search
        metadata = {
            "query": search_query,
            "result_count": len(results),
            "search_time_seconds": round(search_time, 2)
        }
        
        return json.dumps({"results": results, "metadata": metadata}, indent=2)
        
    except Exception as e:
        logger.error(f"Error in web_search_tool: {str(e)}")
        return json.dumps({"error": f"Search failed: {str(e)}"})
