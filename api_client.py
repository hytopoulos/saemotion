"""
API client for Nooscope backend communication.
"""
import requests
from typing import Dict, Any
import numpy as np


def search_images(
    embedding: np.ndarray,
    endpoint: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Search for images using the Nooscope API.
    
    Args:
        embedding: L2-normalized embedding vector
        endpoint: API endpoint URL
        timeout: Request timeout in seconds
    
    Returns:
        JSON response from the API
    """
    payload = {
        "terms": [{"embedding": embedding.tolist(), "weight": 1}],
    }
    
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    
    return response.json()
