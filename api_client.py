"""
API client for backend communication.
"""
import requests
from typing import Dict, Any
import numpy as np
from config import ENDPOINT

def search_images(
    embedding: np.ndarray,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Search for images using the API.
    
    Args:
        embedding: L2-normalized embedding vector
        timeout: Request timeout in seconds
    
    Returns:
        JSON response from the API
    """
    payload = {
        "terms": [{"embedding": embedding.tolist(), "weight": 1}],
    }
    
    response = requests.post(ENDPOINT, json=payload, timeout=timeout)
    response.raise_for_status()
    
    return response.json()
