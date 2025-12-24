"""
Prompt template loader utility.

This module provides functions to load prompt templates from YAML files.
"""

import os
import yaml
from pathlib import Path

# Get the directory where this file is located
_PROMPTS_DIR = Path(__file__).parent


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from a YAML file.
    
    Args:
        template_name: Name of the template file (without .yaml extension)
    
    Returns:
        The prompt template string
    
    Raises:
        FileNotFoundError: If the template file doesn't exist
    """
    template_path = _PROMPTS_DIR / f"{template_name}.yaml"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # YAML files should have a 'template' key with the prompt string
    if 'template' not in data:
        raise ValueError(f"YAML file {template_name}.yaml must contain a 'template' key")
    
    return data['template']


def load_prompt_template_with_metadata(template_name: str) -> dict:
    """
    Load a prompt template with its metadata from a YAML file.
    
    Args:
        template_name: Name of the template file (without .yaml extension)
    
    Returns:
        Dictionary with 'template' and any other metadata keys
    """
    template_path = _PROMPTS_DIR / f"{template_name}.yaml"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data

