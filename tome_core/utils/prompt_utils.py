"""
Prompt utilities for OCR processing.
"""

from typing import Dict

try:
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    OLMOC_AVAILABLE = True
except ImportError:
    OLMOC_AVAILABLE = False
    build_no_anchoring_v4_yaml_prompt = None


def build_layout_preservation_prompt() -> str:
    """
    Build a prompt for layout-preserving OCR.
    
    Returns:
        Formatted prompt string
    """
    prompt = """Please analyze this image and extract all text content while preserving the original layout and formatting as much as possible.

Include:
1. All visible text content
2. Structural elements (headings, paragraphs, lists)
3. Special formatting (bold, italic, etc.)
4. Tables and their structure if present
5. Any other textual elements

Maintain the relative positioning and structure of the text elements."""
    
    return prompt


def build_simple_text_extraction_prompt() -> str:
    """
    Build a simple text extraction prompt.
    
    Returns:
        Formatted prompt string
    """
    prompt = "Extract all text content from this image."
    
    return prompt


def build_structured_data_prompt() -> str:
    """
    Build a prompt for structured data extraction.
    
    Returns:
        Formatted prompt string
    """
    prompt = """Extract all text from this image and organize it in a structured format.

Please identify and preserve:
- Document title and headings
- Paragraphs and their hierarchy
- Lists and bullet points
- Tables and their structure
- Any other structured elements

Provide the output in a clear, organized format."""
    
    return prompt


# Dictionary mapping prompt modes to actual prompts (from the original code)
dict_promptmode_to_prompt: Dict[str, str] = {
    "prompt_layout_all_en": build_layout_preservation_prompt(),
    "prompt_simple_extraction": build_simple_text_extraction_prompt(),
    "prompt_structured_data": build_structured_data_prompt(),
}

# Add olmocr prompt if available
if OLMOC_AVAILABLE and build_no_anchoring_v4_yaml_prompt is not None:
    dict_promptmode_to_prompt["prompt_no_anchoring_v4_yaml"] = build_no_anchoring_v4_yaml_prompt()


def get_prompt_by_mode(prompt_mode: str) -> str:
    """
    Get prompt text by mode.
    
    Args:
        prompt_mode: Mode identifier for the prompt
        
    Returns:
        Prompt text for the given mode
        
    Raises:
        ValueError: If prompt mode is not found
    """
    if prompt_mode not in dict_promptmode_to_prompt:
        available_modes = list(dict_promptmode_to_prompt.keys())
        raise ValueError(f"Invalid prompt_mode '{prompt_mode}'. Available modes: {available_modes}")
    
    return dict_promptmode_to_prompt[prompt_mode]


def list_available_prompt_modes() -> list:
    """
    List all available prompt modes.
    
    Returns:
        List of available prompt mode strings
    """
    return list(dict_promptmode_to_prompt.keys())


def add_custom_prompt_mode(mode_name: str, prompt_text: str) -> None:
    """
    Add a custom prompt mode.
    
    Args:
        mode_name: Name for the new prompt mode
        prompt_text: The prompt text
    """
    dict_promptmode_to_prompt[mode_name] = prompt_text