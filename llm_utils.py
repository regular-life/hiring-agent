"""
Utility functions for LLM providers.
"""

import logging
from typing import Any, Dict, Optional
from models import (
    ModelProvider,
    OllamaProvider,
    GeminiProvider,
    OpenRouterProvider,
    NvidiaProvider,
    GroqProvider,
    RoutingLLMProvider,
)
from prompt import (
    MODEL_PROVIDER_MAPPING,
    PROVIDER,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    NVIDIA_API_KEY,
    GROQ_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_MODEL,
    NVIDIA_MODEL,
    GROQ_MODEL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Text with markdown code block syntax removed
    """

    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8 :]

    # Remove leading ```json if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    # Remove trailing ``` if present
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name and environment settings.

    Args:
        model_name: The name of the model to use

    Returns:
        An initialized LLM provider instance
    """
    # 1. If USING ROUTER mode, load-balance / rotate calls among active providers
    if PROVIDER == "router":
        active_providers = []
        
        if GEMINI_API_KEY:
            logger.info(f"➕ Adding Gemini to router (model: {GEMINI_MODEL})")
            active_providers.append(("gemini", GeminiProvider(api_key=GEMINI_API_KEY), GEMINI_MODEL))
            
        if OPENROUTER_API_KEY:
            logger.info(f"➕ Adding OpenRouter to router (model: {OPENROUTER_MODEL})")
            active_providers.append(("openrouter", OpenRouterProvider(api_key=OPENROUTER_API_KEY), OPENROUTER_MODEL))
            
        if NVIDIA_API_KEY:
            logger.info(f"➕ Adding Nvidia to router (model: {NVIDIA_MODEL})")
            active_providers.append(("nvidia", NvidiaProvider(api_key=NVIDIA_API_KEY), NVIDIA_MODEL))
            
        if GROQ_API_KEY:
            logger.info(f"➕ Adding Groq to router (model: {GROQ_MODEL})")
            active_providers.append(("groq", GroqProvider(api_key=GROQ_API_KEY), GROQ_MODEL))

        # Ollama is always added as a local provider / fallback
        logger.info(f"➕ Adding Ollama to router (model: {OLLAMA_MODEL})")
        active_providers.append(("ollama", OllamaProvider(), OLLAMA_MODEL))

        logger.info(f"🔄 Initializing RoutingLLMProvider with {len(active_providers)} active providers.")
        return RoutingLLMProvider(providers=active_providers)

    # 2. Check the model mapping or explicitly set LLM_PROVIDER variables
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, None)
    if model_provider is None:
        try:
            model_provider = ModelProvider(PROVIDER)
        except ValueError:
            model_provider = ModelProvider.OLLAMA

    # Override provider based on LLM_PROVIDER environment config
    if PROVIDER == "gemini":
        model_provider = ModelProvider.GEMINI
    elif PROVIDER == "openrouter":
        model_provider = ModelProvider.OPENROUTER
    elif PROVIDER == "nvidia":
        model_provider = ModelProvider.NVIDIA
    elif PROVIDER == "groq":
        model_provider = ModelProvider.GROQ
    elif PROVIDER == "ollama":
        model_provider = ModelProvider.OLLAMA

    # 3. Instantiate the selected provider
    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
        return GeminiProvider(api_key=GEMINI_API_KEY)
        
    elif model_provider == ModelProvider.OPENROUTER:
        if not OPENROUTER_API_KEY:
            logger.warning("⚠️ OpenRouter API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using OpenRouter API provider with model {model_name}")
        return OpenRouterProvider(api_key=OPENROUTER_API_KEY)
        
    elif model_provider == ModelProvider.NVIDIA:
        if not NVIDIA_API_KEY:
            logger.warning("⚠️ NVIDIA API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using NVIDIA NIM API provider with model {model_name}")
        return NvidiaProvider(api_key=NVIDIA_API_KEY)
        
    elif model_provider == ModelProvider.GROQ:
        if not GROQ_API_KEY:
            logger.warning("⚠️ Groq API key not found. Falling back to Ollama.")
            return OllamaProvider()
        logger.info(f"🔄 Using Groq API provider with model {model_name}")
        return GroqProvider(api_key=GROQ_API_KEY)
        
    else:
        logger.info(f"🔄 Using Ollama provider with model {model_name}")
        return OllamaProvider()
