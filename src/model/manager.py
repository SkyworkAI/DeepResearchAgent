"""
Model manager for OpenAI models.
Provides a unified interface for registering and invoking OpenAI models.
"""

import os
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.model.types import ModelConfig, LLMResponse, LLMExtra
from src.model.openai.chat import ChatOpenAI
from src.model.openai.response import ResponseOpenAI
from src.model.openai.transcribe import TranscribeOpenAI
from src.model.openai.embedding import EmbeddingOpenAI
from src.model.openrouter.chat import ChatOpenRouter
from src.model.anthropic.chat import ChatAnthropic
from src.model.google.chat import ChatGoogle
from src.model.litellm.chat import ChatLiteLLM
from src.message.types import Message
from src.logger import logger

if TYPE_CHECKING:
    from src.tool.types import Tool


class ModelManager:
    """
    Central registry and invoker for OpenAI models.

    Responsibilities:
    1. Register and store model configurations.
    2. Provide a unified invocation interface using ChatOpenAI or ResponseOpenAI.
    """
    
    def __init__(self):
        """Initialize the manager."""
        self.models: Dict[str, ModelConfig] = {}
        self.model_clients: Dict[str, Union[ChatOpenAI, ResponseOpenAI, TranscribeOpenAI, EmbeddingOpenAI, ChatOpenRouter, ChatAnthropic, ChatLiteLLM]] = {}
        
        # Default parameters
        self.max_tokens: int = 16384
        self.default_temperature: float = 0.7
        self.default_reasoning: Dict[str, Any] = {
            "reasoning_effort": "high"
        }
        self.default_plugins: Optional[List[Dict[str, Any]]] = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "mistral-ocr"
                }
            },
            {
                "id": "web", "max_results": 10
            },
            { 
                "id": "response-healing"
            }
        ]
    
    async def initialize(self):
        """Initialize the manager and register default models."""
        await self._initialize_openai_models()
        await self._initialize_openrouter_models()
        await self._initialize_anthropic_models()
        await self._initialize_google_models()
        await self._initialize_litellm_models()
        logger.info(f"| Model manager initialized successfully with {len(self.models)} models.")
    
    async def _initialize_openai_models(self):
        """Initialize OpenAI models."""
        # General chat/completions models
        chat_models = [
            {
                "model_name": "openai/gpt-4o",
                "model_id": "gpt-4o",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-4.1"
            },
            {
                "model_name": "openai/gpt-4.1",
                "model_type": "chat/completions",
                "model_id": "gpt-4.1",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-4o",
            },
        ]
        
        # Responses API models
        response_models = [
            {
                "model_name": "openai/gpt-5",
                "model_type": "responses",
                "model_id": "gpt-5",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/o3",
            },
            {
                "model_name": "openai/gpt-5.1",
                "model_type": "responses",
                "model_id": "gpt-5.1",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/o3",
                "model_type": "responses",
                "model_id": "o3",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5.1",
            },
            {
                "model_name": "openai/gpt-5.2",
                "model_type": "responses",
                "model_id": "gpt-5.1",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.3",
                "model_type": "responses",
                "model_id": "gpt-5.3",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.4",
                "model_type": "responses",
                "model_id": "gpt-5.4",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            },
            {
                "model_name": "openai/gpt-5.4-pro",
                "model_type": "responses",
                "model_id": "gpt-5.4-pro",
                "reasoning": self.default_reasoning,
                "max_output_tokens": self.max_tokens,
                "fallback_model": "openai/gpt-5",
            }
        ]
        
        # Transcription models
        transcribe_models = [
            {
                "model_name": "openai/gpt-4o-transcribe",
                "model_type": "transcriptions",
                "model_id": "gpt-4o-transcribe",
                "fallback_model": "openai/gpt-4o-transcribe",
            }
        ]
        
        # Embedding models
        embedding_models = [
            {
                "model_name": "openai/text-embedding-3-small",
                "model_type": "embeddings",
                "model_id": "text-embedding-3-small",
                "fallback_model": "openai/text-embedding-3-large",
            },
            {
                "model_name": "openai/text-embedding-3-large",
                "model_type": "embeddings",
                "model_id": "text-embedding-3-large",
                "fallback_model": "openai/text-embedding-3-large",
            },
            {
                "model_name": "openai/text-embedding-ada-002",
                "model_type": "embeddings",
                "model_id": "text-embedding-ada-002",
                "fallback_model": "openai/text-embedding-3-large",
            },
        ]
        
        # Register chat models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                api_base=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
        
        # Register response models
        for model in response_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",  
                api_base=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
                reasoning=model.get("reasoning"),
                max_output_tokens=model.get("max_output_tokens"),
                supports_streaming=False,  # Responses API may not support streaming
                supports_functions=False,  # Responses API may not support tools
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
        
        # Register transcription models
        for model in transcribe_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                api_base=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
                supports_streaming=False,
                supports_functions=False,
                supports_vision=False,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
        
        # Register embedding models
        for model in embedding_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openai",
                api_base=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
                supports_streaming=False,
                supports_functions=False,
                supports_vision=False,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_openrouter_models(self):
        """Initialize OpenRouter models (OpenAI models via OpenRouter)."""
        chat_models = [
            # OpenAI models
            {
                "model_name": "openrouter/gpt-4o",
                "model_id": "openai/gpt-4o",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-4.1",
                "model_id": "openai/gpt-4.1",
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5",
                "model_id": "openai/gpt-5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5.1",
                "model_id": "openai/gpt-5.1",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5.2",
                "model_id": "openai/gpt-5.2",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5.3",
                "model_id": "openai/gpt-5.3",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5.4",
                "model_id": "openai/gpt-5.4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/gpt-5.4-pro",
                "model_id": "openai/gpt-5.4-pro",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/o3",
                "model_id": "openai/o3",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": 1.0,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            # Anthropic models
            {
                "model_name": "openrouter/claude-sonnet-3.5",
                "model_id": "anthropic/claude-3.5-sonnet",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-3.7",
                "model_id": "anthropic/claude-3.7-sonnet",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4",
                "model_id": "anthropic/claude-sonnet-4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4",
                "model_id": "anthropic/claude-opus-4",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4.5",
                "model_id": "anthropic/claude-sonnet-4.5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4.5",
                "model_id": "anthropic/claude-opus-4.5",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-sonnet-4.6",
                "model_id": "anthropic/claude-sonnet-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/claude-opus-4.6",
                "model_id": "anthropic/claude-opus-4.6",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            # Gemini models for chat, vision, audio, video, pdf, etc.
            {
                "model_name": "openrouter/gemini-2.5-flash",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-flash",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-2.5-pro",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-pro",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-3-pro-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-3.1-pro-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3.1-pro-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-3-flash-preview",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-flash-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-2.5-flash-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-2.5-flash",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-2.5-flash",
            },
            {
                "model_name": "openrouter/gemini-3-flash-preview-plugins",
                "model_type": "chat/completions",
                "model_id": "google/gemini-3-flash-preview",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "plugins": self.default_plugins,
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/gemini-3-flash-preview",
            },
            # Qwen models
            {
                "model_name": "openrouter/qwen3-coder",
                "model_id": "qwen/qwen3-coder",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            {
                "model_name": "openrouter/qwen3-max",
                "model_id": "qwen/qwen3-max",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            #deepseek models
            {
                "model_name": "openrouter/deepseek-v3.2",
                "model_id": "deepseek/deepseek-v3.2",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            },
            # X-ai models
            {
                "model_name": "openrouter/grok-4.1-fast",
                "model_id": "x-ai/grok-4.1-fast",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "openrouter/o3",
            }
        ]
        
        # Register OpenRouter models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="openrouter",
                api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                reasoning=model.get("reasoning") if model.get("reasoning") else None,
                plugins=model.get("plugins") if model.get("plugins") else None,
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_anthropic_models(self):
        """Initialize Anthropic models."""
        chat_models = [
            # Note: Claude 3.5 Sonnet is deprecated, use Claude 3.7 Sonnet or Claude Sonnet 4.5 instead
            # {
            #     "model_name": "anthropic/claude-sonnet-3.5",
            #     "model_id": "claude-3-5-sonnet-20240620",
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #     "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
            {
                "model_name": "anthropic/claude-sonnet-3.7",
                "model_id": "claude-3-7-sonnet-20250219",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            {
                "model_name": "anthropic/claude-sonnet-4",
                "model_id": "claude-sonnet-4-20250514",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            # {
            #     "model_name": "anthropic/claude-opus-4",
            #     "model_id": "claude-opus-4-20250514",  # Anthropic model ID with version
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #     "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
            {
                "model_name": "anthropic/claude-sonnet-4.5",
                "model_id": "claude-sonnet-4-5-20250929",  # Anthropic model ID with version
                "model_type": "chat/completions",
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
                "fallback_model": "anthropic/claude-sonnet-4.5",
            },
            # {
            #     "model_name": "anthropic/claude-opus-4.5",
            #     "model_id": "claude-opus-4-1-20250805",  # Correct Anthropic model ID (Opus 4.1)
            #     "model_type": "chat/completions",
            #     "temperature": self.default_temperature,
            #     "max_completion_tokens": self.max_tokens,
            #    "fallback_model": "anthropic/claude-sonnet-4.5",
            # },
        ]
        
        # Register Anthropic models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="anthropic",
                api_base=os.getenv("ANTHROPIC_API_BASE"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=model.get("fallback_model"),
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_google_models(self):
        """Initialize Google Gemini models."""
        chat_models = [
            {
                "model_name": "google/gemini-2.5-flash",
                "model_id": "gemini-2.5-flash",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
            },
            {
                "model_name": "google/gemini-2.5-pro",
                "model_id": "gemini-2.5-pro",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
            },
            {
                "model_name": "google/gemini-3-pro-preview",
                "model_id": "gemini-3-pro-preview",
                "model_type": "chat/completions",
                "reasoning": {
                    "reasoning": {
                        "enabled": True
                    }
                },
                "temperature": self.default_temperature,
                "max_completion_tokens": self.max_tokens,
            },
        ]
        
        # Register Google models
        for model in chat_models:
            config = ModelConfig(
                model_name=model["model_name"],
                model_id=model["model_id"],
                model_type=model["model_type"],
                provider="google",
                api_base=None,  # Google Gemini doesn't use custom API base
                api_key=os.getenv("GOOGLE_API_KEY"),
                reasoning=model.get("reasoning"),
                temperature=model.get("temperature"),
                max_completion_tokens=model.get("max_completion_tokens"),
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                output_version=None,
                fallback_model=None,
            )
            self.models[config.model_name] = config
            await self._create_client(config)
    
    async def _initialize_litellm_models(self):
        """Initialize LiteLLM models — only if LITELLM_API_KEY or LITELLM_MODEL is set."""
        if not os.getenv("LITELLM_API_KEY") and not os.getenv("LITELLM_MODEL"):
            return

        litellm_model = os.getenv("LITELLM_MODEL", "openai/gpt-4o")
        config = ModelConfig(
            model_name=f"litellm/{litellm_model}",
            model_id=litellm_model,
            model_type="chat/completions",
            provider="litellm",
            api_base=os.getenv("LITELLM_API_BASE"),
            api_key=os.getenv("LITELLM_API_KEY"),
            temperature=self.default_temperature,
            max_completion_tokens=self.max_tokens,
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
            output_version=None,
            fallback_model=None,
        )
        self.models[config.model_name] = config
        await self._create_client(config)

    async def _create_client(self, config: ModelConfig) -> None:
        """Create and cache a client for the given model config."""
        if config.provider == "litellm":
            client = ChatLiteLLM(
                model=config.model_id,
                api_key=config.api_key,
                api_base=config.api_base,
                reasoning=config.reasoning if config.reasoning else None,
                temperature=config.temperature or self.default_temperature,
                max_completion_tokens=config.max_completion_tokens or self.max_tokens,
            )
            logger.info(f"| Created ChatLiteLLM client for {config.model_name}")
        elif config.provider == "openrouter":
            # OpenRouter models (only chat/completions supported for now)
            if config.model_type == "chat/completions":
                client = ChatOpenRouter(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    reasoning=config.reasoning if config.reasoning else None,
                    plugins=config.plugins if config.plugins else None,
                    temperature=config.temperature or self.default_temperature,
                    max_completion_tokens=config.max_completion_tokens or self.max_tokens,
                    http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
                    x_title=os.getenv("OPENROUTER_X_TITLE"),
                )
                logger.info(f"| Created ChatOpenRouter client for {config.model_name}")
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for OpenRouter provider")
        elif config.provider == "anthropic":
            # Anthropic models (only chat/completions supported for now)
            if config.model_type == "chat/completions":
                client = ChatAnthropic(
                    model=config.model_id,
                    api_key=config.api_key,
                    base_url=config.api_base,
                    reasoning=config.reasoning if config.reasoning else None,
                    temperature=config.temperature or self.default_temperature,
                    max_tokens=config.max_completion_tokens or self.max_tokens,
                )
                logger.info(f"| Created ChatAnthropic client for {config.model_name}")
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for Anthropic provider")
        elif config.provider == "google":
            # Google Gemini models (only chat/completions supported for now)
            if config.model_type == "chat/completions":
                client = ChatGoogle(
                    model=config.model_id,
                    api_key=config.api_key,
                    reasoning=config.reasoning if config.reasoning else None,
                    temperature=config.temperature or self.default_temperature,
                    max_output_tokens=config.max_completion_tokens or self.max_tokens,
                )
                logger.info(f"| Created ChatGoogle client for {config.model_name}")
            else:
                raise ValueError(f"Unsupported model type {config.model_type} for Google provider")
        elif config.model_type == "responses":
            # Create ResponseOpenAI client
            client = ResponseOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                reasoning=config.reasoning if config.reasoning else None,
                max_output_tokens=config.max_output_tokens or self.max_tokens,
            )
            logger.info(f"| Created ResponseOpenAI client for {config.model_name}")
        elif config.model_type == "transcriptions":
            # Create TranscribeOpenAI client
            client = TranscribeOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
            )
            logger.info(f"| Created TranscribeOpenAI client for {config.model_name}")
        elif config.model_type == "embeddings":
            # Create EmbeddingOpenAI client
            client = EmbeddingOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
            )
            logger.info(f"| Created EmbeddingOpenAI client for {config.model_name}")
        else:
            # Create ChatOpenAI client
            client = ChatOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                temperature=config.temperature or self.default_temperature,
                reasoning=config.reasoning if config.reasoning else None,
                max_completion_tokens=config.max_completion_tokens or self.max_tokens,
            )
            logger.info(f"| Created ChatOpenAI client for {config.model_name}")
            
        self.model_clients[config.model_name] = client
    
    async def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        if config.provider not in ["openai", "openrouter", "anthropic", "google", "litellm"]:
            raise ValueError(f"Only OpenAI, OpenRouter, Anthropic, and Google models are supported. Got provider: {config.provider}")
        
        self.models[config.model_name] = config
        await self._create_client(config)
        logger.info(f"Registered model: {config.model_name}")
    
    def _log_usage(self, model_name: str, result: LLMResponse) -> None:
        """Log token usage and estimated cost after a model call."""
        if not result.success or not result.extra or not result.extra.data:
            return
        usage = result.extra.data.get("usage")
        if not usage:
            return

        # Normalize token field names across providers
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count") or 0
        total_tokens = usage.get("total_tokens") or usage.get("total_token_count") or (input_tokens + output_tokens)

        # Some providers (e.g. OpenRouter) may include cost directly
        cost = usage.get("cost")
        # Anthropic may include cache-related fields
        cache_creation = usage.get("cache_creation_input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0

        parts = [
            f"model={model_name}",
            f"input={input_tokens}",
            f"output={output_tokens}",
            f"total={total_tokens}",
        ]
        if cache_creation or cache_read:
            parts.append(f"cache_create={cache_creation}")
            parts.append(f"cache_read={cache_read}")
        if cost is not None:
            parts.append(f"cost=${cost:.6f}")

        logger.info(f"| 💰 Usage: {', '.join(parts)}")

    async def __call__(
        self,
        model: str,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[BaseModel, Dict]] = None,
        stream: bool = False,
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Invoke a model asynchronously.
        
        Args:
            model: Model name
            messages: List of Message objects (for transcription models, should contain ContentPartAudio)
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model or dict)
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with formatted message
        """
        # Validate that tools and response_format are not both provided
        if tools and response_format:
            raise ValueError("tools and response_format cannot be used together")
        
        current_model = model
        try:
            # Get client for the model
            client = self.model_clients.get(current_model)
            if not client:
                raise ValueError(f"Model {current_model} not found. Available models: {list(self.models.keys())}")
            
            # Check model type and call appropriate method
            model_config = self.models.get(current_model)
            
            if model_config and model_config.model_type == "transcriptions":
                # Transcription models use messages parameter (extracts audio from messages)
                if messages is None:
                    raise ValueError("messages parameter is required for transcription models")
                result = await client(
                    messages=messages,
                    **kwargs,
                )
            elif model_config and model_config.model_type == "embeddings":
                # Embedding models use messages parameter (extracts text from messages)
                # Filter out unsupported parameters (tools, response_format, stream)
                embedding_kwargs = {k: v for k, v in kwargs.items() if k not in ['tools', 'response_format', 'stream']}
                if messages is None:
                    raise ValueError("messages parameter is required for embedding models")
                result = await client(
                    messages=messages,
                    **embedding_kwargs,
                )
            else:
                # Chat/response models use messages parameter
                if messages is None:
                    raise ValueError("messages parameter is required for chat/response models")
                result = await client(
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    stream=stream,
                    plugins=plugins,
                    **kwargs,
                )
            
            self._log_usage(current_model, result)
            return result
            
        except Exception as e:
            # Check if we have a fallback model
            model_config = self.models.get(current_model)
            if model_config and model_config.fallback_model:
                fallback_model = model_config.fallback_model
                logger.warning(f"| Primary model {current_model} failed, falling back to {fallback_model}: {e}")
                try:
                    # Retry with fallback model
                    fallback_client = self.model_clients.get(fallback_model)
                    if not fallback_client:
                        raise ValueError(f"Fallback model {fallback_model} not found")
                    
                    # Check fallback model type
                    fallback_config = self.models.get(fallback_model)
                    if fallback_config and fallback_config.model_type == "transcriptions":
                        # Transcription models use messages parameter (extracts audio from messages)
                        if messages is None:
                            raise ValueError("messages parameter is required for transcription models")
                        result = await fallback_client(
                            messages=messages,
                            **kwargs,
                        )
                    elif fallback_config and fallback_config.model_type == "embeddings":
                        # Embedding models use messages parameter (extracts text from messages)
                        # Filter out unsupported parameters (tools, response_format, stream)
                        embedding_kwargs = {k: v for k, v in kwargs.items() if k not in ['tools', 'response_format', 'stream']}
                        if messages is None:
                            raise ValueError("messages parameter is required for embedding models")
                        result = await fallback_client(
                            messages=messages,
                            **embedding_kwargs,
                        )
                    else:
                        # Chat/response models use messages parameter
                        if messages is None:
                            raise ValueError("messages parameter is required for chat/response models")
                        result = await fallback_client(
                            messages=messages,
                            tools=tools,
                            response_format=response_format,
                            stream=stream,
                            plugins=plugins,
                            **kwargs,
                        )
                    logger.info(f"| Fallback model {fallback_model} succeeded")
                    self._log_usage(fallback_model, result)
                    return result
                except Exception as fallback_error:
                    logger.error(f"| Fallback model {fallback_model} also failed: {fallback_error}")
                    return LLMResponse(
                        success=False,
                        message=f"Both primary model ({current_model}) and fallback model ({fallback_model}) failed. Primary error: {e}, Fallback error: {fallback_error}",
                        extra=None
                    )
            
            logger.error(f"Model invocation failed: {e}")
            return LLMResponse(
                success=False,
                message=str(e)
            )
    
    async def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get the configuration for a model."""
        return await self.models.get(model)
    
    async def list(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())


    # Global singleton instance
model_manager = ModelManager()

