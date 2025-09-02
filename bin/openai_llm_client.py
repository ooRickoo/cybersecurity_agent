"""
Centralized OpenAI LLM Client Component

This module provides a unified interface for all OpenAI API interactions
across the Cybersecurity Agent system. It centralizes configuration,
error handling, cost tracking, and provides consistent API patterns.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import openai
from openai import OpenAI


class ModelType(Enum):
    """Supported OpenAI models."""
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"


class ResponseFormat(Enum):
    """Response format options."""
    TEXT = "text"
    JSON = "json_object"
    MARKDOWN = "markdown"


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    model: ModelType = ModelType.GPT_4
    max_tokens: int = 1000
    temperature: float = 0.3
    response_format: ResponseFormat = ResponseFormat.TEXT
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    success: bool
    content: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0
    model_used: str = ""
    error: Optional[str] = None
    response_time: float = 0.0


@dataclass
class UsageStats:
    """Usage statistics for tracking."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0


class OpenAILLMClient:
    """
    Centralized OpenAI LLM client for the Cybersecurity Agent.
    
    Provides a unified interface for all OpenAI API interactions with:
    - Centralized configuration and error handling
    - Cost tracking and usage statistics
    - Retry logic and timeout handling
    - Consistent response formatting
    - Model selection and optimization
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client: Optional[OpenAI] = None
        self.usage_stats = UsageStats()
        self.is_configured = False
        
        # Model cost estimates (per 1K tokens, as of 2024)
        self.cost_estimates = {
            ModelType.GPT_4.value: {"input": 0.03, "output": 0.06},
            ModelType.GPT_4_TURBO.value: {"input": 0.01, "output": 0.03},
            ModelType.GPT_3_5_TURBO.value: {"input": 0.002, "output": 0.002},
            ModelType.GPT_3_5_TURBO_16K.value: {"input": 0.003, "output": 0.004},
        }
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self) -> bool:
        """Initialize the OpenAI client and test connectivity."""
        try:
            if not self.api_key:
                self.logger.warning("OpenAI API key not found - LLM features will be disabled")
                return False
            
            # Create client with optional base URL
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self.client = OpenAI(**client_kwargs)
            
            # Test the connection with a minimal call
            test_response = self.client.chat.completions.create(
                model=ModelType.GPT_3_5_TURBO.value,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            
            self.is_configured = True
            self.logger.info("OpenAI LLM client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.is_configured = False
            return False
    
    def is_available(self) -> bool:
        """Check if the LLM client is available and configured."""
        return self.is_configured and self.client is not None
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            config: LLM configuration (uses defaults if None)
            
        Returns:
            LLMResponse with generated content or error
        """
        if not self.is_available():
            return LLMResponse(
                success=False,
                error="OpenAI client not available or not configured"
            )
        
        config = config or LLMConfig()
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request parameters
            request_params = {
                "model": config.model.value,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "timeout": config.timeout
            }
            
            # Add response format if specified
            if config.response_format == ResponseFormat.JSON:
                request_params["response_format"] = {"type": "json_object"}
            
            # Make the API call with retry logic
            response = self._make_api_call_with_retry(request_params, config)
            
            if response is None:
                return LLMResponse(
                    success=False,
                    error="API call failed after all retry attempts"
                )
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            model_used = response.model
            
            # Calculate cost
            cost = self._calculate_cost(model_used, tokens_used)
            
            # Update usage statistics
            response_time = time.time() - start_time
            self._update_usage_stats(tokens_used, cost, response_time, True)
            
            return LLMResponse(
                success=True,
                content=content,
                tokens_used=tokens_used,
                cost=cost,
                model_used=model_used,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_usage_stats(0, 0.0, response_time, False)
            
            return LLMResponse(
                success=False,
                error=f"LLM generation failed: {str(e)}",
                response_time=response_time
            )
    
    def generate_structured_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate structured JSON response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            schema: JSON schema for response structure
            config: LLM configuration
            
        Returns:
            LLMResponse with JSON content
        """
        if config is None:
            config = LLMConfig()
        
        config.response_format = ResponseFormat.JSON
        
        # Enhance system prompt with JSON schema if provided
        enhanced_system_prompt = system_prompt or ""
        if schema:
            enhanced_system_prompt += f"\n\nRespond with valid JSON matching this schema: {schema}"
        
        return self.generate_text(prompt, enhanced_system_prompt, config)
    
    def analyze_text(
        self,
        text: str,
        analysis_type: str,
        context: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Analyze text for specific purposes (security, sentiment, etc.).
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (e.g., "security", "sentiment", "classification")
            context: Additional context for analysis
            config: LLM configuration
            
        Returns:
            LLMResponse with analysis results
        """
        system_prompt = f"""You are a cybersecurity expert. Analyze the provided text for {analysis_type}.
        
Provide a detailed analysis including:
1. Key findings
2. Risk assessment (if applicable)
3. Recommendations (if applicable)
4. Confidence level

Format your response in clear, structured markdown."""
        
        if context:
            system_prompt += f"\n\nAdditional context: {context}"
        
        prompt = f"Please analyze this text for {analysis_type}:\n\n{text}"
        
        if config is None:
            config = LLMConfig()
            config.response_format = ResponseFormat.MARKDOWN
        
        return self.generate_text(prompt, system_prompt, config)
    
    def summarize_content(
        self,
        content: str,
        max_length: int = 500,
        focus_areas: Optional[List[str]] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Summarize content with optional focus areas.
        
        Args:
            content: Content to summarize
            max_length: Maximum length of summary
            focus_areas: Specific areas to focus on
            config: LLM configuration
            
        Returns:
            LLMResponse with summary
        """
        system_prompt = f"""You are a cybersecurity expert. Create a concise summary of the provided content.
        
Summary requirements:
- Maximum {max_length} words
- Focus on key security insights and findings
- Highlight important technical details
- Include actionable recommendations if applicable"""
        
        if focus_areas:
            system_prompt += f"\n\nFocus on these specific areas: {', '.join(focus_areas)}"
        
        prompt = f"Please summarize this content:\n\n{content}"
        
        if config is None:
            config = LLMConfig()
            config.max_tokens = min(max_length * 2, 1000)  # Rough token estimate
        
        return self.generate_text(prompt, system_prompt, config)
    
    def _make_api_call_with_retry(
        self,
        request_params: Dict[str, Any],
        config: LLMConfig
    ) -> Optional[Any]:
        """Make API call with retry logic."""
        last_exception = None
        
        for attempt in range(config.retry_attempts):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        self.logger.error(f"All {config.retry_attempts} API call attempts failed")
        return None
    
    def _calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate approximate cost for API usage."""
        if model not in self.cost_estimates:
            # Default to GPT-4 pricing for unknown models
            model = ModelType.GPT_4.value
        
        cost_per_1k = self.cost_estimates[model]["input"]  # Simplified: use input cost
        return (tokens / 1000) * cost_per_1k
    
    def _update_usage_stats(
        self,
        tokens: int,
        cost: float,
        response_time: float,
        success: bool
    ) -> None:
        """Update usage statistics."""
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        
        if success:
            self.usage_stats.successful_requests += 1
        else:
            self.usage_stats.failed_requests += 1
        
        # Update average response time
        total_time = self.usage_stats.average_response_time * (self.usage_stats.total_requests - 1)
        self.usage_stats.average_response_time = (total_time + response_time) / self.usage_stats.total_requests
    
    def get_usage_stats(self) -> UsageStats:
        """Get current usage statistics."""
        return self.usage_stats
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = UsageStats()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [model.value for model in ModelType]
    
    def test_connection(self) -> bool:
        """Test the connection to OpenAI API."""
        if not self.is_available():
            return False
        
        try:
            response = self.generate_text(
                "Hello, this is a connection test.",
                config=LLMConfig(
                    model=ModelType.GPT_3_5_TURBO,
                    max_tokens=10,
                    temperature=0.0
                )
            )
            return response.success
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


# Convenience functions for backward compatibility
def create_llm_client(api_key: Optional[str] = None) -> OpenAILLMClient:
    """Create a new OpenAI LLM client instance."""
    return OpenAILLMClient(api_key)


def get_default_llm_client() -> Optional[OpenAILLMClient]:
    """Get a default LLM client instance (singleton pattern could be added here)."""
    client = OpenAILLMClient()
    return client if client.is_available() else None


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the client
    client = OpenAILLMClient()
    
    if client.is_available():
        print("✅ OpenAI LLM client is available")
        
        # Test basic text generation
        response = client.generate_text(
            "What are the key principles of cybersecurity?",
            system_prompt="You are a cybersecurity expert."
        )
        
        if response.success:
            print(f"✅ Generated text: {response.content[:100]}...")
            print(f"📊 Tokens used: {response.tokens_used}")
            print(f"💰 Cost: ${response.cost:.4f}")
        else:
            print(f"❌ Error: {response.error}")
        
        # Test connection
        if client.test_connection():
            print("✅ Connection test passed")
        else:
            print("❌ Connection test failed")
        
        # Show usage stats
        stats = client.get_usage_stats()
        print(f"📈 Usage stats: {stats.total_requests} requests, {stats.total_tokens} tokens, ${stats.total_cost:.4f} cost")
    
    else:
        print("❌ OpenAI LLM client is not available")
        print("   Make sure OPENAI_API_KEY environment variable is set")
