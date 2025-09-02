# Centralized OpenAI LLM Client Guide

## Overview

The Cybersecurity Agent now uses a centralized OpenAI LLM client component that provides a unified interface for all AI-powered features across the system. This centralization improves maintainability, reduces code duplication, and provides consistent error handling, cost tracking, and usage statistics.

## Architecture

### Core Components

1. **`bin/openai_llm_client.py`** - Centralized LLM client
2. **`bin/cs_ai_tools.py`** - Updated to use centralized client
3. **`bin/langgraph_cybersecurity_agent.py`** - Updated to use centralized client

### Key Features

- **Unified Configuration**: Single point for OpenAI API configuration
- **Cost Tracking**: Centralized cost calculation and usage statistics
- **Error Handling**: Consistent retry logic and error management
- **Model Management**: Easy model selection and configuration
- **Response Formatting**: Support for text, JSON, and markdown responses
- **Usage Analytics**: Comprehensive usage statistics and monitoring

## Usage

### Basic Text Generation

```python
from bin.openai_llm_client import OpenAILLMClient, LLMConfig, ModelType

# Create client
client = OpenAILLMClient()

# Check availability
if client.is_available():
    # Generate text
    response = client.generate_text(
        prompt="What are the key principles of cybersecurity?",
        system_prompt="You are a cybersecurity expert.",
        config=LLMConfig(
            model=ModelType.GPT_4,
            max_tokens=500,
            temperature=0.3
        )
    )
    
    if response.success:
        print(f"Generated: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost:.4f}")
    else:
        print(f"Error: {response.error}")
```

### Structured JSON Responses

```python
# Generate structured JSON response
response = client.generate_structured_response(
    prompt="Analyze this security log and extract key events",
    system_prompt="You are a security analyst.",
    schema={
        "type": "object",
        "properties": {
            "threats": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "recommendations": {"type": "array", "items": {"type": "string"}}
        }
    }
)
```

### Text Analysis

```python
# Analyze text for security threats
response = client.analyze_text(
    text="Suspicious network activity detected...",
    analysis_type="security",
    context="Network monitoring logs"
)
```

### Content Summarization

```python
# Summarize content with focus areas
response = client.summarize_content(
    content="Long security report...",
    max_length=300,
    focus_areas=["threats", "vulnerabilities", "recommendations"]
)
```

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key-here"

# Optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # For custom endpoints
```

### Model Selection

```python
from bin.openai_llm_client import ModelType

# Available models
ModelType.GPT_4              # Most capable, highest cost
ModelType.GPT_4_TURBO        # Balanced performance and cost
ModelType.GPT_3_5_TURBO      # Fast and cost-effective
ModelType.GPT_3_5_TURBO_16K  # Extended context window
```

### Response Formats

```python
from bin.openai_llm_client import ResponseFormat

ResponseFormat.TEXT      # Plain text (default)
ResponseFormat.JSON      # Structured JSON
ResponseFormat.MARKDOWN  # Markdown formatted
```

## Integration Points

### MCP Server Integration

The MCP server automatically uses the centralized LLM client for all AI-powered tools:

```python
from bin.cs_ai_tools import MCPServer

server = MCPServer()
# LLM client is automatically initialized and available
# All AI tools use the same client instance
```

### LangGraph Agent Integration

The main agent uses the centralized client for patent analysis and other AI features:

```python
from bin.langgraph_cybersecurity_agent import LangGraphCybersecurityAgent

agent = LangGraphCybersecurityAgent()
# LLM client is automatically available for AI-powered workflows
```

## Cost Management

### Usage Statistics

```python
# Get current usage statistics
stats = client.get_usage_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost:.4f}")
print(f"Success rate: {stats.successful_requests}/{stats.total_requests}")
```

### Cost Estimates

The client provides real-time cost estimates based on current OpenAI pricing:

- **GPT-4**: $0.03 per 1K input tokens, $0.06 per 1K output tokens
- **GPT-4 Turbo**: $0.01 per 1K input tokens, $0.03 per 1K output tokens
- **GPT-3.5 Turbo**: $0.002 per 1K input tokens, $0.002 per 1K output tokens

### Reset Statistics

```python
# Reset usage statistics
client.reset_usage_stats()
```

## Error Handling

### Retry Logic

The client includes automatic retry logic with exponential backoff:

```python
config = LLMConfig(
    retry_attempts=3,      # Number of retry attempts
    retry_delay=1.0,       # Initial delay between retries
    timeout=30             # Request timeout in seconds
)
```

### Error Responses

All methods return structured responses with error information:

```python
response = client.generate_text("test prompt")

if not response.success:
    print(f"Error: {response.error}")
    print(f"Response time: {response.response_time:.2f}s")
```

## Testing

### Connection Test

```python
# Test API connectivity
if client.test_connection():
    print("✅ OpenAI API connection successful")
else:
    print("❌ OpenAI API connection failed")
```

### Availability Check

```python
# Check if client is available
if client.is_available():
    print("✅ LLM client is ready")
else:
    print("❌ LLM client not available (check API key)")
```

## Migration from Legacy Code

### Before (Legacy)

```python
# Old scattered approach
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "prompt"}],
    max_tokens=1000
)
```

### After (Centralized)

```python
# New centralized approach
from bin.openai_llm_client import OpenAILLMClient, LLMConfig, ModelType

client = OpenAILLMClient()
response = client.generate_text(
    prompt="prompt",
    config=LLMConfig(model=ModelType.GPT_4, max_tokens=1000)
)
```

## Benefits

1. **Consistency**: All AI features use the same client and configuration
2. **Maintainability**: Single point of change for OpenAI integration
3. **Cost Control**: Centralized cost tracking and usage monitoring
4. **Error Handling**: Unified retry logic and error management
5. **Performance**: Optimized connection handling and response processing
6. **Extensibility**: Easy to add new models or features
7. **Testing**: Simplified testing with mock clients
8. **Security**: Centralized API key management

## Best Practices

1. **Always check availability** before making requests
2. **Use appropriate models** for the task (GPT-3.5 for simple tasks, GPT-4 for complex analysis)
3. **Monitor usage statistics** to track costs and performance
4. **Handle errors gracefully** with proper fallback mechanisms
5. **Use structured responses** for data extraction tasks
6. **Set appropriate timeouts** for different use cases
7. **Reset statistics** periodically for accurate tracking

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `OPENAI_API_KEY` environment variable is set
   - Check that the API key is valid and has sufficient credits

2. **Connection Timeouts**
   - Increase timeout in `LLMConfig`
   - Check network connectivity
   - Verify API endpoint accessibility

3. **Rate Limiting**
   - Implement request queuing
   - Use exponential backoff
   - Consider upgrading API plan

4. **High Costs**
   - Use GPT-3.5 for simple tasks
   - Optimize prompts to reduce token usage
   - Monitor usage statistics regularly

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Caching**: Implement response caching for repeated queries
- **Streaming**: Support for streaming responses
- **Batch Processing**: Handle multiple requests efficiently
- **Custom Models**: Support for fine-tuned models
- **Multi-Provider**: Support for other LLM providers
- **Advanced Analytics**: Detailed usage analytics and reporting
