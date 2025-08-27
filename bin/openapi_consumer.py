"""
OpenAPI Consumer for Cybersecurity Agent

Provides capabilities to read OpenAPI specifications and quickly consume APIs
with automatic credential management and MCP tool generation.
"""

import json
import logging
import requests
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
import hashlib

from .credential_vault import CredentialVault
from .enhanced_session_manager import EnhancedSessionManager
from .context_memory_manager import ContextMemoryManager

logger = logging.getLogger(__name__)

class OpenAPIConsumer:
    """OpenAPI specification consumer with automatic API consumption capabilities."""
    
    def __init__(self, session_manager: EnhancedSessionManager, 
                 credential_vault: CredentialVault,
                 memory_manager: ContextMemoryManager):
        self.session_manager = session_manager
        self.credential_vault = credential_vault
        self.memory_manager = memory_manager
        self.api_specs = {}
        self.api_clients = {}
        self.generated_tools = {}
        
    def load_openapi_spec(self, spec_source: str, spec_type: str = "auto") -> Dict[str, Any]:
        """
        Load OpenAPI specification from various sources.
        
        Args:
            spec_source: Source of the OpenAPI spec (URL, file path, or raw content)
            spec_type: Type of specification (auto, url, file, content)
            
        Returns:
            Parsed OpenAPI specification
        """
        try:
            if spec_type == "auto":
                # Auto-detect source type
                if spec_source.startswith(('http://', 'https://')):
                    spec_type = "url"
                elif Path(spec_source).exists():
                    spec_type = "file"
                else:
                    spec_type = "content"
            
            if spec_type == "url":
                spec = self._load_from_url(spec_source)
            elif spec_type == "file":
                spec = self._load_from_file(spec_source)
            elif spec_type == "content":
                spec = self._load_from_content(spec_source)
            else:
                raise ValueError(f"Unsupported spec type: {spec_type}")
            
            # Validate and parse the specification
            parsed_spec = self._parse_openapi_spec(spec)
            
            # Store the specification
            spec_id = self._generate_spec_id(parsed_spec)
            self.api_specs[spec_id] = parsed_spec
            
            # Generate API client
            api_client = self._generate_api_client(parsed_spec)
            self.api_clients[spec_id] = api_client
            
            # Generate MCP tools
            mcp_tools = self._generate_mcp_tools(parsed_spec, api_client)
            self.generated_tools[spec_id] = mcp_tools
            
            # Store in memory for future use
            self.memory_manager.import_data(
                "openapi_specification",
                {
                    "spec_id": spec_id,
                    "info": parsed_spec.get("info", {}),
                    "servers": parsed_spec.get("servers", []),
                    "paths": list(parsed_spec.get("paths", {}).keys()),
                    "total_endpoints": len(parsed_spec.get("paths", {})),
                    "loaded_at": datetime.now().isoformat()
                },
                domain="api_integration",
                tier="long_term",
                ttl_days=90,
                metadata={
                    "description": f"OpenAPI specification for {parsed_spec.get('info', {}).get('title', 'Unknown API')}",
                    "type": "openapi_spec",
                    "source": spec_source
                }
            )
            
            logger.info(f"Successfully loaded OpenAPI spec: {spec_id}")
            return parsed_spec
            
        except Exception as e:
            logger.error(f"Failed to load OpenAPI spec: {e}")
            raise
    
    def _load_from_url(self, url: str) -> Dict[str, Any]:
        """Load OpenAPI specification from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'application/yaml' in content_type or 'text/yaml' in content_type:
                return yaml.safe_load(response.text)
            else:
                # Try to parse as JSON first, then YAML
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return yaml.safe_load(response.text)
                    
        except Exception as e:
            logger.error(f"Failed to load OpenAPI spec from URL {url}: {e}")
            raise
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load OpenAPI specification from file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"OpenAPI spec file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine file type and parse accordingly
            if file_path.suffix.lower() in ['.json']:
                return json.loads(content)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            else:
                # Try both formats
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return yaml.safe_load(content)
                    
        except Exception as e:
            logger.error(f"Failed to load OpenAPI spec from file {file_path}: {e}")
            raise
    
    def _load_from_content(self, content: str) -> Dict[str, Any]:
        """Load OpenAPI specification from raw content."""
        try:
            # Try to parse as JSON first, then YAML
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return yaml.safe_load(content)
                
        except Exception as e:
            logger.error(f"Failed to load OpenAPI spec from content: {e}")
            raise
    
    def _parse_openapi_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate OpenAPI specification."""
        try:
            # Basic validation
            if not isinstance(spec, dict):
                raise ValueError("OpenAPI specification must be a dictionary")
            
            # Check for required fields
            if 'openapi' not in spec and 'swagger' not in spec:
                raise ValueError("OpenAPI specification must contain 'openapi' or 'swagger' version")
            
            if 'paths' not in spec:
                raise ValueError("OpenAPI specification must contain 'paths'")
            
            # Normalize the specification
            normalized_spec = spec.copy()
            
            # Ensure info section exists
            if 'info' not in normalized_spec:
                normalized_spec['info'] = {
                    'title': 'Unknown API',
                    'version': '1.0.0',
                    'description': 'API loaded from specification'
                }
            
            # Ensure servers section exists
            if 'servers' not in normalized_spec:
                normalized_spec['servers'] = [
                    {'url': 'http://localhost:8080', 'description': 'Default server'}
                ]
            
            # Process paths and operations
            processed_paths = {}
            for path, path_item in normalized_spec.get('paths', {}).items():
                processed_paths[path] = self._process_path_item(path, path_item)
            
            normalized_spec['paths'] = processed_paths
            
            return normalized_spec
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAPI spec: {e}")
            raise
    
    def _process_path_item(self, path: str, path_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual path item in OpenAPI specification."""
        try:
            processed_item = path_item.copy()
            
            # Process each HTTP method
            for method in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                if method in processed_item:
                    operation = processed_item[method]
                    
                    # Ensure operation has an operationId
                    if 'operationId' not in operation:
                        operation['operationId'] = self._generate_operation_id(method, path)
                    
                    # Process parameters
                    if 'parameters' in operation:
                        operation['parameters'] = self._process_parameters(operation['parameters'])
                    
                    # Process request body
                    if 'requestBody' in operation:
                        operation['requestBody'] = self._process_request_body(operation['requestBody'])
                    
                    # Process responses
                    if 'responses' in operation:
                        operation['responses'] = self._process_responses(operation['responses'])
            
            return processed_item
            
        except Exception as e:
            logger.error(f"Failed to process path item {path}: {e}")
            return path_item
    
    def _generate_operation_id(self, method: str, path: str) -> str:
        """Generate operation ID from method and path."""
        try:
            # Clean path and convert to camelCase
            clean_path = re.sub(r'[{}]', '', path)  # Remove path parameters
            clean_path = re.sub(r'[^a-zA-Z0-9]', '_', clean_path)  # Replace special chars with underscore
            clean_path = clean_path.strip('_')  # Remove leading/trailing underscores
            
            # Convert to camelCase
            parts = clean_path.split('_')
            camel_case = parts[0] + ''.join(word.capitalize() for word in parts[1:])
            
            return f"{method}{camel_case.capitalize()}"
            
        except Exception as e:
            logger.warning(f"Failed to generate operation ID for {method} {path}: {e}")
            return f"{method}{hashlib.sha256(path.encode()).hexdigest()[:8]}"
    
    def _process_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process operation parameters."""
        try:
            processed_params = []
            
            for param in parameters:
                processed_param = param.copy()
                
                # Ensure required fields exist
                if 'name' not in processed_param:
                    continue
                
                if 'in' not in processed_param:
                    processed_param['in'] = 'query'
                
                if 'required' not in processed_param:
                    processed_param['required'] = False
                
                # Process schema if present
                if 'schema' in processed_param:
                    processed_param['schema'] = self._process_schema(processed_param['schema'])
                
                processed_params.append(processed_param)
            
            return processed_params
            
        except Exception as e:
            logger.warning(f"Failed to process parameters: {e}")
            return parameters
    
    def _process_request_body(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Process request body."""
        try:
            processed_body = request_body.copy()
            
            # Ensure required fields exist
            if 'required' not in processed_body:
                processed_body['required'] = False
            
            # Process content if present
            if 'content' in processed_body:
                for content_type, content_item in processed_body['content'].items():
                    if 'schema' in content_item:
                        content_item['schema'] = self._process_schema(content_item['schema'])
            
            return processed_body
            
        except Exception as e:
            logger.warning(f"Failed to process request body: {e}")
            return request_body
    
    def _process_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Process operation responses."""
        try:
            processed_responses = {}
            
            for status_code, response in responses.items():
                processed_response = response.copy()
                
                # Process content if present
                if 'content' in processed_response:
                    for content_type, content_item in processed_response['content'].items():
                        if 'schema' in content_item:
                            content_item['schema'] = self._process_schema(content_item['schema'])
                
                processed_responses[status_code] = processed_response
            
            return processed_responses
            
        except Exception as e:
            logger.warning(f"Failed to process responses: {e}")
            return responses
    
    def _process_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Process schema definition."""
        try:
            processed_schema = schema.copy()
            
            # Ensure type field exists
            if 'type' not in processed_schema:
                processed_schema['type'] = 'object'
            
            # Process properties if present
            if 'properties' in processed_schema:
                for prop_name, prop_schema in processed_schema['properties'].items():
                    processed_schema['properties'][prop_name] = self._process_schema(prop_schema)
            
            # Process items if present (for arrays)
            if 'items' in processed_schema:
                processed_schema['items'] = self._process_schema(processed_schema['items'])
            
            return processed_schema
            
        except Exception as e:
            logger.warning(f"Failed to process schema: {e}")
            return schema
    
    def _generate_spec_id(self, spec: Dict[str, Any]) -> str:
        """Generate unique ID for specification."""
        try:
            info = spec.get('info', {})
            title = info.get('title', 'Unknown')
            version = info.get('version', '1.0.0')
            
            # Create hash from title and version
            content = f"{title}{version}".encode()
            return hashlib.sha256(content).hexdigest()[:12]
            
        except Exception as e:
            logger.warning(f"Failed to generate spec ID: {e}")
            return hashlib.sha256(str(spec).encode()).hexdigest()[:12]
    
    def _generate_api_client(self, spec: Dict[str, Any]) -> 'APIClient':
        """Generate API client from specification."""
        try:
            return APIClient(spec, self.credential_vault, self.session_manager)
        except Exception as e:
            logger.error(f"Failed to generate API client: {e}")
            raise
    
    def _generate_mcp_tools(self, spec: Dict[str, Any], api_client: 'APIClient') -> Dict[str, Dict[str, Any]]:
        """Generate MCP tools from OpenAPI specification."""
        try:
            mcp_tools = {}
            
            for path, path_item in spec.get('paths', {}).items():
                for method, operation in path_item.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        operation_id = operation.get('operationId', f"{method}{path.replace('/', '_')}")
                        
                        # Generate tool definition
                        tool_def = self._create_mcp_tool_definition(operation_id, method, path, operation)
                        mcp_tools[operation_id] = tool_def
            
            return mcp_tools
            
        except Exception as e:
            logger.error(f"Failed to generate MCP tools: {e}")
            return {}
    
    def _create_mcp_tool_definition(self, operation_id: str, method: str, path: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Create MCP tool definition for an operation."""
        try:
            # Extract parameters
            parameters = {}
            if 'parameters' in operation:
                for param in operation['parameters']:
                    param_name = param.get('name', 'unknown')
                    param_type = self._map_openapi_type_to_mcp(param.get('schema', {}).get('type', 'string'))
                    param_desc = param.get('description', f'Parameter: {param_name}')
                    
                    parameters[param_name] = {
                        "type": param_type,
                        "description": param_desc,
                        "required": param.get('required', False)
                    }
            
            # Extract request body parameters
            if 'requestBody' in operation:
                request_body = operation['requestBody']
                if 'content' in request_body:
                    for content_type, content_item in request_body['content'].items():
                        if 'schema' in content_item:
                            schema = content_item['schema']
                            if 'properties' in schema:
                                for prop_name, prop_schema in schema['properties'].items():
                                    prop_type = self._map_openapi_type_to_mcp(prop_schema.get('type', 'string'))
                                    prop_desc = prop_schema.get('description', f'Request body property: {prop_name}')
                                    
                                    parameters[prop_name] = {
                                        "type": prop_type,
                                        "description": prop_desc,
                                        "required": prop_name in schema.get('required', [])
                                    }
            
            # Create tool definition
            tool_def = {
                "name": operation_id,
                "description": operation.get('summary', operation.get('description', f'{method.upper()} {path}')),
                "parameters": parameters,
                "returns": {
                    "type": "object",
                    "description": f"Response from {method.upper()} {path}"
                },
                "metadata": {
                    "method": method.upper(),
                    "path": path,
                    "operation_id": operation_id,
                    "tags": operation.get('tags', []),
                    "deprecated": operation.get('deprecated', False)
                }
            }
            
            return tool_def
            
        except Exception as e:
            logger.error(f"Failed to create MCP tool definition for {operation_id}: {e}")
            return {
                "name": operation_id,
                "description": f"API operation: {method.upper()} {path}",
                "parameters": {},
                "returns": {"type": "object", "description": "API response"},
                "metadata": {"method": method.upper(), "path": path}
            }
    
    def _map_openapi_type_to_mcp(self, openapi_type: str) -> str:
        """Map OpenAPI types to MCP types."""
        type_mapping = {
            'string': 'string',
            'integer': 'integer',
            'number': 'number',
            'boolean': 'boolean',
            'array': 'array',
            'object': 'object'
        }
        
        return type_mapping.get(openapi_type, 'string')
    
    def execute_api_operation(self, spec_id: str, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an API operation.
        
        Args:
            spec_id: ID of the OpenAPI specification
            operation_id: ID of the operation to execute
            **kwargs: Parameters for the operation
            
        Returns:
            API response and metadata
        """
        try:
            if spec_id not in self.api_clients:
                raise ValueError(f"API specification {spec_id} not found")
            
            api_client = self.api_clients[spec_id]
            return api_client.execute_operation(operation_id, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to execute API operation {operation_id}: {e}")
            raise
    
    def get_available_apis(self) -> List[Dict[str, Any]]:
        """Get list of available APIs."""
        try:
            apis = []
            
            for spec_id, spec in self.api_specs.items():
                api_info = {
                    "spec_id": spec_id,
                    "title": spec.get('info', {}).get('title', 'Unknown API'),
                    "version": spec.get('info', {}).get('version', '1.0.0'),
                    "description": spec.get('info', {}).get('description', ''),
                    "servers": spec.get('servers', []),
                    "total_endpoints": len(spec.get('paths', {})),
                    "available_operations": list(self.generated_tools.get(spec_id, {}).keys()),
                    "loaded_at": spec.get('loaded_at', 'unknown')
                }
                apis.append(api_info)
            
            return apis
            
        except Exception as e:
            logger.error(f"Failed to get available APIs: {e}")
            return []
    
    def get_api_operations(self, spec_id: str) -> List[Dict[str, Any]]:
        """Get list of operations for a specific API."""
        try:
            if spec_id not in self.generated_tools:
                return []
            
            operations = []
            for operation_id, tool_def in self.generated_tools[spec_id].items():
                operation_info = {
                    "operation_id": operation_id,
                    "description": tool_def.get('description', ''),
                    "method": tool_def.get('metadata', {}).get('method', ''),
                    "path": tool_def.get('metadata', {}).get('path', ''),
                    "parameters": tool_def.get('parameters', {}),
                    "tags": tool_def.get('metadata', {}).get('tags', []),
                    "deprecated": tool_def.get('metadata', {}).get('deprecated', False)
                }
                operations.append(operation_info)
            
            return operations
            
        except Exception as e:
            logger.error(f"Failed to get API operations for {spec_id}: {e}")
            return []

class APIClient:
    """API client for executing operations from OpenAPI specifications."""
    
    def __init__(self, spec: Dict[str, Any], credential_vault: CredentialVault, 
                 session_manager: EnhancedSessionManager):
        self.spec = spec
        self.credential_vault = credential_vault
        self.session_manager = session_manager
        self.base_url = self._get_base_url()
        self.auth_config = self._get_auth_config()
        
    def _get_base_url(self) -> str:
        """Get base URL from specification."""
        try:
            servers = self.spec.get('servers', [])
            if servers:
                return servers[0].get('url', 'http://localhost:8080')
            return 'http://localhost:8080'
        except Exception as e:
            logger.warning(f"Failed to get base URL: {e}")
            return 'http://localhost:8080'
    
    def _get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration from specification."""
        try:
            # Check for security schemes
            security_schemes = self.spec.get('components', {}).get('securitySchemes', {})
            
            auth_config = {}
            for scheme_name, scheme in security_schemes.items():
                scheme_type = scheme.get('type', 'unknown')
                
                if scheme_type == 'apiKey':
                    auth_config[scheme_name] = {
                        'type': 'apiKey',
                        'in': scheme.get('in', 'header'),
                        'name': scheme.get('name', 'Authorization')
                    }
                elif scheme_type == 'http':
                    auth_config[scheme_name] = {
                        'type': 'http',
                        'scheme': scheme.get('scheme', 'bearer')
                    }
                elif scheme_type == 'oauth2':
                    auth_config[scheme_name] = {
                        'type': 'oauth2',
                        'flows': scheme.get('flows', {})
                    }
            
            return auth_config
            
        except Exception as e:
            logger.warning(f"Failed to get auth config: {e}")
            return {}
    
    def execute_operation(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an API operation.
        
        Args:
            operation_id: ID of the operation to execute
            **kwargs: Parameters for the operation
            
        Returns:
            API response and metadata
        """
        try:
            # Find the operation in the specification
            operation_info = self._find_operation(operation_id)
            if not operation_info:
                raise ValueError(f"Operation {operation_id} not found")
            
            method = operation_info['method']
            path = operation_info['path']
            operation = operation_info['operation']
            
            # Prepare request
            url = urljoin(self.base_url, path)
            headers = self._prepare_headers(operation)
            params = self._prepare_params(operation, kwargs)
            data = self._prepare_data(operation, kwargs)
            
            # Execute request
            start_time = datetime.now()
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                timeout=30
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process response
            result = self._process_response(response, operation)
            
            # Add metadata
            result['metadata'] = {
                'operation_id': operation_id,
                'method': method,
                'path': path,
                'url': url,
                'execution_time': execution_time,
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to session
            self._save_api_response(result, operation_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute operation {operation_id}: {e}")
            raise
    
    def _find_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Find operation by ID in specification."""
        try:
            for path, path_item in self.spec.get('paths', {}).items():
                for method, operation in path_item.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        if operation.get('operationId') == operation_id:
                            return {
                                'method': method.upper(),
                                'path': path,
                                'operation': operation
                            }
            return None
            
        except Exception as e:
            logger.error(f"Failed to find operation {operation_id}: {e}")
            return None
    
    def _prepare_headers(self, operation: Dict[str, Any]) -> Dict[str, str]:
        """Prepare request headers."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Add authentication headers
            for auth_name, auth_config in self.auth_config.items():
                if auth_config['type'] == 'apiKey':
                    if auth_config['in'] == 'header':
                        # Get credential from vault
                        credential = self.credential_vault.get_credential(auth_name)
                        if credential:
                            headers[auth_config['name']] = credential.get('value', '')
                elif auth_config['type'] == 'http':
                    if auth_config['scheme'] == 'bearer':
                        # Get bearer token from vault
                        credential = self.credential_vault.get_credential(auth_name)
                        if credential:
                            headers['Authorization'] = f"Bearer {credential.get('token', '')}"
            
            return headers
            
        except Exception as e:
            logger.warning(f"Failed to prepare headers: {e}")
            return {'Content-Type': 'application/json'}
    
    def _prepare_params(self, operation: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare query parameters."""
        try:
            params = {}
            
            if 'parameters' in operation:
                for param in operation['parameters']:
                    if param.get('in') == 'query':
                        param_name = param.get('name')
                        if param_name in kwargs:
                            params[param_name] = kwargs[param_name]
            
            return params
            
        except Exception as e:
            logger.warning(f"Failed to prepare params: {e}")
            return {}
    
    def _prepare_data(self, operation: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request body data."""
        try:
            data = {}
            
            if 'requestBody' in operation:
                request_body = operation['requestBody']
                if 'content' in request_body:
                    for content_type, content_item in request_body['content'].items():
                        if 'schema' in content_item:
                            schema = content_item['schema']
                            if 'properties' in schema:
                                for prop_name in schema['properties']:
                                    if prop_name in kwargs:
                                        data[prop_name] = kwargs[prop_name]
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to prepare data: {e}")
            return {}
    
    def _process_response(self, response: requests.Response, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process API response."""
        try:
            result = {
                'success': response.status_code < 400,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'raw_response': response.text
            }
            
            # Try to parse JSON response
            try:
                result['data'] = response.json()
            except json.JSONDecodeError:
                result['data'] = response.text
            
            # Add response validation
            if 'responses' in operation:
                expected_responses = operation['responses']
                if str(response.status_code) in expected_responses:
                    result['expected_response'] = expected_responses[str(response.status_code)]
                else:
                    result['unexpected_status'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_code': response.status_code,
                'raw_response': response.text
            }
    
    def _save_api_response(self, result: Dict[str, Any], operation_id: str):
        """Save API response to session."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save response data
            filename = f"api_response_{operation_id}_{timestamp}"
            self.session_manager.save_text_output(
                json.dumps(result, indent=2, default=str),
                filename,
                f"API response for operation {operation_id}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to save API response: {e}")

# MCP Tools for OpenAPI Consumer
class OpenAPIConsumerMCPTools:
    """MCP-compatible tools for OpenAPI consumer."""
    
    def __init__(self, openapi_consumer: OpenAPIConsumer):
        self.consumer = openapi_consumer
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for OpenAPI consumer."""
        return {
            "load_openapi_spec": {
                "name": "load_openapi_spec",
                "description": "Load OpenAPI specification from various sources",
                "parameters": {
                    "spec_source": {"type": "string", "description": "Source of the OpenAPI spec (URL, file path, or raw content)"},
                    "spec_type": {"type": "string", "description": "Type of specification (auto, url, file, content)"}
                },
                "returns": {"type": "object", "description": "Parsed OpenAPI specification"}
            },
            "execute_api_operation": {
                "name": "execute_api_operation",
                "description": "Execute an API operation from loaded specification",
                "parameters": {
                    "spec_id": {"type": "string", "description": "ID of the OpenAPI specification"},
                    "operation_id": {"type": "string", "description": "ID of the operation to execute"},
                    "parameters": {"type": "object", "description": "Parameters for the operation"}
                },
                "returns": {"type": "object", "description": "API response and metadata"}
            },
            "get_available_apis": {
                "name": "get_available_apis",
                "description": "Get list of available APIs",
                "parameters": {},
                "returns": {"type": "object", "description": "List of available APIs with metadata"}
            },
            "get_api_operations": {
                "name": "get_api_operations",
                "description": "Get list of operations for a specific API",
                "parameters": {
                    "spec_id": {"type": "string", "description": "ID of the OpenAPI specification"}
                },
                "returns": {"type": "object", "description": "List of available operations"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute OpenAPI consumer MCP tool."""
        if tool_name == "load_openapi_spec":
            return self.consumer.load_openapi_spec(**kwargs)
        elif tool_name == "execute_api_operation":
            # Extract parameters from kwargs
            spec_id = kwargs.pop("spec_id")
            operation_id = kwargs.pop("operation_id")
            return self.consumer.execute_api_operation(spec_id, operation_id, **kwargs)
        elif tool_name == "get_available_apis":
            return self.consumer.get_available_apis()
        elif tool_name == "get_api_operations":
            return self.consumer.get_api_operations(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

