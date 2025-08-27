#!/usr/bin/env python3
"""
Security MCP Integration Layer - Host Scanning and Hashing Tools
Provides MCP-compatible integration for host scanning and hashing tools.

Features:
- Unified MCP interface for security tools
- Dynamic tool discovery and execution
- Context-aware tool selection
- Performance monitoring and analytics
- Integration with agentic workflow system
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time

# Import our security tools
from host_scanning_tools import HostScanningManager, ScanType, ScanIntensity
from hashing_tools import HashingManager, HashAlgorithm, HashType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityToolCategory(Enum):
    """Categories of security tools."""
    HOST_SCANNING = "host_scanning"
    HASHING = "hashing"
    NETWORK_ANALYSIS = "network_analysis"
    FORENSICS = "forensics"
    SECURITY_ASSESSMENT = "security_assessment"

class SecurityToolCapability(Enum):
    """Capabilities of security tools."""
    PORT_SCANNING = "port_scanning"
    OS_DETECTION = "os_detection"
    SERVICE_DETECTION = "service_detection"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    NETWORK_MAPPING = "network_mapping"
    FILE_HASHING = "file_hashing"
    STRING_HASHING = "string_hashing"
    HASH_VERIFICATION = "hash_verification"
    HMAC_GENERATION = "hmac_generation"
    BATCH_PROCESSING = "batch_processing"

@dataclass
class SecurityToolMetadata:
    """Metadata for security tools."""
    tool_id: str
    name: str
    description: str
    category: SecurityToolCategory
    capabilities: List[SecurityToolCapability]
    input_types: List[str]
    output_types: List[str]
    performance_metrics: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_used: Optional[float] = None

class SecurityMCPIntegrationLayer:
    """MCP integration layer for security tools."""
    
    def __init__(self):
        # Initialize tool managers
        self.host_scanning_manager = HostScanningManager()
        self.hashing_manager = HashingManager()
        
        # Create MCP tool registry
        self.mcp_tools = self._create_mcp_tools()
        self.tool_registry = self._create_tool_registry()
        
        # Performance tracking
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'tool_usage_counts': {}
        }
        
        logger.info("🚀 Security MCP Integration Layer initialized")
    
    def _create_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Create MCP-compatible tool definitions."""
        return {
            # Host Scanning Tools
            "quick_host_scan": {
                "name": "Quick Host Scan",
                "description": "Perform a quick port scan on target hosts",
                "category": SecurityToolCategory.HOST_SCANNING,
                "capabilities": [SecurityToolCapability.PORT_SCANNING],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of target IP addresses or hostnames"
                        },
                        "custom_options": {
                            "type": "object",
                            "description": "Additional scan options"
                        }
                    },
                    "required": ["targets"]
                },
                "handler": self._execute_quick_host_scan
            },
            
            "security_assessment_scan": {
                "name": "Security Assessment Scan",
                "description": "Comprehensive security assessment with vulnerability detection",
                "category": SecurityToolCategory.SECURITY_ASSESSMENT,
                "capabilities": [
                    SecurityToolCapability.PORT_SCANNING,
                    SecurityToolCapability.VULNERABILITY_SCANNING,
                    SecurityToolCapability.OS_DETECTION
                ],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of target IP addresses or hostnames"
                        },
                        "intensity": {
                            "type": "string",
                            "enum": ["polite", "normal", "aggressive"],
                            "description": "Scan intensity level"
                        }
                    },
                    "required": ["targets"]
                },
                "handler": self._execute_security_assessment_scan
            },
            
            "network_discovery_scan": {
                "name": "Network Discovery Scan",
                "description": "Network topology discovery and mapping",
                "category": SecurityToolCategory.NETWORK_ANALYSIS,
                "capabilities": [SecurityToolCapability.NETWORK_MAPPING],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of target IP addresses or hostnames"
                        },
                        "include_ports": {
                            "type": "boolean",
                            "description": "Include port scanning in discovery"
                        }
                    },
                    "required": ["targets"]
                },
                "handler": self._execute_network_discovery_scan
            },
            
            # Hashing Tools
            "hash_string": {
                "name": "Hash String",
                "description": "Hash a string using specified algorithm",
                "category": SecurityToolCategory.HASHING,
                "capabilities": [SecurityToolCapability.STRING_HASHING],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to hash"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"],
                            "description": "Hash algorithm to use"
                        }
                    },
                    "required": ["text"]
                },
                "handler": self._execute_hash_string
            },
            
            "hash_file": {
                "name": "Hash File",
                "description": "Hash a file using specified algorithm",
                "category": SecurityToolCategory.FORENSICS,
                "capabilities": [SecurityToolCapability.FILE_HASHING],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file to hash"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"],
                            "description": "Hash algorithm to use"
                        }
                    },
                    "required": ["file_path"]
                },
                "handler": self._execute_hash_file
            },
            
            "verify_hash": {
                "name": "Verify Hash",
                "description": "Verify hash integrity",
                "category": SecurityToolCategory.HASHING,
                "capabilities": [SecurityToolCapability.HASH_VERIFICATION],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "original_hash": {
                            "type": "string",
                            "description": "Original hash value"
                        },
                        "computed_hash": {
                            "type": "string",
                            "description": "Computed hash value to verify"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"],
                            "description": "Hash algorithm used"
                        }
                    },
                    "required": ["original_hash", "computed_hash"]
                },
                "handler": self._execute_verify_hash
            },
            
            "create_hmac": {
                "name": "Create HMAC",
                "description": "Create HMAC for data authentication",
                "category": SecurityToolCategory.HASHING,
                "capabilities": [SecurityToolCapability.HMAC_GENERATION],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Data to authenticate"
                        },
                        "key": {
                            "type": "string",
                            "description": "Secret key for HMAC"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["sha256", "sha512", "blake2b"],
                            "description": "Hash algorithm to use"
                        }
                    },
                    "required": ["data", "key"]
                },
                "handler": self._execute_create_hmac
            },
            
            "batch_hash_files": {
                "name": "Batch Hash Files",
                "description": "Hash multiple files efficiently",
                "category": SecurityToolCategory.FORENSICS,
                "capabilities": [SecurityToolCapability.FILE_HASHING, SecurityToolCapability.BATCH_PROCESSING],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to hash"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512"],
                            "description": "Hash algorithm to use"
                        }
                    },
                    "required": ["file_paths"]
                },
                "handler": self._execute_batch_hash_files
            }
        }
    
    def _create_tool_registry(self) -> Dict[str, Any]:
        """Create comprehensive tool registry for MCP discovery."""
        registry = {
            "tools": {},
            "categories": {},
            "capabilities": {},
            "search_index": {},
            "performance_metrics": {}
        }
        
        # Populate tools
        for tool_id, tool_info in self.mcp_tools.items():
            registry["tools"][tool_id] = {
                "id": tool_id,
                "name": tool_info["name"],
                "description": tool_info["description"],
                "category": tool_info["category"].value,
                "capabilities": [cap.value for cap in tool_info["capabilities"]],
                "input_schema": tool_info["input_schema"],
                "usage_count": 0,
                "success_rate": 1.0,
                "last_used": None
            }
            
            # Update categories
            category = tool_info["category"].value
            if category not in registry["categories"]:
                registry["categories"][category] = []
            registry["categories"][category].append(tool_id)
            
            # Update capabilities
            for capability in tool_info["capabilities"]:
                cap_value = capability.value
                if cap_value not in registry["capabilities"]:
                    registry["capabilities"][cap_value] = []
                registry["capabilities"][cap_value].append(tool_id)
            
            # Update search index
            search_terms = [
                tool_info["name"].lower(),
                tool_info["description"].lower(),
                tool_info["category"].value.lower()
            ]
            for term in search_terms:
                if term not in registry["search_index"]:
                    registry["search_index"][term] = []
                registry["search_index"][term].append(tool_id)
        
        return registry
    
    # Tool execution handlers
    async def _execute_quick_host_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quick host scan."""
        start_time = time.time()
        
        try:
            targets = params.get("targets", [])
            custom_options = params.get("custom_options", {})
            
            scan_result = await self.host_scanning_manager.execute_scan_template(
                "quick_audit", targets, custom_options
            )
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats("quick_host_scan", True, execution_time)
            
            return {
                "success": True,
                "scan_id": scan_result.scan_id,
                "hosts_found": len(scan_result.hosts),
                "summary": scan_result.summary,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("quick_host_scan", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_security_assessment_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security assessment scan."""
        start_time = time.time()
        
        try:
            targets = params.get("targets", [])
            intensity = params.get("intensity", "normal")
            
            # Map intensity to ScanIntensity
            intensity_map = {
                "polite": ScanIntensity.POLITE,
                "normal": ScanIntensity.NORMAL,
                "aggressive": ScanIntensity.AGGRESSIVE
            }
            
            scan_intensity = intensity_map.get(intensity, ScanIntensity.NORMAL)
            
            scan_result = await self.host_scanning_manager.custom_scan(
                targets, ScanType.VULNERABILITY_SCAN, scan_intensity
            )
            
            # Analyze results
            analysis = self.host_scanning_manager.analyze_scan_results(scan_result)
            
            execution_time = time.time() - start_time
            self._update_performance_stats("security_assessment_scan", True, execution_time)
            
            return {
                "success": True,
                "scan_id": scan_result.scan_id,
                "hosts_found": len(scan_result.hosts),
                "security_risks": analysis.get("security_risks", []),
                "recommendations": analysis.get("recommendations", []),
                "summary": scan_result.summary,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("security_assessment_scan", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_network_discovery_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute network discovery scan."""
        start_time = time.time()
        
        try:
            targets = params.get("targets", [])
            include_ports = params.get("include_ports", False)
            
            scan_type = ScanType.TOPOLOGY_SCAN
            if include_ports:
                scan_type = ScanType.COMPREHENSIVE_SCAN
            
            scan_result = await self.host_scanning_manager.custom_scan(
                targets, scan_type, ScanIntensity.POLITE
            )
            
            execution_time = time.time() - start_time
            self._update_performance_stats("network_discovery_scan", True, execution_time)
            
            return {
                "success": True,
                "scan_id": scan_result.scan_id,
                "hosts_found": len(scan_result.hosts),
                "network_insights": scan_result.summary,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("network_discovery_scan", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_hash_string(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute string hashing."""
        start_time = time.time()
        
        try:
            text = params.get("text", "")
            algorithm_name = params.get("algorithm", "sha256")
            
            # Map algorithm name to HashAlgorithm
            algorithm_map = {
                "md5": HashAlgorithm.MD5,
                "sha1": HashAlgorithm.SHA1,
                "sha256": HashAlgorithm.SHA256,
                "sha512": HashAlgorithm.SHA512,
                "blake2b": HashAlgorithm.BLAKE2B
            }
            
            algorithm = algorithm_map.get(algorithm_name, HashAlgorithm.SHA256)
            
            result = self.hashing_manager.calculator.hash_string(text, algorithm)
            
            execution_time = time.time() - start_time
            self._update_performance_stats("hash_string", True, execution_time)
            
            return {
                "success": True,
                "hash_value": result.hash_value,
                "algorithm": algorithm.value,
                "input_length": len(text),
                "processing_time": result.processing_time,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("hash_string", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_hash_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file hashing."""
        start_time = time.time()
        
        try:
            file_path = params.get("file_path", "")
            algorithm_name = params.get("algorithm", "sha256")
            
            # Map algorithm name to HashAlgorithm
            algorithm_map = {
                "md5": HashAlgorithm.MD5,
                "sha1": HashAlgorithm.SHA1,
                "sha256": HashAlgorithm.SHA256,
                "sha512": HashAlgorithm.SHA512,
                "blake2b": HashAlgorithm.BLAKE2B
            }
            
            algorithm = algorithm_map.get(algorithm_name, HashAlgorithm.SHA256)
            
            result = self.hashing_manager.calculator.hash_file(file_path, algorithm)
            
            execution_time = time.time() - start_time
            self._update_performance_stats("hash_file", True, execution_time)
            
            return {
                "success": True,
                "hash_value": result.hash_value,
                "algorithm": algorithm.value,
                "file_path": file_path,
                "file_size": result.file_size,
                "processing_time": result.processing_time,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("hash_file", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_verify_hash(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hash verification."""
        start_time = time.time()
        
        try:
            original_hash = params.get("original_hash", "")
            computed_hash = params.get("computed_hash", "")
            algorithm_name = params.get("algorithm", "sha256")
            
            # Map algorithm name to HashAlgorithm
            algorithm_map = {
                "md5": HashAlgorithm.MD5,
                "sha1": HashAlgorithm.SHA1,
                "sha256": HashAlgorithm.SHA256,
                "sha512": HashAlgorithm.SHA512,
                "blake2b": HashAlgorithm.BLAKE2B
            }
            
            algorithm = algorithm_map.get(algorithm_name, HashAlgorithm.SHA256)
            
            result = self.hashing_manager.calculator.verify_hash(
                original_hash, computed_hash, algorithm
            )
            
            execution_time = time.time() - start_time
            self._update_performance_stats("verify_hash", True, execution_time)
            
            return {
                "success": True,
                "is_valid": result.is_valid,
                "original_hash": original_hash,
                "computed_hash": computed_hash,
                "verification_time": result.verification_time,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("verify_hash", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_create_hmac(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HMAC creation."""
        start_time = time.time()
        
        try:
            data = params.get("data", "")
            key = params.get("key", "")
            algorithm_name = params.get("algorithm", "sha256")
            
            # Map algorithm name to HashAlgorithm
            algorithm_map = {
                "sha256": HashAlgorithm.SHA256,
                "sha512": HashAlgorithm.SHA512,
                "blake2b": HashAlgorithm.BLAKE2B
            }
            
            algorithm = algorithm_map.get(algorithm_name, HashAlgorithm.SHA256)
            
            result = self.hashing_manager.calculator.create_hmac(data, key, algorithm)
            
            execution_time = time.time() - start_time
            self._update_performance_stats("create_hmac", True, execution_time)
            
            return {
                "success": True,
                "hmac_value": result.hash_value,
                "algorithm": algorithm.value,
                "data_length": len(data),
                "key_length": len(key),
                "processing_time": result.processing_time,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("create_hmac", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_batch_hash_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch file hashing."""
        start_time = time.time()
        
        try:
            file_paths = params.get("file_paths", [])
            algorithm_name = params.get("algorithm", "sha256")
            
            # Map algorithm name to HashAlgorithm
            algorithm_map = {
                "md5": HashAlgorithm.MD5,
                "sha1": HashAlgorithm.SHA1,
                "sha256": HashAlgorithm.SHA256,
                "sha512": HashAlgorithm.SHA512
            }
            
            algorithm = algorithm_map.get(algorithm_name, HashAlgorithm.SHA256)
            
            results = self.hashing_manager.batch_hash_files(file_paths, algorithm)
            
            # Prepare summary
            successful_hashes = [r for r in results if not r.errors]
            failed_hashes = [r for r in results if r.errors]
            
            execution_time = time.time() - start_time
            self._update_performance_stats("batch_hash_files", True, execution_time)
            
            return {
                "success": True,
                "total_files": len(file_paths),
                "successful_hashes": len(successful_hashes),
                "failed_hashes": len(failed_hashes),
                "results": [
                    {
                        "file_path": r.input_source,
                        "hash_value": r.hash_value,
                        "algorithm": r.algorithm.value,
                        "success": not bool(r.errors)
                    }
                    for r in results
                ],
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_stats("batch_hash_files", False, execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _update_performance_stats(self, tool_id: str, success: bool, execution_time: float):
        """Update performance statistics."""
        self.performance_stats['total_executions'] += 1
        
        if success:
            self.performance_stats['successful_executions'] += 1
        else:
            self.performance_stats['failed_executions'] += 1
        
        # Update average execution time
        current_avg = self.performance_stats['average_execution_time']
        total_execs = self.performance_stats['total_executions']
        self.performance_stats['average_execution_time'] = (
            (current_avg * (total_execs - 1) + execution_time) / total_execs
        )
        
        # Update tool usage counts
        if tool_id not in self.performance_stats['tool_usage_counts']:
            self.performance_stats['tool_usage_counts'][tool_id] = 0
        self.performance_stats['tool_usage_counts'][tool_id] += 1
    
    # MCP Interface Methods
    def discover_tools(self, category: Optional[str] = None, 
                      capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover available tools based on criteria."""
        tools = []
        
        for tool_id, tool_info in self.mcp_tools.items():
            # Filter by category if specified
            if category and tool_info["category"].value != category:
                continue
            
            # Filter by capability if specified
            if capability:
                capabilities = [cap.value for cap in tool_info["capabilities"]]
                if capability not in capabilities:
                    continue
            
            tools.append({
                "id": tool_id,
                "name": tool_info["name"],
                "description": tool_info["description"],
                "category": tool_info["category"].value,
                "capabilities": [cap.value for cap in tool_info["capabilities"]],
                "input_schema": tool_info["input_schema"]
            })
        
        return tools
    
    def get_tool_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get input schema for a specific tool."""
        if tool_id in self.mcp_tools:
            tool_info = self.mcp_tools[tool_id]
            return {
                "id": tool_id,
                "name": tool_info["name"],
                "description": tool_info["description"],
                "input_schema": tool_info["input_schema"],
                "category": tool_info["category"].value,
                "capabilities": [cap.value for cap in tool_info["capabilities"]]
            }
        return None
    
    async def execute_tool(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters."""
        if tool_id not in self.mcp_tools:
            return {
                "success": False,
                "error": f"Tool not found: {tool_id}"
            }
        
        tool_info = self.mcp_tools[tool_id]
        handler = tool_info["handler"]
        
        try:
            result = await handler(params)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def get_tool_categories(self) -> List[str]:
        """Get list of available tool categories."""
        return list(set(tool["category"].value for tool in self.mcp_tools.values()))
    
    def get_tool_capabilities(self) -> List[str]:
        """Get list of available tool capabilities."""
        capabilities = set()
        for tool in self.mcp_tools.values():
            capabilities.update(cap.value for cap in tool["capabilities"])
        return list(capabilities)
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics."""
        return {
            "total_tools": len(self.mcp_tools),
            "categories": len(self.get_tool_categories()),
            "capabilities": len(self.get_tool_capabilities()),
            "performance": self.performance_stats.copy(),
            "tool_registry": self.tool_registry
        }
    
    def suggest_tools_for_problem(self, problem_description: str) -> List[str]:
        """Suggest tools based on problem description."""
        problem_lower = problem_description.lower()
        suggestions = []
        
        # Keyword-based tool suggestion
        keyword_mapping = {
            "scan": ["quick_host_scan", "security_assessment_scan", "network_discovery_scan"],
            "port": ["quick_host_scan", "security_assessment_scan"],
            "vulnerability": ["security_assessment_scan"],
            "network": ["network_discovery_scan", "quick_host_scan"],
            "hash": ["hash_string", "hash_file", "verify_hash"],
            "file": ["hash_file", "batch_hash_files"],
            "string": ["hash_string"],
            "verify": ["verify_hash"],
            "hmac": ["create_hmac"],
            "batch": ["batch_hash_files"]
        }
        
        for keyword, tool_ids in keyword_mapping.items():
            if keyword in problem_lower:
                suggestions.extend(tool_ids)
        
        # Remove duplicates and return
        return list(set(suggestions))

class SecurityToolsQueryPathIntegration:
    """Integration layer for Query Path to discover and select security tools."""
    
    def __init__(self, mcp_integration: SecurityMCPIntegrationLayer):
        self.mcp_integration = mcp_integration
        self.tool_selection_history = []
        self.context_analysis_cache = {}
        
        logger.info("🚀 Security Tools Query Path Integration initialized")
    
    def discover_relevant_tools(self, problem_description: str, context: Dict[str, Any]) -> List[str]:
        """Discover tools relevant to the given problem and context."""
        # Get tool suggestions based on problem description
        suggested_tools = self.mcp_integration.suggest_tools_for_problem(problem_description)
        
        # Filter by context if available
        if context:
            context_tools = self._filter_tools_by_context(suggested_tools, context)
            if context_tools:
                suggested_tools = context_tools
        
        # Log tool discovery
        self.tool_selection_history.append({
            "timestamp": time.time(),
            "problem": problem_description,
            "suggested_tools": suggested_tools,
            "context": context
        })
        
        return suggested_tools
    
    def _filter_tools_by_context(self, tools: List[str], context: Dict[str, Any]) -> List[str]:
        """Filter tools based on context information."""
        filtered_tools = []
        
        for tool_id in tools:
            tool_schema = self.mcp_integration.get_tool_schema(tool_id)
            if not tool_schema:
                continue
            
            # Check if tool requirements match context
            if self._tool_matches_context(tool_schema, context):
                filtered_tools.append(tool_id)
        
        return filtered_tools
    
    def _tool_matches_context(self, tool_schema: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a tool matches the given context."""
        # Simple context matching - can be enhanced
        context_keys = set(context.keys())
        
        # Check if tool requires any context keys
        input_schema = tool_schema.get("input_schema", {})
        required_props = input_schema.get("required", [])
        
        # If tool has no required properties, it matches any context
        if not required_props:
            return True
        
        # Check if context provides required properties
        for prop in required_props:
            if prop not in context_keys:
                return False
        
        return True
    
    def get_tool_relevance_score(self, tool_id: str, problem_description: str, 
                                context: Dict[str, Any]) -> float:
        """Calculate relevance score for a tool based on problem and context."""
        score = 0.0
        
        # Base score from problem description matching
        problem_lower = problem_description.lower()
        tool_schema = self.mcp_integration.get_tool_schema(tool_id)
        
        if tool_schema:
            tool_name = tool_schema["name"].lower()
            tool_desc = tool_schema["description"].lower()
            
            # Check for keyword matches
            keywords = problem_lower.split()
            for keyword in keywords:
                if keyword in tool_name:
                    score += 0.3
                if keyword in tool_desc:
                    score += 0.2
            
            # Context relevance
            if context:
                context_score = self._calculate_context_relevance(tool_schema, context)
                score += context_score * 0.3
            
            # Capability relevance
            capabilities = tool_schema.get("capabilities", [])
            capability_score = self._calculate_capability_relevance(capabilities, problem_description)
            score += capability_score * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_context_relevance(self, tool_schema: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how well a tool matches the given context."""
        input_schema = tool_schema.get("input_schema", {})
        required_props = input_schema.get("required", [])
        
        if not required_props:
            return 1.0
        
        context_keys = set(context.keys())
        matching_props = sum(1 for prop in required_props if prop in context_keys)
        
        return matching_props / len(required_props)
    
    def _calculate_capability_relevance(self, capabilities: List[str], problem_description: str) -> float:
        """Calculate how well tool capabilities match the problem."""
        problem_lower = problem_description.lower()
        
        capability_keywords = {
            "port_scanning": ["port", "scan", "network", "host"],
            "vulnerability_scanning": ["vulnerability", "security", "assessment", "risk"],
            "network_mapping": ["network", "topology", "discovery", "mapping"],
            "file_hashing": ["file", "hash", "integrity", "forensics"],
            "string_hashing": ["string", "text", "hash", "verify"],
            "hash_verification": ["verify", "check", "integrity", "hash"],
            "hmac_generation": ["hmac", "authentication", "message", "key"]
        }
        
        total_score = 0.0
        for capability in capabilities:
            if capability in capability_keywords:
                keywords = capability_keywords[capability]
                keyword_matches = sum(1 for keyword in keywords if keyword in problem_lower)
                total_score += keyword_matches / len(keywords)
        
        return min(total_score / len(capabilities), 1.0) if capabilities else 0.0

async def main():
    """Example usage and testing."""
    try:
        # Initialize integration layer
        mcp_integration = SecurityMCPIntegrationLayer()
        query_integration = SecurityToolsQueryPathIntegration(mcp_integration)
        
        print("🔍 Available security tools:")
        tools = mcp_integration.discover_tools()
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        print(f"\n📊 Tool categories:")
        categories = mcp_integration.get_tool_categories()
        for category in categories:
            print(f"  - {category}")
        
        print(f"\n🚀 Testing tool discovery for problem: 'Scan network for vulnerabilities'")
        problem = "Scan network for vulnerabilities"
        context = {"targets": ["192.168.1.1", "192.168.1.100"]}
        
        relevant_tools = query_integration.discover_relevant_tools(problem, context)
        print(f"  Relevant tools: {relevant_tools}")
        
        # Test tool execution
        if relevant_tools:
            print(f"\n🔧 Testing tool execution: {relevant_tools[0]}")
            test_params = {"targets": ["127.0.0.1"]}
            
            result = await mcp_integration.execute_tool(relevant_tools[0], test_params)
            print(f"  Result: {result}")
        
        # Get statistics
        stats = mcp_integration.get_tool_statistics()
        print(f"\n📈 Security tools statistics:")
        print(f"  - Total tools: {stats['total_tools']}")
        print(f"  - Categories: {stats['categories']}")
        print(f"  - Capabilities: {stats['capabilities']}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
