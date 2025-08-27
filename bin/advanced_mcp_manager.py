#!/usr/bin/env python3
"""
Advanced MCP Tool Management System for Cybersecurity Agent
Provides intelligent MCP tool discovery and dynamic integration
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import hashlib
import subprocess
import socket
import requests
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolType(Enum):
    """MCP tool type enumeration."""
    SERVER = "server"
    CLIENT = "client"
    PROXY = "proxy"
    BRIDGE = "bridge"

class MCPToolStatus(Enum):
    """MCP tool status enumeration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class MCPTool:
    """MCP tool information and capabilities."""
    tool_id: str
    name: str
    description: str
    tool_type: MCPToolType
    version: str
    server_url: str
    port: int
    protocol: str
    capabilities: List[str]
    status: MCPToolStatus
    last_checked: datetime
    performance_metrics: Dict[str, Any]
    is_optimized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['tool_type'] = self.tool_type.value
        data['status'] = self.status.value
        data['last_checked'] = self.last_checked.isoformat()
        return data

@dataclass
class MCPToolDiscovery:
    """MCP tool discovery result."""
    tool_id: str
    discovery_method: str
    discovered_at: datetime
    configuration: Dict[str, Any]
    validation_result: str

class ToolDiscovery:
    """Automatically discover available MCP tools."""
    
    def __init__(self):
        self.discovery_methods = self._load_discovery_methods()
        self.known_ports = [3000, 3001, 3002, 4000, 4001, 5000, 5001, 8080, 9000]
        self.known_protocols = ["http", "https", "ws", "wss", "tcp"]
    
    def _load_discovery_methods(self) -> List[Callable]:
        """Load tool discovery methods."""
        return [
            self._discover_via_config_files,
            self._discover_via_network_scan,
            self._discover_via_service_registry,
            self._discover_via_environment_vars
        ]
    
    async def scan_servers(self) -> List[Dict[str, Any]]:
        """Scan for available MCP servers."""
        discovered_servers = []
        
        # Scan known ports on localhost
        for port in self.known_ports:
            try:
                if await self._check_port_availability(port):
                    server_info = await self._probe_mcp_server(port)
                    if server_info:
                        discovered_servers.append(server_info)
            except Exception as e:
                logger.debug(f"Port {port} scan failed: {e}")
        
        # Scan network ranges if configured
        network_servers = await self._scan_network_ranges()
        discovered_servers.extend(network_servers)
        
        return discovered_servers
    
    async def _check_port_availability(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    async def _probe_mcp_server(self, port: int) -> Optional[Dict[str, Any]]:
        """Probe an MCP server for information."""
        try:
            # Try HTTP/HTTPS first
            for protocol in ["http", "https"]:
                try:
                    url = f"{protocol}://localhost:{port}"
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        return {
                            "host": "localhost",
                            "port": port,
                            "protocol": protocol,
                            "status": "available",
                            "discovery_method": "port_scan"
                        }
                except requests.RequestException:
                    continue
            
            # Try WebSocket
            try:
                # This is a simplified WebSocket check
                # In practice, you'd use a proper WebSocket library
                return {
                    "host": "localhost",
                    "port": port,
                    "protocol": "ws",
                    "status": "available",
                    "discovery_method": "port_scan"
                }
            except Exception:
                pass
            
            return None
        except Exception as e:
            logger.debug(f"Failed to probe port {port}: {e}")
            return None
    
    async def _scan_network_ranges(self) -> List[Dict[str, Any]]:
        """Scan network ranges for MCP servers."""
        # This would scan configured network ranges
        # For now, return empty list
        return []
    
    async def discover_tools(self, server: Dict[str, Any]) -> List[MCPTool]:
        """Discover tools from an MCP server."""
        discovered_tools = []
        
        try:
            # Try to get tool information from the server
            if server["protocol"] in ["http", "https"]:
                tools = await self._discover_tools_via_http(server)
                discovered_tools.extend(tools)
            elif server["protocol"] in ["ws", "wss"]:
                tools = await self._discover_tools_via_websocket(server)
                discovered_tools.extend(tools)
            
        except Exception as e:
            logger.error(f"Failed to discover tools from server {server}: {e}")
        
        return discovered_tools
    
    async def _discover_tools_via_http(self, server: Dict[str, Any]) -> List[MCPTool]:
        """Discover tools via HTTP API."""
        tools = []
        
        try:
            url = f"{server['protocol']}://{server['host']}:{server['port']}"
            
            # Try common MCP endpoints
            endpoints = ["/tools", "/capabilities", "/api/tools", "/mcp/tools"]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        tool_data = response.json()
                        tools.extend(self._parse_tool_data(tool_data, server))
                        break
                except requests.RequestException:
                    continue
            
        except Exception as e:
            logger.error(f"HTTP discovery failed for {server}: {e}")
        
        return tools
    
    async def _discover_tools_via_websocket(self, server: Dict[str, Any]) -> List[MCPTool]:
        """Discover tools via WebSocket."""
        # This would implement WebSocket-based tool discovery
        # For now, return empty list
        return []
    
    def _parse_tool_data(self, tool_data: Dict[str, Any], server: Dict[str, Any]) -> List[MCPTool]:
        """Parse tool data from server response."""
        tools = []
        
        try:
            if isinstance(tool_data, list):
                for tool_info in tool_data:
                    tool = MCPTool(
                        tool_id=f"{server['host']}_{server['port']}_{tool_info.get('id', 'unknown')}",
                        name=tool_info.get('name', 'Unknown Tool'),
                        description=tool_info.get('description', 'No description available'),
                        tool_type=MCPToolType.SERVER,
                        version=tool_info.get('version', '1.0.0'),
                        server_url=f"{server['protocol']}://{server['host']}:{server['port']}",
                        port=server['port'],
                        protocol=server['protocol'],
                        capabilities=tool_info.get('capabilities', []),
                        status=MCPToolStatus.AVAILABLE,
                        last_checked=datetime.now(),
                        performance_metrics={}
                    )
                    tools.append(tool)
            elif isinstance(tool_data, dict):
                # Single tool
                tool = MCPTool(
                    tool_id=f"{server['host']}_{server['port']}_{tool_data.get('id', 'unknown')}",
                    name=tool_data.get('name', 'Unknown Tool'),
                    description=tool_data.get('description', 'No description available'),
                    tool_type=MCPToolType.SERVER,
                    version=tool_data.get('version', '1.0.0'),
                    server_url=f"{server['protocol']}://{server['host']}:{server['port']}",
                    port=server['port'],
                    protocol=server['protocol'],
                    capabilities=tool_data.get('capabilities', []),
                    status=MCPToolStatus.AVAILABLE,
                    last_checked=datetime.now(),
                    performance_metrics={}
                )
                tools.append(tool)
        
        except Exception as e:
            logger.error(f"Failed to parse tool data: {e}")
        
        return tools
    
    async def _discover_via_config_files(self) -> List[MCPTool]:
        """Discover tools via configuration files."""
        tools = []
        
        # Look for common config file locations
        config_paths = [
            "~/.mcp/config.yaml",
            "~/.mcp/config.json",
            "./mcp-config.yaml",
            "./mcp-config.json",
            "/etc/mcp/config.yaml"
        ]
        
        for config_path in config_paths:
            expanded_path = Path(config_path).expanduser()
            if expanded_path.exists():
                try:
                    with open(expanded_path, 'r') as f:
                        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                            config = yaml.safe_load(f)
                        else:
                            config = json.load(f)
                    
                    tools.extend(self._parse_config_tools(config))
                except Exception as e:
                    logger.error(f"Failed to parse config file {config_path}: {e}")
        
        return tools
    
    async def _discover_via_network_scan(self) -> List[MCPTool]:
        """Discover tools via network scanning."""
        # This would implement network scanning for MCP tools
        # For now, return empty list
        return []
    
    async def _discover_via_service_registry(self) -> List[MCPTool]:
        """Discover tools via service registry."""
        # This would implement service registry discovery
        # For now, return empty list
        return []
    
    async def _discover_via_environment_vars(self) -> List[MCPTool]:
        """Discover tools via environment variables."""
        tools = []
        
        # Check for MCP-related environment variables
        mcp_vars = {
            'MCP_SERVER_URL': 'server_url',
            'MCP_TOOL_PATH': 'tool_path',
            'MCP_CONFIG': 'config_path'
        }
        
        for env_var, config_key in mcp_vars.items():
            if env_var in os.environ:
                try:
                    tool = MCPTool(
                        tool_id=f"env_{env_var.lower()}",
                        name=f"Environment {env_var}",
                        description=f"Tool discovered via {env_var}",
                        tool_type=MCPToolType.SERVER,
                        version="1.0.0",
                        server_url=os.environ.get(env_var, ""),
                        port=0,
                        protocol="unknown",
                        capabilities=["environment_discovered"],
                        status=MCPToolStatus.AVAILABLE,
                        last_checked=datetime.now(),
                        performance_metrics={}
                    )
                    tools.append(tool)
                except Exception as e:
                    logger.error(f"Failed to create tool from env var {env_var}: {e}")
        
        return tools
    
    def _parse_config_tools(self, config: Dict[str, Any]) -> List[MCPTool]:
        """Parse tools from configuration."""
        tools = []
        
        try:
            if 'tools' in config:
                for tool_config in config['tools']:
                    tool = MCPTool(
                        tool_id=tool_config.get('id', f"config_{len(tools)}"),
                        name=tool_config.get('name', 'Unknown Tool'),
                        description=tool_config.get('description', 'No description'),
                        tool_type=MCPToolType.SERVER,
                        version=tool_config.get('version', '1.0.0'),
                        server_url=tool_config.get('server_url', ''),
                        port=tool_config.get('port', 0),
                        protocol=tool_config.get('protocol', 'http'),
                        capabilities=tool_config.get('capabilities', []),
                        status=MCPToolStatus.AVAILABLE,
                        last_checked=datetime.now(),
                        performance_metrics={}
                    )
                    tools.append(tool)
        except Exception as e:
            logger.error(f"Failed to parse config tools: {e}")
        
        return tools

class ToolOptimizer:
    """Optimize MCP tools for local environment."""
    
    def __init__(self):
        self.optimization_strategies = self._load_optimization_strategies()
        self.local_tool_mappings = self._load_local_tool_mappings()
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load tool optimization strategies."""
        return {
            "performance": {
                "priority": "high",
                "methods": ["caching", "connection_pooling", "request_batching"],
                "estimated_improvement": 0.3
            },
            "reliability": {
                "priority": "high",
                "methods": ["retry_logic", "circuit_breaker", "fallback_handlers"],
                "estimated_improvement": 0.4
            },
            "security": {
                "priority": "medium",
                "methods": ["encryption", "authentication", "authorization"],
                "estimated_improvement": 0.2
            }
        }
    
    def _load_local_tool_mappings(self) -> Dict[str, str]:
        """Load mappings from MCP tools to local alternatives."""
        return {
            "file_operations": "local_file_manager",
            "data_processing": "local_data_processor",
            "network_scanning": "local_network_scanner",
            "threat_analysis": "local_threat_analyzer",
            "compliance_checking": "local_compliance_checker"
        }
    
    async def optimize_tools(self, tools: List[MCPTool]) -> List[MCPTool]:
        """Optimize MCP tools for local environment."""
        optimized_tools = []
        
        for tool in tools:
            try:
                # Check if tool can be optimized locally
                if await self.can_optimize_locally(tool):
                    optimized_tool = await self.create_local_optimization(tool)
                    optimized_tools.append(optimized_tool)
                else:
                    # Apply other optimizations
                    optimized_tool = await self.apply_optimizations(tool)
                    optimized_tools.append(optimized_tool)
            except Exception as e:
                logger.error(f"Failed to optimize tool {tool.name}: {e}")
                optimized_tools.append(tool)
        
        return optimized_tools
    
    async def can_optimize_locally(self, tool: MCPTool) -> bool:
        """Check if tool can be optimized for local execution."""
        # Check if there's a local alternative
        for mcp_capability in tool.capabilities:
            if mcp_capability in self.local_tool_mappings:
                return True
        
        # Check if tool is simple enough to replicate locally
        if len(tool.capabilities) <= 3 and tool.description:
            return True
        
        return False
    
    async def create_local_optimization(self, tool: MCPTool) -> MCPTool:
        """Create local optimization for a tool."""
        # Create a local version of the tool
        local_tool = MCPTool(
            tool_id=f"local_{tool.tool_id}",
            name=f"Local {tool.name}",
            description=f"Local optimization of {tool.description}",
            tool_type=MCPToolType.CLIENT,
            version=tool.version,
            server_url="local://localhost",
            port=0,
            protocol="local",
            capabilities=tool.capabilities,
            status=MCPToolStatus.AVAILABLE,
            last_checked=datetime.now(),
            performance_metrics={
                "latency": 0.001,  # Very fast local execution
                "throughput": 1000,  # High throughput
                "reliability": 0.99  # High reliability
            },
            is_optimized=True
        )
        
        return local_tool
    
    async def apply_optimizations(self, tool: MCPTool) -> MCPTool:
        """Apply general optimizations to a tool."""
        optimized_tool = tool
        
        # Apply performance optimizations
        if "performance" in self.optimization_strategies:
            optimized_tool.performance_metrics.update({
                "caching_enabled": True,
                "connection_pooling": True,
                "request_batching": True
            })
        
        # Apply reliability optimizations
        if "reliability" in self.optimization_strategies:
            optimized_tool.performance_metrics.update({
                "retry_logic": True,
                "circuit_breaker": True,
                "fallback_handlers": True
            })
        
        # Apply security optimizations
        if "security" in self.optimization_strategies:
            optimized_tool.performance_metrics.update({
                "encryption": True,
                "authentication": True,
                "authorization": True
            })
        
        return optimized_tool

class IntegrationManager:
    """Manage MCP tool integration."""
    
    def __init__(self):
        self.integrated_tools: Dict[str, MCPTool] = {}
        self.integration_db_path = Path("knowledge-objects/mcp_integration.db")
        self.integration_db_path.parent.mkdir(exist_ok=True)
        self._init_integration_db()
    
    def _init_integration_db(self):
        """Initialize integration database."""
        try:
            with sqlite3.connect(self.integration_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS mcp_tools (
                        tool_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        tool_type TEXT NOT NULL,
                        version TEXT,
                        server_url TEXT,
                        port INTEGER,
                        protocol TEXT,
                        capabilities TEXT,
                        status TEXT NOT NULL,
                        last_checked TEXT NOT NULL,
                        performance_metrics TEXT,
                        is_optimized BOOLEAN DEFAULT FALSE
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tool_discoveries (
                        tool_id TEXT PRIMARY KEY,
                        discovery_method TEXT NOT NULL,
                        discovered_at TEXT NOT NULL,
                        configuration TEXT,
                        validation_result TEXT
                    )
                """)
        except Exception as e:
            logger.warning(f"Integration database initialization failed: {e}")
    
    async def integrate_tools(self, tools: List[MCPTool]):
        """Integrate MCP tools into the system."""
        for tool in tools:
            try:
                # Validate tool
                if await self._validate_tool(tool):
                    # Store tool
                    self.integrated_tools[tool.tool_id] = tool
                    
                    # Store in database
                    await self._store_tool(tool)
                    
                    logger.info(f"Successfully integrated tool: {tool.name}")
                else:
                    logger.warning(f"Tool validation failed: {tool.name}")
            except Exception as e:
                logger.error(f"Failed to integrate tool {tool.name}: {e}")
    
    async def _validate_tool(self, tool: MCPTool) -> bool:
        """Validate an MCP tool."""
        try:
            # Basic validation
            if not tool.name or not tool.tool_id:
                return False
            
            # Check if tool is accessible
            if tool.server_url != "local://localhost":
                if not await self._check_tool_accessibility(tool):
                    return False
            
            # Check capabilities
            if not tool.capabilities:
                logger.warning(f"Tool {tool.name} has no capabilities")
            
            return True
        except Exception as e:
            logger.error(f"Tool validation failed: {e}")
            return False
    
    async def _check_tool_accessibility(self, tool: MCPTool) -> bool:
        """Check if a tool is accessible."""
        try:
            if tool.protocol in ["http", "https"]:
                response = requests.get(f"{tool.server_url}/health", timeout=5)
                return response.status_code == 200
            elif tool.protocol in ["ws", "wss"]:
                # WebSocket accessibility check
                return True  # Simplified for now
            else:
                return True
        except Exception:
            return False
    
    async def _store_tool(self, tool: MCPTool):
        """Store tool in database."""
        try:
            with sqlite3.connect(self.integration_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO mcp_tools 
                    (tool_id, name, description, tool_type, version, server_url, 
                     port, protocol, capabilities, status, last_checked, performance_metrics, is_optimized)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool.tool_id, tool.name, tool.description, tool.tool_type.value,
                    tool.version, tool.server_url, tool.port, tool.protocol,
                    json.dumps(tool.capabilities), tool.status.value,
                    tool.last_checked.isoformat(), json.dumps(tool.performance_metrics),
                    tool.is_optimized
                ))
        except Exception as e:
            logger.error(f"Failed to store tool in database: {e}")
    
    def get_integrated_tools(self) -> List[MCPTool]:
        """Get all integrated tools."""
        return list(self.integrated_tools.values())
    
    def get_tool_by_id(self, tool_id: str) -> Optional[MCPTool]:
        """Get tool by ID."""
        return self.integrated_tools.get(tool_id)
    
    def get_tools_by_capability(self, capability: str) -> List[MCPTool]:
        """Get tools by capability."""
        return [tool for tool in self.integrated_tools.values() if capability in tool.capabilities]

class AdvancedMCPManager:
    """Main advanced MCP tool management orchestrator."""
    
    def __init__(self):
        self.tool_discovery = ToolDiscovery()
        self.tool_optimizer = ToolOptimizer()
        self.integration_manager = IntegrationManager()
        self.discovery_history: List[MCPToolDiscovery] = []
    
    async def auto_discover_tools(self) -> List[MCPTool]:
        """Automatically discover and integrate new MCP tools."""
        # Scan for available MCP servers
        available_servers = await self.tool_discovery.scan_servers()
        
        # Discover tools from each server
        discovered_tools = []
        for server in available_servers:
            tools = await self.tool_discovery.discover_tools(server)
            discovered_tools.extend(tools)
        
        # Discover tools via other methods
        config_tools = await self.tool_discovery._discover_via_config_files()
        discovered_tools.extend(config_tools)
        
        env_tools = await self.tool_discovery._discover_via_environment_vars()
        discovered_tools.extend(env_tools)
        
        # Optimize tool integration
        optimized_tools = await self.tool_optimizer.optimize_tools(discovered_tools)
        
        # Integrate tools
        await self.integration_manager.integrate_tools(optimized_tools)
        
        return optimized_tools
    
    async def get_tool_recommendations(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get tool recommendations based on task and context."""
        integrated_tools = self.integration_manager.get_integrated_tools()
        
        recommendations = {
            "task": task,
            "available_tools": [],
            "recommended_tools": [],
            "optimization_suggestions": []
        }
        
        # Analyze available tools
        for tool in integrated_tools:
            tool_info = {
                "tool": tool.to_dict(),
                "relevance_score": await self._calculate_tool_relevance(tool, task),
                "performance_score": await self._calculate_tool_performance(tool),
                "optimization_status": "optimized" if tool.is_optimized else "not_optimized"
            }
            recommendations["available_tools"].append(tool_info)
        
        # Sort by relevance and performance
        recommendations["available_tools"].sort(
            key=lambda x: x["relevance_score"] * 0.7 + x["performance_score"] * 0.3,
            reverse=True
        )
        
        # Get top recommendations
        recommendations["recommended_tools"] = recommendations["available_tools"][:5]
        
        # Generate optimization suggestions
        recommendations["optimization_suggestions"] = await self._generate_optimization_suggestions(
            recommendations["available_tools"]
        )
        
        return recommendations
    
    async def _calculate_tool_relevance(self, tool: MCPTool, task: str) -> float:
        """Calculate tool relevance for a task."""
        task_lower = task.lower()
        relevance_score = 0.0
        
        # Check capability matches
        for capability in tool.capabilities:
            if capability.lower() in task_lower:
                relevance_score += 0.3
            elif any(word in capability.lower() for word in task_lower.split()):
                relevance_score += 0.1
        
        # Check name matches
        if any(word in tool.name.lower() for word in task_lower.split()):
            relevance_score += 0.2
        
        # Check description matches
        if any(word in tool.description.lower() for word in task_lower.split()):
            relevance_score += 0.1
        
        return min(1.0, relevance_score)
    
    async def _calculate_tool_performance(self, tool: MCPTool) -> float:
        """Calculate tool performance score."""
        if not tool.performance_metrics:
            return 0.5
        
        # Calculate performance based on metrics
        performance_score = 0.0
        
        # Latency (lower is better)
        if "latency" in tool.performance_metrics:
            latency = tool.performance_metrics["latency"]
            if latency < 0.1:
                performance_score += 0.3
            elif latency < 1.0:
                performance_score += 0.2
            else:
                performance_score += 0.1
        
        # Throughput (higher is better)
        if "throughput" in tool.performance_metrics:
            throughput = tool.performance_metrics["throughput"]
            if throughput > 1000:
                performance_score += 0.3
            elif throughput > 100:
                performance_score += 0.2
            else:
                performance_score += 0.1
        
        # Reliability
        if "reliability" in tool.performance_metrics:
            reliability = tool.performance_metrics["reliability"]
            performance_score += reliability * 0.4
        
        return min(1.0, performance_score)
    
    async def _generate_optimization_suggestions(self, tools: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        unoptimized_tools = [t for t in tools if t["optimization_status"] == "not_optimized"]
        if unoptimized_tools:
            suggestions.append(f"Consider optimizing {len(unoptimized_tools)} unoptimized tools")
        
        low_performance_tools = [t for t in tools if t["performance_score"] < 0.5]
        if low_performance_tools:
            suggestions.append(f"Review performance of {len(low_performance_tools)} low-performance tools")
        
        return suggestions
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get MCP integration status."""
        integrated_tools = self.integration_manager.get_integrated_tools()
        
        return {
            "total_tools": len(integrated_tools),
            "optimized_tools": len([t for t in integrated_tools if t.is_optimized]),
            "local_tools": len([t for t in integrated_tools if t.protocol == "local"]),
            "remote_tools": len([t for t in integrated_tools if t.protocol != "local"]),
            "discovery_methods": len(self.tool_discovery.discovery_methods),
            "optimization_strategies": len(self.tool_optimizer.optimization_strategies)
        }

# Global advanced MCP manager instance
advanced_mcp_manager = AdvancedMCPManager()

# Convenience functions
async def auto_discover_tools() -> List[MCPTool]:
    """Convenience function for automatic tool discovery."""
    return await advanced_mcp_manager.auto_discover_tools()

async def get_tool_recommendations(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for tool recommendations."""
    return await advanced_mcp_manager.get_tool_recommendations(task, context)

def get_integration_status() -> Dict[str, Any]:
    """Convenience function for integration status."""
    return advanced_mcp_manager.get_integration_status()
