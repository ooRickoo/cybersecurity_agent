#!/usr/bin/env python3
"""
Network Tools MCP Integration Layer

Integrates network tools with the MCP (Multi-Agent Communication Protocol) system
for dynamic discovery and execution by the Query Path and Runner Agent.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from network_tools import (
    NetworkToolsManager, NetworkToolCategory, NetworkToolCapability,
    NetworkToolMetadata
)

logger = logging.getLogger(__name__)

@dataclass
class MCPNetworkTool:
    """MCP-compatible network tool wrapper."""
    tool_id: str
    name: str
    description: str
    category: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: NetworkToolMetadata

class NetworkMCPIntegrationLayer:
    """MCP integration layer for network tools."""
    
    def __init__(self):
        self.network_manager = NetworkToolsManager()
        self.mcp_tools = self._create_mcp_tools()
        self.tool_registry = self._create_tool_registry()
        
        logger.info("🚀 Network MCP Integration Layer initialized")
    
    def _create_mcp_tools(self) -> Dict[str, MCPNetworkTool]:
        """Create MCP-compatible tool wrappers."""
        mcp_tools = {}
        
        for tool_id, tool in self.network_manager.tools.items():
            metadata = tool.metadata
            
            # Create input schema based on tool capabilities
            input_schema = self._create_input_schema(tool_id, metadata)
            
            # Create output schema based on tool output types
            output_schema = self._create_output_schema(tool_id, metadata)
            
            mcp_tool = MCPNetworkTool(
                tool_id=tool_id,
                name=metadata.name,
                description=metadata.description,
                category=metadata.category.value,
                capabilities=[cap.value for cap in metadata.capabilities],
                input_schema=input_schema,
                output_schema=output_schema,
                metadata=metadata
            )
            
            mcp_tools[tool_id] = mcp_tool
        
        return mcp_tools
    
    def _create_input_schema(self, tool_id: str, metadata: NetworkToolMetadata) -> Dict[str, Any]:
        """Create input schema for MCP tool."""
        base_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Tool-specific input schemas
        if tool_id == "network_ping":
            base_schema["properties"] = {
                "host": {
                    "type": "string",
                    "description": "Target host (IP address or hostname)",
                    "examples": ["8.8.8.8", "google.com"]
                },
                "count": {
                    "type": "integer",
                    "description": "Number of ping packets to send",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 100
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 10.0
                },
                "size": {
                    "type": "integer",
                    "description": "Packet size in bytes",
                    "default": 56,
                    "minimum": 32,
                    "maximum": 65507
                }
            }
            base_schema["required"] = ["host"]
            
        elif tool_id == "network_dns_lookup":
            base_schema["properties"] = {
                "hostname": {
                    "type": "string",
                    "description": "Hostname to resolve",
                    "examples": ["google.com", "example.org"]
                },
                "record_type": {
                    "type": "string",
                    "description": "DNS record type",
                    "enum": ["A", "AAAA", "CNAME", "MX", "TXT", "NS"],
                    "default": "A"
                },
                "nameserver": {
                    "type": "string",
                    "description": "Custom nameserver (optional)",
                    "examples": ["8.8.8.8", "1.1.1.1"]
                }
            }
            base_schema["required"] = ["hostname"]
            
        elif tool_id == "network_netstat":
            base_schema["properties"] = {
                "protocol": {
                    "type": "string",
                    "description": "Protocol to filter",
                    "enum": ["all", "tcp", "udp"],
                    "default": "all"
                },
                "state": {
                    "type": "string",
                    "description": "Connection state to filter",
                    "enum": ["all", "listening", "established", "time_wait"],
                    "default": "all"
                },
                "interface": {
                    "type": "string",
                    "description": "Network interface to filter (optional)",
                    "examples": ["eth0", "Wi-Fi", "Ethernet"]
                }
            }
            
        elif tool_id == "network_arp":
            base_schema["properties"] = {
                "action": {
                    "type": "string",
                    "description": "ARP operation to perform",
                    "enum": ["show", "add", "delete"],
                    "default": "show"
                },
                "interface": {
                    "type": "string",
                    "description": "Network interface (optional)",
                    "examples": ["eth0", "Wi-Fi"]
                },
                "ip_address": {
                    "type": "string",
                    "description": "IP address for add/delete operations",
                    "examples": ["192.168.1.1"]
                },
                "mac_address": {
                    "type": "string",
                    "description": "MAC address for add operation",
                    "examples": ["00:11:22:33:44:55"]
                }
            }
            base_schema["required"] = ["action"]
            
        elif tool_id == "network_traceroute":
            base_schema["properties"] = {
                "host": {
                    "type": "string",
                    "description": "Target host (IP address or hostname)",
                    "examples": ["8.8.8.8", "google.com"]
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Maximum number of hops",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 255
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 10.0
                },
                "protocol": {
                    "type": "string",
                    "description": "Protocol to use",
                    "enum": ["icmp", "tcp", "udp"],
                    "default": "icmp"
                }
            }
            base_schema["required"] = ["host"]
            
        elif tool_id == "network_port_scanner":
            base_schema["properties"] = {
                "host": {
                    "type": "string",
                    "description": "Target host (IP address or hostname)",
                    "examples": ["192.168.1.1", "example.com"]
                },
                "port_range": {
                    "type": "string",
                    "description": "Port range to scan",
                    "examples": ["1-1024", "80", "443", "22-25"],
                    "default": "1-1024"
                },
                "scan_type": {
                    "type": "string",
                    "description": "Scan type",
                    "enum": ["tcp", "udp"],
                    "default": "tcp"
                },
                "timeout": {
                    "type": "number",
                    "description": "Connection timeout in seconds",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 10.0
                }
            }
            base_schema["required"] = ["host"]
        
        return base_schema
    
    def _create_output_schema(self, tool_id: str, metadata: NetworkToolMetadata) -> Dict[str, Any]:
        """Create output schema for MCP tool."""
        base_schema = {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "tool_id": {
                    "type": "string",
                    "description": "ID of the executed tool"
                },
                "execution_time": {
                    "type": "number",
                    "description": "Execution time in seconds"
                },
                "results": {
                    "type": "object",
                    "description": "Tool-specific results"
                },
                "analysis": {
                    "type": "object",
                    "description": "Analysis and insights from the tool"
                }
            },
            "required": ["success", "tool_id", "execution_time"]
        }
        
        # Add tool-specific result schemas
        if tool_id == "network_ping":
            base_schema["properties"]["results"]["properties"] = {
                "platform": {"type": "string"},
                "responses": {"type": "array"},
                "statistics": {
                    "type": "object",
                    "properties": {
                        "packets_sent": {"type": "integer"},
                        "packets_received": {"type": "integer"},
                        "packet_loss_percent": {"type": "number"},
                        "min_rtt": {"type": "number"},
                        "avg_rtt": {"type": "number"},
                        "max_rtt": {"type": "number"},
                        "mdev_rtt": {"type": "number"}
                    }
                }
            }
            
        elif tool_id == "network_dns_lookup":
            base_schema["properties"]["results"]["properties"] = {
                "hostname": {"type": "string"},
                "record_type": {"type": "string"},
                "resolver": {"type": "string"},
                "answers": {"type": "array"},
                "authority": {"type": "array"},
                "additional": {"type": "array"},
                "query_time": {"type": "integer"},
                "server": {"type": "string"}
            }
            
        elif tool_id == "network_netstat":
            base_schema["properties"]["results"]["properties"] = {
                "connections": {"type": "array"},
                "listening_ports": {"type": "array"},
                "interface_stats": {"type": "array"}
            }
            
        elif tool_id == "network_arp":
            base_schema["properties"]["results"]["properties"] = {
                "interface": {"type": "string"},
                "arp_entries": {"type": "array"},
                "total_entries": {"type": "integer"}
            }
            
        elif tool_id == "network_traceroute":
            base_schema["properties"]["results"]["properties"] = {
                "platform": {"type": "string"},
                "hops": {"type": "array"},
                "total_hops": {"type": "integer"},
                "destination_reached": {"type": "boolean"}
            }
            
        elif tool_id == "network_port_scanner":
            base_schema["properties"]["results"]["properties"] = {
                "host": {"type": "string"},
                "scan_type": {"type": "string"},
                "total_ports": {"type": "integer"},
                "open_ports": {"type": "array"},
                "closed_ports": {"type": "array"},
                "filtered_ports": {"type": "array"},
                "scan_summary": {"type": "object"}
            }
        
        return base_schema
    
    def _create_tool_registry(self) -> Dict[str, Any]:
        """Create tool registry for MCP discovery."""
        registry = {
            "tools": {},
            "categories": {},
            "capabilities": {},
            "search_index": {}
        }
        
        # Populate tools
        for tool_id, mcp_tool in self.mcp_tools.items():
            registry["tools"][tool_id] = {
                "id": mcp_tool.tool_id,
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "category": mcp_tool.category,
                "capabilities": mcp_tool.capabilities,
                "input_schema": mcp_tool.input_schema,
                "output_schema": mcp_tool.output_schema,
                "metadata": {
                    "usage_count": mcp_tool.metadata.usage_count,
                    "success_rate": mcp_tool.metadata.success_rate,
                    "last_used": mcp_tool.metadata.last_used,
                    "avg_execution_time": mcp_tool.metadata.performance_metrics.get("avg_execution_time", 0)
                }
            }
            
            # Populate categories
            if mcp_tool.category not in registry["categories"]:
                registry["categories"][mcp_tool.category] = []
            registry["categories"][mcp_tool.category].append(tool_id)
            
            # Populate capabilities
            for capability in mcp_tool.capabilities:
                if capability not in registry["capabilities"]:
                    registry["capabilities"][capability] = []
                registry["capabilities"][capability].append(tool_id)
            
            # Create search index
            search_terms = [
                tool_id.lower(),
                mcp_tool.name.lower(),
                mcp_tool.description.lower(),
                mcp_tool.category.lower()
            ] + [cap.lower() for cap in mcp_tool.capabilities]
            
            for term in search_terms:
                if term not in registry["search_index"]:
                    registry["search_index"][term] = []
                if tool_id not in registry["search_index"][term]:
                    registry["search_index"][term].append(tool_id)
        
        return registry
    
    def discover_tools(self, query: str = None, category: str = None, capability: str = None) -> List[Dict[str, Any]]:
        """Discover tools based on query, category, or capability."""
        discovered_tools = []
        
        if query:
            # Search by query
            query_lower = query.lower()
            matching_tools = set()
            
            for term, tool_ids in self.tool_registry["search_index"].items():
                if query_lower in term:
                    matching_tools.update(tool_ids)
            
            for tool_id in matching_tools:
                discovered_tools.append(self.tool_registry["tools"][tool_id])
                
        elif category:
            # Search by category
            if category in self.tool_registry["categories"]:
                for tool_id in self.tool_registry["categories"][category]:
                    discovered_tools.append(self.tool_registry["tools"][tool_id])
                    
        elif capability:
            # Search by capability
            if capability in self.tool_registry["capabilities"]:
                for tool_id in self.tool_registry["capabilities"][capability]:
                    discovered_tools.append(self.tool_registry["tools"][tool_id])
                    
        else:
            # Return all tools
            discovered_tools = list(self.tool_registry["tools"].values())
        
        return discovered_tools
    
    def get_tool_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get input and output schema for a specific tool."""
        if tool_id in self.tool_registry["tools"]:
            tool_info = self.tool_registry["tools"][tool_id]
            return {
                "input_schema": tool_info["input_schema"],
                "output_schema": tool_info["output_schema"]
            }
        return None
    
    async def execute_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute a network tool through MCP interface."""
        if tool_id not in self.mcp_tools:
            return {
                "success": False,
                "error": f"Tool not found: {tool_id}",
                "available_tools": list(self.mcp_tools.keys())
            }
        
        try:
            # Execute the tool
            result = await self.network_manager.execute_tool(tool_id, **kwargs)
            
            # Update tool registry with performance metrics
            if result["success"] and tool_id in self.tool_registry["tools"]:
                tool_info = self.tool_registry["tools"][tool_id]
                tool_info["metadata"]["usage_count"] += 1
                tool_info["metadata"]["last_used"] = result.get("execution_time", 0)
                tool_info["metadata"]["avg_execution_time"] = result.get("execution_time", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id
            }
    
    def get_tool_categories(self) -> List[str]:
        """Get available tool categories."""
        return list(self.tool_registry["categories"].keys())
    
    def get_tool_capabilities(self) -> List[str]:
        """Get available tool capabilities."""
        return list(self.tool_registry["capabilities"].keys())
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics."""
        return {
            "total_tools": len(self.tool_registry["tools"]),
            "categories": {
                category: len(tools) for category, tools in self.tool_registry["categories"].items()
            },
            "capabilities": {
                capability: len(tools) for capability, tools in self.tool_registry["capabilities"].items()
            },
            "performance_summary": self.network_manager.get_performance_stats()
        }
    
    def suggest_tools_for_problem(self, problem_description: str) -> List[Dict[str, Any]]:
        """Suggest tools based on problem description."""
        suggestions = []
        problem_lower = problem_description.lower()
        
        # Define problem-to-tool mappings
        problem_mappings = {
            "ping": ["network_ping"],
            "connectivity": ["network_ping", "network_traceroute"],
            "dns": ["network_dns_lookup"],
            "resolution": ["network_dns_lookup"],
            "netstat": ["network_netstat"],
            "connections": ["network_netstat"],
            "ports": ["network_netstat", "network_port_scanner"],
            "arp": ["network_arp"],
            "mac": ["network_arp"],
            "traceroute": ["network_traceroute"],
            "path": ["network_traceroute"],
            "route": ["network_traceroute"],
            "scan": ["network_port_scanner"],
            "vulnerability": ["network_port_scanner"],
            "security": ["network_port_scanner", "network_arp"],
            "monitor": ["network_ping", "network_netstat"],
            "diagnose": ["network_ping", "network_traceroute", "network_dns_lookup"]
        }
        
        # Find matching tools
        matched_tools = set()
        for keyword, tool_ids in problem_mappings.items():
            if keyword in problem_lower:
                matched_tools.update(tool_ids)
        
        # Create suggestions with relevance scores
        for tool_id in matched_tools:
            if tool_id in self.tool_registry["tools"]:
                tool_info = self.tool_registry["tools"][tool_id]
                
                # Calculate relevance score based on keyword matches
                relevance_score = 0
                for keyword, tool_ids in problem_mappings.items():
                    if keyword in problem_lower and tool_id in tool_ids:
                        relevance_score += 1
                
                suggestions.append({
                    "tool_id": tool_id,
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "category": tool_info["category"],
                    "capabilities": tool_info["capabilities"],
                    "relevance_score": relevance_score,
                    "metadata": tool_info["metadata"]
                })
        
        # Sort by relevance score (highest first)
        suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return suggestions

# Integration with Query Path and Runner Agent
class NetworkToolsQueryPathIntegration:
    """Integration layer for Query Path to discover and select network tools."""
    
    def __init__(self, mcp_integration: NetworkMCPIntegrationLayer):
        self.mcp_integration = mcp_integration
        self.tool_selection_history = []
        
    def discover_network_tools(self, problem_description: str) -> List[Dict[str, Any]]:
        """Discover network tools relevant to a problem."""
        return self.mcp_integration.suggest_tools_for_problem(problem_description)
    
    def get_tools_by_capability(self, capability: str) -> List[str]:
        """Get tools that provide a specific capability."""
        if capability in self.mcp_integration.tool_registry["capabilities"]:
            return self.mcp_integration.tool_registry["capabilities"][capability]
        return []
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category."""
        if category in self.mcp_integration.tool_registry["categories"]:
            return self.mcp_integration.tool_registry["categories"][category]
        return []
    
    def score_tool_relevance(self, tool_id: str, problem_description: str, context: Dict[str, Any]) -> float:
        """Score how relevant a tool is to a problem."""
        if tool_id not in self.mcp_integration.tool_registry["tools"]:
            return 0.0
        
        tool_info = self.mcp_integration.tool_registry["tools"][tool_id]
        score = 0.0
        
        # Base score from problem description matching
        problem_lower = problem_description.lower()
        for capability in tool_info["capabilities"]:
            if capability in problem_lower:
                score += 0.3
        
        # Category relevance
        if tool_info["category"] in problem_lower:
            score += 0.2
        
        # Performance-based scoring
        metadata = tool_info["metadata"]
        if metadata["usage_count"] > 0:
            # Success rate contribution
            score += metadata["success_rate"] * 0.2
            
            # Usage frequency contribution (normalized)
            usage_score = min(metadata["usage_count"] / 100.0, 1.0)
            score += usage_score * 0.1
        
        # Context relevance
        if "network" in context.get("domain", "").lower():
            score += 0.2
        
        return min(score, 1.0)

# Example usage and testing
async def main():
    """Example usage of the Network MCP Integration Layer."""
    mcp_integration = NetworkMCPIntegrationLayer()
    
    print("🚀 Network MCP Integration Layer Test")
    print("=" * 60)
    
    # Show available tools
    print("\n📋 Available Tools:")
    for tool_id, tool_info in mcp_integration.tool_registry["tools"].items():
        print(f"  {tool_id}: {tool_info['name']}")
        print(f"    Category: {tool_info['category']}")
        print(f"    Capabilities: {tool_info['capabilities']}")
    
    # Test tool discovery
    print("\n🔍 Tool Discovery Test:")
    print("  Searching for 'ping' tools:")
    ping_tools = mcp_integration.discover_tools(query="ping")
    for tool in ping_tools:
        print(f"    - {tool['name']}: {tool['description']}")
    
    print("  Searching for 'security' tools:")
    security_tools = mcp_integration.discover_tools(query="security")
    for tool in security_tools:
        print(f"    - {tool['name']}: {tool['description']}")
    
    # Test problem-based tool suggestions
    print("\n🎯 Problem-Based Tool Suggestions:")
    problems = [
        "I need to check network connectivity to a server",
        "I want to scan for open ports on a host",
        "I need to resolve DNS issues",
        "I want to analyze network path to a destination"
    ]
    
    for problem in problems:
        print(f"\n  Problem: {problem}")
        suggestions = mcp_integration.suggest_tools_for_problem(problem)
        for suggestion in suggestions[:3]:  # Top 3 suggestions
            print(f"    - {suggestion['name']} (relevance: {suggestion['relevance_score']})")
    
    # Test tool execution
    print("\n🔧 Tool Execution Test:")
    ping_result = await mcp_integration.execute_tool("network_ping", host="8.8.8.8", count=2)
    if ping_result["success"]:
        print(f"  ✅ Ping successful to 8.8.8.8")
        print(f"  📊 Execution time: {ping_result['execution_time']:.3f}s")
    else:
        print(f"  ❌ Ping failed: {ping_result['error']}")
    
    # Show statistics
    print("\n📊 Integration Layer Statistics:")
    stats = mcp_integration.get_tool_statistics()
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Categories: {list(stats['categories'].keys())}")
    print(f"  Capabilities: {list(stats['capabilities'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())
