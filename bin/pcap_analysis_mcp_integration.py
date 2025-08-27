#!/usr/bin/env python3
"""
PCAP Analysis MCP Integration Layer

Provides MCP-compatible interface for PCAP analysis tools with:
- Dynamic tool discovery
- Standardized tool execution
- Query Path integration
- Performance monitoring
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from pcap_analysis_tools import get_pcap_analysis_manager, PCAPAnalysisManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCAPAnalysisToolCategory(Enum):
    """Categories of PCAP analysis tools."""
    TRAFFIC_ANALYSIS = "traffic_analysis"
    TECHNOLOGY_DETECTION = "technology_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    FILE_EXTRACTION = "file_extraction"
    PCAP_MANIPULATION = "pcap_manipulation"
    SCAN_CAPTURE = "scan_capture"

class PCAPAnalysisToolCapability(Enum):
    """Capabilities of PCAP analysis tools."""
    TRAFFIC_SUMMARIZATION = "traffic_summarization"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    FLOW_ANALYSIS = "flow_analysis"
    TECHNOLOGY_FINGERPRINTING = "technology_fingerprinting"
    ANOMALY_DETECTION = "anomaly_detection"
    FILE_EXTRACTION = "file_extraction"
    PCAP_FILTERING = "pcap_filtering"
    PCAP_MERGING = "pcap_merging"
    SCAN_CAPTURE = "scan_capture"
    STATISTICAL_ANALYSIS = "statistical_analysis"

@dataclass
class PCAPAnalysisToolMetadata:
    """Metadata for PCAP analysis tools."""
    tool_id: str
    name: str
    description: str
    category: PCAPAnalysisToolCategory
    capabilities: List[PCAPAnalysisToolCapability]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, str]
    security_considerations: List[str]
    tags: List[str]

class PCAPAnalysisMCPIntegrationLayer:
    """MCP integration layer for PCAP analysis tools."""
    
    def __init__(self):
        self.pcap_manager = get_pcap_analysis_manager()
        self.tool_registry = self._initialize_tool_registry()
        
        logger.info("🚀 PCAP Analysis MCP Integration Layer initialized")
    
    def _initialize_tool_registry(self) -> Dict[str, PCAPAnalysisToolMetadata]:
        """Initialize the tool registry with all available PCAP analysis tools."""
        registry = {
            "analyze_pcap_traffic": PCAPAnalysisToolMetadata(
                tool_id="analyze_pcap_traffic",
                name="PCAP Traffic Analyzer",
                description="Comprehensive analysis of PCAP file traffic including protocols, flows, and statistics",
                category=PCAPAnalysisToolCategory.TRAFFIC_ANALYSIS,
                capabilities=[
                    PCAPAnalysisToolCapability.TRAFFIC_SUMMARIZATION,
                    PCAPAnalysisToolCapability.PROTOCOL_ANALYSIS,
                    PCAPAnalysisToolCapability.FLOW_ANALYSIS,
                    PCAPAnalysisToolCapability.STATISTICAL_ANALYSIS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "Path to PCAP file"},
                        "analysis_type": {"type": "string", "description": "Type of analysis (basic, comprehensive)", "default": "comprehensive"}
                    },
                    "required": ["pcap_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "summary": {"type": "object", "description": "PCAP analysis summary"},
                        "error": {"type": "string", "description": "Error message if analysis failed"}
                    }
                },
                performance_metrics={"expected_response_time": "5-30s"},
                security_considerations=["Analyzes network traffic data, may contain sensitive information"],
                tags=["pcap", "traffic", "analysis", "protocols", "flows"]
            ),
            
            "detect_technology_stack": PCAPAnalysisToolMetadata(
                tool_id="detect_technology_stack",
                name="Technology Stack Detector",
                description="Detect technology stack components from network traffic patterns",
                category=PCAPAnalysisToolCategory.TECHNOLOGY_DETECTION,
                capabilities=[
                    PCAPAnalysisToolCapability.TECHNOLOGY_FINGERPRINTING,
                    PCAPAnalysisToolCapability.PROTOCOL_ANALYSIS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "Path to PCAP file"}
                    },
                    "required": ["pcap_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "technologies": {"type": "array", "description": "Detected technologies"},
                        "error": {"type": "string", "description": "Error message if detection failed"}
                    }
                },
                performance_metrics={"expected_response_time": "3-15s"},
                security_considerations=["Identifies network services and technologies"],
                tags=["technology", "fingerprinting", "detection", "services"]
            ),
            
            "detect_anomalies": PCAPAnalysisToolMetadata(
                tool_id="detect_anomalies",
                name="Network Anomaly Detector",
                description="Detect anomalies in network traffic including suspicious behavior and security threats",
                category=PCAPAnalysisToolCategory.ANOMALY_DETECTION,
                capabilities=[
                    PCAPAnalysisToolCapability.ANOMALY_DETECTION,
                    PCAPAnalysisToolCapability.TRAFFIC_SUMMARIZATION
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "Path to PCAP file"},
                        "anomaly_types": {"type": "array", "description": "Types of anomalies to detect", "items": {"type": "string"}}
                    },
                    "required": ["pcap_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "anomalies": {"type": "array", "description": "Detected anomalies"},
                        "error": {"type": "string", "description": "Error message if detection failed"}
                    }
                },
                performance_metrics={"expected_response_time": "5-20s"},
                security_considerations=["Identifies potential security threats and suspicious behavior"],
                tags=["anomaly", "detection", "security", "threats", "suspicious"]
            ),
            
            "extract_files": PCAPAnalysisToolMetadata(
                tool_id="extract_files",
                name="File Extractor",
                description="Extract files transferred over network protocols from PCAP files",
                category=PCAPAnalysisToolCategory.FILE_EXTRACTION,
                capabilities=[
                    PCAPAnalysisToolCapability.FILE_EXTRACTION,
                    PCAPAnalysisToolCapability.PROTOCOL_ANALYSIS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "Path to PCAP file"},
                        "protocols": {"type": "array", "description": "Protocols to extract files from", "items": {"type": "string"}}
                    },
                    "required": ["pcap_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "extracted_files": {"type": "array", "description": "List of extracted files"},
                        "error": {"type": "string", "description": "Error message if extraction failed"}
                    }
                },
                performance_metrics={"expected_response_time": "10-60s"},
                security_considerations=["Extracts files that may contain sensitive data"],
                tags=["file", "extraction", "protocols", "http", "ftp", "smb"]
            ),
            
            "filter_pcap": PCAPAnalysisToolMetadata(
                tool_id="filter_pcap",
                name="PCAP Filter",
                description="Filter PCAP files based on various criteria like IP addresses, ports, protocols, and time ranges",
                category=PCAPAnalysisToolCategory.PCAP_MANIPULATION,
                capabilities=[
                    PCAPAnalysisToolCapability.PCAP_FILTERING,
                    PCAPAnalysisToolCapability.TRAFFIC_SUMMARIZATION
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "input_pcap": {"type": "string", "description": "Input PCAP file path"},
                        "output_pcap": {"type": "string", "description": "Output PCAP file path"},
                        "filters": {"type": "object", "description": "Filter criteria"}
                    },
                    "required": ["input_pcap", "output_pcap", "filters"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "output_path": {"type": "string", "description": "Path to filtered PCAP"},
                        "message": {"type": "string", "description": "Success message"},
                        "error": {"type": "string", "description": "Error message if filtering failed"}
                    }
                },
                performance_metrics={"expected_response_time": "2-10s"},
                security_considerations=["Filters network traffic data"],
                tags=["pcap", "filtering", "manipulation", "criteria"]
            ),
            
            "merge_pcaps": PCAPAnalysisToolMetadata(
                tool_id="merge_pcaps",
                name="PCAP Merger",
                description="Merge multiple PCAP files into a single file",
                category=PCAPAnalysisToolCategory.PCAP_MANIPULATION,
                capabilities=[
                    PCAPAnalysisToolCapability.PCAP_MERGING
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_files": {"type": "array", "description": "List of PCAP files to merge", "items": {"type": "string"}},
                        "output_pcap": {"type": "string", "description": "Output merged PCAP file path"}
                    },
                    "required": ["pcap_files", "output_pcap"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "output_path": {"type": "string", "description": "Path to merged PCAP"},
                        "message": {"type": "string", "description": "Success message"},
                        "error": {"type": "string", "description": "Error message if merging failed"}
                    }
                },
                performance_metrics={"expected_response_time": "1-5s"},
                security_considerations=["Combines network traffic data"],
                tags=["pcap", "merging", "manipulation", "combine"]
            ),
            
            "create_scan_pcap": PCAPAnalysisToolMetadata(
                tool_id="create_scan_pcap",
                name="Scan PCAP Creator",
                description="Create PCAP files by capturing network traffic during host scanning activities",
                category=PCAPAnalysisToolCategory.SCAN_CAPTURE,
                capabilities=[
                    PCAPAnalysisToolCapability.SCAN_CAPTURE,
                    PCAPAnalysisToolCapability.TRAFFIC_SUMMARIZATION
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "target_hosts": {"type": "array", "description": "List of target host IP addresses", "items": {"type": "string"}},
                        "scan_type": {"type": "string", "description": "Type of scan (basic, comprehensive)", "default": "basic"},
                        "output_path": {"type": "string", "description": "Output PCAP file path"}
                    },
                    "required": ["target_hosts"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "output_path": {"type": "string", "description": "Path to created PCAP"},
                        "message": {"type": "string", "description": "Success message"},
                        "error": {"type": "string", "description": "Error message if creation failed"}
                    }
                },
                performance_metrics={"expected_response_time": "1-10s"},
                security_considerations=["Captures live network traffic during scanning"],
                tags=["scan", "capture", "pcap", "traffic", "monitoring"]
            ),
            
            "get_pcap_statistics": PCAPAnalysisToolMetadata(
                tool_id="get_pcap_statistics",
                name="PCAP Statistics",
                description="Get comprehensive statistics and metrics from PCAP analysis",
                category=PCAPAnalysisToolCategory.TRAFFIC_ANALYSIS,
                capabilities=[
                    PCAPAnalysisToolCapability.STATISTICAL_ANALYSIS,
                    PCAPAnalysisToolCapability.TRAFFIC_SUMMARIZATION
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "Path to PCAP file"}
                    },
                    "required": ["pcap_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "statistics": {"type": "object", "description": "PCAP statistics"},
                        "error": {"type": "string", "description": "Error message if analysis failed"}
                    }
                },
                performance_metrics={"expected_response_time": "3-15s"},
                security_considerations=["Provides statistical analysis of network traffic"],
                tags=["statistics", "metrics", "analysis", "pcap"]
            )
        }
        
        return registry
    
    def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover all available PCAP analysis tools."""
        tools = []
        
        for tool_id, metadata in self.tool_registry.items():
            tools.append({
                "tool_id": tool_id,
                "name": metadata.name,
                "description": metadata.description,
                "category": metadata.category.value,
                "capabilities": [cap.value for cap in metadata.capabilities],
                "input_schema": metadata.input_schema,
                "output_schema": metadata.output_schema,
                "performance_metrics": metadata.performance_metrics,
                "security_considerations": metadata.security_considerations,
                "tags": metadata.tags
            })
        
        logger.info(f"Discovered {len(tools)} PCAP analysis tools")
        return tools
    
    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a PCAP analysis tool."""
        if tool_id not in self.tool_registry:
            return {
                "success": False,
                "error": f"Tool not found: {tool_id}"
            }
        
        metadata = self.tool_registry[tool_id]
        logger.info(f"Executing tool: {tool_id} with parameters: {parameters}")
        
        try:
            if tool_id == "analyze_pcap_traffic":
                result = self.pcap_manager.analyze_pcap_file(
                    pcap_path=parameters["pcap_path"],
                    analysis_type=parameters.get("analysis_type", "comprehensive")
                )
            
            elif tool_id == "detect_technology_stack":
                # First analyze the PCAP, then extract technology information
                analysis_result = self.pcap_manager.analyze_pcap_file(
                    pcap_path=parameters["pcap_path"]
                )
                
                if analysis_result["success"]:
                    result = {
                        "success": True,
                        "technologies": analysis_result["summary"]["technology_stack"]
                    }
                else:
                    result = analysis_result
            
            elif tool_id == "detect_anomalies":
                # First analyze the PCAP, then extract anomaly information
                analysis_result = self.pcap_manager.analyze_pcap_file(
                    pcap_path=parameters["pcap_path"]
                )
                
                if analysis_result["success"]:
                    result = {
                        "success": True,
                        "anomalies": analysis_result["summary"]["anomalies"]
                    }
                else:
                    result = analysis_result
            
            elif tool_id == "extract_files":
                # First analyze the PCAP, then extract file information
                analysis_result = self.pcap_manager.analyze_pcap_file(
                    pcap_path=parameters["pcap_path"]
                )
                
                if analysis_result["success"]:
                    result = {
                        "success": True,
                        "extracted_files": analysis_result["summary"]["extracted_files"]
                    }
                else:
                    result = analysis_result
            
            elif tool_id == "filter_pcap":
                result = self.pcap_manager.filter_pcap_file(
                    input_pcap=parameters["input_pcap"],
                    output_pcap=parameters["output_pcap"],
                    filters=parameters["filters"]
                )
            
            elif tool_id == "merge_pcaps":
                result = self.pcap_manager.merge_pcap_files(
                    pcap_files=parameters["pcap_files"],
                    output_pcap=parameters["output_pcap"]
                )
            
            elif tool_id == "create_scan_pcap":
                result = self.pcap_manager.create_scan_pcap(
                    target_hosts=parameters["target_hosts"],
                    scan_type=parameters.get("scan_type", "basic"),
                    output_path=parameters.get("output_path")
                )
            
            elif tool_id == "get_pcap_statistics":
                # First analyze the PCAP, then extract statistics
                analysis_result = self.pcap_manager.analyze_pcap_file(
                    pcap_path=parameters["pcap_path"]
                )
                
                if analysis_result["success"]:
                    result = {
                        "success": True,
                        "statistics": analysis_result["summary"]["statistics"]
                    }
                else:
                    result = analysis_result
            
            else:
                result = {
                    "success": False,
                    "error": f"Tool execution not implemented: {tool_id}"
                }
            
            logger.info(f"Tool {tool_id} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_metadata(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool."""
        if tool_id in self.tool_registry:
            metadata = self.tool_registry[tool_id]
            return {
                "tool_id": tool_id,
                "name": metadata.name,
                "description": metadata.description,
                "category": metadata.category.value,
                "capabilities": [cap.value for cap in metadata.capabilities],
                "input_schema": metadata.input_schema,
                "output_schema": metadata.output_schema,
                "performance_metrics": metadata.performance_metrics,
                "security_considerations": metadata.security_considerations,
                "tags": metadata.tags
            }
        return None
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tools filtered by category."""
        tools = []
        
        for tool_id, metadata in self.tool_registry.items():
            if metadata.category.value == category:
                tools.append({
                    "tool_id": tool_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "category": metadata.category.value,
                    "capabilities": [cap.value for cap in metadata.capabilities],
                    "tags": metadata.tags
                })
        
        return tools
    
    def get_tools_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get tools filtered by capability."""
        tools = []
        
        for tool_id, metadata in self.tool_registry.items():
            if any(cap.value == capability for cap in metadata.capabilities):
                tools.append({
                    "tool_id": tool_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "category": metadata.category.value,
                    "capabilities": [cap.value for cap in metadata.capabilities],
                    "tags": metadata.tags
                })
        
        return tools
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for PCAP analysis operations."""
        return self.pcap_manager.get_performance_stats()
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.pcap_manager.get_analysis_history()

class PCAPAnalysisToolsQueryPathIntegration:
    """Integration layer for Query Path and PCAP analysis tools."""
    
    def __init__(self):
        self.mcp_layer = PCAPAnalysisMCPIntegrationLayer()
        self.tool_registry = self.mcp_layer.tool_registry
        
        logger.info("🚀 PCAP Analysis Tools Query Path Integration initialized")
    
    def get_tool_relevance_score(self, tool_id: str, query_context: Dict[str, Any]) -> float:
        """Calculate relevance score for a tool based on query context."""
        if tool_id not in self.tool_registry:
            return 0.0
        
        metadata = self.tool_registry[tool_id]
        score = 0.0
        
        # Check if query mentions PCAP-related terms
        query_text = query_context.get("query", "").lower()
        context_text = query_context.get("context", "").lower()
        
        # PCAP-related keywords
        pcap_keywords = ["pcap", "packet", "traffic", "network", "capture", "wireshark", "tshark"]
        analysis_keywords = ["analyze", "analyze", "detect", "extract", "filter", "merge", "scan"]
        
        # Score based on keyword matches
        for keyword in pcap_keywords:
            if keyword in query_text or keyword in context_text:
                score += 0.3
        
        for keyword in analysis_keywords:
            if keyword in query_text or keyword in context_text:
                score += 0.2
        
        # Score based on tool category relevance
        if "traffic" in query_text and metadata.category == PCAPAnalysisToolCategory.TRAFFIC_ANALYSIS:
            score += 0.4
        
        if "technology" in query_text and metadata.category == PCAPAnalysisToolCategory.TECHNOLOGY_DETECTION:
            score += 0.4
        
        if "anomaly" in query_text and metadata.category == PCAPAnalysisToolCategory.ANOMALY_DETECTION:
            score += 0.4
        
        if "file" in query_text and metadata.category == PCAPAnalysisToolCategory.FILE_EXTRACTION:
            score += 0.4
        
        if "filter" in query_text and metadata.category == PCAPAnalysisToolCategory.PCAP_MANIPULATION:
            score += 0.4
        
        if "scan" in query_text and metadata.category == PCAPAnalysisToolCategory.SCAN_CAPTURE:
            score += 0.4
        
        # Cap score at 1.0
        return min(score, 1.0)
    
    def suggest_tools_for_query(self, query_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest relevant tools for a given query."""
        suggestions = []
        
        for tool_id, metadata in self.tool_registry.items():
            relevance_score = self.get_tool_relevance_score(tool_id, query_context)
            
            if relevance_score > 0.2:  # Minimum relevance threshold
                suggestions.append({
                    "tool_id": tool_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "category": metadata.category.value,
                    "relevance_score": relevance_score,
                    "capabilities": [cap.value for cap in metadata.capabilities],
                    "tags": metadata.tags
                })
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return suggestions
    
    def generate_execution_plan(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan for complex PCAP analysis queries."""
        suggested_tools = self.suggest_tools_for_query(query_context)
        
        if not suggested_tools:
            return {
                "success": False,
                "error": "No relevant tools found for the query"
            }
        
        # Create execution plan
        execution_plan = {
            "query": query_context.get("query", ""),
            "suggested_tools": suggested_tools,
            "execution_steps": [],
            "estimated_time": 0,
            "dependencies": []
        }
        
        # Add execution steps based on tool suggestions
        for tool in suggested_tools[:3]:  # Limit to top 3 tools
            step = {
                "step_id": f"step_{len(execution_plan['execution_steps']) + 1}",
                "tool_id": tool["tool_id"],
                "tool_name": tool["name"],
                "description": f"Execute {tool['name']}",
                "estimated_time": "5-30s",
                "dependencies": []
            }
            
            execution_plan["execution_steps"].append(step)
            execution_plan["estimated_time"] += 30  # Add estimated time
        
        return execution_plan
    
    def get_tool_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of all tool capabilities."""
        capabilities_summary = {}
        
        for category in PCAPAnalysisToolCategory:
            capabilities_summary[category.value] = {
                "tools": [],
                "capabilities": set()
            }
        
        for tool_id, metadata in self.tool_registry.items():
            category = metadata.category.value
            capabilities_summary[category]["tools"].append({
                "tool_id": tool_id,
                "name": metadata.name,
                "description": metadata.description
            })
            
            for capability in metadata.capabilities:
                capabilities_summary[category]["capabilities"].add(capability.value)
        
        # Convert sets to lists for JSON serialization
        for category in capabilities_summary:
            capabilities_summary[category]["capabilities"] = list(capabilities_summary[category]["capabilities"])
        
        return capabilities_summary

# Global instance for lazy loading
_pcap_analysis_mcp_integration = None

def get_pcap_analysis_mcp_integration() -> PCAPAnalysisMCPIntegrationLayer:
    """Get or create PCAP analysis MCP integration instance (lazy loading)."""
    global _pcap_analysis_mcp_integration
    if _pcap_analysis_mcp_integration is None:
        _pcap_analysis_mcp_integration = PCAPAnalysisMCPIntegrationLayer()
    return _pcap_analysis_mcp_integration

def get_pcap_analysis_query_path_integration() -> PCAPAnalysisToolsQueryPathIntegration:
    """Get or create PCAP analysis query path integration instance (lazy loading)."""
    return PCAPAnalysisToolsQueryPathIntegration()
