#!/usr/bin/env python3
"""
MCP Integration Layer - Advanced Workflow Integration

Provides:
- Multimodal MCP Integration
- Evolutionary MCP Tool Development
- Reflective MCP Architecture
- Dynamic Tool Discovery and Registration
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Type
from enum import Enum
import inspect
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

# Import from workflow_templates instead
from workflow_templates import WorkflowContext, ProblemType
from enhanced_context_memory import EnhancedContextMemoryManager

logger = logging.getLogger(__name__)

class MCPToolCategory(Enum):
    """Categories of MCP tools."""
    CONTEXT_MEMORY = "context_memory"
    DATA_ANALYSIS = "data_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    HOST_ANALYSIS = "host_analysis"
    THREAT_INTELLIGENCE = "threat_intelligence"
    COMPLIANCE = "compliance"
    INCIDENT_RESPONSE = "incident_response"
    FORENSICS = "forensics"
    REPORTING = "reporting"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class MCPToolCapability(Enum):
    """Capabilities of MCP tools."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    INTEGRATE = "integrate"
    MONITOR = "monitor"
    ALERT = "alert"

@dataclass
class MCPToolMetadata:
    """Metadata for MCP tools."""
    tool_id: str
    name: str
    description: str
    category: MCPToolCategory
    capabilities: List[MCPToolCapability]
    version: str = "1.0.0"
    author: str = "System"
    dependencies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

@dataclass
class MCPToolInstance:
    """Instance of an MCP tool."""
    tool_id: str
    tool_func: Callable
    metadata: MCPToolMetadata
    instance_id: str
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    last_used: float = 0
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    success_rate: float = 1.0

class MCPToolRegistry:
    """Advanced registry for MCP tools with evolutionary capabilities."""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolInstance] = {}
        self.tool_metadata: Dict[str, MCPToolMetadata] = {}
        self.categories: Dict[MCPToolCategory, List[str]] = defaultdict(list)
        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Tool discovery and registration
        self.discovery_plugins: List[Callable] = []
        self.registration_hooks: List[Callable] = []
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.performance_thresholds = {
            'execution_time': 5.0,  # seconds
            'error_rate': 0.1,      # 10%
            'success_rate': 0.9     # 90%
        }
    
    def register_tool(self, tool_id: str, tool_func: Callable, metadata: MCPToolMetadata) -> str:
        """Register a new MCP tool with advanced capabilities."""
        instance_id = f"{tool_id}_{int(time.time())}"
        
        # Create tool instance
        tool_instance = MCPToolInstance(
            tool_id=tool_id,
            tool_func=tool_func,
            metadata=metadata,
            instance_id=instance_id
        )
        
        # Store tool
        self.tools[instance_id] = tool_instance
        self.tool_metadata[tool_id] = metadata
        
        # Categorize tool
        self.categories[metadata.category].append(instance_id)
        
        # Run registration hooks
        for hook in self.registration_hooks:
            try:
                hook(tool_instance)
            except Exception as e:
                logger.warning(f"Registration hook failed: {e}")
        
        logger.info(f"Registered MCP tool: {tool_id} ({instance_id})")
        return instance_id
    
    def get_tool(self, tool_id: str, category: Optional[MCPToolCategory] = None) -> Optional[MCPToolInstance]:
        """Get a tool by ID with optional category filtering."""
        if category:
            # Search in specific category
            for instance_id in self.categories.get(category, []):
                if self.tools[instance_id].tool_id == tool_id:
                    return self.tools[instance_id]
        else:
            # Search in all tools
            for instance in self.tools.values():
                if instance.tool_id == tool_id:
                    return instance
        
        return None
    
    def get_tools_by_category(self, category: MCPToolCategory) -> List[MCPToolInstance]:
        """Get all tools in a specific category."""
        return [self.tools[instance_id] for instance_id in self.categories.get(category, [])]
    
    def get_tools_by_capability(self, capability: MCPToolCapability) -> List[MCPToolInstance]:
        """Get all tools with a specific capability."""
        tools = []
        for instance in self.tools.values():
            if capability in instance.metadata.capabilities:
                tools.append(instance)
        return tools
    
    async def execute_tool(self, tool_id: str, *args, **kwargs) -> Any:
        """Execute a tool with performance monitoring."""
        tool_instance = self.get_tool(tool_id)
        if not tool_instance:
            raise ValueError(f"Tool not found: {tool_id}")
        
        start_time = time.time()
        success = False
        
        try:
            # Execute tool
            if asyncio.iscoroutinefunction(tool_instance.tool_func):
                result = await tool_instance.tool_func(*args, **kwargs)
            else:
                result = tool_instance.tool_func(*args, **kwargs)
            
            success = True
            return result
            
        except Exception as e:
            tool_instance.error_count += 1
            logger.error(f"Tool execution failed: {tool_id} - {e}")
            raise
        
        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_tool_performance(tool_instance, execution_time, success)
    
    def _record_tool_performance(self, tool_instance: MCPToolInstance, execution_time: float, success: bool):
        """Record tool performance metrics."""
        if not self.monitoring_enabled:
            return
        
        # Update instance metrics
        tool_instance.usage_count += 1
        tool_instance.last_used = time.time()
        tool_instance.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': success
        })
        
        # Calculate success rate
        recent_performance = tool_instance.performance_history[-100:]  # Last 100 executions
        success_count = sum(1 for p in recent_performance if p['success'])
        tool_instance.success_rate = success_count / len(recent_performance)
        
        # Store in registry metrics
        self.performance_metrics[tool_instance.tool_id].append({
            'instance_id': tool_instance.instance_id,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': success
        })
        
        # Check performance thresholds
        self._check_performance_thresholds(tool_instance)
    
    def _check_performance_thresholds(self, tool_instance: MCPToolInstance):
        """Check if tool performance meets thresholds."""
        recent_metrics = tool_instance.performance_history[-50:]  # Last 50 executions
        
        if not recent_metrics:
            return
        
        avg_execution_time = sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics)
        error_rate = 1 - tool_instance.success_rate
        
        # Check thresholds
        if avg_execution_time > self.performance_thresholds['execution_time']:
            logger.warning(f"Tool {tool_instance.tool_id} exceeds execution time threshold: {avg_execution_time:.2f}s")
        
        if error_rate > self.performance_thresholds['error_rate']:
            logger.warning(f"Tool {tool_instance.tool_id} exceeds error rate threshold: {error_rate:.2%}")
        
        if tool_instance.success_rate < self.performance_thresholds['success_rate']:
            logger.warning(f"Tool {tool_instance.tool_id} below success rate threshold: {tool_instance.success_rate:.2%}")
    
    def evolve_tool(self, tool_id: str, evolution_data: Dict[str, Any]):
        """Record tool evolution for analysis."""
        evolution_record = {
            'tool_id': tool_id,
            'timestamp': time.time(),
            'evolution_data': evolution_data,
            'current_performance': self._get_tool_performance_summary(tool_id)
        }
        
        self.evolution_history.append(evolution_record)
        
        # Update tool metadata
        if tool_id in self.tool_metadata:
            self.tool_metadata[tool_id].evolution_history.append(evolution_data)
            self.tool_metadata[tool_id].last_updated = time.time()
    
    def _get_tool_performance_summary(self, tool_id: str) -> Dict[str, Any]:
        """Get performance summary for a tool."""
        metrics = self.performance_metrics.get(tool_id, [])
        if not metrics:
            return {}
        
        recent_metrics = metrics[-100:]  # Last 100 executions
        
        return {
            'total_executions': len(metrics),
            'recent_executions': len(recent_metrics),
            'avg_execution_time': sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics),
            'success_rate': sum(1 for m in recent_metrics if m['success']) / len(recent_metrics),
            'last_execution': max(m['timestamp'] for m in metrics) if metrics else 0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'total_tools': len(self.tools),
            'categories': {cat.value: len(instances) for cat, instances in self.categories.items()},
            'performance_summary': {},
            'evolution_summary': {
                'total_evolutions': len(self.evolution_history),
                'recent_evolutions': len([e for e in self.evolution_history if time.time() - e['timestamp'] < 86400])
            }
        }
        
        # Add performance summary for each tool
        for tool_id in self.tool_metadata:
            report['performance_summary'][tool_id] = self._get_tool_performance_summary(tool_id)
        
        return report

class MCPToolDiscovery:
    """Discovers and registers MCP tools automatically."""
    
    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry
        self.discovery_plugins: List[Callable] = []
        self.discovery_history: List[Dict[str, Any]] = []
        
    def add_discovery_plugin(self, plugin: Callable):
        """Add a discovery plugin."""
        self.discovery_plugins.append(plugin)
    
    async def discover_tools(self) -> List[str]:
        """Discover new tools using all plugins."""
        discovered_tools = []
        
        for plugin in self.discovery_plugins:
            try:
                plugin_result = await plugin()
                if isinstance(plugin_result, list):
                    discovered_tools.extend(plugin_result)
                elif isinstance(plugin_result, dict):
                    discovered_tools.append(plugin_result)
                
                # Record discovery
                self.discovery_history.append({
                    'plugin': plugin.__name__,
                    'timestamp': time.time(),
                    'result': plugin_result
                })
                
            except Exception as e:
                logger.error(f"Discovery plugin failed: {plugin.__name__} - {e}")
        
        return discovered_tools
    
    def get_discovery_history(self) -> List[Dict[str, Any]]:
        """Get discovery history."""
        return self.discovery_history

class MCPToolEvolution:
    """Manages tool evolution and adaptation."""
    
    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry
        self.evolution_rules: List[Dict[str, Any]] = []
        self.adaptation_triggers: List[Dict[str, Any]] = []
        
    def add_evolution_rule(self, rule: Dict[str, Any]):
        """Add an evolution rule."""
        self.evolution_rules.append(rule)
    
    def add_adaptation_trigger(self, trigger: Dict[str, Any]):
        """Add an adaptation trigger."""
        self.adaptation_triggers.append(trigger)
    
    def analyze_tool_performance(self, tool_id: str) -> Dict[str, Any]:
        """Analyze tool performance and suggest improvements."""
        performance = self.registry._get_tool_performance_summary(tool_id)
        
        analysis = {
            'tool_id': tool_id,
            'performance': performance,
            'suggestions': [],
            'evolution_opportunities': []
        }
        
        # Analyze performance and suggest improvements
        if performance.get('avg_execution_time', 0) > 2.0:
            analysis['suggestions'].append("Consider optimizing tool execution for better performance")
        
        if performance.get('success_rate', 1.0) < 0.95:
            analysis['suggestions'].append("Investigate error patterns and improve error handling")
        
        # Identify evolution opportunities
        if performance.get('total_executions', 0) > 100:
            analysis['evolution_opportunities'].append("Tool has sufficient usage data for evolution analysis")
        
        return analysis
    
    def suggest_tool_evolution(self, tool_id: str) -> List[Dict[str, Any]]:
        """Suggest specific evolution paths for a tool."""
        analysis = self.analyze_tool_performance(tool_id)
        suggestions = []
        
        # Performance-based suggestions
        if analysis['performance'].get('avg_execution_time', 0) > 2.0:
            suggestions.append({
                'type': 'performance_optimization',
                'description': 'Optimize tool execution for better performance',
                'priority': 'high',
                'estimated_effort': 'medium'
            })
        
        # Usage-based suggestions
        if analysis['performance'].get('total_executions', 0) > 500:
            suggestions.append({
                'type': 'feature_expansion',
                'description': 'Consider adding new features based on usage patterns',
                'priority': 'medium',
                'estimated_effort': 'high'
            })
        
        return suggestions

class MCPIntegrationLayer:
    """Main integration layer for MCP tools and workflows."""
    
    def __init__(self, context_memory: EnhancedContextMemoryManager):
        self.context_memory = context_memory
        self.registry = MCPToolRegistry()
        self.discovery = MCPToolDiscovery(self.registry)
        self.evolution = MCPToolEvolution(self.registry)
        
        # Initialize with context memory tools
        self._initialize_context_memory_tools()
        
        # Register discovery plugins
        self._register_discovery_plugins()
    
    def _initialize_context_memory_tools(self):
        """Initialize context memory MCP tools."""
        # Context memory tools
        self.registry.register_tool(
            "get_workflow_context",
            self.context_memory.get_workflow_context,
            MCPToolMetadata(
                tool_id="get_workflow_context",
                name="Get Workflow Context",
                description="Retrieve relevant context for workflow execution",
                category=MCPToolCategory.CONTEXT_MEMORY,
                capabilities=[MCPToolCapability.READ, MCPToolCapability.ANALYZE]
            )
        )
        
        self.registry.register_tool(
            "add_memory",
            self.context_memory.add_memory,
            MCPToolMetadata(
                tool_id="add_memory",
                name="Add Memory",
                description="Add new memory to context",
                category=MCPToolCategory.CONTEXT_MEMORY,
                capabilities=[MCPToolCapability.WRITE]
            )
        )
        
        self.registry.register_tool(
            "search_memories",
            self.context_memory.search_memories,
            MCPToolMetadata(
                tool_id="search_memories",
                name="Search Memories",
                description="Search across all memories",
                category=MCPToolCategory.CONTEXT_MEMORY,
                capabilities=[MCPToolCapability.READ, MCPToolCapability.ANALYZE]
            )
        )
    
    def _register_discovery_plugins(self):
        """Register tool discovery plugins."""
        # Add discovery plugins for different tool types
        self.discovery.add_discovery_plugin(self._discover_context_memory_tools)
        self.discovery.add_discovery_plugin(self._discover_workflow_tools)
    
    async def _discover_context_memory_tools(self) -> List[Dict[str, Any]]:
        """Discover context memory related tools."""
        # This would integrate with your existing MCP tools
        discovered = []
        
        # Example: Discover available domains
        try:
            domains = self.context_memory.master_catalog.domains.keys()
            for domain in domains:
                discovered.append({
                    'tool_id': f"domain_{domain}_tools",
                    'category': MCPToolCategory.CONTEXT_MEMORY.value,
                    'capabilities': [MCPToolCapability.READ.value, MCPToolCapability.ANALYZE.value]
                })
        except Exception as e:
            logger.warning(f"Context memory discovery failed: {e}")
        
        return discovered
    
    async def _discover_workflow_tools(self) -> List[Dict[str, Any]]:
        """Discover workflow-related tools."""
        # This would integrate with your workflow engine
        discovered = []
        
        # Example: Discover workflow templates
        discovered.append({
            'tool_id': 'workflow_templates',
            'category': MCPToolCategory.INTEGRATION.value,
            'capabilities': [MCPToolCapability.READ.value, MCPToolCapability.EXECUTE.value]
        })
        
        return discovered
    
    async def execute_workflow_with_tools(self, workflow_context: WorkflowContext, required_tools: List[str]) -> Dict[str, Any]:
        """Execute a workflow using MCP tools."""
        # Discover required tools
        available_tools = []
        for tool_id in required_tools:
            tool = self.registry.get_tool(tool_id)
            if tool:
                available_tools.append(tool)
            else:
                logger.warning(f"Required tool not found: {tool_id}")
        
        # Execute workflow using available tools
        workflow_result = {
            'workflow_id': workflow_context.problem_description[:20],
            'tools_used': [tool.tool_id for tool in available_tools],
            'execution_results': {},
            'performance_metrics': {}
        }
        
        # Execute each tool
        for tool in available_tools:
            try:
                result = await self.registry.execute_tool(tool.tool_id, workflow_context.problem_description)
                workflow_result['execution_results'][tool.tool_id] = result
            except Exception as e:
                workflow_result['execution_results'][tool.tool_id] = {'error': str(e)}
        
        return workflow_result
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of MCP integration."""
        return {
            'registry_status': {
                'total_tools': len(self.registry.tools),
                'categories': {cat.value: len(instances) for cat, instances in self.registry.categories.items()}
            },
            'discovery_status': {
                'plugins': len(self.discovery.discovery_plugins),
                'history': len(self.discovery.discovery_history)
            },
            'evolution_status': {
                'rules': len(self.evolution.evolution_rules),
                'triggers': len(self.evolution.adaptation_triggers)
            },
            'performance_report': self.registry.get_performance_report()
        }

# Example usage
async def main():
    """Example usage of MCP integration layer."""
    from enhanced_context_memory import EnhancedContextMemoryManager
    
    # Initialize context memory
    context_memory = EnhancedContextMemoryManager(".")
    
    # Create integration layer
    integration = MCPIntegrationLayer(context_memory)
    
    # Discover tools
    discovered_tools = await integration.discovery.discover_tools()
    print(f"🔍 Discovered {len(discovered_tools)} tools")
    
    # Get integration status
    status = integration.get_integration_status()
    print(f"📊 Integration status: {status['registry_status']['total_tools']} tools registered")
    
    # Execute workflow with tools
    workflow_context = WorkflowContext(
        problem_type=ProblemType.THREAT_HUNTING,
        problem_description="Investigate potential APT29 activity"
    )
    
    result = await integration.execute_workflow_with_tools(
        workflow_context, 
        ['get_workflow_context', 'search_memories']
    )
    
    print(f"✅ Workflow executed with {len(result['tools_used'])} tools")
    print(f"   Results: {list(result['execution_results'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())

