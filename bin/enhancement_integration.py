#!/usr/bin/env python3
"""
Enhancement Integration System for Cybersecurity Agent
Integrates all enhancement systems into a unified interface
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import time

# Import all enhancement systems
from performance_optimizer import performance_optimizer, optimize, parallel_execute
from tool_selection_engine import tool_selection_engine, select_tools, get_recommendations
from dynamic_workflow_orchestrator import dynamic_orchestrator, route_workflow, get_recommendations as get_workflow_recommendations
from adaptive_context_manager import adaptive_context_manager, adapt_context, get_context_health
from enhanced_chat_interface import enhanced_chat_interface, process_enhanced_message
from advanced_mcp_manager import advanced_mcp_manager, auto_discover_tools, get_tool_recommendations as get_mcp_recommendations
from enhanced_knowledge_memory import enhanced_knowledge_memory, import_data_file, search_memory, get_memory_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancementIntegrationManager:
    """Main integration manager for all enhancement systems."""
    
    def __init__(self):
        self.performance_optimizer = performance_optimizer
        self.tool_selection_engine = tool_selection_engine
        self.dynamic_orchestrator = dynamic_orchestrator
        self.adaptive_context_manager = adaptive_context_manager
        self.enhanced_chat_interface = enhanced_chat_interface
        self.advanced_mcp_manager = advanced_mcp_manager
        self.enhanced_knowledge_memory = enhanced_knowledge_memory
        
        self.integration_status = {}
        self.performance_metrics = {}
    
    async def initialize_all_systems(self) -> Dict[str, Any]:
        """Initialize all enhancement systems."""
        initialization_results = {}
        
        try:
            # Initialize performance optimization
            logger.info("Initializing Performance Optimization System...")
            self.performance_optimizer = performance_optimizer
            initialization_results["performance_optimization"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Performance optimization initialization failed: {e}")
            initialization_results["performance_optimization"] = f"❌ Failed: {e}"
        
        try:
            # Initialize tool selection engine
            logger.info("Initializing Tool Selection Engine...")
            self.tool_selection_engine = tool_selection_engine
            initialization_results["tool_selection"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Tool selection initialization failed: {e}")
            initialization_results["tool_selection"] = f"❌ Failed: {e}"
        
        try:
            # Initialize dynamic workflow orchestration
            logger.info("Initializing Dynamic Workflow Orchestration...")
            self.dynamic_orchestrator = dynamic_orchestrator
            initialization_results["dynamic_workflow"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Dynamic workflow initialization failed: {e}")
            initialization_results["dynamic_workflow"] = f"❌ Failed: {e}"
        
        try:
            # Initialize adaptive context management
            logger.info("Initializing Adaptive Context Management...")
            self.adaptive_context_manager = adaptive_context_manager
            initialization_results["adaptive_context"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Adaptive context initialization failed: {e}")
            initialization_results["adaptive_context"] = f"❌ Failed: {e}"
        
        try:
            # Initialize enhanced chat interface
            logger.info("Initializing Enhanced Chat Interface...")
            self.enhanced_chat_interface = enhanced_chat_interface
            initialization_results["enhanced_chat"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Enhanced chat initialization failed: {e}")
            initialization_results["enhanced_chat"] = f"❌ Failed: {e}"
        
        try:
            # Initialize advanced MCP management
            logger.info("Initializing Advanced MCP Management...")
            self.advanced_mcp_manager = advanced_mcp_manager
            initialization_results["advanced_mcp"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Advanced MCP initialization failed: {e}")
            initialization_results["advanced_mcp"] = f"❌ Failed: {e}"
        
        try:
            # Initialize enhanced knowledge memory
            logger.info("Initializing Enhanced Knowledge Memory...")
            self.enhanced_knowledge_memory = enhanced_knowledge_memory
            initialization_results["enhanced_memory"] = "✅ Initialized"
        except Exception as e:
            logger.error(f"Enhanced memory initialization failed: {e}")
            initialization_results["enhanced_memory"] = f"❌ Failed: {e}"
        
        self.integration_status = initialization_results
        return initialization_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all enhancement systems."""
        status = {
            "integration_status": self.integration_status,
            "performance_metrics": await self._get_performance_metrics(),
            "system_health": await self._assess_system_health(),
            "recommendations": await self._generate_system_recommendations()
        }
        
        return status
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from all systems."""
        metrics = {}
        
        try:
            # Performance optimizer metrics
            if hasattr(self.performance_optimizer, 'get_optimization_stats'):
                metrics["performance_optimizer"] = self.performance_optimizer.get_optimization_stats()
        except Exception as e:
            logger.error(f"Failed to get performance optimizer metrics: {e}")
        
        try:
            # Tool selection metrics
            if hasattr(self.tool_selection_engine, 'get_tool_registry'):
                registry = self.tool_selection_engine.tool_registry
                metrics["tool_selection"] = {
                    "total_tools": len(registry.tools),
                    "local_tools": len(registry.get_local_tools()),
                    "mcp_tools": len(registry.get_mcp_tools())
                }
        except Exception as e:
            logger.error(f"Failed to get tool selection metrics: {e}")
        
        try:
            # MCP integration metrics
            if hasattr(self.advanced_mcp_manager, 'get_integration_status'):
                metrics["mcp_integration"] = self.advanced_mcp_manager.get_integration_status()
        except Exception as e:
            logger.error(f"Failed to get MCP integration metrics: {e}")
        
        try:
            # Memory metrics
            if hasattr(self.enhanced_knowledge_memory, 'get_memory_stats'):
                metrics["memory"] = await self.enhanced_knowledge_memory.get_memory_stats()
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
        
        return metrics
    
    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        health_scores = {}
        
        # Check each system's health
        for system_name, status in self.integration_status.items():
            if "✅" in status:
                health_scores[system_name] = "healthy"
            elif "❌" in status:
                health_scores[system_name] = "unhealthy"
            else:
                health_scores[system_name] = "unknown"
        
        # Calculate overall health
        healthy_systems = sum(1 for score in health_scores.values() if score == "healthy")
        total_systems = len(health_scores)
        
        if total_systems == 0:
            overall_health = "unknown"
        elif healthy_systems == total_systems:
            overall_health = "excellent"
        elif healthy_systems >= total_systems * 0.8:
            overall_health = "good"
        elif healthy_systems >= total_systems * 0.6:
            overall_health = "fair"
        else:
            overall_health = "poor"
        
        return {
            "overall_health": overall_health,
            "system_health": health_scores,
            "healthy_systems": healthy_systems,
            "total_systems": total_systems,
            "health_percentage": (healthy_systems / total_systems * 100) if total_systems > 0 else 0
        }
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system improvement recommendations."""
        recommendations = []
        
        # Check integration status
        unhealthy_systems = [name for name, status in self.integration_status.items() if "❌" in status]
        if unhealthy_systems:
            recommendations.append(f"Fix initialization issues in: {', '.join(unhealthy_systems)}")
        
        # Check performance metrics
        try:
            if hasattr(self.performance_optimizer, 'get_optimization_stats'):
                stats = self.performance_optimizer.get_optimization_stats()
                if stats.get("cache", {}).get("cache_size_mb", 0) > 50:
                    recommendations.append("Consider increasing cache size for better performance")
        except Exception:
            pass
        
        # Check tool registry
        try:
            if hasattr(self.tool_selection_engine, 'get_tool_registry'):
                registry = self.tool_selection_engine.tool_registry
                if len(registry.tools) < 5:
                    recommendations.append("Consider adding more tools to the registry")
        except Exception:
            pass
        
        return recommendations
    
    async def execute_enhanced_workflow(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow using all enhancement systems."""
        workflow_result = {
            "user_input": user_input,
            "execution_start": datetime.now().isoformat(),
            "enhancements_applied": [],
            "results": {},
            "performance_metrics": {},
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # 1. Dynamic workflow orchestration
            logger.info("Executing dynamic workflow orchestration...")
            workflow_plan = await self.dynamic_orchestrator.route_workflow(user_input, context)
            workflow_result["results"]["workflow_plan"] = workflow_plan.to_dict()
            workflow_result["enhancements_applied"].append("dynamic_workflow_orchestration")
            
            # 2. Tool selection optimization
            logger.info("Optimizing tool selection...")
            optimal_tools = await self.tool_selection_engine.select_optimal_tools(user_input, context)
            workflow_result["results"]["selected_tools"] = [tool.to_dict() for tool in optimal_tools]
            workflow_result["enhancements_applied"].append("intelligent_tool_selection")
            
            # 3. Context adaptation
            logger.info("Adapting context...")
            adapted_context = await self.adaptive_context_manager.adapt_context(context, "workflow_execution")
            workflow_result["results"]["adapted_context"] = adapted_context
            workflow_result["enhancements_applied"].append("real_time_context_adaptation")
            
            # 4. Enhanced chat processing
            logger.info("Processing enhanced chat...")
            chat_response = await self.enhanced_chat_interface.process_message(user_input, adapted_context)
            workflow_result["results"]["chat_response"] = chat_response.to_dict()
            workflow_result["enhancements_applied"].append("enhanced_chat_interface")
            
            # 5. Performance optimization
            logger.info("Applying performance optimizations...")
            if hasattr(self.performance_optimizer, 'optimize_workflow'):
                optimized_workflow = await self.performance_optimizer.optimize_workflow(workflow_plan.steps)
                workflow_result["results"]["optimized_workflow"] = optimized_workflow
                workflow_result["enhancements_applied"].append("performance_optimization")
            
            # 6. MCP tool integration
            logger.info("Integrating MCP tools...")
            if hasattr(self.advanced_mcp_manager, 'get_tool_recommendations'):
                mcp_recommendations = await self.advanced_mcp_manager.get_tool_recommendations(user_input, context)
                workflow_result["results"]["mcp_recommendations"] = mcp_recommendations
                workflow_result["enhancements_applied"].append("advanced_mcp_integration")
            
        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {e}")
            workflow_result["error"] = str(e)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        workflow_result["execution_time"] = execution_time
        
        # Get performance metrics
        try:
            workflow_result["performance_metrics"] = await self._get_performance_metrics()
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
        
        workflow_result["execution_end"] = datetime.now().isoformat()
        
        return workflow_result
    
    async def import_knowledge_data(self, file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
        """Import data into the enhanced knowledge memory system."""
        try:
            logger.info(f"Importing data from {file_path}...")
            import_result = await self.enhanced_knowledge_memory.import_data_file(file_path, import_config)
            
            return {
                "success": True,
                "import_result": import_result.to_dict(),
                "message": f"Successfully imported {import_result.nodes_created} nodes and {import_result.relationships_created} relationships"
            }
            
        except Exception as e:
            logger.error(f"Data import failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Data import failed"
            }
    
    async def search_knowledge_base(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search the enhanced knowledge base."""
        try:
            logger.info(f"Searching knowledge base for: {query}")
            search_results = await self.enhanced_knowledge_memory.search_memory(query, filters)
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "result_count": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Knowledge search failed"
            }
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all systems."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system_status": await self.get_system_status(),
            "performance_metrics": await self._get_performance_metrics(),
            "memory_stats": await self.enhanced_knowledge_memory.get_memory_stats(),
            "mcp_status": self.advanced_mcp_manager.get_integration_status() if hasattr(self.advanced_mcp_manager, 'get_integration_status') else {},
            "tool_registry": {
                "total_tools": len(self.tool_selection_engine.tool_registry.tools) if hasattr(self.tool_selection_engine, 'tool_registry') else 0
            }
        }
        
        return stats

# Global integration manager instance
enhancement_integration_manager = EnhancementIntegrationManager()

# Convenience functions
async def initialize_enhancements() -> Dict[str, Any]:
    """Initialize all enhancement systems."""
    return await enhancement_integration_manager.initialize_all_systems()

async def get_enhancement_status() -> Dict[str, Any]:
    """Get status of all enhancement systems."""
    return await enhancement_integration_manager.get_system_status()

async def execute_enhanced_workflow(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute enhanced workflow using all systems."""
    return await enhancement_integration_manager.execute_enhanced_workflow(user_input, context)

async def import_knowledge_data(file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
    """Import data into knowledge base."""
    return await enhancement_integration_manager.import_knowledge_data(file_path, import_config)

async def search_knowledge_base(query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Search knowledge base."""
    return await enhancement_integration_manager.search_knowledge_base(query, filters)

async def get_comprehensive_stats() -> Dict[str, Any]:
    """Get comprehensive system statistics."""
    return await enhancement_integration_manager.get_comprehensive_stats()

# CLI interface for testing
async def main():
    """Main CLI interface for testing enhancements."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cybersecurity Agent Enhancement Integration CLI")
    parser.add_argument("--initialize", action="store_true", help="Initialize all enhancement systems")
    parser.add_argument("--status", action="store_true", help="Get system status")
    parser.add_argument("--stats", action="store_true", help="Get comprehensive statistics")
    parser.add_argument("--workflow", type=str, help="Execute enhanced workflow with user input")
    parser.add_argument("--import", dest="import_file", type=str, help="Import data file")
    parser.add_argument("--search", type=str, help="Search knowledge base")
    parser.add_argument("--config", type=str, help="Import configuration file (JSON)")
    
    args = parser.parse_args()
    
    if args.initialize:
        print("🚀 Initializing all enhancement systems...")
        results = await initialize_enhancements()
        print(json.dumps(results, indent=2))
    
    elif args.status:
        print("📊 Getting system status...")
        status = await get_enhancement_status()
        print(json.dumps(status, indent=2))
    
    elif args.stats:
        print("📈 Getting comprehensive statistics...")
        stats = await get_comprehensive_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.workflow:
        print(f"⚡ Executing enhanced workflow: {args.workflow}")
        context = {"user_preferences": {}, "available_tools": []}
        result = await execute_enhanced_workflow(args.workflow, context)
        print(json.dumps(result, indent=2))
    
    elif args.import_file:
        print(f"📥 Importing data file: {args.import_file}")
        import_config = {}
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    import_config = json.load(f)
            except Exception as e:
                print(f"Failed to load config file: {e}")
                return
        
        result = await import_knowledge_data(args.import_file, import_config)
        print(json.dumps(result, indent=2))
    
    elif args.search:
        print(f"🔍 Searching knowledge base: {args.search}")
        result = await search_knowledge_base(args.search)
        print(json.dumps(result, indent=2))
    
    else:
        print("Cybersecurity Agent Enhancement Integration CLI")
        print("Use --help for available options")

if __name__ == "__main__":
    asyncio.run(main())
