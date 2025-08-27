#!/usr/bin/env python3
"""
Intelligent Tool Selection Engine for Cybersecurity Agent
Provides local tool prioritization and performance-based weighting
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Tool type enumeration."""
    LOCAL = "local"
    MCP = "mcp"
    EXTERNAL = "external"
    HYBRID = "hybrid"

class ToolCategory(Enum):
    """Tool category enumeration."""
    THREAT_INTELLIGENCE = "threat_intelligence"
    POLICY_ANALYSIS = "policy_analysis"
    COMPLIANCE = "compliance"
    INCIDENT_RESPONSE = "incident_response"
    FORENSICS = "forensics"
    NETWORK_ANALYSIS = "network_analysis"
    DATA_PROCESSING = "data_processing"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    UTILITY = "utility"

@dataclass
class Tool:
    """Tool information and capabilities."""
    id: str
    name: str
    description: str
    tool_type: ToolType
    category: ToolCategory
    tags: List[str]
    is_local: bool
    performance_score: float
    resource_usage: Dict[str, Any]
    dependencies: List[str]
    version: str
    last_updated: datetime
    is_available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['tool_type'] = self.tool_type.value
        data['category'] = self.category.value
        data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class ToolPerformance:
    """Tool performance metrics."""
    tool_id: str
    execution_count: int
    avg_execution_time: float
    success_rate: float
    cache_hit_rate: float
    resource_efficiency: float
    last_used: datetime
    performance_trend: str  # "improving", "stable", "declining"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['last_used'] = self.last_used.isoformat()
        return data

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.registry_db_path = Path("knowledge-objects/tool_registry.db")
        self.registry_db_path.parent.mkdir(exist_ok=True)
        self._init_registry_db()
        self._load_default_tools()
    
    def _init_registry_db(self):
        """Initialize tool registry database."""
        try:
            with sqlite3.connect(self.registry_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tools (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        tool_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        tags TEXT,
                        is_local BOOLEAN NOT NULL,
                        performance_score REAL DEFAULT 0.0,
                        resource_usage TEXT,
                        dependencies TEXT,
                        version TEXT,
                        last_updated TEXT,
                        is_available BOOLEAN DEFAULT TRUE
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tool_performance (
                        tool_id TEXT PRIMARY KEY,
                        execution_count INTEGER DEFAULT 0,
                        avg_execution_time REAL DEFAULT 0.0,
                        success_rate REAL DEFAULT 1.0,
                        cache_hit_rate REAL DEFAULT 0.0,
                        resource_efficiency REAL DEFAULT 0.0,
                        last_used TEXT,
                        performance_trend TEXT DEFAULT 'stable'
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_type ON tools(tool_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON tools(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_is_local ON tools(is_local)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_performance ON tools(performance_score)")
        except Exception as e:
            logger.warning(f"Tool registry database initialization failed: {e}")
    
    def _load_default_tools(self):
        """Load default tools into registry."""
        default_tools = [
            Tool(
                id="csv_processor",
                name="CSV Processor",
                description="Process and analyze CSV files",
                tool_type=ToolType.LOCAL,
                category=ToolCategory.DATA_PROCESSING,
                tags=["csv", "data", "analysis"],
                is_local=True,
                performance_score=0.9,
                resource_usage={"memory_mb": 10, "cpu_percent": 5},
                dependencies=["pandas"],
                version="1.0.0",
                last_updated=datetime.now()
            ),
            Tool(
                id="threat_analyzer",
                name="Threat Analyzer",
                description="Analyze threat intelligence data",
                tool_type=ToolType.LOCAL,
                category=ToolCategory.THREAT_INTELLIGENCE,
                tags=["threat", "intelligence", "analysis"],
                is_local=True,
                performance_score=0.85,
                resource_usage={"memory_mb": 25, "cpu_percent": 15},
                dependencies=["pandas", "numpy"],
                version="1.0.0",
                last_updated=datetime.now()
            ),
            Tool(
                id="policy_mapper",
                name="Policy Mapper",
                description="Map and analyze security policies",
                tool_type=ToolType.LOCAL,
                category=ToolCategory.POLICY_ANALYSIS,
                tags=["policy", "mapping", "compliance"],
                is_local=True,
                performance_score=0.8,
                resource_usage={"memory_mb": 15, "cpu_percent": 10},
                dependencies=["pandas"],
                version="1.0.0",
                last_updated=datetime.now()
            ),
            Tool(
                id="visualization_engine",
                name="Visualization Engine",
                description="Create charts and visualizations",
                tool_type=ToolType.LOCAL,
                category=ToolCategory.VISUALIZATION,
                tags=["visualization", "charts", "graphs"],
                is_local=True,
                performance_score=0.75,
                resource_usage={"memory_mb": 30, "cpu_percent": 20},
                dependencies=["matplotlib", "seaborn"],
                version="1.0.0",
                last_updated=datetime.now()
            )
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.id] = tool
        
        # Store in database
        try:
            with sqlite3.connect(self.registry_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tools 
                    (id, name, description, tool_type, category, tags, is_local, 
                     performance_score, resource_usage, dependencies, version, last_updated, is_available)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool.id, tool.name, tool.description, tool.tool_type.value,
                    tool.category.value, json.dumps(tool.tags), tool.is_local,
                    tool.performance_score, json.dumps(tool.resource_usage),
                    json.dumps(tool.dependencies), tool.version,
                    tool.last_updated.isoformat(), tool.is_available
                ))
        except Exception as e:
            logger.error(f"Failed to store tool in database: {e}")
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID."""
        return self.tools.get(tool_id)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools by category."""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def get_local_tools(self) -> List[Tool]:
        """Get all local tools."""
        return [tool for tool in self.tools.values() if tool.is_local]
    
    def get_mcp_tools(self) -> List[Tool]:
        """Get all MCP tools."""
        return [tool for tool in self.tools.values() if tool.tool_type == ToolType.MCP]

class PerformanceTracker:
    """Track and analyze tool performance."""
    
    def __init__(self):
        self.performance_db_path = Path("knowledge-objects/tool_performance.db")
        self.performance_db_path.parent.mkdir(exist_ok=True)
        self._init_performance_db()
    
    def _init_performance_db(self):
        """Initialize performance database."""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tool_id TEXT NOT NULL,
                        execution_time REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        resource_usage TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_id ON execution_logs(tool_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON execution_logs(timestamp)")
        except Exception as e:
            logger.warning(f"Performance database initialization failed: {e}")
    
    async def record_execution(self, tool_id: str, execution_time: float, 
                              success: bool, error_message: Optional[str] = None,
                              resource_usage: Optional[Dict[str, Any]] = None):
        """Record tool execution metrics."""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                conn.execute("""
                    INSERT INTO execution_logs 
                    (tool_id, execution_time, success, error_message, resource_usage, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    tool_id, execution_time, success, error_message or "",
                    json.dumps(resource_usage or {}), datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
    
    async def get_tool_performance(self, tool_id: str) -> float:
        """Get tool performance score."""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                # Get recent executions (last 100)
                cursor = conn.execute("""
                    SELECT execution_time, success FROM execution_logs 
                    WHERE tool_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, (tool_id,))
                
                executions = cursor.fetchall()
                if not executions:
                    return 0.5  # Default score for new tools
                
                # Calculate performance score
                total_executions = len(executions)
                successful_executions = sum(1 for _, success in executions if success)
                avg_execution_time = sum(time for time, _ in executions) / total_executions
                
                # Normalize execution time (lower is better)
                time_score = max(0, 1 - (avg_execution_time / 10))  # Assume 10s is "slow"
                
                # Calculate overall score
                success_rate = successful_executions / total_executions
                performance_score = (success_rate * 0.7) + (time_score * 0.3)
                
                return performance_score
        except Exception as e:
            logger.error(f"Failed to get tool performance: {e}")
            return 0.5
    
    async def get_performance_trend(self, tool_id: str) -> str:
        """Get tool performance trend."""
        try:
            with sqlite3.connect(self.performance_db_path) as conn:
                # Get recent vs older performance
                cursor = conn.execute("""
                    SELECT AVG(execution_time) as avg_time, COUNT(*) as count
                    FROM execution_logs 
                    WHERE tool_id = ? 
                    GROUP BY 
                        CASE 
                            WHEN timestamp > datetime('now', '-1 hour') THEN 'recent'
                            ELSE 'older'
                        END
                """, (tool_id,))
                
                results = cursor.fetchall()
                if len(results) < 2:
                    return "stable"
                
                # Compare recent vs older performance
                recent_avg = results[0][0] if results[0][1] > 0 else 0
                older_avg = results[1][0] if results[1][1] > 0 else 0
                
                if recent_avg == 0 or older_avg == 0:
                    return "stable"
                
                improvement = (older_avg - recent_avg) / older_avg
                
                if improvement > 0.1:
                    return "improving"
                elif improvement < -0.1:
                    return "declining"
                else:
                    return "stable"
        except Exception as e:
            logger.error(f"Failed to get performance trend: {e}")
            return "stable"

class ToolSelectionEngine:
    """Intelligent tool selection with local tool prioritization."""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.performance_tracker = PerformanceTracker()
        self.local_tool_boost = 1.5  # 50% boost for local tools
        self.performance_weight = 0.4
        self.local_preference_weight = 0.3
        self.resource_efficiency_weight = 0.2
        self.freshness_weight = 0.1
    
    async def select_optimal_tools(self, task: str, context: Dict[str, Any], 
                                  max_tools: int = 5) -> List[Tool]:
        """Select optimal tools for a given task."""
        # Get available tools
        available_tools = await self._get_available_tools(task, context)
        
        if not available_tools:
            return []
        
        # Score tools based on multiple factors
        scored_tools = []
        for tool in available_tools:
            score = await self._calculate_tool_score(tool, task, context)
            scored_tools.append((tool, score))
        
        # Sort by score and return top tools
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in scored_tools[:max_tools]]
    
    async def _get_available_tools(self, task: str, context: Dict[str, Any]) -> List[Tool]:
        """Get available tools for a task."""
        # Analyze task to determine relevant categories
        relevant_categories = await self._identify_relevant_categories(task)
        
        available_tools = []
        for category in relevant_categories:
            tools = self.tool_registry.get_tools_by_category(category)
            available_tools.extend([tool for tool in tools if tool.is_available])
        
        return available_tools
    
    async def _identify_relevant_categories(self, task: str) -> List[ToolCategory]:
        """Identify relevant tool categories for a task."""
        task_lower = task.lower()
        relevant_categories = []
        
        # Simple keyword-based categorization
        if any(word in task_lower for word in ["threat", "attack", "malware"]):
            relevant_categories.append(ToolCategory.THREAT_INTELLIGENCE)
        
        if any(word in task_lower for word in ["policy", "compliance", "regulation"]):
            relevant_categories.append(ToolCategory.POLICY_ANALYSIS)
            relevant_categories.append(ToolCategory.COMPLIANCE)
        
        if any(word in task_lower for word in ["incident", "breach", "response"]):
            relevant_categories.append(ToolCategory.INCIDENT_RESPONSE)
        
        if any(word in task_lower for word in ["data", "csv", "json", "process"]):
            relevant_categories.append(ToolCategory.DATA_PROCESSING)
        
        if any(word in task_lower for word in ["visual", "chart", "graph", "report"]):
            relevant_categories.append(ToolCategory.VISUALIZATION)
            relevant_categories.append(ToolCategory.REPORTING)
        
        # Always include utility tools
        relevant_categories.append(ToolCategory.UTILITY)
        
        return relevant_categories
    
    async def _calculate_tool_score(self, tool: Tool, task: str, context: Dict[str, Any]) -> float:
        """Calculate comprehensive tool score."""
        base_score = 0.0
        
        # Performance history (40% weight)
        performance = await self.performance_tracker.get_tool_performance(tool.id)
        base_score += performance * self.performance_weight
        
        # Local tool preference (30% weight)
        if tool.is_local:
            local_score = 1.0
        else:
            local_score = 0.5  # MCP tools get lower preference
        base_score += local_score * self.local_preference_weight
        
        # Resource efficiency (20% weight)
        efficiency = await self._calculate_resource_efficiency(tool, context)
        base_score += efficiency * self.resource_efficiency_weight
        
        # Tool freshness (10% weight)
        freshness = await self._calculate_tool_freshness(tool)
        base_score += freshness * self.freshness_weight
        
        return base_score
    
    async def _calculate_resource_efficiency(self, tool: Tool, context: Dict[str, Any]) -> float:
        """Calculate resource efficiency score."""
        # Check if we have resource constraints
        available_memory = context.get("available_memory_mb", 1000)
        available_cpu = context.get("available_cpu_percent", 100)
        
        tool_memory = tool.resource_usage.get("memory_mb", 0)
        tool_cpu = tool.resource_usage.get("cpu_percent", 0)
        
        # Calculate efficiency (lower resource usage = higher efficiency)
        memory_efficiency = max(0, 1 - (tool_memory / available_memory))
        cpu_efficiency = max(0, 1 - (tool_cpu / available_cpu))
        
        return (memory_efficiency + cpu_efficiency) / 2
    
    async def _calculate_tool_freshness(self, tool: Tool) -> float:
        """Calculate tool freshness score."""
        days_since_update = (datetime.now() - tool.last_updated).days
        
        if days_since_update < 7:
            return 1.0  # Very fresh
        elif days_since_update < 30:
            return 0.8  # Fresh
        elif days_since_update < 90:
            return 0.6  # Moderately fresh
        else:
            return 0.4  # Stale
    
    async def get_tool_recommendations(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed tool recommendations for a task."""
        optimal_tools = await self.select_optimal_tools(task, context)
        
        recommendations = {
            "task": task,
            "recommended_tools": [],
            "reasoning": [],
            "performance_insights": []
        }
        
        for i, tool in enumerate(optimal_tools):
            tool_info = {
                "rank": i + 1,
                "tool": tool.to_dict(),
                "score": await self._calculate_tool_score(tool, task, context),
                "why_recommended": await self._explain_recommendation(tool, task, context)
            }
            recommendations["recommended_tools"].append(tool_info)
        
        # Add performance insights
        recommendations["performance_insights"] = await self._get_performance_insights(optimal_tools)
        
        return recommendations
    
    async def _explain_recommendation(self, tool: Tool, task: str, context: Dict[str, Any]) -> str:
        """Explain why a tool was recommended."""
        reasons = []
        
        if tool.is_local:
            reasons.append("Local tool - faster execution and better integration")
        
        performance = await self.performance_tracker.get_tool_performance(tool.id)
        if performance > 0.8:
            reasons.append("High performance track record")
        elif performance < 0.5:
            reasons.append("Lower performance - consider alternatives")
        
        if tool.category.value in task.lower():
            reasons.append("Directly relevant to task category")
        
        return "; ".join(reasons) if reasons else "General purpose tool"
    
    async def _get_performance_insights(self, tools: List[Tool]) -> List[str]:
        """Get performance insights for recommended tools."""
        insights = []
        
        local_tools = [t for t in tools if t.is_local]
        mcp_tools = [t for t in tools if not t.is_local]
        
        if local_tools:
            insights.append(f"Local tools recommended: {len(local_tools)}/{len(tools)} for optimal performance")
        
        if mcp_tools:
            insights.append(f"MCP tools: {len(mcp_tools)} available for specialized tasks")
        
        # Performance trends
        for tool in tools[:3]:  # Top 3 tools
            trend = await self.performance_tracker.get_performance_trend(tool.id)
            if trend == "improving":
                insights.append(f"{tool.name}: Performance improving - good choice")
            elif trend == "declining":
                insights.append(f"{tool.name}: Performance declining - monitor usage")
        
        return insights

# Global tool selection engine instance
tool_selection_engine = ToolSelectionEngine()

# Convenience functions
async def select_tools(task: str, context: Dict[str, Any], max_tools: int = 5) -> List[Tool]:
    """Convenience function for tool selection."""
    return await tool_selection_engine.select_optimal_tools(task, context, max_tools)

async def get_recommendations(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for tool recommendations."""
    return await tool_selection_engine.get_tool_recommendations(task, context)
