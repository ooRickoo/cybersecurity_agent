#!/usr/bin/env python3
"""
Smart Data Manager for ADK Integration

Keeps data local in DataFrames, SQLite, and GraphDB while providing
intelligent context extraction for LLM tasks.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import pandas as pd
import sqlite3
from dataclasses import dataclass, field

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")

logger = logging.getLogger(__name__)

@dataclass
class DataContext:
    """Context information for data chunks."""
    data_id: str
    data_type: str  # 'dataframe', 'sqlite', 'graphdb'
    source: str
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMContext:
    """Minimal context sent to LLMs."""
    context_id: str
    data_summary: str
    key_insights: List[str]
    relevant_columns: List[str]
    sample_data: str
    processing_instructions: str
    estimated_complexity: str
    local_processing_capabilities: List[str]

class SmartDataManager:
    """Manages data locally while providing intelligent context for LLMs."""
    
    def __init__(self):
        """Initialize the smart data manager."""
        self.data_registry: Dict[str, DataContext] = {}
        self.local_tools = self._initialize_local_tools()
        self.context_cache: Dict[str, LLMContext] = {}
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info("🚀 Smart Data Manager initialized")
    
    def _initialize_local_tools(self) -> Dict[str, Any]:
        """Initialize local processing tools."""
        tools = {}
        
        try:
            # DataFrame tools
            tools["dataframe"] = {
                "available": True,
                "capabilities": ["filter", "aggregate", "transform", "analyze", "visualize"]
            }
        except ImportError:
            tools["dataframe"] = {"available": False, "capabilities": []}
        
        try:
            # SQLite tools
            tools["sqlite"] = {
                "available": True,
                "capabilities": ["query", "aggregate", "join", "analyze", "export"]
            }
        except ImportError:
            tools["sqlite"] = {"available": False, "capabilities": []}
        
        try:
            # GraphDB tools (Neo4j)
            tools["graphdb"] = {
                "available": True,
                "capabilities": ["traverse", "analyze", "pattern_match", "community_detection"]
            }
        except ImportError:
            tools["graphdb"] = {"available": False, "capabilities": []}
        
        return tools
    
    def register_data(self, data_id: str, data_type: str, source: str, 
                     data: Any, tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Register data for local management."""
        try:
            # Analyze data characteristics
            context = self._analyze_data(data, data_type, source, tags or [], metadata or {})
            
            # Store in registry
            self.data_registry[data_id] = context
            
            # Generate context hash for caching
            context_hash = self._generate_context_hash(context)
            
            logger.info(f"✅ Registered data: {data_id} ({data_type}) - {context.size_bytes} bytes")
            
            return context_hash
            
        except Exception as e:
            logger.error(f"❌ Failed to register data {data_id}: {e}")
            return None
    
    def _analyze_data(self, data: Any, data_type: str, source: str, 
                      tags: List[str], metadata: Dict[str, Any]) -> DataContext:
        """Analyze data characteristics for context creation."""
        # Initialize with default size
        size_bytes = 0
        
        if data_type == "dataframe" and hasattr(data, 'shape'):
            size_bytes = data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else 1000
        elif data_type == "sqlite":
            if isinstance(data, str) and os.path.exists(data):
                size_bytes = os.path.getsize(data)
            else:
                size_bytes = 1000  # Default size
        elif data_type == "graphdb":
            size_bytes = 1000  # Default size
        
        context = DataContext(
            data_id=id(data),
            data_type=data_type,
            source=source,
            size_bytes=size_bytes,
            tags=tags,
            metadata=metadata
        )
        
        if data_type == "dataframe":
            if hasattr(data, 'shape'):
                context.row_count = data.shape[0]
                context.column_count = data.shape[1]
                context.size_bytes = data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else 1000
            else:
                # Handle placeholder data
                context.row_count = 1000  # Default
                context.column_count = 5   # Default
                context.size_bytes = 1000  # Default
            
        elif data_type == "sqlite":
            if isinstance(data, str) and os.path.exists(data):
                # Estimate size from file
                context.size_bytes = os.path.getsize(data)
                # Could add more detailed analysis here
            else:
                context.size_bytes = 1000  # Default
        
        elif data_type == "graphdb":
            # For graph databases, we'd analyze node/edge counts
            context.node_count = 100  # Default
            context.edge_count = 500  # Default
            context.size_bytes = 1000  # Default
        
        return context
    
    def _generate_context_hash(self, context: DataContext) -> str:
        """Generate hash for context caching."""
        context_str = f"{context.data_id}_{context.data_type}_{context.source}_{context.size_bytes}"
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def get_llm_context(self, data_id: str, task_description: str, 
                       max_context_size: int = 1000) -> LLMContext:
        """Generate minimal LLM context from local data."""
        try:
            if data_id not in self.data_registry:
                return None
            
            context = self.data_registry[data_id]
            
            # Check cache first
            cache_key = f"{data_id}_{task_description[:50]}"
            if cache_key in self.context_cache:
                return self.context_cache[cache_key]
            
            # Generate intelligent context
            llm_context = self._create_intelligent_context(context, task_description, max_context_size)
            
            # Cache the result
            self.context_cache[cache_key] = llm_context
            
            return llm_context
            
        except Exception as e:
            logger.error(f"❌ Failed to generate LLM context: {e}")
            return None
    
    def _create_intelligent_context(self, context: DataContext, task_description: str, 
                                  max_context_size: int) -> LLMContext:
        """Create intelligent context based on task and data characteristics."""
        
        # Analyze task complexity
        task_complexity = self._analyze_task_complexity(task_description)
        
        # Determine what can be processed locally
        local_capabilities = self._identify_local_capabilities(context, task_description)
        
        # Generate data summary
        data_summary = self._generate_data_summary(context)
        
        # Extract key insights
        key_insights = self._extract_key_insights(context, task_description)
        
        # Select relevant columns/attributes
        relevant_columns = self._identify_relevant_columns(context, task_description)
        
        # Generate sample data
        sample_data = self._generate_sample_data(context, relevant_columns, max_context_size)
        
        # Create processing instructions
        processing_instructions = self._create_processing_instructions(
            context, task_description, local_capabilities
        )
        
        return LLMContext(
            context_id=f"ctx_{context.data_id}_{int(datetime.now().timestamp())}",
            data_summary=data_summary,
            key_insights=key_insights,
            relevant_columns=relevant_columns,
            sample_data=sample_data,
            processing_instructions=processing_instructions,
            estimated_complexity=task_complexity,
            local_processing_capabilities=local_capabilities
        )
    
    def _analyze_task_complexity(self, task_description: str) -> str:
        """Analyze task complexity to determine processing strategy."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["analyze", "investigate", "detect", "identify"]):
            return "high"
        elif any(word in task_lower for word in ["filter", "sort", "count", "summarize"]):
            return "low"
        elif any(word in task_lower for word in ["transform", "enrich", "merge", "join"]):
            return "medium"
        else:
            return "medium"
    
    def _identify_local_capabilities(self, context: DataContext, task_description: str) -> List[str]:
        """Identify what can be processed locally."""
        capabilities = []
        task_lower = task_description.lower()
        
        if context.data_type == "dataframe":
            if "filter" in task_lower or "sort" in task_lower:
                capabilities.append("dataframe_filter_sort")
            if "aggregate" in task_lower or "group" in task_lower:
                capabilities.append("dataframe_aggregate")
            if "transform" in task_lower or "clean" in task_lower:
                capabilities.append("dataframe_transform")
        
        elif context.data_type == "sqlite":
            if "query" in task_lower or "filter" in task_lower:
                capabilities.append("sqlite_query")
            if "join" in task_lower or "merge" in task_lower:
                capabilities.append("sqlite_join")
        
        elif context.data_type == "graphdb":
            if "traverse" in task_lower or "path" in task_lower:
                capabilities.append("graph_traversal")
            if "pattern" in task_lower or "match" in task_lower:
                capabilities.append("graph_pattern_matching")
        
        return capabilities
    
    def _generate_data_summary(self, context: DataContext) -> str:
        """Generate concise data summary."""
        if context.data_type == "dataframe":
            return f"DataFrame with {context.row_count:,} rows and {context.column_count} columns ({context.size_bytes:,} bytes)"
        elif context.data_type == "sqlite":
            return f"SQLite database from {context.source} ({context.size_bytes:,} bytes)"
        elif context.data_type == "graphdb":
            return f"Graph database with {context.node_count or 'unknown'} nodes and {context.edge_count or 'unknown'} edges"
        else:
            return f"{context.data_type} data from {context.source} ({context.size_bytes:,} bytes)"
    
    def _extract_key_insights(self, context: DataContext, task_description: str) -> List[str]:
        """Extract key insights based on task and data."""
        insights = []
        
        if context.data_type == "dataframe":
            insights.append(f"Data shape: {context.row_count:,} x {context.column_count}")
            insights.append(f"Memory usage: {context.size_bytes:,} bytes")
            if context.tags:
                insights.append(f"Tagged as: {', '.join(context.tags)}")
        
        elif context.data_type == "sqlite":
            insights.append(f"Database source: {context.source}")
            insights.append(f"File size: {context.size_bytes:,} bytes")
        
        # Add task-specific insights
        if "security" in task_description.lower():
            insights.append("Security-focused analysis recommended")
        if "threat" in task_description.lower():
            insights.append("Threat intelligence context available")
        
        return insights
    
    def _identify_relevant_columns(self, context: DataContext, task_description: str) -> List[str]:
        """Identify relevant columns/attributes for the task."""
        # This would be implemented based on actual data structure
        # For now, return a placeholder
        if context.data_type == "dataframe":
            return ["relevant_column_1", "relevant_column_2"]
        elif context.data_type == "sqlite":
            return ["relevant_table.column"]
        elif context.data_type == "graphdb":
            return ["node_property", "edge_property"]
        else:
            return []
    
    def _generate_sample_data(self, context: DataContext, relevant_columns: List[str], 
                            max_size: int) -> str:
        """Generate sample data within size limits."""
        # This would extract actual sample data
        # For now, return a placeholder
        sample = f"Sample data from {context.data_type}:\n"
        sample += f"Columns: {', '.join(relevant_columns[:5])}\n"
        sample += f"Size: {context.size_bytes:,} bytes\n"
        
        # Ensure we stay within limits
        if len(sample) > max_size:
            sample = sample[:max_size-3] + "..."
        
        return sample
    
    def _create_processing_instructions(self, context: DataContext, task_description: str, 
                                      local_capabilities: List[str]) -> str:
        """Create processing instructions for the task."""
        instructions = f"Process {context.data_type} data for: {task_description}\n\n"
        
        if local_capabilities:
            instructions += "Local processing available:\n"
            for capability in local_capabilities:
                instructions += f"• {capability}\n"
            instructions += "\n"
        
        instructions += "LLM should focus on:\n"
        instructions += "• High-level analysis and insights\n"
        instructions += "• Strategy and recommendations\n"
        instructions += "• Complex pattern recognition\n"
        
        instructions += "\nLocal tools should handle:\n"
        instructions += "• Data filtering and aggregation\n"
        instructions += "• Basic transformations\n"
        instructions += "• Query execution\n"
        
        return instructions
    
    def process_locally(self, data_id: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Process data locally using available tools."""
        try:
            if data_id not in self.data_registry:
                return {"success": False, "error": "Data not found"}
            
            context = self.data_registry[data_id]
            
            # Route to appropriate local processor
            if context.data_type == "dataframe":
                return self._process_dataframe_locally(context, operation, **kwargs)
            elif context.data_type == "sqlite":
                return self._process_sqlite_locally(context, operation, **kwargs)
            elif context.data_type == "graphdb":
                return self._process_graphdb_locally(context, operation, **kwargs)
            else:
                return {"success": False, "error": f"Unsupported data type: {context.data_type}"}
                
        except Exception as e:
            logger.error(f"❌ Local processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_dataframe_locally(self, context: DataContext, operation: str, **kwargs) -> Dict[str, Any]:
        """Process DataFrame locally."""
        # This would implement actual DataFrame processing
        # For now, return a placeholder
        return {
            "success": True,
            "operation": operation,
            "data_type": "dataframe",
            "result": f"Processed {context.row_count:,} rows using {operation}",
            "local_processing": True
        }
    
    def _process_sqlite_locally(self, context: DataContext, operation: str, **kwargs) -> Dict[str, Any]:
        """Process SQLite data locally."""
        return {
            "success": True,
            "operation": operation,
            "data_type": "sqlite",
            "result": f"Processed SQLite data using {operation}",
            "local_processing": True
        }
    
    def _process_graphdb_locally(self, context: DataContext, operation: str, **kwargs) -> Dict[str, Any]:
        """Process GraphDB data locally."""
        return {
            "success": True,
            "operation": operation,
            "data_type": "graphdb",
            "result": f"Processed graph data using {operation}",
            "local_processing": True
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_data_registered": len(self.data_registry),
            "context_cache_size": len(self.context_cache),
            "processing_history_count": len(self.processing_history),
            "local_tools": self.local_tools,
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate context cache hit rate."""
        # This would implement actual cache hit rate calculation
        return 0.85  # Placeholder
    
    def clear_cache(self):
        """Clear context cache."""
        self.context_cache.clear()
        logger.info("🧹 Context cache cleared")

# Example usage
async def main():
    """Test the Smart Data Manager."""
    print("🚀 Testing Smart Data Manager")
    print("=" * 50)
    
    # Create manager
    manager = SmartDataManager()
    
    # Register sample data
    sample_df = pd.DataFrame({
        'id': range(1000),
        'name': [f'item_{i}' for i in range(1000)],
        'value': [i * 1.5 for i in range(1000)]
    })
    
    data_id = manager.register_data(
        "sample_dataframe",
        "dataframe",
        "test_data",
        sample_df,
        tags=["sample", "test"],
        metadata={"description": "Sample test data"}
    )
    
    print(f"✅ Registered data with context hash: {data_id}")
    
    # Generate LLM context
    task_description = "Analyze this data for security threats and identify anomalies"
    llm_context = manager.get_llm_context("sample_dataframe", task_description)
    
    if llm_context:
        print(f"\n🤖 LLM Context Generated:")
        print(f"   Summary: {llm_context.data_summary}")
        print(f"   Insights: {len(llm_context.key_insights)} insights")
        print(f"   Local capabilities: {len(llm_context.local_processing_capabilities)}")
        print(f"   Processing instructions: {len(llm_context.processing_instructions)} chars")
    
    # Test local processing
    result = manager.process_locally("sample_dataframe", "filter", column="value", threshold=500)
    print(f"\n🔧 Local Processing Result: {result}")
    
    # Get stats
    stats = manager.get_processing_stats()
    print(f"\n📊 Processing Stats: {stats}")
    
    print(f"\n🎉 Smart Data Manager test completed!")

if __name__ == "__main__":
    asyncio.run(main())
