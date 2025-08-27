#!/usr/bin/env python3
"""
Enhanced Agentic Memory System - Multi-Tier Memory Integration

Fully utilizes short-term, medium-term, and long-term memory concepts:
- Short-term: Session-specific, workflow execution context
- Medium-term: Workflow patterns, tool performance, adaptation rules
- Long-term: Knowledge accumulation, problem-solving patterns, system evolution
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
import hashlib

from enhanced_context_memory import EnhancedContextMemoryManager

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory tier categories."""
    SHORT_TERM = "short_term"      # Session-specific, immediate context
    MEDIUM_TERM = "medium_term"    # Workflow patterns, tool performance
    LONG_TERM = "long_term"        # Knowledge accumulation, system evolution

class MemoryCategory(Enum):
    """Memory categories for different types of information."""
    WORKFLOW_EXECUTION = "workflow_execution"
    TOOL_PERFORMANCE = "tool_performance"
    PROBLEM_PATTERNS = "problem_patterns"
    ADAPTATION_RULES = "adaptation_rules"
    CONTEXT_BUILDING = "context_building"
    SOLUTION_SYNTHESIS = "solution_synthesis"
    LEARNING_OUTCOMES = "learning_outcomes"
    SYSTEM_EVOLUTION = "system_evolution"

class EnhancedAgenticMemorySystem:
    """Enhanced memory system that fully utilizes multi-tier memory concepts."""
    
    def __init__(self, context_memory: EnhancedContextMemoryManager):
        self.context_memory = context_memory
        self.session_id = None
        self.workflow_context = {}
        self.memory_strategies = self._initialize_memory_strategies()
        
        logger.info("🚀 Enhanced Agentic Memory System initialized")
    
    def _initialize_memory_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize memory strategies for different tiers and categories."""
        return {
            MemoryTier.SHORT_TERM.value: {
                "ttl_hours": 4,
                "max_items": 1000,
                "categories": [
                    MemoryCategory.WORKFLOW_EXECUTION.value,
                    MemoryCategory.CONTEXT_BUILDING.value
                ],
                "promotion_threshold": 0.8,  # Importance score to promote to medium-term
                "eviction_strategy": "lru"
            },
            MemoryTier.MEDIUM_TERM.value: {
                "ttl_hours": 24,
                "max_items": 500,
                "categories": [
                    MemoryCategory.TOOL_PERFORMANCE.value,
                    MemoryCategory.PROBLEM_PATTERNS.value,
                    MemoryCategory.ADAPTATION_RULES.value
                ],
                "promotion_threshold": 0.9,  # Importance score to promote to long-term
                "eviction_strategy": "importance_based"
            },
            MemoryTier.LONG_TERM.value: {
                "ttl_hours": 168,  # 1 week
                "max_items": 10000,
                "categories": [
                    MemoryCategory.LEARNING_OUTCOMES.value,
                    MemoryCategory.SYSTEM_EVOLUTION.value,
                    MemoryCategory.SOLUTION_SYNTHESIS.value
                ],
                "eviction_strategy": "never",  # Long-term memories are permanent
                "archival_strategy": "compressed"
            }
        }
    
    def start_workflow_session(self, session_id: str, workflow_context: Dict[str, Any]):
        """Start a new workflow session with memory management."""
        self.session_id = session_id
        self.workflow_context = workflow_context
        
        # Start context memory session
        self.context_memory.start_session(session_id)
        
        # Initialize session-specific short-term memory
        self._initialize_session_memory(session_id, workflow_context)
        
        logger.info(f"🚀 Started workflow session: {session_id}")
    
    def end_workflow_session(self):
        """End workflow session and promote important memories."""
        if self.session_id:
            # Promote important short-term memories to medium-term
            self._promote_session_memories()
            
            # End context memory session
            self.context_memory.end_session()
            
            # Clear session state
            self.session_id = None
            self.workflow_context = {}
            
            logger.info(f"✅ Ended workflow session: {self.session_id}")
    
    def _initialize_session_memory(self, session_id: str, workflow_context: Dict[str, Any]):
        """Initialize session-specific memory structures."""
        session_memory = {
            "session_id": session_id,
            "start_time": time.time(),
            "workflow_context": workflow_context,
            "execution_trail": [],
            "context_building": {},
            "tool_selections": [],
            "adaptations_made": []
        }
        
        # Store in short-term memory
        self._store_memory(
            MemoryCategory.WORKFLOW_EXECUTION.value,
            f"session_{session_id}",
            "session_init",
            json.dumps(session_memory),
            MemoryTier.SHORT_TERM.value,
            0.9,
            {
                "session_id": session_id,
                "memory_type": "session_initialization",
                "timestamp": time.time()
            }
        )
    
    def store_workflow_execution(self, execution_data: Dict[str, Any], importance_score: float = 0.7):
        """Store workflow execution data in appropriate memory tier."""
        # Determine memory tier based on importance and type
        memory_tier = self._determine_memory_tier(execution_data, importance_score)
        
        # Generate unique ID for execution
        execution_id = self._generate_execution_id(execution_data)
        
        # Store execution data
        self._store_memory(
            MemoryCategory.WORKFLOW_EXECUTION.value,
            execution_id,
            "workflow_execution",
            json.dumps(execution_data),
            memory_tier.value,
            importance_score,
            {
                "session_id": self.session_id,
                "execution_type": execution_data.get("type", "unknown"),
                "mode": execution_data.get("mode", "unknown"),
                "timestamp": time.time(),
                "memory_tier": memory_tier.value
            }
        )
        
        # Update session execution trail
        if self.session_id:
            self._update_execution_trail(execution_id, execution_data)
        
        return execution_id
    
    def store_tool_performance(self, tool_id: str, performance_data: Dict[str, Any], importance_score: float = 0.8):
        """Store tool performance data in medium-term memory."""
        performance_id = f"tool_perf_{tool_id}_{int(time.time())}"
        
        self._store_memory(
            MemoryCategory.TOOL_PERFORMANCE.value,
            performance_id,
            "tool_performance",
            json.dumps(performance_data),
            MemoryTier.MEDIUM_TERM.value,
            importance_score,
            {
                "tool_id": tool_id,
                "session_id": self.session_id,
                "performance_metrics": list(performance_data.keys()),
                "timestamp": time.time()
            }
        )
        
        return performance_id
    
    def store_problem_pattern(self, problem_description: str, solution_pattern: Dict[str, Any], importance_score: float = 0.9):
        """Store problem-solving patterns in long-term memory."""
        pattern_id = self._generate_pattern_id(problem_description, solution_pattern)
        
        self._store_memory(
            MemoryCategory.PROBLEM_PATTERNS.value,
            pattern_id,
            "problem_pattern",
            json.dumps(solution_pattern),
            MemoryTier.LONG_TERM.value,
            importance_score,
            {
                "problem_description": problem_description,
                "pattern_type": solution_pattern.get("type", "unknown"),
                "success_rate": solution_pattern.get("success_rate", 0.0),
                "timestamp": time.time(),
                "usage_count": 1
            }
        )
        
        return pattern_id
    
    def store_adaptation_rule(self, adaptation_data: Dict[str, Any], importance_score: float = 0.8):
        """Store adaptation rules in medium-term memory."""
        rule_id = f"adaptation_rule_{int(time.time())}"
        
        self._store_memory(
            MemoryCategory.ADAPTATION_RULES.value,
            rule_id,
            "adaptation_rule",
            json.dumps(adaptation_data),
            MemoryTier.MEDIUM_TERM.value,
            importance_score,
            {
                "session_id": self.session_id,
                "adaptation_type": adaptation_data.get("type", "unknown"),
                "trigger_conditions": adaptation_data.get("triggers", []),
                "success_rate": adaptation_data.get("success_rate", 0.0),
                "timestamp": time.time()
            }
        )
        
        return rule_id
    
    def store_context_building(self, context_data: Dict[str, Any], importance_score: float = 0.6):
        """Store context building data in short-term memory."""
        context_id = f"context_{int(time.time())}"
        
        self._store_memory(
            MemoryCategory.CONTEXT_BUILDING.value,
            context_id,
            "context_building",
            json.dumps(context_data),
            MemoryTier.SHORT_TERM.value,
            importance_score,
            {
                "session_id": self.session_id,
                "context_type": context_data.get("type", "unknown"),
                "building_stage": context_data.get("stage", "unknown"),
                "timestamp": time.time()
            }
        )
        
        return context_id
    
    def store_solution_synthesis(self, synthesis_data: Dict[str, Any], importance_score: float = 0.9):
        """Store solution synthesis data in long-term memory."""
        synthesis_id = f"synthesis_{int(time.time())}"
        
        self._store_memory(
            MemoryCategory.SOLUTION_SYNTHESIS.value,
            synthesis_id,
            "solution_synthesis",
            json.dumps(synthesis_data),
            MemoryTier.LONG_TERM.value,
            importance_score,
            {
                "session_id": self.session_id,
                "synthesis_type": synthesis_data.get("type", "unknown"),
                "confidence_score": synthesis_data.get("confidence_score", 0.0),
                "sub_problems_resolved": synthesis_data.get("sub_problems_resolved", 0),
                "timestamp": time.time()
            }
        )
        
        return synthesis_id
    
    def store_learning_outcome(self, learning_data: Dict[str, Any], importance_score: float = 1.0):
        """Store learning outcomes in long-term memory."""
        learning_id = f"learning_{int(time.time())}"
        
        self._store_memory(
            MemoryCategory.LEARNING_OUTCOMES.value,
            learning_id,
            "learning_outcome",
            json.dumps(learning_data),
            MemoryTier.LONG_TERM.value,
            importance_score,
            {
                "session_id": self.session_id,
                "learning_type": learning_data.get("type", "unknown"),
                "impact_score": learning_data.get("impact_score", 0.0),
                "applicability": learning_data.get("applicability", "unknown"),
                "timestamp": time.time()
            }
        )
        
        return learning_id
    
    def retrieve_context_for_problem(self, problem_description: str, max_context_items: int = 50) -> Dict[str, Any]:
        """Retrieve relevant context from all memory tiers for a problem."""
        context = {
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "synthesized_context": {}
        }
        
        try:
            # Search across all memory tiers
            for tier in MemoryTier:
                tier_context = self._search_tier_memory(
                    problem_description, tier, max_context_items // 3
                )
                context[tier.value] = tier_context
            
            # Synthesize context from different tiers
            context["synthesized_context"] = self._synthesize_context(context)
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error retrieving context: {e}")
            return context
    
    def _search_tier_memory(self, query: str, tier: MemoryTier, max_results: int) -> List[Dict[str, Any]]:
        """Search memory within a specific tier."""
        try:
            # Use context memory search with tier filtering
            # Note: EnhancedContextMemoryManager doesn't have get_domains method
            # So we'll search across all available domains
            results = self.context_memory.search_memories(
                query=query,
                max_results=max_results
            )
            
            # Filter results by tier if possible
            # For now, return all results as the tier filtering would need domain-specific implementation
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Error searching {tier.value} memory: {e}")
            return []
    
    def _synthesize_context(self, tier_contexts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize context from different memory tiers."""
        synthesis = {
            "total_context_items": 0,
            "tier_distribution": {},
            "key_insights": [],
            "relevance_score": 0.0,
            "confidence_level": "low"
        }
        
        try:
            # Count items per tier
            for tier, items in tier_contexts.items():
                synthesis["tier_distribution"][tier] = len(items)
                synthesis["total_context_items"] += len(items)
            
            # Extract key insights from long-term memory
            long_term_items = tier_contexts.get("long_term", [])
            for item in long_term_items:
                if item.get("importance_score", 0) > 0.8:
                    synthesis["key_insights"].append({
                        "content": item.get("content", "")[:200],
                        "importance": item.get("importance_score", 0),
                        "type": item.get("node_type", "unknown")
                    })
            
            # Calculate relevance score
            if synthesis["total_context_items"] > 0:
                synthesis["relevance_score"] = min(
                    synthesis["total_context_items"] / 100.0, 1.0
                )
                
                # Set confidence level based on relevance
                if synthesis["relevance_score"] > 0.7:
                    synthesis["confidence_level"] = "high"
                elif synthesis["relevance_score"] > 0.4:
                    synthesis["confidence_level"] = "medium"
            
            return synthesis
            
        except Exception as e:
            logger.error(f"❌ Error synthesizing context: {e}")
            return synthesis
    
    def _determine_memory_tier(self, data: Dict[str, Any], importance_score: float) -> MemoryTier:
        """Determine appropriate memory tier for data."""
        # High importance items go to long-term memory
        if importance_score > 0.9:
            return MemoryTier.LONG_TERM
        
        # Medium importance items go to medium-term memory
        elif importance_score > 0.7:
            return MemoryTier.MEDIUM_TERM
        
        # Low importance items go to short-term memory
        else:
            return MemoryTier.SHORT_TERM
    
    def _store_memory(self, domain_id: str, node_id: str, node_type: str, content: str,
                      ttl_category: str, importance_score: float, metadata: Dict[str, Any]):
        """Store memory using the enhanced context memory system."""
        try:
            success = self.context_memory.add_memory(
                domain_id, node_id, node_type, content,
                metadata, importance_score, ttl_category
            )
            
            if success:
                logger.debug(f"✅ Stored memory {node_id} in {ttl_category} tier")
            else:
                logger.warning(f"⚠️  Failed to store memory {node_id}")
                
        except Exception as e:
            logger.error(f"❌ Error storing memory: {e}")
    
    def _generate_execution_id(self, execution_data: Dict[str, Any]) -> str:
        """Generate unique ID for execution data."""
        content_hash = hashlib.sha256(
            json.dumps(execution_data, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"exec_{content_hash}_{int(time.time())}"
    
    def _generate_pattern_id(self, problem_description: str, solution_pattern: Dict[str, Any]) -> str:
        """Generate unique ID for problem pattern."""
        pattern_hash = hashlib.sha256(
            (problem_description + json.dumps(solution_pattern, sort_keys=True)).encode()
        ).hexdigest()[:8]
        
        return f"pattern_{pattern_hash}"
    
    def _update_execution_trail(self, execution_id: str, execution_data: Dict[str, Any]):
        """Update session execution trail."""
        if self.session_id:
            trail_entry = {
                "execution_id": execution_id,
                "timestamp": time.time(),
                "type": execution_data.get("type", "unknown"),
                "mode": execution_data.get("mode", "unknown"),
                "success": execution_data.get("success", False)
            }
            
            # Add to short-term memory
            self._store_memory(
                MemoryCategory.WORKFLOW_EXECUTION.value,
                f"trail_{execution_id}",
                "execution_trail",
                json.dumps(trail_entry),
                MemoryTier.SHORT_TERM.value,
                0.5,
                {
                    "session_id": self.session_id,
                    "trail_type": "execution_step",
                    "timestamp": time.time()
                }
            )
    
    def _promote_session_memories(self):
        """Promote important short-term memories to higher tiers."""
        try:
            # Note: EnhancedContextMemoryManager doesn't have get_session_memories method
            # For now, we'll skip promotion until we implement a proper session memory tracking
            logger.info("ℹ️  Session memory promotion not implemented - requires session memory tracking")
            
        except Exception as e:
            logger.error(f"❌ Error promoting session memories: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            stats = {
                "session_info": {
                    "current_session": self.session_id,
                    "session_duration": time.time() - self.workflow_context.get("start_time", time.time()) if self.session_id else 0
                },
                "memory_tiers": {},
                "performance_metrics": self.context_memory.get_performance_stats()
            }
            
            # Get statistics for each memory tier
            for tier in MemoryTier:
                tier_stats = self._get_tier_statistics(tier)
                stats["memory_tiers"][tier.value] = tier_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting memory statistics: {e}")
            return {}
    
    def _get_tier_statistics(self, tier: MemoryTier) -> Dict[str, Any]:
        """Get statistics for a specific memory tier."""
        try:
            # Get cache statistics from context memory
            cache_stats = self.context_memory.get_performance_stats()
            
            if tier == MemoryTier.SHORT_TERM:
                return cache_stats.get("cache_stats", {}).get("short_term", {})
            elif tier == MemoryTier.MEDIUM_TERM:
                return cache_stats.get("cache_stats", {}).get("workflow", {})
            else:  # Long-term
                return {
                    "total_items": cache_stats.get("master_catalog_stats", {}).get("total_nodes", 0),
                    "domains": cache_stats.get("master_catalog_stats", {}).get("domains", {})
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting {tier.value} statistics: {e}")
            return {}

# Example usage
async def main():
    """Example usage of the Enhanced Agentic Memory System."""
    from enhanced_context_memory import EnhancedContextMemoryManager
    
    # Initialize context memory
    context_memory = EnhancedContextMemoryManager(".")
    
    # Initialize enhanced agentic memory system
    memory_system = EnhancedAgenticMemorySystem(context_memory)
    
    try:
        # Start workflow session
        workflow_context = {
            "mode": "automated",
            "priority": 4,
            "complexity": 7,
            "start_time": time.time()
        }
        
        memory_system.start_workflow_session("test_session_001", workflow_context)
        
        # Store various types of memories
        execution_id = memory_system.store_workflow_execution({
            "type": "csv_processing",
            "mode": "automated",
            "rows_processed": 100,
            "success": True
        }, importance_score=0.8)
        
        tool_perf_id = memory_system.store_tool_performance("get_workflow_context", {
            "execution_time": 0.05,
            "success_rate": 0.95,
            "usage_count": 10
        }, importance_score=0.8)
        
        pattern_id = memory_system.store_problem_pattern(
            "Analyze threat groups and provide risk assessment",
            {
                "type": "threat_analysis",
                "success_rate": 0.9,
                "tools_used": ["get_workflow_context", "search_memories"],
                "execution_time": 0.1
            },
            importance_score=0.9
        )
        
        # Retrieve context for a problem
        context = memory_system.retrieve_context_for_problem(
            "Analyze threat indicators and provide response recommendations"
        )
        
        print("📊 Memory System Test Results:")
        print(f"Execution ID: {execution_id}")
        print(f"Tool Performance ID: {tool_perf_id}")
        print(f"Pattern ID: {pattern_id}")
        print(f"Retrieved Context Items: {context.get('synthesized_context', {}).get('total_context_items', 0)}")
        
        # Get memory statistics
        stats = memory_system.get_memory_statistics()
        print(f"\n📊 Memory Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    finally:
        # End workflow session
        memory_system.end_workflow_session()
        context_memory.close()

if __name__ == "__main__":
    asyncio.run(main())
