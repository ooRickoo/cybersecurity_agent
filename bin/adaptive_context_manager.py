#!/usr/bin/env python3
"""
Real-Time Context Adaptation System for Cybersecurity Agent
Enhances Knowledge Graph Context Memory with dynamic adaptation during execution
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
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Context type enumeration."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    EMERGENT = "emergent"

class ContextPriority(Enum):
    """Context priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ContextElement:
    """Individual context element."""
    element_id: str
    name: str
    content: Any
    context_type: ContextType
    priority: ContextPriority
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    relevance_score: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['context_type'] = self.context_type.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

@dataclass
class ContextGap:
    """Identified context gap."""
    gap_id: str
    description: str
    impact_level: str
    suggested_sources: List[str]
    estimated_fill_time: float
    priority: ContextPriority
    detected_at: datetime

@dataclass
class ContextAdaptation:
    """Context adaptation action."""
    adaptation_id: str
    action_type: str
    description: str
    context_elements: List[str]
    reasoning: str
    confidence: float
    executed_at: datetime

class ContextAnalyzer:
    """Analyze context for gaps and optimization opportunities."""
    
    def __init__(self):
        self.analysis_patterns = self._load_analysis_patterns()
        self.gap_detectors = self._load_gap_detectors()
    
    def _load_analysis_patterns(self) -> Dict[str, Any]:
        """Load context analysis patterns."""
        return {
            "completeness": {
                "threat_analysis": ["threat_indicators", "attack_vectors", "vulnerability_data"],
                "incident_response": ["incident_details", "affected_systems", "timeline"],
                "compliance": ["policy_requirements", "control_measures", "audit_findings"]
            },
            "relevance": {
                "time_decay": 0.1,  # Relevance decreases by 10% per day
                "access_patterns": True,
                "semantic_similarity": True
            }
        }
    
    def _load_gap_detectors(self) -> List[Callable]:
        """Load gap detection functions."""
        return [
            self._detect_missing_entities,
            self._detect_incomplete_relationships,
            self._detect_outdated_information,
            self._detect_insufficient_coverage
        ]
    
    async def analyze_context(self, context: Dict[str, Any], 
                            execution_step: str) -> Dict[str, Any]:
        """Analyze context for gaps and optimization opportunities."""
        analysis = {
            "completeness_score": 0.0,
            "relevance_score": 0.0,
            "gaps_detected": [],
            "optimization_opportunities": [],
            "context_health": "good"
        }
        
        # Analyze completeness
        completeness_score = await self._analyze_completeness(context, execution_step)
        analysis["completeness_score"] = completeness_score
        
        # Analyze relevance
        relevance_score = await self._analyze_relevance(context)
        analysis["relevance_score"] = relevance_score
        
        # Detect gaps
        gaps = await self._detect_context_gaps(context, execution_step)
        analysis["gaps_detected"] = gaps
        
        # Generate optimization opportunities
        opportunities = await self._generate_optimization_opportunities(context, analysis)
        analysis["optimization_opportunities"] = opportunities
        
        # Determine overall health
        overall_score = (completeness_score + relevance_score) / 2
        if overall_score < 0.5:
            analysis["context_health"] = "poor"
        elif overall_score < 0.8:
            analysis["context_health"] = "fair"
        else:
            analysis["context_health"] = "good"
        
        return analysis
    
    async def _analyze_completeness(self, context: Dict[str, Any], execution_step: str) -> float:
        """Analyze context completeness for a specific execution step."""
        required_elements = self.analysis_patterns["completeness"].get(execution_step, [])
        
        if not required_elements:
            return 1.0  # No specific requirements
        
        present_elements = 0
        for element in required_elements:
            if element in context:
                present_elements += 1
        
        return present_elements / len(required_elements)
    
    async def _analyze_relevance(self, context: Dict[str, Any]) -> float:
        """Analyze context relevance."""
        relevance_scores = []
        
        for key, value in context.items():
            if isinstance(value, dict) and "created_at" in value:
                # Calculate time-based relevance
                created_at = datetime.fromisoformat(value["created_at"])
                days_old = (datetime.now() - created_at).days
                time_relevance = max(0, 1 - (days_old * self.analysis_patterns["relevance"]["time_decay"]))
                relevance_scores.append(time_relevance)
            else:
                # Default relevance for non-temporal elements
                relevance_scores.append(0.8)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    async def _detect_context_gaps(self, context: Dict[str, Any], execution_step: str) -> List[ContextGap]:
        """Detect context gaps using multiple detection methods."""
        gaps = []
        
        for detector in self.gap_detectors:
            try:
                detected_gaps = await detector(context, execution_step)
                gaps.extend(detected_gaps)
            except Exception as e:
                logger.error(f"Gap detection failed: {e}")
        
        return gaps
    
    async def _detect_missing_entities(self, context: Dict[str, Any], execution_step: str) -> List[ContextGap]:
        """Detect missing entities in context."""
        gaps = []
        
        # Check for common missing entities
        common_entities = {
            "threat_analysis": ["threat_actors", "attack_vectors", "vulnerabilities"],
            "incident_response": ["affected_systems", "timeline", "evidence"],
            "compliance": ["policies", "controls", "standards"]
        }
        
        required_entities = common_entities.get(execution_step, [])
        for entity in required_entities:
            if entity not in context:
                gap = ContextGap(
                    gap_id=f"missing_{entity}_{hashlib.sha256(entity.encode()).hexdigest()[:8]}",
                    description=f"Missing {entity} information",
                    impact_level="medium",
                    suggested_sources=[f"query_{entity}", f"search_{entity}"],
                    estimated_fill_time=5.0,
                    priority=ContextPriority.MEDIUM,
                    detected_at=datetime.now()
                )
                gaps.append(gap)
        
        return gaps
    
    async def _detect_incomplete_relationships(self, context: Dict[str, Any], execution_step: str) -> List[ContextGap]:
        """Detect incomplete relationships in context."""
        gaps = []
        
        # This is a simplified implementation
        # In practice, you'd analyze graph relationships and identify missing connections
        
        return gaps
    
    async def _detect_outdated_information(self, context: Dict[str, Any], execution_step: str) -> List[ContextGap]:
        """Detect outdated information in context."""
        gaps = []
        
        for key, value in context.items():
            if isinstance(value, dict) and "created_at" in value:
                created_at = datetime.fromisoformat(value["created_at"])
                days_old = (datetime.now() - created_at).days
                
                if days_old > 30:  # Consider information older than 30 days as potentially outdated
                    gap = ContextGap(
                        gap_id=f"outdated_{key}_{hashlib.sha256(key.encode()).hexdigest()[:8]}",
                        description=f"Information for {key} may be outdated ({days_old} days old)",
                        impact_level="low",
                        suggested_sources=[f"refresh_{key}", f"update_{key}"],
                        estimated_fill_time=2.0,
                        priority=ContextPriority.LOW,
                        detected_at=datetime.now()
                    )
                    gaps.append(gap)
        
        return gaps
    
    async def _detect_insufficient_coverage(self, context: Dict[str, Any], execution_step: str) -> List[ContextGap]:
        """Detect insufficient coverage in context."""
        gaps = []
        
        # Check if context has sufficient depth
        if len(context) < 5:  # Arbitrary threshold
            gap = ContextGap(
                                        gap_id=f"insufficient_coverage_{hashlib.sha256(execution_step.encode()).hexdigest()[:8]}",
                description=f"Insufficient context coverage for {execution_step}",
                impact_level="high",
                suggested_sources=["expand_context", "gather_more_data"],
                estimated_fill_time=10.0,
                priority=ContextPriority.HIGH,
                detected_at=datetime.now()
            )
            gaps.append(gap)
        
        return gaps
    
    async def _generate_optimization_opportunities(self, context: Dict[str, Any], 
                                                 analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization opportunities based on analysis."""
        opportunities = []
        
        if analysis["completeness_score"] < 0.8:
            opportunities.append("Context completeness can be improved by filling identified gaps")
        
        if analysis["relevance_score"] < 0.7:
            opportunities.append("Context relevance can be improved by updating outdated information")
        
        if len(analysis["gaps_detected"]) > 3:
            opportunities.append("Multiple context gaps detected - consider comprehensive context refresh")
        
        if analysis["context_health"] == "poor":
            opportunities.append("Context health is poor - immediate attention required")
        
        return opportunities

class AdaptationEngine:
    """Generate and execute context adaptations."""
    
    def __init__(self):
        self.adaptation_strategies = self._load_adaptation_strategies()
        self.adaptation_history: List[ContextAdaptation] = []
    
    def _load_adaptation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load context adaptation strategies."""
        return {
            "gap_filling": {
                "priority": "high",
                "methods": ["query_knowledge_base", "search_external_sources", "infer_from_context"],
                "estimated_time": 5.0
            },
            "relevance_improvement": {
                "priority": "medium",
                "methods": ["update_outdated_info", "remove_irrelevant_data", "recalculate_scores"],
                "estimated_time": 3.0
            },
            "completeness_enhancement": {
                "priority": "medium",
                "methods": ["expand_relationships", "add_missing_entities", "deepen_coverage"],
                "estimated_time": 8.0
            }
        }
    
    async def generate_adaptations(self, context_gaps: List[ContextGap], 
                                 execution_state: Dict[str, Any]) -> List[ContextAdaptation]:
        """Generate context adaptations based on identified gaps."""
        adaptations = []
        
        for gap in context_gaps:
            # Determine adaptation strategy based on gap type and priority
            strategy = self._select_adaptation_strategy(gap)
            
            # Generate adaptation actions
            actions = await self._generate_adaptation_actions(gap, strategy)
            
            for action in actions:
                adaptation = ContextAdaptation(
                                            adaptation_id=f"adapt_{gap.gap_id}_{hashlib.sha256(action.encode()).hexdigest()[:8]}",
                    action_type=action,
                    description=f"Adapt context to address: {gap.description}",
                    context_elements=[gap.gap_id],
                    reasoning=f"Gap detected: {gap.description}",
                    confidence=0.8,
                    executed_at=datetime.now()
                )
                adaptations.append(adaptation)
        
        return adaptations
    
    def _select_adaptation_strategy(self, gap: ContextGap) -> str:
        """Select appropriate adaptation strategy for a gap."""
        if gap.priority == ContextPriority.CRITICAL:
            return "gap_filling"
        elif gap.priority == ContextPriority.HIGH:
            return "gap_filling"
        elif gap.priority == ContextPriority.MEDIUM:
            return "completeness_enhancement"
        else:
            return "relevance_improvement"
    
    async def _generate_adaptation_actions(self, gap: ContextGap, strategy: str) -> List[str]:
        """Generate specific adaptation actions for a gap."""
        strategy_config = self.adaptation_strategies.get(strategy, {})
        methods = strategy_config.get("methods", [])
        
        actions = []
        for method in methods:
            if method == "query_knowledge_base":
                actions.append(f"query_kb_for_{gap.description.lower().replace(' ', '_')}")
            elif method == "search_external_sources":
                actions.append(f"search_external_for_{gap.description.lower().replace(' ', '_')}")
            elif method == "infer_from_context":
                actions.append(f"infer_{gap.description.lower().replace(' ', '_')}_from_context")
            else:
                actions.append(method)
        
        return actions

class AdaptiveContextManager:
    """Main manager for real-time context adaptation."""
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.adaptation_engine = AdaptationEngine()
        self.context_cache = {}
        self.adaptation_history: List[ContextAdaptation] = []
        self.context_db_path = Path("knowledge-objects/adaptive_context.db")
        self.context_db_path.parent.mkdir(exist_ok=True)
        self._init_context_db()
    
    def _init_context_db(self):
        """Initialize context database."""
        try:
            with sqlite3.connect(self.context_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_elements (
                        element_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        content TEXT,
                        context_type TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        relevance_score REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.0
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_adaptations (
                        adaptation_id TEXT PRIMARY KEY,
                        action_type TEXT NOT NULL,
                        description TEXT,
                        context_elements TEXT,
                        reasoning TEXT,
                        confidence REAL DEFAULT 0.0,
                        executed_at TEXT NOT NULL
                    )
                """)
        except Exception as e:
            logger.warning(f"Context database initialization failed: {e}")
    
    async def adapt_context(self, current_context: Dict[str, Any], 
                           execution_step: str) -> Dict[str, Any]:
        """Dynamically adapt context based on execution progress."""
        # Analyze current context
        context_analysis = await self.context_analyzer.analyze_context(current_context, execution_step)
        
        # Generate adaptations if needed
        if context_analysis["gaps_detected"]:
            adaptations = await self.adaptation_engine.generate_adaptations(
                context_analysis["gaps_detected"], 
                {"execution_step": execution_step, "context_analysis": context_analysis}
            )
            
            # Execute adaptations
            adapted_context = await self._execute_adaptations(current_context, adaptations)
            
            # Store adaptation history
            self.adaptation_history.extend(adaptations)
            
            return adapted_context
        else:
            # No adaptations needed, return original context
            return current_context
    
    async def _execute_adaptations(self, context: Dict[str, Any], 
                                 adaptations: List[ContextAdaptation]) -> Dict[str, Any]:
        """Execute context adaptations."""
        adapted_context = context.copy()
        
        for adaptation in adaptations:
            try:
                # Execute adaptation based on action type
                if "query_kb" in adaptation.action_type:
                    adapted_context = await self._query_knowledge_base(adapted_context, adaptation)
                elif "search_external" in adaptation.action_type:
                    adapted_context = await self._search_external_sources(adapted_context, adaptation)
                elif "infer" in adaptation.action_type:
                    adapted_context = await self._infer_from_context(adapted_context, adaptation)
                elif "update" in adaptation.action_type:
                    adapted_context = await self._update_context(adapted_context, adaptation)
                
                # Mark adaptation as executed
                adaptation.executed_at = datetime.now()
                
            except Exception as e:
                logger.error(f"Adaptation execution failed: {e}")
                adaptation.confidence = 0.0
        
        return adapted_context
    
    async def _query_knowledge_base(self, context: Dict[str, Any], 
                                   adaptation: ContextAdaptation) -> Dict[str, Any]:
        """Query knowledge base for missing information."""
        # This would integrate with your existing knowledge base
        # For now, add placeholder information
        context[f"kb_query_{adaptation.adaptation_id}"] = {
            "source": "knowledge_base",
            "content": f"Information retrieved from KB for {adaptation.description}",
            "confidence": adaptation.confidence,
            "retrieved_at": datetime.now().isoformat()
        }
        
        return context
    
    async def _search_external_sources(self, context: Dict[str, Any], 
                                     adaptation: ContextAdaptation) -> Dict[str, Any]:
        """Search external sources for missing information."""
        # This would integrate with external APIs, databases, etc.
        context[f"external_search_{adaptation.adaptation_id}"] = {
            "source": "external_search",
            "content": f"Information retrieved from external sources for {adaptation.description}",
            "confidence": adaptation.confidence * 0.8,  # External sources have lower confidence
            "retrieved_at": datetime.now().isoformat()
        }
        
        return context
    
    async def _infer_from_context(self, context: Dict[str, Any], 
                                 adaptation: ContextAdaptation) -> Dict[str, Any]:
        """Infer missing information from existing context."""
        # Use pattern matching and inference to fill gaps
        inferred_content = await self._infer_content_from_patterns(context, adaptation)
        
        context[f"inferred_{adaptation.adaptation_id}"] = {
            "source": "inference",
            "content": inferred_content,
            "confidence": adaptation.confidence * 0.6,  # Inferred information has lower confidence
            "inferred_at": datetime.now().isoformat()
        }
        
        return context
    
    async def _infer_content_from_patterns(self, context: Dict[str, Any], 
                                         adaptation: ContextAdaptation) -> str:
        """Infer content from existing context patterns."""
        # Simple pattern-based inference
        # In practice, this would use more sophisticated NLP and ML techniques
        
        if "threat" in adaptation.description.lower():
            return "Threat information inferred from existing threat patterns"
        elif "incident" in adaptation.description.lower():
            return "Incident information inferred from existing incident patterns"
        elif "compliance" in adaptation.description.lower():
            return "Compliance information inferred from existing compliance patterns"
        else:
            return "Information inferred from existing context patterns"
    
    async def _update_context(self, context: Dict[str, Any], 
                             adaptation: ContextAdaptation) -> Dict[str, Any]:
        """Update existing context information."""
        # Update timestamps and relevance scores
        for element_id in adaptation.context_elements:
            if element_id in context:
                context[element_id]["last_updated"] = datetime.now().isoformat()
                context[element_id]["relevance_score"] = min(1.0, context[element_id].get("relevance_score", 0.0) + 0.1)
        
        return context
    
    async def get_context_health_report(self) -> Dict[str, Any]:
        """Get comprehensive context health report."""
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len([a for a in self.adaptation_history if 
                                     (datetime.now() - a.executed_at).days < 1]),
            "adaptation_success_rate": self._calculate_adaptation_success_rate(),
            "context_quality_metrics": await self._calculate_context_quality_metrics(),
            "recommendations": self._generate_context_recommendations()
        }
    
    def _calculate_adaptation_success_rate(self) -> float:
        """Calculate adaptation success rate."""
        if not self.adaptation_history:
            return 0.0
        
        successful_adaptations = [a for a in self.adaptation_history if a.confidence > 0.5]
        return len(successful_adaptations) / len(self.adaptation_history)
    
    async def _calculate_context_quality_metrics(self) -> Dict[str, float]:
        """Calculate context quality metrics."""
        # This would analyze the actual context data
        return {
            "completeness": 0.8,  # Placeholder
            "relevance": 0.7,     # Placeholder
            "consistency": 0.9,   # Placeholder
            "freshness": 0.6      # Placeholder
        }
    
    def _generate_context_recommendations(self) -> List[str]:
        """Generate context improvement recommendations."""
        recommendations = []
        
        if len(self.adaptation_history) > 10:
            recommendations.append("High number of adaptations - consider proactive context management")
        
        success_rate = self._calculate_adaptation_success_rate()
        if success_rate < 0.7:
            recommendations.append("Low adaptation success rate - review adaptation strategies")
        
        return recommendations

# Global adaptive context manager instance
adaptive_context_manager = AdaptiveContextManager()

# Convenience functions
async def adapt_context(current_context: Dict[str, Any], execution_step: str) -> Dict[str, Any]:
    """Convenience function for context adaptation."""
    return await adaptive_context_manager.adapt_context(current_context, execution_step)

async def get_context_health() -> Dict[str, Any]:
    """Convenience function for context health report."""
    return await adaptive_context_manager.get_context_health_report()
