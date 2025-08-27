#!/usr/bin/env python3
"""
Dynamic Workflow Orchestration System for Cybersecurity Agent
Provides LLM-powered workflow generation and real-time adaptation
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Workflow type enumeration."""
    THREAT_HUNTING = "threat_hunting"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    INVESTIGATION = "investigation"
    ANALYSIS = "analysis"
    DATA_PROCESSING = "data_processing"
    REPORTING = "reporting"
    HYBRID = "hybrid"

class WorkflowComplexity(Enum):
    """Workflow complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    step_id: str
    name: str
    description: str
    tool_id: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_time: float
    retry_count: int = 0
    max_retries: int = 3
    is_parallel: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

@dataclass
class WorkflowPlan:
    """Complete workflow execution plan."""
    plan_id: str
    workflow_type: WorkflowType
    complexity: WorkflowComplexity
    steps: List[WorkflowStep]
    estimated_total_time: float
    resource_requirements: Dict[str, Any]
    created_at: datetime
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['workflow_type'] = self.workflow_type.value
        data['complexity'] = self.complexity.value
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    user_input: str
    available_tools: List[str]
    resource_constraints: Dict[str, Any]
    knowledge_base: Dict[str, Any]
    session_data: Dict[str, Any]
    user_preferences: Dict[str, Any]

class LLMRouter:
    """LLM-powered workflow routing and analysis."""
    
    def __init__(self):
        self.routing_prompts = self._load_routing_prompts()
        self.workflow_templates = self._load_workflow_templates()
    
    def _load_routing_prompts(self) -> Dict[str, str]:
        """Load routing prompts for different workflow types."""
        return {
            "threat_hunting": """
            Analyze the user request and determine if this is a threat hunting task.
            Look for keywords like: threat, attack, malware, APT, suspicious, investigation, hunting.
            Consider the context and available tools.
            """,
            "incident_response": """
            Analyze the user request and determine if this is an incident response task.
            Look for keywords like: incident, breach, response, emergency, urgent, containment.
            Consider the context and available tools.
            """,
            "compliance": """
            Analyze the user request and determine if this is a compliance task.
            Look for keywords like: compliance, policy, regulation, audit, standard, requirement.
            Consider the context and available tools.
            """,
            "data_processing": """
            Analyze the user request and determine if this is a data processing task.
            Look for keywords like: data, csv, json, process, analyze, transform, clean.
            Consider the context and available tools.
            """
        }
    
    def _load_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined workflow templates."""
        return {
            "threat_hunting": {
                "steps": [
                    {"name": "threat_intelligence_gathering", "tool": "threat_analyzer"},
                    {"name": "data_collection", "tool": "data_collector"},
                    {"name": "pattern_analysis", "tool": "pattern_analyzer"},
                    {"name": "threat_assessment", "tool": "threat_assessor"},
                    {"name": "reporting", "tool": "report_generator"}
                ],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 30.0
            },
            "incident_response": {
                "steps": [
                    {"name": "incident_assessment", "tool": "incident_assessor"},
                    {"name": "containment", "tool": "containment_tool"},
                    {"name": "evidence_collection", "tool": "evidence_collector"},
                    {"name": "analysis", "tool": "incident_analyzer"},
                    {"name": "remediation", "tool": "remediation_tool"},
                    {"name": "documentation", "tool": "documentation_tool"}
                ],
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_time": 60.0
            },
            "compliance": {
                "steps": [
                    {"name": "policy_review", "tool": "policy_analyzer"},
                    {"name": "gap_analysis", "tool": "gap_analyzer"},
                    {"name": "compliance_check", "tool": "compliance_checker"},
                    {"name": "reporting", "tool": "compliance_reporter"}
                ],
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_time": 45.0
            }
        }
    
    async def analyze_intent(self, user_input: str, context: ExecutionContext) -> Dict[str, Any]:
        """Analyze user intent and determine workflow type."""
        # Simple keyword-based analysis (can be enhanced with actual LLM)
        user_input_lower = user_input.lower()
        
        intent_analysis = {
            "primary_intent": None,
            "confidence": 0.0,
            "secondary_intents": [],
            "complexity_indicators": [],
            "urgency_level": "normal"
        }
        
        # Analyze primary intent
        if any(word in user_input_lower for word in ["threat", "attack", "malware", "apt", "hunting"]):
            intent_analysis["primary_intent"] = "threat_hunting"
            intent_analysis["confidence"] = 0.9
        elif any(word in user_input_lower for word in ["incident", "breach", "response", "emergency"]):
            intent_analysis["primary_intent"] = "incident_response"
            intent_analysis["confidence"] = 0.9
        elif any(word in user_input_lower for word in ["compliance", "policy", "regulation", "audit"]):
            intent_analysis["primary_intent"] = "compliance"
            intent_analysis["confidence"] = 0.9
        elif any(word in user_input_lower for word in ["data", "csv", "json", "process", "analyze"]):
            intent_analysis["primary_intent"] = "data_processing"
            intent_analysis["confidence"] = 0.8
        else:
            intent_analysis["primary_intent"] = "analysis"
            intent_analysis["confidence"] = 0.6
        
        # Analyze complexity
        if any(word in user_input_lower for word in ["complex", "advanced", "expert", "comprehensive"]):
            intent_analysis["complexity_indicators"].append("high_complexity")
        elif any(word in user_input_lower for word in ["simple", "basic", "quick", "fast"]):
            intent_analysis["complexity_indicators"].append("low_complexity")
        
        # Analyze urgency
        if any(word in user_input_lower for word in ["urgent", "emergency", "critical", "immediate"]):
            intent_analysis["urgency_level"] = "high"
        
        return intent_analysis
    
    async def generate_workflow_plan(self, intent: Dict[str, Any], context: ExecutionContext) -> WorkflowPlan:
        """Generate workflow plan based on intent analysis."""
        workflow_type = WorkflowType(intent["primary_intent"])
        
        # Get base template
        base_template = self.workflow_templates.get(intent["primary_intent"], {})
        
        # Generate steps
        steps = []
        step_id_counter = 1
        
        for step_template in base_template.get("steps", []):
            step = WorkflowStep(
                step_id=f"step_{step_id_counter}",
                name=step_template["name"],
                description=f"Execute {step_template['name']}",
                tool_id=step_template["tool"],
                parameters={},
                dependencies=[],
                estimated_time=5.0,  # Default step time
                is_parallel=False
            )
            steps.append(step)
            step_id_counter += 1
        
        # Calculate total time
        estimated_total_time = sum(step.estimated_time for step in steps)
        
        # Determine complexity
        if intent["complexity_indicators"]:
            if "high_complexity" in intent["complexity_indicators"]:
                complexity = WorkflowComplexity.EXPERT
            elif "low_complexity" in intent["complexity_indicators"]:
                complexity = WorkflowComplexity.SIMPLE
            else:
                complexity = WorkflowComplexity.MODERATE
        else:
            complexity = base_template.get("complexity", WorkflowComplexity.MODERATE)
        
        # Create workflow plan
        plan = WorkflowPlan(
            plan_id=f"plan_{hashlib.sha256(f'{workflow_type}_{datetime.now()}'.encode()).hexdigest()[:8]}",
            workflow_type=workflow_type,
            complexity=complexity,
            steps=steps,
            estimated_total_time=estimated_total_time,
            resource_requirements={
                "memory_mb": len(steps) * 50,  # Rough estimate
                "cpu_percent": len(steps) * 10,
                "tools_required": [step.tool_id for step in steps]
            },
            created_at=datetime.now(),
            context=context.__dict__
        )
        
        return plan

class WorkflowGenerator:
    """Generate and optimize workflow plans."""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load workflow optimization rules."""
        return {
            "parallelization_threshold": 3,
            "resource_optimization": True,
            "tool_efficiency_boost": 1.2,
            "complexity_penalty": 0.8
        }
    
    async def create_workflow(self, intent: Dict[str, Any], context: ExecutionContext) -> WorkflowPlan:
        """Create optimized workflow based on intent and context."""
        # Generate base workflow
        workflow_plan = await self._generate_base_workflow(intent, context)
        
        # Optimize workflow
        optimized_plan = await self._optimize_workflow(workflow_plan, context)
        
        return optimized_plan
    
    async def _generate_base_workflow(self, intent: Dict[str, Any], context: ExecutionContext) -> WorkflowPlan:
        """Generate base workflow plan."""
        # This would integrate with the LLM router
        # For now, create a simple workflow
        workflow_type = WorkflowType(intent.get("primary_intent", "analysis"))
        
        steps = [
            WorkflowStep(
                step_id="step_1",
                name="data_collection",
                description="Collect relevant data",
                tool_id="data_collector",
                parameters={},
                dependencies=[],
                estimated_time=10.0,
                is_parallel=False
            ),
            WorkflowStep(
                step_id="step_2",
                name="analysis",
                description="Analyze collected data",
                tool_id="analyzer",
                parameters={},
                dependencies=["step_1"],
                estimated_time=15.0,
                is_parallel=False
            ),
            WorkflowStep(
                step_id="step_3",
                name="reporting",
                description="Generate report",
                tool_id="report_generator",
                parameters={},
                dependencies=["step_2"],
                estimated_time=5.0,
                is_parallel=False
            )
        ]
        
        plan = WorkflowPlan(
            plan_id=f"base_plan_{hashlib.sha256(f'{workflow_type}_{datetime.now()}'.encode()).hexdigest()[:8]}",
            workflow_type=workflow_type,
            complexity=WorkflowComplexity.MODERATE,
            steps=steps,
            estimated_total_time=30.0,
            resource_requirements={
                "memory_mb": 150,
                "cpu_percent": 30,
                "tools_required": ["data_collector", "analyzer", "report_generator"]
            },
            created_at=datetime.now(),
            context=context.__dict__
        )
        
        return plan
    
    async def _optimize_workflow(self, plan: WorkflowPlan, context: ExecutionContext) -> WorkflowPlan:
        """Optimize workflow for performance and efficiency."""
        optimized_steps = []
        
        # Analyze dependencies and optimize parallelization
        for step in plan.steps:
            # Check if step can be parallelized
            if len(step.dependencies) == 0 and len(optimized_steps) > 0:
                # Independent step, can be parallel
                step.is_parallel = True
            
            optimized_steps.append(step)
        
        # Update resource requirements
        parallel_steps = [s for s in optimized_steps if s.is_parallel]
        sequential_steps = [s for s in optimized_steps if not s.is_parallel]
        
        # Adjust time estimates for parallel execution
        if parallel_steps:
            parallel_time = max(step.estimated_time for step in parallel_steps)
            sequential_time = sum(step.estimated_time for step in sequential_steps)
            total_time = parallel_time + sequential_time
        else:
            total_time = sum(step.estimated_time for step in optimized_steps)
        
        # Create optimized plan
        optimized_plan = WorkflowPlan(
            plan_id=f"opt_{plan.plan_id}",
            workflow_type=plan.workflow_type,
            complexity=plan.complexity,
            steps=optimized_steps,
            estimated_total_time=total_time,
            resource_requirements={
                "memory_mb": len(optimized_steps) * 50,
                "cpu_percent": len(optimized_steps) * 10,
                "tools_required": [step.tool_id for step in optimized_steps],
                "parallel_steps": len(parallel_steps),
                "sequential_steps": len(sequential_steps)
            },
            created_at=datetime.now(),
            context=context.__dict__
        )
        
        return optimized_plan

class ContextAnalyzer:
    """Analyze and understand execution context."""
    
    def __init__(self):
        self.context_patterns = self._load_context_patterns()
    
    def _load_context_patterns(self) -> Dict[str, Any]:
        """Load context analysis patterns."""
        return {
            "resource_intensive": ["large_data", "complex_analysis", "multiple_sources"],
            "time_sensitive": ["urgent", "emergency", "real_time", "live"],
            "security_critical": ["threat", "breach", "incident", "vulnerability"],
            "compliance_related": ["audit", "policy", "regulation", "standard"]
        }
    
    async def analyze_context(self, context: ExecutionContext) -> Dict[str, Any]:
        """Analyze execution context for optimization opportunities."""
        analysis = {
            "context_type": "standard",
            "optimization_opportunities": [],
            "resource_constraints": [],
            "security_considerations": []
        }
        
        # Analyze user input for context clues
        user_input_lower = context.user_input.lower()
        
        # Check for resource-intensive operations
        if any(word in user_input_lower for word in ["large", "complex", "multiple", "batch"]):
            analysis["context_type"] = "resource_intensive"
            analysis["optimization_opportunities"].append("parallel_processing")
            analysis["optimization_opportunities"].append("caching")
        
        # Check for time-sensitive operations
        if any(word in user_input_lower for word in ["urgent", "emergency", "real-time", "live"]):
            analysis["context_type"] = "time_sensitive"
            analysis["optimization_opportunities"].append("priority_execution")
            analysis["optimization_opportunities"].append("resource_allocation")
        
        # Check for security-critical operations
        if any(word in user_input_lower for word in ["threat", "breach", "incident", "vulnerability"]):
            analysis["security_considerations"].append("high_priority")
            analysis["security_considerations"].append("audit_logging")
        
        # Analyze available tools
        if len(context.available_tools) < 5:
            analysis["resource_constraints"].append("limited_tools")
        
        return analysis

class DynamicWorkflowOrchestrator:
    """Main orchestrator for dynamic workflow generation and execution."""
    
    def __init__(self):
        self.llm_router = LLMRouter()
        self.workflow_generator = WorkflowGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.workflow_history: List[WorkflowPlan] = []
    
    async def route_workflow(self, user_input: str, context: ExecutionContext) -> WorkflowPlan:
        """Route user input to appropriate workflow."""
        # Analyze user intent
        intent = await self.llm_router.analyze_intent(user_input, context)
        
        # Analyze context
        context_analysis = await self.context_analyzer.analyze_context(context)
        
        # Generate workflow
        workflow = await self.workflow_generator.create_workflow(intent, context)
        
        # Store in history
        self.workflow_history.append(workflow)
        
        return workflow
    
    async def adapt_workflow(self, workflow: WorkflowPlan, new_context: Dict[str, Any]) -> WorkflowPlan:
        """Adapt existing workflow to new context."""
        # Create new execution context
        adapted_context = ExecutionContext(
            user_input=workflow.context.get("user_input", ""),
            available_tools=new_context.get("available_tools", []),
            resource_constraints=new_context.get("resource_constraints", {}),
            knowledge_base=new_context.get("knowledge_base", {}),
            session_data=new_context.get("session_data", {}),
            user_preferences=new_context.get("user_preferences", {})
        )
        
        # Re-analyze intent
        intent = await self.llm_router.analyze_intent(
            workflow.context.get("user_input", ""), 
            adapted_context
        )
        
        # Generate adapted workflow
        adapted_workflow = await self.workflow_generator.create_workflow(intent, adapted_context)
        
        return adapted_workflow
    
    async def get_workflow_recommendations(self, user_input: str, context: ExecutionContext) -> Dict[str, Any]:
        """Get workflow recommendations based on user input."""
        # Analyze intent
        intent = await self.llm_router.analyze_intent(user_input, context)
        
        # Get context analysis
        context_analysis = await self.context_analyzer.analyze_context(context)
        
        # Generate workflow plan
        workflow = await self.workflow_generator.create_workflow(intent, context)
        
        recommendations = {
            "workflow_type": workflow.workflow_type.value,
            "complexity": workflow.complexity.value,
            "estimated_time": workflow.estimated_total_time,
            "steps": [step.to_dict() for step in workflow.steps],
            "resource_requirements": workflow.resource_requirements,
            "context_analysis": context_analysis,
            "optimization_suggestions": self._generate_optimization_suggestions(workflow, context_analysis)
        }
        
        return recommendations
    
    def _generate_optimization_suggestions(self, workflow: WorkflowPlan, context_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for workflow."""
        suggestions = []
        
        if workflow.complexity == WorkflowComplexity.COMPLEX:
            suggestions.append("Consider breaking down into smaller sub-workflows")
        
        if len(workflow.steps) > 5:
            suggestions.append("Workflow has many steps - consider parallelization")
        
        if "resource_intensive" in context_analysis.get("context_type", ""):
            suggestions.append("Resource-intensive operation - enable caching and parallel processing")
        
        if "time_sensitive" in context_analysis.get("context_type", ""):
            suggestions.append("Time-sensitive operation - prioritize execution and resource allocation")
        
        return suggestions
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        return [workflow.to_dict() for workflow in self.workflow_history]

# Global orchestrator instance
dynamic_orchestrator = DynamicWorkflowOrchestrator()

# Convenience functions
async def route_workflow(user_input: str, context: ExecutionContext) -> WorkflowPlan:
    """Convenience function for workflow routing."""
    return await dynamic_orchestrator.route_workflow(user_input, context)

async def get_recommendations(user_input: str, context: ExecutionContext) -> Dict[str, Any]:
    """Convenience function for workflow recommendations."""
    return await dynamic_orchestrator.get_workflow_recommendations(user_input, context)
