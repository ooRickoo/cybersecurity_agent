#!/usr/bin/env python3
"""
Agentic Workflow System - Query Path and Runner Agent Implementation

Implements dynamic workflow execution with:
- Query Path: Intelligent tool selection and routing
- Runner Agent: Dynamic workflow execution and adaptation
- Context-Aware Tool Selection: Based on knowledge graph context
- Automated and Manual Execution Paths
- Enhanced Multi-Tier Memory Integration: Short-term, medium-term, and long-term memory
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
import pandas as pd
from pathlib import Path
import csv
import io
import uuid

from workflow_templates import WorkflowContext, ProblemType
from enhanced_context_memory import EnhancedContextMemoryManager
from enhanced_agentic_memory_system import EnhancedAgenticMemorySystem, MemoryTier, MemoryCategory
from mcp_integration_layer import MCPIntegrationLayer, MCPToolCategory, MCPToolCapability

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Workflow execution modes."""
    AUTOMATED = "automated"  # CSV input → batch processing → enriched output
    MANUAL = "manual"        # Interactive chat → problem solving → larger resolution
    HYBRID = "hybrid"        # Combination of both approaches

class QueryPath:
    """Intelligent tool selection and routing based on context."""
    
    def __init__(self, mcp_integration: MCPIntegrationLayer, context_memory: EnhancedContextMemoryManager, enhanced_memory: EnhancedAgenticMemorySystem):
        self.mcp_integration = mcp_integration
        self.context_memory = context_memory
        self.enhanced_memory = enhanced_memory
        self.tool_selection_history = []
        self.path_optimization_rules = []
        
    async def select_tools_for_problem(self, problem_description: str, context: Dict[str, Any]) -> List[str]:
        """Select optimal tools for a given problem using context-aware routing."""
        # Store context building in short-term memory
        self.enhanced_memory.store_context_building({
            "type": "tool_selection",
            "stage": "problem_analysis",
            "problem_description": problem_description,
            "context_summary": {
                "priority": context.get('priority', 1),
                "complexity": context.get('complexity', 1),
                "mode": context.get('mode', 'unknown')
            }
        }, importance_score=0.6)
        
        # Get relevant context from all memory tiers
        relevant_context = await self._get_multi_tier_context(problem_description, context)
        
        # Analyze problem requirements
        requirements = self._analyze_problem_requirements(problem_description, context)
        
        # Get available tools by capability
        available_tools = self._get_available_tools_by_requirements(requirements)
        
        # Score tools based on context relevance and memory tiers
        scored_tools = await self._score_tools_by_context_and_memory(available_tools, relevant_context, requirements)
        
        # Select optimal tool combination
        selected_tools = self._select_optimal_tool_combination(scored_tools, requirements)
        
        # Record selection for learning and optimization
        self._record_tool_selection(problem_description, selected_tools, requirements, relevant_context)
        
        # Store tool selection pattern in medium-term memory
        self.enhanced_memory.store_adaptation_rule({
            "type": "tool_selection_pattern",
            "triggers": [requirements['capabilities'], requirements['data_types']],
            "selected_tools": selected_tools,
            "success_rate": 0.8,  # Initial estimate
            "context_used": list(relevant_context.keys())
        }, importance_score=0.8)
        
        return selected_tools
    
    async def _get_multi_tier_context(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context from all memory tiers using enhanced memory system."""
        try:
            # Retrieve context from all memory tiers
            multi_tier_context = self.enhanced_memory.retrieve_context_for_problem(
                problem_description, max_context_items=100
            )
            
            # Store context retrieval in short-term memory
            self.enhanced_memory.store_context_building({
                "type": "context_retrieval",
                "stage": "multi_tier_search",
                "problem_description": problem_description,
                "context_items_found": multi_tier_context['total_context_items'],
                "tier_distribution": multi_tier_context['tier_distribution'],
                "confidence_level": multi_tier_context['confidence_level']
            }, importance_score=0.7)
            
            return multi_tier_context
            
        except Exception as e:
            logger.error(f"❌ Error getting multi-tier context: {e}")
            return {"short_term": [], "medium_term": [], "long_term": [], "synthesized_context": {}}
    
    def _analyze_problem_requirements(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem to determine tool requirements."""
        requirements = {
            'capabilities': [],
            'data_types': [],
            'complexity_level': 1,
            'priority': context.get('priority', 1),
            'time_constraints': context.get('time_constraints', 'flexible'),
            'resource_constraints': context.get('resource_constraints', 'standard')
        }
        
        # Analyze description for capability requirements
        description_lower = problem_description.lower()
        
        if any(word in description_lower for word in ['analyze', 'investigate', 'examine']):
            requirements['capabilities'].extend(['read', 'analyze'])
        
        if any(word in description_lower for word in ['transform', 'convert', 'modify']):
            requirements['capabilities'].extend(['transform', 'write'])
        
        if any(word in description_lower for word in ['integrate', 'connect', 'combine']):
            requirements['capabilities'].extend(['integrate', 'execute'])
        
        if any(word in description_lower for word in ['monitor', 'track', 'observe']):
            requirements['capabilities'].extend(['monitor', 'alert'])
        
        # Analyze for data type requirements
        if any(word in description_lower for word in ['csv', 'excel', 'spreadsheet']):
            requirements['data_types'].append('tabular')
        
        if any(word in description_lower for word in ['log', 'text', 'document']):
            requirements['data_types'].append('text')
        
        if any(word in description_lower for word in ['network', 'graph', 'relationship']):
            requirements['data_types'].append('graph')
        
        # Set complexity based on description length and context
        requirements['complexity_level'] = min(len(problem_description.split()) // 10 + 1, 10)
        
        return requirements
    
    def _get_available_tools_by_requirements(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available tools that match the requirements."""
        available_tools = []
        
        for capability in requirements['capabilities']:
            try:
                capability_enum = MCPToolCapability(capability)
                tools = self.mcp_integration.registry.get_tools_by_capability(capability_enum)
                available_tools.extend([{
                    'tool_id': tool.tool_id,
                    'metadata': tool.metadata,
                    'performance': tool.success_rate,
                    'capability': capability
                } for tool in tools])
            except ValueError:
                continue
        
        return available_tools
    
    async def _score_tools_by_context_and_memory(self, tools: List[Dict[str, Any]], context: Dict[str, Any], requirements: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Score tools based on context relevance, memory tiers, and requirements match."""
        scored_tools = []
        
        for tool in tools:
            score = 0.0
            
            # Base score from performance (30%)
            score += tool['performance'] * 0.3
            
            # Capability match score (40%)
            if tool['capability'] in requirements['capabilities']:
                score += 0.4
            
            # Context relevance score with memory tier weighting (30%)
            context_relevance = self._calculate_context_relevance_with_memory_tiers(tool, context)
            score += context_relevance * 0.3
            
            scored_tools.append((tool['tool_id'], score))
        
        # Sort by score (highest first)
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools
    
    def _calculate_context_relevance_with_memory_tiers(self, tool: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant a tool is to the current context using memory tiers."""
        relevance = 0.0
        
        # Check short-term memory relevance (session-specific)
        short_term_items = context.get('short_term', [])
        for item in short_term_items:
            if tool['tool_id'] in str(item.get('content', '')):
                relevance += 0.3  # Short-term relevance
        
        # Check medium-term memory relevance (workflow patterns)
        medium_term_items = context.get('medium_term', [])
        for item in medium_term_items:
            if tool['tool_id'] in str(item.get('content', '')):
                relevance += 0.4  # Medium-term relevance (higher weight)
        
        # Check long-term memory relevance (knowledge accumulation)
        long_term_items = context.get('long_term', [])
        for item in long_term_items:
            if tool['tool_id'] in str(item.get('content', '')):
                relevance += 0.5  # Long-term relevance (highest weight)
        
        # Check tool category relevance
        if tool['metadata'].category.value in ['context_memory', 'data_analysis']:
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _select_optimal_tool_combination(self, scored_tools: List[Tuple[str, float]], requirements: Dict[str, Any]) -> List[str]:
        """Select optimal combination of tools for the requirements."""
        selected_tools = []
        
        # Select tools based on requirements and scores
        for tool_id, score in scored_tools:
            if len(selected_tools) >= 5:  # Limit to 5 tools max
                break
            
            if score > 0.5:  # Only select tools with good scores
                selected_tools.append(tool_id)
        
        # Ensure we have at least one tool
        if not selected_tools and scored_tools:
            selected_tools.append(scored_tools[0][0])
        
        return selected_tools
    
    def _record_tool_selection(self, problem_description: str, selected_tools: List[str], requirements: Dict[str, Any], context: Dict[str, Any]):
        """Record tool selection for learning and optimization."""
        selection_record = {
            'timestamp': time.time(),
            'problem_description': problem_description,
            'selected_tools': selected_tools,
            'requirements': requirements,
            'context_summary': {
                'total_items': context.get('synthesized_context', {}).get('total_context_items', 0),
                'tier_distribution': context.get('synthesized_context', {}).get('tier_distribution', {}),
                'confidence_level': context.get('synthesized_context', {}).get('confidence_level', 'low')
            }
        }
        
        self.tool_selection_history.append(selection_record)
        
        # Keep only last 1000 selections
        if len(self.tool_selection_history) > 1000:
            self.tool_selection_history = self.tool_selection_history[-1000:]

class RunnerAgent:
    """Dynamic workflow execution and adaptation agent."""
    
    def __init__(self, mcp_integration: MCPIntegrationLayer, context_memory: EnhancedContextMemoryManager, enhanced_memory: EnhancedAgenticMemorySystem):
        self.mcp_integration = mcp_integration
        self.context_memory = context_memory
        self.enhanced_memory = enhanced_memory
        self.execution_history = []
        self.adaptation_rules = []
        self.performance_metrics = {}
        
    async def execute_workflow(self, problem_description: str, context: Dict[str, Any], mode: ExecutionMode) -> Dict[str, Any]:
        """Execute workflow using the appropriate mode."""
        # Start workflow session in enhanced memory system
        session_id = f"workflow_{uuid.uuid4().hex[:8]}"
        workflow_context = {
            **context,
            "mode": mode.value,
            "problem_description": problem_description,
            "start_time": time.time()
        }
        
        self.enhanced_memory.start_workflow_session(session_id, workflow_context)
        
        try:
            if mode == ExecutionMode.AUTOMATED:
                result = await self._execute_automated_workflow(problem_description, context)
            elif mode == ExecutionMode.MANUAL:
                result = await self._execute_manual_workflow(problem_description, context)
            else:  # HYBRID
                result = await self._execute_hybrid_workflow(problem_description, context)
            
            # Store workflow execution result in appropriate memory tier
            self.enhanced_memory.store_workflow_execution({
                "type": "workflow_completion",
                "mode": mode.value,
                "problem_description": problem_description,
                "success": result.get('success', False),
                "execution_summary": result.get('execution_summary', {}),
                "session_id": session_id
            }, importance_score=0.9)
            
            return result
            
        finally:
            # End workflow session
            self.enhanced_memory.end_workflow_session()
    
    async def _execute_automated_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automated workflow for batch processing."""
        # Store workflow execution start in short-term memory
        self.enhanced_memory.store_workflow_execution({
            "type": "automated_workflow_start",
            "mode": "automated",
            "problem_description": problem_description,
            "stage": "initialization"
        }, importance_score=0.7)
        
        # Parse CSV input if provided
        csv_data = context.get('csv_input', None)
        if csv_data:
            df = pd.read_csv(io.StringIO(csv_data))
            total_rows = len(df)
        else:
            df = pd.DataFrame()
            total_rows = 0
        
        results = []
        execution_summary = {
            'mode': 'automated',
            'total_rows_processed': total_rows,
            'successful_rows': 0,
            'failed_rows': 0,
            'execution_time': 0,
            'tools_used': [],
            'adaptations_made': 0
        }
        
        start_time = time.time()
        
        try:
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Create row-specific context
                    row_context = {
                        **context,
                        'row_index': index,
                        'row_data': row.to_dict(),
                        'current_row': index + 1,
                        'total_rows': total_rows
                    }
                    
                    # Store row processing start in short-term memory
                    self.enhanced_memory.store_context_building({
                        "type": "row_processing",
                        "stage": f"row_{index + 1}",
                        "row_data": row.to_dict(),
                        "progress": f"{index + 1}/{total_rows}"
                    }, importance_score=0.5)
                    
                    # Execute workflow for this row
                    row_result = await self._execute_single_row_workflow(problem_description, row_context)
                    
                    # Enrich row with results
                    enriched_row = {**row.to_dict(), 'workflow_results': row_result}
                    results.append(enriched_row)
                    
                    execution_summary['successful_rows'] += 1
                    
                    # Store successful row processing in medium-term memory
                    self.enhanced_memory.store_tool_performance(f"row_processing_{index}", {
                        "row_index": index,
                        "success": True,
                        "execution_time": row_result.get('execution_time', 0),
                        "tools_used": row_result.get('tools_used', [])
                    }, importance_score=0.7)
                    
                except Exception as e:
                    logger.error(f"Failed to process row {index}: {e}")
                    # Add error information to row
                    error_row = {**row.to_dict(), 'workflow_error': str(e)}
                    results.append(error_row)
                    execution_summary['failed_rows'] += 1
                    
                    # Store failed row processing in medium-term memory for learning
                    self.enhanced_memory.store_tool_performance(f"row_processing_{index}", {
                        "row_index": index,
                        "success": False,
                        "error": str(e),
                        "execution_time": 0
                    }, importance_score=0.8)  # Higher importance for learning from failures
            
            # Create enriched CSV output
            enriched_df = pd.DataFrame(results)
            csv_output = enriched_df.to_csv(index=False)
            
            execution_summary['execution_time'] = time.time() - start_time
            execution_summary['csv_output'] = csv_output
            
            # Store successful workflow completion in long-term memory
            self.enhanced_memory.store_solution_synthesis({
                "type": "automated_csv_processing",
                "problem_description": problem_description,
                "total_rows": total_rows,
                "success_rate": execution_summary['successful_rows'] / max(total_rows, 1),
                "execution_time": execution_summary['execution_time'],
                "tools_used": execution_summary['tools_used'],
                "confidence_score": min(execution_summary['successful_rows'] / max(total_rows, 1), 1.0)
            }, importance_score=0.9)
            
            return {
                'success': True,
                'mode': 'automated',
                'execution_summary': execution_summary,
                'enriched_data': results,
                'csv_output': csv_output
            }
            
        except Exception as e:
            logger.error(f"Automated workflow execution failed: {e}")
            
            # Store workflow failure in medium-term memory for learning
            self.enhanced_memory.store_learning_outcome({
                "type": "workflow_failure",
                "problem_description": problem_description,
                "error": str(e),
                "impact_score": 0.9,  # High impact for learning
                "applicability": "future_automated_workflows"
            }, importance_score=1.0)
            
            return {
                'success': False,
                'mode': 'automated',
                'error': str(e),
                'execution_summary': execution_summary
            }
    
    async def _execute_manual_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manual workflow for interactive problem solving."""
        # Store workflow execution start in short-term memory
        self.enhanced_memory.store_workflow_execution({
            "type": "manual_workflow_start",
            "mode": "manual",
            "problem_description": problem_description,
            "stage": "initialization"
        }, importance_score=0.7)
        
        execution_summary = {
            'mode': 'manual',
            'steps_executed': 0,
            'tools_used': [],
            'adaptations_made': 0,
            'execution_time': 0,
            'problem_resolution': 'in_progress'
        }
        
        start_time = time.time()
        
        try:
            # Break down problem into smaller sub-problems
            sub_problems = self._decompose_problem(problem_description, context)
            
            # Store problem decomposition in medium-term memory
            self.enhanced_memory.store_problem_pattern(problem_description, {
                "type": "problem_decomposition",
                "sub_problems_count": len(sub_problems),
                "sub_problems": sub_problems,
                "decomposition_strategy": "natural_language_analysis",
                "success_rate": 0.8
            }, importance_score=0.8)
            
            resolved_sub_problems = []
            current_context = context.copy()
            
            # Solve each sub-problem
            for i, sub_problem in enumerate(sub_problems):
                logger.info(f"Solving sub-problem {i+1}/{len(sub_problems)}: {sub_problem}")
                
                # Store sub-problem start in short-term memory
                self.enhanced_memory.store_context_building({
                    "type": "sub_problem_execution",
                    "stage": f"sub_problem_{i + 1}",
                    "sub_problem": sub_problem,
                    "progress": f"{i + 1}/{len(sub_problems)}"
                }, importance_score=0.6)
                
                # Execute workflow for sub-problem
                sub_result = await self._execute_single_row_workflow(sub_problem, current_context)
                
                resolved_sub_problems.append({
                    'sub_problem': sub_problem,
                    'solution': sub_result,
                    'step_number': i + 1
                })
                
                # Update context with solution
                current_context = self._update_context_with_solution(current_context, sub_result)
                execution_summary['steps_executed'] += 1
                
                # Store sub-problem solution in medium-term memory
                self.enhanced_memory.store_tool_performance(f"sub_problem_{i + 1}", {
                    "sub_problem": sub_problem,
                    "success": sub_result.get('success', False),
                    "execution_time": sub_result.get('execution_time', 0),
                    "tools_used": sub_result.get('tools_used', [])
                }, importance_score=0.7)
            
            # Synthesize final solution
            final_solution = self._synthesize_solution(resolved_sub_problems, problem_description)
            
            execution_summary['execution_time'] = time.time() - start_time
            execution_summary['problem_resolution'] = 'completed'
            
            # Store final solution synthesis in long-term memory
            self.enhanced_memory.store_solution_synthesis({
                "type": "manual_problem_solving",
                "problem_description": problem_description,
                "sub_problems_resolved": len(resolved_sub_problems),
                "final_solution": final_solution,
                "execution_time": execution_summary['execution_time'],
                "confidence_score": final_solution.get('confidence_score', 0.0)
            }, importance_score=0.9)
            
            return {
                'success': True,
                'mode': 'manual',
                'execution_summary': execution_summary,
                'sub_problems': resolved_sub_problems,
                'final_solution': final_solution,
                'synthesized_context': current_context
            }
            
        except Exception as e:
            logger.error(f"Manual workflow execution failed: {e}")
            
            # Store workflow failure in medium-term memory for learning
            self.enhanced_memory.store_learning_outcome({
                "type": "manual_workflow_failure",
                "problem_description": problem_description,
                "error": str(e),
                "impact_score": 0.9,
                "applicability": "future_manual_workflows"
            }, importance_score=1.0)
            
            return {
                'success': False,
                'mode': 'manual',
                'error': str(e),
                'execution_summary': execution_summary
            }
    
    async def _execute_hybrid_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hybrid workflow combining automated and manual approaches."""
        # Store workflow execution start in short-term memory
        self.enhanced_memory.store_workflow_execution({
            "type": "hybrid_workflow_start",
            "mode": "hybrid",
            "problem_description": problem_description,
            "stage": "initialization"
        }, importance_score=0.7)
        
        # Start with automated approach
        automated_result = await self._execute_automated_workflow(problem_description, context)
        
        # If automated approach has high failure rate, switch to manual
        if automated_result['success']:
            failure_rate = automated_result['execution_summary']['failed_rows'] / max(automated_result['execution_summary']['total_rows_processed'], 1)
            
            if failure_rate > 0.3:  # More than 30% failure rate
                logger.info("High failure rate detected, switching to manual approach")
                
                # Store adaptation decision in medium-term memory
                self.enhanced_memory.store_adaptation_rule({
                    "type": "automated_to_manual_fallback",
                    "triggers": ["high_failure_rate"],
                    "failure_rate": failure_rate,
                    "threshold": 0.3,
                    "success_rate": 0.8
                }, importance_score=0.8)
                
                manual_result = await self._execute_manual_workflow(problem_description, context)
                
                # Store hybrid workflow completion in long-term memory
                self.enhanced_memory.store_solution_synthesis({
                    "type": "hybrid_workflow_with_fallback",
                    "problem_description": problem_description,
                    "automated_success": False,
                    "manual_fallback_success": manual_result.get('success', False),
                    "adaptation_reason": "high_failure_rate",
                    "confidence_score": manual_result.get('final_solution', {}).get('confidence_score', 0.0)
                }, importance_score=0.9)
                
                return {
                    'success': True,
                    'mode': 'hybrid',
                    'automated_attempt': automated_result,
                    'manual_fallback': manual_result,
                    'adaptation_reason': 'high_failure_rate',
                    'final_result': manual_result
                }
        
        # Store successful hybrid workflow in long-term memory
        self.enhanced_memory.store_solution_synthesis({
            "type": "hybrid_workflow_success",
            "problem_description": problem_description,
            "automated_success": True,
            "manual_fallback_needed": False,
            "adaptation_reason": "automated_success",
            "confidence_score": 0.9
        }, importance_score=0.9)
        
        return {
            'success': True,
            'mode': 'hybrid',
            'automated_result': automated_result,
            'manual_fallback': None,
            'adaptation_reason': 'automated_success'
        }
    
    async def _execute_single_row_workflow(self, problem_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow for a single row or sub-problem."""
        # Use query path to select tools
        query_path = QueryPath(self.mcp_integration, self.context_memory, self.enhanced_memory)
        selected_tools = await query_path.select_tools_for_problem(problem_description, context)
        
        # Execute workflow with selected tools
        workflow_result = await self.mcp_integration.execute_workflow_with_tools(
            WorkflowContext(
                problem_type=ProblemType.ANALYSIS,
                problem_description=problem_description,
                priority=context.get('priority', 1),
                complexity=context.get('complexity', 1)
            ),
            selected_tools
        )
        
        # Record execution
        self._record_execution(problem_description, selected_tools, workflow_result, context)
        
        return workflow_result
    
    def _decompose_problem(self, problem_description: str, context: Dict[str, Any]) -> List[str]:
        """Decompose a complex problem into smaller sub-problems."""
        # Simple decomposition based on problem description
        sub_problems = []
        
        # Look for natural breakpoints in the problem
        if 'and' in problem_description.lower():
            parts = problem_description.split(' and ')
            sub_problems.extend([part.strip() for part in parts if part.strip()])
        elif 'then' in problem_description.lower():
            parts = problem_description.split(' then ')
            sub_problems.extend([part.strip() for part in parts if part.strip()])
        else:
            # Break down by complexity
            words = problem_description.split()
            if len(words) > 20:
                # Split into chunks
                chunk_size = len(words) // 3
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    if chunk.strip():
                        sub_problems.append(chunk.strip())
            else:
                sub_problems.append(problem_description)
        
        return sub_problems
    
    def _update_context_with_solution(self, context: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """Update context with solution from sub-problem."""
        updated_context = context.copy()
        
        # Add solution to context
        if 'solutions' not in updated_context:
            updated_context['solutions'] = []
        
        updated_context['solutions'].append(solution)
        
        # Update knowledge graph with new information
        if solution.get('execution_results'):
            for tool_id, result in solution['execution_results'].items():
                if isinstance(result, dict) and 'result' in result:
                    # Add to context memory using enhanced memory system
                    self.enhanced_memory.store_context_building({
                        "type": "solution_integration",
                        "stage": "context_update",
                        "tool_id": tool_id,
                        "solution_content": str(result['result']),
                        "integration_timestamp": time.time()
                    }, importance_score=0.7)
        
        return updated_context
    
    def _synthesize_solution(self, resolved_sub_problems: List[Dict[str, Any]], original_problem: str) -> Dict[str, Any]:
        """Synthesize final solution from resolved sub-problems."""
        synthesis = {
            'original_problem': original_problem,
            'sub_problems_resolved': len(resolved_sub_problems),
            'synthesis_summary': '',
            'key_findings': [],
            'recommendations': [],
            'confidence_score': 0.0
        }
        
        # Extract key findings from sub-problems
        for sub_problem in resolved_sub_problems:
            if 'solution' in sub_problem and 'execution_results' in sub_problem['solution']:
                for tool_id, result in sub_problem['solution']['execution_results'].items():
                    if isinstance(result, dict) and 'result' in result:
                        synthesis['key_findings'].append({
                            'sub_problem': sub_problem['sub_problem'],
                            'tool': tool_id,
                            'finding': str(result['result'])
                        })
        
        # Generate synthesis summary
        if synthesis['key_findings']:
            synthesis['synthesis_summary'] = f"Successfully resolved {len(resolved_sub_problems)} sub-problems using {len(set(f['tool'] for f in synthesis['key_findings']))} different tools."
            synthesis['confidence_score'] = min(len(resolved_sub_problems) * 0.2, 1.0)
        
        return synthesis
    
    def _record_execution(self, problem_description: str, tools_used: List[str], result: Dict[str, Any], context: Dict[str, Any]):
        """Record execution for learning and optimization."""
        execution_record = {
            'timestamp': time.time(),
            'problem_description': problem_description,
            'tools_used': tools_used,
            'result_success': result.get('success', False),
            'execution_time': context.get('execution_time', 0),
            'context_summary': {
                'priority': context.get('priority', 1),
                'complexity': context.get('complexity', 1),
                'mode': context.get('mode', 'unknown')
            }
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

class AgenticWorkflowSystem:
    """Main system that orchestrates query path and runner agent."""
    
    def __init__(self, base_path: str = "knowledge-objects"):
        self.base_path = Path(base_path)
        self.context_memory = EnhancedContextMemoryManager(str(self.base_path))
        self.enhanced_memory = EnhancedAgenticMemorySystem(self.context_memory)
        self.mcp_integration = MCPIntegrationLayer(self.context_memory)
        self.query_path = QueryPath(self.mcp_integration, self.context_memory, self.enhanced_memory)
        self.runner_agent = RunnerAgent(self.mcp_integration, self.context_memory, self.enhanced_memory)
        
        logger.info("🚀 Agentic Workflow System initialized with Enhanced Multi-Tier Memory")
    
    async def execute_workflow(self, problem_description: str, context: Dict[str, Any], mode: ExecutionMode) -> Dict[str, Any]:
        """Execute workflow using the specified mode."""
        logger.info(f"🚀 Executing {mode.value} workflow: {problem_description}")
        
        # Execute workflow using runner agent
        result = await self.runner_agent.execute_workflow(problem_description, context, mode)
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'query_path': {
                'tool_selections': len(self.query_path.tool_selection_history),
                'optimization_rules': len(self.query_path.path_optimization_rules)
            },
            'runner_agent': {
                'executions': len(self.runner_agent.execution_history),
                'adaptation_rules': len(self.runner_agent.adaptation_rules)
            },
            'context_memory': self.context_memory.get_performance_stats(),
            'enhanced_memory': self.enhanced_memory.get_memory_statistics(),
            'mcp_integration': self.mcp_integration.get_integration_status()
        }
    
    def close(self):
        """Clean up resources."""
        try:
            self.enhanced_memory.end_workflow_session()
            self.context_memory.close()
            logger.info("✅ Agentic Workflow System closed")
        except Exception as e:
            logger.error(f"❌ Error closing system: {e}")

# Example usage
async def main():
    """Example usage of the Agentic Workflow System with Enhanced Memory."""
    system = AgenticWorkflowSystem(".")
    
    try:
        # Test automated workflow
        csv_input = """id,name,description
1,APT29,Russian APT group
2,APT28,Fancy Bear group
3,APT41,Chinese APT group"""
        
        automated_result = await system.execute_workflow(
            "Analyze threat groups and provide risk assessment",
            {
                'csv_input': csv_input,
                'priority': 4,
                'complexity': 7,
                'mode': 'automated'
            },
            ExecutionMode.AUTOMATED
        )
        
        print("📊 Automated Workflow Result:")
        print(json.dumps(automated_result, indent=2, default=str))
        
        # Test manual workflow
        manual_result = await system.execute_workflow(
            "Investigate potential APT29 activity and provide containment recommendations",
            {
                'priority': 5,
                'complexity': 8,
                'mode': 'manual'
            },
            ExecutionMode.MANUAL
        )
        
        print("\n📊 Manual Workflow Result:")
        print(json.dumps(manual_result, indent=2, default=str))
        
        # Get system status with enhanced memory
        status = system.get_system_status()
        print("\n📊 System Status with Enhanced Memory:")
        print(json.dumps(status, indent=2, default=str))
        
    finally:
        system.close()

if __name__ == "__main__":
    asyncio.run(main())
