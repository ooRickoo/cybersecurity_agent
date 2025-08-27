#!/usr/bin/env python3
"""
Cryptography Evaluation MCP Integration Layer
Provides MCP-compatible interface for cryptography evaluation tools.

Features:
- Unified MCP interface for cryptography evaluation
- Dynamic tool discovery and execution
- Query Path integration for intelligent tool selection
- Comprehensive security assessment capabilities
- Integration with agentic workflows
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
import base64
import secrets

# Import cryptography evaluation tools
from cryptography_evaluation import (
    CryptographyEvaluationManager,
    SecurityLevel,
    VulnerabilityType,
    EvaluationCategory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptographyEvaluationToolCategory(Enum):
    """Categories of cryptography evaluation tools."""
    ALGORITHM_EVALUATION = "algorithm_evaluation"
    IMPLEMENTATION_EVALUATION = "implementation_evaluation"
    KEY_QUALITY_EVALUATION = "key_quality_evaluation"
    RANDOMNESS_EVALUATION = "randomness_evaluation"
    COMPREHENSIVE_EVALUATION = "comprehensive_evaluation"
    SECURITY_ANALYSIS = "security_analysis"

class CryptographyEvaluationToolCapability(Enum):
    """Capabilities of cryptography evaluation tools."""
    ALGORITHM_STRENGTH_ANALYSIS = "algorithm_strength_analysis"
    SECURITY_VULNERABILITY_DETECTION = "security_vulnerability_detection"
    KEY_QUALITY_ASSESSMENT = "key_quality_assessment"
    RANDOMNESS_QUALITY_ANALYSIS = "randomness_quality_analysis"
    IMPLEMENTATION_SECURITY_REVIEW = "implementation_security_review"
    SECURITY_RECOMMENDATIONS = "security_recommendations"
    COMPREHENSIVE_SECURITY_AUDIT = "comprehensive_security_audit"

@dataclass
class CryptographyEvaluationToolMetadata:
    """Metadata for cryptography evaluation tools."""
    tool_id: str
    name: str
    description: str
    category: CryptographyEvaluationToolCategory
    capabilities: List[CryptographyEvaluationToolCapability]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    security_considerations: List[str]
    tags: List[str]

class CryptographyEvaluationMCPIntegrationLayer:
    """MCP integration layer for cryptography evaluation tools."""
    
    def __init__(self):
        self.evaluation_manager = CryptographyEvaluationManager()
        self.tool_registry: Dict[str, CryptographyEvaluationToolMetadata] = {}
        self.performance_stats = {
            'total_tool_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }
        
        self._initialize_tool_registry()
        logger.info("🚀 Cryptography Evaluation MCP Integration Layer initialized")
    
    def _initialize_tool_registry(self):
        """Initialize the tool registry with available cryptography evaluation tools."""
        tools = [
            CryptographyEvaluationToolMetadata(
                tool_id="evaluate_algorithm_security",
                name="Algorithm Security Evaluator",
                description="Evaluate the security strength of cryptographic algorithms",
                category=CryptographyEvaluationToolCategory.ALGORITHM_EVALUATION,
                capabilities=[
                    CryptographyEvaluationToolCapability.ALGORITHM_STRENGTH_ANALYSIS,
                    CryptographyEvaluationToolCapability.SECURITY_VULNERABILITY_DETECTION,
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "algorithm": {"type": "string", "description": "Cryptographic algorithm name"},
                        "key_length": {"type": "integer", "description": "Key length in bits"},
                        "mode": {"type": "string", "description": "Encryption mode (if applicable)"}
                    },
                    "required": ["algorithm"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "security_score": {"type": "number", "description": "Security score (0-100)"},
                        "vulnerabilities": {"type": "array", "description": "List of security vulnerabilities"},
                        "recommendations": {"type": "array", "description": "Security recommendations"}
                    }
                },
                performance_metrics={"expected_response_time": "0.1-1.0s"},
                security_considerations=["Analyzes cryptographic algorithms for security weaknesses"],
                tags=["algorithm", "security", "cryptography", "evaluation"]
            ),
            
            CryptographyEvaluationToolMetadata(
                tool_id="evaluate_implementation_security",
                name="Implementation Security Evaluator",
                description="Evaluate the security of cryptographic implementations",
                category=CryptographyEvaluationToolCategory.IMPLEMENTATION_EVALUATION,
                capabilities=[
                    CryptographyEvaluationToolCapability.IMPLEMENTATION_SECURITY_REVIEW,
                    CryptographyEvaluationToolCapability.SECURITY_VULNERABILITY_DETECTION,
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "implementation_data": {
                            "type": "object",
                            "description": "Implementation details including padding, IV generation, etc."
                        }
                    },
                    "required": ["implementation_data"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "security_score": {"type": "number", "description": "Security score (0-100)"},
                        "vulnerabilities": {"type": "array", "description": "List of security vulnerabilities"},
                        "recommendations": {"type": "array", "description": "Security recommendations"}
                    }
                },
                performance_metrics={"expected_response_time": "0.1-0.5s"},
                security_considerations=["Reviews implementation for common security pitfalls"],
                tags=["implementation", "security", "cryptography", "evaluation"]
            ),
            
            CryptographyEvaluationToolMetadata(
                tool_id="evaluate_key_quality",
                name="Key Quality Evaluator",
                description="Evaluate the quality and randomness of cryptographic keys",
                category=CryptographyEvaluationToolCategory.KEY_QUALITY_EVALUATION,
                capabilities=[
                    CryptographyEvaluationToolCapability.KEY_QUALITY_ASSESSMENT,
                    CryptographyEvaluationToolCapability.RANDOMNESS_QUALITY_ANALYSIS,
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "key_data": {"type": "string", "description": "Base64-encoded key data"},
                        "algorithm": {"type": "string", "description": "Algorithm the key is used with"}
                    },
                    "required": ["key_data", "algorithm"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "security_score": {"type": "number", "description": "Security score (0-100)"},
                        "key_length": {"type": "integer", "description": "Key length in bits"},
                        "entropy_estimate": {"type": "number", "description": "Estimated entropy"},
                        "recommendations": {"type": "array", "description": "Security recommendations"}
                    }
                },
                performance_metrics={"expected_response_time": "0.1-0.3s"},
                security_considerations=["Analyzes key randomness and quality"],
                tags=["key", "quality", "randomness", "cryptography", "evaluation"]
            ),
            
            CryptographyEvaluationToolMetadata(
                tool_id="evaluate_randomness_quality",
                name="Randomness Quality Evaluator",
                description="Evaluate the quality of random data and number generation",
                category=CryptographyEvaluationToolCategory.RANDOMNESS_EVALUATION,
                capabilities=[
                    CryptographyEvaluationToolCapability.RANDOMNESS_QUALITY_ANALYSIS,
                    CryptographyEvaluationToolCapability.SECURITY_VULNERABILITY_DETECTION,
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "random_data": {"type": "string", "description": "Base64-encoded random data"}
                    },
                    "required": ["random_data"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "security_score": {"type": "number", "description": "Security score (0-100)"},
                        "entropy_estimate": {"type": "number", "description": "Estimated entropy"},
                        "distribution_analysis": {"type": "object", "description": "Statistical distribution analysis"},
                        "recommendations": {"type": "array", "description": "Security recommendations"}
                    }
                },
                performance_metrics={"expected_response_time": "0.1-0.3s"},
                security_considerations=["Analyzes randomness quality for cryptographic use"],
                tags=["randomness", "quality", "cryptography", "evaluation"]
            ),
            
            CryptographyEvaluationToolMetadata(
                tool_id="execute_evaluation_template",
                name="Evaluation Template Executor",
                description="Execute predefined evaluation templates for comprehensive analysis",
                category=CryptographyEvaluationToolCategory.COMPREHENSIVE_EVALUATION,
                capabilities=[
                    CryptographyEvaluationToolCapability.COMPREHENSIVE_SECURITY_AUDIT,
                    CryptographyEvaluationToolCapability.ALGORITHM_STRENGTH_ANALYSIS,
                    CryptographyEvaluationToolCapability.IMPLEMENTATION_SECURITY_REVIEW,
                    CryptographyEvaluationToolCapability.KEY_QUALITY_ASSESSMENT,
                    CryptographyEvaluationToolCapability.RANDOMNESS_QUALITY_ANALYSIS,
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "template_name": {"type": "string", "enum": ["comprehensive_security", "algorithm_focus", "implementation_focus", "key_quality_focus", "quick_assessment"], "description": "Evaluation template to execute"},
                        "parameters": {"type": "object", "description": "Parameters for the evaluation"}
                    },
                    "required": ["template_name"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "evaluations": {"type": "array", "description": "List of evaluation results"},
                        "summary": {"type": "object", "description": "Summary of all evaluations"},
                        "overall_security_score": {"type": "number", "description": "Overall security score"}
                    }
                },
                performance_metrics={"expected_response_time": "0.5-2.0s"},
                security_considerations=["Comprehensive security analysis of cryptographic systems"],
                tags=["template", "comprehensive", "security", "cryptography", "evaluation"]
            ),
            
            CryptographyEvaluationToolMetadata(
                tool_id="get_evaluation_statistics",
                name="Evaluation Statistics Provider",
                description="Get performance statistics and analysis of evaluation patterns",
                category=CryptographyEvaluationToolCategory.SECURITY_ANALYSIS,
                capabilities=[
                    CryptographyEvaluationToolCapability.SECURITY_RECOMMENDATIONS
                ],
                input_schema={
                    "type": "object",
                    "properties": {}
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "performance_stats": {"type": "object", "description": "Performance statistics"},
                        "evaluation_patterns": {"type": "object", "description": "Analysis of evaluation patterns"},
                        "recommendations": {"type": "array", "description": "Recommendations based on analysis"}
                    }
                },
                performance_metrics={"expected_response_time": "0.1-0.2s"},
                security_considerations=["Provides insights into evaluation patterns and performance"],
                tags=["statistics", "analysis", "cryptography", "evaluation"]
            )
        ]
        
        for tool in tools:
            self.tool_registry[tool.tool_id] = tool
        
        logger.info(f"Registered {len(tools)} cryptography evaluation tools")
    
    def discover_tools(self) -> List[CryptographyEvaluationToolMetadata]:
        """Discover all available cryptography evaluation tools."""
        return list(self.tool_registry.values())
    
    def get_tool_categories(self) -> List[str]:
        """Get list of available tool categories."""
        return list(set(tool.category.value for tool in self.tool_registry.values()))
    
    def get_tool_capabilities(self) -> List[str]:
        """Get list of available tool capabilities."""
        capabilities = set()
        for tool in self.tool_registry.values():
            capabilities.update(cap.value for cap in tool.capabilities)
        return list(capabilities)
    
    def get_tools_by_category(self, category: str) -> List[CryptographyEvaluationToolMetadata]:
        """Get tools filtered by category."""
        return [tool for tool in self.tool_registry.values() if tool.category.value == category]
    
    def get_tools_by_capability(self, capability: str) -> List[CryptographyEvaluationToolMetadata]:
        """Get tools filtered by capability."""
        return [tool for tool in self.tool_registry.values() 
                if any(cap.value == capability for cap in tool.capabilities)]
    
    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cryptography evaluation tool."""
        start_time = time.time()
        self.performance_stats['total_tool_calls'] += 1
        
        try:
            if tool_id not in self.tool_registry:
                raise ValueError(f"Unknown tool: {tool_id}")
            
            tool_metadata = self.tool_registry[tool_id]
            logger.info(f"Executing tool: {tool_id} with parameters: {parameters}")
            
            result = None
            
            if tool_id == "evaluate_algorithm_security":
                result = self._execute_algorithm_security_evaluation(parameters)
                
            elif tool_id == "evaluate_implementation_security":
                result = self._execute_implementation_security_evaluation(parameters)
                
            elif tool_id == "evaluate_key_quality":
                result = self._execute_key_quality_evaluation(parameters)
                
            elif tool_id == "evaluate_randomness_quality":
                result = self._execute_randomness_quality_evaluation(parameters)
                
            elif tool_id == "execute_evaluation_template":
                result = self._execute_evaluation_template(parameters)
                
            elif tool_id == "get_evaluation_statistics":
                result = self._execute_get_evaluation_statistics(parameters)
                
            else:
                raise ValueError(f"Tool {tool_id} not implemented")
            
            # Update performance stats
            response_time = time.time() - start_time
            self.performance_stats['successful_calls'] += 1
            self.performance_stats['average_response_time'] = (
                (self.performance_stats['average_response_time'] * (self.performance_stats['successful_calls'] - 1) + response_time) /
                self.performance_stats['successful_calls']
            )
            
            logger.info(f"Tool {tool_id} executed successfully in {response_time:.3f}s")
            return result
            
        except Exception as e:
            self.performance_stats['failed_calls'] += 1
            logger.error(f"Tool {tool_id} execution failed: {e}")
            raise
    
    def _execute_algorithm_security_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute algorithm security evaluation."""
        algorithm = parameters.get('algorithm')
        key_length = parameters.get('key_length')
        mode = parameters.get('mode')
        
        if not algorithm:
            raise ValueError("Algorithm parameter is required")
        
        evaluation = self.evaluation_manager.evaluator.evaluate_algorithm_security(
            algorithm, key_length, mode
        )
        
        return {
            'security_score': evaluation.security_score,
            'vulnerabilities': [
                {
                    'type': vuln.type.value,
                    'severity': vuln.severity.value,
                    'description': vuln.description,
                    'recommendation': vuln.recommendation
                }
                for vuln in evaluation.vulnerabilities
            ],
            'recommendations': evaluation.recommendations,
            'metadata': evaluation.metadata
        }
    
    def _execute_implementation_security_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute implementation security evaluation."""
        implementation_data = parameters.get('implementation_data', {})
        
        evaluation = self.evaluation_manager.evaluator.evaluate_implementation_security(
            implementation_data
        )
        
        return {
            'security_score': evaluation.security_score,
            'vulnerabilities': [
                {
                    'type': vuln.type.value,
                    'severity': vuln.severity.value,
                    'description': vuln.description,
                    'recommendation': vuln.recommendation
                }
                for vuln in evaluation.vulnerabilities
            ],
            'recommendations': evaluation.recommendations,
            'metadata': evaluation.metadata
        }
    
    def _execute_key_quality_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute key quality evaluation."""
        key_data_b64 = parameters.get('key_data')
        algorithm = parameters.get('algorithm')
        
        if not key_data_b64 or not algorithm:
            raise ValueError("Both key_data and algorithm parameters are required")
        
        try:
            key_data = base64.b64decode(key_data_b64)
        except Exception as e:
            raise ValueError(f"Invalid base64 key data: {e}")
        
        evaluation = self.evaluation_manager.evaluator.evaluate_key_quality(
            key_data, algorithm
        )
        
        return {
            'security_score': evaluation.security_score,
            'key_length': evaluation.metadata['key_analysis']['key_length'],
            'entropy_estimate': evaluation.metadata['key_analysis']['entropy_estimate'],
            'randomness_quality': evaluation.metadata['key_analysis']['randomness_quality'],
            'recommendations': evaluation.recommendations,
            'metadata': evaluation.metadata
        }
    
    def _execute_randomness_quality_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute randomness quality evaluation."""
        random_data_b64 = parameters.get('random_data')
        
        if not random_data_b64:
            raise ValueError("random_data parameter is required")
        
        try:
            random_data = base64.b64decode(random_data_b64)
        except Exception as e:
            raise ValueError(f"Invalid base64 random data: {e}")
        
        evaluation = self.evaluation_manager.evaluator.evaluate_randomness_quality(
            random_data
        )
        
        return {
            'security_score': evaluation.security_score,
            'entropy_estimate': evaluation.metadata['randomness_analysis']['entropy_estimate'],
            'distribution_analysis': evaluation.metadata['randomness_analysis']['distribution_analysis'],
            'recommendations': evaluation.recommendations,
            'metadata': evaluation.metadata
        }
    
    def _execute_evaluation_template(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation template."""
        template_name = parameters.get('template_name')
        template_params = parameters.get('parameters', {})
        
        if not template_name:
            raise ValueError("template_name parameter is required")
        
        evaluations = self.evaluation_manager.execute_evaluation_template(
            template_name, **template_params
        )
        
        # Calculate overall security score
        if evaluations:
            overall_score = sum(eval_result.security_score for eval_result in evaluations) / len(evaluations)
        else:
            overall_score = 0.0
        
        # Create summary
        summary = {
            'template_name': template_name,
            'total_evaluations': len(evaluations),
            'evaluation_types': [eval_result.evaluation_type.value for eval_result in evaluations],
            'average_security_score': overall_score
        }
        
        return {
            'evaluations': [
                {
                    'evaluation_type': eval_result.evaluation_type.value,
                    'security_score': eval_result.security_score,
                    'vulnerabilities_count': len(eval_result.vulnerabilities),
                    'recommendations_count': len(eval_result.recommendations)
                }
                for eval_result in evaluations
            ],
            'summary': summary,
            'overall_security_score': overall_score
        }
    
    def _execute_get_evaluation_statistics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get evaluation statistics."""
        performance_stats = self.evaluation_manager.get_evaluation_statistics()
        evaluation_patterns = self.evaluation_manager.analyze_evaluation_patterns()
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if evaluation_patterns:
            if 'vulnerability_patterns' in evaluation_patterns:
                vuln_patterns = evaluation_patterns['vulnerability_patterns']
                if vuln_patterns:
                    most_common_vuln = max(vuln_patterns.items(), key=lambda x: x[1])
                    recommendations.append(f"Most common vulnerability type: {most_common_vuln[0]} ({most_common_vuln[1]} occurrences)")
            
            if 'average_scores_by_type' in evaluation_patterns:
                avg_scores = evaluation_patterns['average_scores_by_type']
                if avg_scores:
                    lowest_avg_score = min(avg_scores.items(), key=lambda x: x[1])
                    if lowest_avg_score[1] < 70:
                        recommendations.append(f"Consider focusing on {lowest_avg_score[0]} evaluations - average score is {lowest_avg_score[1]:.1f}")
        
        return {
            'performance_stats': performance_stats,
            'evaluation_patterns': evaluation_patterns,
            'recommendations': recommendations
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the MCP integration layer."""
        return self.performance_stats.copy()

class CryptographyEvaluationToolsQueryPathIntegration:
    """Integration layer for Query Path to intelligently select cryptography evaluation tools."""
    
    def __init__(self, mcp_layer: CryptographyEvaluationMCPIntegrationLayer):
        self.mcp_layer = mcp_layer
        self.tool_relevance_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("🚀 Cryptography Evaluation Tools Query Path Integration initialized")
    
    def get_tool_relevance_score(self, tool_id: str, query_context: Dict[str, Any]) -> float:
        """Calculate relevance score for a tool based on query context."""
        if tool_id not in self.mcp_layer.tool_registry:
            return 0.0
        
        tool_metadata = self.mcp_layer.tool_registry[tool_id]
        
        # Check cache first
        cache_key = f"{tool_id}_{hash(str(query_context))}"
        if cache_key in self.tool_relevance_cache:
            return self.tool_relevance_cache[cache_key]
        
        relevance_score = 0.0
        
        # Analyze query context for relevant keywords and intent
        query_text = query_context.get('query', '').lower()
        query_intent = query_context.get('intent', '')
        
        # Score based on tool category
        if tool_metadata.category.value in query_text:
            relevance_score += 30
        
        # Score based on capabilities
        for capability in tool_metadata.capabilities:
            if capability.value in query_text:
                relevance_score += 20
        
        # Score based on tags
        for tag in tool_metadata.tags:
            if tag in query_text:
                relevance_score += 15
        
        # Score based on intent matching
        if 'algorithm' in query_intent and 'algorithm' in tool_metadata.category.value:
            relevance_score += 25
        elif 'implementation' in query_intent and 'implementation' in tool_metadata.category.value:
            relevance_score += 25
        elif 'key' in query_intent and 'key' in tool_metadata.category.value:
            relevance_score += 25
        elif 'randomness' in query_intent and 'randomness' in tool_metadata.category.value:
            relevance_score += 25
        elif 'comprehensive' in query_intent and 'comprehensive' in tool_metadata.category.value:
            relevance_score += 25
        
        # Score based on security focus
        if any(security_term in query_text for security_term in ['security', 'vulnerability', 'weakness', 'risk']):
            if 'security' in tool_metadata.category.value or 'security' in str(tool_metadata.capabilities):
                relevance_score += 20
        
        # Normalize score to 0-100 range
        relevance_score = min(100.0, relevance_score)
        
        # Cache the result
        self.tool_relevance_cache[cache_key] = relevance_score
        
        return relevance_score
    
    def get_recommended_tools(self, query_context: Dict[str, Any], limit: int = 5) -> List[Tuple[str, float]]:
        """Get recommended tools for a query, sorted by relevance score."""
        tool_scores = []
        
        for tool_id in self.mcp_layer.tool_registry:
            score = self.get_tool_relevance_score(tool_id, query_context)
            if score > 0:
                tool_scores.append((tool_id, score))
        
        # Sort by score (descending) and limit results
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[:limit]
    
    def get_tool_execution_plan(self, query_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate an execution plan for complex queries requiring multiple tools."""
        query_text = query_context.get('query', '').lower()
        execution_plan = []
        
        # Check if this is a comprehensive security evaluation request
        if any(term in query_text for term in ['comprehensive', 'full', 'complete', 'audit', 'assessment']):
            execution_plan.append({
                'tool_id': 'execute_evaluation_template',
                'parameters': {
                    'template_name': 'comprehensive_security',
                    'parameters': query_context.get('parameters', {})
                },
                'description': 'Execute comprehensive security evaluation template',
                'priority': 'high'
            })
        
        # Check for specific evaluation needs
        if 'algorithm' in query_text:
            execution_plan.append({
                'tool_id': 'evaluate_algorithm_security',
                'parameters': query_context.get('algorithm_params', {}),
                'description': 'Evaluate algorithm security',
                'priority': 'medium'
            })
        
        if 'implementation' in query_text:
            execution_plan.append({
                'tool_id': 'evaluate_implementation_security',
                'parameters': query_context.get('implementation_params', {}),
                'description': 'Evaluate implementation security',
                'priority': 'medium'
            })
        
        if 'key' in query_text:
            execution_plan.append({
                'tool_id': 'evaluate_key_quality',
                'parameters': query_context.get('key_params', {}),
                'description': 'Evaluate key quality',
                'priority': 'medium'
            })
        
        if 'randomness' in query_text:
            execution_plan.append({
                'tool_id': 'evaluate_randomness_quality',
                'parameters': query_context.get('randomness_params', {}),
                'description': 'Evaluate randomness quality',
                'priority': 'medium'
            })
        
        # Add statistics if multiple evaluations are planned
        if len(execution_plan) > 1:
            execution_plan.append({
                'tool_id': 'get_evaluation_statistics',
                'parameters': {},
                'description': 'Get evaluation statistics and patterns',
                'priority': 'low'
            })
        
        return execution_plan
    
    def clear_cache(self):
        """Clear the tool relevance cache."""
        self.tool_relevance_cache.clear()
        logger.info("Cryptography evaluation tools relevance cache cleared")

async def main():
    """Example usage and testing."""
    try:
        # Initialize MCP integration layer
        mcp_layer = CryptographyEvaluationMCPIntegrationLayer()
        
        print("🔍 Available cryptography evaluation tools:")
        tools = mcp_layer.discover_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            print(f"    Category: {tool.category.value}")
            print(f"    Capabilities: {[cap.value for cap in tool.capabilities]}")
            print()
        
        print("📊 Tool categories:")
        categories = mcp_layer.get_tool_categories()
        for category in categories:
            print(f"  - {category}")
        
        print("\n🚀 Tool capabilities:")
        capabilities = mcp_layer.get_tool_capabilities()
        for capability in capabilities:
            print(f"  - {capability}")
        
        # Test tool execution
        print(f"\n🧪 Testing tool execution...")
        
        # Test algorithm security evaluation
        algo_result = mcp_layer.execute_tool("evaluate_algorithm_security", {
            "algorithm": "AES-256",
            "key_length": 256,
            "mode": "GCM"
        })
        
        print(f"  ✅ Algorithm evaluation result:")
        print(f"    - Security score: {algo_result['security_score']:.1f}/100")
        print(f"    - Vulnerabilities: {len(algo_result['vulnerabilities'])}")
        print(f"    - Recommendations: {len(algo_result['recommendations'])}")
        
        # Test weak algorithm evaluation
        weak_algo_result = mcp_layer.execute_tool("evaluate_algorithm_security", {
            "algorithm": "MD5",
            "key_length": 128
        })
        
        print(f"  ⚠️  Weak algorithm evaluation result:")
        print(f"    - Security score: {weak_algo_result['security_score']:.1f}/100")
        print(f"    - Vulnerabilities: {len(weak_algo_result['vulnerabilities'])}")
        
        # Test key quality evaluation
        test_key = secrets.token_bytes(32)
        key_result = mcp_layer.execute_tool("evaluate_key_quality", {
            "key_data": base64.b64encode(test_key).decode(),
            "algorithm": "AES-256"
        })
        
        print(f"  🔑 Key quality evaluation result:")
        print(f"    - Security score: {key_result['security_score']:.1f}/100")
        print(f"    - Key length: {key_result['key_length']} bits")
        print(f"    - Entropy estimate: {key_result['entropy_estimate']:.2f}")
        
        # Test comprehensive evaluation template
        template_result = mcp_layer.execute_tool("execute_evaluation_template", {
            "template_name": "comprehensive_security",
            "parameters": {
                "algorithm": "RSA-2048",
                "key_length": 2048,
                "mode": "OAEP",
                "key_data": base64.b64encode(test_key).decode(),
                "random_data": base64.b64encode(secrets.token_bytes(64)).decode(),
                "implementation_data": {
                    "padding": "OAEP",
                    "iv_generation": "random",
                    "key_derivation": {"iterations": 100000}
                }
            }
        })
        
        print(f"  🔍 Comprehensive evaluation result:")
        print(f"    - Total evaluations: {template_result['summary']['total_evaluations']}")
        print(f"    - Overall security score: {template_result['overall_security_score']:.1f}/100")
        
        # Test Query Path integration
        print(f"\n🧠 Testing Query Path integration...")
        query_integration = CryptographyEvaluationToolsQueryPathIntegration(mcp_layer)
        
        # Test tool relevance scoring
        query_context = {
            "query": "I need to evaluate the security of AES-256 encryption algorithm",
            "intent": "algorithm_security_evaluation"
        }
        
        recommended_tools = query_integration.get_recommended_tools(query_context)
        print(f"  📋 Recommended tools for algorithm security query:")
        for tool_id, score in recommended_tools:
            print(f"    - {tool_id}: {score:.1f} relevance")
        
        # Test execution plan generation
        execution_plan = query_integration.get_tool_execution_plan(query_context)
        print(f"  📝 Execution plan for algorithm security query:")
        for step in execution_plan:
            print(f"    - {step['tool_id']}: {step['description']} (Priority: {step['priority']})")
        
        # Get performance statistics
        stats = mcp_layer.get_performance_stats()
        print(f"\n📈 MCP integration performance statistics:")
        print(f"  - Total tool calls: {stats['total_tool_calls']}")
        print(f"  - Successful calls: {stats['successful_calls']}")
        print(f"  - Failed calls: {stats['failed_calls']}")
        print(f"  - Average response time: {stats['average_response_time']:.3f}s")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
