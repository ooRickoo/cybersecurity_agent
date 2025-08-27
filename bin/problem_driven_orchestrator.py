#!/usr/bin/env python3
"""
Problem-Driven Workflow Orchestrator

Automatically selects and orchestrates the best tools and workflows based on:
- Problem description and context
- Available tools and capabilities
- Data types and formats
- Desired outcomes
- Performance requirements
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProblemType(Enum):
    """Automatically detected problem types."""
    DATA_ENRICHMENT = "data_enrichment"
    THREAT_ANALYSIS = "threat_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_INVESTIGATION = "incident_investigation"
    RISK_ASSESSMENT = "risk_assessment"
    NETWORK_ANALYSIS = "network_analysis"
    LOG_ANALYSIS = "log_analysis"
    FILE_ANALYSIS = "file_analysis"
    MITRE_MAPPING = "mitre_mapping"
    SECURITY_AUDIT = "security_audit"
    UNKNOWN = "unknown"

class DataType(Enum):
    """Data types that can be processed."""
    CSV = "csv"
    JSON = "json"
    LOG = "log"
    PCAP = "pcap"
    TEXT = "text"
    BINARY = "binary"
    NETWORK = "network"
    POLICY = "policy"
    CATALOG = "catalog"

@dataclass
class ProblemContext:
    """Context for problem analysis and workflow selection."""
    problem_description: str
    input_files: List[str]
    data_types: List[DataType]
    desired_outputs: List[str]
    complexity_level: str  # simple, moderate, complex
    urgency: str  # low, medium, high, critical
    constraints: Dict[str, Any]
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}

@dataclass
class WorkflowRecommendation:
    """Recommended workflow with reasoning."""
    workflow_name: str
    confidence_score: float
    reasoning: str
    required_tools: List[str]
    estimated_time: str
    complexity: str
    prerequisites: List[str]

@dataclass
class ToolRecommendation:
    """Recommended tool with reasoning."""
    tool_name: str
    category: str
    confidence_score: float
    reasoning: str
    input_requirements: List[str]
    output_capabilities: List[str]

class ProblemDrivenOrchestrator:
    """Orchestrates workflows and tools based on problem context."""
    
    def __init__(self, tool_manager=None, workflow_manager=None):
        self.tool_manager = tool_manager
        self.workflow_manager = workflow_manager
        self.problem_patterns = self._initialize_problem_patterns()
        self.workflow_mappings = self._initialize_workflow_mappings()
        self.tool_capabilities = self._initialize_tool_capabilities()
    
    def _initialize_problem_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for problem detection."""
        return {
            "data_enrichment": {
                "keywords": ["enrich", "add columns", "enhance", "augment", "supplement"],
                "file_patterns": [r"\.csv$", r"\.xlsx?$", r"\.json$"],
                "output_patterns": ["enriched", "enhanced", "augmented"],
                "complexity": "moderate"
            },
            "threat_analysis": {
                "keywords": ["threat", "attack", "malware", "suspicious", "anomaly", "breach"],
                "file_patterns": [r"\.log$", r"\.pcap$", r"\.csv$", r"\.json$"],
                "output_patterns": ["threat_level", "risk_score", "attack_type"],
                "complexity": "complex"
            },
            "compliance_check": {
                "keywords": ["compliance", "policy", "standard", "requirement", "audit"],
                "file_patterns": [r"\.txt$", r"\.md$", r"\.pdf$", r"\.docx?$"],
                "output_patterns": ["compliance_status", "violations", "recommendations"],
                "complexity": "moderate"
            },
            "incident_investigation": {
                "keywords": ["incident", "investigation", "forensics", "evidence", "timeline"],
                "file_patterns": [r"\.log$", r"\.pcap$", r"\.csv$", r"\.json$"],
                "output_patterns": ["timeline", "evidence", "conclusions"],
                "complexity": "complex"
            },
            "mitre_mapping": {
                "keywords": ["mitre", "attack", "framework", "tactic", "technique", "mapping"],
                "file_patterns": [r"\.csv$", r"\.txt$", r"\.json$", r"\.md$"],
                "output_patterns": ["mitre_mapping", "tactic", "technique", "reasoning"],
                "complexity": "moderate"
            },
            "log_analysis": {
                "keywords": ["log", "logs", "events", "entries", "traffic", "access"],
                "file_patterns": [r"\.log$", r"\.txt$", r"\.csv$"],
                "output_patterns": ["patterns", "anomalies", "insights"],
                "complexity": "moderate"
            },
            "network_analysis": {
                "keywords": ["network", "traffic", "packets", "connections", "protocols"],
                "file_patterns": [r"\.pcap$", r"\.pcapng$", r"\.log$"],
                "output_patterns": ["traffic_summary", "protocols", "anomalies"],
                "complexity": "complex"
            }
        }
    
    def _initialize_workflow_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mappings from problem types to workflows."""
        return {
            "data_enrichment": {
                "primary_workflow": "csv_enrichment",
                "alternative_workflows": ["dataframe_analysis", "ml_enrichment"],
                "required_tools": ["create_dataframe", "query_dataframe"],
                "output_format": "enriched_csv"
            },
            "threat_analysis": {
                "primary_workflow": "threat_hunting",
                "alternative_workflows": ["anomaly_detection", "pattern_analysis"],
                "required_tools": ["detect_anomalies_isolation_forest", "extract_features_statistical"],
                "output_format": "threat_report"
            },
            "mitre_mapping": {
                "primary_workflow": "mitre_attack_mapping",
                "alternative_workflows": ["threat_classification", "attack_pattern_analysis"],
                "required_tools": ["ai_categorization", "ai_threat_intelligence"],
                "output_format": "mitre_mapped_csv"
            },
            "log_analysis": {
                "primary_workflow": "log_analysis",
                "alternative_workflows": ["pattern_detection", "anomaly_analysis"],
                "required_tools": ["extract_text_features", "detect_patterns_correlation"],
                "output_format": "analysis_report"
            },
            "network_analysis": {
                "primary_workflow": "network_analysis",
                "alternative_workflows": ["traffic_analysis", "protocol_analysis"],
                "required_tools": ["pcap_analysis", "traffic_summarization"],
                "output_format": "network_report"
            }
        }
    
    def _initialize_tool_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tool capability mappings."""
        return {
            "dataframe_manager": {
                "capabilities": ["data_import", "data_export", "data_query", "data_analysis"],
                "input_types": ["csv", "json", "excel"],
                "output_types": ["csv", "json", "excel", "html"],
                "complexity": "simple"
            },
            "ml_tools": {
                "capabilities": ["anomaly_detection", "clustering", "classification", "feature_extraction"],
                "input_types": ["numeric_data", "text_data"],
                "output_types": ["predictions", "clusters", "features"],
                "complexity": "moderate"
            },
            "nlp_tools": {
                "capabilities": ["text_analysis", "sentiment_analysis", "keyword_extraction", "text_classification"],
                "input_types": ["text", "logs", "documents"],
                "output_types": ["analysis", "classification", "keywords"],
                "complexity": "moderate"
            },
            "pcap_analysis_tools": {
                "capabilities": ["traffic_analysis", "protocol_analysis", "file_extraction", "anomaly_detection"],
                "input_types": ["pcap", "pcapng"],
                "output_types": ["traffic_summary", "protocol_stats", "extracted_files"],
                "complexity": "complex"
            },
            "ai_integration": {
                "capabilities": ["reasoning", "categorization", "summarization", "threat_intelligence"],
                "input_types": ["text", "data", "queries"],
                "output_types": ["insights", "analysis", "recommendations"],
                "complexity": "complex"
            }
        }
    
    async def analyze_problem(self, user_query: str, input_files: List[str] = None) -> ProblemContext:
        """Analyze user query and determine problem context."""
        try:
            logger.info(f"🔍 Analyzing problem: {user_query}")
            
            # Detect problem type
            problem_type = self._detect_problem_type(user_query)
            
            # Analyze input files
            data_types = self._analyze_input_files(input_files) if input_files else []
            
            # Determine desired outputs
            desired_outputs = self._extract_desired_outputs(user_query)
            
            # Assess complexity
            complexity_level = self._assess_complexity(user_query, problem_type)
            
            # Determine urgency
            urgency = self._assess_urgency(user_query)
            
            # Identify constraints
            constraints = self._identify_constraints(user_query)
            
            context = ProblemContext(
                problem_description=user_query,
                input_files=input_files or [],
                data_types=data_types,
                desired_outputs=desired_outputs,
                complexity_level=complexity_level,
                urgency=urgency,
                constraints=constraints
            )
            
            logger.info(f"✅ Problem analyzed: {problem_type.value} - {complexity_level} complexity")
            return context
            
        except Exception as e:
            logger.error(f"❌ Error analyzing problem: {e}")
            raise
    
    def _detect_problem_type(self, query: str) -> ProblemType:
        """Detect the type of problem from user query."""
        query_lower = query.lower()
        
        # Check for specific MITRE ATT&CK mapping requests FIRST (highest priority)
        if any(term in query_lower for term in ["mitre", "attack framework", "tactic", "technique", "map to mitre", "mitre mapping"]):
            return ProblemType.MITRE_MAPPING
        
        # Check for file analysis requests
        if any(term in query_lower for term in ["analyze file", "check file", "process file"]):
            return ProblemType.FILE_ANALYSIS
        
        # Check each problem pattern (lower priority)
        for problem_type, pattern_info in self.problem_patterns.items():
            keywords = pattern_info["keywords"]
            if any(keyword in query_lower for keyword in keywords):
                return ProblemType(problem_type)
        
        return ProblemType.UNKNOWN
    
    def _analyze_input_files(self, file_paths: List[str]) -> List[DataType]:
        """Analyze input files to determine data types."""
        data_types = []
        
        for file_path in file_paths:
            if Path(file_path).exists():
                extension = Path(file_path).suffix.lower()
                
                if extension in ['.csv', '.xlsx', '.xls']:
                    data_types.append(DataType.CSV)
                elif extension == '.json':
                    data_types.append(DataType.JSON)
                elif extension in ['.log', '.txt']:
                    data_types.append(DataType.LOG)
                elif extension in ['.pcap', '.pcapng']:
                    data_types.append(DataType.PCAP)
                elif extension in ['.md', '.doc', '.docx']:
                    data_types.append(DataType.POLICY)
                else:
                    data_types.append(DataType.TEXT)
        
        return list(set(data_types))  # Remove duplicates
    
    def _extract_desired_outputs(self, query: str) -> List[str]:
        """Extract desired outputs from user query."""
        outputs = []
        query_lower = query.lower()
        
        # Common output patterns
        output_patterns = {
            "enriched": ["enriched", "enhanced", "augmented", "additional columns"],
            "analysis": ["analysis", "insights", "findings", "patterns"],
            "mapping": ["mapping", "classification", "categorization", "map to"],
            "report": ["report", "summary", "overview"],
            "mitre": ["mitre", "attack framework", "tactics", "techniques", "mitre mapping"]
        }
        
        for output_type, patterns in output_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                outputs.append(output_type)
        
        return outputs
    
    def _assess_complexity(self, query: str, problem_type: ProblemType) -> str:
        """Assess the complexity of the problem."""
        word_count = len(query.split())
        
        # Base complexity from problem type
        base_complexity = self.problem_patterns.get(problem_type.value, {}).get("complexity", "moderate")
        
        # Adjust based on query length and content
        if word_count > 20 or "complex" in query.lower():
            return "complex"
        elif word_count > 10 or "detailed" in query.lower():
            return "moderate"
        else:
            return "simple"
    
    def _assess_urgency(self, query: str) -> str:
        """Assess the urgency of the problem."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["urgent", "critical", "emergency", "immediate"]):
            return "critical"
        elif any(term in query_lower for term in ["asap", "soon", "quick", "fast"]):
            return "high"
        elif any(term in query_lower for term in ["when possible", "no rush", "take time"]):
            return "low"
        else:
            return "medium"
    
    def _identify_constraints(self, query: str) -> Dict[str, Any]:
        """Identify constraints from user query."""
        constraints = {}
        query_lower = query.lower()
        
        # Time constraints
        if "within" in query_lower or "by" in query_lower:
            constraints["time_constraint"] = True
        
        # Resource constraints
        if "memory" in query_lower or "cpu" in query_lower:
            constraints["resource_constraint"] = True
        
        # Quality constraints
        if "high quality" in query_lower or "accurate" in query_lower:
            constraints["quality_constraint"] = True
        
        return constraints
    
    async def recommend_workflow(self, context: ProblemContext) -> WorkflowRecommendation:
        """Recommend the best workflow for the problem."""
        try:
            problem_type = self._detect_problem_type(context.problem_description)
            
            if problem_type == ProblemType.UNKNOWN:
                return WorkflowRecommendation(
                    workflow_name="general_analysis",
                    confidence_score=0.3,
                    reasoning="Problem type not clearly identified, using general analysis workflow",
                    required_tools=["ai_reasoning", "ai_categorization"],
                    estimated_time="variable",
                    complexity="moderate",
                    prerequisites=["data_import", "basic_analysis"]
                )
            
            # Get workflow mapping
            workflow_info = self.workflow_mappings.get(problem_type.value, {})
            primary_workflow = workflow_info.get("primary_workflow", "general_analysis")
            
            # Special handling for MITRE mapping
            if problem_type == ProblemType.MITRE_MAPPING:
                primary_workflow = "mitre_attack_mapping"
            
            # Calculate confidence score
            confidence_score = self._calculate_workflow_confidence(context, problem_type)
            
            # Determine required tools
            required_tools = workflow_info.get("required_tools", [])
            
            # Estimate time
            estimated_time = self._estimate_workflow_time(context, primary_workflow)
            
            # Assess complexity
            complexity = self._assess_workflow_complexity(context, primary_workflow)
            
            # Identify prerequisites
            prerequisites = self._identify_prerequisites(context, primary_workflow)
            
            recommendation = WorkflowRecommendation(
                workflow_name=primary_workflow,
                confidence_score=confidence_score,
                reasoning=f"Problem identified as {problem_type.value}, {primary_workflow} is the most suitable workflow",
                required_tools=required_tools,
                estimated_time=estimated_time,
                complexity=complexity,
                prerequisites=prerequisites
            )
            
            logger.info(f"✅ Workflow recommended: {primary_workflow} (confidence: {confidence_score:.2f})")
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Error recommending workflow: {e}")
            raise
    
    def _calculate_workflow_confidence(self, context: ProblemContext, problem_type: ProblemType) -> float:
        """Calculate confidence score for workflow recommendation."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on clear problem type
        if problem_type != ProblemType.UNKNOWN:
            confidence += 0.2
        
        # Boost confidence based on clear data types
        if context.data_types:
            confidence += 0.1
        
        # Boost confidence based on clear desired outputs
        if context.desired_outputs:
            confidence += 0.1
        
        # Boost confidence based on appropriate complexity
        if context.complexity_level in ["moderate", "complex"]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _estimate_workflow_time(self, context: ProblemContext, workflow_name: str) -> str:
        """Estimate time for workflow execution."""
        # Base estimates
        time_estimates = {
            "csv_enrichment": "5-30 minutes",
            "threat_hunting": "15-60 minutes",
            "mitre_attack_mapping": "10-45 minutes",
            "log_analysis": "10-30 minutes",
            "network_analysis": "20-90 minutes"
        }
        
        base_time = time_estimates.get(workflow_name, "variable")
        
        # Adjust based on data size
        if context.input_files:
            total_size = sum(Path(f).stat().st_size for f in context.input_files if Path(f).exists())
            if total_size > 100 * 1024 * 1024:  # > 100MB
                base_time = "longer than usual"
        
        return base_time
    
    def _assess_workflow_complexity(self, context: ProblemContext, workflow_name: str) -> str:
        """Assess the complexity of the recommended workflow."""
        complexity_mapping = {
            "csv_enrichment": "moderate",
            "threat_hunting": "complex",
            "mitre_attack_mapping": "moderate",
            "log_analysis": "moderate",
            "network_analysis": "complex"
        }
        
        return complexity_mapping.get(workflow_name, "moderate")
    
    def _identify_prerequisites(self, context: ProblemContext, workflow_name: str) -> List[str]:
        """Identify prerequisites for the workflow."""
        prerequisites = []
        
        # Data import prerequisites
        if context.input_files:
            prerequisites.append("data_import")
        
        # Tool availability prerequisites
        if workflow_name == "mitre_attack_mapping":
            prerequisites.append("ai_categorization")
            prerequisites.append("ai_threat_intelligence")
        elif workflow_name == "threat_hunting":
            prerequisites.append("anomaly_detection")
            prerequisites.append("pattern_analysis")
        
        return prerequisites
    
    async def recommend_tools(self, context: ProblemContext, workflow: WorkflowRecommendation) -> List[ToolRecommendation]:
        """Recommend tools for the workflow execution."""
        try:
            tool_recommendations = []
            
            # Get available tools from tool manager
            if self.tool_manager:
                available_tools = self.tool_manager.get_tool_status()
                
                for tool_name, is_available in available_tools.items():
                    if is_available:
                        # Check if tool is relevant to the workflow
                        relevance_score = self._calculate_tool_relevance(tool_name, context, workflow)
                        
                        if relevance_score > 0.3:  # Only recommend relevant tools
                            tool_info = self._get_tool_info(tool_name)
                            
                            recommendation = ToolRecommendation(
                                tool_name=tool_name,
                                category=tool_info.get("category", "general"),
                                confidence_score=relevance_score,
                                reasoning=f"Tool {tool_name} is relevant for {workflow.workflow_name}",
                                input_requirements=tool_info.get("input_requirements", []),
                                output_capabilities=tool_info.get("output_capabilities", [])
                            )
                            
                            tool_recommendations.append(recommendation)
            
            # Sort by confidence score
            tool_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            
            logger.info(f"✅ {len(tool_recommendations)} tools recommended")
            return tool_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error recommending tools: {e}")
            return []
    
    def _calculate_tool_relevance(self, tool_name: str, context: ProblemContext, workflow: WorkflowRecommendation) -> float:
        """Calculate relevance score for a tool."""
        relevance = 0.0
        
        # Check if tool is required by workflow
        if tool_name in workflow.required_tools:
            relevance += 0.5
        
        # Check tool capabilities against problem requirements
        tool_caps = self.tool_capabilities.get(tool_name, {})
        capabilities = tool_caps.get("capabilities", [])
        
        # Match capabilities to problem needs
        if "data_import" in capabilities and context.input_files:
            relevance += 0.2
        
        if "anomaly_detection" in capabilities and "threat" in context.problem_description.lower():
            relevance += 0.2
        
        if "text_analysis" in capabilities and any(dt in [DataType.LOG, DataType.TEXT, DataType.POLICY] for dt in context.data_types):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        # This would typically come from the tool manager
        # For now, return basic info
        return {
            "category": "general",
            "input_requirements": [],
            "output_capabilities": []
        }
    
    async def execute_workflow(self, context: ProblemContext, workflow: WorkflowRecommendation) -> Dict[str, Any]:
        """Execute the recommended workflow."""
        try:
            logger.info(f"🚀 Executing workflow: {workflow.workflow_name}")
            
            # Check prerequisites
            if not await self._check_prerequisites(workflow.prerequisites):
                raise Exception(f"Prerequisites not met: {workflow.prerequisites}")
            
            # Execute workflow based on type
            if workflow.workflow_name == "csv_enrichment":
                return await self._execute_csv_enrichment(context)
            elif workflow.workflow_name == "mitre_attack_mapping":
                return await self._execute_mitre_mapping(context)
            elif workflow.workflow_name == "threat_hunting":
                return await self._execute_threat_hunting(context)
            else:
                return await self._execute_general_workflow(context, workflow)
                
        except Exception as e:
            logger.error(f"❌ Error executing workflow: {e}")
            raise
    
    async def _check_prerequisites(self, prerequisites: List[str]) -> bool:
        """Check if workflow prerequisites are met."""
        # This would check tool availability, data readiness, etc.
        # For now, assume prerequisites are met
        return True
    
    async def _execute_csv_enrichment(self, context: ProblemContext) -> Dict[str, Any]:
        """Execute CSV enrichment workflow."""
        # This would integrate with the existing CSV enrichment executor
        return {
            "workflow": "csv_enrichment",
            "status": "ready_to_execute",
            "message": "CSV enrichment workflow ready - use csv-enrich command"
        }
    
    async def _execute_mitre_mapping(self, context: ProblemContext) -> Dict[str, Any]:
        """Execute MITRE ATT&CK mapping workflow."""
        # This would be the new MITRE mapping workflow
        return {
            "workflow": "mitre_attack_mapping",
            "status": "ready_to_execute",
            "message": "MITRE ATT&CK mapping workflow ready"
        }
    
    async def _execute_threat_hunting(self, context: ProblemContext) -> Dict[str, Any]:
        """Execute threat hunting workflow."""
        return {
            "workflow": "threat_hunting",
            "status": "ready_to_execute",
            "message": "Threat hunting workflow ready"
        }
    
    async def _execute_general_workflow(self, context: ProblemContext, workflow: WorkflowRecommendation) -> Dict[str, Any]:
        """Execute general workflow."""
        return {
            "workflow": workflow.workflow_name,
            "status": "ready_to_execute",
            "message": f"{workflow.workflow_name} workflow ready"
        }

# Example usage
async def main():
    """Example usage of problem-driven orchestrator."""
    orchestrator = ProblemDrivenOrchestrator()
    
    # Example problem
    query = "I have a policy catalog file and need to map it to MITRE ATT&CK framework, enriching with recommendations and reasoning"
    input_files = ["policy_catalog.csv"]
    
    # Analyze problem
    context = await orchestrator.analyze_problem(query, input_files)
    print(f"Problem Type: {context.problem_description}")
    print(f"Data Types: {[dt.value for dt in context.data_types]}")
    print(f"Desired Outputs: {context.desired_outputs}")
    print(f"Complexity: {context.complexity_level}")
    
    # Get workflow recommendation
    workflow = await orchestrator.recommend_workflow(context)
    print(f"Recommended Workflow: {workflow.workflow_name}")
    print(f"Confidence: {workflow.confidence_score:.2f}")
    print(f"Reasoning: {workflow.reasoning}")
    
    # Get tool recommendations
    tools = await orchestrator.recommend_tools(context, workflow)
    print(f"Recommended Tools: {[t.tool_name for t in tools]}")

if __name__ == "__main__":
    asyncio.run(main())
