#!/usr/bin/env python3
"""
Dynamic Workflow Generator

Creates workflows on-the-fly based on problem analysis, making the system more adaptive
while maintaining speed through intelligent caching and pattern recognition.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WorkflowComponentType(Enum):
    """Types of workflow components that can be dynamically assembled."""
    DATA_IMPORT = "data_import"
    ANALYSIS = "analysis"
    PROCESSING = "processing"
    VALIDATION = "validation"
    EXPORT = "export"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    RESPONSE = "response"

class ComponentComplexity(Enum):
    """Complexity levels for workflow components."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class WorkflowComponent:
    """A reusable workflow component that can be dynamically assembled."""
    component_id: str
    component_type: WorkflowComponentType
    name: str
    description: str
    complexity: ComponentComplexity
    execution_time: float  # Estimated execution time in seconds
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}
        if self.validation_rules is None:
            self.validation_rules = []
        if self.optimization_hints is None:
            self.optimization_hints = {}

@dataclass
class DynamicWorkflow:
    """A dynamically generated workflow."""
    workflow_id: str
    problem_description: str
    components: List[WorkflowComponent]
    execution_plan: List[Dict[str, Any]]
    estimated_duration: float
    confidence_score: float
    adaptation_points: List[Dict[str, Any]]
    
    def __post_init__(self):
        if self.adaptation_points is None:
            self.adaptation_points = []

class DynamicWorkflowGenerator:
    """Generates workflows dynamically based on problem analysis."""
    
    def __init__(self):
        self.component_library = self._initialize_component_library()
        self.pattern_cache = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.performance_metrics = {}
        
    def _initialize_component_library(self) -> Dict[str, WorkflowComponent]:
        """Initialize the library of reusable workflow components."""
        components = {}
        
        # Data Import Components
        components["csv_import"] = WorkflowComponent(
            component_id="csv_import",
            component_type=WorkflowComponentType.DATA_IMPORT,
            name="CSV Data Import",
            description="Import data from CSV files with automatic format detection",
            complexity=ComponentComplexity.SIMPLE,
            execution_time=2.0,
            parameters={"file_path": "string", "encoding": "utf-8", "delimiter": "auto"},
            optimization_hints={"parallel": False, "caching": True, "batch_size": 1000}
        )
        
        components["json_import"] = WorkflowComponent(
            component_id="json_import",
            component_type=WorkflowComponentType.DATA_IMPORT,
            name="JSON Data Import",
            description="Import data from JSON files with schema validation",
            complexity=ComponentComplexity.SIMPLE,
            execution_time=1.5,
            parameters={"file_path": "string", "schema_validation": True},
            optimization_hints={"parallel": False, "caching": True}
        )
        
        # Analysis Components
        components["pattern_analysis"] = WorkflowComponent(
            component_id="pattern_analysis",
            component_type=WorkflowComponentType.ANALYSIS,
            name="Pattern Analysis",
            description="Analyze data for patterns, anomalies, and trends",
            complexity=ComponentComplexity.MODERATE,
            execution_time=15.0,
            parameters={"analysis_type": "auto", "sensitivity": 0.8},
            optimization_hints={"parallel": True, "caching": True, "batch_size": 500}
        )
        
        components["threat_intelligence"] = WorkflowComponent(
            component_id="threat_intelligence",
            component_type=WorkflowComponentType.ANALYSIS,
            name="Threat Intelligence Analysis",
            description="Analyze data against threat intelligence feeds",
            complexity=ComponentComplexity.COMPLEX,
            execution_time=45.0,
            parameters={"feed_sources": "auto", "confidence_threshold": 0.7},
            optimization_hints={"parallel": True, "caching": True, "external_api": True}
        )
        
        # Processing Components
        components["data_enrichment"] = WorkflowComponent(
            component_id="data_enrichment",
            component_type=WorkflowComponentType.PROCESSING,
            name="Data Enrichment",
            description="Enhance data with additional context and information",
            complexity=ComponentComplexity.MODERATE,
            execution_time=30.0,
            parameters={"enrichment_sources": "auto", "quality_threshold": 0.8},
            optimization_hints={"parallel": True, "caching": True, "batch_size": 200}
        )
        
        components["ml_classification"] = WorkflowComponent(
            component_id="ml_classification",
            component_type=WorkflowComponentType.PROCESSING,
            name="ML Classification",
            description="Apply machine learning classification to data",
            complexity=ComponentComplexity.COMPLEX,
            execution_time=60.0,
            parameters={"model_type": "auto", "confidence_threshold": 0.7},
            optimization_hints={"parallel": False, "caching": True, "gpu_acceleration": True}
        )
        
        # Validation Components
        components["data_quality_check"] = WorkflowComponent(
            component_id="data_quality_check",
            component_type=WorkflowComponentType.VALIDATION,
            name="Data Quality Validation",
            description="Validate data quality and integrity",
            complexity=ComponentComplexity.SIMPLE,
            execution_time=5.0,
            parameters={"quality_metrics": "auto", "threshold": 0.9},
            optimization_hints={"parallel": True, "caching": False}
        )
        
        # Export Components
        components["csv_export"] = WorkflowComponent(
            component_id="csv_export",
            component_type=WorkflowComponentType.EXPORT,
            name="CSV Export",
            description="Export processed data to CSV format",
            complexity=ComponentComplexity.SIMPLE,
            execution_time=3.0,
            parameters={"file_path": "string", "include_metadata": True},
            optimization_hints={"parallel": False, "caching": False}
        )
        
        components["json_export"] = WorkflowComponent(
            component_id="json_export",
            component_type=WorkflowComponentType.EXPORT,
            name="JSON Export",
            description="Export processed data to JSON format",
            complexity=ComponentComplexity.SIMPLE,
            execution_time=2.0,
            parameters={"file_path": "string", "pretty_print": True},
            optimization_hints={"parallel": False, "caching": False}
        )
        
        # Integration Components
        components["mitre_mapping"] = WorkflowComponent(
            component_id="mitre_mapping",
            component_type=WorkflowComponentType.INTEGRATION,
            name="MITRE ATT&CK Mapping",
            description="Map data to MITRE ATT&CK framework",
            complexity=ComponentComplexity.MODERATE,
            execution_time=25.0,
            parameters={"confidence_threshold": 0.6, "include_subtechniques": True},
            optimization_hints={"parallel": True, "caching": True, "batch_size": 100}
        )
        
        return components
    
    def _initialize_workflow_templates(self) -> Dict[str, List[str]]:
        """Initialize common workflow patterns."""
        return {
            "data_analysis": ["csv_import", "pattern_analysis", "data_quality_check", "csv_export"],
            "threat_hunting": ["csv_import", "threat_intelligence", "pattern_analysis", "data_enrichment", "csv_export"],
            "compliance_check": ["csv_import", "data_quality_check", "mitre_mapping", "data_enrichment", "json_export"],
            "ml_processing": ["csv_import", "data_quality_check", "ml_classification", "data_enrichment", "csv_export"],
            "quick_export": ["csv_import", "csv_export"],
            "data_enrichment": ["csv_import", "data_enrichment", "data_quality_check", "csv_export"]
        }
    
    async def generate_workflow(self, problem_description: str, 
                               input_files: List[str] = None,
                               desired_outputs: List[str] = None,
                               constraints: Dict[str, Any] = None) -> DynamicWorkflow:
        """Generate a workflow dynamically based on problem analysis."""
        
        start_time = time.time()
        
        # Analyze the problem
        problem_analysis = self._analyze_problem(problem_description, input_files, desired_outputs, constraints)
        
        # Check pattern cache first
        cache_key = self._generate_cache_key(problem_analysis)
        if cache_key in self.pattern_cache:
            logger.info(f"Using cached workflow pattern: {cache_key}")
            cached_workflow = self.pattern_cache[cache_key]
            cached_workflow.workflow_id = f"dynamic_{int(time.time())}"
            return cached_workflow
        
        # Generate workflow components
        selected_components = self._select_components(problem_analysis)
        
        # Create execution plan
        execution_plan = self._create_execution_plan(selected_components, problem_analysis)
        
        # Estimate duration and confidence
        estimated_duration = self._estimate_duration(selected_components)
        confidence_score = self._calculate_confidence(selected_components, problem_analysis)
        
        # Create adaptation points
        adaptation_points = self._identify_adaptation_points(selected_components, problem_analysis)
        
        # Create the workflow
        workflow = DynamicWorkflow(
            workflow_id=f"dynamic_{int(time.time())}",
            problem_description=problem_description,
            components=selected_components,
            execution_plan=execution_plan,
            estimated_duration=estimated_duration,
            confidence_score=confidence_score,
            adaptation_points=adaptation_points
        )
        
        # Cache the workflow pattern
        self.pattern_cache[cache_key] = workflow
        
        # Update performance metrics
        generation_time = time.time() - start_time
        self.performance_metrics[workflow.workflow_id] = {
            "generation_time": generation_time,
            "component_count": len(selected_components),
            "complexity_score": self._calculate_complexity_score(selected_components)
        }
        
        logger.info(f"Generated dynamic workflow in {generation_time:.2f}s with {len(selected_components)} components")
        
        return workflow
    
    def _analyze_problem(self, problem_description: str, 
                         input_files: List[str] = None,
                         desired_outputs: List[str] = None,
                         constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the problem to determine requirements."""
        
        analysis = {
            "problem_type": self._classify_problem(problem_description),
            "input_types": self._analyze_input_files(input_files) if input_files else [],
            "output_requirements": self._analyze_outputs(desired_outputs) if desired_outputs else [],
            "constraints": constraints or {},
            "complexity": self._assess_complexity(problem_description),
            "urgency": self._assess_urgency(problem_description),
            "keywords": self._extract_keywords(problem_description)
        }
        
        return analysis
    
    def _classify_problem(self, description: str) -> str:
        """Classify the type of problem."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["threat", "attack", "malware", "intrusion"]):
            return "threat_hunting"
        elif any(word in description_lower for word in ["compliance", "audit", "regulation", "policy"]):
            return "compliance"
        elif any(word in description_lower for word in ["analysis", "investigation", "forensics"]):
            return "investigation"
        elif any(word in description_lower for word in ["enrich", "enhance", "add columns"]):
            return "data_enrichment"
        elif any(word in description_lower for word in ["mitre", "attack framework", "tactic"]):
            return "mitre_mapping"
        elif any(word in description_lower for word in ["ml", "machine learning", "classification"]):
            return "ml_processing"
        else:
            return "general_analysis"
    
    def _analyze_input_files(self, input_files: List[str]) -> List[Dict[str, str]]:
        """Analyze input file types and characteristics."""
        file_analysis = []
        
        for file_path in input_files:
            path = Path(file_path)
            file_info = {
                "path": str(file_path),
                "extension": path.suffix.lower(),
                "type": self._determine_file_type(path.suffix),
                "estimated_size": "unknown"  # Could be enhanced with actual file size
            }
            file_analysis.append(file_info)
        
        return file_analysis
    
    def _determine_file_type(self, extension: str) -> str:
        """Determine the type of file based on extension."""
        extension_map = {
            ".csv": "tabular_data",
            ".json": "structured_data",
            ".xlsx": "spreadsheet",
            ".xls": "spreadsheet",
            ".txt": "text_data",
            ".log": "log_data",
            ".pcap": "network_data",
            ".xml": "structured_data"
        }
        return extension_map.get(extension, "unknown")
    
    def _analyze_outputs(self, outputs: List[str]) -> List[Dict[str, str]]:
        """Analyze desired output requirements."""
        output_analysis = []
        
        for output in outputs:
            output_info = {
                "type": self._classify_output_type(output),
                "format": self._determine_output_format(output),
                "requirements": self._extract_output_requirements(output)
            }
            output_analysis.append(output_info)
        
        return output_analysis
    
    def _classify_output_type(self, output: str) -> str:
        """Classify the type of output needed."""
        output_lower = output.lower()
        
        if any(word in output_lower for word in ["report", "summary", "analysis"]):
            return "report"
        elif any(word in output_lower for word in ["enriched", "enhanced", "processed"]):
            return "enriched_data"
        elif any(word in output_lower for word in ["mapped", "mitre", "framework"]):
            return "mapped_data"
        elif any(word in output_lower for word in ["visualization", "chart", "graph"]):
            return "visualization"
        else:
            return "processed_data"
    
    def _determine_output_format(self, output: str) -> str:
        """Determine the output format."""
        output_lower = output.lower()
        
        if any(word in output_lower for word in ["csv", "excel", "spreadsheet"]):
            return "csv"
        elif any(word in output_lower for word in ["json", "api"]):
            return "json"
        elif any(word in output_lower for word in ["pdf", "report", "document"]):
            return "pdf"
        elif any(word in output_lower for word in ["html", "web", "dashboard"]):
            return "html"
        else:
            return "auto"
    
    def _extract_output_requirements(self, output: str) -> List[str]:
        """Extract specific output requirements."""
        requirements = []
        output_lower = output.lower()
        
        if "mitre" in output_lower:
            requirements.append("mitre_mapping")
        if "enriched" in output_lower:
            requirements.append("data_enrichment")
        if "analysis" in output_lower:
            requirements.append("pattern_analysis")
        if "validation" in output_lower:
            requirements.append("data_quality_check")
        
        return requirements
    
    def _assess_complexity(self, description: str) -> str:
        """Assess the complexity of the problem."""
        description_lower = description.lower()
        
        # Count complexity indicators
        complexity_indicators = {
            "simple": ["simple", "basic", "quick", "export", "import"],
            "moderate": ["analysis", "enrich", "process", "validate"],
            "complex": ["ml", "machine learning", "threat intelligence", "forensics", "investigation"]
        }
        
        scores = {}
        for level, indicators in complexity_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in description_lower)
        
        # Determine complexity based on scores
        if scores["complex"] > 0:
            return "complex"
        elif scores["moderate"] > 0:
            return "moderate"
        else:
            return "simple"
    
    def _assess_urgency(self, description: str) -> str:
        """Assess the urgency of the problem."""
        description_lower = description.lower()
        
        urgency_indicators = {
            "high": ["urgent", "critical", "emergency", "immediate", "asap"],
            "medium": ["soon", "today", "this week"],
            "low": ["when convenient", "no rush", "low priority"]
        }
        
        for level, indicators in urgency_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                return level
        
        return "medium"  # Default urgency
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract relevant keywords from the problem description."""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = description.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))
    
    def _select_components(self, problem_analysis: Dict[str, Any]) -> List[WorkflowComponent]:
        """Select appropriate components based on problem analysis."""
        selected_components = []
        
        # Start with input components based on file types
        input_components = self._select_input_components(problem_analysis["input_types"])
        selected_components.extend(input_components)
        
        # Add processing components based on problem type and requirements
        processing_components = self._select_processing_components(problem_analysis)
        selected_components.extend(processing_components)
        
        # Add output components based on requirements
        output_components = self._select_output_components(problem_analysis["output_requirements"])
        selected_components.extend(output_components)
        
        # Add validation components if needed
        if problem_analysis["complexity"] in ["moderate", "complex"]:
            validation_components = self._select_validation_components(problem_analysis)
            selected_components.extend(validation_components)
        
        return selected_components
    
    def _select_input_components(self, input_types: List[Dict[str, str]]) -> List[WorkflowComponent]:
        """Select input components based on file types."""
        components = []
        
        for input_type in input_types:
            if input_type["type"] == "tabular_data":
                components.append(self.component_library["csv_import"])
            elif input_type["type"] == "structured_data":
                components.append(self.component_library["json_import"])
            # Add more input type handlers as needed
        
        return components
    
    def _select_processing_components(self, problem_analysis: Dict[str, Any]) -> List[WorkflowComponent]:
        """Select processing components based on problem analysis."""
        components = []
        
        problem_type = problem_analysis["problem_type"]
        complexity = problem_analysis["complexity"]
        requirements = problem_analysis.get("output_requirements", [])
        
        # Add components based on problem type
        if problem_type == "threat_hunting":
            components.append(self.component_library["threat_intelligence"])
            components.append(self.component_library["pattern_analysis"])
        elif problem_type == "compliance":
            components.append(self.component_library["data_quality_check"])
            components.append(self.component_library["mitre_mapping"])
        elif problem_type == "data_enrichment":
            components.append(self.component_library["data_enrichment"])
        elif problem_type == "ml_processing":
            components.append(self.component_library["ml_classification"])
        
        # Add pattern analysis for moderate+ complexity
        if complexity in ["moderate", "complex"]:
            components.append(self.component_library["pattern_analysis"])
        
        # Add components based on output requirements
        for req in requirements:
            if req == "mitre_mapping":
                components.append(self.component_library["mitre_mapping"])
            elif req == "data_enrichment":
                components.append(self.component_library["data_enrichment"])
            elif req == "pattern_analysis":
                components.append(self.component_library["pattern_analysis"])
        
        return components
    
    def _select_output_components(self, output_requirements: List[Dict[str, Any]]) -> List[WorkflowComponent]:
        """Select output components based on requirements."""
        components = []
        
        for req in output_requirements:
            format_type = req["format"]
            
            if format_type == "csv":
                components.append(self.component_library["csv_export"])
            elif format_type == "json":
                components.append(self.component_library["json_export"])
            elif format_type == "auto":
                # Default to CSV for most cases
                components.append(self.component_library["csv_export"])
        
        return components
    
    def _select_validation_components(self, problem_analysis: Dict[str, Any]) -> List[WorkflowComponent]:
        """Select validation components based on problem analysis."""
        components = []
        
        # Always add data quality check for moderate+ complexity
        components.append(self.component_library["data_quality_check"])
        
        return components
    
    def _create_execution_plan(self, components: List[WorkflowComponent], 
                              problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create an execution plan for the selected components."""
        execution_plan = []
        
        # Group components by type for parallel execution where possible
        component_groups = {
            WorkflowComponentType.DATA_IMPORT: [],
            WorkflowComponentType.ANALYSIS: [],
            WorkflowComponentType.PROCESSING: [],
            WorkflowComponentType.VALIDATION: [],
            WorkflowComponentType.EXPORT: []
        }
        
        for component in components:
            component_groups[component.component_type].append(component)
        
        # Create execution steps
        step_number = 1
        
        # Step 1: Data Import (sequential)
        if component_groups[WorkflowComponentType.DATA_IMPORT]:
            execution_plan.append({
                "step": step_number,
                "name": "Data Import",
                "components": [comp.component_id for comp in component_groups[WorkflowComponentType.DATA_IMPORT]],
                "execution_type": "sequential",
                "dependencies": [],
                "estimated_time": sum(comp.execution_time for comp in component_groups[WorkflowComponentType.DATA_IMPORT])
            })
            step_number += 1
        
        # Step 2: Analysis and Processing (parallel where possible)
        analysis_processing = component_groups[WorkflowComponentType.ANALYSIS] + component_groups[WorkflowComponentType.PROCESSING]
        if analysis_processing:
            # Group by complexity for parallel execution
            simple_components = [comp for comp in analysis_processing if comp.complexity == ComponentComplexity.SIMPLE]
            complex_components = [comp for comp in analysis_processing if comp.complexity != ComponentComplexity.SIMPLE]
            
            if simple_components:
                execution_plan.append({
                    "step": step_number,
                    "name": "Parallel Analysis",
                    "components": [comp.component_id for comp in simple_components],
                    "execution_type": "parallel",
                    "dependencies": [step_number - 1] if step_number > 1 else [],
                    "estimated_time": max(comp.execution_time for comp in simple_components)
                })
                step_number += 1
            
            if complex_components:
                for comp in complex_components:
                    execution_plan.append({
                        "step": step_number,
                        "name": f"{comp.name}",
                        "components": [comp.component_id],
                        "execution_type": "sequential",
                        "dependencies": [step_number - 1] if step_number > 1 else [],
                        "estimated_time": comp.execution_time
                    })
                    step_number += 1
        
        # Step 3: Validation (parallel)
        if component_groups[WorkflowComponentType.VALIDATION]:
            execution_plan.append({
                "step": step_number,
                "name": "Data Validation",
                "components": [comp.component_id for comp in component_groups[WorkflowComponentType.VALIDATION]],
                "execution_type": "parallel",
                "dependencies": [step_number - 1] if step_number > 1 else [],
                "estimated_time": max(comp.execution_time for comp in component_groups[WorkflowComponentType.VALIDATION])
            })
            step_number += 1
        
        # Step 4: Export (sequential)
        if component_groups[WorkflowComponentType.EXPORT]:
            execution_plan.append({
                "step": step_number,
                "name": "Data Export",
                "components": [comp.component_id for comp in component_groups[WorkflowComponentType.EXPORT]],
                "execution_type": "sequential",
                "dependencies": [step_number - 1] if step_number > 1 else [],
                "estimated_time": sum(comp.execution_time for comp in component_groups[WorkflowComponentType.EXPORT])
            })
        
        return execution_plan
    
    def _estimate_duration(self, components: List[WorkflowComponent]) -> float:
        """Estimate the total execution duration."""
        # Simple estimation - could be enhanced with historical data
        total_time = 0
        
        for component in components:
            if component.component_type == WorkflowComponentType.DATA_IMPORT:
                total_time += component.execution_time
            elif component.component_type == WorkflowComponentType.ANALYSIS:
                total_time += component.execution_time * 0.8  # Parallel execution discount
            elif component.component_type == WorkflowComponentType.PROCESSING:
                total_time += component.execution_time
            elif component.component_type == WorkflowComponentType.VALIDATION:
                total_time += component.execution_time * 0.7  # Parallel execution discount
            elif component.component_type == WorkflowComponentType.EXPORT:
                total_time += component.execution_time
        
        return total_time
    
    def _calculate_confidence(self, components: List[WorkflowComponent], 
                             problem_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated workflow."""
        confidence = 0.5  # Base confidence
        
        # Component coverage score
        required_types = set()
        if problem_analysis["problem_type"] == "threat_hunting":
            required_types.update([WorkflowComponentType.ANALYSIS, WorkflowComponentType.PROCESSING])
        elif problem_analysis["problem_type"] == "data_enrichment":
            required_types.update([WorkflowComponentType.PROCESSING])
        
        covered_types = set(comp.component_type for comp in components)
        coverage_score = len(required_types.intersection(covered_types)) / len(required_types) if required_types else 1.0
        confidence += coverage_score * 0.3
        
        # Complexity match score
        complexity_match = 1.0
        if problem_analysis["complexity"] == "simple" and any(comp.complexity == ComponentComplexity.COMPLEX for comp in components):
            complexity_match = 0.7
        elif problem_analysis["complexity"] == "complex" and all(comp.complexity == ComponentComplexity.SIMPLE for comp in components):
            complexity_match = 0.6
        
        confidence += complexity_match * 0.2
        
        return min(confidence, 1.0)
    
    def _identify_adaptation_points(self, components: List[WorkflowComponent], 
                                   problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify points where the workflow can adapt during execution."""
        adaptation_points = []
        
        for i, component in enumerate(components):
            if component.complexity == ComponentComplexity.COMPLEX:
                adaptation_points.append({
                    "component_id": component.component_id,
                    "adaptation_type": "fallback",
                    "trigger": "execution_timeout",
                    "fallback_component": self._find_fallback_component(component),
                    "description": f"Fallback to simpler alternative if {component.name} takes too long"
                })
            
            if component.component_type == WorkflowComponentType.ANALYSIS:
                adaptation_points.append({
                    "component_id": component.component_id,
                    "adaptation_type": "parameter_adjustment",
                    "trigger": "low_confidence",
                    "adjustment": "increase_sensitivity",
                    "description": f"Adjust {component.name} parameters if results have low confidence"
                })
        
        return adaptation_points
    
    def _find_fallback_component(self, component: WorkflowComponent) -> Optional[str]:
        """Find a simpler fallback component."""
        if component.component_id == "ml_classification":
            return "pattern_analysis"
        elif component.component_id == "threat_intelligence":
            return "pattern_analysis"
        elif component.component_id == "data_enrichment":
            return "data_quality_check"
        
        return None
    
    def _calculate_complexity_score(self, components: List[WorkflowComponent]) -> float:
        """Calculate overall complexity score for the workflow."""
        complexity_weights = {
            ComponentComplexity.SIMPLE: 1.0,
            ComponentComplexity.MODERATE: 2.0,
            ComponentComplexity.COMPLEX: 3.0
        }
        
        total_score = sum(complexity_weights[comp.complexity] for comp in components)
        return total_score / len(components) if components else 0.0
    
    def _generate_cache_key(self, problem_analysis: Dict[str, Any]) -> str:
        """Generate a cache key for the problem analysis."""
        key_parts = [
            problem_analysis["problem_type"],
            problem_analysis["complexity"],
            ",".join(sorted(problem_analysis["keywords"][:5])),  # Limit keywords for cache efficiency
            str(len(problem_analysis.get("input_types", []))),
            str(len(problem_analysis.get("output_requirements", [])))
        ]
        return "|".join(key_parts)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for workflow generation."""
        if not self.performance_metrics:
            return {}
        
        total_workflows = len(self.performance_metrics)
        avg_generation_time = sum(metrics["generation_time"] for metrics in self.performance_metrics.values()) / total_workflows
        avg_components = sum(metrics["component_count"] for metrics in self.performance_metrics.values()) / total_workflows
        avg_complexity = sum(metrics["complexity_score"] for metrics in self.performance_metrics.values()) / total_workflows
        
        return {
            "total_workflows_generated": total_workflows,
            "average_generation_time": avg_generation_time,
            "average_component_count": avg_components,
            "average_complexity_score": avg_complexity,
            "cache_hit_rate": len(self.pattern_cache) / (len(self.pattern_cache) + total_workflows) if self.pattern_cache else 0
        }
    
    def clear_cache(self):
        """Clear the pattern cache."""
        self.pattern_cache.clear()
        logger.info("Pattern cache cleared")
    
    def export_workflow(self, workflow: DynamicWorkflow, format: str = "json") -> str:
        """Export the workflow to a specific format."""
        if format.lower() == "json":
            return json.dumps({
                "workflow_id": workflow.workflow_id,
                "problem_description": workflow.problem_description,
                "components": [asdict(comp) for comp in workflow.components],
                "execution_plan": workflow.execution_plan,
                "estimated_duration": workflow.estimated_duration,
                "confidence_score": workflow.confidence_score,
                "adaptation_points": workflow.adaptation_points
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

def asdict(obj):
    """Convert dataclass to dictionary."""
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)
