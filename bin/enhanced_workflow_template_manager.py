#!/usr/bin/env python3
"""
Enhanced Workflow Template Manager - Dynamic and Intelligent Workflow Management

Provides advanced workflow template management with:
- Dynamic template generation based on context
- Intelligent workflow adaptation
- Performance optimization
- Learning from execution patterns
- Multi-modal workflow composition
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)

class WorkflowComplexity(Enum):
    """Workflow complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class WorkflowCategory(Enum):
    """Workflow categories."""
    ANALYSIS = "analysis"
    INVESTIGATION = "investigation"
    RESPONSE = "response"
    PREVENTION = "prevention"
    COMPLIANCE = "compliance"
    RESEARCH = "research"
    AUTOMATION = "automation"

class ExecutionMode(Enum):
    """Execution modes for workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    INTERACTIVE = "interactive"

@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    step_id: str
    name: str
    description: str
    tool_name: str
    tool_category: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    retry_count: int = 0
    timeout: float = 300.0
    parallel_execution: bool = False
    conditional_execution: Optional[str] = None
    output_mapping: Dict[str, str] = field(default_factory=dict)

@dataclass
class WorkflowTemplate:
    """Enhanced workflow template definition."""
    template_id: str
    name: str
    description: str
    category: WorkflowCategory
    complexity: WorkflowComplexity
    steps: List[WorkflowStep]
    execution_mode: ExecutionMode
    required_tools: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    adaptation_rules: List[Dict[str, Any]] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution tracking."""
    execution_id: str
    template_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    adaptations_made: List[Dict[str, Any]] = field(default_factory=list)
    user_feedback: Optional[Dict[str, Any]] = None

class EnhancedWorkflowTemplateManager:
    """Enhanced workflow template manager with dynamic capabilities."""
    
    def __init__(self, templates_dir: str = "workflow_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.performance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.adaptation_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Learning and optimization
        self.usage_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Dynamic generation
        self.component_library: Dict[str, Dict[str, Any]] = {}
        self.composition_rules: List[Dict[str, Any]] = []
        
        # Initialize
        self._initialize_component_library()
        self._initialize_composition_rules()
        self._load_existing_templates()
        self._create_default_templates()
    
    def _initialize_component_library(self):
        """Initialize the component library for dynamic workflow generation."""
        self.component_library = {
            # Data Processing Components
            "data_import": {
                "name": "Data Import",
                "description": "Import data from various sources",
                "category": "data_processing",
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_duration": 5.0,
                "required_tools": ["file_reader", "csv_parser", "json_parser"],
                "outputs": ["imported_data"],
                "tags": ["data", "import", "file"]
            },
            "data_validation": {
                "name": "Data Validation",
                "description": "Validate data quality and integrity",
                "category": "data_processing",
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_duration": 3.0,
                "required_tools": ["data_validator"],
                "outputs": ["validation_report"],
                "tags": ["data", "validation", "quality"]
            },
            "data_enrichment": {
                "name": "Data Enrichment",
                "description": "Enhance data with additional information",
                "category": "data_processing",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 15.0,
                "required_tools": ["api_client", "data_merger"],
                "outputs": ["enriched_data"],
                "tags": ["data", "enrichment", "enhancement"]
            },
            "data_export": {
                "name": "Data Export",
                "description": "Export processed data to various formats",
                "category": "data_processing",
                "complexity": WorkflowComplexity.SIMPLE,
                "estimated_duration": 5.0,
                "required_tools": ["file_writer", "csv_writer", "json_writer"],
                "outputs": ["exported_files"],
                "tags": ["data", "export", "file"]
            },
            
            # Analysis Components
            "pattern_analysis": {
                "name": "Pattern Analysis",
                "description": "Analyze patterns in data",
                "category": "analysis",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 20.0,
                "required_tools": ["pattern_detector", "statistical_analyzer"],
                "outputs": ["pattern_report"],
                "tags": ["analysis", "pattern", "statistics"]
            },
            "anomaly_detection": {
                "name": "Anomaly Detection",
                "description": "Detect anomalies in data",
                "category": "analysis",
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_duration": 30.0,
                "required_tools": ["anomaly_detector", "ml_engine"],
                "outputs": ["anomaly_report"],
                "tags": ["analysis", "anomaly", "ml"]
            },
            "threat_analysis": {
                "name": "Threat Analysis",
                "description": "Analyze threats and security indicators",
                "category": "analysis",
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_duration": 25.0,
                "required_tools": ["threat_analyzer", "ioc_matcher"],
                "outputs": ["threat_report"],
                "tags": ["analysis", "threat", "security"]
            },
            "compliance_check": {
                "name": "Compliance Check",
                "description": "Check compliance against standards",
                "category": "analysis",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 20.0,
                "required_tools": ["compliance_checker", "framework_mapper"],
                "outputs": ["compliance_report"],
                "tags": ["analysis", "compliance", "standards"]
            },
            
            # Security Components
            "malware_analysis": {
                "name": "Malware Analysis",
                "description": "Analyze malware and suspicious files",
                "category": "security",
                "complexity": WorkflowComplexity.EXPERT,
                "estimated_duration": 45.0,
                "required_tools": ["malware_analyzer", "yara_engine", "sandbox"],
                "outputs": ["malware_report"],
                "tags": ["security", "malware", "analysis"]
            },
            "network_analysis": {
                "name": "Network Analysis",
                "description": "Analyze network traffic and protocols",
                "category": "security",
                "complexity": WorkflowComplexity.COMPLEX,
                "estimated_duration": 35.0,
                "required_tools": ["pcap_analyzer", "protocol_analyzer"],
                "outputs": ["network_report"],
                "tags": ["security", "network", "traffic"]
            },
            "vulnerability_scan": {
                "name": "Vulnerability Scan",
                "description": "Scan for vulnerabilities",
                "category": "security",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 30.0,
                "required_tools": ["vulnerability_scanner", "port_scanner"],
                "outputs": ["vulnerability_report"],
                "tags": ["security", "vulnerability", "scan"]
            },
            "incident_response": {
                "name": "Incident Response",
                "description": "Respond to security incidents",
                "category": "security",
                "complexity": WorkflowComplexity.EXPERT,
                "estimated_duration": 60.0,
                "required_tools": ["incident_manager", "forensics_tools"],
                "outputs": ["incident_report"],
                "tags": ["security", "incident", "response"]
            },
            
            # Reporting Components
            "report_generation": {
                "name": "Report Generation",
                "description": "Generate comprehensive reports",
                "category": "reporting",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 15.0,
                "required_tools": ["report_generator", "template_engine"],
                "outputs": ["report"],
                "tags": ["reporting", "documentation"]
            },
            "visualization": {
                "name": "Data Visualization",
                "description": "Create visualizations from data",
                "category": "reporting",
                "complexity": WorkflowComplexity.MODERATE,
                "estimated_duration": 20.0,
                "required_tools": ["chart_generator", "graph_visualizer"],
                "outputs": ["visualizations"],
                "tags": ["reporting", "visualization", "charts"]
            }
        }
    
    def _initialize_composition_rules(self):
        """Initialize rules for composing workflows from components."""
        self.composition_rules = [
            {
                "name": "data_processing_chain",
                "pattern": ["data_import", "data_validation", "data_enrichment", "data_export"],
                "category": WorkflowCategory.ANALYSIS,
                "complexity": WorkflowComplexity.MODERATE
            },
            {
                "name": "security_analysis_chain",
                "pattern": ["data_import", "threat_analysis", "anomaly_detection", "report_generation"],
                "category": WorkflowCategory.INVESTIGATION,
                "complexity": WorkflowComplexity.COMPLEX
            },
            {
                "name": "compliance_workflow",
                "pattern": ["data_import", "compliance_check", "report_generation"],
                "category": WorkflowCategory.COMPLIANCE,
                "complexity": WorkflowComplexity.MODERATE
            },
            {
                "name": "incident_response_workflow",
                "pattern": ["data_import", "incident_response", "malware_analysis", "network_analysis", "report_generation"],
                "category": WorkflowCategory.RESPONSE,
                "complexity": WorkflowComplexity.EXPERT
            },
            {
                "name": "threat_hunting_workflow",
                "pattern": ["data_import", "threat_analysis", "anomaly_detection", "pattern_analysis", "visualization"],
                "category": WorkflowCategory.INVESTIGATION,
                "complexity": WorkflowComplexity.COMPLEX
            }
        ]
    
    def _load_existing_templates(self):
        """Load existing templates from storage."""
        try:
            templates_file = self.templates_dir / "templates.json"
            if templates_file.exists():
                with open(templates_file, 'r') as f:
                    templates_data = json.load(f)
                
                for template_id, template_data in templates_data.items():
                    # Convert back to WorkflowTemplate object
                    steps = [WorkflowStep(**step) for step in template_data.get('steps', [])]
                    template_data['steps'] = steps
                    template_data['category'] = WorkflowCategory(template_data['category'])
                    template_data['complexity'] = WorkflowComplexity(template_data['complexity'])
                    template_data['execution_mode'] = ExecutionMode(template_data['execution_mode'])
                    
                    self.templates[template_id] = WorkflowTemplate(**template_data)
                
                logger.info(f"Loaded {len(self.templates)} existing templates")
        except Exception as e:
            logger.warning(f"Error loading existing templates: {e}")
    
    def _create_default_templates(self):
        """Create default workflow templates."""
        # Patent Analysis Template
        patent_template = WorkflowTemplate(
            template_id="patent_analysis_v1",
            name="Patent Analysis Workflow",
            description="Comprehensive patent analysis with LLM insights",
            category=WorkflowCategory.RESEARCH,
            complexity=WorkflowComplexity.MODERATE,
            steps=[
                WorkflowStep(
                    step_id="import_patents",
                    name="Import Patent Data",
                    description="Import patent data from CSV file",
                    tool_name="csv_reader",
                    tool_category="data_processing",
                    estimated_duration=5.0
                ),
                WorkflowStep(
                    step_id="fetch_patent_details",
                    name="Fetch Patent Details",
                    description="Fetch detailed patent information from USPTO",
                    tool_name="patent_lookup",
                    tool_category="api",
                    dependencies=["import_patents"],
                    estimated_duration=30.0
                ),
                WorkflowStep(
                    step_id="generate_insights",
                    name="Generate LLM Insights",
                    description="Generate value propositions and categories using LLM",
                    tool_name="ai_patent_analysis",
                    tool_category="ai",
                    dependencies=["fetch_patent_details"],
                    estimated_duration=15.0
                ),
                WorkflowStep(
                    step_id="export_results",
                    name="Export Results",
                    description="Export enhanced patent data to CSV",
                    tool_name="csv_writer",
                    tool_category="data_processing",
                    dependencies=["generate_insights"],
                    estimated_duration=5.0
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            required_tools=["csv_reader", "patent_lookup", "ai_patent_analysis", "csv_writer"],
            required_inputs=["patent_csv_file"],
            expected_outputs=["enhanced_patent_csv", "analysis_summary"],
            tags=["patent", "analysis", "research", "llm"]
        )
        
        # Malware Analysis Template
        malware_template = WorkflowTemplate(
            template_id="malware_analysis_v1",
            name="Malware Analysis Workflow",
            description="Comprehensive malware analysis and threat assessment",
            category=WorkflowCategory.INVESTIGATION,
            complexity=WorkflowComplexity.EXPERT,
            steps=[
                WorkflowStep(
                    step_id="file_analysis",
                    name="File Analysis",
                    description="Analyze file properties and metadata",
                    tool_name="file_analyzer",
                    tool_category="forensics",
                    estimated_duration=10.0
                ),
                WorkflowStep(
                    step_id="malware_detection",
                    name="Malware Detection",
                    description="Detect malware using signatures and heuristics",
                    tool_name="malware_analyzer",
                    tool_category="security",
                    dependencies=["file_analysis"],
                    estimated_duration=20.0
                ),
                WorkflowStep(
                    step_id="behavioral_analysis",
                    name="Behavioral Analysis",
                    description="Analyze malware behavior patterns",
                    tool_name="behavior_analyzer",
                    tool_category="security",
                    dependencies=["malware_detection"],
                    estimated_duration=25.0
                ),
                WorkflowStep(
                    step_id="threat_assessment",
                    name="Threat Assessment",
                    description="Assess threat level and impact",
                    tool_name="threat_assessor",
                    tool_category="security",
                    dependencies=["behavioral_analysis"],
                    estimated_duration=15.0
                ),
                WorkflowStep(
                    step_id="generate_report",
                    name="Generate Report",
                    description="Generate comprehensive malware analysis report",
                    tool_name="report_generator",
                    tool_category="reporting",
                    dependencies=["threat_assessment"],
                    estimated_duration=10.0
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            required_tools=["file_analyzer", "malware_analyzer", "behavior_analyzer", "threat_assessor", "report_generator"],
            required_inputs=["suspicious_file"],
            expected_outputs=["malware_report", "threat_assessment"],
            tags=["malware", "analysis", "security", "forensics"]
        )
        
        # Network Analysis Template
        network_template = WorkflowTemplate(
            template_id="network_analysis_v1",
            name="Network Analysis Workflow",
            description="Comprehensive network traffic analysis",
            category=WorkflowCategory.INVESTIGATION,
            complexity=WorkflowComplexity.COMPLEX,
            steps=[
                WorkflowStep(
                    step_id="pcap_analysis",
                    name="PCAP Analysis",
                    description="Analyze PCAP file for traffic patterns",
                    tool_name="pcap_analyzer",
                    tool_category="network",
                    estimated_duration=20.0
                ),
                WorkflowStep(
                    step_id="protocol_analysis",
                    name="Protocol Analysis",
                    description="Analyze network protocols and services",
                    tool_name="protocol_analyzer",
                    tool_category="network",
                    dependencies=["pcap_analysis"],
                    estimated_duration=15.0
                ),
                WorkflowStep(
                    step_id="security_indicators",
                    name="Security Indicators",
                    description="Detect security indicators and anomalies",
                    tool_name="threat_detector",
                    tool_category="security",
                    dependencies=["protocol_analysis"],
                    estimated_duration=10.0
                ),
                WorkflowStep(
                    step_id="traffic_visualization",
                    name="Traffic Visualization",
                    description="Create traffic visualizations",
                    tool_name="chart_generator",
                    tool_category="visualization",
                    dependencies=["security_indicators"],
                    estimated_duration=15.0
                ),
                WorkflowStep(
                    step_id="generate_report",
                    name="Generate Report",
                    description="Generate network analysis report",
                    tool_name="report_generator",
                    tool_category="reporting",
                    dependencies=["traffic_visualization"],
                    estimated_duration=10.0
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            required_tools=["pcap_analyzer", "protocol_analyzer", "threat_detector", "chart_generator", "report_generator"],
            required_inputs=["pcap_file"],
            expected_outputs=["network_report", "traffic_stats", "security_indicators"],
            tags=["network", "analysis", "traffic", "security"]
        )
        
        # Add templates to manager
        self.templates[patent_template.template_id] = patent_template
        self.templates[malware_template.template_id] = malware_template
        self.templates[network_template.template_id] = network_template
        
        # Save templates
        self._save_templates()
    
    def _save_templates(self):
        """Save templates to storage."""
        try:
            templates_file = self.templates_dir / "templates.json"
            templates_data = {}
            
            for template_id, template in self.templates.items():
                template_dict = template.__dict__.copy()
                # Convert enums to strings for JSON serialization
                template_dict['category'] = template.category.value
                template_dict['complexity'] = template.complexity.value
                template_dict['execution_mode'] = template.execution_mode.value
                template_dict['steps'] = [step.__dict__ for step in template.steps]
                templates_data[template_id] = template_dict
            
            with open(templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            logger.info(f"Saved {len(self.templates)} templates")
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
    
    def create_dynamic_template(self, problem_description: str, 
                              input_files: List[str] = None,
                              desired_outputs: List[str] = None,
                              complexity: WorkflowComplexity = None) -> WorkflowTemplate:
        """Create a dynamic workflow template based on problem analysis."""
        template_id = f"dynamic_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Analyze problem to determine components
        components = self._analyze_problem_for_components(
            problem_description, input_files, desired_outputs
        )
        
        # Determine complexity if not provided
        if complexity is None:
            complexity = self._assess_complexity(components)
        
        # Create workflow steps from components
        steps = self._create_steps_from_components(components)
        
        # Determine execution mode
        execution_mode = self._determine_execution_mode(steps, complexity)
        
        # Create template
        template = WorkflowTemplate(
            template_id=template_id,
            name=f"Dynamic Workflow - {problem_description[:50]}...",
            description=f"Dynamically generated workflow for: {problem_description}",
            category=self._determine_category(components),
            complexity=complexity,
            steps=steps,
            execution_mode=execution_mode,
            required_tools=self._extract_required_tools(steps),
            required_inputs=input_files or [],
            expected_outputs=desired_outputs or [],
            tags=self._generate_tags(components, problem_description)
        )
        
        # Add to templates
        self.templates[template_id] = template
        
        # Save templates
        self._save_templates()
        
        return template
    
    def _analyze_problem_for_components(self, problem_description: str, 
                                      input_files: List[str] = None,
                                      desired_outputs: List[str] = None) -> List[str]:
        """Analyze problem to determine required components."""
        components = []
        problem_lower = problem_description.lower()
        
        # Always start with data import if files are provided
        if input_files:
            components.append("data_import")
        
        # Analyze problem keywords to determine components
        if any(keyword in problem_lower for keyword in ["analyze", "analysis", "examine"]):
            if any(keyword in problem_lower for keyword in ["malware", "virus", "threat"]):
                components.extend(["malware_analysis", "threat_analysis"])
            elif any(keyword in problem_lower for keyword in ["network", "traffic", "pcap"]):
                components.extend(["network_analysis", "protocol_analysis"])
            elif any(keyword in problem_lower for keyword in ["vulnerability", "scan", "security"]):
                components.extend(["vulnerability_scan", "compliance_check"])
            else:
                components.append("pattern_analysis")
        
        if any(keyword in problem_lower for keyword in ["enrich", "enhance", "augment"]):
            components.append("data_enrichment")
        
        if any(keyword in problem_lower for keyword in ["compliance", "policy", "standard"]):
            components.append("compliance_check")
        
        if any(keyword in problem_lower for keyword in ["incident", "response", "breach"]):
            components.extend(["incident_response", "malware_analysis"])
        
        if any(keyword in problem_lower for keyword in ["anomaly", "unusual", "suspicious"]):
            components.append("anomaly_detection")
        
        # Always end with reporting if outputs are expected
        if desired_outputs or any(keyword in problem_lower for keyword in ["report", "summary", "result"]):
            components.append("report_generation")
        
        # Add data export if needed
        if any(keyword in problem_lower for keyword in ["export", "save", "output"]):
            components.append("data_export")
        
        return list(set(components))  # Remove duplicates
    
    def _assess_complexity(self, components: List[str]) -> WorkflowComplexity:
        """Assess workflow complexity based on components."""
        complexity_scores = {
            WorkflowComplexity.SIMPLE: 0,
            WorkflowComplexity.MODERATE: 0,
            WorkflowComplexity.COMPLEX: 0,
            WorkflowComplexity.EXPERT: 0
        }
        
        for component in components:
            if component in self.component_library:
                comp_complexity = self.component_library[component]["complexity"]
                complexity_scores[comp_complexity] += 1
        
        # Return the highest complexity level
        return max(complexity_scores.items(), key=lambda x: x[1])[0]
    
    def _create_steps_from_components(self, components: List[str]) -> List[WorkflowStep]:
        """Create workflow steps from components."""
        steps = []
        step_counter = 1
        
        for component in components:
            if component in self.component_library:
                comp_info = self.component_library[component]
                
                step = WorkflowStep(
                    step_id=f"step_{step_counter}",
                    name=comp_info["name"],
                    description=comp_info["description"],
                    tool_name=comp_info["required_tools"][0] if comp_info["required_tools"] else "generic_tool",
                    tool_category=comp_info["category"],
                    estimated_duration=comp_info["estimated_duration"],
                    dependencies=[f"step_{step_counter-1}"] if step_counter > 1 else []
                )
                
                steps.append(step)
                step_counter += 1
        
        return steps
    
    def _determine_execution_mode(self, steps: List[WorkflowStep], 
                                complexity: WorkflowComplexity) -> ExecutionMode:
        """Determine execution mode based on steps and complexity."""
        if complexity == WorkflowComplexity.EXPERT:
            return ExecutionMode.ADAPTIVE
        elif len(steps) > 5:
            return ExecutionMode.PARALLEL
        elif any(step.parallel_execution for step in steps):
            return ExecutionMode.PARALLEL
        else:
            return ExecutionMode.SEQUENTIAL
    
    def _determine_category(self, components: List[str]) -> WorkflowCategory:
        """Determine workflow category based on components."""
        category_scores = {
            WorkflowCategory.ANALYSIS: 0,
            WorkflowCategory.INVESTIGATION: 0,
            WorkflowCategory.RESPONSE: 0,
            WorkflowCategory.PREVENTION: 0,
            WorkflowCategory.COMPLIANCE: 0,
            WorkflowCategory.RESEARCH: 0,
            WorkflowCategory.AUTOMATION: 0
        }
        
        for component in components:
            if component in self.component_library:
                comp_category = self.component_library[component]["category"]
                if comp_category == "analysis":
                    category_scores[WorkflowCategory.ANALYSIS] += 1
                elif comp_category == "security":
                    category_scores[WorkflowCategory.INVESTIGATION] += 1
                elif comp_category == "data_processing":
                    category_scores[WorkflowCategory.AUTOMATION] += 1
        
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_required_tools(self, steps: List[WorkflowStep]) -> List[str]:
        """Extract required tools from workflow steps."""
        tools = set()
        for step in steps:
            tools.add(step.tool_name)
        return list(tools)
    
    def _generate_tags(self, components: List[str], problem_description: str) -> List[str]:
        """Generate tags for the workflow."""
        tags = set()
        
        # Add component tags
        for component in components:
            if component in self.component_library:
                tags.update(self.component_library[component]["tags"])
        
        # Add problem-specific tags
        problem_lower = problem_description.lower()
        if "patent" in problem_lower:
            tags.add("patent")
        if "malware" in problem_lower:
            tags.add("malware")
        if "network" in problem_lower:
            tags.add("network")
        if "security" in problem_lower:
            tags.add("security")
        
        return list(tags)
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: WorkflowCategory = None, 
                      complexity: WorkflowComplexity = None,
                      tags: List[str] = None) -> List[WorkflowTemplate]:
        """List workflow templates with optional filtering."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return templates
    
    def update_template_performance(self, template_id: str, execution_time: float, 
                                  success: bool, adaptations: List[Dict[str, Any]] = None):
        """Update template performance metrics."""
        if template_id in self.templates:
            template = self.templates[template_id]
            template.usage_count += 1
            
            # Update success rate
            if success:
                template.success_rate = (template.success_rate * (template.usage_count - 1) + 1.0) / template.usage_count
            else:
                template.success_rate = (template.success_rate * (template.usage_count - 1)) / template.usage_count
            
            # Update average execution time
            template.avg_execution_time = (template.avg_execution_time * (template.usage_count - 1) + execution_time) / template.usage_count
            
            # Update last modified
            template.last_modified = time.time()
            
            # Store adaptations
            if adaptations:
                self.adaptation_patterns[template_id].extend(adaptations)
            
            # Save templates
            self._save_templates()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all templates."""
        stats = {
            "total_templates": len(self.templates),
            "total_executions": sum(t.usage_count for t in self.templates.values()),
            "avg_success_rate": sum(t.success_rate for t in self.templates.values()) / len(self.templates) if self.templates else 0,
            "avg_execution_time": sum(t.avg_execution_time for t in self.templates.values()) / len(self.templates) if self.templates else 0,
            "templates_by_category": {},
            "templates_by_complexity": {},
            "most_used_templates": []
        }
        
        # Templates by category
        for template in self.templates.values():
            category = template.category.value
            if category not in stats["templates_by_category"]:
                stats["templates_by_category"][category] = 0
            stats["templates_by_category"][category] += 1
        
        # Templates by complexity
        for template in self.templates.values():
            complexity = template.complexity.value
            if complexity not in stats["templates_by_complexity"]:
                stats["templates_by_complexity"][complexity] = 0
            stats["templates_by_complexity"][complexity] += 1
        
        # Most used templates
        sorted_templates = sorted(self.templates.values(), key=lambda t: t.usage_count, reverse=True)
        stats["most_used_templates"] = [
            {
                "template_id": t.template_id,
                "name": t.name,
                "usage_count": t.usage_count,
                "success_rate": t.success_rate
            }
            for t in sorted_templates[:10]
        ]
        
        return stats
