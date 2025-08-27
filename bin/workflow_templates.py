#!/usr/bin/env python3
"""
Workflow Templates and Execution Logic

Provides specialized workflow templates for different problem types with:
- Hierarchical execution patterns
- Self-modifying capabilities
- Intelligent resource allocation
- Performance optimization
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Define missing classes locally
from enum import Enum

class ProblemType(Enum):
    """Problem types for workflow templates."""
    THREAT_HUNTING = "threat_hunting"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    INVESTIGATION = "investigation"
    CSV_ENRICHMENT = "csv_enrichment"

@dataclass
class WorkflowNode:
    """Workflow node definition."""
    node_id: str
    node_type: str
    name: str
    description: str
    execution_order: int
    timeout: int
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class WorkflowContext:
    """Workflow execution context."""
    workflow_id: str
    problem_type: ProblemType
    parameters: Dict[str, Any] = None
    state: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.state is None:
            self.state = {}

class ExecutionStrategy(Enum):
    """Execution strategies for workflow nodes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    ADAPTIVE = "adaptive"

@dataclass
class ExecutionTemplate:
    """Template for workflow execution."""
    strategy: ExecutionStrategy
    nodes: List[WorkflowNode]
    dependencies: Dict[str, List[str]]
    optimization_rules: List[Dict[str, Any]]
    adaptation_triggers: List[Dict[str, Any]]

class WorkflowTemplateLibrary:
    """Library of workflow templates for different problem types."""
    
    def __init__(self):
        self.templates = {}
        self.execution_patterns = {}
        self.optimization_rules = {}
        
        # Initialize templates
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all workflow templates."""
        self.templates[ProblemType.THREAT_HUNTING] = self._create_threat_hunting_template()
        self.templates[ProblemType.INCIDENT_RESPONSE] = self._create_incident_response_template()
        self.templates[ProblemType.COMPLIANCE] = self._create_compliance_template()
        self.templates[ProblemType.RISK_ASSESSMENT] = self._create_risk_assessment_template()
        self.templates[ProblemType.INVESTIGATION] = self._create_investigation_template()
        self.templates[ProblemType.CSV_ENRICHMENT] = self._create_csv_enrichment_template()
    
    def _create_threat_hunting_template(self) -> ExecutionTemplate:
        """Create threat hunting workflow template."""
        nodes = [
            WorkflowNode(
                node_id="threat_analysis",
                node_type="analysis",
                name="Threat Analysis",
                description="Analyze threat indicators and context",
                execution_order=1,
                timeout=120
            ),
            WorkflowNode(
                node_id="intelligence_gathering",
                node_type="gathering",
                name="Intelligence Gathering",
                description="Gather relevant threat intelligence",
                dependencies=["threat_analysis"],
                execution_order=2,
                timeout=180
            ),
            WorkflowNode(
                node_id="network_analysis",
                node_type="analysis",
                name="Network Analysis",
                description="Analyze network traffic and logs",
                dependencies=["intelligence_gathering"],
                execution_order=3,
                timeout=300
            ),
            WorkflowNode(
                node_id="host_analysis",
                node_type="analysis",
                name="Host Analysis",
                description="Analyze host-based indicators",
                dependencies=["network_analysis"],
                execution_order=4,
                timeout=240
            ),
            WorkflowNode(
                node_id="threat_assessment",
                node_type="assessment",
                name="Threat Assessment",
                description="Assess threat level and impact",
                dependencies=["host_analysis"],
                execution_order=5,
                timeout=120
            ),
            WorkflowNode(
                node_id="response_planning",
                node_type="planning",
                name="Response Planning",
                description="Plan response actions",
                dependencies=["threat_assessment"],
                execution_order=6,
                timeout=180
            )
        ]
        
        dependencies = {
            "threat_analysis": [],
            "intelligence_gathering": ["threat_analysis"],
            "network_analysis": ["intelligence_gathering"],
            "host_analysis": ["network_analysis"],
            "threat_assessment": ["host_analysis"],
            "response_planning": ["threat_assessment"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["network_analysis", "host_analysis"], "condition": "complexity > 5"},
            {"type": "caching", "nodes": ["intelligence_gathering"], "strategy": "aggressive"},
            {"type": "resource_allocation", "nodes": ["threat_analysis"], "priority": "high"}
        ]
        
        adaptation_triggers = [
            {"trigger": "threat_level_high", "action": "add_validation_node", "condition": "threat_assessment.result.level > 7"},
            {"trigger": "complexity_increase", "action": "add_parallel_execution", "condition": "context.complexity > 8"},
            {"trigger": "resource_constraint", "action": "optimize_execution_order", "condition": "available_resources < required_resources"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.ADAPTIVE,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def _create_incident_response_template(self) -> ExecutionTemplate:
        """Create incident response workflow template."""
        nodes = [
            WorkflowNode(
                node_id="incident_triage",
                node_type="triage",
                name="Incident Triage",
                description="Initial incident assessment and classification",
                execution_order=1,
                timeout=60
            ),
            WorkflowNode(
                node_id="containment",
                node_type="action",
                name="Containment",
                description="Contain the incident to prevent spread",
                dependencies=["incident_triage"],
                execution_order=2,
                timeout=300
            ),
            WorkflowNode(
                node_id="evidence_collection",
                node_type="collection",
                name="Evidence Collection",
                description="Collect forensic evidence",
                dependencies=["containment"],
                execution_order=3,
                timeout=600
            ),
            WorkflowNode(
                node_id="analysis",
                node_type="analysis",
                name="Analysis",
                description="Analyze collected evidence",
                dependencies=["evidence_collection"],
                execution_order=4,
                timeout=900
            ),
            WorkflowNode(
                node_id="eradication",
                node_type="action",
                name="Eradication",
                description="Remove threat from environment",
                dependencies=["analysis"],
                execution_order=5,
                timeout=600
            ),
            WorkflowNode(
                node_id="recovery",
                node_type="action",
                name="Recovery",
                description="Restore affected systems",
                dependencies=["eradication"],
                execution_order=6,
                timeout=1200
            ),
            WorkflowNode(
                node_id="lessons_learned",
                node_type="documentation",
                name="Lessons Learned",
                description="Document lessons learned",
                dependencies=["recovery"],
                execution_order=7,
                timeout=180
            )
        ]
        
        dependencies = {
            "incident_triage": [],
            "containment": ["incident_triage"],
            "evidence_collection": ["containment"],
            "analysis": ["evidence_collection"],
            "eradication": ["analysis"],
            "recovery": ["eradication"],
            "lessons_learned": ["recovery"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["evidence_collection", "containment"], "condition": "incident_scope > 5"},
            {"type": "priority", "nodes": ["containment"], "priority": "critical"},
            {"type": "resource_allocation", "nodes": ["analysis"], "priority": "high"}
        ]
        
        adaptation_triggers = [
            {"trigger": "incident_escalation", "action": "add_escalation_node", "condition": "incident_triage.result.severity > 8"},
            {"trigger": "scope_expansion", "action": "add_parallel_containment", "condition": "affected_systems > 10"},
            {"trigger": "regulatory_requirement", "action": "add_compliance_node", "condition": "regulatory_impact > 0"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.SEQUENTIAL,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def _create_compliance_template(self) -> ExecutionTemplate:
        """Create compliance workflow template."""
        nodes = [
            WorkflowNode(
                node_id="scope_definition",
                node_type="planning",
                name="Scope Definition",
                description="Define compliance assessment scope",
                execution_order=1,
                timeout=120
            ),
            WorkflowNode(
                node_id="policy_review",
                node_type="review",
                name="Policy Review",
                description="Review relevant policies and standards",
                dependencies=["scope_definition"],
                execution_order=2,
                timeout=300
            ),
            WorkflowNode(
                node_id="gap_analysis",
                node_type="analysis",
                name="Gap Analysis",
                description="Analyze gaps between current state and requirements",
                dependencies=["policy_review"],
                execution_order=3,
                timeout=600
            ),
            WorkflowNode(
                node_id="risk_assessment",
                node_type="assessment",
                name="Risk Assessment",
                description="Assess compliance risks",
                dependencies=["gap_analysis"],
                execution_order=4,
                timeout=300
            ),
            WorkflowNode(
                node_id="remediation_planning",
                node_type="planning",
                name="Remediation Planning",
                description="Plan remediation actions",
                dependencies=["risk_assessment"],
                execution_order=5,
                timeout=240
            ),
            WorkflowNode(
                node_id="implementation",
                node_type="action",
                name="Implementation",
                description="Implement remediation actions",
                dependencies=["remediation_planning"],
                execution_order=6,
                timeout=1800
            ),
            WorkflowNode(
                node_id="validation",
                node_type="validation",
                name="Validation",
                description="Validate compliance improvements",
                dependencies=["implementation"],
                execution_order=7,
                timeout=600
            )
        ]
        
        dependencies = {
            "scope_definition": [],
            "policy_review": ["scope_definition"],
            "gap_analysis": ["policy_review"],
            "risk_assessment": ["gap_analysis"],
            "remediation_planning": ["risk_assessment"],
            "implementation": ["remediation_planning"],
            "validation": ["implementation"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["policy_review", "gap_analysis"], "condition": "scope_size > 10"},
            {"type": "caching", "nodes": ["policy_review"], "strategy": "moderate"},
            {"type": "resource_allocation", "nodes": ["implementation"], "priority": "high"}
        ]
        
        adaptation_triggers = [
            {"trigger": "scope_change", "action": "reassess_scope", "condition": "scope_modification > 0.3"},
            {"trigger": "high_risk", "action": "add_mitigation_node", "condition": "risk_assessment.result.level > 7"},
            {"trigger": "resource_constraint", "action": "prioritize_remediation", "condition": "available_resources < required_resources * 0.8"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.SEQUENTIAL,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def _create_risk_assessment_template(self) -> ExecutionTemplate:
        """Create risk assessment workflow template."""
        nodes = [
            WorkflowNode(
                node_id="asset_inventory",
                node_type="inventory",
                name="Asset Inventory",
                description="Identify and catalog assets",
                execution_order=1,
                timeout=300
            ),
            WorkflowNode(
                node_id="threat_identification",
                node_type="identification",
                name="Threat Identification",
                description="Identify potential threats",
                dependencies=["asset_inventory"],
                execution_order=2,
                timeout=240
            ),
            WorkflowNode(
                node_id="vulnerability_assessment",
                node_type="assessment",
                name="Vulnerability Assessment",
                description="Assess asset vulnerabilities",
                dependencies=["asset_inventory"],
                execution_order=2,
                timeout=600
            ),
            WorkflowNode(
                node_id="risk_calculation",
                node_type="calculation",
                name="Risk Calculation",
                description="Calculate risk scores",
                dependencies=["threat_identification", "vulnerability_assessment"],
                execution_order=3,
                timeout=180
            ),
            WorkflowNode(
                node_id="risk_prioritization",
                node_type="prioritization",
                name="Risk Prioritization",
                description="Prioritize risks for treatment",
                dependencies=["risk_calculation"],
                execution_order=4,
                timeout=120
            ),
            WorkflowNode(
                node_id="treatment_planning",
                node_type="planning",
                name="Treatment Planning",
                description="Plan risk treatment actions",
                dependencies=["risk_prioritization"],
                execution_order=5,
                timeout=300
            )
        ]
        
        dependencies = {
            "asset_inventory": [],
            "threat_identification": ["asset_inventory"],
            "vulnerability_assessment": ["asset_inventory"],
            "risk_calculation": ["threat_identification", "vulnerability_assessment"],
            "risk_prioritization": ["risk_calculation"],
            "treatment_planning": ["risk_prioritization"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["threat_identification", "vulnerability_assessment"], "condition": "asset_count > 50"},
            {"type": "caching", "nodes": ["asset_inventory"], "strategy": "aggressive"},
            {"type": "resource_allocation", "nodes": ["vulnerability_assessment"], "priority": "high"}
        ]
        
        adaptation_triggers = [
            {"trigger": "high_risk_discovery", "action": "add_mitigation_node", "condition": "risk_score > 8"},
            {"trigger": "scope_expansion", "action": "add_parallel_processing", "condition": "asset_count > 100"},
            {"trigger": "time_constraint", "action": "prioritize_critical_assets", "condition": "time_remaining < total_estimated_time * 0.5"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.ADAPTIVE,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def _create_investigation_template(self) -> ExecutionTemplate:
        """Create general investigation workflow template."""
        nodes = [
            WorkflowNode(
                node_id="hypothesis_formation",
                node_type="planning",
                name="Hypothesis Formation",
                description="Form initial hypotheses",
                execution_order=1,
                timeout=120
            ),
            WorkflowNode(
                node_id="data_collection",
                node_type="collection",
                name="Data Collection",
                description="Collect relevant data",
                dependencies=["hypothesis_formation"],
                execution_order=2,
                timeout=600
            ),
            WorkflowNode(
                node_id="data_analysis",
                node_type="analysis",
                name="Data Analysis",
                description="Analyze collected data",
                dependencies=["data_collection"],
                execution_order=3,
                timeout=900
            ),
            WorkflowNode(
                node_id="hypothesis_testing",
                node_type="testing",
                name="Hypothesis Testing",
                description="Test hypotheses against data",
                dependencies=["data_analysis"],
                execution_order=4,
                timeout=300
            ),
            WorkflowNode(
                node_id="conclusion_formation",
                node_type="synthesis",
                name="Conclusion Formation",
                description="Form conclusions based on analysis",
                dependencies=["hypothesis_testing"],
                execution_order=5,
                timeout=180
            ),
            WorkflowNode(
                node_id="report_generation",
                node_type="documentation",
                name="Report Generation",
                description="Generate investigation report",
                dependencies=["conclusion_formation"],
                execution_order=6,
                timeout=240
            )
        ]
        
        dependencies = {
            "hypothesis_formation": [],
            "data_collection": ["hypothesis_formation"],
            "data_analysis": ["data_collection"],
            "hypothesis_testing": ["data_analysis"],
            "conclusion_formation": ["hypothesis_testing"],
            "report_generation": ["conclusion_formation"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["data_collection", "data_analysis"], "condition": "data_volume > 1000"},
            {"type": "caching", "nodes": ["data_collection"], "strategy": "moderate"},
            {"type": "resource_allocation", "nodes": ["data_analysis"], "priority": "high"}
        ]
        
        adaptation_triggers = [
            {"trigger": "hypothesis_rejection", "action": "add_hypothesis_node", "condition": "hypothesis_testing.result.rejected > 0.5"},
            {"trigger": "data_insufficiency", "action": "add_collection_node", "condition": "data_quality < 0.7"},
            {"trigger": "complexity_increase", "action": "add_validation_node", "condition": "analysis_complexity > 8"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.SEQUENTIAL,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def _create_csv_enrichment_template(self) -> ExecutionTemplate:
        """Create CSV enrichment workflow template."""
        nodes = [
            WorkflowNode(
                node_id="csv_import",
                node_type="import",
                name="CSV Import",
                description="Import CSV file into DataFrame",
                execution_order=1,
                timeout=60
            ),
            WorkflowNode(
                node_id="column_analysis",
                node_type="analysis",
                name="Column Analysis",
                description="Analyze existing columns and identify new columns needed",
                dependencies=["csv_import"],
                execution_order=2,
                timeout=120
            ),
            WorkflowNode(
                node_id="column_creation",
                node_type="preparation",
                name="Column Creation",
                description="Create new columns based on analysis",
                dependencies=["column_analysis"],
                execution_order=3,
                timeout=90
            ),
            WorkflowNode(
                node_id="llm_processing",
                node_type="processing",
                name="LLM Processing",
                description="Process each row using LLM to enrich data",
                dependencies=["column_creation"],
                execution_order=4,
                timeout=1800  # 30 minutes for LLM processing
            ),
            WorkflowNode(
                node_id="data_validation",
                node_type="validation",
                name="Data Validation",
                description="Validate enriched data quality",
                dependencies=["llm_processing"],
                execution_order=5,
                timeout=180
            ),
            WorkflowNode(
                node_id="export_results",
                node_type="export",
                name="Export Results",
                description="Export enriched DataFrame to output file",
                dependencies=["data_validation"],
                execution_order=6,
                timeout=120
            )
        ]
        
        dependencies = {
            "csv_import": [],
            "column_analysis": ["csv_import"],
            "column_creation": ["column_analysis"],
            "llm_processing": ["column_creation"],
            "data_validation": ["llm_processing"],
            "export_results": ["data_validation"]
        }
        
        optimization_rules = [
            {"type": "parallel", "nodes": ["column_analysis", "column_creation"], "condition": "csv_size > 1000"},
            {"type": "caching", "nodes": ["csv_import"], "strategy": "aggressive"},
            {"type": "resource_allocation", "nodes": ["llm_processing"], "priority": "high"},
            {"type": "batch_processing", "nodes": ["llm_processing"], "batch_size": 100}
        ]
        
        adaptation_triggers = [
            {"trigger": "large_dataset", "action": "add_batch_processing", "condition": "row_count > 10000"},
            {"trigger": "llm_timeout", "action": "add_retry_logic", "condition": "llm_timeout_count > 3"},
            {"trigger": "data_quality_issue", "action": "add_cleaning_node", "condition": "validation_score < 0.8"},
            {"trigger": "memory_constraint", "action": "add_streaming_processing", "condition": "memory_usage > 0.8"}
        ]
        
        return ExecutionTemplate(
            strategy=ExecutionStrategy.ITERATIVE,
            nodes=nodes,
            dependencies=dependencies,
            optimization_rules=optimization_rules,
            adaptation_triggers=adaptation_triggers
        )
    
    def get_template(self, problem_type: ProblemType) -> Optional[ExecutionTemplate]:
        """Get workflow template for problem type."""
        return self.templates.get(problem_type)
    
    def customize_template(self, template: ExecutionTemplate, context: WorkflowContext) -> ExecutionTemplate:
        """Customize template based on context."""
        customized_nodes = template.nodes.copy()
        
        # Add complexity-based nodes
        if context.complexity > 7:
            validation_node = WorkflowNode(
                node_id="validation",
                node_type="validation",
                name="Validation",
                description="Validate findings and conclusions",
                dependencies=[node.node_id for node in customized_nodes if node.execution_order > 0],
                execution_order=max(node.execution_order for node in customized_nodes) + 1,
                timeout=300
            )
            customized_nodes.append(validation_node)
        
        # Add priority-based nodes
        if context.priority > 3:
            review_node = WorkflowNode(
                node_id="peer_review",
                node_type="review",
                name="Peer Review",
                description="Peer review of findings",
                dependencies=[node.node_id for node in customized_nodes if node.execution_order > 0],
                execution_order=max(node.execution_order for node in customized_nodes) + 1,
                timeout=180
            )
            customized_nodes.append(review_node)
        
        # Update dependencies
        updated_dependencies = template.dependencies.copy()
        for node in customized_nodes:
            if node.node_id not in updated_dependencies:
                updated_dependencies[node.node_id] = []
        
        return ExecutionTemplate(
            strategy=template.strategy,
            nodes=customized_nodes,
            dependencies=updated_dependencies,
            optimization_rules=template.optimization_rules,
            adaptation_triggers=template.adaptation_triggers
        )

class WorkflowExecutor:
    """Executes workflow templates with optimization and adaptation."""
    
    def __init__(self, template_library: WorkflowTemplateLibrary):
        self.template_library = template_library
        self.execution_history = []
        self.performance_metrics = {}
        
    async def execute_template(self, template: ExecutionTemplate, context: WorkflowContext) -> Dict[str, Any]:
        """Execute a workflow template."""
        start_time = time.time()
        
        # Apply optimization rules
        optimized_template = self._apply_optimization_rules(template, context)
        
        # Execute nodes based on strategy
        if optimized_template.strategy == ExecutionStrategy.SEQUENTIAL:
            results = await self._execute_sequential(optimized_template, context)
        elif optimized_template.strategy == ExecutionStrategy.PARALLEL:
            results = await self._execute_parallel(optimized_template, context)
        elif optimized_template.strategy == ExecutionStrategy.ITERATIVE:
            results = await self._execute_iterative(optimized_template, context)
        elif optimized_template.strategy == ExecutionStrategy.ADAPTIVE:
            results = await self._execute_adaptive(optimized_template, context)
        else:
            results = await self._execute_sequential(optimized_template, context)
        
        # Record execution
        execution_time = time.time() - start_time
        self.execution_history.append({
            'template_id': id(template),
            'context': context,
            'execution_time': execution_time,
            'results': results
        })
        
        return results
    
    async def _execute_sequential(self, template: ExecutionTemplate, context: WorkflowContext) -> Dict[str, Any]:
        """Execute nodes sequentially."""
        results = {}
        
        for node in sorted(template.nodes, key=lambda x: x.execution_order):
            # Check dependencies
            if not self._check_dependencies(node, results):
                continue
            
            # Execute node
            node_result = await self._execute_node(node, context, results)
            results[node.node_id] = node_result
            
            # Check for adaptation triggers
            if self._should_adapt(node, node_result, template, context):
                template = self._adapt_template(template, node, node_result, context)
        
        return results
    
    async def _execute_parallel(self, template: ExecutionTemplate, context: WorkflowContext) -> Dict[str, Any]:
        """Execute nodes in parallel where possible."""
        results = {}
        
        # Group nodes by execution order
        execution_groups = {}
        for node in template.nodes:
            if node.execution_order not in execution_groups:
                execution_groups[node.execution_order] = []
            execution_groups[node.execution_order].append(node)
        
        # Execute groups sequentially, nodes within groups in parallel
        for order in sorted(execution_groups.keys()):
            group_nodes = execution_groups[order]
            
            # Check dependencies for all nodes in group
            executable_nodes = [node for node in group_nodes if self._check_dependencies(node, results)]
            
            if executable_nodes:
                # Execute nodes in parallel
                tasks = [self._execute_node(node, context, results) for node in executable_nodes]
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results
                for node, result in zip(executable_nodes, group_results):
                    if isinstance(result, Exception):
                        results[node.node_id] = {'error': str(result)}
                    else:
                        results[node.node_id] = result
        
        return results
    
    async def _execute_iterative(self, template: ExecutionTemplate, context: WorkflowContext) -> Dict[str, Any]:
        """Execute nodes with iterative processing for data workflows."""
        results = {}
        
        for node in sorted(template.nodes, key=lambda x: x.execution_order):
            # Check dependencies
            if not self._check_dependencies(node, results):
                continue
            
            # Execute node
            node_result = await self._execute_node(node, context, results)
            results[node.node_id] = node_result
            
            # For iterative workflows, check if we need to repeat certain nodes
            if node.node_type == "processing" and self._should_iterate(node, node_result, context):
                # Re-execute processing node with updated context
                iteration_result = await self._execute_node_iteration(node, context, results, node_result)
                results[f"{node.node_id}_iteration"] = iteration_result
            
            # Check for adaptation triggers
            if self._should_adapt(node, node_result, template, context):
                template = self._adapt_template(template, node, node_result, context)
        
        return results
    
    def _should_iterate(self, node: WorkflowNode, result: Any, context: WorkflowContext) -> bool:
        """Determine if a node should be executed again iteratively."""
        if node.node_type == "processing":
            # Check if there are more items to process
            if isinstance(result, dict) and result.get('remaining_items', 0) > 0:
                return True
            # Check if quality threshold not met
            if isinstance(result, dict) and result.get('quality_score', 1.0) < 0.8:
                return True
        return False
    
    async def _execute_node_iteration(self, node: WorkflowNode, context: WorkflowContext, 
                                    results: Dict[str, Any], previous_result: Any) -> Any:
        """Execute a node iteration with updated context."""
        # Update context with previous results
        updated_context = WorkflowContext(
            workflow_id=context.workflow_id,
            problem_type=context.problem_type,
            parameters=context.parameters.copy(),
            state=context.state.copy()
        )
        
        # Add iteration state
        updated_context.state['iteration_count'] = updated_context.state.get('iteration_count', 0) + 1
        updated_context.state['previous_result'] = previous_result
        
        # Execute node with updated context
        return await self._execute_node(node, updated_context, results)
    
    async def _execute_adaptive(self, template: ExecutionTemplate, context: WorkflowContext) -> Dict[str, Any]:
        """Execute with adaptive behavior."""
        results = {}
        current_template = template
        
        for node in sorted(template.nodes, key=lambda x: x.execution_order):
            # Check dependencies
            if not self._check_dependencies(node, current_template, results):
                continue
            
            # Execute node
            node_result = await self._execute_node(node, context, results)
            results[node.node_id] = node_result
            
            # Check for adaptation
            if self._should_adapt(node, node_result, current_template, context):
                current_template = self._adapt_template(current_template, node, node_result, context)
                
                # Re-evaluate execution order
                current_template.nodes = sorted(current_template.nodes, key=lambda x: x.execution_order)
        
        return results
    
    async def _execute_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Any:
        """Execute a single workflow node."""
        node_start_time = time.time()
        
        try:
            # Simulate node execution based on type
            if node.node_type == "analysis":
                result = await self._execute_analysis_node(node, context, results)
            elif node.node_type == "gathering":
                result = await self._execute_gathering_node(node, context, results)
            elif node.node_type == "assessment":
                result = await self._execute_assessment_node(node, context, results)
            elif node.node_type == "action":
                result = await self._execute_action_node(node, context, results)
            else:
                result = await self._execute_generic_node(node, context, results)
            
            node.execution_time = time.time() - node_start_time
            node.status = "completed"
            node.result = result
            
            return result
            
        except Exception as e:
            node.error = str(e)
            node.status = "failed"
            return {'error': str(e)}
    
    async def _execute_analysis_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis node."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'analysis',
            'node_id': node.node_id,
            'result': f"Analysis completed for {node.name}",
            'complexity_level': context.complexity,
            'requires_adaptation': context.complexity > 8
        }
    
    async def _execute_gathering_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gathering node."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'gathering',
            'node_id': node.node_id,
            'result': f"Data gathered for {node.name}",
            'data_volume': context.complexity * 100,
            'data_quality': 0.8
        }
    
    async def _execute_assessment_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assessment node."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'assessment',
            'node_id': node.node_id,
            'result': f"Assessment completed for {node.name}",
            'risk_level': min(context.complexity, 10),
            'confidence': 0.9
        }
    
    async def _execute_action_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action node."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'action',
            'node_id': node.node_id,
            'result': f"Action completed for {node.name}",
            'success': True,
            'impact_level': context.priority
        }
    
    async def _execute_generic_node(self, node: WorkflowNode, context: WorkflowContext, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic node."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'type': 'generic',
            'node_id': node.node_id,
            'result': f"Generic execution completed for {node.name}",
            'status': 'success'
        }
    
    def _check_dependencies(self, node: WorkflowNode, template: ExecutionTemplate, results: Dict[str, Any]) -> bool:
        """Check if node dependencies are satisfied."""
        for dep_id in node.dependencies:
            if dep_id not in results:
                return False
        return True
    
    def _should_adapt(self, node: WorkflowNode, result: Any, template: ExecutionTemplate, context: WorkflowContext) -> bool:
        """Determine if adaptation is needed."""
        # Check adaptation triggers
        for trigger in template.adaptation_triggers:
            if self._evaluate_trigger(trigger, node, result, context):
                return True
        return False
    
    def _evaluate_trigger(self, trigger: Dict[str, Any], node: WorkflowNode, result: Any, context: WorkflowContext) -> bool:
        """Evaluate if a trigger condition is met."""
        # This is a simplified trigger evaluation
        # In a real implementation, this would be more sophisticated
        return True  # Simplified for demo
    
    def _adapt_template(self, template: ExecutionTemplate, node: WorkflowNode, result: Any, context: WorkflowContext) -> ExecutionTemplate:
        """Adapt the workflow template."""
        # Create adapted template
        adapted_nodes = template.nodes.copy()
        
        # Add adaptation node if needed
        if result.get('requires_adaptation', False):
            adaptation_node = WorkflowNode(
                node_id=f"adaptation_{len(adapted_nodes)}",
                node_type="adaptation",
                name="Workflow Adaptation",
                description="Adapt workflow based on execution results",
                dependencies=[node.node_id],
                execution_order=node.execution_order + 1,
                timeout=120
            )
            adapted_nodes.append(adaptation_node)
        
        return ExecutionTemplate(
            strategy=template.strategy,
            nodes=adapted_nodes,
            dependencies=template.dependencies,
            optimization_rules=template.optimization_rules,
            adaptation_triggers=template.adaptation_triggers
        )
    
    def _apply_optimization_rules(self, template: ExecutionTemplate, context: WorkflowContext) -> ExecutionTemplate:
        """Apply optimization rules to template."""
        optimized_nodes = template.nodes.copy()
        
        for rule in template.optimization_rules:
            if rule['type'] == 'parallel' and self._evaluate_condition(rule['condition'], context):
                # Apply parallel execution optimization
                for node_id in rule['nodes']:
                    node = next((n for n in optimized_nodes if n.node_id == node_id), None)
                    if node:
                        node.execution_order = min(node.execution_order, 2)
        
        return ExecutionTemplate(
            strategy=template.strategy,
            nodes=optimized_nodes,
            dependencies=template.dependencies,
            optimization_rules=template.optimization_rules,
            adaptation_triggers=template.adaptation_triggers
        )
    
    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """Evaluate optimization condition."""
        # Simplified condition evaluation
        # In a real implementation, this would parse and evaluate conditions
        return True  # Simplified for demo

# Example usage
async def main():
    """Example usage of workflow templates and executor."""
    template_library = WorkflowTemplateLibrary()
    executor = WorkflowExecutor(template_library)
    
    # Get template for threat hunting
    template = template_library.get_template(ProblemType.THREAT_HUNTING)
    
    if template:
        # Create context
        context = WorkflowContext(
            problem_type=ProblemType.THREAT_HUNTING,
            problem_description="Investigate potential APT29 activity",
            priority=4,
            complexity=7
        )
        
        # Customize template
        customized_template = template_library.customize_template(template, context)
        
        print(f"🚀 Executing {context.problem_type.value} workflow")
        print(f"   Priority: {context.priority}")
        print(f"   Complexity: {context.complexity}")
        print(f"   Nodes: {len(customized_template.nodes)}")
        
        # Execute workflow
        results = await executor.execute_template(customized_template, context)
        
        print(f"✅ Workflow completed")
        print(f"   Results: {len(results)} nodes executed")
        for node_id, result in results.items():
            print(f"   {node_id}: {result.get('result', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())

