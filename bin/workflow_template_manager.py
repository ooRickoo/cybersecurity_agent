#!/usr/bin/env python3
"""
Workflow Template Manager for Cybersecurity Agent
Handles backtracking and alternative approach generation when verification fails
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import sys

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from workflow_verification_system import get_workflow_verifier, VerificationResult
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowTemplateType(Enum):
    """Workflow template type enumeration."""
    THREAT_ANALYSIS = "threat_analysis"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    COMPLIANCE_AUDIT = "compliance_audit"
    GENERAL_INVESTIGATION = "general_investigation"
    SECURITY_ASSESSMENT = "security_assessment"
    FORENSIC_ANALYSIS = "forensic_analysis"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class WorkflowTemplate:
    """Workflow template definition."""
    template_id: str
    template_type: WorkflowTemplateType
    name: str
    description: str
    steps: List[Dict[str, Any]]
    prerequisites: List[str]
    expected_outcomes: List[str]
    success_criteria: Dict[str, Any]
    alternative_templates: List[str]
    created_at: datetime
    last_used: datetime
    success_rate: float
    usage_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['template_type'] = self.template_type.value
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTemplate':
        """Create from dictionary."""
        data['template_type'] = WorkflowTemplateType(data['template_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)

@dataclass
class BacktrackDecision:
    """Decision made during backtracking."""
    decision_id: str
    original_execution_id: str
    reason: str
    selected_template: str
    alternative_approaches: List[str]
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

class WorkflowTemplateManager:
    """Manages workflow templates and handles backtracking decisions."""
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.backtrack_history: Dict[str, BacktrackDecision] = {}
        self.execution_templates: Dict[str, str] = {}  # execution_id -> template_id
        self._load_default_templates()
        
        if VERIFICATION_AVAILABLE:
            self.verifier = get_workflow_verifier()
        else:
            self.verifier = None
    
    def _load_default_templates(self):
        """Load default workflow templates."""
        default_templates = {
            "threat_analysis_basic": {
                "template_id": "threat_analysis_basic",
                "template_type": WorkflowTemplateType.THREAT_ANALYSIS,
                "name": "Basic Threat Analysis",
                "description": "Standard threat analysis workflow for common scenarios",
                "steps": [
                    {
                        "step_id": "threat_intel_gathering",
                        "step_type": "data_collection",
                        "description": "Gather threat intelligence from multiple sources",
                        "tools": ["threat_intel_tool", "osint_tools"],
                        "expected_duration": 5.0,
                        "success_criteria": {"data_sources": 3, "indicators_found": True}
                    },
                    {
                        "step_id": "threat_assessment",
                        "step_type": "analysis",
                        "description": "Assess threat level and potential impact",
                        "tools": ["risk_assessment_tool", "threat_modeling"],
                        "expected_duration": 8.0,
                        "success_criteria": {"risk_score": "calculated", "impact_assessed": True}
                    },
                    {
                        "step_id": "mitigation_planning",
                        "step_type": "planning",
                        "description": "Develop mitigation strategies and response plan",
                        "tools": ["planning_tools", "documentation"],
                        "expected_duration": 6.0,
                        "success_criteria": {"strategies_defined": True, "plan_documented": True}
                    }
                ],
                "prerequisites": ["threat_intel_access", "risk_assessment_capability"],
                "expected_outcomes": ["threat_level_assessed", "mitigation_plan_created"],
                "success_criteria": {"accuracy_threshold": 0.8, "completeness": 0.9},
                "alternative_templates": ["threat_analysis_advanced", "threat_analysis_rapid"],
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "success_rate": 0.85,
                "usage_count": 0
            },
            "threat_analysis_advanced": {
                "template_id": "threat_analysis_advanced",
                "template_type": WorkflowTemplateType.THREAT_ANALYSIS,
                "name": "Advanced Threat Analysis",
                "description": "Comprehensive threat analysis with deep investigation",
                "steps": [
                    {
                        "step_id": "comprehensive_intel_gathering",
                        "step_type": "data_collection",
                        "description": "Comprehensive threat intelligence collection",
                        "tools": ["threat_intel_tool", "osint_tools", "dark_web_monitoring"],
                        "expected_duration": 10.0,
                        "success_criteria": {"data_sources": 5, "indicators_found": True}
                    },
                    {
                        "step_id": "deep_threat_analysis",
                        "step_type": "analysis",
                        "description": "Deep analysis of threat patterns and behaviors",
                        "tools": ["malware_analysis", "behavioral_analysis", "pattern_recognition"],
                        "expected_duration": 15.0,
                        "success_criteria": {"patterns_identified": True, "behaviors_analyzed": True}
                    },
                    {
                        "step_id": "threat_modeling",
                        "step_type": "modeling",
                        "description": "Create detailed threat models and attack trees",
                        "tools": ["threat_modeling_tool", "attack_tree_builder"],
                        "expected_duration": 12.0,
                        "success_criteria": {"models_created": True, "attack_paths_identified": True}
                    },
                    {
                        "step_id": "comprehensive_mitigation",
                        "step_type": "planning",
                        "description": "Develop comprehensive mitigation and response strategies",
                        "tools": ["planning_tools", "documentation", "stakeholder_management"],
                        "expected_duration": 10.0,
                        "success_criteria": {"strategies_defined": True, "stakeholders_engaged": True}
                    }
                ],
                "prerequisites": ["advanced_threat_intel", "malware_analysis_capability", "threat_modeling_expertise"],
                "expected_outcomes": ["comprehensive_threat_assessment", "detailed_mitigation_plan"],
                "success_criteria": {"accuracy_threshold": 0.9, "completeness": 0.95},
                "alternative_templates": ["threat_analysis_basic", "threat_analysis_rapid"],
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "success_rate": 0.92,
                "usage_count": 0
            },
            "incident_response_standard": {
                "template_id": "incident_response_standard",
                "template_type": WorkflowTemplateType.INCIDENT_RESPONSE,
                "name": "Standard Incident Response",
                "description": "Standard incident response workflow for security incidents",
                "steps": [
                    {
                        "step_id": "incident_detection",
                        "step_type": "detection",
                        "description": "Detect and classify security incident",
                        "tools": ["siem", "alerting_system", "monitoring_tools"],
                        "expected_duration": 2.0,
                        "success_criteria": {"incident_classified": True, "severity_assessed": True}
                    },
                    {
                        "step_id": "initial_assessment",
                        "step_type": "assessment",
                        "description": "Perform initial incident assessment",
                        "tools": ["forensic_tools", "log_analysis", "network_analysis"],
                        "expected_duration": 6.0,
                        "success_criteria": {"scope_defined": True, "impact_assessed": True}
                    },
                    {
                        "step_id": "containment",
                        "step_type": "response",
                        "description": "Contain the incident and prevent further damage",
                        "tools": ["network_isolation", "access_control", "system_quarantine"],
                        "expected_duration": 4.0,
                        "success_criteria": {"incident_contained": True, "further_damage_prevented": True}
                    },
                    {
                        "step_id": "eradication",
                        "step_type": "response",
                        "description": "Remove threat and restore systems",
                        "tools": ["malware_removal", "system_restoration", "security_patching"],
                        "expected_duration": 8.0,
                        "success_criteria": {"threat_removed": True, "systems_restored": True}
                    },
                    {
                        "step_id": "recovery",
                        "step_type": "recovery",
                        "description": "Restore normal operations and implement lessons learned",
                        "tools": ["system_validation", "documentation", "training"],
                        "expected_duration": 6.0,
                        "success_criteria": {"operations_restored": True, "lessons_learned": True}
                    }
                ],
                "prerequisites": ["incident_response_team", "forensic_capability", "communication_plan"],
                "expected_outcomes": ["incident_resolved", "lessons_learned_documented"],
                "success_criteria": {"resolution_time": "within_sla", "lessons_learned": True},
                "alternative_templates": ["incident_response_rapid", "incident_response_comprehensive"],
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "success_rate": 0.88,
                "usage_count": 0
            },
            "vulnerability_assessment_comprehensive": {
                "template_id": "vulnerability_assessment_comprehensive",
                "template_type": WorkflowTemplateType.VULNERABILITY_ASSESSMENT,
                "name": "Comprehensive Vulnerability Assessment",
                "description": "Complete vulnerability assessment with remediation planning",
                "steps": [
                    {
                        "step_id": "asset_discovery",
                        "step_type": "discovery",
                        "description": "Discover and inventory all assets",
                        "tools": ["asset_discovery_tool", "network_scanner", "inventory_management"],
                        "expected_duration": 4.0,
                        "success_criteria": {"assets_inventoried": True, "scope_defined": True}
                    },
                    {
                        "step_id": "vulnerability_scanning",
                        "step_type": "scanning",
                        "description": "Perform comprehensive vulnerability scanning",
                        "tools": ["vulnerability_scanner", "configuration_analyzer", "patch_analyzer"],
                        "expected_duration": 8.0,
                        "success_criteria": {"vulnerabilities_identified": True, "scan_complete": True}
                    },
                    {
                        "step_id": "risk_assessment",
                        "step_type": "assessment",
                        "description": "Assess risk and prioritize vulnerabilities",
                        "tools": ["risk_assessment_tool", "threat_modeling", "business_impact_analysis"],
                        "expected_duration": 10.0,
                        "success_criteria": {"risks_assessed": True, "vulnerabilities_prioritized": True}
                    },
                    {
                        "step_id": "remediation_planning",
                        "step_type": "planning",
                        "description": "Develop remediation plan and timeline",
                        "resources": ["remediation_tools", "stakeholder_management", "project_planning"],
                        "expected_duration": 6.0,
                        "success_criteria": {"plan_developed": True, "timeline_established": True}
                    }
                ],
                "prerequisites": ["vulnerability_scanning_capability", "risk_assessment_expertise", "stakeholder_access"],
                "expected_outcomes": ["vulnerabilities_assessed", "remediation_plan_created"],
                "success_criteria": {"assessment_complete": True, "plan_actionable": True},
                "alternative_templates": ["vulnerability_assessment_rapid", "vulnerability_assessment_focused"],
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "success_rate": 0.90,
                "usage_count": 0
            }
        }
        
        for template_data in default_templates.values():
            template = WorkflowTemplate(**template_data)
            self.templates[template.template_id] = template
    
    def select_template_for_question(self, question: str, question_type: str = None) -> Optional[WorkflowTemplate]:
        """Select the most appropriate template for a given question."""
        try:
            if not question_type:
                question_type = self._classify_question(question)
            
            # Get templates of the appropriate type
            available_templates = [
                template for template in self.templates.values()
                if template.template_type.value == question_type
            ]
            
            if not available_templates:
                # Fallback to general investigation
                available_templates = [
                    template for template in self.templates.values()
                    if template.template_type == WorkflowTemplateType.GENERAL_INVESTIGATION
                ]
            
            if not available_templates:
                return None
            
            # Score templates based on question characteristics
            scored_templates = []
            for template in available_templates:
                score = self._score_template_for_question(template, question)
                scored_templates.append((template, score))
            
            # Sort by score and return the best match
            scored_templates.sort(key=lambda x: x[1], reverse=True)
            return scored_templates[0][0]
            
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return None
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked."""
        question_lower = question.lower()
        
        # Define classification patterns
        classification_patterns = {
            "threat_analysis": ["threat", "threats", "malware", "attack", "attacks", "campaign", "landscape"],
            "incident_response": ["incident", "breach", "compromise", "intrusion", "alert", "response"],
            "vulnerability_assessment": ["vulnerability", "vulnerabilities", "patch", "security", "weakness"],
            "compliance_audit": ["compliance", "audit", "policy", "regulation", "standard", "requirement"],
            "forensic_analysis": ["forensic", "evidence", "investigation", "timeline", "chain", "custody"],
            "risk_assessment": ["risk", "assessment", "evaluate", "probability", "impact", "mitigation"]
        }
        
        for question_type, patterns in classification_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return question_type
        
        return "general_investigation"
    
    def _score_template_for_question(self, template: WorkflowTemplate, question: str) -> float:
        """Score how well a template fits a given question."""
        score = 0.0
        
        # Base score from success rate
        score += template.success_rate * 0.4
        
        # Question complexity matching
        question_complexity = self._assess_question_complexity(question)
        template_complexity = len(template.steps)
        
        if question_complexity == "simple" and template_complexity <= 3:
            score += 0.3
        elif question_complexity == "moderate" and 3 < template_complexity <= 5:
            score += 0.3
        elif question_complexity == "complex" and template_complexity > 5:
            score += 0.3
        
        # Recency bonus (prefer recently used templates)
        days_since_use = (datetime.now() - template.last_used).days
        if days_since_use < 7:
            score += 0.1
        elif days_since_use < 30:
            score += 0.05
        
        # Usage count bonus (prefer proven templates)
        if template.usage_count > 10:
            score += 0.1
        elif template.usage_count > 5:
            score += 0.05
        
        return min(1.0, score)
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess the complexity of a question."""
        word_count = len(question.split())
        question_lower = question.lower()
        
        # Complexity indicators
        simple_indicators = ["what is", "how to", "basic", "simple", "quick"]
        complex_indicators = ["comprehensive", "detailed", "thorough", "complete", "extensive", "analysis"]
        
        if any(indicator in question_lower for indicator in complex_indicators):
            return "complex"
        elif any(indicator in question_lower for indicator in simple_indicators) or word_count < 10:
            return "simple"
        else:
            return "moderate"
    
    def handle_backtracking(self, 
                           original_execution_id: str,
                           verification_result: VerificationResult,
                           original_question: str) -> Dict[str, Any]:
        """Handle backtracking when verification fails."""
        try:
            logger.info(f"Handling backtracking for execution {original_execution_id}")
            
            # Create backtrack decision
            decision_id = str(uuid.uuid4())
            decision = BacktrackDecision(
                decision_id=decision_id,
                original_execution_id=original_execution_id,
                reason=verification_result.backtrack_reason or "Verification failed",
                selected_template="",
                alternative_approaches=verification_result.alternative_approaches or [],
                confidence=0.0,
                created_at=datetime.now()
            )
            
            # Analyze why the original approach failed
            failure_analysis = self._analyze_failure_reasons(verification_result)
            
            # Select alternative template
            alternative_template = self._select_alternative_template(
                original_question, failure_analysis, verification_result
            )
            
            if alternative_template:
                decision.selected_template = alternative_template.template_id
                decision.confidence = self._calculate_backtrack_confidence(
                    alternative_template, failure_analysis
                )
                
                # Update template usage
                alternative_template.last_used = datetime.now()
                alternative_template.usage_count += 1
                
                # Store decision
                self.backtrack_history[decision_id] = decision
                
                return {
                    "success": True,
                    "decision_id": decision_id,
                    "selected_template": alternative_template.template_id,
                    "template_name": alternative_template.name,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "alternative_approaches": verification_result.alternative_approaches,
                    "recommended_steps": alternative_template.steps
                }
            else:
                # No suitable alternative template found
                decision.confidence = 0.0
                self.backtrack_history[decision_id] = decision
                
                return {
                    "success": False,
                    "decision_id": decision_id,
                    "reason": "No suitable alternative template found",
                    "recommendations": [
                        "Break down the problem into smaller components",
                        "Use manual analysis approach",
                        "Consult with domain experts",
                        "Consider using different tools or data sources"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error handling backtracking: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Review workflow manually", "Check system logs"]
            }
    
    def _analyze_failure_reasons(self, verification_result: VerificationResult) -> Dict[str, Any]:
        """Analyze why the original workflow failed."""
        failure_analysis = {
            "accuracy_issues": [],
            "completeness_issues": [],
            "evidence_issues": [],
            "logical_issues": [],
            "tool_issues": []
        }
        
        for issue in verification_result.issues_found:
            issue_lower = issue.lower()
            
            if "answer" in issue_lower and "question" in issue_lower:
                failure_analysis["accuracy_issues"].append(issue)
            elif "incomplete" in issue_lower or "steps" in issue_lower:
                failure_analysis["completeness_issues"].append(issue)
            elif "evidence" in issue_lower or "insufficient" in issue_lower:
                failure_analysis["evidence_issues"].append(issue)
            elif "logical" in issue_lower or "inconsistencies" in issue_lower:
                failure_analysis["logical_issues"].append(issue)
            elif "tool" in issue_lower or "usage" in issue_lower:
                failure_analysis["tool_issues"].append(issue)
            else:
                # Default to accuracy issues
                failure_analysis["accuracy_issues"].append(issue)
        
        return failure_analysis
    
    def _select_alternative_template(self, 
                                   question: str,
                                   failure_analysis: Dict[str, Any],
                                   verification_result: VerificationResult) -> Optional[WorkflowTemplate]:
        """Select an alternative template based on failure analysis."""
        try:
            # Determine question type
            question_type = self._classify_question(question)
            
            # Get available templates
            available_templates = [
                template for template in self.templates.values()
                if template.template_type.value == question_type
            ]
            
            if not available_templates:
                return None
            
            # Score templates based on failure analysis
            scored_templates = []
            for template in available_templates:
                score = self._score_template_for_failure_recovery(
                    template, failure_analysis, verification_result
                )
                scored_templates.append((template, score))
            
            # Sort by score and return the best match
            scored_templates.sort(key=lambda x: x[1], reverse=True)
            
            if scored_templates and scored_templates[0][1] > 0.5:
                return scored_templates[0][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting alternative template: {e}")
            return None
    
    def _score_template_for_failure_recovery(self, 
                                           template: WorkflowTemplate,
                                           failure_analysis: Dict[str, Any],
                                           verification_result: VerificationResult) -> float:
        """Score how well a template addresses the failure reasons."""
        score = 0.0
        
        # Base score from template success rate
        score += template.success_rate * 0.3
        
        # Address completeness issues
        if failure_analysis["completeness_issues"]:
            if len(template.steps) > 3:  # More comprehensive template
                score += 0.2
        
        # Address evidence issues
        if failure_analysis["evidence_issues"]:
            # Check if template has data collection steps
            data_collection_steps = [step for step in template.steps 
                                   if step.get("step_type") == "data_collection"]
            if data_collection_steps:
                score += 0.2
        
        # Address logical issues
        if failure_analysis["logical_issues"]:
            # Check for structured analysis steps
            analysis_steps = [step for step in template.steps 
                            if step.get("step_type") in ["analysis", "assessment", "validation"]]
            if len(analysis_steps) >= 2:
                score += 0.2
        
        # Address tool issues
        if failure_analysis["tool_issues"]:
            # Check for diverse tool usage
            all_tools = []
            for step in template.steps:
                if step.get("tools"):
                    all_tools.extend(step["tools"])
            if len(set(all_tools)) > 2:  # Multiple tools
                score += 0.1
        
        # Prefer templates with higher success rates
        if template.success_rate > 0.9:
            score += 0.1
        elif template.success_rate > 0.8:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_backtrack_confidence(self, 
                                      template: WorkflowTemplate,
                                      failure_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the backtracking decision."""
        confidence = 0.5  # Base confidence
        
        # Template success rate
        confidence += template.success_rate * 0.3
        
        # Address failure reasons
        addressed_issues = 0
        total_issues = sum(len(issues) for issues in failure_analysis.values())
        
        if total_issues > 0:
            if failure_analysis["completeness_issues"] and len(template.steps) > 3:
                addressed_issues += len(failure_analysis["completeness_issues"])
            
            if failure_analysis["evidence_issues"]:
                data_steps = [step for step in template.steps 
                             if step.get("step_type") == "data_collection"]
                if data_steps:
                    addressed_issues += len(failure_analysis["evidence_issues"])
            
            if failure_analysis["logical_issues"]:
                analysis_steps = [step for step in template.steps 
                                if step.get("step_type") in ["analysis", "assessment"]]
                if len(analysis_steps) >= 2:
                    addressed_issues += len(failure_analysis["logical_issues"])
            
            confidence += (addressed_issues / total_issues) * 0.2
        
        return min(1.0, confidence)
    
    def get_template_variations(self, template_id: str) -> List[WorkflowTemplate]:
        """Get variations of a template for different complexity levels."""
        if template_id not in self.templates:
            return []
        
        base_template = self.templates[template_id]
        variations = []
        
        # Create simplified version
        if len(base_template.steps) > 3:
            simplified_steps = base_template.steps[:3]  # Take first 3 steps
            simplified_template = WorkflowTemplate(
                template_id=f"{template_id}_simplified",
                template_type=base_template.template_type,
                name=f"{base_template.name} (Simplified)",
                description=f"Simplified version of {base_template.name}",
                steps=simplified_steps,
                prerequisites=base_template.prerequisites[:2],  # Reduce prerequisites
                expected_outcomes=base_template.expected_outcomes,
                success_criteria=base_template.success_criteria,
                alternative_templates=[template_id],
                created_at=datetime.now(),
                last_used=datetime.now(),
                success_rate=base_template.success_rate * 0.9,  # Slightly lower success rate
                usage_count=0
            )
            variations.append(simplified_template)
        
        # Create comprehensive version
        if len(base_template.steps) < 6:
            # Add additional analysis and validation steps
            comprehensive_steps = base_template.steps.copy()
            comprehensive_steps.extend([
                {
                    "step_id": "validation",
                    "step_type": "validation",
                    "description": "Validate results and conclusions",
                    "tools": ["validation_tools", "peer_review"],
                    "expected_duration": 4.0,
                    "success_criteria": {"results_validated": True, "peer_review_complete": True}
                },
                {
                    "step_id": "documentation",
                    "step_type": "documentation",
                    "description": "Comprehensive documentation and reporting",
                    "tools": ["documentation_tools", "reporting_framework"],
                    "expected_duration": 3.0,
                    "success_criteria": {"documentation_complete": True, "report_generated": True}
                }
            ])
            
            comprehensive_template = WorkflowTemplate(
                template_id=f"{template_id}_comprehensive",
                template_type=base_template.template_type,
                name=f"{base_template.name} (Comprehensive)",
                description=f"Comprehensive version of {base_template.name}",
                steps=comprehensive_steps,
                prerequisites=base_template.prerequisites + ["validation_capability", "documentation_tools"],
                expected_outcomes=base_template.expected_outcomes + ["results_validated", "comprehensive_documentation"],
                success_criteria=base_template.success_criteria,
                alternative_templates=[template_id],
                created_at=datetime.now(),
                last_used=datetime.now(),
                success_rate=min(1.0, base_template.success_rate * 1.05),  # Slightly higher success rate
                usage_count=0
            )
            variations.append(comprehensive_template)
        
        return variations
    
    def get_backtrack_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get summary of backtracking decisions for an execution."""
        backtrack_decisions = [
            decision for decision in self.backtrack_history.values()
            if decision.original_execution_id == execution_id
        ]
        
        if not backtrack_decisions:
            return {"backtrack_count": 0, "decisions": []}
        
        return {
            "backtrack_count": len(backtrack_decisions),
            "decisions": [
                {
                    "decision_id": decision.decision_id,
                    "reason": decision.reason,
                    "selected_template": decision.selected_template,
                    "confidence": decision.confidence,
                    "created_at": decision.created_at.isoformat()
                }
                for decision in backtrack_decisions
            ]
        }
    
    def get_template_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all templates."""
        stats = {}
        
        for template in self.templates.values():
            template_type = template.template_type.value
            if template_type not in stats:
                stats[template_type] = {
                    "total_templates": 0,
                    "total_usage": 0,
                    "avg_success_rate": 0.0,
                    "best_template": None,
                    "best_success_rate": 0.0
                }
            
            stats[template_type]["total_templates"] += 1
            stats[template_type]["total_usage"] += template.usage_count
            
            if template.success_rate > stats[template_type]["best_success_rate"]:
                stats[template_type]["best_success_rate"] = template.success_rate
                stats[template_type]["best_template"] = template.template_id
        
        # Calculate averages
        for template_type in stats:
            templates_of_type = [t for t in self.templates.values() 
                               if t.template_type.value == template_type]
            if templates_of_type:
                avg_success = sum(t.success_rate for t in templates_of_type) / len(templates_of_type)
                stats[template_type]["avg_success_rate"] = round(avg_success, 3)
        
        return stats

# Global instance
_workflow_template_manager = None

def get_workflow_template_manager() -> WorkflowTemplateManager:
    """Get or create the global workflow template manager instance."""
    global _workflow_template_manager
    if _workflow_template_manager is None:
        _workflow_template_manager = WorkflowTemplateManager()
    return _workflow_template_manager

if __name__ == "__main__":
    # Test the template manager
    manager = get_workflow_template_manager()
    
    print("🧪 Testing Workflow Template Manager...")
    
    # Test template selection
    question = "What is the current threat landscape and how should we respond?"
    template = manager.select_template_for_question(question)
    
    if template:
        print(f"✅ Selected template: {template.name}")
        print(f"   Type: {template.template_type.value}")
        print(f"   Steps: {len(template.steps)}")
        print(f"   Success Rate: {template.success_rate}")
    else:
        print("❌ No template selected")
    
    # Test backtracking (simulated)
    if VERIFICATION_AVAILABLE:
        from workflow_verification_system import VerificationResult
        
        # Simulate failed verification
        failed_verification = VerificationResult(
            accuracy_score=0.4,
            confidence_level="low",
            issues_found=["Insufficient evidence", "Incomplete analysis"],
            recommendations=["Gather more data", "Use comprehensive approach"],
            needs_backtrack=True,
            backtrack_reason="Accuracy score below threshold"
        )
        
        backtrack_result = manager.handle_backtracking(
            "test_exec_001",
            failed_verification,
            question
        )
        
        print(f"Backtracking result: {backtrack_result}")
    
    # Get performance stats
    stats = manager.get_template_performance_stats()
    print(f"Template performance stats: {stats}")
