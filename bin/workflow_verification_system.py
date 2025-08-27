#!/usr/bin/env python3
"""
Workflow Verification System for Cybersecurity Agent
Provides "Check our math" verification with intelligent backtracking and loop prevention
"""

import json
import hashlib
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Verification status enumeration."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    LOOP_DETECTED = "loop_detected"

class VerificationLevel(Enum):
    """Verification level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"

@dataclass
class WorkflowStep:
    """Individual workflow step information."""
    step_id: str
    step_type: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    tools_used: List[str]
    execution_time: float
    status: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

@dataclass
class WorkflowExecution:
    """Complete workflow execution record."""
    execution_id: str
    original_question: str
    question_hash: str
    workflow_template: str
    steps: List[WorkflowStep]
    final_answer: str
    verification_status: VerificationStatus
    verification_details: Dict[str, Any]
    execution_path: List[str]
    backtrack_count: int
    created_at: datetime
    completed_at: datetime
    total_execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['completed_at'] = self.completed_at.isoformat()
        data['verification_status'] = self.verification_status.value
        return data

@dataclass
class VerificationResult:
    """Result of verification analysis."""
    accuracy_score: float
    confidence_level: str
    issues_found: List[str]
    recommendations: List[str]
    needs_backtrack: bool
    backtrack_reason: Optional[str] = None
    alternative_approaches: List[str] = None

class WorkflowPathTracker:
    """Tracks execution paths to prevent loops."""
    
    def __init__(self):
        self.execution_paths: Dict[str, List[str]] = {}
        self.path_hashes: Dict[str, str] = {}
        self.loop_detection_threshold = 3
    
    def add_execution_path(self, execution_id: str, path: List[str]) -> bool:
        """Add execution path and check for loops."""
        path_hash = self._hash_path(path)
        
        # Check if this path has been used before
        if path_hash in self.path_hashes:
            existing_execution = self.path_hashes[path_hash]
            if existing_execution != execution_id:
                # Check if this path has been used too many times
                usage_count = sum(1 for p in self.execution_paths.values() if self._hash_path(p) == path_hash)
                if usage_count >= self.loop_detection_threshold:
                    logger.warning(f"Loop detected in execution path: {path}")
                    return False
        
        self.execution_paths[execution_id] = path
        self.path_hashes[path_hash] = execution_id
        return True
    
    def _hash_path(self, path: List[str]) -> str:
        """Create hash of execution path."""
        path_string = "->".join(path)
        return hashlib.md5(path_string.encode()).hexdigest()
    
    def get_path_variations(self, base_path: List[str]) -> List[List[str]]:
        """Get variations of a base path to suggest alternatives."""
        variations = []
        
        # Simple variations: reorder steps, combine steps, split steps
        if len(base_path) > 2:
            # Reorder variation
            reordered = base_path.copy()
            reordered[1], reordered[2] = reordered[2], reordered[1]
            variations.append(reordered)
            
            # Combine adjacent steps
            if len(base_path) > 3:
                combined = base_path.copy()
                combined[1] = f"{combined[1]}+{combined[2]}"
                combined.pop(2)
                variations.append(combined)
        
        return variations

class WorkflowVerifier:
    """Main workflow verification system."""
    
    def __init__(self):
        self.path_tracker = WorkflowPathTracker()
        self.execution_history: Dict[str, WorkflowExecution] = {}
        self.verification_templates: Dict[str, Dict[str, Any]] = {}
        self._load_verification_templates()
    
    def _load_verification_templates(self):
        """Load verification templates for different question types."""
        self.verification_templates = {
            "threat_analysis": {
                "accuracy_checks": [
                    "threat_indicators_present",
                    "risk_assessment_consistent",
                    "mitigation_strategies_appropriate"
                ],
                "confidence_thresholds": {
                    "high": 0.8,
                    "medium": 0.6,
                    "low": 0.4
                }
            },
            "incident_response": {
                "accuracy_checks": [
                    "timeline_consistent",
                    "evidence_chain_complete",
                    "response_actions_appropriate"
                ],
                "confidence_thresholds": {
                    "high": 0.85,
                    "medium": 0.65,
                    "low": 0.45
                }
            },
            "vulnerability_assessment": {
                "accuracy_checks": [
                    "vulnerability_identification_accurate",
                    "severity_assessment_consistent",
                    "patch_recommendations_valid"
                ],
                "confidence_thresholds": {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.5
                }
            },
            "compliance_audit": {
                "accuracy_checks": [
                    "requirements_coverage_complete",
                    "evidence_quality_adequate",
                    "gap_analysis_accurate"
                ],
                "confidence_thresholds": {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.5
                }
            },
            "general_investigation": {
                "accuracy_checks": [
                    "question_answered_completely",
                    "evidence_sufficient",
                    "conclusions_logical"
                ],
                "confidence_thresholds": {
                    "high": 0.8,
                    "medium": 0.6,
                    "low": 0.4
                }
            }
        }
    
    def verify_workflow(self, 
                       execution_id: str,
                       original_question: str,
                       workflow_steps: List[WorkflowStep],
                       final_answer: str,
                       question_type: str = "general_investigation") -> VerificationResult:
        """Perform comprehensive workflow verification."""
        try:
            logger.info(f"Starting verification for execution {execution_id}")
            
            # Create execution record
            execution = self._create_execution_record(
                execution_id, original_question, workflow_steps, final_answer
            )
            
            # Track execution path
            path = [step.step_type for step in workflow_steps]
            loop_detected = not self.path_tracker.add_execution_path(execution_id, path)
            
            if loop_detected:
                execution.verification_status = VerificationStatus.LOOP_DETECTED
                return VerificationResult(
                    accuracy_score=0.0,
                    confidence_level="none",
                    issues_found=["Execution loop detected - same path used multiple times"],
                    recommendations=["Use different approach or break down problem differently"],
                    needs_backtrack=True,
                    backtrack_reason="Loop detected in execution path",
                    alternative_approaches=self.path_tracker.get_path_variations(path)
                )
            
            # Perform verification analysis
            verification_result = self._perform_verification_analysis(
                original_question, workflow_steps, final_answer, question_type
            )
            
            # Update execution record
            execution.verification_status = (
                VerificationStatus.PASSED if verification_result.accuracy_score >= 0.8
                else VerificationStatus.NEEDS_REVIEW if verification_result.accuracy_score >= 0.6
                else VerificationStatus.FAILED
            )
            execution.verification_details = {
                "accuracy_score": verification_result.accuracy_score,
                "confidence_level": verification_result.confidence_level,
                "issues_found": verification_result.issues_found,
                "recommendations": verification_result.recommendations
            }
            
            # Store execution record
            self.execution_history[execution_id] = execution
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error during workflow verification: {e}")
            return VerificationResult(
                accuracy_score=0.0,
                confidence_level="none",
                issues_found=[f"Verification error: {str(e)}"],
                recommendations=["Review workflow execution manually"],
                needs_backtrack=True,
                backtrack_reason="Verification system error"
            )
    
    def _create_execution_record(self, 
                                execution_id: str,
                                original_question: str,
                                workflow_steps: List[WorkflowStep],
                                final_answer: str) -> WorkflowExecution:
        """Create execution record from workflow data."""
        question_hash = hashlib.md5(original_question.encode()).hexdigest()
        execution_path = [step.step_type for step in workflow_steps]
        total_time = sum(step.execution_time for step in workflow_steps)
        
        return WorkflowExecution(
            execution_id=execution_id,
            original_question=original_question,
            question_hash=question_hash,
            workflow_template="",  # Will be set by workflow manager
            steps=workflow_steps,
            final_answer=final_answer,
            verification_status=VerificationStatus.PENDING,
            verification_details={},
            execution_path=execution_path,
            backtrack_count=0,
            created_at=datetime.now(),
            completed_at=datetime.now(),
            total_execution_time=total_time
        )
    
    def _perform_verification_analysis(self,
                                     original_question: str,
                                     workflow_steps: List[WorkflowStep],
                                     final_answer: str,
                                     question_type: str) -> VerificationResult:
        """Perform detailed verification analysis."""
        issues_found = []
        recommendations = []
        
        # 1. Question-Answer Alignment Check
        alignment_score = self._check_question_answer_alignment(original_question, final_answer)
        
        # 2. Workflow Completeness Check
        completeness_score = self._check_workflow_completeness(workflow_steps, original_question)
        
        # 3. Evidence Quality Check
        evidence_score = self._check_evidence_quality(workflow_steps, final_answer)
        
        # 4. Logical Consistency Check
        consistency_score = self._check_logical_consistency(workflow_steps, final_answer)
        
        # 5. Tool Usage Appropriateness Check
        tool_score = self._check_tool_usage_appropriateness(workflow_steps, original_question)
        
        # Calculate overall accuracy score
        accuracy_score = (alignment_score + completeness_score + evidence_score + 
                         consistency_score + tool_score) / 5.0
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(accuracy_score, question_type)
        
        # Generate specific issues and recommendations
        if alignment_score < 0.7:
            issues_found.append("Answer doesn't fully address the original question")
            recommendations.append("Review question requirements and ensure complete coverage")
        
        if completeness_score < 0.7:
            issues_found.append("Workflow steps may be incomplete for the given question")
            recommendations.append("Consider additional analysis steps or data sources")
        
        if evidence_score < 0.7:
            issues_found.append("Insufficient evidence to support conclusions")
            recommendations.append("Gather additional data or use alternative analysis methods")
        
        if consistency_score < 0.7:
            issues_found.append("Logical inconsistencies detected in workflow or conclusions")
            recommendations.append("Review reasoning chain and validate assumptions")
        
        if tool_score < 0.7:
            issues_found.append("Tool usage may not be optimal for the given question")
            recommendations.append("Consider alternative tools or analysis approaches")
        
        # Determine if backtracking is needed
        needs_backtrack = accuracy_score < 0.6 or len(issues_found) > 2
        
        return VerificationResult(
            accuracy_score=accuracy_score,
            confidence_level=confidence_level,
            issues_found=issues_found,
            recommendations=recommendations,
            needs_backtrack=needs_backtrack,
            backtrack_reason=f"Accuracy score {accuracy_score:.2f} below threshold" if needs_backtrack else None,
            alternative_approaches=self._suggest_alternative_approaches(workflow_steps, original_question)
        )
    
    def _check_question_answer_alignment(self, question: str, answer: str) -> float:
        """Check how well the answer aligns with the question."""
        try:
            # Simple keyword matching for now - could be enhanced with NLP
            question_keywords = set(question.lower().split())
            answer_keywords = set(answer.lower().split())
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            question_keywords -= common_words
            answer_keywords -= common_words
            
            if not question_keywords:
                return 0.5  # Neutral score if no meaningful keywords
            
            # Calculate overlap
            overlap = len(question_keywords.intersection(answer_keywords))
            total_unique = len(question_keywords.union(answer_keywords))
            
            if total_unique == 0:
                return 0.5
            
            alignment_score = overlap / total_unique
            
            # Boost score if answer is comprehensive
            if len(answer) > len(question) * 2:
                alignment_score = min(1.0, alignment_score * 1.2)
            
            return min(1.0, max(0.0, alignment_score))
            
        except Exception as e:
            logger.error(f"Error in question-answer alignment check: {e}")
            return 0.5
    
    def _check_workflow_completeness(self, steps: List[WorkflowStep], question: str) -> float:
        """Check if workflow steps are complete for the given question."""
        try:
            # Define expected step patterns for different question types
            expected_patterns = {
                "threat": ["analysis", "assessment", "mitigation"],
                "incident": ["detection", "analysis", "response"],
                "vulnerability": ["scanning", "assessment", "remediation"],
                "compliance": ["audit", "gap_analysis", "recommendations"],
                "investigation": ["data_collection", "analysis", "conclusions"]
            }
            
            # Determine question type
            question_lower = question.lower()
            question_type = "investigation"  # default
            
            for pattern_type, keywords in expected_patterns.items():
                if any(keyword in question_lower for keyword in keywords):
                    question_type = pattern_type
                    break
            
            # Check if expected steps are present
            step_types = [step.step_type.lower() for step in steps]
            expected_steps = expected_patterns.get(question_type, [])
            
            if not expected_steps:
                return 0.7  # Neutral score for unknown question types
            
            # Calculate completeness
            present_steps = sum(1 for expected in expected_steps 
                              if any(expected in step_type for step_type in step_types))
            
            completeness_score = present_steps / len(expected_steps)
            
            # Bonus for having more steps than minimum
            if len(steps) > len(expected_steps):
                completeness_score = min(1.0, completeness_score * 1.1)
            
            return min(1.0, max(0.0, completeness_score))
            
        except Exception as e:
            logger.error(f"Error in workflow completeness check: {e}")
            return 0.7
    
    def _check_evidence_quality(self, steps: List[WorkflowStep], answer: str) -> float:
        """Check the quality of evidence supporting the answer."""
        try:
            evidence_score = 0.5  # Base score
            
            # Check for data sources used
            data_sources = []
            for step in steps:
                if step.tools_used:
                    data_sources.extend(step.tools_used)
            
            if data_sources:
                evidence_score += 0.2
            
            # Check for specific data in outputs
            specific_data_count = 0
            for step in steps:
                if step.outputs and isinstance(step.outputs, dict):
                    # Count non-empty outputs
                    specific_data_count += sum(1 for v in step.outputs.values() if v)
            
            if specific_data_count > 0:
                evidence_score += min(0.3, specific_data_count * 0.1)
            
            # Check answer for specific details
            if any(char.isdigit() for char in answer):
                evidence_score += 0.1
            
            if len(answer.split()) > 50:  # Comprehensive answer
                evidence_score += 0.1
            
            return min(1.0, max(0.0, evidence_score))
            
        except Exception as e:
            logger.error(f"Error in evidence quality check: {e}")
            return 0.5
    
    def _check_logical_consistency(self, steps: List[WorkflowStep], answer: str) -> float:
        """Check logical consistency of workflow and conclusions."""
        try:
            consistency_score = 0.7  # Base score
            
            # Check for logical flow in step sequence
            step_types = [step.step_type for step in steps]
            
            # Look for logical patterns
            if len(step_types) >= 2:
                # Check for analysis before conclusions
                if "analysis" in step_types and "conclusion" in step_types:
                    analysis_index = step_types.index("analysis")
                    conclusion_index = step_types.index("conclusion")
                    if analysis_index < conclusion_index:
                        consistency_score += 0.2
                
                # Check for data collection before analysis
                if "data_collection" in step_types and "analysis" in step_types:
                    collection_index = step_types.index("data_collection")
                    analysis_index = step_types.index("analysis")
                    if collection_index < analysis_index:
                        consistency_score += 0.1
            
            # Check answer for logical indicators
            logical_indicators = ["because", "therefore", "as a result", "consequently", "thus"]
            if any(indicator in answer.lower() for indicator in logical_indicators):
                consistency_score += 0.1
            
            return min(1.0, max(0.0, consistency_score))
            
        except Exception as e:
            logger.error(f"Error in logical consistency check: {e}")
            return 0.7
    
    def _check_tool_usage_appropriateness(self, steps: List[WorkflowStep], question: str) -> float:
        """Check if tools were used appropriately for the question."""
        try:
            tool_score = 0.7  # Base score
            
            # Check for appropriate tool usage
            question_lower = question.lower()
            tools_used = []
            for step in steps:
                tools_used.extend(step.tools_used)
            
            # Define appropriate tools for different question types
            appropriate_tools = {
                "threat": ["threat_intel", "malware_analysis", "network_analysis"],
                "incident": ["log_analysis", "forensics", "timeline_analysis"],
                "vulnerability": ["vulnerability_scanner", "penetration_testing", "risk_assessment"],
                "compliance": ["audit_tools", "policy_analyzer", "gap_analyzer"],
                "investigation": ["data_collection", "analysis_tools", "reporting"]
            }
            
            # Determine question type
            question_type = "investigation"
            for pattern_type in appropriate_tools.keys():
                if pattern_type in question_lower:
                    question_type = pattern_type
                    break
            
            # Check tool appropriateness
            expected_tools = appropriate_tools.get(question_type, [])
            if expected_tools:
                appropriate_usage = sum(1 for tool in tools_used 
                                     if any(expected in tool.lower() for expected in expected_tools))
                if appropriate_usage > 0:
                    tool_score += 0.2
            
            # Bonus for using multiple tools
            if len(set(tools_used)) > 1:
                tool_score += 0.1
            
            return min(1.0, max(0.0, tool_score))
            
        except Exception as e:
            logger.error(f"Error in tool usage appropriateness check: {e}")
            return 0.7
    
    def _determine_confidence_level(self, accuracy_score: float, question_type: str) -> str:
        """Determine confidence level based on accuracy score and question type."""
        template = self.verification_templates.get(question_type, self.verification_templates["general_investigation"])
        thresholds = template["confidence_thresholds"]
        
        if accuracy_score >= thresholds["high"]:
            return "high"
        elif accuracy_score >= thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _suggest_alternative_approaches(self, steps: List[WorkflowStep], question: str) -> List[str]:
        """Suggest alternative approaches for the workflow."""
        alternatives = []
        
        # Get current approach
        current_approach = [step.step_type for step in steps]
        
        # Suggest variations
        if len(current_approach) > 2:
            # Alternative 1: Reorder steps
            reordered = current_approach.copy()
            if len(reordered) >= 3:
                reordered[1], reordered[2] = reordered[2], reordered[1]
                alternatives.append(f"Reorder approach: {' -> '.join(reordered)}")
            
            # Alternative 2: Combine steps
            if len(current_approach) >= 4:
                combined = current_approach.copy()
                combined[1] = f"{combined[1]}+{combined[2]}"
                combined.pop(2)
                alternatives.append(f"Combined approach: {' -> '.join(combined)}")
            
            # Alternative 3: Split complex steps
            for i, step in enumerate(current_approach):
                if step in ["comprehensive_analysis", "detailed_investigation", "full_assessment"]:
                    split_approach = current_approach.copy()
                    split_approach[i:i+1] = ["preliminary_analysis", "detailed_analysis", "validation"]
                    alternatives.append(f"Split approach: {' -> '.join(split_approach)}")
                    break
        
        # Add general alternatives
        alternatives.extend([
            "Use different data sources",
            "Apply alternative analysis methodologies",
            "Break down into smaller sub-problems",
            "Use expert consultation or external tools"
        ])
        
        return alternatives
    
    def get_verification_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive verification summary for an execution."""
        if execution_id not in self.execution_history:
            return {"error": "Execution not found"}
        
        execution = self.execution_history[execution_id]
        
        return {
            "execution_id": execution_id,
            "original_question": execution.original_question,
            "verification_status": execution.verification_status.value,
            "accuracy_score": execution.verification_details.get("accuracy_score", 0.0),
            "confidence_level": execution.verification_details.get("confidence_level", "unknown"),
            "issues_found": execution.verification_details.get("issues_found", []),
            "recommendations": execution.verification_details.get("recommendations", []),
            "execution_path": execution.execution_path,
            "total_steps": len(execution.steps),
            "total_execution_time": execution.total_execution_time,
            "backtrack_count": execution.backtrack_count,
            "created_at": execution.created_at.isoformat(),
            "completed_at": execution.completed_at.isoformat()
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        sorted_executions = sorted(
            self.execution_history.values(),
            key=lambda x: x.completed_at,
            reverse=True
        )
        
        return [
            {
                "execution_id": exec.execution_id,
                "question": exec.original_question[:100] + "..." if len(exec.original_question) > 100 else exec.original_question,
                "status": exec.verification_status.value,
                "accuracy": exec.verification_details.get("accuracy_score", 0.0),
                "steps": len(exec.steps),
                "completed_at": exec.completed_at.isoformat()
            }
            for exec in sorted_executions[:limit]
        ]

# Global instance
_workflow_verifier = None

def get_workflow_verifier() -> WorkflowVerifier:
    """Get or create the global workflow verifier instance."""
    global _workflow_verifier
    if _workflow_verifier is None:
        _workflow_verifier = WorkflowVerifier()
    return _workflow_verifier

if __name__ == "__main__":
    # Test the verification system
    verifier = get_workflow_verifier()
    
    print("🧪 Testing Workflow Verification System...")
    
    # Test verification
    test_steps = [
        WorkflowStep(
            step_id="step1",
            step_type="data_collection",
            description="Collect threat intelligence data",
            inputs={"source": "threat_feeds"},
            outputs={"threat_data": "malware_indicators"},
            tools_used=["threat_intel_tool"],
            execution_time=2.5,
            status="completed"
        ),
        WorkflowStep(
            step_id="step2",
            step_type="analysis",
            description="Analyze threat patterns",
            inputs={"threat_data": "malware_indicators"},
            outputs={"analysis_results": "threat_assessment"},
            tools_used=["analysis_tool"],
            execution_time=3.2,
            status="completed"
        ),
        WorkflowStep(
            step_id="step3",
            step_type="conclusion",
            description="Provide threat assessment",
            inputs={"analysis_results": "threat_assessment"},
            outputs={"final_answer": "High-risk malware campaign detected"},
            tools_used=["reporting_tool"],
            execution_time=1.8,
            status="completed"
        )
    ]
    
    result = verifier.verify_workflow(
        execution_id="test_exec_001",
        original_question="What is the current threat landscape?",
        workflow_steps=test_steps,
        final_answer="Based on analysis of threat intelligence feeds, we have detected a high-risk malware campaign targeting financial institutions. The campaign uses sophisticated techniques and has been observed in multiple regions.",
        question_type="threat_analysis"
    )
    
    print(f"Verification Result: {result}")
    
    # Get summary
    summary = verifier.get_verification_summary("test_exec_001")
    print(f"Verification Summary: {summary}")
