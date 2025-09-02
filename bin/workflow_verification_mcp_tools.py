#!/usr/bin/env python3
"""
Workflow Verification MCP Tools for Cybersecurity Agent
Provides NLP-friendly workflow verification and backtracking capabilities
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys
from datetime import datetime

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from workflow_verification_system import get_workflow_verifier, WorkflowStep
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    # Define WorkflowStep locally if not available
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class WorkflowStep:
        step_id: str
        step_type: str
        description: str
        input_data: Dict[str, Any] = None
        output_data: Dict[str, Any] = None
        status: str = "pending"
        error_message: str = None
        execution_time: float = 0.0
    
    # Fallback function for get_workflow_verifier
    def get_workflow_verifier():
        return None

try:
    from workflow_template_manager import get_workflow_template_manager
    TEMPLATE_MANAGER_AVAILABLE = True
except ImportError:
    TEMPLATE_MANAGER_AVAILABLE = False

class WorkflowVerificationMCPTools:
    """MCP tools for workflow verification and backtracking with NLP support."""
    
    def __init__(self):
        self.verifier = get_workflow_verifier() if VERIFICATION_AVAILABLE else None
        self.template_manager = get_workflow_template_manager() if TEMPLATE_MANAGER_AVAILABLE else None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools."""
        if not VERIFICATION_AVAILABLE or not TEMPLATE_MANAGER_AVAILABLE:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_our_math",
                    "description": "Verify workflow accuracy by comparing question, steps taken, and final output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "execution_id": {
                                "type": "string",
                                "description": "Unique identifier for the workflow execution"
                            },
                            "original_question": {
                                "type": "string",
                                "description": "The original question or problem that was asked"
                            },
                            "workflow_steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_id": {"type": "string"},
                                        "step_type": {"type": "string"},
                                        "description": {"type": "string"},
                                        "inputs": {"type": "object"},
                                        "outputs": {"type": "object"},
                                        "tools_used": {"type": "array", "items": {"type": "string"}},
                                        "execution_time": {"type": "number"},
                                        "status": {"type": "string"}
                                    }
                                },
                                "description": "List of workflow steps that were executed"
                            },
                            "final_answer": {
                                "type": "string",
                                "description": "The final answer or result produced by the workflow"
                            },
                            "question_type": {
                                "type": "string",
                                "description": "Type of question for specialized verification (threat_analysis, incident_response, etc.)",
                                "enum": ["threat_analysis", "incident_response", "vulnerability_assessment", "compliance_audit", "general_investigation"]
                            }
                        },
                        "required": ["execution_id", "original_question", "workflow_steps", "final_answer"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handle_verification_failure",
                    "description": "Handle verification failures by selecting alternative workflow templates and approaches",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "execution_id": {
                                "type": "string",
                                "description": "ID of the failed workflow execution"
                            },
                            "verification_result": {
                                "type": "object",
                                "description": "Result from the verification process showing what failed"
                            },
                            "original_question": {
                                "type": "string",
                                "description": "The original question that needs to be answered"
                            }
                        },
                        "required": ["execution_id", "verification_result", "original_question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_verification_summary",
                    "description": "Get comprehensive summary of workflow verification results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "execution_id": {
                                "type": "string",
                                "description": "ID of the workflow execution to summarize"
                            }
                        },
                        "required": ["execution_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_execution_history",
                    "description": "Get recent workflow execution history with verification status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of executions to return",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_workflow_template",
                    "description": "Select the most appropriate workflow template for a given question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question or problem to be solved"
                            },
                            "question_type": {
                                "type": "string",
                                "description": "Optional question type classification"
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_template_variations",
                    "description": "Get variations of a workflow template for different complexity levels",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "template_id": {
                                "type": "string",
                                "description": "ID of the base template to get variations for"
                            }
                        },
                        "required": ["template_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_backtrack_summary",
                    "description": "Get summary of backtracking decisions for a specific execution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "execution_id": {
                                "type": "string",
                                "description": "ID of the execution to get backtrack summary for"
                            }
                        },
                        "required": ["execution_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_template_performance_stats",
                    "description": "Get performance statistics for all workflow templates",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def check_our_math(self, 
                       execution_id: str,
                       original_question: str,
                       workflow_steps: List[Dict[str, Any]],
                       final_answer: str,
                       question_type: str = "general_investigation") -> Dict[str, Any]:
        """Verify workflow accuracy by comparing question, steps taken, and final output."""
        if not VERIFICATION_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow verification system not available",
                "message": "Verification tools not available"
            }
        
        try:
            # Convert workflow steps to WorkflowStep objects
            converted_steps = []
            for step_data in workflow_steps:
                step = WorkflowStep(
                    step_id=step_data.get("step_id", str(uuid.uuid4())),
                    step_type=step_data.get("step_type", "unknown"),
                    description=step_data.get("description", ""),
                    inputs=step_data.get("inputs", {}),
                    outputs=step_data.get("outputs", {}),
                    tools_used=step_data.get("tools_used", []),
                    execution_time=step_data.get("execution_time", 0.0),
                    status=step_data.get("status", "completed")
                )
                converted_steps.append(step)
            
            # Perform verification
            verification_result = self.verifier.verify_workflow(
                execution_id=execution_id,
                original_question=original_question,
                workflow_steps=converted_steps,
                final_answer=final_answer,
                question_type=question_type
            )
            
            # Generate comprehensive CLI output
            cli_output = self._generate_verification_cli_output(
                execution_id, original_question, converted_steps, final_answer, verification_result
            )
            
            return {
                "success": True,
                "execution_id": execution_id,
                "verification_result": {
                    "accuracy_score": verification_result.accuracy_score,
                    "confidence_level": verification_result.confidence_level,
                    "issues_found": verification_result.issues_found,
                    "recommendations": verification_result.recommendations,
                    "needs_backtrack": verification_result.needs_backtrack,
                    "backtrack_reason": verification_result.backtrack_reason,
                    "alternative_approaches": verification_result.alternative_approaches
                },
                "message": f"Verification completed with {verification_result.confidence_level} confidence",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error during verification: {e}",
                "cli_output": f"❌ **Verification Error**\n   **Error:** {e}"
            }
    
    def _generate_verification_cli_output(self, 
                                        execution_id: str,
                                        original_question: str,
                                        workflow_steps: List[WorkflowStep],
                                        final_answer: str,
                                        verification_result) -> str:
        """Generate comprehensive CLI output for verification results."""
        cli_output = f"🔍 **Workflow Verification: Check Our Math**\n"
        cli_output += f"   **Execution ID:** {execution_id}\n"
        cli_output += f"   **Original Question:** {original_question}\n"
        cli_output += f"   **Total Steps:** {len(workflow_steps)}\n"
        cli_output += f"   **Final Answer Length:** {len(final_answer)} characters\n\n"
        
        # Verification Results
        cli_output += f"📊 **Verification Results:**\n"
        cli_output += f"   **Accuracy Score:** {verification_result.accuracy_score:.2f}/1.00\n"
        cli_output += f"   **Confidence Level:** {verification_result.confidence_level.upper()}\n"
        cli_output += f"   **Status:** {'✅ PASSED' if verification_result.accuracy_score >= 0.8 else '⚠️ NEEDS REVIEW' if verification_result.accuracy_score >= 0.6 else '❌ FAILED'}\n\n"
        
        # Workflow Steps Summary
        cli_output += f"🔄 **Workflow Steps Executed:**\n"
        for i, step in enumerate(workflow_steps, 1):
            cli_output += f"   **{i}.** {step.step_type.title()}: {step.description}\n"
            cli_output += f"       Tools: {', '.join(step.tools_used) if step.tools_used else 'None'}\n"
            cli_output += f"       Time: {step.execution_time:.1f}s\n"
            cli_output += f"       Status: {step.status}\n\n"
        
        # Issues and Recommendations
        if verification_result.issues_found:
            cli_output += f"⚠️ **Issues Found:**\n"
            for issue in verification_result.issues_found:
                cli_output += f"   • {issue}\n"
            cli_output += "\n"
        
        if verification_result.recommendations:
            cli_output += f"💡 **Recommendations:**\n"
            for recommendation in verification_result.recommendations:
                cli_output += f"   • {recommendation}\n"
            cli_output += "\n"
        
        # Backtracking Decision
        if verification_result.needs_backtrack:
            cli_output += f"🔄 **Backtracking Required:**\n"
            cli_output += f"   **Reason:** {verification_result.backtrack_reason}\n"
            cli_output += f"   **Action:** Use 'handle_verification_failure' to select alternative approach\n\n"
            
            if verification_result.alternative_approaches:
                cli_output += f"🛤️ **Alternative Approaches:**\n"
                for approach in verification_result.alternative_approaches:
                    cli_output += f"   • {approach}\n"
                cli_output += "\n"
        else:
            cli_output += f"✅ **Verification Passed:**\n"
            cli_output += f"   The workflow successfully answered the question with high accuracy.\n"
            cli_output += f"   No backtracking required.\n\n"
        
        # Next Steps
        cli_output += f"🎯 **Next Steps:**\n"
        if verification_result.needs_backtrack:
            cli_output += f"   • Use 'handle_verification_failure' to select alternative workflow\n"
            cli_output += f"   • Review and refine the approach based on recommendations\n"
            cli_output += f"   • Consider breaking down the problem into smaller components\n"
        else:
            cli_output += f"   • Workflow completed successfully\n"
            cli_output += f"   • Results are ready for use\n"
            cli_output += f"   • Consider documenting the successful approach for future use\n"
        
        return cli_output
    
    def handle_verification_failure(self, 
                                  execution_id: str,
                                  verification_result: Dict[str, Any],
                                  original_question: str) -> Dict[str, Any]:
        """Handle verification failures by selecting alternative workflow templates."""
        if not TEMPLATE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow template manager not available",
                "message": "Template management tools not available"
            }
        
        try:
            # Convert verification result dict to VerificationResult object if needed
            if isinstance(verification_result, dict):
                # Create a mock VerificationResult object for the template manager
                from workflow_verification_system import VerificationResult
                mock_result = VerificationResult(
                    accuracy_score=verification_result.get("accuracy_score", 0.0),
                    confidence_level=verification_result.get("confidence_level", "low"),
                    issues_found=verification_result.get("issues_found", []),
                    recommendations=verification_result.get("recommendations", []),
                    needs_backtrack=verification_result.get("needs_backtrack", True),
                    backtrack_reason=verification_result.get("backtrack_reason", "Verification failed"),
                    alternative_approaches=verification_result.get("alternative_approaches", [])
                )
                verification_result = mock_result
            
            # Handle backtracking
            backtrack_result = self.template_manager.handle_backtracking(
                original_execution_id=execution_id,
                verification_result=verification_result,
                original_question=original_question
            )
            
            # Generate CLI output
            cli_output = self._generate_backtrack_cli_output(
                execution_id, original_question, backtrack_result
            )
            
            return {
                "success": True,
                "execution_id": execution_id,
                "backtrack_result": backtrack_result,
                "message": "Backtracking handled successfully",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error handling backtracking: {e}",
                "cli_output": f"❌ **Backtracking Error**\n   **Error:** {e}"
            }
    
    def _generate_backtrack_cli_output(self, 
                                     execution_id: str,
                                     original_question: str,
                                     backtrack_result: Dict[str, Any]) -> str:
        """Generate CLI output for backtracking results."""
        cli_output = f"🔄 **Workflow Backtracking: Alternative Approach Selection**\n"
        cli_output += f"   **Execution ID:** {execution_id}\n"
        cli_output += f"   **Original Question:** {original_question}\n\n"
        
        if backtrack_result.get("success"):
            cli_output += f"✅ **Alternative Template Selected:**\n"
            cli_output += f"   **Template ID:** {backtrack_result['selected_template']}\n"
            cli_output += f"   **Template Name:** {backtrack_result['template_name']}\n"
            cli_output += f"   **Confidence:** {backtrack_result['confidence']:.2f}\n"
            cli_output += f"   **Reason:** {backtrack_result['reason']}\n\n"
            
            # Recommended Steps
            if backtrack_result.get("recommended_steps"):
                cli_output += f"📋 **Recommended Steps:**\n"
                for i, step in enumerate(backtrack_result["recommended_steps"], 1):
                    cli_output += f"   **{i}.** {step['step_type'].title()}: {step['description']}\n"
                    if step.get("tools"):
                        cli_output += f"       Tools: {', '.join(step['tools'])}\n"
                    cli_output += f"       Expected Duration: {step.get('expected_duration', 'Unknown')}s\n\n"
            
            # Alternative Approaches
            if backtrack_result.get("alternative_approaches"):
                cli_output += f"🛤️ **Alternative Approaches:**\n"
                for approach in backtrack_result["alternative_approaches"]:
                    cli_output += f"   • {approach}\n"
                cli_output += "\n"
            
            cli_output += f"💡 **Next Steps:**\n"
            cli_output += f"   • Execute the new workflow template\n"
            cli_output += f"   • Monitor for improved accuracy\n"
            cli_output += f"   • Document successful alternative approaches\n"
            
        else:
            cli_output += f"❌ **No Suitable Alternative Found:**\n"
            cli_output += f"   **Reason:** {backtrack_result['reason']}\n\n"
            
            # Recommendations
            if backtrack_result.get("recommendations"):
                cli_output += f"💡 **Recommendations:**\n"
                for recommendation in backtrack_result["recommendations"]:
                    cli_output += f"   • {recommendation}\n"
                cli_output += "\n"
            
            cli_output += f"🔄 **Alternative Actions:**\n"
            cli_output += f"   • Break down the problem into smaller components\n"
            cli_output += f"   • Use manual analysis approach\n"
            cli_output += f"   • Consult with domain experts\n"
            cli_output += f"   • Consider using different tools or data sources\n"
        
        return cli_output
    
    def get_verification_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of workflow verification results."""
        if not VERIFICATION_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow verification system not available",
                "message": "Verification tools not available"
            }
        
        try:
            summary = self.verifier.get_verification_summary(execution_id)
            
            if "error" in summary:
                return {
                    "success": False,
                    "error": summary["error"],
                    "message": "Execution not found"
                }
            
            # Generate CLI output
            cli_output = f"📊 **Verification Summary**\n"
            cli_output += f"   **Execution ID:** {summary['execution_id']}\n"
            cli_output += f"   **Question:** {summary['original_question']}\n"
            cli_output += f"   **Status:** {summary['verification_status'].upper()}\n"
            cli_output += f"   **Accuracy Score:** {summary['accuracy_score']:.2f}/1.00\n"
            cli_output += f"   **Confidence Level:** {summary['confidence_level'].upper()}\n"
            cli_output += f"   **Total Steps:** {summary['total_steps']}\n"
            cli_output += f"   **Execution Time:** {summary['total_execution_time']:.1f}s\n"
            cli_output += f"   **Backtrack Count:** {summary['backtrack_count']}\n"
            cli_output += f"   **Created:** {summary['created_at'][:19]}\n"
            cli_output += f"   **Completed:** {summary['completed_at'][:19]}\n\n"
            
            # Issues and Recommendations
            if summary.get('issues_found'):
                cli_output += f"⚠️ **Issues Found:**\n"
                for issue in summary['issues_found']:
                    cli_output += f"   • {issue}\n"
                cli_output += "\n"
            
            if summary.get('recommendations'):
                cli_output += f"💡 **Recommendations:**\n"
                for recommendation in summary['recommendations']:
                    cli_output += f"   • {recommendation}\n"
                cli_output += "\n"
            
            # Execution Path
            if summary.get('execution_path'):
                cli_output += f"🛤️ **Execution Path:**\n"
                cli_output += f"   {' → '.join(summary['execution_path'])}\n\n"
            
            return {
                "success": True,
                "summary": summary,
                "message": "Verification summary retrieved successfully",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting verification summary: {e}"
            }
    
    def get_execution_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent workflow execution history with verification status."""
        if not VERIFICATION_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow verification system not available",
                "message": "Verification tools not available"
            }
        
        try:
            history = self.verifier.get_execution_history(limit)
            
            # Generate CLI output
            cli_output = f"📚 **Workflow Execution History**\n"
            cli_output += f"   **Total Executions:** {len(history)}\n\n"
            
            if history:
                for i, execution in enumerate(history, 1):
                    status_icon = "✅" if execution['status'] == 'passed' else "⚠️" if execution['status'] == 'needs_review' else "❌"
                    cli_output += f"{status_icon} **{i}.** {execution['execution_id'][:8]}...\n"
                    cli_output += f"   **Question:** {execution['question']}\n"
                    cli_output += f"   **Status:** {execution['status'].upper()}\n"
                    cli_output += f"   **Accuracy:** {execution['accuracy']:.2f}\n"
                    cli_output += f"   **Steps:** {execution['steps']}\n"
                    cli_output += f"   **Completed:** {execution['completed_at'][:10]}\n\n"
            else:
                cli_output += f"   **No executions found**\n"
            
            return {
                "success": True,
                "history": history,
                "message": f"Retrieved {len(history)} execution records",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting execution history: {e}"
            }
    
    def select_workflow_template(self, question: str, question_type: str = None) -> Dict[str, Any]:
        """Select the most appropriate workflow template for a given question."""
        if not TEMPLATE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow template manager not available",
                "message": "Template management tools not available"
            }
        
        try:
            template = self.template_manager.select_template_for_question(question, question_type)
            
            if not template:
                return {
                    "success": False,
                    "error": "No suitable template found",
                    "message": "Could not find appropriate workflow template"
                }
            
            # Generate CLI output
            cli_output = f"📋 **Workflow Template Selected**\n"
            cli_output += f"   **Template ID:** {template.template_id}\n"
            cli_output += f"   **Name:** {template.name}\n"
            cli_output += f"   **Type:** {template.template_type.value.replace('_', ' ').title()}\n"
            cli_output += f"   **Description:** {template.description}\n"
            cli_output += f"   **Success Rate:** {template.success_rate:.1%}\n"
            cli_output += f"   **Usage Count:** {template.usage_count}\n"
            cli_output += f"   **Total Steps:** {len(template.steps)}\n\n"
            
            # Prerequisites
            if template.prerequisites:
                cli_output += f"🔑 **Prerequisites:**\n"
                for prereq in template.prerequisites:
                    cli_output += f"   • {prereq}\n"
                cli_output += "\n"
            
            # Expected Outcomes
            if template.expected_outcomes:
                cli_output += f"🎯 **Expected Outcomes:**\n"
                for outcome in template.expected_outcomes:
                    cli_output += f"   • {outcome}\n"
                cli_output += "\n"
            
            # Workflow Steps
            cli_output += f"🔄 **Workflow Steps:**\n"
            for i, step in enumerate(template.steps, 1):
                cli_output += f"   **{i}.** {step['step_type'].title()}: {step['description']}\n"
                if step.get("tools"):
                    cli_output += f"       Tools: {', '.join(step['tools'])}\n"
                cli_output += f"       Duration: {step.get('expected_duration', 'Unknown')}s\n"
                cli_output += "\n"
            
            # Alternative Templates
            if template.alternative_templates:
                cli_output += f"🔄 **Alternative Templates:**\n"
                for alt in template.alternative_templates:
                    cli_output += f"   • {alt}\n"
                cli_output += "\n"
            
            return {
                "success": True,
                "template": template.to_dict(),
                "message": f"Selected template: {template.name}",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error selecting template: {e}"
            }
    
    def get_template_variations(self, template_id: str) -> Dict[str, Any]:
        """Get variations of a workflow template for different complexity levels."""
        if not TEMPLATE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow template manager not available",
                "message": "Template management tools not available"
            }
        
        try:
            variations = self.template_manager.get_template_variations(template_id)
            
            if not variations:
                return {
                    "success": False,
                    "error": "No variations found",
                    "message": f"No variations available for template {template_id}"
                }
            
            # Generate CLI output
            cli_output = f"🔄 **Template Variations**\n"
            cli_output += f"   **Base Template:** {template_id}\n"
            cli_output += f"   **Variations Found:** {len(variations)}\n\n"
            
            for i, variation in enumerate(variations, 1):
                cli_output += f"**{i}. {variation.name}**\n"
                cli_output += f"   **ID:** {variation.template_id}\n"
                cli_output += f"   **Description:** {variation.description}\n"
                cli_output += f"   **Steps:** {len(variation.steps)}\n"
                cli_output += f"   **Success Rate:** {variation.success_rate:.1%}\n"
                cli_output += f"   **Prerequisites:** {len(variation.prerequisites)}\n\n"
            
            return {
                "success": True,
                "variations": [v.to_dict() for v in variations],
                "message": f"Found {len(variations)} template variations",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting template variations: {e}"
            }
    
    def get_backtrack_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get summary of backtracking decisions for a specific execution."""
        if not TEMPLATE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow template manager not available",
                "message": "Template management tools not available"
            }
        
        try:
            summary = self.template_manager.get_backtrack_summary(execution_id)
            
            # Generate CLI output
            cli_output = f"🔄 **Backtracking Summary**\n"
            cli_output += f"   **Execution ID:** {execution_id}\n"
            cli_output += f"   **Backtrack Count:** {summary['backtrack_count']}\n\n"
            
            if summary['decisions']:
                cli_output += f"📋 **Backtrack Decisions:**\n"
                for i, decision in enumerate(summary['decisions'], 1):
                    cli_output += f"   **{i}.** Decision ID: {decision['decision_id'][:8]}...\n"
                    cli_output += f"       Reason: {decision['reason']}\n"
                    cli_output += f"       Template: {decision['selected_template']}\n"
                    cli_output += f"       Confidence: {decision['confidence']:.2f}\n"
                    cli_output += f"       Created: {decision['created_at'][:19]}\n\n"
            else:
                cli_output += f"   **No backtrack decisions found**\n"
            
            return {
                "success": True,
                "summary": summary,
                "message": f"Backtrack summary retrieved: {summary['backtrack_count']} decisions",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting backtrack summary: {e}"
            }
    
    def get_template_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all workflow templates."""
        if not TEMPLATE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Workflow template manager not available",
                "message": "Template management tools not available"
            }
        
        try:
            stats = self.template_manager.get_template_performance_stats()
            
            # Generate CLI output
            cli_output = f"📊 **Template Performance Statistics**\n\n"
            
            for template_type, type_stats in stats.items():
                cli_output += f"**{template_type.replace('_', ' ').title()}:**\n"
                cli_output += f"   **Total Templates:** {type_stats['total_templates']}\n"
                cli_output += f"   **Total Usage:** {type_stats['total_usage']}\n"
                cli_output += f"   **Average Success Rate:** {type_stats['avg_success_rate']:.1%}\n"
                cli_output += f"   **Best Template:** {type_stats['best_template']}\n"
                cli_output += f"   **Best Success Rate:** {type_stats['best_success_rate']:.1%}\n\n"
            
            return {
                "success": True,
                "stats": stats,
                "message": "Template performance statistics retrieved successfully",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting template performance stats: {e}"
            }

# Global instance
_workflow_verification_mcp_tools = None

def get_workflow_verification_mcp_tools() -> WorkflowVerificationMCPTools:
    """Get or create the global workflow verification MCP tools instance."""
    global _workflow_verification_mcp_tools
    if _workflow_verification_mcp_tools is None:
        _workflow_verification_mcp_tools = WorkflowVerificationMCPTools()
    return _workflow_verification_mcp_tools

if __name__ == "__main__":
    # Test the MCP tools
    tools = get_workflow_verification_mcp_tools()
    
    print("🧪 Testing Workflow Verification MCP Tools...")
    
    # Test getting tools
    available_tools = tools.get_tools()
    print(f"Available tools: {len(available_tools)}")
    for tool in available_tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    
    # Test template selection
    result = tools.select_workflow_template("What is the current threat landscape?")
    print(f"Template selection: {result}")
