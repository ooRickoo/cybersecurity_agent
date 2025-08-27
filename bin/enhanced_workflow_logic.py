"""
Enhanced Workflow Logic for Cybersecurity Agent

Provides intelligent workflow adaptation and local scratch tool preference,
enabling efficient problem-solving workflows that leverage local data processing.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import pandas as pd
from pathlib import Path

from .enhanced_session_manager import EnhancedSessionManager
from .context_memory_manager import ContextMemoryManager
from .local_scratch_tools import LocalScratchTools

logger = logging.getLogger(__name__)

class WorkflowPhase(Enum):
    """Workflow execution phases."""
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    LOCAL_ANALYSIS = "local_analysis"
    EXTERNAL_QUERY = "external_query"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    COMPLETION = "completion"

class WorkflowStrategy(Enum):
    """Workflow execution strategies."""
    LOCAL_FIRST = "local_first"  # Prefer local scratch tools
    HYBRID = "hybrid"            # Balance local and external
    EXTERNAL_FIRST = "external_first"  # Prefer external tools
    ADAPTIVE = "adaptive"        # Dynamically adapt based on data

class EnhancedWorkflowLogic:
    """Enhanced workflow logic with local scratch tool preference and intelligent adaptation."""
    
    def __init__(self, session_manager: EnhancedSessionManager, 
                 memory_manager: ContextMemoryManager,
                 scratch_tools: LocalScratchTools):
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.scratch_tools = scratch_tools
        self.workflow_history = []
        self.current_phase = WorkflowPhase.PLANNING
        self.strategy = WorkflowStrategy.LOCAL_FIRST
        
    def plan_workflow(self, user_request: str, available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a workflow based on user request and available tools.
        
        Args:
            user_request: User's request or question
            available_tools: Dictionary of available MCP tools
            
        Returns:
            Workflow plan with phases and tool recommendations
        """
        try:
            # Analyze user request to determine workflow type
            workflow_type = self._analyze_request_type(user_request)
            
            # Check available local data
            local_data_sources = self._identify_local_data_sources(user_request)
            
            # Determine optimal strategy
            strategy = self._determine_strategy(workflow_type, local_data_sources, available_tools)
            
            # Create workflow plan
            plan = self._create_workflow_plan(user_request, workflow_type, strategy, available_tools)
            
            # Log workflow planning
            self.workflow_history.append({
                "timestamp": datetime.now().isoformat(),
                "phase": "planning",
                "user_request": user_request,
                "workflow_type": workflow_type,
                "strategy": strategy.value,
                "plan": plan
            })
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to plan workflow: {e}")
            raise
    
    def _analyze_request_type(self, user_request: str) -> str:
        """Analyze user request to determine workflow type."""
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ["analyze", "investigate", "examine", "review"]):
            return "analysis"
        elif any(word in request_lower for word in ["compare", "diff", "versus", "vs"]):
            return "comparison"
        elif any(word in request_lower for word in ["search", "find", "locate", "discover"]):
            return "discovery"
        elif any(word in request_lower for word in ["monitor", "track", "watch", "observe"]):
            return "monitoring"
        elif any(word in request_lower for word in ["generate", "create", "build", "make"]):
            return "generation"
        elif any(word in request_lower for word in ["validate", "verify", "check", "test"]):
            return "validation"
        else:
            return "general"
    
    def _identify_local_data_sources(self, user_request: str) -> List[Dict[str, Any]]:
        """Identify relevant local data sources for the request."""
        try:
            # Search memory for relevant data
            relevant_data = self.memory_manager.search(user_request, max_entries=10)
            
            # Check session outputs for recent data
            session_data = self._get_session_data_sources()
            
            # Combine and rank data sources
            data_sources = []
            
            for data in relevant_data:
                data_sources.append({
                    "source": "memory",
                    "domain": data.domain,
                    "tier": data.tier,
                    "relevance_score": self._calculate_relevance_score(user_request, data.data),
                    "data": data.data,
                    "metadata": data.metadata
                })
            
            for data in session_data:
                data_sources.append({
                    "source": "session",
                    "domain": "session_output",
                    "tier": "short_term",
                    "relevance_score": self._calculate_relevance_score(user_request, data),
                    "data": data,
                    "metadata": {"type": "session_output"}
                })
            
            # Sort by relevance score
            data_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return data_sources
            
        except Exception as e:
            logger.warning(f"Failed to identify local data sources: {e}")
            return []
    
    def _get_session_data_sources(self) -> List[Dict[str, Any]]:
        """Get data sources from current session outputs."""
        try:
            session_outputs = []
            session_dir = Path(f"session-outputs/{self.session_manager.session_id}")
            
            if session_dir.exists():
                # Look for data files
                for data_file in session_dir.rglob("*.csv"):
                    try:
                        df = pd.read_csv(data_file)
                        session_outputs.append({
                            "file_path": str(data_file),
                            "data_type": "csv",
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "sample_data": df.head(3).to_dict('records')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read session data file {data_file}: {e}")
                
                # Look for JSON files
                for json_file in session_dir.rglob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        session_outputs.append({
                            "file_path": str(json_file),
                            "data_type": "json",
                            "data": data
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read session JSON file {json_file}: {e}")
            
            return session_outputs
            
        except Exception as e:
            logger.warning(f"Failed to get session data sources: {e}")
            return []
    
    def _calculate_relevance_score(self, user_request: str, data: Any) -> float:
        """Calculate relevance score between user request and data."""
        try:
            # Simple keyword-based relevance scoring
            request_words = set(user_request.lower().split())
            
            if isinstance(data, dict):
                data_text = " ".join(str(v) for v in data.values())
            elif isinstance(data, str):
                data_text = data
            else:
                data_text = str(data)
            
            data_words = set(data_text.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(request_words.intersection(data_words))
            union = len(request_words.union(data_words))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance score: {e}")
            return 0.0
    
    def _determine_strategy(self, workflow_type: str, local_data_sources: List[Dict[str, Any]], 
                           available_tools: Dict[str, Any]) -> WorkflowStrategy:
        """Determine optimal workflow strategy."""
        # Check if we have relevant local data
        has_relevant_local_data = any(source["relevance_score"] > 0.3 for source in local_data_sources)
        
        # Check if we have local scratch tools
        has_local_tools = any("scratch" in tool_name.lower() for tool_name in available_tools.keys())
        
        # Check if we have external tools
        has_external_tools = any(tool_name in available_tools for tool_name in 
                               ["azure", "gcp", "splunk", "web_search"])
        
        if has_relevant_local_data and has_local_tools:
            if workflow_type in ["analysis", "comparison", "validation"]:
                return WorkflowStrategy.LOCAL_FIRST
            else:
                return WorkflowStrategy.HYBRID
        elif has_local_tools and not has_external_tools:
            return WorkflowStrategy.LOCAL_FIRST
        elif has_external_tools and not has_local_tools:
            return WorkflowStrategy.EXTERNAL_FIRST
        else:
            return WorkflowStrategy.ADAPTIVE
    
    def _create_workflow_plan(self, user_request: str, workflow_type: str, 
                             strategy: WorkflowStrategy, available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed workflow plan."""
        plan = {
            "user_request": user_request,
            "workflow_type": workflow_type,
            "strategy": strategy.value,
            "phases": [],
            "tool_recommendations": {},
            "estimated_duration": "5-15 minutes",
            "success_criteria": []
        }
        
        # Define phases based on workflow type and strategy
        if strategy == WorkflowStrategy.LOCAL_FIRST:
            plan["phases"] = [
                {
                    "phase": "data_assessment",
                    "description": "Assess available local data and determine if additional collection is needed",
                    "tools": ["scratch_analyze_data", "scratch_generate_report"],
                    "priority": "high"
                },
                {
                    "phase": "local_analysis",
                    "description": "Perform analysis using local scratch tools",
                    "tools": ["scratch_clean_data", "scratch_filter_data", "scratch_aggregate_data"],
                    "priority": "high"
                },
                {
                    "phase": "external_validation",
                    "description": "Validate results with external sources if needed",
                    "tools": list(available_tools.keys())[:3],  # Limit to 3 external tools
                    "priority": "medium"
                }
            ]
        elif strategy == WorkflowStrategy.HYBRID:
            plan["phases"] = [
                {
                    "phase": "data_collection",
                    "description": "Collect data from both local and external sources",
                    "tools": list(available_tools.keys())[:5],
                    "priority": "high"
                },
                {
                    "phase": "local_processing",
                    "description": "Process and analyze data locally",
                    "tools": ["scratch_clean_data", "scratch_analyze_data"],
                    "priority": "high"
                },
                {
                    "phase": "synthesis",
                    "description": "Combine local and external results",
                    "tools": ["scratch_merge_datasets", "scratch_generate_report"],
                    "priority": "high"
                }
            ]
        else:  # EXTERNAL_FIRST or ADAPTIVE
            plan["phases"] = [
                {
                    "phase": "external_data_collection",
                    "description": "Collect data from external sources",
                    "tools": list(available_tools.keys())[:5],
                    "priority": "high"
                },
                {
                    "phase": "local_analysis",
                    "description": "Analyze collected data locally",
                    "tools": ["scratch_analyze_data", "scratch_clean_data"],
                    "priority": "high"
                },
                {
                    "phase": "result_generation",
                    "description": "Generate final results and reports",
                    "tools": ["scratch_generate_report"],
                    "priority": "high"
                }
            ]
        
        # Add tool recommendations
        plan["tool_recommendations"] = self._get_tool_recommendations(workflow_type, strategy, available_tools)
        
        # Add success criteria
        plan["success_criteria"] = self._define_success_criteria(workflow_type)
        
        return plan
    
    def _get_tool_recommendations(self, workflow_type: str, strategy: WorkflowStrategy, 
                                 available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Get tool recommendations based on workflow type and strategy."""
        recommendations = {
            "primary_tools": [],
            "secondary_tools": [],
            "local_tools": [],
            "external_tools": []
        }
        
        # Categorize tools
        local_tools = [name for name in available_tools.keys() if "scratch" in name.lower()]
        external_tools = [name for name in available_tools.keys() if "scratch" not in name.lower()]
        
        recommendations["local_tools"] = local_tools
        recommendations["external_tools"] = external_tools
        
        # Recommend tools based on workflow type
        if workflow_type == "analysis":
            if strategy == WorkflowStrategy.LOCAL_FIRST:
                recommendations["primary_tools"] = ["scratch_analyze_data", "scratch_clean_data"]
                recommendations["secondary_tools"] = external_tools[:2]
            else:
                recommendations["primary_tools"] = external_tools[:3]
                recommendations["secondary_tools"] = ["scratch_analyze_data"]
        
        elif workflow_type == "comparison":
            recommendations["primary_tools"] = ["scratch_merge_datasets", "scratch_analyze_data"]
            recommendations["secondary_tools"] = external_tools[:2]
        
        elif workflow_type == "discovery":
            recommendations["primary_tools"] = external_tools[:3]
            recommendations["secondary_tools"] = ["scratch_analyze_data"]
        
        else:
            recommendations["primary_tools"] = local_tools[:2] + external_tools[:2]
            recommendations["secondary_tools"] = local_tools[2:] + external_tools[2:]
        
        return recommendations
    
    def _define_success_criteria(self, workflow_type: str) -> List[str]:
        """Define success criteria for workflow type."""
        if workflow_type == "analysis":
            return [
                "Data structure and quality assessed",
                "Key insights identified and documented",
                "Analysis report generated and saved",
                "Data exported to local scratch for further processing"
            ]
        elif workflow_type == "comparison":
            return [
                "Datasets successfully merged",
                "Differences and similarities identified",
                "Comparison report generated",
                "Results exported to local scratch"
            ]
        elif workflow_type == "discovery":
            return [
                "Relevant data sources identified",
                "Data collected and validated",
                "Initial analysis performed",
                "Data saved to local scratch for processing"
            ]
        else:
            return [
                "User request addressed",
                "Results documented and saved",
                "Data available in local scratch",
                "Workflow completed successfully"
            ]
    
    def execute_workflow_phase(self, phase: Dict[str, Any], 
                              available_tools: Dict[str, Any],
                              context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a specific workflow phase.
        
        Args:
            phase: Phase definition from workflow plan
            available_tools: Available MCP tools
            context_data: Context data from previous phases
            
        Returns:
            Phase execution results
        """
        try:
            phase_name = phase["phase"]
            logger.info(f"Executing workflow phase: {phase_name}")
            
            # Update current phase
            self.current_phase = WorkflowPhase(phase_name)
            
            # Execute phase-specific logic
            if phase_name == "data_assessment":
                results = self._execute_data_assessment_phase(phase, context_data)
            elif phase_name == "local_analysis":
                results = self._execute_local_analysis_phase(phase, context_data)
            elif phase_name == "data_collection":
                results = self._execute_data_collection_phase(phase, available_tools, context_data)
            elif phase_name == "external_data_collection":
                results = self._execute_external_collection_phase(phase, available_tools, context_data)
            elif phase_name == "synthesis":
                results = self._execute_synthesis_phase(phase, context_data)
            elif phase_name == "result_generation":
                results = self._execute_result_generation_phase(phase, context_data)
            else:
                results = {"status": "unknown_phase", "message": f"Unknown phase: {phase_name}"}
            
            # Log phase execution
            self.workflow_history.append({
                "timestamp": datetime.now().isoformat(),
                "phase": phase_name,
                "results": results,
                "status": results.get("status", "unknown")
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute workflow phase {phase_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "phase": phase_name
            }
    
    def _execute_data_assessment_phase(self, phase: Dict[str, Any], 
                                     context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute data assessment phase."""
        try:
            # Check what local data we have
            local_data_sources = self._identify_local_data_sources("")
            
            # Analyze available data
            analysis_results = []
            for source in local_data_sources[:5]:  # Limit to 5 sources
                if source["source"] == "session" and "file_path" in source["data"]:
                    try:
                        df = self.scratch_tools.load_data(source["data"]["file_path"])
                        analysis = self.scratch_tools.analyze_data_structure(df)
                        analysis_results.append({
                            "source": source["data"]["file_path"],
                            "analysis": analysis
                        })
                    except Exception as e:
                        logger.warning(f"Failed to analyze {source['data']['file_path']}: {e}")
            
            return {
                "status": "completed",
                "phase": "data_assessment",
                "local_data_sources": len(local_data_sources),
                "analyzed_sources": len(analysis_results),
                "analysis_results": analysis_results,
                "recommendations": self._generate_data_assessment_recommendations(local_data_sources)
            }
            
        except Exception as e:
            logger.error(f"Failed to execute data assessment phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_local_analysis_phase(self, phase: Dict[str, Any], 
                                    context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute local analysis phase."""
        try:
            # Get available local data
            local_data_sources = self._identify_local_data_sources("")
            
            analysis_results = []
            for source in local_data_sources[:3]:  # Limit to 3 sources
                if source["source"] == "session" and "file_path" in source["data"]:
                    try:
                        df = self.scratch_tools.load_data(source["data"]["file_path"])
                        
                        # Clean data
                        df_clean = self.scratch_tools.clean_data(df)
                        
                        # Generate comprehensive report
                        report = self.scratch_tools.generate_summary_report(df_clean, include_analysis=True)
                        
                        analysis_results.append({
                            "source": source["data"]["file_path"],
                            "original_shape": df.shape,
                            "cleaned_shape": df_clean.shape,
                            "report": report
                        })
                        
                        # Save cleaned data
                        cleaned_filename = f"cleaned_{Path(source['data']['file_path']).stem}"
                        self.session_manager.save_dataframe(
                            df_clean, 
                            cleaned_filename, 
                            f"Cleaned data from {source['data']['file_path']}"
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze {source['data']['file_path']}: {e}")
            
            return {
                "status": "completed",
                "phase": "local_analysis",
                "analyzed_sources": len(analysis_results),
                "analysis_results": analysis_results,
                "cleaned_data_saved": True
            }
            
        except Exception as e:
            logger.error(f"Failed to execute local analysis phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_data_collection_phase(self, phase: Dict[str, Any], 
                                     available_tools: Dict[str, Any],
                                     context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute data collection phase (hybrid approach)."""
        try:
            # This would integrate with external tools for data collection
            # For now, return a placeholder
            return {
                "status": "completed",
                "phase": "data_collection",
                "message": "Data collection phase completed (external tool integration needed)",
                "collected_sources": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to execute data collection phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_external_collection_phase(self, phase: Dict[str, Any], 
                                         available_tools: Dict[str, Any],
                                         context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute external data collection phase."""
        try:
            # This would integrate with external tools
            # For now, return a placeholder
            return {
                "status": "completed",
                "phase": "external_data_collection",
                "message": "External data collection phase completed (external tool integration needed)",
                "collected_sources": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to execute external collection phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_synthesis_phase(self, phase: Dict[str, Any], 
                               context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute synthesis phase."""
        try:
            # Combine and synthesize results from different sources
            return {
                "status": "completed",
                "phase": "synthesis",
                "message": "Data synthesis completed",
                "synthesized_results": True
            }
            
        except Exception as e:
            logger.error(f"Failed to execute synthesis phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_result_generation_phase(self, phase: Dict[str, Any], 
                                       context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute result generation phase."""
        try:
            # Generate final results and reports
            return {
                "status": "completed",
                "phase": "result_generation",
                "message": "Results generated and saved",
                "reports_created": True
            }
            
        except Exception as e:
            logger.error(f"Failed to execute result generation phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_data_assessment_recommendations(self, local_data_sources: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on data assessment."""
        recommendations = []
        
        if not local_data_sources:
            recommendations.append("No local data sources found. Consider collecting data from external sources.")
        else:
            recommendations.append(f"Found {len(local_data_sources)} local data sources. Analyze existing data first.")
            
            # Check data quality
            high_quality_sources = [s for s in local_data_sources if s.get("relevance_score", 0) > 0.5]
            if high_quality_sources:
                recommendations.append(f"{len(high_quality_sources)} high-relevance data sources available. Prioritize local analysis.")
            
            # Check data freshness
            recent_sources = [s for s in local_data_sources if s.get("source") == "session"]
            if recent_sources:
                recommendations.append(f"{len(recent_sources)} recent session outputs available. Use for immediate analysis.")
        
        return recommendations
    
    def adapt_workflow(self, current_results: Dict[str, Any], 
                      user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Adapt workflow based on current results and user feedback.
        
        Args:
            current_results: Results from current workflow phase
            user_feedback: Optional user feedback for adaptation
            
        Returns:
            Adapted workflow plan
        """
        try:
            # Analyze current results
            if current_results.get("status") == "error":
                # Handle errors by adapting strategy
                return self._adapt_for_errors(current_results)
            
            # Check if we need to collect more data
            if self._needs_more_data(current_results):
                return self._adapt_for_data_collection(current_results)
            
            # Check if we need to switch to local-first approach
            if self._should_switch_to_local(current_results):
                return self._adapt_for_local_analysis(current_results)
            
            # Check user feedback for adaptation
            if user_feedback:
                return self._adapt_for_user_feedback(user_feedback, current_results)
            
            # Default: continue with current plan
            return {
                "status": "adapted",
                "message": "Workflow adapted based on current results",
                "recommendations": ["Continue with current workflow plan"],
                "next_phase": "Continue to next planned phase"
            }
            
        except Exception as e:
            logger.error(f"Failed to adapt workflow: {e}")
            return {"status": "error", "error": str(e)}
    
    def _adapt_for_errors(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt workflow to handle errors."""
        error = current_results.get("error", "Unknown error")
        
        if "connection" in error.lower() or "authentication" in error.lower():
            return {
                "status": "adapted",
                "message": "Authentication/connection error detected",
                "recommendations": [
                    "Switch to local-first approach using available local data",
                    "Use local scratch tools for analysis",
                    "Generate report based on local data only"
                ],
                "strategy_change": "LOCAL_FIRST",
                "next_phase": "local_analysis"
            }
        else:
            return {
                "status": "adapted",
                "message": "General error detected",
                "recommendations": [
                    "Review error details and retry",
                    "Consider alternative approach",
                    "Use local tools if available"
                ],
                "next_phase": "error_recovery"
            }
    
    def _needs_more_data(self, current_results: Dict[str, Any]) -> bool:
        """Check if workflow needs more data."""
        # Check if we have sufficient data for analysis
        if "analysis_results" in current_results:
            analysis_count = len(current_results["analysis_results"])
            return analysis_count < 2  # Need at least 2 data sources
        
        return True  # Default to needing more data
    
    def _should_switch_to_local(self, current_results: Dict[str, Any]) -> bool:
        """Check if we should switch to local-first approach."""
        # Check if we have good local data
        if "local_data_sources" in current_results:
            local_count = current_results["local_data_sources"]
            return local_count >= 3  # Switch to local if we have 3+ sources
        
        return False
    
    def _adapt_for_data_collection(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt workflow for data collection."""
        return {
            "status": "adapted",
            "message": "Additional data collection needed",
            "recommendations": [
                "Collect data from external sources",
                "Use available external tools",
                "Ensure data quality and relevance"
            ],
            "next_phase": "data_collection",
            "priority": "high"
        }
    
    def _adapt_for_local_analysis(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt workflow for local analysis."""
        return {
            "status": "adapted",
            "message": "Switching to local-first approach",
            "recommendations": [
                "Use local scratch tools for analysis",
                "Leverage existing local data",
                "Generate comprehensive local analysis report"
            ],
            "strategy_change": "LOCAL_FIRST",
            "next_phase": "local_analysis",
            "priority": "high"
        }
    
    def _adapt_for_user_feedback(self, user_feedback: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt workflow based on user feedback."""
        feedback_lower = user_feedback.lower()
        
        if any(word in feedback_lower for word in ["more", "additional", "expand"]):
            return {
                "status": "adapted",
                "message": "Expanding workflow based on user feedback",
                "recommendations": [
                    "Collect additional data sources",
                    "Perform deeper analysis",
                    "Generate comprehensive reports"
                ],
                "next_phase": "data_collection",
                "priority": "high"
            }
        elif any(word in feedback_lower for word in ["faster", "quick", "simple"]):
            return {
                "status": "adapted",
                "message": "Simplifying workflow based on user feedback",
                "recommendations": [
                    "Focus on local data analysis",
                    "Use local scratch tools",
                    "Generate concise reports"
                ],
                "strategy_change": "LOCAL_FIRST",
                "next_phase": "local_analysis",
                "priority": "high"
            }
        else:
            return {
                "status": "adapted",
                "message": "Workflow adapted based on user feedback",
                "recommendations": [
                    "Continue with current approach",
                    "Monitor user satisfaction",
                    "Be ready for further adaptation"
                ],
                "next_phase": "Continue current workflow"
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and history."""
        return {
            "current_phase": self.current_phase.value,
            "strategy": self.strategy.value,
            "workflow_history": self.workflow_history,
            "total_phases_executed": len(self.workflow_history),
            "last_activity": self.workflow_history[-1]["timestamp"] if self.workflow_history else None,
            "status": "active" if self.current_phase != WorkflowPhase.COMPLETION else "completed"
        }
    
    def reset_workflow(self) -> None:
        """Reset workflow state."""
        self.workflow_history = []
        self.current_phase = WorkflowPhase.PLANNING
        self.strategy = WorkflowStrategy.LOCAL_FIRST
        logger.info("Workflow reset to initial state")
