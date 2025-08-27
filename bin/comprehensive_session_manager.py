#!/usr/bin/env python3
"""
Comprehensive Session Manager for ADK Integration

This module provides a unified interface for creating both session-logs and session-outputs
when working with the ADK framework.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import uuid
from typing import Dict, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from session_output_manager import create_session, add_output_file, save_session

class ComprehensiveSessionManager:
    """Manages both session-logs and session-outputs for ADK integration."""
    
    def __init__(self, session_logs_dir="session-logs", session_output_dir="session-output"):
        self.session_logs_dir = Path(session_logs_dir)
        self.session_output_dir = Path(session_output_dir)
        self.session_logs_dir.mkdir(exist_ok=True)
        self.session_output_dir.mkdir(exist_ok=True)
        
    def create_comprehensive_session(self, session_name: str = None, session_metadata: Dict[str, Any] = None) -> str:
        """Create a session with both logging and output capabilities."""
        # Create session in output manager
        session_id = create_session(session_name, session_metadata)
        
        # Create session log entry
        self._create_session_log(session_id, session_name or "adk_session")
        
        return session_id
    
    def _create_session_log(self, session_id: str, session_name: str):
        """Create a session log entry."""
        timestamp = datetime.now()
        log_filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}.json"
        
        log_data = {
            "session_metadata": {
                "session_id": session_id,
                "session_name": session_name,
                "start_time": timestamp.isoformat(),
                "version": "1.0.0",
                "framework": "ADK Cybersecurity Agent",
                "end_time": None,
                "duration_ms": None
            },
            "agent_interactions": [
                {
                    "timestamp": timestamp.isoformat(),
                    "level": "info",
                    "category": "session_creation",
                    "action": "session_start",
                    "details": {
                        "message": f"Session created by ADK Agent: {session_name}",
                        "session_id": session_id,
                        "framework": "ADK Cybersecurity Agent"
                    },
                    "session_id": session_id,
                    "agent_type": "CybersecurityAgent",
                    "workflow_step": "session_initialization",
                    "execution_id": str(uuid.uuid4()),
                    "parent_action": None,
                    "duration_ms": None,
                    "success": True,
                    "error_message": None,
                    "metadata": {
                        "source": "adk_agent_integration",
                        "trigger_system": True
                    }
                }
            ],
            "workflow_executions": [],
            "tool_calls": [],
            "data_operations": [],
            "decision_points": [],
            "errors": [],
            "performance_metrics": {
                "total_tool_calls": 0,
                "total_workflow_steps": 0,
                "total_errors": 0,
                "session_duration_ms": None
            }
        }
        
        log_file_path = self.session_logs_dir / log_filename
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Created session log: {log_file_path}")
        return log_file_path
    
    def add_workflow_execution(self, session_id: str, workflow_name: str, execution_details: Dict[str, Any] = None):
        """Add a workflow execution to the session log."""
        # Find the session log file
        log_files = list(self.session_logs_dir.glob(f"*{session_id[:8]}*.json"))
        if not log_files:
            print(f"⚠️  No session log found for session {session_id}")
            return
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Add workflow execution
            workflow_execution = {
                "timestamp": datetime.now().isoformat(),
                "workflow_name": workflow_name,
                "execution_id": str(uuid.uuid4()),
                "status": "completed",
                "details": execution_details or {},
                "duration_ms": None,
                "success": True,
                "error_message": None
            }
            
            log_data["workflow_executions"].append(workflow_execution)
            
            # Update performance metrics
            log_data["performance_metrics"]["total_workflow_steps"] += 1
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"📊 Added workflow execution to session log: {workflow_name}")
            
        except Exception as e:
            print(f"❌ Error updating session log: {e}")
    
    def add_detailed_workflow_step(self, session_id: str, step_name: str, step_details: Dict[str, Any] = None, 
                                  step_type: str = "workflow_step", parent_workflow: str = None):
        """Add a detailed workflow step with comprehensive logging."""
        # Find the session log file
        log_files = list(self.session_logs_dir.glob(f"*{session_id[:8]}*.json"))
        if not log_files:
            print(f"⚠️  No session log found for session {session_id}")
            return
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Create detailed step entry
            detailed_step = {
                "timestamp": datetime.now().isoformat(),
                "step_name": step_name,
                "step_type": step_type,
                "step_id": str(uuid.uuid4()),
                "parent_workflow": parent_workflow,
                "status": "completed",
                "details": step_details or {},
                "duration_ms": None,
                "success": True,
                "error_message": None,
                "metadata": {
                    "execution_environment": "external_cli",
                    "code_controlled": True,
                    "workflow_tracking": True
                }
            }
            
            # Ensure all values in details are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif hasattr(obj, 'dtype'):  # pandas dtype objects
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj)
            
            detailed_step["details"] = make_json_serializable(detailed_step["details"])
            
            # Add to workflow executions for tracking
            log_data["workflow_executions"].append(detailed_step)
            
            # Update performance metrics
            log_data["performance_metrics"]["total_workflow_steps"] += 1
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"📝 Added detailed workflow step: {step_name}")
            
        except Exception as e:
            print(f"❌ Error adding detailed workflow step: {e}")
    
    def add_workflow_progress(self, session_id: str, progress_step: int, total_steps: int, 
                             step_description: str, step_details: Dict[str, Any] = None):
        """Add workflow progress tracking with step-by-step details."""
        # Find the session log file
        log_files = list(self.session_logs_dir.glob(f"*{session_id[:8]}*.json"))
        if not log_files:
            print(f"⚠️  No session log found for session {session_id}")
            return
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Create progress entry
            progress_entry = {
                "timestamp": datetime.now().isoformat(),
                "progress_step": progress_step,
                "total_steps": total_steps,
                "step_description": step_description,
                "step_id": str(uuid.uuid4()),
                "step_type": "progress_tracking",
                "status": "in_progress",
                "details": step_details or {},
                "completion_percentage": (progress_step / total_steps) * 100,
                "metadata": {
                    "workflow_tracking": True,
                    "progress_monitoring": True
                }
            }
            
            # Ensure all values in details are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif hasattr(obj, 'dtype'):  # pandas dtype objects
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj)
            
            progress_entry["details"] = make_json_serializable(progress_entry["details"])
            
            # Add to workflow executions
            log_data["workflow_executions"].append(progress_entry)
            
            # Update performance metrics
            log_data["performance_metrics"]["total_workflow_steps"] += 1
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"📊 Progress: Step {progress_step}/{total_steps} - {step_description}")
            
        except Exception as e:
            print(f"❌ Error adding workflow progress: {e}")
    
    def finalize_session(self, session_id: str, success: bool = True, error_message: str = None):
        """Finalize a session by updating the log and calculating duration."""
        # Find the session log file
        log_files = list(self.session_logs_dir.glob(f"*{session_id[:8]}*.json"))
        if not log_files:
            print(f"⚠️  No session log found for session {session_id}")
            return
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Calculate duration
            start_time = datetime.fromisoformat(log_data["session_metadata"]["start_time"])
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update session metadata
            log_data["session_metadata"]["end_time"] = end_time.isoformat()
            log_data["session_metadata"]["duration_ms"] = duration_ms
            
            # Update performance metrics
            log_data["performance_metrics"]["session_duration_ms"] = duration_ms
            
            # Add final interaction
            final_interaction = {
                "timestamp": end_time.isoformat(),
                "level": "info" if success else "error",
                "category": "session_completion",
                "action": "session_end",
                "details": {
                    "message": "Session completed successfully" if success else f"Session failed: {error_message}",
                    "session_id": session_id,
                    "duration_ms": duration_ms
                },
                "session_id": session_id,
                "agent_type": "CybersecurityAgent",
                "workflow_step": "session_completion",
                "execution_id": str(uuid.uuid4()),
                "parent_action": None,
                "duration_ms": duration_ms,
                "success": success,
                "error_message": error_message,
                "metadata": {
                    "source": "adk_agent_integration",
                    "finalized": True
                }
            }
            
            log_data["agent_interactions"].append(final_interaction)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"✅ Finalized session log: {log_file}")
            
        except Exception as e:
            print(f"❌ Error finalizing session log: {e}")

# Global comprehensive session manager instance
_comprehensive_session_manager = None

def get_comprehensive_session_manager() -> ComprehensiveSessionManager:
    """Get or create the global comprehensive session manager instance."""
    global _comprehensive_session_manager
    if _comprehensive_session_manager is None:
        _comprehensive_session_manager = ComprehensiveSessionManager()
    return _comprehensive_session_manager

def create_comprehensive_session(session_name: str = None, session_metadata: Dict[str, Any] = None) -> str:
    """Create a comprehensive session with both logging and output."""
    manager = get_comprehensive_session_manager()
    return manager.create_comprehensive_session(session_name, session_metadata)

def add_workflow_execution(session_id: str, workflow_name: str, execution_details: Dict[str, Any] = None):
    """Add workflow execution to session log."""
    manager = get_comprehensive_session_manager()
    manager.add_workflow_execution(session_id, workflow_name, execution_details)

def add_detailed_workflow_step(session_id: str, step_name: str, step_details: Dict[str, Any] = None, 
                              step_type: str = "workflow_step", parent_workflow: str = None):
    """Add a detailed workflow step with comprehensive logging."""
    manager = get_comprehensive_session_manager()
    manager.add_detailed_workflow_step(session_id, step_name, step_details, step_type, parent_workflow)

def add_workflow_progress(session_id: str, progress_step: int, total_steps: int, 
                         step_description: str, step_details: Dict[str, Any] = None):
    """Add workflow progress tracking with step-by-step details."""
    manager = get_comprehensive_session_manager()
    manager.add_workflow_progress(session_id, progress_step, total_steps, step_description, step_details)

def finalize_comprehensive_session(session_id: str, success: bool = True, error_message: str = None):
    """Finalize a comprehensive session."""
    manager = get_comprehensive_session_manager()
    manager.finalize_session(session_id, success, error_message)

if __name__ == "__main__":
    # Test the comprehensive session manager
    print("🧪 Testing Comprehensive Session Manager...")
    
    try:
        # Create a session
        session_id = create_comprehensive_session("test_comprehensive_session")
        print(f"✅ Created comprehensive session: {session_id}")
        
        # Add workflow execution
        add_workflow_execution(session_id, "policy_mapping", {"policies_processed": 5, "mitre_mappings": 7})
        
        # Add output file
        add_output_file(session_id, "test_output.txt", "This is a test output file", "text")
        
        # Save session outputs
        save_result = save_session(session_id)
        print(f"✅ Saved session outputs: {save_result}")
        
        # Finalize session
        finalize_comprehensive_session(session_id, success=True)
        
        print("🎉 Comprehensive session test completed successfully!")
        
    except Exception as e:
        print(f"❌ Comprehensive session test failed: {e}")
        import traceback
        traceback.print_exc()
