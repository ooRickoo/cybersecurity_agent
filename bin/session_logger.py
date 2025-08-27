#!/usr/bin/env python3
"""
Comprehensive Session Logging System
Tracks all agent interactions, LLM calls, tool executions, and outputs.
"""

import os
import sys
import json
import uuid
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import traceback
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class LogLevel(Enum):
    """Log levels for session logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SessionEventType(Enum):
    """Types of session events."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    USER_INPUT = "user_input"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_END = "workflow_end"
    MEMORY_OPERATION = "memory_operation"
    VISUALIZATION_CREATED = "visualization_created"
    FILE_OPERATION = "file_operation"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_EVENT = "system_event"

@dataclass
class SessionEvent:
    """Represents a single session event."""
    event_id: str
    event_type: SessionEventType
    timestamp: str
    session_id: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    
    # Event-specific data
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    memory_usage: Optional[Dict[str, Any]] = None
    
    # Context information
    context: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class SessionLogger:
    """Comprehensive session logging system."""
    
    def __init__(self, session_id: str = None, user_id: str = None, agent_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or "anonymous"
        self.agent_id = agent_id or "cybersecurity_agent"
        
        # Setup logging directories
        self.session_logs_dir = Path("session-logs")
        self.session_outputs_dir = Path("session-outputs")
        self.session_logs_dir.mkdir(exist_ok=True)
        self.session_outputs_dir.mkdir(exist_ok=True)
        
        # Create session-specific directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{timestamp}_{self.session_id[:8]}"
        self.session_log_dir = self.session_logs_dir / self.session_name
        self.session_output_dir = self.session_outputs_dir / self.session_name
        self.session_log_dir.mkdir(exist_ok=True)
        self.session_output_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Session metadata
        self.session_metadata = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'start_time': datetime.now().isoformat(),
            'session_name': self.session_name,
            'platform': sys.platform,
            'python_version': sys.version,
            'working_directory': str(Path.cwd())
        }
        
        # Event tracking
        self.events: List[SessionEvent] = []
        self.current_workflow: Optional[str] = None
        self.current_step: Optional[str] = None
        
        # Performance tracking
        self.start_time = time.time()
        self.llm_calls = 0
        self.tool_executions = 0
        self.errors = 0
        
        # Log session start
        self.log_event(
            SessionEventType.AGENT_START,
            metadata=self.session_metadata,
            tags=['session_start', 'system']
        )
        
        print(f"📝 Session logging initialized: {self.session_name}")
        print(f"   Logs: {self.session_log_dir}")
        print(f"   Outputs: {self.session_output_dir}")
    
    def _setup_logging(self):
        """Setup file-based logging."""
        log_file = self.session_log_dir / "session.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(f"session_{self.session_id[:8]}")
    
    def log_event(self, event_type: SessionEventType, input_data: Dict[str, Any] = None,
                  output_data: Dict[str, Any] = None, metadata: Dict[str, Any] = None,
                  error_info: Dict[str, Any] = None, duration_ms: float = None,
                  context: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """Log a session event."""
        try:
            event = SessionEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now().isoformat(),
                session_id=self.session_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                workflow_id=self.current_workflow,
                step_id=self.current_step,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
                error_info=error_info,
                duration_ms=duration_ms,
                memory_usage=self._get_memory_usage(),
                context=context,
                tags=tags or []
            )
            
            # Add to events list
            self.events.append(event)
            
            # Log to file
            self._write_event_to_file(event)
            
            # Update counters
            if event_type == SessionEventType.LLM_CALL:
                self.llm_calls += 1
            elif event_type == SessionEventType.TOOL_EXECUTION:
                self.tool_executions += 1
            elif event_type == SessionEventType.ERROR_OCCURRED:
                self.errors += 1
            
            return event.event_id
            
        except Exception as e:
            print(f"❌ Error logging event: {e}")
            return ""
    
    def _write_event_to_file(self, event: SessionEvent):
        """Write event to JSON log file."""
        try:
            log_file = self.session_log_dir / "events.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ Error writing event to file: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def log_user_input(self, user_input: str, context: Dict[str, Any] = None):
        """Log user input."""
        return self.log_event(
            SessionEventType.USER_INPUT,
            input_data={'user_input': user_input},
            context=context,
            tags=['user_interaction', 'input']
        )
    
    def log_llm_call(self, prompt: str, model: str = None, parameters: Dict[str, Any] = None,
                     response: str = None, duration_ms: float = None, tokens_used: Dict[str, int] = None):
        """Log LLM call details."""
        input_data = {
            'prompt': prompt,
            'model': model,
            'parameters': parameters or {}
        }
        
        output_data = {
            'response': response,
            'tokens_used': tokens_used or {}
        }
        
        return self.log_event(
            SessionEventType.LLM_CALL,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            tags=['llm', 'ai', 'generation']
        )
    
    def log_tool_execution(self, tool_name: str, tool_input: Dict[str, Any], 
                          tool_output: Dict[str, Any], duration_ms: float = None,
                          success: bool = True, error_message: str = None):
        """Log tool execution details."""
        input_data = {
            'tool_name': tool_name,
            'tool_input': tool_input,
            'success': success
        }
        
        output_data = {
            'tool_output': tool_output,
            'error_message': error_message
        }
        
        error_info = None
        if not success:
            error_info = {'error_message': error_message, 'success': success}
        
        return self.log_event(
            SessionEventType.TOOL_EXECUTION,
            input_data=input_data,
            output_data=output_data,
            error_info=error_info,
            duration_ms=duration_ms,
            tags=['tool_execution', 'automation']
        )
    
    def log_workflow_start(self, workflow_name: str, workflow_config: Dict[str, Any]):
        """Log workflow start."""
        self.current_workflow = workflow_name
        return self.log_event(
            SessionEventType.WORKFLOW_START,
            input_data={'workflow_name': workflow_name, 'workflow_config': workflow_config},
            tags=['workflow', 'start']
        )
    
    def log_workflow_step(self, step_name: str, step_input: Dict[str, Any], 
                         step_output: Dict[str, Any], step_duration_ms: float = None):
        """Log workflow step execution."""
        self.current_step = step_name
        return self.log_event(
            SessionEventType.WORKFLOW_STEP,
            input_data={'step_name': step_name, 'step_input': step_input},
            output_data={'step_output': step_output},
            duration_ms=step_duration_ms,
            tags=['workflow', 'step']
        )
    
    def log_workflow_end(self, workflow_name: str, final_output: Dict[str, Any], 
                        total_duration_ms: float = None, success: bool = True):
        """Log workflow completion."""
        self.current_workflow = None
        self.current_step = None
        
        return self.log_event(
            SessionEventType.WORKFLOW_END,
            input_data={'workflow_name': workflow_name, 'success': success},
            output_data={'final_output': final_output},
            duration_ms=total_duration_ms,
            tags=['workflow', 'end']
        )
    
    def log_memory_operation(self, operation_type: str, operation_data: Dict[str, Any],
                            result: Dict[str, Any], duration_ms: float = None):
        """Log memory management operations."""
        return self.log_event(
            SessionEventType.MEMORY_OPERATION,
            input_data={'operation_type': operation_type, 'operation_data': operation_data},
            output_data={'result': result},
            duration_ms=duration_ms,
            tags=['memory', 'management']
        )
    
    def log_visualization_created(self, viz_type: str, viz_data: Dict[str, Any],
                                 output_path: str, metadata: Dict[str, Any] = None):
        """Log visualization creation."""
        return self.log_event(
            SessionEventType.VISUALIZATION_CREATED,
            input_data={'visualization_type': viz_type, 'visualization_data': viz_data},
            output_data={'output_path': output_path, 'metadata': metadata},
            tags=['visualization', 'output']
        )
    
    def log_file_operation(self, operation: str, file_path: str, file_size: int = None,
                          operation_result: Dict[str, Any] = None):
        """Log file operations."""
        return self.log_event(
            SessionEventType.FILE_OPERATION,
            input_data={'operation': operation, 'file_path': file_path, 'file_size': file_size},
            output_data={'operation_result': operation_result},
            tags=['file_operation', 'io']
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, 
                  error_type: str = "exception", severity: str = "error"):
        """Log error occurrences."""
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'error_class': error.__class__.__name__,
            'traceback': traceback.format_exc(),
            'severity': severity
        }
        
        return self.log_event(
            SessionEventType.ERROR_OCCURRED,
            error_info=error_info,
            context=context,
            tags=['error', 'exception', severity]
        )
    
    def save_output_file(self, filename: str, content: Union[str, bytes, Dict[str, Any]], 
                        content_type: str = "text") -> str:
        """Save output file to session output directory."""
        try:
            output_path = self.session_output_dir / filename
            
            if content_type == "json" and isinstance(content, dict):
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
            elif content_type == "text" and isinstance(content, str):
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            elif content_type == "bytes" and isinstance(content, bytes):
                with open(output_path, 'wb') as f:
                    f.write(content)
            else:
                # Convert to string if needed
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            
            # Log file operation
            self.log_file_operation(
                operation="save_output",
                file_path=str(output_path),
                file_size=len(str(content)),
                operation_result={'success': True, 'path': str(output_path)}
            )
            
            return str(output_path)
            
        except Exception as e:
            self.log_error(e, context={'filename': filename, 'content_type': content_type})
            return ""
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate comprehensive session summary."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        summary = {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'start_time': self.session_metadata['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'total_events': len(self.events),
            'llm_calls': self.llm_calls,
            'tool_executions': self.tool_executions,
            'errors': self.errors,
            'event_breakdown': {},
            'performance_metrics': {
                'avg_event_duration_ms': 0,
                'total_duration_ms': 0
            },
            'output_files': [],
            'session_directory': str(self.session_log_dir)
        }
        
        # Calculate event breakdown
        for event in self.events:
            event_type = event.event_type.value
            summary['event_breakdown'][event_type] = summary['event_breakdown'].get(event_type, 0) + 1
            
            if event.duration_ms:
                summary['performance_metrics']['total_duration_ms'] += event.duration_ms
        
        if self.events:
            summary['performance_metrics']['avg_event_duration_ms'] = (
                summary['performance_metrics']['total_duration_ms'] / len(self.events)
            )
        
        # List output files
        if self.session_output_dir.exists():
            for file_path in self.session_output_dir.iterdir():
                if file_path.is_file():
                    summary['output_files'].append({
                        'filename': file_path.name,
                        'size_bytes': file_path.stat().st_size,
                        'path': str(file_path)
                    })
        
        return summary
    
    def finalize_session(self) -> str:
        """Finalize session and save summary."""
        try:
            # Log session end
            self.log_event(
                SessionEventType.AGENT_END,
                metadata={'total_events': len(self.events)},
                tags=['session_end', 'system']
            )
            
            # Generate and save summary
            summary = self.generate_session_summary()
            summary_path = self.session_log_dir / "session_summary.json"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Save summary to output directory as well
            self.save_output_file("session_summary.json", summary, "json")
            
            print(f"📝 Session finalized: {self.session_name}")
            print(f"   Total events: {len(self.events)}")
            print(f"   LLM calls: {self.llm_calls}")
            print(f"   Tool executions: {self.tool_executions}")
            print(f"   Errors: {self.errors}")
            print(f"   Duration: {summary['total_duration_seconds']:.2f}s")
            print(f"   Summary: {summary_path}")
            
            return str(summary_path)
            
        except Exception as e:
            print(f"❌ Error finalizing session: {e}")
            return ""

# Context manager for automatic session management
class SessionContext:
    """Context manager for automatic session logging."""
    
    def __init__(self, session_id: str = None, user_id: str = None, agent_id: str = None):
        self.session_logger = SessionLogger(session_id, user_id, agent_id)
    
    def __enter__(self):
        return self.session_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session_logger.log_error(exc_val, error_type="context_exit")
        self.session_logger.finalize_session()

# Example usage
if __name__ == "__main__":
    # Test session logging
    with SessionContext(user_id="test_user", agent_id="test_agent") as logger:
        logger.log_user_input("Hello, can you help me with cybersecurity analysis?")
        
        logger.log_llm_call(
            prompt="Analyze this security policy...",
            model="gpt-4",
            response="Based on the analysis...",
            duration_ms=1500
        )
        
        logger.log_tool_execution(
            tool_name="policy_analyzer",
            tool_input={"policy_text": "Sample policy..."},
            tool_output={"risk_score": 0.8, "recommendations": ["..."], "mitre_mappings": []},
            duration_ms=2500
        )
        
        logger.save_output_file("analysis_report.txt", "This is the analysis report...")
        
        print("✅ Session logging test completed!")
