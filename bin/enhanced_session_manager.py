#!/usr/bin/env python3
"""
Enhanced Session Manager
Comprehensive session logging, output organization, and activity tracking.
"""

import os
import sys
import json
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class EnhancedSessionManager:
    """Enhanced session management with comprehensive logging and output organization."""
    
    def __init__(self, session_id: str = None, enable_logging: bool = True):
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.enable_logging = enable_logging
        
        # Create session directories
        self.session_logs_dir = Path("session-logs")
        self.session_outputs_dir = Path("session-outputs")
        self.session_logs_dir.mkdir(exist_ok=True)
        self.session_outputs_dir.mkdir(exist_ok=True)
        
        # Create unique session folder
        self.session_folder = self.session_outputs_dir / f"{self.session_id}"
        self.session_folder.mkdir(exist_ok=True)
        
        # Initialize logging
        if self.enable_logging:
            self._setup_logging()
        
        # Session tracking
        self.workflow_steps = []
        self.llm_calls = []
        self.tool_executions = []
        self.outputs_generated = []
        self.errors_encountered = []
        self.performance_metrics = {}
        
        # Start session log
        self._log_session_start()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"
    
    def _setup_logging(self):
        """Setup comprehensive logging for the session."""
        # Create session log file
        log_file = self.session_logs_dir / f"{self.session_id}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(f"session_{self.session_id}")
        self.logger.info(f"Session {self.session_id} started at {self.start_time}")
    
    def _log_session_start(self):
        """Log session start information."""
        session_info = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'session_logs_dir': str(self.session_logs_dir),
            'session_outputs_dir': str(self.session_outputs_dir),
            'session_folder': str(self.session_folder)
        }
        
        if self.enable_logging:
            self.logger.info(f"Session initialized: {json.dumps(session_info, indent=2)}")
        
        # Save session info
        self._save_session_info(session_info)
    
    def _save_session_info(self, session_info: Dict[str, Any]):
        """Save session information to session folder."""
        info_file = self.session_folder / "session_info.json"
        with open(info_file, 'w') as f:
            json.dump(session_info, f, indent=2, default=str)
    
    def log_workflow_step(self, step_name: str, step_data: Dict[str, Any], 
                         inputs: Any = None, outputs: Any = None, 
                         duration: float = None, status: str = "completed"):
        """Log a workflow step with detailed information."""
        step_info = {
            'timestamp': datetime.now().isoformat(),
            'step_name': step_name,
            'step_data': step_data,
            'inputs': inputs,
            'outputs': outputs,
            'duration': duration,
            'status': status
        }
        
        self.workflow_steps.append(step_info)
        
        if self.enable_logging:
            self.logger.info(f"Workflow step: {step_name} - {status}")
            if duration:
                self.logger.info(f"  Duration: {duration:.2f}s")
            if inputs:
                self.logger.debug(f"  Inputs: {inputs}")
            if outputs:
                self.logger.debug(f"  Outputs: {outputs}")
        
        # Save step details
        self._save_workflow_step(step_info)
    
    def log_llm_call(self, model: str, prompt: str, response: str, 
                     tokens_used: int = None, duration: float = None,
                     metadata: Dict[str, Any] = None):
        """Log LLM call details."""
        llm_info = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt': prompt,
            'response': response,
            'tokens_used': tokens_used,
            'duration': duration,
            'metadata': metadata or {}
        }
        
        self.llm_calls.append(llm_info)
        
        if self.enable_logging:
            self.logger.info(f"LLM call to {model} - {duration:.2f}s" if duration else f"LLM call to {model}")
            self.logger.debug(f"  Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"  Prompt: {prompt}")
            self.logger.debug(f"  Response: {response[:200]}..." if len(response) > 200 else f"  Response: {response}")
        
        # Save LLM call details
        self._save_llm_call(llm_info)
    
    def log_tool_execution(self, tool_name: str, tool_inputs: Dict[str, Any],
                          tool_outputs: Dict[str, Any], duration: float = None,
                          status: str = "completed", error: str = None):
        """Log tool execution details."""
        tool_info = {
            'timestamp': datetime.now().isoformat(),
            'tool_name': tool_name,
            'tool_inputs': tool_inputs,
            'tool_outputs': tool_outputs,
            'duration': duration,
            'status': status,
            'error': error
        }
        
        self.tool_executions.append(tool_info)
        
        if self.enable_logging:
            self.logger.info(f"Tool execution: {tool_name} - {status}")
            if duration:
                self.logger.info(f"  Duration: {duration:.2f}s")
            if error:
                self.logger.error(f"  Error: {error}")
        
        # Save tool execution details
        self._save_tool_execution(tool_info)
    
    def log_output_generation(self, output_type: str, output_data: Any,
                             output_path: str = None, metadata: Dict[str, Any] = None):
        """Log output generation."""
        output_info = {
            'timestamp': datetime.now().isoformat(),
            'output_type': output_type,
            'output_data': output_data,
            'output_path': output_path,
            'metadata': metadata or {}
        }
        
        self.outputs_generated.append(output_info)
        
        if self.enable_logging:
            self.logger.info(f"Output generated: {output_type}")
            if output_path:
                self.logger.info(f"  Path: {output_path}")
        
        # Save output details
        self._save_output_info(output_info)
    
    def log_error(self, error_type: str, error_message: str, 
                  error_traceback: str = None, context: Dict[str, Any] = None):
        """Log error information."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'error_traceback': error_traceback,
            'context': context or {}
        }
        
        self.errors_encountered.append(error_info)
        
        if self.enable_logging:
            self.logger.error(f"Error: {error_type} - {error_message}")
            if error_traceback:
                self.logger.error(f"  Traceback: {error_traceback}")
        
        # Save error details
        self._save_error_info(error_info)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, 
                      description: str = "", metadata: Dict[str, Any] = None) -> str:
        """Save DataFrame to session folder with metadata."""
        try:
            # Create data subfolder
            data_folder = self.session_folder / "data"
            data_folder.mkdir(exist_ok=True)
            
            # Save DataFrame
            file_path = data_folder / filename
            if filename.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif filename.endswith('.json'):
                df.to_json(file_path, orient='records', indent=2)
            elif filename.endswith('.parquet'):
                df.to_parquet(file_path, index=False)
            else:
                # Default to CSV
                file_path = file_path.with_suffix('.csv')
                df.to_csv(file_path, index=False)
            
            # Log output generation
            self.log_output_generation(
                output_type='dataframe',
                output_data={'shape': df.shape, 'columns': list(df.columns)},
                output_path=str(file_path),
                metadata={
                    'description': description,
                    'file_type': file_path.suffix,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    **(metadata or {})
                }
            )
            
            return str(file_path)
            
        except Exception as e:
            self.log_error('dataframe_save', str(e), traceback.format_exc(), 
                          {'filename': filename, 'dataframe_shape': df.shape if hasattr(df, 'shape') else 'unknown'})
            raise
    
    def save_visualization(self, visualization_data: Any, filename: str,
                          visualization_type: str, metadata: Dict[str, Any] = None) -> str:
        """Save visualization to session folder."""
        try:
            # Create visualizations subfolder
            viz_folder = self.session_folder / "visualizations"
            viz_folder.mkdir(exist_ok=True)
            
            file_path = viz_folder / filename
            
            # Handle different visualization types
            if visualization_type == 'html':
                if hasattr(visualization_data, 'save'):
                    visualization_data.save(str(file_path))
                else:
                    with open(file_path, 'w') as f:
                        f.write(str(visualization_data))
            elif visualization_type in ['png', 'jpg', 'jpeg']:
                if hasattr(visualization_data, 'savefig'):
                    visualization_data.savefig(file_path, dpi=300, bbox_inches='tight')
                else:
                    # Assume it's already an image
                    with open(file_path, 'wb') as f:
                        f.write(visualization_data)
            elif visualization_type == 'svg':
                if hasattr(visualization_data, 'savefig'):
                    visualization_data.savefig(file_path, format='svg', bbox_inches='tight')
                else:
                    with open(file_path, 'w') as f:
                        f.write(str(visualization_data))
            else:
                # Default to saving as text
                with open(file_path, 'w') as f:
                    f.write(str(visualization_data))
            
            # Log output generation
            self.log_output_generation(
                output_type='visualization',
                output_data={'visualization_type': visualization_type},
                output_path=str(file_path),
                metadata={
                    'visualization_type': visualization_type,
                    'file_size': file_path.stat().st_size if file_path.exists() else 0,
                    **(metadata or {})
                }
            )
            
            return str(file_path)
            
        except Exception as e:
            self.log_error('visualization_save', str(e), traceback.format_exc(),
                          {'filename': filename, 'visualization_type': visualization_type})
            raise
    
    def save_text_output(self, content: str, filename: str, 
                        content_type: str = "text", metadata: Dict[str, Any] = None) -> str:
        """Save text content to session folder."""
        try:
            # Create text subfolder
            text_folder = self.session_folder / "text"
            text_folder.mkdir(exist_ok=True)
            
            file_path = text_folder / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log output generation
            self.log_output_generation(
                output_type='text',
                output_data={'content_type': content_type, 'length': len(content)},
                output_path=str(file_path),
                metadata={
                    'content_type': content_type,
                    'file_size': len(content.encode('utf-8')),
                    'lines': content.count('\n') + 1,
                    **(metadata or {})
                }
            )
            
            return str(file_path)
            
        except Exception as e:
            self.log_error('text_save', str(e), traceback.format_exc(),
                          {'filename': filename, 'content_type': content_type})
            raise
    
    def _save_workflow_step(self, step_info: Dict[str, Any]):
        """Save workflow step information."""
        steps_file = self.session_folder / "workflow_steps.json"
        with open(steps_file, 'w') as f:
            json.dump(self.workflow_steps, f, indent=2, default=str)
    
    def _save_llm_call(self, llm_info: Dict[str, Any]):
        """Save LLM call information."""
        llm_file = self.session_folder / "llm_calls.json"
        with open(llm_file, 'w') as f:
            json.dump(self.llm_calls, f, indent=2, default=str)
    
    def _save_tool_execution(self, tool_info: Dict[str, Any]):
        """Save tool execution information."""
        tools_file = self.session_folder / "tool_executions.json"
        with open(tools_file, 'w') as f:
            json.dump(self.tool_executions, f, indent=2, default=str)
    
    def _save_output_info(self, output_info: Dict[str, Any]):
        """Save output information."""
        outputs_file = self.session_folder / "outputs.json"
        with open(outputs_file, 'w') as f:
            json.dump(self.outputs_generated, f, indent=2, default=str)
    
    def _save_error_info(self, error_info: Dict[str, Any]):
        """Save error information."""
        errors_file = self.session_folder / "errors.json"
        with open(errors_file, 'w') as f:
            json.dump(self.errors_encountered, f, indent=2, default=str)
    
    def generate_session_report(self) -> str:
        """Generate comprehensive session report."""
        try:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            report_data = {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'duration_formatted': f"{duration:.2f}s",
                'summary': {
                    'workflow_steps': len(self.workflow_steps),
                    'llm_calls': len(self.llm_calls),
                    'tool_executions': len(self.tool_executions),
                    'outputs_generated': len(self.outputs_generated),
                    'errors_encountered': len(self.errors_encountered)
                },
                'workflow_steps': self.workflow_steps,
                'llm_calls': self.llm_calls,
                'tool_executions': self.tool_executions,
                'outputs_generated': self.outputs_generated,
                'errors_encountered': self.errors_encountered,
                'performance_metrics': self.performance_metrics
            }
            
            # Save comprehensive report
            report_file = self.session_folder / "session_report.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Generate human-readable summary
            summary_file = self.session_folder / "session_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Session Report: {self.session_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Start Time: {self.start_time}\n")
                f.write(f"End Time: {end_time}\n")
                f.write(f"Duration: {duration:.2f} seconds\n\n")
                f.write(f"Workflow Steps: {len(self.workflow_steps)}\n")
                f.write(f"LLM Calls: {len(self.llm_calls)}\n")
                f.write(f"Tool Executions: {len(self.tool_executions)}\n")
                f.write(f"Outputs Generated: {len(self.outputs_generated)}\n")
                f.write(f"Errors Encountered: {len(self.errors_encountered)}\n\n")
                
                if self.workflow_steps:
                    f.write("Workflow Steps:\n")
                    f.write("-" * 20 + "\n")
                    for step in self.workflow_steps:
                        f.write(f"• {step['step_name']} ({step['status']})\n")
                        if step.get('duration'):
                            f.write(f"  Duration: {step['duration']:.2f}s\n")
                
                if self.outputs_generated:
                    f.write("\nOutputs Generated:\n")
                    f.write("-" * 20 + "\n")
                    for output in self.outputs_generated:
                        f.write(f"• {output['output_type']}\n")
                        if output.get('output_path'):
                            f.write(f"  Path: {output['output_path']}\n")
                
                if self.errors_encountered:
                    f.write("\nErrors Encountered:\n")
                    f.write("-" * 20 + "\n")
                    for error in self.errors_encountered:
                        f.write(f"• {error['error_type']}: {error['error_message']}\n")
            
            # Log report generation
            self.log_output_generation(
                output_type='session_report',
                output_data={'report_files': [str(report_file), str(summary_file)]},
                output_path=str(report_file),
                metadata={'duration': duration, 'total_activities': len(self.workflow_steps) + len(self.llm_calls) + len(self.tool_executions)}
            )
            
            return str(report_file)
            
        except Exception as e:
            self.log_error('report_generation', str(e), traceback.format_exc())
            raise
    
    def end_session(self):
        """End session and generate final report."""
        try:
            if self.enable_logging:
                self.logger.info(f"Session {self.session_id} ending")
            
            # Generate final report
            report_path = self.generate_session_report()
            
            # Log session end
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            if self.enable_logging:
                self.logger.info(f"Session {self.session_id} ended after {duration:.2f} seconds")
                self.logger.info(f"Final report: {report_path}")
            
            return report_path
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error ending session: {e}")
            raise
    
    def get_session_path(self) -> Path:
        """Get the session folder path."""
        return self.session_folder
    
    def get_session_logs_path(self) -> Path:
        """Get the session logs directory path."""
        return self.session_logs_dir
    
    def get_session_outputs_path(self) -> Path:
        """Get the session outputs directory path."""
        return self.session_outputs_dir

# Example usage
if __name__ == "__main__":
    # Test session manager
    session_mgr = EnhancedSessionManager()
    
    # Log some activities
    session_mgr.log_workflow_step("data_import", {"source": "test.csv"}, duration=1.5)
    session_mgr.log_llm_call("gpt-4", "Analyze this data", "Analysis complete", duration=2.3)
    session_mgr.log_tool_execution("pandas_processor", {"operation": "groupby"}, {"result": "success"}, duration=0.5)
    
    # Save some outputs
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    session_mgr.save_dataframe(df, "test_output.csv", "Test data output")
    
    # End session and generate report
    report_path = session_mgr.end_session()
    print(f"Session completed. Report: {report_path}")
