#!/usr/bin/env python3
"""
Enhanced Input Processor
Main integration layer combining local processing with existing cybersecurity agent
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add the bin directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from local_processing.tiny_bert_classifier import TinyBERTClassifier, IntentResult
from local_processing.rule_based_extractor import CyberSecurityParameterExtractor, ExtractedParameters
from conversational.star_trek_interface import StarTrekInterface

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of enhanced input processing"""
    response: str
    source: str  # 'local', 'dynamic_workflow', 'mcp', 'llm_handoff'
    processing_time_ms: float
    confidence: float
    suggestions: List[str]
    intent: str
    parameters: Dict[str, Any]
    reasoning: str
    local_processing_used: bool

class EnhancedInputProcessor:
    """Main processor combining all enhancements with existing agent"""
    
    def __init__(self, existing_agent=None, llm_client=None):
        # Local processing components
        self.intent_classifier = TinyBERTClassifier(llm_client=llm_client)
        self.param_extractor = CyberSecurityParameterExtractor()
        self.conversational = StarTrekInterface(llm_client=llm_client)
        
        # Existing agent integration
        self.existing_agent = existing_agent
        self.llm_client = llm_client
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'local_processing': 0,
            'dynamic_workflows': 0,
            'mcp_optimizations': 0,
            'llm_handoffs': 0,
            'avg_processing_time_ms': 0.0,
            'user_satisfaction_score': 0.0
        }
        
        # Local tool mappings (these would be connected to your existing tools)
        self.local_tools = {
            'malware_analysis': self._local_malware_analysis,
            'vulnerability_scan': self._local_vulnerability_scan,
            'network_analysis': self._local_network_analysis,
            'file_forensics': self._local_file_forensics,
            'threat_hunting': self._local_threat_hunting
        }
        
        logger.info("Enhanced Input Processor initialized")
    
    def process_input(self, problem_text: str) -> ProcessingResult:
        """
        Enhanced processing pipeline with all improvements
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 1. Local intent classification and parameter extraction (fast)
            intent_result = self.intent_classifier.classify_intent(problem_text)
            extracted_params = self.param_extractor.extract_all_parameters(problem_text)
            
            # 2. Generate user-friendly acknowledgment
            acknowledgment = self.conversational.generate_acknowledgment(
                intent_result.intent, extracted_params.__dict__
            )
            
            # 3. Decision logic: local vs dynamic workflow vs MCP vs external LLM
            if self.can_handle_with_local_tools(intent_result, extracted_params):
                return self.execute_local_workflow(intent_result, extracted_params, problem_text, acknowledgment, start_time)
                
            elif self.requires_dynamic_workflow(intent_result, extracted_params):
                return self.execute_dynamic_workflow(intent_result, extracted_params, problem_text, acknowledgment, start_time)
                
            elif self.requires_mcp_tools(intent_result, extracted_params):
                return self.execute_enhanced_mcp_workflow(intent_result, extracted_params, problem_text, acknowledgment, start_time)
                
            else:
                return self.handoff_to_existing_agent(problem_text, intent_result, extracted_params, acknowledgment, start_time)
                
        except Exception as e:
            logger.error(f"Error in enhanced processing: {e}")
            return self._create_error_result(str(e), start_time)
    
    def can_handle_with_local_tools(self, intent_result: IntentResult, params: ExtractedParameters) -> bool:
        """Determine if query can be handled with existing local cybersecurity tools"""
        return (intent_result.confidence > 0.9 and 
                intent_result.intent in ['malware_analysis', 'vulnerability_scan', 'file_forensics'] and
                len(params.targets) > 0 and
                intent_result.can_handle_locally)
    
    def requires_dynamic_workflow(self, intent_result: IntentResult, params: ExtractedParameters) -> bool:
        """Check if query needs intelligent tool orchestration"""
        return (intent_result.intent in ['network_analysis', 'incident_response'] and
                len(params.targets) > 1 and
                intent_result.confidence > 0.7)
    
    def requires_mcp_tools(self, intent_result: IntentResult, params: ExtractedParameters) -> bool:
        """Check if query needs MCP tools (Drive, Hugging Face, Chrome, etc.)"""
        mcp_keywords = ['search drive', 'hugging face', 'browser', 'gmail', 'cloudflare', 'google drive']
        return any(keyword in intent_result.original_text.lower() for keyword in mcp_keywords)
    
    def execute_local_workflow(self, intent_result: IntentResult, params: ExtractedParameters, 
                              problem_text: str, acknowledgment: str, start_time: float) -> ProcessingResult:
        """Execute workflow using local tools"""
        self.stats['local_processing'] += 1
        
        try:
            # Get local tool function
            tool_function = self.local_tools.get(intent_result.intent)
            if not tool_function:
                return self._create_error_result(f"No local tool available for {intent_result.intent}", start_time)
            
            # Execute local tool
            results = tool_function(params)
            
            # Generate response
            response = f"{acknowledgment}\n\n{results.get('summary', 'Analysis completed.')}"
            
            # Generate suggestions
            suggestions = self.conversational.suggest_follow_up_actions(results, intent_result.intent)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, True)
            
            return ProcessingResult(
                response=response,
                source='local',
                processing_time_ms=processing_time,
                confidence=intent_result.confidence,
                suggestions=suggestions,
                intent=intent_result.intent,
                parameters=params.__dict__,
                reasoning=f"Local processing: {intent_result.reasoning}",
                local_processing_used=True
            )
            
        except Exception as e:
            logger.error(f"Error in local workflow execution: {e}")
            return self._create_error_result(f"Local processing failed: {str(e)}", start_time)
    
    def execute_dynamic_workflow(self, intent_result: IntentResult, params: ExtractedParameters,
                                problem_text: str, acknowledgment: str, start_time: float) -> ProcessingResult:
        """Execute workflow using dynamic tool orchestration"""
        self.stats['dynamic_workflows'] += 1
        
        try:
            # This would integrate with your existing workflow system
            # For now, we'll simulate dynamic workflow execution
            response = f"{acknowledgment}\n\nExecuting dynamic workflow for {intent_result.intent}..."
            
            # Simulate workflow execution
            time.sleep(0.1)  # Simulate processing time
            
            results = {
                'summary': f"Dynamic workflow completed for {len(params.targets)} targets",
                'targets_processed': len(params.targets),
                'workflow_type': intent_result.intent
            }
            
            suggestions = self.conversational.suggest_follow_up_actions(results, intent_result.intent)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, False)
            
            return ProcessingResult(
                response=response,
                source='dynamic_workflow',
                processing_time_ms=processing_time,
                confidence=intent_result.confidence,
                suggestions=suggestions,
                intent=intent_result.intent,
                parameters=params.__dict__,
                reasoning=f"Dynamic workflow: {intent_result.reasoning}",
                local_processing_used=False
            )
            
        except Exception as e:
            logger.error(f"Error in dynamic workflow execution: {e}")
            return self._create_error_result(f"Dynamic workflow failed: {str(e)}", start_time)
    
    def execute_enhanced_mcp_workflow(self, intent_result: IntentResult, params: ExtractedParameters,
                                     problem_text: str, acknowledgment: str, start_time: float) -> ProcessingResult:
        """Execute workflow using enhanced MCP integration"""
        self.stats['mcp_optimizations'] += 1
        
        try:
            # This would integrate with your existing MCP system
            # For now, we'll simulate MCP workflow execution
            response = f"{acknowledgment}\n\nExecuting MCP workflow for {intent_result.intent}..."
            
            # Simulate MCP execution
            time.sleep(0.2)  # Simulate processing time
            
            results = {
                'summary': f"MCP workflow completed for {intent_result.intent}",
                'mcp_tools_used': ['drive_search', 'hugging_face_analysis'],
                'targets_processed': len(params.targets)
            }
            
            suggestions = self.conversational.suggest_follow_up_actions(results, intent_result.intent)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, False)
            
            return ProcessingResult(
                response=response,
                source='mcp',
                processing_time_ms=processing_time,
                confidence=intent_result.confidence,
                suggestions=suggestions,
                intent=intent_result.intent,
                parameters=params.__dict__,
                reasoning=f"MCP workflow: {intent_result.reasoning}",
                local_processing_used=False
            )
            
        except Exception as e:
            logger.error(f"Error in MCP workflow execution: {e}")
            return self._create_error_result(f"MCP workflow failed: {str(e)}", start_time)
    
    def handoff_to_existing_agent(self, problem_text: str, intent_result: IntentResult,
                                 params: ExtractedParameters, acknowledgment: str, start_time: float) -> ProcessingResult:
        """Hand off to existing agent for complex processing"""
        self.stats['llm_handoffs'] += 1
        
        try:
            if self.existing_agent:
                # Use existing agent
                response = f"{acknowledgment}\n\nHanding off to advanced analysis system..."
                
                # Simulate existing agent processing
                time.sleep(0.3)  # Simulate processing time
                
                results = {
                    'summary': "Advanced analysis completed using external LLM",
                    'complexity': 'high',
                    'llm_used': True
                }
                
                suggestions = [
                    "Review the detailed analysis report",
                    "Consider additional specialized analysis",
                    "Update your threat intelligence feeds",
                    "Schedule follow-up assessments"
                ]
                
                processing_time = (time.time() - start_time) * 1000
                self._update_stats(processing_time, False)
                
                return ProcessingResult(
                    response=response,
                    source='llm_handoff',
                    processing_time_ms=processing_time,
                    confidence=intent_result.confidence,
                    suggestions=suggestions,
                    intent=intent_result.intent,
                    parameters=params.__dict__,
                    reasoning=f"LLM handoff: {intent_result.reasoning}",
                    local_processing_used=False
                )
            else:
                return self._create_error_result("No existing agent available for handoff", start_time)
                
        except Exception as e:
            logger.error(f"Error in LLM handoff: {e}")
            return self._create_error_result(f"LLM handoff failed: {str(e)}", start_time)
    
    def _local_malware_analysis(self, params: ExtractedParameters) -> Dict[str, Any]:
        """Simulate local malware analysis"""
        # This would integrate with your actual malware analysis tools
        return {
            'summary': f"Analyzed {len(params.targets)} files for malware",
            'threats_found': 2 if len(params.targets) > 0 else 0,
            'files_processed': len(params.targets),
            'analysis_type': 'malware_analysis'
        }
    
    def _local_vulnerability_scan(self, params: ExtractedParameters) -> Dict[str, Any]:
        """Simulate local vulnerability scanning"""
        return {
            'summary': f"Scanned {len(params.targets)} targets for vulnerabilities",
            'vulnerabilities_found': 3 if len(params.targets) > 0 else 0,
            'targets_scanned': len(params.targets),
            'analysis_type': 'vulnerability_scan'
        }
    
    def _local_network_analysis(self, params: ExtractedParameters) -> Dict[str, Any]:
        """Simulate local network analysis"""
        return {
            'summary': f"Analyzed network traffic for {len(params.targets)} targets",
            'anomalies_found': 1 if len(params.targets) > 0 else 0,
            'packets_analyzed': 1000,
            'analysis_type': 'network_analysis'
        }
    
    def _local_file_forensics(self, params: ExtractedParameters) -> Dict[str, Any]:
        """Simulate local file forensics"""
        return {
            'summary': f"Performed forensic analysis on {len(params.targets)} files",
            'artifacts_found': 5 if len(params.targets) > 0 else 0,
            'files_analyzed': len(params.targets),
            'analysis_type': 'file_forensics'
        }
    
    def _local_threat_hunting(self, params: ExtractedParameters) -> Dict[str, Any]:
        """Simulate local threat hunting"""
        return {
            'summary': f"Conducted threat hunting across {len(params.targets)} targets",
            'iocs_found': 4 if len(params.targets) > 0 else 0,
            'threats_detected': 1 if len(params.targets) > 0 else 0,
            'analysis_type': 'threat_hunting'
        }
    
    def _create_error_result(self, error_message: str, start_time: float) -> ProcessingResult:
        """Create error result"""
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            response=f"I encountered an error: {error_message}",
            source='error',
            processing_time_ms=processing_time,
            confidence=0.0,
            suggestions=["Please try rephrasing your request", "Check your input parameters"],
            intent='error',
            parameters={},
            reasoning=f"Error: {error_message}",
            local_processing_used=False
        )
    
    def _update_stats(self, processing_time: float, local_processing: bool) -> None:
        """Update performance statistics"""
        # Update average processing time
        total_time = self.stats['avg_processing_time_ms'] * (self.stats['total_requests'] - 1)
        self.stats['avg_processing_time_ms'] = (total_time + processing_time) / self.stats['total_requests']
        
        # Update satisfaction score (simplified)
        if local_processing and processing_time < 100:  # Fast local processing
            self.stats['user_satisfaction_score'] = min(1.0, self.stats['user_satisfaction_score'] + 0.1)
        elif not local_processing and processing_time > 1000:  # Slow external processing
            self.stats['user_satisfaction_score'] = max(0.0, self.stats['user_satisfaction_score'] - 0.05)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['local_processing_rate'] = stats['local_processing'] / stats['total_requests']
            stats['llm_handoff_rate'] = stats['llm_handoffs'] / stats['total_requests']
            stats['mcp_usage_rate'] = stats['mcp_optimizations'] / stats['total_requests']
            stats['dynamic_workflow_rate'] = stats['dynamic_workflows'] / stats['total_requests']
        else:
            stats['local_processing_rate'] = 0.0
            stats['llm_handoff_rate'] = 0.0
            stats['mcp_usage_rate'] = 0.0
            stats['dynamic_workflow_rate'] = 0.0
        
        # Add classifier stats
        classifier_stats = self.intent_classifier.get_performance_stats()
        stats['classifier_stats'] = classifier_stats
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.stats = {
            'total_requests': 0,
            'local_processing': 0,
            'dynamic_workflows': 0,
            'mcp_optimizations': 0,
            'llm_handoffs': 0,
            'avg_processing_time_ms': 0.0,
            'user_satisfaction_score': 0.0
        }
        self.intent_classifier.reset_stats()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the enhanced processor
    processor = EnhancedInputProcessor()
    
    test_cases = [
        "analyze sample X-47 for malware",
        "scan 192.168.1.1 for vulnerabilities",
        "hello, how are you?",
        "search my Google Drive for incident reports and analyze them",
        "provide detailed attribution analysis correlating multiple APT campaigns"
    ]
    
    print("🧪 Testing Enhanced Input Processor")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        result = processor.process_input(text)
        
        print(f"Response: {result.response}")
        print(f"Source: {result.source}")
        print(f"Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Local Processing: {result.local_processing_used}")
        print(f"Intent: {result.intent}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.suggestions:
            print("Suggestions:")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")
    
    # Show performance stats
    print(f"\n📊 Performance Stats:")
    stats = processor.get_performance_stats()
    for key, value in stats.items():
        if key != 'classifier_stats':
            print(f"{key}: {value}")
    
    if 'classifier_stats' in stats:
        print(f"\nClassifier Stats:")
        for key, value in stats['classifier_stats'].items():
            print(f"  {key}: {value}")
