#!/usr/bin/env python3
"""
Star Trek Computer-Style Conversational Interface
Natural conversation interface for cybersecurity operations
"""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context for maintaining conversation state"""
    user_name: str = "User"
    session_start: datetime = None
    last_intent: str = ""
    conversation_history: List[Dict] = None
    current_operation: str = ""
    operation_progress: float = 0.0

class StarTrekInterface:
    """Natural conversation interface for cybersecurity operations"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.context = ConversationContext()
        self.context.session_start = datetime.now()
        self.context.conversation_history = []
        
        # Response templates
        self.templates = {
            'greetings': [
                "Good {time}! How can I assist with your cybersecurity analysis?",
                "Hello! Ready to tackle some threat hunting?",
                "Hi there! What security challenge can I help you with?",
                "Greetings! I'm here to help with your security operations.",
                "Welcome! What cybersecurity task shall we work on today?"
            ],
            'acknowledgments': [
                "Understood. I'll {action} {target} for {analysis_type}.",
                "Got it! Analyzing {target} for {analysis_type} indicators.",
                "Roger that. Running {analysis_type} analysis on {target}.",
                "Acknowledged. Initiating {analysis_type} on {target}.",
                "Confirmed. Beginning {analysis_type} assessment of {target}."
            ],
            'working': [
                "Scanning {target}... {progress}%",
                "Processing {analysis_type} analysis... Please stand by.",
                "Analyzing {count} targets... ETA {eta} seconds.",
                "Running security assessment... {progress}% complete.",
                "Executing {analysis_type}... {progress}% finished."
            ],
            'confirmations': [
                "I found {count} targets to analyze: {targets}. Proceed?",
                "Detected {count} potential targets: {targets}. Shall I continue?",
                "Located {count} items for {analysis_type}: {targets}. Proceed with analysis?",
                "Identified {count} targets: {targets}. Ready to begin {analysis_type}?",
                "Found {count} candidates: {targets}. Initiate {analysis_type}?"
            ],
            'completions': [
                "Analysis complete. {summary}",
                "Task finished. {summary}",
                "Operation successful. {summary}",
                "Assessment completed. {summary}",
                "Analysis done. {summary}"
            ],
            'errors': [
                "I encountered an issue: {error}. Let me try an alternative approach.",
                "Error detected: {error}. Attempting to resolve...",
                "Problem identified: {error}. Switching to backup method.",
                "Issue found: {error}. Implementing workaround...",
                "Error occurred: {error}. Trying different approach..."
            ],
            'clarifications': [
                "I need more information about {aspect}. Could you clarify?",
                "To proceed, I need details about {aspect}. Please provide more context.",
                "I'm not sure about {aspect}. Can you be more specific?",
                "I need clarification on {aspect}. Could you elaborate?",
                "I'm unclear about {aspect}. Please provide more details."
            ]
        }
        
        # Casual conversation responses
        self.casual_responses = {
            'greeting': [
                "Hello! I'm your cybersecurity assistant. How can I help today?",
                "Hi there! Ready to work on some security tasks?",
                "Hey! What security challenge can we tackle together?",
                "Good to see you! What's on your security agenda today?",
                "Hello! I'm here to help with your cybersecurity needs."
            ],
            'how_are_you': [
                "I'm functioning optimally and ready to assist with security operations.",
                "All systems operational! How can I help with your security tasks?",
                "I'm running smoothly and prepared for any cybersecurity challenges.",
                "Everything's working perfectly! What security work shall we do?",
                "I'm in top form and ready to tackle any security issues."
            ],
            'thanks': [
                "You're welcome! Happy to help with your security needs.",
                "My pleasure! Always here to assist with cybersecurity tasks.",
                "Glad I could help! Let me know if you need anything else.",
                "You're very welcome! Ready for the next security challenge.",
                "Happy to be of service! What else can I help with?"
            ],
            'goodbye': [
                "Goodbye! Stay secure out there!",
                "Farewell! Keep your systems protected!",
                "See you later! Remember to stay vigilant!",
                "Take care! Don't forget to update your security patches!",
                "Until next time! Keep your threat hunting skills sharp!"
            ]
        }
    
    def process_command(self, user_input: str, intent: str = "", params: Dict = None) -> str:
        """Main command processing with Star Trek computer personality"""
        if not params:
            params = {}
        
        # Update context
        self.context.last_intent = intent
        self.context.conversation_history.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'intent': intent,
            'params': params
        })
        
        # Handle casual conversation
        if intent == 'casual_conversation':
            return self.handle_casual_conversation(user_input)
        
        # Generate appropriate response based on intent and params
        if intent in ['malware_analysis', 'vulnerability_scan', 'network_analysis', 'file_forensics']:
            return self._generate_analysis_response(intent, params)
        elif intent == 'clarification_needed':
            return self._generate_clarification_response(user_input, params)
        else:
            return self._generate_general_response(intent, params)
    
    def generate_acknowledgment(self, intent: str, params: Dict) -> str:
        """Generate approachable acknowledgment that builds user confidence"""
        if not params:
            params = {}
        
        # Extract key information
        targets = params.get('targets', ['target'])
        analysis_type = self._get_analysis_type_name(intent)
        action = self._get_action_verb(intent)
        
        # Choose appropriate template
        template = random.choice(self.templates['acknowledgments'])
        
        # Format the response
        if len(targets) == 1:
            target = targets[0]
        else:
            target = f"{len(targets)} targets"
        
        response = template.format(
            action=action,
            target=target,
            analysis_type=analysis_type
        )
        
        # Add context if available
        if params.get('priority') == 'critical':
            response += " (High priority - processing immediately)"
        elif params.get('priority') == 'high':
            response += " (High priority - expediting analysis)"
        
        return response
    
    def create_progress_feedback(self, operation: str, status: str, eta: int = None) -> str:
        """Create clear, non-technical progress updates"""
        self.context.current_operation = operation
        self.context.operation_progress = self._calculate_progress(status)
        
        # Choose appropriate template
        template = random.choice(self.templates['working'])
        
        # Format progress
        progress = int(self.context.operation_progress * 100)
        
        # Format ETA
        eta_text = ""
        if eta:
            if eta < 60:
                eta_text = f"{eta} seconds"
            else:
                eta_text = f"{eta // 60} minutes"
        
        response = template.format(
            target=operation,
            analysis_type=self._get_analysis_type_name(operation),
            progress=progress,
            count=1,
            eta=eta_text
        )
        
        # Add status-specific details
        if status == "starting":
            response += " Initializing systems..."
        elif status == "processing":
            response += " Deep analysis in progress..."
        elif status == "finalizing":
            response += " Compiling results..."
        
        return response
    
    def suggest_follow_up_actions(self, results: Dict, original_intent: str) -> List[str]:
        """Suggest helpful next steps based on analysis results"""
        suggestions = []
        
        # Base suggestions on intent
        if original_intent == 'malware_analysis':
            if results.get('threats_found', 0) > 0:
                suggestions.extend([
                    "Run additional malware analysis on related files",
                    "Check for similar threats in your network",
                    "Update your antivirus signatures",
                    "Review security policies for prevention"
                ])
            else:
                suggestions.extend([
                    "Continue monitoring for similar threats",
                    "Update your threat intelligence feeds",
                    "Review file access logs for anomalies"
                ])
        
        elif original_intent == 'vulnerability_scan':
            if results.get('vulnerabilities_found', 0) > 0:
                suggestions.extend([
                    "Prioritize patching critical vulnerabilities",
                    "Run additional scans on affected systems",
                    "Review security configurations",
                    "Implement additional monitoring"
                ])
            else:
                suggestions.extend([
                    "Schedule regular vulnerability scans",
                    "Review security hardening guidelines",
                    "Update scanning tools and signatures"
                ])
        
        elif original_intent == 'network_analysis':
            if results.get('anomalies_found', 0) > 0:
                suggestions.extend([
                    "Investigate anomalous network traffic",
                    "Review firewall rules and policies",
                    "Check for unauthorized access attempts",
                    "Implement network segmentation"
                ])
            else:
                suggestions.extend([
                    "Continue monitoring network traffic",
                    "Review network security policies",
                    "Update network monitoring tools"
                ])
        
        # Add general suggestions
        suggestions.extend([
            "Review the detailed analysis report",
            "Update your security documentation",
            "Schedule follow-up assessments",
            "Share findings with your security team"
        ])
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def handle_casual_conversation(self, text: str) -> str:
        """Handle non-work conversation in a professional but friendly way"""
        text_lower = text.lower()
        
        # Greeting responses
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return random.choice(self.casual_responses['greeting'])
        
        # How are you responses
        elif any(phrase in text_lower for phrase in ['how are you', 'how are things', 'how\'s it going']):
            return random.choice(self.casual_responses['how_are_you'])
        
        # Thanks responses
        elif any(word in text_lower for word in ['thanks', 'thank you', 'appreciate']):
            return random.choice(self.casual_responses['thanks'])
        
        # Goodbye responses
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return random.choice(self.casual_responses['goodbye'])
        
        # Default casual response
        else:
            return "I'm here to help with your cybersecurity needs. What security task can we work on together?"
    
    def _generate_analysis_response(self, intent: str, params: Dict) -> str:
        """Generate response for analysis intents"""
        targets = params.get('targets', [])
        priority = params.get('priority', 'medium')
        
        if targets:
            if len(targets) == 1:
                target_text = targets[0]
            else:
                target_text = f"{len(targets)} targets"
            
            response = f"Understood. I'll analyze {target_text} for {self._get_analysis_type_name(intent)}."
            
            if priority == 'critical':
                response += " (High priority - processing immediately)"
            elif priority == 'high':
                response += " (High priority - expediting analysis)"
        else:
            response = f"Ready to perform {self._get_analysis_type_name(intent)}. What would you like me to analyze?"
        
        return response
    
    def _generate_clarification_response(self, user_input: str, params: Dict) -> str:
        """Generate response when clarification is needed"""
        # Try to identify what needs clarification
        if not params.get('targets'):
            aspect = "what you'd like me to analyze"
        elif not params.get('analysis_type'):
            aspect = "what type of analysis you need"
        else:
            aspect = "the specific requirements"
        
        template = random.choice(self.templates['clarifications'])
        return template.format(aspect=aspect)
    
    def _generate_general_response(self, intent: str, params: Dict) -> str:
        """Generate general response for other intents"""
        if intent == 'complex_analysis_request':
            return "I understand you need a comprehensive analysis. Let me connect you with our advanced analysis capabilities."
        else:
            return "I'm ready to help with your cybersecurity needs. What would you like me to do?"
    
    def _get_analysis_type_name(self, intent: str) -> str:
        """Convert intent to human-readable analysis type"""
        type_names = {
            'malware_analysis': 'malware analysis',
            'vulnerability_scan': 'vulnerability scanning',
            'network_analysis': 'network analysis',
            'file_forensics': 'forensic analysis',
            'threat_hunting': 'threat hunting',
            'incident_response': 'incident response'
        }
        return type_names.get(intent, 'security analysis')
    
    def _get_action_verb(self, intent: str) -> str:
        """Get action verb for the intent"""
        action_verbs = {
            'malware_analysis': 'analyze',
            'vulnerability_scan': 'scan',
            'network_analysis': 'analyze',
            'file_forensics': 'examine',
            'threat_hunting': 'hunt for threats in',
            'incident_response': 'investigate'
        }
        return action_verbs.get(intent, 'process')
    
    def _calculate_progress(self, status: str) -> float:
        """Calculate progress based on status"""
        progress_map = {
            'starting': 0.1,
            'processing': 0.5,
            'finalizing': 0.9,
            'complete': 1.0
        }
        return progress_map.get(status, 0.0)
    
    def create_confirmation_message(self, targets: List[str], analysis_type: str) -> str:
        """Create confirmation message before starting analysis"""
        if len(targets) == 1:
            target_text = targets[0]
        else:
            target_text = f"{len(targets)} targets: {', '.join(targets[:3])}"
            if len(targets) > 3:
                target_text += f" and {len(targets) - 3} more"
        
        template = random.choice(self.templates['confirmations'])
        return template.format(
            count=len(targets),
            targets=target_text,
            analysis_type=analysis_type
        )
    
    def create_completion_message(self, results: Dict) -> str:
        """Create completion message with results summary"""
        summary_parts = []
        
        if results.get('threats_found', 0) > 0:
            summary_parts.append(f"Found {results['threats_found']} potential threats")
        elif results.get('vulnerabilities_found', 0) > 0:
            summary_parts.append(f"Identified {results['vulnerabilities_found']} vulnerabilities")
        elif results.get('anomalies_found', 0) > 0:
            summary_parts.append(f"Detected {results['anomalies_found']} anomalies")
        else:
            summary_parts.append("No significant issues found")
        
        if results.get('files_processed', 0) > 0:
            summary_parts.append(f"Processed {results['files_processed']} files")
        
        summary = ". ".join(summary_parts)
        
        template = random.choice(self.templates['completions'])
        return template.format(summary=summary)
    
    def create_error_message(self, error: str) -> str:
        """Create user-friendly error message"""
        template = random.choice(self.templates['errors'])
        return template.format(error=error)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        session_duration = datetime.now() - self.context.session_start
        
        return {
            'session_duration': str(session_duration),
            'total_interactions': len(self.context.conversation_history),
            'current_operation': self.context.current_operation,
            'operation_progress': self.context.operation_progress,
            'last_intent': self.context.last_intent
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the interface
    interface = StarTrekInterface()
    
    test_cases = [
        ("analyze sample X-47 for malware", "malware_analysis", {"targets": ["X-47"], "priority": "high"}),
        ("hello, how are you?", "casual_conversation", {}),
        ("scan 192.168.1.1 for vulnerabilities", "vulnerability_scan", {"targets": ["192.168.1.1"]}),
        ("thanks for your help", "casual_conversation", {}),
        ("I need help with something", "clarification_needed", {})
    ]
    
    print("🧪 Testing Star Trek Conversational Interface")
    print("=" * 60)
    
    for user_input, intent, params in test_cases:
        print(f"\nInput: {user_input}")
        print(f"Intent: {intent}")
        response = interface.process_command(user_input, intent, params)
        print(f"Response: {response}")
    
    # Test progress feedback
    print(f"\n📊 Progress Feedback:")
    print(interface.create_progress_feedback("malware_analysis", "processing", 30))
    print(interface.create_progress_feedback("malware_analysis", "finalizing", 5))
    
    # Test suggestions
    print(f"\n💡 Follow-up Suggestions:")
    results = {"threats_found": 3, "files_processed": 15}
    suggestions = interface.suggest_follow_up_actions(results, "malware_analysis")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # Show session summary
    print(f"\n📈 Session Summary:")
    summary = interface.get_session_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
