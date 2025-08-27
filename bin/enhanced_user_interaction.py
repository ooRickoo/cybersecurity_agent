"""
Enhanced User Interaction System for Cybersecurity Agent

Provides fluid, natural conversation capabilities with intelligent context understanding,
dynamic workflow adaptation, and seamless problem-solving experience.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import spacy
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class EnhancedUserInteraction:
    """Enhanced user interaction system for natural conversation flow."""
    
    def __init__(self, session_manager, memory_manager, workflow_manager):
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.workflow_manager = workflow_manager
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            logger.warning("spaCy English model not available. Install with: python -m spacy download en_core_web_sm")
        
        # Conversation context tracking
        self.conversation_context = {
            'current_topic': None,
            'recent_questions': [],
            'user_preferences': {},
            'workflow_history': [],
            'suggested_actions': []
        }
        
        # Intent patterns for natural language understanding
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Context-aware response templates
        self.response_templates = self._initialize_response_templates()
        
        # Workflow suggestion patterns
        self.workflow_suggestions = self._initialize_workflow_suggestions()
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for intent recognition."""
        return {
            'threat_hunting': [
                r'\b(hunt|find|detect|identify|locate|search for)\b.*\b(threat|malware|attack|intrusion|breach|compromise)\b',
                r'\b(analyze|examine|investigate)\b.*\b(suspicious|anomalous|unusual|strange)\b',
                r'\b(look for|check|scan)\b.*\b(ioc|indicator|signature)\b',
                r'\b(security|threat)\b.*\b(incident|event|alert)\b'
            ],
            'policy_analysis': [
                r'\b(analyze|review|examine|assess)\b.*\b(policy|procedure|standard|guideline)\b',
                r'\b(compliance|audit|governance)\b.*\b(check|verify|validate)\b',
                r'\b(security|cyber)\b.*\b(framework|control|requirement)\b',
                r'\b(map|align)\b.*\b(framework|standard|regulation)\b'
            ],
            'incident_response': [
                r'\b(respond|handle|manage|coordinate)\b.*\b(incident|breach|attack|alert)\b',
                r'\b(contain|mitigate|remediate)\b.*\b(threat|vulnerability|risk)\b',
                r'\b(escalate|notify|alert)\b.*\b(stakeholder|team|management)\b',
                r'\b(forensic|evidence|timeline)\b.*\b(collect|preserve|analyze)\b'
            ],
            'vulnerability_assessment': [
                r'\b(scan|assess|evaluate|test)\b.*\b(vulnerability|weakness|exposure)\b',
                r'\b(penetration|pen|security)\b.*\b(test|assessment|audit)\b',
                r'\b(risk|threat)\b.*\b(analysis|assessment|evaluation)\b',
                r'\b(patch|update|upgrade)\b.*\b(security|vulnerability)\b'
            ],
            'data_analysis': [
                r'\b(analyze|examine|process|review)\b.*\b(data|dataset|log|event)\b',
                r'\b(extract|identify|find)\b.*\b(pattern|trend|anomaly|insight)\b',
                r'\b(visualize|chart|graph|plot)\b.*\b(data|information|results)\b',
                r'\b(correlate|compare|contrast)\b.*\b(data|events|sources)\b'
            ],
            'compliance_assessment': [
                r'\b(assess|evaluate|check)\b.*\b(compliance|conformance|adherence)\b',
                r'\b(audit|review|examine)\b.*\b(standard|regulation|requirement)\b',
                r'\b(gap|gap analysis)\b.*\b(compliance|requirement|standard)\b',
                r'\b(report|document)\b.*\b(compliance|audit|assessment)\b'
            ],
            'memory_management': [
                r'\b(backup|restore|save|export)\b.*\b(memory|data|information)\b',
                r'\b(delete|remove|clean)\b.*\b(memory|data|information)\b',
                r'\b(import|load|add)\b.*\b(data|information|knowledge)\b',
                r'\b(search|query|find)\b.*\b(memory|stored|saved)\b'
            ],
            'output_distribution': [
                r'\b(send|distribute|upload|transfer)\b.*\b(file|data|report)\b',
                r'\b(export|share|deliver)\b.*\b(result|output|finding)\b',
                r'\b(stream|transmit|forward)\b.*\b(data|information|alert)\b',
                r'\b(backup|archive|store)\b.*\b(file|data|information)\b'
            ],
            'general_inquiry': [
                r'\b(what|how|why|when|where|who)\b.*\b(is|are|does|can|should)\b',
                r'\b(explain|describe|tell me about|show me)\b',
                r'\b(help|assist|support|guide)\b',
                r'\b(capability|feature|function|tool)\b'
            ]
        }
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize context-aware response templates."""
        return {
            'threat_hunting': [
                "I can help you hunt for threats! Let me understand what you're looking for. Are you investigating a specific incident, or do you want to run proactive threat hunting?",
                "Great! Let's set up a threat hunting workflow. I can help you analyze logs, search for IOCs, and identify suspicious patterns. What type of threat are you concerned about?",
                "I'm ready to help with threat hunting! I can search through various data sources, analyze patterns, and help you identify potential threats. What's your starting point?"
            ],
            'policy_analysis': [
                "I can help analyze security policies and map them to frameworks! Let me understand what policies you want to examine and what frameworks you need to align with.",
                "Policy analysis is one of my strengths! I can help you review policies, identify gaps, and map them to security frameworks like NIST, ISO, or MITRE. What policies should we start with?",
                "Let's analyze your security policies! I can help you assess compliance, identify improvements, and map to industry standards. What's your current policy landscape?"
            ],
            'incident_response': [
                "I'm here to help with incident response! Let me understand the situation so I can guide you through the appropriate response steps.",
                "Incident response requires a systematic approach. I can help you coordinate the response, collect evidence, and ensure proper containment. What type of incident are we dealing with?",
                "Let's handle this incident properly! I can help you follow incident response procedures, coordinate with teams, and ensure proper documentation. What's the current status?"
            ],
            'vulnerability_assessment': [
                "I can help assess vulnerabilities and identify security weaknesses! Let me understand your environment and what you want to test.",
                "Vulnerability assessment is crucial for security! I can help you scan systems, identify weaknesses, and prioritize remediation efforts. What's your scope?",
                "Let's identify and assess vulnerabilities! I can help you run scans, analyze results, and create remediation plans. What systems should we focus on?"
            ],
            'data_analysis': [
                "I can help analyze data and extract insights! Let me understand what data you have and what patterns you're looking for.",
                "Data analysis is one of my core capabilities! I can help you process logs, identify trends, and visualize results. What data should we start with?",
                "Let's analyze your data! I can help you extract patterns, identify anomalies, and create visualizations. What's your data source?"
            ],
            'compliance_assessment': [
                "I can help assess compliance and identify gaps! Let me understand what standards you need to meet and what you want to evaluate.",
                "Compliance assessment is essential! I can help you evaluate your current state, identify gaps, and create improvement plans. What standards are you targeting?",
                "Let's assess your compliance posture! I can help you evaluate controls, identify gaps, and create remediation plans. What frameworks are you working with?"
            ],
            'memory_management': [
                "I can help manage your knowledge base! Let me understand what you want to do with your stored information.",
                "Memory management is important for maintaining your knowledge base! I can help you backup, restore, or clean up stored data. What operation do you need?",
                "Let's manage your knowledge base! I can help you organize, backup, or clean up stored information. What would you like to do?"
            ],
            'output_distribution': [
                "I can help distribute your outputs! Let me understand where you want to send files and in what format.",
                "Output distribution is flexible! I can help you send files to various destinations in multiple formats. Where do you want to send your results?",
                "Let's get your outputs where they need to go! I can help you distribute files to various systems and formats. What's your destination?"
            ],
            'general_inquiry': [
                "I'm here to help! Let me understand what you need assistance with.",
                "I can help with various cybersecurity tasks! What would you like to work on?",
                "I'm ready to assist! What cybersecurity challenge can I help you with today?"
            ]
        }
    
    def _initialize_workflow_suggestions(self) -> Dict[str, List[str]]:
        """Initialize workflow suggestion patterns."""
        return {
            'threat_hunting': [
                "I can suggest a threat hunting workflow that includes data collection, pattern analysis, and IOC identification.",
                "For threat hunting, I recommend starting with data discovery, then moving to pattern analysis and threat correlation.",
                "Let me create a comprehensive threat hunting workflow that covers all the essential steps."
            ],
            'policy_analysis': [
                "I can suggest a policy analysis workflow that includes policy review, gap analysis, and framework mapping.",
                "For policy analysis, I recommend starting with policy collection, then analyzing gaps and mapping to standards.",
                "Let me create a systematic policy analysis workflow that ensures comprehensive coverage."
            ],
            'incident_response': [
                "I can suggest an incident response workflow that follows industry best practices and ensures proper coordination.",
                "For incident response, I recommend following a structured approach with clear phases and responsibilities.",
                "Let me create an incident response workflow that ensures proper containment and documentation."
            ]
        }
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate intelligent response.
        
        Args:
            user_input: User's input text
            
        Returns:
            Processed response with intent, suggestions, and actions
        """
        try:
            # Update conversation context
            self._update_conversation_context(user_input)
            
            # Analyze intent
            intent = self._analyze_intent(user_input)
            
            # Generate context-aware response
            response = self._generate_context_response(intent, user_input)
            
            # Suggest relevant workflows
            workflow_suggestions = self._suggest_workflows(intent, user_input)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(intent, user_input)
            
            # Update context with current interaction
            self.conversation_context['current_topic'] = intent
            self.conversation_context['recent_questions'].append({
                'question': user_input,
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 questions
            if len(self.conversation_context['recent_questions']) > 10:
                self.conversation_context['recent_questions'] = self.conversation_context['recent_questions'][-10:]
            
            return {
                'success': True,
                'intent': intent,
                'response': response,
                'workflow_suggestions': workflow_suggestions,
                'follow_up_questions': follow_up_questions,
                'context': self.conversation_context,
                'suggested_actions': self._generate_suggested_actions(intent, user_input)
            }
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I apologize, but I encountered an error processing your request. Could you please rephrase or try a different approach?"
            }
    
    def _analyze_intent(self, user_input: str) -> str:
        """Analyze user input to determine intent."""
        try:
            user_input_lower = user_input.lower()
            
            # Check for exact matches first
            if any(word in user_input_lower for word in ['hunt', 'threat', 'malware', 'attack']):
                return 'threat_hunting'
            elif any(word in user_input_lower for word in ['policy', 'compliance', 'framework']):
                return 'policy_analysis'
            elif any(word in user_input_lower for word in ['incident', 'breach', 'response']):
                return 'incident_response'
            elif any(word in user_input_lower for word in ['vulnerability', 'scan', 'penetration']):
                return 'vulnerability_assessment'
            elif any(word in user_input_lower for word in ['data', 'analyze', 'log']):
                return 'data_analysis'
            elif any(word in user_input_lower for word in ['backup', 'restore', 'memory']):
                return 'memory_management'
            elif any(word in user_input_lower for word in ['send', 'distribute', 'upload']):
                return 'output_distribution'
            
            # Use pattern matching for more complex intents
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, user_input_lower):
                        return intent
            
            # Use NLP if available for more sophisticated analysis
            if self.nlp:
                return self._analyze_intent_nlp(user_input)
            
            # Default to general inquiry
            return 'general_inquiry'
            
        except Exception as e:
            logger.error(f"Failed to analyze intent: {e}")
            return 'general_inquiry'
    
    def _analyze_intent_nlp(self, user_input: str) -> str:
        """Use NLP to analyze intent more sophisticatedly."""
        try:
            doc = self.nlp(user_input)
            
            # Extract key entities and concepts
            entities = [ent.text.lower() for ent in doc.ents]
            tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
            
            # Check for security-related terms
            security_terms = ['threat', 'malware', 'attack', 'breach', 'vulnerability', 'policy', 'compliance']
            if any(term in tokens for term in security_terms):
                if any(term in tokens for term in ['hunt', 'find', 'detect']):
                    return 'threat_hunting'
                elif any(term in tokens for term in ['policy', 'framework', 'standard']):
                    return 'policy_analysis'
                elif any(term in tokens for term in ['incident', 'response', 'contain']):
                    return 'incident_response'
                elif any(term in tokens for term in ['vulnerability', 'scan', 'test']):
                    return 'vulnerability_assessment'
            
            # Check for data analysis terms
            data_terms = ['data', 'log', 'analyze', 'pattern', 'trend']
            if any(term in tokens for term in data_terms):
                return 'data_analysis'
            
            # Check for memory management terms
            memory_terms = ['backup', 'restore', 'memory', 'store', 'delete']
            if any(term in tokens for term in memory_terms):
                return 'memory_management'
            
            # Check for distribution terms
            distribution_terms = ['send', 'distribute', 'upload', 'export', 'share']
            if any(term in tokens for term in distribution_terms):
                return 'output_distribution'
            
            return 'general_inquiry'
            
        except Exception as e:
            logger.error(f"Failed to analyze intent with NLP: {e}")
            return 'general_inquiry'
    
    def _generate_context_response(self, intent: str, user_input: str) -> str:
        """Generate context-aware response based on intent."""
        try:
            if intent in self.response_templates:
                # Select appropriate template
                templates = self.response_templates[intent]
                
                # Use first template for now (could be enhanced with context selection)
                response = templates[0]
                
                # Personalize response based on user input
                response = self._personalize_response(response, user_input)
                
                return response
            else:
                return "I understand you're asking about cybersecurity. Let me help you with that. What specific aspect would you like to work on?"
                
        except Exception as e:
            logger.error(f"Failed to generate context response: {e}")
            return "I'm here to help with cybersecurity tasks. What would you like to work on?"
    
    def _personalize_response(self, response: str, user_input: str) -> str:
        """Personalize response based on user input."""
        try:
            # Extract specific details from user input
            user_input_lower = user_input.lower()
            
            # Add specific details if mentioned
            if 'splunk' in user_input_lower:
                response += " I can help you analyze Splunk data and discover what's feeding into your indexes."
            elif 'policy' in user_input_lower:
                response += " I can help you analyze security policies and map them to frameworks like NIST or ISO."
            elif 'threat' in user_input_lower:
                response += " I can help you hunt for threats and analyze security events."
            elif 'compliance' in user_input_lower:
                response += " I can help you assess compliance and identify gaps in your security posture."
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to personalize response: {e}")
            return response
    
    def _suggest_workflows(self, intent: str, user_input: str) -> List[str]:
        """Suggest relevant workflows based on intent."""
        try:
            suggestions = []
            
            if intent in self.workflow_suggestions:
                suggestions.extend(self.workflow_suggestions[intent])
            
            # Add specific workflow suggestions based on user input
            user_input_lower = user_input.lower()
            
            if 'splunk' in user_input_lower:
                suggestions.append("I can run a Splunk data discovery workflow to show you what's feeding into your indexes.")
                suggestions.append("Let me analyze your Splunk index performance and data flow patterns.")
            
            if 'policy' in user_input_lower:
                suggestions.append("I can analyze your security policies and map them to industry frameworks.")
                suggestions.append("Let me help you identify policy gaps and create improvement recommendations.")
            
            if 'threat' in user_input_lower:
                suggestions.append("I can set up a comprehensive threat hunting workflow with data analysis and IOC identification.")
                suggestions.append("Let me help you investigate potential threats and analyze security events.")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest workflows: {e}")
            return []
    
    def _generate_follow_up_questions(self, intent: str, user_input: str) -> List[str]:
        """Generate relevant follow-up questions."""
        try:
            questions = []
            
            if intent == 'threat_hunting':
                questions.extend([
                    "What type of threat are you concerned about?",
                    "Do you have specific IOCs or indicators to investigate?",
                    "What time period should we focus on?",
                    "Which data sources should we analyze?"
                ])
            elif intent == 'policy_analysis':
                questions.extend([
                    "What security policies do you want to analyze?",
                    "Which frameworks do you need to align with?",
                    "Are you looking for compliance gaps or improvement opportunities?",
                    "What's your current policy review process?"
                ])
            elif intent == 'incident_response':
                questions.extend([
                    "What type of incident are you dealing with?",
                    "What's the current status and severity?",
                    "Which teams need to be involved?",
                    "What evidence has been collected so far?"
                ])
            elif intent == 'data_analysis':
                questions.extend([
                    "What type of data do you want to analyze?",
                    "What patterns or insights are you looking for?",
                    "What time range should we focus on?",
                    "How would you like the results visualized?"
                ])
            elif intent == 'memory_management':
                questions.extend([
                    "What operation do you need (backup, restore, cleanup)?",
                    "What type of data do you want to manage?",
                    "Do you need to export or import information?",
                    "What's your backup retention policy?"
                ])
            elif intent == 'output_distribution':
                questions.extend([
                    "Where do you want to send your outputs?",
                    "What format do you need (JSON, CSV, CEF, etc.)?",
                    "Do you need real-time streaming or batch transfer?",
                    "What authentication or credentials are required?"
                ])
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []
    
    def _generate_suggested_actions(self, intent: str, user_input: str) -> List[Dict[str, Any]]:
        """Generate suggested actions based on intent."""
        try:
            actions = []
            
            if intent == 'threat_hunting':
                actions.extend([
                    {
                        'action': 'discover_data_sources',
                        'description': 'Discover what data is available in your Splunk indexes',
                        'parameters': {'time_range': '-4h'}
                    },
                    {
                        'action': 'analyze_logs',
                        'description': 'Analyze logs for suspicious patterns and anomalies',
                        'parameters': {'source': 'splunk', 'time_range': '-24h'}
                    }
                ])
            elif intent == 'policy_analysis':
                actions.extend([
                    {
                        'action': 'analyze_policies',
                        'description': 'Analyze your security policies for gaps and improvements',
                        'parameters': {'framework': 'nist'}
                    },
                    {
                        'action': 'map_frameworks',
                        'description': 'Map your policies to industry frameworks',
                        'parameters': {'source': 'local', 'target': 'nist'}
                    }
                ])
            elif intent == 'data_analysis':
                actions.extend([
                    {
                        'action': 'process_data',
                        'description': 'Process and analyze your data for insights',
                        'parameters': {'format': 'auto'}
                    },
                    {
                        'action': 'create_visualizations',
                        'description': 'Create visualizations of your data analysis results',
                        'parameters': {'type': 'auto'}
                    }
                ])
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to generate suggested actions: {e}")
            return []
    
    def _update_conversation_context(self, user_input: str):
        """Update conversation context with new information."""
        try:
            # Extract potential preferences or context from user input
            user_input_lower = user_input.lower()
            
            # Update user preferences
            if 'prefer' in user_input_lower or 'like' in user_input_lower:
                # Extract preference information
                if 'visual' in user_input_lower or 'chart' in user_input_lower:
                    self.conversation_context['user_preferences']['visualization'] = True
                if 'detailed' in user_input_lower or 'comprehensive' in user_input_lower:
                    self.conversation_context['user_preferences']['detail_level'] = 'high'
                if 'quick' in user_input_lower or 'fast' in user_input_lower:
                    self.conversation_context['user_preferences']['speed'] = 'fast'
            
            # Update current topic if it's a new subject
            if self.conversation_context['current_topic']:
                # Check if topic has changed significantly
                topic_similarity = SequenceMatcher(None, 
                                                 self.conversation_context['current_topic'], 
                                                 user_input).ratio()
                if topic_similarity < 0.3:  # Low similarity indicates topic change
                    self.conversation_context['current_topic'] = None
            
        except Exception as e:
            logger.error(f"Failed to update conversation context: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation context."""
        return {
            'current_topic': self.conversation_context['current_topic'],
            'recent_questions_count': len(self.conversation_context['recent_questions']),
            'user_preferences': self.conversation_context['user_preferences'],
            'workflow_history_count': len(self.conversation_context['workflow_history']),
            'suggested_actions_count': len(self.conversation_context['suggested_actions'])
        }
    
    def reset_conversation_context(self):
        """Reset conversation context."""
        self.conversation_context = {
            'current_topic': None,
            'recent_questions': [],
            'user_preferences': {},
            'workflow_history': [],
            'suggested_actions': []
        }
    
    def add_workflow_to_history(self, workflow_name: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Add workflow execution to history."""
        try:
            workflow_record = {
                'name': workflow_name,
                'parameters': parameters,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversation_context['workflow_history'].append(workflow_record)
            
            # Keep only last 20 workflows
            if len(self.conversation_context['workflow_history']) > 20:
                self.conversation_context['workflow_history'] = self.conversation_context['workflow_history'][-20:]
                
        except Exception as e:
            logger.error(f"Failed to add workflow to history: {e}")

# MCP Tools for Enhanced User Interaction
class EnhancedUserInteractionMCPTools:
    """MCP-compatible tools for enhanced user interaction."""
    
    def __init__(self, user_interaction: EnhancedUserInteraction):
        self.user_interaction = user_interaction
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for enhanced user interaction."""
        return {
            "process_user_input": {
                "name": "process_user_input",
                "description": "Process user input and generate intelligent response",
                "parameters": {
                    "user_input": {"type": "string", "description": "User's input text"}
                },
                "returns": {"type": "object", "description": "Processed response with intent, suggestions, and actions"}
            },
            "get_conversation_summary": {
                "name": "get_conversation_summary",
                "description": "Get summary of current conversation context",
                "parameters": {},
                "returns": {"type": "object", "description": "Conversation context summary"}
            },
            "reset_conversation_context": {
                "name": "reset_conversation_context",
                "description": "Reset conversation context",
                "parameters": {},
                "returns": {"type": "object", "description": "Reset confirmation"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute enhanced user interaction MCP tool."""
        if tool_name == "process_user_input":
            return self.user_interaction.process_user_input(**kwargs)
        elif tool_name == "get_conversation_summary":
            return self.user_interaction.get_conversation_summary()
        elif tool_name == "reset_conversation_context":
            self.user_interaction.reset_conversation_context()
            return {"success": True, "message": "Conversation context reset"}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

