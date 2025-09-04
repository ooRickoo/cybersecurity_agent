#!/usr/bin/env python3
"""
TinyBERT Classifier for Cybersecurity Intent Classification
Lightweight 14MB model for local intent classification
"""

import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    confidence: float
    processing_time_ms: float
    can_handle_locally: bool
    original_text: str
    reasoning: str = ""

class TinyBERTClassifier:
    """14MB intent classifier for cybersecurity commands"""
    
    def __init__(self, model_path: str = "models/tiny_models", llm_client=None):
        self.model_path = Path(model_path)
        self.llm_client = llm_client
        
        # Intent labels for cybersecurity domain
        self.intent_labels = {
            0: "casual_conversation",
            1: "malware_analysis", 
            2: "vulnerability_scan",
            3: "network_analysis",
            4: "file_forensics",
            5: "incident_response",
            6: "threat_hunting",
            7: "complex_analysis_request",
            8: "clarification_needed"
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        # Performance tracking
        self.stats = {
            'total_classifications': 0,
            'local_handles': 0,
            'llm_handoffs': 0,
            'avg_processing_time_ms': 0.0,
            'accuracy_tracking': []
        }
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        # Load model
        self._load_model()
    
    def _get_device(self) -> str:
        """Get optimal device for inference"""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load TinyBERT model and tokenizer"""
        try:
            # For now, skip the actual model loading to avoid TensorFlow issues
            # This will use rule-based classification as fallback
            logger.info("Using rule-based classification (TinyBERT model loading disabled for compatibility)")
            self.model = None
            self.tokenizer = None
            
        except Exception as e:
            logger.error(f"Failed to load TinyBERT model: {e}")
            # Fallback to rule-based classification
            self.model = None
            self.tokenizer = None
    
    def classify_intent(self, text: str) -> IntentResult:
        """
        Main classification method
        Returns: IntentResult with classification details
        """
        start_time = time.time()
        original_text = text.strip()
        
        try:
            if self.model is None:
                # Fallback to rule-based classification
                return self._rule_based_classification(original_text, start_time)
            
            # Preprocess text
            processed_text = self._preprocess_text(original_text)
            
            # Tokenize and encode
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get intent label
            intent = self.intent_labels.get(predicted_class, "clarification_needed")
            
            # Calibrate confidence based on cybersecurity patterns
            calibrated_confidence = self._calibrate_confidence(original_text, confidence, intent)
            
            # Determine if can handle locally
            can_handle_locally = self.can_handle_locally(intent, calibrated_confidence)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self._update_stats(processing_time, can_handle_locally)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(intent, calibrated_confidence, original_text)
            
            return IntentResult(
                intent=intent,
                confidence=calibrated_confidence,
                processing_time_ms=processing_time,
                can_handle_locally=can_handle_locally,
                original_text=original_text,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._rule_based_classification(original_text, start_time)
    
    def _rule_based_classification(self, text: str, start_time: float) -> IntentResult:
        """Fallback rule-based classification when model is unavailable"""
        text_lower = text.lower()
        
        # Cybersecurity patterns
        patterns = {
            'malware_analysis': [
                r'analyze.*malware', r'check.*virus', r'scan.*trojan',
                r'malware.*sample', r'suspicious.*file', r'virus.*detection'
            ],
            'vulnerability_scan': [
                r'scan.*vulnerabilit', r'check.*vuln', r'security.*scan',
                r'penetration.*test', r'vuln.*assessment', r'security.*audit'
            ],
            'network_analysis': [
                r'analyze.*network', r'network.*traffic', r'pcap.*analysis',
                r'packet.*capture', r'network.*scan', r'traffic.*analysis'
            ],
            'file_forensics': [
                r'forensic.*analysis', r'file.*investigation', r'digital.*forensics',
                r'evidence.*analysis', r'timeline.*analysis', r'file.*recovery'
            ],
            'incident_response': [
                r'incident.*response', r'security.*incident', r'breach.*analysis',
                r'incident.*investigation', r'security.*breach', r'incident.*handling'
            ],
            'threat_hunting': [
                r'threat.*hunting', r'hunt.*threats', r'ioc.*analysis',
                r'threat.*intelligence', r'apt.*analysis', r'threat.*detection'
            ],
            'casual_conversation': [
                r'hello', r'hi', r'how are you', r'what.*up', r'good morning',
                r'good afternoon', r'good evening', r'hey', r'thanks'
            ]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, pattern_list in patterns.items():
            score = 0
            for pattern in pattern_list:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                intent = best_intent[0]
                confidence = min(0.8, best_intent[1] * 0.2)  # Rule-based confidence
            else:
                intent = "clarification_needed"
                confidence = 0.3
        else:
            intent = "clarification_needed"
            confidence = 0.3
        
        # Determine if complex analysis needed
        if any(word in text_lower for word in ['complex', 'detailed', 'comprehensive', 'advanced', 'sophisticated']):
            intent = "complex_analysis_request"
            confidence = 0.7
        
        can_handle_locally = self.can_handle_locally(intent, confidence)
        processing_time = (time.time() - start_time) * 1000
        
        self._update_stats(processing_time, can_handle_locally)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            processing_time_ms=processing_time,
            can_handle_locally=can_handle_locally,
            original_text=text,
            reasoning=f"Rule-based classification: {intent} (score: {intent_scores.get(intent, 0)})"
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize cybersecurity terms
        cyber_terms = {
            'malware': 'malware',
            'virus': 'malware',
            'trojan': 'malware',
            'vuln': 'vulnerability',
            'vulns': 'vulnerability',
            'pcap': 'network packet capture',
            'ioc': 'indicator of compromise',
            'apt': 'advanced persistent threat'
        }
        
        text_lower = text.lower()
        for term, normalized in cyber_terms.items():
            text_lower = text_lower.replace(term, normalized)
        
        return text_lower
    
    def _calibrate_confidence(self, text: str, confidence: float, intent: str) -> float:
        """Adjust confidence based on cybersecurity domain patterns"""
        # Boost confidence for clear cybersecurity patterns
        cyber_patterns = [
            r'analyze\s+\w+', r'scan\s+\w+', r'check\s+\w+',
            r'malware', r'vulnerability', r'forensic', r'network',
            r'threat', r'incident', r'security'
        ]
        
        pattern_matches = sum(1 for pattern in cyber_patterns if re.search(pattern, text.lower()))
        
        # Boost confidence for cybersecurity terms
        if pattern_matches > 0:
            confidence = min(0.95, confidence + (pattern_matches * 0.05))
        
        # Boost confidence for specific intent patterns
        if intent in ['malware_analysis', 'vulnerability_scan', 'network_analysis']:
            if any(word in text.lower() for word in ['analyze', 'scan', 'check', 'investigate']):
                confidence = min(0.95, confidence + 0.1)
        
        # Reduce confidence for casual conversation
        if intent == 'casual_conversation':
            if any(word in text.lower() for word in ['malware', 'security', 'vulnerability', 'threat']):
                confidence = max(0.3, confidence - 0.2)
        
        return confidence
    
    def can_handle_locally(self, intent: str, confidence: float) -> bool:
        """Determine if intent can be handled with local tools"""
        # High confidence local intents
        local_intents = [
            'malware_analysis', 'vulnerability_scan', 'file_forensics',
            'network_analysis', 'threat_hunting'
        ]
        
        if intent in local_intents and confidence >= self.confidence_thresholds['high']:
            return True
        
        # Medium confidence for simple operations
        if intent in ['casual_conversation', 'clarification_needed'] and confidence >= self.confidence_thresholds['medium']:
            return True
        
        # Complex analysis always goes to LLM
        if intent == 'complex_analysis_request':
            return False
        
        return False
    
    def _update_stats(self, processing_time: float, handled_locally: bool) -> None:
        """Update performance statistics"""
        self.stats['total_classifications'] += 1
        
        if handled_locally:
            self.stats['local_handles'] += 1
        else:
            self.stats['llm_handoffs'] += 1
        
        # Update average processing time
        total_time = self.stats['avg_processing_time_ms'] * (self.stats['total_classifications'] - 1)
        self.stats['avg_processing_time_ms'] = (total_time + processing_time) / self.stats['total_classifications']
    
    def _generate_reasoning(self, intent: str, confidence: float, text: str) -> str:
        """Generate reasoning for classification decision"""
        reasoning_parts = []
        
        # Intent reasoning
        reasoning_parts.append(f"Classified as '{intent}' with {confidence:.2f} confidence")
        
        # Confidence level
        if confidence >= self.confidence_thresholds['high']:
            reasoning_parts.append("High confidence - clear cybersecurity intent")
        elif confidence >= self.confidence_thresholds['medium']:
            reasoning_parts.append("Medium confidence - some uncertainty")
        else:
            reasoning_parts.append("Low confidence - may need clarification")
        
        # Local handling capability
        if self.can_handle_locally(intent, confidence):
            reasoning_parts.append("Can be handled with local tools")
        else:
            reasoning_parts.append("Requires external LLM processing")
        
        return ". ".join(reasoning_parts)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        if stats['total_classifications'] > 0:
            stats['local_handle_rate'] = stats['local_handles'] / stats['total_classifications']
            stats['llm_handoff_rate'] = stats['llm_handoffs'] / stats['total_classifications']
        else:
            stats['local_handle_rate'] = 0.0
            stats['llm_handoff_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.stats = {
            'total_classifications': 0,
            'local_handles': 0,
            'llm_handoffs': 0,
            'avg_processing_time_ms': 0.0,
            'accuracy_tracking': []
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the classifier
    classifier = TinyBERTClassifier()
    
    test_cases = [
        "analyze sample X-47 for malware",
        "scan 192.168.1.1 for vulnerabilities",
        "hello, how are you?",
        "perform comprehensive threat hunting analysis",
        "check this file for forensic evidence"
    ]
    
    print("🧪 Testing TinyBERT Classifier")
    print("=" * 50)
    
    for text in test_cases:
        result = classifier.classify_intent(text)
        print(f"\nInput: {text}")
        print(f"Intent: {result.intent}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Local: {result.can_handle_locally}")
        print(f"Time: {result.processing_time_ms:.1f}ms")
        print(f"Reasoning: {result.reasoning}")
    
    # Show performance stats
    stats = classifier.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"Total classifications: {stats['total_classifications']}")
    print(f"Local handle rate: {stats['local_handle_rate']:.1%}")
    print(f"Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
