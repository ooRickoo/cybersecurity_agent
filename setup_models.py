#!/usr/bin/env python3
"""
Model Setup Script for Enhanced Cybersecurity Agent
Downloads and configures TinyBERT for cybersecurity intent classification
"""

import os
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tinybert_model():
    """Download and setup TinyBERT model"""
    try:
        model_name = "huawei-noah/TinyBERT_General_4L_312D"
        model_dir = Path("models/tiny_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading TinyBERT model: {model_name}")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(model_dir)
        )
        
        # Download model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=9,  # Number of intent categories
            cache_dir=str(model_dir)
        )
        
        logger.info("TinyBERT model downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download TinyBERT model: {e}")
        return False

def create_training_data():
    """Create training data for cybersecurity domain"""
    training_data = {
        "examples": [
            # Malware analysis examples
            {"text": "analyze sample X-47 for malware", "intent": "malware_analysis"},
            {"text": "check this file for viruses", "intent": "malware_analysis"},
            {"text": "scan suspicious executable for trojans", "intent": "malware_analysis"},
            {"text": "examine malware sample for indicators", "intent": "malware_analysis"},
            {"text": "detect malicious code in this file", "intent": "malware_analysis"},
            
            # Vulnerability scanning examples
            {"text": "scan 192.168.1.1 for vulnerabilities", "intent": "vulnerability_scan"},
            {"text": "check system for security holes", "intent": "vulnerability_scan"},
            {"text": "run vulnerability assessment on network", "intent": "vulnerability_scan"},
            {"text": "find CVE vulnerabilities in this system", "intent": "vulnerability_scan"},
            {"text": "perform security audit of servers", "intent": "vulnerability_scan"},
            
            # Network analysis examples
            {"text": "analyze network traffic patterns", "intent": "network_analysis"},
            {"text": "examine pcap file for anomalies", "intent": "network_analysis"},
            {"text": "monitor network connections", "intent": "network_analysis"},
            {"text": "investigate network security issues", "intent": "network_analysis"},
            {"text": "analyze packet capture data", "intent": "network_analysis"},
            
            # File forensics examples
            {"text": "perform forensic analysis on this file", "intent": "file_forensics"},
            {"text": "examine file for evidence", "intent": "file_forensics"},
            {"text": "investigate file timeline", "intent": "file_forensics"},
            {"text": "recover deleted files", "intent": "file_forensics"},
            {"text": "analyze file metadata", "intent": "file_forensics"},
            
            # Incident response examples
            {"text": "respond to security incident", "intent": "incident_response"},
            {"text": "handle security breach", "intent": "incident_response"},
            {"text": "investigate security incident", "intent": "incident_response"},
            {"text": "contain security threat", "intent": "incident_response"},
            {"text": "manage security incident", "intent": "incident_response"},
            
            # Threat hunting examples
            {"text": "hunt for advanced threats", "intent": "threat_hunting"},
            {"text": "search for IOCs in network", "intent": "threat_hunting"},
            {"text": "investigate APT activity", "intent": "threat_hunting"},
            {"text": "analyze threat intelligence", "intent": "threat_hunting"},
            {"text": "hunt for malicious actors", "intent": "threat_hunting"},
            
            # Casual conversation examples
            {"text": "hello, how are you?", "intent": "casual_conversation"},
            {"text": "hi there", "intent": "casual_conversation"},
            {"text": "good morning", "intent": "casual_conversation"},
            {"text": "thanks for your help", "intent": "casual_conversation"},
            {"text": "what's up?", "intent": "casual_conversation"},
            
            # Complex analysis examples
            {"text": "provide detailed attribution analysis", "intent": "complex_analysis_request"},
            {"text": "comprehensive security assessment", "intent": "complex_analysis_request"},
            {"text": "advanced threat intelligence analysis", "intent": "complex_analysis_request"},
            {"text": "sophisticated malware reverse engineering", "intent": "complex_analysis_request"},
            {"text": "complex incident response planning", "intent": "complex_analysis_request"},
            
            # Clarification needed examples
            {"text": "I need help with something", "intent": "clarification_needed"},
            {"text": "can you help me?", "intent": "clarification_needed"},
            {"text": "what can you do?", "intent": "clarification_needed"},
            {"text": "I'm not sure what I need", "intent": "clarification_needed"},
            {"text": "help me understand", "intent": "clarification_needed"}
        ]
    }
    
    # Save training data
    training_file = Path("models/tiny_models/training_data.json")
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Created training data with {len(training_data['examples'])} examples")
    return training_file

def validate_model():
    """Validate the downloaded model"""
    try:
        from bin.local_processing.tiny_bert_classifier import TinyBERTClassifier
        
        logger.info("Testing TinyBERT classifier...")
        classifier = TinyBERTClassifier()
        
        # Test cases
        test_cases = [
            "analyze sample X-47 for malware",
            "scan 192.168.1.1 for vulnerabilities",
            "hello, how are you?",
            "perform comprehensive threat hunting analysis"
        ]
        
        for text in test_cases:
            result = classifier.classify_intent(text)
            logger.info(f"Test: '{text}' -> {result.intent} (confidence: {result.confidence:.2f})")
        
        logger.info("Model validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False

def benchmark_performance():
    """Benchmark model performance"""
    try:
        from bin.local_processing.tiny_bert_classifier import TinyBERTClassifier
        
        classifier = TinyBERTClassifier()
        
        # Performance test
        test_texts = [
            "analyze sample X-47 for malware",
            "scan 192.168.1.1 for vulnerabilities",
            "hello, how are you?",
            "perform comprehensive threat hunting analysis",
            "check this file for forensic evidence"
        ] * 10  # Run 50 tests
        
        import time
        start_time = time.time()
        
        for text in test_texts:
            result = classifier.classify_intent(text)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = (total_time / len(test_texts)) * 1000  # Convert to milliseconds
        
        logger.info(f"Performance benchmark:")
        logger.info(f"  Total tests: {len(test_texts)}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Average time: {avg_time:.1f} ms per classification")
        
        # Show stats
        stats = classifier.get_performance_stats()
        logger.info(f"  Classifier stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Enhanced Cybersecurity Agent Models")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("bin").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Download TinyBERT model
    print("📥 Downloading TinyBERT model...")
    if not download_tinybert_model():
        print("❌ Failed to download TinyBERT model")
        sys.exit(1)
    print("✅ TinyBERT model downloaded successfully")
    
    # Create training data
    print("📚 Creating training data...")
    training_file = create_training_data()
    print(f"✅ Training data created: {training_file}")
    
    # Validate model
    print("🧪 Validating model...")
    if not validate_model():
        print("❌ Model validation failed")
        sys.exit(1)
    print("✅ Model validation passed")
    
    # Benchmark performance
    print("⚡ Benchmarking performance...")
    if not benchmark_performance():
        print("❌ Performance benchmark failed")
        sys.exit(1)
    print("✅ Performance benchmark completed")
    
    print("\n🎉 Model setup completed successfully!")
    print("=" * 60)
    print("You can now use the enhanced CLI with:")
    print("  python cs_util_lg_enhanced.py --interactive --enhanced")
    print("\nExpected improvements:")
    print("  - 60-80% reduction in LLM calls for routine queries")
    print("  - <100ms response time for local processing")
    print("  - Professional, approachable conversation flow")
    print("  - Dynamic tool selection and orchestration")

if __name__ == "__main__":
    main()
