#!/usr/bin/env python3
"""
Comprehensive tests for the conversational enhancement system
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from bin.local_processing.tiny_bert_classifier import TinyBERTClassifier, IntentResult
from bin.local_processing.rule_based_extractor import CyberSecurityParameterExtractor, ExtractedParameters
from bin.conversational.star_trek_interface import StarTrekInterface
from bin.enhanced_input_processor import EnhancedInputProcessor, ProcessingResult

class TestTinyBERTClassifier(unittest.TestCase):
    """Test TinyBERT classifier functionality"""
    
    def setUp(self):
        self.classifier = TinyBERTClassifier()
    
    def test_classify_malware_analysis(self):
        """Test malware analysis intent classification"""
        result = self.classifier.classify_intent("analyze sample X-47 for malware")
        self.assertIsInstance(result, IntentResult)
        self.assertIn(result.intent, ['malware_analysis', 'clarification_needed'])
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.processing_time_ms, 1000)  # Should be fast
    
    def test_classify_vulnerability_scan(self):
        """Test vulnerability scanning intent classification"""
        result = self.classifier.classify_intent("scan 192.168.1.1 for vulnerabilities")
        self.assertIsInstance(result, IntentResult)
        self.assertIn(result.intent, ['vulnerability_scan', 'clarification_needed'])
        self.assertGreater(result.confidence, 0.0)
    
    def test_classify_casual_conversation(self):
        """Test casual conversation intent classification"""
        result = self.classifier.classify_intent("hello, how are you?")
        self.assertIsInstance(result, IntentResult)
        self.assertIn(result.intent, ['casual_conversation', 'clarification_needed'])
        self.assertGreater(result.confidence, 0.0)
    
    def test_can_handle_locally(self):
        """Test local handling capability"""
        result = self.classifier.classify_intent("analyze sample X-47 for malware")
        self.assertIsInstance(result.can_handle_locally, bool)
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        # Run a few classifications
        self.classifier.classify_intent("analyze sample X-47 for malware")
        self.classifier.classify_intent("scan 192.168.1.1 for vulnerabilities")
        
        stats = self.classifier.get_performance_stats()
        self.assertIn('total_classifications', stats)
        self.assertIn('avg_processing_time_ms', stats)
        self.assertGreaterEqual(stats['total_classifications'], 2)

class TestRuleBasedExtractor(unittest.TestCase):
    """Test rule-based parameter extractor functionality"""
    
    def setUp(self):
        self.extractor = CyberSecurityParameterExtractor()
    
    def test_extract_ip_addresses(self):
        """Test IP address extraction"""
        params = self.extractor.extract_all_parameters("scan 192.168.1.1 and 10.0.0.1 for vulnerabilities")
        self.assertIsInstance(params, ExtractedParameters)
        self.assertIn('ip_addresses', params.entities)
        self.assertIn('192.168.1.1', params.entities['ip_addresses'])
        self.assertIn('10.0.0.1', params.entities['ip_addresses'])
    
    def test_extract_domains(self):
        """Test domain extraction"""
        params = self.extractor.extract_all_parameters("analyze example.com and test.org for threats")
        self.assertIsInstance(params, ExtractedParameters)
        self.assertIn('domains', params.entities)
        self.assertIn('example.com', params.entities['domains'])
        self.assertIn('test.org', params.entities['domains'])
    
    def test_extract_file_paths(self):
        """Test file path extraction"""
        params = self.extractor.extract_all_parameters("check /tmp/suspicious.exe and C:\\temp\\malware.bin")
        self.assertIsInstance(params, ExtractedParameters)
        self.assertIn('file_paths', params.entities)
        self.assertIn('/tmp/suspicious.exe', params.entities['file_paths'])
        self.assertIn('C:\\temp\\malware.bin', params.entities['file_paths'])
    
    def test_detect_analysis_type(self):
        """Test analysis type detection"""
        params = self.extractor.extract_all_parameters("analyze sample X-47 for malware")
        self.assertEqual(params.analysis_type, 'malware')
        
        params = self.extractor.extract_all_parameters("scan 192.168.1.1 for vulnerabilities")
        self.assertEqual(params.analysis_type, 'vulnerability')
    
    def test_detect_priority(self):
        """Test priority detection"""
        params = self.extractor.extract_all_parameters("URGENT: analyze this critical malware sample")
        self.assertEqual(params.priority, 'critical')
        
        params = self.extractor.extract_all_parameters("scan this when possible")
        self.assertEqual(params.priority, 'low')
    
    def test_extract_targets(self):
        """Test target extraction"""
        params = self.extractor.extract_all_parameters("analyze 192.168.1.1 and example.com for threats")
        self.assertIsInstance(params.targets, list)
        self.assertIn('192.168.1.1', params.targets)
        self.assertIn('example.com', params.targets)
    
    def test_validate_parameters(self):
        """Test parameter validation"""
        params = self.extractor.extract_all_parameters("analyze 192.168.1.1 for malware")
        is_valid, issues = self.extractor.validate_extracted_params(params)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(issues, list)

class TestStarTrekInterface(unittest.TestCase):
    """Test Star Trek conversational interface functionality"""
    
    def setUp(self):
        self.interface = StarTrekInterface()
    
    def test_process_command_malware_analysis(self):
        """Test malware analysis command processing"""
        response = self.interface.process_command(
            "analyze sample X-47 for malware",
            "malware_analysis",
            {"targets": ["X-47"], "priority": "high"}
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_process_command_casual_conversation(self):
        """Test casual conversation processing"""
        response = self.interface.process_command(
            "hello, how are you?",
            "casual_conversation",
            {}
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_generate_acknowledgment(self):
        """Test acknowledgment generation"""
        acknowledgment = self.interface.generate_acknowledgment(
            "malware_analysis",
            {"targets": ["X-47"], "priority": "high"}
        )
        self.assertIsInstance(acknowledgment, str)
        self.assertGreater(len(acknowledgment), 0)
    
    def test_create_progress_feedback(self):
        """Test progress feedback creation"""
        feedback = self.interface.create_progress_feedback(
            "malware_analysis",
            "processing",
            30
        )
        self.assertIsInstance(feedback, str)
        self.assertGreater(len(feedback), 0)
    
    def test_suggest_follow_up_actions(self):
        """Test follow-up action suggestions"""
        results = {"threats_found": 3, "files_processed": 15}
        suggestions = self.interface.suggest_follow_up_actions(results, "malware_analysis")
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_handle_casual_conversation(self):
        """Test casual conversation handling"""
        response = self.interface.handle_casual_conversation("hello there")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_get_session_summary(self):
        """Test session summary generation"""
        summary = self.interface.get_session_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('session_duration', summary)
        self.assertIn('total_interactions', summary)

class TestEnhancedInputProcessor(unittest.TestCase):
    """Test enhanced input processor functionality"""
    
    def setUp(self):
        self.processor = EnhancedInputProcessor()
    
    def test_process_input_malware_analysis(self):
        """Test malware analysis input processing"""
        result = self.processor.process_input("analyze sample X-47 for malware")
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.response, str)
        self.assertIsInstance(result.source, str)
        self.assertIsInstance(result.processing_time_ms, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.suggestions, list)
    
    def test_process_input_casual_conversation(self):
        """Test casual conversation input processing"""
        result = self.processor.process_input("hello, how are you?")
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.response, str)
        self.assertGreater(len(result.response), 0)
    
    def test_can_handle_with_local_tools(self):
        """Test local tool handling capability"""
        # Create mock intent result and parameters
        intent_result = type('IntentResult', (), {
            'confidence': 0.95,
            'intent': 'malware_analysis',
            'can_handle_locally': True
        })()
        
        params = type('ExtractedParameters', (), {
            'targets': ['X-47']
        })()
        
        can_handle = self.processor.can_handle_with_local_tools(intent_result, params)
        self.assertIsInstance(can_handle, bool)
    
    def test_requires_dynamic_workflow(self):
        """Test dynamic workflow requirement detection"""
        intent_result = type('IntentResult', (), {
            'intent': 'network_analysis',
            'confidence': 0.8
        })()
        
        params = type('ExtractedParameters', (), {
            'targets': ['192.168.1.1', '192.168.1.2']
        })()
        
        requires_dynamic = self.processor.requires_dynamic_workflow(intent_result, params)
        self.assertIsInstance(requires_dynamic, bool)
    
    def test_requires_mcp_tools(self):
        """Test MCP tool requirement detection"""
        intent_result = type('IntentResult', (), {
            'original_text': 'search my Google Drive for incident reports'
        })()
        
        params = type('ExtractedParameters', (), {})()
        
        requires_mcp = self.processor.requires_mcp_tools(intent_result, params)
        self.assertIsInstance(requires_mcp, bool)
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval"""
        # Process a few inputs to generate stats
        self.processor.process_input("analyze sample X-47 for malware")
        self.processor.process_input("scan 192.168.1.1 for vulnerabilities")
        
        stats = self.processor.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_requests', stats)
        self.assertIn('avg_processing_time_ms', stats)
        self.assertGreaterEqual(stats['total_requests'], 2)

class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        self.processor = EnhancedInputProcessor()
    
    def test_end_to_end_malware_analysis(self):
        """Test end-to-end malware analysis workflow"""
        result = self.processor.process_input("analyze sample X-47 for malware")
        
        # Verify result structure
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.response, str)
        self.assertGreater(len(result.response), 0)
        
        # Verify processing time is reasonable
        self.assertLess(result.processing_time_ms, 5000)  # Should be under 5 seconds
        
        # Verify confidence is reasonable
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_end_to_end_casual_conversation(self):
        """Test end-to-end casual conversation workflow"""
        result = self.processor.process_input("hello, how are you?")
        
        # Verify result structure
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.response, str)
        self.assertGreater(len(result.response), 0)
        
        # Verify it's handled as casual conversation
        self.assertIn(result.intent, ['casual_conversation', 'clarification_needed'])
    
    def test_performance_benchmark(self):
        """Test performance benchmark"""
        test_cases = [
            "analyze sample X-47 for malware",
            "scan 192.168.1.1 for vulnerabilities",
            "hello, how are you?",
            "perform comprehensive threat hunting analysis",
            "check this file for forensic evidence"
        ]
        
        total_time = 0
        for text in test_cases:
            result = self.processor.process_input(text)
            total_time += result.processing_time_ms
        
        avg_time = total_time / len(test_cases)
        
        # Verify average processing time is reasonable
        self.assertLess(avg_time, 1000)  # Should be under 1 second on average
        
        # Verify all results are valid
        for text in test_cases:
            result = self.processor.process_input(text)
            self.assertIsInstance(result, ProcessingResult)
            self.assertIsInstance(result.response, str)
            self.assertGreater(len(result.response), 0)

def run_performance_tests():
    """Run performance tests"""
    print("🧪 Running Performance Tests")
    print("=" * 50)
    
    processor = EnhancedInputProcessor()
    
    test_cases = [
        "analyze sample X-47 for malware",
        "scan 192.168.1.1 for vulnerabilities",
        "hello, how are you?",
        "perform comprehensive threat hunting analysis",
        "check this file for forensic evidence"
    ] * 10  # Run 50 tests
    
    import time
    start_time = time.time()
    
    for text in test_cases:
        result = processor.process_input(text)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = (total_time / len(test_cases)) * 1000
    
    print(f"Total tests: {len(test_cases)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time: {avg_time:.1f} ms per classification")
    
    # Show stats
    stats = processor.get_performance_stats()
    print(f"\nPerformance Stats:")
    for key, value in stats.items():
        if key != 'classifier_stats':
            print(f"  {key}: {value}")
    
    return avg_time < 1000  # Should be under 1 second on average

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "=" * 50)
    performance_passed = run_performance_tests()
    
    if performance_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
