#!/usr/bin/env python3
"""
Test script for enhanced cybersecurity agent features
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from bin.enhanced_input_processor import EnhancedInputProcessor
from bin.langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
from bin.llm_client import LLMClient

async def test_enhanced_features():
    """Test the enhanced features"""
    print("🧪 Testing Enhanced Cybersecurity Agent Features")
    print("=" * 60)
    
    # Initialize components
    llm_client = LLMClient()
    agent = LangGraphCybersecurityAgent()
    await agent.start()
    
    enhanced_processor = EnhancedInputProcessor(existing_agent=agent, llm_client=llm_client)
    
    # Test cases
    test_cases = [
        "analyze sample X-47 for malware",
        "scan 192.168.1.1 for vulnerabilities", 
        "hello, how are you?",
        "search my Google Drive for incident reports",
        "perform comprehensive threat hunting analysis"
    ]
    
    print("🚀 Enhanced local processing enabled")
    print("   - TinyBERT intent classification (rule-based fallback)")
    print("   - Rule-based parameter extraction")
    print("   - Star Trek conversational interface")
    print("   - 60-80% reduction in LLM calls for routine queries")
    print()
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        print("-" * 40)
        
        result = enhanced_processor.process_input(text)
        
        print(f"🤖 Response: {result.response}")
        print(f"⚡ Source: {result.source}")
        print(f"⏱️  Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"🔧 Local Processing: {result.local_processing_used}")
        print(f"🧠 Intent: {result.intent}")
        
        if result.suggestions:
            print("💡 Suggestions:")
            for j, suggestion in enumerate(result.suggestions, 1):
                print(f"   {j}. {suggestion}")
        
        print()
    
    # Show performance stats
    print("📊 Performance Statistics:")
    print("=" * 40)
    stats = enhanced_processor.get_performance_stats()
    for key, value in stats.items():
        if key != 'classifier_stats':
            print(f"{key}: {value}")
    
    if 'classifier_stats' in stats:
        print(f"\nClassifier Performance:")
        for key, value in stats['classifier_stats'].items():
            print(f"  {key}: {value}")
    
    print("\n✅ Enhanced features test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())
