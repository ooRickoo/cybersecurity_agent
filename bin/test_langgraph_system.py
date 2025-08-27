#!/usr/bin/env python3
"""
Test script for LangGraph Cybersecurity Agent
Verifies basic functionality without requiring full MCP setup.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test that basic imports work."""
    try:
        from bin.mcp_tools import (
            SessionManager, 
            FrameworkProcessor, 
            EncryptionManager,
            KnowledgeGraphManager
        )
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_session_manager():
    """Test session manager functionality."""
    try:
        from bin.mcp_tools import SessionManager
        
        session_mgr = SessionManager()
        session_id = session_mgr.create_session("test_session")
        
        print(f"✅ Session created: {session_id}")
        
        # Test output file saving
        test_content = "This is a test output file"
        output_path = session_mgr.save_output_file(
            session_id, 
            "test_output.txt", 
            test_content
        )
        
        print(f"✅ Output file saved: {output_path}")
        
        # Clean up test session
        import shutil
        test_output_dir = Path("session-outputs") / session_id
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Session manager test failed: {e}")
        return False

def test_framework_processor():
    """Test framework processor functionality."""
    try:
        from bin.mcp_tools import FrameworkProcessor
        
        processor = FrameworkProcessor()
        
        # Test CSV processing
        test_csv = """policy_id,policy_name,description
POL001,Password Policy,Strong password requirements
POL002,Access Control,Role-based access control"""
        
        result = processor.process_framework(test_csv, "csv")
        
        if result.get("processed") and result.get("total_items") == 2:
            print("✅ Framework processor CSV test successful")
            return True
        else:
            print(f"❌ Framework processor CSV test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Framework processor test failed: {e}")
        return False

def test_encryption_manager():
    """Test encryption manager functionality."""
    try:
        from bin.mcp_tools import EncryptionManager
        
        # Test without encryption enabled
        enc_mgr = EncryptionManager("test_password")
        
        test_data = b"This is test data to encrypt"
        
        # Test encryption (should return original data if crypto not available)
        encrypted = enc_mgr.encrypt_data(test_data)
        decrypted = enc_mgr.decrypt_data(encrypted)
        
        if decrypted == test_data:
            print("✅ Encryption manager test successful")
            return True
        else:
            print("❌ Encryption manager test failed: data mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Encryption manager test failed: {e}")
        return False

def test_knowledge_manager():
    """Test knowledge manager functionality."""
    try:
        from bin.mcp_tools import KnowledgeGraphManager, EncryptionManager
        
        enc_mgr = EncryptionManager()
        knowledge_mgr = KnowledgeGraphManager(enc_mgr)
        
        # Test adding framework
        test_framework = {
            "name": "Test Framework",
            "version": "1.0",
            "items": ["item1", "item2", "item3"]
        }
        
        framework_id = knowledge_mgr.add_framework("TestFramework", test_framework)
        
        if framework_id:
            print(f"✅ Framework added: {framework_id}")
            
            # Test querying
            results = knowledge_mgr.query_knowledge("Test Framework")
            
            if results.get("short_term"):
                print("✅ Knowledge query successful")
                return True
            else:
                print("❌ Knowledge query failed")
                return False
        else:
            print("❌ Framework addition failed")
            return False
            
    except Exception as e:
        print(f"❌ Knowledge manager test failed: {e}")
        return False

async def test_langgraph_agent():
    """Test LangGraph agent initialization."""
    try:
        from langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
        
        agent = LangGraphCybersecurityAgent()
        
        # Test basic initialization
        if hasattr(agent, 'knowledge_manager') and hasattr(agent, 'workflow_manager'):
            print("✅ LangGraph agent initialization successful")
            return True
        else:
            print("❌ LangGraph agent missing required components")
            return False
            
    except Exception as e:
        print(f"❌ LangGraph agent test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing LangGraph Cybersecurity Agent System")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Session Manager", test_session_manager),
        ("Framework Processor", test_framework_processor),
        ("Encryption Manager", test_encryption_manager),
        ("Knowledge Manager", test_knowledge_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the system:")
        print("   python cybersecurity_cli.py")
        print("\n📚 For more information, see README_LANGGRAPH_SYSTEM.md")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common issues:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check Python version (3.9+)")
        print("   - Verify file permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
