#!/usr/bin/env python3
"""
Test Workflow Verification Integration
Simple test to verify the verification system works correctly
"""

import sys
from pathlib import Path

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

def test_verification_system():
    """Test the verification system independently."""
    print("🧪 **Testing Workflow Verification System**\n")
    
    try:
        from workflow_verification_system import get_workflow_verifier
        verifier = get_workflow_verifier()
        print("✅ Verification system imported successfully")
    except ImportError as e:
        print(f"❌ Verification system import failed: {e}")
        return False
    
    try:
        from workflow_template_manager import get_workflow_template_manager
        template_manager = get_workflow_template_manager()
        print("✅ Template manager imported successfully")
    except ImportError as e:
        print(f"❌ Template manager import failed: {e}")
        return False
    
    try:
        from workflow_verification_mcp_tools import get_workflow_verification_mcp_tools
        mcp_tools = get_workflow_verification_mcp_tools()
        print("✅ MCP tools imported successfully")
    except ImportError as e:
        print(f"❌ MCP tools import failed: {e}")
        return False
    
    print("\n✅ **All verification components imported successfully!**")
    return True

def test_agent_state_integration():
    """Test that the agent state can be imported and used."""
    print("\n🧪 **Testing Agent State Integration**\n")
    
    try:
        # Try to import the agent state class
        sys.path.append(str(Path(__file__).parent.parent))
        from langgraph_cybersecurity_agent import AgentState
        
        # Test creating an agent state with verification fields
        state = AgentState(
            messages=[{"role": "user", "content": "Test question"}],
            workflow_steps=[],
            verification_required=False
        )
        
        print("✅ AgentState imported and created successfully")
        print(f"   Verification fields: {hasattr(state, 'verification_required')}")
        print(f"   Workflow steps: {hasattr(state, 'workflow_steps')}")
        print(f"   Execution ID: {hasattr(state, 'execution_id')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ AgentState import failed: {e}")
        print("   This is expected if the agent file has syntax errors")
        return False
    except Exception as e:
        print(f"❌ AgentState creation failed: {e}")
        return False

def test_verification_workflow():
    """Test a simple verification workflow."""
    print("\n🧪 **Testing Verification Workflow**\n")
    
    try:
        from workflow_verification_mcp_tools import get_workflow_verification_mcp_tools
        tools = get_workflow_verification_mcp_tools()
        
        # Test template selection
        template_result = tools.select_workflow_template("What is the current threat landscape?")
        
        if template_result.get("success"):
            print("✅ Template selection successful")
            template = template_result["template"]
            print(f"   Selected: {template['name']}")
            print(f"   Type: {template['template_type']}")
            print(f"   Steps: {len(template['steps'])}")
        else:
            print(f"❌ Template selection failed: {template_result.get('error')}")
            return False
        
        # Test getting available tools
        available_tools = tools.get_tools()
        print(f"\n✅ Available MCP tools: {len(available_tools)}")
        
        tool_names = [tool['function']['name'] for tool in available_tools]
        print(f"   Tools: {', '.join(tool_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification workflow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 **Workflow Verification Integration Tests**\n")
    
    tests = [
        ("Verification System", test_verification_system),
        ("Agent State Integration", test_agent_state_integration),
        ("Verification Workflow", test_verification_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 **Test Results Summary**")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("=" * 40)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 **All tests passed! The verification system is ready for use.**")
        print("\nNext steps:")
        print("1. ✅ Verification system components are working")
        print("2. ✅ Agent state integration is ready")
        print("3. ✅ MCP tools are available")
        print("4. 🔄 Test with your actual agent workflow")
    else:
        print(f"\n⚠️ **{total - passed} tests failed. Please check the errors above.**")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
