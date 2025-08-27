#!/usr/bin/env python3
"""
Test Session Viewer Integration
Verifies that the session viewer can be launched and managed from the cybersecurity agent
"""

import sys
import time
from pathlib import Path

# Add the bin directory to the path
bin_path = Path(__file__).parent / "bin"
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

def test_session_viewer_manager():
    """Test the session viewer manager directly."""
    print("🧪 Testing Session Viewer Manager...")
    print("=" * 50)
    
    try:
        from session_viewer_manager import get_session_viewer_manager
        
        manager = get_session_viewer_manager()
        
        # Test status
        print("\n📊 Getting initial status...")
        status = manager.get_status()
        print(f"Status: {status}")
        
        # Test launch
        print("\n🚀 Launching session viewer...")
        result = manager.start_viewer(auto_open=False)  # Don't auto-open browser for testing
        print(f"Launch result: {result}")
        
        if result['success']:
            print("✅ Session viewer launched successfully!")
            
            # Test status again
            print("\n📊 Getting status after launch...")
            status = manager.get_status()
            print(f"Status: {status}")
            
            # Test extend session
            print("\n⏰ Extending session...")
            extend_result = manager.extend_session()
            print(f"Extend result: {extend_result}")
            
            # Wait a moment
            print("\n⏳ Waiting 5 seconds...")
            time.sleep(5)
            
            # Test stop
            print("\n🛑 Stopping session viewer...")
            stop_result = manager.stop_viewer()
            print(f"Stop result: {stop_result}")
            
            # Final status
            print("\n📊 Final status...")
            status = manager.get_status()
            print(f"Status: {status}")
            
        else:
            print(f"❌ Failed to launch: {result['error']}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_mcp_tools():
    """Test the MCP tools integration."""
    print("\n🧪 Testing Session Viewer MCP Tools...")
    print("=" * 50)
    
    try:
        from session_viewer_mcp_tools import get_session_viewer_tools
        
        tools = get_session_viewer_tools()
        
        # Test getting tools
        print("\n🔧 Getting available tools...")
        available_tools = tools.get_tools()
        print(f"Available tools: {len(available_tools)}")
        for tool in available_tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")
        
        # Test status
        print("\n📊 Testing get_session_viewer_status...")
        status_result = tools.get_session_viewer_status()
        print(f"Status result: {status_result}")
        
        # Test launch
        print("\n🚀 Testing launch_session_viewer...")
        launch_result = tools.launch_session_viewer(reason="Testing MCP tools")
        print(f"Launch result: {launch_result}")
        
        if launch_result['success']:
            print("✅ Launch successful!")
            print(f"CLI Output:\n{launch_result['cli_output']}")
            
            # Test extend
            print("\n⏰ Testing extend_session_viewer...")
            extend_result = tools.extend_session_viewer(reason="Testing extend")
            print(f"Extend result: {extend_result}")
            
            # Test stop
            print("\n🛑 Testing stop_session_viewer...")
            stop_result = tools.stop_session_viewer(reason="Testing stop")
            print(f"Stop result: {stop_result}")
            
        else:
            print(f"❌ Launch failed: {launch_result['error']}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_agent_integration():
    """Test integration with the main agent."""
    print("\n🧪 Testing Agent Integration...")
    print("=" * 50)
    
    try:
        # Import the main agent
        sys.path.insert(0, str(Path(__file__).parent))
        from langgraph_cybersecurity_agent import LangGraphCybersecurityAgent
        
        print("✅ Successfully imported LangGraphCybersecurityAgent")
        
        # Check if session viewer manager is available
        if hasattr(LangGraphCybersecurityAgent, 'session_viewer_manager'):
            print("✅ Session viewer manager attribute found in agent")
        else:
            print("❌ Session viewer manager attribute not found in agent")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Session Viewer Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Direct manager
    test_session_viewer_manager()
    
    # Test 2: MCP tools
    test_mcp_tools()
    
    # Test 3: Agent integration
    test_agent_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")
    print("\n💡 Next steps:")
    print("   1. Install Node.js dependencies: cd session-viewer && npm install")
    print("   2. Build React client: npm run build-client")
    print("   3. Test the viewer: python bin/session_viewer_manager.py")
    print("   4. Integrate with workflows using MCP tools")

if __name__ == "__main__":
    main()
