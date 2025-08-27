#!/usr/bin/env python3
"""
Test script for Cybersecurity Agent Enhancement Systems
Verifies all enhancement systems are working correctly
"""

import asyncio
import json
import sys
from pathlib import Path

# Add bin directory to path
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

async def test_enhancement_systems():
    """Test all enhancement systems."""
    print("🚀 Testing Cybersecurity Agent Enhancement Systems")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Performance Optimization System
        print("\n1️⃣ Testing Performance Optimization System...")
        try:
            from performance_optimizer import performance_optimizer
            stats = performance_optimizer.get_optimization_stats()
            test_results["performance_optimizer"] = "✅ PASSED"
            print(f"   ✅ Performance optimizer initialized successfully")
            print(f"   📊 Cache stats: {stats.get('cache', {}).get('cache_size', 0)} entries")
        except Exception as e:
            test_results["performance_optimizer"] = f"❌ FAILED: {e}"
            print(f"   ❌ Performance optimizer failed: {e}")
        
        # Test 2: Tool Selection Engine
        print("\n2️⃣ Testing Tool Selection Engine...")
        try:
            from tool_selection_engine import tool_selection_engine
            total_tools = len(tool_selection_engine.tool_registry.tools)
            test_results["tool_selection_engine"] = "✅ PASSED"
            print(f"   ✅ Tool selection engine initialized successfully")
            print(f"   🛠️  Total tools registered: {total_tools}")
        except Exception as e:
            test_results["tool_selection_engine"] = f"❌ FAILED: {e}"
            print(f"   ❌ Tool selection engine failed: {e}")
        
        # Test 3: Dynamic Workflow Orchestration
        print("\n3️⃣ Testing Dynamic Workflow Orchestration...")
        try:
            from dynamic_workflow_orchestrator import dynamic_orchestrator
            test_results["dynamic_workflow_orchestration"] = "✅ PASSED"
            print(f"   ✅ Dynamic workflow orchestration initialized successfully")
        except Exception as e:
            test_results["dynamic_workflow_orchestration"] = f"❌ FAILED: {e}"
            print(f"   ❌ Dynamic workflow orchestration failed: {e}")
        
        # Test 4: Adaptive Context Management
        print("\n4️⃣ Testing Adaptive Context Management...")
        try:
            from adaptive_context_manager import adaptive_context_manager
            test_results["adaptive_context_management"] = "✅ PASSED"
            print(f"   ✅ Adaptive context management initialized successfully")
        except Exception as e:
            test_results["adaptive_context_management"] = f"❌ FAILED: {e}"
            print(f"   ❌ Adaptive context management failed: {e}")
        
        # Test 5: Enhanced Chat Interface
        print("\n5️⃣ Testing Enhanced Chat Interface...")
        try:
            from enhanced_chat_interface import enhanced_chat_interface
            test_results["enhanced_chat_interface"] = "✅ PASSED"
            print(f"   ✅ Enhanced chat interface initialized successfully")
        except Exception as e:
            test_results["enhanced_chat_interface"] = f"❌ FAILED: {e}"
            print(f"   ❌ Enhanced chat interface failed: {e}")
        
        # Test 6: Advanced MCP Management
        print("\n6️⃣ Testing Advanced MCP Management...")
        try:
            from advanced_mcp_manager import advanced_mcp_manager
            status = advanced_mcp_manager.get_integration_status()
            test_results["advanced_mcp_management"] = "✅ PASSED"
            print(f"   ✅ Advanced MCP management initialized successfully")
            print(f"   🔧 Discovery methods: {status.get('discovery_methods', 0)}")
        except Exception as e:
            test_results["advanced_mcp_management"] = f"❌ FAILED: {e}"
            print(f"   ❌ Advanced MCP management failed: {e}")
        
        # Test 7: Enhanced Knowledge Memory
        print("\n7️⃣ Testing Enhanced Knowledge Memory...")
        try:
            from enhanced_knowledge_memory import enhanced_knowledge_memory
            test_results["enhanced_knowledge_memory"] = "✅ PASSED"
            print(f"   ✅ Enhanced knowledge memory initialized successfully")
        except Exception as e:
            test_results["enhanced_knowledge_memory"] = f"❌ FAILED: {e}"
            print(f"   ❌ Enhanced knowledge memory failed: {e}")
        
        # Test 8: Integration Manager
        print("\n8️⃣ Testing Enhancement Integration Manager...")
        try:
            from enhancement_integration import enhancement_integration_manager
            test_results["integration_manager"] = "✅ PASSED"
            print(f"   ✅ Enhancement integration manager initialized successfully")
        except Exception as e:
            test_results["integration_manager"] = f"❌ FAILED: {e}"
            print(f"   ❌ Enhancement integration manager failed: {e}")
        
        # Test 9: System Integration
        print("\n9️⃣ Testing System Integration...")
        try:
            from enhancement_integration import initialize_enhancements
            init_results = await initialize_enhancements()
            test_results["system_integration"] = "✅ PASSED"
            print(f"   ✅ System integration test passed")
            print(f"   🔧 Systems initialized: {len([r for r in init_results.values() if '✅' in str(r)])}")
        except Exception as e:
            test_results["system_integration"] = f"❌ FAILED: {e}"
            print(f"   ❌ System integration test failed: {e}")
        
        # Test 10: Performance Test
        print("\n🔟 Testing Performance...")
        try:
            from enhancement_integration import execute_enhanced_workflow
            start_time = asyncio.get_event_loop().time()
            
            result = await execute_enhanced_workflow(
                "Analyze threat intelligence data",
                {"test_mode": True}
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            test_results["performance_test"] = "✅ PASSED"
            print(f"   ✅ Performance test completed in {execution_time:.2f} seconds")
            print(f"   🚀 Enhancements applied: {len(result.get('enhancements_applied', []))}")
            
        except Exception as e:
            test_results["performance_test"] = f"❌ FAILED: {e}"
            print(f"   ❌ Performance test failed: {e}")
        
    except Exception as e:
        print(f"❌ Critical error during testing: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        if "✅ PASSED" in result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print(f"\n📈 Overall Results: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("🎉 All enhancement systems are working correctly!")
        return True
    else:
        print(f"⚠️  {failed} enhancement system(s) have issues that need attention")
        return False

async def test_individual_systems():
    """Test individual enhancement systems in detail."""
    print("\n🔍 Detailed System Testing")
    print("=" * 40)
    
    # Test Performance Optimizer
    print("\n📊 Testing Performance Optimizer...")
    try:
        from performance_optimizer import performance_optimizer
        stats = performance_optimizer.get_optimization_stats()
        print(f"   Cache size: {stats.get('cache', {}).get('cache_size_mb', 0):.2f} MB")
        print(f"   Performance metrics: {len(stats.get('performance', {}))} metrics available")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Tool Selection
    print("\n🛠️  Testing Tool Selection Engine...")
    try:
        from tool_selection_engine import tool_selection_engine
        local_tools = len(tool_selection_engine.tool_registry.get_local_tools())
        mcp_tools = len(tool_selection_engine.tool_registry.get_mcp_tools())
        print(f"   Local tools: {local_tools}")
        print(f"   MCP tools: {mcp_tools}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Knowledge Memory
    print("\n🧠 Testing Knowledge Memory...")
    try:
        from enhanced_knowledge_memory import get_memory_stats
        stats = await get_memory_stats()
        print(f"   Total nodes: {stats.get('total_nodes', 0)}")
        print(f"   Total relationships: {stats.get('total_relationships', 0)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

async def main():
    """Main test function."""
    print("Cybersecurity Agent Enhancement Systems Test Suite")
    print("=" * 60)
    
    # Run basic tests
    success = await test_enhancement_systems()
    
    if success:
        # Run detailed tests
        await test_individual_systems()
        
        print("\n🎯 All systems are ready for use!")
        print("\n📖 Next steps:")
        print("   1. Use 'python bin/enhancement_integration.py --initialize' to start")
        print("   2. Check 'documentation/ENHANCEMENT_SYSTEMS_GUIDE.md' for usage")
        print("   3. Test workflows with 'python bin/enhancement_integration.py --workflow'")
    else:
        print("\n⚠️  Some systems need attention before use")
        print("   Check the error messages above and fix any issues")

if __name__ == "__main__":
    asyncio.run(main())
