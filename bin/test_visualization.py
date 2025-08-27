#!/usr/bin/env python3
"""
Test script for visualization tools
"""

import sys
from pathlib import Path

# Add the bin directory to the path
bin_path = Path(__file__).parent / "bin"
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from visualization_manager import VisualizationManager
    from enhanced_session_manager import EnhancedSessionManager
    
    print("🔍 Testing visualization tools...")
    
    # Initialize session manager
    session_manager = EnhancedSessionManager()
    
    # Initialize visualization manager
    viz_manager = VisualizationManager(session_manager)
    
    print("✅ Visualization manager initialized successfully")
    
    # Test dataframe viewer
    try:
        import pandas as pd
        test_df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Department': ['IT', 'HR', 'Finance']
        })
        
        print("📊 Testing dataframe viewer...")
        result = viz_manager.show_dataframe_viewer(test_df, "Test Data")
        print(f"✅ Dataframe viewer result: {result}")
        
    except Exception as e:
        print(f"❌ Dataframe viewer failed: {e}")
    
    # Test workflow diagram
    try:
        print("📋 Testing workflow diagram...")
        workflow_data = {
            'nodes': [
                {'id': 'start', 'label': 'Start', 'type': 'start'},
                {'id': 'process', 'label': 'Process', 'type': 'process'},
                {'id': 'end', 'label': 'End', 'type': 'end'}
            ],
            'edges': [
                {'from': 'start', 'to': 'process'},
                {'from': 'process', 'to': 'end'}
            ]
        }
        result = viz_manager.create_workflow_diagram(workflow_data, "Test Workflow")
        print(f"✅ Workflow diagram result: {result}")
        
    except Exception as e:
        print(f"❌ Workflow diagram failed: {e}")
    
    # Test Neo4j graph visualization
    try:
        print("🕸️ Testing Neo4j graph visualization...")
        graph_data = {
            'nodes': [
                {'id': '1', 'label': 'Host A', 'type': 'host'},
                {'id': '2', 'label': 'Process B', 'type': 'process'},
                {'id': '3', 'label': 'File C', 'type': 'file'}
            ],
            'relationships': [
                {'from': '1', 'to': '2', 'type': 'runs'},
                {'from': '2', 'to': '3', 'type': 'accesses'}
            ]
        }
        result = viz_manager.create_neo4j_graph_visualization(graph_data, "Test Graph")
        print(f"✅ Neo4j graph visualization result: {result}")
        
    except Exception as e:
        print(f"❌ Neo4j graph visualization failed: {e}")
    
    # Test Vega-Lite chart
    try:
        print("📈 Testing Vega-Lite chart...")
        chart_data = {
            'data': [
                {'category': 'A', 'value': 10},
                {'category': 'B', 'value': 20},
                {'category': 'C', 'value': 15}
            ],
            'chart_type': 'bar'
        }
        result = viz_manager.create_vega_lite_chart(chart_data, "Test Chart")
        print(f"✅ Vega-Lite chart result: {result}")
        
    except Exception as e:
        print(f"❌ Vega-Lite chart failed: {e}")
    
    print("\n🎯 Visualization tools test completed!")
    
except ImportError as e:
    print(f"❌ Failed to import visualization modules: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
