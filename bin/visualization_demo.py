#!/usr/bin/env python3
"""
Visualization Demo Script
Demonstrates all visualization features of the cybersecurity agent.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def create_sample_data():
    """Create sample data for visualization demos."""
    
    # Sample DataFrame for data validation
    df_data = {
        'id': range(1, 101),
        'name': [f'Item_{i}' for i in range(1, 101)],
        'category': np.random.choice(['Threat', 'Vulnerability', 'Asset', 'Control'], 100),
        'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 100),
        'status': np.random.choice(['Open', 'In Progress', 'Resolved', 'Closed'], 100),
        'score': np.random.randint(1, 101, 100),
        'created_date': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(df_data)
    
    # Sample workflow steps
    workflow_steps = [
        {'id': 'start', 'name': 'Start Analysis', 'type': 'start'},
        {'id': 'collect', 'name': 'Collect Data', 'type': 'process'},
        {'id': 'analyze', 'name': 'Analyze Threats', 'type': 'process'},
        {'id': 'assess', 'name': 'Assess Risk', 'type': 'decision'},
        {'id': 'mitigate', 'name': 'Mitigate Threats', 'type': 'process'},
        {'id': 'monitor', 'name': 'Monitor Results', 'type': 'process'},
        {'id': 'end', 'name': 'Complete Analysis', 'type': 'end'}
    ]
    
    # Sample Neo4j graph data
    graph_data = {
        'nodes': [
            {'id': '1', 'labels': ['Threat'], 'properties': {'name': 'Malware', 'severity': 'High'}},
            {'id': '2', 'labels': ['Asset'], 'properties': {'name': 'Database Server', 'type': 'Critical'}},
            {'id': '3', 'labels': ['Control'], 'properties': {'name': 'Firewall', 'type': 'Network'}},
            {'id': '4', 'labels': ['Vulnerability'], 'properties': {'name': 'SQL Injection', 'severity': 'Critical'}},
            {'id': '5', 'labels': ['Threat'], 'properties': {'name': 'Phishing', 'severity': 'Medium'}}
        ],
        'relationships': [
            {'start': {'id': '1'}, 'end': {'id': '2'}, 'type': 'TARGETS'},
            {'start': {'id': '3'}, 'end': {'id': '1'}, 'type': 'MITIGATES'},
            {'start': {'id': '4'}, 'end': {'id': '2'}, 'type': 'EXPLOITS'},
            {'start': {'id': '5'}, 'end': {'id': '2'}, 'type': 'TARGETS'},
            {'start': {'id': '3'}, 'end': {'id': '4'}, 'type': 'PROTECTS_AGAINST'}
        ]
    }
    
    # Sample Vega-Lite chart specification
    chart_spec = {
        'x': 'severity:N',
        'y': 'count():Q',
        'color': 'category:N',
        'tooltip': ['category', 'severity', 'count()']
    }
    
    return df, workflow_steps, graph_data, chart_spec

def demo_dataframe_viewer(viz_manager):
    """Demo DataFrame viewer."""
    print("\n🔍 Demo: DataFrame Viewer")
    print("=" * 50)
    
    df, _, _, _ = create_sample_data()
    print(f"Created sample DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show DataFrame info
    print("\nDataFrame Preview:")
    print(df.head())
    
    # Create interactive viewer
    print("\n🎨 Opening DataFrame viewer...")
    result = viz_manager.create_dataframe_viewer(
        df, 
        title="Cybersecurity Data Validation", 
        description="Sample threat and vulnerability data for validation"
    )
    
    if result:
        print(f"✅ DataFrame viewer completed: {result}")
    else:
        print("❌ DataFrame viewer failed")

def demo_workflow_diagram(viz_manager):
    """Demo workflow diagram creation."""
    print("\n🔄 Demo: Workflow Diagram")
    print("=" * 50)
    
    _, workflow_steps, _, _ = create_sample_data()
    print(f"Created workflow with {len(workflow_steps)} steps")
    
    for step in workflow_steps:
        print(f"  • {step['name']} ({step['type']})")
    
    # Create workflow visualization
    print("\n🎨 Creating workflow diagram...")
    result = viz_manager.create_workflow_diagram(
        workflow_steps, 
        title="Cybersecurity Analysis Workflow"
    )
    
    if result:
        print(f"✅ Workflow diagram created: {result}")
    else:
        print("❌ Workflow diagram creation failed")

def demo_neo4j_visualization(viz_manager):
    """Demo Neo4j graph visualization."""
    print("\n🕸️  Demo: Neo4j Graph Visualization")
    print("=" * 50)
    
    _, _, graph_data, _ = create_sample_data()
    print(f"Created graph with {len(graph_data['nodes'])} nodes and {len(graph_data['relationships'])} relationships")
    
    for node in graph_data['nodes']:
        print(f"  • {node['labels'][0]}: {node['properties']['name']}")
    
    # Create graph visualization
    print("\n🎨 Creating Neo4j graph visualization...")
    result = viz_manager.create_neo4j_graph_visualization(
        graph_data, 
        title="Cybersecurity Resource Relationships"
    )
    
    if result:
        print(f"✅ Neo4j graph visualization created: {result}")
    else:
        print("❌ Neo4j graph visualization creation failed")

def demo_vega_lite_charts(viz_manager):
    """Demo Vega-Lite chart creation."""
    print("\n📊 Demo: Vega-Lite Charts")
    print("=" * 50)
    
    df, _, _, chart_spec = create_sample_data()
    
    # Create aggregated data for charting
    chart_data = df.groupby(['severity', 'category']).size().reset_index(name='count')
    print(f"Created chart data: {len(chart_data)} data points")
    
    # Create Vega-Lite visualization
    print("\n🎨 Creating Vega-Lite chart...")
    result = viz_manager.create_vega_lite_visualization(
        chart_data, 
        chart_spec, 
        title="Cybersecurity Data Distribution"
    )
    
    if result:
        print(f"✅ Vega-Lite chart created: {result}")
    else:
        print("❌ Vega-Lite chart creation failed")

def demo_export_features(viz_manager):
    """Demo export features."""
    print("\n📤 Demo: Export Features")
    print("=" * 50)
    
    df, _, _, _ = create_sample_data()
    
    # Export DataFrame to HTML
    print("\n🎨 Exporting DataFrame to HTML...")
    viz_manager._export_dataframe_html(df, "Cybersecurity Data Export")
    
    print("✅ Export demo completed")

def main():
    """Main demo function."""
    print("🎨 Cybersecurity Agent Visualization Demo")
    print("=" * 60)
    
    try:
        from bin.visualization_manager import VisualizationManager
        
        # Initialize visualization manager
        viz_manager = VisualizationManager()
        
        # Check available engines
        print("\n🔧 Available Visualization Engines:")
        for engine, available in viz_manager.engines.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"   {engine}: {status}")
        
        # Run demos for available engines
        if viz_manager.engines['dataframe_viewer']:
            demo_dataframe_viewer(viz_manager)
        
        if viz_manager.engines['workflow_diagram']:
            demo_workflow_diagram(viz_manager)
        
        if viz_manager.engines['neo4j_visualizer']:
            demo_neo4j_visualization(viz_manager)
        
        if viz_manager.engines['vega_lite']:
            demo_vega_lite_charts(viz_manager)
        
        # Always demo export features
        demo_export_features(viz_manager)
        
        print("\n🎉 Visualization demo completed!")
        print(f"📁 Check the 'session-outputs/visualizations' folder for generated files")
        
    except ImportError as e:
        print(f"❌ Could not import visualization manager: {e}")
        print("   Please ensure all dependencies are installed:")
        print("   pip install matplotlib networkx altair pandas numpy")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
