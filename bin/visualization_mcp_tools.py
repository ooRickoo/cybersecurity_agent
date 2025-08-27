#!/usr/bin/env python3
"""
Visualization MCP Tools
Comprehensive visualization tools for the Runner Agent to use dynamically in workflows.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class VisualizationMCPTools:
    """MCP-compatible visualization tools for dynamic workflow integration."""
    
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.visualization_dir = Path("session-outputs/visualizations")
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization engines
        self._initialize_visualization_engines()
    
    def _initialize_visualization_engines(self):
        """Initialize available visualization engines."""
        self.engines = {
            'dataframe_viewer': self._check_dataframe_viewer(),
            'workflow_diagram': self._check_workflow_diagram(),
            'neo4j_visualizer': self._check_neo4j_visualizer(),
            'vega_lite': self._check_vega_lite()
        }
    
    def _check_dataframe_viewer(self) -> bool:
        """Check if DataFrame viewer is available."""
        try:
            import tkinter as tk
            from tkinter import ttk
            return True
        except ImportError:
            return False
    
    def _check_workflow_diagram(self) -> bool:
        """Check if workflow diagram tools are available."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            return True
        except ImportError:
            return False
    
    def _check_neo4j_visualizer(self) -> bool:
        """Check if Neo4j visualization tools are available."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            return True
        except ImportError:
            return False
    
    def _check_vega_lite(self) -> bool:
        """Check if Vega-Lite is available."""
        try:
            import altair as alt
            return True
        except ImportError:
            return False
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available visualization MCP tools."""
        return {
            'dataframe_viewer': {
                'name': 'DataFrame Viewer',
                'description': 'Interactive DataFrame viewer for data validation and analysis',
                'category': 'visualization',
                'parameters': {
                    'data': {'type': 'DataFrame', 'description': 'Data to visualize'},
                    'title': {'type': 'string', 'description': 'Title for the visualization'},
                    'description': {'type': 'string', 'description': 'Description of the data'}
                },
                'returns': {'type': 'string', 'description': 'Path to generated visualization'},
                'available': self.engines['dataframe_viewer']
            },
            'workflow_diagram': {
                'name': 'Workflow Diagram',
                'description': 'Create beautiful workflow step visualizations',
                'category': 'visualization',
                'parameters': {
                    'workflow_steps': {'type': 'list', 'description': 'List of workflow steps'},
                    'title': {'type': 'string', 'description': 'Title for the workflow diagram'}
                },
                'returns': {'type': 'string', 'description': 'Path to generated workflow diagram'},
                'available': self.engines['workflow_diagram']
            },
            'neo4j_graph_visualizer': {
                'name': 'Neo4j Graph Visualizer',
                'description': 'Visualize resource relationships and graph data',
                'category': 'visualization',
                'parameters': {
                    'graph_data': {'type': 'dict', 'description': 'Graph data with nodes and edges'},
                    'title': {'type': 'string', 'description': 'Title for the graph visualization'}
                },
                'returns': {'type': 'string', 'description': 'Path to generated graph visualization'},
                'available': self.engines['neo4j_visualizer']
            },
            'vega_lite_charts': {
                'name': 'Vega-Lite Charts',
                'description': 'Create professional data visualizations with Vega-Lite',
                'category': 'visualization',
                'parameters': {
                    'data': {'type': 'DataFrame', 'description': 'Data for visualization'},
                    'chart_spec': {'type': 'dict', 'description': 'Vega-Lite chart specification'},
                    'title': {'type': 'string', 'description': 'Title for the chart'}
                },
                'returns': {'type': 'string', 'description': 'Path to generated chart'},
                'available': self.engines['vega_lite']
            },
            'visualization_exporter': {
                'name': 'Visualization Exporter',
                'description': 'Export visualizations to multiple formats (HTML, PNG, SVG)',
                'category': 'visualization',
                'parameters': {
                    'visualization_type': {'type': 'string', 'description': 'Type of visualization to export'},
                    'data': {'type': 'any', 'description': 'Data for visualization'},
                    'title': {'type': 'string', 'description': 'Title for the visualization'},
                    'export_formats': {'type': 'list', 'description': 'List of export formats'}
                },
                'returns': {'type': 'dict', 'description': 'Paths to exported files'},
                'available': True
            },
            'workflow_visualization_suggester': {
                'name': 'Workflow Visualization Suggester',
                'description': 'Suggest appropriate visualizations for workflow data',
                'category': 'visualization',
                'parameters': {
                    'workflow_data': {'type': 'dict', 'description': 'Workflow data to analyze'}
                },
                'returns': {'type': 'list', 'description': 'List of suggested visualizations'},
                'available': True
            }
        }
    
    def execute_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute a visualization tool based on MCP tool ID."""
        try:
            if tool_id == 'dataframe_viewer':
                return self._execute_dataframe_viewer(**kwargs)
            elif tool_id == 'workflow_diagram':
                return self._execute_workflow_diagram(**kwargs)
            elif tool_id == 'neo4j_graph_visualizer':
                return self._execute_neo4j_visualizer(**kwargs)
            elif tool_id == 'vega_lite_charts':
                return self._execute_vega_lite_charts(**kwargs)
            elif tool_id == 'visualization_exporter':
                return self._execute_visualization_exporter(**kwargs)
            elif tool_id == 'workflow_visualization_suggester':
                return self._execute_workflow_visualization_suggester(**kwargs)
            else:
                return {'error': f'Unknown tool: {tool_id}', 'success': False}
                
        except Exception as e:
            return {'error': f'Tool execution error: {e}', 'success': False}
    
    def _execute_dataframe_viewer(self, **kwargs) -> Dict[str, Any]:
        """Execute DataFrame viewer tool."""
        if not self.engines['dataframe_viewer']:
            return {'error': 'DataFrame viewer not available', 'success': False}
        
        try:
            data = kwargs.get('data')
            title = kwargs.get('title', 'Data Validation')
            description = kwargs.get('description', '')
            
            if data is None:
                return {'error': 'Data required for DataFrame viewer', 'success': False}
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except Exception as e:
                    return {'error': f'Cannot convert data to DataFrame: {e}', 'success': False}
            
            # Create visualization
            result = self._create_dataframe_viewer(data, title, description)
            
            return {
                'tool': 'dataframe_viewer',
                'result': result,
                'success': result is not None,
                'data_shape': data.shape,
                'data_columns': list(data.columns)
            }
            
        except Exception as e:
            return {'error': f'DataFrame viewer error: {e}', 'success': False}
    
    def _execute_workflow_diagram(self, **kwargs) -> Dict[str, Any]:
        """Execute workflow diagram tool."""
        if not self.engines['workflow_diagram']:
            return {'error': 'Workflow diagram tools not available', 'success': False}
        
        try:
            workflow_steps = kwargs.get('workflow_steps')
            title = kwargs.get('title', 'Workflow Steps')
            
            if workflow_steps is None:
                return {'error': 'Workflow steps required', 'success': False}
            
            # Create visualization
            result = self._create_workflow_diagram(workflow_steps, title)
            
            return {
                'tool': 'workflow_diagram',
                'result': result,
                'success': result is not None,
                'steps_count': len(workflow_steps)
            }
            
        except Exception as e:
            return {'error': f'Workflow diagram error: {e}', 'success': False}
    
    def _execute_neo4j_visualizer(self, **kwargs) -> Dict[str, Any]:
        """Execute Neo4j graph visualizer tool."""
        if not self.engines['neo4j_visualizer']:
            return {'error': 'Neo4j visualization tools not available', 'success': False}
        
        try:
            graph_data = kwargs.get('graph_data')
            title = kwargs.get('title', 'Resource Relationships')
            
            if graph_data is None:
                return {'error': 'Graph data required', 'success': False}
            
            # Create visualization
            result = self._create_neo4j_visualization(graph_data, title)
            
            return {
                'tool': 'neo4j_graph_visualizer',
                'result': result,
                'success': result is not None,
                'nodes_count': len(graph_data.get('nodes', [])),
                'edges_count': len(graph_data.get('edges', []))
            }
            
        except Exception as e:
            return {'error': f'Neo4j visualizer error: {e}', 'success': False}
    
    def _execute_vega_lite_charts(self, **kwargs) -> Dict[str, Any]:
        """Execute Vega-Lite charts tool."""
        if not self.engines['vega_lite']:
            return {'error': 'Vega-Lite not available', 'success': False}
        
        try:
            data = kwargs.get('data')
            chart_spec = kwargs.get('chart_spec', {})
            title = kwargs.get('title', 'Data Visualization')
            
            if data is None or chart_spec is None:
                return {'error': 'Data and chart specification required', 'success': False}
            
            # Create visualization
            result = self._create_vega_lite_chart(data, chart_spec, title)
            
            return {
                'tool': 'vega_lite_charts',
                'result': result,
                'success': result is not None,
                'chart_type': chart_spec.get('mark', 'unknown')
            }
            
        except Exception as e:
            return {'error': f'Vega-Lite charts error: {e}', 'success': False}
    
    def _execute_visualization_exporter(self, **kwargs) -> Dict[str, Any]:
        """Execute visualization exporter tool."""
        try:
            visualization_type = kwargs.get('visualization_type')
            data = kwargs.get('data')
            title = kwargs.get('title', 'Exported Visualization')
            export_formats = kwargs.get('export_formats', ['html', 'png', 'svg'])
            
            if visualization_type is None or data is None:
                return {'error': 'Visualization type and data required', 'success': False}
            
            # Export visualization
            result = self._export_visualization(visualization_type, data, title, export_formats)
            
            return {
                'tool': 'visualization_exporter',
                'result': result,
                'success': result is not None,
                'exported_formats': export_formats
            }
            
        except Exception as e:
            return {'error': f'Visualization exporter error: {e}', 'success': False}
    
    def _execute_workflow_visualization_suggester(self, **kwargs) -> Dict[str, Any]:
        """Execute workflow visualization suggester tool."""
        try:
            workflow_data = kwargs.get('workflow_data')
            
            if workflow_data is None:
                return {'error': 'Workflow data required', 'success': False}
            
            # Suggest visualizations
            suggestions = self._suggest_workflow_visualizations(workflow_data)
            
            return {
                'tool': 'workflow_visualization_suggester',
                'result': suggestions,
                'success': True,
                'suggestions_count': len(suggestions)
            }
            
        except Exception as e:
            return {'error': f'Visualization suggester error: {e}', 'success': False}
    
    def _create_dataframe_viewer(self, df: pd.DataFrame, title: str, description: str) -> Optional[str]:
        """Create DataFrame viewer visualization."""
        try:
            # Create HTML export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataframe_viewer_{timestamp}.html"
            filepath = self.visualization_dir / filename
            
            # Create styled HTML
            html_content = self._create_dataframe_html(df, title, description)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error creating DataFrame viewer: {e}")
            return None
    
    def _create_workflow_diagram(self, workflow_steps: List[Dict[str, Any]], title: str) -> Optional[str]:
        """Create workflow diagram visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, step in enumerate(workflow_steps):
                G.add_node(i, label=step.get('name', f'Step {i+1}'))
            
            # Add edges
            for i in range(len(workflow_steps) - 1):
                G.add_edge(i, i + 1)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=2000, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                  arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels
            labels = {node: G.nodes[node]['label'] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_diagram_{timestamp}.png"
            filepath = self.visualization_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error creating workflow diagram: {e}")
            return None
    
    def _create_neo4j_visualization(self, graph_data: Dict[str, Any], title: str) -> Optional[str]:
        """Create Neo4j graph visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes
            for node in graph_data.get('nodes', []):
                G.add_node(node.get('id', node.get('name', 'Unknown')), 
                          **{k: v for k, v in node.items() if k not in ['id', 'name']})
            
            # Add edges
            for edge in graph_data.get('edges', []):
                G.add_edge(edge.get('source', edge.get('from')), 
                          edge.get('target', edge.get('to')),
                          **{k: v for k, v in edge.items() if k not in ['source', 'target', 'from', 'to']})
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                                  node_size=1500, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='darkgreen', 
                                  alpha=0.6, width=1.5)
            
            # Draw labels
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neo4j_graph_{timestamp}.png"
            filepath = self.visualization_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error creating Neo4j visualization: {e}")
            return None
    
    def _create_vega_lite_chart(self, data: pd.DataFrame, chart_spec: Dict[str, Any], title: str) -> Optional[str]:
        """Create Vega-Lite chart visualization."""
        try:
            import altair as alt
            
            # Convert DataFrame to Altair chart
            chart = alt.Chart(data).mark_bar().encode(
                x=chart_spec.get('x', alt.X('*:Q', bin=True)),
                y=chart_spec.get('y', 'count()')
            ).properties(
                title=title,
                width=600,
                height=400
            )
            
            # Save as HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vega_lite_chart_{timestamp}.html"
            filepath = self.visualization_dir / filename
            
            chart.save(filepath)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error creating Vega-Lite chart: {e}")
            return None
    
    def _export_visualization(self, visualization_type: str, data: Any, title: str, 
                             export_formats: List[str]) -> Optional[Dict[str, str]]:
        """Export visualization to multiple formats."""
        try:
            results = {}
            
            for fmt in export_formats:
                if fmt == 'html':
                    if visualization_type == 'dataframe':
                        results['html'] = self._create_dataframe_html(data, title, "")
                    else:
                        results['html'] = f"HTML export not available for {visualization_type}"
                
                elif fmt == 'png':
                    if visualization_type in ['workflow_diagram', 'neo4j_graph_visualizer']:
                        # PNG already created
                        results['png'] = "PNG export available"
                    else:
                        results['png'] = f"PNG export not available for {visualization_type}"
                
                elif fmt == 'svg':
                    if visualization_type in ['workflow_diagram', 'neo4j_graph_visualizer']:
                        # Convert to SVG
                        results['svg'] = self._convert_to_svg(visualization_type, data, title)
                    else:
                        results['svg'] = f"SVG export not available for {visualization_type}"
            
            return results
            
        except Exception as e:
            print(f"❌ Error exporting visualization: {e}")
            return None
    
    def _create_dataframe_html(self, df: pd.DataFrame, title: str, description: str) -> str:
        """Create styled HTML for DataFrame."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 10px; }}
                .description {{ color: #7f8c8d; text-align: center; margin-bottom: 30px; font-style: italic; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e8f4f8; }}
                .info {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .timestamp {{ color: #95a5a6; font-size: 12px; text-align: center; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                {f'<div class="description">{description}</div>' if description else ''}
                
                <div class="info">
                    <strong>Data Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
                    <strong>Columns:</strong> {', '.join(df.columns.tolist())}
                </div>
                
                {df.to_html(classes='dataframe', index=False, escape=False)}
                
                <div class="timestamp">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _convert_to_svg(self, visualization_type: str, data: Any, title: str) -> str:
        """Convert visualization to SVG format."""
        try:
            # This would require additional implementation for SVG conversion
            # For now, return a placeholder
            return f"SVG conversion for {visualization_type} not yet implemented"
        except Exception as e:
            return f"SVG conversion error: {e}"
    
    def _suggest_workflow_visualizations(self, workflow_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest appropriate visualizations for workflow data."""
        suggestions = []
        
        try:
            # Analyze workflow data and suggest visualizations
            if 'dataframes' in workflow_data:
                for df_name, df_data in workflow_data['dataframes'].items():
                    suggestions.append({
                        'type': 'dataframe_viewer',
                        'title': f'Data Validation: {df_name}',
                        'description': f'Interactive validation of {df_name} data',
                        'data_key': df_name
                    })
            
            if 'workflow_steps' in workflow_data:
                suggestions.append({
                    'type': 'workflow_diagram',
                    'title': 'Workflow Execution Steps',
                    'description': 'Visual representation of workflow progress',
                    'data_key': 'workflow_steps'
                })
            
            if 'graph_data' in workflow_data:
                suggestions.append({
                    'type': 'neo4j_graph_visualizer',
                    'title': 'Resource Relationships',
                    'description': 'Visualization of resource connections and dependencies',
                    'data_key': 'graph_data'
                })
            
            if 'analysis_results' in workflow_data:
                suggestions.append({
                    'type': 'vega_lite_charts',
                    'title': 'Analysis Results',
                    'description': 'Interactive charts of analysis outcomes',
                    'data_key': 'analysis_results'
                })
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️  Error suggesting visualizations: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_df = pd.DataFrame({
        'Policy_ID': ['POL001', 'POL002', 'POL003'],
        'Policy_Name': ['User Auth', 'Remote Access', 'Data Encryption'],
        'Risk_Level': ['High', 'High', 'Medium'],
        'MITRE_Technique': ['T1078', 'T1021', 'T1005']
    })
    
    sample_workflow = [
        {'name': 'Data Loading', 'status': 'completed'},
        {'name': 'Analysis', 'status': 'in_progress'},
        {'name': 'Visualization', 'status': 'pending'}
    ]
    
    sample_graph = {
        'nodes': [
            {'id': 'User', 'type': 'entity'},
            {'id': 'System', 'type': 'resource'},
            {'id': 'Data', 'type': 'asset'}
        ],
        'edges': [
            {'source': 'User', 'target': 'System', 'relationship': 'accesses'},
            {'source': 'System', 'target': 'Data', 'relationship': 'contains'}
        ]
    }
    
    # Initialize tools
    viz_tools = VisualizationMCPTools()
    
    print("🎨 Visualization MCP Tools Test")
    print("=" * 50)
    
    # Test available tools
    tools = viz_tools.get_available_tools()
    print(f"Available tools: {len(tools)}")
    for tool_id, tool_info in tools.items():
        status = "✅" if tool_info['available'] else "❌"
        print(f"  {status} {tool_info['name']}: {tool_info['description']}")
    
    print("\n🧪 Testing Tools:")
    
    # Test DataFrame viewer
    result = viz_tools.execute_tool('dataframe_viewer', 
                                   data=sample_df, 
                                   title='Policy Analysis Results')
    print(f"  DataFrame Viewer: {'✅' if result.get('success') else '❌'}")
    
    # Test workflow diagram
    result = viz_tools.execute_tool('workflow_diagram', 
                                   workflow_steps=sample_workflow, 
                                   title='Security Analysis Workflow')
    print(f"  Workflow Diagram: {'✅' if result.get('success') else '❌'}")
    
    # Test Neo4j visualizer
    result = viz_tools.execute_tool('neo4j_graph_visualizer', 
                                   graph_data=sample_graph, 
                                   title='Security Architecture')
    print(f"  Neo4j Visualizer: {'✅' if result.get('success') else '❌'}")
    
    # Test visualization suggester
    workflow_data = {
        'dataframes': {'policies': sample_df},
        'workflow_steps': sample_workflow,
        'graph_data': sample_graph
    }
    result = viz_tools.execute_tool('workflow_visualization_suggester', 
                                   workflow_data=workflow_data)
    print(f"  Visualization Suggester: {'✅' if result.get('success') else '❌'}")
    
    print(f"\n📁 Visualizations saved to: {viz_tools.visualization_dir}")
