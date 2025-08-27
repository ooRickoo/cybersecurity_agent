#!/usr/bin/env python3
"""
Visualization Manager
Professional-grade visualization tools for cybersecurity analysis.
"""

import os
import sys
import json
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class VisualizationManager:
    """Manages professional-grade visualizations for cybersecurity analysis."""
    
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
        
        print("🎨 Visualization Engines:")
        for engine, available in self.engines.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"   {engine}: {status}")
    
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
    
    def create_dataframe_viewer(self, df: pd.DataFrame, title: str = "Data Validation", 
                               description: str = "") -> Optional[str]:
        """Create an interactive DataFrame viewer window."""
        if not self.engines['dataframe_viewer']:
            print("❌ DataFrame viewer not available")
            return None
        
        try:
            import tkinter as tk
            from tkinter import ttk
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create main window
            root = tk.Tk()
            root.title(f"Data Validation: {title}")
            root.geometry("1200x800")
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            
            # Create main frame
            main_frame = ttk.Frame(root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Title and description
            title_label = ttk.Label(main_frame, text=title, font=('Arial', 16, 'bold'))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
            
            if description:
                desc_label = ttk.Label(main_frame, text=description, font=('Arial', 10))
                desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
            
            # DataFrame info
            info_frame = ttk.LabelFrame(main_frame, text="DataFrame Information", padding="5")
            info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            info_text = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
            info_label = ttk.Label(info_frame, text=info_text)
            info_label.pack()
            
            # Create Treeview for DataFrame
            tree_frame = ttk.Frame(main_frame)
            tree_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(3, weight=1)
            tree_frame.columnconfigure(0, weight=1)
            tree_frame.rowconfigure(0, weight=1)
            
            # Create Treeview
            tree = ttk.Treeview(tree_frame)
            tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
            
            # Grid the tree and scrollbars
            tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            tree_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
            tree_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            # Configure columns
            tree['columns'] = list(df.columns)
            tree['show'] = 'headings'
            
            # Set column headings
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=150, minwidth=100)
            
            # Insert data (limit to first 1000 rows for performance)
            max_rows = min(1000, len(df))
            for i, row in df.head(max_rows).iterrows():
                tree.insert('', 'end', values=list(row))
            
            if len(df) > max_rows:
                info_text += f" (showing first {max_rows} rows)"
                info_label.config(text=info_text)
            
            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
            
            # Export button
            export_btn = ttk.Button(button_frame, text="Export to HTML", 
                                  command=lambda: self._export_dataframe_html(df, title))
            export_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            # Close button
            close_btn = ttk.Button(button_frame, text="Close", command=root.destroy)
            close_btn.pack(side=tk.LEFT)
            
            # Log the visualization creation
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "visualization", f"Created interactive viewer for {df.shape[0]}x{df.shape[1]} DataFrame",
                    {'visualization_type': 'dataframe_viewer', 'title': title, 'dataframe_shape': f"{df.shape[0]}x{df.shape[1]}"}
                )
            
            # Start the GUI event loop
            root.mainloop()
            
            return f"DataFrame viewer completed for {title}"
            
        except Exception as e:
            print(f"❌ Error creating DataFrame viewer: {e}")
            return None
    
    def create_workflow_diagram(self, workflow_steps: List[Dict[str, Any]], 
                               title: str = "Workflow Visualization") -> Optional[str]:
        """Create a beautiful, modern workflow diagram visualization with enhanced spacing and styling."""
        if not self.engines['workflow_diagram']:
            print("❌ Workflow diagram tools not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
            import numpy as np
            
            # Create figure with modern dark theme and better proportions
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(18, 12))
            
            # Set background with gradient effect
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            node_positions = {}
            for i, step in enumerate(workflow_steps):
                step_id = step.get('id', f'step_{i}')
                step_name = step.get('name', f'Step {i+1}')
                step_type = step.get('type', 'process')
                
                G.add_node(step_id, name=step_name, type=step_type)
                
                # Position nodes with much better spacing - significantly increased
                x = i * 8  # Increased from 4 to 8 for much better spacing
                y = 0
                node_positions[step_id] = (x, y)
                
                # Add edges between consecutive steps
                if i > 0:
                    prev_step_id = workflow_steps[i-1].get('id', f'step_{i-1}')
                    G.add_edge(prev_step_id, step_id)
            
            # Use custom layout with better spacing
            pos = nx.spring_layout(G, pos=node_positions, k=6, iterations=100, scale=2.0)
            
            # Modern color palette with better contrast
            modern_colors = {
                'start': '#00D4AA',      # Modern teal
                'end': '#FF6B6B',        # Modern coral
                'decision': '#FFA726',   # Modern orange
                'process': '#42A5F5',    # Modern blue
                'analysis': '#AB47BC',   # Modern purple
                'output': '#26A69A',     # Modern green
                'input': '#FF7043'       # Modern deep orange
            }
            
            # Node colors based on type with fallback
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                node_type = G.nodes[node].get('type', 'process')
                color = modern_colors.get(node_type, modern_colors['process'])
                node_colors.append(color)
                
                # Vary node sizes based on type for visual hierarchy
                if node_type == 'start':
                    node_sizes.append(4000)
                elif node_type == 'end':
                    node_sizes.append(4000)
                elif node_type == 'decision':
                    node_sizes.append(3500)
                else:
                    node_sizes.append(3000)
            
            # Draw nodes with enhanced styling
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.9, 
                                 edgecolors='#ffffff', linewidths=3)
            
            # Draw edges with modern styling and better arrows
            nx.draw_networkx_edges(G, pos, edge_color='#E0E0E0', 
                                 arrows=True, arrowsize=25, 
                                 arrowstyle='->', width=3, alpha=0.8,
                                 connectionstyle='arc3,rad=0.1')  # Curved edges
            
            # Enhanced node labels with better typography
            nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', 
                                  font_color='#ffffff', font_family='sans-serif')
            
            # Enhanced edge labels for workflow transitions
            edge_labels = {}
            for i in range(len(workflow_steps) - 1):
                current_step = workflow_steps[i]
                next_step = workflow_steps[i + 1]
                edge_labels[(current_step.get('id', f'step_{i}'), 
                           next_step.get('id', f'step_{i+1}'))] = '→'
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=14, 
                                       font_color='#FFD700', font_weight='bold')
            
            # Enhanced plot customization with better spacing
            ax.set_title(title, fontsize=24, fontweight='bold', color='#ffffff', pad=30)
            ax.set_xlim(-4, len(workflow_steps) * 8 + 4)  # Updated to match new spacing
            ax.set_ylim(-3, 3)  # Increased vertical space
            ax.axis('off')
            
            # Enhanced legend with modern styling
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=modern_colors['start'], 
                          markersize=20, label='Start', markeredgecolor='#ffffff', markeredgewidth=2),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=modern_colors['process'], 
                          markersize=20, label='Process', markeredgecolor='#ffffff', markeredgewidth=2),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=modern_colors['decision'], 
                          markersize=20, label='Decision', markeredgecolor='#ffffff', markeredgewidth=2),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=modern_colors['analysis'], 
                          markersize=20, label='Analysis', markeredgecolor='#ffffff', markeredgewidth=2),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=modern_colors['end'], 
                          markersize=20, label='End', markeredgecolor='#ffffff', markeredgewidth=2)
            ]
            
            legend = ax.legend(handles=legend_elements, loc='upper right', 
                             fontsize=14, framealpha=0.9, 
                             facecolor='#2d2d44', edgecolor='#ffffff',
                             title='Node Types', title_fontsize=16)
            legend.get_title().set_color('#ffffff')
            
            # Add subtle grid lines for better visual structure
            ax.grid(True, alpha=0.1, color='#ffffff', linestyle='-', linewidth=0.5)
            
            # Add watermark or branding
            fig.text(0.99, 0.01, 'Cybersecurity Agent Agent', fontsize=10, color='#666666', 
                    ha='right', va='bottom', alpha=0.7)
            
            # Save the visualization
            output_path = self._save_visualization(fig, title, 'workflow_diagram')
            
            # Log the visualization creation
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "visualization", f"Created workflow diagram with {len(workflow_steps)} steps",
                    {'visualization_type': 'workflow_diagram', 'title': title, 'steps_count': len(workflow_steps)}
                )
            
            plt.close()
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating workflow diagram: {e}")
            return None
    
    def create_neo4j_graph_visualization(self, graph_data: Dict[str, Any], 
                                        title: str = "Neo4j Graph Visualization") -> Optional[str]:
        """Create a beautiful, modern Neo4j graph visualization with enhanced spacing and styling."""
        if not self.engines['neo4j_visualizer']:
            print("❌ Neo4j visualization tools not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
            
            # Create figure with modern dark theme and better proportions
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(20, 16))
            
            # Set background with gradient effect
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            
            # Create graph from Neo4j data
            G = nx.Graph()
            
            # Add nodes and edges from graph data
            if 'nodes' in graph_data:
                for node in graph_data['nodes']:
                    node_id = node.get('id', str(hash(str(node))))
                    node_type = node.get('labels', ['Unknown'])[0]
                    
                    # Create meaningful node labels from properties
                    properties = node.get('properties', {})
                    if properties:
                        # Try to find meaningful names from properties
                        for key in ['name', 'hostname', 'username', 'ip', 'path', 'service']:
                            if key in properties and properties[key]:
                                node_label = str(properties[key])
                                break
                        else:
                            # Fallback to node type + ID
                            node_label = f"{node_type}_{node_id}"
                    else:
                        node_label = f"{node_type}_{node_id}"
                    
                    G.add_node(node_id, type=node_type, properties=properties, label=node_label)
            
            if 'relationships' in graph_data:
                for rel in graph_data['relationships']:
                    start_node = rel.get('start', {}).get('id')
                    end_node = rel.get('end', {}).get('id')
                    rel_type = rel.get('type', 'RELATES_TO')
                    if start_node and end_node:
                        G.add_edge(start_node, end_node, type=rel_type)
            
            # Use spring layout with better spacing parameters
            pos = nx.spring_layout(G, k=4, iterations=100, seed=42, scale=3.0)
            
            # Modern color palette for node types
            modern_colors = {
                'host': '#00D4AA',       # Modern teal
                'process': '#42A5F5',    # Modern blue
                'file': '#FF6B6B',       # Modern coral
                'network': '#AB47BC',    # Modern purple
                'user': '#FFA726',       # Modern orange
                'database': '#26A69A',   # Modern green
                'ip': '#FF7043',         # Modern deep orange
                'url': '#EC407A',        # Modern pink
                'control': '#8D6E63',    # Modern brown
                'Unknown': '#9E9E9E'     # Modern gray
            }
            
            # Node colors based on type with modern palette
            node_types = nx.get_node_attributes(G, 'type')
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                node_type = node_types.get(node, 'Unknown')
                color = modern_colors.get(node_type, modern_colors['Unknown'])
                node_colors.append(color)
                
                # Vary node sizes based on type for visual hierarchy
                if node_type in ['host', 'database']:
                    node_sizes.append(2500)
                elif node_type in ['process', 'network']:
                    node_sizes.append(2000)
                elif node_type in ['file', 'user']:
                    node_sizes.append(1800)
                else:
                    node_sizes.append(1500)
            
            # Draw the graph with enhanced styling
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.9, 
                                 edgecolors='#ffffff', linewidths=3)
            
            # Enhanced edges with better styling
            nx.draw_networkx_edges(G, pos, edge_color='#E0E0E0', 
                                 width=2.5, alpha=0.7, 
                                 edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1)
            
            # Enhanced node labels with better typography - use meaningful labels
            labels = {node: G.nodes[node].get('label', str(node)) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', 
                                  font_color='#ffffff', font_family='sans-serif')
            
            # Enhanced edge labels
            edge_labels = nx.get_edge_attributes(G, 'type')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, 
                                       font_color='#FFD700', font_weight='bold')
            
            # Enhanced plot customization
            ax.set_title(title, fontsize=24, fontweight='bold', color='#ffffff', pad=30)
            ax.axis('off')
            
            # Enhanced legend with modern styling
            legend_elements = []
            for node_type in set(node_types.values()):
                color = modern_colors.get(node_type, modern_colors['Unknown'])
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=20, label=node_type,
                              markeredgecolor='#ffffff', markeredgewidth=2)
                )
            
            legend = ax.legend(handles=legend_elements, loc='upper right', 
                             fontsize=14, framealpha=0.9, 
                             facecolor='#2d2d44', edgecolor='#ffffff',
                             title='Entity Types', title_fontsize=16)
            legend.get_title().set_color('#ffffff')
            
            # Add subtle grid for better visual structure
            ax.grid(True, alpha=0.05, color='#ffffff', linestyle='-', linewidth=0.3)
            
            # Add watermark
            fig.text(0.99, 0.01, 'Cybersecurity Agent Agent', fontsize=10, color='#666666', 
                    ha='right', va='bottom', alpha=0.7)
            
            # Save the visualization
            output_path = self._save_visualization(fig, title, 'neo4j_graph')
            
            # Log the visualization creation
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "visualization", f"Created Neo4j graph visualization with {len(G.nodes())} nodes",
                    {'visualization_type': 'neo4j_graph', 'title': title, 'nodes_count': len(G.nodes())}
                )
            
            plt.close()
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating Neo4j graph visualization: {e}")
            return None
    
    def create_vega_lite_visualization(self, data: pd.DataFrame, chart_spec: Dict[str, Any], 
                                      title: str = "Data Visualization") -> Optional[str]:
        """Create a Vega-Lite visualization using Altair."""
        if not self.engines['vega_lite']:
            print("❌ Vega-Lite (Altair) not available")
            return None
        
        try:
            import altair as alt
            
            # Configure Altair for better styling
            alt.themes.enable('dark')
            
            # Create the chart
            chart = alt.Chart(data).mark_circle().encode(
                x=chart_spec.get('x', alt.X('index:Q', title='Index')),
                y=chart_spec.get('y', alt.Y('value:Q', title='Value')),
                color=chart_spec.get('color', alt.Color('category:N', title='Category')),
                size=chart_spec.get('size', alt.Size('size:Q', title='Size')),
                tooltip=chart_spec.get('tooltip', alt.Tooltip('*'))
            ).properties(
                title=title,
                width=800,
                height=500
            ).interactive()
            
            # Save as HTML
            output_path = self._save_vega_lite_html(chart, title)
            
            # Log the visualization creation
            if self.session_manager:
                self.session_manager.log_workflow_step(
                    "visualization", f"Created Vega-Lite visualization: {chart_spec.get('type', 'chart')}",
                    {'visualization_type': 'vega_lite', 'title': title, 'chart_type': chart_spec.get('type', 'chart')}
                )
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating Vega-Lite visualization: {e}")
            return None
    
    def create_vega_lite_chart(self, chart_data: Dict[str, Any], title: str = "Data Chart") -> Optional[str]:
        """Create a Vega-Lite chart from chart data (alias for create_vega_lite_visualization)."""
        try:
            import pandas as pd
            
            # Convert chart data to DataFrame if it's not already
            if isinstance(chart_data, dict) and 'data' in chart_data:
                data = pd.DataFrame(chart_data['data'])
                chart_spec = chart_data
            else:
                # Assume chart_data is already a DataFrame
                data = chart_data if isinstance(chart_data, pd.DataFrame) else pd.DataFrame(chart_data)
                chart_spec = {'type': 'chart'}
            
            return self.create_vega_lite_visualization(data, chart_spec, title)
            
        except Exception as e:
            print(f"❌ Error creating Vega-Lite chart: {e}")
            return None
    
    def _save_visualization(self, fig, title: str, viz_type: str) -> str:
        """Save matplotlib visualization to file."""
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Save as PNG
            png_path = self.visualization_dir / f"{safe_title}_{timestamp}_{viz_type}.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            
            # Save as SVG
            svg_path = self.visualization_dir / f"{safe_title}_{timestamp}_{viz_type}.svg"
            fig.savefig(svg_path, format='svg', bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            
            print(f"✅ Visualization saved: {png_path}")
            print(f"✅ SVG version saved: {svg_path}")
            
            return str(png_path)
            
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
            return ""
    
    def _save_vega_lite_html(self, chart, title: str) -> str:
        """Save Vega-Lite chart as HTML."""
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Save as HTML
            html_path = self.visualization_dir / f"{safe_title}_{timestamp}_vega_lite.html"
            chart.save(str(html_path))
            
            print(f"✅ Vega-Lite HTML saved: {html_path}")
            return str(html_path)
            
        except Exception as e:
            print(f"❌ Error saving Vega-Lite HTML: {e}")
            return ""
    
    def _export_dataframe_html(self, df: pd.DataFrame, title: str):
        """Export DataFrame to HTML with styling."""
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Save as HTML with custom styling
            html_path = self.visualization_dir / f"{safe_title}_{timestamp}_dataframe.html"
            
            # Create styled HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        margin: 0;
                        padding: 20px;
                        color: white;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 15px;
                        padding: 20px;
                        backdrop-filter: blur(10px);
                    }}
                    h1 {{
                        text-align: center;
                        color: #FFD700;
                        margin-bottom: 30px;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                    }}
                    .info {{
                        background: rgba(255, 255, 255, 0.1);
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        border-left: 4px solid #4CAF50;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        background: rgba(255, 255, 255, 0.95);
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    }}
                    th {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px;
                        text-align: left;
                        font-weight: bold;
                    }}
                    td {{
                        padding: 12px;
                        border-bottom: 1px solid #ddd;
                        color: #333;
                    }}
                    tr:nth-child(even) {{
                        background-color: rgba(102, 126, 234, 0.1);
                    }}
                    tr:hover {{
                        background-color: rgba(102, 126, 234, 0.2);
                        transform: scale(1.01);
                        transition: all 0.2s ease;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    <div class="info">
                        <strong>DataFrame Information:</strong><br>
                        Shape: {df.shape[0]} rows × {df.shape[1]} columns<br>
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    {df.to_html(index=False, classes='dataframe')}
                </div>
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ DataFrame HTML exported: {html_path}")
            
        except Exception as e:
            print(f"❌ Error exporting DataFrame HTML: {e}")

def main():
    """Command line interface for visualization management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage cybersecurity visualizations")
    parser.add_argument('--info', action='store_true', help='Show visualization engine status')
    parser.add_argument('--test', action='store_true', help='Test visualization engines')
    
    args = parser.parse_args()
    
    viz_manager = VisualizationManager()
    
    if args.info:
        print("🎨 Visualization Manager Status:")
        for engine, available in viz_manager.engines.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"   {engine}: {status}")
    
    elif args.test:
        print("🧪 Testing visualization engines...")
        
        # Test DataFrame viewer
        if viz_manager.engines['dataframe_viewer']:
            print("✅ DataFrame viewer: Available")
        else:
            print("❌ DataFrame viewer: Not available")
        
        # Test workflow diagram
        if viz_manager.engines['workflow_diagram']:
            print("✅ Workflow diagram: Available")
        else:
            print("❌ Workflow diagram: Not available")
        
        # Test Neo4j visualizer
        if viz_manager.engines['neo4j_visualizer']:
            print("✅ Neo4j visualizer: Available")
        else:
            print("❌ Neo4j visualizer: Not available")
        
        # Test Vega-Lite
        if viz_manager.engines['vega_lite']:
            print("✅ Vega-Lite: Available")
        else:
            print("❌ Vega-Lite: Not available")
    
    else:
        # Default: show info
        print("🎨 Visualization Manager")
        print("Available engines:")
        for engine, available in viz_manager.engines.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"   {engine}: {status}")
        
        print("\n💡 Use --info for detailed status")
        print("   Use --test to test engines")

if __name__ == "__main__":
    main()
