#!/usr/bin/env python3
"""
Test script for enhanced visualizations and JSON Graph Analyzer
"""

import sys
from pathlib import Path
import json

# Add the bin directory to the path
bin_path = Path(__file__).parent / "bin"
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from visualization_manager import VisualizationManager
    from enhanced_session_manager import EnhancedSessionManager
    from json_graph_mcp_tools import JSONGraphMCPTools
    
    print("🔍 Testing enhanced visualization tools and JSON Graph Analyzer...")
    
    # Initialize session manager
    session_manager = EnhancedSessionManager()
    
    # Initialize visualization manager
    viz_manager = VisualizationManager(session_manager)
    
    print("✅ Enhanced visualization manager initialized successfully")
    
    # Test enhanced workflow diagram with better spacing
    try:
        print("\n📋 Testing enhanced workflow diagram...")
        workflow_data = [
            {'id': 'start', 'name': 'Start Analysis', 'type': 'start'},
            {'id': 'collect', 'name': 'Collect Data', 'type': 'process'},
            {'id': 'analyze', 'name': 'Analyze Structure', 'type': 'analysis'},
            {'id': 'decompose', 'name': 'Decompose Document', 'type': 'process'},
            {'id': 'build_graph', 'name': 'Build Graph', 'type': 'process'},
            {'id': 'export', 'name': 'Export Results', 'type': 'output'},
            {'id': 'end', 'name': 'Complete', 'type': 'end'}
        ]
        result = viz_manager.create_workflow_diagram(workflow_data, "Enhanced Cybersecurity Analysis Workflow")
        print(f"✅ Enhanced workflow diagram result: {result}")
        
    except Exception as e:
        print(f"❌ Enhanced workflow diagram failed: {e}")
    
    # Test enhanced Neo4j graph visualization
    try:
        print("\n🕸️ Testing enhanced Neo4j graph visualization...")
        graph_data = {
            'nodes': [
                {'id': '1', 'labels': ['host'], 'properties': {'hostname': 'web-server-01', 'ip': '192.168.1.100'}},
                {'id': '2', 'labels': ['process'], 'properties': {'name': 'nginx', 'pid': '1234'}},
                {'id': '3', 'labels': ['file'], 'properties': {'path': '/var/www/html', 'permissions': '755'}},
                {'id': '4', 'labels': ['network'], 'properties': {'interface': 'eth0', 'subnet': '192.168.1.0/24'}},
                {'id': '5', 'labels': ['user'], 'properties': {'username': 'webadmin', 'role': 'admin'}}
            ],
            'relationships': [
                {'start': {'id': '1'}, 'end': {'id': '2'}, 'type': 'runs'},
                {'start': {'id': '2'}, 'end': {'id': '3'}, 'type': 'accesses'},
                {'start': {'id': '1'}, 'end': {'id': '4'}, 'type': 'connected_to'},
                {'start': {'id': '5'}, 'end': {'id': '1'}, 'type': 'manages'}
            ]
        }
        result = viz_manager.create_neo4j_graph_visualization(graph_data, "Enhanced Cybersecurity Entity Graph")
        print(f"✅ Enhanced Neo4j graph visualization result: {result}")
        
    except Exception as e:
        print(f"❌ Enhanced Neo4j graph visualization failed: {e}")
    
    # Test JSON Graph Analyzer
    try:
        print("\n🔍 Testing JSON Graph Analyzer...")
        
        # Initialize the analyzer
        json_tools = JSONGraphMCPTools()
        
        # Create example cybersecurity documents
        example_app = {
            "app_name": "Web Application",
            "version": "1.0.0",
            "associated_hosts": [
                {"hostname": "web-server-01", "ip": "192.168.1.100", "os": "Ubuntu 20.04"},
                {"hostname": "web-server-02", "ip": "192.168.1.101", "os": "Ubuntu 20.04"}
            ],
            "associated_databases": [
                {"db_name": "app_db", "type": "PostgreSQL", "version": "13.0"}
            ],
            "associated_controls": [
                {"control_id": "AC-1", "name": "Access Control Policy", "status": "implemented"}
            ],
            "metadata": {
                "owner": "IT Department",
                "risk_level": "medium"
            }
        }
        
        example_host = {
            "hostname": "database-server-01",
            "ip_address": "192.168.1.200",
            "os_type": "CentOS 8",
            "associated_ips": [
                {"ip": "192.168.1.200", "interface": "eth0"},
                {"ip": "10.0.0.100", "interface": "eth1"}
            ],
            "associated_controls": [
                {"control_id": "SC-1", "name": "System Configuration", "status": "implemented"},
                {"control_id": "SC-2", "name": "Access Control", "status": "implemented"}
            ],
            "listening_ports": [
                {"port": 5432, "service": "PostgreSQL", "status": "open"},
                {"port": 22, "service": "SSH", "status": "open"}
            ],
            "privileged_users": [
                {"username": "dbadmin", "role": "database_admin"},
                {"username": "sysadmin", "role": "system_admin"}
            ]
        }
        
        print("📊 Analyzing application document structure...")
        result = json_tools.analyze_json_structure(json.dumps(example_app), "application")
        print(f"✅ Structure Analysis: {result['message']}")
        
        print("🔧 Decomposing application document...")
        result = json_tools.decompose_json_document(json.dumps(example_app), "app_001", "application")
        print(f"✅ Document Decomposition: {result['message']}")
        
        print("🔧 Decomposing host document...")
        result = json_tools.decompose_json_document(json.dumps(example_host), "host_001", "host")
        print(f"✅ Host Decomposition: {result['message']}")
        
        print("📈 Getting graph statistics...")
        result = json_tools.get_graph_statistics()
        print(f"✅ Graph Statistics: {result['message']}")
        if result['success']:
            stats = result['statistics']
            print(f"   - Total Nodes: {stats['total_nodes']}")
            print(f"   - Total Edges: {stats['total_edges']}")
            print(f"   - Node Types: {stats['node_types']}")
            print(f"   - Relationship Types: {stats['relationship_types']}")
        
        print("💾 Exporting graph to JSON...")
        result = json_tools.export_graph("json")
        print(f"✅ Graph Export: {result['message']}")
        
        print("💾 Exporting graph to SQLite...")
        result = json_tools.export_graph("sqlite")
        print(f"✅ SQLite Export: {result['message']}")
        
        print("💾 Exporting graph to Cypher...")
        result = json_tools.export_graph("cypher")
        print(f"✅ Cypher Export: {result['message']}")
        
        print("📊 Saving analysis report...")
        result = json_tools.save_analysis_report()
        print(f"✅ Analysis Report: {result['message']}")
        
    except Exception as e:
        print(f"❌ JSON Graph Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Enhanced visualization and JSON Graph Analyzer test completed!")
    
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
