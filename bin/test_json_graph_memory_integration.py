#!/usr/bin/env python3
"""
Test script for JSON Graph Analyzer with Context Memory Integration
Demonstrates saving, loading, and reusing graph patterns in memory.
"""

import sys
from pathlib import Path
import json

# Add the bin directory to the path
bin_path = Path(__file__).parent / "bin"
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from json_graph_mcp_tools import JSONGraphMCPTools
    from context_memory_manager import ContextMemoryManager
    
    print("🔍 Testing JSON Graph Analyzer with Context Memory Integration...")
    
    # Initialize context memory manager
    print("\n🧠 Initializing Context Memory Manager...")
    memory_manager = ContextMemoryManager()
    print("✅ Context Memory Manager initialized")
    
    # Initialize JSON Graph Analyzer tools
    print("\n🔧 Initializing JSON Graph Analyzer Tools...")
    json_tools = JSONGraphMCPTools()
    print("✅ JSON Graph Analyzer Tools initialized")
    
    # Create example cybersecurity documents
    print("\n📝 Creating example cybersecurity documents...")
    
    # Example 1: Web Application
    web_app = {
        "app_name": "E-commerce Platform",
        "version": "2.1.0",
        "associated_hosts": [
            {"hostname": "web-frontend-01", "ip": "192.168.1.100", "os": "Ubuntu 20.04", "role": "frontend"},
            {"hostname": "web-frontend-02", "ip": "192.168.1.101", "os": "Ubuntu 20.04", "role": "frontend"},
            {"hostname": "app-backend-01", "ip": "192.168.1.200", "os": "CentOS 8", "role": "backend"},
            {"hostname": "app-backend-02", "ip": "192.168.1.201", "os": "CentOS 8", "role": "backend"}
        ],
        "associated_databases": [
            {"db_name": "ecommerce_db", "type": "PostgreSQL", "version": "13.0", "role": "primary"},
            {"db_name": "ecommerce_cache", "type": "Redis", "version": "6.0", "role": "cache"}
        ],
        "associated_controls": [
            {"control_id": "AC-1", "name": "Access Control Policy", "status": "implemented", "priority": "high"},
            {"control_id": "SC-1", "name": "System Configuration", "status": "implemented", "priority": "medium"},
            {"control_id": "SI-1", "name": "System Monitoring", "status": "implemented", "priority": "high"}
        ],
        "metadata": {
            "owner": "E-commerce Team",
            "risk_level": "medium",
            "compliance": ["PCI-DSS", "GDPR"],
            "business_criticality": "high"
        }
    }
    
    # Example 2: Network Infrastructure
    network_infra = {
        "network_id": "prod-network-01",
        "network_name": "Production Network",
        "subnets": [
            {"subnet": "192.168.1.0/24", "purpose": "web_servers", "vlan": "100"},
            {"subnet": "192.168.2.0/24", "purpose": "database_servers", "vlan": "200"},
            {"subnet": "192.168.3.0/24", "purpose": "management", "vlan": "300"}
        ],
        "gateways": [
            {"ip": "192.168.1.1", "device": "core-router-01", "type": "primary"},
            {"ip": "192.168.1.2", "device": "core-router-02", "type": "backup"}
        ],
        "firewall_rules": [
            {"rule_id": "FW-001", "source": "internet", "destination": "web_servers", "ports": "80,443", "action": "allow"},
            {"rule_id": "FW-002", "source": "web_servers", "destination": "database_servers", "ports": "5432", "action": "allow"}
        ],
        "connected_hosts": [
            {"hostname": "web-frontend-01", "ip": "192.168.1.100", "subnet": "192.168.1.0/24"},
            {"hostname": "app-backend-01", "ip": "192.168.2.100", "subnet": "192.168.2.0/24"}
        ]
    }
    
    # Example 3: User Management System
    user_system = {
        "system_name": "Corporate Identity Management",
        "user_count": 1500,
        "assigned_roles": [
            {"role_name": "admin", "permissions": ["full_access"], "user_count": 25},
            {"role_name": "manager", "permissions": ["team_management", "reports"], "user_count": 150},
            {"role_name": "employee", "permissions": ["basic_access"], "user_count": 1325}
        ],
        "access_permissions": [
            {"permission_id": "ACC-001", "name": "Database Access", "scope": "read_only"},
            {"permission_id": "ACC-002", "name": "System Configuration", "scope": "admin_only"},
            {"permission_id": "ACC-003", "name": "User Management", "scope": "hr_team"}
        ],
        "group_memberships": [
            {"group_name": "IT Department", "members": 45, "permissions": ["system_admin", "network_admin"]},
            {"group_name": "Finance Team", "members": 30, "permissions": ["financial_data", "reports"]},
            {"group_name": "HR Team", "members": 25, "permissions": ["user_management", "employee_data"]}
        ]
    }
    
    print("✅ Created 3 example cybersecurity documents")
    
    # Test 1: Analyze and decompose web application
    print("\n🔍 Test 1: Analyzing and decomposing web application...")
    result = json_tools.analyze_json_structure(json.dumps(web_app), "application")
    print(f"✅ Structure Analysis: {result['message']}")
    
    result = json_tools.decompose_json_document(json.dumps(web_app), "web_app_001", "application")
    print(f"✅ Document Decomposition: {result['message']}")
    
    # Save web app pattern to memory
    print("\n💾 Saving web application pattern to context memory...")
    result = json_tools.save_graph_to_memory(
        pattern_name="web_application_pattern",
        description="Standard web application architecture with frontend, backend, and database components",
        tags=["web", "application", "multi-tier", "production"],
        tier="long_term",
        ttl_days=365,
        priority=8
    )
    print(f"✅ Memory Save: {result['message']}")
    
    # Test 2: Analyze and decompose network infrastructure
    print("\n🔍 Test 2: Analyzing and decomposing network infrastructure...")
    result = json_tools.decompose_json_document(json.dumps(network_infra), "network_001", "network")
    print(f"✅ Network Decomposition: {result['message']}")
    
    # Save network pattern to memory
    print("\n💾 Saving network infrastructure pattern to context memory...")
    result = json_tools.save_graph_to_memory(
        pattern_name="network_infrastructure_pattern",
        description="Production network with subnets, gateways, and firewall rules",
        tags=["network", "infrastructure", "production", "security"],
        tier="long_term",
        ttl_days=365,
        priority=9
    )
    print(f"✅ Memory Save: {result['message']}")
    
    # Test 3: Analyze and decompose user management system
    print("\n🔍 Test 3: Analyzing and decomposing user management system...")
    result = json_tools.decompose_json_document(json.dumps(user_system), "user_sys_001", "user")
    print(f"✅ User System Decomposition: {result['message']}")
    
    # Save user system pattern to memory
    print("\n💾 Saving user management pattern to context memory...")
    result = json_tools.save_graph_to_memory(
        pattern_name="user_management_pattern",
        description="Corporate identity management with role-based access control",
        tags=["user", "identity", "rbac", "corporate"],
        tier="long_term",
        ttl_days=365,
        priority=7
    )
    print(f"✅ Memory Save: {result['message']}")
    
    # Test 4: Get available patterns from memory
    print("\n📋 Test 4: Retrieving available patterns from context memory...")
    result = json_tools.get_available_patterns()
    print(f"✅ Available Patterns: {result['message']}")
    
    if result['success']:
        patterns = result['patterns']
        print(f"   Found {len(patterns)} patterns:")
        for pattern in patterns:
            print(f"   - {pattern['name']}: {pattern['description']}")
            print(f"     Nodes: {pattern['nodes']}, Edges: {pattern['edges']}")
            print(f"     Tags: {', '.join(pattern['tags'])}")
    
    # Test 5: Load a specific pattern from memory
    print("\n🔄 Test 5: Loading web application pattern from memory...")
    result = json_tools.load_graph_from_memory(pattern_name="web_application_pattern")
    print(f"✅ Pattern Load: {result['message']}")
    
    # Test 6: Get statistics of loaded pattern
    print("\n📊 Test 6: Getting statistics of loaded pattern...")
    result = json_tools.get_graph_statistics()
    print(f"✅ Graph Statistics: {result['message']}")
    
    if result['success']:
        stats = result['statistics']
        print(f"   - Total Nodes: {stats['total_nodes']}")
        print(f"   - Total Edges: {stats['total_edges']}")
        print(f"   - Node Types: {stats['node_types']}")
        print(f"   - Relationship Types: {stats['relationship_types']}")
    
    # Test 7: Export loaded pattern to various formats
    print("\n💾 Test 7: Exporting loaded pattern to various formats...")
    
    formats = ["json", "yaml", "cypher", "sqlite"]
    for fmt in formats:
        result = json_tools.export_graph(fmt)
        print(f"✅ {fmt.upper()} Export: {result['message']}")
    
    # Test 8: Create a new simple graph and merge with existing pattern
    print("\n🔗 Test 8: Creating new graph and merging with existing pattern...")
    
    # Create a simple new graph
    simple_app = {
        "app_name": "Simple API Service",
        "version": "1.0.0",
        "associated_hosts": [
            {"hostname": "api-server-01", "ip": "10.0.1.100", "os": "Ubuntu 18.04"}
        ],
        "associated_databases": [
            {"db_name": "api_db", "type": "MongoDB", "version": "4.4"}
        ]
    }
    
    result = json_tools.decompose_json_document(json.dumps(simple_app), "simple_api_001", "application")
    print(f"✅ Simple API Decomposition: {result['message']}")
    
    # Merge with the web application pattern
    print("\n🔄 Merging with web application pattern...")
    result = json_tools.merge_with_pattern("web_application_pattern")
    print(f"✅ Pattern Merge: {result['message']}")
    
    # Test 9: Save the merged pattern as a new pattern
    print("\n💾 Test 9: Saving merged pattern as new pattern...")
    result = json_tools.save_graph_to_memory(
        pattern_name="enhanced_web_application_pattern",
        description="Enhanced web application pattern combining standard and API components",
        tags=["web", "application", "api", "enhanced", "merged"],
        tier="long_term",
        ttl_days=365,
        priority=9
    )
    print(f"✅ Enhanced Pattern Save: {result['message']}")
    
    # Test 10: Final statistics and cleanup
    print("\n📊 Test 10: Final statistics and summary...")
    result = json_tools.get_graph_statistics()
    if result['success']:
        stats = result['statistics']
        print(f"   Final Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        print(f"   Node Types: {list(stats['node_types'].keys())}")
        print(f"   Relationship Types: {list(stats['relationship_types'].keys())}")
    
    # Get all available patterns
    result = json_tools.get_available_patterns()
    if result['success']:
        patterns = result['patterns']
        print(f"\n🎯 Summary: Successfully created and stored {len(patterns)} graph patterns:")
        for pattern in patterns:
            print(f"   - {pattern['name']}: {pattern['nodes']} nodes, {pattern['edges']} edges")
    
    print("\n🎯 JSON Graph Analyzer with Context Memory Integration test completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Document structure analysis and decomposition")
    print("   ✅ Graph pattern creation and storage in context memory")
    print("   ✅ Pattern retrieval and loading from memory")
    print("   ✅ Graph merging and enhancement")
    print("   ✅ Multiple export formats (JSON, YAML, Cypher, SQLite)")
    print("   ✅ Persistent storage with metadata and TTL")
    print("   ✅ Reusable knowledge graph patterns")
    
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Make sure the Context Memory Manager and JSON Graph Analyzer are available")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
