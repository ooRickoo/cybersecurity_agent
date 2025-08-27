#!/usr/bin/env python3
"""
JSON Graph Analyzer - Comprehensive tool for decomposing multidimensional JSON documents
into graph representations with SQLite and graph database support.
"""

import json
import sqlite3
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import yaml
import uuid
from dataclasses import dataclass, asdict
import logging
import sys

# Add the bin directory to the path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

@dataclass
class EntityNode:
    """Represents an entity node in the graph."""
    id: str
    type: str
    properties: Dict[str, Any]
    source_document: str
    created_at: str
    metadata: Dict[str, Any] = None

@dataclass
class EntityRelationship:
    """Represents a relationship between entities."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    source_document: str
    created_at: str

class JSONGraphAnalyzer:
    """Comprehensive JSON document decomposition and graph analysis tool."""
    
    def __init__(self, output_dir: str = "session-outputs/graph_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        
        # Graph metadata storage for reuse
        self.graph_metadata = {}
        self.saved_graphs = {}  # Store saved graph representations
        
        # Entity type mappings for cybersecurity domains
        self.entity_type_mappings = {
            # Application entities
            'application': {
                'nested_arrays': ['associated_hosts', 'associated_ips', 'associated_locations', 
                                'associated_datacenters', 'associated_databases', 'associated_controls',
                                'associated_urls', 'associated_networks', 'associated_tags'],
                'nested_objects': ['metadata', 'configuration', 'security_profile']
            },
            # Host entities
            'host': {
                'nested_arrays': ['associated_ips', 'associated_controls', 'listening_ports', 
                                'privileged_users', 'installed_software', 'network_interfaces',
                                'security_events', 'vulnerabilities'],
                'nested_objects': ['system_info', 'network_config', 'security_status']
            },
            # Network entities
            'network': {
                'nested_arrays': ['subnets', 'gateways', 'dns_servers', 'dhcp_servers',
                                'connected_hosts', 'routing_rules', 'firewall_rules'],
                'nested_objects': ['network_config', 'security_policies']
            },
            # User entities
            'user': {
                'nested_arrays': ['assigned_roles', 'access_permissions', 'login_history',
                                'associated_devices', 'group_memberships'],
                'nested_objects': ['profile', 'security_clearance']
            },
            # Database entities
            'database': {
                'nested_arrays': ['tables', 'users', 'backup_schedules', 'replication_configs'],
                'nested_objects': ['connection_info', 'security_settings']
            },
            # Control entities
            'control': {
                'nested_arrays': ['implemented_by', 'tested_by', 'monitored_by'],
                'nested_objects': ['specification', 'implementation_details']
            }
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_document_structure(self, json_data: Dict[str, Any], 
                                 document_type: str = None) -> Dict[str, Any]:
        """Analyze the structure of a JSON document and identify components."""
        analysis = {
            'document_type': document_type or self._infer_document_type(json_data),
            'main_entity': {},
            'nested_arrays': {},
            'nested_objects': {},
            'relationships': [],
            'metadata': {
                'total_fields': 0,
                'array_fields': 0,
                'object_fields': 0,
                'primitive_fields': 0
            }
        }
        
        # Analyze main document structure
        for key, value in json_data.items():
            analysis['metadata']['total_fields'] += 1
            
            if isinstance(value, list):
                analysis['metadata']['array_fields'] += 1
                analysis['nested_arrays'][key] = {
                    'type': 'array',
                    'count': len(value),
                    'sample_items': value[:3] if len(value) > 3 else value,
                    'item_types': list(set(type(item).__name__ for item in value)) if value else []
                }
            elif isinstance(value, dict):
                analysis['metadata']['object_fields'] += 1
                analysis['nested_objects'][key] = {
                    'type': 'object',
                    'fields': list(value.keys()),
                    'field_count': len(value)
                }
            else:
                analysis['metadata']['primitive_fields'] += 1
                analysis['main_entity'][key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]  # Truncate long values
                }
        
        return analysis
    
    def _infer_document_type(self, json_data: Dict[str, Any]) -> str:
        """Infer the document type based on content."""
        # Look for common cybersecurity entity indicators
        indicators = {
            'application': ['app_name', 'application_name', 'software', 'version'],
            'host': ['hostname', 'ip_address', 'os_type', 'system_info'],
            'network': ['network_id', 'subnet', 'cidr', 'routing'],
            'user': ['username', 'user_id', 'email', 'role'],
            'database': ['db_name', 'connection_string', 'schema', 'tables'],
            'control': ['control_id', 'control_name', 'framework', 'compliance']
        }
        
        for doc_type, fields in indicators.items():
            if any(field in json_data for field in fields):
                return doc_type
        
        return 'unknown'
    
    def decompose_document(self, json_data: Dict[str, Any], 
                          document_id: str = None,
                          document_type: str = None) -> Dict[str, Any]:
        """Decompose a JSON document into its components."""
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        if document_type is None:
            document_type = self._infer_document_type(json_data)
        
        decomposition = {
            'document_id': document_id,
            'document_type': document_type,
            'main_entity': {},
            'nested_entities': [],
            'relationships': [],
            'metadata': {
                'decomposed_at': datetime.now().isoformat(),
                'source_type': 'json'
            }
        }
        
        # Extract main entity properties
        main_entity_props = {}
        for key, value in json_data.items():
            if not isinstance(value, (list, dict)):
                main_entity_props[key] = value
        
        # Create main entity node
        main_entity = EntityNode(
            id=document_id,
            type=document_type,
            properties=main_entity_props,
            source_document=document_id,
            created_at=datetime.now().isoformat(),
            metadata={'is_main_entity': True}
        )
        
        decomposition['main_entity'] = asdict(main_entity)
        
        # Process nested arrays and objects
        entity_counter = 0
        for key, value in json_data.items():
            if isinstance(value, list):
                # Process array items
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        entity_id = f"{document_id}_{key}_{i}"
                        nested_entity = EntityNode(
                            id=entity_id,
                            type=key.rstrip('s'),  # Remove plural
                            properties=item,
                            source_document=document_id,
                            created_at=datetime.now().isoformat(),
                            metadata={'array_key': key, 'array_index': i}
                        )
                        
                        decomposition['nested_entities'].append(asdict(nested_entity))
                        
                        # Create relationship
                        relationship = EntityRelationship(
                            id=f"{document_id}_rel_{entity_counter}",
                            source_id=document_id,
                            target_id=entity_id,
                            type=f"has_{key.rstrip('s')}",
                            properties={'array_key': key, 'array_index': i},
                            source_document=document_id,
                            created_at=datetime.now().isoformat()
                        )
                        
                        decomposition['relationships'].append(asdict(relationship))
                        entity_counter += 1
                        
            elif isinstance(value, dict):
                # Process nested object
                entity_id = f"{document_id}_{key}"
                nested_entity = EntityNode(
                    id=entity_id,
                    type=key,
                    properties=value,
                    source_document=document_id,
                    created_at=datetime.now().isoformat(),
                    metadata={'object_key': key}
                )
                
                decomposition['nested_entities'].append(asdict(nested_entity))
                
                # Create relationship
                relationship = EntityRelationship(
                    id=f"{document_id}_rel_{entity_counter}",
                    source_id=document_id,
                    target_id=entity_id,
                    type=f"has_{key}",
                    properties={'object_key': key},
                    source_document=document_id,
                    created_at=datetime.now().isoformat()
                )
                
                decomposition['relationships'].append(asdict(relationship))
                entity_counter += 1
        
        return decomposition
    
    def build_graph_from_decomposition(self, decomposition: Dict[str, Any]) -> nx.MultiDiGraph:
        """Build a NetworkX graph from document decomposition."""
        # Add main entity node
        main_entity = decomposition['main_entity']
        self.graph.add_node(
            main_entity['id'],
            **{k: v for k, v in main_entity.items() if k != 'id'}
        )
        
        # Add nested entity nodes
        for entity in decomposition['nested_entities']:
            self.graph.add_node(
                entity['id'],
                **{k: v for k, v in entity.items() if k != 'id'}
            )
        
        # Add relationships
        for rel in decomposition['relationships']:
            self.graph.add_edge(
                rel['source_id'],
                rel['target_id'],
                **{k: v for k, v in rel.items() if k not in ['source_id', 'target_id']}
            )
        
        return self.graph
    
    def export_to_sqlite(self, db_path: str = None) -> str:
        """Export the graph to SQLite database."""
        if db_path is None:
            db_path = self.output_dir / f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT,
                properties TEXT,
                source_document TEXT,
                created_at TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                type TEXT,
                properties TEXT,
                source_document TEXT,
                created_at TEXT,
                FOREIGN KEY (source_id) REFERENCES entities (id),
                FOREIGN KEY (target_id) REFERENCES entities (id)
            )
        ''')
        
        # Insert entities
        for node_id, node_data in self.graph.nodes(data=True):
            cursor.execute('''
                INSERT OR REPLACE INTO entities 
                (id, type, properties, source_document, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                node_data.get('type', 'unknown'),
                json.dumps(node_data.get('properties', {})),
                node_data.get('source_document', ''),
                node_data.get('created_at', ''),
                json.dumps(node_data.get('metadata', {}))
            ))
        
        # Insert relationships
        for source, target, edge_data in self.graph.edges(data=True):
            cursor.execute('''
                INSERT OR REPLACE INTO relationships
                (id, source_id, target_id, type, properties, source_document, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge_data.get('id', f"{source}_{target}"),
                source,
                target,
                edge_data.get('type', 'relates_to'),
                json.dumps(edge_data.get('properties', {})),
                edge_data.get('source_document', ''),
                edge_data.get('created_at', '')
            ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Graph exported to SQLite: {db_path}")
        return str(db_path)
    
    def export_to_graphml(self, file_path: str = None) -> str:
        """Export the graph to GraphML format."""
        if file_path is None:
            file_path = self.output_dir / f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml"
        
        nx.write_graphml(self.graph, file_path)
        self.logger.info(f"Graph exported to GraphML: {file_path}")
        return str(file_path)
    
    def export_to_cypher(self, file_path: str = None) -> str:
        """Export the graph to Cypher queries for Neo4j."""
        if file_path is None:
            file_path = self.output_dir / f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cypher"
        
        cypher_queries = []
        
        # Create constraints and indexes
        cypher_queries.append("// Create constraints and indexes")
        cypher_queries.append("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;")
        cypher_queries.append("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);")
        cypher_queries.append("")
        
        # Create entities
        cypher_queries.append("// Create entities")
        for node_id, node_data in self.graph.nodes(data=True):
            properties = node_data.get('properties', {})
            properties_str = ', '.join([f"{k}: {repr(v)}" for k, v in properties.items()])
            
            query = f"CREATE (e:Entity {{id: '{node_id}', type: '{node_data.get('type', 'unknown')}'"
            if properties_str:
                query += f", {properties_str}"
            query += "});"
            cypher_queries.append(query)
        
        cypher_queries.append("")
        
        # Create relationships
        cypher_queries.append("// Create relationships")
        for source, target, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('type', 'RELATES_TO').upper()
            properties = edge_data.get('properties', {})
            properties_str = ', '.join([f"{k}: {repr(v)}" for k, v in properties.items()])
            
            query = f"MATCH (a:Entity {{id: '{source}'}}), (b:Entity {{id: '{target}'}})"
            query += f" CREATE (a)-[r:{rel_type}"
            if properties_str:
                query += f" {{{properties_str}}}"
            query += "]->(b);"
            cypher_queries.append(query)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(cypher_queries))
        
        self.logger.info(f"Graph exported to Cypher: {file_path}")
        return str(file_path)
    
    def export_to_json(self, file_path: str = None) -> str:
        """Export the graph to JSON format."""
        if file_path is None:
            file_path = self.output_dir / f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        graph_data = {
            'nodes': [],
            'relationships': [],
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_nodes': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges()
            }
        }
        
        # Export nodes
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node_id,
                **node_data
            })
        
        # Export relationships
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data['relationships'].append({
                'source': source,
                'target': target,
                **edge_data
            })
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        self.logger.info(f"Graph exported to JSON: {file_path}")
        return str(file_path)
    
    def export_to_yaml(self, file_path: str = None) -> str:
        """Export the graph to YAML format."""
        if file_path is None:
            file_path = self.output_dir / f"graph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        graph_data = {
            'nodes': [],
            'relationships': [],
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_nodes': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges()
            }
        }
        
        # Export nodes
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node_id,
                **node_data
            })
        
        # Export relationships
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data['relationships'].append({
                'source': source,
                'target': target,
                **edge_data
            })
        
        # Write to file
        with open(file_path, 'w') as f:
            yaml.dump(graph_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Graph exported to YAML: {file_path}")
        return str(file_path)
    
    def load_from_json(self, file_path: str) -> nx.MultiDiGraph:
        """Load a graph from JSON file."""
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for node in graph_data['nodes']:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)
        
        # Add relationships
        for rel in graph_data['relationships']:
            source = rel.pop('source')
            target = rel.pop('target')
            self.graph.add_edge(source, target, **rel)
        
        self.logger.info(f"Graph loaded from JSON: {file_path}")
        return self.graph
    
    def load_from_yaml(self, file_path: str) -> nx.MultiDiGraph:
        """Load a graph from YAML file."""
        with open(file_path, 'r') as f:
            graph_data = yaml.safe_load(f)
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for node in graph_data['nodes']:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)
        
        # Add relationships
        for rel in graph_data['relationships']:
            source = rel.pop('source')
            target = rel.pop('target')
            self.graph.add_edge(source, target, **rel)
        
        self.logger.info(f"Graph loaded from YAML: {file_path}")
        return self.graph
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph."""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'relationship_types': {},
            'source_documents': set(),
            'largest_connected_component': 0,
            'density': 0.0,
            'average_clustering': 0.0
        }
        
        # Analyze node types
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            stats['source_documents'].add(node_data.get('source_document', 'unknown'))
        
        # Analyze relationship types
        for source, target, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('type', 'unknown')
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        # Convert sets to lists for JSON serialization
        stats['source_documents'] = list(stats['source_documents'])
        
        # Calculate graph metrics
        if self.graph.number_of_nodes() > 0:
            try:
                # Largest connected component
                largest_cc = max(nx.connected_components(self.graph.to_undirected()), key=len)
                stats['largest_connected_component'] = len(largest_cc)
                
                # Graph density
                stats['density'] = nx.density(self.graph)
                
                # Average clustering coefficient
                stats['average_clustering'] = nx.average_clustering(self.graph.to_undirected())
            except:
                pass
        
        return stats
    
    def query_entities(self, entity_type: str = None, 
                      source_document: str = None,
                      properties_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query entities based on filters."""
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Apply filters
            if entity_type and node_data.get('type') != entity_type:
                continue
            
            if source_document and node_data.get('source_document') != source_document:
                continue
            
            if properties_filter:
                node_props = node_data.get('properties', {})
                if not all(node_props.get(k) == v for k, v in properties_filter.items()):
                    continue
            
            results.append({
                'id': node_id,
                **node_data
            })
        
        return results
    
    def find_relationships(self, source_id: str = None, 
                          target_id: str = None,
                          relationship_type: str = None) -> List[Dict[str, Any]]:
        """Find relationships based on filters."""
        results = []
        
        for source, target, edge_data in self.graph.edges(data=True):
            # Apply filters
            if source_id and source != source_id:
                continue
            
            if target_id and target != target_id:
                continue
            
            if relationship_type and edge_data.get('type') != relationship_type:
                continue
            
            results.append({
                'source': source,
                'target': target,
                **edge_data
            })
        
        return results
    
    def get_entity_neighborhood(self, entity_id: str, 
                               max_depth: int = 2) -> Dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth."""
        if entity_id not in self.graph:
            return {}
        
        neighborhood = {
            'entity': dict(self.graph.nodes[entity_id]),
            'neighbors': [],
            'relationships': []
        }
        
        # Get immediate neighbors
        neighbors = list(self.graph.neighbors(entity_id))
        neighborhood['neighbors'] = [
            {
                'id': neighbor_id,
                **dict(self.graph.nodes[neighbor_id])
            }
            for neighbor_id in neighbors
        ]
        
        # Get relationships
        for neighbor_id in neighbors:
            edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
            for edge_key, edge_props in edge_data.items():
                neighborhood['relationships'].append({
                    'source': entity_id,
                    'target': neighbor_id,
                    **edge_props
                })
        
        return neighborhood
    
    def merge_graphs(self, other_analyzer: 'JSONGraphAnalyzer') -> None:
        """Merge another graph analyzer's graph into this one."""
        # Add nodes
        for node_id, node_data in other_analyzer.graph.nodes(data=True):
            if node_id not in self.graph:
                self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for source, target, edge_data in other_analyzer.graph.edges(data=True):
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, **edge_data)
        
        self.logger.info(f"Merged graph with {other_analyzer.graph.number_of_nodes()} nodes and {other_analyzer.graph.number_of_edges()} edges")
    
    def save_analysis_report(self, file_path: str = None) -> str:
        """Save a comprehensive analysis report."""
        if file_path is None:
            file_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        stats = self.get_graph_statistics()
        
        report = f"""# Graph Analysis Report

## Overview
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Nodes**: {stats['total_nodes']}
- **Total Edges**: {stats['total_edges']}
- **Source Documents**: {len(stats['source_documents'])}

## Node Types
"""
        
        for node_type, count in stats['node_types'].items():
            report += f"- **{node_type}**: {count}\n"
        
        report += f"""
## Relationship Types
"""
        
        for rel_type, count in stats['relationship_types'].items():
            report += f"- **{rel_type}**: {count}\n"
        
        report += f"""
## Graph Metrics
- **Density**: {stats['density']:.4f}
- **Largest Connected Component**: {stats['largest_connected_component']}
- **Average Clustering**: {stats['average_clustering']:.4f}

## Source Documents
"""
        
        for doc in stats['source_documents']:
            report += f"- {doc}\n"
        
        # Write report
        with open(file_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Analysis report saved: {file_path}")
        return str(file_path)
    
    def save_to_context_memory(self, memory_manager=None, pattern_name: str = None, 
                              description: str = None, tags: List[str] = None,
                              tier: str = "medium_term", ttl_days: int = 90,
                              priority: int = 5) -> Optional[str]:
        """Save the current graph as a reusable pattern in context memory."""
        try:
            if memory_manager is None:
                # Try to import and initialize memory manager
                try:
                    from context_memory_manager import ContextMemoryManager, MemoryDomain, MemoryTier
                    memory_manager = ContextMemoryManager()
                except ImportError:
                    self.logger.warning("Context Memory Manager not available")
                    return None
            
            # Prepare graph data for storage
            graph_data = {
                'nodes': [],
                'relationships': [],
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_relationships': self.graph.number_of_edges(),
                    'pattern_name': pattern_name or f"graph_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'description': description or f"Graph pattern with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges",
                    'source_documents': list(set(node_data.get('source_document', 'unknown') 
                                              for node_id, node_data in self.graph.nodes(data=True))),
                    'node_types': dict(self.get_graph_statistics()['node_types']),
                    'relationship_types': dict(self.get_graph_statistics()['relationship_types'])
                }
            }
            
            # Export nodes
            for node_id, node_data in self.graph.nodes(data=True):
                graph_data['nodes'].append({
                    'id': node_id,
                    **node_data
                })
            
            # Export relationships
            for source, target, edge_data in self.graph.edges(data=True):
                graph_data['relationships'].append({
                    'source': source,
                    'target': target,
                    **edge_data
                })
            
            # Determine memory domain based on content
            domain = self._determine_memory_domain(graph_data)
            
            # Determine memory tier
            memory_tier = MemoryTier.MEDIUM_TERM
            if tier == "short_term":
                memory_tier = MemoryTier.SHORT_TERM
            elif tier == "long_term":
                memory_tier = MemoryTier.LONG_TERM
            
            # Prepare tags
            if tags is None:
                tags = []
            tags.extend([
                'graph_pattern',
                'reusable',
                f"nodes_{self.graph.number_of_nodes()}",
                f"edges_{self.graph.number_of_edges()}"
            ])
            
            # Import into context memory
            memory_id = memory_manager.import_data(
                domain=domain,
                data=graph_data,
                source=f"json_graph_analyzer_{pattern_name or 'pattern'}",
                tier=memory_tier,
                ttl_days=ttl_days,
                tags=tags,
                description=description or f"Reusable graph pattern: {pattern_name or 'Unnamed Pattern'}",
                priority=priority
            )
            
            self.logger.info(f"Graph pattern saved to context memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to save graph to context memory: {e}")
            return None
    
    def _determine_memory_domain(self, graph_data: Dict[str, Any]) -> 'MemoryDomain':
        """Determine the appropriate memory domain for the graph data."""
        try:
            from context_memory_manager import MemoryDomain
            
            # Analyze node types to determine domain
            node_types = graph_data['metadata']['node_types']
            
            # Check for specific cybersecurity domains
            if any('host' in node_type.lower() for node_type in node_types):
                return MemoryDomain.HOST_INVENTORY
            elif any('application' in node_type.lower() or 'app' in node_type.lower() for node_type in node_types):
                return MemoryDomain.APPLICATION_INVENTORY
            elif any('user' in node_type.lower() for node_type in node_types):
                return MemoryDomain.USER_INVENTORY
            elif any('network' in node_type.lower() for node_type in node_types):
                return MemoryDomain.NETWORK_INVENTORY
            elif any('threat' in node_type.lower() or 'ioc' in node_type.lower() for node_type in node_types):
                return MemoryDomain.THREAT_INTELLIGENCE
            elif any('control' in node_type.lower() or 'policy' in node_type.lower() for node_type in node_types):
                return MemoryDomain.GRC_POLICIES
            else:
                # Default to graph patterns for generic graphs
                return MemoryDomain.GRAPH_PATTERNS
                
        except ImportError:
            # Fallback if memory manager not available
            return None
    
    def load_from_context_memory(self, memory_manager=None, memory_id: str = None,
                               pattern_name: str = None, domain: str = None) -> bool:
        """Load a graph pattern from context memory."""
        try:
            if memory_manager is None:
                # Try to import and initialize memory manager
                try:
                    from context_memory_manager import ContextMemoryManager
                    memory_manager = ContextMemoryManager()
                except ImportError:
                    self.logger.warning("Context Memory Manager not available")
                    return False
            
            # Search for graph patterns
            if memory_id:
                # Load specific memory entry
                memory_entry = memory_manager.get_memory_entry(memory_id)
                if memory_entry and memory_entry.data:
                    return self._load_graph_from_memory_data(memory_entry.data)
            else:
                # Search for patterns by name or domain
                search_results = memory_manager.retrieve_context(
                    query=pattern_name or "graph pattern",
                    domains=[memory_manager._get_domain_from_string(domain or "graph_patterns")],
                    max_results=10
                )
                
                if search_results:
                    # Load the first matching pattern
                    for result in search_results:
                        if result.get('data') and isinstance(result['data'], dict):
                            if self._load_graph_from_memory_data(result['data']):
                                self.logger.info(f"Loaded graph pattern from memory: {result.get('id', 'Unknown')}")
                                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load graph from context memory: {e}")
            return False
    
    def _load_graph_from_memory_data(self, memory_data: Dict[str, Any]) -> bool:
        """Load graph from memory data structure."""
        try:
            if not isinstance(memory_data, dict):
                return False
            
            # Clear existing graph
            self.graph.clear()
            
            # Load nodes
            if 'nodes' in memory_data:
                for node in memory_data['nodes']:
                    node_id = node.pop('id')
                    self.graph.add_node(node_id, **node)
            
            # Load relationships
            if 'relationships' in memory_data:
                for rel in memory_data['relationships']:
                    source = rel.pop('source')
                    target = rel.pop('target')
                    self.graph.add_edge(source, target, **rel)
            
            self.logger.info(f"Successfully loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load graph from memory data: {e}")
            return False
    
    def get_available_patterns(self, memory_manager=None, domain: str = None) -> List[Dict[str, Any]]:
        """Get list of available graph patterns from context memory."""
        try:
            if memory_manager is None:
                # Try to import and initialize memory manager
                try:
                    from context_memory_manager import ContextMemoryManager
                    memory_manager = ContextMemoryManager()
                except ImportError:
                    self.logger.warning("Context Memory Manager not available")
                    return []
            
            # Search for graph patterns
            search_results = memory_manager.search_memory(
                query="graph pattern",
                domain=domain or "graph_patterns",
                limit=50
            )
            
            patterns = []
            for result in search_results:
                if result.get('data') and isinstance(result['data'], dict):
                    metadata = result['data'].get('metadata', {})
                    patterns.append({
                        'id': result.get('id', 'Unknown'),
                        'name': metadata.get('pattern_name', 'Unnamed Pattern'),
                        'description': metadata.get('description', ''),
                        'nodes': metadata.get('total_nodes', 0),
                        'edges': metadata.get('total_relationships', 0),
                        'node_types': metadata.get('node_types', {}),
                        'source_documents': metadata.get('source_documents', []),
                        'created_at': result.get('import_timestamp', ''),
                        'tags': result.get('tags', [])
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to get available patterns: {e}")
            return []
    
    def merge_with_pattern(self, pattern_name: str, memory_manager=None) -> bool:
        """Merge the current graph with a stored pattern from memory."""
        try:
            if memory_manager is None:
                # Try to import and initialize memory manager
                try:
                    from context_memory_manager import ContextMemoryManager
                    memory_manager = ContextMemoryManager()
                except ImportError:
                    self.logger.warning("Context Memory Manager not available")
                    return False
            
            # Search for the specific pattern
            search_results = memory_manager.search_memory(
                query=pattern_name,
                domain="graph_patterns",
                limit=5
            )
            
            if search_results:
                for result in search_results:
                    if result.get('data') and isinstance(result['data'], dict):
                        metadata = result['data'].get('metadata', {})
                        if metadata.get('pattern_name') == pattern_name:
                            # Create temporary analyzer to load the pattern
                            temp_analyzer = JSONGraphAnalyzer()
                            if temp_analyzer._load_graph_from_memory_data(result['data']):
                                # Merge the graphs
                                self.merge_graphs(temp_analyzer)
                                self.logger.info(f"Successfully merged with pattern: {pattern_name}")
                                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to merge with pattern: {e}")
            return False
    
    def save_graph_to_memory(self, graph_name: str, description: str = "", 
                            tags: List[str] = None, domain: str = "graph_patterns") -> str:
        """Save the current graph representation to memory with metadata for reuse."""
        try:
            # Generate unique ID for the saved graph
            graph_id = f"graph_{uuid.uuid4().hex[:8]}"
            
            # Create comprehensive metadata
            metadata = {
                'graph_id': graph_id,
                'graph_name': graph_name,
                'description': description,
                'tags': tags or [],
                'domain': domain,
                'total_nodes': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges(),
                'node_types': {},
                'relationship_types': {},
                'source_documents': set(),
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0,
                'version': '1.0',
                'export_formats': ['json', 'yaml', 'sqlite', 'cypher', 'graphml']
            }
            
            # Analyze node types and relationships
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get('type', 'unknown')
                metadata['node_types'][node_type] = metadata['node_types'].get(node_type, 0) + 1
                metadata['source_documents'].add(node_data.get('source_document', 'unknown'))
            
            for source, target, edge_data in self.graph.edges(data=True):
                rel_type = edge_data.get('type', 'unknown')
                metadata['relationship_types'][rel_type] = metadata['relationship_types'].get(rel_type, 0) + 1
            
            # Convert sets to lists for JSON serialization
            metadata['source_documents'] = list(metadata['source_documents'])
            
            # Create the graph representation data
            graph_data = {
                'metadata': metadata,
                'nodes': [],
                'relationships': []
            }
            
            # Export nodes
            for node_id, node_data in self.graph.nodes(data=True):
                graph_data['nodes'].append({
                    'id': node_id,
                    **node_data
                })
            
            # Export relationships
            for source, target, edge_data in self.graph.edges(data=True):
                graph_data['relationships'].append({
                    'source': source,
                    'target': target,
                    **edge_data
                })
            
            # Store in memory
            self.saved_graphs[graph_id] = {
                'data': graph_data,
                'metadata': metadata,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0
            }
            
            self.logger.info(f"Graph '{graph_name}' saved to memory with ID: {graph_id}")
            return graph_id
            
        except Exception as e:
            self.logger.error(f"Failed to save graph to memory: {e}")
            return None
    
    def load_graph_from_memory(self, graph_id: str) -> bool:
        """Load a graph representation from memory by ID."""
        try:
            if graph_id not in self.saved_graphs:
                self.logger.error(f"Graph ID {graph_id} not found in memory")
                return False
            
            saved_graph = self.saved_graphs[graph_id]
            graph_data = saved_graph['data']
            
            # Update access metadata
            saved_graph['last_accessed'] = datetime.now().isoformat()
            saved_graph['access_count'] += 1
            
            # Clear current graph
            self.graph.clear()
            
            # Load nodes
            for node in graph_data['nodes']:
                node_id = node.pop('id')
                self.graph.add_node(node_id, **node)
            
            # Load relationships
            for rel in graph_data['relationships']:
                source = rel.pop('source')
                target = rel.pop('target')
                self.graph.add_edge(source, target, **rel)
            
            self.logger.info(f"Successfully loaded graph '{saved_graph['metadata']['graph_name']}' from memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load graph from memory: {e}")
            return False
    
    def list_saved_graphs(self) -> List[Dict[str, Any]]:
        """List all saved graph representations in memory."""
        graphs = []
        for graph_id, saved_graph in self.saved_graphs.items():
            metadata = saved_graph['metadata']
            graphs.append({
                'graph_id': graph_id,
                'name': metadata.get('graph_name', 'Unnamed'),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'domain': metadata.get('domain', ''),
                'total_nodes': metadata.get('total_nodes', 0),
                'total_relationships': metadata.get('total_relationships', 0),
                'created_at': saved_graph['created_at'],
                'last_accessed': saved_graph['last_accessed'],
                'access_count': saved_graph['access_count']
            })
        return graphs
    
    def search_saved_graphs(self, query: str = None, tags: List[str] = None, 
                           domain: str = None) -> List[Dict[str, Any]]:
        """Search saved graphs by query, tags, or domain."""
        results = []
        
        for graph_id, saved_graph in self.saved_graphs.items():
            metadata = saved_graph['metadata']
            
            # Apply filters
            if query and query.lower() not in metadata.get('graph_name', '').lower() and \
               query.lower() not in metadata.get('description', '').lower():
                continue
            
            if tags and not any(tag in metadata.get('tags', []) for tag in tags):
                continue
            
            if domain and metadata.get('domain') != domain:
                continue
            
            results.append({
                'graph_id': graph_id,
                'name': metadata.get('graph_name', 'Unnamed'),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'domain': metadata.get('domain', ''),
                'total_nodes': metadata.get('total_nodes', 0),
                'total_relationships': metadata.get('total_relationships', 0),
                'created_at': saved_graph['created_at'],
                'last_accessed': saved_graph['last_accessed'],
                'access_count': saved_graph['access_count']
            })
        
        return results
    
    def delete_saved_graph(self, graph_id: str) -> bool:
        """Delete a saved graph representation from memory."""
        try:
            if graph_id in self.saved_graphs:
                graph_name = self.saved_graphs[graph_id]['metadata']['graph_name']
                del self.saved_graphs[graph_id]
                self.logger.info(f"Graph '{graph_name}' deleted from memory")
                return True
            else:
                self.logger.warning(f"Graph ID {graph_id} not found in memory")
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete saved graph: {e}")
            return False
    
    def export_memory_to_file(self, file_path: str = None) -> str:
        """Export all saved graphs from memory to a file for persistence."""
        if file_path is None:
            file_path = self.output_dir / f"graph_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            memory_data = {
                'exported_at': datetime.now().isoformat(),
                'total_saved_graphs': len(self.saved_graphs),
                'graphs': self.saved_graphs
            }
            
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
            
            self.logger.info(f"Graph memory exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export graph memory: {e}")
            return None
    
    def import_memory_from_file(self, file_path: str) -> bool:
        """Import saved graphs from a file into memory."""
        try:
            with open(file_path, 'r') as f:
                memory_data = json.load(f)
            
            if 'graphs' in memory_data:
                imported_count = 0
                for graph_id, saved_graph in memory_data['graphs'].items():
                    if graph_id not in self.saved_graphs:
                        self.saved_graphs[graph_id] = saved_graph
                        imported_count += 1
                
                self.logger.info(f"Successfully imported {imported_count} graphs from file")
                return True
            else:
                self.logger.error("Invalid memory file format")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to import graph memory: {e}")
            return False
