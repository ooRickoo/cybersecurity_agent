#!/usr/bin/env python3
"""
MCP Tools for JSON Graph Analyzer
Provides MCP-compatible interface for JSON document decomposition and graph analysis.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add the bin directory to the path
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

from json_graph_analyzer import JSONGraphAnalyzer

class JSONGraphMCPTools:
    """MCP-compatible tools for JSON Graph Analyzer."""
    
    def __init__(self):
        self.analyzer = JSONGraphAnalyzer()
        self.session_analyzers = {}  # Store analyzers per session
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return list of available MCP tools."""
        return [
            {
                "name": "analyze_json_structure",
                "description": "Analyze the structure of a JSON document and identify components",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "json_data": {
                            "type": "string",
                            "description": "JSON data as string or file path"
                        },
                        "document_type": {
                            "type": "string",
                            "description": "Optional document type (application, host, network, user, database, control)"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for tracking multiple documents"
                        }
                    },
                    "required": ["json_data"]
                }
            },
            {
                "name": "decompose_json_document",
                "description": "Decompose a JSON document into its components and build a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "json_data": {
                            "type": "string",
                            "description": "JSON data as string or file path"
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Optional document ID"
                        },
                        "document_type": {
                            "type": "string",
                            "description": "Optional document type"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for tracking multiple documents"
                        }
                    },
                    "required": ["json_data"]
                }
            },
            {
                "name": "export_graph",
                "description": "Export the current graph to various formats",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["sqlite", "graphml", "cypher", "json", "yaml"],
                            "description": "Export format"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional custom file path"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    },
                    "required": ["format"]
                }
            },
            {
                "name": "query_graph_entities",
                "description": "Query entities in the graph based on filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Filter by entity type"
                        },
                        "source_document": {
                            "type": "string",
                            "description": "Filter by source document"
                        },
                        "properties_filter": {
                            "type": "object",
                            "description": "Filter by entity properties"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            },
            {
                "name": "get_graph_statistics",
                "description": "Get comprehensive statistics about the current graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            },
            {
                "name": "merge_graphs",
                "description": "Merge multiple graph analyzers into one",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of session IDs to merge"
                        },
                        "target_session_id": {
                            "type": "string",
                            "description": "Target session ID for merged graph"
                        }
                    },
                    "required": ["session_ids", "target_session_id"]
                }
            },
            {
                "name": "load_graph_from_file",
                "description": "Load a graph from a JSON or YAML file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the graph file"
                        },
                        "file_format": {
                            "type": "string",
                            "enum": ["json", "yaml"],
                            "description": "File format"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    },
                    "required": ["file_path", "file_format"]
                }
            },
            {
                "name": "get_entity_neighborhood",
                "description": "Get the neighborhood of an entity in the graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to analyze"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth for neighborhood analysis",
                            "default": 2
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "save_analysis_report",
                "description": "Save a comprehensive analysis report",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Optional custom file path"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            },
            {
                "name": "save_graph_to_memory",
                "description": "Save the current graph as a reusable pattern in context memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern_name": {
                            "type": "string",
                            "description": "Name for the graph pattern"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the pattern"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "tier": {
                            "type": "string",
                            "enum": ["short_term", "medium_term", "long_term"],
                            "description": "Memory tier for storage"
                        },
                        "ttl_days": {
                            "type": "integer",
                            "description": "Time to live in days",
                            "default": 90
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority level (1-10)",
                            "default": 5
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    },
                    "required": ["pattern_name"]
                }
            },
            {
                "name": "load_graph_from_memory",
                "description": "Load a graph pattern from context memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Specific memory entry ID"
                        },
                        "pattern_name": {
                            "type": "string",
                            "description": "Pattern name to search for"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Memory domain to search in"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            },
            {
                "name": "get_available_patterns",
                "description": "Get list of available graph patterns from context memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Memory domain to search in"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            },
            {
                "name": "merge_with_pattern",
                "description": "Merge the current graph with a stored pattern from memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern_name": {
                            "type": "string",
                            "description": "Pattern name to merge with"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    },
                    "required": ["pattern_name"]
                }
            }
        ]
    
    def _get_or_create_analyzer(self, session_id: str = None) -> JSONGraphAnalyzer:
        """Get or create an analyzer for a session."""
        if session_id is None:
            session_id = "default"
        
        if session_id not in self.session_analyzers:
            self.session_analyzers[session_id] = JSONGraphAnalyzer()
        
        return self.session_analyzers[session_id]
    
    def _parse_json_input(self, json_data: str) -> Dict[str, Any]:
        """Parse JSON input which could be a string or file path."""
        try:
            # Try to parse as JSON string first
            return json.loads(json_data)
        except json.JSONDecodeError:
            # If not JSON string, try as file path
            try:
                file_path = Path(json_data)
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        return json.load(f)
                else:
                    raise ValueError(f"File not found: {json_data}")
            except Exception as e:
                raise ValueError(f"Invalid JSON input: {e}")
    
    def analyze_json_structure(self, json_data: str, document_type: str = None, 
                             session_id: str = None) -> Dict[str, Any]:
        """Analyze the structure of a JSON document."""
        try:
            parsed_data = self._parse_json_input(json_data)
            analyzer = self._get_or_create_analyzer(session_id)
            
            analysis = analyzer.analyze_document_structure(parsed_data, document_type)
            
            return {
                "success": True,
                "analysis": analysis,
                "message": f"Successfully analyzed {analysis['document_type']} document structure"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to analyze JSON structure: {e}"
            }
    
    def decompose_json_document(self, json_data: str, document_id: str = None,
                               document_type: str = None, session_id: str = None) -> Dict[str, Any]:
        """Decompose a JSON document into its components."""
        try:
            parsed_data = self._parse_json_input(json_data)
            analyzer = self._get_or_create_analyzer(session_id)
            
            # Decompose document
            decomposition = analyzer.decompose_document(parsed_data, document_id, document_type)
            
            # Build graph
            graph = analyzer.build_graph_from_decomposition(decomposition)
            
            return {
                "success": True,
                "decomposition": decomposition,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                },
                "message": f"Successfully decomposed document into {decomposition['document_type']} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to decompose JSON document: {e}"
            }
    
    def export_graph(self, format: str, file_path: str = None, session_id: str = None) -> Dict[str, Any]:
        """Export the current graph to various formats."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            if format == "sqlite":
                output_path = analyzer.export_to_sqlite(file_path)
            elif format == "graphml":
                output_path = analyzer.export_to_graphml(file_path)
            elif format == "cypher":
                output_path = analyzer.export_to_cypher(file_path)
            elif format == "json":
                output_path = analyzer.export_to_json(file_path)
            elif format == "yaml":
                output_path = analyzer.export_to_yaml(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return {
                "success": True,
                "format": format,
                "output_path": output_path,
                "message": f"Successfully exported graph to {format} format"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to export graph: {e}"
            }
    
    def query_graph_entities(self, entity_type: str = None, source_document: str = None,
                           properties_filter: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
        """Query entities in the graph based on filters."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            results = analyzer.query_entities(entity_type, source_document, properties_filter)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "message": f"Found {len(results)} entities matching the query"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to query entities: {e}"
            }
    
    def get_graph_statistics(self, session_id: str = None) -> Dict[str, Any]:
        """Get comprehensive statistics about the current graph."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            stats = analyzer.get_graph_statistics()
            
            return {
                "success": True,
                "statistics": stats,
                "message": f"Graph statistics retrieved successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get graph statistics: {e}"
            }
    
    def merge_graphs(self, session_ids: List[str], target_session_id: str) -> Dict[str, Any]:
        """Merge multiple graph analyzers into one."""
        try:
            target_analyzer = self._get_or_create_analyzer(target_session_id)
            
            total_nodes = 0
            total_edges = 0
            
            for session_id in session_ids:
                if session_id in self.session_analyzers:
                    source_analyzer = self.session_analyzers[session_id]
                    target_analyzer.merge_graphs(source_analyzer)
                    total_nodes += source_analyzer.graph.number_of_nodes()
                    total_edges += source_analyzer.graph.number_of_edges()
            
            final_stats = target_analyzer.get_graph_statistics()
            
            return {
                "success": True,
                "merged_sessions": session_ids,
                "target_session": target_session_id,
                "final_stats": final_stats,
                "message": f"Successfully merged {len(session_ids)} graphs into target session"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to merge graphs: {e}"
            }
    
    def load_graph_from_file(self, file_path: str, file_format: str, session_id: str = None) -> Dict[str, Any]:
        """Load a graph from a JSON or YAML file."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            if file_format == "json":
                graph = analyzer.load_from_json(file_path)
            elif file_format == "yaml":
                graph = analyzer.load_from_yaml(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            stats = analyzer.get_graph_statistics()
            
            return {
                "success": True,
                "file_path": file_path,
                "format": file_format,
                "graph_stats": stats,
                "message": f"Successfully loaded graph from {file_format} file"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load graph from file: {e}"
            }
    
    def get_entity_neighborhood(self, entity_id: str, max_depth: int = 2, 
                              session_id: str = None) -> Dict[str, Any]:
        """Get the neighborhood of an entity in the graph."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            neighborhood = analyzer.get_entity_neighborhood(entity_id, max_depth)
            
            if not neighborhood:
                return {
                    "success": False,
                    "error": "Entity not found",
                    "message": f"Entity {entity_id} not found in the graph"
                }
            
            return {
                "success": True,
                "entity_id": entity_id,
                "neighborhood": neighborhood,
                "message": f"Successfully retrieved neighborhood for entity {entity_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get entity neighborhood: {e}"
            }
    
    def save_analysis_report(self, file_path: str = None, session_id: str = None) -> Dict[str, Any]:
        """Save a comprehensive analysis report."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            output_path = analyzer.save_analysis_report(file_path)
            
            return {
                "success": True,
                "report_path": output_path,
                "message": "Analysis report saved successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save analysis report: {e}"
            }
    
    def save_graph_to_memory(self, pattern_name: str, description: str = None, 
                           tags: List[str] = None, tier: str = "medium_term",
                           ttl_days: int = 90, priority: int = 5, 
                           session_id: str = None) -> Dict[str, Any]:
        """Save the current graph as a reusable pattern in context memory."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            memory_id = analyzer.save_to_context_memory(
                pattern_name=pattern_name,
                description=description,
                tags=tags,
                tier=tier,
                ttl_days=ttl_days,
                priority=priority
            )
            
            if memory_id:
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "pattern_name": pattern_name,
                    "message": f"Graph pattern '{pattern_name}' saved to context memory successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save to context memory",
                    "message": "Could not save graph pattern to context memory"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save graph to memory: {e}"
            }
    
    def load_graph_from_memory(self, memory_id: str = None, pattern_name: str = None,
                             domain: str = None, session_id: str = None) -> Dict[str, Any]:
        """Load a graph pattern from context memory."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            success = analyzer.load_from_context_memory(
                memory_id=memory_id,
                pattern_name=pattern_name,
                domain=domain
            )
            
            if success:
                stats = analyzer.get_graph_statistics()
                return {
                    "success": True,
                    "loaded_from": memory_id or pattern_name or domain,
                    "message": f"Successfully loaded graph pattern from memory"
                }
            else:
                return {
                    "success": False,
                    "error": "Pattern not found",
                    "message": f"Could not find graph pattern: {memory_id or pattern_name or domain}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load graph from memory: {e}"
            }
    
    def get_available_patterns(self, domain: str = None, session_id: str = None) -> Dict[str, Any]:
        """Get list of available graph patterns from context memory."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            patterns = analyzer.get_available_patterns(domain=domain)
            
            return {
                "success": True,
                "patterns": patterns,
                "count": len(patterns),
                "message": f"Found {len(patterns)} available graph patterns"
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get available patterns: {e}"
            }
    
    def merge_with_pattern(self, pattern_name: str, session_id: str = None) -> Dict[str, Any]:
        """Merge the current graph with a stored pattern from memory."""
        try:
            analyzer = self._get_or_create_analyzer(session_id)
            
            stats = analyzer.get_graph_statistics()
            return {
                "success": True,
                "pattern_name": pattern_name,
                "merged_graph_stats": stats,
                "message": f"Successfully merged with pattern: {pattern_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to merge with pattern: {e}"
            }

# Example usage and testing
if __name__ == "__main__":
    # Create example JSON data
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
    
    # Test the tools
    tools = JSONGraphMCPTools()
    
    print("🔍 Testing JSON Graph Analyzer MCP Tools...")
    
    # Test structure analysis
    result = tools.analyze_json_structure(json.dumps(example_app), "application")
    print(f"✅ Structure Analysis: {result['message']}")
    
    # Test decomposition
    result = tools.decompose_json_document(json.dumps(example_app), "app_001", "application")
    print(f"✅ Document Decomposition: {result['message']}")
    
    # Test statistics
    result = tools.get_graph_statistics()
    print(f"✅ Graph Statistics: {result['message']}")
    
    # Test export
    result = tools.export_graph("json")
    print(f"✅ Graph Export: {result['message']}")
    
    print("🎯 JSON Graph Analyzer MCP Tools test completed!")
