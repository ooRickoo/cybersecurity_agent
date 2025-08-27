#!/usr/bin/env python3
"""
Database MCP Tools for Cybersecurity Agent
Provides NLP-friendly database operations with intelligent credential management
"""

import json
import pandas as pd
import sqlite3
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import tempfile
import os
import sys

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from database_connector import get_database_manager
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False

class DatabaseMCPTools:
    """MCP tools for database operations with NLP support."""
    
    def __init__(self):
        self.db_manager = get_database_manager() if DATABASE_MANAGER_AVAILABLE else None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools."""
        if not DATABASE_MANAGER_AVAILABLE:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "connect_to_database",
                    "description": "Connect to a database with automatic credential management and schema discovery",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database_type": {
                                "type": "string",
                                "description": "Type of database (mssql, mongodb, postgres, mysql, sqlite)",
                                "enum": ["mssql", "mongodb", "postgres", "mysql", "sqlite", "azure-sql"]
                            },
                            "host": {
                                "type": "string",
                                "description": "Database host or server address"
                            },
                            "port": {
                                "type": "integer",
                                "description": "Database port number"
                            },
                            "database_name": {
                                "type": "string",
                                "description": "Name of the database to connect to"
                            },
                            "username": {
                                "type": "string",
                                "description": "Database username (will be prompted if not provided)"
                            },
                            "password": {
                                "type": "string",
                                "description": "Database password (will be prompted if not provided)"
                            },
                            "additional_params": {
                                "type": "object",
                                "description": "Additional connection parameters as key-value pairs"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for connecting (for logging and memory)"
                            }
                        },
                        "required": ["database_type", "host", "port", "database_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "discover_database_schema",
                    "description": "Discover and analyze the schema of a connected database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection to analyze"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for schema discovery (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Execute a query on a connected database with NLP-friendly parameter handling",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection to query"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query or MongoDB aggregation pipeline"
                            },
                            "params": {
                                "type": "object",
                                "description": "Query parameters (for parameterized queries)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 1000)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for query execution (for logging)"
                            }
                        },
                        "required": ["connection_id", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explore_table",
                    "description": "Explore a specific table or collection in detail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection"
                            },
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table or collection to explore"
                            },
                            "include_sample_data": {
                                "type": "boolean",
                                "description": "Whether to include sample data in the exploration",
                                "default": True
                            },
                            "sample_limit": {
                                "type": "integer",
                                "description": "Number of sample records to include",
                                "default": 50
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for table exploration (for logging)"
                            }
                        },
                        "required": ["connection_id", "table_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "export_to_dataframe",
                    "description": "Export database query results to a pandas DataFrame for local analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query or MongoDB aggregation pipeline"
                            },
                            "params": {
                                "type": "object",
                                "description": "Query parameters"
                            },
                            "dataframe_name": {
                                "type": "string",
                                "description": "Name for the exported DataFrame"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for export (for logging)"
                            }
                        },
                        "required": ["connection_id", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "export_to_sqlite",
                    "description": "Export database query results to a local SQLite database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query or MongoDB aggregation pipeline"
                            },
                            "params": {
                                "type": "object",
                                "description": "Query parameters"
                            },
                            "table_name": {
                                "type": "string",
                                "description": "Name for the table in SQLite"
                            },
                            "database_path": {
                                "type": "string",
                                "description": "Path for the SQLite database file"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for export (for logging)"
                            }
                        },
                        "required": ["connection_id", "query", "table_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_database_connections",
                    "description": "List all available database connections with their status",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_connection_status",
                    "description": "Get detailed status of a specific database connection",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection to check"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "disconnect_from_database",
                    "description": "Disconnect from a database and clean up resources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection to disconnect"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for disconnection (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_database_relationships",
                    "description": "Analyze relationships between tables and suggest join strategies",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the database connection"
                            },
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of table names to analyze relationships for"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for relationship analysis (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            }
        ]
    
    def connect_to_database(self, 
                           database_type: str,
                           host: str,
                           port: int,
                           database_name: str,
                           username: str = None,
                           password: str = None,
                           additional_params: Dict[str, Any] = None,
                           reason: str = None) -> Dict[str, Any]:
        """Connect to a database with intelligent credential management."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Create connection configuration
            connection_id = self.db_manager.create_connection(
                database_type=database_type,
                host=host,
                port=port,
                database_name=database_name,
                username=username,
                password=password,
                additional_params=additional_params
            )
            
            if not connection_id:
                return {
                    "success": False,
                    "error": "Connection creation failed",
                    "message": "Failed to create database connection configuration"
                }
            
            # Attempt to connect
            if self.db_manager.connect_to_database(connection_id):
                # Discover schema automatically
                schema = self.db_manager.discover_schema(connection_id)
                
                cli_output = f"🔗 **Database Connected Successfully!**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Type:** {database_type.upper()}\n"
                cli_output += f"   **Host:** {host}:{port}\n"
                cli_output += f"   **Database:** {database_name}\n"
                cli_output += f"   **Username:** {username or 'N/A'}\n"
                
                if schema:
                    cli_output += f"   **Schema Discovered:** ✅\n"
                    cli_output += f"   **Tables:** {len(schema.tables)}\n"
                    cli_output += f"   **Views:** {len(schema.views)}\n"
                    cli_output += f"   **Stored Procedures:** {len(schema.stored_procedures)}\n"
                    cli_output += f"   **Functions:** {len(schema.functions)}\n"
                else:
                    cli_output += f"   **Schema Discovery:** ⚠️ Failed\n"
                
                if reason:
                    cli_output += f"   **Reason:** {reason}\n"
                
                cli_output += f"\n💡 **Next Steps:**\n"
                cli_output += f"   • Use 'explore_table' to examine specific tables\n"
                cli_output += f"   • Use 'query_database' to run custom queries\n"
                cli_output += f"   • Use 'export_to_dataframe' to load data locally\n"
                cli_output += f"   • Use 'disconnect_from_database' when finished\n"
                
                return {
                    "success": True,
                    "connection_id": connection_id,
                    "schema_discovered": schema is not None,
                    "message": f"Successfully connected to {database_type} database",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Connection failed",
                    "message": f"Failed to establish connection to {database_type} database",
                    "connection_id": connection_id,
                    "cli_output": f"❌ **Database Connection Failed**\n   **Type:** {database_type}\n   **Host:** {host}:{port}\n   **Database:** {database_name}\n   **Error:** Connection establishment failed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error connecting to database: {e}",
                "cli_output": f"❌ **Database Connection Error**\n   **Error:** {e}"
            }
    
    def discover_database_schema(self, connection_id: str, reason: str = None) -> Dict[str, Any]:
        """Discover and analyze database schema."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            schema = self.db_manager.discover_schema(connection_id)
            
            if schema:
                cli_output = f"🔍 **Database Schema Discovered**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Database:** {schema.database_name}\n"
                cli_output += f"   **Type:** {schema.database_type}\n"
                cli_output += f"   **Discovered:** {schema.discovered_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                cli_output += f"   **Schema Hash:** {schema.schema_hash[:16]}...\n\n"
                
                cli_output += f"📊 **Schema Summary:**\n"
                cli_output += f"   **Tables:** {len(schema.tables)} collections\n"
                cli_output += f"   **Views:** {len(schema.views)} views\n"
                cli_output += f"   **Stored Procedures:** {len(schema.stored_procedures)} procedures\n"
                cli_output += f"   **Functions:** {len(schema.functions)} functions\n"
                cli_output += f"   **Indexes:** {len(schema.indexes)} indexes\n"
                cli_output += f"   **Foreign Keys:** {len(schema.foreign_keys)} relationships\n\n"
                
                if schema.tables:
                    cli_output += f"📋 **Tables/Collections:**\n"
                    for table in schema.tables[:10]:  # Show first 10
                        cli_output += f"   • {table['name']} ({table.get('row_count', 'N/A')} rows)\n"
                    if len(schema.tables) > 10:
                        cli_output += f"   ... and {len(schema.tables) - 10} more\n"
                
                if reason:
                    cli_output += f"\n💡 **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "schema": schema.to_dict(),
                    "message": "Database schema discovered successfully",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Schema discovery failed",
                    "message": "Failed to discover database schema",
                    "cli_output": f"❌ **Schema Discovery Failed**\n   **Connection ID:** {connection_id}\n   **Error:** Could not discover schema"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error discovering schema: {e}",
                "cli_output": f"❌ **Schema Discovery Error**\n   **Error:** {e}"
            }
    
    def query_database(self, 
                      connection_id: str, 
                      query: str, 
                      params: Optional[Dict[str, Any]] = None,
                      limit: int = 1000,
                      reason: str = None) -> Dict[str, Any]:
        """Execute a query on the database."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Execute query
            results = self.db_manager.execute_query(connection_id, query, params)
            
            if results is not None:
                # Apply limit
                if len(results) > limit:
                    results = results[:limit]
                    limited = True
                else:
                    limited = False
                
                cli_output = f"🔍 **Database Query Executed**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Query:** {query[:100]}{'...' if len(query) > 100 else ''}\n"
                cli_output += f"   **Results:** {len(results)} records\n"
                if limited:
                    cli_output += f"   **Limited:** First {limit} results shown\n"
                
                if results:
                    cli_output += f"   **Columns:** {', '.join(results[0].keys())}\n"
                    
                    # Show sample data
                    cli_output += f"\n📊 **Sample Results:**\n"
                    for i, row in enumerate(results[:3]):  # Show first 3 rows
                        cli_output += f"   **Row {i+1}:** {dict(list(row.items())[:5])}\n"
                        if i >= 2:
                            break
                    
                    if len(results) > 3:
                        cli_output += f"   ... and {len(results) - 3} more rows\n"
                else:
                    cli_output += f"   **Note:** Query returned no results\n"
                
                if reason:
                    cli_output += f"\n💡 **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "results": results,
                    "result_count": len(results),
                    "limited": limited,
                    "message": f"Query executed successfully, returned {len(results)} results",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Query execution failed",
                    "message": "Failed to execute database query",
                    "cli_output": f"❌ **Query Execution Failed**\n   **Connection ID:** {connection_id}\n   **Query:** {query[:100]}...\n   **Error:** Query returned no results or failed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error executing query: {e}",
                "cli_output": f"❌ **Query Execution Error**\n   **Error:** {e}"
            }
    
    def explore_table(self, 
                     connection_id: str, 
                     table_name: str, 
                     include_sample_data: bool = True,
                     sample_limit: int = 50,
                     reason: str = None) -> Dict[str, Any]:
        """Explore a specific table in detail."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Get table information
            table_info = self.db_manager.get_table_info(connection_id, table_name)
            
            if not table_info:
                return {
                    "success": False,
                    "error": "Table not found",
                    "message": f"Table {table_name} not found or not accessible"
                }
            
            # Get sample data if requested
            sample_data = []
            if include_sample_data:
                sample_data = self.db_manager.get_sample_data(connection_id, table_name, sample_limit)
            
            cli_output = f"🔍 **Table Exploration: {table_name}**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Row Count:** {table_info.get('row_count', 'Unknown')}\n"
            cli_output += f"   **Columns:** {len(table_info.get('columns', []))}\n\n"
            
            if table_info.get('columns'):
                cli_output += f"📋 **Column Details:**\n"
                for col in table_info['columns'][:10]:  # Show first 10 columns
                    cli_output += f"   • {col['name']} ({col['data_type']})"
                    if col.get('is_nullable'):
                        cli_output += " [nullable]"
                    if col.get('default_value'):
                        cli_output += f" [default: {col['default_value']}]"
                    cli_output += "\n"
                
                if len(table_info['columns']) > 10:
                    cli_output += f"   ... and {len(table_info['columns']) - 10} more columns\n"
            
            if sample_data:
                cli_output += f"\n📊 **Sample Data ({len(sample_data)} rows):**\n"
                for i, row in enumerate(sample_data[:3]):  # Show first 3 rows
                    cli_output += f"   **Row {i+1}:** {dict(list(row.items())[:5])}\n"
                    if i >= 2:
                        break
                
                if len(sample_data) > 3:
                    cli_output += f"   ... and {len(sample_data) - 3} more rows\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "table_info": table_info,
                "sample_data": sample_data,
                "message": f"Table {table_name} explored successfully",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error exploring table: {e}",
                "cli_output": f"❌ **Table Exploration Error**\n   **Error:** {e}"
            }
    
    def export_to_dataframe(self, 
                           connection_id: str, 
                           query: str, 
                           params: Optional[Dict[str, Any]] = None,
                           dataframe_name: str = None,
                           reason: str = None) -> Dict[str, Any]:
        """Export database query results to a pandas DataFrame."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Execute query
            results = self.db_manager.execute_query(connection_id, query, params)
            
            if not results:
                return {
                    "success": False,
                    "error": "No results to export",
                    "message": "Query returned no results to export"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Generate name if not provided
            if not dataframe_name:
                dataframe_name = f"db_export_{connection_id[:8]}_{int(pd.Timestamp.now().timestamp())}"
            
            # Store DataFrame in memory (you could extend this to store in the memory manager)
            # For now, we'll return the DataFrame info
            
            cli_output = f"📊 **Data Exported to DataFrame**\n"
            cli_output += f"   **DataFrame Name:** {dataframe_name}\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"
            cli_output += f"   **Columns:** {', '.join(df.columns.tolist())}\n"
            cli_output += f"   **Data Types:** {dict(df.dtypes)}\n\n"
            
            cli_output += f"💡 **DataFrame Available for Analysis:**\n"
            cli_output += f"   • Use pandas operations on the data\n"
            cli_output += f"   • Export to CSV, Excel, or other formats\n"
            cli_output += f"   • Perform data analysis and visualization\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "dataframe_name": dataframe_name,
                "dataframe_info": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": dict(df.dtypes),
                    "memory_usage": df.memory_usage(deep=True).sum()
                },
                "message": f"Data exported to DataFrame {dataframe_name} successfully",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error exporting to DataFrame: {e}",
                "cli_output": f"❌ **DataFrame Export Error**\n   **Error:** {e}"
            }
    
    def export_to_sqlite(self, 
                         connection_id: str, 
                         query: str, 
                         table_name: str,
                         params: Optional[Dict[str, Any]] = None,
                         database_path: str = None,
                         reason: str = None) -> Dict[str, Any]:
        """Export database query results to a local SQLite database."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Execute query
            results = self.db_manager.execute_query(connection_id, query, params)
            
            if not results:
                return {
                    "success": False,
                    "error": "No results to export",
                    "message": "Query returned no results to export"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Generate database path if not provided
            if not database_path:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                database_path = f"session-outputs/db_export_{connection_id[:8]}_{timestamp}.db"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            
            # Export to SQLite
            with sqlite3.connect(database_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Get table info
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM ?", (table_name,))
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
            
            cli_output = f"💾 **Data Exported to SQLite**\n"
            cli_output += f"   **Database Path:** {database_path}\n"
            cli_output += f"   **Table Name:** {table_name}\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Exported Rows:** {row_count}\n"
            cli_output += f"   **Exported Columns:** {len(columns)}\n"
            cli_output += f"   **File Size:** {os.path.getsize(database_path) / 1024:.1f} KB\n\n"
            
            cli_output += f"💡 **SQLite Database Ready:**\n"
            cli_output += f"   • Use any SQLite client to open: {database_path}\n"
            cli_output += f"   • Query with: SELECT * FROM {table_name.replace('`', '')}\n"
            cli_output += f"   • Import into other tools or databases\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "database_path": database_path,
                "table_name": table_name,
                "exported_rows": row_count,
                "exported_columns": len(columns),
                "file_size_kb": os.path.getsize(database_path) / 1024,
                "message": f"Data exported to SQLite database successfully",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error exporting to SQLite: {e}",
                "cli_output": f"❌ **SQLite Export Error**\n   **Error:** {e}"
            }
    
    def list_database_connections(self) -> Dict[str, Any]:
        """List all available database connections."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            connections = self.db_manager.list_connections()
            
            cli_output = f"🔗 **Database Connections**\n"
            cli_output += f"   **Total Connections:** {len(connections)}\n\n"
            
            if connections:
                for conn in connections:
                    status_icon = "🟢" if conn['is_connected'] else "🔴"
                    cli_output += f"{status_icon} **{conn['connection_id'][:8]}...**\n"
                    cli_output += f"   **Type:** {conn['database_type'].upper()}\n"
                    cli_output += f"   **Host:** {conn['host']}:{conn['port']}\n"
                    cli_output += f"   **Database:** {conn['database_name']}\n"
                    cli_output += f"   **User:** {conn['username'] or 'N/A'}\n"
                    cli_output += f"   **Status:** {'Connected' if conn['is_connected'] else 'Disconnected'}\n"
                    cli_output += f"   **Created:** {conn['created_at'][:10]}\n"
                    cli_output += f"   **Last Used:** {conn['last_used'][:10]}\n\n"
            else:
                cli_output += f"   **No connections available**\n"
                cli_output += f"   Use 'connect_to_database' to create a new connection\n"
            
            return {
                "success": True,
                "connections": connections,
                "connection_count": len(connections),
                "message": f"Found {len(connections)} database connections",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error listing connections: {e}",
                "cli_output": f"❌ **Connection List Error**\n   **Error:** {e}"
            }
    
    def get_connection_status(self, connection_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific database connection."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            status = self.db_manager.get_connection_status(connection_id)
            
            if status:
                cli_output = f"📊 **Connection Status**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Database Type:** {status['database_type'].upper()}\n"
                cli_output += f"   **Host:** {status['host']}:{status['port']}\n"
                cli_output += f"   **Database:** {status['database_name']}\n"
                cli_output += f"   **Connected:** {'Yes' if status['is_connected'] else 'No'}\n"
                cli_output += f"   **Working:** {'Yes' if status['connection_working'] else 'No'}\n"
                cli_output += f"   **Created:** {status['created_at'][:10]}\n"
                cli_output += f"   **Last Used:** {status['last_used'][:10]}\n"
                
                return {
                    "success": True,
                    "status": status,
                    "message": "Connection status retrieved successfully",
                    "cli_output": cli_output
                }
            else:
                return {
                    "success": False,
                    "error": "Connection not found",
                    "message": f"Connection ID {connection_id} not found",
                    "cli_output": f"❌ **Connection Not Found**\n   **Connection ID:** {connection_id}\n   **Error:** Connection does not exist"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting connection status: {e}",
                "cli_output": f"❌ **Status Check Error**\n   **Error:** {e}"
            }
    
    def disconnect_from_database(self, connection_id: str, reason: str = None) -> Dict[str, Any]:
        """Disconnect from a database and clean up resources."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            success = self.db_manager.disconnect_from_database(connection_id)
            
            if success:
                cli_output = f"🔌 **Database Disconnected**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Status:** Successfully disconnected\n"
                cli_output += f"   **Resources:** Cleaned up and reclaimed\n"
                
                if reason:
                    cli_output += f"   **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "message": "Successfully disconnected from database",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Disconnection failed",
                    "message": "Failed to disconnect from database",
                    "cli_output": f"❌ **Disconnection Failed**\n   **Connection ID:** {connection_id}\n   **Error:** Could not disconnect"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error disconnecting from database: {e}",
                "cli_output": f"❌ **Disconnection Error**\n   **Error:** {e}"
            }
    
    def analyze_database_relationships(self, 
                                     connection_id: str, 
                                     tables: List[str] = None,
                                     reason: str = None) -> Dict[str, Any]:
        """Analyze relationships between tables and suggest join strategies."""
        if not DATABASE_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Database manager not available",
                "message": "Database connector system not available"
            }
        
        try:
            # Get schema for the connection
            schema = self.db_manager.schemas.get(connection_id)
            
            if not schema:
                return {
                    "success": False,
                    "error": "Schema not available",
                    "message": "Database schema not discovered. Use 'discover_database_schema' first."
                }
            
            # Analyze relationships
            relationships = []
            join_suggestions = []
            
            # Find foreign key relationships
            for fk in schema.foreign_keys:
                relationships.append({
                    'type': 'foreign_key',
                    'from_table': fk['table_name'],
                    'from_column': fk['column_name'],
                    'to_table': fk['referenced_table'],
                    'to_column': fk['referenced_column'],
                    'constraint': fk['constraint_name']
                })
                
                # Suggest join
                join_suggestions.append({
                    'tables': [fk['table_name'], fk['referenced_table']],
                    'join_type': 'INNER JOIN',
                    'condition': f"{fk['table_name']}.{fk['column_name']} = {fk['referenced_table']}.{fk['referenced_column']}",
                    'description': f"Join {fk['table_name']} to {fk['referenced_table']} via foreign key"
                })
            
            # Analyze potential relationships based on column names
            if tables:
                for table1 in tables:
                    for table2 in tables:
                        if table1 != table2:
                            # Look for common column names that might indicate relationships
                            table1_info = self.db_manager.get_table_info(connection_id, table1)
                            table2_info = self.db_manager.get_table_info(connection_id, table2)
                            
                            if table1_info and table2_info:
                                table1_columns = {col['name'].lower() for col in table1_info.get('columns', [])}
                                table2_columns = {col['name'].lower() for col in table2_info.get('columns', [])}
                                
                                # Common patterns
                                common_patterns = ['id', 'name', 'code', 'type', 'category']
                                for pattern in common_patterns:
                                    if pattern in table1_columns and pattern in table2_columns:
                                        relationships.append({
                                            'type': 'potential_relationship',
                                            'from_table': table1,
                                            'from_column': pattern,
                                            'to_table': table2,
                                            'to_column': pattern,
                                            'confidence': 'medium',
                                            'description': f'Potential relationship via {pattern} column'
                                        })
                                        
                                        join_suggestions.append({
                                            'tables': [table1, table2],
                                            'join_type': 'LEFT JOIN',
                                            'condition': f"{table1}.{pattern} = {table2}.{pattern}",
                                            'description': f"Potential join via {pattern} column (verify data types)",
                                            'confidence': 'medium'
                                        })
            
            cli_output = f"🔗 **Database Relationship Analysis**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Database:** {schema.database_name}\n"
            cli_output += f"   **Relationships Found:** {len(relationships)}\n"
            cli_output += f"   **Join Suggestions:** {len(join_suggestions)}\n\n"
            
            if relationships:
                cli_output += f"📋 **Relationships:**\n"
                for rel in relationships[:5]:  # Show first 5
                    cli_output += f"   • {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}\n"
                    cli_output += f"     Type: {rel['type']}\n"
                    if rel.get('description'):
                        cli_output += f"     Note: {rel['description']}\n"
                    cli_output += "\n"
                
                if len(relationships) > 5:
                    cli_output += f"   ... and {len(relationships) - 5} more relationships\n\n"
            
            if join_suggestions:
                cli_output += f"🔗 **Join Suggestions:**\n"
                for suggestion in join_suggestions[:3]:  # Show first 3
                    cli_output += f"   • **Tables:** {', '.join(suggestion['tables'])}\n"
                    cli_output += f"     **Join:** {suggestion['join_type']}\n"
                    cli_output += f"     **Condition:** {suggestion['condition']}\n"
                    cli_output += f"     **Description:** {suggestion['description']}\n"
                    if suggestion.get('confidence'):
                        cli_output += f"     **Confidence:** {suggestion['confidence']}\n"
                    cli_output += "\n"
                
                if len(join_suggestions) > 3:
                    cli_output += f"   ... and {len(join_suggestions) - 3} more suggestions\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "relationships": relationships,
                "join_suggestions": join_suggestions,
                "message": f"Relationship analysis completed, found {len(relationships)} relationships",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error analyzing relationships: {e}",
                "cli_output": f"❌ **Relationship Analysis Error**\n   **Error:** {e}"
            }

# Global instance
_database_mcp_tools = None

def get_database_mcp_tools() -> DatabaseMCPTools:
    """Get or create the global database MCP tools instance."""
    global _database_mcp_tools
    if _database_mcp_tools is None:
        _database_mcp_tools = DatabaseMCPTools()
    return _database_mcp_tools

if __name__ == "__main__":
    # Test the MCP tools
    tools = get_database_mcp_tools()
    
    print("🧪 Testing Database MCP Tools...")
    
    # Test getting tools
    available_tools = tools.get_tools()
    print(f"Available tools: {len(available_tools)}")
    for tool in available_tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    
    # Test connection listing
    result = tools.list_database_connections()
    print(f"List connections: {result}")
