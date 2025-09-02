#!/usr/bin/env python3
"""
Database Connector System for Cybersecurity Agent
Provides read-only access to various database types with intelligent credential management
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Database drivers
try:
    import pyodbc
    MSSQL_AVAILABLE = True
except ImportError:
    MSSQL_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from credential_vault import CredentialVault
    CREDENTIAL_VAULT_AVAILABLE = True
except ImportError:
    CREDENTIAL_VAILABLE = False

try:
    from context_memory_manager import ContextMemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConnection:
    """Database connection configuration."""
    connection_id: str
    database_type: str
    host: str
    port: int
    database_name: str
    username: str
    connection_string: str
    additional_params: Dict[str, Any]
    created_at: datetime
    last_used: datetime
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConnection':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)

@dataclass
class DatabaseSchema:
    """Database schema information."""
    connection_id: str
    database_type: str
    database_name: str
    tables: List[Dict[str, Any]]
    views: List[Dict[str, Any]]
    stored_procedures: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    foreign_keys: List[Dict[str, Any]]
    discovered_at: datetime
    schema_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['discovered_at'] = self.discovered_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseSchema':
        """Create from dictionary."""
        data['discovered_at'] = datetime.fromisoformat(data['discovered_at'])
        return cls(**data)

class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    def __init__(self, connection_config: DatabaseConnection):
        self.connection_config = connection_config
        self.connection = None
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close database connection."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is working."""
        pass
    
    @abstractmethod
    def discover_schema(self) -> Optional[DatabaseSchema]:
        """Discover database schema."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read-only query."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific table."""
        pass
    
    @abstractmethod
    def get_sample_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from a table."""
        pass

class MSSQLConnector(DatabaseConnector):
    """Microsoft SQL Server connector."""
    
    def __init__(self, connection_config: DatabaseConnection):
        super().__init__(connection_config)
        if not MSSQL_AVAILABLE:
            raise ImportError("pyodbc not available for MSSQL connections")
    
    def connect(self) -> bool:
        """Establish MSSQL connection."""
        try:
            self.connection = pyodbc.connect(self.connection_config.connection_string)
            self.is_connected = True
            self.connection_config.last_used = datetime.now()
            self.connection_config.is_active = True
            logger.info(f"Connected to MSSQL: {self.connection_config.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MSSQL: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Close MSSQL connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.is_connected = False
                self.connection_config.is_active = False
                logger.info("MSSQL connection closed")
                return True
        except Exception as e:
            logger.error(f"Error closing MSSQL connection: {e}")
        return False
    
    def test_connection(self) -> bool:
        """Test MSSQL connection."""
        try:
            if not self.is_connected:
                return False
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"MSSQL connection test failed: {e}")
            return False
    
    def discover_schema(self) -> Optional[DatabaseSchema]:
        """Discover MSSQL schema."""
        try:
            if not self.is_connected:
                return None
            
            cursor = self.connection.cursor()
            
            # Get tables
            tables = []
            cursor.execute("""
                SELECT 
                    t.TABLE_NAME,
                    t.TABLE_TYPE,
                    p.ROWS as row_count,
                    s.name as schema_name
                FROM INFORMATION_SCHEMA.TABLES t
                LEFT JOIN sys.partitions p ON t.TABLE_NAME = OBJECT_NAME(p.OBJECT_ID)
                LEFT JOIN sys.schemas s ON t.TABLE_SCHEMA = s.name
                WHERE t.TABLE_TYPE = 'BASE TABLE'
                ORDER BY t.TABLE_NAME
            """)
            
            for row in cursor.fetchall():
                tables.append({
                    'name': row[0],
                    'type': row[1],
                    'row_count': row[2] if row[2] else 0,
                    'schema': row[3] if row[3] else 'dbo'
                })
            
            # Get views
            views = []
            cursor.execute("""
                SELECT 
                    TABLE_NAME,
                    VIEW_DEFINITION,
                    TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.VIEWS
                ORDER BY TABLE_NAME
            """)
            
            for row in cursor.fetchall():
                views.append({
                    'name': row[0],
                    'definition': row[1],
                    'schema': row[2]
                })
            
            # Get stored procedures
            procedures = []
            cursor.execute("""
                SELECT 
                    ROUTINE_NAME,
                    ROUTINE_DEFINITION,
                    ROUTINE_SCHEMA
                FROM INFORMATION_SCHEMA.ROUTINES
                WHERE ROUTINE_TYPE = 'PROCEDURE'
                ORDER BY ROUTINE_NAME
            """)
            
            for row in cursor.fetchall():
                procedures.append({
                    'name': row[0],
                    'definition': row[1],
                    'schema': row[2]
                })
            
            # Get functions
            functions = []
            cursor.execute("""
                SELECT 
                    ROUTINE_NAME,
                    ROUTINE_DEFINITION,
                    ROUTINE_SCHEMA
                FROM INFORMATION_SCHEMA.ROUTINES
                WHERE ROUTINE_TYPE = 'FUNCTION'
                ORDER BY ROUTINE_NAME
            """)
            
            for row in cursor.fetchall():
                functions.append({
                    'name': row[0],
                    'definition': row[1],
                    'schema': row[2]
                })
            
            # Get indexes
            indexes = []
            cursor.execute("""
                SELECT 
                    i.name as index_name,
                    t.name as table_name,
                    i.type_desc as index_type,
                    i.is_unique,
                    i.is_primary_key
                FROM sys.indexes i
                JOIN sys.tables t ON i.object_id = t.object_id
                ORDER BY t.name, i.name
            """)
            
            for row in cursor.fetchall():
                indexes.append({
                    'name': row[0],
                    'table': row[1],
                    'type': row[2],
                    'is_unique': bool(row[3]),
                    'is_primary_key': bool(row[4])
                })
            
            # Get foreign keys
            foreign_keys = []
            cursor.execute("""
                SELECT 
                    fk.name as constraint_name,
                    OBJECT_NAME(fk.parent_object_id) as table_name,
                    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) as column_name,
                    OBJECT_NAME(fk.referenced_object_id) as referenced_table,
                    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) as referenced_column
                FROM sys.foreign_keys fk
                JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                ORDER BY table_name, column_name
            """)
            
            for row in cursor.fetchall():
                foreign_keys.append({
                    'constraint_name': row[0],
                    'table_name': row[1],
                    'column_name': row[2],
                    'referenced_table': row[3],
                    'referenced_column': row[4]
                })
            
            cursor.close()
            
            # Create schema hash
            schema_data = json.dumps({
                'tables': tables,
                'views': views,
                'procedures': procedures,
                'functions': functions,
                'indexes': indexes,
                'foreign_keys': foreign_keys
            }, sort_keys=True)
            schema_hash = hashlib.sha256(schema_data.encode()).hexdigest()
            
            return DatabaseSchema(
                connection_id=self.connection_config.connection_id,
                database_type=self.connection_config.database_type,
                database_name=self.connection_config.database_name,
                tables=tables,
                views=views,
                stored_procedures=procedures,
                functions=functions,
                indexes=indexes,
                foreign_keys=foreign_keys,
                discovered_at=datetime.now(),
                schema_hash=schema_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to discover MSSQL schema: {e}")
            return None
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute MSSQL query."""
        try:
            if not self.is_connected:
                return []
            
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get column names
            columns = [column[0] for column in cursor.description] if cursor.description else []
            
            # Fetch results
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"MSSQL query execution failed: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get MSSQL table information."""
        try:
            if not self.is_connected:
                return None
            
            cursor = self.connection.cursor()
            
            # Get column information
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT,
                    CHARACTER_MAXIMUM_LENGTH,
                    NUMERIC_PRECISION,
                    NUMERIC_SCALE,
                    ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, table_name)
            
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2] == 'YES',
                    'default_value': row[3],
                    'max_length': row[4],
                    'precision': row[5],
                    'scale': row[6],
                    'position': row[7]
                })
            
            # Get row count
            cursor.execute("SELECT COUNT(*) FROM ?", (table_name,))
            row_count = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                'table_name': table_name,
                'columns': columns,
                'row_count': row_count,
                'connection_id': self.connection_config.connection_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get MSSQL table info: {e}")
            return None
    
    def get_sample_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from MSSQL table."""
        try:
            query = "SELECT TOP ? * FROM ?"
            return self.execute_query(query, (limit, table_name))
        except Exception as e:
            logger.error(f"Failed to get sample data from MSSQL: {e}")
            return []

class MongoDBConnector(DatabaseConnector):
    """MongoDB connector."""
    
    def __init__(self, connection_config: DatabaseConnection):
        super().__init__(connection_config)
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not available for MongoDB connections")
    
    def connect(self) -> bool:
        """Establish MongoDB connection."""
        try:
            # Parse connection string for MongoDB
            if self.connection_config.connection_string.startswith('mongodb://'):
                self.connection = pymongo.MongoClient(self.connection_config.connection_string)
            else:
                # Build connection string from components
                auth_part = f"{self.connection_config.username}:" if self.connection_config.username else ""
                connection_string = f"mongodb://{auth_part}@{self.connection_config.host}:{self.connection_config.port}/{self.connection_config.database_name}"
                self.connection = pymongo.MongoClient(connection_string)
            
            # Test connection
            self.connection.admin.command('ping')
            self.is_connected = True
            self.connection_config.last_used = datetime.now()
            self.connection_config.is_active = True
            logger.info(f"Connected to MongoDB: {self.connection_config.database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Close MongoDB connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.is_connected = False
                self.connection_config.is_active = False
                logger.info("MongoDB connection closed")
                return True
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
        return False
    
    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            if not self.is_connected:
                return False
            self.connection.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def discover_schema(self) -> Optional[DatabaseSchema]:
        """Discover MongoDB schema."""
        try:
            if not self.is_connected:
                return None
            
            db = self.connection[self.connection_config.database_name]
            
            # Get collections (tables)
            collections = db.list_collection_names()
            tables = []
            
            for collection_name in collections:
                collection = db[collection_name]
                
                # Get document count
                document_count = collection.count_documents({})
                
                # Get sample document for schema analysis
                sample_doc = collection.find_one()
                
                # Analyze document structure
                if sample_doc:
                    columns = self._analyze_mongo_document(sample_doc)
                else:
                    columns = []
                
                tables.append({
                    'name': collection_name,
                    'type': 'collection',
                    'row_count': document_count,
                    'schema': 'default',
                    'columns': columns
                })
            
            # MongoDB doesn't have traditional views, stored procedures, etc.
            # but we can discover indexes
            indexes = []
            for collection_name in collections:
                collection = db[collection_name]
                collection_indexes = collection.list_indexes()
                for index in collection_indexes:
                    indexes.append({
                        'name': index['name'],
                        'table': collection_name,
                        'type': 'index',
                        'is_unique': index.get('unique', False),
                        'is_primary_key': index.get('key', []) == [('_id', 1)]
                    })
            
            # Create schema hash
            schema_data = json.dumps({
                'tables': tables,
                'indexes': indexes
            }, sort_keys=True)
            schema_hash = hashlib.sha256(schema_data.encode()).hexdigest()
            
            return DatabaseSchema(
                connection_id=self.connection_config.connection_id,
                database_type=self.connection_config.database_type,
                database_name=self.connection_config.database_name,
                tables=tables,
                views=[],
                stored_procedures=[],
                functions=[],
                indexes=indexes,
                foreign_keys=[],
                discovered_at=datetime.now(),
                schema_hash=schema_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to discover MongoDB schema: {e}")
            return None
    
    def _analyze_mongo_document(self, doc: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Analyze MongoDB document structure recursively."""
        columns = []
        
        for key, value in doc.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Nested document
                nested_columns = self._analyze_mongo_document(value, full_key)
                columns.extend(nested_columns)
            elif isinstance(value, list) and value:
                # Array - analyze first element
                if isinstance(value[0], dict):
                    nested_columns = self._analyze_mongo_document(value[0], f"{full_key}[]")
                    columns.extend(nested_columns)
                else:
                    columns.append({
                        'name': full_key,
                        'data_type': f'array[{type(value[0]).__name__}]',
                        'is_nullable': True,
                        'default_value': None,
                        'max_length': None,
                        'precision': None,
                        'scale': None,
                        'position': len(columns)
                    })
            else:
                # Simple field
                columns.append({
                    'name': full_key,
                    'data_type': type(value).__name__,
                    'is_nullable': value is None,
                    'default_value': None,
                    'max_length': None,
                    'precision': None,
                    'scale': None,
                    'position': len(columns)
                })
        
        return columns
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute MongoDB query (aggregation pipeline or find)."""
        try:
            if not self.is_connected:
                return []
            
            # Parse query - could be JSON aggregation pipeline or collection name
            try:
                # Try to parse as aggregation pipeline
                pipeline = json.loads(query)
                if isinstance(pipeline, list):
                    # It's an aggregation pipeline
                    collection_name = params.get('collection') if params else None
                    if not collection_name:
                        raise ValueError("Collection name required for aggregation pipeline")
                    
                    db = self.connection[self.connection_config.database_name]
                    collection = db[collection_name]
                    results = list(collection.aggregate(pipeline))
                    return results
                else:
                    raise ValueError("Invalid aggregation pipeline")
                    
            except (json.JSONDecodeError, ValueError):
                # Treat as collection name and return sample documents
                collection_name = query
                limit = params.get('limit', 100) if params else 100
                
                db = self.connection[self.connection_config.database_name]
                collection = db[collection_name]
                results = list(collection.find().limit(limit))
                return results
                
        except Exception as e:
            logger.error(f"MongoDB query execution failed: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get MongoDB collection information."""
        try:
            if not self.is_connected:
                return None
            
            db = self.connection[self.connection_config.database_name]
            collection = db[table_name]
            
            # Get document count
            document_count = collection.count_documents({})
            
            # Get sample document for schema analysis
            sample_doc = collection.find_one()
            columns = self._analyze_mongo_document(sample_doc) if sample_doc else []
            
            return {
                'table_name': table_name,
                'columns': columns,
                'row_count': document_count,
                'connection_id': self.connection_config.connection_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get MongoDB collection info: {e}")
            return None
    
    def get_sample_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from MongoDB collection."""
        try:
            if not self.is_connected:
                return []
            
            db = self.connection[self.connection_config.database_name]
            collection = db[table_name]
            results = list(collection.find().limit(limit))
            return results
            
        except Exception as e:
            logger.error(f"Failed to get sample data from MongoDB: {e}")
            return []

class DatabaseManager:
    """Main database manager for handling connections and operations."""
    
    def __init__(self):
        self.connections: Dict[str, DatabaseConnector] = {}
        self.connection_configs: Dict[str, DatabaseConnection] = {}
        self.schemas: Dict[str, DatabaseSchema] = {}
        
        # Initialize credential vault and memory if available
        self.credential_vault = None
        self.memory_manager = None
        
        if CREDENTIAL_VAULT_AVAILABLE:
            try:
                from bin.credential_vault import get_global_credential_vault
                self.credential_vault = get_global_credential_vault()
                logger.info("Credential vault initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize credential vault: {e}")
        
        if MEMORY_AVAILABLE:
            try:
                self.memory_manager = ContextMemoryManager()
                logger.info("Memory manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory manager: {e}")
    
    def create_connection(self, 
                         database_type: str,
                         host: str,
                         port: int,
                         database_name: str,
                         username: str = None,
                         password: str = None,
                         additional_params: Dict[str, Any] = None) -> Optional[str]:
        """Create a new database connection."""
        try:
            # Generate connection ID
            connection_id = self._generate_connection_id(database_type, host, port, database_name)
            
            # Store credentials in vault if available
            if self.credential_vault and password:
                credential_key = f"db_{connection_id}"
                self.credential_vault.add_credential(
                    credential_key,
                    username,
                    password,
                    f"Database connection for {database_type} at {host}:{port}/{database_name}"
                )
            
            # Build connection string
            connection_string = self._build_connection_string(
                database_type, host, port, database_name, username, password, additional_params
            )
            
            # Create connection config
            connection_config = DatabaseConnection(
                connection_id=connection_id,
                database_type=database_type,
                host=host,
                port=port,
                database_name=database_name,
                username=username,
                connection_string=connection_string,
                additional_params=additional_params or {},
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            # Store connection config
            self.connection_configs[connection_id] = connection_config
            
            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.store_in_memory(
                    'DATABASE_CONNECTIONS',
                    connection_id,
                    connection_config.to_dict(),
                    'database_connection',
                    ttl_hours=24 * 7  # 1 week
                )
            
            logger.info(f"Created connection config: {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None
    
    def connect_to_database(self, connection_id: str) -> bool:
        """Establish connection to database."""
        try:
            if connection_id not in self.connection_configs:
                logger.error(f"Connection ID not found: {connection_id}")
                return False
            
            connection_config = self.connection_configs[connection_id]
            
            # Create appropriate connector
            if connection_config.database_type.lower() in ['mssql', 'sqlserver', 'azure-sql']:
                connector = MSSQLConnector(connection_config)
            elif connection_config.database_type.lower() in ['mongodb', 'mongo']:
                connector = MongoDBConnector(connection_config)
            else:
                logger.error(f"Unsupported database type: {connection_config.database_type}")
                return False
            
            # Establish connection
            if connector.connect():
                self.connections[connection_id] = connector
                logger.info(f"Connected to database: {connection_id}")
                return True
            else:
                logger.error(f"Failed to connect to database: {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def disconnect_from_database(self, connection_id: str) -> bool:
        """Disconnect from database."""
        try:
            if connection_id in self.connections:
                connector = self.connections[connection_id]
                if connector.disconnect():
                    del self.connections[connection_id]
                    logger.info(f"Disconnected from database: {connection_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            return False
    
    def discover_schema(self, connection_id: str) -> Optional[DatabaseSchema]:
        """Discover database schema."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return False
            
            connector = self.connections[connection_id]
            schema = connector.discover_schema()
            
            if schema:
                self.schemas[connection_id] = schema
                
                # Store in memory if available
                if self.memory_manager:
                    self.memory_manager.store_in_memory(
                        'DATABASE_SCHEMAS',
                        connection_id,
                        schema.to_dict(),
                        'database_schema',
                        ttl_hours=24 * 30  # 30 days
                    )
                
                logger.info(f"Schema discovered for: {connection_id}")
                return schema
            
            return None
            
        except Exception as e:
            logger.error(f"Error discovering schema: {e}")
            return None
    
    def execute_query(self, connection_id: str, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query on database."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def get_table_info(self, connection_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return None
            
            connector = self.connections[connection_id]
            return connector.get_table_info(table_name)
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return None
    
    def get_sample_data(self, connection_id: str, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from table."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.get_sample_data(table_name, limit)
            
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return []
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all available connections."""
        connections = []
        for connection_id, config in self.connection_configs.items():
            connection_info = {
                'connection_id': connection_id,
                'database_type': config.database_type,
                'host': config.host,
                'port': config.port,
                'database_name': config.database_name,
                'username': config.username,
                'is_connected': connection_id in self.connections,
                'created_at': config.created_at.isoformat(),
                'last_used': config.last_used.isoformat()
            }
            connections.append(connection_info)
        
        return connections
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection status."""
        if connection_id not in self.connection_configs:
            return None
        
        config = self.connection_configs[connection_id]
        connector = self.connections.get(connection_id)
        
        status = {
            'connection_id': connection_id,
            'database_type': config.database_type,
            'host': config.host,
            'port': config.port,
            'database_name': config.database_name,
            'is_connected': connector is not None,
            'connection_working': connector.test_connection() if connector else False,
            'created_at': config.created_at.isoformat(),
            'last_used': config.last_used.isoformat()
        }
        
        return status
    
    def _generate_connection_id(self, database_type: str, host: str, port: int, database_name: str) -> str:
        """Generate unique connection ID."""
        unique_string = f"{database_type}_{host}_{port}_{database_name}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _build_connection_string(self, 
                                database_type: str,
                                host: str,
                                port: int,
                                database_name: str,
                                username: str = None,
                                password: str = None,
                                additional_params: Dict[str, Any] = None) -> str:
        """Build database connection string."""
        if database_type.lower() in ['mssql', 'sqlserver', 'azure-sql']:
            # MSSQL connection string
            auth_part = f"{username}:{password}@" if username and password else ""
            params = additional_params or {}
            param_string = ";".join([f"{k}={v}" for k, v in params.items()]) if params else ""
            
            if param_string:
                return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host},{port};DATABASE={database_name};UID={username};PWD={password};{param_string}"
            else:
                return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host},{port};DATABASE={database_name};UID={username};PWD={password}"
        
        elif database_type.lower() in ['mongodb', 'mongo']:
            # MongoDB connection string
            auth_part = f"{username}:{password}@" if username and password else ""
            return f"mongodb://{auth_part}{host}:{port}/{database_name}"
        
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    def cleanup(self):
        """Cleanup all connections."""
        for connection_id in list(self.connections.keys()):
            self.disconnect_from_database(connection_id)

# Global instance
_database_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

if __name__ == "__main__":
    # Test the database manager
    manager = get_database_manager()
    
    print("🧪 Testing Database Manager...")
    print(f"MSSQL Available: {MSSQL_AVAILABLE}")
    print(f"MongoDB Available: {MONGODB_AVAILABLE}")
    print(f"Credential Vault: {CREDENTIAL_VAULT_AVAILABLE}")
    print(f"Memory Manager: {MEMORY_AVAILABLE}")
    
    # Test connection creation
    connection_id = manager.create_connection(
        database_type="mssql",
        host="localhost",
        port=1433,
        database_name="testdb",
        username="testuser",
                    password="test_password"
    )
    
    if connection_id:
        print(f"✅ Connection created: {connection_id}")
        
        # List connections
        connections = manager.list_connections()
        print(f"Connections: {connections}")
        
        # Get status
        status = manager.get_connection_status(connection_id)
        print(f"Status: {status}")
    else:
        print("❌ Failed to create connection")
