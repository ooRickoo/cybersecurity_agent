#!/usr/bin/env python3
"""
SQLite Manager - Comprehensive Database Operations for Cybersecurity Analysis
Provides SQLite database management capabilities for storing and querying cybersecurity data.
"""

import sqlite3
import json
import logging
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Types of databases supported."""
    THREAT_INTEL = "threat_intel"
    INCIDENT_DATA = "incident_data"
    VULNERABILITY_DATA = "vulnerability_data"
    NETWORK_DATA = "network_data"
    LOG_DATA = "log_data"
    FORENSIC_DATA = "forensic_data"
    GENERAL = "general"

@dataclass
class DatabaseSchema:
    """Database schema definition."""
    table_name: str
    columns: Dict[str, str]  # column_name: data_type
    indexes: List[str] = None
    constraints: List[str] = None

class SQLiteManager:
    """Comprehensive SQLite database manager for cybersecurity data."""
    
    def __init__(self, db_path: str = "cybersecurity_data.db"):
        """Initialize SQLite manager."""
        self.db_path = Path(db_path)
        self.connection = None
        self.schemas = self._initialize_schemas()
        self.logger = logging.getLogger(__name__)
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"🚀 SQLite Manager initialized with database: {self.db_path}")
    
    def _initialize_schemas(self) -> Dict[str, DatabaseSchema]:
        """Initialize database schemas for cybersecurity data."""
        return {
            "threat_indicators": DatabaseSchema(
                table_name="threat_indicators",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "indicator": "TEXT NOT NULL",
                    "indicator_type": "TEXT NOT NULL",  # IP, Domain, Hash, URL
                    "threat_type": "TEXT",  # Malware, Phishing, C2, etc.
                    "confidence": "REAL DEFAULT 0.0",
                    "source": "TEXT",
                    "first_seen": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "last_seen": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "tags": "TEXT",  # JSON array of tags
                    "metadata": "TEXT"  # JSON metadata
                },
                indexes=["indicator", "indicator_type", "threat_type", "first_seen"],
                constraints=["UNIQUE(indicator, indicator_type)"]
            ),
            "incidents": DatabaseSchema(
                table_name="incidents",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "incident_id": "TEXT UNIQUE NOT NULL",
                    "title": "TEXT NOT NULL",
                    "description": "TEXT",
                    "severity": "TEXT",  # Low, Medium, High, Critical
                    "status": "TEXT DEFAULT 'Open'",  # Open, In Progress, Closed
                    "assigned_to": "TEXT",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "resolved_at": "TIMESTAMP",
                    "affected_assets": "TEXT",  # JSON array
                    "indicators": "TEXT",  # JSON array of threat indicators
                    "evidence": "TEXT",  # JSON array of evidence files
                    "notes": "TEXT"
                },
                indexes=["incident_id", "severity", "status", "created_at"],
                constraints=[]
            ),
            "vulnerabilities": DatabaseSchema(
                table_name="vulnerabilities",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "cve_id": "TEXT UNIQUE",
                    "title": "TEXT NOT NULL",
                    "description": "TEXT",
                    "severity": "TEXT",  # Low, Medium, High, Critical
                    "cvss_score": "REAL",
                    "cvss_vector": "TEXT",
                    "affected_products": "TEXT",  # JSON array
                    "published_date": "DATE",
                    "last_modified": "DATE",
                    "references": "TEXT",  # JSON array of URLs
                    "exploit_available": "BOOLEAN DEFAULT FALSE",
                    "patch_available": "BOOLEAN DEFAULT FALSE",
                    "tags": "TEXT"  # JSON array
                },
                indexes=["cve_id", "severity", "cvss_score", "published_date"],
                constraints=[]
            ),
            "network_events": DatabaseSchema(
                table_name="network_events",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "timestamp": "TIMESTAMP NOT NULL",
                    "source_ip": "TEXT",
                    "dest_ip": "TEXT",
                    "source_port": "INTEGER",
                    "dest_port": "INTEGER",
                    "protocol": "TEXT",
                    "packet_size": "INTEGER",
                    "flags": "TEXT",
                    "payload_hash": "TEXT",
                    "event_type": "TEXT",  # Connection, Data Transfer, etc.
                    "threat_level": "TEXT",  # Low, Medium, High, Critical
                    "metadata": "TEXT"  # JSON metadata
                },
                indexes=["timestamp", "source_ip", "dest_ip", "event_type", "threat_level"],
                constraints=[]
            ),
            "log_entries": DatabaseSchema(
                table_name="log_entries",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "timestamp": "TIMESTAMP NOT NULL",
                    "source": "TEXT NOT NULL",  # System, Application, Security
                    "log_level": "TEXT",  # INFO, WARN, ERROR, CRITICAL
                    "message": "TEXT NOT NULL",
                    "user": "TEXT",
                    "ip_address": "TEXT",
                    "event_id": "TEXT",
                    "category": "TEXT",
                    "tags": "TEXT",  # JSON array
                    "raw_log": "TEXT"  # Original log line
                },
                indexes=["timestamp", "source", "log_level", "user", "ip_address"],
                constraints=[]
            ),
            "forensic_artifacts": DatabaseSchema(
                table_name="forensic_artifacts",
                columns={
                    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                    "artifact_id": "TEXT UNIQUE NOT NULL",
                    "artifact_type": "TEXT NOT NULL",  # File, Registry, Memory, etc.
                    "file_path": "TEXT",
                    "file_hash": "TEXT",
                    "file_size": "INTEGER",
                    "created_time": "TIMESTAMP",
                    "modified_time": "TIMESTAMP",
                    "accessed_time": "TIMESTAMP",
                    "owner": "TEXT",
                    "permissions": "TEXT",
                    "content_hash": "TEXT",
                    "extracted_data": "TEXT",  # JSON extracted data
                    "analysis_results": "TEXT",  # JSON analysis results
                    "tags": "TEXT",  # JSON array
                    "case_id": "TEXT"
                },
                indexes=["artifact_id", "artifact_type", "file_hash", "case_id"],
                constraints=[]
            )
        }
    
    def _initialize_database(self):
        """Initialize database with schemas."""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create tables
            for schema_name, schema in self.schemas.items():
                self._create_table(schema)
            
            # Create indexes
            for schema_name, schema in self.schemas.items():
                if schema.indexes:
                    self._create_indexes(schema)
            
            self.connection.commit()
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _create_table(self, schema: DatabaseSchema):
        """Create a table from schema."""
        columns = []
        for col_name, col_type in schema.columns.items():
            columns.append(f"{col_name} {col_type}")
        
        # Add constraints
        if schema.constraints:
            columns.extend(schema.constraints)
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema.table_name} (
            {', '.join(columns)}
        )
        """
        
        self.connection.execute(create_sql)
    
    def _create_indexes(self, schema: DatabaseSchema):
        """Create indexes for a table."""
        for index_col in schema.indexes:
            index_name = f"idx_{schema.table_name}_{index_col}"
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {schema.table_name} ({index_col})
            """
            self.connection.execute(create_index_sql)
    
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> int:
        """Insert data into a table."""
        try:
            # Convert lists/dicts to JSON strings
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    processed_data[key] = json.dumps(value)
                else:
                    processed_data[key] = value
            
            columns = list(processed_data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            values = list(processed_data.values())
            
            sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            cursor = self.connection.execute(sql, values)
            self.connection.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"❌ Failed to insert data into {table_name}: {e}")
            self.connection.rollback()
            raise
    
    def query_data(self, sql: str, parameters: Tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        try:
            cursor = self.connection.execute(sql, parameters or ())
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                result = dict(row)
                # Parse JSON fields
                for key, value in result.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            result[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
    
    def update_data(self, table_name: str, data: Dict[str, Any], where_clause: str, where_params: Tuple = None) -> int:
        """Update data in a table."""
        try:
            # Convert lists/dicts to JSON strings
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    processed_data[key] = json.dumps(value)
                else:
                    processed_data[key] = value
            
            set_clause = ', '.join([f"{key} = ?" for key in processed_data.keys()])
            values = list(processed_data.values())
            
            if where_params:
                values.extend(where_params)
            
            sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            
            cursor = self.connection.execute(sql, values)
            self.connection.commit()
            
            return cursor.rowcount
            
        except Exception as e:
            logger.error(f"❌ Failed to update data in {table_name}: {e}")
            self.connection.rollback()
            raise
    
    def delete_data(self, table_name: str, where_clause: str, where_params: Tuple = None) -> int:
        """Delete data from a table."""
        try:
            sql = f"DELETE FROM {table_name} WHERE {where_clause}"
            cursor = self.connection.execute(sql, where_params or ())
            self.connection.commit()
            
            return cursor.rowcount
            
        except Exception as e:
            logger.error(f"❌ Failed to delete data from {table_name}: {e}")
            self.connection.rollback()
            raise
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        try:
            sql = f"PRAGMA table_info({table_name})"
            return self.query_data(sql)
        except Exception as e:
            logger.error(f"❌ Failed to get table info for {table_name}: {e}")
            return []
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics."""
        try:
            # Get row count
            count_sql = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = self.query_data(count_sql)
            row_count = count_result[0]['count'] if count_result else 0
            
            # Get table size
            size_sql = f"SELECT COUNT(*) * AVG(LENGTH(CAST(*) AS TEXT)) as size FROM {table_name}"
            size_result = self.query_data(size_sql)
            estimated_size = size_result[0]['size'] if size_result else 0
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "estimated_size_bytes": estimated_size,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get table stats for {table_name}: {e}")
            return {"table_name": table_name, "error": str(e)}
    
    def export_to_csv(self, table_name: str, output_path: str, where_clause: str = None, where_params: Tuple = None) -> bool:
        """Export table data to CSV."""
        try:
            sql = f"SELECT * FROM {table_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            
            df = pd.read_sql_query(sql, self.connection, params=where_params)
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Exported {len(df)} rows from {table_name} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export {table_name} to CSV: {e}")
            return False
    
    def import_from_csv(self, table_name: str, csv_path: str, if_exists: str = "append") -> bool:
        """Import data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            self.connection.commit()
            
            logger.info(f"✅ Imported {len(df)} rows from {csv_path} to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to import from {csv_path} to {table_name}: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Close current connection
            if self.connection:
                self.connection.close()
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            # Reconnect
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            logger.info(f"✅ Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to backup database: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("🔒 Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Example usage and testing
if __name__ == "__main__":
    # Test the SQLite manager
    with SQLiteManager("test_cybersecurity.db") as db:
        # Insert sample threat indicator
        threat_data = {
            "indicator": "192.168.1.100",
            "indicator_type": "IP",
            "threat_type": "Malware",
            "confidence": 0.85,
            "source": "Threat Intelligence Feed",
            "tags": ["malware", "c2", "suspicious"],
            "metadata": {"country": "US", "asn": "12345"}
        }
        
        indicator_id = db.insert_data("threat_indicators", threat_data)
        print(f"✅ Inserted threat indicator with ID: {indicator_id}")
        
        # Query threat indicators
        results = db.query_data("SELECT * FROM threat_indicators WHERE threat_type = ?", ("Malware",))
        print(f"✅ Found {len(results)} malware indicators")
        
        # Get table stats
        stats = db.get_table_stats("threat_indicators")
        print(f"✅ Table stats: {stats}")
        
        # Export to CSV
        db.export_to_csv("threat_indicators", "threat_indicators_export.csv")
        print("✅ Exported to CSV")
