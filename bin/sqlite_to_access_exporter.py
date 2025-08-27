#!/usr/bin/env python3
"""
SQLite to MS Access Exporter
Exports SQLite databases to MS Access format (.mdb/.accdb) for further analysis and use.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime
import os

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

try:
    import win32com.client
    WIN32COM_AVAILABLE = True
except ImportError:
    WIN32COM_AVAILABLE = False

class SQLiteToAccessExporter:
    """Export SQLite databases to MS Access format."""
    
    def __init__(self, output_dir: str = "session-outputs/access_exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Export status
        self.export_status = {
            'success': False,
            'tables_exported': 0,
            'total_rows': 0,
            'errors': [],
            'warnings': [],
            'output_file': None
        }
    
    def export_sqlite_to_access(self, sqlite_db_path: str, 
                               access_filename: str = None,
                               include_data: bool = True,
                               include_indexes: bool = True,
                               include_constraints: bool = True,
                               max_rows_per_table: int = 10000) -> Dict[str, Any]:
        """
        Export SQLite database to MS Access format.
        
        Args:
            sqlite_db_path: Path to SQLite database file
            access_filename: Name for the Access file (without extension)
            include_data: Whether to include data in tables
            include_indexes: Whether to include indexes
            include_constraints: Whether to include constraints
            max_rows_per_table: Maximum rows to export per table
        
        Returns:
            Dictionary with export status and details
        """
        try:
            # Validate SQLite file
            if not Path(sqlite_db_path).exists():
                raise FileNotFoundError(f"SQLite database not found: {sqlite_db_path}")
            
            # Generate Access filename if not provided
            if access_filename is None:
                db_name = Path(sqlite_db_path).stem
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                access_filename = f"{db_name}_export_{timestamp}"
            
            # Connect to SQLite database
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_conn.row_factory = sqlite3.Row
            
            # Get database schema
            schema_info = self._get_sqlite_schema(sqlite_conn)
            
            # Create Access database
            access_file_path = self._create_access_database(access_filename, schema_info)
            
            if access_file_path:
                # Export tables
                self._export_tables(sqlite_conn, access_file_path, schema_info, 
                                  include_data, max_rows_per_table)
                
                # Export indexes if requested
                if include_indexes:
                    self._export_indexes(sqlite_conn, access_file_path, schema_info)
                
                # Export constraints if requested
                if include_constraints:
                    self._export_constraints(sqlite_conn, access_file_path, schema_info)
                
                # Create metadata file
                self._create_metadata_file(access_file_path, schema_info)
                
                self.export_status['success'] = True
                self.export_status['output_file'] = str(access_file_path)
                
                self.logger.info(f"Successfully exported SQLite database to: {access_file_path}")
            
            sqlite_conn.close()
            
        except Exception as e:
            self.export_status['errors'].append(str(e))
            self.logger.error(f"Export failed: {e}")
        
        return self.export_status
    
    def _get_sqlite_schema(self, sqlite_conn: sqlite3.Connection) -> Dict[str, Any]:
        """Extract schema information from SQLite database."""
        schema_info = {
            'tables': [],
            'indexes': [],
            'constraints': [],
            'database_info': {}
        }
        
        try:
            cursor = sqlite_conn.cursor()
            
            # Get table information
            cursor.execute("""
                SELECT name, sql FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            tables = cursor.fetchall()
            for table in tables:
                table_name = table['name']
                table_sql = table['sql']
                
                # Get column information
                cursor.execute("PRAGMA table_info(?)", (table_name,))
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute("SELECT COUNT(*) as count FROM ?", (table_name,))
                row_count = cursor.fetchone()['count']
                
                table_info = {
                    'name': table_name,
                    'sql': table_sql,
                    'columns': [],
                    'row_count': row_count,
                    'primary_key': None,
                    'foreign_keys': []
                }
                
                # Process columns
                for col in columns:
                    column_info = {
                        'name': col['name'],
                        'type': col['type'],
                        'not_null': bool(col['notnull']),
                        'default_value': col['dflt_value'],
                        'primary_key': bool(col['pk'])
                    }
                    
                    if col['pk']:
                        table_info['primary_key'] = col['name']
                    
                    table_info['columns'].append(column_info)
                
                # Get foreign key information
                cursor.execute("PRAGMA foreign_key_list(?)", (table_name,))
                foreign_keys = cursor.fetchall()
                for fk in foreign_keys:
                    fk_info = {
                        'column': fk['from'],
                        'references_table': fk['table'],
                        'references_column': fk['to']
                    }
                    table_info['foreign_keys'].append(fk_info)
                
                schema_info['tables'].append(table_info)
            
            # Get index information
            cursor.execute("""
                SELECT name, tbl_name, sql FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name, name
            """)
            
            indexes = cursor.fetchall()
            for idx in indexes:
                index_info = {
                    'name': idx['name'],
                    'table': idx['tbl_name'],
                    'sql': idx['sql']
                }
                schema_info['indexes'].append(index_info)
            
            # Get database information
            cursor.execute("PRAGMA database_list")
            db_list = cursor.fetchall()
            schema_info['database_info'] = {
                'main_db': db_list[0]['file'] if db_list else None,
                'total_tables': len(schema_info['tables']),
                'total_indexes': len(schema_info['indexes']),
                'exported_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract schema: {e}")
            raise
        
        return schema_info
    
    def _create_access_database(self, access_filename: str, schema_info: Dict[str, Any]) -> Optional[Path]:
        """Create MS Access database file."""
        try:
            # Try different methods to create Access database
            
            # Method 1: Using pyodbc (if available)
            if PYODBC_AVAILABLE:
                return self._create_access_with_pyodbc(access_filename, schema_info)
            
            # Method 2: Using win32com (if available)
            elif WIN32COM_AVAILABLE:
                return self._create_access_with_win32com(access_filename, schema_info)
            
            # Method 3: Create CSV exports with Access import instructions
            else:
                return self._create_csv_with_access_instructions(access_filename, schema_info)
                
        except Exception as e:
            self.logger.error(f"Failed to create Access database: {e}")
            return None
    
    def _create_access_with_pyodbc(self, access_filename: str, schema_info: Dict[str, Any]) -> Optional[Path]:
        """Create Access database using pyodbc."""
        try:
            access_file_path = self.output_dir / f"{access_filename}.accdb"
            
            # Create connection string for Access
            conn_str = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={access_file_path}"
            
            # Create database using pyodbc
            conn = pyodbc.connect(conn_str, autocommit=True)
            cursor = conn.cursor()
            
            # Create tables
            for table_info in schema_info['tables']:
                self._create_access_table_pyodbc(cursor, table_info)
            
            conn.close()
            
            self.logger.info(f"Created Access database using pyodbc: {access_file_path}")
            return access_file_path
            
        except Exception as e:
            self.logger.error(f"pyodbc method failed: {e}")
            return None
    
    def _create_access_with_win32com(self, access_filename: str, schema_info: Dict[str, Any]) -> Optional[Path]:
        """Create Access database using win32com."""
        try:
            access_file_path = self.output_dir / f"{access_filename}.accdb"
            
            # Create Access application object
            access_app = win32com.client.Dispatch("Access.Application")
            access_app.Visible = False
            
            # Create new database
            access_app.DBEngine.CreateDatabase(str(access_file_path), win32com.client.constants.dbLangGeneral)
            
            # Open database
            db = access_app.OpenCurrentDatabase(str(access_file_path))
            
            # Create tables
            for table_info in schema_info['tables']:
                self._create_access_table_win32com(access_app, table_info)
            
            # Close database and application
            access_app.CloseCurrentDatabase()
            access_app.Quit()
            
            self.logger.info(f"Created Access database using win32com: {access_file_path}")
            return access_file_path
            
        except Exception as e:
            self.logger.error(f"win32com method failed: {e}")
            return None
    
    def _create_csv_with_access_instructions(self, access_filename: str, schema_info: Dict[str, Any]) -> Path:
        """Create CSV exports with Access import instructions."""
        csv_dir = self.output_dir / access_filename
        csv_dir.mkdir(exist_ok=True)
        
        # Create Access import instructions
        instructions = self._generate_access_import_instructions(schema_info)
        
        instructions_file = csv_dir / "ACCESS_IMPORT_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        self.logger.info(f"Created CSV exports with Access import instructions in: {csv_dir}")
        return csv_dir
    
    def _generate_access_import_instructions(self, schema_info: Dict[str, Any]) -> str:
        """Generate Markdown instructions for importing CSV files into Access."""
        instructions = f"""# MS Access Import Instructions

## Overview
This document provides step-by-step instructions for importing the exported CSV files into Microsoft Access.

## Database Information
- **Total Tables**: {schema_info['database_info']['total_tables']}
- **Total Indexes**: {schema_info['database_info']['total_indexes']}
- **Exported At**: {schema_info['database_info']['exported_at']}

## Import Steps

### 1. Create New Access Database
1. Open Microsoft Access
2. Click "Blank database"
3. Choose a location and name for your database
4. Click "Create"

### 2. Import Tables
For each table in the list below:

1. Go to External Data tab
2. Click "Text File" in the Import & Link group
3. Browse to the CSV file location
4. Select "Import the source data into a new table in the current database"
5. Click "OK"
6. Follow the import wizard:
   - Choose "Delimited" and click "Next"
   - Select "Comma" as delimiter and click "Next"
   - Review field names and types, click "Next"
   - Choose "Let Access add primary key" or specify your own
   - Click "Finish"

### 3. Table Details

"""
        
        for table_info in schema_info['tables']:
            instructions += f"""
#### Table: {table_info['name']}
- **Rows**: {table_info['row_count']}
- **Columns**: {len(table_info['columns'])}
- **Primary Key**: {table_info['primary_key'] or 'None'}
- **CSV File**: `{table_info['name']}.csv`

**Columns:**
"""
            for col in table_info['columns']:
                instructions += f"- {col['name']} ({col['type']})"
                if col['not_null']:
                    instructions += " - NOT NULL"
                if col['default_value']:
                    instructions += f" - Default: {col['default_value']}"
                instructions += "\n"
        
        instructions += f"""
### 4. Create Relationships
After importing all tables, create relationships based on foreign keys:

"""
        
        for table_info in schema_info['tables']:
            if table_info['foreign_keys']:
                instructions += f"\n**Table: {table_info['name']}**\n"
                for fk in table_info['foreign_keys']:
                    instructions += f"- {fk['column']} → {fk['references_table']}.{fk['references_column']}\n"
        
        instructions += """
### 5. Create Indexes
Consider creating indexes on frequently queried columns for better performance.

### 6. Verify Data
- Check row counts match expected values
- Verify data types are correct
- Test relationships between tables

## Notes
- Some SQLite data types may be converted to Access equivalents
- Large tables may take time to import
- Consider splitting very large tables if Access performance is slow
"""
        
        return instructions
    
    def _export_tables(self, sqlite_conn: sqlite3.Connection, access_file_path: Path, 
                       schema_info: Dict[str, Any], include_data: bool, max_rows_per_table: int):
        """Export table data to Access or CSV."""
        try:
            cursor = sqlite_conn.cursor()
            
            for table_info in schema_info['tables']:
                table_name = table_info['name']
                
                if include_data:
                    # Export data
                    cursor.execute("SELECT * FROM ?", (table_name,))
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Limit rows if specified
                        if len(rows) > max_rows_per_table:
                            rows = rows[:max_rows_per_table]
                            self.export_status['warnings'].append(
                                f"Table {table_name}: Limited to {max_rows_per_table} rows (total: {len(rows)})"
                            )
                        
                        # Convert to DataFrame for easier handling
                        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
                        
                        # Export based on method
                        if access_file_path.is_file() and access_file_path.suffix in ['.mdb', '.accdb']:
                            # Export to Access
                            self._export_table_to_access(access_file_path, table_name, df)
                        else:
                            # Export to CSV
                            csv_file = access_file_path / f"{table_name}.csv"
                            df.to_csv(csv_file, index=False)
                        
                        self.export_status['tables_exported'] += 1
                        self.export_status['total_rows'] += len(rows)
                        
                        self.logger.info(f"Exported table {table_name}: {len(rows)} rows")
                    else:
                        self.logger.info(f"Table {table_name}: No data to export")
                else:
                    self.logger.info(f"Table {table_name}: Data export skipped")
                    
        except Exception as e:
            self.export_status['errors'].append(f"Table export failed: {e}")
            self.logger.error(f"Table export failed: {e}")
    
    def _export_table_to_access(self, access_file_path: Path, table_name: str, df: pd.DataFrame):
        """Export table data to Access database."""
        try:
            if PYODBC_AVAILABLE:
                self._export_table_to_access_pyodbc(access_file_path, table_name, df)
            elif WIN32COM_AVAILABLE:
                self._export_table_to_access_win32com(access_file_path, table_name, df)
        except Exception as e:
            self.logger.error(f"Failed to export table {table_name} to Access: {e}")
            # Fallback to CSV
            csv_file = access_file_path.parent / f"{table_name}.csv"
            df.to_csv(csv_file, index=False)
    
    def _export_table_to_access_pyodbc(self, access_file_path: Path, table_name: str, df: pd.DataFrame):
        """Export table to Access using pyodbc."""
        conn_str = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={access_file_path}"
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        columns = []
        for col_name, dtype in df.dtypes.items():
            if 'int' in str(dtype):
                col_type = 'INTEGER'
            elif 'float' in str(dtype):
                col_type = 'DOUBLE'
            elif 'datetime' in str(dtype):
                col_type = 'DATETIME'
            else:
                col_type = 'TEXT(255)'
            
            columns.append(f"[{col_name}] {col_type}")
        
        create_sql = f"CREATE TABLE [{table_name}] ({', '.join(columns)})"
        cursor.execute(create_sql)
        
        # Insert data
        for _, row in df.iterrows():
            placeholders = ', '.join(['?' for _ in row])
            insert_sql = f"INSERT INTO [{table_name}] VALUES ({placeholders})"
            cursor.execute(insert_sql, tuple(row))
        
        conn.close()
    
    def _export_table_to_access_win32com(self, access_file_path: Path, table_name: str, df: pd.DataFrame):
        """Export table to Access using win32com."""
        access_app = win32com.client.Dispatch("Access.Application")
        access_app.Visible = False
        
        db = access_app.OpenCurrentDatabase(str(access_file_path))
        
        # Create table
        table_def = db.TableDefs.CreateTableDef(table_name)
        
        # Add fields
        for col_name, dtype in df.dtypes.items():
            field_def = table_def.CreateField(col_name)
            
            if 'int' in str(dtype):
                field_def.Type = 3  # dbInteger
            elif 'float' in str(dtype):
                field_def.Type = 7  # dbDouble
            elif 'datetime' in str(dtype):
                field_def.Type = 8  # dbDate
            else:
                field_def.Type = 10  # dbText
                field_def.Size = 255
            
            table_def.Fields.Append(field_def)
        
        db.TableDefs.Append(table_def)
        
        # Insert data
        for _, row in df.iterrows():
            # Implementation depends on specific Access version
            pass
        
        access_app.CloseCurrentDatabase()
        access_app.Quit()
    
    def _export_indexes(self, sqlite_conn: sqlite3.Connection, access_file_path: Path, schema_info: Dict[str, Any]):
        """Export index information."""
        try:
            if access_file_path.is_file() and access_file_path.suffix in ['.mdb', '.accdb']:
                # Export indexes to Access
                pass
            else:
                # Export index information to text file
                index_file = access_file_path / "indexes.txt"
                with open(index_file, 'w') as f:
                    f.write("SQLite Indexes\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for idx in schema_info['indexes']:
                        f.write(f"Index: {idx['name']}\n")
                        f.write(f"Table: {idx['table']}\n")
                        f.write(f"SQL: {idx['sql']}\n")
                        f.write("-" * 30 + "\n\n")
                
                self.logger.info(f"Exported index information to: {index_file}")
                
        except Exception as e:
            self.logger.error(f"Index export failed: {e}")
    
    def _export_constraints(self, sqlite_conn: sqlite3.Connection, access_file_path: Path, schema_info: Dict[str, Any]):
        """Export constraint information."""
        try:
            if access_file_path.is_file() and access_file_path.suffix in ['.mdb', '.accdb']:
                # Export constraints to Access
                pass
            else:
                # Export constraint information to text file
                constraint_file = access_file_path / "constraints.txt"
                with open(constraint_file, 'w') as f:
                    f.write("SQLite Constraints\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for table_info in schema_info['tables']:
                        f.write(f"Table: {table_info['name']}\n")
                        
                        if table_info['primary_key']:
                            f.write(f"  Primary Key: {table_info['primary_key']}\n")
                        
                        if table_info['foreign_keys']:
                            f.write("  Foreign Keys:\n")
                            for fk in table_info['foreign_keys']:
                                f.write(f"    {fk['column']} → {fk['references_table']}.{fk['references_column']}\n")
                        
                        f.write("-" * 30 + "\n\n")
                
                self.logger.info(f"Exported constraint information to: {constraint_file}")
                
        except Exception as e:
            self.logger.error(f"Constraint export failed: {e}")
    
    def _create_metadata_file(self, access_file_path: Path, schema_info: Dict[str, Any]):
        """Create metadata file with export information."""
        try:
            metadata = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'export_tool': 'SQLiteToAccessExporter',
                    'version': '1.0'
                },
                'source_database': {
                    'type': 'SQLite',
                    'total_tables': schema_info['database_info']['total_tables'],
                    'total_indexes': schema_info['database_info']['total_indexes']
                },
                'export_status': self.export_status,
                'schema_summary': {
                    'tables': [
                        {
                            'name': table['name'],
                            'columns': len(table['columns']),
                            'rows': table['row_count'],
                            'primary_key': table['primary_key']
                        }
                        for table in schema_info['tables']
                    ]
                }
            }
            
            metadata_file = access_file_path / "export_metadata.json"
            if access_file_path.is_file():
                metadata_file = access_file_path.parent / "export_metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Created metadata file: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create metadata file: {e}")
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of the last export operation."""
        return {
            'status': self.export_status,
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir)
        }

# Example usage
if __name__ == "__main__":
    exporter = SQLiteToAccessExporter()
    
    # Example export
    result = exporter.export_sqlite_to_access(
        sqlite_db_path="example.db",
        access_filename="exported_database",
        include_data=True,
        max_rows_per_table=5000
    )
    
    print("Export Result:", json.dumps(result, indent=2))
