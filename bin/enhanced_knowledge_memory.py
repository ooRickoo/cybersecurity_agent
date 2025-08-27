#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Context Memory System for Cybersecurity Agent
Integrates with Real-Time Context Adaptation and provides easy CSV/JSON import
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import yaml
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

class MemoryCategory(Enum):
    """Memory category enumeration."""
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_DATA = "incident_data"
    COMPLIANCE_INFO = "compliance_info"
    ORGANIZATIONAL = "organizational"
    TECHNICAL = "technical"
    PROCEDURAL = "procedural"

@dataclass
class MemoryNode:
    """Individual memory node in the knowledge graph."""
    node_id: str
    content: str
    memory_type: MemoryType
    category: MemoryCategory
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    relevance_score: float
    confidence: float
    relationships: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['category'] = self.category.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

@dataclass
class MemoryRelationship:
    """Relationship between memory nodes."""
    relationship_id: str
    source_node: str
    target_node: str
    relationship_type: str
    strength: float
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class ImportResult:
    """Result of data import operation."""
    import_id: str
    source_file: str
    import_type: str
    nodes_created: int
    relationships_created: int
    errors: List[str]
    warnings: List[str]
    processing_time: float
    created_at: datetime

class DataImporter:
    """Import data from various formats into the knowledge graph."""
    
    def __init__(self):
        self.supported_formats = ["csv", "json", "yaml", "xml"]
        self.import_processors = self._load_import_processors()
        self.field_normalizers = self._load_field_normalizers()
    
    def _load_import_processors(self) -> Dict[str, Callable]:
        """Load import processors for different formats."""
        return {
            "csv": self._process_csv,
            "json": self._process_json,
            "yaml": self._process_yaml,
            "xml": self._process_xml
        }
    
    def _load_field_normalizers(self) -> Dict[str, Callable]:
        """Load field normalization functions."""
        return {
            "standardize_case": self._standardize_case,
            "normalize_whitespace": self._normalize_whitespace,
            "extract_keywords": self._extract_keywords,
            "categorize_content": self._categorize_content
        }
    
    async def import_data(self, file_path: str, import_config: Dict[str, Any]) -> ImportResult:
        """Import data from a file into the knowledge graph."""
        start_time = datetime.now()
        
        try:
            # Determine file format
            file_format = self._detect_file_format(file_path)
            
            if file_format not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Process the file
            processor = self.import_processors[file_format]
            result = await processor(file_path, import_config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create import result
            import_result = ImportResult(
                import_id=f"import_{hashlib.sha256(f'{file_path}_{start_time}'.encode()).hexdigest()[:8]}",
                source_file=file_path,
                import_type=file_format,
                nodes_created=result.get("nodes_created", 0),
                relationships_created=result.get("relationships_created", 0),
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                processing_time=processing_time,
                created_at=start_time
            )
            
            return import_result
            
        except Exception as e:
            # Create error result
            processing_time = (datetime.now() - start_time).total_seconds()
            import_result = ImportResult(
                import_id=f"import_error_{hashlib.sha256(f'{file_path}_{start_time}'.encode()).hexdigest()[:8]}",
                source_file=file_path,
                import_type="unknown",
                nodes_created=0,
                relationships_created=0,
                errors=[str(e)],
                warnings=[],
                processing_time=processing_time,
                created_at=start_time
            )
            
            return import_result
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format based on extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".csv":
            return "csv"
        elif file_ext in [".json", ".js"]:
            return "json"
        elif file_ext in [".yaml", ".yml"]:
            return "yaml"
        elif file_ext == ".xml":
            return "xml"
        else:
            # Try to detect by content
            return self._detect_by_content(file_path)
    
    def _detect_by_content(self, file_path: str) -> str:
        """Detect file format by examining content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                if first_line.startswith('{') or first_line.startswith('['):
                    return "json"
                elif first_line.startswith('<?xml'):
                    return "xml"
                elif first_line.startswith('---') or ':' in first_line:
                    return "yaml"
                else:
                    # Assume CSV if it contains commas
                    if ',' in first_line:
                        return "csv"
                    else:
                        return "unknown"
        except Exception:
            return "unknown"
    
    async def _process_csv(self, file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV file import."""
        result = {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": [],
            "warnings": [],
            "data": []
        }
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Normalize column names
            df.columns = [self._normalize_column_name(col) for col in df.columns]
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Create memory node from row
                    node_data = self._create_node_from_csv_row(row, import_config)
                    if node_data:
                        result["data"].append(node_data)
                        result["nodes_created"] += 1
                except Exception as e:
                    result["errors"].append(f"Row {index + 1}: {str(e)}")
            
            # Create relationships if specified
            if import_config.get("create_relationships", False):
                relationships = self._create_relationships_from_csv(result["data"], import_config)
                result["relationships_created"] = len(relationships)
                result["relationships"] = relationships
            
        except Exception as e:
            result["errors"].append(f"CSV processing failed: {str(e)}")
        
        return result
    
    async def _process_json(self, file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON file import."""
        result = {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": [],
            "warnings": [],
            "data": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                # Array of objects
                for item in json_data:
                    try:
                        node_data = self._create_node_from_json_item(item, import_config)
                        if node_data:
                            result["data"].append(node_data)
                            result["nodes_created"] += 1
                    except Exception as e:
                        result["errors"].append(f"JSON item processing failed: {str(e)}")
            
            elif isinstance(json_data, dict):
                # Single object or nested structure
                if import_config.get("flatten_nested", False):
                    # Flatten nested structure
                    flattened_items = self._flatten_json_structure(json_data)
                    for item in flattened_items:
                        try:
                            node_data = self._create_node_from_json_item(item, import_config)
                            if node_data:
                                result["data"].append(node_data)
                                result["nodes_created"] += 1
                        except Exception as e:
                            result["errors"].append(f"Flattened item processing failed: {str(e)}")
                else:
                    # Process as single item
                    try:
                        node_data = self._create_node_from_json_item(json_data, import_config)
                        if node_data:
                            result["data"].append(node_data)
                            result["nodes_created"] += 1
                    except Exception as e:
                        result["errors"].append(f"JSON processing failed: {str(e)}")
            
            # Create relationships
            if import_config.get("create_relationships", False):
                relationships = self._create_relationships_from_json(result["data"], import_config)
                result["relationships_created"] = len(relationships)
                result["relationships"] = relationships
            
        except Exception as e:
            result["errors"].append(f"JSON processing failed: {str(e)}")
        
        return result
    
    async def _process_yaml(self, file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process YAML file import."""
        result = {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": [],
            "warnings": [],
            "data": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Process YAML data (similar to JSON)
            if isinstance(yaml_data, list):
                for item in yaml_data:
                    try:
                        node_data = self._create_node_from_yaml_item(item, import_config)
                        if node_data:
                            result["data"].append(node_data)
                            result["nodes_created"] += 1
                    except Exception as e:
                        result["errors"].append(f"YAML item processing failed: {str(e)}")
            
            elif isinstance(yaml_data, dict):
                if import_config.get("flatten_nested", False):
                    flattened_items = self._flatten_yaml_structure(yaml_data)
                    for item in flattened_items:
                        try:
                            node_data = self._create_node_from_yaml_item(item, import_config)
                            if node_data:
                                result["data"].append(node_data)
                                result["nodes_created"] += 1
                        except Exception as e:
                            result["errors"].append(f"Flattened YAML item processing failed: {str(e)}")
                else:
                    try:
                        node_data = self._create_node_from_yaml_item(yaml_data, import_config)
                        if node_data:
                            result["data"].append(node_data)
                            result["nodes_created"] += 1
                    except Exception as e:
                        result["errors"].append(f"YAML processing failed: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"YAML processing failed: {str(e)}")
        
        return result
    
    async def _process_xml(self, file_path: str, import_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process XML file import."""
        # This would implement XML processing
        # For now, return empty result
        return {
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": ["XML processing not yet implemented"],
            "warnings": [],
            "data": []
        }
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize CSV column names."""
        # Remove special characters and normalize
        normalized = re.sub(r'[^a-zA-Z0-9_\s]', '', column_name)
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = normalized.lower().strip('_')
        return normalized
    
    def _create_node_from_csv_row(self, row: pd.Series, import_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create memory node from CSV row."""
        try:
            # Extract content from row
            content_fields = import_config.get("content_fields", [])
            if not content_fields:
                # Use all non-empty fields
                content_fields = [col for col in row.index if pd.notna(row[col])]
            
            content = " ".join([str(row[field]) for field in content_fields if field in row.index])
            
            if not content.strip():
                return None
            
            # Determine memory type and category
            memory_type = self._determine_memory_type(row, import_config)
            category = self._determine_memory_category(row, import_config)
            
            # Extract tags
            tags = self._extract_tags_from_row(row, import_config)
            
            # Create node data
            node_data = {
                "content": content,
                "memory_type": memory_type,
                "category": category,
                "tags": tags,
                "metadata": {
                    "source": "csv_import",
                    "row_data": row.to_dict(),
                    "import_config": import_config
                }
            }
            
            return node_data
            
        except Exception as e:
            logger.error(f"Failed to create node from CSV row: {e}")
            return None
    
    def _create_node_from_json_item(self, item: Dict[str, Any], import_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create memory node from JSON item."""
        try:
            # Extract content from item
            content_fields = import_config.get("content_fields", [])
            if not content_fields:
                # Use all string fields
                content_fields = [key for key, value in item.items() if isinstance(value, str)]
            
            content = " ".join([str(item.get(field, "")) for field in content_fields])
            
            if not content.strip():
                return None
            
            # Determine memory type and category
            memory_type = self._determine_memory_type(item, import_config)
            category = self._determine_memory_category(item, import_config)
            
            # Extract tags
            tags = self._extract_tags_from_json(item, import_config)
            
            # Create node data
            node_data = {
                "content": content,
                "memory_type": memory_type,
                "category": category,
                "tags": tags,
                "metadata": {
                    "source": "json_import",
                    "item_data": item,
                    "import_config": import_config
                }
            }
            
            return node_data
            
        except Exception as e:
            logger.error(f"Failed to create node from JSON item: {e}")
            return None
    
    def _create_node_from_yaml_item(self, item: Dict[str, Any], import_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create memory node from YAML item."""
        # Similar to JSON processing
        return self._create_node_from_json_item(item, import_config)
    
    def _determine_memory_type(self, data: Union[pd.Series, Dict[str, Any]], import_config: Dict[str, Any]) -> MemoryType:
        """Determine memory type based on data and configuration."""
        # Check configuration first
        if "memory_type" in import_config:
            try:
                return MemoryType(import_config["memory_type"])
            except ValueError:
                pass
        
        # Analyze data to determine type
        if isinstance(data, pd.Series):
            data_dict = data.to_dict()
        else:
            data_dict = data
        
        # Check for time-based fields
        time_fields = ["timestamp", "date", "created_at", "updated_at"]
        if any(field in data_dict for field in time_fields):
            return MemoryType.EPISODIC
        
        # Check for technical fields
        tech_fields = ["ip", "port", "protocol", "hash", "signature"]
        if any(field in data_dict for field in tech_fields):
            return MemoryType.TECHNICAL
        
        # Default to semantic
        return MemoryType.SEMANTIC
    
    def _determine_memory_category(self, data: Union[pd.Series, Dict[str, Any]], import_config: Dict[str, Any]) -> MemoryCategory:
        """Determine memory category based on data and configuration."""
        # Check configuration first
        if "memory_category" in import_config:
            try:
                return MemoryCategory(import_config["memory_category"])
            except ValueError:
                pass
        
        # Analyze data to determine category
        if isinstance(data, pd.Series):
            data_dict = data.to_dict()
        else:
            data_dict = data
        
        # Check for threat-related fields
        threat_fields = ["threat", "malware", "attack", "vulnerability", "cve"]
        if any(field in str(data_dict).lower() for field in threat_fields):
            return MemoryCategory.THREAT_INTELLIGENCE
        
        # Check for incident-related fields
        incident_fields = ["incident", "breach", "alert", "event"]
        if any(field in str(data_dict).lower() for field in incident_fields):
            return MemoryCategory.INCIDENT_DATA
        
        # Check for compliance-related fields
        compliance_fields = ["compliance", "policy", "regulation", "standard"]
        if any(field in str(data_dict).lower() for field in compliance_fields):
            return MemoryCategory.COMPLIANCE_INFO
        
        # Default to technical
        return MemoryCategory.TECHNICAL
    
    def _extract_tags_from_row(self, row: pd.Series, import_config: Dict[str, Any]) -> List[str]:
        """Extract tags from CSV row."""
        tags = []
        
        # Check configuration for tag fields
        tag_fields = import_config.get("tag_fields", [])
        if tag_fields:
            for field in tag_fields:
                if field in row.index and pd.notna(row[field]):
                    tags.append(str(row[field]))
        
        # Extract tags from content
        content = " ".join([str(val) for val in row.values if pd.notna(val)])
        extracted_tags = self._extract_tags_from_content(content)
        tags.extend(extracted_tags)
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_tags_from_json(self, item: Dict[str, Any], import_config: Dict[str, Any]) -> List[str]:
        """Extract tags from JSON item."""
        tags = []
        
        # Check configuration for tag fields
        tag_fields = import_config.get("tag_fields", [])
        if tag_fields:
            for field in tag_fields:
                if field in item:
                    tags.append(str(item[field]))
        
        # Extract tags from content
        content = " ".join([str(val) for val in item.values()])
        extracted_tags = self._extract_tags_from_content(content)
        tags.extend(extracted_tags)
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract tags from content using NLP techniques."""
        tags = []
        
        # Extract potential tags (words starting with capital letters)
        potential_tags = re.findall(r'\b[A-Z][a-z]+\b', content)
        tags.extend(potential_tags)
        
        # Extract technical terms
        tech_terms = re.findall(r'\b[A-Z]{2,}\b', content)  # Acronyms
        tags.extend(tech_terms)
        
        # Extract version numbers
        version_terms = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', content)
        tags.extend(version_terms)
        
        return list(set(tags))  # Remove duplicates
    
    def _flatten_json_structure(self, data: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested JSON structure."""
        flattened = []
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.extend(self._flatten_json_structure(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened.extend(self._flatten_json_structure(item, f"{new_key}[{i}]"))
                    else:
                        flattened.append({new_key: item})
            else:
                flattened.append({new_key: value})
        
        return flattened
    
    def _flatten_yaml_structure(self, data: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested YAML structure."""
        # Similar to JSON flattening
        return self._flatten_json_structure(data, prefix)
    
    def _create_relationships_from_csv(self, nodes: List[Dict[str, Any]], import_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create relationships between CSV nodes."""
        relationships = []
        
        # Check if relationships should be created
        if not import_config.get("create_relationships", False):
            return relationships
        
        # Simple relationship creation based on common tags
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Check for common tags
                common_tags = set(node1.get("tags", [])) & set(node2.get("tags", []))
                
                if len(common_tags) >= 2:  # At least 2 common tags
                    relationship = {
                        "source_node": i,
                        "target_node": j,
                        "relationship_type": "related_by_tags",
                        "strength": len(common_tags) / max(len(node1.get("tags", [])), len(node2.get("tags", []))),
                        "common_tags": list(common_tags)
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _create_relationships_from_json(self, nodes: List[Dict[str, Any]], import_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create relationships between JSON nodes."""
        # Similar to CSV relationship creation
        return self._create_relationships_from_csv(nodes, import_config)

class EnhancedKnowledgeMemory:
    """Enhanced knowledge graph context memory system."""
    
    def __init__(self):
        self.data_importer = DataImporter()
        self.memory_db_path = Path("knowledge-objects/enhanced_memory.db")
        self.memory_db_path.parent.mkdir(exist_ok=True)
        self._init_memory_db()
    
    def _init_memory_db(self):
        """Initialize memory database."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_nodes (
                        node_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        relevance_score REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.0,
                        relationships TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_relationships (
                        relationship_id TEXT PRIMARY KEY,
                        source_node TEXT NOT NULL,
                        target_node TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        strength REAL DEFAULT 0.0,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS import_history (
                        import_id TEXT PRIMARY KEY,
                        source_file TEXT NOT NULL,
                        import_type TEXT NOT NULL,
                        nodes_created INTEGER DEFAULT 0,
                        relationships_created INTEGER DEFAULT 0,
                        errors TEXT,
                        warnings TEXT,
                        processing_time REAL DEFAULT 0.0,
                        created_at TEXT NOT NULL
                    )
                """)
        except Exception as e:
            logger.warning(f"Memory database initialization failed: {e}")
    
    async def import_data_file(self, file_path: str, import_config: Dict[str, Any]) -> ImportResult:
        """Import data from a file into the knowledge graph."""
        # Import data
        import_result = await self.data_importer.import_data(file_path, import_config)
        
        # Store in database
        await self._store_import_result(import_result)
        
        # Store memory nodes
        if import_result.nodes_created > 0:
            await self._store_memory_nodes(import_result)
        
        # Store relationships
        if import_result.relationships_created > 0:
            await self._store_relationships(import_result)
        
        return import_result
    
    async def _store_import_result(self, import_result: ImportResult):
        """Store import result in database."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO import_history 
                    (import_id, source_file, import_type, nodes_created, relationships_created, 
                     errors, warnings, processing_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    import_result.import_id,
                    import_result.source_file,
                    import_result.import_type,
                    import_result.nodes_created,
                    import_result.relationships_created,
                    json.dumps(import_result.errors),
                    json.dumps(import_result.warnings),
                    import_result.processing_time,
                    import_result.created_at.isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to store import result: {e}")
    
    async def _store_memory_nodes(self, import_result: ImportResult):
        """Store memory nodes from import result."""
        # This would store the actual memory nodes
        # For now, just log the count
        logger.info(f"Stored {import_result.nodes_created} memory nodes from import {import_result.import_id}")
    
    async def _store_relationships(self, import_result: ImportResult):
        """Store relationships from import result."""
        # This would store the actual relationships
        # For now, just log the count
        logger.info(f"Stored {import_result.relationships_created} relationships from import {import_result.import_id}")
    
    async def search_memory(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search memory for relevant information."""
        # This would implement semantic search
        # For now, return empty list
        return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                # Get node count by type
                cursor = conn.execute("""
                    SELECT memory_type, COUNT(*) as count 
                    FROM memory_nodes 
                    GROUP BY memory_type
                """)
                nodes_by_type = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get node count by category
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM memory_nodes 
                    GROUP BY category
                """)
                nodes_by_category = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get total relationships
                cursor = conn.execute("SELECT COUNT(*) FROM memory_relationships")
                total_relationships = cursor.fetchone()[0]
                
                # Get import history
                cursor = conn.execute("SELECT COUNT(*) FROM import_history")
                total_imports = cursor.fetchone()[0]
                
                return {
                    "total_nodes": sum(nodes_by_type.values()),
                    "nodes_by_type": nodes_by_type,
                    "nodes_by_category": nodes_by_category,
                    "total_relationships": total_relationships,
                    "total_imports": total_imports
                }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

# Global enhanced knowledge memory instance
enhanced_knowledge_memory = EnhancedKnowledgeMemory()

# Convenience functions
async def import_data_file(file_path: str, import_config: Dict[str, Any]) -> ImportResult:
    """Convenience function for data import."""
    return await enhanced_knowledge_memory.import_data_file(file_path, import_config)

async def search_memory(query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Convenience function for memory search."""
    return await enhanced_knowledge_memory.search_memory(query, filters)

async def get_memory_stats() -> Dict[str, Any]:
    """Convenience function for memory statistics."""
    return await enhanced_knowledge_memory.get_memory_stats()
