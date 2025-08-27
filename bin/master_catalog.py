#!/usr/bin/env python3
"""
Master Catalog for Distributed Knowledge Graph

This system manages multiple domain-specific SQLite databases and provides
intelligent routing, discovery, and optimization for large-scale knowledge graphs.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class KnowledgeDomain:
    """Represents a knowledge domain with its database and metadata."""
    domain_id: str
    name: str
    description: str
    db_path: str
    node_count: int = 0
    last_updated: str = ""
    tags: List[str] = None
    access_patterns: Dict[str, int] = None
    size_mb: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.access_patterns is None:
            self.access_patterns = {}

class MasterCatalog:
    """Master catalog for distributed knowledge graph management."""
    
    def __init__(self, base_path: str = "knowledge-objects"):
        self.base_path = Path(base_path).resolve()
        self.master_db_path = self.base_path / "master_catalog.db"
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.domain_connections: Dict[str, sqlite3.Connection] = {}
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MasterCatalog")
        
        # Initialize master database
        self._init_master_database()
        
        # Discover and register existing domains
        self._discover_domains()
    
    def _init_master_database(self):
        """Initialize the master catalog database."""
        try:
            self.master_conn = sqlite3.connect(self.master_db_path)
            self.master_cursor = self.master_conn.cursor()
            
            # Create master catalog tables
            self.master_cursor.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_domains (
                    domain_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    db_path TEXT NOT NULL,
                    node_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,
                    access_patterns TEXT,
                    size_mb REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS domain_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_domain TEXT NOT NULL,
                    target_domain TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_domain) REFERENCES knowledge_domains (domain_id),
                    FOREIGN KEY (target_domain) REFERENCES knowledge_domains (domain_id)
                );
                
                CREATE TABLE IF NOT EXISTS global_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    access_frequency INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,
                    metadata TEXT,
                    FOREIGN KEY (domain_id) REFERENCES knowledge_domains (domain_id)
                );
                
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    node_count INTEGER DEFAULT 0,
                    response_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT,
                    FOREIGN KEY (domain_id) REFERENCES knowledge_domains (domain_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_global_index_node_id ON global_index (node_id);
                CREATE INDEX IF NOT EXISTS idx_global_index_domain ON global_index (domain_id);
                CREATE INDEX IF NOT EXISTS idx_global_index_type ON global_index (node_type);
                CREATE INDEX IF NOT EXISTS idx_global_index_hash ON global_index (content_hash);
                CREATE INDEX IF NOT EXISTS idx_access_log_domain ON access_log (domain_id);
                CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON access_log (timestamp);
            """)
            
            self.master_conn.commit()
            self.logger.info("✅ Master catalog database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize master database: {e}")
            raise
    
    def _discover_domains(self):
        """Discover and register existing knowledge domains."""
        domain_configs = {
            "grc": {
                "name": "Governance, Risk & Compliance",
                "description": "Policies, controls, risk assessments, compliance frameworks",
                "tags": ["policies", "controls", "compliance", "risk", "audit"]
            },
            "organization": {
                "name": "Organizational Information",
                "description": "Company structure, terms, processes, business units",
                "tags": ["company", "structure", "processes", "business", "terms"]
            },
            "networks": {
                "name": "Network Infrastructure",
                "description": "Network topology, segments, routing, connectivity",
                "tags": ["network", "infrastructure", "topology", "segments", "routing"]
            },
            "hosts": {
                "name": "Host Systems",
                "description": "Servers, workstations, devices, endpoints",
                "tags": ["hosts", "servers", "workstations", "devices", "endpoints"]
            },
            "threat_intelligence": {
                "name": "Threat Intelligence",
                "description": "Threat actors, indicators, campaigns, TTPs",
                "tags": ["threats", "intelligence", "actors", "indicators", "ttps"]
            },
            "users": {
                "name": "User Management",
                "description": "User accounts, roles, permissions, access",
                "tags": ["users", "accounts", "roles", "permissions", "access"]
            },
            "applications": {
                "name": "Applications & Systems",
                "description": "Software applications, systems, integrations",
                "tags": ["applications", "software", "systems", "integrations"]
            },
            "incidents": {
                "name": "Security Incidents",
                "description": "Incident reports, investigations, lessons learned",
                "tags": ["incidents", "investigations", "reports", "lessons"]
            },
            "compliance": {
                "name": "Compliance Frameworks",
                "description": "NIST, ISO, SOC2, PCI-DSS, SOX compliance",
                "tags": ["compliance", "frameworks", "standards", "certifications"]
            }
        }
        
        for domain_id, config in domain_configs.items():
            domain_path = self.base_path / domain_id
            if domain_path.exists():
                self._register_domain(domain_id, config)
    
    def _register_domain(self, domain_id: str, config: Dict[str, Any]):
        """Register a knowledge domain."""
        try:
            db_path = str(self.base_path / domain_id / f"{domain_id}.db")
            
            # Create domain database if it doesn't exist
            if not os.path.exists(db_path):
                self._create_domain_database(domain_id, db_path)
            
            # Get domain statistics
            stats = self._get_domain_statistics(domain_id, db_path)
            
            # Create domain object
            domain = KnowledgeDomain(
                domain_id=domain_id,
                name=config["name"],
                description=config["description"],
                db_path=db_path,
                node_count=stats["node_count"],
                last_updated=stats["last_updated"],
                tags=config["tags"],
                size_mb=stats["size_mb"]
            )
            
            self.domains[domain_id] = domain
            
            # Update master catalog
            self._update_domain_in_catalog(domain)
            
            self.logger.info(f"✅ Registered domain: {domain_id} ({stats['node_count']} nodes)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register domain {domain_id}: {e}")
    
    def _create_domain_database(self, domain_id: str, db_path: str):
        """Create a new domain-specific database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create domain-specific tables
            cursor.executescript(f"""
                CREATE TABLE IF NOT EXISTS {domain_id}_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT UNIQUE NOT NULL,
                    node_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance_score REAL DEFAULT 0.5,
                    access_frequency INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_category TEXT DEFAULT 'long_term',
                    expires_at TIMESTAMP,
                    content_hash TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS {domain_id}_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_node_id) REFERENCES {domain_id}_nodes (node_id),
                    FOREIGN KEY (target_node_id) REFERENCES {domain_id}_nodes (node_id)
                );
                
                CREATE TABLE IF NOT EXISTS {domain_id}_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    session_id TEXT,
                    workflow_id TEXT,
                    context TEXT,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES {domain_id}_nodes (node_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_{domain_id}_nodes_type ON {domain_id}_nodes (node_type);
                CREATE INDEX IF NOT EXISTS idx_{domain_id}_nodes_hash ON {domain_id}_nodes (content_hash);
                CREATE INDEX IF NOT EXISTS idx_{domain_id}_nodes_ttl ON {domain_id}_nodes (ttl_category);
                CREATE INDEX IF NOT EXISTS idx_{domain_id}_relationships_source ON {domain_id}_relationships (source_node_id);
                CREATE INDEX IF NOT EXISTS idx_{domain_id}_relationships_target ON {domain_id}_relationships (target_node_id);
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Created domain database: {db_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create domain database {db_path}: {e}")
            raise
    
    def _get_domain_statistics(self, domain_id: str, db_path: str) -> Dict[str, Any]:
        """Get statistics for a domain database."""
        try:
            if not os.path.exists(db_path):
                return {"node_count": 0, "last_updated": "", "size_mb": 0.0}
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get node count
            cursor.execute(f"SELECT COUNT(*) FROM {domain_id}_nodes")
            node_count = cursor.fetchone()[0]
            
            # Get last updated
            cursor.execute(f"SELECT MAX(updated_at) FROM {domain_id}_nodes")
            last_updated = cursor.fetchone()[0] or ""
            
            # Get database size
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            
            conn.close()
            
            return {
                "node_count": node_count,
                "last_updated": last_updated,
                "size_mb": round(size_mb, 2)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics for {domain_id}: {e}")
            return {"node_count": 0, "last_updated": "", "size_mb": 0.0}
    
    def _update_domain_in_catalog(self, domain: KnowledgeDomain):
        """Update domain information in master catalog."""
        try:
            self.master_cursor.execute("""
                INSERT OR REPLACE INTO knowledge_domains 
                (domain_id, name, description, db_path, node_count, last_updated, tags, size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                domain.domain_id, domain.name, domain.description, domain.db_path,
                domain.node_count, domain.last_updated, json.dumps(domain.tags), domain.size_mb
            ))
            
            self.master_conn.commit()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update domain in catalog: {e}")
    
    def get_domain_connection(self, domain_id: str) -> Optional[sqlite3.Connection]:
        """Get or create a connection to a domain database."""
        if domain_id not in self.domain_connections:
            if domain_id in self.domains:
                try:
                    conn = sqlite3.connect(self.domains[domain_id].db_path)
                    conn.row_factory = sqlite3.Row  # Enable dict-like access
                    self.domain_connections[domain_id] = conn
                except Exception as e:
                    self.logger.error(f"❌ Failed to connect to domain {domain_id}: {e}")
                    return None
            else:
                return None
        
        return self.domain_connections[domain_id]
    
    def add_node(self, domain_id: str, node_id: str, node_type: str, content: str,
                 metadata: Dict[str, Any] = None, importance_score: float = 0.5,
                 ttl_category: str = "long_term") -> bool:
        """Add a node to a specific domain."""
        try:
            conn = self.get_domain_connection(domain_id)
            if not conn:
                return False
            
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check for duplicates
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT node_id FROM {domain_id}_nodes 
                WHERE content_hash = ? OR node_id = ?
            """, (content_hash, node_id))
            
            if cursor.fetchone():
                self.logger.warning(f"⚠️ Node already exists in {domain_id}: {node_id}")
                return False
            
            # Add node
            cursor.execute(f"""
                INSERT INTO {domain_id}_nodes 
                (node_id, node_type, content, metadata, importance_score, ttl_category, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id, node_type, content, json.dumps(metadata or {}),
                importance_score, ttl_category, content_hash
            ))
            
            # Update global index
            self._update_global_index(domain_id, node_id, node_type, content_hash, importance_score, metadata)
            
            # Update domain statistics
            self._update_domain_statistics(domain_id)
            
            conn.commit()
            
            self.logger.info(f"✅ Added node {node_id} to domain {domain_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add node to {domain_id}: {e}")
            return False
    
    def _update_global_index(self, domain_id: str, node_id: str, node_type: str,
                            content_hash: str, importance_score: float, metadata: Dict[str, Any]):
        """Update the global index with new node information."""
        try:
            # Extract tags from metadata
            tags = metadata.get("tags", []) if metadata else []
            
            self.master_cursor.execute("""
                INSERT OR REPLACE INTO global_index 
                (node_id, domain_id, node_type, content_hash, importance_score, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id, domain_id, node_type, content_hash,
                importance_score, json.dumps(tags), json.dumps(metadata or {})
            ))
            
            self.master_conn.commit()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update global index: {e}")
    
    def _update_domain_statistics(self, domain_id: str):
        """Update domain statistics after changes."""
        try:
            stats = self._get_domain_statistics(domain_id, self.domains[domain_id].db_path)
            self.domains[domain_id].node_count = stats["node_count"]
            self.domains[domain_id].last_updated = stats["last_updated"]
            self.domains[domain_id].size_mb = stats["size_mb"]
            
            self._update_domain_in_catalog(self.domains[domain_id])
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update domain statistics: {e}")
    
    def search_across_domains(self, query: str, node_types: List[str] = None,
                             domains: List[str] = None, min_importance: float = 0.0,
                             max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for nodes across multiple domains."""
        try:
            # Start with global index search
            search_query = """
                SELECT node_id, domain_id, node_type, importance_score, tags, metadata
                FROM global_index
                WHERE 1=1
            """
            params = []
            
            if node_types:
                placeholders = ",".join(["?" for _ in node_types])
                search_query += f" AND node_type IN ({placeholders})"
                params.extend(node_types)
            
            if domains:
                placeholders = ",".join(["?" for _ in domains])
                search_query += f" AND domain_id IN ({placeholders})"
                params.extend(domains)
            
            search_query += " AND importance_score >= ?"
            params.append(min_importance)
            
            search_query += " ORDER BY importance_score DESC, access_frequency DESC"
            search_query += " LIMIT ?"
            params.append(max_results)
            
            self.master_cursor.execute(search_query, params)
            global_results = self.master_cursor.fetchall()
            
            # Fetch detailed content from domain databases
            detailed_results = []
            for row in global_results:
                node_id, domain_id, node_type, importance_score, tags, metadata = row
                
                # Get detailed content from domain
                domain_content = self._get_node_from_domain(domain_id, node_id)
                if domain_content:
                    detailed_results.append({
                        "node_id": node_id,
                        "domain_id": domain_id,
                        "node_type": node_type,
                        "content": domain_content["content"],
                        "importance_score": importance_score,
                        "tags": json.loads(tags) if tags else [],
                        "metadata": json.loads(metadata) if metadata else {},
                        "source": domain_id
                    })
            
            return detailed_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"❌ Cross-domain search failed: {e}")
            return []
    
    def _get_node_from_domain(self, domain_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed node content from a specific domain."""
        try:
            conn = self.get_domain_connection(domain_id)
            if not conn:
                return None
            
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT content, metadata, importance_score, access_frequency
                FROM {domain_id}_nodes WHERE node_id = ?
            """, (node_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "content": row[0],
                    "metadata": json.loads(row[1]) if row[1] else {},
                    "importance_score": row[2],
                    "access_frequency": row[3]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get node from domain {domain_id}: {e}")
            return None
    
    def get_workflow_context(self, workflow_context: str, max_nodes: int = 50) -> Dict[str, Any]:
        """Get relevant context for a workflow across all domains."""
        try:
            # Search across all domains
            relevant_nodes = self.search_across_domains(
                query=workflow_context,
                max_results=max_nodes * 2
            )
            
            # Group by domain and select top relevant
            context = {
                "domains": {},
                "summary": {
                    "total_nodes": 0,
                    "domains_used": set(),
                    "node_types": {},
                    "importance_distribution": {"low": 0, "medium": 0, "high": 0}
                }
            }
            
            for node in relevant_nodes[:max_nodes]:
                domain_id = node["domain_id"]
                
                if domain_id not in context["domains"]:
                    context["domains"][domain_id] = []
                
                context["domains"][domain_id].append({
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "content": node["content"],
                    "importance_score": node["importance_score"],
                    "tags": node["tags"],
                    "metadata": node["metadata"]
                })
                
                # Update summary
                context["summary"]["total_nodes"] += 1
                context["summary"]["domains_used"].add(domain_id)
                context["summary"]["node_types"][node["node_type"]] = \
                    context["summary"]["node_types"].get(node["node_type"], 0) + 1
                
                # Categorize importance
                if node["importance_score"] < 0.33:
                    context["summary"]["importance_distribution"]["low"] += 1
                elif node["importance_score"] < 0.67:
                    context["summary"]["importance_distribution"]["medium"] += 1
                else:
                    context["summary"]["importance_distribution"]["high"] += 1
            
            # Convert set to list for JSON serialization
            context["summary"]["domains_used"] = list(context["summary"]["domains_used"])
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get workflow context: {e}")
            return {"error": str(e)}
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all domains."""
        try:
            stats = {
                "total_domains": len(self.domains),
                "total_nodes": 0,
                "total_size_mb": 0.0,
                "domains": {},
                "performance_metrics": {}
            }
            
            for domain_id, domain in self.domains.items():
                stats["domains"][domain_id] = asdict(domain)
                stats["total_nodes"] += domain.node_count
                stats["total_size_mb"] += domain.size_mb
            
            # Get performance metrics from access log
            self.master_cursor.execute("""
                SELECT domain_id, AVG(response_time_ms) as avg_response,
                       COUNT(*) as total_operations
                FROM access_log 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY domain_id
            """)
            
            performance_data = self.master_cursor.fetchall()
            for row in performance_data:
                domain_id, avg_response, total_ops = row
                stats["performance_metrics"][domain_id] = {
                    "avg_response_time_ms": round(avg_response or 0, 2),
                    "operations_24h": total_ops
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get domain statistics: {e}")
            return {"error": str(e)}
    
    def optimize_domain(self, domain_id: str) -> Dict[str, Any]:
        """Optimize a specific domain database."""
        try:
            if domain_id not in self.domains:
                return {"error": f"Domain {domain_id} not found"}
            
            conn = self.get_domain_connection(domain_id)
            if not conn:
                return {"error": f"Failed to connect to domain {domain_id}"}
            
            cursor = conn.cursor()
            
            # Analyze table statistics
            cursor.execute(f"ANALYZE {domain_id}_nodes")
            cursor.execute(f"ANALYZE {domain_id}_relationships")
            
            # Get optimization recommendations
            cursor.execute(f"""
                SELECT name, sql FROM sqlite_master 
                WHERE type='table' AND name LIKE '{domain_id}_%'
            """)
            
            tables = cursor.fetchall()
            optimization_results = {
                "domain_id": domain_id,
                "tables_analyzed": len(tables),
                "recommendations": []
            }
            
            # Check for missing indexes
            for table_name, table_sql in tables:
                if "CREATE INDEX" not in table_sql.upper():
                    optimization_results["recommendations"].append(
                        f"Consider adding indexes to {table_name}"
                    )
            
            # Vacuum database
            cursor.execute("VACUUM")
            
            # Update statistics
            self._update_domain_statistics(domain_id)
            
            optimization_results["status"] = "completed"
            optimization_results["message"] = "Database optimized and analyzed"
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize domain {domain_id}: {e}")
            return {"error": str(e)}
    
    def close_all_connections(self):
        """Close all domain database connections."""
        for domain_id, conn in self.domain_connections.items():
            try:
                conn.close()
                self.logger.info(f"✅ Closed connection to {domain_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed to close connection to {domain_id}: {e}")
        
        self.domain_connections.clear()
        
        if hasattr(self, 'master_conn'):
            self.master_conn.close()
            self.logger.info("✅ Closed master catalog connection")

if __name__ == "__main__":
    # Test the master catalog
    catalog = MasterCatalog(".")
    
    print("🔍 Testing Master Catalog...")
    print(f"✅ Discovered {len(catalog.domains)} domains")
    
    for domain_id, domain in catalog.domains.items():
        print(f"   - {domain_id}: {domain.name} ({domain.node_count} nodes, {domain.size_mb} MB)")
    
    # Test adding a node
    success = catalog.add_node(
        "grc",
        "test_policy_001",
        "policy",
        "Test policy for demonstration purposes",
        metadata={"category": "test", "priority": "medium"},
        importance_score=0.7
    )
    
    if success:
        print("✅ Test node added successfully")
    else:
        print("❌ Failed to add test node")
    
    # Get statistics
    stats = catalog.get_domain_statistics()
    print(f"✅ Total nodes across all domains: {stats['total_nodes']}")
    
    catalog.close_all_connections()
