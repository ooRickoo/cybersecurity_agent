#!/usr/bin/env python3
"""
Context Memory Manager
Distributed, multi-dimensional memory management for cybersecurity analysis.
Handles short-term, medium-term, and long-term memory with TTL features.
"""

import os
import sys
import json
import hashlib
import sqlite3
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class MemoryDomain(Enum):
    """Memory domains for different types of cybersecurity data."""
    THREAT_INTELLIGENCE = "threat_intelligence"
    GRC_POLICIES = "grc_policies"
    HOST_INVENTORY = "host_inventory"
    APPLICATION_INVENTORY = "application_inventory"
    USER_INVENTORY = "user_inventory"
    NETWORK_INVENTORY = "network_inventory"
    IOC_COLLECTION = "ioc_collection"
    THREAT_ACTORS = "threat_actors"
    INVESTIGATION_ENTITIES = "investigation_entities"
    SPLUNK_SCHEMAS = "splunk_schemas"
    MITRE_ATTACK = "mitre_attack"
    MITRE_D3FEND = "mitre_d3fend"
    NIST_FRAMEWORKS = "nist_frameworks"
    CUSTOM_FRAMEWORKS = "custom_frameworks"
    GRAPH_PATTERNS = "graph_patterns"
    ENTITY_RELATIONSHIPS = "entity_relationships"

class MemoryTier(Enum):
    """Memory tiers with different TTL characteristics."""
    SHORT_TERM = "short_term"      # TTL: 1-7 days
    MEDIUM_TERM = "medium_term"    # TTL: 7-30 days
    LONG_TERM = "long_term"        # TTL: 30+ days

class DataType(Enum):
    """Data types for efficient storage and retrieval."""
    STRUCTURED = "structured"      # CSV, JSON, Database
    SEMI_STRUCTURED = "semi_structured"  # XML, YAML, Logs
    UNSTRUCTURED = "unstructured"  # Text, Documents
    BINARY = "binary"              # Images, Files, PCAPs
    GRAPH = "graph"                # Network graphs, relationships

@dataclass
class MemoryMetadata:
    """Metadata for memory entries."""
    id: str
    domain: MemoryDomain
    tier: MemoryTier
    data_type: DataType
    source: str
    import_timestamp: datetime
    last_accessed: datetime
    access_count: int
    ttl_days: int
    expires_at: datetime
    size_bytes: int
    compression_ratio: float
    tags: List[str]
    relationships: List[str]
    priority: int
    description: str

@dataclass
class MemoryEntry:
    """A memory entry with data and metadata."""
    metadata: MemoryMetadata
    data: Any
    compressed_data: Optional[bytes] = None

class ContextMemoryManager:
    """Distributed, multi-dimensional context memory management system."""
    
    def __init__(self, base_path: str = "knowledge-objects"):
        self.base_path = Path(base_path)
        self.memory_path = self.base_path / "context_memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory databases
        self._init_memory_databases()
        
        # Memory caches for fast access
        self.short_term_cache = {}
        self.medium_term_cache = {}
        self.long_term_cache = {}
        
        # Relationship graph for entity connections
        self.relationship_graph = defaultdict(set)
        
        # Index for fast searching
        self.search_index = {}
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'total_size_bytes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_savings': 0
        }
        
        # Background maintenance thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        print("🧠 Context Memory Manager initialized")
        print(f"   Memory path: {self.memory_path}")
        print(f"   Domains: {len(MemoryDomain)}")
        print(f"   Tiers: {len(MemoryTier)}")
    
    def _init_memory_databases(self):
        """Initialize SQLite databases for memory management."""
        # Main memory database
        self.memory_db = self.memory_path / "context_memory.db"
        self._create_memory_tables()
        
        # Relationship database
        self.relationship_db = self.memory_path / "relationships.db"
        self._create_relationship_tables()
        
        # Search index database
        self.search_db = self.memory_path / "search_index.db"
        self._create_search_tables()
    
    def _create_memory_tables(self):
        """Create memory management tables."""
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    import_timestamp TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_days INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compression_ratio REAL DEFAULT 1.0,
                    tags TEXT,
                    relationships TEXT,
                    priority INTEGER DEFAULT 5,
                    description TEXT,
                    metadata_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain_tier ON memory_entries(domain, tier)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON memory_entries(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON memory_entries(tags)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_priority ON memory_entries(priority)
            """)
    
    def _create_relationship_tables(self):
        """Create relationship tracking tables."""
        with sqlite3.connect(self.relationship_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_entity ON entity_relationships(source_entity)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_target_entity ON entity_relationships(target_entity)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_type ON entity_relationships(relationship_type)
            """)
    
    def _create_search_tables(self):
        """Create search indexing tables."""
        with sqlite3.connect(self.search_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    vector_data BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_text ON search_index(search_text)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_id ON search_index(memory_id)
            """)
    
    def import_data(self, domain: MemoryDomain, data: Any, source: str, 
                    tier: MemoryTier = MemoryTier.MEDIUM_TERM, ttl_days: int = 30,
                    tags: List[str] = None, description: str = "", 
                    priority: int = 5) -> str:
        """Import data into context memory."""
        try:
            # Generate unique ID
            memory_id = self._generate_memory_id(domain, source, data)
            
            # Determine data type
            data_type = self._classify_data_type(data)
            
            # Calculate size and compress if beneficial
            original_size = self._calculate_data_size(data)
            compressed_data, compression_ratio = self._compress_data(data)
            
            # Create metadata
            now = datetime.now()
            expires_at = now + timedelta(days=ttl_days)
            
            metadata = MemoryMetadata(
                id=memory_id,
                domain=domain,
                tier=tier,
                data_type=data_type,
                source=source,
                import_timestamp=now,
                last_accessed=now,
                access_count=0,
                ttl_days=ttl_days,
                expires_at=expires_at,
                size_bytes=len(compressed_data) if compressed_data else original_size,
                compression_ratio=compression_ratio,
                tags=tags or [],
                relationships=[],
                priority=priority,
                description=description
            )
            
            # Create memory entry
            entry = MemoryEntry(
                metadata=metadata,
                data=data,
                compressed_data=compressed_data
            )
            
            # Store in database
            self._store_memory_entry(entry)
            
            # Update caches
            self._update_cache(entry)
            
            # Update search index
            self._update_search_index(entry)
            
            # Update statistics
            self.stats['total_entries'] += 1
            self.stats['total_size_bytes'] += entry.metadata.size_bytes
            self.stats['compression_savings'] += int(original_size * (1 - compression_ratio))
            
            print(f"✅ Imported {domain.value} data: {memory_id}")
            print(f"   Size: {original_size} → {entry.metadata.size_bytes} bytes")
            print(f"   Compression: {compression_ratio:.2f}")
            print(f"   TTL: {ttl_days} days")
            
            return memory_id
            
        except Exception as e:
            print(f"❌ Error importing data: {e}")
            raise
    
    def _generate_memory_id(self, domain: MemoryDomain, source: str, data: Any) -> str:
        """Generate unique memory ID."""
        content_hash = hashlib.sha256(str(data).encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{domain.value}_{source}_{timestamp}_{content_hash}"
    
    def _classify_data_type(self, data: Any) -> DataType:
        """Classify data type for efficient storage."""
        if isinstance(data, (pd.DataFrame, dict, list)):
            return DataType.STRUCTURED
        elif isinstance(data, str) and any(marker in data for marker in ['<', '>', ':', '|']):
            return DataType.SEMI_STRUCTURED
        elif isinstance(data, str):
            return DataType.UNSTRUCTURED
        elif isinstance(data, bytes):
            return DataType.BINARY
        elif hasattr(data, 'nodes') and hasattr(data, 'edges'):
            return DataType.GRAPH
        else:
            return DataType.STRUCTURED
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate data size in bytes."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str).encode())
            elif isinstance(data, str):
                return len(data.encode())
            elif isinstance(data, bytes):
                return len(data)
            else:
                return len(str(data).encode())
        except:
            return len(str(data).encode())
    
    def _compress_data(self, data: Any) -> Tuple[Optional[bytes], float]:
        """Compress data if beneficial."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            
            # Compress
            compressed = gzip.compress(serialized)
            
            # Check if compression is beneficial
            compression_ratio = len(compressed) / len(serialized)
            
            if compression_ratio < 0.8:  # Only use if 20%+ savings
                return compressed, compression_ratio
            else:
                return None, 1.0
                
        except Exception:
            return None, 1.0
    
    def _store_memory_entry(self, entry: MemoryEntry):
        """Store memory entry in database."""
        with sqlite3.connect(self.memory_db) as conn:
            # Store metadata
            conn.execute("""
                INSERT OR REPLACE INTO memory_entries 
                (id, domain, tier, data_type, source, import_timestamp, last_accessed,
                 access_count, ttl_days, expires_at, size_bytes, compression_ratio,
                 tags, relationships, priority, description, metadata_hash, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.metadata.id,
                entry.metadata.domain.value,
                entry.metadata.tier.value,
                entry.metadata.data_type.value,
                entry.metadata.source,
                entry.metadata.import_timestamp.isoformat(),
                entry.metadata.last_accessed.isoformat(),
                entry.metadata.access_count,
                entry.metadata.ttl_days,
                entry.metadata.expires_at.isoformat(),
                entry.metadata.size_bytes,
                entry.metadata.compression_ratio,
                json.dumps(entry.metadata.tags),
                json.dumps(entry.metadata.relationships),
                entry.metadata.priority,
                entry.metadata.description,
                hashlib.sha256(str(asdict(entry.metadata)).encode()).hexdigest(),
                hashlib.sha256(str(entry.data).encode()).hexdigest()
            ))
            
            # Store compressed data
            data_file = self.memory_path / f"{entry.metadata.id}.data"
            if entry.compressed_data:
                data_file.write_bytes(entry.compressed_data)
            else:
                data_file.write_bytes(pickle.dumps(entry.data))
    
    def _update_cache(self, entry: MemoryEntry):
        """Update memory caches."""
        cache_key = entry.metadata.id
        
        if entry.metadata.tier == MemoryTier.SHORT_TERM:
            self.short_term_cache[cache_key] = entry
        elif entry.metadata.tier == MemoryTier.MEDIUM_TERM:
            self.medium_term_cache[cache_key] = entry
        elif entry.metadata.tier == MemoryTier.LONG_TERM:
            self.long_term_cache[cache_key] = entry
        
        # Limit cache sizes
        max_cache_size = 1000
        if len(self.short_term_cache) > max_cache_size:
            self._cleanup_cache(self.short_term_cache)
        if len(self.medium_term_cache) > max_cache_size:
            self._cleanup_cache(self.medium_term_cache)
        if len(self.long_term_cache) > max_cache_size:
            self._cleanup_cache(self.long_term_cache)
    
    def _cleanup_cache(self, cache: Dict):
        """Clean up cache by removing least recently used entries."""
        if len(cache) > 800:  # Only cleanup when 80% full
            # Sort by last accessed and remove oldest 20%
            sorted_entries = sorted(cache.items(), 
                                  key=lambda x: x[1].metadata.last_accessed)
            entries_to_remove = sorted_entries[:len(sorted_entries) // 5]
            
            for key, _ in entries_to_remove:
                del cache[key]
    
    def _update_search_index(self, entry: MemoryEntry):
        """Update search index for fast retrieval."""
        try:
            # Extract searchable text
            search_text = self._extract_searchable_text(entry.data)
            
            # Create vector representation (simplified)
            vector_data = self._create_vector_representation(search_text)
            
            with sqlite3.connect(self.search_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO search_index 
                    (memory_id, domain, tier, search_text, vector_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entry.metadata.id,
                    entry.metadata.domain.value,
                    entry.metadata.tier.value,
                    search_text,
                    vector_data
                ))
                
        except Exception as e:
            print(f"⚠️  Search index update failed: {e}")
    
    def _extract_searchable_text(self, data: Any) -> str:
        """Extract searchable text from data."""
        try:
            if isinstance(data, pd.DataFrame):
                return " ".join(str(col) for col in data.columns) + " " + \
                       " ".join(str(val) for row in data.head().values for val in row)
            elif isinstance(data, dict):
                return " ".join(str(k) + " " + str(v) for k, v in data.items())
            elif isinstance(data, list):
                return " ".join(str(item) for item in data[:100])  # Limit to first 100
            elif isinstance(data, str):
                return data
            else:
                return str(data)
        except:
            return str(data)
    
    def _create_vector_representation(self, text: str) -> bytes:
        """Create simple vector representation for search."""
        # Simplified vector creation - in production, use proper embeddings
        try:
            # Create a simple hash-based vector
            words = text.lower().split()
            word_freq = defaultdict(int)
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_freq[word] += 1
            
            # Convert to bytes (simplified)
            vector_str = json.dumps(dict(word_freq))
            return vector_str.encode()
        except:
            return b""
    
    def retrieve_context(self, query: str, domains: List[MemoryDomain] = None,
                        tiers: List[MemoryTier] = None, max_results: int = 10,
                        min_relevance: float = 0.5) -> List[MemoryEntry]:
        """Retrieve relevant context for a query."""
        try:
            # Search across specified domains and tiers
            if domains is None:
                domains = list(MemoryDomain)
            if tiers is None:
                tiers = list(MemoryTier)
            
            # Search in caches first
            cache_results = self._search_caches(query, domains, tiers, max_results)
            
            # If not enough results, search database
            if len(cache_results) < max_results:
                db_results = self._search_database(query, domains, tiers, 
                                                max_results - len(cache_results))
                cache_results.extend(db_results)
            
            # Sort by relevance and priority
            cache_results.sort(key=lambda x: (x.metadata.priority, x.metadata.access_count), 
                             reverse=True)
            
            # Update access statistics
            for entry in cache_results:
                self._update_access_stats(entry.metadata.id)
            
            return cache_results[:max_results]
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return []
    
    def _search_caches(self, query: str, domains: List[MemoryDomain],
                       tiers: List[MemoryTier], max_results: int) -> List[MemoryEntry]:
        """Search memory caches for relevant entries."""
        results = []
        query_lower = query.lower()
        
        # Search short-term cache
        if MemoryTier.SHORT_TERM in tiers:
            for entry in self.short_term_cache.values():
                if entry.metadata.domain in domains:
                    if self._is_relevant(entry, query_lower):
                        results.append(entry)
                        if len(results) >= max_results:
                            break
        
        # Search medium-term cache
        if MemoryTier.MEDIUM_TERM in tiers and len(results) < max_results:
            for entry in self.medium_term_cache.values():
                if entry.metadata.domain in domains:
                    if self._is_relevant(entry, query_lower):
                        results.append(entry)
                        if len(results) >= max_results:
                            break
        
        # Search long-term cache
        if MemoryTier.LONG_TERM in tiers and len(results) < max_results:
            for entry in self.long_term_cache.values():
                if entry.metadata.domain in domains:
                    if self._is_relevant(entry, query_lower):
                        results.append(entry)
                        if len(results) >= max_results:
                            break
        
        return results
    
    def _search_database(self, query: str, domains: List[MemoryDomain],
                         tiers: List[MemoryTier], max_results: int) -> List[MemoryEntry]:
        """Search database for relevant entries."""
        results = []
        query_lower = query.lower()
        
        domain_values = [d.value for d in domains]
        tier_values = [t.value for t in tiers]
        
        with sqlite3.connect(self.memory_db) as conn:
            # Use parameterized query to prevent SQL injection
            placeholders = ','.join(['?' for _ in domain_values])
            tier_placeholders = ','.join(['?' for _ in tier_values])
            query = f"""
                SELECT id FROM memory_entries 
                WHERE domain IN ({placeholders}) AND tier IN ({tier_placeholders})
                ORDER BY priority DESC, access_count DESC
                LIMIT ?
            """
            cursor = conn.execute(query, domain_values + tier_values + [max_results * 2])  # Get more for filtering
            
            for row in cursor:
                entry = self._load_memory_entry(row[0])
                if entry and self._is_relevant(entry, query_lower):
                    results.append(entry)
                    if len(results) >= max_results:
                        break
        
        return results
    
    def _is_relevant(self, entry: MemoryEntry, query: str) -> bool:
        """Check if memory entry is relevant to query."""
        try:
            # Check tags
            if any(tag.lower() in query for tag in entry.metadata.tags):
                return True
            
            # Check description
            if query in entry.metadata.description.lower():
                return True
            
            # Check data content (simplified)
            if isinstance(entry.data, str) and query in entry.data.lower():
                return True
            elif isinstance(entry.data, pd.DataFrame):
                if any(query in str(col).lower() for col in entry.data.columns):
                    return True
            
            return False
            
        except:
            return False
    
    def _load_memory_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Load memory entry from database."""
        try:
            with sqlite3.connect(self.memory_db) as conn:
                cursor = conn.execute("""
                    SELECT * FROM memory_entries WHERE id = ?
                """, (memory_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Reconstruct metadata
                metadata = MemoryMetadata(
                    id=row[0],
                    domain=MemoryDomain(row[1]),
                    tier=MemoryTier(row[2]),
                    data_type=DataType(row[3]),
                    source=row[4],
                    import_timestamp=datetime.fromisoformat(row[5]),
                    last_accessed=datetime.fromisoformat(row[6]),
                    access_count=row[7],
                    ttl_days=row[8],
                    expires_at=datetime.fromisoformat(row[9]),
                    size_bytes=row[10],
                    compression_ratio=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    relationships=json.loads(row[13]) if row[13] else [],
                    priority=row[14],
                    description=row[15]
                )
                
                # Load data
                data_file = self.memory_path / f"{memory_id}.data"
                if data_file.exists():
                    compressed_data = data_file.read_bytes()
                    
                                        # Check if data is compressed
                    if metadata.compression_ratio < 1.0:
                        # Security: Only load data from trusted sources
                        data = pickle.loads(gzip.decompress(compressed_data))
                    else:
                        # Security: Only load data from trusted sources
                        data = pickle.loads(compressed_data)
                    
                    return MemoryEntry(
                        metadata=metadata,
                        data=data,
                        compressed_data=compressed_data
                    )
                
                return None
                
        except Exception as e:
            print(f"⚠️  Error loading memory entry {memory_id}: {e}")
            return None
    
    def _update_access_stats(self, memory_id: str):
        """Update access statistics for memory entry."""
        try:
            with sqlite3.connect(self.memory_db) as conn:
                conn.execute("""
                    UPDATE memory_entries 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (datetime.now().isoformat(), memory_id))
                
        except Exception as e:
            print(f"⚠️  Error updating access stats: {e}")
    
    def add_relationship(self, source_entity: str, target_entity: str, 
                        relationship_type: str, strength: float = 1.0,
                        metadata: Dict = None):
        """Add relationship between entities."""
        try:
            with sqlite3.connect(self.relationship_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entity_relationships
                    (source_entity, target_entity, relationship_type, strength, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source_entity,
                    target_entity,
                    relationship_type,
                    strength,
                    json.dumps(metadata) if metadata else None
                ))
            
            # Update in-memory graph
            self.relationship_graph[source_entity].add(target_entity)
            self.relationship_graph[target_entity].add(source_entity)
            
        except Exception as e:
            print(f"❌ Error adding relationship: {e}")
    
    def get_related_entities(self, entity: str, max_depth: int = 2) -> List[str]:
        """Get related entities up to specified depth."""
        visited = set()
        to_visit = [(entity, 0)]
        related = []
        
        while to_visit:
            current, depth = to_visit.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            if current != entity:
                related.append(current)
            
            # Add neighbors
            for neighbor in self.relationship_graph[current]:
                if neighbor not in visited:
                    to_visit.append((neighbor, depth + 1))
        
        return related
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            with sqlite3.connect(self.memory_db) as conn:
                # Count by domain
                domain_counts = {}
                for domain in MemoryDomain:
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM memory_entries WHERE domain = ?
                    """, (domain.value,))
                    domain_counts[domain.value] = cursor.fetchone()[0]
                
                # Count by tier
                tier_counts = {}
                for tier in MemoryTier:
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM memory_entries WHERE tier = ?
                    """, (tier.value,))
                    tier_counts[tier.value] = cursor.fetchone()[0]
                
                # Size statistics
                cursor = conn.execute("""
                    SELECT SUM(size_bytes), AVG(size_bytes), COUNT(*) 
                    FROM memory_entries
                """)
                total_size, avg_size, total_entries = cursor.fetchone()
                
                # TTL statistics
                cursor = conn.execute("""
                    SELECT AVG(ttl_days), MIN(ttl_days), MAX(ttl_days)
                    FROM memory_entries
                """)
                avg_ttl, min_ttl, max_ttl = cursor.fetchone()
                
                return {
                    'total_entries': total_entries or 0,
                    'total_size_bytes': total_size or 0,
                    'average_size_bytes': avg_size or 0,
                    'domain_counts': domain_counts,
                    'tier_counts': tier_counts,
                    'ttl_stats': {
                        'average_days': avg_ttl or 0,
                        'minimum_days': min_ttl or 0,
                        'maximum_days': max_ttl or 0
                    },
                    'cache_stats': {
                        'short_term': len(self.short_term_cache),
                        'medium_term': len(self.medium_term_cache),
                        'long_term': len(self.long_term_cache)
                    },
                    'relationship_stats': {
                        'total_entities': len(self.relationship_graph),
                        'total_relationships': sum(len(neighbors) for neighbors in self.relationship_graph.values()) // 2
                    },
                    'performance_stats': self.stats
                }
                
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
            return {}
    
    def cleanup_expired_memory(self) -> int:
        """Clean up expired memory entries."""
        try:
            expired_count = 0
            now = datetime.now()
            
            with sqlite3.connect(self.memory_db) as conn:
                # Find expired entries
                cursor = conn.execute("""
                    SELECT id FROM memory_entries WHERE expires_at < ?
                """, (now.isoformat(),))
                
                expired_ids = [row[0] for row in cursor]
                
                for memory_id in expired_ids:
                    # Remove from database
                    conn.execute("DELETE FROM memory_entries WHERE id = ?", (memory_id,))
                    
                    # Remove data file
                    data_file = self.memory_path / f"{memory_id}.data"
                    if data_file.exists():
                        data_file.unlink()
                    
                    # Remove from search index
                    with sqlite3.connect(self.search_db) as search_conn:
                        search_conn.execute("DELETE FROM search_index WHERE memory_id = ?", (memory_id,))
                    
                    expired_count += 1
                
                # Remove from caches
                for cache in [self.short_term_cache, self.medium_term_cache, self.long_term_cache]:
                    expired_keys = [key for key, entry in cache.items() 
                                  if entry.metadata.expires_at < now]
                    for key in expired_keys:
                        del cache[key]
            
            print(f"🧹 Cleaned up {expired_count} expired memory entries")
            return expired_count
            
        except Exception as e:
            print(f"❌ Error cleaning up expired memory: {e}")
            return 0
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while True:
            try:
                # Clean up expired memory every hour
                self.cleanup_expired_memory()
                
                # Update statistics
                self.stats['total_entries'] = len(self._get_all_memory_ids())
                
                # Sleep for an hour
                time.sleep(3600)
                
            except Exception as e:
                print(f"⚠️  Maintenance loop error: {e}")
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _get_all_memory_ids(self) -> List[str]:
        """Get all memory IDs from database."""
        try:
            with sqlite3.connect(self.memory_db) as conn:
                cursor = conn.execute("SELECT id FROM memory_entries")
                return [row[0] for row in cursor]
        except:
            return []
    
    def export_memory_snapshot(self, output_path: str = None) -> str:
        """Export memory snapshot for backup or analysis."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"memory_snapshot_{timestamp}.json"
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.get_memory_stats(),
                'memory_entries': []
            }
            
            # Export metadata for all entries
            with sqlite3.connect(self.memory_db) as conn:
                cursor = conn.execute("SELECT * FROM memory_entries")
                for row in cursor:
                    snapshot['memory_entries'].append({
                        'id': row[0],
                        'domain': row[1],
                        'tier': row[2],
                        'data_type': row[3],
                        'source': row[4],
                        'import_timestamp': row[5],
                        'last_accessed': row[6],
                        'access_count': row[7],
                        'ttl_days': row[8],
                        'expires_at': row[9],
                        'size_bytes': row[10],
                        'compression_ratio': row[11],
                        'tags': row[12],
                        'relationships': row[13],
                        'priority': row[14],
                        'description': row[15]
                    })
            
            # Write snapshot
            with open(output_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            print(f"✅ Memory snapshot exported to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error exporting memory snapshot: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize memory manager
    memory_manager = ContextMemoryManager()
    
    print("\n🧪 Testing Context Memory Manager")
    print("=" * 50)
    
    # Test data import
    test_data = {
        'hosts': ['host1.example.com', 'host2.example.com', 'host3.example.com'],
        'ips': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
        'status': 'active'
    }
    
    # Import test data
    memory_id = memory_manager.import_data(
        domain=MemoryDomain.HOST_INVENTORY,
        data=test_data,
        source='test_import',
        tier=MemoryTier.SHORT_TERM,
        ttl_days=7,
        tags=['test', 'hosts', 'inventory'],
        description='Test host inventory data'
    )
    
    print(f"📝 Imported test data with ID: {memory_id}")
    
    # Test context retrieval
    results = memory_manager.retrieve_context(
        query='host inventory',
        domains=[MemoryDomain.HOST_INVENTORY],
        max_results=5
    )
    
    print(f"🔍 Retrieved {len(results)} context entries")
    for entry in results:
        print(f"   • {entry.metadata.domain.value}: {entry.metadata.description}")
    
    # Test relationship management
    memory_manager.add_relationship(
        source_entity='host1.example.com',
        target_entity='192.168.1.1',
        relationship_type='resolves_to',
        strength=1.0
    )
    
    related = memory_manager.get_related_entities('host1.example.com')
    print(f"🔗 Related entities for host1.example.com: {related}")
    
    # Get memory statistics
    stats = memory_manager.get_memory_stats()
    print(f"\n📊 Memory Statistics:")
    print(f"   Total entries: {stats.get('total_entries', 0)}")
    print(f"   Total size: {stats.get('total_size_bytes', 0)} bytes")
    print(f"   Cache sizes: {stats.get('cache_stats', {})}")
    
    # Export snapshot
    snapshot_path = memory_manager.export_memory_snapshot()
    print(f"💾 Memory snapshot exported to: {snapshot_path}")
