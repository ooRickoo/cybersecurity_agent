#!/usr/bin/env python3
"""
Framework Knowledge Graph Memory Context

Integrates cybersecurity frameworks with knowledge graph memory context,
providing TTL-based memory management and workflow task context.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import sqlite3
import threading
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")

logger = logging.getLogger(__name__)

@dataclass
class FrameworkMemory:
    """Framework memory object with TTL."""
    framework_id: str
    framework_name: str
    framework_type: str
    data_hash: str
    flattened_data: Dict[str, Any]
    query_index: Dict[str, List[str]]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    ttl_days: int
    access_count: int
    memory_size: int
    tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowTaskContext:
    """Context for specific workflow tasks."""
    task_id: str
    task_name: str
    task_type: str
    frameworks_required: List[str]
    created_at: datetime
    last_updated: datetime
    context_data: Dict[str, Any]
    status: str  # 'active', 'completed', 'failed'
    priority: int

class FrameworkKnowledgeGraph:
    """Knowledge graph for cybersecurity frameworks with memory management."""
    
    def __init__(self, db_path: str = "knowledge-objects/framework_knowledge.db"):
        """Initialize the framework knowledge graph."""
        self.db_path = db_path
        self.db_lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Memory cache
        self.memory_cache: Dict[str, FrameworkMemory] = {}
        
        # Task contexts
        self.task_contexts: Dict[str, WorkflowTaskContext] = {}
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("🚀 Framework Knowledge Graph initialized")
    
    def _initialize_database(self):
        """Initialize the SQLite database."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create frameworks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS frameworks (
                        framework_id TEXT PRIMARY KEY,
                        framework_name TEXT NOT NULL,
                        framework_type TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        flattened_data TEXT NOT NULL,
                        query_index TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        ttl_days INTEGER NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        memory_size INTEGER NOT NULL,
                        tags TEXT NOT NULL
                    )
                ''')
                
                # Create task contexts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS task_contexts (
                        task_id TEXT PRIMARY KEY,
                        task_name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        frameworks_required TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        context_data TEXT NOT NULL,
                        status TEXT NOT NULL,
                        priority INTEGER DEFAULT 0
                    )
                ''')
                
                # Create access log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        framework_id TEXT NOT NULL,
                        access_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        task_id TEXT,
                        access_details TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_frameworks_type ON frameworks(framework_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_frameworks_tags ON frameworks(tags)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_frameworks_ttl ON frameworks(ttl_days)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_contexts_type ON task_contexts(task_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_contexts_status ON task_contexts(status)')
                
                conn.commit()
                conn.close()
                
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    async def store_framework(self, framework_data: Dict[str, Any], ttl_days: int = 30) -> Dict[str, Any]:
        """Store a framework in the knowledge graph."""
        try:
            framework_id = framework_data.get("id")
            framework_name = framework_data.get("name")
            framework_type = framework_data.get("framework_type", "unknown")
            
            # Calculate data hash
            data_str = json.dumps(framework_data, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Check if already exists
            existing = await self.get_framework(framework_id)
            if existing:
                # Update existing framework
                return await self._update_existing_framework(framework_id, framework_data, ttl_days)
            
            # Extract query index from the flattened data structure
            query_index = {}
            if "query_index" in framework_data:
                query_index = framework_data["query_index"]
            elif "query_index_keys" in framework_data:
                # If we only have keys, we need to rebuild the index
                logger.info(f"📊 Rebuilding query index from {len(framework_data.get('query_index_keys', []))} keys")
                # For now, we'll use an empty index and rebuild it later
                query_index = {}
            
            # Create new framework memory
            framework_memory = FrameworkMemory(
                framework_id=framework_id,
                framework_name=framework_name,
                framework_type=framework_type,
                data_hash=data_hash,
                flattened_data=framework_data,
                query_index=query_index,
                metadata=framework_data.get("metadata", {}),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_days=ttl_days,
                access_count=1,
                memory_size=len(data_str),
                tags=framework_data.get("metadata", {}).get("tags", [])
            )
            
            # Store in database
            await self._store_framework_in_db(framework_memory)
            
            # Store in cache
            self.memory_cache[framework_id] = framework_memory
            
            logger.info(f"✅ Stored framework: {framework_name} (TTL: {ttl_days} days)")
            
            return {
                "success": True,
                "framework_id": framework_id,
                "framework_name": framework_name,
                "stored_at": framework_memory.created_at.isoformat(),
                "ttl_days": ttl_days,
                "memory_size": framework_memory.memory_size
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store framework: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_existing_framework(self, framework_id: str, framework_data: Dict[str, Any], ttl_days: int) -> Dict[str, Any]:
        """Update an existing framework in the knowledge graph."""
        try:
            # Calculate new data hash
            data_str = json.dumps(framework_data, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Update existing framework memory
            existing = await self.get_framework(framework_id)
            if existing:
                existing.flattened_data = framework_data
                existing.query_index = framework_data.get("query_index", {})
                existing.metadata = framework_data.get("metadata", {})
                existing.last_accessed = datetime.now()
                existing.ttl_days = ttl_days
                existing.memory_size = len(data_str)
                existing.tags = framework_data.get("metadata", {}).get("tags", [])
                
                # Update in database
                await self._store_framework_in_db(existing)
                
                # Update in cache
                self.memory_cache[framework_id] = existing
                
                logger.info(f"✅ Updated framework: {existing.framework_name}")
                
                return {
                    "success": True,
                    "framework_id": framework_id,
                    "framework_name": existing.framework_name,
                    "updated_at": existing.last_accessed.isoformat(),
                    "ttl_days": ttl_days,
                    "memory_size": existing.memory_size
                }
            
            return {"success": False, "error": "Framework not found for update"}
            
        except Exception as e:
            logger.error(f"❌ Failed to update framework: {e}")
            return {"success": False, "error": str(e)}
    
    async def _store_framework_in_db(self, framework_memory: FrameworkMemory):
        """Store framework in database."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO frameworks 
                    (framework_id, framework_name, framework_type, data_hash, flattened_data, 
                     query_index, metadata, created_at, last_accessed, ttl_days, 
                     access_count, memory_size, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    framework_memory.framework_id,
                    framework_memory.framework_name,
                    framework_memory.framework_type,
                    framework_memory.data_hash,
                    json.dumps(framework_memory.flattened_data),
                    json.dumps(framework_memory.query_index),
                    json.dumps(framework_memory.metadata),
                    framework_memory.created_at.isoformat(),
                    framework_memory.last_accessed.isoformat(),
                    framework_memory.ttl_days,
                    framework_memory.access_count,
                    framework_memory.memory_size,
                    json.dumps(framework_memory.tags)
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Database storage failed: {e}")
            raise
    
    async def get_framework(self, framework_id: str) -> Optional[FrameworkMemory]:
        """Get a framework from the knowledge graph."""
        try:
            # Check cache first
            if framework_id in self.memory_cache:
                framework = self.memory_cache[framework_id]
                
                # Check TTL
                if self._is_expired(framework):
                    logger.info(f"🗑️  Framework expired: {framework_id}")
                    await self.remove_framework(framework_id)
                    return None
                
                # Update access count and timestamp
                framework.access_count += 1
                framework.last_accessed = datetime.now()
                await self._update_access_log(framework_id, "cache_hit")
                
                return framework
            
            # Check database
            framework = await self._get_framework_from_db(framework_id)
            if framework:
                # Check TTL
                if self._is_expired(framework):
                    logger.info(f"🗑️  Framework expired: {framework_id}")
                    await self.remove_framework(framework_id)
                    return None
                
                # Add to cache
                self.memory_cache[framework_id] = framework
                
                # Update access count and timestamp
                framework.access_count += 1
                framework.last_accessed = datetime.now()
                await self._update_access_log(framework_id, "db_hit")
                
                return framework
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get framework: {e}")
            return None
    
    async def _get_framework_from_db(self, framework_id: str) -> Optional[FrameworkMemory]:
        """Get framework from database."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT framework_id, framework_name, framework_type, data_hash, 
                           flattened_data, query_index, metadata, created_at, 
                           last_accessed, ttl_days, access_count, memory_size, tags
                    FROM frameworks WHERE framework_id = ?
                ''', (framework_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return FrameworkMemory(
                        framework_id=row[0],
                        framework_name=row[1],
                        framework_type=row[2],
                        data_hash=row[3],
                        flattened_data=json.loads(row[4]),
                        query_index=json.loads(row[5]),
                        metadata=json.loads(row[6]),
                        created_at=datetime.fromisoformat(row[7]),
                        last_accessed=datetime.fromisoformat(row[8]),
                        ttl_days=row[9],
                        access_count=row[10],
                        memory_size=row[11],
                        tags=json.loads(row[12])
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Database retrieval failed: {e}")
            return None
    
    def _is_expired(self, framework: FrameworkMemory) -> bool:
        """Check if framework is expired."""
        expiry_date = framework.created_at + timedelta(days=framework.ttl_days)
        return datetime.now() > expiry_date
    
    async def remove_framework(self, framework_id: str) -> bool:
        """Remove a framework from the knowledge graph."""
        try:
            # Remove from cache
            if framework_id in self.memory_cache:
                del self.memory_cache[framework_id]
            
            # Remove from database
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM frameworks WHERE framework_id = ?', (framework_id,))
                
                conn.commit()
                conn.close()
            
            logger.info(f"🗑️  Removed framework: {framework_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove framework: {e}")
            return False
    
    async def query_framework(self, framework_id: str, query: str, max_results: int = 50) -> Dict[str, Any]:
        """Query a framework using the knowledge graph."""
        try:
            framework = await self.get_framework(framework_id)
            if not framework:
                return {"success": False, "error": f"Framework not found: {framework_id}"}
            
            # Update access log
            await self._update_access_log(framework_id, "query", query=query)
            
            # Simple text search in query index
            query_lower = query.lower()
            results = []
            
            # Debug logging
            logger.info(f"🔍 Querying framework {framework_id} for '{query}'")
            logger.info(f"📊 Query index keys: {len(framework.query_index)}")
            logger.info(f"📊 Flattened data keys: {list(framework.flattened_data.keys())}")
            
            # Check if query index exists
            if not framework.query_index:
                logger.warning(f"⚠️  No query index found for framework {framework_id}")
                return {
                    "success": True,
                    "framework_id": framework_id,
                    "framework_name": framework.framework_name,
                    "query": query,
                    "results": [],
                    "total_found": 0,
                    "query_time": datetime.now().isoformat(),
                    "warning": "No query index available"
                }
            
            # Search through query index
            for word, item_ids in framework.query_index.items():
                if query_lower in word:
                    logger.info(f"🔍 Found matching word: '{word}' with {len(item_ids)} items")
                    for item_id in item_ids:
                        # Find the item in flattened data
                        items = framework.flattened_data.get("items", [])
                        if not items:
                            logger.warning(f"⚠️  No items found in flattened data")
                            break
                        
                        for item in items:
                            if item.get("id") == item_id:
                                results.append(item)
                                break
                        
                        if len(results) >= max_results:
                            break
                
                if len(results) >= max_results:
                    break
            
            return {
                "success": True,
                "framework_id": framework_id,
                "framework_name": framework.framework_name,
                "query": query,
                "results": results,
                "total_found": len(results),
                "query_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Framework query failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_task_context(self, task_name: str, task_type: str, 
                                frameworks_required: List[str], priority: int = 0) -> Dict[str, Any]:
        """Create a workflow task context."""
        try:
            task_id = f"task_{int(datetime.now().timestamp())}_{hash(task_name)}"
            
            task_context = WorkflowTaskContext(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                frameworks_required=frameworks_required,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                context_data={},
                status="active",
                priority=priority
            )
            
            # Store in memory
            self.task_contexts[task_id] = task_context
            
            # Store in database
            await self._store_task_context_in_db(task_context)
            
            logger.info(f"✅ Created task context: {task_name} (ID: {task_id})")
            
            return {
                "success": True,
                "task_id": task_id,
                "task_name": task_name,
                "status": "active",
                "frameworks_required": frameworks_required
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create task context: {e}")
            return {"success": False, "error": str(e)}
    
    async def _store_task_context_in_db(self, task_context: WorkflowTaskContext):
        """Store task context in database."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO task_contexts 
                    (task_id, task_name, task_type, frameworks_required, created_at, 
                     last_updated, context_data, status, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_context.task_id,
                    task_context.task_name,
                    task_context.task_type,
                    json.dumps(task_context.frameworks_required),
                    task_context.created_at.isoformat(),
                    task_context.last_updated.isoformat(),
                    json.dumps(task_context.context_data),
                    task_context.status,
                    task_context.priority
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Task context storage failed: {e}")
            raise
    
    async def update_task_context(self, task_id: str, context_data: Dict[str, Any], 
                                status: str = None) -> Dict[str, Any]:
        """Update a workflow task context."""
        try:
            if task_id not in self.task_contexts:
                return {"success": False, "error": f"Task context not found: {task_id}"}
            
            task_context = self.task_contexts[task_id]
            task_context.context_data.update(context_data)
            task_context.last_updated = datetime.now()
            
            if status:
                task_context.status = status
            
            # Update database
            await self._store_task_context_in_db(task_context)
            
            logger.info(f"✅ Updated task context: {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "status": task_context.status,
                "last_updated": task_context.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to update task context: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_task_context(self, task_id: str) -> Optional[WorkflowTaskContext]:
        """Get a workflow task context."""
        return self.task_contexts.get(task_id)
    
    async def _update_access_log(self, framework_id: str, access_type: str, 
                                task_id: str = None, query: str = None):
        """Update access log."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO access_log (framework_id, access_type, timestamp, task_id, access_details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    framework_id,
                    access_type,
                    datetime.now().isoformat(),
                    task_id,
                    json.dumps({"query": query}) if query else None
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Access log update failed: {e}")
    
    def _background_cleanup(self):
        """Background thread for cleaning up expired frameworks."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_expired_frameworks()
            except Exception as e:
                logger.error(f"❌ Background cleanup failed: {e}")
    
    def _cleanup_expired_frameworks(self):
        """Clean up expired frameworks."""
        try:
            expired_frameworks = []
            
            # Check cache
            for framework_id, framework in self.memory_cache.items():
                if self._is_expired(framework):
                    expired_frameworks.append(framework_id)
            
            # Remove expired frameworks (use threading instead of asyncio)
            for framework_id in expired_frameworks:
                try:
                    # Remove from cache
                    if framework_id in self.memory_cache:
                        del self.memory_cache[framework_id]
                    
                    # Remove from database with timeout
                    if self.db_lock.acquire(timeout=0.5):
                        try:
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute('DELETE FROM frameworks WHERE framework_id = ?', (framework_id,))
                            conn.commit()
                            conn.close()
                        finally:
                            self.db_lock.release()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to remove expired framework {framework_id}: {e}")
            
            if expired_frameworks:
                logger.info(f"🧹 Cleaned up {len(expired_frameworks)} expired frameworks")
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Use a timeout for the lock to prevent hanging
            if self.db_lock.acquire(timeout=1.0):
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Framework stats
                    cursor.execute('SELECT COUNT(*) FROM frameworks')
                    total_frameworks = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT SUM(memory_size) FROM frameworks')
                    total_memory = cursor.fetchone()[0] or 0
                    
                    cursor.execute('SELECT COUNT(*) FROM task_contexts WHERE status = "active"')
                    active_tasks = cursor.fetchone()[0]
                    
                    # TTL distribution
                    cursor.execute('''
                        SELECT ttl_days, COUNT(*) FROM frameworks GROUP BY ttl_days
                    ''')
                    ttl_distribution = dict(cursor.fetchall())
                    
                    conn.close()
                    
                    return {
                        "success": True,
                        "total_frameworks": total_frameworks,
                        "cached_frameworks": len(self.memory_cache),
                        "total_memory_bytes": total_memory,
                        "total_memory_mb": round(total_memory / (1024 * 1024), 2),
                        "active_tasks": active_tasks,
                        "ttl_distribution": ttl_distribution,
                        "cache_hit_rate": self._calculate_cache_hit_rate()
                    }
                finally:
                    self.db_lock.release()
            else:
                logger.warning("⚠️  Could not acquire database lock for memory stats")
                return {
                    "success": True,
                    "total_frameworks": len(self.memory_cache),
                    "cached_frameworks": len(self.memory_cache),
                    "total_memory_bytes": 0,
                    "total_memory_mb": 0.0,
                    "active_tasks": len(self.task_contexts),
                    "ttl_distribution": {},
                    "cache_hit_rate": 0.0
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        try:
            # Use a timeout for the lock to prevent hanging
            if self.db_lock.acquire(timeout=1.0):
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT COUNT(*) FROM access_log')
                    total_accesses = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM access_log WHERE access_type = "cache_hit"')
                    cache_hits = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    if total_accesses > 0:
                        return round(cache_hits / total_accesses, 3)
                    else:
                        return 0.0
                finally:
                    self.db_lock.release()
            else:
                logger.warning("⚠️  Could not acquire database lock for cache hit rate calculation")
                return 0.0
                    
        except Exception as e:
            logger.error(f"❌ Cache hit rate calculation failed: {e}")
            return 0.0
    
    def clear_cache(self):
        """Clear memory cache."""
        self.memory_cache.clear()
        logger.info("🧹 Memory cache cleared")

# Example usage
async def main():
    """Test Framework Knowledge Graph."""
    print("🚀 Testing Framework Knowledge Graph")
    print("=" * 50)
    
    # Create knowledge graph
    kg = FrameworkKnowledgeGraph()
    
    # Test storing a framework
    print("📊 Testing framework storage...")
    framework_data = {
        "id": "mitre_attack_test",
        "name": "MITRE ATT&CK Test",
        "framework_type": "stix",
        "query_index": {
            "attack": ["T1234", "T5678"],
            "technique": ["T1234", "T5678"]
        },
        "metadata": {
            "tags": ["threat_intelligence", "attack_patterns"],
            "version": "1.0"
        }
    }
    
    store_result = await kg.store_framework(framework_data, ttl_days=30)
    print(f"Storage result: {store_result}")
    
    # Test retrieving framework
    print("\n🔍 Testing framework retrieval...")
    framework = await kg.get_framework("mitre_attack_test")
    if framework:
        print(f"Retrieved: {framework.framework_name}")
        print(f"Memory size: {framework.memory_size} bytes")
        print(f"TTL: {framework.ttl_days} days")
    
    # Test querying framework
    print("\n🔍 Testing framework query...")
    query_result = await kg.query_framework("mitre_attack_test", "attack")
    print(f"Query result: {query_result}")
    
    # Test task context
    print("\n📋 Testing task context...")
    task_result = await kg.create_task_context(
        "threat_analysis",
        "security_analysis",
        ["mitre_attack_test"],
        priority=1
    )
    print(f"Task context result: {task_result}")
    
    # Get memory stats
    print("\n📊 Memory statistics...")
    stats = kg.get_memory_stats()
    print(f"Total frameworks: {stats.get('total_frameworks', 0)}")
    print(f"Cached frameworks: {stats.get('cached_frameworks', 0)}")
    print(f"Total memory: {stats.get('total_memory_mb', 0)} MB")
    print(f"Cache hit rate: {stats.get('cache_hit_rate', 0)}")
    
    print(f"\n🎉 Framework Knowledge Graph test completed!")

if __name__ == "__main__":
    asyncio.run(main())
