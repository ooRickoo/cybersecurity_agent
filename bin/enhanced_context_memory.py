#!/usr/bin/env python3
"""
Enhanced Context Memory Manager

This system provides optimized memory management using the distributed knowledge graph,
with intelligent caching, lazy loading, and performance optimizations for large-scale
knowledge graphs with 2M+ nodes.
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import logging
from collections import defaultdict, OrderedDict
import threading
from functools import lru_cache

from master_catalog import MasterCatalog

class MemoryCache:
    """Intelligent memory cache with LRU and importance-based eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.current_size = 0
        self.current_memory_mb = 0.0
        
        # LRU cache for frequently accessed items
        self.lru_cache = OrderedDict()
        
        # Importance-based cache for high-value items
        self.importance_cache = {}
        
        # Access tracking
        self.access_counts = defaultdict(int)
        self.last_access = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        with self.lock:
            # Check LRU cache first
            if key in self.lru_cache:
                item = self.lru_cache.pop(key)
                self.lru_cache[key] = item
                self._update_access_stats(key)
                return item
            
            # Check importance cache
            if key in self.importance_cache:
                self._update_access_stats(key)
                return self.importance_cache[key]
            
            return None
    
    def put(self, key: str, value: Any, importance_score: float = 0.5):
        """Put item in appropriate cache based on importance."""
        with self.lock:
            # Estimate memory usage
            item_size_mb = self._estimate_size_mb(value)
            
            # High importance items go to importance cache
            if importance_score > 0.8:
                self.importance_cache[key] = value
                self.current_memory_mb += item_size_mb
                self._update_access_stats(key)
            else:
                # Regular items go to LRU cache
                self.lru_cache[key] = value
                self.current_size += 1
                self.current_memory_mb += item_size_mb
                self._update_access_stats(key)
            
            # Evict if needed
            self._evict_if_needed()
    
    def _estimate_size_mb(self, value: Any) -> float:
        """Estimate memory usage of a value in MB."""
        try:
            # Rough estimation based on content length
            if isinstance(value, str):
                return len(value.encode('utf-8')) / (1024 * 1024)
            elif isinstance(value, dict):
                return len(json.dumps(value)) / (1024 * 1024)
            else:
                return 0.001  # Default small size
        except:
            return 0.001
    
    def _update_access_stats(self, key: str):
        """Update access statistics for a key."""
        self.access_counts[key] += 1
        self.last_access[key] = time.time()
    
    def _evict_if_needed(self):
        """Evict items if cache limits are exceeded."""
        # Evict from LRU cache if size limit exceeded
        while self.current_size > self.max_size and self.lru_cache:
            key, value = self.lru_cache.popitem(last=False)
            self.current_size -= 1
            self.current_memory_mb -= self._estimate_size_mb(value)
        
        # Evict from importance cache if memory limit exceeded
        if self.current_memory_mb > self.max_memory_mb:
            # Sort by importance and access frequency
            sorted_items = sorted(
                self.importance_cache.items(),
                key=lambda x: (x[1].get('importance_score', 0), self.access_counts.get(x[0], 0)),
                reverse=True
            )
            
            # Keep top items, remove rest
            keep_count = len(sorted_items) // 2  # Keep top 50%
            items_to_remove = sorted_items[keep_count:]
            
            for key, value in items_to_remove:
                del self.importance_cache[key]
                self.current_memory_mb -= self._estimate_size_mb(value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "lru_cache_size": self.current_size,
                "importance_cache_size": len(self.importance_cache),
                "total_memory_mb": round(self.current_memory_mb, 2),
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_mb,
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        return len(self.access_counts) / total_accesses

class EnhancedContextMemoryManager:
    """Enhanced context memory manager with distributed knowledge graph support."""
    
    def __init__(self, base_path: str = "knowledge-objects"):
        self.base_path = Path(base_path)
        self.master_catalog = MasterCatalog(base_path)
        
        # Memory caches
        self.short_term_cache = MemoryCache(max_size=500, max_memory_mb=50)
        self.workflow_cache = MemoryCache(max_size=200, max_memory_mb=25)
        
        # Session management
        self.current_session_id = None
        self.session_start_time = None
        self.session_memories: Set[str] = set()
        
        # Performance tracking
        self.performance_metrics = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time_ms": 0.0,
            "total_query_time_ms": 0.0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EnhancedContextMemory")
        
        # Background optimization thread
        self._start_optimization_thread()
    
    def _start_optimization_thread(self):
        """Start background thread for database optimization."""
        def optimization_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._background_optimization()
                except Exception as e:
                    self.logger.error(f"❌ Background optimization failed: {e}")
        
        thread = threading.Thread(target=optimization_worker, daemon=True)
        thread.start()
        self.logger.info("✅ Background optimization thread started")
    
    def _background_optimization(self):
        """Background database optimization."""
        try:
            # Optimize domains with high access frequency
            stats = self.master_catalog.get_domain_statistics()
            
            for domain_id, metrics in stats.get("performance_metrics", {}).items():
                if metrics.get("operations_24h", 0) > 1000:  # High activity domains
                    self.logger.info(f"🔧 Optimizing high-activity domain: {domain_id}")
                    self.master_catalog.optimize_domain(domain_id)
            
            # Clean up expired cache entries
            self._cleanup_expired_cache()
            
        except Exception as e:
            self.logger.error(f"❌ Background optimization failed: {e}")
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean short-term cache (older than 4 hours)
        with self.short_term_cache.lock:
            expired_keys = [
                key for key, last_access in self.short_term_cache.last_access.items()
                if current_time - last_access > 4 * 3600
            ]
            
            for key in expired_keys:
                if key in self.short_term_cache.lru_cache:
                    del self.short_term_cache.lru_cache[key]
                    self.short_term_cache.current_size -= 1
                if key in self.short_term_cache.importance_cache:
                    del self.short_term_cache.importance_cache[key]
        
        # Clean workflow cache (older than 24 hours)
        with self.workflow_cache.lock:
            expired_keys = [
                key for key, last_access in self.workflow_cache.last_access.items()
                if current_time - last_access > 24 * 3600
            ]
            
            for key in expired_keys:
                if key in self.workflow_cache.lru_cache:
                    del self.workflow_cache.lru_cache[key]
                    self.workflow_cache.current_size -= 1
                if key in self.workflow_cache.importance_cache:
                    del self.workflow_cache.importance_cache[key]
    
    def start_session(self, session_id: str):
        """Start a new memory session."""
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        self.session_memories.clear()
        
        self.logger.info(f"✅ Started memory session: {session_id}")
    
    def end_session(self):
        """End current session and cleanup."""
        if self.current_session_id:
            # Promote important session memories
            self._promote_session_memories()
            
            # Clear session-specific caches
            self.session_memories.clear()
            
            self.logger.info(f"✅ Ended memory session: {self.current_session_id}")
            
            self.current_session_id = None
            self.session_start_time = None
    
    def add_memory(self, domain_id: str, node_id: str, node_type: str, content: str,
                   metadata: Dict[str, Any] = None, importance_score: float = 0.5,
                   ttl_category: str = "long_term", relationships: List[Dict] = None) -> bool:
        """Add a memory node to a specific domain."""
        try:
            start_time = time.time()
            
            # Add to master catalog
            success = self.master_catalog.add_node(
                domain_id, node_id, node_type, content,
                metadata, importance_score, ttl_category
            )
            
            if success:
                # Add to appropriate cache
                cache_key = f"{domain_id}:{node_id}"
                
                if ttl_category == "short_term":
                    self.short_term_cache.put(cache_key, {
                        "domain_id": domain_id,
                        "node_id": node_id,
                        "node_type": node_type,
                        "content": content,
                        "metadata": metadata or {},
                        "importance_score": importance_score,
                        "ttl_category": ttl_category
                    }, importance_score)
                    
                    # Track in session
                    if self.current_session_id:
                        self.session_memories.add(cache_key)
                
                elif ttl_category == "medium_term":
                    self.workflow_cache.put(cache_key, {
                        "domain_id": domain_id,
                        "node_id": node_id,
                        "node_type": node_type,
                        "content": content,
                        "metadata": metadata or {},
                        "importance_score": importance_score,
                        "ttl_category": ttl_category
                    }, importance_score)
                
                # Add relationships if provided
                if relationships:
                    self._add_relationships(domain_id, node_id, relationships)
                
                # Update performance metrics
                query_time = (time.time() - start_time) * 1000
                self._update_performance_metrics(query_time, cache_hit=False)
                
                self.logger.info(f"✅ Added memory {node_id} to domain {domain_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add memory: {e}")
            return False
    
    def _add_relationships(self, domain_id: str, node_id: str, relationships: List[Dict]):
        """Add relationships for a node."""
        try:
            conn = self.master_catalog.get_domain_connection(domain_id)
            if not conn:
                return
            
            cursor = conn.cursor()
            
            for rel in relationships:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {domain_id}_relationships 
                    (source_node_id, target_node_id, relationship_type, strength, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    node_id, rel["target"], rel["type"],
                    rel.get("strength", 1.0), json.dumps(rel.get("metadata", {}))
                ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add relationships: {e}")
    
    def get_memory(self, node_id: str, domain_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a memory node by ID."""
        try:
            start_time = time.time()
            
            # Try cache first
            if domain_id:
                cache_key = f"{domain_id}:{node_id}"
                
                # Check short-term cache
                cached_item = self.short_term_cache.get(cache_key)
                if cached_item:
                    self._update_performance_metrics(0, cache_hit=True)
                    return cached_item
                
                # Check workflow cache
                cached_item = self.workflow_cache.get(cache_key)
                if cached_item:
                    self._update_performance_metrics(0, cache_hit=True)
                    return cached_item
            
            # If not in cache, search across domains
            if not domain_id:
                # Search global index
                results = self.master_catalog.search_across_domains(
                    query=node_id, max_results=1
                )
                
                if results:
                    result = results[0]
                    domain_id = result["domain_id"]
                    cache_key = f"{domain_id}:{node_id}"
                    
                    # Cache the result
                    self.workflow_cache.put(cache_key, result, result["importance_score"])
                    
                    query_time = (time.time() - start_time) * 1000
                    self._update_performance_metrics(query_time, cache_hit=False)
                    
                    return result
            
            # Direct domain lookup
            if domain_id:
                conn = self.master_catalog.get_domain_connection(domain_id)
                if conn:
                    cursor = conn.cursor()
                    # Use parameterized query to prevent SQL injection
                    query = f"""
                        SELECT node_id, node_type, content, metadata, importance_score, ttl_category
                        FROM {domain_id}_nodes WHERE node_id = ?
                    """
                    cursor.execute(query, (node_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        result = {
                            "domain_id": domain_id,
                            "node_id": row[0],
                            "node_type": row[1],
                            "content": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {},
                            "importance_score": row[4],
                            "ttl_category": row[5]
                        }
                        
                        # Cache the result
                        cache_key = f"{domain_id}:{node_id}"
                        self.workflow_cache.put(cache_key, result, result["importance_score"])
                        
                        query_time = (time.time() - start_time) * 1000
                        self._update_performance_metrics(query_time, cache_hit=False)
                        
                        return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory: {e}")
            return None
    
    def search_memories(self, query: str, node_types: List[str] = None,
                       domains: List[str] = None, min_importance: float = 0.0,
                       max_results: int = 100) -> List[Dict[str, Any]]:
        """Search memories across domains with intelligent caching."""
        try:
            start_time = time.time()
            
            # Generate cache key for search
            search_key = hashlib.sha256(
                f"{query}:{':'.join(node_types or [])}:{':'.join(domains or [])}:{min_importance}:{max_results}".encode()
            ).hexdigest()
            
            # Check workflow cache for search results
            cached_results = self.workflow_cache.get(search_key)
            if cached_results:
                self._update_performance_metrics(0, cache_hit=True)
                return cached_results
            
            # Execute search
            results = self.master_catalog.search_across_domains(
                query, node_types, domains, min_importance, max_results
            )
            
            # Cache search results
            self.workflow_cache.put(search_key, results, 0.6)  # Medium importance for search results
            
            # Update performance metrics
            query_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(query_time, cache_hit=False)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Memory search failed: {e}")
            return []
    
    def get_workflow_context(self, workflow_context: str, max_nodes: int = 50) -> Dict[str, Any]:
        """Get optimized workflow context with intelligent domain selection."""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = hashlib.sha256(f"workflow:{workflow_context}:{max_nodes}".encode()).hexdigest()
            
            # Check cache first
            cached_context = self.workflow_cache.get(cache_key)
            if cached_context:
                self._update_performance_metrics(0, cache_hit=True)
                return cached_context
            
            # Get context from master catalog
            context = self.master_catalog.get_workflow_context(workflow_context, max_nodes)
            
            # Cache the context
            self.workflow_cache.put(cache_key, context, 0.7)
            
            # Update performance metrics
            query_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(query_time, cache_hit=False)
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get workflow context: {e}")
            return {"error": str(e)}
    
    def suggest_memory_promotion(self, workflow_output: Dict[str, Any],
                                context_used: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest memory promotion with domain-specific recommendations."""
        try:
            suggestions = {
                "promote_to_medium": [],
                "promote_to_long": [],
                "domain_recommendations": {},
                "reasoning": []
            }
            
            # Analyze workflow output
            if "dataframes" in workflow_output:
                for df_name, df_info in workflow_output["dataframes"].items():
                    if df_info.get("rows", 0) > 100:
                        suggestions["promote_to_medium"].append({
                            "type": "dataframe",
                            "name": df_name,
                            "reason": "Large dataset with potential reuse value",
                            "suggested_domain": "organization"  # General business data
                        })
            
            if "analysis_results" in workflow_output:
                for result in workflow_output["analysis_results"]:
                    if result.get("confidence", 0) > 0.8:
                        # Determine appropriate domain based on content
                        domain = self._suggest_domain_for_content(result.get("summary", ""))
                        
                        suggestions["promote_to_long"].append({
                            "type": "analysis_result",
                            "content": result.get("summary", "High confidence analysis"),
                            "reason": "High confidence result with organizational value",
                            "suggested_domain": domain
                        })
            
            # Domain-specific recommendations
            for domain_id in context_used.get("domains", {}):
                if domain_id in ["grc", "compliance"]:
                    suggestions["domain_recommendations"][domain_id] = "High-value domain for long-term retention"
                elif domain_id in ["threat-intelligence", "incidents"]:
                    suggestions["domain_recommendations"][domain_id] = "Critical for threat analysis and incident response"
                elif domain_id in ["networks", "hosts"]:
                    suggestions["domain_recommendations"][domain_id] = "Infrastructure information for operational use"
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ Memory promotion suggestion failed: {e}")
            return {"error": str(e)}
    
    def _suggest_domain_for_content(self, content: str) -> str:
        """Suggest appropriate domain for content based on keywords."""
        content_lower = content.lower()
        
        # Domain-specific keyword matching
        domain_keywords = {
            "grc": ["policy", "control", "compliance", "risk", "audit", "governance"],
            "threat-intelligence": ["threat", "actor", "indicator", "ttp", "campaign", "malware"],
            "networks": ["network", "segment", "vlan", "routing", "connectivity", "topology"],
            "hosts": ["host", "server", "endpoint", "device", "workstation"],
            "incidents": ["incident", "breach", "alert", "investigation", "response"],
            "compliance": ["nist", "iso", "soc2", "pci", "sox", "framework"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return domain
        
        return "organization"  # Default domain
    
    def _promote_session_memories(self):
        """Promote important session memories to longer-term storage."""
        try:
            for cache_key in self.session_memories:
                # Get memory from cache
                memory = self.short_term_cache.get(cache_key)
                if memory and memory.get("importance_score", 0) > 0.7:
                    # Promote to medium-term
                    domain_id = memory["domain_id"]
                    node_id = memory["node_id"]
                    
                    # Update TTL in domain database
                    conn = self.master_catalog.get_domain_connection(domain_id)
                    if conn:
                        cursor = conn.cursor()
                        # Use parameterized query to prevent SQL injection
                        query = f"""
                            UPDATE {domain_id}_nodes 
                            SET ttl_category = 'medium_term', updated_at = CURRENT_TIMESTAMP
                            WHERE node_id = ?
                        """
                        cursor.execute(query, (node_id,))
                        conn.commit()
                        
                        self.logger.info(f"✅ Promoted {node_id} to medium-term memory")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to promote session memories: {e}")
    
    def _update_performance_metrics(self, query_time_ms: float, cache_hit: bool):
        """Update performance tracking metrics."""
        self.performance_metrics["queries_executed"] += 1
        self.performance_metrics["total_query_time_ms"] += query_time_ms
        
        if cache_hit:
            self.performance_metrics["cache_hits"] += 1
        else:
            self.performance_metrics["cache_misses"] += 1
        
        # Update average query time
        self.performance_metrics["avg_query_time_ms"] = (
            self.performance_metrics["total_query_time_ms"] / 
            self.performance_metrics["queries_executed"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.performance_metrics,
            "cache_stats": {
                "short_term": self.short_term_cache.get_stats(),
                "workflow": self.workflow_cache.get_stats()
            },
            "master_catalog_stats": self.master_catalog.get_domain_statistics(),
            "session_info": {
                "current_session": self.current_session_id,
                "session_memories_count": len(self.session_memories),
                "session_duration": (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance."""
        try:
            optimization_results = {
                "cache_optimization": {},
                "database_optimization": {},
                "recommendations": []
            }
            
            # Optimize caches
            self.short_term_cache._evict_if_needed()
            self.workflow_cache._evict_if_needed()
            
            optimization_results["cache_optimization"] = {
                "short_term": self.short_term_cache.get_stats(),
                "workflow": self.workflow_cache.get_stats()
            }
            
            # Optimize databases
            stats = self.master_catalog.get_domain_statistics()
            for domain_id in stats.get("domains", {}):
                if stats["domains"][domain_id]["node_count"] > 10000:  # Large domains
                    self.logger.info(f"🔧 Optimizing large domain: {domain_id}")
                    result = self.master_catalog.optimize_domain(domain_id)
                    optimization_results["database_optimization"][domain_id] = result
            
            # Generate recommendations
            hit_rate = self.performance_metrics["cache_hits"] / max(self.performance_metrics["queries_executed"], 1)
            if hit_rate < 0.5:
                optimization_results["recommendations"].append("Consider increasing cache sizes for better hit rates")
            
            if self.performance_metrics["avg_query_time_ms"] > 100:
                optimization_results["recommendations"].append("Query performance below threshold, consider database optimization")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Performance optimization failed: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close all connections and cleanup."""
        try:
            # Close master catalog connections
            self.master_catalog.close_all_connections()
            
            # Clear caches
            self.short_term_cache.lru_cache.clear()
            self.short_term_cache.importance_cache.clear()
            self.workflow_cache.lru_cache.clear()
            self.workflow_cache.importance_cache.clear()
            
            self.logger.info("✅ Enhanced Context Memory Manager closed")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to close manager: {e}")

if __name__ == "__main__":
    # Test the enhanced context memory manager
    memory_manager = EnhancedContextMemoryManager()
    
    print("🧠 Testing Enhanced Context Memory Manager...")
    
    # Start session
    memory_manager.start_session("test_session_001")
    
    # Add test memories
    success = memory_manager.add_memory(
        "grc", "enhanced_policy_001", "policy",
        "Enhanced policy management with distributed storage",
        metadata={"category": "test", "priority": "high"},
        importance_score=0.9
    )
    
    if success:
        print("✅ Test memory added successfully")
        
        # Test retrieval
        memory = memory_manager.get_memory("enhanced_policy_001", "grc")
        if memory:
            print(f"✅ Memory retrieved: {memory['content'][:50]}...")
        
        # Test search
        results = memory_manager.search_memories("policy", domains=["grc"])
        print(f"✅ Search returned {len(results)} results")
        
        # Get performance stats
        stats = memory_manager.get_performance_stats()
        print(f"✅ Performance stats: {stats['queries_executed']} queries executed")
    
    # End session
    memory_manager.end_session()
    
    # Close manager
    memory_manager.close()

