#!/usr/bin/env python3
"""
Performance Optimization System for Cybersecurity Agent
Provides intelligent caching, parallelization, and resource optimization
"""

import asyncio
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import pickle
import sqlite3
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['accessed_at'] = self.accessed_at.isoformat()
        return data

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    execution_time: float
    memory_usage: int
    cpu_usage: float
    cache_hit_rate: float
    parallelization_efficiency: float
    tool_usage_stats: Dict[str, Dict[str, Any]]

class CacheManager:
    """Intelligent caching system with TTL and size management."""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_db_path = Path("knowledge-objects/performance_cache.db")
        self.cache_db_path.parent.mkdir(exist_ok=True)
        self._init_cache_db()
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def _init_cache_db(self):
        """Initialize cache database."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        created_at TEXT,
                        accessed_at TEXT,
                        access_count INTEGER,
                        size_bytes INTEGER,
                        ttl_seconds INTEGER
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
        except Exception as e:
            logger.warning(f"Cache database initialization failed: {e}")
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        # Create a stable representation of arguments
        args_repr = str(sorted(args)) + str(sorted(kwargs.items()))
        return hashlib.sha256(f"{func_name}:{args_repr}".encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                return entry.value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            # Estimate size
            size_bytes = len(pickle.dumps(value))
            
            # Check if we have space
            if not self._has_space(size_bytes):
                self._evict_entries(size_bytes)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache[key] = entry
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def _has_space(self, required_bytes: int) -> bool:
        """Check if cache has space for new entry."""
        current_size = sum(entry.size_bytes for entry in self.cache.values())
        return (current_size + required_bytes) <= self.max_size_bytes
    
    def _evict_entries(self, required_bytes: int):
        """Evict cache entries to make space."""
        # Sort by access count and last access time (LRU with frequency)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda x: (x.access_count, x.accessed_at)
        )
        
        freed_bytes = 0
        for entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break
            del self.cache[entry.key]
            freed_bytes += entry.size_bytes
    
    def _background_cleanup(self):
        """Background cleanup of expired entries."""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]
                for key in expired_keys:
                    del self.cache[key]
            except Exception as e:
                logger.error(f"Background cleanup failed: {e}")

class ParallelExecutor:
    """Parallel execution engine for workflow steps."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def execute_parallel(self, tasks: List[Callable], use_processes: bool = False) -> List[Any]:
        """Execute tasks in parallel."""
        if not tasks:
            return []
        
        if len(tasks) == 1:
            # Single task, no need for parallelization
            return [await self._execute_task(tasks[0])]
        
        # Choose executor based on task type
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Execute tasks in parallel
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(executor, self._execute_task, task)
            for task in tasks
        ]
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _execute_task(self, task: Callable) -> Any:
        """Execute a single task."""
        try:
            if asyncio.iscoroutinefunction(task):
                # Handle async tasks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(task())
                finally:
                    loop.close()
            else:
                # Handle sync tasks
                return task()
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.metrics_db_path = Path("knowledge-objects/performance_metrics.db")
        self.metrics_db_path.parent.mkdir(exist_ok=True)
        self._init_metrics_db()
    
    def _init_metrics_db(self):
        """Initialize metrics database."""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_time REAL,
                        memory_usage INTEGER,
                        cpu_usage REAL,
                        cache_hit_rate REAL,
                        parallelization_efficiency REAL,
                        tool_usage_stats TEXT,
                        timestamp TEXT
                    )
                """)
        except Exception as e:
            logger.warning(f"Metrics database initialization failed: {e}")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Store in database
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (execution_time, memory_usage, cpu_usage, cache_hit_rate, 
                     parallelization_efficiency, tool_usage_stats, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.execution_time,
                    metrics.memory_usage,
                    metrics.parallelization_efficiency,
                    metrics.cache_hit_rate,
                    metrics.parallelization_efficiency,
                    json.dumps(metrics.tool_usage_stats),
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        execution_times = [m.execution_time for m in self.metrics_history]
        cache_hit_rates = [m.cache_hit_rate for m in self.metrics_history]
        parallelization_efficiencies = [m.parallelization_efficiency for m in self.metrics_history]
        
        return {
            "total_executions": len(self.metrics_history),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
            "avg_parallelization_efficiency": sum(parallelization_efficiencies) / len(parallelization_efficiencies),
            "uptime_seconds": time.time() - self.start_time
        }

class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.parallel_executor = ParallelExecutor()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules from configuration."""
        return {
            "cache_ttl": {
                "threat_intelligence": 1800,  # 30 minutes
                "policy_analysis": 3600,      # 1 hour
                "compliance_check": 7200,     # 2 hours
                "default": 3600
            },
            "parallelization_threshold": 3,  # Minimum tasks for parallelization
            "process_pool_threshold": 5,     # Use process pool for CPU-intensive tasks
            "memory_threshold_mb": 100       # Memory threshold for optimization
        }
    
    def optimize_function(self, cache_key: Optional[str] = None, ttl: Optional[int] = None):
        """Decorator to optimize function execution."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Generate cache key if not provided
                if cache_key is None:
                    key = self.cache_manager._generate_key(func.__name__, args, kwargs)
                else:
                    key = cache_key
                
                # Check cache first
                cached_result = self.cache_manager.get(key)
                if cached_result is not None:
                    # Record cache hit
                    self.performance_monitor.record_metrics(PerformanceMetrics(
                        execution_time=0.001,  # Cache hit is very fast
                        memory_usage=0,
                        cpu_usage=0,
                        cache_hit_rate=1.0,
                        parallelization_efficiency=1.0,
                        tool_usage_stats={"cache_hit": True}
                    ))
                    return cached_result
                
                # Execute function
                try:
                    result = await func(*args, **kwargs)
                    
                    # Cache result
                    cache_ttl = ttl or self.optimization_rules["cache_ttl"]["default"]
                    self.cache_manager.set(key, result, cache_ttl)
                    
                    # Record metrics
                    execution_time = time.time() - start_time
                    self.performance_monitor.record_metrics(PerformanceMetrics(
                        execution_time=execution_time,
                        memory_usage=0,  # Could add memory monitoring
                        cpu_usage=0,     # Could add CPU monitoring
                        cache_hit_rate=0.0,
                        parallelization_efficiency=1.0,
                        tool_usage_stats={"cache_hit": False, "execution_time": execution_time}
                    ))
                    
                    return result
                except Exception as e:
                    logger.error(f"Function execution failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    async def optimize_workflow(self, workflow_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize workflow for maximum performance."""
        optimized_steps = []
        
        # Group independent steps for parallelization
        dependency_groups = self._analyze_dependencies(workflow_steps)
        
        for group in dependency_groups:
            if len(group) > self.optimization_rules["parallelization_threshold"]:
                # Mark for parallel execution
                for step in group:
                    step["execution_mode"] = "parallel"
                    step["group_id"] = id(group)
                optimized_steps.extend(group)
            else:
                # Sequential execution
                for step in group:
                    step["execution_mode"] = "sequential"
                optimized_steps.extend(group)
        
        return optimized_steps
    
    def _analyze_dependencies(self, steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Analyze workflow steps for dependencies."""
        # Simple dependency analysis - can be enhanced
        dependency_groups = []
        current_group = []
        
        for step in steps:
            if not step.get("depends_on"):
                current_group.append(step)
            else:
                if current_group:
                    dependency_groups.append(current_group)
                current_group = [step]
        
        if current_group:
            dependency_groups.append(current_group)
        
        return dependency_groups
    
    async def execute_optimized_workflow(self, workflow_steps: List[Dict[str, Any]]) -> List[Any]:
        """Execute optimized workflow with parallelization."""
        results = []
        
        # Group steps by execution mode
        parallel_groups = {}
        sequential_steps = []
        
        for step in workflow_steps:
            if step["execution_mode"] == "parallel":
                group_id = step["group_id"]
                if group_id not in parallel_groups:
                    parallel_groups[group_id] = []
                parallel_groups[group_id].append(step)
            else:
                sequential_steps.append(step)
        
        # Execute parallel groups
        for group_id, group_steps in parallel_groups.items():
            group_results = await self.parallel_executor.execute_parallel(
                [step["function"] for step in group_steps]
            )
            results.extend(group_results)
        
        # Execute sequential steps
        for step in sequential_steps:
            result = await step["function"]()
            results.append(result)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        cache_stats = {
            "cache_size": len(self.cache_manager.cache),
            "cache_size_mb": sum(entry.size_bytes for entry in self.cache_manager.cache.values()) / (1024 * 1024),
            "max_cache_size_mb": self.cache_manager.max_size_bytes / (1024 * 1024)
        }
        
        performance_stats = self.performance_monitor.get_performance_summary()
        
        return {
            "cache": cache_stats,
            "performance": performance_stats,
            "parallelization": {
                "max_workers": self.parallel_executor.max_workers,
                "active_threads": len(self.parallel_executor.thread_pool._threads),
                "active_processes": len(self.parallel_executor.process_pool._processes)
            }
        }

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience decorators
def optimize(cache_key: Optional[str] = None, ttl: Optional[int] = None):
    """Convenience decorator for function optimization."""
    return performance_optimizer.optimize_function(cache_key, ttl)

def parallel_execute(tasks: List[Callable], use_processes: bool = False):
    """Convenience function for parallel execution."""
    return performance_optimizer.parallel_executor.execute_parallel(tasks, use_processes)
