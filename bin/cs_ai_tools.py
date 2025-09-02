#!/usr/bin/env python3
"""
CS AI Tools - Comprehensive Data Management for Agent Workflows
Combines pandas dataframe and SQLite database management with enhanced NLP capabilities.

Neo4j Graph Database Features:
- In-memory mode enabled by default for immediate use
- Automatic fallback to memory if Neo4j Desktop is unavailable
- No manual setup or password configuration required
- Full graph database functionality without external dependencies

MCP Server Features:
- Self-describing tools for dynamic agent discovery
- Lazy loading of tool managers
- Standardized tool interface and schemas
- Dynamic tool registration and execution
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import uuid
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from .openai_llm_client import OpenAILLMClient, LLMConfig, ModelType, ResponseFormat
from dataclasses import dataclass, asdict
from enum import Enum
import os
import re
import subprocess
import zipfile
import tarfile
import gzip
import bz2
import lzma
import tempfile
import random
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Protocol Constants
class MCPMessageType(Enum):
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMISES = "promises/list"
    CALL_PROMISE = "promises/call"

@dataclass
class MCPTool:
    """MCP Tool definition with full schema."""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    handler: Callable
    category: str = "general"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class MCPServer:
    """In-memory MCP server for dynamic tool discovery and execution."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Any] = {}
        self.promises: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.tool_manager = None  # Will be set when available
        self.session_logger = None  # Will be set when available
        
        # OpenAI configuration
        self.llm_client = OpenAILLMClient()
        self.openai_configured = self.llm_client.is_available()
        
        # Register built-in tools
        self._register_builtin_tools()
    

    
    def set_tool_manager(self, tool_manager):
        """Set the tool manager for dynamic tool discovery."""
        self.tool_manager = tool_manager
        self.logger.info("Tool manager connected to MCP server")
    
    def set_session_logger(self, session_logger):
        """Set the session logger for comprehensive logging."""
        self.session_logger = session_logger
        self.logger.info("Session logger connected to MCP server")
    
    def register_tool(self, tool: MCPTool):
        """Register a tool with the MCP server."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered MCP tool: {tool.name}")
        
        # Log tool registration
        if self.session_logger:
            self.session_logger.log_info("tool_registered", f"Tool '{tool.name}' registered", metadata={
                "tool_name": tool.name,
                "category": tool.category,
                "tags": tool.tags
            })
    
    def discover_available_tools(self, force=False):
        """Dynamically discover and register all available tools."""
        if not self.tool_manager:
            return
        
        # Log discovery start
        if self.session_logger:
            self.session_logger.log_info("tool_discovery_started", "Starting dynamic tool discovery")
        
        # Track which tool categories have been registered
        if not hasattr(self, '_registered_categories'):
            self._registered_categories = set()
        
        # Discover DataFrame tools (only if not already registered)
        if (hasattr(self.tool_manager, 'df_manager') and 
            self.tool_manager.df_manager and 
            ('dataframe' not in self._registered_categories or force)):
            self._register_dataframe_tools(self.tool_manager.df_manager)
            self._registered_categories.add('dataframe')
        
        # Discover SQLite tools (only if not already registered)
        if (hasattr(self.tool_manager, 'sqlite_manager') and 
            self.tool_manager.sqlite_manager and 
            ('sqlite' not in self._registered_categories or force)):
            # Check if SQLite manager has required methods
            if hasattr(self.tool_manager.sqlite_manager, 'create_database'):
                self._register_sqlite_tools(self.tool_manager.sqlite_manager)
                self._registered_categories.add('sqlite')
        
        # Discover Neo4j tools (only if not already registered)
        if (hasattr(self.tool_manager, 'neo4j_manager') and 
            self.tool_manager.neo4j_manager and 
            ('neo4j' not in self._registered_categories or force)):
            # Check if Neo4j manager has required methods
            if hasattr(self.tool_manager.neo4j_manager, 'create_graph'):
                self._register_neo4j_tools(self.tool_manager.neo4j_manager)
                self._registered_categories.add('neo4j')
        
        # Discover File tools (only if not already registered)
        if (hasattr(self.tool_manager, 'file_tools') and 
            self.tool_manager.file_tools and 
            ('file' not in self._registered_categories or force)):
            # Check if File tools have required methods
            if hasattr(self.tool_manager.file_tools, 'convert_file'):
                self._register_file_tools(self.tool_manager.file_tools)
                self._registered_categories.add('file')
        
        # Discover Compression tools (only if not already registered)
        if (hasattr(self.tool_manager, 'compression_tools') and 
            self.tool_manager.compression_tools and 
            ('compression' not in self._registered_categories or force)):
            # Check if Compression tools have required methods
            if hasattr(self.tool_manager.compression_tools, 'extract_archive'):
                self._register_compression_tools(self.tool_manager.compression_tools)
                self._registered_categories.add('compression')
        
        # Discover ML tools (only if not already registered)
        if (hasattr(self.tool_manager, 'ml_tools') and 
            self.tool_manager.ml_tools and 
            ('ml' not in self._registered_categories or force)):
            self._register_ml_tools(self.tool_manager.ml_tools)
            self._registered_categories.add('ml')
        
        # Discover NLP tools (only if not already registered)
        if (hasattr(self.tool_manager, 'nlp_tools') and 
            self.tool_manager.nlp_tools and 
            ('nlp' not in self._registered_categories or force)):
            self._register_nlp_tools(self.tool_manager.nlp_tools)
            self._registered_categories.add('nlp')
        
        # Discover Context Memory tools (only if not already registered)
        if (hasattr(self.tool_manager, 'context_memory') and 
            self.tool_manager.context_memory and 
            ('context_memory' not in self._registered_categories or force)):
            self._register_context_memory_tools(self.tool_manager.context_memory)
            self._registered_categories.add('context_memory')
        
        # Discover Security tools (only if not already registered)
        if (hasattr(self.tool_manager, 'security_tools') and 
            self.tool_manager.security_tools and 
            ('security' not in self._registered_categories or force)):
            self._register_security_tools(self.tool_manager.security_tools)
            self._registered_categories.add('security')
        
        # Discover Cryptography Evaluation tools (only if not already registered)
        if (hasattr(self.tool_manager, 'cryptography_evaluation_tools') and 
            self.tool_manager.cryptography_evaluation_tools and 
            ('cryptography_evaluation' not in self._registered_categories or force)):
            self._register_cryptography_evaluation_tools(self.tool_manager.cryptography_evaluation_tools)
            self._registered_categories.add('cryptography_evaluation')
        
        # Log discovery completion
        if self.session_logger:
            self.session_logger.log_info("tool_discovery_completed", f"Tool discovery completed. Total tools: {len(self.tools)}")
    
    def get_dynamic_tools(self, force_discovery=False):
        """Get tools with optional dynamic discovery."""
        if force_discovery:
            self.discover_available_tools(force=True)
        return self._list_tools_handler()
    
    def clear_tool_registrations(self):
        """Clear all tool registrations and reset discovery cache."""
        self.tools.clear()
        if hasattr(self, '_registered_categories'):
            self._registered_categories.clear()
        if self.session_logger:
            self.session_logger.log_info("tool_registrations_cleared", "All tool registrations cleared")
    
    def _register_compression_tools(self, compression_tools):
        """Register compression and archive MCP tools."""
        compression_tools_list = [
            MCPTool(
                name="extract_archive",
                description="Extract compressed archives (ZIP, TAR, GZ, BZ2, XZ)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "archive_path": {"type": "string", "description": "Path to archive file"},
                        "extract_to": {"type": "string", "description": "Extraction directory"},
                        "password": {"type": "string", "description": "Archive password if needed"},
                        "overwrite": {"type": "boolean", "description": "Overwrite existing files", "default": False}
                    },
                    "required": ["archive_path"]
                },
                handler=compression_tools.extract_archive,
                category="compression",
                tags=["extraction", "archive", "decompression", "security", "forensics"]
            ),
            
            MCPTool(
                name="create_archive",
                description="Create compressed archives with various formats",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "description": "Source file or directory"},
                        "archive_path": {"type": "string", "description": "Output archive path"},
                        "archive_type": {"type": "string", "enum": ["zip", "tar", "tar.gz", "tar.bz2", "tar.xz"], "description": "Archive format"},
                        "compression_level": {"type": "integer", "minimum": 0, "maximum": 9, "description": "Compression level"},
                        "options": {"type": "object", "description": "Additional archive options"}
                    },
                    "required": ["source_path", "archive_path"]
                },
                handler=compression_tools.create_archive,
                category="compression",
                tags=["compression", "archive", "creation", "backup", "storage"]
            ),
            
            MCPTool(
                name="list_archive_contents",
                description="List contents of compressed archives without extraction",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "archive_path": {"type": "string", "description": "Path to archive file"},
                        "detailed": {"type": "boolean", "description": "Include file details", "default": False}
                    },
                    "required": ["archive_path"]
                },
                handler=compression_tools.list_archive_contents,
                category="compression",
                tags=["inspection", "archive", "contents", "analysis"]
            )
        ]
        
        for tool in compression_tools_list:
            self.register_tool(tool)
    
    def _register_ml_tools(self, ml_tools):
        """Register machine learning MCP tools."""
        ml_tools_list = [
            MCPTool(
                name="detect_anomalies_isolation_forest",
                description="Detect anomalies using Isolation Forest algorithm for cybersecurity data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for anomaly detection"},
                        "contamination": {"type": "number", "description": "Expected proportion of anomalies", "default": 0.1},
                        "random_state": {"type": "integer", "description": "Random seed for reproducibility", "default": 42}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_anomalies_isolation_forest,
                category="machine_learning",
                tags=["anomaly_detection", "isolation_forest", "cybersecurity", "outlier_detection", "ml"]
            ),
            
            MCPTool(
                name="detect_anomalies_lof",
                description="Detect anomalies using Local Outlier Factor algorithm",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for anomaly detection"},
                        "n_neighbors": {"type": "integer", "description": "Number of neighbors for LOF", "default": 20},
                        "contamination": {"type": "number", "description": "Expected proportion of anomalies", "default": 0.1}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_anomalies_lof,
                category="machine_learning",
                tags=["anomaly_detection", "lof", "local_outlier_factor", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="cluster_data_kmeans",
                description="Cluster data using K-Means algorithm for pattern recognition",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for clustering"},
                        "n_clusters": {"type": "integer", "description": "Number of clusters", "default": 3},
                        "random_state": {"type": "integer", "description": "Random seed for reproducibility", "default": 42}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.cluster_data_kmeans,
                category="machine_learning",
                tags=["clustering", "kmeans", "pattern_recognition", "data_analysis", "ml"]
            ),
            
            MCPTool(
                name="find_optimal_clusters",
                description="Find optimal number of clusters using elbow method and silhouette analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for cluster analysis"},
                        "max_clusters": {"type": "integer", "description": "Maximum number of clusters to test", "default": 10},
                        "random_state": {"type": "integer", "description": "Random seed for reproducibility", "default": 42}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.find_optimal_clusters,
                category="machine_learning",
                tags=["clustering", "optimization", "elbow_method", "silhouette_analysis", "ml"]
            ),
            
            MCPTool(
                name="extract_features_statistical",
                description="Extract comprehensive statistical features from cybersecurity data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for feature extraction"}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.extract_features_statistical,
                category="machine_learning",
                tags=["feature_engineering", "statistics", "data_analysis", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="detect_patterns_correlation",
                description="Detect correlation patterns in cybersecurity datasets",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for correlation analysis"},
                        "threshold": {"type": "number", "description": "Correlation threshold for pattern detection", "default": 0.7}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_patterns_correlation,
                category="machine_learning",
                tags=["correlation_analysis", "pattern_detection", "data_analysis", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="detect_outliers_zscore",
                description="Detect outliers using Z-score method for cybersecurity data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for outlier detection"},
                        "threshold": {"type": "number", "description": "Z-score threshold for outlier detection", "default": 3.0}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_outliers_zscore,
                category="machine_learning",
                tags=["outlier_detection", "zscore", "statistics", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="detect_outliers_iqr",
                description="Detect outliers using Interquartile Range (IQR) method",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for outlier detection"},
                        "multiplier": {"type": "number", "description": "IQR multiplier for outlier detection", "default": 1.5}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_outliers_iqr,
                category="machine_learning",
                tags=["outlier_detection", "iqr", "statistics", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="analyze_data_distribution",
                description="Analyze data distribution and identify patterns in cybersecurity data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input data for distribution analysis"},
                        "bins": {"type": "integer", "description": "Number of histogram bins", "default": 10}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.analyze_data_distribution,
                category="machine_learning",
                tags=["distribution_analysis", "histogram", "statistics", "cybersecurity", "ml"]
            ),
            
            MCPTool(
                name="detect_change_points",
                description="Detect change points in time series cybersecurity data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Input time series data"},
                        "window_size": {"type": "integer", "description": "Window size for change detection", "default": 10},
                        "threshold": {"type": "number", "description": "Change detection threshold", "default": 2.0}
                    },
                    "required": ["data"]
                },
                handler=ml_tools.detect_change_points,
                category="machine_learning",
                tags=["change_detection", "time_series", "anomaly_detection", "cybersecurity", "ml"]
            )
        ]
        
        for tool in ml_tools_list:
            self.register_tool(tool)
    
    def _register_nlp_tools(self, nlp_tools):
        """Register local NLP MCP tools."""
        nlp_tools_list = [
            MCPTool(
                name="extract_text_features",
                description="Extract comprehensive text features for cybersecurity analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Input text for analysis"},
                        "include_advanced": {"type": "boolean", "description": "Include advanced NLP features", "default": True}
                    },
                    "required": ["text"]
                },
                handler=nlp_tools.extract_text_features,
                category="nlp",
                tags=["text_analysis", "feature_extraction", "cybersecurity", "nlp"]
            ),
            
            MCPTool(
                name="classify_text_naive_bayes",
                description="Classify text using Naive Bayes with automatic training",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to classify"},
                        "training_data": {"type": "array", "items": {"type": "string"}, "description": "Training text samples"},
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "Training labels"},
                        "test_size": {"type": "number", "description": "Test set proportion", "default": 0.2}
                    },
                    "required": ["text", "training_data", "labels"]
                },
                handler=nlp_tools.classify_text_naive_bayes,
                category="nlp",
                tags=["classification", "naive_bayes", "machine_learning", "nlp"]
            ),
            
            MCPTool(
                name="extract_text_embeddings",
                description="Extract text embeddings using sentence-transformers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {"type": "array", "items": {"type": "string"}, "description": "Texts to embed"},
                        "model_name": {"type": "string", "description": "Embedding model name", "default": "all-MiniLM-L6-v2"}
                    },
                    "required": ["texts"]
                },
                handler=nlp_tools.extract_text_embeddings,
                category="nlp",
                tags=["embeddings", "sentence_transformers", "semantic_analysis", "nlp"]
            ),
            
            MCPTool(
                name="calculate_text_similarity",
                description="Calculate semantic similarity between two texts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text1": {"type": "string", "description": "First text"},
                        "text2": {"type": "string", "description": "Second text"},
                        "model_name": {"type": "string", "description": "Embedding model name", "default": "all-MiniLM-L6-v2"}
                    },
                    "required": ["text1", "text2"]
                },
                handler=nlp_tools.calculate_text_similarity,
                category="nlp",
                tags=["similarity", "semantic_analysis", "text_comparison", "nlp"]
            ),
            
            MCPTool(
                name="cluster_texts",
                description="Cluster texts based on semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {"type": "array", "items": {"type": "string"}, "description": "Texts to cluster"},
                        "n_clusters": {"type": "integer", "description": "Number of clusters", "default": 3},
                        "model_name": {"type": "string", "description": "Embedding model name", "default": "all-MiniLM-L6-v2"}
                    },
                    "required": ["texts"]
                },
                handler=nlp_tools.cluster_texts,
                category="nlp",
                tags=["clustering", "semantic_analysis", "text_grouping", "nlp"]
            ),
            
            MCPTool(
                name="analyze_text_sentiment",
                description="Analyze text sentiment using lexicon-based analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                        "method": {"type": "string", "enum": ["lexicon", "ml"], "description": "Sentiment analysis method", "default": "lexicon"}
                    },
                    "required": ["text"]
                },
                handler=nlp_tools.analyze_text_sentiment,
                category="nlp",
                tags=["sentiment_analysis", "lexicon", "text_analysis", "nlp"]
            ),
            
            MCPTool(
                name="extract_keywords",
                description="Extract keywords from text using TF-IDF or frequency analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                        "top_k": {"type": "integer", "description": "Number of top keywords", "default": 10},
                        "method": {"type": "string", "enum": ["tfidf", "frequency"], "description": "Keyword extraction method", "default": "tfidf"}
                    },
                    "required": ["text"]
                },
                handler=nlp_tools.extract_keywords,
                category="nlp",
                tags=["keyword_extraction", "tfidf", "frequency_analysis", "nlp"]
            ),
            
            MCPTool(
                name="preprocess_text",
                description="Apply various text preprocessing operations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to preprocess"},
                        "operations": {"type": "array", "items": {"type": "string"}, "description": "Preprocessing operations to apply"}
                    },
                    "required": ["text"]
                },
                handler=nlp_tools.preprocess_text,
                category="nlp",
                tags=["text_preprocessing", "cleaning", "normalization", "nlp"]
            )
        ]
        
        for tool in nlp_tools_list:
            self.register_tool(tool)
    
    def _register_context_memory_tools(self, context_memory):
        """Register context memory MCP tools."""
        context_memory_tools_list = [
            MCPTool(
                name="start_memory_session",
                description="Start a new memory session for short-term memory management",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Unique session identifier"}
                    },
                    "required": ["session_id"]
                },
                handler=context_memory.start_session,
                category="context_memory",
                tags=["session_management", "memory", "workflow"]
            ),
            
            MCPTool(
                name="end_memory_session",
                description="End current memory session and cleanup short-term memory",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=context_memory.end_session,
                category="context_memory",
                tags=["session_management", "memory", "cleanup"]
            ),
            
            MCPTool(
                name="add_memory",
                description="Add a new memory node to the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Unique node identifier"},
                        "node_type": {"type": "string", "description": "Type of memory node (policy, control, network, device, user, application)"},
                        "content": {"type": "string", "description": "Memory content or description"},
                        "ttl_category": {"type": "string", "enum": ["short_term", "medium_term", "long_term"], "description": "Memory TTL category", "default": "short_term"},
                        "metadata": {"type": "object", "description": "Additional metadata for the memory"},
                        "importance_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Importance score (0.0 to 1.0)", "default": 0.5},
                        "relationships": {"type": "array", "items": {"type": "object"}, "description": "List of relationships to other nodes"}
                    },
                    "required": ["node_id", "node_type", "content"]
                },
                handler=context_memory.add_memory,
                category="context_memory",
                tags=["memory_creation", "knowledge_graph", "workflow"]
            ),
            
            MCPTool(
                name="get_memory",
                description="Retrieve a memory node by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Memory node identifier"},
                        "include_relationships": {"type": "boolean", "description": "Include relationship information", "default": True}
                    },
                    "required": ["node_id"]
                },
                handler=context_memory.get_memory,
                category="context_memory",
                tags=["memory_retrieval", "knowledge_graph", "workflow"]
            ),
            
            MCPTool(
                name="search_memories",
                description="Search memories using semantic similarity and filters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "node_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by node types"},
                        "ttl_categories": {"type": "array", "items": {"type": "string"}, "description": "Filter by TTL categories"},
                        "min_importance": {"type": "number", "minimum": 0, "maximum": 1, "description": "Minimum importance score", "default": 0.0},
                        "max_results": {"type": "integer", "description": "Maximum number of results", "default": 50}
                    },
                    "required": ["query"]
                },
                handler=context_memory.search_memories,
                category="context_memory",
                tags=["memory_search", "semantic_search", "knowledge_graph"]
            ),
            
            MCPTool(
                name="get_workflow_context",
                description="Get relevant context nodes for a specific workflow or task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_context": {"type": "string", "description": "Workflow or task context description"},
                        "max_nodes": {"type": "integer", "description": "Maximum number of context nodes", "default": 20}
                    },
                    "required": ["workflow_context"]
                },
                handler=context_memory.get_context_for_workflow,
                category="context_memory",
                tags=["workflow_context", "context_loading", "task_support"]
            ),
            
            MCPTool(
                name="suggest_memory_promotion",
                description="Suggest which workflow outputs should be promoted to longer-term memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_output": {"type": "object", "description": "Output from workflow execution"},
                        "context_used": {"type": "object", "description": "Context that was used in the workflow"}
                    },
                    "required": ["workflow_output", "context_used"]
                },
                handler=context_memory.suggest_memory_promotion,
                category="context_memory",
                tags=["memory_promotion", "workflow_analysis", "long_term_storage"]
            ),
            
            MCPTool(
                name="promote_memory",
                description="Promote a memory node to a longer TTL category",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Memory node identifier"},
                        "new_ttl_category": {"type": "string", "enum": ["medium_term", "long_term"], "description": "New TTL category"},
                        "reason": {"type": "string", "description": "Reason for promotion"}
                    },
                    "required": ["node_id", "new_ttl_category"]
                },
                handler=context_memory.promote_memory,
                category="context_memory",
                tags=["memory_promotion", "ttl_management", "knowledge_graph"]
            ),
            
            MCPTool(
                name="get_memory_stats",
                description="Get comprehensive memory statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=context_memory.get_memory_stats,
                category="context_memory",
                tags=["memory_analytics", "system_status", "monitoring"]
            ),
            
            MCPTool(
                name="cleanup_expired_memories",
                description="Clean up expired memories from all categories",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=context_memory.cleanup_expired_memories,
                category="context_memory",
                tags=["memory_cleanup", "maintenance", "system_health"]
            )
        ]
        
        for tool in context_memory_tools_list:
            self.register_tool(tool)
    
    def _register_security_tools(self, security_tools):
        """Register security and forensics MCP tools."""
        security_tools_list = [
            MCPTool(
                name="quick_host_scan",
                description="Perform a quick port scan on target hosts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "targets": {"type": "array", "items": {"type": "string"}, "description": "List of target IP addresses or hostnames"},
                        "custom_options": {"type": "object", "description": "Additional scan options"}
                    },
                    "required": ["targets"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["host_scanning", "port_scanning", "network_analysis", "security_assessment"]
            ),
            
            MCPTool(
                name="security_assessment_scan",
                description="Comprehensive security assessment with vulnerability detection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "targets": {"type": "array", "items": {"type": "string"}, "description": "List of target IP addresses or hostnames"},
                        "intensity": {"type": "string", "enum": ["polite", "normal", "aggressive"], "description": "Scan intensity level"}
                    },
                    "required": ["targets"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["vulnerability_scanning", "security_assessment", "os_detection", "comprehensive_scan"]
            ),
            
            MCPTool(
                name="network_discovery_scan",
                description="Network topology discovery and mapping",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "targets": {"type": "array", "items": {"type": "string"}, "description": "List of target IP addresses or hostnames"},
                        "include_ports": {"type": "boolean", "description": "Include port scanning in discovery"}
                    },
                    "required": ["targets"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["network_mapping", "topology_discovery", "network_analysis"]
            ),
            
            MCPTool(
                name="hash_string",
                description="Hash a string using specified algorithm",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to hash"},
                        "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"], "description": "Hash algorithm to use"}
                    },
                    "required": ["text"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["string_hashing", "cryptography", "data_integrity", "forensics"]
            ),
            
            MCPTool(
                name="hash_file",
                description="Hash a file using specified algorithm",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to hash"},
                        "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"], "description": "Hash algorithm to use"}
                    },
                    "required": ["file_path"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["file_hashing", "forensics", "data_integrity", "malware_analysis"]
            ),
            
            MCPTool(
                name="verify_hash",
                description="Verify hash integrity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "original_hash": {"type": "string", "description": "Original hash value"},
                        "computed_hash": {"type": "string", "description": "Computed hash value to verify"},
                        "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"], "description": "Hash algorithm used"}
                    },
                    "required": ["original_hash", "computed_hash"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["hash_verification", "data_integrity", "forensics", "compliance"]
            ),
            
            MCPTool(
                name="create_hmac",
                description="Create HMAC for data authentication",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Data to authenticate"},
                        "key": {"type": "string", "description": "Secret key for HMAC"},
                        "algorithm": {"type": "string", "enum": ["sha256", "sha512", "blake2b"], "description": "Hash algorithm to use"}
                    },
                    "required": ["data", "key"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["hmac_generation", "authentication", "message_integrity", "cryptography"]
            ),
            
            MCPTool(
                name="batch_hash_files",
                description="Hash multiple files efficiently",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_paths": {"type": "array", "items": {"type": "string"}, "description": "List of file paths to hash"},
                        "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512"], "description": "Hash algorithm to use"}
                    },
                    "required": ["file_paths"]
                },
                handler=security_tools.execute_tool,
                category="security",
                tags=["batch_processing", "file_hashing", "forensics", "efficiency"]
            )
        ]
        
        for tool in security_tools_list:
            self.register_tool(tool)
    
    def _register_cryptography_evaluation_tools(self, cryptography_evaluation_tools):
        """Register cryptography evaluation MCP tools."""
        cryptography_evaluation_tools_list = [
            MCPTool(
                name="evaluate_algorithm_security",
                description="Evaluate the security strength of cryptographic algorithms",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "algorithm": {"type": "string", "description": "Cryptographic algorithm name"},
                        "key_length": {"type": "integer", "description": "Key length in bits"},
                        "mode": {"type": "string", "description": "Encryption mode (if applicable)"}
                    },
                    "required": ["algorithm"]
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["algorithm", "security", "cryptography", "evaluation"]
            ),
            MCPTool(
                name="evaluate_implementation_security",
                description="Evaluate the security of cryptographic implementations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "implementation_data": {
                            "type": "object",
                            "description": "Implementation details including padding, IV generation, etc."
                        }
                    },
                    "required": ["implementation_data"]
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["implementation", "security", "cryptography", "evaluation"]
            ),
            MCPTool(
                name="evaluate_key_quality",
                description="Evaluate the quality and randomness of cryptographic keys",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_data": {"type": "string", "description": "Base64-encoded key data"},
                        "algorithm": {"type": "string", "description": "Algorithm the key is used with"}
                    },
                    "required": ["key_data", "algorithm"]
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["key", "quality", "randomness", "cryptography", "evaluation"]
            ),
            MCPTool(
                name="evaluate_randomness_quality",
                description="Evaluate the quality of random data and number generation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "random_data": {"type": "string", "description": "Base64-encoded random data"}
                    },
                    "required": ["random_data"]
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["randomness", "quality", "cryptography", "evaluation"]
            ),
            MCPTool(
                name="execute_evaluation_template",
                description="Execute predefined evaluation templates for comprehensive analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "template_name": {"type": "string", "enum": ["comprehensive_security", "algorithm_focus", "implementation_focus", "key_quality_focus", "quick_assessment"], "description": "Evaluation template to execute"},
                        "parameters": {"type": "object", "description": "Parameters for the evaluation"}
                    },
                    "required": ["template_name"]
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["template", "comprehensive", "security", "cryptography", "evaluation"]
            ),
            MCPTool(
                name="get_evaluation_statistics",
                description="Get performance statistics and analysis of evaluation patterns",
                inputSchema={
                    "type": "object",
                    "properties": {}
                },
                handler=cryptography_evaluation_tools.execute_tool,
                category="cryptography_evaluation",
                tags=["statistics", "analysis", "cryptography", "evaluation"]
            )
        ]
        
        for tool in cryptography_evaluation_tools_list:
            self.register_tool(tool)
    
    def _register_file_tools(self, file_tools):
        """Register file manipulation MCP tools."""
        file_tools_list = [
            MCPTool(
                name="convert_file",
                description="Convert files between different formats (JSON, CSV, XLSX, HTML, MD)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input_file": {"type": "string", "description": "Path to input file"},
                        "output_file": {"type": "string", "description": "Path for output file"},
                        "input_format": {"type": "string", "description": "Input file format (auto-detected if not specified)"},
                        "output_format": {"type": "string", "description": "Output file format"},
                        "options": {"type": "object", "description": "Conversion options"}
                    },
                    "required": ["input_file", "output_file"]
                },
                handler=file_tools.convert_file,
                category="file",
                tags=["conversion", "format", "transformation", "data", "reporting"]
            ),
            
            MCPTool(
                name="write_html_report",
                description="Generate professional HTML reports from data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to include in report"},
                        "output_file": {"type": "string", "description": "Output HTML file path"},
                        "template": {"type": "string", "description": "HTML template to use"},
                        "title": {"type": "string", "description": "Report title"},
                        "options": {"type": "object", "description": "Report generation options"}
                    },
                    "required": ["data", "output_file"]
                },
                handler=file_tools.write_html_report,
                category="file",
                tags=["report", "html", "generation", "presentation", "analysis"]
            ),
            
            MCPTool(
                name="write_markdown_report",
                description="Generate Markdown reports for documentation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to include in report"},
                        "output_file": {"type": "string", "description": "Output Markdown file path"},
                        "title": {"type": "string", "description": "Report title"},
                        "sections": {"type": "array", "items": {"type": "string"}, "description": "Report sections"},
                        "options": {"type": "object", "description": "Report generation options"}
                    },
                    "required": ["data", "output_file"]
                },
                handler=file_tools.write_markdown_report,
                category="file",
                tags=["report", "markdown", "documentation", "analysis"]
            )
        ]
        
        for tool in file_tools_list:
            self.register_tool(tool)
    
    def _register_neo4j_tools(self, neo4j_manager):
        """Register Neo4j-related MCP tools."""
        neo4j_tools = [
            MCPTool(
                name="create_node",
                description="Create a node in the graph database with labels and properties",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "Node labels"},
                        "properties": {"type": "object", "description": "Node properties"},
                        "options": {"type": "object", "description": "Additional creation options"}
                    },
                    "required": ["labels", "properties"]
                },
                handler=neo4j_manager.create_node,
                category="neo4j",
                tags=["graph", "node", "creation", "threat", "entity"]
            ),
            
            MCPTool(
                name="query_graph",
                description="Query the graph database using Cypher or natural language",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Cypher query or natural language query"},
                        "query_type": {"type": "string", "enum": ["cypher", "nlp", "pattern"], "description": "Type of query"},
                        "parameters": {"type": "object", "description": "Query parameters"},
                        "options": {"type": "object", "description": "Additional query options"}
                    },
                    "required": ["query"]
                },
                handler=neo4j_manager.query_graph,
                category="neo4j",
                tags=["graph", "query", "cypher", "nlp", "investigation"]
            ),
            
            MCPTool(
                name="get_graph_schema",
                description="Get comprehensive graph schema information for analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_samples": {"type": "boolean", "description": "Include sample data", "default": True},
                        "include_counts": {"type": "boolean", "description": "Include node/relationship counts", "default": True},
                        "options": {"type": "object", "description": "Additional schema options"}
                    }
                },
                handler=neo4j_manager.get_graph_schema,
                category="neo4j",
                tags=["graph", "schema", "metadata", "discovery", "analysis"]
            ),
            
            MCPTool(
                name="bulk_import_graph",
                description="Bulk import nodes and relationships from CSV files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nodes_file": {"type": "string", "description": "Path to nodes CSV file"},
                        "relationships_file": {"type": "string", "description": "Path to relationships CSV file"},
                        "options": {"type": "object", "description": "Import options and configuration"}
                    },
                    "required": ["nodes_file", "relationships_file"]
                },
                handler=neo4j_manager.bulk_import_graph,
                category="neo4j",
                tags=["graph", "import", "bulk", "csv", "data_ingestion"]
            )
        ]
        
        for tool in neo4j_tools:
            self.register_tool(tool)
    
    def _register_sqlite_tools(self, sqlite_manager):
        """Register SQLite-related MCP tools."""
        sqlite_tools = [
            MCPTool(
                name="create_database",
                description="Create a new SQLite database for data storage",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the database"},
                        "path": {"type": "string", "description": "File path for the database"},
                        "options": {"type": "object", "description": "Additional database options"}
                    },
                    "required": ["name"]
                },
                handler=sqlite_manager.create_database,
                category="sqlite",
                tags=["database", "creation", "storage", "data"]
            ),
            
            MCPTool(
                name="execute_sql",
                description="Execute SQL queries on SQLite databases",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Name of the database"},
                        "query": {"type": "string", "description": "SQL query to execute"},
                        "parameters": {"type": "array", "items": {"type": "string"}, "description": "Query parameters"},
                        "options": {"type": "object", "description": "Additional execution options"}
                    },
                    "required": ["database", "query"]
                },
                handler=sqlite_manager.execute_sql,
                category="sqlite",
                tags=["database", "query", "sql", "execution", "analysis"]
            ),
            
            MCPTool(
                name="get_database_schema",
                description="Get database schema information for analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Name of the database"},
                        "include_metadata": {"type": "boolean", "description": "Include detailed metadata", "default": False}
                    },
                    "required": ["database"]
                },
                handler=sqlite_manager.get_database_schema,
                category="sqlite",
                tags=["database", "schema", "metadata", "discovery"]
            )
        ]
        
        for tool in sqlite_tools:
            self.register_tool(tool)
    
    def _register_dataframe_tools(self, df_manager):
        """Register DataFrame-related MCP tools."""
        dataframe_tools = [
            MCPTool(
                name="create_dataframe",
                description="Create a new DataFrame from various data sources",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the DataFrame"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Column names"},
                        "data": {"type": "array", "description": "Optional initial data"},
                        "options": {"type": "object", "description": "Additional options"}
                    },
                    "required": ["name", "columns"]
                },
                handler=df_manager.create_dataframe,
                category="dataframe",
                tags=["data", "creation", "import"]
            ),
            
            MCPTool(
                name="list_dataframes",
                description="List all available DataFrames with metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_metadata": {"type": "boolean", "description": "Include detailed metadata", "default": False}
                    }
                },
                handler=df_manager.list_dataframes,
                category="dataframe",
                tags=["data", "discovery", "metadata"]
            ),
            
            MCPTool(
                name="query_dataframe",
                description="Query DataFrame using natural language or pandas query syntax",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the DataFrame to query"},
                        "query": {"type": "string", "description": "Natural language query or pandas query string"},
                        "query_type": {"type": "string", "enum": ["nlp", "pandas", "sql_like"], "description": "Type of query"},
                        "options": {"type": "object", "description": "Additional query options"}
                    },
                    "required": ["name", "query"]
                },
                handler=df_manager.query_dataframe,
                category="dataframe",
                tags=["data", "query", "nlp", "analysis"]
            )
        ]
        
        for tool in dataframe_tools:
            self.register_tool(tool)
    
    def _register_builtin_tools(self):
        """Register built-in MCP server tools."""
        builtin_tools = [
            MCPTool(
                name="list_tools",
                description="List all available MCP tools with descriptions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "Filter by tool category"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                        "detailed": {"type": "boolean", "description": "Include detailed tool information", "default": False}
                    }
                },
                handler=self._list_tools_handler,
                category="mcp",
                tags=["discovery", "metadata", "tools"]
            ),
            
            MCPTool(
                name="get_tool_schema",
                description="Get detailed schema for a specific tool",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "Name of the tool"}
                    },
                    "required": ["tool_name"]
                },
                handler=self._get_tool_schema_handler,
                category="mcp",
                tags=["discovery", "schema", "tools"]
            ),
            
            MCPTool(
                name="get_server_info",
                description="Get MCP server information and capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detailed": {"type": "boolean", "description": "Include detailed server information", "default": False}
                    }
                },
                handler=self._get_server_info_handler,
                category="mcp",
                tags=["server", "info", "capabilities"]
            ),
            
            MCPTool(
                name="suggest_workflow",
                description="Get workflow suggestions based on a specific goal",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Description of what you want to accomplish"},
                        "available_data": {"type": "array", "items": {"type": "string"}, "description": "List of available data sources"}
                    },
                    "required": ["goal"]
                },
                handler=self.suggest_workflow,
                category="mcp",
                tags=["workflow", "planning", "orchestration", "intelligence"]
            ),
            
            MCPTool(
                name="find_tools_by_capability",
                description="Find tools that can perform a specific capability",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "capability": {"type": "string", "description": "Description of the capability needed"}
                    },
                    "required": ["capability"]
                },
                handler=self.find_tools_by_capability,
                category="mcp",
                tags=["discovery", "capability", "intelligence", "search"]
            ),
            
            MCPTool(
                name="get_workflow_templates",
                description="Get predefined workflow templates for common cybersecurity tasks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_type": {"type": "string", "enum": ["threat_investigation", "data_analysis", "forensic_analysis"], "description": "Type of workflow template"}
                    }
                },
                handler=self.get_workflow_templates,
                category="mcp",
                tags=["workflow", "templates", "orchestration", "cybersecurity"]
            ),
            
            MCPTool(
                name="validate_workflow",
                description="Validate a proposed workflow for feasibility and completeness",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_steps": {"type": "array", "items": {"type": "object"}, "description": "List of workflow steps to validate"}
                    },
                    "required": ["workflow_steps"]
                },
                handler=self.validate_workflow,
                category="mcp",
                tags=["workflow", "validation", "quality", "orchestration"]
            ),
            
            MCPTool(
                name="log_agent_question",
                description="Log a question asked by an agent or user",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question being asked"},
                        "agent_type": {"type": "string", "description": "Type of agent (user, planner, runner, etc.)", "default": "user"},
                        "context": {"type": "object", "description": "Additional context for the question"},
                        "workflow_step": {"type": "string", "description": "Current workflow step if applicable"}
                    },
                    "required": ["question"]
                },
                handler=self._log_agent_question_handler,
                category="logging",
                tags=["logging", "agent", "question", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="log_agent_response",
                description="Log a response given by an agent",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "response": {"type": "string", "description": "The response given"},
                        "agent_type": {"type": "string", "description": "Type of agent (assistant, planner, runner, etc.)", "default": "assistant"},
                        "question_context": {"type": "string", "description": "Context of the question being answered"},
                        "workflow_step": {"type": "string", "description": "Current workflow step if applicable"}
                    },
                    "required": ["response"]
                },
                handler=self._log_agent_response_handler,
                category="logging",
                tags=["logging", "agent", "response", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="log_workflow_step",
                description="Log a workflow execution step",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "description": "Name or description of the workflow step"},
                        "action": {"type": "string", "description": "Action being performed"},
                        "details": {"type": "object", "description": "Detailed information about the step"},
                        "execution_id": {"type": "string", "description": "Unique execution identifier"},
                        "agent_type": {"type": "string", "description": "Type of agent executing the step"}
                    },
                    "required": ["step", "action"]
                },
                handler=self._log_workflow_step_handler,
                category="logging",
                tags=["logging", "workflow", "execution", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="log_decision_point",
                description="Log a decision point in the workflow",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "description": "Description of the decision being made"},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Available options"},
                        "selected_option": {"type": "string", "description": "The option that was selected"},
                        "reasoning": {"type": "string", "description": "Reasoning behind the selection"},
                        "agent_type": {"type": "string", "description": "Type of agent making the decision"}
                    },
                    "required": ["decision", "options", "selected_option"]
                },
                handler=self._log_decision_point_handler,
                category="logging",
                tags=["logging", "decision", "reasoning", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="get_session_summary",
                description="Get a summary of the current session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detailed": {"type": "boolean", "description": "Include detailed session information", "default": False}
                    }
                },
                handler=self._get_session_summary_handler,
                category="logging",
                tags=["logging", "session", "summary", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="end_session",
                description="End the current session and save the final log",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Summary of the session"}
                    }
                },
                handler=self._end_session_handler,
                category="logging",
                tags=["logging", "session", "completion", "debugging", "forensics"]
            ),
            
            MCPTool(
                name="generate_ascii_art",
                description="Generate ASCII art for welcome/exit messages and general flair",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to convert to ASCII art"},
                        "style": {"type": "string", "enum": ["cyber", "matrix", "hacker", "simple", "block"], "description": "ASCII art style to use", "default": "cyber"},
                        "width": {"type": "integer", "description": "Maximum width of the art", "default": 80},
                        "color": {"type": "string", "enum": ["none", "red", "green", "blue", "cyan", "yellow", "magenta"], "description": "Color theme for the art", "default": "none"}
                    },
                    "required": ["text"]
                },
                handler=self._generate_ascii_art_handler,
                category="presentation",
                tags=["ascii_art", "presentation", "style", "cybersecurity", "cli"]
            ),
            
            MCPTool(
                name="get_cyber_message",
                description="Get a random cyber-themed welcome or exit message",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message_type": {"type": "string", "enum": ["welcome", "exit", "random"], "description": "Type of message to get", "default": "random"},
                        "tone": {"type": "string", "enum": ["professional", "hacker", "threatening", "inspirational", "random"], "description": "Tone of the message", "default": "random"},
                        "include_ascii": {"type": "boolean", "description": "Include ASCII art with the message", "default": False},
                        "ascii_style": {"type": "string", "enum": ["cyber", "matrix", "hacker", "simple", "block"], "description": "ASCII art style if included", "default": "cyber"}
                    }
                },
                handler=self._get_cyber_message_handler,
                category="presentation",
                tags=["cyber_messages", "presentation", "style", "cybersecurity", "cli"]
            ),
            
            MCPTool(
                name="ai_reasoning",
                description="Use OpenAI to perform complex reasoning and analysis on data or problems",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The reasoning prompt or question"},
                        "context": {"type": "string", "description": "Additional context or data to consider"},
                        "reasoning_type": {"type": "string", "enum": ["threat_analysis", "risk_assessment", "pattern_recognition", "causal_analysis", "hypothesis_generation"], "description": "Type of reasoning to perform", "default": "threat_analysis"},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens for response", "default": 1000},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level (0=deterministic, 2=creative)", "default": 0.3}
                    },
                    "required": ["prompt"]
                },
                handler=self._ai_reasoning_handler,
                category="ai_integration",
                tags=["openai", "reasoning", "analysis", "intelligence", "threat_analysis"]
            ),
            
            MCPTool(
                name="ai_categorization",
                description="Use OpenAI to categorize and classify data, threats, or information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Data to categorize"},
                        "categories": {"type": "array", "items": {"type": "string"}, "description": "Available categories for classification"},
                        "categorization_type": {"type": "string", "enum": ["threat_classification", "data_classification", "severity_assessment", "priority_ranking", "custom_classification"], "description": "Type of categorization", "default": "threat_classification"},
                        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "description": "Minimum confidence for classification", "default": 0.7},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "include_reasoning": {"type": "boolean", "description": "Include reasoning for classification", "default": True}
                    },
                    "required": ["data", "categories"]
                },
                handler=self._ai_categorization_handler,
                category="ai_integration",
                tags=["openai", "categorization", "classification", "threat_intelligence", "data_analysis"]
            ),
            
            MCPTool(
                name="ai_summarization",
                description="Use OpenAI to summarize large amounts of data, logs, or information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to summarize"},
                        "summary_type": {"type": "string", "enum": ["executive_summary", "technical_summary", "threat_summary", "incident_summary", "custom_summary"], "description": "Type of summary to generate", "default": "executive_summary"},
                        "max_length": {"type": "integer", "description": "Maximum length of summary", "default": 500},
                        "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Specific areas to focus on in summary"},
                        "include_key_points": {"type": "boolean", "description": "Include key points list", "default": True},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.2}
                    },
                    "required": ["content"]
                },
                handler=self._ai_summarization_handler,
                category="ai_integration",
                tags=["openai", "summarization", "content_analysis", "reporting", "data_compression"]
            ),
            
            MCPTool(
                name="ai_code_transformation",
                description="Use OpenAI to transform, refactor, or convert code and queries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code or query to transform"},
                        "transformation_type": {"type": "string", "enum": ["language_conversion", "query_optimization", "code_refactoring", "format_conversion", "security_enhancement"], "description": "Type of transformation to perform", "default": "language_conversion"},
                        "target_language": {"type": "string", "description": "Target programming language or format"},
                        "source_language": {"type": "string", "description": "Source programming language or format"},
                        "requirements": {"type": "string", "description": "Specific requirements or constraints"},
                        "include_explanation": {"type": "boolean", "description": "Include explanation of changes", "default": True},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.1}
                    },
                    "required": ["code", "transformation_type"]
                },
                handler=self._ai_code_transformation_handler,
                category="ai_integration",
                tags=["openai", "code_transformation", "query_optimization", "refactoring", "security"]
            ),
            
            MCPTool(
                name="ai_query_enhancement",
                description="Use OpenAI to enhance, optimize, or generate complex queries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original query or search request"},
                        "enhancement_type": {"type": "string", "enum": ["query_expansion", "query_optimization", "query_generation", "query_translation", "query_validation"], "description": "Type of enhancement to perform", "default": "query_optimization"},
                        "context": {"type": "string", "description": "Context about the data or system being queried"},
                        "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints or limitations to consider"},
                        "target_format": {"type": "string", "description": "Desired output format (SQL, Cypher, etc.)", "default": "natural_language"},
                        "include_alternatives": {"type": "boolean", "description": "Include alternative query suggestions", "default": False},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.3}
                    },
                    "required": ["query"]
                },
                handler=self._ai_query_enhancement_handler,
                category="ai_integration",
                tags=["openai", "query_enhancement", "search_optimization", "query_generation", "intelligence"]
            ),
            
            MCPTool(
                name="ai_threat_intelligence",
                description="Use OpenAI to analyze and generate threat intelligence insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threat_data": {"type": "string", "description": "Threat data or indicators to analyze"},
                        "analysis_type": {"type": "string", "enum": ["threat_assessment", "indicator_analysis", "attack_pattern_recognition", "threat_hunting", "vulnerability_assessment"], "description": "Type of threat analysis", "default": "threat_assessment"},
                        "threat_context": {"type": "string", "description": "Additional context about the threat environment"},
                        "include_mitigation": {"type": "boolean", "description": "Include mitigation strategies", "default": True},
                        "include_indicators": {"type": "boolean", "description": "Include threat indicators", "default": True},
                        "severity_level": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Threat severity level", "default": "medium"},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.2}
                    },
                    "required": ["threat_data"]
                },
                handler=self._ai_threat_intelligence_handler,
                category="ai_integration",
                tags=["openai", "threat_intelligence", "security_analysis", "threat_assessment", "cybersecurity"]
            ),
            
            MCPTool(
                name="ai_workflow_optimization",
                description="Use OpenAI to analyze and optimize workflow execution plans",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow": {"type": "object", "description": "Workflow definition to optimize"},
                        "optimization_goal": {"type": "string", "enum": ["performance", "efficiency", "security", "reliability", "cost"], "description": "Primary optimization goal", "default": "performance"},
                        "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints to consider during optimization"},
                        "include_alternatives": {"type": "boolean", "description": "Include alternative workflow suggestions", "default": True},
                        "include_metrics": {"type": "boolean", "description": "Include performance metrics", "default": True},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.3}
                    },
                    "required": ["workflow"]
                },
                handler=self._ai_workflow_optimization_handler,
                category="ai_integration",
                tags=["openai", "workflow_optimization", "performance", "efficiency", "planning"]
            ),
            
            MCPTool(
                name="ai_patent_analysis",
                description="Use OpenAI to analyze patents and generate value propositions and categorizations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patent_title": {"type": "string", "description": "Patent title"},
                        "patent_abstract": {"type": "string", "description": "Patent abstract"},
                        "analysis_type": {"type": "string", "enum": ["value_proposition", "categorization", "both"], "description": "Type of analysis to perform", "default": "both"},
                        "model": {"type": "string", "description": "OpenAI model to use", "default": "gpt-4"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "description": "Creativity level", "default": 0.3}
                    },
                    "required": ["patent_title", "patent_abstract"]
                },
                handler=self._ai_patent_analysis_handler,
                category="ai_integration",
                tags=["openai", "patent_analysis", "intellectual_property", "cybersecurity", "categorization"]
            )
        ]
        
        for tool in builtin_tools:
            self.register_tool(tool)
    
    def _list_tools_handler(self, **kwargs):
        """Handler for listing available tools."""
        category_filter = kwargs.get('category')
        tags_filter = kwargs.get('tags', [])
        detailed = kwargs.get('detailed', False)
        
        filtered_tools = {}
        
        for name, tool in self.tools.items():
            # Apply category filter
            if category_filter and tool.category != category_filter:
                continue
            
            # Apply tags filter
            if tags_filter and not any(tag in tool.tags for tag in tags_filter):
                continue
            
            if detailed:
                filtered_tools[name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                    "category": tool.category,
                    "tags": tool.tags
                }
            else:
                filtered_tools[name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "tags": tool.tags
                }
        
        return {
            "success": True,
            "tools": filtered_tools,
            "total_count": len(filtered_tools),
            "filters_applied": {
                "category": category_filter,
                "tags": tags_filter
            }
        }
    
    def _get_tool_schema_handler(self, **kwargs):
        """Handler for getting tool schema."""
        tool_name = kwargs.get('tool_name')
        
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        tool = self.tools[tool_name]
        return {
            "success": True,
            "tool": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "category": tool.category,
                "tags": tool.tags
            }
        }
    
    def _get_server_info_handler(self, **kwargs):
        """Handler for getting server information."""
        detailed = kwargs.get('detailed', False)
        
        info = {
            "server_name": "CS-AI MCP Server",
            "version": "1.0.0",
            "capabilities": ["tools", "resources", "promises", "workflows", "orchestration"],
            "tool_count": len(self.tools),
            "categories": list(set(tool.category for tool in self.tools.values())),
            "tags": list(set(tag for tool in self.tools.values() for tag in tool.tags))
        }
        
        if detailed:
            info.update({
                "tools": {name: {"category": tool.category, "tags": tool.tags} 
                          for name, tool in self.tools.items()},
                "resources": list(self.resources.keys()),
                "promises": list(self.promises.keys()),
                "workflow_capabilities": {
                    "tool_chaining": True,
                    "conditional_execution": True,
                    "parallel_execution": True,
                    "error_handling": True,
                    "workflow_persistence": True
                }
            })
        
        return {
            "success": True,
            "server_info": info
        }
    
    def suggest_workflow(self, goal: str, available_data: List[str] = None) -> Dict[str, Any]:
        """Suggest a workflow based on the goal and available data."""
        suggestions = []
        
        # Analyze goal and suggest relevant tools
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["extract", "unzip", "decompress"]):
            suggestions.append({
                "step": "extract_archive",
                "reason": "Goal involves extracting compressed files",
                "category": "compression",
                "priority": "high"
            })
        
        if any(word in goal_lower for word in ["convert", "transform", "format"]):
            suggestions.append({
                "step": "convert_file",
                "reason": "Goal involves file format conversion",
                "category": "file",
                "priority": "high"
            })
        
        if any(word in goal_lower for word in ["analyze", "query", "investigate"]):
            suggestions.append({
                "step": "query_dataframe",
                "reason": "Goal involves data analysis",
                "category": "dataframe",
                "priority": "medium"
            })
        
        if any(word in goal_lower for word in ["graph", "relationship", "network"]):
            suggestions.append({
                "step": "query_graph",
                "reason": "Goal involves graph analysis",
                "category": "neo4j",
                "priority": "medium"
            })
        
        if any(word in goal_lower for word in ["report", "document", "present"]):
            suggestions.append({
                "step": "write_html_report",
                "reason": "Goal involves report generation",
                "category": "file",
                "priority": "medium"
            })
        
        return {
            "success": True,
            "goal": goal,
            "suggested_workflow": suggestions,
            "total_steps": len(suggestions),
            "categories_involved": list(set(s["category"] for s in suggestions))
        }
    
    def find_tools_by_capability(self, capability: str) -> Dict[str, Any]:
        """Find tools that can perform a specific capability."""
        matching_tools = {}
        
        for name, tool in self.tools.items():
            # Check if capability is mentioned in description, tags, or category
            if (capability.lower() in tool.description.lower() or
                capability.lower() in tool.category.lower() or
                any(capability.lower() in tag.lower() for tag in tool.tags)):
                matching_tools[name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "tags": tool.tags,
                    "relevance_score": self._calculate_relevance_score(tool, capability)
                }
        
        # Sort by relevance score
        sorted_tools = dict(sorted(
            matching_tools.items(), 
            key=lambda x: x[1]["relevance_score"], 
            reverse=True
        ))
        
        return {
            "success": True,
            "capability": capability,
            "matching_tools": sorted_tools,
            "total_matches": len(sorted_tools)
        }
    
    def _calculate_relevance_score(self, tool: MCPTool, capability: str) -> float:
        """Calculate how relevant a tool is to a specific capability."""
        score = 0.0
        
        # Exact matches get higher scores
        if capability.lower() in tool.name.lower():
            score += 3.0
        if capability.lower() in tool.description.lower():
            score += 2.0
        if capability.lower() in tool.category.lower():
            score += 1.5
        if any(capability.lower() in tag.lower() for tag in tool.tags):
            score += 1.0
        
        # Bonus for security/forensics related tools
        if any(tag in ["security", "forensics", "threat", "investigation"] for tag in tool.tags):
            score += 0.5
        
        return score
    
    def get_workflow_templates(self, workflow_type: str = None) -> Dict[str, Any]:
        """Get predefined workflow templates for common cybersecurity tasks."""
        templates = {
            "threat_investigation": {
                "name": "Threat Investigation Workflow",
                "description": "Complete workflow for investigating security threats",
                "steps": [
                    {
                        "step": 1,
                        "tool": "extract_archive",
                        "description": "Extract suspicious archive files",
                        "category": "compression"
                    },
                    {
                        "step": 2,
                        "tool": "convert_file",
                        "description": "Convert logs to analyzable format",
                        "category": "file"
                    },
                    {
                        "step": 3,
                        "tool": "query_dataframe",
                        "description": "Analyze log data for anomalies",
                        "category": "dataframe"
                    },
                    {
                        "step": 4,
                        "tool": "create_node",
                        "description": "Create threat entities in graph",
                        "category": "neo4j"
                    },
                    {
                        "step": 5,
                        "tool": "write_html_report",
                        "description": "Generate investigation report",
                        "category": "file"
                    }
                ],
                "estimated_duration": "15-30 minutes",
                "complexity": "medium"
            },
            "data_analysis": {
                "name": "Data Analysis Workflow",
                "description": "Workflow for comprehensive data analysis",
                "steps": [
                    {
                        "step": 1,
                        "tool": "convert_file",
                        "description": "Convert data to analysis format",
                        "category": "file"
                    },
                    {
                        "step": 2,
                        "tool": "query_dataframe",
                        "description": "Perform data analysis",
                        "category": "dataframe"
                    },
                    {
                        "step": 3,
                        "tool": "write_markdown_report",
                        "description": "Document analysis results",
                        "category": "file"
                    }
                ],
                "estimated_duration": "10-20 minutes",
                "complexity": "low"
            },
            "forensic_analysis": {
                "name": "Forensic Analysis Workflow",
                "description": "Complete forensic analysis workflow",
                "steps": [
                    {
                        "step": 1,
                        "tool": "extract_archive",
                        "description": "Extract forensic images",
                        "category": "compression"
                    },
                    {
                        "step": 2,
                        "tool": "list_archive_contents",
                        "description": "Inventory extracted contents",
                        "category": "compression"
                    },
                    {
                        "step": 3,
                        "tool": "convert_file",
                        "description": "Convert evidence to analyzable format",
                        "category": "file"
                    },
                    {
                        "step": 4,
                        "tool": "query_dataframe",
                        "description": "Analyze evidence data",
                        "category": "dataframe"
                    },
                    {
                        "step": 5,
                        "tool": "create_node",
                        "description": "Build evidence graph",
                        "category": "neo4j"
                    },
                    {
                        "step": 6,
                        "tool": "write_html_report",
                        "description": "Generate forensic report",
                        "category": "file"
                    }
                ],
                "estimated_duration": "30-60 minutes",
                "complexity": "high"
            }
        }
        
        if workflow_type and workflow_type in templates:
            return {
                "success": True,
                "template": templates[workflow_type]
            }
        
        return {
            "success": True,
            "available_templates": list(templates.keys()),
            "templates": templates
        }
    
    def validate_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a proposed workflow for feasibility and completeness."""
        validation_results = []
        total_score = 0
        max_score = len(workflow_steps) * 10
        
        for i, step in enumerate(workflow_steps):
            step_score = 0
            issues = []
            
            # Check if tool exists
            tool_name = step.get("tool")
            if tool_name in self.tools:
                step_score += 5
            else:
                issues.append(f"Tool '{tool_name}' not found")
            
            # Check if required parameters are specified
            tool = self.tools.get(tool_name)
            if tool:
                required_params = tool.inputSchema.get("required", [])
                provided_params = step.get("parameters", {})
                
                for param in required_params:
                    if param in provided_params:
                        step_score += 3
                    else:
                        issues.append(f"Required parameter '{param}' missing")
                
                # Check for extra parameters
                extra_params = set(provided_params.keys()) - set(tool.inputSchema.get("properties", {}).keys())
                if extra_params:
                    issues.append(f"Unknown parameters: {list(extra_params)}")
                    step_score -= 1
            
            # Check step dependencies
            if i > 0:
                prev_step = workflow_steps[i-1]
                if prev_step.get("output") and step.get("input"):
                    step_score += 2
            
            validation_results.append({
                "step": i + 1,
                "tool": tool_name,
                "score": step_score,
                "issues": issues,
                "status": "valid" if step_score >= 7 else "needs_attention"
            })
            
            total_score += step_score
        
        overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        return {
            "success": True,
            "workflow_validation": {
                "overall_score": overall_score,
                "step_validations": validation_results,
                "total_steps": len(workflow_steps),
                "status": "ready" if overall_score >= 80 else "needs_review",
                "recommendations": self._generate_workflow_recommendations(validation_results)
            }
        }
    
    def _generate_workflow_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving workflow validation."""
        recommendations = []
        
        low_score_steps = [r for r in validation_results if r["score"] < 7]
        if low_score_steps:
            recommendations.append(f"Review {len(low_score_steps)} steps with low validation scores")
        
        missing_tools = [r for r in validation_results if "Tool" in str(r.get("issues", []))]
        if missing_tools:
            recommendations.append("Ensure all required tools are available")
        
        param_issues = [r for r in validation_results if any("parameter" in issue.lower() for issue in r.get("issues", []))]
        if param_issues:
            recommendations.append("Review parameter specifications for all steps")
        
        if not recommendations:
            recommendations.append("Workflow is well-structured and ready for execution")
        
        return recommendations
    
    def _log_agent_question_handler(self, **kwargs):
        """Handler for logging agent questions."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            self.session_logger.log_agent_question(
                question=kwargs.get("question"),
                agent_type=kwargs.get("agent_type", "user"),
                context=kwargs.get("context"),
                workflow_step=kwargs.get("workflow_step")
            )
            return {"success": True, "message": "Question logged successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to log question: {str(e)}"}
    
    def _log_agent_response_handler(self, **kwargs):
        """Handler for logging agent responses."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            self.session_logger.log_agent_response(
                response=kwargs.get("response"),
                agent_type=kwargs.get("agent_type", "assistant"),
                question_context=kwargs.get("question_context"),
                workflow_step=kwargs.get("workflow_step")
            )
            return {"success": True, "message": "Response logged successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to log response: {str(e)}"}
    
    def _log_workflow_step_handler(self, **kwargs):
        """Handler for logging workflow execution steps."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            self.session_logger.log_workflow_execution(
                step=kwargs.get("step"),
                action=kwargs.get("action"),
                details=kwargs.get("details", {}),
                execution_id=kwargs.get("execution_id"),
                agent_type=kwargs.get("agent_type")
            )
            return {"success": True, "message": "Workflow step logged successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to log workflow step: {str(e)}"}
    
    def _log_decision_point_handler(self, **kwargs):
        """Handler for logging decision points."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            self.session_logger.log_decision_point(
                decision=kwargs.get("decision"),
                options=kwargs.get("options", []),
                selected_option=kwargs.get("selected_option"),
                reasoning=kwargs.get("reasoning"),
                agent_type=kwargs.get("agent_type")
            )
            return {"success": True, "message": "Decision point logged successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to log decision point: {str(e)}"}
    
    def _get_session_summary_handler(self, **kwargs):
        """Handler for getting session summary."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            summary = self.session_logger.get_session_summary()
            return {"success": True, "session_summary": summary}
        except Exception as e:
            return {"success": False, "error": f"Failed to get session summary: {str(e)}"}
    
    def _end_session_handler(self, **kwargs):
        """Handler for ending the session."""
        if not hasattr(self, 'session_logger') or not self.session_logger:
            return {"success": False, "error": "Session logger not available"}
        
        try:
            summary = kwargs.get("summary", "Session ended by agent request")
            self.session_logger.end_session(summary)
            return {"success": True, "message": "Session ended successfully", "summary": summary}
        except Exception as e:
            return {"success": False, "error": f"Failed to end session: {str(e)}"}
    
    def _generate_ascii_art_handler(self, **kwargs):
        """Handler for generating ASCII art."""
        try:
            text = kwargs.get("text", "")
            style = kwargs.get("style", "cyber")
            width = kwargs.get("width", 80)
            color = kwargs.get("color", "none")
            
            if not text:
                return {"success": False, "error": "Text is required for ASCII art generation"}
            
            # Generate ASCII art based on style
            ascii_art = self._create_ascii_art(text, style, width)
            
            # Apply color if specified
            if color != "none":
                ascii_art = self._apply_color(ascii_art, color)
            
            return {
                "success": True,
                "ascii_art": ascii_art,
                "text": text,
                "style": style,
                "width": width,
                "color": color
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to generate ASCII art: {str(e)}"}
    
    def _get_cyber_message_handler(self, **kwargs):
        """Handler for getting cyber-themed messages."""
        try:
            message_type = kwargs.get("message_type", "random")
            tone = kwargs.get("tone", "random")
            include_ascii = kwargs.get("include_ascii", False)
            ascii_style = kwargs.get("ascii_style", "cyber")
            
            # Get the message
            message_data = self._get_random_cyber_message(message_type, tone)
            
            result = {
                "success": True,
                "message": message_data["message"],
                "message_type": message_data["type"],
                "tone": message_data["tone"],
                "author": message_data.get("author", "Unknown"),
                "ascii_art": None
            }
            
            # Generate ASCII art if requested
            if include_ascii:
                ascii_result = self._generate_ascii_art_handler(
                    text=message_data["message"],
                    style=ascii_style,
                    width=80,
                    color="none"
                )
                if ascii_result.get("success"):
                    result["ascii_art"] = ascii_result["ascii_art"]
                    result["ascii_style"] = ascii_style
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Failed to get cyber message: {str(e)}"}
    
    def _ai_reasoning_handler(self, **kwargs):
        """Handler for AI-powered reasoning and analysis."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            prompt = kwargs.get("prompt")
            context = kwargs.get("context", "")
            reasoning_type = kwargs.get("reasoning_type", "threat_analysis")
            model = kwargs.get("model", "gpt-4")
            max_tokens = kwargs.get("max_tokens", 1000)
            temperature = kwargs.get("temperature", 0.3)
            
            if not prompt:
                return {"success": False, "error": "Prompt is required for AI reasoning"}
            
            # Build the reasoning prompt
            system_prompt = self._get_reasoning_system_prompt(reasoning_type)
            user_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "reasoning": response.get("content"),
                    "reasoning_type": reasoning_type,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI reasoning failed: {str(e)}"}
    
    def _ai_categorization_handler(self, **kwargs):
        """Handler for AI-powered categorization and classification."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            data = kwargs.get("data")
            categories = kwargs.get("categories", [])
            categorization_type = kwargs.get("categorization_type", "threat_classification")
            confidence_threshold = kwargs.get("confidence_threshold", 0.7)
            model = kwargs.get("model", "gpt-4")
            include_reasoning = kwargs.get("include_reasoning", True)
            
            if not data or not categories:
                return {"success": False, "error": "Data and categories are required for categorization"}
            
            # Build the categorization prompt
            system_prompt = self._get_categorization_system_prompt(categorization_type)
            user_prompt = f"Data: {data}\n\nCategories: {', '.join(categories)}\n\nPlease categorize the data and provide confidence scores."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=800,
                temperature=0.2
            )
            
            if response.get("success"):
                # Parse the response for categorization results
                categorization_result = self._parse_categorization_response(
                    response.get("content"),
                    categories,
                    confidence_threshold,
                    include_reasoning
                )
                
                return {
                    "success": True,
                    "categorization": categorization_result,
                    "categorization_type": categorization_type,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI categorization failed: {str(e)}"}
    
    def _ai_summarization_handler(self, **kwargs):
        """Handler for AI-powered summarization."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            content = kwargs.get("content")
            summary_type = kwargs.get("summary_type", "executive_summary")
            max_length = kwargs.get("max_length", 500)
            focus_areas = kwargs.get("focus_areas", [])
            include_key_points = kwargs.get("include_key_points", True)
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.2)
            
            if not content:
                return {"success": False, "error": "Content is required for summarization"}
            
            # Build the summarization prompt
            system_prompt = self._get_summarization_system_prompt(summary_type)
            user_prompt = f"Content to summarize:\n\n{content}\n\n"
            
            if focus_areas:
                user_prompt += f"Focus areas: {', '.join(focus_areas)}\n\n"
            
            user_prompt += f"Generate a summary of maximum {max_length} characters"
            
            if include_key_points:
                user_prompt += " and include key points"
            
            user_prompt += "."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=max_length + 200,  # Allow some extra for formatting
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "summary": response.get("content"),
                    "summary_type": summary_type,
                    "max_length": max_length,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI summarization failed: {str(e)}"}
    
    def _ai_code_transformation_handler(self, **kwargs):
        """Handler for AI-powered code transformation."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            code = kwargs.get("code")
            transformation_type = kwargs.get("transformation_type", "language_conversion")
            target_language = kwargs.get("target_language", "")
            source_language = kwargs.get("source_language", "")
            requirements = kwargs.get("requirements", "")
            include_explanation = kwargs.get("include_explanation", True)
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.1)
            
            if not code:
                return {"success": False, "error": "Code is required for transformation"}
            
            # Build the transformation prompt
            system_prompt = self._get_code_transformation_system_prompt(transformation_type)
            user_prompt = f"Code to transform:\n\n```{source_language or 'text'}\n{code}\n```\n\n"
            
            if target_language:
                user_prompt += f"Target language: {target_language}\n\n"
            
            if requirements:
                user_prompt += f"Requirements: {requirements}\n\n"
            
            user_prompt += "Please transform the code"
            
            if include_explanation:
                user_prompt += " and include an explanation of the changes"
            
            user_prompt += "."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=1500,
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "transformed_code": response.get("content"),
                    "transformation_type": transformation_type,
                    "source_language": source_language,
                    "target_language": target_language,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI code transformation failed: {str(e)}"}
    
    def _ai_query_enhancement_handler(self, **kwargs):
        """Handler for AI-powered query enhancement."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            query = kwargs.get("query")
            enhancement_type = kwargs.get("enhancement_type", "query_optimization")
            context = kwargs.get("context", "")
            constraints = kwargs.get("constraints", [])
            target_format = kwargs.get("target_format", "natural_language")
            include_alternatives = kwargs.get("include_alternatives", False)
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.3)
            
            if not query:
                return {"success": False, "error": "Query is required for enhancement"}
            
            # Build the enhancement prompt
            system_prompt = self._get_query_enhancement_system_prompt(enhancement_type)
            user_prompt = f"Original query: {query}\n\n"
            
            if context:
                user_prompt += f"Context: {context}\n\n"
            
            if constraints:
                user_prompt += f"Constraints: {', '.join(constraints)}\n\n"
            
            user_prompt += f"Target format: {target_format}\n\n"
            user_prompt += "Please enhance the query"
            
            if include_alternatives:
                user_prompt += " and provide alternative versions"
            
            user_prompt += "."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=1000,
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "enhanced_query": response.get("content"),
                    "enhancement_type": enhancement_type,
                    "target_format": target_format,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI query enhancement failed: {str(e)}"}
    
    def _ai_threat_intelligence_handler(self, **kwargs):
        """Handler for AI-powered threat intelligence analysis."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            threat_data = kwargs.get("threat_data")
            analysis_type = kwargs.get("analysis_type", "threat_assessment")
            threat_context = kwargs.get("threat_context", "")
            include_mitigation = kwargs.get("include_mitigation", True)
            include_indicators = kwargs.get("include_indicators", True)
            severity_level = kwargs.get("severity_level", "medium")
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.2)
            
            if not threat_data:
                return {"success": False, "error": "Threat data is required for analysis"}
            
            # Build the threat analysis prompt
            system_prompt = self._get_threat_intelligence_system_prompt(analysis_type)
            user_prompt = f"Threat data: {threat_data}\n\n"
            
            if threat_context:
                user_prompt += f"Threat context: {threat_context}\n\n"
            
            user_prompt += f"Severity level: {severity_level}\n\n"
            user_prompt += "Please analyze this threat"
            
            if include_mitigation:
                user_prompt += ", include mitigation strategies"
            
            if include_indicators:
                user_prompt += ", and identify threat indicators"
            
            user_prompt += "."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=1200,
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "threat_analysis": response.get("content"),
                    "analysis_type": analysis_type,
                    "severity_level": severity_level,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI threat intelligence failed: {str(e)}"}
    
    def _ai_workflow_optimization_handler(self, **kwargs):
        """Handler for AI-powered workflow optimization."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            workflow = kwargs.get("workflow")
            optimization_goal = kwargs.get("optimization_goal", "performance")
            constraints = kwargs.get("constraints", [])
            include_alternatives = kwargs.get("include_alternatives", True)
            include_metrics = kwargs.get("include_metrics", True)
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.3)
            
            if not workflow:
                return {"success": False, "error": "Workflow is required for optimization"}
            
            # Build the optimization prompt
            system_prompt = self._get_workflow_optimization_system_prompt(optimization_goal)
            user_prompt = f"Workflow to optimize:\n\n{json.dumps(workflow, indent=2)}\n\n"
            user_prompt += f"Optimization goal: {optimization_goal}\n\n"
            
            if constraints:
                user_prompt += f"Constraints: {', '.join(constraints)}\n\n"
            
            user_prompt += "Please optimize this workflow"
            
            if include_alternatives:
                user_prompt += " and provide alternative approaches"
            
            if include_metrics:
                user_prompt += " with performance metrics"
            
            user_prompt += "."
            
            # Call OpenAI API
            response = self._call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=1500,
                temperature=temperature
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "optimized_workflow": response.get("content"),
                    "optimization_goal": optimization_goal,
                    "model": model,
                    "tokens_used": response.get("tokens_used"),
                    "cost": response.get("cost")
                }
            else:
                return {"success": False, "error": response.get("error")}
                
        except Exception as e:
            return {"success": False, "error": f"AI workflow optimization failed: {str(e)}"}
    
    def _ai_patent_analysis_handler(self, **kwargs):
        """Handler for AI-powered patent analysis."""
        try:
            # Check if OpenAI is configured
            if not self.openai_configured:
                return {"success": False, "error": "OpenAI API not configured. Please set OPENAI_API_KEY environment variable."}
            
            patent_title = kwargs.get("patent_title")
            patent_abstract = kwargs.get("patent_abstract")
            analysis_type = kwargs.get("analysis_type", "both")
            model = kwargs.get("model", "gpt-4")
            temperature = kwargs.get("temperature", 0.3)
            
            if not patent_title or not patent_abstract:
                return {"success": False, "error": "Patent title and abstract are required for analysis"}
            
            results = {}
            
            # Generate value proposition if requested
            if analysis_type in ["value_proposition", "both"]:
                kvp_system_prompt = self._get_patent_kvp_system_prompt()
                kvp_user_prompt = f"""Analyze this patent and provide a concise 1-3 line summary of its key value proposition:

Title: {patent_title}
Abstract: {patent_abstract}

Focus on:
- What problem does it solve?
- What makes it innovative?
- What are the key benefits?

Provide a clear, business-focused summary in 1-3 lines:"""

                kvp_response = self._call_openai_api(
                    system_prompt=kvp_system_prompt,
                    user_prompt=kvp_user_prompt,
                    model=model,
                    max_tokens=300,
                    temperature=temperature
                )
                
                if kvp_response.get("success"):
                    results["value_proposition"] = kvp_response.get("content")
                    results["kvp_tokens_used"] = kvp_response.get("tokens_used")
                    results["kvp_cost"] = kvp_response.get("cost")
                else:
                    results["value_proposition"] = f"Analysis failed: {kvp_response.get('error')}"
            
            # Generate categorization if requested
            if analysis_type in ["categorization", "both"]:
                cat_system_prompt = self._get_patent_category_system_prompt()
                cat_user_prompt = f"""Based on this patent information, assign it to ONE of these cybersecurity categories:

Title: {patent_title}
Abstract: {patent_abstract}

Categories:
- Network Security
- Endpoint Protection
- Identity & Access Management
- Data Protection & Encryption
- Threat Detection & Response
- Vulnerability Management
- Security Operations
- Compliance & Governance
- Application Security
- Infrastructure Security
- Cloud Security
- IoT Security
- Mobile Security
- Incident Response
- Forensic Analysis
- Other

Choose the SINGLE most appropriate category:"""

                cat_response = self._call_openai_api(
                    system_prompt=cat_system_prompt,
                    user_prompt=cat_user_prompt,
                    model=model,
                    max_tokens=100,
                    temperature=temperature
                )
                
                if cat_response.get("success"):
                    results["category"] = cat_response.get("content")
                    results["cat_tokens_used"] = cat_response.get("tokens_used")
                    results["cat_cost"] = cat_response.get("cost")
                else:
                    results["category"] = f"Analysis failed: {cat_response.get('error')}"
            
            # Calculate total tokens and cost
            total_tokens = results.get("kvp_tokens_used", 0) + results.get("cat_tokens_used", 0)
            total_cost = results.get("kvp_cost", 0) + results.get("cat_cost", 0)
            
            return {
                "success": True,
                "patent_title": patent_title,
                "analysis_type": analysis_type,
                "model": model,
                "total_tokens_used": total_tokens,
                "total_cost": total_cost,
                **results
            }
                
        except Exception as e:
            return {"success": False, "error": f"AI patent analysis failed: {str(e)}"}
    
    def _create_ascii_art(self, text: str, style: str, width: int) -> str:
        """Create ASCII art from text using different styles."""
        text = text.upper()
        
        if style == "simple":
            return self._simple_ascii_art(text, width)
        elif style == "block":
            return self._block_ascii_art(text, width)
        elif style == "matrix":
            return self._matrix_ascii_art(text, width)
        elif style == "hacker":
            return self._hacker_ascii_art(text, width)
        else:  # cyber style (default)
            return self._cyber_ascii_art(text, width)
    
    def _cyber_ascii_art(self, text: str, width: int) -> str:
        """Generate cyber-style ASCII art."""
        # Cyber-style font mapping
        cyber_font = {
            'A': [' ██████╗', '██╔═══██╗', '███████╔╝', '██╔═══██╗', '███████╔╝', '██║  ██║', '╚██████╔╝'],
            'B': ['██████╗ ', '██╔══██╗', '██████╔╝', '██╔══██╗', '██████╔╝', '██╔══██╗', '██████╔╝'],
            'C': [' ██████╗', '██╔═══██╗', '██║     ', '██║     ', '██║     ', '╚██████╔╝', ' ╚═════╝'],
            'D': ['██████╗ ', '██╔══██╗', '██║  ██║', '██║  ██║', '██║  ██║', '██╔══██╗', '██████╔╝'],
            'E': ['███████╗', '██╔════╝', '███████╗', '██╔════╝', '███████╗', '██╔════╝', '███████╗'],
            'F': ['███████╗', '██╔════╝', '███████╗', '██╔════╝', '██║     ', '██║     ', '╚═╝     '],
            'G': [' ██████╗', '██╔═══██╗', '██║     ', '██║ ███╗', '██║  ██║', '╚██████╔╝', ' ╚═════╝'],
            'H': ['██╗  ██╗', '██║  ██║', '███████║', '██╔═══██║', '██║  ██║', '██║  ██║', '╚═╝  ╚═╝'],
            'I': ['██╗', '██║', '██║', '██║', '██║', '██║', '╚═╝'],
            'J': ['     ██╗', '     ██║', '     ██║', '     ██║', '██╗  ██║', '╚█████╔╝', ' ╚════╝ '],
            'K': ['██╗  ██╗', '██║ ██╔╝', '█████╔╝ ', '██╔═██╗ ', '██║╚██╗', '██║ ╚██╗', '╚═╝  ╚═╝'],
            'L': ['██╗     ', '██║     ', '██║     ', '██║     ', '██║     ', '███████╗', '╚══════╝'],
            'M': ['███╗   ███╗', '████╗ ████║', '██╔████╔██║', '██║╚██╔╝██║', '██║ ╚═╝ ██║', '██║     ██║', '╚═╝     ╚═╝'],
            'N': ['███╗   ██╗', '████╗  ██║', '██╔██╗ ██║', '██║╚██╗██║', '██║ ╚████║', '██║  ╚██║', '╚═╝   ╚═╝'],
            'O': [' ██████╗', '██╔═══██╗', '██║   ██║', '██║   ██║', '██║   ██║', '╚██████╔╝', ' ╚═════╝'],
            'P': ['██████╗ ', '██╔══██╗', '██████╔╝', '██╔═══╝ ', '██║     ', '██║     ', '╚═╝     '],
            'Q': [' ██████╗', '██╔═══██╗', '██║   ██║', '██║   ██║', '██║   ██║', '╚██████╔╝', ' ╚═════╝'],
            'R': ['██████╗ ', '██╔══██╗', '██████╔╝', '██╔══██╗', '██║  ██║', '██║  ██║', '╚═╝  ╚═╝'],
            'S': [' ██████╗', '██╔═══██╗', '╚█████╔╝', ' ╚═══██╗', '██████╔╝', '╚═════╝ ', '        '],
            'T': ['████████╗', '╚══██╔══╝', '   ██║   ', '   ██║   ', '   ██║   ', '   ██║   ', '   ╚═╝   '],
            'U': ['██╗   ██╗', '██║   ██║', '██║   ██║', '██║   ██║', '██║   ██║', '╚██████╔╝', ' ╚═════╝'],
            'V': ['██╗   ██╗', '██║   ██║', '██║   ██║', '╚██╗ ██╔╝', ' ╚████╔╝ ', '  ╚██╔╝  ', '   ╚═╝   '],
            'W': ['██╗    ██╗', '██║    ██║', '██║ █╗ ██║', '██║███╗██║', '╚███╔███╔╝', ' ╚══╝╚══╝ ', '        '],
            'X': ['██╗  ██╗', '╚██╗██╔╝', ' ╚███╔╝ ', ' ██╔██╗ ', '██╔╝╚██╗', '██║  ╚██╗', '╚═╝  ╚═╝'],
            'Y': ['██╗   ██╗', ' ╚██╗ ██╔╝', '  ╚████╔╝ ', '   ╚██╔╝  ', '   ██║   ', '   ██║   ', '   ╚═╝   '],
            'Z': ['███████╗', '╚════██║', '    ██╔╝', '   ██╔╝ ', '  ██╔╝  ', ' ██╔╝   ', '███████╗'],
            ' ': ['   ', '   ', '   ', '   ', '   ', '   ', '   '],
            '0': [' ██████╗', '██╔═══██╗', '██║██╗██║', '██║██║██║', '██║╚████║', '╚██████╔╝', ' ╚═════╝'],
            '1': ['  ██╗ ', ' ███║ ', ' ╚██║ ', '  ██║ ', '  ██║ ', '  ██║ ', '  ╚═╝ '],
            '2': ['██████╗ ', '╚════██╗', ' █████╔╝', '██╔═══╝ ', '███████╗', '╚══════╝', '        '],
            '3': ['██████╗ ', '╚════██╗', ' █████╔╝', ' ╚═══██╗', '██████╔╝', '╚═════╝ ', '        '],
            '4': ['██╗  ██╗', '██║  ██║', '███████║', '╚════██║', '     ██║', '     ██║', '     ╚═╝'],
            '5': ['███████╗', '██╔════╝', '███████╗', '╚════██╗', '███████╗', '╚══════╝', '        '],
            '6': [' ██████╗', '██╔════╝', '███████╗', '██╔═══██╗', '╚██████╔╝', ' ╚═════╝', '        '],
            '7': ['███████╗', '╚════██║', '    ██╔╝', '   ██╔╝ ', '  ██╔╝  ', ' ██╔╝   ', '╚═╝     '],
            '8': [' ██████╗', '██╔═══██╗', '╚█████╔╝', '██╔═══██╗', '██████╔╝', '╚═════╝ ', '        '],
            '9': [' ██████╗', '██╔═══██╗', '╚██████╔╝', ' ╚═══██║', ' █████╔╝', ' ╚═════╝', '        '],
            '!': ['██╗', '██║', '██║', '██║', '╚═╝', '██╗', '╚═╝'],
            '?': [' ██████╗', '██╔═══██╗', '    ██╔╝', '   ██╔╝ ', '   ╚═╝  ', '   ██╗  ', '   ╚═╝  '],
            '.': ['   ', '   ', '   ', '   ', '   ', '██╗', '╚═╝'],
            '-': ['        ', '        ', '██████╗', '        ', '        ', '        ', '        '],
            '_': ['        ', '        ', '        ', '        ', '        ', '        ', '███████╗']
        }
        
        # Generate art line by line
        lines = []
        for i in range(7):  # 7 lines tall
            line = ""
            for char in text:
                if char in cyber_font:
                    line += cyber_font[char][i]
                else:
                    line += "   "  # Default spacing for unknown chars
            lines.append(line)
        
        # Center the art within the specified width
        centered_lines = []
        for line in lines:
            if len(line) < width:
                padding = (width - len(line)) // 2
                centered_line = " " * padding + line
            else:
                centered_line = line[:width]
            centered_lines.append(centered_line)
        
        return "\n".join(centered_lines)
    
    def _simple_ascii_art(self, text: str, width: int) -> str:
        """Generate simple ASCII art."""
        art = f"""    ╔{'═' * (len(text) + 4)}╗
    ║  {text}  ║
    ╚{'═' * (len(text) + 4)}╝
"""
        return art
    
    def _block_ascii_art(self, text: str, width: int) -> str:
        """Generate block-style ASCII art."""
        art = f"""    ┌{'─' * (len(text) + 4)}┐
    │  {text}  │
    └{'─' * (len(text) + 4)}┘
"""
        return art
    
    def _matrix_ascii_art(self, text: str, width: int) -> str:
        """Generate matrix-style ASCII art."""
        matrix_chars = "01"
        art = f"""    ╔{'═' * (len(text) + 4)}╗
    ║  {text}  ║
    ╚{'═' * (len(text) + 4)}╝
"""
        # Add matrix-style decoration
        matrix_lines = []
        for i in range(3):
            line = "  " + "".join([matrix_chars[i % 2] for _ in range(len(text) + 4)]) + "  "
            matrix_lines.append(line)
        
        return "\n".join(matrix_lines) + "\n" + art + "\n" + "\n".join(matrix_lines)
    
    def _hacker_ascii_art(self, text: str, width: int) -> str:
        """Generate hacker-style ASCII art."""
        art = f"""    ╔{'═' * (len(text) + 4)}╗
    ║  {text}  ║
    ╚{'═' * (len(text) + 4)}╝
"""
        # Add hacker-style decoration
        hacker_lines = [
            "  /\\_/\\",
            " ( o.o )",
            "  > ^ <",
            "  /   \\",
            " /     \\"
        ]
        
        return "\n".join(hacker_lines) + "\n" + art + "\n" + "\n".join(hacker_lines[::-1])
    
    def _apply_color(self, ascii_art: str, color: str) -> str:
        """Apply color to ASCII art (placeholder for future color support)."""
        # For now, return the art as-is
        # In the future, this could add ANSI color codes
        return ascii_art
    
    def _get_random_cyber_message(self, message_type: str, tone: str) -> Dict[str, str]:
        """Get a random cyber-themed message."""
        import random
        
        # Cyber-themed message collections
        welcome_messages = [
            {"message": "Welcome to the digital battlefield, soldier.", "type": "welcome", "tone": "professional", "author": "CS-AI Commander"},
            {"message": "Initiating threat analysis protocols...", "type": "welcome", "tone": "professional", "author": "Security AI"},
            {"message": "The matrix has you... follow the white rabbit.", "type": "welcome", "tone": "hacker", "author": "Neo"},
            {"message": "Access granted. Welcome to the secure zone.", "type": "welcome", "tone": "professional", "author": "System Admin"},
            {"message": "Time to hack the planet!", "type": "welcome", "tone": "hacker", "author": "Hackerman"},
            {"message": "Security clearance: Level 10. Welcome aboard.", "type": "welcome", "tone": "professional", "author": "Security Protocol"},
            {"message": "The code is strong with this one.", "type": "welcome", "tone": "inspirational", "author": "Code Master"},
            {"message": "Entering the realm of digital forensics.", "type": "welcome", "tone": "professional", "author": "Forensics AI"},
            {"message": "Welcome to the future of cybersecurity.", "type": "welcome", "tone": "inspirational", "author": "Future Tech"},
            {"message": "Infiltrating the system... access granted.", "type": "welcome", "tone": "hacker", "author": "Ghost in the Shell"}
        ]
        
        exit_messages = [
            {"message": "Mission accomplished. Shutting down.", "type": "exit", "tone": "professional", "author": "CS-AI Commander"},
            {"message": "Threat neutralized. System secure.", "type": "exit", "tone": "professional", "author": "Security AI"},
            {"message": "Exiting the matrix... until next time.", "type": "exit", "tone": "hacker", "author": "Neo"},
            {"message": "Session terminated. Stay vigilant.", "type": "exit", "tone": "professional", "author": "System Admin"},
            {"message": "Hack complete. Mission successful.", "type": "exit", "tone": "hacker", "author": "Hackerman"},
            {"message": "Security protocols deactivated. Goodbye.", "type": "exit", "tone": "professional", "author": "Security Protocol"},
            {"message": "The code has spoken. Farewell.", "type": "exit", "tone": "inspirational", "author": "Code Master"},
            {"message": "Forensics complete. Evidence secured.", "type": "exit", "tone": "professional", "author": "Forensics AI"},
            {"message": "Future secured. Until we meet again.", "type": "exit", "tone": "inspirational", "author": "Future Tech"},
            {"message": "Ghost in the shell... signing off.", "type": "exit", "tone": "hacker", "author": "Ghost in the Shell"}
        ]
        
        threatening_messages = [
            {"message": "Your security is an illusion.", "type": "threatening", "tone": "threatening", "author": "Shadow Hacker"},
            {"message": "I see through your defenses.", "type": "threatening", "tone": "threatening", "author": "Digital Phantom"},
            {"message": "Your firewall is like paper to me.", "type": "threatening", "tone": "threatening", "author": "Cyber Predator"},
            {"message": "I am the virus in your system.", "type": "threatening", "tone": "threatening", "author": "Digital Plague"},
            {"message": "Your encryption is child's play.", "type": "threatening", "tone": "threatening", "author": "Code Breaker"}
        ]
        
        inspirational_messages = [
            {"message": "In code we trust, in security we believe.", "type": "inspirational", "tone": "inspirational", "author": "Digital Prophet"},
            {"message": "The best defense is a good offense.", "type": "inspirational", "tone": "inspirational", "author": "Security Sage"},
            {"message": "Innovation is the key to survival.", "type": "inspirational", "tone": "inspirational", "author": "Tech Visionary"},
            {"message": "Adapt, evolve, secure.", "type": "inspirational", "tone": "inspirational", "author": "Digital Darwin"},
            {"message": "The future belongs to the prepared.", "type": "inspirational", "tone": "inspirational", "author": "Future Guardian"}
        ]
        
        # Select message collection based on type and tone
        if message_type == "welcome":
            collection = welcome_messages
        elif message_type == "exit":
            collection = exit_messages
        elif tone == "threatening":
            collection = threatening_messages
        elif tone == "inspirational":
            collection = inspirational_messages
        elif tone == "hacker":
            collection = [m for m in welcome_messages + exit_messages if m["tone"] == "hacker"]
        elif tone == "professional":
            collection = [m for m in welcome_messages + exit_messages if m["tone"] == "professional"]
        else:  # random
            collection = welcome_messages + exit_messages + threatening_messages + inspirational_messages
        
        # Return random message from selected collection
        return random.choice(collection)
    
    def _call_openai_api(self, system_prompt: str, user_prompt: str, model: str = "gpt-4", 
                         max_tokens: int = 1000, temperature: float = 0.3) -> Dict[str, Any]:
        """Make a call to the OpenAI API using the centralized LLM client."""
        if not self.llm_client.is_available():
            return {"success": False, "error": "OpenAI LLM client not available"}
        
        # Map model string to ModelType enum
        model_mapping = {
            "gpt-4": ModelType.GPT_4,
            "gpt-4-turbo": ModelType.GPT_4_TURBO,
            "gpt-3.5-turbo": ModelType.GPT_3_5_TURBO,
            "gpt-3.5-turbo-16k": ModelType.GPT_3_5_TURBO_16K
        }
        
        model_type = model_mapping.get(model, ModelType.GPT_4)
        
        # Create LLM config
        config = LLMConfig(
            model=model_type,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Make the API call using centralized client
        response = self.llm_client.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            config=config
        )
        
        # Convert response to expected format
        if response.success:
            return {
                "success": True,
                "content": response.content,
                "tokens_used": response.tokens_used,
                "cost": response.cost
            }
        else:
            return {
                "success": False,
                "error": response.error
            }
    

    
    def _get_reasoning_system_prompt(self, reasoning_type: str) -> str:
        """Get system prompt for reasoning tasks."""
        prompts = {
            "threat_analysis": """You are a cybersecurity expert specializing in threat analysis. 
            Analyze the given information to identify potential threats, assess risks, and provide 
            actionable insights. Focus on security implications and practical recommendations.""",
            
            "risk_assessment": """You are a risk assessment specialist. Evaluate the given information 
            to identify risks, assess their likelihood and impact, and provide risk mitigation strategies. 
            Use a structured approach to risk analysis.""",
            
            "pattern_recognition": """You are a pattern recognition expert. Analyze the given data to 
            identify patterns, trends, and anomalies. Look for correlations, sequences, and hidden 
            relationships that could be significant.""",
            
            "causal_analysis": """You are a causal analysis expert. Examine the given information to 
            identify cause-and-effect relationships, root causes, and contributing factors. Provide 
            logical reasoning for your conclusions.""",
            
            "hypothesis_generation": """You are a hypothesis generation specialist. Based on the given 
            information, generate plausible hypotheses, explain your reasoning, and suggest ways to 
            test or validate these hypotheses."""
        }
        
        return prompts.get(reasoning_type, prompts["threat_analysis"])
    
    def _get_categorization_system_prompt(self, categorization_type: str) -> str:
        """Get system prompt for categorization tasks."""
        prompts = {
            "threat_classification": """You are a threat classification expert. Categorize the given 
            threat data into appropriate categories. Provide confidence scores and reasoning for 
            your classifications. Focus on cybersecurity threat taxonomy.""",
            
            "data_classification": """You are a data classification specialist. Categorize the given 
            data according to sensitivity, type, and purpose. Use standard data classification 
            frameworks and provide confidence scores.""",
            
            "severity_assessment": """You are a severity assessment expert. Evaluate the given 
            information and assign severity levels (Low, Medium, High, Critical). Provide 
            confidence scores and reasoning for your assessments.""",
            
            "priority_ranking": """You are a priority ranking specialist. Rank the given items 
            by priority based on urgency, importance, and impact. Provide confidence scores 
            and reasoning for your rankings.""",
            
            "custom_classification": """You are a classification expert. Categorize the given 
            data according to the specified categories. Provide confidence scores and reasoning 
            for your classifications."""
        }
        
        return prompts.get(categorization_type, prompts["threat_classification"])
    
    def _get_summarization_system_prompt(self, summary_type: str) -> str:
        """Get system prompt for summarization tasks."""
        prompts = {
            "executive_summary": """You are an executive summary specialist. Create concise, 
            high-level summaries suitable for executive audiences. Focus on key insights, 
            implications, and actionable recommendations.""",
            
            "technical_summary": """You are a technical summary specialist. Create detailed, 
            technical summaries for technical audiences. Include technical details, methodologies, 
            and technical implications.""",
            
            "threat_summary": """You are a threat summary specialist. Create focused summaries 
            of threat-related information. Highlight threat indicators, risks, and mitigation 
            strategies.""",
            
            "incident_summary": """You are an incident summary specialist. Create structured 
            summaries of security incidents. Include timeline, impact, response, and lessons 
            learned.""",
            
            "custom_summary": """You are a summary specialist. Create summaries according to 
            the specified requirements. Focus on clarity, accuracy, and usefulness."""
        }
        
        return prompts.get(summary_type, prompts["executive_summary"])
    
    def _get_code_transformation_system_prompt(self, transformation_type: str) -> str:
        """Get system prompt for code transformation tasks."""
        prompts = {
            "language_conversion": """You are a code conversion expert. Convert the given code 
            to the target programming language while maintaining functionality and best practices. 
            Ensure the converted code is idiomatic for the target language.""",
            
            "query_optimization": """You are a query optimization expert. Optimize the given 
            query for better performance, readability, and maintainability. Consider indexing, 
            query structure, and database-specific optimizations.""",
            
            "code_refactoring": """You are a code refactoring expert. Refactor the given code 
            to improve structure, readability, and maintainability. Follow clean code principles 
            and design patterns.""",
            
            "format_conversion": """You are a format conversion expert. Convert the given code 
            or data to the target format while preserving structure and meaning. Ensure the 
            output follows the target format specifications.""",
            
            "security_enhancement": """You are a security enhancement expert. Improve the 
            security of the given code by identifying and fixing security vulnerabilities. 
            Follow security best practices and coding standards."""
        }
        
        return prompts.get(transformation_type, prompts["language_conversion"])
    
    def _get_query_enhancement_system_prompt(self, enhancement_type: str) -> str:
        """Get system prompt for query enhancement tasks."""
        prompts = {
            "query_expansion": """You are a query expansion expert. Enhance the given query 
            by adding relevant terms, synonyms, and related concepts. Maintain the original 
            intent while improving search coverage and relevance.""",
            
            "query_optimization": """You are a query optimization expert. Optimize the given 
            query for better search performance and results. Consider query structure, 
            terminology, and search engine best practices.""",
            
            "query_generation": """You are a query generation expert. Generate effective 
            queries based on the given information and requirements. Focus on clarity, 
            specificity, and search effectiveness.""",
            
            "query_translation": """You are a query translation expert. Translate the given 
            query to the target format or language while preserving meaning and intent. 
            Ensure the translated query is effective and natural.""",
            
            "query_validation": """You are a query validation expert. Validate the given 
            query for correctness, completeness, and effectiveness. Identify potential 
            issues and suggest improvements."""
        }
        
        return prompts.get(enhancement_type, prompts["query_optimization"])
    
    def _get_threat_intelligence_system_prompt(self, analysis_type: str) -> str:
        """Get system prompt for threat intelligence tasks."""
        prompts = {
            "threat_assessment": """You are a threat assessment expert. Analyze the given 
            threat data to assess the nature, scope, and impact of potential threats. 
            Provide actionable intelligence and recommendations.""",
            
            "indicator_analysis": """You are an indicator analysis expert. Analyze the given 
            threat indicators to identify patterns, relationships, and implications. 
            Focus on actionable intelligence and threat hunting opportunities.""",
            
            "attack_pattern_recognition": """You are an attack pattern recognition expert. 
            Identify attack patterns, techniques, and tactics in the given threat data. 
            Use MITRE ATT&CK framework and provide detailed analysis.""",
            
            "threat_hunting": """You are a threat hunting expert. Analyze the given data 
            to identify potential threats, anomalies, and suspicious activities. Provide 
            hunting hypotheses and investigation guidance.""",
            
            "vulnerability_assessment": """You are a vulnerability assessment expert. 
            Analyze the given information to identify vulnerabilities, assess their 
            severity, and provide remediation recommendations."""
        }
        
        return prompts.get(analysis_type, prompts["threat_assessment"])
    
    def _get_workflow_optimization_system_prompt(self, optimization_goal: str) -> str:
        """Get system prompt for workflow optimization tasks."""
        prompts = {
            "performance": """You are a performance optimization expert. Analyze the given 
            workflow to identify performance bottlenecks and optimization opportunities. 
            Focus on speed, efficiency, and resource utilization improvements.""",
            
            "efficiency": """You are an efficiency optimization expert. Analyze the given 
            workflow to identify inefficiencies and improvement opportunities. Focus on 
            process optimization, resource allocation, and workflow streamlining.""",
            
            "security": """You are a security optimization expert. Analyze the given 
            workflow to identify security risks and improvement opportunities. Focus on 
            security controls, risk mitigation, and compliance requirements.""",
            
            "reliability": """You are a reliability optimization expert. Analyze the given 
            workflow to identify reliability issues and improvement opportunities. Focus on 
            error handling, fault tolerance, and system resilience.""",
            
            "cost": """You are a cost optimization expert. Analyze the given workflow to 
            identify cost optimization opportunities. Focus on resource utilization, 
            efficiency improvements, and cost-benefit analysis."""
        }
        
        return prompts.get(optimization_goal, prompts["performance"])
    
    def _get_patent_kvp_system_prompt(self) -> str:
        """Get system prompt for patent key value proposition analysis."""
        return """You are a cybersecurity patent analyst specializing in identifying key value propositions. 
        
Your task is to analyze patent information and provide a concise 1-3 line summary of the patent's key value proposition.

Focus on:
- What problem does this patent solve?
- What makes it innovative or unique?
- What are the key benefits for cybersecurity?

Provide a clear, business-focused summary that would be valuable for:
- Technology assessment teams
- Investment decision makers
- Competitive intelligence analysts

Keep your response concise and actionable."""

    def _get_patent_category_system_prompt(self) -> str:
        """Get system prompt for patent categorization."""
        return """You are a cybersecurity patent classification expert. 
        
Your task is to assign the given patent to ONE of these cybersecurity categories:

- Network Security
- Endpoint Protection
- Identity & Access Management
- Data Protection & Encryption
- Threat Detection & Response
- Vulnerability Management
- Security Operations
- Compliance & Governance
- Application Security
- Infrastructure Security
- Cloud Security
- IoT Security
- Mobile Security
- Incident Response
- Forensic Analysis
- Other

Choose the SINGLE most appropriate category based on the patent's title, abstract, and technical focus.

Provide only the category name as your response."""
    
    def _parse_categorization_response(self, response: str, categories: List[str], 
                                     confidence_threshold: float, include_reasoning: bool) -> Dict[str, Any]:
        """Parse the AI response for categorization results."""
        try:
            # Try to extract structured information from the response
            result = {
                "primary_category": None,
                "confidence_score": 0.0,
                "alternative_categories": [],
                "reasoning": None
            }
            
            # Simple parsing logic - can be enhanced with more sophisticated parsing
            response_lower = response.lower()
            
            # Find the most likely category
            for category in categories:
                if category.lower() in response_lower:
                    result["primary_category"] = category
                    break
            
            # Extract confidence score if present
            confidence_match = re.search(r'confidence[:\s]*(\d+(?:\.\d+)?)', response_lower)
            if confidence_match:
                result["confidence_score"] = float(confidence_match.group(1))
            
            # Extract reasoning if requested
            if include_reasoning:
                result["reasoning"] = response
            
            return result
            
        except Exception as e:
            # Fallback to basic parsing
            return {
                "primary_category": categories[0] if categories else None,
                "confidence_score": 0.5,
                "alternative_categories": categories[1:3] if len(categories) > 1 else [],
                "reasoning": response,
                "parsing_error": str(e)
            }
    
    async def _call_tool_handler(self, **kwargs):
        """Handler for calling tools with comprehensive logging."""
        start_time = time.time()
        tool_name = kwargs.get('name')
        arguments = kwargs.get('arguments', {})
        
        # Log tool call start
        if self.session_logger:
            self.session_logger.log_info("tool_call_started", f"Starting tool call: {tool_name}", metadata={
                "tool_name": tool_name,
                "arguments": arguments
            })
        
        try:
            # Find the tool
            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found"
                if self.session_logger:
                    self.session_logger.log_error("tool_not_found", error_msg, metadata={
                        "tool_name": tool_name,
                        "available_tools": list(self.tools.keys())
                    })
                return {"success": False, "error": error_msg}
            
            tool = self.tools[tool_name]
            
            # Log tool execution
            if self.session_logger:
                self.session_logger.log_workflow_execution(
                    step=f"tool_execution_{tool_name}",
                    action="tool_executing",
                    details={
                        "tool_name": tool_name,
                        "category": tool.category,
                        "tags": tool.tags
                    }
                )
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            
            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful tool execution
            if self.session_logger:
                self.session_logger.log_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration_ms,
                    success=True
                )
                
                self.session_logger.log_workflow_execution(
                    step=f"tool_execution_{tool_name}",
                    action="tool_completed",
                    details={
                        "tool_name": tool_name,
                        "duration_ms": duration_ms,
                        "success": True
                    }
                )
            
            return {
                "success": True,
                "result": result,
                "execution_time_ms": duration_ms
            }
            
        except Exception as e:
            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000
            
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            
            # Log tool execution error
            if self.session_logger:
                self.session_logger.log_error("tool_execution_failed", error_msg, metadata={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error": str(e),
                    "duration_ms": duration_ms
                })
                
                self.session_logger.log_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e)
                )
                
                self.session_logger.log_workflow_execution(
                    step=f"tool_execution_{tool_name}",
                    action="tool_failed",
                    details={
                        "tool_name": tool_name,
                        "error": str(e),
                        "duration_ms": duration_ms
                    }
                )
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time_ms": duration_ms
            }
    
    def _list_resources_handler(self, **kwargs):
        """Handler for listing resources."""
        return {
            "success": True,
            "resources": list(self.resources.keys())
        }
    
    async def _read_resource_handler(self, **kwargs):
        """Handler for reading resources."""
        resource_name = kwargs.get('name')
        
        if resource_name not in self.resources:
            return {
                "success": False,
                "error": f"Resource '{resource_name}' not found"
            }
        
        return {
            "success": True,
            "resource": self.resources[resource_name]
        }
    
    async def execute_tool(self, tool_name: str, parameters: dict) -> dict:
        """Execute a tool by name with given parameters."""
        try:
            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                }
            
            tool = self.tools[tool_name]
            result = await self._call_tool_handler(name=tool_name, arguments=parameters)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name
            }

# Core Tool Classes
class DataFrameManager:
    """Comprehensive pandas dataframe management for agent workflows."""
    
    def __init__(self):
        self.dataframes = {}
        self.schemas = {}
        self.workflow_history = []
    
    def create_dataframe(self, name: str, columns: List[str], data: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
        """Create a new dataframe with specified columns."""
        try:
            if name in self.dataframes:
                return {"success": False, "error": f"Dataframe '{name}' already exists"}
            
            if data:
                df = pd.DataFrame(data, columns=columns)
            else:
                df = pd.DataFrame(columns=columns)
            
            self.dataframes[name] = df
            self._update_schema(name)
            
            return {
                "success": True,
                "message": f"Dataframe '{name}' created successfully",
                "shape": df.shape,
                "columns": list(df.columns)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_dataframes(self) -> Dict[str, Any]:
        """List all available dataframes."""
        try:
            df_list = []
            for name, df in self.dataframes.items():
                df_info = {
                    "name": name,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict()
                }
                df_list.append(df_info)
            
            return {
                "success": True,
                "dataframes": df_list,
                "total_count": len(df_list)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def query_dataframe(self, name: str, query: str) -> Dict[str, Any]:
        """Query dataframe using natural language or pandas syntax."""
        try:
            if name not in self.dataframes:
                return {"success": False, "error": f"Dataframe '{name}' not found"}
            
            df = self.dataframes[name]
            
            # Simple natural language query processing
            if "first" in query.lower() and "rows" in query.lower():
                num = 5  # Default to 5
                if "10" in query:
                    num = 10
                elif "20" in query:
                    num = 20
                result = df.head(num)
                return {
                    "success": True,
                    "result": result.to_dict('records'),
                    "query_type": "nlp",
                    "rows_returned": len(result)
                }
            
            # Try pandas query syntax
            try:
                result = df.query(query)
                return {
                    "success": True,
                    "result": result.to_dict('records'),
                    "query_type": "pandas",
                    "rows_returned": len(result)
                }
            except:
                return {"success": False, "error": f"Invalid query syntax: {query}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_schema(self, name: str):
        """Update schema information for a dataframe."""
        if name in self.dataframes:
            df = self.dataframes[name]
            self.schemas[name] = {
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "shape": df.shape,
                "last_updated": datetime.now().isoformat()
            }
    
    def export_dataframe(self, name: str, file_path: str = None, format: str = "csv", 
                        use_session_output: bool = True) -> Dict[str, Any]:
        """Export dataframe to file in specified format."""
        try:
            if name not in self.dataframes:
                return {"success": False, "error": f"Dataframe '{name}' not found"}
            
            df = self.dataframes[name]
            
            # Generate default filename if not provided
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"dataframe_{name}_{timestamp}.{format}"
            
            # Use session output directory if requested and available
            if use_session_output and hasattr(session_logger, 'output_dir'):
                # Ensure filename has proper extension
                if not file_path.endswith(f".{format}"):
                    file_path = f"{file_path}.{format}"
                
                # Create session-specific filename
                session_filename = f"dataframe_{name}_{timestamp}.{format}"
                file_path = str(session_logger.output_dir / session_filename)
                
                # Track this output in session logger
                session_logger.create_output_file(
                    filename=session_filename,
                    description=f"Dataframe '{name}' exported in {format.upper()} format",
                    category="dataframe_export"
                )
            
            # Export based on format
            if format.lower() == "csv":
                df.to_csv(file_path, index=False)
            elif format.lower() == "excel":
                df.to_excel(file_path, index=False)
            elif format.lower() == "json":
                df.to_json(file_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}
            
            return {
                "success": True,
                "message": f"Dataframe '{name}' exported successfully",
                "file_path": file_path,
                "format": format,
                "rows_exported": len(df),
                "columns_exported": len(df.columns),
                "session_output": use_session_output and hasattr(session_logger, 'output_dir')
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_dataframe(self, name: str) -> Dict[str, Any]:
        """Delete a dataframe from memory."""
        try:
            if name not in self.dataframes:
                return {"success": False, "error": f"Dataframe '{name}' not found"}
            
            del self.dataframes[name]
            if name in self.schemas:
                del self.schemas[name]
            
            return {
                "success": True,
                "message": f"Dataframe '{name}' deleted successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_dataframe_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a dataframe."""
        try:
            if name not in self.dataframes:
                return {"success": False, "error": f"Dataframe '{name}' not found"}
            
            df = self.dataframes[name]
            schema = self.schemas.get(name, {})
            
            return {
                "success": True,
                "name": name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": schema.get("memory_usage", 0),
                "null_counts": schema.get("null_counts", {}),
                "sample_data": df.head(3).to_dict('records')
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

class ToolManager:
    """Lazy loading manager for all tools with MCP server integration."""
    
    def __init__(self):
        self._df_manager = None
        self._sqlite_manager = None
        self._neo4j_manager = None
        self._file_tools = None
        self._compression_tools = None
        self._ml_tools = None
        self._nlp_tools = None
        self._context_memory = None
        self._mcp_server = None
    
    @property
    def mcp_server(self):
        """Lazy load MCP server instance."""
        if self._mcp_server is None:
            self._mcp_server = MCPServer()
            # Connect the tool manager to the MCP server
            self._mcp_server.set_tool_manager(self)
            # Connect the session logger to the MCP server
            self._mcp_server.set_session_logger(session_logger)
        
        return self._mcp_server
    
    @property
    def df_manager(self):
        """Lazy load DataFrameManager only when needed."""
        if self._df_manager is None:
            self._df_manager = DataFrameManager()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._df_manager
    
    @property
    def sqlite_manager(self):
        """Lazy load SQLiteManager only when needed."""
        if self._sqlite_manager is None:
            try:
                from bin.sqlite_manager import SQLiteManager
                self._sqlite_manager = SQLiteManager()
                logger.info("✅ SQLite Manager initialized")
            except ImportError as e:
                logger.warning(f"⚠️ SQLite Manager not available: {e}")
                # Create a minimal placeholder
                self._sqlite_manager = type('SQLiteManager', (), {
                    'insert_data': lambda *args, **kwargs: {"success": False, "error": "SQLite Manager not available"},
                    'query_data': lambda *args, **kwargs: [],
                    'update_data': lambda *args, **kwargs: {"success": False, "error": "SQLite Manager not available"},
                    'delete_data': lambda *args, **kwargs: {"success": False, "error": "SQLite Manager not available"}
                })()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._sqlite_manager
    
    @property
    def neo4j_manager(self):
        """Lazy load Neo4jManager only when needed."""
        if self._neo4j_manager is None:
            # For now, create a placeholder - you can add the actual Neo4jManager class later
            self._neo4j_manager = type('Neo4jManager', (), {})()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._neo4j_manager
    
    @property
    def file_tools(self):
        """Lazy load FileTools only when needed."""
        if self._file_tools is None:
            try:
                from bin.file_tools_manager import FileToolsManager
                self._file_tools = FileToolsManager()
                logger.info("✅ File Tools Manager initialized")
            except ImportError as e:
                logger.warning(f"⚠️ File Tools Manager not available: {e}")
                # Create a minimal placeholder
                self._file_tools = type('FileToolsManager', (), {
                    'get_file_metadata': lambda *args, **kwargs: {"error": "File Tools Manager not available"},
                    'analyze_file': lambda *args, **kwargs: {"error": "File Tools Manager not available"},
                    'extract_archive': lambda *args, **kwargs: {"success": False, "error": "File Tools Manager not available"},
                    'find_files': lambda *args, **kwargs: []
                })()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._file_tools
    
    @property
    def compression_tools(self):
        """Lazy load CompressionTools only when needed."""
        if self._compression_tools is None:
            # For now, create a placeholder - you can add the actual CompressionTools class later
            self._compression_tools = type('CompressionTools', (), {})()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._compression_tools
    
    @property
    def ml_tools(self):
        """Lazy load MLTools only when needed."""
        if self._ml_tools is None:
            self._ml_tools = MLTools()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._ml_tools
    
    @property
    def nlp_tools(self):
        """Lazy load LocalNLP only when needed."""
        if self._nlp_tools is None:
            self._nlp_tools = LocalNLP()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._nlp_tools
    
    @property
    def context_memory(self):
        """Lazy load ContextMemoryManager only when needed."""
        if self._context_memory is None:
            self._context_memory = ContextMemoryManager()
            # Trigger dynamic tool discovery only if MCP server exists
            if self._mcp_server:
                self._mcp_server.discover_available_tools()
        return self._context_memory
    
    @property
    def pcap_analysis_tools(self):
        """Lazy load PCAPAnalysisMCPIntegrationLayer only when needed."""
        if not hasattr(self, '_pcap_analysis_tools') or self._pcap_analysis_tools is None:
            try:
                from pcap_analysis_mcp_integration import get_pcap_analysis_mcp_integration
                self._pcap_analysis_tools = get_pcap_analysis_mcp_integration()
                # Trigger dynamic tool discovery only if MCP server exists
                if self._mcp_server:
                    self._mcp_server.discover_available_tools()
            except ImportError as e:
                logger.warning(f"PCAP analysis tools not available: {e}")
                self._pcap_analysis_tools = None
        return self._pcap_analysis_tools
    
    def get_mcp_tools(self, category=None, tags=None, detailed=False, force_discovery=False):
        """Get MCP tools through the server with optional dynamic discovery."""
        if force_discovery:
            return self.mcp_server.get_dynamic_tools(force_discovery=True)
        return self.mcp_server._list_tools_handler(
            category=category, 
            tags=tags, 
            detailed=detailed
        )
    
    async def call_mcp_tool(self, tool_name, arguments):
        """Call an MCP tool through the server."""
        return await self.mcp_server._call_tool_handler(
            name=tool_name, 
            arguments=arguments
        )
    
    def discover_all_tools(self):
        """Force discovery of all available tools."""
        if self._mcp_server:
            self._mcp_server.discover_available_tools()
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of which tools have been initialized."""
        return {
            "dataframe_manager": self._df_manager is not None,
            "sqlite_manager": self._sqlite_manager is not None,
            "neo4j_manager": self._neo4j_manager is not None,
            "file_tools": self._file_tools is not None,
            "compression_tools": self._compression_tools is not None,
            "ml_tools": self._ml_tools is not None,
            "nlp_tools": self._nlp_tools is not None,
            "context_memory": self._context_memory is not None,
            "pcap_analysis_tools": hasattr(self, '_pcap_analysis_tools') and self._pcap_analysis_tools is not None,
            "mcp_server": self._mcp_server is not None
        }

# Initialize tool manager (lazy loading)
tool_manager = ToolManager()

# Backward compatibility functions
def get_df_manager():
    """Get DataFrameManager instance (lazy loaded)."""
    return tool_manager.df_manager

def get_sqlite_manager():
    """Get SQLiteManager instance (lazy loaded)."""
    return tool_manager.sqlite_manager

def get_neo4j_manager():
    """Get Neo4jManager instance (lazy loaded)."""
    return tool_manager.neo4j_manager

def get_file_tools():
    """Get FileTools instance (lazy loaded)."""
    return tool_manager.file_tools

def get_compression_tools():
    """Get CompressionTools instance (lazy loaded)."""
    return tool_manager.compression_tools

def get_ml_tools():
    """Get MLTools instance (lazy loaded)."""
    return tool_manager.ml_tools

def get_nlp_tools():
    """Get LocalNLP instance (lazy loaded)."""
    return tool_manager.nlp_tools

def get_context_memory():
    """Get ContextMemoryManager instance (lazy loaded)."""
    return tool_manager.context_memory

def get_pcap_analysis_tools():
    """Get PCAPAnalysisMCPIntegrationLayer instance (lazy loaded)."""
    return tool_manager.pcap_analysis_tools

# MCP Tools for backward compatibility
MCP_TOOLS = {
    "create_dataframe": {
        "description": "Create a new DataFrame",
        "parameters": {
            "name": "Name for the DataFrame",
            "columns": "List of column names",
            "data": "Optional initial data"
        }
    },
    "query_dataframe": {
        "description": "Query DataFrame using natural language or pandas syntax",
        "parameters": {
            "name": "Name of the DataFrame to query",
            "query": "Query string (natural language or pandas syntax)"
        }
    }
}

# MCP wrapper functions for backward compatibility
def mcp_create_dataframe(name: str, columns: List[str], data: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
    """MCP wrapper for create_dataframe."""
    return tool_manager.df_manager.create_dataframe(name, columns, data)

def mcp_query_dataframe(name: str, query: str) -> Dict[str, Any]:
    """MCP wrapper for query_dataframe."""
    return tool_manager.df_manager.query_dataframe(name, query)

def mcp_list_dataframes() -> Dict[str, Any]:
    """MCP wrapper for list_dataframes."""
    return tool_manager.df_manager.list_dataframes()

def mcp_export_dataframe(name: str, file_path: str = None, format: str = "csv") -> Dict[str, Any]:
    """MCP wrapper for exporting dataframes."""
    return tool_manager.df_manager.export_dataframe(name, file_path, format)

def mcp_delete_dataframe(name: str) -> Dict[str, Any]:
    """MCP wrapper for deleting dataframes."""
    return tool_manager.df_manager.delete_dataframe(name)

def mcp_get_dataframe_info(name: str) -> Dict[str, Any]:
    """MCP wrapper for getting dataframe information."""
    return tool_manager.df_manager.get_dataframe_info(name)

def mcp_get_schema(name: str) -> Dict[str, Any]:
    """MCP wrapper for get_schema."""
    return tool_manager.df_manager.get_schema(name)

class LogLevel(Enum):
    """Log levels for session logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Individual log entry with metadata."""
    timestamp: str
    level: str
    category: str
    action: str
    details: Dict[str, Any]
    session_id: str
    agent_type: Optional[str] = None
    workflow_step: Optional[str] = None
    execution_id: Optional[str] = None
    parent_action: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Convert any additional kwargs to metadata
        if self.metadata is None:
            self.metadata = {}

class SessionLogger:
    """Comprehensive session logging system for agentic workflows."""
    
    def __init__(self, session_name: str = None, max_sessions: int = 200):
        self.session_id = str(uuid.uuid4())
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now(timezone.utc)
        self.max_sessions = max_sessions
        
        # Create session-logs directory
        self.logs_dir = Path("session-logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create session-specific output directory
        self.output_dir = Path("session-outputs") / f"{self.session_name}_{self.session_id[:8]}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session log file
        self.session_file = self.logs_dir / f"{self.session_name}_{self.session_id[:8]}.json"
        
        # Track outputs generated in this session
        self.outputs_generated = []
        self.output_count = 0
        
        # Initialize log structure
        self.session_log = {
            "session_metadata": {
                "session_id": self.session_id,
                "session_name": self.session_name,
                "start_time": self.start_time.isoformat(),
                "version": "1.0.0",
                "framework": "CS-AI Dynamic Agentic Workflow"
            },
            "agent_interactions": [],
            "workflow_executions": [],
            "tool_calls": [],
            "data_operations": [],
            "decision_points": [],
            "errors": [],
            "performance_metrics": {
                "total_tool_calls": 0,
                "total_workflow_steps": 0,
                "total_errors": 0,
                "session_duration_ms": 0
            }
        }
        
        # Cleanup old sessions
        self._cleanup_old_sessions()
        
        # Log session start
        self.log_info("session_start", "Session logging initialized")
    
    def _cleanup_old_sessions(self):
        """Remove old session files, keeping only max_sessions."""
        try:
            session_files = sorted(
                self.logs_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if len(session_files) > self.max_sessions:
                files_to_delete = session_files[self.max_sessions:]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        print(f"🗑️  Cleaned up old session log: {file_path.name}")
                    except Exception as e:
                        print(f"⚠️  Failed to delete old session log {file_path.name}: {e}")
        except Exception as e:
            print(f"⚠️  Session cleanup failed: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def _create_log_entry(self, level: str, category: str, action: str, 
                         details: Dict[str, Any], **kwargs) -> LogEntry:
        """Create a standardized log entry."""
        return LogEntry(
            timestamp=self._get_timestamp(),
            level=level,
            category=category,
            action=action,
            details=details,
            session_id=self.session_id,
            **kwargs
        )
    
    def _save_session_log(self):
        """Save the current session log to file."""
        try:
            # Update session duration
            current_time = datetime.now(timezone.utc)
            duration = (current_time - self.start_time).total_seconds() * 1000
            self.session_log["session_metadata"]["end_time"] = current_time.isoformat()
            self.session_log["session_metadata"]["duration_ms"] = duration
            self.session_log["performance_metrics"]["session_duration_ms"] = duration
            
            # Save to file
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_log, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"❌ Failed to save session log: {e}")
    
    def log_debug(self, action: str, details: Union[str, Dict[str, Any]], **kwargs):
        """Log debug information."""
        if isinstance(details, str):
            details = {"message": details}
        
        entry = self._create_log_entry("debug", "debug", action, details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_info(self, action: str, details: Union[str, Dict[str, Any]], **kwargs):
        """Log informational messages."""
        if isinstance(details, str):
            details = {"message": details}
        
        entry = self._create_log_entry("info", "info", action, details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_warning(self, action: str, details: Union[str, Dict[str, Any]], **kwargs):
        """Log warning messages."""
        if isinstance(details, str):
            details = {"message": details}
        
        entry = self._create_log_entry("warning", "warning", action, details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_error(self, action: str, details: Union[str, Dict[str, Any]], **kwargs):
        """Log error messages."""
        if isinstance(details, str):
            details = {"message": details}
        
        entry = self._create_log_entry("error", "error", action, details, **kwargs)
        self.session_log["errors"].append(asdict(entry))
        self.session_log["performance_metrics"]["total_errors"] += 1
        self._save_session_log()
    
    def log_agent_question(self, question: str, agent_type: str = "user", 
                          context: Dict[str, Any] = None, **kwargs):
        """Log questions asked by agents or users."""
        details = {
            "question": question,
            "context": context or {},
            "agent_type": agent_type
        }
        
        entry = self._create_log_entry("info", "agent_question", "question_asked", 
                                     details, agent_type=agent_type, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_agent_response(self, response: str, agent_type: str = "assistant",
                          question_context: str = None, **kwargs):
        """Log responses given by agents."""
        details = {
            "response": response,
            "question_context": question_context,
            "agent_type": agent_type
        }
        
        entry = self._create_log_entry("info", "agent_response", "response_given", 
                                     details, agent_type=agent_type, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_workflow_planning(self, plan: Dict[str, Any], agent_type: str = "planner",
                             reasoning: str = None, **kwargs):
        """Log workflow planning activities."""
        details = {
            "plan": plan,
            "reasoning": reasoning,
            "agent_type": agent_type
        }
        
        entry = self._create_log_entry("info", "workflow_planning", "plan_created", 
                                     details, agent_type=agent_type, **kwargs)
        self.session_log["workflow_executions"].append(asdict(entry))
        self.session_log["performance_metrics"]["total_workflow_steps"] += 1
        self._save_session_log()
    
    def log_workflow_execution(self, step: str, action: str, details: Dict[str, Any],
                              execution_id: str = None, **kwargs):
        """Log individual workflow execution steps."""
        details.update({
            "step": step,
            "execution_id": execution_id or str(uuid.uuid4())
        })
        
        entry = self._create_log_entry("info", "workflow_execution", action, 
                                     details, workflow_step=step, 
                                     execution_id=execution_id, **kwargs)
        self.session_log["workflow_executions"].append(asdict(entry))
        self._save_session_log()
    
    def log_plan_change(self, original_plan: Dict[str, Any], new_plan: Dict[str, Any],
                       reason: str, agent_type: str = "planner", **kwargs):
        """Log changes in workflow plans."""
        details = {
            "original_plan": original_plan,
            "reason": reason,
            "agent_type": agent_type
        }
        
        entry = self._create_log_entry("info", "workflow_planning", "plan_modified", 
                                     details, agent_type=agent_type, **kwargs)
        self.session_log["workflow_executions"].append(asdict(entry))
        self._save_session_log()
    
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], 
                      result: Any, duration_ms: float = None, success: bool = True,
                      error_message: str = None, **kwargs):
        """Log tool calls and their results."""
        details = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "success": success,
            "error_message": error_message
        }
        
        entry = self._create_log_entry("info", "tool_execution", "tool_called", 
                                     details, duration_ms=duration_ms, 
                                     success=success, error_message=error_message, **kwargs)
        self.session_log["tool_calls"].append(asdict(entry))
        self.session_log["performance_metrics"]["total_tool_calls"] += 1
        self._save_session_log()
    
    def log_data_operation(self, operation: str, data_type: str, data_source: str,
                          data_summary: Dict[str, Any], **kwargs):
        """Log data operations (pull, transform, store)."""
        details = {
            "operation": operation,
            "data_type": data_type,
            "data_source": data_source,
            "data_summary": data_summary
        }
        
        entry = self._create_log_entry("info", "data_operation", operation, 
                                     details, **kwargs)
        self.session_log["data_operations"].append(asdict(entry))
        self._save_session_log()
    
    def log_decision_point(self, decision: str, options: List[str], 
                          selected_option: str, reasoning: str, **kwargs):
        """Log decision points in the workflow."""
        details = {
            "decision": decision,
            "options": options,
            "selected_option": selected_option,
            "reasoning": reasoning
        }
        
        entry = self._create_log_entry("info", "decision_making", "decision_made", 
                                     details, **kwargs)
        self.session_log["decision_points"].append(asdict(entry))
        self._save_session_log()
    
    def log_variable_change(self, variable_name: str, old_value: Any, new_value: Any,
                           context: str = None, **kwargs):
        """Log changes in variables during execution."""
        details = {
            "variable_name": variable_name,
            "old_value": old_value,
            "new_value": new_value,
            "context": context
        }
        
        entry = self._create_log_entry("debug", "variable_tracking", "variable_changed", 
                                     details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_logic_step(self, logic_type: str, input_data: Any, output_data: Any,
                       algorithm: str = None, **kwargs):
        """Log logical reasoning steps."""
        details = {
            "logic_type": logic_type,
            "input_data": input_data,
            "output_data": output_data,
            "algorithm": algorithm
        }
        
        entry = self._create_log_entry("debug", "variable_tracking", "logic_applied", 
                                     details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def log_performance_metric(self, metric_name: str, value: Any, unit: str = None, **kwargs):
        """Log performance metrics."""
        details = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit
        }
        
        entry = self._create_log_entry("info", "performance", "metric_recorded", 
                                     details, **kwargs)
        self.session_log["agent_interactions"].append(asdict(entry))
        self._save_session_log()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "start_time": self.start_time.isoformat(),
            "current_duration_ms": (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000,
            "log_file": str(self.session_file),
            "performance_metrics": self.session_log["performance_metrics"],
            "total_interactions": len(self.session_log["agent_interactions"]),
            "total_workflow_steps": len(self.session_log["workflow_executions"]),
            "total_tool_calls": len(self.session_log["tool_calls"]),
            "total_errors": len(self.session_log["errors"])
        }
    
    def end_session(self, summary: str = None):
        """End the session and save final log."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds() * 1000
        
        # Log session end
        self.log_info("session_end", f"Session ended: {summary or 'Normal completion'}")
        
        # Update final metadata
        self.session_log["session_metadata"]["end_time"] = end_time.isoformat()
        self.session_log["session_metadata"]["duration_ms"] = duration
        self.session_log["session_metadata"]["status"] = "completed"
        
        # Save final log
        self._save_session_log()
        
        print(f"📝 Session log saved: {self.session_file}")
        print(f"📊 Session summary: {self.get_session_summary()}")
    
    def export_session_log(self, format: str = "json") -> str:
        """Export session log in specified format."""
        if format.lower() == "json":
            return json.dumps(self.session_log, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_output_file(self, filename: str, content: str = None, binary_content: bytes = None, 
                          description: str = None, category: str = "general") -> str:
        """Create an output file in the session-specific directory."""
        try:
            # Generate timestamped filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output_{self.output_count:03d}_{timestamp}.txt"
            
            # Ensure filename has extension
            if not Path(filename).suffix:
                filename += ".txt"
            
            # Create full file path
            file_path = self.output_dir / filename
            
            # Write content to file
            if binary_content:
                with open(file_path, 'wb') as f:
                    f.write(binary_content)
            elif content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Create empty file
                file_path.touch()
            
            # Track the output
            output_info = {
                "filename": filename,
                "file_path": str(file_path),
                "description": description or f"Output file {self.output_count}",
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "size_bytes": file_path.stat().st_size
            }
            
            self.outputs_generated.append(output_info)
            self.output_count += 1
            
            # Log the output creation
            self.log_info("output_created", f"Output file created: {filename}", 
                         metadata={"output_info": output_info})
            
            return str(file_path)
            
        except Exception as e:
            self.log_error("output_creation_failed", f"Failed to create output file: {e}")
            return None
    
    def get_session_outputs(self) -> List[Dict[str, Any]]:
        """Get list of all outputs generated in this session."""
        return self.outputs_generated.copy()
    
    def cleanup_session_outputs(self):
        """Clean up session output directory if no outputs were generated."""
        if not self.outputs_generated:
            try:
                import shutil
                shutil.rmtree(self.output_dir)
                self.log_info("outputs_cleaned", "Session output directory cleaned up (no outputs generated)")
            except Exception as e:
                self.log_error("cleanup_failed", f"Failed to cleanup output directory: {e}")
    
    def end_session_with_cleanup(self):
        """End session and cleanup if no outputs were generated."""
        if not self.outputs_generated:
            self.cleanup_session_outputs()
        self.end_session()

# Global session logger instance
session_logger = SessionLogger()

class ContextMemoryManager:
    """Advanced context memory manager with knowledge graph and human-like memory hierarchy."""
    
    def __init__(self, memory_db_path: str = "context_memory.db"):
        self.memory_db_path = memory_db_path
        self.short_term_memory = {}  # Session-based, in-memory
        self.medium_term_memory = {}  # Workflow context, persistent
        self.long_term_memory = {}    # Organizational knowledge, persistent
        
        # Memory TTL settings (in seconds)
        self.ttl_settings = {
            "short_term": 4 * 60 * 60,      # 4 hours
            "medium_term": 7 * 24 * 60 * 60, # 7 days
            "long_term": 30 * 24 * 60 * 60   # 30 days
        }
        
        # Initialize knowledge graph database
        self._init_memory_database()
        
        # Load existing memories
        self._load_persistent_memories()
        
        # Session tracking
        self.current_session_id = None
        self.session_start_time = None
    
    def _init_memory_database(self):
        """Initialize SQLite database for persistent memory storage."""
        try:
            import sqlite3
            self.conn = sqlite3.connect(self.memory_db_path)
            self.cursor = self.conn.cursor()
            
            # Create memory tables
            self.cursor.executescript("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT UNIQUE NOT NULL,
                    node_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance_score REAL DEFAULT 0.5,
                    access_frequency INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_category TEXT NOT NULL,
                    expires_at TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_node_id) REFERENCES memory_nodes (node_id),
                    FOREIGN KEY (target_node_id) REFERENCES memory_nodes (node_id)
                );
                
                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    session_id TEXT,
                    workflow_id TEXT,
                    context TEXT,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES memory_nodes (node_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_type ON memory_nodes (node_type);
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_ttl ON memory_nodes (ttl_category);
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_expires ON memory_nodes (expires_at);
                CREATE INDEX IF NOT EXISTS idx_memory_relationships_source ON memory_relationships (source_node_id);
                CREATE INDEX IF NOT EXISTS idx_memory_relationships_target ON memory_relationships (target_node_id);
            """)
            
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Failed to initialize memory database: {e}")
            self.conn = None
            self.cursor = None
    
    def _load_persistent_memories(self):
        """Load medium and long-term memories from database."""
        if not self.conn:
            return
        
        try:
            # Load medium-term memories
            self.cursor.execute("""
                SELECT node_id, node_type, content, metadata, importance_score, 
                       access_frequency, last_accessed, expires_at
                FROM memory_nodes 
                WHERE ttl_category = 'medium_term' AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY importance_score DESC, access_frequency DESC
            """)
            
            for row in self.cursor.fetchall():
                node_id, node_type, content, metadata, importance_score, \
                access_frequency, last_accessed, expires_at = row
                
                self.medium_term_memory[node_id] = {
                    "node_type": node_type,
                    "content": content,
                    "metadata": json.loads(metadata) if metadata else {},
                    "importance_score": importance_score,
                    "access_frequency": access_frequency,
                    "last_accessed": last_accessed,
                    "expires_at": expires_at
                }
            
            # Load long-term memories
            self.cursor.execute("""
                SELECT node_id, node_type, content, metadata, importance_score, 
                       access_frequency, last_accessed, expires_at
                FROM memory_nodes 
                WHERE ttl_category = 'long_term' AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY importance_score DESC, access_frequency DESC
            """)
            
            for row in self.cursor.fetchall():
                node_id, node_type, content, metadata, importance_score, \
                access_frequency, last_accessed, expires_at = row
                
                self.long_term_memory[node_id] = {
                    "node_type": node_type,
                    "content": content,
                    "metadata": json.loads(metadata) if metadata else {},
                    "importance_score": importance_score,
                    "access_frequency": access_frequency,
                    "last_accessed": last_accessed,
                    "expires_at": expires_at
                }
                
        except Exception as e:
            print(f"❌ Failed to load persistent memories: {e}")
    
    def start_session(self, session_id: str):
        """Start a new memory session for short-term memory."""
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        
        # Clear expired short-term memories
        self._cleanup_expired_short_term_memory()
        
        # Log session start
        self._log_memory_access("session_start", session_id, "session_management")
    
    def end_session(self):
        """End current session and cleanup short-term memory."""
        if self.current_session_id:
            # Move important short-term memories to medium-term if needed
            self._promote_important_memories()
            
            # Clear short-term memory
            self.short_term_memory.clear()
            
            # Log session end
            self._log_memory_access("session_end", self.current_session_id, "session_management")
            
            self.current_session_id = None
            self.session_start_time = None
    
    def add_memory(self, node_id: str, node_type: str, content: str, 
                   ttl_category: str = "short_term", metadata: dict = None,
                   importance_score: float = 0.5, relationships: list = None) -> bool:
        """Add a new memory node to the knowledge graph."""
        try:
            memory_data = {
                "node_type": node_type,
                "content": content,
                "metadata": metadata or {},
                "importance_score": max(0.0, min(1.0, importance_score)),
                "access_frequency": 0,
                "last_accessed": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "expires_at": self._calculate_expiry(ttl_category)
            }
            
            # Add to appropriate memory store
            if ttl_category == "short_term":
                self.short_term_memory[node_id] = memory_data
            elif ttl_category == "medium_term":
                self.medium_term_memory[node_id] = memory_data
                self._persist_memory_node(node_id, memory_data, ttl_category)
            elif ttl_category == "long_term":
                self.long_term_memory[node_id] = memory_data
                self._persist_memory_node(node_id, memory_data, ttl_category)
            else:
                raise ValueError(f"Invalid TTL category: {ttl_category}")
            
            # Add relationships if provided
            if relationships:
                for rel in relationships:
                    self._add_relationship(node_id, rel["target"], rel["type"], rel.get("strength", 1.0))
            
            # Log memory creation
            self._log_memory_access("memory_created", node_id, "memory_management")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add memory: {e}")
            return False
    
    def get_memory(self, node_id: str, include_relationships: bool = True) -> dict:
        """Retrieve a memory node by ID."""
        # Check short-term memory first
        if node_id in self.short_term_memory:
            memory = self.short_term_memory[node_id].copy()
            memory["source"] = "short_term"
            self._update_access_stats(node_id, "short_term")
            return memory
        
        # Check medium-term memory
        if node_id in self.medium_term_memory:
            memory = self.medium_term_memory[node_id].copy()
            memory["source"] = "medium_term"
            self._update_access_stats(node_id, "medium_term")
            return memory
        
        # Check long-term memory
        if node_id in self.long_term_memory:
            memory = self.long_term_memory[node_id].copy()
            memory["source"] = "long_term"
            self._update_access_stats(node_id, "long_term")
            return memory
        
        return None
    
    def search_memories(self, query: str, node_types: list = None, 
                       ttl_categories: list = None, min_importance: float = 0.0,
                       max_results: int = 50) -> list:
        """Search memories using semantic similarity and filters."""
        try:
            results = []
            
            # Search short-term memory
            if not ttl_categories or "short_term" in ttl_categories:
                for node_id, memory in self.short_term_memory.items():
                    if self._matches_search_criteria(memory, query, node_types, min_importance):
                        memory_copy = memory.copy()
                        memory_copy["node_id"] = node_id
                        memory_copy["source"] = "short_term"
                        memory_copy["relevance_score"] = self._calculate_relevance(memory, query)
                        results.append(memory_copy)
            
            # Search medium-term memory
            if not ttl_categories or "medium_term" in ttl_categories:
                for node_id, memory in self.medium_term_memory.items():
                    if self._matches_search_criteria(memory, query, node_types, min_importance):
                        memory_copy = memory.copy()
                        memory_copy["node_id"] = node_id
                        memory_copy["source"] = "medium_term"
                        memory_copy["relevance_score"] = self._calculate_relevance(memory, query)
                        results.append(memory_copy)
            
            # Search long-term memory
            if not ttl_categories or "long_term" in ttl_categories:
                for node_id, memory in self.long_term_memory.items():
                    if self._matches_search_criteria(memory, query, node_types, min_importance):
                        memory_copy = memory.copy()
                        memory_copy["node_id"] = node_id
                        memory_copy["source"] = "long_term"
                        memory_copy["relevance_score"] = self._calculate_relevance(memory, query)
                        results.append(memory_copy)
            
            # Sort by relevance and importance
            results.sort(key=lambda x: (x["relevance_score"], x["importance_score"]), reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"❌ Memory search failed: {e}")
            return []
    
    def get_context_for_workflow(self, workflow_context: str, max_nodes: int = 20) -> dict:
        """Get relevant context nodes for a specific workflow or task."""
        try:
            # Search for relevant memories
            relevant_memories = self.search_memories(
                query=workflow_context,
                max_results=max_nodes * 2  # Get more to filter
            )
            
            # Group by source and select top relevant
            context = {
                "short_term": [],
                "medium_term": [],
                "long_term": [],
                "relationships": [],
                "summary": {
                    "total_nodes": 0,
                    "node_types": {},
                    "importance_distribution": {"low": 0, "medium": 0, "high": 0}
                }
            }
            
            for memory in relevant_memories[:max_nodes]:
                source = memory["source"]
                context[source].append({
                    "node_id": memory.get("node_id", "unknown"),
                    "node_type": memory["node_type"],
                    "content": memory["content"],
                    "importance_score": memory["importance_score"],
                    "relevance_score": memory["relevance_score"],
                    "metadata": memory["metadata"]
                })
                
                # Update summary
                context["summary"]["total_nodes"] += 1
                context["summary"]["node_types"][memory["node_type"]] = \
                    context["summary"]["node_types"].get(memory["node_type"], 0) + 1
                
                # Categorize importance
                if memory["importance_score"] < 0.33:
                    context["summary"]["importance_distribution"]["low"] += 1
                elif memory["importance_score"] < 0.67:
                    context["summary"]["importance_distribution"]["medium"] += 1
                else:
                    context["summary"]["importance_distribution"]["high"] += 1
            
            # Get relationships for context nodes
            context["relationships"] = self._get_context_relationships(context)
            
            return context
            
        except Exception as e:
            print(f"❌ Failed to get workflow context: {e}")
            return {"error": str(e)}
    
    def suggest_memory_promotion(self, workflow_output: dict, 
                                context_used: dict) -> dict:
        """Suggest which workflow outputs should be promoted to longer-term memory."""
        try:
            suggestions = {
                "promote_to_medium": [],
                "promote_to_long": [],
                "reasoning": []
            }
            
            # Analyze workflow output for potential long-term value
            if "dataframes" in workflow_output:
                for df_name, df_info in workflow_output["dataframes"].items():
                    if df_info.get("rows", 0) > 100:  # Large datasets
                        suggestions["promote_to_medium"].append({
                            "type": "dataframe",
                            "name": df_name,
                            "reason": "Large dataset with potential reuse value",
                            "suggested_ttl": "medium_term"
                        })
            
            if "analysis_results" in workflow_output:
                for result in workflow_output["analysis_results"]:
                    if result.get("confidence", 0) > 0.8:  # High confidence results
                        suggestions["promote_to_long"].append({
                            "type": "analysis_result",
                            "content": result.get("summary", "High confidence analysis"),
                            "reason": "High confidence result with organizational value",
                            "suggested_ttl": "long_term"
                        })
            
            if "policy_mappings" in workflow_output:
                suggestions["promote_to_long"].append({
                    "type": "policy_mapping",
                    "content": "Policy compliance mapping results",
                    "reason": "Organizational policy information",
                    "suggested_ttl": "long_term"
                })
            
            # Add reasoning based on context usage
            if context_used.get("long_term"):
                suggestions["reasoning"].append(
                    "Long-term context was used, suggesting similar outputs may have long-term value"
                )
            
            return suggestions
            
        except Exception as e:
            print(f"❌ Memory promotion suggestion failed: {e}")
            return {"error": str(e)}
    
    def promote_memory(self, node_id: str, new_ttl_category: str, 
                       reason: str = None) -> bool:
        """Promote a memory node to a longer TTL category."""
        try:
            # Find the memory in current stores
            memory = None
            current_category = None
            
            if node_id in self.short_term_memory:
                memory = self.short_term_memory[node_id]
                current_category = "short_term"
                del self.short_term_memory[node_id]
            elif node_id in self.medium_term_memory:
                memory = self.medium_term_memory[node_id]
                current_category = "medium_term"
                del self.medium_term_memory[node_id]
            elif node_id in self.long_term_memory:
                memory = self.long_term_memory[node_id]
                current_category = "long_term"
            else:
                return False
            
            # Update memory with new TTL
            memory["ttl_category"] = new_ttl_category
            memory["expires_at"] = self._calculate_expiry(new_ttl_category)
            memory["metadata"]["promotion_history"] = memory["metadata"].get("promotion_history", [])
            memory["metadata"]["promotion_history"].append({
                "from": current_category,
                "to": new_ttl_category,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
            
            # Add to new category
            if new_ttl_category == "medium_term":
                self.medium_term_memory[node_id] = memory
                self._persist_memory_node(node_id, memory, new_ttl_category)
            elif new_ttl_category == "long_term":
                self.long_term_memory[node_id] = memory
                self._persist_memory_node(node_id, memory, new_ttl_category)
            
            # Log promotion
            self._log_memory_access("memory_promoted", node_id, "memory_management", 
                                  metadata={"from": current_category, "to": new_ttl_category, "reason": reason})
            
            return True
            
        except Exception as e:
            print(f"❌ Memory promotion failed: {e}")
            return False
    
    def _calculate_expiry(self, ttl_category: str) -> str:
        """Calculate expiry timestamp for TTL category."""
        if ttl_category == "short_term":
            return (datetime.now() + timedelta(seconds=self.ttl_settings["short_term"])).isoformat()
        elif ttl_category == "medium_term":
            return (datetime.now() + timedelta(seconds=self.ttl_settings["medium_term"])).isoformat()
        elif ttl_category == "long_term":
            return (datetime.now() + timedelta(seconds=self.ttl_settings["long_term"])).isoformat()
        else:
            return None
    
    def _matches_search_criteria(self, memory: dict, query: str, 
                               node_types: list, min_importance: float) -> bool:
        """Check if memory matches search criteria."""
        # Check node type filter
        if node_types and memory["node_type"] not in node_types:
            return False
        
        # Check importance filter
        if memory["importance_score"] < min_importance:
            return False
        
        # Check content relevance (simple text matching for now)
        query_lower = query.lower()
        content_lower = memory["content"].lower()
        
        # Check for exact matches or key terms
        if query_lower in content_lower or any(term in content_lower for term in query_lower.split()):
            return True
        
        # Check metadata for matches
        metadata_str = str(memory["metadata"]).lower()
        if query_lower in metadata_str:
            return True
        
        return False
    
    def _calculate_relevance(self, memory: dict, query: str) -> float:
        """Calculate relevance score for search results."""
        # Simple relevance scoring (can be enhanced with embeddings)
        relevance = 0.0
        
        # Content match
        query_terms = set(query.lower().split())
        content_terms = set(memory["content"].lower().split())
        
        if query_terms & content_terms:  # Intersection
            relevance += 0.4
        
        # Importance boost
        relevance += memory["importance_score"] * 0.3
        
        # Recency boost - check if source exists
        if "source" in memory:
            if memory["source"] == "short_term":
                relevance += 0.2
            elif memory["source"] == "medium_term":
                relevance += 0.1
        
        # Access frequency boost
        relevance += min(memory["access_frequency"] * 0.01, 0.1)
        
        return min(relevance, 1.0)
    
    def _persist_memory_node(self, node_id: str, memory: dict, ttl_category: str):
        """Persist memory node to database."""
        if not self.conn:
            return
        
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO memory_nodes 
                (node_id, node_type, content, metadata, importance_score, 
                 access_frequency, last_accessed, ttl_category, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id, memory["node_type"], memory["content"],
                json.dumps(memory["metadata"]), memory["importance_score"],
                memory["access_frequency"], memory["last_accessed"],
                ttl_category, memory["expires_at"]
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Failed to persist memory node: {e}")
    
    def _add_relationship(self, source_id: str, target_id: str, 
                         relationship_type: str, strength: float = 1.0):
        """Add relationship between memory nodes."""
        if not self.conn:
            return
        
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO memory_relationships 
                (source_node_id, target_node_id, relationship_type, strength)
                VALUES (?, ?, ?, ?)
            """, (source_id, target_id, relationship_type, strength))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Failed to add relationship: {e}")
    
    def _get_context_relationships(self, context: dict) -> list:
        """Get relationships for context nodes."""
        if not self.conn:
            return []
        
        try:
            # Get all node IDs from context
            node_ids = []
            for category in ["short_term", "medium_term", "long_term"]:
                node_ids.extend([node["node_id"] for node in context[category]])
            
            if not node_ids:
                return []
            
            # Query relationships
            # Use parameterized query to prevent SQL injection
            placeholders = ",".join(["?" for _ in node_ids])
            query = f"""
                SELECT source_node_id, target_node_id, relationship_type, strength
                FROM memory_relationships 
                WHERE source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders})
            """
            self.cursor.execute(query, node_ids + node_ids)
            
            relationships = []
            for row in self.cursor.fetchall():
                source_id, target_id, rel_type, strength = row
                relationships.append({
                    "source": source_id,
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "strength": strength
                })
            
            return relationships
            
        except Exception as e:
            print(f"❌ Failed to get context relationships: {e}")
            return []
    
    def _update_access_stats(self, node_id: str, source: str):
        """Update access statistics for a memory node."""
        try:
            if source == "short_term" and node_id in self.short_term_memory:
                self.short_term_memory[node_id]["access_frequency"] += 1
                self.short_term_memory[node_id]["last_accessed"] = datetime.now().isoformat()
            elif source == "medium_term" and node_id in self.medium_term_memory:
                self.medium_term_memory[node_id]["access_frequency"] += 1
                self.medium_term_memory[node_id]["last_accessed"] = datetime.now().isoformat()
                self._persist_memory_node(node_id, self.medium_term_memory[node_id], "medium_term")
            elif source == "long_term" and node_id in self.medium_term_memory:
                self.long_term_memory[node_id]["access_frequency"] += 1
                self.long_term_memory[node_id]["last_accessed"] = datetime.now().isoformat()
                self._persist_memory_node(node_id, self.long_term_memory[node_id], "long_term")
            
            # Log access
            self._log_memory_access("memory_accessed", node_id, "memory_access")
            
        except Exception as e:
            print(f"❌ Failed to update access stats: {e}")
    
    def _log_memory_access(self, access_type: str, node_id: str, context: str, 
                          metadata: dict = None):
        """Log memory access for analytics."""
        if not self.conn:
            return
        
        try:
            self.cursor.execute("""
                INSERT INTO memory_access_log 
                (node_id, access_type, session_id, workflow_id, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                node_id, access_type, self.current_session_id, 
                metadata.get("workflow_id") if metadata else None, context
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Failed to log memory access: {e}")
    
    def _cleanup_expired_short_term_memory(self):
        """Clean up expired short-term memories."""
        if not self.session_start_time:
            return
        
        current_time = datetime.now()
        expired_nodes = []
        
        for node_id, memory in self.short_term_memory.items():
            created_at = datetime.fromisoformat(memory["created_at"])
            if (current_time - created_at).total_seconds() > self.ttl_settings["short_term"]:
                expired_nodes.append(node_id)
        
        for node_id in expired_nodes:
            del self.short_term_memory[node_id]
    
    def _promote_important_memories(self):
        """Promote important short-term memories to medium-term."""
        for node_id, memory in list(self.short_term_memory.items()):
            if memory["importance_score"] > 0.7 or memory["access_frequency"] > 5:
                self.promote_memory(node_id, "medium_term", 
                                  "Auto-promoted due to high importance/frequency")
    
    def get_memory_stats(self) -> dict:
        """Get comprehensive memory statistics."""
        return {
            "short_term": {
                "count": len(self.short_term_memory),
                "ttl_hours": self.ttl_settings["short_term"] / 3600
            },
            "medium_term": {
                "count": len(self.medium_term_memory),
                "ttl_days": self.ttl_settings["medium_term"] / (24 * 3600)
            },
            "long_term": {
                "count": len(self.long_term_memory),
                "ttl_days": self.ttl_settings["long_term"] / (24 * 3600)
            },
            "total_nodes": len(self.short_term_memory) + len(self.medium_term_memory) + len(self.long_term_memory),
            "current_session": self.current_session_id,
            "session_duration": (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0
        }
    
    def cleanup_expired_memories(self):
        """Clean up expired memories from all categories."""
        if not self.conn:
            return
        
        try:
            # Clean up expired medium and long-term memories
            self.cursor.execute("""
                DELETE FROM memory_nodes 
                WHERE expires_at IS NOT NULL AND expires_at < datetime('now')
            """)
            
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            
            # Reload memories to sync with database
            self._load_persistent_memories()
            
            return deleted_count
            
        except Exception as e:
            print(f"❌ Failed to cleanup expired memories: {e}")
            return 0
    
    def close(self):
        """Close database connection and cleanup."""
        if self.conn:
            self.conn.close()

class MLTools:
    """Machine Learning tools for cybersecurity analysis and anomaly detection."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.anomaly_thresholds = {}
        
    def detect_anomalies_isolation_forest(self, data, contamination=0.1, random_state=42):
        """Detect anomalies using Isolation Forest algorithm."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Reshape if 1D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
            predictions = iso_forest.fit_predict(data_scaled)
            
            # Get anomaly scores
            scores = iso_forest.score_samples(data_scaled)
            
            # Identify anomalies
            anomalies = data_array[predictions == -1]
            normal = data_array[predictions == 1]
            
            return {
                "success": True,
                "anomalies": anomalies.tolist(),
                "normal_data": normal.tolist(),
                "anomaly_scores": scores.tolist(),
                "anomaly_indices": np.where(predictions == -1)[0].tolist(),
                "contamination": contamination,
                "total_samples": len(data_array),
                "anomaly_count": len(anomalies)
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Install with: pip install scikit-learn"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_anomalies_lof(self, data, n_neighbors=20, contamination=0.1):
        """Detect anomalies using Local Outlier Factor algorithm."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Reshape if 1D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Fit LOF
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            predictions = lof.fit_predict(data_scaled)
            
            # Get outlier scores
            scores = lof.negative_outlier_factor_
            
            # Identify anomalies
            anomalies = data_array[predictions == -1]
            normal = data_array[predictions == 1]
            
            return {
                "success": True,
                "anomalies": anomalies.tolist(),
                "normal_data": normal.tolist(),
                "outlier_scores": scores.tolist(),
                "anomaly_indices": np.where(predictions == -1)[0].tolist(),
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                "total_samples": len(data_array),
                "anomaly_count": len(anomalies)
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Install with: pip install scikit-learn"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cluster_data_kmeans(self, data, n_clusters=3, random_state=42):
        """Cluster data using K-Means algorithm."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Reshape if 1D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(data_scaled)
            
            # Get cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Group data by clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_data = data_array[cluster_labels == i]
                clusters[f"cluster_{i}"] = {
                    "data": cluster_data.tolist(),
                    "count": len(cluster_data),
                    "center": cluster_centers[i].tolist()
                }
            
            return {
                "success": True,
                "clusters": clusters,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": cluster_centers.tolist(),
                "inertia": float(inertia),
                "n_clusters": n_clusters,
                "total_samples": len(data_array)
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Install with: pip install scikit-learn"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def find_optimal_clusters(self, data, max_clusters=10, random_state=42):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Reshape if 1D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Calculate metrics for different numbers of clusters
            inertias = []
            silhouette_scores = []
            cluster_range = range(2, min(max_clusters + 1, len(data_array) // 2))
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                cluster_labels = kmeans.fit_predict(data_scaled)
                
                inertias.append(kmeans.inertia_)
                
                if k > 1:  # Silhouette score requires at least 2 clusters
                    silhouette_scores.append(silhouette_score(data_scaled, cluster_labels))
                else:
                    silhouette_scores.append(0)
            
            # Find elbow point (approximate)
            elbow_k = self._find_elbow_point(cluster_range, inertias)
            
            # Find best silhouette score
            best_silhouette_k = cluster_range[np.argmax(silhouette_scores)]
            
            return {
                "success": True,
                "cluster_range": list(cluster_range),
                "inertias": [float(x) for x in inertias],
                "silhouette_scores": [float(x) for x in silhouette_scores],
                "elbow_recommendation": elbow_k,
                "silhouette_recommendation": best_silhouette_k,
                "recommendations": {
                    "elbow_method": elbow_k,
                    "silhouette_analysis": best_silhouette_k,
                    "final_recommendation": best_silhouette_k if best_silhouette_k else elbow_k
                }
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Install with: pip install scikit-learn"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _find_elbow_point(self, x, y):
        """Find elbow point using the second derivative method."""
        try:
            import numpy as np
            
            # Calculate second derivative
            dy = np.diff(y)
            d2y = np.diff(dy)
            
            # Find the point with maximum second derivative
            elbow_idx = np.argmax(np.abs(d2y)) + 1
            
            return x[elbow_idx] if elbow_idx < len(x) else x[-1]
            
        except Exception:
            # Fallback: return middle point
            return x[len(x) // 2]
    
    def extract_features_statistical(self, data):
        """Extract statistical features from data."""
        try:
            import numpy as np
            import pandas as pd
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Ensure 2D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            features = {}
            
            for col in range(data_array.shape[1]):
                col_data = data_array[:, col]
                col_name = f"feature_{col}"
                
                # Basic statistics
                features[col_name] = {
                    "mean": float(np.mean(col_data)),
                    "std": float(np.std(col_data)),
                    "min": float(np.min(col_data)),
                    "max": float(np.max(col_data)),
                    "median": float(np.median(col_data)),
                    "q25": float(np.percentile(col_data, 25)),
                    "q75": float(np.percentile(col_data, 75)),
                    "skewness": float(self._calculate_skewness(col_data)),
                    "kurtosis": float(self._calculate_kurtosis(col_data)),
                    "variance": float(np.var(col_data)),
                    "range": float(np.max(col_data) - np.min(col_data)),
                    "iqr": float(np.percentile(col_data, 75) - np.percentile(col_data, 25))
                }
            
            # Overall dataset features
            features["dataset"] = {
                "total_samples": int(data_array.shape[0]),
                "total_features": int(data_array.shape[1]),
                "missing_values": int(np.isnan(data_array).sum()),
                "zero_values": int((data_array == 0).sum()),
                "negative_values": int((data_array < 0).sum()),
                "positive_values": int((data_array > 0).sum())
            }
            
            return {
                "success": True, 
                "features": features,
                "total_features": len(features),
                "dataset": {
                    "total_samples": int(data_array.shape[0]),
                    "total_features": int(data_array.shape[1]),
                    "missing_values": int(np.isnan(data_array).sum()),
                    "zero_values": int((data_array == 0).sum()),
                    "negative_values": int((data_array < 0).sum()),
                    "positive_values": int((data_array > 0).sum())
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        try:
            import numpy as np
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)
            
            skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
            return skewness
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        try:
            import numpy as np
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)
            
            kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
            return kurtosis
        except:
            return 0.0
    
    def detect_patterns_correlation(self, data, threshold=0.7):
        """Detect correlation patterns in data."""
        try:
            import numpy as np
            import pandas as pd
            
            # Convert to pandas DataFrame if needed
            if hasattr(data, 'corr'):
                df = data
            else:
                df = pd.DataFrame(data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        high_correlations.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) >= 0.8 else "moderate"
                        })
            
            # Sort by absolute correlation value
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "success": True,
                "correlation_matrix": corr_matrix.values.tolist(),
                "feature_names": corr_matrix.columns.tolist(),
                "high_correlations": high_correlations,
                "threshold": threshold,
                "total_correlations": len(high_correlations)
            }
            
        except ImportError:
            return {"success": False, "error": "pandas not installed. Install with: pip install pandas"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_outliers_zscore(self, data, threshold=3.0):
        """Detect outliers using Z-score method."""
        try:
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Ensure 2D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            outliers = {}
            
            for col in range(data_array.shape[1]):
                col_data = data_array[:, col]
                col_name = f"feature_{col}"
                
                # Calculate Z-scores
                z_scores = np.abs((col_data - np.mean(col_data)) / np.std(col_data))
                
                # Find outliers
                outlier_indices = np.where(z_scores > threshold)[0]
                outlier_values = col_data[outlier_indices]
                
                outliers[col_name] = {
                    "outlier_indices": outlier_indices.tolist(),
                    "outlier_values": outlier_values.tolist(),
                    "outlier_count": len(outlier_indices),
                    "z_scores": z_scores.tolist(),
                    "threshold": threshold,
                    "mean": float(np.mean(col_data)),
                    "std": float(np.std(col_data))
                }
            
            return {
                "success": True,
                "outliers": outliers,
                "threshold": threshold,
                "total_outliers": sum(out["outlier_count"] for out in outliers.values())
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_outliers_iqr(self, data, multiplier=1.5):
        """Detect outliers using IQR method."""
        try:
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Ensure 2D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            outliers = {}
            
            for col in range(data_array.shape[1]):
                col_data = data_array[:, col]
                col_name = f"feature_{col}"
                
                # Calculate quartiles
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                
                # Find outliers
                outlier_indices = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]
                outlier_values = col_data[outlier_indices]
                
                outliers[col_name] = {
                    "outlier_indices": outlier_indices.tolist(),
                    "outlier_values": outlier_values.tolist(),
                    "outlier_count": len(outlier_indices),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "multiplier": multiplier
                }
            
            return {
                "success": True,
                "outliers": outliers,
                "multiplier": multiplier,
                "total_outliers": sum(out["outlier_count"] for out in outliers.values())
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_data_distribution(self, data, bins=10):
        """Analyze data distribution and identify patterns."""
        try:
            import numpy as np
            import pandas as pd
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Ensure 2D
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)
            
            distributions = {}
            
            for col in range(data_array.shape[1]):
                col_data = data_array[:, col]
                col_name = f"feature_{col}"
                
                # Remove NaN values
                clean_data = col_data[~np.isnan(col_data)]
                
                if len(clean_data) == 0:
                    distributions[col_name] = {"error": "No valid data"}
                    continue
                
                # Basic statistics
                mean_val = np.mean(clean_data)
                median_val = np.median(clean_data)
                std_val = np.std(clean_data)
                
                # Distribution shape
                skewness = self._calculate_skewness(clean_data)
                kurtosis = self._calculate_kurtosis(clean_data)
                
                # Histogram
                hist, bin_edges = np.histogram(clean_data, bins=bins)
                
                # Identify distribution type
                distribution_type = self._identify_distribution_type(skewness, kurtosis)
                
                distributions[col_name] = {
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "std": float(std_val),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "distribution_type": distribution_type,
                    "histogram": {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                        "bin_centers": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                    },
                    "data_range": {
                        "min": float(np.min(clean_data)),
                        "max": float(np.max(clean_data)),
                        "range": float(np.max(clean_data) - np.min(clean_data))
                    },
                    "percentiles": {
                        "p10": float(np.percentile(clean_data, 10)),
                        "p25": float(np.percentile(clean_data, 25)),
                        "p50": float(np.percentile(clean_data, 50)),
                        "p75": float(np.percentile(clean_data, 75)),
                        "p90": float(np.percentile(clean_data, 90))
                    }
                }
            
            return {
                "success": True,
                "distributions": distributions,
                "bins": bins,
                "total_features": len(distributions)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _identify_distribution_type(self, skewness, kurtosis):
        """Identify the type of distribution based on skewness and kurtosis."""
        if abs(skewness) < 0.5:
            if abs(kurtosis) < 0.5:
                return "normal"
            elif kurtosis > 0.5:
                return "leptokurtic"
            else:
                return "platykurtic"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"
    
    def detect_change_points(self, data, window_size=10, threshold=2.0):
        """Detect change points in time series data."""
        try:
            import numpy as np
            
            # Convert to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
            
            # Ensure 1D
            if len(data_array.shape) > 1:
                data_array = data_array.flatten()
            
            change_points = []
            
            # Simple change point detection using rolling statistics
            for i in range(window_size, len(data_array) - window_size):
                # Calculate statistics for windows before and after
                before_window = data_array[i-window_size:i]
                after_window = data_array[i:i+window_size]
                
                # Calculate mean and std for both windows
                before_mean = np.mean(before_window)
                after_mean = np.mean(after_window)
                before_std = np.std(before_window)
                after_std = np.std(after_window)
                
                # Calculate change score
                mean_change = abs(after_mean - before_mean) / (before_std + 1e-8)
                std_change = abs(after_std - before_std) / (before_std + 1e-8)
                
                # Combined change score
                change_score = mean_change + std_change
                
                if change_score > threshold:
                    change_points.append({
                        "index": int(i),
                        "value": float(data_array[i]),
                        "change_score": float(change_score),
                        "mean_change": float(mean_change),
                        "std_change": float(std_change),
                        "before_stats": {
                            "mean": float(before_mean),
                            "std": float(before_std)
                        },
                        "after_stats": {
                            "mean": float(after_mean),
                            "std": float(after_std)
                        }
                    })
            
            # Sort by change score
            change_points.sort(key=lambda x: x["change_score"], reverse=True)
            
            return {
                "success": True,
                "change_points": change_points,
                "window_size": window_size,
                "threshold": threshold,
                "total_change_points": len(change_points),
                "data_length": len(data_array)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

class LocalNLP:
    """Local NLP tools for cybersecurity text processing and classification."""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.classifiers = {}
        self.embeddings = {}
        self.nlp_pipeline = None
        
        # Initialize basic NLP components
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components lazily."""
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            
            # Download required NLTK data (only once)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            self.nlp_pipeline = {
                'tokenizer': word_tokenize,
                'sent_tokenizer': sent_tokenize,
                'stopwords': set(stopwords.words('english')),
                'stemmer': PorterStemmer(),
                'lemmatizer': WordNetLemmatizer()
            }
            
        except ImportError:
            self.nlp_pipeline = None
    
    def extract_text_features(self, text, include_advanced=True):
        """Extract comprehensive text features for cybersecurity analysis."""
        try:
            import numpy as np
            
            if not text or not isinstance(text, str):
                return {"success": False, "error": "Invalid text input"}
            
            # Basic text statistics
            features = {
                "length": len(text),
                "word_count": len(text.split()),
                "char_count": len(text.replace(" ", "")),
                "sentence_count": len(text.split('.')),
                "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
                "avg_sentence_length": len(text.split()) / len(text.split('.')) if text.split('.') else 0
            }
            
            if include_advanced and self.nlp_pipeline:
                # Advanced NLP features
                tokens = self.nlp_pipeline['tokenizer'](text.lower())
                sentences = self.nlp_pipeline['sent_tokenizer'](text)
                
                # Remove stopwords and punctuation
                clean_tokens = [token for token in tokens 
                              if token.isalnum() and token not in self.nlp_pipeline['stopwords']]
                
                # Stemming and lemmatization
                stems = [self.nlp_pipeline['stemmer'].stem(token) for token in clean_tokens]
                lemmas = [self.nlp_pipeline['lemmatizer'].lemmatize(token) for token in clean_tokens]
                
                features.update({
                    "unique_words": len(set(clean_tokens)),
                    "stopword_ratio": len([t for t in tokens if t in self.nlp_pipeline['stopwords']]) / len(tokens) if tokens else 0,
                    "avg_sentence_complexity": len(clean_tokens) / len(sentences) if sentences else 0,
                    "lexical_diversity": len(set(clean_tokens)) / len(clean_tokens) if clean_tokens else 0,
                    "stem_count": len(set(stems)),
                    "lemma_count": len(set(lemmas))
                })
            
            # Cybersecurity-specific features
            security_keywords = [
                'attack', 'breach', 'hack', 'malware', 'virus', 'trojan', 'phishing',
                'ddos', 'sql injection', 'xss', 'csrf', 'exploit', 'vulnerability',
                'firewall', 'antivirus', 'encryption', 'authentication', 'authorization'
            ]
            
            security_score = sum(1 for keyword in security_keywords 
                               if keyword.lower() in text.lower())
            
            features.update({
                "security_keywords": security_score,
                "security_density": security_score / features["word_count"] if features["word_count"] > 0 else 0
            })
            
            return {
                "success": True,
                "features": features,
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def classify_text_naive_bayes(self, text, training_data, labels, test_size=0.2):
        """Classify text using Naive Bayes with automatic training."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            import numpy as np
            
            if not text or not training_data or not labels:
                return {"success": False, "error": "Missing required parameters"}
            
            if len(training_data) != len(labels):
                return {"success": False, "error": "Training data and labels must have same length"}
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Vectorize training data
            X = vectorizer.fit_transform(training_data)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train classifier
            classifier = MultinomialNB()
            classifier.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classify input text
            text_vector = vectorizer.transform([text])
            prediction = classifier.predict(text_vector)[0]
            probabilities = classifier.predict_proba(text_vector)[0]
            
            # Get confidence scores
            confidence = max(probabilities)
            predicted_class = classifier.classes_[np.argmax(probabilities)]
            
            return {
                "success": True,
                "prediction": predicted_class,
                "confidence": float(confidence),
                "all_probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(classifier.classes_, probabilities)
                },
                "model_performance": {
                    "accuracy": float(accuracy),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "classes": classifier.classes_.tolist()
                },
                "text_features": {
                    "vector_dimensions": text_vector.shape[1],
                    "non_zero_features": text_vector.nnz
                }
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed. Install with: pip install scikit-learn"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_text_embeddings(self, texts, model_name="all-MiniLM-L6-v2"):
        """Extract text embeddings using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            if not texts:
                return {"success": False, "error": "No texts provided"}
            
            if isinstance(texts, str):
                texts = [texts]
            
            # Load model (cached after first use)
            if model_name not in self.embeddings:
                self.embeddings[model_name] = SentenceTransformer(model_name)
            
            model = self.embeddings[model_name]
            
            # Generate embeddings
            embeddings = model.encode(texts, convert_to_tensor=False)
            
            # Convert to lists for JSON serialization
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            return {
                "success": True,
                "embeddings": embeddings_list,
                "model": model_name,
                "embedding_dimensions": embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings_list[0]),
                "text_count": len(texts),
                "text_previews": [text[:100] + "..." if len(text) > 100 else text for text in texts]
            }
            
        except ImportError:
            return {"success": False, "error": "sentence-transformers not installed. Install with: pip install sentence-transformers"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_text_similarity(self, text1, text2, model_name="all-MiniLM-L6-v2"):
        """Calculate semantic similarity between two texts."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Load model
            if model_name not in self.embeddings:
                self.embeddings[model_name] = Sentence_transformers
            model = self.embeddings[model_name]
            
            # Generate embeddings
            embeddings = model.encode([text1, text2])
            
            # Calculate similarity
            similarity_matrix = cosine_similarity(embeddings)
            similarity_score = float(similarity_matrix[0, 1])
            
            return {
                "success": True,
                "similarity_score": similarity_score,
                "similarity_percentage": similarity_score * 100,
                "model": model_name,
                "text1_preview": text1[:100] + "..." if len(text1) > 100 else text1,
                "text2_preview": text2[:100] + "..." if len(text2) > 100 else text2,
                "text1_length": len(text1),
                "text2_length": len(text2)
            }
            
        except ImportError:
            return {"success": False, "error": "Required libraries not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cluster_texts(self, texts, n_clusters=3, model_name="all-MiniLM-L6-v2"):
        """Cluster texts based on semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            if not texts or len(texts) < n_clusters:
                return {"success": False, "error": "Insufficient texts for clustering"}
            
            # Load model and generate embeddings
            if model_name not in self.embeddings:
                self.embeddings[model_name] = SentenceTransformer(model_name)
            
            model = self.embeddings[model_name]
            embeddings = model.encode(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            
            # Organize results by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_key = f"cluster_{label}"
                if cluster_key not in clusters:
                    clusters[cluster_key] = {
                        "texts": [],
                        "indices": [],
                        "text_previews": []
                    }
                
                clusters[cluster_key]["texts"].append(texts[i])
                clusters[cluster_key]["indices"].append(i)
                clusters[cluster_key]["text_previews"].append(
                    texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
                )
            
            # Add cluster statistics
            for cluster_key in clusters:
                clusters[cluster_key]["size"] = len(clusters[cluster_key]["texts"])
                clusters[cluster_key]["center"] = kmeans.cluster_centers_[int(cluster_key.split('_')[1])].tolist()
            
            return {
                "success": True,
                "clusters": clusters,
                "cluster_labels": cluster_labels.tolist(),
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette_avg),
                "total_texts": len(texts),
                "model": model_name,
                "embedding_dimensions": embeddings.shape[1]
            }
            
        except ImportError:
            return {"success": False, "error": "Required libraries not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_text_sentiment(self, text, method="lexicon"):
        """Analyze text sentiment using multiple methods."""
        try:
            if method == "lexicon":
                return self._sentiment_lexicon_analysis(text)
            elif method == "ml":
                return self._sentiment_ml_analysis(text)
            else:
                return {"success": False, "error": "Unsupported sentiment method"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _sentiment_lexicon_analysis(self, text):
        """Simple lexicon-based sentiment analysis."""
        try:
            # Basic sentiment lexicons
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'secure', 'safe', 'protected', 'trusted', 'reliable', 'stable',
                'success', 'victory', 'win', 'positive', 'beneficial', 'helpful'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'dangerous', 'threat',
                'attack', 'breach', 'hack', 'malware', 'virus', 'exploit',
                'vulnerability', 'weak', 'unsecure', 'compromised', 'failed'
            }
            
            # Tokenize and analyze
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            # Calculate sentiment score
            total_words = len(words)
            if total_words == 0:
                sentiment_score = 0
            else:
                sentiment_score = (positive_count - negative_count) / total_words
            
            # Determine sentiment category
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "success": True,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": total_words,
                "method": "lexicon",
                "confidence": min(abs(sentiment_score) * 2, 1.0)  # Simple confidence metric
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _sentiment_ml_analysis(self, text):
        """ML-based sentiment analysis using pre-trained models."""
        try:
            # For now, fall back to lexicon method
            # In production, you could integrate with a local sentiment model
            return self._sentiment_lexicon_analysis(text)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_keywords(self, text, top_k=10, method="tfidf"):
        """Extract keywords from text using multiple methods."""
        try:
            if method == "tfidf":
                return self._extract_keywords_tfidf(text, top_k)
            elif method == "frequency":
                return self._extract_keywords_frequency(text, top_k)
            else:
                return {"success": False, "error": "Unsupported keyword extraction method"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_keywords_tfidf(self, text, top_k):
        """Extract keywords using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
            keywords = [
                {
                    "keyword": feature_names[i],
                    "score": float(tfidf_scores[i]),
                    "rank": rank + 1
                }
                for rank, i in enumerate(top_indices)
                if tfidf_scores[i] > 0
            ]
            
            return {
                "success": True,
                "keywords": keywords,
                "method": "tfidf",
                "total_keywords": len(keywords),
                "text_length": len(text)
            }
            
        except ImportError:
            return {"success": False, "error": "scikit-learn not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_keywords_frequency(self, text, top_k):
        """Extract keywords using frequency analysis."""
        try:
            from collections import Counter
            import re
            
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            # Remove common stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [word for word in words if word not in stopwords and len(word) > 2]
            
            # Count frequencies
            word_counts = Counter(words)
            
            # Get top keywords
            top_keywords = word_counts.most_common(top_k)
            keywords = [
                {
                    "keyword": word,
                    "count": count,
                    "rank": rank + 1
                }
                for rank, (word, count) in enumerate(top_keywords)
            ]
            
            return {
                "success": True,
                "keywords": keywords,
                "method": "frequency",
                "total_keywords": len(keywords),
                "text_length": len(text),
                "unique_words": len(word_counts)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def preprocess_text(self, text, operations=None):
        """Apply various text preprocessing operations."""
        try:
            if not text:
                return {"success": False, "error": "No text provided"}
            
            if operations is None:
                operations = ["lowercase", "remove_punctuation", "remove_stopwords", "stemming"]
            
            processed_text = text
            applied_operations = []
            
            # Apply requested operations
            if "lowercase" in operations:
                processed_text = processed_text.lower()
                applied_operations.append("lowercase")
            
            if "remove_punctuation" in operations:
                import re
                processed_text = re.sub(r'[^\w\s]', '', processed_text)
                applied_operations.append("remove_punctuation")
            
            if "remove_stopwords" in operations and self.nlp_pipeline:
                words = processed_text.split()
                processed_text = " ".join([word for word in words 
                                        if word not in self.nlp_pipeline['stopwords']])
                applied_operations.append("remove_stopwords")
            
            if "stemming" in operations and self.nlp_pipeline:
                words = processed_text.split()
                processed_text = " ".join([self.nlp_pipeline["stemmer"].stem(word) for word in words])
                applied_operations.append("stemming")
            
            if "lemmatization" in operations and self.nlp_pipeline:
                words = processed_text.split()
                processed_text = " ".join([self.nlp_pipeline['lemmatizer'].lemmatize(word) 
                                        for word in words])
                applied_operations.append("lemmatization")
            
            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "applied_operations": applied_operations,
                "original_length": len(text),
                "processed_length": len(processed_text),
                "reduction_percentage": ((len(text) - len(processed_text)) / len(text)) * 100
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# ... existing code ...

if __name__ == "__main__":
    print("CS AI Tools loaded successfully!")
    print(f"MCP Server tools: {len(tool_manager.mcp_server.tools)}")
    print(f"Tool Manager status: {tool_manager.get_tool_status()}")
