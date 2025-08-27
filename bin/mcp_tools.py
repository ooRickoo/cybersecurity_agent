#!/usr/bin/env python3
"""
MCP Tools for LangGraph Cybersecurity Agent
Provides all the tool functionality in MCP-compatible format.
"""

import os
import sys
import json
import uuid
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Manages session creation and output organization."""
    
    def __init__(self):
        self.session_logs_dir = Path("session-logs")
        self.session_outputs_dir = Path("session-outputs")
        
        # Create directories if they don't exist
        self.session_logs_dir.mkdir(exist_ok=True)
        self.session_outputs_dir.mkdir(exist_ok=True)
    
    def create_session(self, session_name: str = None) -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session output directory
        session_output_dir = self.session_outputs_dir / session_id
        session_output_dir.mkdir(exist_ok=True)
        
        # Create session log
        self._create_session_log(session_id, session_name)
        
        return session_id
    
    def _create_session_log(self, session_id: str, session_name: str):
        """Create a session log entry."""
        timestamp = datetime.now()
        log_filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}.json"
        
        log_data = {
            "session_metadata": {
                "session_id": session_id,
                "session_name": session_name,
                "start_time": timestamp.isoformat(),
                "version": "2.0.0",
                "framework": "LangGraph Cybersecurity Agent",
                "end_time": None,
                "duration_ms": None
            },
            "agent_interactions": [
                {
                    "timestamp": timestamp.isoformat(),
                    "level": "info",
                    "category": "session_creation",
                    "action": "session_start",
                    "details": {
                        "message": f"Session created: {session_name}",
                        "session_id": session_id,
                        "framework": "LangGraph Cybersecurity Agent"
                    },
                    "session_id": session_id,
                    "agent_type": "LangGraphAgent",
                    "workflow_step": "session_initialization"
                }
            ],
            "workflow_executions": [],
            "tool_calls": [],
            "data_operations": [],
            "decision_points": [],
            "errors": [],
            "performance_metrics": {
                "total_tool_calls": 0,
                "total_workflow_steps": 0,
                "total_errors": 0,
                "session_duration_ms": None
            }
        }
        
        log_file_path = self.session_logs_dir / log_filename
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_file_path
    
    def add_workflow_execution(self, session_id: str, workflow_name: str, execution_details: Dict[str, Any] = None):
        """Add a workflow execution to the session log."""
        log_files = list(self.session_logs_dir.glob(f"*{session_id[:8]}*.json"))
        if not log_files:
            return
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            workflow_execution = {
                "timestamp": datetime.now().isoformat(),
                "workflow_name": workflow_name,
                "execution_id": str(uuid.uuid4()),
                "status": "completed",
                "details": execution_details or {},
                "duration_ms": None,
                "success": True,
                "error_message": None
            }
            
            log_data["workflow_executions"].append(workflow_execution)
            log_data["performance_metrics"]["total_workflow_steps"] += 1
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to update session log: {e}")
    
    def save_output_file(self, session_id: str, filename: str, content: str, file_type: str = "text"):
        """Save an output file to the session directory."""
        session_output_dir = self.session_outputs_dir / session_id
        session_output_dir.mkdir(exist_ok=True)
        
        file_path = session_output_dir / filename
        
        if file_type == "csv":
            # Handle CSV content
            if isinstance(content, pd.DataFrame):
                content.to_csv(file_path, index=False)
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
        else:
            # Handle text content
            with open(file_path, 'w') as f:
                f.write(content)
        
        return str(file_path)

# ============================================================================
# FRAMEWORK PROCESSOR
# ============================================================================

class FrameworkProcessor:
    """Processes and flattens cybersecurity frameworks."""
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv', 'ttl', 'stix']
    
    def process_framework(self, framework_data: str, framework_type: str) -> Dict[str, Any]:
        """Process a framework and return flattened data."""
        try:
            if framework_type.lower() == 'json':
                return self._process_json(framework_data)
            elif framework_type.lower() == 'csv':
                return self._process_csv(framework_data)
            elif framework_type.lower() == 'xml':
                return self._process_xml(framework_data)
            elif framework_type.lower() == 'ttl':
                return self._process_ttl(framework_data)
            elif framework_type.lower() == 'stix':
                return self._process_stix(framework_data)
            else:
                raise ValueError(f"Unsupported framework type: {framework_type}")
        except Exception as e:
            return {"error": f"Failed to process framework: {str(e)}"}
    
    def _process_json(self, data: str) -> Dict[str, Any]:
        """Process JSON framework data."""
        try:
            if isinstance(data, str):
                json_data = json.loads(data)
            else:
                json_data = data
            
            # Flatten JSON structure
            flattened = self._flatten_json(json_data)
            
            return {
                "type": "json",
                "processed": True,
                "total_items": len(flattened),
                "flattened_data": flattened,
                "structure": self._analyze_structure(json_data)
            }
        except Exception as e:
            return {"error": f"JSON processing failed: {str(e)}"}
    
    def _process_csv(self, data: str) -> Dict[str, Any]:
        """Process CSV framework data."""
        try:
            if isinstance(data, str):
                df = pd.read_csv(pd.StringIO(data))
            else:
                df = data
            
            return {
                "type": "csv",
                "processed": True,
                "total_items": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "flattened_data": df.to_dict('records'),
                "structure": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "has_headers": True
                }
            }
        except Exception as e:
            return {"error": f"CSV processing failed: {str(e)}"}
    
    def _process_xml(self, data: str) -> Dict[str, Any]:
        """Process XML framework data."""
        try:
            import xml.etree.ElementTree as ET
            
            if isinstance(data, str):
                root = ET.fromstring(data)
            else:
                root = data
            
            # Convert XML to dictionary
            xml_dict = self._xml_to_dict(root)
            flattened = self._flatten_json(xml_dict)
            
            return {
                "type": "xml",
                "processed": True,
                "total_items": len(flattened),
                "root_tag": root.tag,
                "flattened_data": flattened,
                "structure": self._analyze_xml_structure(root)
            }
        except Exception as e:
            return {"error": f"XML processing failed: {str(e)}"}
    
    def _process_ttl(self, data: str) -> Dict[str, Any]:
        """Process TTL (Turtle) framework data."""
        try:
            # Simple TTL parsing - in production, use rdflib
            lines = data.split('\n') if isinstance(data, str) else data
            
            triples = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and ' ' in line:
                    parts = line.split(' ', 2)
                    if len(parts) >= 2:
                        triples.append({
                            "subject": parts[0],
                            "predicate": parts[1],
                            "object": parts[2] if len(parts) > 2 else ""
                        })
            
            return {
                "type": "ttl",
                "processed": True,
                "total_items": len(triples),
                "triples": triples,
                "structure": {
                    "format": "rdf_triple",
                    "triple_count": len(triples)
                }
            }
        except Exception as e:
            return {"error": f"TTL processing failed: {str(e)}"}
    
    def _process_stix(self, data: str) -> Dict[str, Any]:
        """Process STIX framework data."""
        try:
            if isinstance(data, str):
                stix_data = json.loads(data)
            else:
                stix_data = data
            
            # Extract STIX objects
            stix_objects = stix_data.get('objects', [])
            
            flattened = []
            for obj in stix_objects:
                obj_type = obj.get('type', 'unknown')
                obj_id = obj.get('id', 'unknown')
                
                flattened.append({
                    "stix_type": obj_type,
                    "stix_id": obj_id,
                    "properties": obj,
                    "extracted_at": datetime.now().isoformat()
                })
            
            return {
                "type": "stix",
                "processed": True,
                "total_items": len(flattened),
                "stix_objects": flattened,
                "structure": {
                    "format": "stix_2_1",
                    "object_types": list(set(obj.get('type') for obj in stix_objects))
                }
            }
        except Exception as e:
            return {"error": f"STIX processing failed: {str(e)}"}
    
    def _flatten_json(self, obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested JSON structure."""
        flattened = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                flattened.extend(self._flatten_json(value, new_prefix))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                flattened.extend(self._flatten_json(item, new_prefix))
        else:
            flattened.append({
                "path": prefix,
                "value": obj,
                "type": type(obj).__name__
            })
        
        return flattened
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result['@text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            child_tag = child.tag
            
            if child_tag in result:
                if not isinstance(result[child_tag], list):
                    result[child_tag] = [result[child_tag]]
                result[child_tag].append(child_data)
            else:
                result[child_tag] = child_data
        
        return result
    
    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze the structure of processed data."""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "depth": self._calculate_depth(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list(set(type(item).__name__ for item in data))
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100]  # Truncate long values
            }
    
    def _analyze_xml_structure(self, root) -> Dict[str, Any]:
        """Analyze XML structure."""
        def count_elements(element):
            count = 1
            for child in element:
                count += count_elements(child)
            return count
        
        return {
            "root_tag": root.tag,
            "total_elements": count_elements(root),
            "max_depth": self._calculate_xml_depth(root)
        }
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of nested structure."""
        if isinstance(obj, dict):
            return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            return max(self._calculate_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _calculate_xml_depth(self, element, current_depth: int = 0) -> int:
        """Calculate the maximum depth of XML structure."""
        if not element:
            return current_depth
        
        max_depth = current_depth
        for child in element:
            child_depth = self._calculate_xml_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth

# ============================================================================
# ENCRYPTION MANAGER (Moved from main file)
# ============================================================================

class EncryptionManager:
    """Manages encryption/decryption of sensitive data."""
    
    def __init__(self, password_hash: str = None, salt: str = None):
        if password_hash is None:
            password_hash = os.getenv('ENCRYPTION_PASSWORD_HASH', '')
            salt = os.getenv('ENCRYPTION_SALT', 'cybersecurity_agent_salt')
            
            # Default password hash for 'Vosteen2025' if none provided
            if not password_hash:
                default_password = ''  # No default password for security
                password_hash = hashlib.sha256(default_password.encode()).hexdigest()
        
        self.password_hash = password_hash
        self.salt = salt.encode() if isinstance(salt, str) else salt
        self.key = self._derive_key_from_hash(password_hash)
        self.cipher = self._create_cipher()
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password (legacy method)."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            salt = b'cybersecurity_agent_salt'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key
        except ImportError:
            # Fallback to simple hash if cryptography not available
            import hashlib
            return hashlib.sha256(password.encode()).digest()
    
    def _derive_key_from_hash(self, password_hash: str) -> bytes:
        """Derive encryption key from password hash using salt."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Use the password hash directly with the salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            # The hash is already a hex string, so we use it directly
            key = base64.urlsafe_b64encode(kdf.derive(password_hash.encode()))
            return key
        except ImportError:
            # Fallback to simple hash if cryptography not available
            import hashlib
            return hashlib.md5(password_hash.encode()).digest()
        except Exception as e:
            print(f"Warning: Failed to derive key from hash: {e}")
            return None
    
    def _create_cipher(self):
        """Create encryption cipher."""
        try:
            from cryptography.fernet import Fernet
            return Fernet(self.key)
        except ImportError:
            return None
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not self.cipher:
            return data
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.cipher:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file and return path to encrypted version."""
        if not self.cipher:
            return file_path
        
        encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_path: Path) -> Path:
        """Decrypt a file and return path to decrypted version."""
        if not self.cipher:
            return encrypted_path
        
        decrypted_path = encrypted_path.with_suffix('').with_suffix(
            encrypted_path.suffix.replace('.encrypted', '')
        )
        
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.decrypt_data(encrypted_data)
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        return decrypted_path

# ============================================================================
# KNOWLEDGE GRAPH MANAGER (Moved from main file)
# ============================================================================

class KnowledgeGraphManager:
    """Manages multi-dimensional knowledge graphs and memory."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.knowledge_base_path = Path("knowledge-objects")
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Memory dimensions
        self.short_term = {}      # Current session context
        self.running_term = {}    # Active workflows
        self.long_term = {}       # Persistent knowledge
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load existing knowledge base."""
        master_catalog = self.knowledge_base_path / "master_catalog.db"
        if master_catalog.exists():
            # Load encrypted database if encryption is enabled
            if os.getenv('ENCRYPTION_ENABLED', 'false').lower() == 'true':
                decrypted_path = self.encryption_manager.decrypt_file(master_catalog)
                # Load from decrypted path
                decrypted_path.unlink()  # Clean up
            else:
                # Load directly
                pass
    
    def add_framework(self, framework_name: str, framework_data: Dict[str, Any]) -> str:
        """Add a new framework to the knowledge base."""
        framework_id = str(uuid.uuid4())
        
        # Store in appropriate memory dimension
        if len(str(framework_data)) < 1000:
            # Small framework - store in short-term
            self.short_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'short_term'
            }
        elif len(str(framework_data)) < 10000:
            # Medium framework - store in running-term
            self.running_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'running_term'
            }
        else:
            # Large framework - store in long-term
            self.long_term[framework_id] = {
                'name': framework_name,
                'type': 'framework',
                'data': framework_data,
                'timestamp': datetime.now().isoformat(),
                'dimension': 'long_term'
            }
        
        return framework_id
    
    def query_knowledge(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query knowledge across all dimensions."""
        results = {
            'short_term': [],
            'running_term': [],
            'long_term': [],
            'relevance_scores': {}
        }
        
        # Search across dimensions
        for dimension_name, dimension_data in [
            ('short_term', self.short_term),
            ('running_term', self.running_term),
            ('long_term', self.long_term)
        ]:
            for item_id, item_data in dimension_data.items():
                relevance = self._calculate_relevance(query, item_data, context)
                if relevance > 0.5:  # Threshold for relevance
                    results[dimension_name].append({
                        'id': item_id,
                        'data': item_data,
                        'relevance': relevance
                    })
                    results['relevance_scores'][item_id] = relevance
        
        # Sort by relevance
        for dimension in ['short_term', 'running_term', 'long_term']:
            results[dimension].sort(key=lambda x: x['relevance'], reverse=True)
        
        return results
    
    def _calculate_relevance(self, query: str, item_data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for a query against an item."""
        # Simple keyword matching for now
        query_lower = query.lower()
        item_name = item_data.get('name', '').lower()
        item_type = item_data.get('type', '').lower()
        
        score = 0.0
        
        # Exact name match
        if query_lower in item_name:
            score += 0.8
        
        # Type match
        if query_lower in item_type:
            score += 0.6
        
        # Context relevance
        if context:
            context_keywords = ' '.join(str(v) for v in context.values()).lower()
            if any(keyword in context_keywords for keyword in query_lower.split()):
                score += 0.4
        
        return min(score, 1.0)
    
    def rehydrate_context(self, context_keys: List[str]) -> Dict[str, Any]:
        """Rehydrate context from memory dimensions."""
        context = {}
        
        for key in context_keys:
            # Search across all dimensions
            for dimension_data in [self.short_term, self.running_term, self.long_term]:
                if key in dimension_data:
                    context[key] = dimension_data[key]
                    break
        
        return context

# ============================================================================
# MCP TOOL REGISTRATION
# ============================================================================

def register_mcp_tools():
    """Register all MCP tools."""
    tools = {
        "framework_processor": {
            "description": "Process and flatten cybersecurity frameworks",
            "input_schema": {
                "type": "object",
                "properties": {
                    "framework_data": {"type": "string", "description": "Framework data to process"},
                    "framework_type": {"type": "string", "description": "Type of framework (json, csv, xml, ttl, stix)"}
                },
                "required": ["framework_data", "framework_type"]
            }
        },
        "session_manager": {
            "description": "Manage sessions and output files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action to perform (create, save, log)"},
                    "session_name": {"type": "string", "description": "Name for the session"},
                    "filename": {"type": "string", "description": "Output filename"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["action"]
            }
        },
        "knowledge_manager": {
            "description": "Manage knowledge graph and memory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action to perform (add, query, rehydrate)"},
                    "framework_name": {"type": "string", "description": "Name of the framework"},
                    "framework_data": {"type": "object", "description": "Framework data"},
                    "query": {"type": "string", "description": "Query string"},
                    "context_keys": {"type": "array", "items": {"type": "string"}, "description": "Context keys to rehydrate"}
                },
                "required": ["action"]
            }
        }
    }
    
    return tools
