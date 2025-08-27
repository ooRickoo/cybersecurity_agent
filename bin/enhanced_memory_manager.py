"""
Enhanced Memory Manager for Cybersecurity Agent

Provides comprehensive memory management including deletion, backup/restore,
and lifecycle management capabilities.
"""

import json
import logging
import zipfile
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import hashlib
import getpass
import pyzipper  # For AES encryption

from .context_memory_manager import ContextMemoryManager
from .enhanced_session_manager import EnhancedSessionManager
from .credential_vault import CredentialVault

logger = logging.getLogger(__name__)

class EnhancedMemoryManager:
    """Enhanced memory manager with deletion, backup, and restore capabilities."""
    
    def __init__(self, memory_manager: ContextMemoryManager, 
                 session_manager: EnhancedSessionManager,
                 credential_vault: CredentialVault):
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self.credential_vault = credential_vault
        self.backup_dir = Path("knowledge-objects/backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory deletion tracking
        self.deletion_history = []
        
    def delete_memory(self, memory_id: str, domain: Optional[str] = None, 
                     tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a single memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            domain: Optional domain filter
            tier: Optional tier filter
            
        Returns:
            Deletion result
        """
        try:
            # Get memory details before deletion
            memory_info = self.memory_manager.get_memory_by_id(memory_id)
            
            if not memory_info:
                return {
                    'success': False,
                    'error': f'Memory with ID {memory_id} not found'
                }
            
            # Verify deletion criteria if filters provided
            if domain and memory_info.get('domain') != domain:
                return {
                    'success': False,
                    'error': f'Memory domain mismatch: expected {domain}, got {memory_info.get("domain")}'
                }
            
            if tier and memory_info.get('tier') != tier:
                return {
                    'success': False,
                    'error': f'Memory tier mismatch: expected {tier}, got {memory_info.get("tier")}'
                }
            
            # Delete the memory
            deletion_result = self.memory_manager.delete_memory(memory_id)
            
            if deletion_result.get('success', False):
                # Record deletion in history
                deletion_record = {
                    'memory_id': memory_id,
                    'deleted_at': datetime.now().isoformat(),
                    'memory_info': memory_info,
                    'deletion_reason': 'manual_deletion'
                }
                self.deletion_history.append(deletion_record)
                
                # Save deletion history
                self._save_deletion_history()
                
                return {
                    'success': True,
                    'message': f'Memory {memory_id} deleted successfully',
                    'deleted_memory': memory_info
                }
            else:
                return deletion_result
                
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_memories_by_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete multiple memories based on criteria.
        
        Args:
            criteria: Dictionary of criteria (domain, tier, data_type, etc.)
            
        Returns:
            Bulk deletion result
        """
        try:
            # Search for memories matching criteria
            search_results = self.memory_manager.search_memories(criteria)
            
            if not search_results.get('success', False):
                return search_results
            
            memories_to_delete = search_results.get('memories', [])
            
            if not memories_to_delete:
                return {
                    'success': True,
                    'message': 'No memories found matching criteria',
                    'deleted_count': 0
                }
            
            # Confirm deletion if large number
            if len(memories_to_delete) > 100:
                confirmation = input(f"About to delete {len(memories_to_delete)} memories. Type 'CONFIRM' to proceed: ")
                if confirmation != 'CONFIRM':
                    return {
                        'success': False,
                        'message': 'Deletion cancelled by user',
                        'deleted_count': 0
                    }
            
            # Delete memories
            deleted_count = 0
            failed_deletions = []
            
            for memory in memories_to_delete:
                memory_id = memory.get('id')
                if memory_id:
                    deletion_result = self.delete_memory(memory_id)
                    if deletion_result.get('success', False):
                        deleted_count += 1
                    else:
                        failed_deletions.append({
                            'memory_id': memory_id,
                            'error': deletion_result.get('error', 'Unknown error')
                        })
            
            return {
                'success': True,
                'message': f'Deleted {deleted_count} memories',
                'deleted_count': deleted_count,
                'failed_deletions': failed_deletions,
                'total_found': len(memories_to_delete)
            }
            
        except Exception as e:
            logger.error(f"Failed to delete memories by criteria: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_all_memories(self, confirmation: str = None) -> Dict[str, Any]:
        """
        Delete all memories (dangerous operation).
        
        Args:
            confirmation: Must be 'DELETE_ALL_MEMORIES' to proceed
            
        Returns:
            Deletion result
        """
        try:
            if confirmation != 'DELETE_ALL_MEMORIES':
                return {
                    'success': False,
                    'error': 'Confirmation required. Pass "DELETE_ALL_MEMORIES" to proceed.'
                }
            
            # Get memory statistics before deletion
            stats = self.memory_manager.get_memory_statistics()
            
            # Delete all memories
            deletion_result = self.memory_manager.delete_all_memories()
            
            if deletion_result.get('success', False):
                # Record bulk deletion
                bulk_deletion_record = {
                    'deletion_type': 'bulk_all',
                    'deleted_at': datetime.now().isoformat(),
                    'previous_stats': stats,
                    'deletion_reason': 'manual_bulk_deletion'
                }
                self.deletion_history.append(bulk_deletion_record)
                
                # Save deletion history
                self._save_deletion_history()
                
                return {
                    'success': True,
                    'message': 'All memories deleted successfully',
                    'previous_stats': stats
                }
            else:
                return deletion_result
                
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def backup_memories(self, backup_name: Optional[str] = None, 
                       include_deleted: bool = False) -> Dict[str, Any]:
        """
        Create an encrypted backup of all memories.
        
        Args:
            backup_name: Optional custom name for the backup
            include_deleted: Whether to include deletion history
            
        Returns:
            Backup result
        """
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if backup_name:
                backup_filename = f"{backup_name}_{timestamp}.zip"
            else:
                backup_filename = f"memory_backup_{timestamp}.zip"
            
            backup_path = self.backup_dir / backup_filename
            
            # Prompt for encryption password
            print("🔐 Creating encrypted memory backup...")
            password = getpass.getpass("Enter backup encryption password: ")
            if not password:
                return {
                    'success': False,
                    'error': 'Password required for backup encryption'
                }
            
            # Create temporary directory for backup data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Export memories to temporary files
                self._export_memories_to_temp(temp_path)
                
                # Include deletion history if requested
                if include_deleted:
                    deletion_file = temp_path / "deletion_history.json"
                    with open(deletion_file, 'w') as f:
                        json.dump(self.deletion_history, f, indent=2, default=str)
                
                # Create metadata file
                metadata = {
                    'backup_created_at': datetime.now().isoformat(),
                    'backup_version': '1.0',
                    'memory_count': self.memory_manager.get_memory_statistics().get('total_entries', 0),
                    'domains': list(self.memory_manager.get_memory_statistics().get('domains', {}).keys()),
                    'tiers': list(self.memory_manager.get_memory_statistics().get('tiers', {}).keys()),
                    'include_deleted': include_deleted
                }
                
                metadata_file = temp_path / "backup_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create encrypted zip file
                self._create_encrypted_zip(temp_path, backup_path, password)
            
            # Verify backup file
            if backup_path.exists() and backup_path.stat().st_size > 0:
                return {
                    'success': True,
                    'message': 'Memory backup created successfully',
                    'backup_file': str(backup_path),
                    'backup_size': backup_path.stat().st_size,
                    'metadata': metadata
                }
            else:
                return {
                    'success': False,
                    'error': 'Backup file creation failed'
                }
                
        except Exception as e:
            logger.error(f"Failed to create memory backup: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _export_memories_to_temp(self, temp_path: Path):
        """Export memories to temporary directory."""
        try:
            # Get all memories
            all_memories = self.memory_manager.get_all_memories()
            
            if all_memories.get('success', False):
                memories = all_memories.get('memories', [])
                
                # Group by domain and tier
                for memory in memories:
                    domain = memory.get('domain', 'unknown')
                    tier = memory.get('tier', 'unknown')
                    
                    # Create directory structure
                    memory_dir = temp_path / domain / tier
                    memory_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save memory to file
                    memory_id = memory.get('id', 'unknown')
                    memory_file = memory_dir / f"{memory_id}.json"
                    
                    with open(memory_file, 'w') as f:
                        json.dump(memory, f, indent=2, default=str)
                        
        except Exception as e:
            logger.error(f"Failed to export memories to temp: {e}")
            raise
    
    def _create_encrypted_zip(self, source_path: Path, dest_path: Path, password: str):
        """Create encrypted zip file."""
        try:
            with pyzipper.AESZipFile(
                dest_path, 'w', compression=pyzipper.ZIP_LZMA, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password.encode())
                
                # Add all files from source directory
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        arc_name = file_path.relative_to(source_path)
                        zf.write(file_path, arc_name)
                        
        except Exception as e:
            logger.error(f"Failed to create encrypted zip: {e}")
            raise
    
    def restore_memories(self, backup_file: str, password: str) -> Dict[str, Any]:
        """
        Restore memories from an encrypted backup.
        
        Args:
            backup_file: Path to the backup file
            password: Backup encryption password
            
        Returns:
            Restore result
        """
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup file not found: {backup_file}'
                }
            
            print("🔓 Restoring memories from backup...")
            
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract encrypted zip
                self._extract_encrypted_zip(backup_path, temp_path, password)
                
                # Read metadata
                metadata_file = temp_path / "backup_metadata.json"
                if not metadata_file.exists():
                    return {
                        'success': False,
                        'error': 'Backup metadata not found'
                    }
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Restore memories
                restore_result = self._restore_memories_from_temp(temp_path)
                
                if restore_result.get('success', False):
                    # Restore deletion history if present
                    deletion_file = temp_path / "deletion_history.json"
                    if deletion_file.exists():
                        with open(deletion_file, 'r') as f:
                            self.deletion_history = json.load(f)
                        self._save_deletion_history()
                    
                    return {
                        'success': True,
                        'message': 'Memories restored successfully',
                        'restored_count': restore_result.get('restored_count', 0),
                        'backup_metadata': metadata
                    }
                else:
                    return restore_result
                    
        except Exception as e:
            logger.error(f"Failed to restore memories: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_encrypted_zip(self, zip_path: Path, extract_path: Path, password: str):
        """Extract encrypted zip file."""
        try:
            with pyzipper.AESZipFile(zip_path, 'r') as zf:
                zf.setpassword(password.encode())
                zf.extractall(extract_path)
                
        except Exception as e:
            logger.error(f"Failed to extract encrypted zip: {e}")
            raise
    
    def _restore_memories_from_temp(self, temp_path: Path) -> Dict[str, Any]:
        """Restore memories from temporary directory."""
        try:
            restored_count = 0
            failed_restores = []
            
            # Find all memory files
            memory_files = list(temp_path.rglob("*.json"))
            
            for memory_file in memory_files:
                if memory_file.name in ['backup_metadata.json', 'deletion_history.json']:
                    continue
                
                try:
                    # Read memory data
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                    
                    # Restore memory
                    restore_result = self.memory_manager.import_data(
                        memory_data.get('data_type', 'unknown'),
                        memory_data.get('data', {}),
                        domain=memory_data.get('domain', 'unknown'),
                        tier=memory_data.get('tier', 'medium_term'),
                        ttl_days=memory_data.get('ttl_days', 30),
                        metadata=memory_data.get('metadata', {})
                    )
                    
                    if restore_result.get('success', False):
                        restored_count += 1
                    else:
                        failed_restores.append({
                            'file': str(memory_file),
                            'error': restore_result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    failed_restores.append({
                        'file': str(memory_file),
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'restored_count': restored_count,
                'failed_restores': failed_restores,
                'total_files': len(memory_files) - 2  # Exclude metadata files
            }
            
        except Exception as e:
            logger.error(f"Failed to restore memories from temp: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> Dict[str, Any]:
        """List available memory backups."""
        try:
            backup_files = []
            
            for backup_file in self.backup_dir.glob("*.zip"):
                try:
                    stat = backup_file.stat()
                    backup_info = {
                        'filename': backup_file.name,
                        'size': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'path': str(backup_file)
                    }
                    backup_files.append(backup_info)
                except Exception as e:
                    logger.warning(f"Failed to get info for backup {backup_file}: {e}")
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x['created_at'], reverse=True)
            
            return {
                'success': True,
                'backups': backup_files,
                'total_backups': len(backup_files)
            }
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_backup(self, backup_file: str) -> Dict[str, Any]:
        """Delete a memory backup file."""
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup file not found: {backup_file}'
                }
            
            # Confirm deletion
            confirmation = input(f"Delete backup file {backup_file}? Type 'DELETE' to confirm: ")
            if confirmation != 'DELETE':
                return {
                    'success': False,
                    'message': 'Backup deletion cancelled by user'
                }
            
            # Delete the file
            backup_path.unlink()
            
            return {
                'success': True,
                'message': f'Backup {backup_file} deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_file}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_deletion_history(self) -> Dict[str, Any]:
        """Get deletion history."""
        try:
            return {
                'success': True,
                'deletion_history': self.deletion_history,
                'total_deletions': len(self.deletion_history)
            }
        except Exception as e:
            logger.error(f"Failed to get deletion history: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_deletion_history(self):
        """Save deletion history to file."""
        try:
            history_file = self.backup_dir / "deletion_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.deletion_history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save deletion history: {e}")
    
    def load_deletion_history(self):
        """Load deletion history from file."""
        try:
            history_file = self.backup_dir / "deletion_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.deletion_history = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load deletion history: {e}")
            self.deletion_history = []

# MCP Tools for Enhanced Memory Management
class EnhancedMemoryMCPTools:
    """MCP-compatible tools for enhanced memory management."""
    
    def __init__(self, enhanced_memory: EnhancedMemoryManager):
        self.memory_manager = enhanced_memory
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for enhanced memory management."""
        return {
            "delete_memory": {
                "name": "delete_memory",
                "description": "Delete a single memory entry",
                "parameters": {
                    "memory_id": {"type": "string", "description": "ID of the memory to delete"},
                    "domain": {"type": "string", "description": "Optional domain filter"},
                    "tier": {"type": "string", "description": "Optional tier filter"}
                },
                "returns": {"type": "object", "description": "Deletion result"}
            },
            "delete_memories_by_criteria": {
                "name": "delete_memories_by_criteria",
                "description": "Delete multiple memories based on criteria",
                "parameters": {
                    "criteria": {"type": "object", "description": "Dictionary of deletion criteria"}
                },
                "returns": {"type": "object", "description": "Bulk deletion result"}
            },
            "delete_all_memories": {
                "name": "delete_all_memories",
                "description": "Delete all memories (dangerous operation)",
                "parameters": {
                    "confirmation": {"type": "string", "description": "Must be 'DELETE_ALL_MEMORIES' to proceed"}
                },
                "returns": {"type": "object", "description": "Deletion result"}
            },
            "backup_memories": {
                "name": "backup_memories",
                "description": "Create an encrypted backup of all memories",
                "parameters": {
                    "backup_name": {"type": "string", "description": "Optional custom name for the backup"},
                    "include_deleted": {"type": "boolean", "description": "Whether to include deletion history"}
                },
                "returns": {"type": "object", "description": "Backup result"}
            },
            "restore_memories": {
                "name": "restore_memories",
                "description": "Restore memories from an encrypted backup",
                "parameters": {
                    "backup_file": {"type": "string", "description": "Path to the backup file"},
                    "password": {"type": "string", "description": "Backup encryption password"}
                },
                "returns": {"type": "object", "description": "Restore result"}
            },
            "list_backups": {
                "name": "list_backups",
                "description": "List available memory backups",
                "parameters": {},
                "returns": {"type": "object", "description": "List of available backups"}
            },
            "get_deletion_history": {
                "name": "get_deletion_history",
                "description": "Get deletion history",
                "parameters": {},
                "returns": {"type": "object", "description": "Deletion history"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute enhanced memory management MCP tool."""
        if tool_name == "delete_memory":
            return self.memory_manager.delete_memory(**kwargs)
        elif tool_name == "delete_memories_by_criteria":
            return self.memory_manager.delete_memories_by_criteria(**kwargs)
        elif tool_name == "delete_all_memories":
            return self.memory_manager.delete_all_memories(**kwargs)
        elif tool_name == "backup_memories":
            return self.memory_manager.backup_memories(**kwargs)
        elif tool_name == "restore_memories":
            return self.memory_manager.restore_memories(**kwargs)
        elif tool_name == "list_backups":
            return self.memory_manager.list_backups()
        elif tool_name == "get_deletion_history":
            return self.memory_manager.get_deletion_history()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

