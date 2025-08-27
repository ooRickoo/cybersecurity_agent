#!/usr/bin/env python3
"""
Backup Manager for Knowledge Objects System

Provides comprehensive backup and restore functionality for the distributed knowledge graph,
including automatic backup rotation, compression, and restoration capabilities.
"""

import os
import shutil
import tarfile
import zipfile
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
import tempfile

class BackupManager:
    """Manages backup and restore operations for the knowledge-objects system."""
    
    def __init__(self, base_path: str = "knowledge-objects", max_backups: int = 10):
        self.base_path = Path(base_path).resolve()
        self.backup_path = self.base_path / "backup"
        self.max_backups = max_backups
        self.backup_metadata_file = self.backup_path / "backup_metadata.json"
        
        # Setup logging
        self.logger = logging.getLogger("BackupManager")
        self.logger.setLevel(logging.INFO)
        
        # Ensure backup directory exists
        self.backup_path.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load backup metadata from file."""
        if self.backup_metadata_file.exists():
            try:
                with open(self.backup_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load backup metadata: {e}")
                return {"backups": [], "last_backup": None, "total_backups": 0}
        return {"backups": [], "last_backup": None, "total_backups": 0}
    
    def _save_metadata(self):
        """Save backup metadata to file."""
        try:
            with open(self.backup_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save backup metadata: {e}")
    
    def _get_backup_filename(self, timestamp: datetime, backup_type: str = "full") -> str:
        """Generate backup filename with timestamp and type."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"knowledge_objects_{backup_type}_{timestamp_str}.tar.gz"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _get_backup_size(self, backup_file: Path) -> int:
        """Get size of backup file in bytes."""
        try:
            return backup_file.stat().st_size
        except Exception:
            return 0
    
    def _get_backup_info(self, backup_file: Path) -> Dict:
        """Get detailed information about a backup file."""
        try:
            stat = backup_file.stat()
            return {
                "filename": backup_file.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "path": str(backup_file)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get backup info for {backup_file}: {e}")
            return {}
    
    def create_backup(self, backup_type: str = "full", description: str = "") -> Dict:
        """
        Create a new backup of the knowledge-objects system.
        
        Args:
            backup_type: Type of backup ("full", "incremental", "database_only")
            description: Optional description of the backup
            
        Returns:
            Dict containing backup information
        """
        try:
            timestamp = datetime.now()
            backup_filename = self._get_backup_filename(timestamp, backup_type)
            backup_file_path = self.backup_path / backup_filename
            
            self.logger.info(f"Creating {backup_type} backup: {backup_filename}")
            
            # Create temporary directory for backup preparation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if backup_type == "full":
                    # Full backup - everything except backup folder
                    self._create_full_backup(temp_path)
                elif backup_type == "database_only":
                    # Database-only backup
                    self._create_database_backup(temp_path)
                elif backup_type == "incremental":
                    # Incremental backup - only changed files
                    self._create_incremental_backup(temp_path)
                else:
                    raise ValueError(f"Unknown backup type: {backup_type}")
                
                # Create compressed archive
                self._create_compressed_archive(temp_path, backup_file_path)
            
            # Calculate backup hash
            backup_hash = self._calculate_file_hash(backup_file_path)
            
            # Get backup information
            backup_info = self._get_backup_info(backup_file_path)
            backup_info.update({
                "backup_type": backup_type,
                "description": description,
                "hash": backup_hash,
                "created_by": "BackupManager"
            })
            
            # Add to metadata
            self.metadata["backups"].append(backup_info)
            self.metadata["last_backup"] = timestamp.isoformat()
            self.metadata["total_backups"] += 1
            
            # Rotate old backups
            self._rotate_backups()
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Backup created successfully: {backup_filename}")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise
    
    def _create_full_backup(self, temp_path: Path):
        """Create a full backup excluding the backup folder."""
        self.logger.info("Creating full backup...")
        
        # Copy all files and directories except backup folder
        for item in self.base_path.iterdir():
            if item.name == "backup":
                continue
            
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, temp_path / item.name, dirs_exist_ok=True)
        
        # Create backup manifest
        self._create_backup_manifest(temp_path)
    
    def _create_database_backup(self, temp_path: Path):
        """Create a database-only backup."""
        self.logger.info("Creating database-only backup...")
        
        # Create databases directory
        db_path = temp_path / "databases"
        db_path.mkdir(exist_ok=True)
        
        # Copy master catalog database
        master_db = self.base_path / "master_catalog.db"
        if master_db.exists():
            shutil.copy2(master_db, db_path / "master_catalog.db")
        
        # Copy domain databases
        for domain_dir in self.base_path.iterdir():
            if domain_dir.is_dir() and domain_dir.name != "backup":
                domain_db = domain_dir / f"{domain_dir.name}.db"
                if domain_db.exists():
                    domain_db_backup = db_path / f"{domain_dir.name}.db"
                    shutil.copy2(domain_db, domain_db_backup)
        
        # Create backup manifest
        self._create_backup_manifest(temp_path, backup_type="database_only")
    
    def _create_incremental_backup(self, temp_path: Path):
        """Create an incremental backup based on last backup."""
        self.logger.info("Creating incremental backup...")
        
        if not self.metadata["backups"]:
            self.logger.info("No previous backups found, creating full backup instead")
            self._create_full_backup(temp_path)
            return
        
        # Get last backup timestamp
        last_backup = max(self.metadata["backups"], key=lambda x: x.get("created_at", ""))
        last_backup_time = datetime.fromisoformat(last_backup["created_at"])
        
        # Create incremental directory
        incremental_path = temp_path / "incremental"
        incremental_path.mkdir(exist_ok=True)
        
        # Find changed files since last backup
        changed_files = self._find_changed_files(last_backup_time)
        
        # Copy changed files
        for file_path in changed_files:
            if file_path.name == "backup":
                continue
            
            relative_path = file_path.relative_to(self.base_path)
            backup_file_path = incremental_path / relative_path
            backup_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_file_path)
        
        # Create backup manifest
        self._create_backup_manifest(temp_path, backup_type="incremental", changed_files=changed_files)
    
    def _find_changed_files(self, since_time: datetime) -> List[Path]:
        """Find files that have changed since the specified time."""
        changed_files = []
        
        for item in self.base_path.rglob("*"):
            if item.is_file() and item.name != "backup":
                try:
                    if item.stat().st_mtime > since_time.timestamp():
                        changed_files.append(item)
                except Exception:
                    continue
        
        return changed_files
    
    def _create_backup_manifest(self, temp_path: Path, backup_type: str = "full", changed_files: List[Path] = None):
        """Create a manifest file describing the backup contents."""
        manifest = {
            "backup_type": backup_type,
            "created_at": datetime.now().isoformat(),
            "source_path": str(self.base_path),
            "contents": []
        }
        
        if backup_type == "incremental" and changed_files:
            manifest["changed_files"] = [str(f.relative_to(self.base_path)) for f in changed_files]
        
        # List all files in backup
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    relative_path = file_path.relative_to(temp_path)
                    manifest["contents"].append({
                        "path": str(relative_path),
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hash": self._calculate_file_hash(file_path)
                    })
                except Exception:
                    continue
        
        # Save manifest
        with open(temp_path / "backup_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def _create_compressed_archive(self, source_path: Path, target_path: Path):
        """Create a compressed tar.gz archive."""
        with tarfile.open(target_path, "w:gz") as tar:
            tar.add(source_path, arcname="")
    
    def _rotate_backups(self):
        """Remove old backups to maintain max_backups limit."""
        if len(self.metadata["backups"]) <= self.max_backups:
            return
        
        # Sort backups by creation time (oldest first)
        sorted_backups = sorted(self.metadata["backups"], 
                              key=lambda x: x.get("created_at", ""))
        
        # Remove oldest backups
        backups_to_remove = sorted_backups[:-self.max_backups]
        
        for backup in backups_to_remove:
            try:
                backup_file = Path(backup["path"])
                if backup_file.exists():
                    backup_file.unlink()
                    self.logger.info(f"Removed old backup: {backup_file.name}")
                
                # Remove from metadata
                self.metadata["backups"].remove(backup)
                
            except Exception as e:
                self.logger.warning(f"Failed to remove old backup {backup.get('filename', 'Unknown')}: {e}")
    
    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        return self.metadata["backups"]
    
    def get_backup_info(self, backup_filename: str) -> Optional[Dict]:
        """Get information about a specific backup."""
        for backup in self.metadata["backups"]:
            if backup.get("filename") == backup_filename:
                return backup
        return None
    
    def restore_backup(self, backup_filename: str, restore_path: str = None, 
                      verify_hash: bool = True) -> Dict:
        """
        Restore a backup to the specified path.
        
        Args:
            backup_filename: Name of the backup file to restore
            restore_path: Path to restore to (defaults to original location)
            verify_hash: Whether to verify backup integrity
            
        Returns:
            Dict containing restore information
        """
        try:
            # Find backup file
            backup_file = self.backup_path / backup_filename
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            # Get backup info
            backup_info = self.get_backup_info(backup_filename)
            if not backup_info:
                raise ValueError(f"Backup not found in metadata: {backup_filename}")
            
            # Verify hash if requested
            if verify_hash:
                current_hash = self._calculate_file_hash(backup_file)
                if current_hash != backup_info.get("hash", ""):
                    raise ValueError(f"Backup hash verification failed. Expected: {backup_info.get('hash')}, Got: {current_hash}")
            
            # Determine restore path
            if restore_path is None:
                restore_path = self.base_path.parent / f"restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            restore_path = Path(restore_path)
            restore_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Restoring backup {backup_filename} to {restore_path}")
            
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                # Security: Validate tar members before extraction
                for member in tar.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        raise ValueError(f"Potentially dangerous path in tar file: {member.name}")
                tar.extractall(restore_path)
            
            # Verify manifest if it exists
            manifest_path = restore_path / "backup_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Verify file integrity
                verification_results = self._verify_restored_files(restore_path, manifest)
                
                restore_info = {
                    "backup_filename": backup_filename,
                    "restore_path": str(restore_path),
                    "backup_type": manifest.get("backup_type", "unknown"),
                    "files_restored": len(manifest.get("contents", [])),
                    "verification_results": verification_results,
                    "restored_at": datetime.now().isoformat()
                }
            else:
                restore_info = {
                    "backup_filename": backup_filename,
                    "restore_path": str(restore_path),
                    "backup_type": "unknown",
                    "files_restored": "unknown",
                    "verification_results": "no_manifest",
                    "restored_at": datetime.now().isoformat()
                }
            
            self.logger.info(f"Backup restored successfully to {restore_path}")
            return restore_info
            
        except Exception as e:
            self.logger.error(f"Backup restoration failed: {e}")
            raise
    
    def _verify_restored_files(self, restore_path: Path, manifest: Dict) -> Dict:
        """Verify the integrity of restored files."""
        verification_results = {
            "total_files": len(manifest.get("contents", [])),
            "verified_files": 0,
            "failed_verifications": 0,
            "errors": []
        }
        
        for file_info in manifest.get("contents", []):
            try:
                file_path = restore_path / file_info["path"]
                if file_path.exists():
                    current_hash = self._calculate_file_hash(file_path)
                    if current_hash == file_info.get("hash", ""):
                        verification_results["verified_files"] += 1
                    else:
                        verification_results["failed_verifications"] += 1
                        verification_results["errors"].append(
                            f"Hash mismatch for {file_info['path']}"
                        )
                else:
                    verification_results["failed_verifications"] += 1
                    verification_results["errors"].append(
                        f"File not found: {file_info['path']}"
                    )
            except Exception as e:
                verification_results["failed_verifications"] += 1
                verification_results["errors"].append(
                    f"Error verifying {file_info.get('path', 'Unknown')}: {e}"
                )
        
        return verification_results
    
    def delete_backup(self, backup_filename: str) -> bool:
        """Delete a specific backup."""
        try:
            backup_file = self.backup_path / backup_filename
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            # Remove from metadata
            backup_info = self.get_backup_info(backup_filename)
            if backup_info:
                self.metadata["backups"].remove(backup_info)
                self._save_metadata()
            
            # Delete file
            backup_file.unlink()
            
            self.logger.info(f"Backup deleted: {backup_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_filename}: {e}")
            return False
    
    def cleanup_old_backups(self, days_old: int = 30) -> int:
        """Clean up backups older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        backups_to_delete = []
        for backup in self.metadata["backups"]:
            try:
                backup_date = datetime.fromisoformat(backup["created_at"])
                if backup_date < cutoff_date:
                    backups_to_delete.append(backup)
            except Exception:
                continue
        
        for backup in backups_to_delete:
            if self.delete_backup(backup["filename"]):
                deleted_count += 1
        
        self.logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    def get_backup_statistics(self) -> Dict:
        """Get comprehensive backup statistics."""
        total_size = sum(backup.get("size_bytes", 0) for backup in self.metadata["backups"])
        total_size_mb = round(total_size / (1024 * 1024), 2)
        
        backup_types = {}
        for backup in self.metadata["backups"]:
            backup_type = backup.get("backup_type", "unknown")
            backup_types[backup_type] = backup_types.get(backup_type, 0) + 1
        
        return {
            "total_backups": len(self.metadata["backups"]),
            "total_size_bytes": total_size,
            "total_size_mb": total_size_mb,
            "backup_types": backup_types,
            "max_backups": self.max_backups,
            "available_space": self._get_available_space(),
            "last_backup": self.metadata.get("last_backup"),
            "backup_path": str(self.backup_path)
        }
    
    def _get_available_space(self) -> Dict:
        """Get available disk space information."""
        try:
            stat = shutil.disk_usage(self.backup_path)
            return {
                "total_gb": round(stat.total / (1024**3), 2),
                "used_gb": round(stat.used / (1024**3), 2),
                "free_gb": round(stat.free / (1024**3), 2),
                "free_percent": round((stat.free / stat.total) * 100, 2)
            }
        except Exception:
            return {"error": "Unable to determine disk space"}
    
    def validate_backup_integrity(self, backup_filename: str) -> Dict:
        """Validate the integrity of a backup file."""
        try:
            backup_file = self.backup_path / backup_filename
            if not backup_file.exists():
                return {"valid": False, "error": "Backup file not found"}
            
            # Get backup info
            backup_info = self.get_backup_info(backup_filename)
            if not backup_info:
                return {"valid": False, "error": "Backup not found in metadata"}
            
            # Verify hash
            current_hash = self._calculate_file_hash(backup_file)
            hash_valid = current_hash == backup_info.get("hash", "")
            
            # Check if file is readable
            try:
                with tarfile.open(backup_file, "r:gz") as tar:
                    file_list = tar.getnames()
                tar_valid = True
            except Exception as e:
                tar_valid = False
                tar_error = str(e)
            
            validation_result = {
                "valid": hash_valid and tar_valid,
                "backup_filename": backup_filename,
                "hash_valid": hash_valid,
                "tar_valid": tar_valid,
                "expected_hash": backup_info.get("hash", ""),
                "actual_hash": current_hash,
                "file_count": len(file_list) if tar_valid else 0,
                "file_size_bytes": backup_file.stat().st_size,
                "file_size_mb": round(backup_file.stat().st_size / (1024 * 1024), 2)
            }
            
            if not tar_valid:
                validation_result["tar_error"] = tar_error
            
            return validation_result
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def close(self):
        """Clean up resources."""
        self._save_metadata()


def main():
    """Command-line interface for backup manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Objects Backup Manager")
    parser.add_argument("action", choices=["backup", "restore", "list", "info", "delete", "cleanup", "validate", "stats"],
                       help="Action to perform")
    parser.add_argument("--backup-file", help="Backup filename for restore/delete/validate actions")
    parser.add_argument("--backup-type", choices=["full", "incremental", "database_only"], 
                       default="full", help="Type of backup to create")
    parser.add_argument("--description", help="Description for the backup")
    parser.add_argument("--restore-path", help="Path to restore backup to")
    parser.add_argument("--max-backups", type=int, default=10, help="Maximum number of backups to keep")
    parser.add_argument("--days-old", type=int, default=30, help="Days old for cleanup action")
    parser.add_argument("--verify-hash", action="store_true", default=True, 
                       help="Verify backup hash during restore")
    
    args = parser.parse_args()
    
    # Initialize backup manager
    backup_manager = BackupManager(".", max_backups=args.max_backups)
    
    try:
        if args.action == "backup":
            result = backup_manager.create_backup(args.backup_type, args.description or "")
            print(f"✅ Backup created successfully: {result['filename']}")
            print(f"   Size: {result['size_mb']} MB")
            print(f"   Type: {result['backup_type']}")
            print(f"   Hash: {result['hash'][:16]}...")
        
        elif args.action == "restore":
            if not args.backup_file:
                print("❌ Error: --backup-file is required for restore action")
                return
            
            result = backup_manager.restore_backup(
                args.backup_file, 
                args.restore_path, 
                args.verify_hash
            )
            print(f"✅ Backup restored successfully to: {result['restore_path']}")
            print(f"   Files restored: {result['files_restored']}")
            print(f"   Backup type: {result['backup_type']}")
        
        elif args.action == "list":
            backups = backup_manager.list_backups()
            if not backups:
                print("📁 No backups found")
                return
            
            print(f"📁 Found {len(backups)} backups:")
            print("-" * 80)
            for backup in sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True):
                print(f"📦 {backup['filename']}")
                print(f"   Size: {backup['size_mb']} MB")
                print(f"   Type: {backup.get('backup_type', 'unknown')}")
                print(f"   Created: {backup.get('created_at', 'unknown')}")
                if backup.get('description'):
                    print(f"   Description: {backup['description']}")
                print()
        
        elif args.action == "info":
            if not args.backup_file:
                print("❌ Error: --backup-file is required for info action")
                return
            
            backup_info = backup_manager.get_backup_info(args.backup_file)
            if not backup_info:
                print(f"❌ Backup not found: {args.backup_file}")
                return
            
            print(f"📋 Backup Information: {args.backup_file}")
            print("-" * 40)
            for key, value in backup_info.items():
                print(f"{key}: {value}")
        
        elif args.action == "delete":
            if not args.backup_file:
                print("❌ Error: --backup-file is required for delete action")
                return
            
            if backup_manager.delete_backup(args.backup_file):
                print(f"✅ Backup deleted: {args.backup_file}")
            else:
                print(f"❌ Failed to delete backup: {args.backup_file}")
        
        elif args.action == "cleanup":
            deleted_count = backup_manager.cleanup_old_backups(args.days_old)
            print(f"✅ Cleaned up {deleted_count} backups older than {args.days_old} days")
        
        elif args.action == "validate":
            if not args.backup_file:
                print("❌ Error: --backup-file is required for validate action")
                return
            
            result = backup_manager.validate_backup_integrity(args.backup_file)
            if result["valid"]:
                print(f"✅ Backup validation successful: {args.backup_file}")
                print(f"   File count: {result['file_count']}")
                print(f"   Size: {result['file_size_mb']} MB")
                print(f"   Hash: {result['actual_hash'][:16]}...")
            else:
                print(f"❌ Backup validation failed: {args.backup_file}")
                if "error" in result:
                    print(f"   Error: {result['error']}")
        
        elif args.action == "stats":
            stats = backup_manager.get_backup_statistics()
            print("📊 Backup Statistics:")
            print("-" * 30)
            print(f"Total backups: {stats['total_backups']}")
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Max backups: {stats['max_backups']}")
            print(f"Last backup: {stats['last_backup']}")
            print(f"Backup path: {stats['backup_path']}")
            
            if "backup_types" in stats:
                print("\nBackup types:")
                for backup_type, count in stats["backup_types"].items():
                    print(f"  {backup_type}: {count}")
            
            if "available_space" in stats and "error" not in stats["available_space"]:
                space = stats["available_space"]
                print(f"\nDisk space:")
                print(f"  Free: {space['free_gb']} GB ({space['free_percent']}%)")
                print(f"  Used: {space['used_gb']} GB")
                print(f"  Total: {space['total_gb']} GB")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        backup_manager.close()


if __name__ == "__main__":
    main()
