#!/usr/bin/env python3
"""
File Tools Manager - Comprehensive File Operations for Cybersecurity Analysis
Provides file manipulation, analysis, and forensics capabilities.
"""

import os
import hashlib
import mimetypes
import magic
import json
import logging
import shutil
import tempfile
import zipfile
import tarfile
import gzip
import bz2
import lzma
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import time

logger = logging.getLogger(__name__)

class FileType(Enum):
    """File type categories."""
    TEXT = "text"
    BINARY = "binary"
    ARCHIVE = "archive"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EXECUTABLE = "executable"
    DOCUMENT = "document"
    UNKNOWN = "unknown"

class SecurityLevel(Enum):
    """File security levels."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"

@dataclass
class FileMetadata:
    """File metadata structure."""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
    created_time: str
    modified_time: str
    accessed_time: str
    permissions: str
    owner: str
    group: str
    md5_hash: str
    sha1_hash: str
    sha256_hash: str
    security_level: str
    entropy: float
    strings: List[str]
    metadata: Dict[str, Any]

@dataclass
class FileAnalysisResult:
    """File analysis result."""
    file_metadata: FileMetadata
    analysis_timestamp: str
    analysis_duration: float
    findings: List[str]
    recommendations: List[str]
    risk_score: float
    tags: List[str]

class FileToolsManager:
    """Comprehensive file operations manager for cybersecurity analysis."""
    
    def __init__(self):
        """Initialize file tools manager."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.gettempdir()) / "cybersecurity_agent"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file type detection
        self._initialize_file_detection()
        
        logger.info("🚀 File Tools Manager initialized")
    
    def _initialize_file_detection(self):
        """Initialize file type detection capabilities."""
        try:
            # Try to use python-magic for better file type detection
            import magic
            self.magic_available = True
            self.magic_instance = magic.Magic(mime=True)
        except ImportError:
            self.magic_available = False
            self.logger.warning("python-magic not available. Install with: pip install python-magic")
        
        # Initialize MIME types
        mimetypes.init()
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> FileMetadata:
        """Get comprehensive file metadata."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            
            # Get file type
            mime_type = self._get_mime_type(file_path)
            file_type = self._categorize_file_type(file_path, mime_type)
            
            # Calculate hashes
            hashes = self._calculate_hashes(file_path)
            
            # Get file permissions
            permissions = oct(stat.st_mode)[-3:]
            
            # Get owner/group (Unix-like systems)
            try:
                import pwd
                import grp
                owner = pwd.getpwuid(stat.st_uid).pw_name
                group = grp.getgrgid(stat.st_gid).gr_name
            except (ImportError, KeyError):
                owner = str(stat.st_uid)
                group = str(stat.st_gid)
            
            # Calculate entropy
            entropy = self._calculate_entropy(file_path)
            
            # Extract strings
            strings = self._extract_strings(file_path)
            
            # Determine security level
            security_level = self._assess_security_level(file_path, file_type, entropy)
            
            return FileMetadata(
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_size=stat.st_size,
                file_type=file_type.value,
                mime_type=mime_type,
                created_time=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                accessed_time=datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat(),
                permissions=permissions,
                owner=owner,
                group=group,
                md5_hash=hashes['md5'],
                sha1_hash=hashes['sha1'],
                sha256_hash=hashes['sha256'],
                security_level=security_level.value,
                entropy=entropy,
                strings=strings[:100],  # Limit to first 100 strings
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get file metadata for {file_path}: {e}")
            raise
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type of file."""
        try:
            if self.magic_available:
                return self.magic_instance.from_file(str(file_path))
            else:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                return mime_type or "application/octet-stream"
        except Exception:
            return "application/octet-stream"
    
    def _categorize_file_type(self, file_path: Path, mime_type: str) -> FileType:
        """Categorize file type."""
        extension = file_path.suffix.lower()
        
        # Text files
        if mime_type.startswith('text/') or extension in ['.txt', '.log', '.csv', '.json', '.xml', '.yaml', '.yml']:
            return FileType.TEXT
        
        # Executables
        if (mime_type in ['application/x-executable', 'application/x-msdownload'] or 
            extension in ['.exe', '.dll', '.so', '.bin', '.app']):
            return FileType.EXECUTABLE
        
        # Archives
        if (mime_type.startswith('application/') and 
            any(arch in mime_type for arch in ['zip', 'tar', 'gzip', 'bzip2', 'x-tar', 'x-7z'])):
            return FileType.ARCHIVE
        
        # Images
        if mime_type.startswith('image/'):
            return FileType.IMAGE
        
        # Audio
        if mime_type.startswith('audio/'):
            return FileType.AUDIO
        
        # Video
        if mime_type.startswith('video/'):
            return FileType.VIDEO
        
        # Documents
        if mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return FileType.DOCUMENT
        
        # Check if it's binary
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return FileType.BINARY
        except Exception:
            pass
        
        return FileType.UNKNOWN
    
    def _calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate file hashes."""
        hashes = {'md5': '', 'sha1': '', 'sha256': ''}
        
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks for large files
                md5_hash = hashlib.md5()
                sha1_hash = hashlib.sha1()
                sha256_hash = hashlib.sha256()
                
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                    sha1_hash.update(chunk)
                    sha256_hash.update(chunk)
                
                hashes['md5'] = md5_hash.hexdigest()
                hashes['sha1'] = sha1_hash.hexdigest()
                hashes['sha256'] = sha256_hash.hexdigest()
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate hashes for {file_path}: {e}")
        
        return hashes
    
    def _calculate_entropy(self, file_path: Path) -> float:
        """Calculate file entropy."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            if not data:
                return 0.0
            
            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0.0
            data_len = len(data)
            
            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * (probability.bit_length() - 1)
            
            return entropy
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate entropy for {file_path}: {e}")
            return 0.0
    
    def _extract_strings(self, file_path: Path, min_length: int = 4) -> List[str]:
        """Extract printable strings from file."""
        strings = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            current_string = ""
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= min_length:
                        strings.append(current_string)
                    current_string = ""
            
            # Add final string if it meets criteria
            if len(current_string) >= min_length:
                strings.append(current_string)
                
        except Exception as e:
            logger.error(f"❌ Failed to extract strings from {file_path}: {e}")
        
        return strings
    
    def _assess_security_level(self, file_path: Path, file_type: FileType, entropy: float) -> SecurityLevel:
        """Assess file security level."""
        # High entropy might indicate packed/encrypted content
        if entropy > 7.5:
            return SecurityLevel.SUSPICIOUS
        
        # Executables are potentially risky
        if file_type == FileType.EXECUTABLE:
            return SecurityLevel.SUSPICIOUS
        
        # Check for suspicious file extensions
        suspicious_extensions = ['.scr', '.pif', '.bat', '.cmd', '.com', '.vbs', '.js', '.jar']
        if file_path.suffix.lower() in suspicious_extensions:
            return SecurityLevel.SUSPICIOUS
        
        # Check for double extensions (common malware technique)
        if '.' in file_path.stem and file_path.suffix.lower() in ['.exe', '.scr', '.pif']:
            return SecurityLevel.SUSPICIOUS
        
        return SecurityLevel.SAFE
    
    def analyze_file(self, file_path: Union[str, Path]) -> FileAnalysisResult:
        """Perform comprehensive file analysis."""
        start_time = time.time()
        
        try:
            # Get file metadata
            metadata = self.get_file_metadata(file_path)
            
            # Analyze file content
            findings = []
            recommendations = []
            risk_score = 0.0
            tags = []
            
            # Check for suspicious characteristics
            if metadata.entropy > 7.5:
                findings.append("High entropy detected - file may be packed or encrypted")
                risk_score += 0.3
                tags.append("high-entropy")
            
            if metadata.file_type == "executable":
                findings.append("Executable file detected")
                risk_score += 0.2
                tags.append("executable")
            
            # Check for suspicious strings
            suspicious_strings = [
                "cmd.exe", "powershell", "regsvr32", "rundll32", "wscript", "cscript",
                "download", "execute", "payload", "backdoor", "trojan", "keylogger"
            ]
            
            found_suspicious = []
            for string in metadata.strings:
                for suspicious in suspicious_strings:
                    if suspicious.lower() in string.lower():
                        found_suspicious.append(string)
                        break
            
            if found_suspicious:
                findings.append(f"Suspicious strings found: {found_suspicious[:5]}")
                risk_score += 0.4
                tags.append("suspicious-strings")
            
            # Generate recommendations
            if risk_score > 0.5:
                recommendations.append("File should be quarantined and analyzed further")
                recommendations.append("Run additional malware scanning tools")
            elif risk_score > 0.2:
                recommendations.append("File should be monitored for suspicious activity")
            else:
                recommendations.append("File appears safe for normal use")
            
            analysis_duration = time.time() - start_time
            
            return FileAnalysisResult(
                file_metadata=metadata,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(),
                analysis_duration=analysis_duration,
                findings=findings,
                recommendations=recommendations,
                risk_score=min(risk_score, 1.0),
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze file {file_path}: {e}")
            raise
    
    def extract_archive(self, archive_path: Union[str, Path], extract_to: Union[str, Path] = None) -> Dict[str, Any]:
        """Extract archive file."""
        try:
            archive_path = Path(archive_path)
            
            if not archive_path.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")
            
            if extract_to is None:
                extract_to = self.temp_dir / f"extracted_{archive_path.stem}"
            else:
                extract_to = Path(extract_to)
            
            extract_to.mkdir(parents=True, exist_ok=True)
            
            extracted_files = []
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    extracted_files = zip_ref.namelist()
            
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
                    extracted_files = tar_ref.getnames()
            
            elif archive_path.suffix.lower() == '.gz':
                with gzip.open(archive_path, 'rb') as gz_ref:
                    output_file = extract_to / archive_path.stem
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(gz_ref, out_ref)
                    extracted_files = [archive_path.stem]
            
            elif archive_path.suffix.lower() == '.bz2':
                with bz2.open(archive_path, 'rb') as bz2_ref:
                    output_file = extract_to / archive_path.stem
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(bz2_ref, out_ref)
                    extracted_files = [archive_path.stem]
            
            elif archive_path.suffix.lower() == '.xz':
                with lzma.open(archive_path, 'rb') as xz_ref:
                    output_file = extract_to / archive_path.stem
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(xz_ref, out_ref)
                    extracted_files = [archive_path.stem]
            
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            return {
                "success": True,
                "extract_path": str(extract_to),
                "extracted_files": extracted_files,
                "file_count": len(extracted_files)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract archive {archive_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_archive(self, source_path: Union[str, Path], archive_path: Union[str, Path], 
                      archive_type: str = "zip") -> bool:
        """Create archive from source path."""
        try:
            source_path = Path(source_path)
            archive_path = Path(archive_path)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")
            
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if archive_type.lower() == "zip":
                if source_path.is_file():
                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        zip_ref.write(source_path, source_path.name)
                else:
                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                        for file_path in source_path.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(source_path)
                                zip_ref.write(file_path, arcname)
            
            elif archive_type.lower() == "tar":
                with tarfile.open(archive_path, 'w') as tar_ref:
                    if source_path.is_file():
                        tar_ref.add(source_path, arcname=source_path.name)
                    else:
                        tar_ref.add(source_path, arcname=source_path.name)
            
            else:
                raise ValueError(f"Unsupported archive type: {archive_type}")
            
            logger.info(f"✅ Created {archive_type} archive: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create archive: {e}")
            return False
    
    def find_files(self, search_path: Union[str, Path], pattern: str = "*", 
                   file_type: FileType = None, min_size: int = None, 
                   max_size: int = None) -> List[Path]:
        """Find files matching criteria."""
        try:
            search_path = Path(search_path)
            
            if not search_path.exists():
                return []
            
            files = []
            
            for file_path in search_path.rglob(pattern):
                if file_path.is_file():
                    # Check file type
                    if file_type:
                        mime_type = self._get_mime_type(file_path)
                        if self._categorize_file_type(file_path, mime_type) != file_type:
                            continue
                    
                    # Check file size
                    if min_size is not None and file_path.stat().st_size < min_size:
                        continue
                    
                    if max_size is not None and file_path.stat().st_size > max_size:
                        continue
                    
                    files.append(file_path)
            
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to find files: {e}")
            return []
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path], 
                  preserve_metadata: bool = True) -> bool:
        """Copy file with optional metadata preservation."""
        try:
            source = Path(source)
            destination = Path(destination)
            
            if not source.exists():
                raise FileNotFoundError(f"Source not found: {source}")
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            if preserve_metadata:
                shutil.copy2(source, destination)
            else:
                shutil.copy(source, destination)
            
            logger.info(f"✅ Copied {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy file: {e}")
            return False
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move file to new location."""
        try:
            source = Path(source)
            destination = Path(destination)
            
            if not source.exists():
                raise FileNotFoundError(f"Source not found: {source}")
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            
            logger.info(f"✅ Moved {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to move file: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path], secure_delete: bool = False) -> bool:
        """Delete file with optional secure deletion."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return True  # Already deleted
            
            if secure_delete:
                # Overwrite file with random data before deletion
                file_size = file_path.stat().st_size
                with open(file_path, 'r+b') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            file_path.unlink()
            
            logger.info(f"✅ Deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete file: {e}")
            return False
    
    def get_directory_tree(self, directory_path: Union[str, Path], max_depth: int = 3) -> Dict[str, Any]:
        """Get directory tree structure."""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                return {}
            
            def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
                if current_depth >= max_depth:
                    return {"type": "directory", "truncated": True}
                
                tree = {
                    "name": path.name,
                    "type": "directory" if path.is_dir() else "file",
                    "path": str(path),
                    "size": path.stat().st_size if path.is_file() else 0,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                }
                
                if path.is_dir():
                    tree["children"] = []
                    try:
                        for child in sorted(path.iterdir()):
                            tree["children"].append(build_tree(child, current_depth + 1))
                    except PermissionError:
                        tree["error"] = "Permission denied"
                
                return tree
            
            return build_tree(directory_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to get directory tree: {e}")
            return {}
    
    def cleanup_temp_files(self) -> int:
        """Clean up temporary files."""
        try:
            if not self.temp_dir.exists():
                return 0
            
            deleted_count = 0
            for file_path in self.temp_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception:
                        pass
            
            logger.info(f"✅ Cleaned up {deleted_count} temporary files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup temp files: {e}")
            return 0

# Example usage and testing
if __name__ == "__main__":
    # Test the file tools manager
    ftm = FileToolsManager()
    
    # Test with a sample file
    test_file = Path("test_file.txt")
    test_file.write_text("This is a test file for cybersecurity analysis.")
    
    try:
        # Get file metadata
        metadata = ftm.get_file_metadata(test_file)
        print(f"✅ File metadata: {metadata.file_name}, Size: {metadata.file_size}, Type: {metadata.file_type}")
        
        # Analyze file
        analysis = ftm.analyze_file(test_file)
        print(f"✅ Analysis complete: Risk Score: {analysis.risk_score}, Findings: {len(analysis.findings)}")
        
        # Create archive
        archive_path = Path("test_archive.zip")
        ftm.create_archive(test_file, archive_path)
        print(f"✅ Created archive: {archive_path}")
        
        # Extract archive
        extract_result = ftm.extract_archive(archive_path)
        print(f"✅ Extracted archive: {extract_result}")
        
    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
        ftm.cleanup_temp_files()
