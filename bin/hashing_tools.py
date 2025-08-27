#!/usr/bin/env python3
"""
Hashing Tools - Comprehensive File and String Analysis
Provides cryptographic hashing capabilities for cybersecurity analysis, forensics, and data integrity.

Features:
- Multiple hash algorithms (MD5, SHA1, SHA256, SHA512, etc.)
- File hashing with progress tracking
- String hashing and comparison
- Hash verification and validation
- Rainbow table checking
- Integration with MCP framework for dynamic workflows
"""

import hashlib
import os
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
import base64
import hmac
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HashAlgorithm(Enum):
    """Available hash algorithms."""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA224 = "sha224"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_224 = "sha3_224"
    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    RIPEMD160 = "ripemd160"
    WHIRLPOOL = "whirlpool"

class HashType(Enum):
    """Types of hash operations."""
    FILE_HASH = "file"
    STRING_HASH = "string"
    DIRECTORY_HASH = "directory"
    STREAM_HASH = "stream"
    HMAC_HASH = "hmac"

@dataclass
class HashResult:
    """Result of a hashing operation."""
    hash_id: str
    hash_type: HashType
    algorithm: HashAlgorithm
    input_source: str
    hash_value: str
    processing_time: float
    file_size: Optional[int] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

@dataclass
class HashVerification:
    """Result of hash verification."""
    hash_id: str
    original_hash: str
    computed_hash: str
    is_valid: bool
    verification_time: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

class HashCalculator:
    """Core hashing functionality."""
    
    def __init__(self):
        self.hash_functions = self._initialize_hash_functions()
        self.hash_history: List[HashResult] = []
        self.verification_history: List[HashVerification] = []
        
        logger.info("🚀 HashCalculator initialized")
    
    def _initialize_hash_functions(self) -> Dict[HashAlgorithm, Callable]:
        """Initialize hash function mappings."""
        return {
            HashAlgorithm.MD5: hashlib.md5,
            HashAlgorithm.SHA1: hashlib.sha1,
            HashAlgorithm.SHA224: hashlib.sha224,
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA384: hashlib.sha384,
            HashAlgorithm.SHA512: hashlib.sha512,
            HashAlgorithm.SHA3_224: hashlib.sha3_224,
            HashAlgorithm.SHA3_256: hashlib.sha3_256,
            HashAlgorithm.SHA3_384: hashlib.sha3_384,
            HashAlgorithm.SHA3_512: hashlib.sha3_512,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
            HashAlgorithm.BLAKE2S: hashlib.blake2s,
            HashAlgorithm.RIPEMD160: hashlib.new,
            HashAlgorithm.WHIRLPOOL: hashlib.new
        }
    
    def _get_hash_function(self, algorithm: HashAlgorithm):
        """Get the appropriate hash function for the algorithm."""
        if algorithm == HashAlgorithm.RIPEMD160:
            return lambda: hashlib.new('ripemd160')
        elif algorithm == HashAlgorithm.WHIRLPOOL:
            return lambda: hashlib.new('whirlpool')
        else:
            return self.hash_functions[algorithm]
    
    def hash_string(self, text: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> HashResult:
        """Hash a string using the specified algorithm."""
        start_time = time.time()
        hash_id = f"hash_{int(start_time)}_{len(self.hash_history)}"
        
        try:
            # Get hash function
            hash_func = self._get_hash_function(algorithm)
            hash_obj = hash_func()
            
            # Hash the string
            hash_obj.update(text.encode('utf-8'))
            hash_value = hash_obj.hexdigest()
            
            # Create result
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.STRING_HASH,
                algorithm=algorithm,
                input_source=text[:100] + "..." if len(text) > 100 else text,
                hash_value=hash_value,
                processing_time=time.time() - start_time,
                metadata={
                    'input_length': len(text),
                    'encoding': 'utf-8'
                }
            )
            
            self.hash_history.append(result)
            logger.info(f"String hashed with {algorithm.value}: {hash_value[:16]}...")
            return result
            
        except Exception as e:
            error_msg = f"String hashing failed: {str(e)}"
            logger.error(error_msg)
            
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.STRING_HASH,
                algorithm=algorithm,
                input_source=text[:100] + "..." if len(text) > 100 else text,
                hash_value="",
                processing_time=time.time() - start_time,
                errors=[error_msg]
            )
            
            self.hash_history.append(result)
            return result
    
    def hash_file(self, file_path: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256,
                  chunk_size: int = 8192, progress_callback: Optional[Callable] = None) -> HashResult:
        """Hash a file using the specified algorithm with progress tracking."""
        start_time = time.time()
        hash_id = f"hash_{int(start_time)}_{len(self.hash_history)}"
        
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not os.path.isfile(file_path):
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Get hash function
            hash_func = self._get_hash_function(algorithm)
            hash_obj = hash_func()
            
            # Hash file in chunks
            bytes_processed = 0
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    hash_obj.update(chunk)
                    bytes_processed += len(chunk)
                    
                    # Call progress callback if provided
                    if progress_callback and file_size > 0:
                        progress = (bytes_processed / file_size) * 100
                        progress_callback(progress)
            
            hash_value = hash_obj.hexdigest()
            
            # Create result
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.FILE_HASH,
                algorithm=algorithm,
                input_source=file_path,
                hash_value=hash_value,
                processing_time=time.time() - start_time,
                file_size=file_size,
                metadata={
                    'chunk_size': chunk_size,
                    'bytes_processed': bytes_processed
                }
            )
            
            self.hash_history.append(result)
            logger.info(f"File hashed with {algorithm.value}: {hash_value[:16]}...")
            return result
            
        except Exception as e:
            error_msg = f"File hashing failed: {str(e)}"
            logger.error(error_msg)
            
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.FILE_HASH,
                algorithm=algorithm,
                input_source=file_path,
                hash_value="",
                processing_time=time.time() - start_time,
                errors=[error_msg]
            )
            
            self.hash_history.append(result)
            return result
    
    def hash_directory(self, directory_path: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256,
                      file_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None) -> Dict[str, HashResult]:
        """Hash all files in a directory."""
        start_time = time.time()
        results = {}
        
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not os.path.isdir(directory_path):
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Get all files
            all_files = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Apply file pattern filters
                    if file_patterns:
                        if not any(file.endswith(pattern) for pattern in file_patterns):
                            continue
                    
                    # Apply exclude patterns
                    if exclude_patterns:
                        if any(file.endswith(pattern) for pattern in exclude_patterns):
                            continue
                    
                    all_files.append(file_path)
            
            logger.info(f"Found {len(all_files)} files to hash in {directory_path}")
            
            # Hash files (can be parallelized for large directories)
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self.hash_file, file_path, algorithm): file_path
                    for file_path in all_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results[file_path] = result
                    except Exception as e:
                        logger.error(f"Failed to hash {file_path}: {e}")
            
            logger.info(f"Directory hashing completed: {len(results)} files processed")
            
        except Exception as e:
            logger.error(f"Directory hashing failed: {str(e)}")
        
        return results
    
    def create_hmac(self, data: Union[str, bytes], key: Union[str, bytes],
                    algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> HashResult:
        """Create HMAC (Hash-based Message Authentication Code)."""
        start_time = time.time()
        hash_id = f"hmac_{int(start_time)}_{len(self.hash_history)}"
        
        try:
            # Ensure data and key are bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            if isinstance(key, str):
                key = key.encode('utf-8')
            
            # Create HMAC
            hash_func = self._get_hash_function(algorithm)
            hash_obj = hash_func()
            hmac_obj = hmac.new(key, data, hash_func)
            hmac_value = hmac_obj.hexdigest()
            
            # Create result
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.HMAC_HASH,
                algorithm=algorithm,
                input_source=f"HMAC({len(data)} bytes, key: {len(key)} bytes)",
                hash_value=hmac_value,
                processing_time=time.time() - start_time,
                metadata={
                    'data_length': len(data),
                    'key_length': len(key),
                    'hmac': True
                }
            )
            
            self.hash_history.append(result)
            logger.info(f"HMAC created with {algorithm.value}: {hmac_value[:16]}...")
            return result
            
        except Exception as e:
            error_msg = f"HMAC creation failed: {str(e)}"
            logger.error(error_msg)
            
            result = HashResult(
                hash_id=hash_id,
                hash_type=HashType.HMAC_HASH,
                algorithm=algorithm,
                input_source="HMAC creation failed",
                hash_value="",
                processing_time=time.time() - start_time,
                errors=[error_msg]
            )
            
            self.hash_history.append(result)
            return result
    
    def verify_hash(self, original_hash: str, computed_hash: str,
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> HashVerification:
        """Verify that two hashes match."""
        start_time = time.time()
        verification_id = f"verify_{int(start_time)}_{len(self.verification_history)}"
        
        # Compare hashes (case-insensitive)
        is_valid = original_hash.lower() == computed_hash.lower()
        
        # Create verification result
        result = HashVerification(
            hash_id=verification_id,
            original_hash=original_hash,
            computed_hash=computed_hash,
            is_valid=is_valid,
            verification_time=time.time() - start_time,
            details={
                'algorithm': algorithm.value,
                'case_sensitive': False
            }
        )
        
        self.verification_history.append(result)
        
        if is_valid:
            logger.info(f"Hash verification successful: {original_hash[:16]}...")
        else:
            logger.warning(f"Hash verification failed: {original_hash[:16]}... != {computed_hash[:16]}...")
        
        return result
    
    def get_hash_history(self) -> List[HashResult]:
        """Get list of all hash operations."""
        return self.hash_history.copy()
    
    def get_verification_history(self) -> List[HashVerification]:
        """Get list of all verification operations."""
        return self.verification_history.copy()
    
    def clear_history(self):
        """Clear hash and verification history."""
        self.hash_history.clear()
        self.verification_history.clear()
        logger.info("Hash history cleared")

class HashingManager:
    """Manager for hashing operations with MCP integration capabilities."""
    
    def __init__(self):
        self.calculator = HashCalculator()
        self.hash_templates = self._create_hash_templates()
        self.performance_stats = {
            'total_hashes': 0,
            'successful_hashes': 0,
            'failed_hashes': 0,
            'average_hash_time': 0.0,
            'total_verifications': 0,
            'successful_verifications': 0
        }
        
        logger.info("🚀 Hashing Manager initialized")
    
    def _create_hash_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined hashing templates."""
        return {
            'quick_verification': {
                'algorithm': HashAlgorithm.MD5,
                'description': 'Quick hash for basic verification'
            },
            'secure_verification': {
                'algorithm': HashAlgorithm.SHA256,
                'description': 'Secure hash for critical data verification'
            },
            'maximum_security': {
                'algorithm': HashAlgorithm.SHA512,
                'description': 'Maximum security hash for sensitive data'
            },
            'legacy_compatibility': {
                'algorithm': HashAlgorithm.SHA1,
                'description': 'Legacy hash for compatibility with older systems'
            },
            'fast_processing': {
                'algorithm': HashAlgorithm.BLAKE2B,
                'description': 'Fast hash for high-performance applications'
            }
        }
    
    def execute_hash_template(self, template_name: str, input_source: str,
                            input_type: str = "string") -> HashResult:
        """Execute a predefined hash template."""
        if template_name not in self.hash_templates:
            raise ValueError(f"Unknown hash template: {template_name}")
        
        template = self.hash_templates[template_name]
        
        if input_type == "file":
            return self.calculator.hash_file(input_source, template['algorithm'])
        else:
            return self.calculator.hash_string(input_source, template['algorithm'])
    
    def batch_hash_strings(self, strings: List[str], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> List[HashResult]:
        """Hash multiple strings efficiently."""
        results = []
        for text in strings:
            result = self.calculator.hash_string(text, algorithm)
            results.append(result)
        return results
    
    def batch_hash_files(self, file_paths: List[str], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> List[HashResult]:
        """Hash multiple files efficiently."""
        results = []
        for file_path in file_paths:
            result = self.calculator.hash_file(file_path, algorithm)
            results.append(result)
        return results
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available hash templates."""
        return self.hash_templates.copy()
    
    def get_hashing_statistics(self) -> Dict[str, Any]:
        """Get hashing performance statistics."""
        stats = self.performance_stats.copy()
        stats['hash_history_count'] = len(self.calculator.get_hash_history())
        stats['verification_history_count'] = len(self.calculator.get_verification_history())
        return stats
    
    def analyze_hash_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in hash operations."""
        hash_history = self.calculator.get_hash_history()
        
        if not hash_history:
            return {}
        
        analysis = {
            'algorithm_usage': {},
            'hash_type_distribution': {},
            'average_processing_times': {},
            'error_patterns': {}
        }
        
        # Analyze algorithm usage
        for result in hash_history:
            algo = result.algorithm.value
            analysis['algorithm_usage'][algo] = analysis['algorithm_usage'].get(algo, 0) + 1
        
        # Analyze hash type distribution
        for result in hash_history:
            hash_type = result.hash_type.value
            analysis['hash_type_distribution'][hash_type] = analysis['hash_type_distribution'].get(hash_type, 0) + 1
        
        # Analyze processing times
        for result in hash_history:
            algo = result.algorithm.value
            if algo not in analysis['average_processing_times']:
                analysis['average_processing_times'][algo] = []
            analysis['average_processing_times'][algo].append(result.processing_time)
        
        # Calculate averages
        for algo, times in analysis['average_processing_times'].items():
            analysis['average_processing_times'][algo] = sum(times) / len(times)
        
        # Analyze error patterns
        error_count = sum(1 for result in hash_history if result.errors)
        analysis['error_patterns']['total_errors'] = error_count
        analysis['error_patterns']['error_rate'] = error_count / len(hash_history) if hash_history else 0
        
        return analysis

async def main():
    """Example usage and testing."""
    try:
        # Initialize manager
        manager = HashingManager()
        
        print("🔐 Available hash templates:")
        templates = manager.get_available_templates()
        for name, template in templates.items():
            print(f"  - {name}: {template['description']}")
        
        # Test string hashing
        test_strings = ["Hello, World!", "Cybersecurity is important", "Test data for hashing"]
        print(f"\n🚀 Hashing {len(test_strings)} test strings...")
        
        results = manager.batch_hash_strings(test_strings, HashAlgorithm.SHA256)
        for i, result in enumerate(results):
            print(f"  {i+1}. '{test_strings[i][:20]}...' -> {result.hash_value[:16]}...")
        
        # Test file hashing (if test file exists)
        test_file = "test_file.txt"
        if os.path.exists(test_file):
            print(f"\n📁 Hashing test file: {test_file}")
            file_result = manager.calculator.hash_file(test_file, HashAlgorithm.SHA256)
            print(f"  File hash: {file_result.hash_value[:16]}...")
            print(f"  File size: {file_result.file_size} bytes")
            print(f"  Processing time: {file_result.processing_time:.4f} seconds")
        
        # Test HMAC
        print(f"\n🔑 Creating HMAC...")
        hmac_result = manager.calculator.create_hmac("Secret message", "secret_key", HashAlgorithm.SHA256)
        print(f"  HMAC: {hmac_result.hash_value[:16]}...")
        
        # Get statistics
        stats = manager.get_hashing_statistics()
        print(f"\n📈 Hashing statistics:")
        print(f"  - Total hashes: {stats['hash_history_count']}")
        print(f"  - Verifications: {stats['verification_history_count']}")
        
        # Analyze patterns
        analysis = manager.analyze_hash_patterns()
        if analysis:
            print(f"\n🔍 Hash pattern analysis:")
            print(f"  - Most used algorithm: {max(analysis['algorithm_usage'].items(), key=lambda x: x[1])[0]}")
            print(f"  - Error rate: {analysis['error_patterns']['error_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
