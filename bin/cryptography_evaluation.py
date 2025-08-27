#!/usr/bin/env python3
"""
Cryptography Evaluation Tools - Comprehensive Cryptographic Analysis and Security Assessment
Provides tools for evaluating cryptographic implementations, analyzing security properties, and assessing cryptographic strength.

Features:
- Cryptographic strength analysis
- Implementation security assessment
- Key quality evaluation
- Cryptographic protocol analysis
- Security recommendations
- Integration with MCP framework for dynamic workflows
"""

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
import secrets
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hmac
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.exceptions import InvalidKey, InvalidSignature, UnsupportedAlgorithm
import re
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for cryptographic analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class VulnerabilityType(Enum):
    """Types of cryptographic vulnerabilities."""
    WEAK_ALGORITHM = "weak_algorithm"
    INSECURE_MODE = "insecure_mode"
    SHORT_KEY_LENGTH = "short_key_length"
    WEAK_RANDOMNESS = "weak_randomness"
    IMPROPER_PADDING = "improper_padding"
    TIMING_ATTACK = "timing_attack"
    SIDE_CHANNEL = "side_channel"
    PROTOCOL_FLAW = "protocol_flaw"

class EvaluationCategory(Enum):
    """Categories of cryptographic evaluation."""
    ALGORITHM_STRENGTH = "algorithm_strength"
    IMPLEMENTATION_SECURITY = "implementation_security"
    KEY_QUALITY = "key_quality"
    PROTOCOL_SECURITY = "protocol_security"
    RANDOMNESS_QUALITY = "randomness_quality"

@dataclass
class SecurityVulnerability:
    """Information about a security vulnerability."""
    vulnerability_id: str
    type: VulnerabilityType
    severity: SecurityLevel
    description: str
    affected_component: str
    recommendation: str
    cve_references: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cve_references is None:
            self.cve_references = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CryptographicEvaluation:
    """Result of a cryptographic evaluation."""
    evaluation_id: str
    target: str
    evaluation_type: EvaluationCategory
    security_score: float
    vulnerabilities: List[SecurityVulnerability]
    recommendations: List[str]
    metadata: Dict[str, Any]
    evaluation_time: float
    
    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}

class CryptographyEvaluator:
    """Core cryptography evaluation functionality."""
    
    def __init__(self):
        self.evaluation_history: List[CryptographicEvaluation] = []
        self.vulnerability_database = self._initialize_vulnerability_database()
        
        logger.info("🚀 CryptographyEvaluator initialized")
    
    def _initialize_vulnerability_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of known cryptographic vulnerabilities."""
        return {
            "md5": {
                "severity": SecurityLevel.CRITICAL,
                "description": "MD5 is cryptographically broken and should not be used for security purposes",
                "cve_references": ["CVE-2004-2761", "CVE-2008-1662"],
                "recommendation": "Replace with SHA-256 or SHA-3"
            },
            "sha1": {
                "severity": SecurityLevel.HIGH,
                "description": "SHA-1 is cryptographically broken and should not be used for security purposes",
                "cve_references": ["CVE-2005-4900", "CVE-2017-3732"],
                "recommendation": "Replace with SHA-256 or SHA-3"
            },
            "des": {
                "severity": SecurityLevel.CRITICAL,
                "description": "DES has insufficient key length and is vulnerable to brute force attacks",
                "cve_references": ["CVE-2008-1662"],
                "recommendation": "Replace with AES-256"
            },
            "rc4": {
                "severity": SecurityLevel.CRITICAL,
                "description": "RC4 has known biases and should not be used for security purposes",
                "cve_references": ["CVE-2013-2566", "CVE-2015-2808"],
                "recommendation": "Replace with ChaCha20 or AES-GCM"
            },
            "ecb_mode": {
                "severity": SecurityLevel.HIGH,
                "description": "ECB mode does not provide confidentiality and is vulnerable to pattern analysis",
                "cve_references": ["CVE-2011-3389"],
                "recommendation": "Use CBC, GCM, or CTR mode with proper IVs"
            },
            "weak_rsa": {
                "severity": SecurityLevel.HIGH,
                "description": "RSA keys shorter than 2048 bits are vulnerable to factorization attacks",
                "cve_references": ["CVE-2008-1662"],
                "recommendation": "Use RSA-2048 or longer, or prefer ECC"
            }
        }
    
    def _calculate_algorithm_strength(self, algorithm: str, key_length: int = None) -> Dict[str, Any]:
        """Calculate the cryptographic strength of an algorithm."""
        strength_info = {
            'algorithm': algorithm,
            'security_bits': 0,
            'security_level': SecurityLevel.LOW,
            'estimated_break_time': 'unknown',
            'recommendations': []
        }
        
        # Algorithm-specific strength calculations
        if algorithm.lower() in ['aes-128', 'aes128']:
            strength_info['security_bits'] = 128
            strength_info['security_level'] = SecurityLevel.HIGH
            strength_info['estimated_break_time'] = '> 100 years (quantum-resistant)'
            
        elif algorithm.lower() in ['aes-256', 'aes256']:
            strength_info['security_bits'] = 256
            strength_info['security_level'] = SecurityLevel.VERY_HIGH
            strength_info['estimated_break_time'] = '> 100 years (quantum-resistant)'
            
        elif algorithm.lower() in ['rsa-2048', 'rsa2048']:
            strength_info['security_bits'] = 112
            strength_info['security_level'] = SecurityLevel.HIGH
            strength_info['estimated_break_time'] = '> 20 years'
            strength_info['recommendations'].append("Consider upgrading to RSA-4096 or ECC for long-term security")
            
        elif algorithm.lower() in ['rsa-4096', 'rsa4096']:
            strength_info['security_bits'] = 128
            strength_info['security_level'] = SecurityLevel.VERY_HIGH
            strength_info['estimated_break_time'] = '> 50 years'
            
        elif algorithm.lower() in ['ecc-p256', 'secp256r1']:
            strength_info['security_bits'] = 128
            strength_info['security_level'] = SecurityLevel.VERY_HIGH
            strength_info['estimated_break_time'] = '> 100 years'
            
        elif algorithm.lower() in ['ecc-p384', 'secp384r1']:
            strength_info['security_bits'] = 192
            strength_info['security_level'] = SecurityLevel.CRITICAL
            strength_info['estimated_break_time'] = '> 100 years'
            
        elif algorithm.lower() in ['sha-256', 'sha256']:
            strength_info['security_bits'] = 128
            strength_info['security_level'] = SecurityLevel.VERY_HIGH
            strength_info['estimated_break_time'] = '> 100 years'
            
        elif algorithm.lower() in ['sha-512', 'sha512']:
            strength_info['security_bits'] = 256
            strength_info['security_level'] = SecurityLevel.CRITICAL
            strength_info['estimated_break_time'] = '> 100 years'
            
        elif algorithm.lower() in ['md5', 'md5']:
            strength_info['security_bits'] = 0
            strength_info['security_level'] = SecurityLevel.CRITICAL
            strength_info['estimated_break_time'] = 'Already broken'
            strength_info['recommendations'].append("MD5 is cryptographically broken - replace immediately")
            
        elif algorithm.lower() in ['sha1', 'sha1']:
            strength_info['security_bits'] = 0
            strength_info['security_level'] = SecurityLevel.HIGH
            strength_info['estimated_break_time'] = 'Already broken'
            strength_info['recommendations'].append("SHA-1 is cryptographically broken - replace immediately")
        
        # Adjust based on key length if provided
        if key_length:
            if key_length < 128:
                strength_info['security_level'] = SecurityLevel.LOW
                strength_info['recommendations'].append(f"Key length {key_length} bits is too short for security")
            elif key_length < 256:
                strength_info['security_level'] = SecurityLevel.MEDIUM
                strength_info['recommendations'].append(f"Consider increasing key length to 256+ bits")
        
        return strength_info
    
    def _evaluate_encryption_mode(self, mode: str) -> Dict[str, Any]:
        """Evaluate the security of an encryption mode."""
        mode_evaluation = {
            'mode': mode,
            'security_level': SecurityLevel.LOW,
            'vulnerabilities': [],
            'recommendations': []
        }
        
        mode_lower = mode.lower()
        
        if mode_lower == 'ecb':
            mode_evaluation['security_level'] = SecurityLevel.CRITICAL
            mode_evaluation['vulnerabilities'].append(
                SecurityVulnerability(
                    vulnerability_id="ecb_mode_vulnerability",
                    type=VulnerabilityType.INSECURE_MODE,
                    severity=SecurityLevel.CRITICAL,
                    description="ECB mode does not provide confidentiality and is vulnerable to pattern analysis",
                    affected_component="Encryption mode",
                    recommendation="Use CBC, GCM, or CTR mode with proper IVs",
                    cve_references=["CVE-2011-3389"]
                )
            )
            
        elif mode_lower == 'cbc':
            mode_evaluation['security_level'] = SecurityLevel.MEDIUM
            mode_evaluation['recommendations'].append("Ensure proper IV generation and padding oracle protection")
            mode_evaluation['recommendations'].append("Consider using GCM mode for authenticated encryption")
            
        elif mode_lower == 'gcm':
            mode_evaluation['security_level'] = SecurityLevel.VERY_HIGH
            mode_evaluation['recommendations'].append("Excellent choice - provides authenticated encryption")
            
        elif mode_lower == 'ctr':
            mode_evaluation['security_level'] = SecurityLevel.HIGH
            mode_evaluation['recommendations'].append("Good choice - ensure nonce uniqueness")
            mode_evaluation['recommendations'].append("Consider adding MAC for authentication")
            
        elif mode_lower in ['cfb', 'ofb']:
            mode_evaluation['security_level'] = SecurityLevel.MEDIUM
            mode_evaluation['recommendations'].append("Ensure proper IV generation")
            mode_evaluation['recommendations'].append("Consider using GCM mode for better security")
        
        return mode_evaluation
    
    def _evaluate_key_quality(self, key_data: bytes, algorithm: str) -> Dict[str, Any]:
        """Evaluate the quality of a cryptographic key."""
        key_evaluation = {
            'key_length': len(key_data) * 8,
            'entropy_estimate': 0.0,
            'randomness_quality': 'unknown',
            'security_level': SecurityLevel.LOW,
            'recommendations': []
        }
        
        # Calculate entropy estimate
        byte_counts = {}
        for byte in key_data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        if len(byte_counts) > 1:
            entropy = 0
            total_bytes = len(key_data)
            for count in byte_counts.values():
                probability = count / total_bytes
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            key_evaluation['entropy_estimate'] = entropy
        
        # Assess randomness quality
        if key_evaluation['entropy_estimate'] > 7.5:
            key_evaluation['randomness_quality'] = 'excellent'
            key_evaluation['security_level'] = SecurityLevel.VERY_HIGH
        elif key_evaluation['entropy_estimate'] > 6.5:
            key_evaluation['randomness_quality'] = 'good'
            key_evaluation['security_level'] = SecurityLevel.HIGH
        elif key_evaluation['entropy_estimate'] > 5.5:
            key_evaluation['randomness_quality'] = 'fair'
            key_evaluation['security_level'] = SecurityLevel.MEDIUM
        else:
            key_evaluation['randomness_quality'] = 'poor'
            key_evaluation['security_level'] = SecurityLevel.LOW
            key_evaluation['recommendations'].append("Key shows poor randomness - regenerate with cryptographically secure RNG")
        
        # Check key length
        if key_evaluation['key_length'] < 128:
            key_evaluation['security_level'] = SecurityLevel.LOW
            key_evaluation['recommendations'].append(f"Key length {key_evaluation['key_length']} bits is too short for security")
        elif key_evaluation['key_length'] < 256:
            key_evaluation['recommendations'].append(f"Consider increasing key length to 256+ bits for long-term security")
        
        return key_evaluation
    
    def _evaluate_randomness_quality(self, random_data: bytes) -> Dict[str, Any]:
        """Evaluate the quality of random data."""
        randomness_evaluation = {
            'data_length': len(random_data),
            'entropy_estimate': 0.0,
            'distribution_analysis': {},
            'security_level': SecurityLevel.LOW,
            'recommendations': []
        }
        
        # Calculate entropy
        byte_counts = {}
        for byte in random_data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        if len(byte_counts) > 1:
            entropy = 0
            total_bytes = len(random_data)
            for count in byte_counts.values():
                probability = count / total_bytes
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            randomness_evaluation['entropy_estimate'] = entropy
        
        # Analyze distribution
        expected_frequency = len(random_data) / 256
        chi_square = 0
        for count in byte_counts.values():
            chi_square += ((count - expected_frequency) ** 2) / expected_frequency
        
        randomness_evaluation['distribution_analysis']['chi_square'] = chi_square
        randomness_evaluation['distribution_analysis']['degrees_of_freedom'] = 255
        
        # Assess quality
        if randomness_evaluation['entropy_estimate'] > 7.8:
            randomness_evaluation['security_level'] = SecurityLevel.VERY_HIGH
            randomness_evaluation['recommendations'].append("Excellent randomness quality")
        elif randomness_evaluation['entropy_estimate'] > 7.5:
            randomness_evaluation['security_level'] = SecurityLevel.HIGH
            randomness_evaluation['recommendations'].append("Good randomness quality")
        elif randomness_evaluation['entropy_estimate'] > 7.0:
            randomness_evaluation['security_level'] = SecurityLevel.MEDIUM
            randomness_evaluation['recommendations'].append("Acceptable randomness quality")
        else:
            randomness_evaluation['security_level'] = SecurityLevel.LOW
            randomness_evaluation['recommendations'].append("Poor randomness quality - review RNG implementation")
        
        return randomness_evaluation
    
    def evaluate_algorithm_security(self, algorithm: str, key_length: int = None, 
                                  mode: str = None) -> CryptographicEvaluation:
        """Evaluate the security of a cryptographic algorithm."""
        start_time = time.time()
        evaluation_id = f"algo_eval_{int(start_time)}_{len(self.evaluation_history)}"
        
        vulnerabilities = []
        recommendations = []
        
        # Check for known vulnerabilities
        if algorithm.lower() in self.vulnerability_database:
            vuln_info = self.vulnerability_database[algorithm.lower()]
            vulnerabilities.append(
                SecurityVulnerability(
                    vulnerability_id=f"{algorithm}_known_vuln",
                    type=VulnerabilityType.WEAK_ALGORITHM, # Assuming WEAK_ALGORITHM is the correct type for known vulnerabilities
                    severity=vuln_info['severity'],
                    description=vuln_info['description'],
                    affected_component=f"Algorithm: {algorithm}",
                    recommendation=vuln_info['recommendation'],
                    cve_references=vuln_info['cve_references']
                )
            )
        
        # Evaluate algorithm strength
        strength_info = self._calculate_algorithm_strength(algorithm, key_length)
        recommendations.extend(strength_info['recommendations'])
        
        # Evaluate encryption mode if provided
        mode_evaluation = None
        if mode:
            mode_evaluation = self._evaluate_encryption_mode(mode)
            vulnerabilities.extend(mode_evaluation['vulnerabilities'])
            recommendations.extend(mode_evaluation['recommendations'])
        
        # Calculate security score
        base_score = 100.0
        
        # Deduct points for vulnerabilities
        for vuln in vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                base_score -= 50
            elif vuln.severity == SecurityLevel.HIGH:
                base_score -= 30
            elif vuln.severity == SecurityLevel.MEDIUM:
                base_score -= 15
            elif vuln.severity == SecurityLevel.LOW:
                base_score -= 5
        
        # Deduct points for weak algorithms
        if strength_info['security_level'] == SecurityLevel.LOW:
            base_score -= 40
        elif strength_info['security_level'] == SecurityLevel.MEDIUM:
            base_score -= 20
        
        # Deduct points for weak modes
        if mode_evaluation and mode_evaluation['security_level'] == SecurityLevel.LOW:
            base_score -= 30
        elif mode_evaluation and mode_evaluation['security_level'] == SecurityLevel.MEDIUM:
            base_score -= 15
        
        security_score = max(0.0, base_score)
        
        # Create evaluation result
        evaluation = CryptographicEvaluation(
            evaluation_id=evaluation_id,
            target=f"Algorithm: {algorithm}",
            evaluation_type=EvaluationCategory.ALGORITHM_STRENGTH,
            security_score=security_score,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            metadata={
                'algorithm': algorithm,
                'key_length': key_length,
                'mode': mode,
                'strength_info': strength_info,
                'mode_evaluation': mode_evaluation
            },
            evaluation_time=time.time() - start_time
        )
        
        self.evaluation_history.append(evaluation)
        logger.info(f"Algorithm security evaluation completed for {algorithm}")
        return evaluation
    
    def evaluate_implementation_security(self, implementation_data: Dict[str, Any]) -> CryptographicEvaluation:
        """Evaluate the security of a cryptographic implementation."""
        start_time = time.time()
        evaluation_id = f"impl_eval_{int(start_time)}_{len(self.evaluation_history)}"
        
        vulnerabilities = []
        recommendations = []
        
        # Check for common implementation issues
        if 'padding' in implementation_data:
            padding_type = implementation_data['padding']
            if padding_type.lower() == 'pkcs1_v1_5':
                vulnerabilities.append(
                    SecurityVulnerability(
                        vulnerability_id="pkcs1_v1_5_padding",
                        type=VulnerabilityType.IMPROPER_PADDING,
                        severity=SecurityLevel.HIGH,
                        description="PKCS1 v1.5 padding is vulnerable to padding oracle attacks",
                        affected_component="Padding implementation",
                        recommendation="Use OAEP padding for RSA operations",
                        cve_references=["CVE-2017-1000430"]
                    )
                )
        
        if 'iv_generation' in implementation_data:
            iv_gen = implementation_data['iv_generation']
            if iv_gen.lower() == 'static':
                vulnerabilities.append(
                    SecurityVulnerability(
                        vulnerability_id="static_iv",
                        type=VulnerabilityType.WEAK_RANDOMNESS,
                        severity=SecurityLevel.HIGH,
                        description="Static IVs make encryption deterministic and vulnerable to analysis",
                        affected_component="IV generation",
                        recommendation="Generate cryptographically secure random IVs for each encryption"
                    )
                )
        
        if 'key_derivation' in implementation_data:
            kdf_info = implementation_data['key_derivation']
            if 'iterations' in kdf_info:
                iterations = kdf_info['iterations']
                if iterations < 100000:
                    vulnerabilities.append(
                        SecurityVulnerability(
                            vulnerability_id="insufficient_kdf_iterations",
                            type=VulnerabilityType.WEAK_RANDOMNESS,
                            severity=SecurityLevel.MEDIUM,
                            description=f"KDF iterations ({iterations}) are too low for security",
                            affected_component="Key derivation function",
                            recommendation="Use at least 100,000 iterations for PBKDF2"
                        )
                    )
        
        # Calculate security score
        base_score = 100.0
        for vuln in vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                base_score -= 50
            elif vuln.severity == SecurityLevel.HIGH:
                base_score -= 30
            elif vuln.severity == SecurityLevel.MEDIUM:
                base_score -= 15
            elif vuln.severity == SecurityLevel.LOW:
                base_score -= 5
        
        security_score = max(0.0, base_score)
        
        # Create evaluation result
        evaluation = CryptographicEvaluation(
            evaluation_id=evaluation_id,
            target="Cryptographic implementation",
            evaluation_type=EvaluationCategory.IMPLEMENTATION_SECURITY,
            security_score=security_score,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            metadata={
                'implementation_data': implementation_data,
                'vulnerability_count': len(vulnerabilities)
            },
            evaluation_time=time.time() - start_time
        )
        
        self.evaluation_history.append(evaluation)
        logger.info("Implementation security evaluation completed")
        return evaluation
    
    def evaluate_key_quality(self, key_data: bytes, algorithm: str) -> CryptographicEvaluation:
        """Evaluate the quality of a cryptographic key."""
        start_time = time.time()
        evaluation_id = f"key_eval_{int(start_time)}_{len(self.evaluation_history)}"
        
        # Perform key quality analysis
        key_analysis = self._evaluate_key_quality(key_data, algorithm)
        
        vulnerabilities = []
        recommendations = key_analysis['recommendations']
        
        # Create vulnerabilities for poor key quality
        if key_analysis['security_level'] == SecurityLevel.LOW:
            vulnerabilities.append(
                SecurityVulnerability(
                    vulnerability_id="poor_key_quality",
                    type=VulnerabilityType.SHORT_KEY_LENGTH,
                    severity=SecurityLevel.LOW,
                    description=f"Key shows poor quality with {key_analysis['key_length']} bits and {key_analysis['entropy_estimate']:.2f} entropy",
                    affected_component="Cryptographic key",
                    recommendation="Regenerate key with cryptographically secure RNG and adequate length"
                )
            )
        
        # Calculate security score
        base_score = 100.0
        for vuln in vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                base_score -= 50
            elif vuln.severity == SecurityLevel.HIGH:
                base_score -= 30
            elif vuln.severity == SecurityLevel.MEDIUM:
                base_score -= 15
            elif vuln.severity == SecurityLevel.LOW:
                base_score -= 5
        
        # Adjust score based on key analysis
        if key_analysis['security_level'] == SecurityLevel.LOW:
            base_score -= 40
        elif key_analysis['security_level'] == SecurityLevel.MEDIUM:
            base_score -= 20
        
        security_score = max(0.0, base_score)
        
        # Create evaluation result
        evaluation = CryptographicEvaluation(
            evaluation_id=evaluation_id,
            target=f"Key for {algorithm}",
            evaluation_type=EvaluationCategory.KEY_QUALITY,
            security_score=security_score,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            metadata={
                'algorithm': algorithm,
                'key_analysis': key_analysis
            },
            evaluation_time=time.time() - start_time
        )
        
        self.evaluation_history.append(evaluation)
        logger.info(f"Key quality evaluation completed for {algorithm}")
        return evaluation
    
    def evaluate_randomness_quality(self, random_data: bytes) -> CryptographicEvaluation:
        """Evaluate the quality of random data."""
        start_time = time.time()
        evaluation_id = f"rand_eval_{int(start_time)}_{len(self.evaluation_history)}"
        
        # Perform randomness analysis
        randomness_analysis = self._evaluate_randomness_quality(random_data)
        
        vulnerabilities = []
        recommendations = randomness_analysis['recommendations']
        
        # Create vulnerabilities for poor randomness
        if randomness_analysis['security_level'] == SecurityLevel.LOW:
            vulnerabilities.append(
                SecurityVulnerability(
                    vulnerability_id="poor_randomness",
                    type=VulnerabilityType.WEAK_RANDOMNESS,
                    severity=SecurityLevel.LOW,
                    description=f"Random data shows poor quality with {randomness_analysis['entropy_estimate']:.2f} entropy",
                    affected_component="Random number generation",
                    recommendation="Review and improve RNG implementation"
                )
            )
        
        # Calculate security score
        base_score = 100.0
        for vuln in vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                base_score -= 50
            elif vuln.severity == SecurityLevel.HIGH:
                base_score -= 30
            elif vuln.severity == SecurityLevel.MEDIUM:
                base_score -= 15
            elif vuln.severity == SecurityLevel.LOW:
                base_score -= 5
        
        # Adjust score based on randomness analysis
        if randomness_analysis['security_level'] == SecurityLevel.LOW:
            base_score -= 40
        elif randomness_analysis['security_level'] == SecurityLevel.MEDIUM:
            base_score -= 20
        
        security_score = max(0.0, base_score)
        
        # Create evaluation result
        evaluation = CryptographicEvaluation(
            evaluation_id=evaluation_id,
            target="Random data quality",
            evaluation_type=EvaluationCategory.RANDOMNESS_QUALITY,
            security_score=security_score,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            metadata={
                'randomness_analysis': randomness_analysis
            },
            evaluation_time=time.time() - start_time
        )
        
        self.evaluation_history.append(evaluation)
        logger.info("Randomness quality evaluation completed")
        return evaluation
    
    def get_evaluation_history(self) -> List[CryptographicEvaluation]:
        """Get list of all evaluations."""
        return self.evaluation_history.copy()
    
    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history.clear()
        logger.info("Cryptography evaluation history cleared")

class CryptographyEvaluationManager:
    """Manager for cryptography evaluation operations with MCP integration capabilities."""
    
    def __init__(self):
        self.evaluator = CryptographyEvaluator()
        self.evaluation_templates = self._create_evaluation_templates()
        self.performance_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_evaluation_time': 0.0
        }
        
        logger.info("🚀 Cryptography Evaluation Manager initialized")
    
    def _create_evaluation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined evaluation templates."""
        return {
            'comprehensive_security': {
                'description': 'Comprehensive security evaluation of cryptographic implementation',
                'evaluations': ['algorithm', 'implementation', 'key_quality', 'randomness']
            },
            'algorithm_focus': {
                'description': 'Focus on algorithm strength and security',
                'evaluations': ['algorithm']
            },
            'implementation_focus': {
                'description': 'Focus on implementation security and best practices',
                'evaluations': ['implementation']
            },
            'key_quality_focus': {
                'description': 'Focus on key quality and randomness',
                'evaluations': ['key_quality', 'randomness']
            },
            'quick_assessment': {
                'description': 'Quick security assessment for common issues',
                'evaluations': ['algorithm', 'implementation']
            }
        }
    
    def execute_evaluation_template(self, template_name: str, **kwargs) -> List[CryptographicEvaluation]:
        """Execute a predefined evaluation template."""
        if template_name not in self.evaluation_templates:
            raise ValueError(f"Unknown evaluation template: {template_name}")
        
        template = self.evaluation_templates[template_name]
        evaluations = []
        
        for eval_type in template['evaluations']:
            try:
                if eval_type == 'algorithm':
                    eval_result = self.evaluator.evaluate_algorithm_security(
                        kwargs.get('algorithm', 'unknown'),
                        kwargs.get('key_length'),
                        kwargs.get('mode')
                    )
                    evaluations.append(eval_result)
                    
                elif eval_type == 'implementation':
                    eval_result = self.evaluator.evaluate_implementation_security(
                        kwargs.get('implementation_data', {})
                    )
                    evaluations.append(eval_result)
                    
                elif eval_type == 'key_quality':
                    eval_result = self.evaluator.evaluate_key_quality(
                        kwargs.get('key_data', b''),
                        kwargs.get('algorithm', 'unknown')
                    )
                    evaluations.append(eval_result)
                    
                elif eval_type == 'randomness':
                    eval_result = self.evaluator.evaluate_randomness_quality(
                        kwargs.get('random_data', b'')
                    )
                    evaluations.append(eval_result)
                    
            except Exception as e:
                logger.error(f"Evaluation {eval_type} failed: {e}")
        
        return evaluations
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available evaluation templates."""
        return self.evaluation_templates.copy()
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation performance statistics."""
        stats = self.performance_stats.copy()
        stats['evaluation_history_count'] = len(self.evaluator.get_evaluation_history())
        return stats
    
    def analyze_evaluation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in evaluations."""
        evaluation_history = self.evaluator.get_evaluation_history()
        
        if not evaluation_history:
            return {}
        
        analysis = {
            'evaluation_type_distribution': {},
            'security_score_distribution': {},
            'vulnerability_patterns': {},
            'average_scores_by_type': {}
        }
        
        # Analyze evaluation type distribution
        for evaluation in evaluation_history:
            eval_type = evaluation.evaluation_type.value
            analysis['evaluation_type_distribution'][eval_type] = analysis['evaluation_type_distribution'].get(eval_type, 0) + 1
        
        # Analyze security score distribution
        for evaluation in evaluation_history:
            score_range = f"{(evaluation.security_score // 20) * 20}-{(evaluation.security_score // 20) * 20 + 19}"
            analysis['security_score_distribution'][score_range] = analysis['security_score_distribution'].get(score_range, 0) + 1
        
        # Analyze vulnerability patterns
        for evaluation in evaluation_history:
            for vuln in evaluation.vulnerabilities:
                vuln_type = vuln.type.value
                analysis['vulnerability_patterns'][vuln_type] = analysis['vulnerability_patterns'].get(vuln_type, 0) + 1
        
        # Calculate average scores by type
        for evaluation in evaluation_history:
            eval_type = evaluation.evaluation_type.value
            if eval_type not in analysis['average_scores_by_type']:
                analysis['average_scores_by_type'][eval_type] = []
            analysis['average_scores_by_type'][eval_type].append(evaluation.security_score)
        
        # Calculate averages
        for eval_type, scores in analysis['average_scores_by_type'].items():
            analysis['average_scores_by_type'][eval_type] = sum(scores) / len(scores)
        
        return analysis

async def main():
    """Example usage and testing."""
    try:
        # Initialize manager
        manager = CryptographyEvaluationManager()
        
        print("🔍 Available evaluation templates:")
        templates = manager.get_available_templates()
        for name, template in templates.items():
            print(f"  - {name}: {template['description']}")
        
        # Test algorithm security evaluation
        print(f"\n🚀 Testing algorithm security evaluation...")
        algo_eval = manager.evaluator.evaluate_algorithm_security(
            "AES-256", 
            key_length=256, 
            mode="GCM"
        )
        
        print(f"  ✅ Algorithm evaluation completed:")
        print(f"    - Security score: {algo_eval.security_score:.1f}/100")
        print(f"    - Vulnerabilities found: {len(algo_eval.vulnerabilities)}")
        print(f"    - Recommendations: {len(algo_eval.recommendations)}")
        
        # Test weak algorithm evaluation
        print(f"\n⚠️  Testing weak algorithm evaluation...")
        weak_algo_eval = manager.evaluator.evaluate_algorithm_security(
            "MD5", 
            key_length=128
        )
        
        print(f"  ⚠️  Weak algorithm evaluation completed:")
        print(f"    - Security score: {weak_algo_eval.security_score:.1f}/100")
        print(f"    - Vulnerabilities found: {len(weak_algo_eval.vulnerabilities)}")
        
        # Test key quality evaluation
        print(f"\n🔑 Testing key quality evaluation...")
        test_key = secrets.token_bytes(32)  # 256-bit key
        key_eval = manager.evaluator.evaluate_key_quality(test_key, "AES-256")
        
        print(f"  ✅ Key quality evaluation completed:")
        print(f"    - Security score: {key_eval.security_score:.1f}/100")
        print(f"    - Key length: {key_eval.metadata['key_analysis']['key_length']} bits")
        print(f"    - Entropy estimate: {key_eval.metadata['key_analysis']['entropy_estimate']:.2f}")
        
        # Test randomness quality evaluation
        print(f"\n🎲 Testing randomness quality evaluation...")
        test_random = secrets.token_bytes(64)  # 512 bits of random data
        rand_eval = manager.evaluator.evaluate_randomness_quality(test_random)
        
        print(f"  ✅ Randomness evaluation completed:")
        print(f"    - Security score: {rand_eval.security_score:.1f}/100")
        print(f"    - Entropy estimate: {rand_eval.metadata['randomness_analysis']['entropy_estimate']:.2f}")
        
        # Test comprehensive evaluation template
        print(f"\n🔍 Testing comprehensive evaluation template...")
        comprehensive_evals = manager.execute_evaluation_template(
            'comprehensive_security',
            algorithm='RSA-2048',
            key_length=2048,
            mode='OAEP',
            key_data=test_key,
            random_data=test_random,
            implementation_data={
                'padding': 'OAEP',
                'iv_generation': 'random',
                'key_derivation': {'iterations': 100000}
            }
        )
        
        print(f"  ✅ Comprehensive evaluation completed:")
        print(f"    - Total evaluations: {len(comprehensive_evals)}")
        for eval_result in comprehensive_evals:
            print(f"    - {eval_result.evaluation_type.value}: {eval_result.security_score:.1f}/100")
        
        # Get statistics
        stats = manager.get_evaluation_statistics()
        print(f"\n📈 Evaluation statistics:")
        print(f"  - Total evaluations: {stats['evaluation_history_count']}")
        
        # Analyze patterns
        analysis = manager.analyze_evaluation_patterns()
        if analysis:
            print(f"\n🔍 Evaluation pattern analysis:")
            print(f"  - Most common evaluation type: {max(analysis['evaluation_type_distribution'].items(), key=lambda x: x[1])[0]}")
            print(f"  - Average security score: {sum(analysis['average_scores_by_type'].values()) / len(analysis['average_scores_by_type']):.1f}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
