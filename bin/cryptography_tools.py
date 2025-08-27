#!/usr/bin/env python3
"""
Cryptography Tools - Comprehensive Encryption, Decryption, and Cryptographic Analysis
Provides cryptographic capabilities for cybersecurity analysis, secure communication, and data protection.

Features:
- Symmetric encryption (AES, ChaCha20, etc.)
- Asymmetric encryption (RSA, ECC)
- Hash functions and digital signatures
- Key generation and management
- Cryptographic evaluation and analysis
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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hmac
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.exceptions import InvalidKey, InvalidSignature, UnsupportedAlgorithm
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Available encryption algorithms."""
    AES_128 = "aes_128"
    AES_256 = "aes_256"
    ChaCha20 = "chacha20"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECC_P256 = "ecc_p256"
    ECC_P384 = "ecc_p384"
    ECC_P521 = "ecc_p521"

class EncryptionMode(Enum):
    """Available encryption modes."""
    ECB = "ecb"
    CBC = "cbc"
    GCM = "gcm"
    CTR = "ctr"
    CFB = "cfb"
    OFB = "ofb"

class KeyDerivationFunction(Enum):
    """Available key derivation functions."""
    PBKDF2 = "pbkdf2"
    HKDF = "hkdf"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"

class CryptographyOperation(Enum):
    """Types of cryptographic operations."""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    KEY_GENERATION = "key_generation"
    KEY_EXCHANGE = "key_exchange"

@dataclass
class CryptographyResult:
    """Result of a cryptographic operation."""
    operation_id: str
    operation_type: CryptographyOperation
    algorithm: EncryptionAlgorithm
    success: bool
    result_data: Optional[bytes] = None
    key_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

@dataclass
class KeyPair:
    """Public/private key pair."""
    public_key: bytes
    private_key: bytes
    algorithm: EncryptionAlgorithm
    key_size: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CryptographyEngine:
    """Core cryptography functionality."""
    
    def __init__(self):
        self.operation_history: List[CryptographyResult] = []
        self.key_cache: Dict[str, bytes] = {}
        
        logger.info("🚀 CryptographyEngine initialized")
    
    def _generate_random_bytes(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)
    
    def _generate_symmetric_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate symmetric encryption key."""
        key_sizes = {
            EncryptionAlgorithm.AES_128: 16,
            EncryptionAlgorithm.AES_256: 32,
            EncryptionAlgorithm.ChaCha20: 32
        }
        key_size = key_sizes.get(algorithm, 32)
        return self._generate_random_bytes(key_size)
    
    def _generate_iv(self, algorithm: EncryptionAlgorithm, mode: EncryptionMode) -> bytes:
        """Generate initialization vector."""
        if mode == EncryptionMode.GCM:
            return self._generate_random_bytes(12)
        elif mode in [EncryptionMode.CBC, EncryptionMode.CFB, EncryptionMode.OFB]:
            return self._generate_random_bytes(16)
        elif mode == EncryptionMode.CTR:
            return self._generate_random_bytes(16)
        else:
            return b''
    
    def _derive_key_from_password(self, password: str, salt: bytes, 
                                 kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2,
                                 key_length: int = 32) -> bytes:
        """Derive encryption key from password."""
        try:
            if kdf == KeyDerivationFunction.PBKDF2:
                kdf_obj = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    iterations=100000
                )
                return kdf_obj.derive(password.encode('utf-8'))
            
            elif kdf == KeyDerivationFunction.HKDF:
                kdf_obj = HKDF(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    info=b'key_derivation'
                )
                return kdf_obj.derive(password.encode('utf-8'))
            
            elif kdf == KeyDerivationFunction.SCRYPT:
                kdf_obj = Scrypt(
                    salt=salt,
                    length=key_length,
                    n=2**14,
                    r=8,
                    p=1
                )
                return kdf_obj.derive(password.encode('utf-8'))
            
            else:
                raise ValueError(f"Unsupported KDF: {kdf}")
                
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise
    
    def encrypt_symmetric(self, data: Union[str, bytes], 
                         algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256,
                         mode: EncryptionMode = EncryptionMode.GCM,
                         key: Optional[bytes] = None,
                         password: Optional[str] = None,
                         salt: Optional[bytes] = None) -> CryptographyResult:
        """Encrypt data using symmetric encryption."""
        start_time = time.time()
        operation_id = f"encrypt_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Generate or derive key
            if key is None:
                if password and salt:
                    key = self._derive_key_from_password(password, salt)
                else:
                    key = self._generate_symmetric_key(algorithm)
            
            # Generate IV
            iv = self._generate_iv(algorithm, mode)
            
            # Encrypt data
            if algorithm in [EncryptionAlgorithm.AES_128, EncryptionAlgorithm.AES_256]:
                if mode == EncryptionMode.GCM:
                    cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                    tag = encryptor.tag
                    
                    # Combine IV, ciphertext, and tag
                    result_data = iv + tag + ciphertext
                    
                elif mode == EncryptionMode.CBC:
                    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
                    encryptor = cipher.encryptor()
                    
                    # Pad data to block size
                    block_size = 16
                    padding_length = block_size - (len(data_bytes) % block_size)
                    padded_data = data_bytes + bytes([padding_length] * padding_length)
                    
                    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                    result_data = iv + ciphertext
                    
                elif mode == EncryptionMode.CTR:
                    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                    result_data = iv + ciphertext
                    
                else:
                    raise ValueError(f"Unsupported mode {mode} for AES")
                    
            elif algorithm == EncryptionAlgorithm.ChaCha20:
                cipher = Cipher(algorithms.ChaCha20(key, iv), modes.CTR(iv))
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                result_data = iv + ciphertext
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.ENCRYPT,
                algorithm=algorithm,
                success=True,
                result_data=result_data,
                metadata={
                    'mode': mode.value,
                    'key_length': len(key),
                    'iv_length': len(iv),
                    'data_length': len(data_bytes),
                    'encrypted_length': len(result_data)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Data encrypted with {algorithm.value} in {mode.value} mode")
            return result
            
        except Exception as e:
            error_msg = f"Encryption failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.ENCRYPT,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def decrypt_symmetric(self, encrypted_data: bytes,
                         algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256,
                         mode: EncryptionMode = EncryptionMode.GCM,
                         key: bytes = None,
                         password: str = None,
                         salt: bytes = None) -> CryptographyResult:
        """Decrypt data using symmetric decryption."""
        start_time = time.time()
        operation_id = f"decrypt_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Derive key if password provided
            if key is None and password and salt:
                key = self._derive_key_from_password(password, salt)
            
            if key is None:
                raise ValueError("Key or password must be provided")
            
            # Extract IV and ciphertext based on mode
            if mode == EncryptionMode.GCM:
                iv = encrypted_data[:12]
                tag = encrypted_data[12:28]
                ciphertext = encrypted_data[28:]
                
                cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
                decryptor = cipher.decryptor()
                decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
                
            elif mode == EncryptionMode.CBC:
                iv = encrypted_data[:16]
                ciphertext = encrypted_data[16:]
                
                cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
                decryptor = cipher.decryptor()
                decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
                
                # Remove padding
                padding_length = decrypted_data[-1]
                decrypted_data = decrypted_data[:-padding_length]
                
            elif mode == EncryptionMode.CTR:
                iv = encrypted_data[:16]
                ciphertext = encrypted_data[16:]
                
                cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
                decryptor = cipher.decryptor()
                decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
                
            else:
                raise ValueError(f"Unsupported mode {mode} for decryption")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.DECRYPT,
                algorithm=algorithm,
                success=True,
                result_data=decrypted_data,
                metadata={
                    'mode': mode.value,
                    'key_length': len(key),
                    'iv_length': len(iv),
                    'encrypted_length': len(encrypted_data),
                    'decrypted_length': len(decrypted_data)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Data decrypted with {algorithm.value} in {mode.value} mode")
            return result
            
        except Exception as e:
            error_msg = f"Decryption failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.DECRYPT,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def generate_key_pair(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> CryptographyResult:
        """Generate public/private key pair."""
        start_time = time.time()
        operation_id = f"keygen_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            if algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size
                )
                public_key = private_key.public_key()
                
                # Serialize keys
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                key_pair = KeyPair(
                    public_key=public_pem,
                    private_key=private_pem,
                    algorithm=algorithm,
                    key_size=key_size
                )
                
            elif algorithm in [EncryptionAlgorithm.ECC_P256, EncryptionAlgorithm.ECC_P384, EncryptionAlgorithm.ECC_P521]:
                if algorithm == EncryptionAlgorithm.ECC_P256:
                    curve = ec.SECP256R1()
                elif algorithm == EncryptionAlgorithm.ECC_P384:
                    curve = ec.SECP384R1()
                else:
                    curve = ec.SECP521R1()
                
                private_key = ec.generate_private_key(curve)
                public_key = private_key.public_key()
                
                # Serialize keys
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                key_pair = KeyPair(
                    public_key=public_pem,
                    private_key=private_pem,
                    algorithm=algorithm,
                    key_size=curve.key_size
                )
                
            else:
                raise ValueError(f"Unsupported algorithm for key generation: {algorithm}")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.KEY_GENERATION,
                algorithm=algorithm,
                success=True,
                key_info={
                    'public_key': base64.b64encode(key_pair.public_key).decode('utf-8'),
                    'private_key': base64.b64encode(key_pair.private_key).decode('utf-8'),
                    'key_size': key_pair.key_size
                },
                metadata={
                    'algorithm': algorithm.value,
                    'key_size': key_pair.key_size
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Key pair generated for {algorithm.value}")
            return result
            
        except Exception as e:
            error_msg = f"Key generation failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.KEY_GENERATION,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def encrypt_asymmetric(self, data: Union[str, bytes], public_key_pem: bytes,
                          algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> CryptographyResult:
        """Encrypt data using asymmetric encryption."""
        start_time = time.time()
        operation_id = f"asym_encrypt_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Load public key
            public_key = load_pem_public_key(public_key_pem)
            
            if isinstance(public_key, rsa.RSAPublicKey):
                # RSA encryption
                encrypted_data = public_key.encrypt(
                    data_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            elif isinstance(public_key, ec.EllipticCurvePublicKey):
                # ECC encryption (using ECDH + symmetric encryption)
                # For simplicity, we'll use a hybrid approach
                ephemeral_private_key = ec.generate_private_key(public_key.curve)
                shared_key = ephemeral_private_key.exchange(ec.ECDH(), public_key)
                
                # Use shared key for symmetric encryption
                cipher = Cipher(algorithms.AES(shared_key[:32]), modes.GCM())
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                
                # Combine ephemeral public key and ciphertext
                ephemeral_public_key = ephemeral_private_key.public_key()
                ephemeral_pem = ephemeral_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                encrypted_data = ephemeral_pem + b'\n' + encryptor.nonce + encryptor.tag + ciphertext
            else:
                raise ValueError("Unsupported public key type")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.ENCRYPT,
                algorithm=algorithm,
                success=True,
                result_data=encrypted_data,
                metadata={
                    'encryption_type': 'asymmetric',
                    'data_length': len(data_bytes),
                    'encrypted_length': len(encrypted_data)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Data encrypted with {algorithm.value} asymmetric encryption")
            return result
            
        except Exception as e:
            error_msg = f"Asymmetric encryption failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.ENCRYPT,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def decrypt_asymmetric(self, encrypted_data: bytes, private_key_pem: bytes,
                          algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> CryptographyResult:
        """Decrypt data using asymmetric decryption."""
        start_time = time.time()
        operation_id = f"asym_decrypt_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Load private key
            private_key = load_pem_private_key(private_key_pem, password=None)
            
            if isinstance(private_key, rsa.RSAPrivateKey):
                # RSA decryption
                decrypted_data = private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            elif isinstance(private_key, ec.EllipticCurvePrivateKey):
                # ECC decryption (hybrid approach)
                # Extract ephemeral public key and ciphertext
                parts = encrypted_data.split(b'\n', 1)
                if len(parts) != 2:
                    raise ValueError("Invalid encrypted data format")
                
                ephemeral_pem, ciphertext_data = parts
                ephemeral_public_key = load_pem_public_key(ephemeral_pem)
                
                # Extract nonce, tag, and ciphertext
                nonce = ciphertext_data[:12]
                tag = ciphertext_data[12:28]
                ciphertext = ciphertext_data[28:]
                
                # Derive shared key
                shared_key = private_key.exchange(ec.ECDH(), ephemeral_public_key)
                
                # Decrypt with symmetric key
                cipher = Cipher(algorithms.AES(shared_key[:32]), modes.GCM(nonce, tag))
                decryptor = cipher.decryptor()
                decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            else:
                raise ValueError("Unsupported private key type")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.DECRYPT,
                algorithm=algorithm,
                success=True,
                result_data=decrypted_data,
                metadata={
                    'decryption_type': 'asymmetric',
                    'encrypted_length': len(encrypted_data),
                    'decrypted_length': len(decrypted_data)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Data decrypted with {algorithm.value} asymmetric decryption")
            return result
            
        except Exception as e:
            error_msg = f"Asymmetric decryption failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.DECRYPT,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def sign_data(self, data: Union[str, bytes], private_key_pem: bytes,
                  algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> CryptographyResult:
        """Sign data using private key."""
        start_time = time.time()
        operation_id = f"sign_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Load private key
            private_key = load_pem_private_key(private_key_pem, password=None)
            
            if isinstance(private_key, rsa.RSAPrivateKey):
                # RSA signing
                signature = private_key.sign(
                    data_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            elif isinstance(private_key, ec.EllipticCurvePrivateKey):
                # ECC signing
                signature = private_key.sign(
                    data_bytes,
                    ec.ECDSA(hashes.SHA256())
                )
            else:
                raise ValueError("Unsupported private key type")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.SIGN,
                algorithm=algorithm,
                success=True,
                result_data=signature,
                metadata={
                    'data_length': len(data_bytes),
                    'signature_length': len(signature)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Data signed with {algorithm.value}")
            return result
            
        except Exception as e:
            error_msg = f"Signing failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.SIGN,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes, public_key_pem: bytes,
                        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> CryptographyResult:
        """Verify data signature using public key."""
        start_time = time.time()
        operation_id = f"verify_{int(start_time)}_{len(self.operation_history)}"
        
        try:
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Load public key
            public_key = load_pem_public_key(public_key_pem)
            
            if isinstance(public_key, rsa.RSAPublicKey):
                # RSA verification
                public_key.verify(
                    signature,
                    data_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            elif isinstance(public_key, ec.EllipticCurvePublicKey):
                # ECC verification
                public_key.verify(
                    signature,
                    data_bytes,
                    ec.ECDSA(hashes.SHA256())
                )
            else:
                raise ValueError("Unsupported public key type")
            
            # Create result
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.VERIFY,
                algorithm=algorithm,
                success=True,
                metadata={
                    'verification': 'success',
                    'data_length': len(data_bytes),
                    'signature_length': len(signature)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.info(f"Signature verified with {algorithm.value}")
            return result
            
        except InvalidSignature:
            # Create result for invalid signature
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.VERIFY,
                algorithm=algorithm,
                success=False,
                errors=["Invalid signature"],
                metadata={
                    'verification': 'failed',
                    'data_length': len(data_bytes),
                    'signature_length': len(signature)
                },
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            logger.warning(f"Signature verification failed for {algorithm.value}")
            return result
            
        except Exception as e:
            error_msg = f"Signature verification failed: {str(e)}"
            logger.error(error_msg)
            
            result = CryptographyResult(
                operation_id=operation_id,
                operation_type=CryptographyOperation.VERIFY,
                algorithm=algorithm,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time
            )
            
            self.operation_history.append(result)
            return result
    
    def get_operation_history(self) -> List[CryptographyResult]:
        """Get list of all cryptographic operations."""
        return self.operation_history.copy()
    
    def clear_history(self):
        """Clear operation history."""
        self.operation_history.clear()
        logger.info("Cryptography operation history cleared")

class CryptographyManager:
    """Manager for cryptography operations with MCP integration capabilities."""
    
    def __init__(self):
        self.engine = CryptographyEngine()
        self.cryptography_templates = self._create_cryptography_templates()
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("🚀 Cryptography Manager initialized")
    
    def _create_cryptography_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined cryptography templates."""
        return {
            'secure_encryption': {
                'algorithm': EncryptionAlgorithm.AES_256,
                'mode': EncryptionMode.GCM,
                'description': 'High-security encryption with authenticated encryption'
            },
            'fast_encryption': {
                'algorithm': EncryptionAlgorithm.AES_128,
                'mode': EncryptionMode.CTR,
                'description': 'Fast encryption for performance-critical applications'
            },
            'compatible_encryption': {
                'algorithm': EncryptionAlgorithm.AES_256,
                'mode': EncryptionMode.CBC,
                'description': 'Compatible encryption for legacy systems'
            },
            'strong_key_pair': {
                'algorithm': EncryptionAlgorithm.RSA_4096,
                'description': 'Strong RSA key pair for high-security applications'
            },
            'efficient_key_pair': {
                'algorithm': EncryptionAlgorithm.ECC_P256,
                'description': 'Efficient ECC key pair with strong security'
            }
        }
    
    def execute_cryptography_template(self, template_name: str, operation: str, **kwargs) -> CryptographyResult:
        """Execute a predefined cryptography template."""
        if template_name not in self.cryptography_templates:
            raise ValueError(f"Unknown cryptography template: {template_name}")
        
        template = self.cryptography_templates[template_name]
        
        if operation == "encrypt":
            return self.engine.encrypt_symmetric(
                kwargs.get('data', ''),
                template['algorithm'],
                template.get('mode', EncryptionMode.GCM)
            )
        elif operation == "decrypt":
            return self.engine.decrypt_symmetric(
                kwargs.get('encrypted_data', b''),
                template['algorithm'],
                template.get('mode', EncryptionMode.GCM),
                kwargs.get('key'),
                kwargs.get('password'),
                kwargs.get('salt')
            )
        elif operation == "key_generation":
            return self.engine.generate_key_pair(template['algorithm'])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available cryptography templates."""
        return self.cryptography_templates.copy()
    
    def get_cryptography_statistics(self) -> Dict[str, Any]:
        """Get cryptography performance statistics."""
        stats = self.performance_stats.copy()
        stats['operation_history_count'] = len(self.engine.get_operation_history())
        return stats
    
    def analyze_cryptography_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in cryptography operations."""
        operation_history = self.engine.get_operation_history()
        
        if not operation_history:
            return {}
        
        analysis = {
            'algorithm_usage': {},
            'operation_type_distribution': {},
            'average_processing_times': {},
            'error_patterns': {}
        }
        
        # Analyze algorithm usage
        for result in operation_history:
            algo = result.algorithm.value
            analysis['algorithm_usage'][algo] = analysis['algorithm_usage'].get(algo, 0) + 1
        
        # Analyze operation type distribution
        for result in operation_history:
            op_type = result.operation_type.value
            analysis['operation_type_distribution'][op_type] = analysis['operation_type_distribution'].get(op_type, 0) + 1
        
        # Analyze processing times
        for result in operation_history:
            algo = result.algorithm.value
            if algo not in analysis['average_processing_times']:
                analysis['average_processing_times'][algo] = []
            analysis['average_processing_times'][algo].append(result.processing_time)
        
        # Calculate averages
        for algo, times in analysis['average_processing_times'].items():
            analysis['average_processing_times'][algo] = sum(times) / len(times)
        
        # Analyze error patterns
        error_count = sum(1 for result in operation_history if not result.success)
        analysis['error_patterns']['total_errors'] = error_count
        analysis['error_patterns']['error_rate'] = error_count / len(operation_history) if operation_history else 0
        
        return analysis

async def main():
    """Example usage and testing."""
    try:
        # Initialize manager
        manager = CryptographyManager()
        
        print("🔐 Available cryptography templates:")
        templates = manager.get_available_templates()
        for name, template in templates.items():
            print(f"  - {name}: {template['description']}")
        
        # Test symmetric encryption/decryption
        test_data = "Hello, Cryptography World!"
        print(f"\n🚀 Testing symmetric encryption/decryption...")
        
        # Encrypt
        encrypt_result = manager.engine.encrypt_symmetric(
            test_data, 
            EncryptionAlgorithm.AES_256, 
            EncryptionMode.GCM
        )
        
        if encrypt_result.success:
            print(f"  ✅ Encryption successful: {len(encrypt_result.result_data)} bytes")
            
            # Decrypt
            decrypt_result = manager.engine.decrypt_symmetric(
                encrypt_result.result_data,
                EncryptionAlgorithm.AES_256,
                EncryptionMode.GCM,
                key=manager.engine._generate_symmetric_key(EncryptionAlgorithm.AES_256)
            )
            
            if decrypt_result.success:
                decrypted_text = decrypt_result.result_data.decode('utf-8')
                print(f"  ✅ Decryption successful: '{decrypted_text}'")
                print(f"  ✅ Data integrity: {test_data == decrypted_text}")
            else:
                print(f"  ❌ Decryption failed: {decrypt_result.errors}")
        else:
            print(f"  ❌ Encryption failed: {encrypt_result.errors}")
        
        # Test key generation
        print(f"\n🔑 Testing key generation...")
        key_result = manager.engine.generate_key_pair(EncryptionAlgorithm.RSA_2048)
        
        if key_result.success:
            print(f"  ✅ Key pair generated: {key_result.key_info['key_size']} bits")
            print(f"  ✅ Public key: {key_result.key_info['public_key'][:50]}...")
        else:
            print(f"  ❌ Key generation failed: {key_result.errors}")
        
        # Test asymmetric encryption/decryption
        if key_result.success:
            print(f"\n🔐 Testing asymmetric encryption/decryption...")
            
            # Encrypt with public key
            asym_encrypt_result = manager.engine.encrypt_asymmetric(
                "Secret message", 
                key_result.key_info['public_key'].encode('utf-8'),
                EncryptionAlgorithm.RSA_2048
            )
            
            if asym_encrypt_result.success:
                print(f"  ✅ Asymmetric encryption successful: {len(asym_encrypt_result.result_data)} bytes")
                
                # Decrypt with private key
                asym_decrypt_result = manager.engine.decrypt_asymmetric(
                    asym_encrypt_result.result_data,
                    key_result.key_info['private_key'].encode('utf-8'),
                    EncryptionAlgorithm.RSA_2048
                )
                
                if asym_decrypt_result.success:
                    decrypted_text = asym_decrypt_result.result_data.decode('utf-8')
                    print(f"  ✅ Asymmetric decryption successful: '{decrypted_text}'")
                else:
                    print(f"  ❌ Asymmetric decryption failed: {asym_decrypt_result.errors}")
            else:
                print(f"  ❌ Asymmetric encryption failed: {asym_encrypt_result.errors}")
        
        # Test digital signatures
        if key_result.success:
            print(f"\n✍️  Testing digital signatures...")
            
            # Sign data
            sign_result = manager.engine.sign_data(
                "Data to sign", 
                key_result.key_info['private_key'].encode('utf-8'),
                EncryptionAlgorithm.RSA_2048
            )
            
            if sign_result.success:
                print(f"  ✅ Data signed: {len(sign_result.result_data)} bytes")
                
                # Verify signature
                verify_result = manager.engine.verify_signature(
                    "Data to sign", 
                    sign_result.result_data,
                    key_result.key_info['public_key'].encode('utf-8'),
                    EncryptionAlgorithm.RSA_2048
                )
                
                if verify_result.success:
                    print(f"  ✅ Signature verified successfully")
                else:
                    print(f"  ❌ Signature verification failed: {verify_result.errors}")
            else:
                print(f"  ❌ Signing failed: {sign_result.errors}")
        
        # Get statistics
        stats = manager.get_cryptography_statistics()
        print(f"\n📈 Cryptography statistics:")
        print(f"  - Total operations: {stats['operation_history_count']}")
        
        # Analyze patterns
        analysis = manager.analyze_cryptography_patterns()
        if analysis:
            print(f"\n🔍 Cryptography pattern analysis:")
            print(f"  - Most used algorithm: {max(analysis['algorithm_usage'].items(), key=lambda x: x[1])[0]}")
            print(f"  - Error rate: {analysis['error_patterns']['error_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
