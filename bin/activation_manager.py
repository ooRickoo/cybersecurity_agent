#!/usr/bin/env python3
"""
Activation Management System for Cybersecurity Agent
Creates and verifies host-bound activation tokens for secure access control.
"""

import os
import sys
import hashlib
import secrets
import getpass
from pathlib import Path
from typing import Optional, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import json
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from bin.device_identifier import DeviceIdentifier
    DEVICE_ID_AVAILABLE = True
except ImportError:
    DEVICE_ID_AVAILABLE = False
    print("⚠️  Device identifier not available, using fallback activation")

class ActivationManager:
    """Manages host-bound activation tokens for the cybersecurity agent."""
    
    def __init__(self, activation_file: str = ".activation"):
        self.activation_file = Path(activation_file)
        self.device_id = DeviceIdentifier() if DEVICE_ID_AVAILABLE else None
        self.activation_salt_length = 32  # 32 bytes for strong security
        self.activation_token_length = 64  # 64 bytes for activation token
        
    def create_activation(self, password: str) -> Tuple[bool, str]:
        """
        Create a new activation token bound to this host.
        
        Args:
            password: Master password for activation
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Generate device fingerprint
            device_fingerprint = self._get_device_fingerprint()
            if not device_fingerprint:
                return False, "Failed to generate device fingerprint"
            
            # Derive activation key from password and device fingerprint
            activation_key = self._derive_activation_key(password, device_fingerprint)
            
            # Generate activation token
            activation_token = self._generate_activation_token(activation_key, device_fingerprint)
            
            # Create activation data
            activation_data = {
                'device_fingerprint': device_fingerprint,
                'activation_token': activation_token,
                'created_at': time.time(),
                'version': '1.0'
            }
            
            # Encrypt and store activation data
            encrypted_data = self._encrypt_activation_data(activation_data, activation_key)
            self._store_activation_data(encrypted_data)
            
            return True, f"✅ Activation created successfully for this host"
            
        except Exception as e:
            return False, f"❌ Failed to create activation: {str(e)}"
    
    def verify_activation(self, password: str) -> Tuple[bool, str]:
        """
        Verify activation token for this host.
        
        Args:
            password: Master password for verification
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if activation file exists
            if not self.activation_file.exists():
                return False, "❌ No activation file found. Please run activation first."
            
            # Load and decrypt activation data
            activation_data = self._load_activation_data(password)
            if not activation_data:
                return False, "❌ Failed to decrypt activation data. Check password or activation file."
            
            # Verify device fingerprint
            current_fingerprint = self._get_device_fingerprint()
            if not current_fingerprint:
                return False, "❌ Failed to generate current device fingerprint"
            
            if activation_data['device_fingerprint'] != current_fingerprint:
                return False, "❌ Device fingerprint mismatch. Activation is bound to a different host."
            
            # Verify activation token
            activation_key = self._derive_activation_key(password, current_fingerprint)
            expected_token = self._generate_activation_token(activation_key, current_fingerprint)
            
            if activation_data['activation_token'] != expected_token:
                return False, "❌ Activation token verification failed"
            
            return True, "✅ Activation verified successfully"
            
        except Exception as e:
            return False, f"❌ Activation verification failed: {str(e)}"
    
    def _get_device_fingerprint(self) -> Optional[str]:
        """Generate a unique device fingerprint."""
        try:
            if self.device_id:
                # Use the existing device identifier
                fingerprint_data = self.device_id.generate_device_fingerprint()
                return hashlib.sha256(fingerprint_data.encode()).hexdigest()
            else:
                # Fallback to basic system info
                import platform
                basic_info = f"{platform.system()}-{platform.machine()}-{platform.processor()}"
                return hashlib.sha256(basic_info.encode()).hexdigest()
        except Exception as e:
            print(f"Warning: Failed to generate device fingerprint: {e}")
            return None
    
    def _derive_activation_key(self, password: str, device_fingerprint: str) -> bytes:
        """Derive activation key from password and device fingerprint."""
        # Use device fingerprint as salt for key derivation
        salt = hashlib.sha256(device_fingerprint.encode()).digest()[:16]
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 32 bytes = 256 bits
            salt=salt,
            iterations=100000,  # High iteration count for security
            backend=default_backend()
        )
        
        return kdf.derive(password.encode())
    
    def _generate_activation_token(self, activation_key: bytes, device_fingerprint: str) -> str:
        """Generate activation token using activation key and device fingerprint."""
        # Combine activation key with device fingerprint for token generation
        token_data = activation_key + device_fingerprint.encode()
        return hashlib.sha256(token_data).hexdigest()
    
    def _encrypt_activation_data(self, data: dict, key: bytes) -> bytes:
        """Encrypt activation data using AES encryption."""
        # Convert data to JSON
        json_data = json.dumps(data).encode()
        
        # Generate random IV
        iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(json_data) % 16)
        padded_data = json_data + bytes([padding_length] * padding_length)
        
        # Encrypt data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and encrypted data
        return iv + encrypted_data
    
    def _decrypt_activation_data(self, encrypted_data: bytes, key: bytes) -> Optional[dict]:
        """Decrypt activation data."""
        try:
            # Extract IV and encrypted data
            iv = encrypted_data[:16]
            encrypted = encrypted_data[16:]
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            # Decrypt data
            decrypted_data = decryptor.update(encrypted) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted_data[-1]
            json_data = decrypted_data[:-padding_length]
            
            # Parse JSON
            return json.loads(json_data.decode())
            
        except Exception as e:
            print(f"Warning: Failed to decrypt activation data: {e}")
            return None
    
    def _store_activation_data(self, encrypted_data: bytes):
        """Store encrypted activation data to file."""
        # Add integrity hash
        data_hash = hashlib.sha256(encrypted_data).digest()
        complete_data = encrypted_data + data_hash
        
        # Write to file with restricted permissions
        self.activation_file.write_bytes(complete_data)
        self.activation_file.chmod(0o600)  # Read/write for owner only
    
    def _load_activation_data(self, password: str) -> Optional[dict]:
        """Load and decrypt activation data from file."""
        try:
            # Read file
            complete_data = self.activation_file.read_bytes()
            
            # Verify integrity
            if len(complete_data) < 32:  # Minimum size check
                return None
            
            encrypted_data = complete_data[:-32]
            stored_hash = complete_data[-32:]
            
            if hashlib.sha256(encrypted_data).digest() != stored_hash:
                return None
            
            # Get device fingerprint for key derivation
            device_fingerprint = self._get_device_fingerprint()
            if not device_fingerprint:
                return None
            
            # Derive key
            activation_key = self._derive_activation_key(password, device_fingerprint)
            
            # Decrypt data
            return self._decrypt_activation_data(encrypted_data, activation_key)
            
        except Exception as e:
            print(f"Warning: Failed to load activation data: {e}")
            return None
    
    def get_activation_status(self) -> dict:
        """Get current activation status."""
        status = {
            'activated': False,
            'device_fingerprint': None,
            'created_at': None,
            'version': None
        }
        
        if self.activation_file.exists():
            try:
                # Try to load activation data (without password for status check)
                device_fingerprint = self._get_device_fingerprint()
                if device_fingerprint:
                    status['device_fingerprint'] = device_fingerprint
                    status['activated'] = True
            except Exception:
                pass
        
        return status

def main():
    """Command-line interface for activation management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cybersecurity Agent Activation Manager')
    parser.add_argument('action', choices=['create', 'verify', 'status'], 
                       help='Action to perform')
    parser.add_argument('--password', help='Activation password (will prompt if not provided)')
    parser.add_argument('--file', default='.activation', help='Activation file path')
    
    args = parser.parse_args()
    
    # Initialize activation manager
    activation_manager = ActivationManager(args.file)
    
    # Get password if not provided
    password = args.password
    if not password:
        password = getpass.getpass("Enter activation password: ")
    
    if args.action == 'create':
        success, message = activation_manager.create_activation(password)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.action == 'verify':
        success, message = activation_manager.verify_activation(password)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.action == 'status':
        status = activation_manager.get_activation_status()
        print("🔐 Activation Status:")
        print(f"   Activated: {'✅ Yes' if status['activated'] else '❌ No'}")
        if status['device_fingerprint']:
            print(f"   Device ID: {status['device_fingerprint'][:16]}...")
        if status['created_at']:
            import datetime
            created_time = datetime.datetime.fromtimestamp(status['created_at'])
            print(f"   Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
