#!/usr/bin/env python3
"""
Output Encryption Manager
Handles encryption of output objects using session-specific salts (not device-bound).
"""

import os
import sys
import base64
import hashlib
from pathlib import Path
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class OutputEncryptionManager:
    """Manages encryption of output objects using session-specific salts."""
    
    def __init__(self, password_hash: str = None):
        self.password_hash = password_hash or os.getenv('ENCRYPTION_PASSWORD_HASH', '')
        
        # Get session-specific salt (not device-bound)
        try:
            from bin.salt_manager import SaltManager
            salt_manager = SaltManager()
            self.session_salt = salt_manager.get_or_create_session_salt()
            print("🔐 Output encryption using session-specific salt")
        except ImportError:
            # Fallback to random salt if salt manager not available
            import secrets
            self.session_salt = secrets.token_bytes(32).hex()
            print("⚠️  Salt manager not available, using random session salt")
        
        # Derive encryption key
        self.key = self._derive_key_from_hash(self.password_hash)
        self.cipher = Fernet(self.key) if self.key else None
    
    def _derive_key_from_hash(self, password_hash: str) -> Optional[bytes]:
        """Derive encryption key from password hash using session salt."""
        try:
            if not password_hash:
                print("⚠️  No password hash provided for output encryption")
                return None
            
            # Convert session salt to bytes
            salt_bytes = bytes.fromhex(self.session_salt) if isinstance(self.session_salt, str) else self.session_salt
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            # Derive key from password hash
            key = base64.urlsafe_b64encode(kdf.derive(password_hash.encode()))
            return key
            
        except Exception as e:
            print(f"❌ Failed to derive output encryption key: {e}")
            return None
    
    def encrypt_output_data(self, data: Union[str, bytes]) -> Optional[bytes]:
        """Encrypt output data using session-specific salt."""
        if not self.cipher:
            print("⚠️  Output encryption not available (no cipher)")
            return data.encode() if isinstance(data, str) else data
        
        try:
            # Convert to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt the data
            encrypted_data = self.cipher.encrypt(data_bytes)
            return encrypted_data
            
        except Exception as e:
            print(f"❌ Output encryption failed: {e}")
            return data_bytes if 'data_bytes' in locals() else (data.encode() if isinstance(data, str) else data)
    
    def decrypt_output_data(self, encrypted_data: bytes) -> Optional[bytes]:
        """Decrypt output data using session-specific salt."""
        if not self.cipher:
            print("⚠️  Output decryption not available (no cipher)")
            return encrypted_data
        
        try:
            # Decrypt the data
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return decrypted_data
            
        except Exception as e:
            print(f"❌ Output decryption failed: {e}")
            return encrypted_data
    
    def encrypt_output_file(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Encrypt an output file and return path to encrypted version."""
        if not self.cipher:
            print("⚠️  Output file encryption not available (no cipher)")
            return Path(file_path)
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ Output file not found: {file_path}")
                return None
            
            # Read file content
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Encrypt content
            encrypted_data = self.encrypt_output_data(data)
            if encrypted_data is None:
                return file_path
            
            # Create encrypted file path
            encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
            
            # Write encrypted content
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            print(f"🔐 Output file encrypted: {encrypted_path}")
            return encrypted_path
            
        except Exception as e:
            print(f"❌ Output file encryption failed: {e}")
            return Path(file_path)
    
    def decrypt_output_file(self, encrypted_path: Union[str, Path]) -> Optional[Path]:
        """Decrypt an encrypted output file and return path to decrypted version."""
        if not self.cipher:
            print("⚠️  Output file decryption not available (no cipher)")
            return Path(encrypted_path)
        
        try:
            encrypted_path = Path(encrypted_path)
            if not encrypted_path.exists():
                print(f"❌ Encrypted output file not found: {encrypted_path}")
                return None
            
            # Read encrypted content
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt content
            decrypted_data = self.decrypt_output_data(encrypted_data)
            if decrypted_data is None:
                return encrypted_path
            
            # Create decrypted file path
            decrypted_path = encrypted_path.with_suffix('').with_suffix(
                encrypted_path.suffix.replace('.encrypted', '')
            )
            
            # Write decrypted content
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            print(f"🔓 Output file decrypted: {decrypted_path}")
            return decrypted_path
            
        except Exception as e:
            print(f"❌ Output file decryption failed: {e}")
            return Path(encrypted_path)
    
    def get_encryption_info(self) -> dict:
        """Get information about the output encryption setup."""
        return {
            'enabled': self.cipher is not None,
            'password_hash_provided': bool(self.password_hash),
            'session_salt_preview': self.session_salt[:16] + "..." if self.session_salt else "None",
            'salt_type': 'Session-specific (not device-bound)',
            'key_derived': self.key is not None,
            'cipher_available': self.cipher is not None
        }
    
    def change_session_salt(self) -> bool:
        """Generate a new session salt for output encryption."""
        try:
            from bin.salt_manager import SaltManager
            salt_manager = SaltManager()
            
            # Remove existing session salt file to force regeneration
            if salt_manager.session_salt_file.exists():
                salt_manager.session_salt_file.unlink()
            
            # Generate new session salt
            new_session_salt = salt_manager.get_or_create_session_salt()
            self.session_salt = new_session_salt
            
            # Re-derive encryption key
            self.key = self._derive_key_from_hash(self.password_hash)
            self.cipher = Fernet(self.key) if self.key else None
            
            print(f"✅ New session salt generated: {new_session_salt[:16]}...")
            print("   Output encryption now uses new session salt")
            return True
            
        except Exception as e:
            print(f"❌ Failed to change session salt: {e}")
            return False

def main():
    """Command line interface for output encryption management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage output encryption")
    parser.add_argument('--info', action='store_true', help='Show encryption information')
    parser.add_argument('--encrypt', metavar='FILE', help='Encrypt an output file')
    parser.add_argument('--decrypt', metavar='FILE', help='Decrypt an encrypted output file')
    parser.add_argument('--change-salt', action='store_true', help='Generate new session salt')
    
    args = parser.parse_args()
    
    # Check if encryption is enabled
    if not os.getenv('ENCRYPTION_ENABLED', 'false').lower() == 'true':
        print("⚠️  Output encryption is not enabled")
        print("   Set ENCRYPTION_ENABLED=true to enable")
        return
    
    # Check if password hash is provided
    if not os.getenv('ENCRYPTION_PASSWORD_HASH'):
        print("⚠️  No password hash provided for output encryption")
        print("   Set ENCRYPTION_PASSWORD_HASH to enable output encryption")
        return
    
    output_encryption = OutputEncryptionManager()
    
    if args.info:
        info = output_encryption.get_encryption_info()
        print("🔐 Output Encryption Information:")
        print(f"   Enabled: {info['enabled']}")
        print(f"   Password Hash: {'✅ Provided' if info['password_hash_provided'] else '❌ Missing'}")
        print(f"   Session Salt: {info['session_salt_preview']}")
        print(f"   Salt Type: {info['salt_type']}")
        print(f"   Key Derived: {info['key_derived']}")
        print(f"   Cipher Available: {info['cipher_available']}")
    
    elif args.encrypt:
        result = output_encryption.encrypt_output_file(args.encrypt)
        if result:
            print(f"✅ File encrypted successfully: {result}")
        else:
            print("❌ File encryption failed")
    
    elif args.decrypt:
        result = output_encryption.decrypt_output_file(args.decrypt)
        if result:
            print(f"✅ File decrypted successfully: {result}")
        else:
            print("❌ File decryption failed")
    
    elif args.change_salt:
        if output_encryption.change_session_salt():
            print("✅ Session salt changed successfully")
        else:
            print("❌ Failed to change session salt")
    
    else:
        # Default: show info
        info = output_encryption.get_encryption_info()
        print("🔐 Output Encryption Status:")
        print(f"   Status: {'✅ Active' if info['enabled'] else '❌ Inactive'}")
        print(f"   Session Salt: {info['session_salt_preview']}")
        print(f"   Type: {info['salt_type']}")
        print("\n💡 Use --info for detailed information")
        print("   Use --encrypt FILE to encrypt an output file")
        print("   Use --decrypt FILE to decrypt an encrypted file")
        print("   Use --change-salt to generate new session salt")

if __name__ == "__main__":
    main()
