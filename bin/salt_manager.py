#!/usr/bin/env python3
"""
Salt Management Utility
Generates and manages persistent encryption salts for the cybersecurity agent.
"""

import os
import sys
import secrets
import hashlib
from pathlib import Path
from typing import Optional

class SaltManager:
    """Manages both device-bound and session-specific encryption salts."""
    
    def __init__(self, salt_file: str = ".salt"):
        self.salt_file = Path(salt_file)
        self.session_salt_file = Path(".session_salt")
        self.salt_length = 32  # 32 bytes = 256 bits for strong security
        
        # Device identifier for device-bound salts
        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from device_identifier import DeviceIdentifier
            self.device_id = DeviceIdentifier()
        except ImportError:
            self.device_id = None
            print("⚠️  Device identifier not available, using fallback salt generation")
    
    def get_or_create_salt(self) -> str:
        """Get existing salt or create a new one if none exists."""
        if self.salt_file.exists():
            return self._read_salt()
        else:
            return self._generate_and_store_salt()
    
    def _read_salt(self) -> str:
        """Read existing salt from file."""
        try:
            salt_data = self.salt_file.read_bytes()
            # Verify salt integrity with hash
            if len(salt_data) >= self.salt_length + 32:  # salt + hash
                salt = salt_data[:self.salt_length]
                stored_hash = salt_data[self.salt_length:]
                # Verify hash
                if hashlib.sha256(salt).digest() == stored_hash:
                    return salt.hex()
                else:
                    print("⚠️  Salt file corrupted, generating new salt...")
                    return self._generate_and_store_salt()
            else:
                print("⚠️  Salt file format invalid, generating new salt...")
                return self._generate_and_store_salt()
        except Exception as e:
            print(f"⚠️  Error reading salt file: {e}")
            print("   Generating new salt...")
            return self._generate_and_store_salt()
    
    def _generate_and_store_salt(self) -> str:
        """Generate a new cryptographically secure salt and store it."""
        try:
            # Generate cryptographically secure random salt
            salt = secrets.token_bytes(self.salt_length)
            
            # Create hash for integrity verification
            salt_hash = hashlib.sha256(salt).digest()
            
            # Store salt + hash
            salt_data = salt + salt_hash
            self.salt_file.write_bytes(salt_data)
            
            # Set restrictive permissions (owner read/write only)
            self.salt_file.chmod(0o600)
            
            print(f"🔐 Generated new persistent salt: {salt.hex()[:16]}...")
            print(f"   Salt stored in: {self.salt_file}")
            print(f"   Salt length: {self.salt_length * 8} bits")
            print("   ⚠️  Keep this file secure - it's required for decryption!")
            
            return salt.hex()
            
        except Exception as e:
            print(f"❌ Failed to generate salt: {e}")
            # Fallback to a deterministic salt (less secure but functional)
            fallback_salt = hashlib.sha256(b"cybersecurity_agent_fallback").digest()[:self.salt_length]
            print("⚠️  Using fallback salt (less secure)")
            return fallback_salt.hex()
    
    def verify_salt_integrity(self) -> bool:
        """Verify the stored salt's integrity."""
        if not self.salt_file.exists():
            print("❌ No salt file found")
            return False
        
        try:
            salt_data = self.salt_file.read_bytes()
            
            # New format: salt (24 bytes) + device_fingerprint (32 bytes) + hash (32 bytes) = 88 bytes
            if len(salt_data) >= 88:
                salt = salt_data[:24]  # First 24 bytes are the actual salt
                stored_hash = salt_data[56:88]  # Last 32 bytes are the hash
                expected_hash = hashlib.sha256(salt).digest()
                
                if stored_hash == expected_hash:
                    print(f"✅ Salt integrity verified: {salt.hex()[:16]}...")
                    return True
                else:
                    print("❌ Salt file corrupted (hash mismatch)")
                    return False
            
            # Old format: salt (32 bytes) + hash (32 bytes) = 64 bytes
            elif len(salt_data) >= 64:
                salt = salt_data[:32]
                stored_hash = salt_data[32:64]
                expected_hash = hashlib.sha256(salt).digest()
                
                if stored_hash == expected_hash:
                    print(f"✅ Salt integrity verified: {salt.hex()[:16]}...")
                    return True
                else:
                    print("❌ Salt file corrupted (hash mismatch)")
                    return False
            else:
                print("❌ Salt file corrupted (invalid length)")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying salt: {e}")
            return False
    
    def backup_salt(self, backup_path: str) -> bool:
        """Create a backup of the salt file."""
        try:
            backup_file = Path(backup_path)
            if self.salt_file.exists():
                import shutil
                shutil.copy2(self.salt_file, backup_file)
                backup_file.chmod(0o600)  # Restrictive permissions
                print(f"✅ Salt backed up to: {backup_path}")
                return True
            else:
                print("❌ No salt file to backup")
                return False
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def restore_salt(self, backup_path: str) -> bool:
        """Restore salt from backup."""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                print(f"❌ Backup file not found: {backup_path}")
                return False
            
            # Verify backup integrity before restoring
            temp_manager = SaltManager(backup_path)
            if temp_manager.verify_salt_integrity():
                import shutil
                shutil.copy2(backup_file, self.salt_file)
                self.salt_file.chmod(0o600)
                print(f"✅ Salt restored from: {backup_path}")
                return True
            else:
                print("❌ Backup file corrupted, cannot restore")
                return False
                
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def get_salt_info(self) -> dict:
        """Get information about all salts."""
        info = {
            'device_bound_salt': {
                'file': str(self.salt_file),
                'exists': self.salt_file.exists(),
                'salt_length_bits': self.salt_length * 8,
                'device_bound': False,
                'integrity_verified': False
            },
            'session_salt': {
                'file': str(self.session_salt_file),
                'exists': self.session_salt_file.exists(),
                'salt_length_bits': self.salt_length * 8,
                'integrity_verified': False
            },
            'device_info': None
        }
        
        # Device-bound salt info
        if self.salt_file.exists():
            try:
                salt_data = self.salt_file.read_bytes()
                
                # New format: salt (24 bytes) + device_fingerprint (32 bytes) + hash (32 bytes) = 88 bytes
                if len(salt_data) >= 88:
                    salt = salt_data[:24]  # First 24 bytes are the actual salt
                    device_fingerprint_stored = salt_data[24:56].decode('utf-8')  # Stored device fingerprint
                    info['device_bound_salt']['salt_hex'] = salt.hex()
                    info['device_bound_salt']['salt_preview'] = f"{salt.hex()[:16]}..."
                    info['device_bound_salt']['file_size'] = len(salt_data)
                    info['device_bound_salt']['integrity_verified'] = self.verify_salt_integrity()
                    info['device_bound_salt']['stored_device_fingerprint'] = device_fingerprint_stored
                    
                    # Check if it's device-bound
                    if self.device_id:
                        device_fingerprint = self.device_id.generate_device_fingerprint()
                        info['device_bound_salt']['device_bound'] = self._verify_device_bound_salt(salt.hex(), device_fingerprint)
                        info['device_bound_salt']['device_fingerprint'] = device_fingerprint
                    
                    # Check file permissions
                    stat = self.salt_file.stat()
                    info['device_bound_salt']['permissions'] = oct(stat.st_mode)[-3:]
                    info['device_bound_salt']['owner'] = stat.st_uid
                
                # Old format: salt (32 bytes) + hash (32 bytes) = 64 bytes
                elif len(salt_data) >= 64:
                    salt = salt_data[:32]
                    info['device_bound_salt']['salt_hex'] = salt.hex()
                    info['device_bound_salt']['salt_preview'] = f"{salt.hex()[:16]}..."
                    info['device_bound_salt']['file_size'] = len(salt_data)
                    info['device_bound_salt']['integrity_verified'] = self.verify_salt_integrity()
                    info['device_bound_salt']['format'] = 'legacy'
                    
                    # Check if it's device-bound
                    if self.device_id:
                        device_fingerprint = self.device_id.generate_device_fingerprint()
                        info['device_bound_salt']['device_bound'] = self._verify_device_bound_salt(salt.hex(), device_fingerprint)
                        info['device_bound_salt']['device_fingerprint'] = device_fingerprint
                    
                    # Check file permissions
                    stat = self.salt_file.stat()
                    info['device_bound_salt']['permissions'] = oct(stat.st_mode)[-3:]
                    info['device_bound_salt']['owner'] = stat.st_uid
                    
            except Exception as e:
                info['device_bound_salt']['error'] = str(e)
        
        # Session salt info
        if self.session_salt_file.exists():
            try:
                salt_data = self.session_salt_file.read_bytes()
                if len(salt_data) >= self.salt_length + 32:
                    salt = salt_data[:self.salt_length]
                    info['session_salt']['salt_hex'] = salt.hex()
                    info['session_salt']['salt_preview'] = f"{salt.hex()[:16]}..."
                    info['session_salt']['file_size'] = len(salt_data)
                    info['session_salt']['integrity_verified'] = self._verify_session_salt_integrity()
                    
                    # Check file permissions
                    stat = self.session_salt_file.stat()
                    info['session_salt']['permissions'] = oct(stat.st_mode)[-3:]
                    info['session_salt']['owner'] = stat.st_uid
                    
            except Exception as e:
                info['session_salt']['error'] = str(e)
        
        # Device info
        if self.device_id:
            try:
                device_info = self.device_id.get_device_info()
                info['device_info'] = {
                    'fingerprint': device_info['fingerprint'],
                    'system': f"{device_info['system']['system']} {device_info['system']['machine']}",
                    'hostname': device_info['system']['hostname']
                }
            except Exception as e:
                info['device_info'] = {'error': str(e)}
        
        return info
    
    def get_or_create_device_bound_salt(self) -> str:
        """Get or create a device-bound salt for knowledge graph context memory."""
        if self.device_id is None:
            # Fallback to standard salt if device identifier not available
            return self.get_or_create_salt()
        
        try:
            # Generate device fingerprint
            device_fingerprint = self.device_id.generate_device_fingerprint()
            
            # Create device-bound salt by combining device fingerprint with a random component
            if self.salt_file.exists():
                # Read existing salt and verify it's device-bound
                existing_salt = self._read_device_bound_salt()
                if existing_salt and self._verify_device_bound_salt(existing_salt, device_fingerprint):
                    return existing_salt
                else:
                    print("⚠️  Existing salt is not device-bound, generating new one...")
            
            # Generate new device-bound salt
            return self._generate_device_bound_salt(device_fingerprint)
            
        except Exception as e:
            print(f"⚠️  Error generating device-bound salt: {e}")
            return self.get_or_create_salt()
    
    def _read_device_bound_salt(self) -> Optional[str]:
        """Read existing device-bound salt from file."""
        try:
            salt_data = self.salt_file.read_bytes()
            # New format: salt (24 bytes) + device_fingerprint (32 bytes) + hash (32 bytes) = 88 bytes
            if len(salt_data) >= 88:
                salt = salt_data[:24]  # First 24 bytes are the actual salt
                stored_hash = salt_data[56:88]  # Last 32 bytes are the hash
                # Verify hash
                if hashlib.sha256(salt).digest() == stored_hash:
                    return salt.hex()
                else:
                    print("⚠️  Device-bound salt file corrupted (hash mismatch)")
                    return None
            # Old format: salt (32 bytes) + hash (32 bytes) = 64 bytes
            elif len(salt_data) >= 64:
                salt = salt_data[:32]
                stored_hash = salt_data[32:64]
                # Verify hash
                if hashlib.sha256(salt).digest() == stored_hash:
                    return salt.hex()
                else:
                    print("⚠️  Device-bound salt file corrupted (hash mismatch)")
                    return None
            else:
                print("⚠️  Device-bound salt file format invalid")
                return None
        except Exception as e:
            print(f"⚠️  Error reading device-bound salt file: {e}")
            return None
    
    def _verify_device_bound_salt(self, salt: str, device_fingerprint: str) -> bool:
        """Verify if a salt is bound to the current device."""
        try:
            # The salt format is: salt (48 chars) + device_fingerprint (32 chars) + hash (32 bytes)
            # We need to read the actual file to get the stored device fingerprint
            if not self.salt_file.exists():
                return False
            
            salt_data = self.salt_file.read_bytes()
            # Format: salt (48 chars = 24 bytes) + device_fingerprint (32 chars = 32 bytes) + hash (32 bytes)
            if len(salt_data) >= 24 + 32 + 32:  # 88 bytes total
                stored_device_fingerprint = salt_data[24:56].decode('utf-8')  # Extract stored device fingerprint
                return stored_device_fingerprint == device_fingerprint
            else:
                # Fallback for older salt formats
                if len(salt) >= 32:
                    salt_device_part = salt[:32]
                    return salt_device_part == device_fingerprint
                else:
                    return False
        except Exception:
            return False
    
    def _generate_device_bound_salt(self, device_fingerprint: str) -> str:
        """Generate a new device-bound salt."""
        try:
            # Use the full 32-character device fingerprint as the base
            # Add a small random component for additional security
            random_part = secrets.token_bytes(8).hex()  # 16 random hex characters
            
            device_bound_salt = device_fingerprint + random_part
            
            # Store the device fingerprint separately for verification
            # Format: device_fingerprint (32 chars) + random_part (16 chars) + device_fingerprint (32 chars) + hash (32 bytes)
            salt_bytes = bytes.fromhex(device_bound_salt)
            device_fingerprint_bytes = device_fingerprint.encode('utf-8')
            salt_hash = hashlib.sha256(salt_bytes).digest()
            
            # Store: salt + device_fingerprint + hash
            salt_data = salt_bytes + device_fingerprint_bytes + salt_hash
            self.salt_file.write_bytes(salt_data)
            
            # Set restrictive permissions
            self.salt_file.chmod(0o600)
            
            print(f"🔐 Generated new device-bound salt: {device_bound_salt[:16]}...")
            print(f"   Device fingerprint: {device_fingerprint[:16]}...")
            print(f"   Salt stored in: {self.salt_file}")
            print(f"   ⚠️  This salt is bound to this device!")
            
            return device_bound_salt
            
        except Exception as e:
            print(f"❌ Failed to generate device-bound salt: {e}")
            return self.get_or_create_salt()
    
    def get_or_create_session_salt(self) -> str:
        """Get or create a session-specific salt for output objects (not device-bound)."""
        try:
            if self.session_salt_file.exists():
                return self._read_session_salt()
            else:
                return self._generate_session_salt()
        except Exception as e:
            print(f"⚠️  Error with session salt: {e}")
            # Fallback to random salt
            return secrets.token_bytes(self.salt_length).hex()
    
    def _read_session_salt(self) -> str:
        """Read existing session salt from file."""
        try:
            salt_data = self.session_salt_file.read_bytes()
            if len(salt_data) >= self.salt_length + 32:
                salt = salt_data[:self.salt_length]
                stored_hash = salt_data[self.salt_length:]
                # Verify hash
                if hashlib.sha256(salt).digest() == stored_hash:
                    return salt.hex()
                else:
                    print("⚠️  Session salt file corrupted, generating new one...")
                    return self._generate_session_salt()
            else:
                print("⚠️  Session salt file format invalid, generating new one...")
                return self._generate_session_salt()
        except Exception as e:
            print(f"⚠️  Error reading session salt file: {e}")
            return self._generate_session_salt()
    
    def _generate_session_salt(self) -> str:
        """Generate a new session-specific salt."""
        try:
            # Generate completely random salt (not device-bound)
            session_salt = secrets.token_bytes(self.salt_length)
            
            # Create hash for integrity verification
            salt_hash = hashlib.sha256(session_salt).digest()
            
            # Store salt + hash
            salt_data = session_salt + salt_hash
            self.session_salt_file.write_bytes(salt_data)
            
            # Set restrictive permissions
            self.session_salt_file.chmod(0o600)
            
            print(f"🔐 Generated new session salt: {session_salt.hex()[:16]}...")
            print(f"   Session salt stored in: {self.session_salt_file}")
            print(f"   ⚠️  This salt is NOT device-bound!")
            
            return session_salt.hex()
            
        except Exception as e:
            print(f"❌ Failed to generate session salt: {e}")
            # Fallback to deterministic salt
            fallback_salt = hashlib.sha256(b"session_fallback").digest()[:self.salt_length]
            return fallback_salt.hex()
    
    def _verify_session_salt_integrity(self) -> bool:
        """Verify the stored session salt's integrity."""
        if not self.session_salt_file.exists():
            return False
        
        try:
            salt_data = self.session_salt_file.read_bytes()
            if len(salt_data) < self.salt_length + 32:
                return False
            
            salt = salt_data[:self.salt_length]
            stored_hash = salt_data[self.salt_length:]
            expected_hash = hashlib.sha256(salt).digest()
            
            return stored_hash == expected_hash
                
        except Exception:
            return False

def main():
    """Command line interface for salt management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Salt Management Utility")
    parser.add_argument("--info", action="store_true", help="Show detailed salt information")
    parser.add_argument("--verify", action="store_true", help="Verify salt integrity")
    parser.add_argument("--backup", metavar="PATH", help="Backup salt to specified path")
    parser.add_argument("--restore", metavar="PATH", help="Restore salt from specified path")
    parser.add_argument("--generate", action="store_true", help="Generate new salt (overwrites existing)")
    parser.add_argument("--device-bound", action="store_true", help="Generate device-bound salt")
    parser.add_argument("--session", action="store_true", help="Generate session salt")
    parser.add_argument("--device-info", action="store_true", help="Show device information")
    
    args = parser.parse_args()
    
    salt_manager = SaltManager()
    
    if args.info:
        info = salt_manager.get_salt_info()
        print("🔐 Salt Information:")
        
        # Device-bound salt info
        print(f"\n📱 Device-Bound Salt (Knowledge Graph Context Memory):")
        print(f"   File: {info['device_bound_salt']['file']}")
        print(f"   Exists: {info['device_bound_salt']['exists']}")
        print(f"   Length: {info['device_bound_salt']['salt_length_bits']} bits")
        print(f"   Device Bound: {info['device_bound_salt'].get('device_bound', 'Unknown')}")
        
        if info['device_bound_salt']['exists']:
            print(f"   Salt: {info['device_bound_salt'].get('salt_preview', 'N/A')}")
            print(f"   File Size: {info['device_bound_salt'].get('file_size', 'N/A')} bytes")
            print(f"   Permissions: {info['device_bound_salt'].get('permissions', 'N/A')}")
            print(f"   Integrity: {'✅ Verified' if info['device_bound_salt']['integrity_verified'] else '❌ Failed'}")
            if info['device_bound_salt'].get('device_fingerprint'):
                print(f"   Device: {info['device_bound_salt']['device_fingerprint']}")
            if info['device_bound_salt'].get('stored_device_fingerprint'):
                print(f"   Stored Device: {info['device_bound_salt']['stored_device_fingerprint']}")
        
        # Session salt info
        print(f"\n🔄 Session Salt (Output Objects):")
        print(f"   File: {info['session_salt']['file']}")
        print(f"   Exists: {info['session_salt']['exists']}")
        print(f"   Length: {info['session_salt']['salt_length_bits']} bits")
        
        if info['session_salt']['exists']:
            print(f"   Salt: {info['session_salt'].get('salt_preview', 'N/A')}")
            print(f"   File Size: {info['session_salt'].get('file_size', 'N/A')} bytes")
            print(f"   Permissions: {info['session_salt'].get('permissions', 'N/A')}")
            print(f"   Integrity: {'✅ Verified' if info['session_salt']['integrity_verified'] else '❌ Failed'}")
        
        # Device info
        if info['device_info']:
            print(f"\n🖥️  Device Information:")
            print(f"   Fingerprint: {info['device_info'].get('fingerprint', 'N/A')}")
            print(f"   System: {info['device_info'].get('system', 'N/A')}")
            print(f"   Hostname: {info['device_info'].get('hostname', 'N/A')}")
        
        # Error handling
        for salt_type in ['device_bound_salt', 'session_salt']:
            if 'error' in info[salt_type]:
                print(f"   ❌ {salt_type} Error: {info[salt_type]['error']}")
    
    elif args.verify:
        print("🔐 Verifying salt integrity...")
        device_ok = salt_manager.verify_salt_integrity()
        session_ok = salt_manager._verify_session_salt_integrity()
        
        if device_ok and session_ok:
            print("✅ All salt integrity verifications passed")
        else:
            print("❌ Some salt integrity verifications failed")
            if not device_ok:
                print("   ❌ Device-bound salt verification failed")
            if not session_ok:
                print("   ❌ Session salt verification failed")
    
    elif args.backup:
        salt_manager.backup_salt(args.backup)
    
    elif args.restore:
        salt_manager.restore_salt(args.backup)
    
    elif args.generate:
        print("⚠️  This will overwrite the existing device-bound salt!")
        print("   All encrypted knowledge graph data will become inaccessible!")
        response = input("Continue? (type 'YES' to confirm): ")
        if response == 'YES':
            # Remove existing salt file and generate new one
            if salt_manager.salt_file.exists():
                salt_manager.salt_file.unlink()
            new_salt = salt_manager.get_or_create_device_bound_salt()
            print(f"✅ New device-bound salt generated: {new_salt[:16]}...")
        else:
            print("❌ Salt generation cancelled")
    
    elif args.device_bound:
        print("🔐 Generating device-bound salt for knowledge graph context memory...")
        salt = salt_manager.get_or_create_device_bound_salt()
        print(f"✅ Device-bound salt: {salt[:16]}...")
        print("   This salt is bound to this specific device!")
    
    elif args.session:
        print("🔐 Generating session salt for output objects...")
        salt = salt_manager.get_or_create_session_salt()
        print(f"✅ Session salt: {salt[:16]}...")
        print("   This salt is NOT device-bound and can be shared!")
    
    elif args.device_info:
        if salt_manager.device_id:
            device_info = salt_manager.device_id.get_device_info()
            print("🖥️  Device Information:")
            print(f"   Fingerprint: {device_info['fingerprint']}")
            print(f"   System: {device_info['system']['system']} {device_info['system']['machine']}")
            print(f"   Hostname: {device_info['system']['hostname']}")
            print(f"   Platform: {device_info['system']['platform']}")
            
            if device_info['cpu']:
                print("   CPU:")
                for key, value in device_info['cpu'].items():
                    print(f"     {key}: {value}")
            
            if device_info['disk']:
                print("   Disk:")
                for key, value in device_info['disk'].items():
                    print(f"     {key}: {value}")
            
            if device_info['network']['mac_addresses']:
                print("   Network Interfaces:")
                for i, mac in enumerate(device_info['network']['mac_addresses']):
                    print(f"     MAC {i+1}: {mac}")
        else:
            print("❌ Device identifier not available")
    
    else:
        # Default: show current salts
        print("🔐 Current Salts:")
        
        # Device-bound salt
        try:
            device_salt = salt_manager.get_or_create_device_bound_salt()
            print(f"   📱 Device-Bound: {device_salt[:16]}... (Knowledge Graph)")
        except Exception as e:
            print(f"   📱 Device-Bound: Error - {e}")
        
        # Session salt
        try:
            session_salt = salt_manager.get_or_create_session_salt()
            print(f"   🔄 Session: {session_salt[:16]}... (Output Objects)")
        except Exception as e:
            print(f"   🔄 Session: Error - {e}")
        
        print("\n💡 Use --info for detailed information")
        print("   Use --device-bound to generate device-bound salt")
        print("   Use --session to generate session salt")

if __name__ == "__main__":
    main()
