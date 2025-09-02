#!/usr/bin/env python3
"""
Credential Vault System
Secure storage and management of credentials, secrets, and web credentials.
"""

import os
import sys
import json
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from bin.salt_manager import SaltManager

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Global password manager for all modules
_global_password_hash = None
_global_vault_instance = None
_vault_initialized = False

def get_global_password_hash():
    """Get the global password hash, loading from file if needed."""
    global _global_password_hash
    
    if _global_password_hash is not None:
        return _global_password_hash
    
    # Try to load from saved file FIRST (highest priority)
    password_file = Path("etc/.password_hash")
    if password_file.exists():
        try:
            with open(password_file, 'r') as f:
                password_hash = f.read().strip()
                if password_hash:
                    # Set both global and environment variable
                    _global_password_hash = password_hash
                    os.environ['ENCRYPTION_PASSWORD_HASH'] = password_hash
                    print("🔐 Loaded password hash from file and set environment variable")
                    return password_hash
        except Exception as e:
            print(f"⚠️  Failed to load password hash from file: {e}")
    
    # Try environment variable second
    password_hash = os.getenv('ENCRYPTION_PASSWORD_HASH', '')
    
    if password_hash:
        _global_password_hash = password_hash
        return password_hash
    
    # Only create default if no saved password exists
    if not password_file.exists():
        print("🔐 No saved password found, will prompt for new one")
        return None
    
    return None

def set_global_password_hash(password_hash: str):
    """Set the global password hash."""
    global _global_password_hash
    _global_password_hash = password_hash
    os.environ['ENCRYPTION_PASSWORD_HASH'] = password_hash

def get_global_credential_vault():
    """Get or create the global credential vault instance."""
    global _global_vault_instance, _vault_initialized
    
    print(f"🔍 DEBUG: get_global_credential_vault called, instance exists: {_global_vault_instance is not None}")
    
    if _global_vault_instance is None:
        # Ensure password is loaded first
        password_hash = get_global_password_hash()
        if not password_hash:
            # Create a default password if none exists
            default_password = "Vosteen2025"
            password_hash = hashlib.sha256(default_password.encode()).hexdigest()
            set_global_password_hash(password_hash)
            print("🔐 Using default password for credential vault")
        
        print(f"🔍 DEBUG: Creating new global vault instance with password hash: {password_hash[:8]}...")
        _global_vault_instance = CredentialVault(password_hash)
        _vault_initialized = True
        print(f"🔍 DEBUG: Global vault instance created successfully")
    else:
        print(f"🔍 DEBUG: Returning existing global vault instance")
    
    return _global_vault_instance

def ensure_vault_initialized():
    """Ensure the credential vault is initialized with proper password."""
    if not _vault_initialized:
        get_global_credential_vault()
    return _global_vault_instance

def initialize_vault_system():
    """Initialize the entire vault system with proper password loading."""
    global _global_password_hash, _global_vault_instance, _vault_initialized
    
    print(f"🔍 DEBUG: initialize_vault_system called")
    print(f"🔍 DEBUG: Current state - password_hash: {'set' if _global_password_hash else 'None'}, vault_instance: {'exists' if _global_vault_instance else 'None'}, initialized: {_vault_initialized}")
    
    # Load password hash first
    password_hash = get_global_password_hash()
    
    if not password_hash:
        # Create a default password if none exists
        default_password = "Vosteen2025"
        password_hash = hashlib.sha256(default_password.encode()).hexdigest()
        set_global_password_hash(password_hash)
        print("🔐 Using default password for credential vault")
    
    print(f"🔍 DEBUG: Password hash ready: {password_hash[:8]}...")
    
    # Create vault instance
    if _global_vault_instance is None:
        print(f"🔍 DEBUG: Creating new vault instance")
        _global_vault_instance = CredentialVault(password_hash)
        _vault_initialized = True
        print(f"🔍 DEBUG: Vault instance created and marked as initialized")
    else:
        print(f"🔍 DEBUG: Using existing vault instance")
    
    return _global_vault_instance

class CredentialVault:
    """Secure credential vault with host-bound encryption."""
    
    def __init__(self, password_hash: str = None, salt: str = None):
        """Initialize the credential vault."""
        print(f"🔍 DEBUG: CredentialVault.__init__ called with password_hash: {'provided' if password_hash else 'None'}")
        
        # Initialize file paths
        self.vault_file = Path("etc/credential_vault.db")
        self.password_file = Path("etc/.password_hash")
        
        # Load or create salt
        self.salt_manager = SaltManager()
        self.host_salt = self.salt_manager.get_or_create_device_bound_salt()
        
        # Set password hash
        if password_hash:
            self.password_hash = password_hash
            print(f"🔍 DEBUG: Using provided password hash: {password_hash[:8]}...")
        else:
            # Try to load saved password hash
            self.password_hash = self._load_saved_password_hash()
            if self.password_hash:
                print(f"🔍 DEBUG: Loaded saved password hash: {self.password_hash[:8]}...")
            else:
                print("🔍 DEBUG: No password hash available")
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load or create vault data
        self._load_or_create_vault()
        
        print(f"🔍 DEBUG: CredentialVault initialization complete")
    
    def _load_saved_password_hash(self) -> str:
        """Load saved password hash from file."""
        try:
            if self.password_file.exists():
                with open(self.password_file, 'r') as f:
                    saved_hash = f.read().strip()
                    if saved_hash:
                        print("🔐 Loaded saved password hash from file")
                        # Update global password manager
                        set_global_password_hash(saved_hash)
                        return saved_hash
        except Exception as e:
            print(f"⚠️  Failed to load saved password hash: {e}")
        return ""
    
    def _save_password_hash(self, password_hash: str):
        """Save password hash to file for persistence."""
        try:
            # Ensure etc directory exists
            self.password_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save password hash
            with open(self.password_file, 'w') as f:
                f.write(password_hash)
            
            # Set restrictive permissions (owner read/write only)
            self.password_file.chmod(0o600)
            print("🔐 Password hash saved for future sessions")
            
        except Exception as e:
            print(f"⚠️  Failed to save password hash: {e}")
    
    def set_password(self, password: str):
        """Set password and save hash for persistence."""
        if password:
            import hashlib
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.password_hash = password_hash
            
            # Save password hash for future sessions
            self._save_password_hash(password_hash)
            
            # Reinitialize encryption with new password
            self._initialize_encryption()
            
            # Save vault with new encryption
            self._save_vault()
            
            print("🔐 Password set and vault re-encrypted")
        else:
            print("❌ No password provided")
    
    def _initialize_encryption(self):
        """Initialize encryption using host-bound salt."""
        try:
            # Use device-bound salt for credential vault
            host_salt = self.salt_manager.get_or_create_device_bound_salt()
            
            if not self.password_hash:
                print("🔐 No encryption password hash provided")
                print("   Vault will be created without encryption")
                return
            
            # Derive encryption key from password hash and host salt
            self.encryption_key = self._derive_key_from_hash(self.password_hash, host_salt)
            self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
            
            if self.cipher:
                print("🔐 Credential vault encryption initialized with host-bound salt")
            else:
                print("⚠️  Failed to initialize credential vault encryption")
                
        except Exception as e:
            print(f"❌ Error initializing credential vault encryption: {e}")
            self.cipher = None
    
    def _derive_key_from_hash(self, password_hash: str, salt: str) -> Optional[bytes]:
        """Derive encryption key from password hash using host salt."""
        try:
            # Convert salt to bytes
            salt_bytes = bytes.fromhex(salt) if isinstance(salt, str) else salt
            
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
            print(f"❌ Failed to derive credential vault key: {e}")
            return None
    
    def _load_or_create_vault(self):
        """Load existing vault or create new one."""
        try:
            if self.vault_file.exists():
                print(f"🔍 DEBUG: Vault file exists, attempting to load: {self.vault_file}")
                self._load_vault()
            else:
                print(f"🔍 DEBUG: Vault file does not exist, creating new vault: {self.vault_file}")
                self._create_new_vault()
        except Exception as e:
            print(f"🔍 DEBUG: Error in _load_or_create_vault: {e}")
            print(f"   Creating new vault...")
            self._create_new_vault()
    
    def _load_vault(self):
        """Load existing vault data from file."""
        try:
            with open(self.vault_file, 'rb') as f:
                file_content = f.read()
            
            if self.cipher:
                try:
                    decrypted_data = self.cipher.decrypt(file_content)
                    self.vault_data = json.loads(decrypted_data.decode('utf-8'))
                    print(f"🔐 Encrypted credential vault loaded: {len(self.vault_data)} entries")
                except Exception as e:
                    print(f"⚠️  Failed to decrypt vault: {e}")
                    print("   Creating new vault...")
                    self.vault_data = self._create_default_vault()
            else:
                try:
                    self.vault_data = json.loads(file_content.decode('utf-8'))
                    print(f"⚠️  Unencrypted credential vault loaded: {len(self.vault_data)} entries")
                except Exception as e:
                    print(f"⚠️  Failed to parse vault as JSON: {e}")
                    print("   Creating new vault...")
                    self.vault_data = self._create_default_vault()
        except Exception as e:
            print(f"⚠️  Failed to read vault file: {e}")
            print("   Creating new vault...")
            self.vault_data = self._create_default_vault()
    
    def _create_new_vault(self):
        """Create a new vault data structure."""
        print("🔐 Creating new encrypted vault")
        self.vault_data = self._create_default_vault()
        # Save the initial vault
        self._save_vault()
    
    def _create_default_vault(self) -> Dict[str, Any]:
        """Create default vault structure."""
        return {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'host_fingerprint': self._get_host_fingerprint(),
                'last_updated': datetime.now().isoformat()
            },
            'credentials': {},
            'secrets': {},
            'web_credentials': {},
            'api_keys': {},
            'certificates': {},
            'ssh_keys': {}
        }
    
    def _get_host_fingerprint(self) -> str:
        """Get current host fingerprint."""
        try:
            from bin.device_identifier import DeviceIdentifier
            device_id = DeviceIdentifier()
            return device_id.generate_device_fingerprint()
        except ImportError:
            return "unknown_host"
    
    def _save_vault(self):
        """Save vault data with encryption or plain text."""
        try:
            # Ensure etc directory exists
            self.vault_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            self.vault_data['metadata']['last_updated'] = datetime.now().isoformat()
            self.vault_data['metadata']['host_fingerprint'] = self._get_host_fingerprint()
            
            if self.cipher:
                # Encrypt and save
                vault_json = json.dumps(self.vault_data, indent=2)
                encrypted_data = self.cipher.encrypt(vault_json.encode('utf-8'))
                
                with open(self.vault_file, 'wb') as f:
                    f.write(encrypted_data)
                
                print("🔐 Vault saved with encryption")
            else:
                # Save as plain text (less secure but functional)
                vault_json = json.dumps(self.vault_data, indent=2)
                
                with open(self.vault_file, 'w') as f:
                    f.write(vault_json)
                
                print("⚠️  Vault saved without encryption (less secure)")
            
            # Set restrictive permissions
            self.vault_file.chmod(0o600)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save vault: {e}")
            return False
    
    def add_credential(self, name: str, username: str, password: str, 
                       description: str = "", tags: List[str] = None, 
                       expiry: Optional[datetime] = None) -> bool:
        """Add a new credential to the vault."""
        try:
            credential_id = self._generate_credential_id()
            
            self.vault_data['credentials'][credential_id] = {
                'name': name,
                'username': username,
                'password': password,  # Will be encrypted
                'description': description,
                'tags': tags or [],
                'created': datetime.now().isoformat(),
                'expiry': expiry.isoformat() if expiry else None,
                'last_used': None,
                'usage_count': 0
            }
            
            # Encrypt sensitive fields
            self._encrypt_credential_fields(credential_id)
            
            success = self._save_vault()
            if success:
                print(f"✅ Credential '{name}' added to vault")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to add credential: {e}")
            return False
    
    def add_secret(self, name: str, secret_value: str, 
                   description: str = "", tags: List[str] = None,
                   expiry: Optional[datetime] = None) -> bool:
        """Add a new secret to the vault."""
        try:
            secret_id = self._generate_credential_id()
            
            self.vault_data['secrets'][secret_id] = {
                'name': name,
                'value': secret_value,  # Will be encrypted
                'description': description,
                'tags': tags or [],
                'created': datetime.now().isoformat(),
                'expiry': expiry.isoformat() if expiry else None,
                'last_used': None,
                'usage_count': 0
            }
            
            # Encrypt sensitive fields
            self._encrypt_secret_fields(secret_id)
            
            success = self._save_vault()
            if success:
                print(f"✅ Secret '{name}' added to vault")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to add secret: {e}")
            return False
    
    def add_web_credential(self, name: str, url: str, username: str, 
                           password: str, description: str = "",
                           tags: List[str] = None) -> bool:
        """Add web credentials to the vault."""
        try:
            credential_id = self._generate_credential_id()
            
            self.vault_data['web_credentials'][credential_id] = {
                'name': name,
                'url': url,
                'username': username,
                'password': password,  # Will be encrypted
                'description': description,
                'tags': tags or [],
                'created': datetime.now().isoformat(),
                'last_used': None,
                'usage_count': 0
            }
            
            # Encrypt sensitive fields
            self._encrypt_web_credential_fields(credential_id)
            
            success = self._save_vault()
            if success:
                print(f"✅ Web credential '{name}' added to vault")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to add web credential: {e}")
            return False
    
    def add_api_key(self, name: str, api_key: str, 
                    description: str = "", tags: List[str] = None,
                    expiry: Optional[datetime] = None) -> bool:
        """Add an API key to the vault."""
        try:
            key_id = self._generate_credential_id()
            
            self.vault_data['api_keys'][key_id] = {
                'name': name,
                'key': api_key,  # Will be encrypted
                'description': description,
                'tags': tags or [],
                'created': datetime.now().isoformat(),
                'expiry': expiry.isoformat() if expiry else None,
                'last_used': None,
                'usage_count': 0
            }
            
            # Encrypt sensitive fields
            self._encrypt_api_key_fields(key_id)
            
            success = self._save_vault()
            if success:
                print(f"✅ API key '{name}' added to vault")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to add API key: {e}")
            return False
    
    def _generate_credential_id(self) -> str:
        """Generate unique credential ID."""
        return secrets.token_hex(16)
    
    def _encrypt_credential_fields(self, credential_id: str):
        """Encrypt sensitive fields in a credential."""
        if not self.cipher:
            return
        
        try:
            credential = self.vault_data['credentials'][credential_id]
            
            # Encrypt password
            if credential['password']:
                encrypted_password = self.cipher.encrypt(credential['password'].encode())
                credential['password'] = base64.b64encode(encrypted_password).decode()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not encrypt credential fields: {e}")
    
    def _encrypt_secret_fields(self, secret_id: str):
        """Encrypt sensitive fields in a secret."""
        if not self.cipher:
            return
        
        try:
            secret = self.vault_data['secrets'][secret_id]
            
            # Encrypt value
            if secret['value']:
                encrypted_value = self.cipher.encrypt(secret['value'].encode())
                secret['value'] = base64.b64encode(encrypted_value).decode()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not encrypt secret fields: {e}")
    
    def _encrypt_web_credential_fields(self, credential_id: str):
        """Encrypt sensitive fields in a web credential."""
        if not self.cipher:
            return
        
        try:
            credential = self.vault_data['web_credentials'][credential_id]
            
            # Encrypt password
            if credential['password']:
                encrypted_password = self.cipher.encrypt(credential['password'].encode())
                credential['password'] = base64.b64encode(encrypted_password).decode()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not encrypt web credential fields: {e}")
    
    def _encrypt_api_key_fields(self, key_id: str):
        """Encrypt sensitive fields in an API key."""
        if not self.cipher:
            return
        
        try:
            api_key = self.vault_data['api_keys'][key_id]
            
            # Encrypt key
            if api_key['key']:
                encrypted_key = self.cipher.encrypt(api_key['key'].encode())
                api_key['key'] = base64.b64encode(encrypted_key).decode()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not encrypt API key fields: {e}")
    
    def get_credential(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a credential by name."""
        try:
            for credential_id, credential in self.vault_data['credentials'].items():
                if credential['name'] == name:
                    # Decrypt password
                    decrypted_credential = credential.copy()
                    if credential['password']:
                        try:
                            encrypted_password = base64.b64decode(credential['password'])
                            decrypted_password = self.cipher.decrypt(encrypted_password)
                            decrypted_credential['password'] = decrypted_password.decode()
                            
                            # Update usage stats
                            credential['last_used'] = datetime.now().isoformat()
                            credential['usage_count'] += 1
                            self._save_vault()
                            
                        except Exception as e:
                            print(f"⚠️  Warning: Could not decrypt password: {e}")
                            decrypted_credential['password'] = None
                    
                    return decrypted_credential
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving credential: {e}")
            return None
    
    def get_secret(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a secret by name."""
        try:
            for secret_id, secret in self.vault_data['secrets'].items():
                if secret['name'] == name:
                    # Decrypt value
                    decrypted_secret = secret.copy()
                    if secret['value']:
                        try:
                            encrypted_value = base64.b64decode(secret['value'])
                            decrypted_value = self.cipher.decrypt(encrypted_value)
                            decrypted_secret['value'] = decrypted_value.decode()
                            
                            # Update usage stats
                            secret['last_used'] = datetime.now().isoformat()
                            secret['usage_count'] += 1
                            self._save_vault()
                            
                        except Exception as e:
                            print(f"⚠️  Warning: Could not decrypt secret value: {e}")
                            decrypted_secret['value'] = None
                    
                    return decrypted_secret
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving secret: {e}")
            return None
    
    def get_web_credential(self, name: str) -> Optional[Dict[str, Any]]:
        """Get web credentials by name."""
        try:
            for credential_id, credential in self.vault_data['web_credentials'].items():
                if credential['name'] == name:
                    # Decrypt password
                    decrypted_credential = credential.copy()
                    if credential['password']:
                        try:
                            encrypted_password = base64.b64decode(credential['password'])
                            decrypted_password = self.cipher.decrypt(encrypted_password)
                            decrypted_credential['password'] = decrypted_password.decode()
                            
                            # Update usage stats
                            credential['last_used'] = datetime.now().isoformat()
                            credential['usage_count'] += 1
                            self._save_vault()
                            
                        except Exception as e:
                            print(f"⚠️  Warning: Could not decrypt web credential password: {e}")
                            decrypted_credential['password'] = None
                    
                    return decrypted_credential
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving web credential: {e}")
            return None
    
    def get_api_key(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an API key by name."""
        try:
            for key_id, api_key in self.vault_data['api_keys'].items():
                if api_key['name'] == name:
                    # Decrypt key
                    decrypted_api_key = api_key.copy()
                    if api_key['key']:
                        try:
                            encrypted_key = base64.b64decode(api_key['key'])
                            decrypted_key = self.cipher.decrypt(encrypted_key)
                            decrypted_api_key['key'] = decrypted_key.decode()
                            
                            # Update usage stats
                            api_key['last_used'] = datetime.now().isoformat()
                            api_key['usage_count'] += 1
                            self._save_vault()
                            
                        except Exception as e:
                            print(f"⚠️  Warning: Could not decrypt API key: {e}")
                            decrypted_api_key['key'] = None
                    
                    return decrypted_api_key
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving API key: {e}")
            return None
    
    def list_credentials(self, category: str = None) -> Dict[str, List[str]]:
        """List all credentials in the vault."""
        result = {}
        
        if category is None or category == 'credentials':
            result['credentials'] = [cred['name'] for cred in self.vault_data['credentials'].values()]
        
        if category is None or category == 'secrets':
            result['secrets'] = [secret['name'] for secret in self.vault_data['secrets'].values()]
        
        if category is None or category == 'web_credentials':
            result['web_credentials'] = [cred['name'] for cred in self.vault_data['web_credentials'].values()]
        
        if category is None or category == 'api_keys':
            result['api_keys'] = [key['name'] for key in self.vault_data['api_keys'].values()]
        
        return result
    
    def search_credentials(self, query: str, category: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search credentials by name, description, or tags."""
        results = {}
        query_lower = query.lower()
        
        categories = [category] if category else ['credentials', 'secrets', 'web_credentials', 'api_keys']
        
        for cat in categories:
            if cat in self.vault_data:
                results[cat] = []
                for item_id, item in self.vault_data[cat].items():
                    # Search in name, description, and tags
                    if (query_lower in item['name'].lower() or
                        query_lower in item.get('description', '').lower() or
                        any(query_lower in tag.lower() for tag in item.get('tags', []))):
                        
                        # Create safe copy without sensitive data
                        safe_item = {
                            'id': item_id,
                            'name': item['name'],
                            'description': item.get('description', ''),
                            'tags': item.get('tags', []),
                            'created': item['created'],
                            'last_used': item.get('last_used'),
                            'usage_count': item.get('usage_count', 0)
                        }
                        
                        # Add category-specific fields
                        if cat == 'credentials':
                            safe_item['username'] = item['username']
                        elif cat == 'web_credentials':
                            safe_item['url'] = item['url']
                            safe_item['username'] = item['username']
                        
                        results[cat].append(safe_item)
        
        return results
    
    def delete_credential(self, name: str, category: str = 'credentials') -> bool:
        """Delete a credential from the vault."""
        try:
            if category not in self.vault_data:
                print(f"❌ Invalid category: {category}")
                return False
            
            # Find and remove the credential
            for item_id, item in list(self.vault_data[category].items()):
                if item['name'] == name:
                    del self.vault_data[category][item_id]
                    
                    if self._save_vault():
                        print(f"✅ {category[:-1].title()} '{name}' deleted from vault")
                        return True
                    else:
                        return False
            
            print(f"❌ {category[:-1].title()} '{name}' not found")
            return False
            
        except Exception as e:
            print(f"❌ Failed to delete {category[:-1]}: {e}")
            return False
    
    def update_credential(self, name: str, updates: Dict[str, Any], 
                         category: str = 'credentials') -> bool:
        """Update a credential in the vault."""
        try:
            if category not in self.vault_data:
                print(f"❌ Invalid category: {category}")
                return False
            
            # Find the credential
            for item_id, item in self.vault_data[category].items():
                if item['name'] == name:
                    # Update fields
                    for key, value in updates.items():
                        if key in item and key not in ['id', 'created']:
                            item[key] = value
                    
                    # Re-encrypt if sensitive fields changed
                    if category == 'credentials' and 'password' in updates:
                        self._encrypt_credential_fields(item_id)
                    elif category == 'secrets' and 'value' in updates:
                        self._encrypt_secret_fields(item_id)
                    elif category == 'web_credentials' and 'password' in updates:
                        self._encrypt_web_credential_fields(item_id)
                    elif category == 'api_keys' and 'key' in updates:
                        self._encrypt_api_key_fields(item_id)
                    
                    if self._save_vault():
                        print(f"✅ {category[:-1].title()} '{name}' updated in vault")
                        return True
                    else:
                        return False
            
            print(f"❌ {category[:-1].title()} '{name}' not found")
            return False
            
        except Exception as e:
            print(f"❌ Failed to update {category[:-1]}: {e}")
            return False
    
    def export_vault(self, export_path: str, include_sensitive: bool = False) -> bool:
        """Export vault data (optionally including sensitive information)."""
        try:
            export_data = self.vault_data.copy()
            
            if not include_sensitive:
                # Remove sensitive fields
                for category in ['credentials', 'secrets', 'web_credentials', 'api_keys']:
                    if category in export_data:
                        for item_id in export_data[category]:
                            item = export_data[category][item_id]
                            if 'password' in item:
                                item['password'] = '[ENCRYPTED]'
                            if 'value' in item:
                                item['value'] = '[ENCRYPTED]'
                            if 'key' in item:
                                item['key'] = '[ENCRYPTED]'
            
            # Write export file
            export_file = Path(export_path)
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Vault exported to: {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export vault: {e}")
            return False
    
    def import_vault(self, import_path: str, merge: bool = True) -> bool:
        """Import vault data from file."""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                print(f"❌ Import file not found: {import_path}")
                return False
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            if merge:
                # Merge with existing data
                for category in import_data:
                    if category in self.vault_data:
                        self.vault_data[category].update(import_data[category])
                    else:
                        self.vault_data[category] = import_data[category]
            else:
                # Replace existing data
                self.vault_data = import_data
            
            # Re-encrypt all sensitive fields
            self._re_encrypt_all_fields()
            
            if self._save_vault():
                print(f"✅ Vault imported from: {import_path}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to import vault: {e}")
            return False
    
    def _re_encrypt_all_fields(self):
        """Re-encrypt all sensitive fields after import."""
        if not self.cipher:
            return
        
        try:
            # Re-encrypt credentials
            for credential_id in self.vault_data['credentials']:
                self._encrypt_credential_fields(credential_id)
            
            # Re-encrypt secrets
            for secret_id in self.vault_data['secrets']:
                self._encrypt_secret_fields(secret_id)
            
            # Re-encrypt web credentials
            for credential_id in self.vault_data['web_credentials']:
                self._encrypt_web_credential_fields(credential_id)
            
            # Re-encrypt API keys
            for key_id in self.vault_data['api_keys']:
                self._encrypt_api_key_fields(key_id)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not re-encrypt all fields: {e}")
    
    def get_vault_info(self) -> Dict[str, Any]:
        """Get information about the vault."""
        return {
            'file_path': str(self.vault_file),
            'exists': self.vault_file.exists(),
            'encryption_enabled': self.cipher is not None,
            'password_hash_provided': bool(self.password_hash),
            'host_fingerprint': self._get_host_fingerprint(),
            'total_entries': sum(len(self.vault_data.get(cat, {})) for cat in ['credentials', 'secrets', 'web_credentials', 'api_keys']),
            'categories': {
                'credentials': len(self.vault_data.get('credentials', {})),
                'secrets': len(self.vault_data.get('secrets', {})),
                'web_credentials': len(self.vault_data.get('web_credentials', {})),
                'api_keys': len(self.vault_data.get('api_keys', {}))
            },
            'metadata': self.vault_data.get('metadata', {})
        }

def main():
    """Command line interface for credential vault management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage credential vault")
    parser.add_argument('--info', action='store_true', help='Show vault information')
    parser.add_argument('--list', metavar='CATEGORY', help='List credentials in category (credentials, secrets, web_credentials, api_keys)')
    parser.add_argument('--search', metavar='QUERY', help='Search credentials')
    parser.add_argument('--add-credential', nargs=4, metavar=('NAME', 'USERNAME', 'PASSWORD', 'DESCRIPTION'), help='Add new credential')
    parser.add_argument('--add-secret', nargs=3, metavar=('NAME', 'VALUE', 'DESCRIPTION'), help='Add new secret')
    parser.add_argument('--add-web-credential', nargs=5, metavar=('NAME', 'URL', 'USERNAME', 'PASSWORD', 'DESCRIPTION'), help='Add new web credential')
    parser.add_argument('--add-api-key', nargs=3, metavar=('NAME', 'KEY', 'DESCRIPTION'), help='Add new API key')
    parser.add_argument('--get', nargs=2, metavar=('CATEGORY', 'NAME'), help='Get credential by name')
    parser.add_argument('--delete', nargs=2, metavar=('CATEGORY', 'NAME'), help='Delete credential by name')
    parser.add_argument('--export', metavar='PATH', help='Export vault to file')
    parser.add_argument('--import', metavar='PATH', dest='import_path', help='Import vault from file')
    
    args = parser.parse_args()
    
    # Check if encryption is enabled
    if not os.getenv('ENCRYPTION_ENABLED', 'true').lower() == 'true':
        print("⚠️  Encryption is not enabled")
        print("   Set ENCRYPTION_ENABLED=true to enable")
        return
    
    # Check if password hash is provided
    if not os.getenv('ENCRYPTION_PASSWORD_HASH'):
        print("⚠️  No encryption password hash provided")
        print("   Set ENCRYPTION_PASSWORD_HASH to enable vault")
        return
    
    vault = CredentialVault()
    
    if args.info:
        info = vault.get_vault_info()
        print("🔐 Credential Vault Information:")
        print(f"   File: {info['file_path']}")
        print(f"   Exists: {info['exists']}")
        print(f"   Encryption: {'✅ Enabled' if info['encryption_enabled'] else '❌ Disabled'}")
        print(f"   Password Hash: {'✅ Provided' if info['password_hash_provided'] else '❌ Missing'}")
        print(f"   Host Fingerprint: {info['host_fingerprint'][:16]}...")
        print(f"   Total Entries: {info['total_entries']}")
        print(f"   Categories:")
        for cat, count in info['categories'].items():
            print(f"     {cat}: {count}")
    
    elif args.list:
        credentials = vault.list_credentials(args.list)
        if credentials:
            print(f"📋 {args.list.title()} in vault:")
            for category, names in credentials.items():
                if names:
                    print(f"   {category}:")
                    for name in names:
                        print(f"     • {name}")
        else:
            print(f"ℹ️  No {args.list} found in vault")
    
    elif args.search:
        results = vault.search_credentials(args.search)
        if any(results.values()):
            print(f"🔍 Search results for '{args.search}':")
            for category, items in results.items():
                if items:
                    print(f"   {category}:")
                    for item in items:
                        print(f"     • {item['name']}: {item.get('description', 'No description')}")
        else:
            print(f"ℹ️  No results found for '{args.search}'")
    
    elif args.add_credential:
        name, username, password, description = args.add_credential
        vault.add_credential(name, username, password, description)
    
    elif args.add_secret:
        name, value, description = args.add_secret
        vault.add_secret(name, value, description)
    
    elif args.add_web_credential:
        name, url, username, password, description = args.add_web_credential
        vault.add_web_credential(name, url, username, password, description)
    
    elif args.add_api_key:
        name, key, description = args.add_api_key
        vault.add_api_key(name, key, description)
    
    elif args.get:
        category, name = args.get
        if category == 'credentials':
            result = vault.get_credential(name)
        elif category == 'secrets':
            result = vault.get_secret(name)
        elif category == 'web_credentials':
            result = vault.get_web_credential(name)
        elif category == 'api_keys':
            result = vault.get_api_key(name)
        else:
            print(f"❌ Invalid category: {category}")
            return
        
        if result:
            print(f"📋 {category[:-1].title()} '{name}':")
            for key, value in result.items():
                if key not in ['id']:
                    print(f"   {key}: {value}")
        else:
            print(f"❌ {category[:-1].title()} '{name}' not found")
    
    elif args.delete:
        category, name = args.delete
        vault.delete_credential(name, category)
    
    elif args.export:
        vault.export_vault(args.export)
    
    elif args.import_path:
        vault.import_vault(args.import_path)
    
    else:
        # Default: show info
        info = vault.get_vault_info()
        print("🔐 Credential Vault Status:")
        print(f"   Status: {'✅ Active' if info['encryption_enabled'] else '❌ Inactive'}")
        print(f"   Total Entries: {info['total_entries']}")
        print(f"   Host: {info['host_fingerprint'][:16]}...")
        print("\n💡 Use --info for detailed information")
        print("   Use --list CATEGORY to list credentials")
        print("   Use --search QUERY to search credentials")
        print("   Use --add-credential to add new credentials")

if __name__ == "__main__":
    main()
