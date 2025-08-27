#!/usr/bin/env python3
"""
Secure Credential Interaction System
Ensures credentials are NEVER passed to LLMs or stored in agent memory.
"""

import os
import sys
import getpass
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class CredentialRequest:
    """Secure credential request that never stores sensitive data."""
    request_id: str
    credential_type: str  # 'username', 'password', 'api_key', 'secret'
    prompt: str
    required: bool = True
    validation_regex: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    description: str = ""
    
    def __post_init__(self):
        # Ensure no sensitive data is stored
        self._sensitive_data = None

class SecureCredentialCollector:
    """Collects credentials securely without storing them in memory."""
    
    def __init__(self):
        self.credential_vault = None
        self._initialize_vault()
    
    def _initialize_vault(self):
        """Initialize credential vault if available."""
        try:
            from bin.credential_vault import CredentialVault
            self.credential_vault = CredentialVault()
        except ImportError:
            print("⚠️  Credential vault not available")
    
    def collect_credential(self, request: CredentialRequest) -> Optional[str]:
        """Collect a single credential securely."""
        try:
            # Check if credential exists in vault
            if self.credential_vault and self.credential_vault.cipher:
                vault_result = self._search_vault_for_credential(request)
                if vault_result:
                    print(f"🔐 Found credential in vault: {vault_result['name']}")
                    use_vault = input("Use stored credential? (Y/n): ").strip().lower() != 'n'
                    if use_vault:
                        return vault_result.get('password') or vault_result.get('value') or vault_result.get('key')
            
            # Collect credential from user
            if request.credential_type == 'password':
                credential = getpass.getpass(f"{request.prompt}: ")
            else:
                credential = input(f"{request.prompt}: ").strip()
            
            # Validate credential
            if self._validate_credential(credential, request):
                # Offer to save to vault
                if self.credential_vault and self.credential_vault.cipher:
                    self._offer_to_save_credential(credential, request)
                
                return credential
            else:
                print("❌ Invalid credential format")
                return None
                
        except Exception as e:
            print(f"❌ Error collecting credential: {e}")
            return None
    
    def _search_vault_for_credential(self, request: CredentialRequest) -> Optional[Dict[str, Any]]:
        """Search vault for matching credentials."""
        try:
            if not self.credential_vault or not self.credential_vault.cipher:
                return None
            
            # Search based on prompt and description
            search_terms = [request.prompt, request.description]
            results = self.credential_vault.search_credentials(' '.join(search_terms))
            
            if results:
                # Return first match
                return results[0]
            
        except Exception as e:
            print(f"⚠️  Vault search error: {e}")
        
        return None
    
    def _validate_credential(self, credential: str, request: CredentialRequest) -> bool:
        """Validate credential format."""
        if not credential and request.required:
            return False
        
        if credential:
            # Length validation
            if request.min_length and len(credential) < request.min_length:
                print(f"❌ Credential too short (min: {request.min_length})")
                return False
            
            if request.max_length and len(credential) > request.max_length:
                print(f"❌ Credential too long (max: {request.max_length})")
                return False
            
            # Regex validation
            if request.validation_regex:
                import re
                if not re.match(request.validation_regex, credential):
                    print("❌ Credential format invalid")
                    return False
        
        return True
    
    def _offer_to_save_credential(self, credential: str, request: CredentialRequest):
        """Offer to save credential to vault."""
        try:
            save = input("Save credential to vault? (y/N): ").strip().lower()
            if save == 'y':
                name = input("Name for this credential: ").strip()
                description = input("Description (optional): ").strip()
                
                if request.credential_type == 'password':
                    username = input("Username (optional): ").strip()
                    self.credential_vault.add_credential(name, username, credential, description)
                elif request.credential_type == 'api_key':
                    self.credential_vault.add_api_key(name, credential, description)
                elif request.credential_type == 'secret':
                    self.credential_vault.add_secret(name, credential, description)
                
                print("✅ Credential saved to vault")
                
        except Exception as e:
            print(f"⚠️  Failed to save credential: {e}")
    
    def collect_credentials_batch(self, requests: List[CredentialRequest]) -> Dict[str, str]:
        """Collect multiple credentials securely."""
        collected_credentials = {}
        
        for request in requests:
            credential = self.collect_credential(request)
            if credential is not None:
                collected_credentials[request.request_id] = credential
            elif request.required:
                print(f"❌ Required credential '{request.request_id}' not provided")
                return {}
        
        return collected_credentials
    
    def get_credential_safely(self, credential_id: str, category: str = None) -> Optional[str]:
        """Get credential from vault safely (returns only the value, not metadata)."""
        try:
            if not self.credential_vault or not self.credential_vault.cipher:
                return None
            
            if category == 'credentials' or category is None:
                credential = self.credential_vault.get_credential(credential_id)
                return credential.get('password') if credential else None
            elif category == 'secrets':
                secret = self.credential_vault.get_secret(credential_id)
                return secret.get('value') if secret else None
            elif category == 'web_credentials':
                web_cred = self.credential_vault.get_web_credential(credential_id)
                return web_cred.get('password') if web_cred else None
            elif category == 'api_keys':
                api_key = self.credential_vault.get_api_key(credential_id)
                return api_key.get('key') if api_key else None
            
        except Exception as e:
            print(f"⚠️  Error retrieving credential: {e}")
        
        return None

class CredentialProtection:
    """Ensures credentials are never exposed to LLMs or agent memory."""
    
    def __init__(self):
        self.collector = SecureCredentialCollector()
        self._credential_cache = {}  # Temporary cache, cleared after use
        self._protection_enabled = True
    
    def enable_protection(self):
        """Enable credential protection."""
        self._protection_enabled = True
        print("🔒 Credential protection enabled")
    
    def disable_protection(self):
        """Disable credential protection (use with extreme caution)."""
        self._protection_enabled = False
        print("⚠️  Credential protection disabled - USE WITH EXTREME CAUTION")
    
    def collect_web_credentials(self, url: str, description: str = "") -> Dict[str, str]:
        """Collect web credentials securely."""
        requests = [
            CredentialRequest(
                request_id="username",
                credential_type="username",
                prompt=f"Username for {url}",
                description=description,
                required=True
            ),
            CredentialRequest(
                request_id="password",
                credential_type="password",
                prompt=f"Password for {url}",
                description=description,
                required=True,
                min_length=1
            )
        ]
        
        credentials = self.collector.collect_credentials_batch(requests)
        
        if credentials:
            # Store in temporary cache (not in agent memory)
            cache_key = f"web_{hashlib.md5(url.encode()).hexdigest()}"
            self._credential_cache[cache_key] = credentials.copy()
            
            # Clear sensitive data from credentials dict
            safe_credentials = {
                'username': credentials.get('username', ''),
                'password': '[PROTECTED]' if credentials.get('password') else '',
                'url': url,
                'description': description
            }
            
            return safe_credentials
        
        return {}
    
    def collect_api_credentials(self, service_name: str, description: str = "") -> Dict[str, str]:
        """Collect API credentials securely."""
        requests = [
            CredentialRequest(
                request_id="api_key",
                credential_type="api_key",
                prompt=f"API Key for {service_name}",
                description=description,
                required=True,
                min_length=10
            ),
            CredentialRequest(
                request_id="api_secret",
                credential_type="secret",
                prompt=f"API Secret for {service_name}",
                description=description,
                required=False,
                min_length=1
            )
        ]
        
        credentials = self.collector.collect_credentials_batch(requests)
        
        if credentials:
            # Store in temporary cache
            cache_key = f"api_{hashlib.md5(service_name.encode()).hexdigest()}"
            self._credential_cache[cache_key] = credentials.copy()
            
            # Return safe version
            safe_credentials = {
                'api_key': credentials.get('api_key', ''),
                'api_secret': '[PROTECTED]' if credentials.get('api_secret') else '',
                'service': service_name,
                'description': description
            }
            
            return safe_credentials
        
        return {}
    
    def collect_database_credentials(self, database_type: str, host: str, description: str = "") -> Dict[str, str]:
        """Collect database credentials securely."""
        requests = [
            CredentialRequest(
                request_id="username",
                credential_type="username",
                prompt=f"Database username for {database_type} on {host}",
                description=description,
                required=True
            ),
            CredentialRequest(
                request_id="password",
                credential_type="password",
                prompt=f"Database password for {database_type} on {host}",
                description=description,
                required=True,
                min_length=1
            ),
            CredentialRequest(
                request_id="database",
                credential_type="secret",
                prompt=f"Database name (optional)",
                description=description,
                required=False
            )
        ]
        
        credentials = self.collector.collect_credentials_batch(requests)
        
        if credentials:
            # Store in temporary cache
            cache_key = f"db_{hashlib.md5(f"{database_type}_{host}".encode()).hexdigest()}"
            self._credential_cache[cache_key] = credentials.copy()
            
            # Return safe version
            safe_credentials = {
                'username': credentials.get('username', ''),
                'password': '[PROTECTED]' if credentials.get('password') else '',
                'database': credentials.get('database', ''),
                'type': database_type,
                'host': host,
                'description': description
            }
            
            return safe_credentials
        
        return {}
    
    def get_cached_credential(self, cache_key: str, field: str) -> Optional[str]:
        """Get credential from temporary cache (cleared after use)."""
        if not self._protection_enabled:
            print("⚠️  Credential protection disabled - credentials may be exposed!")
            return None
        
        if cache_key in self._credential_cache:
            credential = self._credential_cache[cache_key].get(field)
            if credential:
                # Clear from cache after use
                del self._credential_cache[cache_key]
                return credential
        
        return None
    
    def clear_credential_cache(self):
        """Clear all cached credentials."""
        self._credential_cache.clear()
        print("🧹 Credential cache cleared")
    
    def get_credential_summary(self) -> Dict[str, Any]:
        """Get summary of collected credentials (no sensitive data)."""
        summary = {
            'protection_enabled': self._protection_enabled,
            'cached_credentials': len(self._credential_cache),
            'vault_available': self.collector.credential_vault is not None,
            'vault_status': 'Active' if self.collector.credential_vault and self.collector.credential_vault.cipher else 'Inactive'
        }
        
        return summary

def main():
    """Test the secure credential interaction system."""
    print("🔐 Secure Credential Interaction System")
    print("=" * 50)
    
    protection = CredentialProtection()
    
    # Test web credentials collection
    print("\n🌐 Testing web credentials collection...")
    web_creds = protection.collect_web_credentials("https://example.com", "Test website")
    print(f"Collected: {web_creds}")
    
    # Test API credentials collection
    print("\n🔑 Testing API credentials collection...")
    api_creds = protection.collect_api_credentials("TestService", "Test API service")
    print(f"Collected: {api_creds}")
    
    # Show protection status
    print("\n🛡️  Protection Status:")
    summary = protection.get_credential_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Clear cache
    protection.clear_credential_cache()

if __name__ == "__main__":
    main()

