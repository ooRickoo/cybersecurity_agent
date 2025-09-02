#!/usr/bin/env python3
"""
Standardized Credential Vault Manager
Ensures consistent credential vault usage across all components
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class StandardizedCredentialVault:
    """Standardized credential vault manager for consistent usage across all components."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            cls._instance = super(StandardizedCredentialVault, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the standardized credential vault."""
        if not self._initialized:
            self.vault = None
            self.initialization_error = None
            self._initialize_vault()
            StandardizedCredentialVault._initialized = True
    
    def _initialize_vault(self):
        """Initialize the credential vault with error handling."""
        try:
            from bin.credential_vault import initialize_vault_system
            self.vault = initialize_vault_system()
            logger.info("✅ Standardized credential vault initialized successfully")
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Failed to initialize standardized credential vault: {e}")
            self.vault = None
    
    def is_available(self) -> bool:
        """Check if credential vault is available."""
        return self.vault is not None and self.initialization_error is None
    
    def get_vault(self):
        """Get the credential vault instance."""
        if not self.is_available():
            logger.warning("⚠️  Credential vault not available")
            return None
        return self.vault
    
    def add_credential(self, name: str, username: str, password: str, 
                      description: str = "", tags: list = None) -> bool:
        """Add a credential to the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot add credential: vault not available")
            return False
        
        try:
            self.vault.add_credential(name, username, password, description, tags or [])
            logger.info(f"✅ Credential '{name}' added to vault")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add credential '{name}': {e}")
            return False
    
    def get_credential(self, name: str) -> Optional[Dict[str, str]]:
        """Get a credential from the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot get credential: vault not available")
            return None
        
        try:
            return self.vault.get_credential(name)
        except Exception as e:
            logger.error(f"❌ Failed to get credential '{name}': {e}")
            return None
    
    def add_api_key(self, name: str, api_key: str, 
                   description: str = "", tags: list = None) -> bool:
        """Add an API key to the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot add API key: vault not available")
            return False
        
        try:
            self.vault.add_api_key(name, api_key, description, tags or [])
            logger.info(f"✅ API key '{name}' added to vault")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add API key '{name}': {e}")
            return False
    
    def get_api_key(self, name: str) -> Optional[str]:
        """Get an API key from the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot get API key: vault not available")
            return None
        
        try:
            return self.vault.get_api_key(name)
        except Exception as e:
            logger.error(f"❌ Failed to get API key '{name}': {e}")
            return None
    
    def add_secret(self, name: str, secret: str, 
                  description: str = "", tags: list = None) -> bool:
        """Add a secret to the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot add secret: vault not available")
            return False
        
        try:
            self.vault.add_secret(name, secret, description, tags or [])
            logger.info(f"✅ Secret '{name}' added to vault")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add secret '{name}': {e}")
            return False
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret from the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot get secret: vault not available")
            return None
        
        try:
            return self.vault.get_secret(name)
        except Exception as e:
            logger.error(f"❌ Failed to get secret '{name}': {e}")
            return None
    
    def search_credentials(self, search_term: str) -> Dict[str, Any]:
        """Search for credentials in the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot search credentials: vault not available")
            return {}
        
        try:
            return self.vault.search_credentials(search_term)
        except Exception as e:
            logger.error(f"❌ Failed to search credentials: {e}")
            return {}
    
    def list_credentials(self, category: str = None) -> Dict[str, Any]:
        """List all credentials in the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot list credentials: vault not available")
            return {}
        
        try:
            return self.vault.list_credentials(category)
        except Exception as e:
            logger.error(f"❌ Failed to list credentials: {e}")
            return {}
    
    def delete_credential(self, name: str, credential_type: str = "credential") -> bool:
        """Delete a credential from the vault."""
        if not self.is_available():
            logger.warning("⚠️  Cannot delete credential: vault not available")
            return False
        
        try:
            self.vault.delete_credential(name, credential_type)
            logger.info(f"✅ Credential '{name}' deleted from vault")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete credential '{name}': {e}")
            return False
    
    def get_vault_status(self) -> Dict[str, Any]:
        """Get vault status information."""
        if not self.is_available():
            return {
                "available": False,
                "error": self.initialization_error,
                "vault_file": None,
                "encryption_enabled": False
            }
        
        try:
            status = self.vault.get_vault_status()
            status["available"] = True
            status["error"] = None
            return status
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "vault_file": None,
                "encryption_enabled": False
            }
    
    def save_password_hash(self, password_hash: str) -> bool:
        """Save password hash to file for persistence."""
        try:
            password_file = Path("etc/.password_hash")
            password_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(password_file, 'w') as f:
                f.write(password_hash)
            
            logger.info("✅ Password hash saved to file")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save password hash: {e}")
            return False
    
    def load_password_hash(self) -> Optional[str]:
        """Load password hash from file."""
        try:
            password_file = Path("etc/.password_hash")
            if password_file.exists():
                with open(password_file, 'r') as f:
                    password_hash = f.read().strip()
                    if password_hash:
                        logger.info("✅ Password hash loaded from file")
                        return password_hash
        except Exception as e:
            logger.error(f"❌ Failed to load password hash: {e}")
        
        return None

# Global instance
_global_standardized_vault = None

def get_standardized_credential_vault() -> StandardizedCredentialVault:
    """Get the global standardized credential vault instance."""
    global _global_standardized_vault
    if _global_standardized_vault is None:
        _global_standardized_vault = StandardizedCredentialVault()
    return _global_standardized_vault

def is_credential_vault_available() -> bool:
    """Check if credential vault is available."""
    vault = get_standardized_credential_vault()
    return vault.is_available()

def safe_get_credential(name: str) -> Optional[Dict[str, str]]:
    """Safely get a credential from the vault."""
    vault = get_standardized_credential_vault()
    return vault.get_credential(name)

def safe_get_api_key(name: str) -> Optional[str]:
    """Safely get an API key from the vault."""
    vault = get_standardized_credential_vault()
    return vault.get_api_key(name)

def safe_get_secret(name: str) -> Optional[str]:
    """Safely get a secret from the vault."""
    vault = get_standardized_credential_vault()
    return vault.get_secret(name)

def safe_add_credential(name: str, username: str, password: str, 
                       description: str = "", tags: list = None) -> bool:
    """Safely add a credential to the vault."""
    vault = get_standardized_credential_vault()
    return vault.add_credential(name, username, password, description, tags)

def safe_add_api_key(name: str, api_key: str, 
                    description: str = "", tags: list = None) -> bool:
    """Safely add an API key to the vault."""
    vault = get_standardized_credential_vault()
    return vault.add_api_key(name, api_key, description, tags)

def safe_add_secret(name: str, secret: str, 
                   description: str = "", tags: list = None) -> bool:
    """Safely add a secret to the vault."""
    vault = get_standardized_credential_vault()
    return vault.add_secret(name, secret, description, tags)
