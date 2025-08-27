#!/usr/bin/env python3
"""
Encryption Management Utility
Handles password changes and file re-encryption for the cybersecurity agent.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mcp_tools import EncryptionManager

class EncryptionUtility:
    """Utility for managing encryption across the system."""
    
    def __init__(self):
        self.encryption_enabled = os.getenv('ENCRYPTION_ENABLED', 'false').lower() == 'true'
        self.current_password_hash = os.getenv('ENCRYPTION_PASSWORD_HASH', '')
        self.current_salt = os.getenv('ENCRYPTION_SALT', 'cybersecurity_agent_salt')
        
        # Default password hash for 'Vosteen2025' if none provided
        if not self.current_password_hash:
            default_password = ''  # No default password for security
            self.current_password_hash = hashlib.sha256(default_password.encode()).hexdigest()
    
    def change_password(self, new_password: str, re_encrypt_files: bool = True) -> bool:
        """Change the encryption password and optionally re-encrypt files."""
        try:
            print(f"🔐 Changing encryption password...")
            
            # Create old and new encryption managers
            old_manager = EncryptionManager(self.current_password_hash, self.current_salt)
            new_password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            new_manager = EncryptionManager(new_password_hash, self.current_salt)
            
            if re_encrypt_files:
                print("🔄 Re-encrypting files with new password...")
                
                # Find all encrypted files
                encrypted_files = self._find_encrypted_files()
                
                if not encrypted_files:
                    print("ℹ️  No encrypted files found to re-encrypt.")
                else:
                    print(f"📁 Found {len(encrypted_files)} encrypted files to re-encrypt...")
                    
                    for file_path in encrypted_files:
                        try:
                            print(f"  🔓 Decrypting: {file_path.name}")
                            decrypted_path = old_manager.decrypt_file(file_path)
                            
                            print(f"  🔒 Re-encrypting: {file_path.name}")
                            new_encrypted_path = new_manager.encrypt_file(decrypted_path)
                            
                            # Remove old encrypted file and decrypted temp file
                            file_path.unlink()
                            decrypted_path.unlink()
                            
                            # Rename new encrypted file to original name
                            new_encrypted_path.rename(file_path)
                            
                            print(f"  ✅ Successfully re-encrypted: {file_path.name}")
                            
                        except Exception as e:
                            print(f"  ❌ Failed to re-encrypt {file_path.name}: {e}")
            
            # Update environment variables
            self._update_environment(new_password_hash)
            
            print("✅ Password change completed successfully!")
            print(f"🔑 New password hash: {new_password_hash}")
            print("\n📝 To apply the new password, restart your terminal or run:")
            print(f"   export ENCRYPTION_PASSWORD_HASH='{new_password_hash}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Password change failed: {e}")
            return False
    
    def _find_encrypted_files(self) -> List[Path]:
        """Find all encrypted files in the project."""
        encrypted_files = []
        
        # Search in common directories
        search_dirs = [
            Path("knowledge-objects"),
            Path("session-outputs"),
            Path("session-logs"),
            Path(".")
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for file_path in search_dir.rglob("*.encrypted"):
                    encrypted_files.append(file_path)
        
        return encrypted_files
    
    def _update_environment(self, new_password_hash: str):
        """Update environment configuration."""
        env_file = Path(".env")
        env_example = Path("documentation/env_config.txt")
        
        if env_file.exists():
            # Update existing .env file
            self._update_env_file(env_file, new_password_hash)
        elif env_example.exists():
            # Create .env from example
            self._create_env_file(env_example, env_file, new_password_hash)
        else:
            # Create basic .env file
            self._create_basic_env_file(env_file, new_password_hash)
    
    def _update_env_file(self, env_file: Path, new_password_hash: str):
        """Update existing .env file with new password hash."""
        try:
            content = env_file.read_text()
            
            # Replace password hash line
            lines = content.split('\n')
            updated_lines = []
            
            for line in lines:
                if line.startswith('ENCRYPTION_PASSWORD_HASH='):
                    updated_lines.append(f'ENCRYPTION_PASSWORD_HASH={new_password_hash}')
                else:
                    updated_lines.append(line)
            
            # Write updated content
            env_file.write_text('\n'.join(updated_lines))
            print("📝 Updated .env file with new password hash")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not update .env file: {e}")
    
    def _create_env_file(self, example_file: Path, env_file: Path, new_password_hash: str):
        """Create .env file from example with new password hash."""
        try:
            content = example_file.read_text()
            content = content.replace('ENCRYPTION_PASSWORD_HASH=', f'ENCRYPTION_PASSWORD_HASH={new_password_hash}')
            content = content.replace('ENCRYPTION_ENABLED=false', 'ENCRYPTION_ENABLED=true')
            
            env_file.write_text(content)
            print("📝 Created .env file from example with new password hash")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create .env file: {e}")
    
    def _create_basic_env_file(self, env_file: Path, new_password_hash: str):
        """Create basic .env file with new password hash."""
        try:
            content = f"""# Encryption Configuration
ENCRYPTION_ENABLED=true
ENCRYPTION_PASSWORD_HASH={new_password_hash}
ENCRYPTION_SALT={self.current_salt.decode() if isinstance(self.current_salt, bytes) else self.current_salt}

# Session Management
SESSION_LOGS_DIR=./session-logs
SESSION_OUTPUTS_DIR=./session-outputs

# Knowledge Base
KNOWLEDGE_BASE_DIR=./knowledge-objects
MASTER_CATALOG_FILE=master_catalog.db
"""
            env_file.write_text(content)
            print("📝 Created basic .env file with new password hash")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create .env file: {e}")
    
    def list_encrypted_files(self):
        """List all encrypted files in the system."""
        encrypted_files = self._find_encrypted_files()
        
        if not encrypted_files:
            print("ℹ️  No encrypted files found in the system.")
            return
        
        print(f"🔐 Found {len(encrypted_files)} encrypted files:")
        for file_path in encrypted_files:
            file_size = file_path.stat().st_size
            print(f"  📁 {file_path} ({file_size} bytes)")
    
    def encrypt_file(self, file_path: str) -> bool:
        """Encrypt a specific file."""
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                return False
            
            if not self.encryption_enabled:
                print("⚠️  Encryption is not enabled. Enable it first with:")
                print("   export ENCRYPTION_ENABLED=true")
                return False
            
            manager = EncryptionManager(self.current_password)
            encrypted_path = manager.encrypt_file(path)
            
            print(f"✅ File encrypted: {encrypted_path}")
            return True
            
        except Exception as e:
            print(f"❌ Encryption failed: {e}")
            return False
    
    def decrypt_file(self, file_path: str) -> bool:
        """Decrypt a specific file."""
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                return False
            
            if not self.encryption_enabled:
                print("⚠️  Encryption is not enabled. Enable it first with:")
                print("   export ENCRYPTION_ENABLED=true")
                return False
            
            manager = EncryptionManager(self.current_password)
            decrypted_path = manager.decrypt_file(path)
            
            print(f"✅ File decrypted: {decrypted_path}")
            return True
            
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            return False

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Encryption Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Change password and re-encrypt all files
  python bin/encryption_manager.py --change-password "NewSecurePassword123"
  
  # List all encrypted files
  python bin/encryption_manager.py --list-files
  
  # Encrypt a specific file
  python bin/encryption_manager.py --encrypt-file sensitive_data.csv
  
  # Decrypt a specific file
  python bin/encryption_manager.py --decrypt-file sensitive_data.csv.encrypted
        """
    )
    
    parser.add_argument(
        '--change-password',
        type=str,
        help='Change encryption password and re-encrypt files'
    )
    
    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List all encrypted files in the system'
    )
    
    parser.add_argument(
        '--encrypt-file',
        type=str,
        help='Encrypt a specific file'
    )
    
    parser.add_argument(
        '--decrypt-file',
        type=str,
        help='Decrypt a specific file'
    )
    
    parser.add_argument(
        '--no-re-encrypt',
        action='store_true',
        help='Skip re-encrypting files when changing password'
    )
    
    args = parser.parse_args()
    
    utility = EncryptionUtility()
    
    if args.change_password:
        re_encrypt = not args.no_re_encrypt
        success = utility.change_password(args.change_password, re_encrypt)
        sys.exit(0 if success else 1)
    
    elif args.list_files:
        utility.list_encrypted_files()
    
    elif args.encrypt_file:
        success = utility.encrypt_file(args.encrypt_file)
        sys.exit(0 if success else 1)
    
    elif args.decrypt_file:
        success = utility.decrypt_file(args.decrypt_file)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        print("\n🔐 Current encryption status:")
        print(f"   Enabled: {utility.encryption_enabled}")
        print(f"   Password Hash: {utility.current_password_hash[:16]}...")
        print(f"   Salt: {utility.current_salt[:16] if isinstance(utility.current_salt, bytes) else utility.current_salt[:16]}...")

if __name__ == "__main__":
    main()
