#!/usr/bin/env python3
"""
Host Verification System
Verifies host compatibility and prompts for encrypted data reset when moving to new hosts.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class HostVerification:
    """Verifies host compatibility and manages encrypted data migration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.encrypted_files = []
        self.host_mismatch_detected = False
        
    def verify_host_compatibility(self) -> bool:
        """Verify that the current host is compatible with existing encrypted data."""
        try:
            from bin.device_identifier import DeviceIdentifier
            from bin.salt_manager import SaltManager
            
            device_id = DeviceIdentifier()
            salt_manager = SaltManager()
            
            current_host_fingerprint = device_id.generate_device_fingerprint()
            print(f"🔍 Current host fingerprint: {current_host_fingerprint[:16]}...")
            
            # Check device-bound salt compatibility
            device_salt_file = Path(".salt")
            if device_salt_file.exists():
                try:
                    # Get the host fingerprint from the salt file
                    salt_host_fingerprint = self._extract_host_fingerprint_from_salt()
                    
                    if salt_host_fingerprint and salt_host_fingerprint == current_host_fingerprint:
                        print("✅ Host compatibility verified")
                        return True
                    elif salt_host_fingerprint and salt_host_fingerprint != current_host_fingerprint:
                        print("⚠️  Host mismatch detected!")
                        print(f"   Salt host: {salt_host_fingerprint[:16]}...")
                        print(f"   Current host: {current_host_fingerprint[:16]}...")
                        
                        self.host_mismatch_detected = True
                        return self._handle_host_mismatch()
                    else:
                        print("ℹ️  Could not verify salt host fingerprint")
                        print("   Continuing with current host...")
                        return True
                        
                except Exception as e:
                    print(f"⚠️  Could not verify host compatibility: {e}")
                    return self._handle_host_mismatch()
            else:
                print("ℹ️  No device-bound salt found - first run on this host")
                return True
                
        except ImportError as e:
            print(f"⚠️  Could not import required modules: {e}")
            return True  # Allow continuation if modules not available
    
    def _extract_host_fingerprint_from_salt(self) -> Optional[str]:
        """Extract host fingerprint from existing salt file."""
        try:
            import sys
            from pathlib import Path
            
            # Add the bin directory to the path
            bin_path = Path(__file__).parent
            if str(bin_path) not in sys.path:
                sys.path.insert(0, str(bin_path))
            
            from salt_manager import SaltManager
            salt_manager = SaltManager()
            
            # Get the device fingerprint that was used to create the salt
            salt_info = salt_manager.get_salt_info()
            if salt_info and 'device_bound_salt' in salt_info:
                device_info = salt_info['device_bound_salt']
                # Check for stored device fingerprint first (new format)
                if device_info.get('stored_device_fingerprint'):
                    return device_info['stored_device_fingerprint']
                # Fallback to current device fingerprint
                elif device_info.get('device_fingerprint'):
                    return device_info['device_fingerprint']
                elif device_info.get('device_id'):
                    return device_info['device_id']
            
            return None
            
        except Exception as e:
            print(f"Debug: Error extracting host fingerprint: {e}")
            return None
    
    def _handle_host_mismatch(self) -> bool:
        """Handle host mismatch by prompting user for action."""
        print("\n🚨 HOST MISMATCH DETECTED")
        print("=" * 50)
        print("The encrypted data on this system was created on a different host.")
        print("This means:")
        print("  • Knowledge graph context memory cannot be decrypted")
        print("  • Credential vault cannot be accessed")
        print("  • All encrypted data is inaccessible")
        print("\nOptions:")
        print("  1. Reset encrypted data and start fresh on this host")
        print("  2. Exit and restore data from the original host")
        print("  3. Try to continue (may fail)")
        
        while True:
            try:
                choice = input("\nChoose option (1/2/3): ").strip()
                
                if choice == '1':
                    return self._reset_encrypted_data()
                elif choice == '2':
                    print("❌ Exiting - please restore data from original host")
                    return False
                elif choice == '3':
                    print("⚠️  Continuing with potentially inaccessible data...")
                    return True
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Exiting due to user interruption")
                return False
    
    def _reset_encrypted_data(self) -> bool:
        """Reset all encrypted data for the current host."""
        print("\n🔄 Resetting encrypted data for current host...")
        print("This will:")
        print("  • Remove all encrypted files")
        print("  • Clear knowledge graph context memory")
        print("  • Reset credential vault")
        print("  • Generate new host-bound encryption")
        
        try:
            confirmation = input("\nType 'RESET' to confirm: ").strip()
            if confirmation != 'RESET':
                print("❌ Reset cancelled")
                return False
            
            # Remove encrypted files
            encrypted_files_removed = self._remove_encrypted_files()
            
            # Remove salt files
            salt_files_removed = self._remove_salt_files()
            
            # Remove credential vault
            vault_removed = self._remove_credential_vault()
            
            # Remove knowledge graph files
            knowledge_files_removed = self._remove_knowledge_graph_files()
            
            print("\n✅ Encrypted data reset completed!")
            print(f"   Files removed: {encrypted_files_removed}")
            print(f"   Salt files removed: {salt_files_removed}")
            print(f"   Vault removed: {vault_removed}")
            print(f"   Knowledge files removed: {knowledge_files_removed}")
            
            print("\n🔄 System will now generate new host-bound encryption on next run")
            return True
            
        except Exception as e:
            print(f"❌ Error during reset: {e}")
            return False
    
    def _remove_encrypted_files(self) -> int:
        """Remove all encrypted files in the project."""
        removed_count = 0
        
        try:
            # Search for encrypted files
            for file_path in self.project_root.rglob("*.encrypted"):
                try:
                    file_path.unlink()
                    removed_count += 1
                    print(f"   🔓 Removed: {file_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {file_path}: {e}")
            
            return removed_count
            
        except Exception as e:
            print(f"   ⚠️  Error searching for encrypted files: {e}")
            return removed_count
    
    def _remove_salt_files(self) -> int:
        """Remove salt files to force regeneration."""
        removed_count = 0
        
        salt_files = ['.salt', '.session_salt']
        
        for salt_file in salt_files:
            salt_path = self.project_root / salt_file
            if salt_path.exists():
                try:
                    salt_path.unlink()
                    removed_count += 1
                    print(f"   🔓 Removed salt: {salt_file}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {salt_file}: {e}")
        
        return removed_count
    
    def _remove_credential_vault(self) -> bool:
        """Remove the credential vault file."""
        vault_path = self.project_root / 'etc' / 'credential_vault.db'
        
        if vault_path.exists():
            try:
                vault_path.unlink()
                print(f"   🔓 Removed credential vault")
                return True
            except Exception as e:
                print(f"   ⚠️  Could not remove credential vault: {e}")
                return False
        else:
            return True
    
    def _remove_knowledge_graph_files(self) -> int:
        """Remove knowledge graph context memory files."""
        removed_count = 0
        
        try:
            # Remove master catalog
            master_catalog = self.project_root / 'knowledge-objects' / 'master_catalog.db'
            if master_catalog.exists():
                try:
                    master_catalog.unlink()
                    removed_count += 1
                    print(f"   🔓 Removed: master_catalog.db")
                except Exception as e:
                    print(f"   ⚠️  Could not remove master_catalog.db: {e}")
            
            # Remove other knowledge files
            knowledge_dir = self.project_root / 'knowledge-objects'
            if knowledge_dir.exists():
                for file_path in knowledge_dir.glob('*.db'):
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"   🔓 Removed: {file_path.name}")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove {file_path.name}: {e}")
            
            return removed_count
            
        except Exception as e:
            print(f"   ⚠️  Error removing knowledge graph files: {e}")
            return removed_count
    
    def scan_encrypted_files(self) -> List[Path]:
        """Scan for all encrypted files in the project."""
        encrypted_files = []
        
        try:
            for file_path in self.project_root.rglob("*.encrypted"):
                encrypted_files.append(file_path)
            
            return encrypted_files
            
        except Exception as e:
            print(f"⚠️  Error scanning for encrypted files: {e}")
            return []
    
    def get_encryption_status(self) -> Dict[str, any]:
        """Get current encryption status and host information."""
        try:
            from bin.device_identifier import DeviceIdentifier
            from bin.salt_manager import SaltManager
            
            device_id = DeviceIdentifier()
            salt_manager = SaltManager()
            
            current_host = device_id.generate_device_fingerprint()
            
            # Check salt files
            device_salt_exists = Path(".salt").exists()
            session_salt_exists = Path(".session_salt").exists()
            
            # Check credential vault
            vault_exists = Path("etc/credential_vault.db").exists()
            
            # Check knowledge graph files
            knowledge_files = list(Path("knowledge-objects").glob("*.db")) if Path("knowledge-objects").exists() else []
            
            # Check encrypted files
            encrypted_files = self.scan_encrypted_files()
            
            return {
                'current_host': current_host,
                'current_host_preview': current_host[:16] + "...",
                'device_salt_exists': device_salt_exists,
                'session_salt_exists': session_salt_exists,
                'vault_exists': vault_exists,
                'knowledge_files_count': len(knowledge_files),
                'encrypted_files_count': len(encrypted_files),
                'encrypted_files': [str(f) for f in encrypted_files],
                'knowledge_files': [str(f) for f in knowledge_files]
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'current_host': 'unknown',
                'current_host_preview': 'unknown'
            }

def main():
    """Command line interface for host verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify host compatibility and manage encrypted data")
    parser.add_argument('--verify', action='store_true', help='Verify host compatibility')
    parser.add_argument('--status', action='store_true', help='Show encryption status')
    parser.add_argument('--scan', action='store_true', help='Scan for encrypted files')
    parser.add_argument('--reset', action='store_true', help='Reset encrypted data for current host')
    
    args = parser.parse_args()
    
    host_verification = HostVerification()
    
    if args.verify:
        print("🔍 Verifying host compatibility...")
        if host_verification.verify_host_compatibility():
            print("✅ Host verification completed successfully")
        else:
            print("❌ Host verification failed")
    
    elif args.status:
        status = host_verification.get_encryption_status()
        print("🔐 Encryption Status:")
        print(f"   Current Host: {status['current_host_preview']}")
        print(f"   Device Salt: {'✅ Exists' if status['device_salt_exists'] else '❌ Missing'}")
        print(f"   Session Salt: {'✅ Exists' if status['session_salt_exists'] else '❌ Missing'}")
        print(f"   Credential Vault: {'✅ Exists' if status['vault_exists'] else '❌ Missing'}")
        print(f"   Knowledge Files: {status['knowledge_files_count']}")
        print(f"   Encrypted Files: {status['encrypted_files_count']}")
        
        if status['encrypted_files_count'] > 0:
            print("\n   Encrypted Files:")
            for file_path in status['encrypted_files'][:10]:  # Show first 10
                print(f"     • {file_path}")
            if status['encrypted_files_count'] > 10:
                print(f"     ... and {status['encrypted_files_count'] - 10} more")
    
    elif args.scan:
        encrypted_files = host_verification.scan_encrypted_files()
        print(f"🔍 Found {len(encrypted_files)} encrypted files:")
        for file_path in encrypted_files:
            print(f"   • {file_path}")
    
    elif args.reset:
        print("⚠️  RESET ENCRYPTED DATA")
        print("This will remove ALL encrypted data and start fresh!")
        confirmation = input("Type 'RESET' to confirm: ").strip()
        if confirmation == 'RESET':
            if host_verification._reset_encrypted_data():
                print("✅ Reset completed successfully")
            else:
                print("❌ Reset failed")
        else:
            print("❌ Reset cancelled")
    
    else:
        # Default: verify host compatibility
        print("🔍 Verifying host compatibility...")
        if host_verification.verify_host_compatibility():
            print("✅ Host verification completed successfully")
        else:
            print("❌ Host verification failed")
        
        print("\n💡 Use --status to see encryption status")
        print("   Use --scan to find encrypted files")
        print("   Use --reset to reset encrypted data")

if __name__ == "__main__":
    main()
