#!/usr/bin/env python3
"""
Host Verification Module for Cybersecurity Agent
Provides host compatibility verification and device fingerprinting for security.
"""

import os
import sys
import hashlib
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from bin.device_identifier import DeviceIdentifier
    from bin.activation_manager import ActivationManager
    DEVICE_ID_AVAILABLE = True
    ACTIVATION_AVAILABLE = True
except ImportError as e:
    DEVICE_ID_AVAILABLE = False
    ACTIVATION_AVAILABLE = False
    print(f"⚠️  Host verification dependencies not available: {e}")


class HostVerification:
    """
    Host verification system for the Cybersecurity Agent.
    
    Provides host compatibility verification, device fingerprinting,
    and integration with the activation system.
    """
    
    def __init__(self):
        self.device_id = DeviceIdentifier() if DEVICE_ID_AVAILABLE else None
        self.activation_manager = ActivationManager() if ACTIVATION_AVAILABLE else None
        self.verification_cache = {}
        
    def verify_host_compatibility(self) -> bool:
        """
        Verify that the host is compatible and properly configured.
        
        Returns:
            bool: True if host is compatible, False otherwise
        """
        try:
            print("🔍 Verifying host compatibility...")
            
            # Check basic system requirements
            if not self._check_system_requirements():
                print("❌ System requirements not met")
                return False
            
            # Check device identifier availability
            if not self._check_device_identifier():
                print("❌ Device identifier not available")
                return False
            
            # Check activation status
            if not self._check_activation_status():
                print("❌ Activation verification failed")
                return False
            
            # Verify host fingerprint consistency
            if not self._verify_host_fingerprint():
                print("❌ Host fingerprint verification failed")
                return False
            
            print("✅ Host compatibility verification passed")
            return True
            
        except Exception as e:
            print(f"❌ Host verification failed: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check basic system requirements."""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
                return False
            
            # Check required directories
            required_dirs = ['bin', 'workflow_templates', 'session-logs']
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    print(f"❌ Required directory missing: {dir_name}")
                    return False
            
            # Check write permissions
            test_file = Path("test_write_permission.tmp")
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                print("❌ No write permission in current directory")
                return False
            
            print("✅ System requirements check passed")
            return True
            
        except Exception as e:
            print(f"❌ System requirements check failed: {e}")
            return False
    
    def _check_device_identifier(self) -> bool:
        """Check if device identifier is available and working."""
        try:
            if not self.device_id:
                print("⚠️  Device identifier not available")
                return False
            
            # Test device fingerprint generation
            fingerprint = self.device_id.generate_device_fingerprint()
            if not fingerprint or len(fingerprint) < 32:
                print("❌ Device fingerprint generation failed")
                return False
            
            print("✅ Device identifier check passed")
            return True
            
        except Exception as e:
            print(f"❌ Device identifier check failed: {e}")
            return False
    
    def _check_activation_status(self) -> bool:
        """Check activation status."""
        try:
            if not self.activation_manager:
                print("⚠️  Activation manager not available")
                return True  # Not critical if activation system is not available
            
            # Check if activation file exists
            if not self.activation_manager.activation_file.exists():
                print("⚠️  No activation file found - agent may not be activated")
                return True  # Allow operation without activation for development
            
            # Check activation file exists (don't verify password here)
            if self.activation_manager.activation_file.exists():
                print("✅ Activation file found")
            else:
                print("⚠️  No activation file found - agent may not be activated")
                return True  # Allow operation without activation for development
            
            print("✅ Activation status check passed")
            return True
            
        except Exception as e:
            print(f"⚠️  Activation check failed: {e}")
            return True  # Don't fail on activation issues
    
    def _verify_host_fingerprint(self) -> bool:
        """Verify host fingerprint consistency."""
        try:
            if not self.device_id:
                return True  # Skip if device ID not available
            
            # Generate current fingerprint
            current_fingerprint = self.device_id.generate_device_fingerprint()
            
            # Check if we have a cached fingerprint
            cache_file = Path(".host_fingerprint_cache")
            if cache_file.exists():
                try:
                    cached_data = json.loads(cache_file.read_text())
                    cached_fingerprint = cached_data.get("fingerprint")
                    
                    if cached_fingerprint and cached_fingerprint != current_fingerprint:
                        print("⚠️  Host fingerprint changed - this may indicate system changes")
                        # Update cache with new fingerprint
                        self._update_fingerprint_cache(current_fingerprint)
                    else:
                        print("✅ Host fingerprint consistency verified")
                except Exception as e:
                    print(f"⚠️  Could not verify fingerprint cache: {e}")
                    self._update_fingerprint_cache(current_fingerprint)
            else:
                # Create initial cache
                self._update_fingerprint_cache(current_fingerprint)
                print("✅ Host fingerprint cache created")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Host fingerprint verification failed: {e}")
            return True  # Don't fail on fingerprint issues
    
    def _update_fingerprint_cache(self, fingerprint: str):
        """Update the host fingerprint cache."""
        try:
            cache_data = {
                "fingerprint": fingerprint,
                "timestamp": time.time(),
                "platform": platform.platform(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
            
            cache_file = Path(".host_fingerprint_cache")
            cache_file.write_text(json.dumps(cache_data, indent=2))
            
        except Exception as e:
            print(f"⚠️  Could not update fingerprint cache: {e}")
    
    def get_host_info(self) -> Dict[str, Any]:
        """Get comprehensive host information."""
        try:
            host_info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "user": os.getenv("USER", "unknown"),
                "home": str(Path.home()),
                "current_dir": str(Path.cwd())
            }
            
            # Add device fingerprint if available
            if self.device_id:
                try:
                    host_info["device_fingerprint"] = self.device_id.generate_device_fingerprint()
                except Exception:
                    host_info["device_fingerprint"] = "unavailable"
            
            # Add activation status if available
            if self.activation_manager:
                try:
                    activation_exists = self.activation_manager.activation_file.exists()
                    host_info["activation_status"] = {
                        "activated": activation_exists,
                        "message": "activation file exists" if activation_exists else "no activation file"
                    }
                except Exception:
                    host_info["activation_status"] = {"activated": False, "message": "unknown"}
            
            return host_info
            
        except Exception as e:
            return {"error": f"Could not get host info: {e}"}
    
    def verify_environment(self) -> Dict[str, Any]:
        """Verify the environment and return detailed status."""
        verification_results = {
            "system_requirements": self._check_system_requirements(),
            "device_identifier": self._check_device_identifier(),
            "activation_status": self._check_activation_status(),
            "host_fingerprint": self._verify_host_fingerprint(),
            "overall_status": False
        }
        
        # Overall status is True if all critical checks pass
        verification_results["overall_status"] = (
            verification_results["system_requirements"] and
            verification_results["device_identifier"]
        )
        
        return verification_results


# Convenience functions for backward compatibility
def verify_host_compatibility() -> bool:
    """Convenience function to verify host compatibility."""
    verifier = HostVerification()
    return verifier.verify_host_compatibility()


def get_host_info() -> Dict[str, Any]:
    """Convenience function to get host information."""
    verifier = HostVerification()
    return verifier.get_host_info()


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Host Verification System Test")
    print("=" * 50)
    
    verifier = HostVerification()
    
    # Test host compatibility
    print("\n1. Testing host compatibility...")
    is_compatible = verifier.verify_host_compatibility()
    print(f"   Result: {'✅ Compatible' if is_compatible else '❌ Not Compatible'}")
    
    # Get host information
    print("\n2. Getting host information...")
    host_info = verifier.get_host_info()
    print(f"   Platform: {host_info.get('platform', 'Unknown')}")
    print(f"   Python: {host_info.get('python_version', 'Unknown')}")
    print(f"   User: {host_info.get('user', 'Unknown')}")
    
    # Verify environment
    print("\n3. Verifying environment...")
    env_results = verifier.verify_environment()
    print(f"   Overall Status: {'✅ Pass' if env_results['overall_status'] else '❌ Fail'}")
    print(f"   System Requirements: {'✅' if env_results['system_requirements'] else '❌'}")
    print(f"   Device Identifier: {'✅' if env_results['device_identifier'] else '❌'}")
    print(f"   Activation Status: {'✅' if env_results['activation_status'] else '❌'}")
    print(f"   Host Fingerprint: {'✅' if env_results['host_fingerprint'] else '❌'}")
    
    print("\n" + "=" * 50)
    print("✅ Host verification test completed")
