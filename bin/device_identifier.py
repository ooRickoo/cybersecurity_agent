#!/usr/bin/env python3
"""
Device Identifier for Cybersecurity Agent
Generates unique device fingerprints for encryption key binding.
"""

import hashlib
import platform
import subprocess
import os
import uuid
from typing import Dict, Any, Optional
import json

class DeviceIdentifier:
    """Generates unique device fingerprints for encryption key binding."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.device_attributes = self._get_device_attributes()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_device_attributes(self) -> Dict[str, Any]:
        """Get device-specific attributes for fingerprinting."""
        try:
            attributes = {}
            
            # Get MAC address (more stable than other identifiers)
            mac_address = self._get_mac_address()
            if mac_address:
                attributes['mac_address'] = mac_address
            
            # Get CPU information (more stable than other hardware details)
            cpu_info = self._get_cpu_info()
            if cpu_info:
                attributes['cpu_info'] = cpu_info
            
            # Get disk serial (if available and stable)
            disk_serial = self._get_disk_serial()
            if disk_serial:
                attributes['disk_serial'] = disk_serial
            
            # Get system UUID (if available)
            system_uuid = self._get_system_uuid()
            if system_uuid:
                attributes['system_uuid'] = system_uuid
            
            # Get hostname (stable identifier)
            hostname = platform.node()
            if hostname:
                attributes['hostname'] = hostname
            
            return attributes
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_mac_address(self) -> Optional[str]:
        """Get primary MAC address."""
        try:
            if platform.system() == "Darwin":  # macOS
                # Use networksetup to get primary MAC address
                result = subprocess.run(
                    ['networksetup', '-listallhardwareports'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'Ethernet Address:' in line:
                            mac = line.split(':')[1].strip()
                            if mac and len(mac) == 17:  # Valid MAC format
                                return mac
            elif platform.system() == "Linux":
                # Try to get MAC from /sys/class/net
                for interface in ['eth0', 'wlan0', 'en0', 'en1']:
                    mac_file = f"/sys/class/net/{interface}/address"
                    if os.path.exists(mac_file):
                        with open(mac_file, 'r') as f:
                            mac = f.read().strip()
                            if mac and len(mac) == 17:
                                return mac
            elif platform.system() == "Windows":
                # Use getmac command on Windows
                result = subprocess.run(
                    ['getmac', '/fo', 'csv', '/nh'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                mac = parts[1].strip().strip('"')
                                if mac and len(mac) == 17:
                                    return mac
            
            return None
            
        except Exception:
            return None
    
    def _get_cpu_info(self) -> Optional[str]:
        """Get CPU information for fingerprinting."""
        try:
            if platform.system() == "Darwin":  # macOS
                # Use sysctl to get CPU info
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    cpu_info = result.stdout.strip()
                    if cpu_info:
                        # Extract key parts for consistency
                        parts = cpu_info.split()
                        if len(parts) >= 3:
                            # Use vendor, family, and model for consistency
                            return f"{parts[0]}_{parts[1]}_{parts[2]}"
            elif platform.system() == "Linux":
                # Read from /proc/cpuinfo
                if os.path.exists('/proc/cpuinfo'):
                    with open('/proc/cpuinfo', 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if line.startswith('model name'):
                                cpu_info = line.split(':')[1].strip()
                                if cpu_info:
                                    parts = cpu_info.split()
                                    if len(parts) >= 3:
                                        return f"{parts[0]}_{parts[1]}_{parts[2]}"
            elif platform.system() == "Windows":
                # Use wmic for CPU info on Windows
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name', '/format:csv'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip() and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                cpu_info = parts[1].strip()
                                if cpu_info:
                                    parts = cpu_info.split()
                                    if len(parts) >= 3:
                                        return f"{parts[0]}_{parts[1]}_{parts[2]}"
            
            return None
            
        except Exception:
            return None
    
    def _get_disk_serial(self) -> Optional[str]:
        """Get disk serial number if available."""
        try:
            if platform.system() == "Darwin":  # macOS
                # Use diskutil to get disk info
                result = subprocess.run(
                    ['diskutil', 'info', '/dev/disk0'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Serial Number:' in line:
                            serial = line.split(':')[1].strip()
                            if serial:
                                return serial
            elif platform.system() == "Linux":
                # Try to get disk serial from /sys/block
                for disk in ['sda', 'nvme0n1', 'hda']:
                    serial_file = f"/sys/block/{disk}/device/serial"
                    if os.path.exists(serial_file):
                        with open(serial_file, 'r') as f:
                            serial = f.read().strip()
                            if serial:
                                return serial
            elif platform.system() == "Windows":
                # Use wmic for disk info on Windows
                result = subprocess.run(
                    ['wmic', 'diskdrive', 'get', 'serialnumber', '/format:csv'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip() and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                serial = parts[1].strip()
                                if serial:
                                    return serial
            
            return None
            
        except Exception:
            return None
    
    def _get_system_uuid(self) -> Optional[str]:
        """Get system UUID if available."""
        try:
            if platform.system() == "Darwin":  # macOS
                # Use system_profiler to get hardware UUID
                result = subprocess.run(
                    ['system_profiler', 'SPHardwareDataType'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Hardware UUID:' in line:
                            uuid = line.split(':')[1].strip()
                            if uuid:
                                return uuid
            elif platform.system() == "Linux":
                # Try to read from /sys/class/dmi/id/product_uuid
                uuid_file = '/sys/class/dmi/id/product_uuid'
                if os.path.exists(uuid_file):
                    with open(uuid_file, 'r') as f:
                        uuid = f.read().strip()
                        if uuid:
                            return uuid
            elif platform.system() == "Windows":
                # Use wmic for system UUID on Windows
                result = subprocess.run(
                    ['wmic', 'csproduct', 'get', 'UUID', '/format:csv'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip() and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                uuid = parts[1].strip()
                                if uuid:
                                    return uuid
            
            return None
            
        except Exception:
            return None
    
    def generate_device_fingerprint(self) -> str:
        """
        Generate a device fingerprint using a balanced set of stable attributes.
        
        Returns:
            A 32-character hexadecimal device fingerprint
        """
        try:
            # Create a fingerprint from the most stable attributes
            fingerprint_data = []
            
            # Add system architecture (very stable)
            if 'architecture' in self.system_info:
                fingerprint_data.append(self.system_info['architecture'])
            
            # Add machine type (stable)
            if 'machine' in self.system_info:
                fingerprint_data.append(self.system_info['machine'])
            
            # Add CPU info (stable)
            if 'cpu_info' in self.device_attributes:
                fingerprint_data.append(self.device_attributes['cpu_info'])
            
            # Add hostname (stable)
            if 'hostname' in self.device_attributes:
                fingerprint_data.append(self.device_attributes['hostname'])
            
            # Add MAC address (stable)
            if 'mac_address' in self.device_attributes:
                fingerprint_data.append(self.device_attributes['mac_address'])
            
            # Add system UUID if available (very stable)
            if 'system_uuid' in self.device_attributes:
                fingerprint_data.append(self.device_attributes['system_uuid'])
            
            # If we don't have enough stable attributes, add some fallbacks
            if len(fingerprint_data) < 3:
                fingerprint_data.append(platform.system())
                fingerprint_data.append(platform.machine())
                fingerprint_data.append(str(uuid.getnode()))  # Fallback to MAC-based UUID
            
            # Create a consistent fingerprint
            fingerprint_string = '_'.join(filter(None, fingerprint_data))
            
            # Generate SHA-256 hash and take first 32 characters for consistency
            fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
            
            # Return first 32 characters for a balanced fingerprint
            return fingerprint_hash[:32]
            
        except Exception as e:
            # Fallback to a basic fingerprint
            fallback_data = f"{platform.system()}_{platform.machine()}_{platform.architecture()[0]}"
            fallback_hash = hashlib.sha256(fallback_data.encode()).hexdigest()
            return fallback_hash[:32]
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        try:
            return {
                'system_info': self.system_info,
                'device_attributes': self.device_attributes,
                'fingerprint': self.generate_device_fingerprint(),
                'fingerprint_length': len(self.generate_device_fingerprint()),
                'fingerprint_type': 'balanced_32char'
            }
        except Exception as e:
            return {
                'error': str(e),
                'fallback_fingerprint': hashlib.sha256(b"fallback").hexdigest()[:32]
            }
    
    def verify_device_consistency(self) -> Dict[str, Any]:
        """Verify device consistency across multiple fingerprint generations."""
        try:
            fingerprints = []
            for _ in range(5):
                fingerprints.append(self.generate_device_fingerprint())
            
            # Check if all fingerprints are identical
            is_consistent = len(set(fingerprints)) == 1
            base_fingerprint = fingerprints[0] if fingerprints else None
            
            return {
                'is_consistent': is_consistent,
                'base_fingerprint': base_fingerprint,
                'fingerprint_count': len(fingerprints),
                'unique_fingerprints': len(set(fingerprints)),
                'consistency_score': 1.0 if is_consistent else 0.0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_consistent': False,
                'consistency_score': 0.0
            }

if __name__ == "__main__":
    # Test the device identifier
    device_id = DeviceIdentifier()
    
    print("🔍 Device Identifier Test")
    print("=" * 50)
    
    # Generate fingerprint
    fingerprint = device_id.generate_device_fingerprint()
    print(f"📱 Device Fingerprint: {fingerprint}")
    print(f"📏 Fingerprint Length: {len(fingerprint)} characters")
    
    # Get device info
    device_info = device_id.get_device_info()
    print(f"\n💻 System Info:")
    for key, value in device_info['system_info'].items():
        print(f"   {key}: {value}")
    
    print(f"\n🔧 Device Attributes:")
    for key, value in device_info['device_attributes'].items():
        print(f"   {key}: {value}")
    
    # Test consistency
    consistency = device_id.verify_device_consistency()
    print(f"\n✅ Consistency Test:")
    print(f"   Consistent: {consistency['is_consistent']}")
    print(f"   Score: {consistency['consistency_score']:.2f}")
    
    if not consistency['is_consistent']:
        print("⚠️  Warning: Device fingerprint is not consistent across generations!")
        print("   This may cause encryption issues.")
