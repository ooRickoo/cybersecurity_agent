#!/usr/bin/env python3
"""
Host Scanning Tools - Comprehensive Network Discovery and Security Analysis
Provides nmap-based host scanning capabilities for cybersecurity analysis and network management.

Features:
- Port scanning and service detection
- OS fingerprinting and version detection
- Vulnerability assessment
- Network topology mapping
- Security posture analysis
- Integration with MCP framework for dynamic workflows
"""

import subprocess
import json
import xml.etree.ElementTree as ET
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import ipaddress
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScanType(Enum):
    """Types of nmap scans available."""
    QUICK_SCAN = "quick"
    STEALTH_SCAN = "stealth"
    COMPREHENSIVE_SCAN = "comprehensive"
    VULNERABILITY_SCAN = "vulnerability"
    OS_DETECTION = "os_detection"
    SERVICE_DETECTION = "service_detection"
    TOPOLOGY_SCAN = "topology"
    CUSTOM_SCAN = "custom"

class ScanIntensity(Enum):
    """Scan intensity levels for timing and aggressiveness."""
    PARANOID = "T0"      # Very slow, stealthy
    SNEAKY = "T1"        # Slow, stealthy
    POLITE = "T2"        # Polite, default
    NORMAL = "T3"        # Normal speed
    AGGRESSIVE = "T4"    # Aggressive
    INSANE = "T5"        # Very aggressive

@dataclass
class HostInfo:
    """Information about a scanned host."""
    ip_address: str
    hostname: Optional[str] = None
    status: str = "unknown"
    os_info: Optional[Dict[str, Any]] = None
    ports: List[Dict[str, Any]] = None
    services: List[Dict[str, Any]] = None
    vulnerabilities: List[Dict[str, Any]] = None
    scan_time: Optional[float] = None
    mac_address: Optional[str] = None
    vendor: Optional[str] = None
    
    def __post_init__(self):
        if self.ports is None:
            self.ports = []
        if self.services is None:
            self.services = []
        if self.vulnerabilities is None:
            self.vulnerabilities = []

@dataclass
class ScanResult:
    """Result of a host scanning operation."""
    scan_id: str
    scan_type: ScanType
    target_hosts: List[str]
    scan_start_time: float
    scan_end_time: Optional[float] = None
    hosts: List[HostInfo] = None
    summary: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.hosts is None:
            self.hosts = []
        if self.summary is None:
            self.summary = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class NmapScanner:
    """Core nmap scanning functionality."""
    
    def __init__(self):
        self.nmap_path = self._find_nmap()
        if not self.nmap_path:
            raise RuntimeError("nmap not found. Please install nmap first.")
        
        self.scan_history: List[ScanResult] = []
        self.current_scans: Dict[str, subprocess.Popen] = {}
        
        logger.info(f"🚀 NmapScanner initialized with nmap at: {self.nmap_path}")
    
    def _find_nmap(self) -> Optional[str]:
        """Find nmap executable in system PATH."""
        try:
            result = subprocess.run(['which', 'nmap'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Try common locations
        common_paths = [
            '/usr/bin/nmap',
            '/usr/local/bin/nmap',
            '/opt/local/bin/nmap',
            'C:\\Program Files (x86)\\Nmap\\nmap.exe',
            'C:\\Program Files\\Nmap\\nmap.exe'
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _validate_target(self, target: str) -> bool:
        """Validate target IP address or hostname."""
        try:
            # Try to parse as IP address
            ipaddress.ip_address(target)
            return True
        except ValueError:
            # Check if it's a valid hostname
            if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', target):
                return True
            return False
    
    def _build_nmap_command(self, scan_type: ScanType, targets: List[str], 
                           intensity: ScanIntensity = ScanIntensity.NORMAL,
                           custom_options: Dict[str, Any] = None) -> List[str]:
        """Build nmap command with appropriate flags."""
        command = [self.nmap_path]
        
        # Add timing template
        command.extend(['-T', intensity.value[1:]])  # Remove 'T' prefix
        
        # Add scan type specific options
        if scan_type == ScanType.QUICK_SCAN:
            command.extend(['-F', '-sS'])  # Fast scan, SYN scan
        elif scan_type == ScanType.STEALTH_SCAN:
            command.extend(['-sS', '--min-rate', '100'])  # SYN scan, slow rate
        elif scan_type == ScanType.COMPREHENSIVE_SCAN:
            command.extend(['-sS', '-sV', '-O', '--version-intensity', '5'])
        elif scan_type == ScanType.VULNERABILITY_SCAN:
            command.extend(['-sS', '-sV', '--script', 'vuln'])
        elif scan_type == ScanType.OS_DETECTION:
            command.extend(['-O', '--osscan-guess'])
        elif scan_type == ScanType.SERVICE_DETECTION:
            command.extend(['-sV', '--version-intensity', '9'])
        elif scan_type == ScanType.TOPOLOGY_SCAN:
            command.extend(['-sn', '--traceroute'])
        
        # Add custom options if provided
        if custom_options:
            if custom_options.get('ports'):
                command.extend(['-p', custom_options['ports']])
            if custom_options.get('exclude'):
                command.extend(['--exclude', custom_options['exclude']])
            if custom_options.get('script'):
                command.extend(['--script', custom_options['script']])
            if custom_options.get('output_format'):
                command.extend(['-o' + custom_options['output_format']])
        
        # Add output format (XML for parsing)
        command.extend(['-oX', '-'])
        
        # Add targets
        command.extend(targets)
        
        return command
    
    def _parse_nmap_xml(self, xml_output: str) -> List[HostInfo]:
        """Parse nmap XML output into structured data."""
        try:
            root = ET.fromstring(xml_output)
            hosts = []
            
            for host_elem in root.findall('.//host'):
                host_info = self._parse_host_element(host_elem)
                if host_info:
                    hosts.append(host_info)
            
            return hosts
        except ET.ParseError as e:
            logger.error(f"Failed to parse nmap XML output: {e}")
            return []
    
    def _parse_host_element(self, host_elem) -> Optional[HostInfo]:
        """Parse individual host element from nmap XML."""
        try:
            # Get address information
            address_elem = host_elem.find('.//address')
            if address_elem is None:
                return None
            
            ip_address = address_elem.get('addr')
            if not ip_address:
                return None
            
            # Get hostname
            hostname = None
            hostname_elem = host_elem.find('.//hostname')
            if hostname_elem is not None:
                hostname = hostname_elem.get('name')
            
            # Get status
            status = "unknown"
            status_elem = host_elem.find('.//status')
            if status_elem is not None:
                status = status_elem.get('state', 'unknown')
            
            # Get OS information
            os_info = self._parse_os_info(host_elem)
            
            # Get ports and services
            ports = self._parse_ports(host_elem)
            services = self._parse_services(host_elem)
            
            # Get MAC address and vendor
            mac_address = None
            vendor = None
            for addr_elem in host_elem.findall('.//address'):
                if addr_elem.get('addrtype') == 'mac':
                    mac_address = addr_elem.get('addr')
                    vendor = addr_elem.get('vendor')
                    break
            
            return HostInfo(
                ip_address=ip_address,
                hostname=hostname,
                status=status,
                os_info=os_info,
                ports=ports,
                services=services,
                mac_address=mac_address,
                vendor=vendor
            )
        
        except Exception as e:
            logger.error(f"Failed to parse host element: {e}")
            return None
    
    def _parse_os_info(self, host_elem) -> Optional[Dict[str, Any]]:
        """Parse OS information from host element."""
        os_elem = host_elem.find('.//os')
        if os_elem is None:
            return None
        
        os_info = {}
        
        # Get OS match
        os_match = os_elem.find('.//osmatch')
        if os_match is not None:
            os_info['name'] = os_match.get('name', '')
            os_info['accuracy'] = os_match.get('accuracy', '')
            os_info['line'] = os_match.get('line', '')
        
        # Get OS details
        os_details = []
        for detail in os_elem.findall('.//osdetail'):
            os_details.append(detail.text)
        if os_details:
            os_info['details'] = os_details
        
        return os_info if os_info else None
    
    def _parse_ports(self, host_elem) -> List[Dict[str, Any]]:
        """Parse port information from host element."""
        ports = []
        
        for port_elem in host_elem.findall('.//port'):
            port_info = {
                'port': port_elem.get('portid', ''),
                'protocol': port_elem.get('protocol', ''),
                'state': port_elem.get('state', ''),
                'service': port_elem.get('name', ''),
                'version': port_elem.get('version', ''),
                'product': port_elem.get('product', ''),
                'extrainfo': port_elem.get('extrainfo', '')
            }
            ports.append(port_info)
        
        return ports
    
    def _parse_services(self, host_elem) -> List[Dict[str, Any]]:
        """Parse service information from host element."""
        services = []
        
        for port_elem in host_elem.findall('.//port'):
            service_elem = port_elem.find('.//service')
            if service_elem is not None:
                service_info = {
                    'port': port_elem.get('portid', ''),
                    'name': service_elem.get('name', ''),
                    'product': service_elem.get('product', ''),
                    'version': service_elem.get('version', ''),
                    'extrainfo': service_elem.get('extrainfo', ''),
                    'ostype': service_elem.get('ostype', ''),
                    'method': service_elem.get('method', ''),
                    'conf': service_elem.get('conf', '')
                }
                services.append(service_info)
        
        return services
    
    async def scan_hosts(self, targets: List[str], scan_type: ScanType = ScanType.QUICK_SCAN,
                        intensity: ScanIntensity = ScanIntensity.NORMAL,
                        custom_options: Dict[str, Any] = None) -> ScanResult:
        """Perform nmap scan on target hosts."""
        # Validate targets
        valid_targets = [t for t in targets if self._validate_target(t)]
        if not valid_targets:
            raise ValueError("No valid targets provided")
        
        # Create scan result
        scan_id = f"scan_{int(time.time())}_{len(self.scan_history)}"
        scan_result = ScanResult(
            scan_id=scan_id,
            scan_type=scan_type,
            target_hosts=valid_targets,
            scan_start_time=time.time()
        )
        
        try:
            # Build nmap command
            command = self._build_nmap_command(scan_type, valid_targets, intensity, custom_options)
            logger.info(f"Starting scan {scan_id}: {' '.join(command)}")
            
            # Execute nmap
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process reference
            self.current_scans[scan_id] = process
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            # Remove from current scans
            del self.current_scans[scan_id]
            
            # Check for errors
            if process.returncode != 0:
                error_msg = f"nmap scan failed with return code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr}"
                scan_result.errors.append(error_msg)
                logger.error(error_msg)
                return scan_result
            
            # Parse results
            if stdout:
                hosts = self._parse_nmap_xml(stdout)
                scan_result.hosts = hosts
                
                # Generate summary
                scan_result.summary = self._generate_scan_summary(hosts)
                
                logger.info(f"Scan {scan_id} completed successfully. Found {len(hosts)} hosts.")
            else:
                scan_result.warnings.append("No output from nmap scan")
                logger.warning("No output from nmap scan")
        
        except Exception as e:
            error_msg = f"Scan {scan_id} failed: {str(e)}"
            scan_result.errors.append(error_msg)
            logger.error(error_msg)
        
        finally:
            scan_result.scan_end_time = time.time()
            self.scan_history.append(scan_result)
        
        return scan_result
    
    def _generate_scan_summary(self, hosts: List[HostInfo]) -> Dict[str, Any]:
        """Generate summary statistics from scan results."""
        if not hosts:
            return {}
        
        total_hosts = len(hosts)
        up_hosts = len([h for h in hosts if h.status == 'up'])
        down_hosts = total_hosts - up_hosts
        
        # Port statistics
        all_ports = []
        for host in hosts:
            all_ports.extend(host.ports)
        
        open_ports = len([p for p in all_ports if p.get('state') == 'open'])
        closed_ports = len([p for p in all_ports if p.get('state') == 'closed'])
        
        # Service statistics
        services_found = set()
        for host in hosts:
            for service in host.services:
                if service.get('name'):
                    services_found.add(service['name'])
        
        # OS detection statistics
        os_detected = len([h for h in hosts if h.os_info])
        
        return {
            'total_hosts': total_hosts,
            'up_hosts': up_hosts,
            'down_hosts': down_hosts,
            'total_ports_scanned': len(all_ports),
            'open_ports': open_ports,
            'closed_ports': closed_ports,
            'services_found': len(services_found),
            'os_detected': os_detected,
            'scan_duration': time.time() - self.scan_history[-1].scan_start_time if self.scan_history else 0
        }
    
    def get_scan_history(self) -> List[ScanResult]:
        """Get list of completed scans."""
        return self.scan_history.copy()
    
    def get_scan_by_id(self, scan_id: str) -> Optional[ScanResult]:
        """Get specific scan result by ID."""
        for scan in self.scan_history:
            if scan.scan_id == scan_id:
                return scan
        return None
    
    def cancel_scan(self, scan_id: str) -> bool:
        """Cancel a running scan."""
        if scan_id in self.current_scans:
            process = self.current_scans[scan_id]
            process.terminate()
            del self.current_scans[scan_id]
            logger.info(f"Scan {scan_id} cancelled")
            return True
        return False
    
    def get_running_scans(self) -> List[str]:
        """Get list of currently running scan IDs."""
        return list(self.current_scans.keys())

class HostScanningManager:
    """Manager for host scanning operations with MCP integration capabilities."""
    
    def __init__(self):
        self.scanner = NmapScanner()
        self.scan_templates = self._create_scan_templates()
        self.performance_stats = {
            'total_scans': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'average_scan_time': 0.0
        }
        
        logger.info("🚀 Host Scanning Manager initialized")
    
    def _create_scan_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined scan templates."""
        return {
            'quick_audit': {
                'scan_type': ScanType.QUICK_SCAN,
                'intensity': ScanIntensity.NORMAL,
                'description': 'Quick port scan for basic network audit'
            },
            'security_assessment': {
                'scan_type': ScanType.VULNERABILITY_SCAN,
                'intensity': ScanIntensity.AGGRESSIVE,
                'description': 'Comprehensive security assessment with vulnerability detection'
            },
            'network_discovery': {
                'scan_type': ScanType.TOPOLOGY_SCAN,
                'intensity': ScanIntensity.POLITE,
                'description': 'Network topology discovery and mapping'
            },
            'service_inventory': {
                'scan_type': ScanType.SERVICE_DETECTION,
                'intensity': ScanIntensity.NORMAL,
                'description': 'Detailed service and version detection'
            },
            'stealth_scan': {
                'scan_type': ScanType.STEALTH_SCAN,
                'intensity': ScanIntensity.SNEAKY,
                'description': 'Stealthy scan for sensitive environments'
            }
        }
    
    async def execute_scan_template(self, template_name: str, targets: List[str],
                                  custom_options: Dict[str, Any] = None) -> ScanResult:
        """Execute a predefined scan template."""
        if template_name not in self.scan_templates:
            raise ValueError(f"Unknown scan template: {template_name}")
        
        template = self.scan_templates[template_name]
        return await self.scanner.scan_hosts(
            targets=targets,
            scan_type=template['scan_type'],
            intensity=template['intensity'],
            custom_options=custom_options
        )
    
    async def custom_scan(self, targets: List[str], scan_type: ScanType,
                         intensity: ScanIntensity = ScanIntensity.NORMAL,
                         custom_options: Dict[str, Any] = None) -> ScanResult:
        """Execute a custom scan with specific parameters."""
        return await self.scanner.scan_hosts(
            targets=targets,
            scan_type=scan_type,
            intensity=intensity,
            custom_options=custom_options
        )
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available scan templates."""
        return self.scan_templates.copy()
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanning performance statistics."""
        stats = self.performance_stats.copy()
        stats['scan_history_count'] = len(self.scanner.get_scan_history())
        stats['running_scans'] = len(self.scanner.get_running_scans())
        return stats
    
    def analyze_scan_results(self, scan_result: ScanResult) -> Dict[str, Any]:
        """Analyze scan results for security insights."""
        if not scan_result.hosts:
            return {}
        
        analysis = {
            'security_risks': [],
            'recommendations': [],
            'compliance_issues': [],
            'network_insights': []
        }
        
        # Analyze open ports and services
        for host in scan_result.hosts:
            if host.status == 'up':
                open_ports = [p for p in host.ports if p.get('state') == 'open']
                
                for port in open_ports:
                    port_num = int(port.get('port', 0))
                    service = port.get('service', '')
                    
                    # Check for common security risks
                    if port_num in [21, 23, 3389] and not service.startswith('ssh'):
                        analysis['security_risks'].append({
                            'host': host.ip_address,
                            'port': port_num,
                            'risk': 'Unencrypted remote access service',
                            'severity': 'high'
                        })
                    
                    if port_num == 22 and service.startswith('ssh'):
                        analysis['recommendations'].append({
                            'host': host.ip_address,
                            'action': 'Verify SSH configuration and key-based authentication'
                        })
        
        # Add network insights
        up_hosts = len([h for h in scan_result.hosts if h.status == 'up'])
        if up_hosts > 0:
            analysis['network_insights'].append({
                'total_hosts': len(scan_result.hosts),
                'active_hosts': up_hosts,
                'uptime_percentage': (up_hosts / len(scan_result.hosts)) * 100
            })
        
        return analysis

async def main():
    """Example usage and testing."""
    try:
        # Initialize scanner
        manager = HostScanningManager()
        
        # Test targets
        test_targets = ['127.0.0.1', 'localhost']
        
        print("🔍 Available scan templates:")
        templates = manager.get_available_templates()
        for name, template in templates.items():
            print(f"  - {name}: {template['description']}")
        
        print(f"\n🚀 Starting quick audit scan of {test_targets}...")
        scan_result = await manager.execute_scan_template('quick_audit', test_targets)
        
        print(f"\n📊 Scan completed:")
        print(f"  - Scan ID: {scan_result.scan_id}")
        print(f"  - Status: {'Success' if not scan_result.errors else 'Failed'}")
        print(f"  - Hosts found: {len(scan_result.hosts)}")
        
        if scan_result.summary:
            print(f"  - Open ports: {scan_result.summary.get('open_ports', 0)}")
            print(f"  - Services: {scan_result.summary.get('services_found', 0)}")
        
        # Analyze results
        analysis = manager.analyze_scan_results(scan_result)
        if analysis['security_risks']:
            print(f"\n⚠️  Security risks found: {len(analysis['security_risks'])}")
        
        # Get statistics
        stats = manager.get_scan_statistics()
        print(f"\n📈 Scanner statistics:")
        print(f"  - Total scans: {stats['total_scans']}")
        print(f"  - Scan history: {stats['scan_history_count']}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
