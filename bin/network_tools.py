#!/usr/bin/env python3
"""
MCP Network Tools Collection - Internal Network Analysis

Provides comprehensive network analysis capabilities:
- Connectivity testing (ping, traceroute)
- DNS resolution (nslookup, dig-like functionality)
- Network statistics (netstat, ss)
- ARP table management
- Port scanning and service detection
- Network interface monitoring
- Bandwidth analysis
- Security scanning

All tools are designed to integrate with the Query Path and Runner Agent
for dynamic workflow execution.
"""

import asyncio
import json
import logging
import subprocess
import socket
import time
import platform
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class NetworkToolCategory(Enum):
    """Network tool categories for classification."""
    CONNECTIVITY = "connectivity"
    DNS = "dns"
    STATISTICS = "statistics"
    SECURITY = "security"
    MONITORING = "monitoring"
    DIAGNOSTICS = "diagnostics"

class NetworkToolCapability(Enum):
    """Network tool capabilities for dynamic selection."""
    PING = "ping"
    DNS_LOOKUP = "dns_lookup"
    PORT_SCAN = "port_scan"
    NETWORK_STATS = "network_stats"
    ARP_MANAGEMENT = "arp_management"
    TRACEROUTE = "traceroute"
    INTERFACE_MONITORING = "interface_monitoring"
    BANDWIDTH_ANALYSIS = "bandwidth_analysis"
    SECURITY_SCANNING = "security_scanning"

@dataclass
class NetworkToolMetadata:
    """Metadata for network tools."""
    tool_id: str
    name: str
    description: str
    category: NetworkToolCategory
    capabilities: List[NetworkToolCapability]
    input_types: List[str]
    output_types: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = 0.0

class NetworkPingTool:
    """Advanced ping tool with comprehensive analysis."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_ping",
            name="Network Ping Analysis",
            description="Comprehensive ping tool with latency analysis, packet loss, and jitter calculation",
            category=NetworkToolCategory.CONNECTIVITY,
            capabilities=[NetworkToolCapability.PING],
            input_types=["host", "count", "timeout", "size"],
            output_types=["ping_results", "statistics", "analysis"]
        )
    
    async def execute(self, host: str, count: int = 4, timeout: float = 1.0, size: int = 56) -> Dict[str, Any]:
        """Execute ping analysis."""
        try:
            start_time = time.time()
            
            # Validate host
            if not self._is_valid_host(host):
                return {"success": False, "error": f"Invalid host: {host}"}
            
            # Execute ping based on platform
            if platform.system().lower() == "windows":
                results = await self._ping_windows(host, count, timeout, size)
            else:
                results = await self._ping_unix(host, count, timeout, size)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_ping_results(results)
            }
            
        except Exception as e:
            logger.error(f"Ping execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _is_valid_host(self, host: str) -> bool:
        """Validate host format."""
        try:
            # Check if it's a valid IP address
            ipaddress.ip_address(host)
            return True
        except ValueError:
            # Check if it's a valid hostname
            if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', host):
                return True
            return False
    
    async def _ping_windows(self, host: str, count: int, timeout: float, size: int) -> Dict[str, Any]:
        """Execute ping on Windows."""
        cmd = [
            "ping", "-n", str(count), "-w", str(int(timeout * 1000)), 
            "-l", str(size), host
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_ping_output(stdout.decode(), "windows")
    
    async def _ping_unix(self, host: str, count: int, timeout: float, size: int) -> Dict[str, Any]:
        """Execute ping on Unix-like systems."""
        cmd = [
            "ping", "-c", str(count), "-W", str(int(timeout)), 
            "-s", str(size), host
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_ping_output(stdout.decode(), "unix")
    
    def _parse_ping_output(self, output: str, platform_type: str) -> Dict[str, Any]:
        """Parse ping output for different platforms."""
        results = {
            "platform": platform_type,
            "responses": [],
            "statistics": {
                "packets_sent": 0,
                "packets_received": 0,
                "packet_loss_percent": 0.0,
                "min_rtt": 0.0,
                "avg_rtt": 0.0,
                "max_rtt": 0.0,
                "mdev_rtt": 0.0
            }
        }
        
        if platform_type == "windows":
            return self._parse_windows_ping(output, results)
        else:
            return self._parse_unix_ping(output, results)
    
    def _parse_windows_ping(self, output: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Windows ping output."""
        lines = output.split('\n')
        rtt_values = []
        
        for line in lines:
            line = line.strip()
            if "Reply from" in line:
                # Extract RTT from "Reply from X.X.X.X: bytes=56 time=1ms TTL=128"
                match = re.search(r'time=(\d+)ms', line)
                if match:
                    rtt = int(match.group(1))
                    rtt_values.append(rtt)
                    results["responses"].append({
                        "status": "success",
                        "rtt_ms": rtt,
                        "raw_line": line
                    })
                results["statistics"]["packets_received"] += 1
            elif "Request timed out" in line:
                results["responses"].append({
                    "status": "timeout",
                    "rtt_ms": None,
                    "raw_line": line
                })
            elif "Ping statistics" in line:
                # Extract packet statistics
                stats_match = re.search(r'Packets: Sent = (\d+), Received = (\d+), Lost = (\d+)', output)
                if stats_match:
                    results["statistics"]["packets_sent"] = int(stats_match.group(1))
                    results["statistics"]["packets_received"] = int(stats_match.group(2))
                    lost = int(stats_match.group(3))
                    results["statistics"]["packet_loss_percent"] = (lost / results["statistics"]["packets_sent"]) * 100
        
        # Calculate RTT statistics
        if rtt_values:
            results["statistics"]["min_rtt"] = min(rtt_values)
            results["statistics"]["max_rtt"] = max(rtt_values)
            results["statistics"]["avg_rtt"] = sum(rtt_values) / len(rtt_values)
            results["statistics"]["mdev_rtt"] = self._calculate_mdev(rtt_values)
        
        return results
    
    def _parse_unix_ping(self, output: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Unix ping output."""
        lines = output.split('\n')
        rtt_values = []
        
        for line in lines:
            line = line.strip()
            if "time=" in line:
                # Extract RTT from "64 bytes from X.X.X.X: icmp_seq=1 time=1.234 ms"
                match = re.search(r'time=([\d.]+) ms', line)
                if match:
                    rtt = float(match.group(1))
                    rtt_values.append(rtt)
                    results["responses"].append({
                        "status": "success",
                        "rtt_ms": rtt,
                        "raw_line": line
                    })
                results["statistics"]["packets_received"] += 1
            elif "no answer" in line or "timeout" in line:
                results["responses"].append({
                    "status": "timeout",
                    "rtt_ms": None,
                    "raw_line": line
                })
            elif "packets transmitted" in line:
                # Extract statistics from "4 packets transmitted, 4 received, 0% packet loss"
                stats_match = re.search(r'(\d+) packets transmitted, (\d+) received, ([\d.]+)% packet loss', line)
                if stats_match:
                    results["statistics"]["packets_sent"] = int(stats_match.group(1))
                    results["statistics"]["packets_received"] = int(stats_match.group(2))
                    results["statistics"]["packet_loss_percent"] = float(stats_match.group(3))
            elif "rtt min/avg/max/mdev" in line:
                # Extract RTT statistics from "rtt min/avg/max/mdev = 1.234/2.345/3.456/0.123 ms"
                rtt_match = re.search(r'= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms', line)
                if rtt_match:
                    results["statistics"]["min_rtt"] = float(rtt_match.group(1))
                    results["statistics"]["avg_rtt"] = float(rtt_match.group(2))
                    results["statistics"]["max_rtt"] = float(rtt_match.group(3))
                    results["statistics"]["mdev_rtt"] = float(rtt_match.group(4))
        
        # Calculate RTT statistics if not provided by ping
        if rtt_values and not results["statistics"]["avg_rtt"]:
            results["statistics"]["min_rtt"] = min(rtt_values)
            results["statistics"]["max_rtt"] = max(rtt_values)
            results["statistics"]["avg_rtt"] = sum(rtt_values) / len(rtt_values)
            results["statistics"]["mdev_rtt"] = self._calculate_mdev(rtt_values)
        
        return results
    
    def _calculate_mdev(self, values: List[float]) -> float:
        """Calculate mean deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        deviations = [abs(x - mean) for x in values]
        return sum(deviations) / len(deviations)
    
    def _analyze_ping_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ping results and provide insights."""
        analysis = {
            "connectivity_status": "unknown",
            "quality_assessment": "unknown",
            "recommendations": [],
            "anomalies": []
        }
        
        stats = results.get("statistics", {})
        
        # Determine connectivity status
        if stats.get("packets_received", 0) > 0:
            analysis["connectivity_status"] = "connected"
        else:
            analysis["connectivity_status"] = "disconnected"
            analysis["recommendations"].append("Check network connectivity and firewall settings")
        
        # Assess quality based on packet loss and RTT
        packet_loss = stats.get("packet_loss_percent", 0.0)
        avg_rtt = stats.get("avg_rtt", 0.0)
        
        if packet_loss == 0.0 and avg_rtt < 50:
            analysis["quality_assessment"] = "excellent"
        elif packet_loss < 1.0 and avg_rtt < 100:
            analysis["quality_assessment"] = "good"
        elif packet_loss < 5.0 and avg_rtt < 200:
            analysis["quality_assessment"] = "fair"
        else:
            analysis["quality_assessment"] = "poor"
            analysis["recommendations"].append("Network quality issues detected - investigate further")
        
        # Check for anomalies
        if packet_loss > 10.0:
            analysis["anomalies"].append(f"High packet loss: {packet_loss:.1f}%")
        
        if avg_rtt > 500:
            analysis["anomalies"].append(f"High latency: {avg_rtt:.1f}ms")
        
        # Add specific recommendations
        if analysis["quality_assessment"] == "poor":
            analysis["recommendations"].extend([
                "Check for network congestion",
                "Verify QoS settings",
                "Consider network path optimization"
            ])
        
        return analysis

class NetworkDNSTool:
    """Advanced DNS lookup tool with comprehensive resolution analysis."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_dns_lookup",
            name="DNS Lookup Analysis",
            description="Comprehensive DNS resolution tool with multiple record types and performance analysis",
            category=NetworkToolCategory.DNS,
            capabilities=[NetworkToolCapability.DNS_LOOKUP],
            input_types=["hostname", "record_type", "nameserver"],
            output_types=["dns_results", "resolution_time", "analysis"]
        )
    
    async def execute(self, hostname: str, record_type: str = "A", nameserver: Optional[str] = None) -> Dict[str, Any]:
        """Execute DNS lookup."""
        try:
            start_time = time.time()
            
            # Validate hostname
            if not self._is_valid_hostname(hostname):
                return {"success": False, "error": f"Invalid hostname: {hostname}"}
            
            # Execute DNS lookup
            results = await self._perform_dns_lookup(hostname, record_type, nameserver)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_dns_results(results)
            }
            
        except Exception as e:
            logger.error(f"DNS lookup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _is_valid_hostname(self, hostname: str) -> bool:
        """Validate hostname format."""
        return bool(re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', hostname))
    
    async def _perform_dns_lookup(self, hostname: str, record_type: str, nameserver: Optional[str]) -> Dict[str, Any]:
        """Perform DNS lookup using system tools."""
        try:
            # Use nslookup or dig based on availability
            if self._has_dig():
                return await self._dig_lookup(hostname, record_type, nameserver)
            else:
                return await self._nslookup_lookup(hostname, record_type, nameserver)
        except Exception as e:
            # Fallback to Python socket resolution
            return await self._socket_lookup(hostname, record_type)
    
    def _has_dig(self) -> bool:
        """Check if dig command is available."""
        try:
            subprocess.run(["dig", "-v"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def _dig_lookup(self, hostname: str, record_type: str, nameserver: Optional[str]) -> Dict[str, Any]:
        """Use dig for DNS lookup."""
        cmd = ["dig", record_type, hostname]
        if nameserver:
            cmd.extend(["@", nameserver])
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_dig_output(stdout.decode(), record_type)
    
    async def _nslookup_lookup(self, hostname: str, record_type: str, nameserver: Optional[str]) -> Dict[str, Any]:
        """Use nslookup for DNS lookup."""
        cmd = ["nslookup", "-type=" + record_type, hostname]
        if nameserver:
            cmd.append(nameserver)
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_nslookup_output(stdout.decode(), record_type)
    
    async def _socket_lookup(self, hostname: str, record_type: str) -> Dict[str, Any]:
        """Use Python socket for basic DNS resolution."""
        results = {
            "hostname": hostname,
            "record_type": record_type,
            "resolver": "python_socket",
            "answers": [],
            "authority": [],
            "additional": []
        }
        
        try:
            if record_type.upper() == "A":
                ip = socket.gethostbyname(hostname)
                results["answers"].append({
                    "type": "A",
                    "data": ip,
                    "ttl": None
                })
            elif record_type.upper() == "CNAME":
                # Try to get canonical name
                try:
                    canonical = socket.gethostbyaddr(hostname)[0]
                    results["answers"].append({
                        "type": "CNAME",
                        "data": canonical,
                        "ttl": None
                    })
                except socket.herror:
                    pass
        except socket.gaierror as e:
            results["error"] = str(e)
        
        return results
    
    def _parse_dig_output(self, output: str, record_type: str) -> Dict[str, Any]:
        """Parse dig command output."""
        results = {
            "hostname": "",
            "record_type": record_type,
            "resolver": "dig",
            "answers": [],
            "authority": [],
            "additional": [],
            "query_time": None,
            "server": None
        }
        
        lines = output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith(';'):
                continue
            
            if ';ANSWER SECTION:' in line:
                current_section = 'answers'
            elif ';AUTHORITY SECTION:' in line:
                current_section = 'authority'
            elif ';ADDITIONAL SECTION:' in line:
                current_section = 'additional'
            elif 'Query time:' in line:
                match = re.search(r'Query time: (\d+) msec', line)
                if match:
                    results["query_time"] = int(match.group(1))
            elif 'SERVER:' in line:
                match = re.search(r'SERVER: ([\d.]+)', line)
                if match:
                    results["server"] = match.group(1)
            elif line and not line.startswith(';') and current_section:
                # Parse record line
                parts = line.split()
                if len(parts) >= 5:
                    record = {
                        "type": parts[3],
                        "data": parts[4],
                        "ttl": int(parts[1]) if parts[1].isdigit() else None
                    }
                    results[current_section].append(record)
        
        return results
    
    def _parse_nslookup_output(self, output: str, record_type: str) -> Dict[str, Any]:
        """Parse nslookup command output."""
        results = {
            "hostname": "",
            "record_type": record_type,
            "resolver": "nslookup",
            "answers": [],
            "authority": [],
            "additional": []
        }
        
        lines = output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'Name:' in line:
                results["hostname"] = line.split(':', 1)[1].strip()
            elif 'Address:' in line:
                address = line.split(':', 1)[1].strip()
                if address and address != "0.0.0.0":
                    results["answers"].append({
                        "type": "A",
                        "data": address,
                        "ttl": None
                    })
        
        return results
    
    def _analyze_dns_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DNS results and provide insights."""
        analysis = {
            "resolution_status": "unknown",
            "performance_assessment": "unknown",
            "security_analysis": [],
            "recommendations": []
        }
        
        # Determine resolution status
        if results.get("answers"):
            analysis["resolution_status"] = "resolved"
        else:
            analysis["resolution_status"] = "unresolved"
            analysis["recommendations"].append("DNS resolution failed - check hostname and DNS configuration")
        
        # Assess performance
        query_time = results.get("query_time")
        if query_time is not None:
            if query_time < 50:
                analysis["performance_assessment"] = "excellent"
            elif query_time < 100:
                analysis["performance_assessment"] = "good"
            elif query_time < 200:
                analysis["performance_assessment"] = "fair"
            else:
                analysis["performance_assessment"] = "poor"
                analysis["recommendations"].append("Slow DNS resolution - consider using faster nameservers")
        
        # Security analysis
        if results.get("resolver") == "dig" and results.get("server"):
            analysis["security_analysis"].append(f"Using nameserver: {results['server']}")
        
        # Add recommendations
        if analysis["resolution_status"] == "resolved":
            analysis["recommendations"].append("DNS resolution successful")
        
        return analysis

# Additional network tools will be added in the next part...

class NetworkNetstatTool:
    """Advanced netstat tool for network statistics and connection analysis."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_netstat",
            name="Network Statistics Analysis",
            description="Comprehensive network statistics tool showing connections, listening ports, and interface statistics",
            category=NetworkToolCategory.STATISTICS,
            capabilities=[NetworkToolCapability.NETWORK_STATS],
            input_types=["protocol", "state", "interface"],
            output_types=["connections", "listening_ports", "interface_stats", "analysis"]
        )
    
    async def execute(self, protocol: str = "all", state: str = "all", interface: Optional[str] = None) -> Dict[str, Any]:
        """Execute netstat analysis."""
        try:
            start_time = time.time()
            
            # Execute netstat based on platform
            if platform.system().lower() == "windows":
                results = await self._netstat_windows(protocol, state, interface)
            else:
                results = await self._netstat_unix(protocol, state, interface)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_netstat_results(results)
            }
            
        except Exception as e:
            logger.error(f"Netstat execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _netstat_windows(self, protocol: str, state: str, interface: Optional[str]) -> Dict[str, Any]:
        """Execute netstat on Windows."""
        cmd = ["netstat", "-an"]
        
        if protocol != "all":
            if protocol.upper() == "TCP":
                cmd.append("-p")
                cmd.append("TCP")
            elif protocol.upper() == "UDP":
                cmd.append("-p")
                cmd.append("UDP")
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_windows_netstat(stdout.decode())
    
    async def _netstat_unix(self, protocol: str, state: str, interface: Optional[str]) -> Dict[str, Any]:
        """Execute netstat on Unix-like systems."""
        cmd = ["netstat", "-tuln"]
        
        if protocol != "all":
            if protocol.upper() == "TCP":
                cmd = ["netstat", "-tln"]
            elif protocol.upper() == "UDP":
                cmd = ["netstat", "-uln"]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_unix_netstat(stdout.decode())
    
    def _parse_windows_netstat(self, output: str) -> Dict[str, Any]:
        """Parse Windows netstat output."""
        results = {
            "connections": [],
            "listening_ports": [],
            "interface_stats": []
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Proto'):
                parts = line.split()
                if len(parts) >= 4:
                    connection = {
                        "protocol": parts[0],
                        "local_address": parts[1],
                        "foreign_address": parts[2],
                        "state": parts[3] if len(parts) > 3 else "UNKNOWN"
                    }
                    
                    if connection["state"] == "LISTENING":
                        results["listening_ports"].append(connection)
                    else:
                        results["connections"].append(connection)
        
        return results
    
    def _parse_unix_netstat(self, output: str) -> Dict[str, Any]:
        """Parse Unix netstat output."""
        results = {
            "connections": [],
            "listening_ports": [],
            "interface_stats": []
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Proto'):
                parts = line.split()
                if len(parts) >= 4:
                    connection = {
                        "protocol": parts[0],
                        "recv_q": parts[1],
                        "send_q": parts[2],
                        "local_address": parts[3],
                        "foreign_address": parts[4] if len(parts) > 4 else "*:*",
                        "state": parts[5] if len(parts) > 5 else "LISTEN"
                    }
                    
                    if connection["state"] == "LISTEN":
                        results["listening_ports"].append(connection)
                    else:
                        results["connections"].append(connection)
        
        return results
    
    def _analyze_netstat_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze netstat results and provide insights."""
        analysis = {
            "total_connections": len(results.get("connections", [])),
            "total_listening_ports": len(results.get("listening_ports", [])),
            "security_analysis": [],
            "recommendations": [],
            "anomalies": []
        }
        
        # Analyze listening ports for security
        listening_ports = results.get("listening_ports", [])
        for port_info in listening_ports:
            local_addr = port_info.get("local_address", "")
            if "0.0.0.0:" in local_addr or ":::" in local_addr:
                analysis["security_analysis"].append(f"Port {local_addr} listening on all interfaces")
                analysis["recommendations"].append(f"Consider restricting {local_addr} to specific interfaces")
        
        # Check for common vulnerable ports
        vulnerable_ports = [21, 23, 25, 53, 80, 110, 143, 389, 443, 1433, 3306, 5432, 6379, 27017]
        for port_info in listening_ports:
            local_addr = port_info.get("local_address", "")
            port_match = re.search(r':(\d+)$', local_addr)
            if port_match:
                port = int(port_match.group(1))
                if port in vulnerable_ports:
                    analysis["security_analysis"].append(f"Potentially vulnerable port {port} is open")
        
        # Add recommendations
        if analysis["total_listening_ports"] > 20:
            analysis["recommendations"].append("High number of listening ports - review necessity")
        
        if analysis["total_connections"] > 100:
            analysis["recommendations"].append("High number of active connections - monitor for anomalies")
        
        return analysis

class NetworkARPTool:
    """Advanced ARP tool for address resolution protocol management."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_arp",
            name="ARP Table Management",
            description="Comprehensive ARP table management tool for viewing, adding, and deleting ARP entries",
            category=NetworkToolCategory.STATISTICS,
            capabilities=[NetworkToolCapability.ARP_MANAGEMENT],
            input_types=["interface", "action", "ip_address", "mac_address"],
            output_types=["arp_table", "interface_info", "analysis"]
        )
    
    async def execute(self, interface: Optional[str] = None, action: str = "show", 
                     ip_address: Optional[str] = None, mac_address: Optional[str] = None) -> Dict[str, Any]:
        """Execute ARP operations."""
        try:
            start_time = time.time()
            
            if action == "show":
                results = await self._show_arp_table(interface)
            elif action == "add":
                if not ip_address or not mac_address:
                    return {"success": False, "error": "IP address and MAC address required for add action"}
                results = await self._add_arp_entry(interface, ip_address, mac_address)
            elif action == "delete":
                if not ip_address:
                    return {"success": False, "error": "IP address required for delete action"}
                results = await self._delete_arp_entry(interface, ip_address)
            else:
                return {"success": False, "error": f"Invalid action: {action}"}
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_arp_results(results)
            }
            
        except Exception as e:
            logger.error(f"ARP operation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _show_arp_table(self, interface: Optional[str]) -> Dict[str, Any]:
        """Show ARP table."""
        if platform.system().lower() == "windows":
            cmd = ["arp", "-a"]
        else:
            cmd = ["arp", "-a"]
            if interface:
                cmd.extend(["-i", interface])
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_arp_output(stdout.decode(), interface)
    
    async def _add_arp_entry(self, interface: Optional[str], ip_address: str, mac_address: str) -> Dict[str, Any]:
        """Add ARP entry."""
        if platform.system().lower() == "windows":
            cmd = ["arp", "-s", ip_address, mac_address]
        else:
            cmd = ["arp", "-s", ip_address, mac_address]
            if interface:
                cmd.extend(["-i", interface])
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "action": "add",
            "ip_address": ip_address,
            "mac_address": mac_address,
            "interface": interface,
            "success": process.returncode == 0,
            "output": stdout.decode(),
            "error": stderr.decode() if stderr else None
        }
    
    async def _delete_arp_entry(self, interface: Optional[str], ip_address: str) -> Dict[str, Any]:
        """Delete ARP entry."""
        if platform.system().lower() == "windows":
            cmd = ["arp", "-d", ip_address]
        else:
            cmd = ["arp", "-d", ip_address]
            if interface:
                cmd.extend(["-i", interface])
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "action": "delete",
            "ip_address": ip_address,
            "interface": interface,
            "success": process.returncode == 0,
            "output": stdout.decode(),
            "error": stderr.decode() if stderr else None
        }
    
    def _parse_arp_output(self, output: str, interface: Optional[str]) -> Dict[str, Any]:
        """Parse ARP command output."""
        results = {
            "interface": interface,
            "arp_entries": [],
            "total_entries": 0
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Address'):
                parts = line.split()
                if len(parts) >= 3:
                    entry = {
                        "ip_address": parts[0],
                        "mac_address": parts[1],
                        "type": parts[2] if len(parts) > 2 else "dynamic",
                        "interface": parts[3] if len(parts) > 3 else interface
                    }
                    results["arp_entries"].append(entry)
        
        results["total_entries"] = len(results["arp_entries"])
        return results
    
    def _analyze_arp_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ARP results and provide insights."""
        analysis = {
            "total_entries": results.get("total_entries", 0),
            "security_analysis": [],
            "recommendations": [],
            "anomalies": []
        }
        
        # Check for potential ARP spoofing
        arp_entries = results.get("arp_entries", [])
        ip_to_mac = {}
        
        for entry in arp_entries:
            ip = entry.get("ip_address")
            mac = entry.get("mac_address")
            
            if ip in ip_to_mac and ip_to_mac[ip] != mac:
                analysis["anomalies"].append(f"Multiple MAC addresses for IP {ip}: {ip_to_mac[ip]} and {mac}")
                analysis["security_analysis"].append(f"Potential ARP spoofing detected for IP {ip}")
            
            ip_to_mac[ip] = mac
        
        # Check for suspicious MAC addresses
        for entry in arp_entries:
            mac = entry.get("mac_address", "")
            if mac.startswith("00:00:00:00:00:00"):
                analysis["anomalies"].append(f"Invalid MAC address: {mac}")
                analysis["security_analysis"].append(f"Invalid MAC address detected: {mac}")
        
        # Add recommendations
        if analysis["total_entries"] > 100:
            analysis["recommendations"].append("Large ARP table - consider cleanup of stale entries")
        
        if analysis["anomalies"]:
            analysis["recommendations"].append("Investigate detected anomalies for security implications")
        
        return analysis

class NetworkTracerouteTool:
    """Advanced traceroute tool for network path analysis."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_traceroute",
            name="Network Path Analysis",
            description="Comprehensive traceroute tool for analyzing network paths and identifying bottlenecks",
            category=NetworkToolCategory.CONNECTIVITY,
            capabilities=[NetworkToolCapability.TRACEROUTE],
            input_types=["host", "max_hops", "timeout", "protocol"],
            output_types=["path_analysis", "hop_details", "bottleneck_analysis"]
        )
    
    async def execute(self, host: str, max_hops: int = 30, timeout: float = 1.0, protocol: str = "icmp") -> Dict[str, Any]:
        """Execute traceroute analysis."""
        try:
            start_time = time.time()
            
            # Validate host
            if not self._is_valid_host(host):
                return {"success": False, "error": f"Invalid host: {host}"}
            
            # Execute traceroute based on platform
            if platform.system().lower() == "windows":
                results = await self._traceroute_windows(host, max_hops, timeout, protocol)
            else:
                results = await self._traceroute_unix(host, max_hops, timeout, protocol)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_traceroute_results(results)
            }
            
        except Exception as e:
            logger.error(f"Traceroute execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _is_valid_host(self, host: str) -> bool:
        """Validate host format."""
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', host):
                return True
            return False
    
    async def _traceroute_windows(self, host: str, max_hops: int, timeout: float, protocol: str) -> Dict[str, Any]:
        """Execute traceroute on Windows."""
        cmd = ["tracert", "-h", str(max_hops), "-w", str(int(timeout * 1000)), host]
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_windows_traceroute(stdout.decode())
    
    async def _traceroute_unix(self, host: str, max_hops: int, timeout: float, protocol: str) -> Dict[str, Any]:
        """Execute traceroute on Unix-like systems."""
        cmd = ["traceroute", "-m", str(max_hops), "-w", str(timeout), host]
        
        if protocol.upper() == "TCP":
            cmd.append("-T")
        elif protocol.upper() == "UDP":
            cmd.append("-U")
        
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return self._parse_unix_traceroute(stdout.decode())
    
    def _parse_windows_traceroute(self, output: str) -> Dict[str, Any]:
        """Parse Windows tracert output."""
        results = {
            "platform": "windows",
            "hops": [],
            "total_hops": 0,
            "destination_reached": False
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+', line):
                parts = line.split()
                if len(parts) >= 4:
                    hop = {
                        "hop_number": int(parts[0]),
                        "ip_address": parts[1],
                        "hostname": parts[2] if len(parts) > 2 else None,
                        "response_times": []
                    }
                    
                    # Extract response times
                    for part in parts[3:]:
                        if part.endswith('ms'):
                            try:
                                time_ms = int(part[:-2])
                                hop["response_times"].append(time_ms)
                            except ValueError:
                                pass
                    
                    results["hops"].append(hop)
                    results["total_hops"] = max(results["total_hops"], hop["hop_number"])
        
        # Check if destination was reached
        if results["hops"]:
            last_hop = results["hops"][-1]
            results["destination_reached"] = "*" not in str(last_hop.get("ip_address", ""))
        
        return results
    
    def _parse_unix_traceroute(self, output: str) -> Dict[str, Any]:
        """Parse Unix traceroute output."""
        results = {
            "platform": "unix",
            "hops": [],
            "total_hops": 0,
            "destination_reached": False
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+', line):
                parts = line.split()
                if len(parts) >= 4:
                    hop = {
                        "hop_number": int(parts[0]),
                        "ip_address": parts[1],
                        "hostname": parts[2] if len(parts) > 2 else None,
                        "response_times": []
                    }
                    
                    # Extract response times
                    for part in parts[3:]:
                        if part.endswith('ms'):
                            try:
                                time_ms = float(part[:-2])
                                hop["response_times"].append(time_ms)
                            except ValueError:
                                pass
                    
                    results["hops"].append(hop)
                    results["total_hops"] = max(results["total_hops"], hop["hop_number"])
        
        # Check if destination was reached
        if results["hops"]:
            last_hop = results["hops"][-1]
            results["destination_reached"] = "*" not in str(last_hop.get("ip_address", ""))
        
        return results
    
    def _analyze_traceroute_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze traceroute results and provide insights."""
        analysis = {
            "path_analysis": {
                "total_hops": results.get("total_hops", 0),
                "destination_reached": results.get("destination_reached", False),
                "path_efficiency": "unknown"
            },
            "bottleneck_analysis": [],
            "network_insights": [],
            "recommendations": []
        }
        
        hops = results.get("hops", [])
        if not hops:
            analysis["recommendations"].append("No path information available")
            return analysis
        
        # Analyze path efficiency
        total_distance = len(hops)
        if total_distance <= 5:
            analysis["path_analysis"]["path_efficiency"] = "excellent"
        elif total_distance <= 10:
            analysis["path_analysis"]["path_efficiency"] = "good"
        elif total_distance <= 15:
            analysis["path_analysis"]["path_efficiency"] = "fair"
        else:
            analysis["path_analysis"]["path_efficiency"] = "poor"
            analysis["recommendations"].append("Long network path detected - consider path optimization")
        
        # Identify bottlenecks
        for i, hop in enumerate(hops):
            response_times = hop.get("response_times", [])
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                if avg_time > 100:  # High latency threshold
                    analysis["bottleneck_analysis"].append({
                        "hop_number": hop["hop_number"],
                        "ip_address": hop["ip_address"],
                        "avg_latency": avg_time,
                        "severity": "high" if avg_time > 200 else "medium"
                    })
        
        # Network insights
        if not results.get("destination_reached"):
            analysis["network_insights"].append("Destination not reached - possible network issues")
            analysis["recommendations"].append("Investigate why destination is unreachable")
        
        # Add recommendations based on analysis
        if analysis["bottleneck_analysis"]:
            analysis["recommendations"].append("Network bottlenecks detected - investigate high-latency hops")
        
        if analysis["path_analysis"]["path_efficiency"] == "poor":
            analysis["recommendations"].append("Consider network path optimization")
        
        return analysis

class NetworkPortScannerTool:
    """Advanced port scanner for service detection and security analysis."""
    
    def __init__(self):
        self.metadata = NetworkToolMetadata(
            tool_id="network_port_scanner",
            name="Port Scanner & Service Detection",
            description="Comprehensive port scanner for identifying open ports, services, and potential security vulnerabilities",
            category=NetworkToolCategory.SECURITY,
            capabilities=[NetworkToolCapability.PORT_SCAN],
            input_types=["host", "port_range", "scan_type", "timeout"],
            output_types=["open_ports", "service_identification", "security_analysis", "vulnerability_assessment"]
        )
    
    async def execute(self, host: str, port_range: str = "1-1024", scan_type: str = "tcp", timeout: float = 1.0) -> Dict[str, Any]:
        """Execute port scan."""
        try:
            start_time = time.time()
            
            # Validate host
            if not self._is_valid_host(host):
                return {"success": False, "error": f"Invalid host: {host}"}
            
            # Parse port range
            ports = self._parse_port_range(port_range)
            if not ports:
                return {"success": False, "error": f"Invalid port range: {port_range}"}
            
            # Execute port scan
            results = await self._scan_ports(host, ports, scan_type, timeout)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.metadata.performance_metrics["avg_execution_time"] = (
                (self.metadata.performance_metrics.get("avg_execution_time", 0) * self.metadata.usage_count + execution_time) / 
                (self.metadata.usage_count + 1)
            )
            self.metadata.usage_count += 1
            self.metadata.last_used = time.time()
            
            return {
                "success": True,
                "tool_id": self.metadata.tool_id,
                "execution_time": execution_time,
                "results": results,
                "analysis": self._analyze_port_scan_results(results)
            }
            
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _is_valid_host(self, host: str) -> bool:
        """Validate host format."""
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', host):
                return True
            return False
    
    def _parse_port_range(self, port_range: str) -> List[int]:
        """Parse port range string into list of ports."""
        ports = []
        try:
            if "-" in port_range:
                start, end = port_range.split("-")
                ports = list(range(int(start), int(end) + 1))
            else:
                ports = [int(port_range)]
        except ValueError:
            return []
        
        # Filter valid ports
        return [p for p in ports if 1 <= p <= 65535]
    
    async def _scan_ports(self, host: str, ports: List[int], scan_type: str, timeout: float) -> Dict[str, Any]:
        """Scan ports using concurrent connections."""
        results = {
            "host": host,
            "scan_type": scan_type,
            "total_ports": len(ports),
            "open_ports": [],
            "closed_ports": [],
            "filtered_ports": [],
            "scan_summary": {}
        }
        
        # Use ThreadPoolExecutor for concurrent scanning
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for port in ports:
                future = executor.submit(self._check_port, host, port, scan_type, timeout)
                futures.append((port, future))
            
            # Collect results
            for port, future in futures:
                try:
                    result = future.result(timeout=timeout + 1)
                    if result["status"] == "open":
                        results["open_ports"].append(result)
                    elif result["status"] == "closed":
                        results["closed_ports"].append(result)
                    else:
                        results["filtered_ports"].append(result)
                except Exception as e:
                    results["filtered_ports"].append({
                        "port": port,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Generate summary
        results["scan_summary"] = {
            "open_count": len(results["open_ports"]),
            "closed_count": len(results["closed_ports"]),
            "filtered_count": len(results["filtered_ports"]),
            "open_percentage": (len(results["open_ports"]) / len(ports)) * 100
        }
        
        return results
    
    def _check_port(self, host: str, port: int, scan_type: str, timeout: float) -> Dict[str, Any]:
        """Check individual port status."""
        try:
            if scan_type.upper() == "TCP":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    service = self._identify_service(port)
                    return {
                        "port": port,
                        "status": "open",
                        "service": service,
                        "protocol": "tcp"
                    }
                else:
                    return {
                        "port": port,
                        "status": "closed",
                        "protocol": "tcp"
                    }
            else:
                # UDP scan (basic implementation)
                return {
                    "port": port,
                    "status": "filtered",
                    "protocol": "udp",
                    "note": "UDP scanning requires advanced techniques"
                }
                
        except Exception as e:
            return {
                "port": port,
                "status": "error",
                "error": str(e)
            }
    
    def _identify_service(self, port: int) -> str:
        """Identify common services by port number."""
        common_services = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            1433: "MSSQL",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            27017: "MongoDB"
        }
        
        return common_services.get(port, "Unknown")
    
    def _analyze_port_scan_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze port scan results and provide security insights."""
        analysis = {
            "security_assessment": "unknown",
            "vulnerability_analysis": [],
            "service_analysis": {},
            "recommendations": [],
            "risk_level": "unknown"
        }
        
        open_ports = results.get("open_ports", [])
        total_ports = results.get("total_ports", 0)
        
        # Security assessment
        if len(open_ports) == 0:
            analysis["security_assessment"] = "excellent"
            analysis["risk_level"] = "low"
        elif len(open_ports) <= 5:
            analysis["security_assessment"] = "good"
            analysis["risk_level"] = "low"
        elif len(open_ports) <= 10:
            analysis["security_assessment"] = "fair"
            analysis["risk_level"] = "medium"
        else:
            analysis["security_assessment"] = "poor"
            analysis["risk_level"] = "high"
        
        # Analyze open ports for vulnerabilities
        for port_info in open_ports:
            port = port_info.get("port")
            service = port_info.get("service", "Unknown")
            
            # Check for high-risk services
            if port in [21, 23, 25, 110, 143]:  # Unencrypted services
                analysis["vulnerability_analysis"].append({
                    "port": port,
                    "service": service,
                    "risk": "high",
                    "issue": "Unencrypted service - data transmitted in plain text"
                })
            
            elif port in [80]:  # HTTP
                analysis["vulnerability_analysis"].append({
                    "port": port,
                    "service": service,
                    "risk": "medium",
                    "issue": "HTTP service - consider upgrading to HTTPS"
                })
            
            elif port in [22, 443]:  # Secure services
                analysis["vulnerability_analysis"].append({
                    "port": port,
                    "service": service,
                    "risk": "low",
                    "issue": "Secure service - good security practice"
                })
            
            # Service analysis
            if service not in analysis["service_analysis"]:
                analysis["service_analysis"][service] = 0
            analysis["service_analysis"][service] += 1
        
        # Generate recommendations
        if analysis["risk_level"] == "high":
            analysis["recommendations"].extend([
                "Review and close unnecessary open ports",
                "Implement firewall rules to restrict access",
                "Consider using a DMZ for public services"
            ])
        
        if analysis["vulnerability_analysis"]:
            high_risk_ports = [v for v in analysis["vulnerability_analysis"] if v["risk"] == "high"]
            if high_risk_ports:
                analysis["recommendations"].append("Immediately address high-risk vulnerabilities")
        
        return analysis

# Network Tools Manager for integration with Query Path and Runner Agent
class NetworkToolsManager:
    """Manager for all network tools with MCP integration capabilities."""
    
    def __init__(self):
        self.tools = {
            "network_ping": NetworkPingTool(),
            "network_dns_lookup": NetworkDNSTool(),
            "network_netstat": NetworkNetstatTool(),
            "network_arp": NetworkARPTool(),
            "network_traceroute": NetworkTracerouteTool(),
            "network_port_scanner": NetworkPortScannerTool()
        }
        
        self.tool_categories = {
            NetworkToolCategory.CONNECTIVITY: ["network_ping", "network_traceroute"],
            NetworkToolCategory.DNS: ["network_dns_lookup"],
            NetworkToolCategory.STATISTICS: ["network_netstat", "network_arp"],
            NetworkToolCategory.SECURITY: ["network_port_scanner"],
            NetworkToolCategory.MONITORING: ["network_ping", "network_netstat"],
            NetworkToolCategory.DIAGNOSTICS: ["network_traceroute", "network_dns_lookup"]
        }
        
        logger.info("🚀 Network Tools Manager initialized")
    
    def get_tool(self, tool_id: str) -> Optional[Any]:
        """Get a specific tool by ID."""
        return self.tools.get(tool_id)
    
    def get_tools_by_category(self, category: NetworkToolCategory) -> List[str]:
        """Get tools by category."""
        return self.tool_categories.get(category, [])
    
    def get_tools_by_capability(self, capability: NetworkToolCapability) -> List[str]:
        """Get tools by capability."""
        matching_tools = []
        for tool_id, tool in self.tools.items():
            if capability in tool.metadata.capabilities:
                matching_tools.append(tool_id)
        return matching_tools
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all tools with metadata."""
        return {tool_id: tool.metadata for tool_id, tool in self.tools.items()}
    
    def get_tool_metadata(self, tool_id: str) -> Optional[NetworkToolMetadata]:
        """Get tool metadata."""
        tool = self.tools.get(tool_id)
        return tool.metadata if tool else None
    
    async def execute_tool(self, tool_id: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool."""
        tool = self.tools.get(tool_id)
        if not tool:
            return {"success": False, "error": f"Tool not found: {tool_id}"}
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "tools_by_category": {},
            "performance_summary": {
                "total_executions": 0,
                "avg_success_rate": 0.0,
                "most_used_tool": None,
                "best_performing_tool": None
            }
        }
        
        total_executions = 0
        total_success_rate = 0.0
        tool_performance = {}
        
        for tool_id, tool in self.tools.items():
            # Category breakdown
            category = tool.metadata.category.value
            if category not in stats["tools_by_category"]:
                stats["tools_by_category"][category] = []
            stats["tools_by_category"][category].append(tool_id)
            
            # Performance metrics
            executions = tool.metadata.usage_count
            success_rate = tool.metadata.success_rate
            total_executions += executions
            total_success_rate += success_rate
            
            tool_performance[tool_id] = {
                "executions": executions,
                "success_rate": success_rate,
                "avg_execution_time": tool.metadata.performance_metrics.get("avg_execution_time", 0)
            }
        
        # Calculate summary statistics
        if self.tools:
            stats["performance_summary"]["total_executions"] = total_executions
            stats["performance_summary"]["avg_success_rate"] = total_success_rate / len(self.tools)
            
            # Find most used tool
            most_used = max(tool_performance.items(), key=lambda x: x[1]["executions"])
            stats["performance_summary"]["most_used_tool"] = most_used[0]
            
            # Find best performing tool
            best_performing = max(tool_performance.items(), key=lambda x: x[1]["success_rate"])
            stats["performance_summary"]["best_performing_tool"] = best_performing[0]
        
        stats["tool_performance"] = tool_performance
        return stats

# Example usage and testing
async def main():
    """Example usage of the Network Tools Manager."""
    manager = NetworkToolsManager()
    
    print("🚀 Network Tools Manager Test")
    print("=" * 50)
    
    # Show available tools
    print("\n📋 Available Tools:")
    for tool_id, metadata in manager.get_all_tools().items():
        print(f"  {tool_id}: {metadata.name}")
        print(f"    Category: {metadata.category.value}")
        print(f"    Capabilities: {[cap.value for cap in metadata.capabilities]}")
    
    # Test ping tool
    print("\n🔍 Testing Ping Tool:")
    ping_result = await manager.execute_tool("network_ping", host="8.8.8.8", count=3)
    if ping_result["success"]:
        print(f"  ✅ Ping successful to 8.8.8.8")
        print(f"  📊 Execution time: {ping_result['execution_time']:.3f}s")
        print(f"  📈 Results: {ping_result['results']['statistics']}")
    else:
        print(f"  ❌ Ping failed: {ping_result['error']}")
    
    # Test DNS lookup
    print("\n🔍 Testing DNS Lookup:")
    dns_result = await manager.execute_tool("network_dns_lookup", hostname="google.com")
    if dns_result["success"]:
        print(f"  ✅ DNS lookup successful for google.com")
        print(f"  📊 Execution time: {dns_result['execution_time']:.3f}s")
        print(f"  📈 Answers: {len(dns_result['results']['answers'])}")
    else:
        print(f"  ❌ DNS lookup failed: {dns_result['error']}")
    
    # Show performance statistics
    print("\n📊 Performance Statistics:")
    stats = manager.get_performance_stats()
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Total executions: {stats['performance_summary']['total_executions']}")
    print(f"  Average success rate: {stats['performance_summary']['avg_success_rate']:.2%}")
    print(f"  Most used tool: {stats['performance_summary']['most_used_tool']}")
    print(f"  Best performing tool: {stats['performance_summary']['best_performing_tool']}")

if __name__ == "__main__":
    asyncio.run(main())
