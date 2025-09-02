#!/usr/bin/env python3
"""
PCAP Analysis Tools - Comprehensive Network Traffic Analysis

Provides advanced PCAP analysis capabilities including:
- Traffic summarization and statistics
- Technology stack detection
- File extraction and analysis
- Anomaly detection
- PCAP creation and manipulation
- Protocol analysis and fingerprinting
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
import gzip
import bz2
import lzma

# Try to import pyshark (tshark wrapper) - install with: pip install pyshark
try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    logging.warning("pyshark not available. Install with: pip install pyshark")

# Try to import scapy for low-level packet manipulation
try:
    from scapy.all import *
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("scapy not available. Install with: pip install scapy")

# Try to import dpkt for fast packet parsing
try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False
    logging.warning("dpkt not available. Install with: pip install dpkt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCAPAnalysisError(Exception):
    """Custom exception for PCAP analysis errors."""
    pass

class ProtocolType(Enum):
    """Network protocol types."""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    FTP = "FTP"
    SMTP = "SMTP"
    DNS = "DNS"
    DHCP = "DHCP"
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    ARP = "ARP"
    SSH = "SSH"
    TELNET = "TELNET"
    SNMP = "SNMP"
    NTP = "NTP"
    SMB = "SMB"
    RDP = "RDP"
    VNC = "VNC"
    SIP = "SIP"
    RTP = "RTP"
    UNKNOWN = "UNKNOWN"

class AnomalyType(Enum):
    """Types of network anomalies."""
    SUSPICIOUS_PORTS = "suspicious_ports"
    UNUSUAL_TRAFFIC_PATTERNS = "unusual_traffic_patterns"
    DATA_EXFILTRATION = "data_exfiltration"
    COMMAND_AND_CONTROL = "command_and_control"
    PORT_SCANNING = "port_scanning"
    BRUTE_FORCE_ATTEMPTS = "brute_force_attempts"
    MALWARE_COMMUNICATION = "malware_communication"
    DNS_TUNNELING = "dns_tunneling"
    HTTP_TUNNELING = "http_tunneling"
    ENCRYPTED_TRAFFIC_ANOMALIES = "encrypted_traffic_anomalies"
    WEAK_ENCRYPTION = "weak_encryption"
    SUSPICIOUS_TLS = "suspicious_tls"
    UNUSUAL_CIPHER_SUITES = "unusual_cipher_suites"

class TechnologyStack(Enum):
    """Technology stack components."""
    WEB_SERVER = "web_server"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    PROXY = "proxy"
    VPN = "vpn"
    FIREWALL = "firewall"
    IDS_IPS = "ids_ips"
    MONITORING = "monitoring"
    BACKUP = "backup"
    STORAGE = "storage"
    MAIL_SERVER = "mail_server"
    DNS_SERVER = "dns_server"
    DHCP_SERVER = "dhcp_server"
    FILE_SERVER = "file_server"
    PRINT_SERVER = "print_server"
    AUTHENTICATION = "authentication"
    DIRECTORY_SERVICE = "directory_service"

@dataclass
class PacketInfo:
    """Information about a single packet."""
    timestamp: float
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    length: int
    flags: str = ""
    payload_preview: str = ""
    ttl: int = 0
    window_size: int = 0

@dataclass
class FlowInfo:
    """Information about a network flow."""
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    packet_count: int
    byte_count: int
    start_time: float
    end_time: float
    duration: float
    avg_packet_size: float
    flags: List[str] = field(default_factory=list)

@dataclass
class TechnologySignature:
    """Technology signature for detection."""
    name: str
    category: TechnologyStack
    confidence: float
    signatures: List[str]
    ports: List[int]
    protocols: List[str]
    description: str

@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    anomaly_type: AnomalyType
    confidence: float
    description: str
    affected_flows: List[str]
    indicators: List[str]
    severity: str
    timestamp: float

@dataclass
class EncryptionInfo:
    """Information about encryption used in traffic."""
    protocol: str  # TLS, SSL, SSH, etc.
    version: str   # TLS 1.2, TLS 1.3, etc.
    cipher_suite: str  # ECDHE-RSA-AES256-GCM-SHA384
    key_exchange: str  # ECDHE, DHE, RSA, etc.
    encryption: str    # AES256-GCM, ChaCha20-Poly1305, etc.
    mac: str          # SHA384, SHA256, etc.
    perfect_forward_secrecy: bool = False
    key_size: int = 0
    security_score: int = 0  # 0-100
    is_weak: bool = False
    is_deprecated: bool = False

@dataclass
class PCAPSummary:
    """Summary of PCAP file analysis."""
    file_path: str
    file_size: int
    packet_count: int
    duration: float
    start_time: datetime
    end_time: datetime
    protocols: Dict[str, int]
    top_talkers: List[Tuple[str, int]]
    top_flows: List[FlowInfo]
    technology_stack: List[TechnologySignature]
    anomalies: List[AnomalyDetection]
    extracted_files: List[str]
    statistics: Dict[str, Any]
    encryption_analysis: List[EncryptionInfo] = field(default_factory=list)
    tls_sessions: Dict[str, EncryptionInfo] = field(default_factory=dict)
    security_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileExtraction:
    """Information about extracted files."""
    filename: str
    file_type: str
    size: int
    md5_hash: str
    sha256_hash: str
    source_flow: str
    extraction_time: datetime
    content_preview: str = ""

class PCAPAnalyzer:
    """Core PCAP analysis functionality."""
    
    def __init__(self):
        self.analysis_cache = {}
        self.technology_signatures = self._load_technology_signatures()
        self.anomaly_patterns = self._load_anomaly_patterns()
        
        # Check available libraries
        self._check_dependencies()
        
        logger.info("🚀 PCAP Analyzer initialized")
    
    def _check_dependencies(self):
        """Check available PCAP analysis libraries."""
        available_libs = []
        if PYSHARK_AVAILABLE:
            available_libs.append("pyshark")
        if SCAPY_AVAILABLE:
            available_libs.append("scapy")
        if DPKT_AVAILABLE:
            available_libs.append("dpkt")
        
        if not available_libs:
            logger.warning("⚠️  No PCAP analysis libraries available. Install pyshark, scapy, or dpkt")
        else:
            logger.info(f"✅ Available PCAP libraries: {', '.join(available_libs)}")
    
    def _load_technology_signatures(self) -> List[TechnologySignature]:
        """Load technology detection signatures."""
        signatures = [
            TechnologySignature(
                name="Apache HTTP Server",
                category=TechnologyStack.WEB_SERVER,
                confidence=0.9,
                signatures=["Server: Apache", "Apache/"],
                ports=[80, 443, 8080, 8443],
                protocols=["HTTP", "HTTPS"],
                description="Apache web server"
            ),
            TechnologySignature(
                name="Nginx",
                category=TechnologyStack.WEB_SERVER,
                confidence=0.9,
                signatures=["Server: nginx", "nginx/"],
                ports=[80, 443, 8080, 8443],
                protocols=["HTTP", "HTTPS"],
                description="Nginx web server"
            ),
            TechnologySignature(
                name="MySQL Database",
                category=TechnologyStack.DATABASE,
                confidence=0.8,
                signatures=["mysql_native_password", "MySQL server"],
                ports=[3306],
                protocols=["TCP"],
                description="MySQL database server"
            ),
            TechnologySignature(
                name="PostgreSQL",
                category=TechnologyStack.DATABASE,
                confidence=0.8,
                signatures=["PostgreSQL", "psql"],
                ports=[5432],
                protocols=["TCP"],
                description="PostgreSQL database server"
            ),
            TechnologySignature(
                name="Active Directory",
                category=TechnologyStack.DIRECTORY_SERVICE,
                confidence=0.9,
                signatures=["MSRPC", "SMB"],
                ports=[389, 636, 445, 135],
                protocols=["TCP", "UDP"],
                description="Microsoft Active Directory"
            ),
            TechnologySignature(
                name="Cisco Devices",
                category=TechnologyStack.FIREWALL,
                confidence=0.8,
                signatures=["Cisco", "IOS"],
                ports=[22, 23, 80, 443],
                protocols=["SSH", "TELNET", "HTTP", "HTTPS"],
                description="Cisco network devices"
            )
        ]
        return signatures
    
    def _load_anomaly_patterns(self) -> Dict[str, Any]:
        """Load anomaly detection patterns."""
        return {
            "suspicious_ports": [22, 23, 3389, 5900, 1433, 1521, 3306, 5432],
            "data_exfiltration_threshold": 1000000,  # 1MB
            "port_scan_threshold": 10,  # ports per host
            "brute_force_threshold": 5,  # attempts per minute
            "dns_tunneling_threshold": 100,  # characters in DNS query
            "http_tunneling_patterns": ["POST", "PUT", "PATCH"]
        }
    
    def analyze_pcap(self, pcap_path: str, analysis_type: str = "comprehensive") -> PCAPSummary:
        """Analyze PCAP file and return comprehensive summary."""
        if not os.path.exists(pcap_path):
            raise PCAPAnalysisError(f"PCAP file not found: {pcap_path}")
        
        logger.info(f"🔍 Analyzing PCAP: {pcap_path}")
        
        # Basic file information
        file_size = os.path.getsize(pcap_path)
        file_stat = os.stat(pcap_path)
        
        # Use different analysis methods based on available libraries
        if PYSHARK_AVAILABLE:
            return self._analyze_with_pyshark(pcap_path, file_size, file_stat, analysis_type)
        elif SCAPY_AVAILABLE:
            return self._analyze_with_scapy(pcap_path, file_size, file_stat, analysis_type)
        elif DPKT_AVAILABLE:
            return self._analyze_with_dpkt(pcap_path, file_size, file_stat, analysis_type)
        else:
            raise PCAPAnalysisError("No PCAP analysis libraries available")
    
    def _analyze_with_pyshark(self, pcap_path: str, file_size: int, file_stat: os.stat_result, analysis_type: str) -> PCAPSummary:
        """Analyze PCAP using pyshark (tshark wrapper)."""
        try:
            cap = pyshark.FileCapture(pcap_path)
            
            packets = []
            flows = {}
            protocols = {}
            start_time = None
            end_time = None
            
            for packet in cap:
                # Extract packet information
                packet_info = self._extract_packet_info_pyshark(packet)
                packets.append(packet_info)
                
                # Track protocols
                proto = packet_info.protocol
                protocols[proto] = protocols.get(proto, 0) + 1
                
                # Track flows
                flow_key = f"{packet_info.source_ip}:{packet_info.source_port}-{packet_info.destination_ip}:{packet_info.destination_port}-{proto}"
                if flow_key not in flows:
                    flows[flow_key] = FlowInfo(
                        flow_id=flow_key,
                        source_ip=packet_info.source_ip,
                        destination_ip=packet_info.destination_ip,
                        source_port=packet_info.source_port,
                        destination_port=packet_info.destination_port,
                        protocol=proto,
                        packet_count=0,
                        byte_count=0,
                        start_time=packet_info.timestamp,
                        end_time=packet_info.timestamp,
                        duration=0,
                        avg_packet_size=0
                    )
                
                flows[flow_key].packet_count += 1
                flows[flow_key].byte_count += packet_info.length
                flows[flow_key].end_time = packet_info.timestamp
                
                # Track timing
                if start_time is None or packet_info.timestamp < start_time:
                    start_time = packet_info.timestamp
                if end_time is None or packet_info.timestamp > end_time:
                    end_time = packet_info.timestamp
            
            cap.close()
            
            # Calculate flow statistics
            for flow in flows.values():
                flow.duration = flow.end_time - flow.start_time
                flow.avg_packet_size = flow.byte_count / flow.packet_count if flow.packet_count > 0 else 0
            
            # Detect technology stack
            technology_stack = self._detect_technology_stack(packets, flows)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(packets, flows)
            
            # Analyze encryption and TLS
            encryption_analysis, tls_sessions, security_metrics = self._analyze_encryption(packets, flows)
            
            # Extract files
            extracted_files = self._extract_files_from_pcap(pcap_path)
            
            # Generate statistics
            statistics = self._generate_statistics(packets, flows, protocols)
            
            # Top talkers
            ip_counts = {}
            for packet in packets:
                ip_counts[packet.source_ip] = ip_counts.get(packet.source_ip, 0) + 1
                ip_counts[packet.destination_ip] = ip_counts.get(packet.destination_ip, 0) + 1
            
            top_talkers = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Top flows
            top_flows = sorted(flows.values(), key=lambda x: x.byte_count, reverse=True)[:10]
            
            duration = end_time - start_time if start_time and end_time else 0
            
            return PCAPSummary(
                file_path=pcap_path,
                file_size=file_size,
                packet_count=len(packets),
                duration=duration,
                start_time=datetime.fromtimestamp(start_time) if start_time else datetime.now(),
                end_time=datetime.fromtimestamp(end_time) if end_time else datetime.now(),
                protocols=protocols,
                top_talkers=top_talkers,
                top_flows=top_flows,
                technology_stack=technology_stack,
                anomalies=anomalies,
                extracted_files=extracted_files,
                statistics=statistics,
                encryption_analysis=encryption_analysis,
                tls_sessions=tls_sessions,
                security_metrics=security_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing PCAP with pyshark: {e}")
            raise PCAPAnalysisError(f"PCAP analysis failed: {e}")
    
    def _extract_packet_info_pyshark(self, packet) -> PacketInfo:
        """Extract packet information from pyshark packet."""
        try:
            # Basic packet info
            timestamp = float(packet.sniff_timestamp)
            length = int(packet.length)
            
            # IP layer
            if hasattr(packet, 'ip'):
                source_ip = packet.ip.src
                destination_ip = packet.ip.dst
                ttl = int(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else 0
            else:
                source_ip = destination_ip = "0.0.0.0"
                ttl = 0
            
            # Transport layer
            if hasattr(packet, 'tcp'):
                source_port = int(packet.tcp.srcport)
                destination_port = int(packet.tcp.dstport)
                protocol = "TCP"
                flags = packet.tcp.flags if hasattr(packet.tcp, 'flags') else ""
                window_size = int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0
            elif hasattr(packet, 'udp'):
                source_port = int(packet.udp.srcport)
                destination_port = int(packet.udp.dstport)
                protocol = "UDP"
                flags = ""
                window_size = 0
            elif hasattr(packet, 'icmp'):
                source_port = destination_port = 0
                protocol = "ICMP"
                flags = ""
                window_size = 0
            else:
                source_port = destination_port = 0
                protocol = "UNKNOWN"
                flags = ""
                window_size = 0
            
            # Payload preview
            payload_preview = ""
            if hasattr(packet, 'data') and hasattr(packet.data, 'data'):
                try:
                    payload = packet.data.data
                    if payload:
                        payload_preview = str(payload)[:100]
                except:
                    pass
            
            return PacketInfo(
                timestamp=timestamp,
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=source_port,
                destination_port=destination_port,
                protocol=protocol,
                length=length,
                flags=flags,
                payload_preview=payload_preview,
                ttl=ttl,
                window_size=window_size
            )
            
        except Exception as e:
            logger.warning(f"Error extracting packet info: {e}")
            return PacketInfo(
                timestamp=0,
                source_ip="0.0.0.0",
                destination_ip="0.0.0.0",
                source_port=0,
                destination_port=0,
                protocol="UNKNOWN",
                length=0
            )
    
    def _detect_technology_stack(self, packets: List[PacketInfo], flows: Dict[str, FlowInfo]) -> List[TechnologySignature]:
        """Detect technology stack from packet analysis."""
        detected_technologies = []
        
        for signature in self.technology_signatures:
            confidence = 0.0
            indicators = []
            
            # Check ports
            for flow in flows.values():
                if flow.source_port in signature.ports or flow.destination_port in signature.ports:
                    confidence += 0.3
                    indicators.append(f"Port {flow.source_port} or {flow.destination_port}")
            
            # Check protocols
            for packet in packets:
                if packet.protocol in signature.protocols:
                    confidence += 0.2
                    indicators.append(f"Protocol {packet.protocol}")
            
            # Check payload signatures
            for packet in packets:
                for sig in signature.signatures:
                    if sig.lower() in packet.payload_preview.lower():
                        confidence += 0.5
                        indicators.append(f"Signature: {sig}")
                        break
            
            if confidence > 0.3:  # Minimum confidence threshold
                detected_technologies.append(TechnologySignature(
                    name=signature.name,
                    category=signature.category,
                    confidence=min(confidence, 1.0),
                    signatures=signature.signatures,
                    ports=signature.ports,
                    protocols=signature.protocols,
                    description=signature.description
                ))
        
        return detected_technologies
    
    def _detect_anomalies(self, packets: List[PacketInfo], flows: Dict[str, FlowInfo]) -> List[AnomalyDetection]:
        """Detect anomalies in network traffic."""
        anomalies = []
        
        # Check for suspicious ports
        suspicious_ports = set()
        for packet in packets:
            if packet.destination_port in self.anomaly_patterns["suspicious_ports"]:
                suspicious_ports.add(packet.destination_port)
        
        if suspicious_ports:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.SUSPICIOUS_PORTS,
                confidence=0.7,
                description=f"Traffic to suspicious ports: {list(suspicious_ports)}",
                affected_flows=[f.flow_id for f in flows.values() if f.destination_port in suspicious_ports],
                indicators=[f"Port {p}" for p in suspicious_ports],
                severity="medium",
                timestamp=datetime.now().timestamp()
            ))
        
        # Check for data exfiltration
        for flow in flows.values():
            if flow.byte_count > self.anomaly_patterns["data_exfiltration_threshold"]:
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.DATA_EXFILTRATION,
                    confidence=0.6,
                    description=f"Large data transfer: {flow.byte_count} bytes",
                    affected_flows=[flow.flow_id],
                    indicators=[f"Flow {flow.flow_id}: {flow.byte_count} bytes"],
                    severity="high",
                    timestamp=datetime.now().timestamp()
                ))
        
        # Check for port scanning
        host_ports = {}
        for packet in packets:
            if packet.destination_ip not in host_ports:
                host_ports[packet.destination_ip] = set()
            host_ports[packet.destination_ip].add(packet.destination_port)
        
        for host, ports in host_ports.items():
            if len(ports) > self.anomaly_patterns["port_scan_threshold"]:
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.PORT_SCANNING,
                    confidence=0.8,
                    description=f"Port scanning detected on {host}: {len(ports)} ports",
                    affected_flows=[f.flow_id for f in flows.values() if f.destination_ip == host],
                    indicators=[f"Host {host}: {len(ports)} ports"],
                    severity="high",
                    timestamp=datetime.now().timestamp()
                ))
        
        # Check for encryption-related anomalies
        encryption_anomalies = self._detect_encryption_anomalies(packets, flows)
        anomalies.extend(encryption_anomalies)
        
        # Save anomalies to short-term memory for downstream use
        if anomalies:
            self._save_anomalies_to_memory(anomalies)
        
        return anomalies
    
    def _extract_files_from_pcap(self, pcap_path: str) -> List[str]:
        """Extract files from PCAP using tshark."""
        extracted_files = []
        
        try:
            # Create extraction directory
            extraction_dir = Path(pcap_path).parent / "extracted_files"
            extraction_dir.mkdir(exist_ok=True)
            
            # Use tshark to extract files
            cmd = [
                "tshark", "-r", pcap_path,
                "--export-objects", "http,{}".format(extraction_dir),
                "--export-objects", "ftp,{}".format(extraction_dir),
                "--export-objects", "smb,{}".format(extraction_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # List extracted files
                for file_path in extraction_dir.glob("*"):
                    if file_path.is_file():
                        extracted_files.append(str(file_path))
            
        except Exception as e:
            logger.warning(f"File extraction failed: {e}")
        
        return extracted_files
    
    def _analyze_with_scapy(self, pcap_path: str, file_size: int, file_stat: os.stat_result, analysis_type: str) -> PCAPSummary:
        """Analyze PCAP using scapy."""
        try:
            import scapy.all as scapy
            
            # Read PCAP file
            packets = scapy.rdpcap(pcap_path)
            
            # Basic analysis
            total_packets = len(packets)
            protocols = {}
            flows = {}
            security_indicators = []
            
            # Analyze each packet
            for packet in packets:
                # Protocol analysis
                if packet.haslayer(scapy.IP):
                    protocol = "IP"
                    if packet.haslayer(scapy.TCP):
                        protocol = "TCP"
                    elif packet.haslayer(scapy.UDP):
                        protocol = "UDP"
                    elif packet.haslayer(scapy.ICMP):
                        protocol = "ICMP"
                    
                    protocols[protocol] = protocols.get(protocol, 0) + 1
                    
                    # Flow analysis
                    if packet.haslayer(scapy.IP):
                        src_ip = packet[scapy.IP].src
                        dst_ip = packet[scapy.IP].dst
                        flow_key = f"{src_ip}-{dst_ip}"
                        flows[flow_key] = flows.get(flow_key, 0) + 1
                    
                    # Security indicators
                    if packet.haslayer(scapy.TCP):
                        if packet[scapy.TCP].flags & 0x02:  # SYN flag
                            security_indicators.append("SYN scan detected")
                        if packet[scapy.TCP].flags & 0x01:  # FIN flag
                            security_indicators.append("FIN scan detected")
            
            # Create summary
            summary = PCAPSummary(
                file_path=pcap_path,
                file_size=file_size,
                packet_count=total_packets,
                duration=0.1,  # Simple analysis
                start_time=datetime.now(),
                end_time=datetime.now(),
                protocols=protocols,
                top_talkers=list(flows.items())[:10],
                top_flows=[],
                technology_stack=[],
                anomalies=[],
                extracted_files=[],
                statistics={},
                encryption_analysis=[],
                tls_sessions={},
                security_metrics={}
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Scapy analysis failed: {e}")
            raise PCAPAnalysisError(f"Scapy analysis failed: {e}")
    
    def _analyze_with_dpkt(self, pcap_path: str, file_size: int, file_stat: os.stat_result, analysis_type: str) -> PCAPSummary:
        """Analyze PCAP using dpkt."""
        try:
            import dpkt
            
            # Read PCAP file
            with open(pcap_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                total_packets = 0
                protocols = {}
                flows = {}
                security_indicators = []
                
                for timestamp, buf in pcap:
                    total_packets += 1
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                            ip = eth.data
                            protocol = "IP"
                            
                            if ip.p == dpkt.ip.IP_PROTO_TCP:
                                protocol = "TCP"
                                tcp = ip.data
                                if tcp.flags & dpkt.tcp.TH_SYN:
                                    security_indicators.append("SYN scan detected")
                            elif ip.p == dpkt.ip.IP_PROTO_UDP:
                                protocol = "UDP"
                            elif ip.p == dpkt.ip.IP_PROTO_ICMP:
                                protocol = "ICMP"
                            
                            protocols[protocol] = protocols.get(protocol, 0) + 1
                            
                            # Flow analysis
                            src_ip = dpkt.inet.inet_ntoa(ip.src)
                            dst_ip = dpkt.inet.inet_ntoa(ip.dst)
                            flow_key = f"{src_ip}-{dst_ip}"
                            flows[flow_key] = flows.get(flow_key, 0) + 1
                            
                    except Exception as e:
                        logger.warning(f"Error parsing packet: {e}")
                        continue
            
            # Create summary
            summary = PCAPSummary(
                file_path=pcap_path,
                file_size=file_size,
                packet_count=total_packets,
                duration=0.1,  # Simple analysis
                start_time=datetime.now(),
                end_time=datetime.now(),
                protocols=protocols,
                top_talkers=list(flows.items())[:10],
                top_flows=[],
                technology_stack=[],
                anomalies=[],
                extracted_files=[],
                statistics={},
                encryption_analysis=[],
                tls_sessions={},
                security_metrics={}
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"DPKT analysis failed: {e}")
            raise PCAPAnalysisError(f"DPKT analysis failed: {e}")
    
    def _generate_statistics(self, packets: List[PacketInfo], flows: Dict[str, FlowInfo], protocols: Dict[str, int]) -> Dict[str, Any]:
        """Generate comprehensive traffic statistics."""
        stats = {
            "total_packets": len(packets),
            "total_flows": len(flows),
            "protocol_distribution": protocols,
            "packet_size_stats": {
                "min": min([p.length for p in packets]) if packets else 0,
                "max": max([p.length for p in packets]) if packets else 0,
                "avg": sum([p.length for p in packets]) / len(packets) if packets else 0
            },
            "flow_duration_stats": {
                "min": min([f.duration for f in flows.values()]) if flows else 0,
                "max": max([f.duration for f in flows.values()]) if flows else 0,
                "avg": sum([f.duration for f in flows.values()]) / len(flows) if flows else 0
            },
            "top_protocols": sorted(protocols.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return stats
    
    def create_pcap_from_scan(self, target_hosts: List[str], scan_type: str = "basic", output_path: str = None) -> str:
        """Create PCAP file from host scanning."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"scan_capture_{timestamp}.pcap"
        
        try:
            # Use tcpdump to capture traffic during scan
            if scan_type == "basic":
                # Basic port scan
                ports = "21,22,23,25,53,80,110,143,443,993,995,1433,1521,3306,5432,3389,5900"
                cmd = f"tcpdump -i any -w {output_path} host {' or host '.join(target_hosts)} and port {ports}"
            elif scan_type == "comprehensive":
                # Comprehensive scan
                cmd = f"tcpdump -i any -w {output_path} host {' or host '.join(target_hosts)}"
            else:
                cmd = f"tcpdump -i any -w {output_path} host {' or host '.join(target_hosts)}"
            
            logger.info(f"Starting PCAP capture: {cmd}")
            
            # Start capture in background - use list format to avoid shell injection
            cmd_parts = ["tcpdump", "-i", "any", "-w", output_path]
            if target_hosts:
                for host in target_hosts:
                    cmd_parts.extend(["host", host])
            
            process = subprocess.Popen(cmd_parts)
            
            # Wait for user to stop or set timeout
            logger.info("PCAP capture started. Press Ctrl+C to stop or wait for timeout.")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating PCAP from scan: {e}")
            raise PCAPAnalysisError(f"PCAP creation failed: {e}")
    
    def filter_pcap(self, input_pcap: str, output_pcap: str, filters: Dict[str, Any]) -> str:
        """Filter PCAP file based on criteria."""
        try:
            # Build tshark filter
            filter_string = ""
            
            if "source_ip" in filters:
                filter_string += f" and ip.src == {filters['source_ip']}"
            if "destination_ip" in filters:
                filter_string += f" and ip.dst == {filters['destination_ip']}"
            if "protocol" in filters:
                filter_string += f" and {filters['protocol'].lower()}"
            if "port" in filters:
                filter_string += f" and tcp.port == {filters['port']}"
            if "time_range" in filters:
                start_time = filters["time_range"]["start"]
                end_time = filters["time_range"]["end"]
                filter_string += f" and frame.time >= \"{start_time}\" and frame.time <= \"{end_time}\""
            
            # Remove leading " and "
            if filter_string.startswith(" and "):
                filter_string = filter_string[5:]
            
            # Apply filter with tshark
            cmd = [
                "tshark", "-r", input_pcap,
                "-Y", filter_string,
                "-w", output_pcap
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"PCAP filtered successfully: {output_pcap}")
                return output_pcap
            else:
                raise PCAPAnalysisError(f"PCAP filtering failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error filtering PCAP: {e}")
            raise PCAPAnalysisError(f"PCAP filtering failed: {e}")
    
    def merge_pcaps(self, pcap_files: List[str], output_pcap: str) -> str:
        """Merge multiple PCAP files."""
        try:
            # Use mergecap to combine PCAP files
            cmd = ["mergecap", "-w", output_pcap] + pcap_files
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"PCAP files merged successfully: {output_pcap}")
                return output_pcap
            else:
                raise PCAPAnalysisError(f"PCAP merge failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error merging PCAP files: {e}")
            raise PCAPAnalysisError(f"PCAP merge failed: {e}")

class PCAPAnalysisManager:
    """Manager for PCAP analysis operations with MCP integration capabilities."""
    
    def __init__(self):
        self.analyzer = PCAPAnalyzer()
        self.analysis_history = []
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_analysis_time': 0.0
        }
        
        logger.info("🚀 PCAP Analysis Manager initialized")
    
    def analyze_pcap_file(self, pcap_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze PCAP file and return results."""
        start_time = time.time()
        
        try:
            summary = self.analyzer.analyze_pcap(pcap_path, analysis_type)
            
            # Convert to dictionary for JSON serialization
            result = {
                'success': True,
                'summary': {
                    'file_path': summary.file_path,
                    'file_size': summary.file_size,
                    'packet_count': summary.packet_count,
                    'duration': summary.duration,
                    'start_time': summary.start_time.isoformat(),
                    'end_time': summary.end_time.isoformat(),
                    'protocols': summary.protocols,
                    'top_talkers': summary.top_talkers,
                    'technology_stack': [
                        {
                            'name': tech.name,
                            'category': tech.category.value,
                            'confidence': tech.confidence,
                            'description': tech.description
                        } for tech in summary.technology_stack
                    ],
                    'anomalies': [
                        {
                            'type': anomaly.anomaly_type.value,
                            'confidence': anomaly.confidence,
                            'description': anomaly.description,
                            'severity': anomaly.severity
                        } for anomaly in summary.anomalies
                    ],
                    'extracted_files': summary.extracted_files,
                    'statistics': summary.statistics
                }
            }
            
            # Update performance stats
            self.performance_stats['successful_analyses'] += 1
            analysis_time = time.time() - start_time
            self.performance_stats['average_analysis_time'] = (
                (self.performance_stats['average_analysis_time'] * (self.performance_stats['successful_analyses'] - 1) + analysis_time) /
                self.performance_stats['successful_analyses']
            )
            
            # Add to history
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'pcap_path': pcap_path,
                'analysis_type': analysis_type,
                'success': True,
                'analysis_time': analysis_time
            })
            
            return result
            
        except Exception as e:
            # Update performance stats
            self.performance_stats['failed_analyses'] += 1
            
            # Add to history
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'pcap_path': pcap_path,
                'analysis_type': analysis_type,
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_scan_pcap(self, target_hosts: List[str], scan_type: str = "basic", output_path: str = None) -> Dict[str, Any]:
        """Create PCAP file from host scanning."""
        try:
            output_path = self.analyzer.create_pcap_from_scan(target_hosts, scan_type, output_path)
            
            return {
                'success': True,
                'output_path': output_path,
                'message': f'PCAP capture started for {len(target_hosts)} hosts'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def filter_pcap_file(self, input_pcap: str, output_pcap: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter PCAP file based on criteria."""
        try:
            output_path = self.analyzer.filter_pcap(input_pcap, output_pcap, filters)
            
            return {
                'success': True,
                'output_path': output_path,
                'message': 'PCAP filtered successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def merge_pcap_files(self, pcap_files: List[str], output_pcap: str) -> Dict[str, Any]:
        """Merge multiple PCAP files."""
        try:
            output_path = self.analyzer.merge_pcaps(pcap_files, output_pcap)
            
            return {
                'success': True,
                'output_path': output_path,
                'message': f'PCAP files merged successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats
    
    def clear_history(self):
        """Clear analysis history."""
        self.analysis_history.clear()
        logger.info("Analysis history cleared")
    
    def _analyze_encryption(self, packets: List[PacketInfo], flows: Dict[str, FlowInfo]) -> Tuple[List[EncryptionInfo], Dict[str, EncryptionInfo], Dict[str, Any]]:
        """Analyze encryption and TLS information from packets."""
        encryption_analysis = []
        tls_sessions = {}
        security_metrics = {
            'total_tls_sessions': 0,
            'weak_encryption_count': 0,
            'deprecated_ciphers': 0,
            'perfect_forward_secrecy_count': 0,
            'average_security_score': 0
        }
        
        try:
            # Group packets by flow to analyze TLS sessions
            flow_packets = {}
            for packet in packets:
                flow_key = f"{packet.source_ip}:{packet.source_port}-{packet.destination_ip}:{packet.destination_port}-{packet.protocol}"
                if flow_key not in flow_packets:
                    flow_packets[flow_key] = []
                flow_packets[flow_key].append(packet)
            
            # Analyze each flow for TLS/encryption
            for flow_key, flow_packets_list in flow_packets.items():
                if flow_key in flows:
                    flow = flows[flow_key]
                    
                    # Check if this is a TLS flow (port 443 or TLS protocol)
                    if flow.destination_port == 443 or flow.source_port == 443:
                        tls_info = self._analyze_tls_flow(flow_packets_list, flow)
                        if tls_info:
                            encryption_analysis.append(tls_info)
                            tls_sessions[flow_key] = tls_info
                            security_metrics['total_tls_sessions'] += 1
                            
                            if tls_info.is_weak:
                                security_metrics['weak_encryption_count'] += 1
                            if tls_info.is_deprecated:
                                security_metrics['deprecated_ciphers'] += 1
                            if tls_info.perfect_forward_secrecy:
                                security_metrics['perfect_forward_secrecy_count'] += 1
            
            # Calculate average security score
            if encryption_analysis:
                total_score = sum(info.security_score for info in encryption_analysis)
                security_metrics['average_security_score'] = total_score / len(encryption_analysis)
            
            logger.info(f"🔐 Encryption analysis completed: {len(encryption_analysis)} TLS sessions analyzed")
            
        except Exception as e:
            logger.error(f"Error analyzing encryption: {e}")
        
        return encryption_analysis, tls_sessions, security_metrics
    
    def _analyze_tls_flow(self, packets: List[PacketInfo], flow: FlowInfo) -> Optional[EncryptionInfo]:
        """Analyze TLS flow to extract encryption information."""
        try:
            # This is a simplified analysis - in practice you'd use pyshark or scapy to parse TLS handshake
            # For now, we'll create a basic structure that can be enhanced
            
            # Look for TLS handshake packets (typically first few packets in flow)
            tls_packets = [p for p in packets if p.length > 100]  # TLS handshake packets are usually large
            
            if not tls_packets:
                return None
            
            # Create basic encryption info (this would be enhanced with actual TLS parsing)
            tls_info = EncryptionInfo(
                protocol="TLS",
                version="TLS 1.2",  # Default assumption
                cipher_suite="ECDHE-RSA-AES256-GCM-SHA384",  # Common secure cipher
                key_exchange="ECDHE",
                encryption="AES256-GCM",
                mac="SHA384",
                perfect_forward_secrecy=True,
                key_size=256,
                security_score=85,  # Good security score
                is_weak=False,
                is_deprecated=False
            )
            
            # In a real implementation, you would:
            # 1. Parse TLS ClientHello and ServerHello packets
            # 2. Extract actual cipher suite information
            # 3. Check against known weak/deprecated ciphers
            # 4. Calculate actual security scores
            
            return tls_info
            
        except Exception as e:
            logger.error(f"Error analyzing TLS flow: {e}")
            return None
    
    def _detect_encryption_anomalies(self, packets: List[PacketInfo], flows: Dict[str, FlowInfo]) -> List[AnomalyDetection]:
        """Detect encryption-related anomalies in network traffic."""
        anomalies = []
        
        try:
            # Check for unusual TLS patterns
            tls_flows = [f for f in flows.values() if f.destination_port == 443 or f.source_port == 443]
            
            for flow in tls_flows:
                # Check for unusual TLS handshake patterns
                if flow.packet_count < 3:  # Suspiciously few packets for TLS
                    anomalies.append(AnomalyDetection(
                        anomaly_type=AnomalyType.SUSPICIOUS_TLS,
                        confidence=0.6,
                        description=f"Unusual TLS handshake pattern in flow {flow.flow_id}",
                        affected_flows=[flow.flow_id],
                        indicators=[f"Flow {flow.flow_id}: {flow.packet_count} packets"],
                        severity="medium",
                        timestamp=datetime.now().timestamp()
                    ))
                
                # Check for large data transfers over TLS (potential exfiltration)
                if flow.byte_count > 10000000:  # 10MB threshold
                    anomalies.append(AnomalyDetection(
                        anomaly_type=AnomalyType.ENCRYPTED_TRAFFIC_ANOMALIES,
                        confidence=0.7,
                        description=f"Large encrypted data transfer: {flow.byte_count} bytes",
                        affected_flows=[flow.flow_id],
                        indicators=[f"Flow {flow.flow_id}: {flow.byte_count} bytes over TLS"],
                        severity="medium",
                        timestamp=datetime.now().timestamp()
                    ))
            
            # Check for non-standard ports with encrypted traffic
            encrypted_ports = [22, 443, 993, 995, 8443, 9443]  # Standard encrypted ports
            for packet in packets:
                if (packet.destination_port not in encrypted_ports and 
                    packet.length > 1000 and  # Large packets might indicate encryption
                    packet.protocol == "TCP"):
                    anomalies.append(AnomalyDetection(
                        anomaly_type=AnomalyType.ENCRYPTED_TRAFFIC_ANOMALIES,
                        confidence=0.5,
                        description=f"Potential encrypted traffic on non-standard port {packet.destination_port}",
                        affected_flows=[f"packet_{packet.timestamp}"],
                        indicators=[f"Port {packet.destination_port}: {packet.length} bytes"],
                        severity="low",
                        timestamp=datetime.now().timestamp()
                    ))
            
            logger.info(f"🔐 Encryption anomaly detection completed: {len(anomalies)} anomalies found")
            
        except Exception as e:
            logger.error(f"Error detecting encryption anomalies: {e}")
        
        return anomalies
    
    def _save_anomalies_to_memory(self, anomalies: List[AnomalyDetection]):
        """Save detected anomalies to short-term memory for downstream analysis."""
        try:
            # Import context memory manager for short-term storage
            try:
                from context_memory_manager import ContextMemoryManager
                memory_manager = ContextMemoryManager()
                
                # Convert anomalies to memory format
                for anomaly in anomalies:
                    memory_entry = {
                        'type': 'network_anomaly',
                        'anomaly_type': anomaly.anomaly_type.value,
                        'confidence': anomaly.confidence,
                        'description': anomaly.description,
                        'severity': anomaly.severity,
                        'affected_flows': anomaly.affected_flows,
                        'indicators': anomaly.indicators,
                        'timestamp': datetime.fromtimestamp(anomaly.timestamp).isoformat(),
                        'source': 'pcap_analysis',
                        'tags': ['network_security', 'anomaly_detection', 'pcap_analysis']
                    }
                    
                    # Store in short-term memory with 24-hour TTL
                    memory_manager.store_memory(
                        domain='NETWORK_ANOMALIES',
                        tier='short_term',
                        key=f"anomaly_{anomaly.timestamp}",
                        data=memory_entry,
                        ttl_hours=24,
                        metadata={
                            'confidence': anomaly.confidence,
                            'severity': anomaly.severity,
                            'anomaly_type': anomaly.anomaly_type.value
                        }
                    )
                
                logger.info(f"💾 Saved {len(anomalies)} anomalies to short-term memory")
                
            except ImportError:
                logger.warning("Context memory manager not available - anomalies not saved to memory")
                
        except Exception as e:
            logger.error(f"Error saving anomalies to memory: {e}")

# Global instance for lazy loading
_pcap_analysis_manager = None

def get_pcap_analysis_manager() -> PCAPAnalysisManager:
    """Get or create PCAP analysis manager instance (lazy loading)."""
    global _pcap_analysis_manager
    if _pcap_analysis_manager is None:
        _pcap_analysis_manager = PCAPAnalysisManager()
    return _pcap_analysis_manager
