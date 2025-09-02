# 🌐 Network Analysis Guide

## Overview

The Network Analysis tools provide comprehensive network security assessment capabilities, including PCAP analysis, network traffic monitoring, protocol analysis, and network forensics. These tools are essential for network security investigations, incident response, and threat hunting.

## Features

### **PCAP Analysis**
- **Traffic Summarization**: Network traffic statistics and flow analysis
- **Protocol Analysis**: Deep packet inspection and protocol identification
- **Technology Stack Detection**: Identify services and applications
- **File Extraction**: Extract files from network traffic
- **Anomaly Detection**: Identify suspicious network patterns

### **Network Tools**
- **Connectivity Testing**: Ping, traceroute, DNS resolution
- **Network Statistics**: Netstat, ARP table analysis
- **Port Scanning**: Advanced port scanning with service detection
- **Bandwidth Analysis**: Network performance monitoring
- **Security Scanning**: Vulnerability-focused network assessment

### **Host Scanning**
- **OS Fingerprinting**: Operating system detection
- **Service Detection**: Identify running services and versions
- **Vulnerability Assessment**: Check for known vulnerabilities
- **Network Topology Mapping**: Map network structure
- **Security Posture Analysis**: Overall security assessment

## Usage

### **Command Line Interface**
```bash
# PCAP analysis
python cs_util_lg.py -workflow data_conversion -problem "analyze PCAP: /path/to/traffic.pcap"

# Network connectivity test
python cs_util_lg.py -workflow data_conversion -problem "test connectivity: example.com"

# Port scanning
python cs_util_lg.py -workflow vulnerability_scan -problem "port scan: 192.168.1.1"

# Network monitoring
python cs_util_lg.py -workflow data_conversion -problem "monitor network: eth0"
```

### **Programmatic Usage**
```python
from bin.pcap_analysis_tools import PCAPAnalyzer
from bin.network_tools import NetworkToolsManager
from bin.host_scanning_tools import HostScanningManager

# PCAP analysis
pcap_analyzer = PCAPAnalyzer()
summary = pcap_analyzer.analyze_pcap("/path/to/traffic.pcap")
print(f"Total packets: {summary.total_packets}")
print(f"Top talkers: {summary.top_talkers}")

# Network tools
network_tools = NetworkToolsManager()
ping_result = network_tools.get_tool("network_ping").ping("example.com")
print(f"Ping result: {ping_result}")

# Host scanning
host_scanner = HostScanningManager()
scan_result = host_scanner.scan_host("192.168.1.1")
print(f"Open ports: {scan_result['open_ports']}")
```

## PCAP Analysis

### **Traffic Analysis**

#### **Basic PCAP Analysis**
```python
from bin.pcap_analysis_tools import PCAPAnalyzer, ProtocolType

# Initialize analyzer
pcap_analyzer = PCAPAnalyzer()

# Analyze PCAP file
summary = pcap_analyzer.analyze_pcap("/path/to/traffic.pcap", "comprehensive")

# Basic statistics
print(f"Analysis Summary:")
print(f"  Total packets: {summary.total_packets}")
print(f"  Total bytes: {summary.total_bytes}")
print(f"  Duration: {summary.duration}")
print(f"  Protocols: {summary.protocols}")

# Top talkers
print(f"\nTop Talkers:")
for talker, bytes_sent in summary.top_talkers[:10]:
    print(f"  {talker}: {bytes_sent} bytes")

# Top flows
print(f"\nTop Flows:")
for flow in summary.top_flows[:10]:
    print(f"  {flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}")
    print(f"    Packets: {flow.packet_count}, Bytes: {flow.byte_count}")
```

#### **Protocol Analysis**
```python
# Analyze specific protocols
def analyze_protocols(pcap_path):
    """Analyze network protocols in PCAP file."""
    summary = pcap_analyzer.analyze_pcap(pcap_path)
    
    protocol_stats = {}
    for protocol in summary.protocols:
        if protocol not in protocol_stats:
            protocol_stats[protocol] = {
                "packets": 0,
                "bytes": 0,
                "flows": 0
            }
    
    # Analyze protocol-specific data
    for flow in summary.top_flows:
        protocol = flow.protocol
        if protocol in protocol_stats:
            protocol_stats[protocol]["flows"] += 1
            protocol_stats[protocol]["packets"] += flow.packet_count
            protocol_stats[protocol]["bytes"] += flow.byte_count
    
    return protocol_stats

# Analyze protocols
protocol_stats = analyze_protocols("/path/to/traffic.pcap")
for protocol, stats in protocol_stats.items():
    print(f"{protocol}: {stats['flows']} flows, {stats['packets']} packets, {stats['bytes']} bytes")
```

### **Technology Stack Detection**

#### **Service Identification**
```python
# Detect technology stack
def detect_technology_stack(pcap_path):
    """Detect technology stack from network traffic."""
    summary = pcap_analyzer.analyze_pcap(pcap_path)
    
    detected_services = []
    for tech in summary.technology_stack:
        detected_services.append({
            "name": tech.name,
            "category": tech.category.value,
            "confidence": tech.confidence,
            "description": tech.description
        })
    
    return detected_services

# Detect services
services = detect_technology_stack("/path/to/traffic.pcap")
print("Detected Services:")
for service in services:
    print(f"  {service['name']} ({service['category']}) - Confidence: {service['confidence']:.2f}")
    print(f"    {service['description']}")
```

### **Anomaly Detection**

#### **Suspicious Activity Detection**
```python
# Detect anomalies
def detect_anomalies(pcap_path):
    """Detect suspicious network activity."""
    summary = pcap_analyzer.analyze_pcap(pcap_path)
    anomalies = []
    
    # Check for suspicious patterns
    for flow in summary.top_flows:
        # Large data transfers
        if flow.byte_count > 1000000:  # 1MB
            anomalies.append({
                "type": "Large Data Transfer",
                "flow": f"{flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}",
                "bytes": flow.byte_count,
                "severity": "Medium"
            })
        
        # High packet count
        if flow.packet_count > 10000:
            anomalies.append({
                "type": "High Packet Count",
                "flow": f"{flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}",
                "packets": flow.packet_count,
                "severity": "Low"
            })
        
        # Suspicious ports
        suspicious_ports = [22, 23, 3389, 5900, 1433, 1521, 3306, 5432]
        if flow.dest_port in suspicious_ports:
            anomalies.append({
                "type": "Suspicious Port",
                "flow": f"{flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}",
                "port": flow.dest_port,
                "severity": "High"
            })
    
    return anomalies

# Detect anomalies
anomalies = detect_anomalies("/path/to/traffic.pcap")
print(f"Detected {len(anomalies)} anomalies:")
for anomaly in anomalies:
    print(f"  {anomaly['type']}: {anomaly['flow']} (Severity: {anomaly['severity']})")
```

### **File Extraction**

#### **Extract Files from Traffic**
```python
# Extract files from network traffic
def extract_files_from_pcap(pcap_path, output_dir):
    """Extract files from PCAP traffic."""
    try:
        # Use pcap_analyzer to extract files
        extracted_files = pcap_analyzer.extract_files(pcap_path, output_dir)
        
        print(f"Extracted {len(extracted_files)} files:")
        for file_info in extracted_files:
            print(f"  {file_info['filename']} ({file_info['size']} bytes)")
            print(f"    Source: {file_info['source_ip']}:{file_info['source_port']}")
            print(f"    Destination: {file_info['dest_ip']}:{file_info['dest_port']}")
        
        return extracted_files
        
    except Exception as e:
        print(f"Error extracting files: {e}")
        return []

# Extract files
extracted = extract_files_from_pcap("/path/to/traffic.pcap", "/output/extracted")
```

## Network Tools

### **Connectivity Testing**

#### **Ping Analysis**
```python
from bin.network_tools import NetworkToolsManager

# Initialize network tools
network_tools = NetworkToolsManager()

# Ping test
def ping_analysis(target, count=10):
    """Perform ping analysis."""
    ping_tool = network_tools.get_tool("network_ping")
    result = ping_tool.ping(target, count=count)
    
    print(f"Ping Results for {target}:")
    print(f"  Packets sent: {result['packets_sent']}")
    print(f"  Packets received: {result['packets_received']}")
    print(f"  Packet loss: {result['packet_loss']}%")
    print(f"  Average RTT: {result['avg_rtt']}ms")
    print(f"  Min RTT: {result['min_rtt']}ms")
    print(f"  Max RTT: {result['max_rtt']}ms")
    
    return result

# Ping analysis
ping_result = ping_analysis("example.com", count=20)
```

#### **Traceroute Analysis**
```python
# Traceroute analysis
def traceroute_analysis(target):
    """Perform traceroute analysis."""
    traceroute_tool = network_tools.get_tool("network_traceroute")
    result = traceroute_tool.traceroute(target)
    
    print(f"Traceroute to {target}:")
    for hop in result['hops']:
        print(f"  {hop['hop_number']}: {hop['ip_address']} ({hop['hostname']})")
        print(f"    RTT: {hop['rtt']}ms")
    
    return result

# Traceroute analysis
trace_result = traceroute_analysis("example.com")
```

### **DNS Analysis**

#### **DNS Resolution**
```python
# DNS analysis
def dns_analysis(domain):
    """Perform DNS analysis."""
    dns_tool = network_tools.get_tool("network_dns_lookup")
    result = dns_tool.lookup(domain)
    
    print(f"DNS Analysis for {domain}:")
    print(f"  A Records: {result.get('A', [])}")
    print(f"  AAAA Records: {result.get('AAAA', [])}")
    print(f"  MX Records: {result.get('MX', [])}")
    print(f"  NS Records: {result.get('NS', [])}")
    print(f"  TXT Records: {result.get('TXT', [])}")
    
    return result

# DNS analysis
dns_result = dns_analysis("example.com")
```

### **Network Statistics**

#### **Netstat Analysis**
```python
# Network statistics
def network_statistics():
    """Get network statistics."""
    netstat_tool = network_tools.get_tool("network_netstat")
    result = netstat_tool.get_connections()
    
    print("Network Connections:")
    for conn in result['connections']:
        print(f"  {conn['protocol']} {conn['local_address']}:{conn['local_port']} -> {conn['remote_address']}:{conn['remote_port']}")
        print(f"    State: {conn['state']}, PID: {conn['pid']}")
    
    return result

# Network statistics
netstat_result = network_statistics()
```

#### **ARP Table Analysis**
```python
# ARP table analysis
def arp_analysis():
    """Analyze ARP table."""
    arp_tool = network_tools.get_tool("network_arp")
    result = arp_tool.get_arp_table()
    
    print("ARP Table:")
    for entry in result['arp_entries']:
        print(f"  {entry['ip_address']} -> {entry['mac_address']} ({entry['interface']})")
        print(f"    Type: {entry['type']}")
    
    return result

# ARP analysis
arp_result = arp_analysis()
```

## Host Scanning

### **Port Scanning**

#### **Comprehensive Port Scan**
```python
from bin.host_scanning_tools import HostScanningManager, ScanType, ScanIntensity

# Initialize host scanner
host_scanner = HostScanningManager()

# Comprehensive port scan
def comprehensive_scan(target):
    """Perform comprehensive port scan."""
    scan_result = host_scanner.scan_host(
        target=target,
        scan_type=ScanType.COMPREHENSIVE,
        intensity=ScanIntensity.NORMAL
    )
    
    print(f"Scan Results for {target}:")
    print(f"  Host status: {scan_result['host_status']}")
    print(f"  Open ports: {len(scan_result['open_ports'])}")
    print(f"  Services detected: {len(scan_result['services'])}")
    
    print("\nOpen Ports:")
    for port_info in scan_result['open_ports']:
        print(f"  Port {port_info['port']}: {port_info['service']} ({port_info['state']})")
        if 'version' in port_info:
            print(f"    Version: {port_info['version']}")
    
    print("\nServices:")
    for service in scan_result['services']:
        print(f"  {service['name']}: {service['port']} ({service['protocol']})")
        if 'version' in service:
            print(f"    Version: {service['version']}")
    
    return scan_result

# Comprehensive scan
scan_result = comprehensive_scan("192.168.1.1")
```

#### **Service Detection**
```python
# Service detection
def service_detection(target):
    """Detect services on target."""
    scan_result = host_scanner.scan_host(
        target=target,
        scan_type=ScanType.SERVICE_DETECTION,
        intensity=ScanIntensity.AGGRESSIVE
    )
    
    print(f"Service Detection for {target}:")
    for service in scan_result['services']:
        print(f"  {service['name']} on port {service['port']}")
        print(f"    Protocol: {service['protocol']}")
        print(f"    Version: {service.get('version', 'Unknown')}")
        print(f"    Banner: {service.get('banner', 'None')}")
    
    return scan_result

# Service detection
service_result = service_detection("192.168.1.1")
```

### **OS Detection**

#### **Operating System Fingerprinting**
```python
# OS detection
def os_detection(target):
    """Detect operating system."""
    scan_result = host_scanner.scan_host(
        target=target,
        scan_type=ScanType.OS_DETECTION,
        intensity=ScanIntensity.NORMAL
    )
    
    if 'os_info' in scan_result:
        os_info = scan_result['os_info']
        print(f"OS Detection for {target}:")
        print(f"  OS: {os_info.get('name', 'Unknown')}")
        print(f"  Version: {os_info.get('version', 'Unknown')}")
        print(f"  Architecture: {os_info.get('architecture', 'Unknown')}")
        print(f"  Confidence: {os_info.get('confidence', 0):.2f}")
    else:
        print(f"OS detection failed for {target}")
    
    return scan_result

# OS detection
os_result = os_detection("192.168.1.1")
```

## Network Forensics

### **Incident Response**

#### **Network Incident Analysis**
```python
def network_incident_analysis(pcap_path, incident_type="data_breach"):
    """Analyze network incident."""
    print(f"Analyzing {incident_type} incident...")
    
    # Analyze PCAP
    summary = pcap_analyzer.analyze_pcap(pcap_path, "comprehensive")
    
    # Detect anomalies
    anomalies = detect_anomalies(pcap_path)
    
    # Extract files
    extracted_files = extract_files_from_pcap(pcap_path, "/output/incident")
    
    # Generate incident report
    incident_report = {
        "incident_type": incident_type,
        "analysis_timestamp": datetime.now().isoformat(),
        "pcap_file": pcap_path,
        "traffic_summary": {
            "total_packets": summary.total_packets,
            "total_bytes": summary.total_bytes,
            "duration": summary.duration,
            "protocols": summary.protocols
        },
        "anomalies_detected": len(anomalies),
        "anomalies": anomalies,
        "files_extracted": len(extracted_files),
        "extracted_files": extracted_files,
        "recommendations": generate_incident_recommendations(anomalies)
    }
    
    return incident_report

def generate_incident_recommendations(anomalies):
    """Generate incident response recommendations."""
    recommendations = []
    
    for anomaly in anomalies:
        if anomaly['severity'] == 'High':
            recommendations.append(f"Immediately investigate {anomaly['type']}: {anomaly['flow']}")
        elif anomaly['severity'] == 'Medium':
            recommendations.append(f"Monitor {anomaly['type']}: {anomaly['flow']}")
        else:
            recommendations.append(f"Review {anomaly['type']}: {anomaly['flow']}")
    
    return recommendations

# Incident analysis
incident_report = network_incident_analysis("/path/to/incident.pcap", "data_breach")
print("Incident Analysis Complete:")
print(f"  Anomalies detected: {incident_report['anomalies_detected']}")
print(f"  Files extracted: {incident_report['files_extracted']}")
print("  Recommendations:")
for rec in incident_report['recommendations']:
    print(f"    • {rec}")
```

### **Threat Hunting**

#### **Network Threat Hunting**
```python
def network_threat_hunting(pcap_path, threat_indicators):
    """Perform network threat hunting."""
    print("Starting network threat hunting...")
    
    # Analyze PCAP
    summary = pcap_analyzer.analyze_pcap(pcap_path, "comprehensive")
    
    # Check for threat indicators
    threats_found = []
    
    for flow in summary.top_flows:
        # Check for suspicious IPs
        for indicator in threat_indicators.get('ips', []):
            if indicator in [flow.source_ip, flow.dest_ip]:
                threats_found.append({
                    "type": "Suspicious IP",
                    "indicator": indicator,
                    "flow": f"{flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}",
                    "severity": "High"
                })
        
        # Check for suspicious ports
        for indicator in threat_indicators.get('ports', []):
            if indicator in [flow.source_port, flow.dest_port]:
                threats_found.append({
                    "type": "Suspicious Port",
                    "indicator": indicator,
                    "flow": f"{flow.source_ip}:{flow.source_port} -> {flow.dest_ip}:{flow.dest_port}",
                    "severity": "Medium"
                })
        
        # Check for suspicious domains (if available)
        for indicator in threat_indicators.get('domains', []):
            # This would require DNS analysis
            pass
    
    return {
        "threats_found": len(threats_found),
        "threats": threats_found,
        "analysis_summary": summary
    }

# Threat hunting
threat_indicators = {
    "ips": ["192.168.1.100", "10.0.0.50"],
    "ports": [4444, 6666, 31337],
    "domains": ["malicious.com", "suspicious.org"]
}

threat_hunt_result = network_threat_hunting("/path/to/traffic.pcap", threat_indicators)
print(f"Threat hunting complete: {threat_hunt_result['threats_found']} threats found")
```

## Integration with Other Tools

### **Vulnerability Scanner Integration**
```python
from bin.vulnerability_scanner import VulnerabilityScanner

def network_vulnerability_assessment(target):
    """Combine network analysis with vulnerability scanning."""
    # Network analysis
    network_result = comprehensive_scan(target)
    
    # Vulnerability scanning
    vuln_scanner = VulnerabilityScanner()
    vuln_result = vuln_scanner.scan_target(target)
    
    # Combine results
    combined_result = {
        "target": target,
        "network_analysis": network_result,
        "vulnerability_scan": vuln_result,
        "risk_assessment": {
            "network_risk": len(network_result['open_ports']) * 0.1,
            "vulnerability_risk": vuln_result.risk_score,
            "overall_risk": (len(network_result['open_ports']) * 0.1 + vuln_result.risk_score) / 2
        }
    }
    
    return combined_result

# Combined assessment
assessment = network_vulnerability_assessment("192.168.1.1")
print(f"Overall risk score: {assessment['risk_assessment']['overall_risk']:.2f}")
```

### **Database Integration**
```python
from bin.sqlite_manager import SQLiteManager

def store_network_analysis(analysis_result):
    """Store network analysis results in database."""
    db = SQLiteManager()
    
    # Store network events
    for flow in analysis_result['top_flows']:
        db.insert_data("network_events", {
            "timestamp": datetime.now().isoformat(),
            "source_ip": flow.source_ip,
            "dest_ip": flow.dest_ip,
            "source_port": flow.source_port,
            "dest_port": flow.dest_port,
            "protocol": flow.protocol,
            "packet_size": flow.byte_count,
            "event_type": "Network Flow",
            "threat_level": "Low",
            "metadata": json.dumps({
                "packet_count": flow.packet_count,
                "duration": flow.duration
            })
        })
    
    print("Network analysis stored in database")

# Store analysis
store_network_analysis(summary)
```

## Performance Optimization

### **Large PCAP Handling**
```python
def analyze_large_pcap(pcap_path, chunk_size=10000):
    """Analyze large PCAP files in chunks."""
    try:
        # Get PCAP file size
        file_size = os.path.getsize(pcap_path)
        print(f"Analyzing large PCAP file: {file_size / (1024*1024):.2f} MB")
        
        # Analyze in chunks if file is very large
        if file_size > 100 * 1024 * 1024:  # 100MB
            print("Large file detected, using chunked analysis...")
            # Implement chunked analysis
            return analyze_pcap_chunks(pcap_path, chunk_size)
        else:
            # Standard analysis
            return pcap_analyzer.analyze_pcap(pcap_path)
            
    except Exception as e:
        print(f"Error analyzing PCAP: {e}")
        return None

def analyze_pcap_chunks(pcap_path, chunk_size):
    """Analyze PCAP file in chunks."""
    # This would implement chunked analysis for very large files
    # For now, return basic analysis
    return pcap_analyzer.analyze_pcap(pcap_path)
```

### **Parallel Processing**
```python
import concurrent.futures

def parallel_network_scan(targets):
    """Perform parallel network scans."""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_target = {
            executor.submit(comprehensive_scan, target): target 
            for target in targets
        }
        
        for future in concurrent.futures.as_completed(future_to_target):
            target = future_to_target[future]
            try:
                result = future.result()
                results.append({
                    "target": target,
                    "result": result
                })
            except Exception as e:
                print(f"Error scanning {target}: {e}")
    
    return results

# Parallel scanning
targets = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
results = parallel_network_scan(targets)
print(f"Scanned {len(results)} targets")
```

## Troubleshooting

### **Common Issues**

#### **PCAP Analysis Errors**
```python
def safe_pcap_analysis(pcap_path):
    """Safely analyze PCAP files with error handling."""
    try:
        # Check if file exists and is readable
        if not os.path.exists(pcap_path):
            return {"error": "PCAP file not found"}
        
        if not os.access(pcap_path, os.R_OK):
            return {"error": "PCAP file not readable"}
        
        # Check file size
        file_size = os.path.getsize(pcap_path)
        if file_size == 0:
            return {"error": "PCAP file is empty"}
        
        # Attempt analysis
        result = pcap_analyzer.analyze_pcap(pcap_path)
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"error": str(e)}
```

#### **Network Tool Failures**
```python
def robust_network_scan(target):
    """Perform robust network scan with fallbacks."""
    try:
        # Try comprehensive scan first
        result = comprehensive_scan(target)
        return result
        
    except Exception as e:
        print(f"Comprehensive scan failed: {e}")
        
        try:
            # Fallback to basic ping
            ping_result = ping_analysis(target)
            return {"scan_type": "ping_only", "result": ping_result}
            
        except Exception as e2:
            print(f"Ping scan also failed: {e2}")
            return {"error": "All scan methods failed"}
```

## Best Practices

### **Network Analysis Workflow**
1. **Initial Assessment**: Basic connectivity and port scanning
2. **Traffic Capture**: Capture relevant network traffic
3. **PCAP Analysis**: Analyze captured traffic for anomalies
4. **Service Enumeration**: Identify running services and versions
5. **Vulnerability Assessment**: Check for known vulnerabilities
6. **Threat Hunting**: Look for specific threat indicators
7. **Reporting**: Generate comprehensive analysis report

### **Evidence Collection**
1. **Traffic Capture**: Use proper capture filters
2. **Metadata Preservation**: Maintain packet timestamps and headers
3. **Chain of Custody**: Document all analysis activities
4. **Secure Storage**: Store evidence in encrypted containers
5. **Integrity Verification**: Use hashes to verify evidence integrity

### **Performance Optimization**
1. **Filtered Capture**: Use capture filters to reduce data volume
2. **Chunked Analysis**: Process large files in chunks
3. **Parallel Processing**: Use multiple threads for concurrent operations
4. **Selective Analysis**: Focus on relevant traffic patterns
5. **Caching**: Cache frequently accessed analysis results

This guide provides comprehensive information about using the network analysis tools effectively. For additional support or advanced use cases, refer to the main documentation or contact the development team.
