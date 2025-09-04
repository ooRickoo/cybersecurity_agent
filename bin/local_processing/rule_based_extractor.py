#!/usr/bin/env python3
"""
Rule-Based Parameter Extractor for Cybersecurity Commands
Extracts cybersecurity-specific parameters using regex patterns
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import ipaddress

logger = logging.getLogger(__name__)

@dataclass
class ExtractedParameters:
    """Result of parameter extraction"""
    entities: Dict[str, List[str]]
    analysis_type: str
    priority: str
    targets: List[str]
    confidence: float
    time_parameters: Dict[str, Any]
    file_parameters: Dict[str, Any]
    network_parameters: Dict[str, Any]

class CyberSecurityParameterExtractor:
    """Extract cybersecurity-specific parameters using regex patterns"""
    
    def __init__(self):
        # Regex patterns for cybersecurity entities
        self.patterns = {
            'ip_addresses': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'ipv6_addresses': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'domains': r'\b[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\b',
            'urls': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'sample_ids': r'(?:sample|file|hash)\s+([A-Z0-9\-_]+)',
            'md5_hashes': r'\b[a-fA-F0-9]{32}\b',
            'sha1_hashes': r'\b[a-fA-F0-9]{40}\b',
            'sha256_hashes': r'\b[a-fA-F0-9]{64}\b',
            'cve_ids': r'CVE-\d{4}-\d{4,}',
            'file_paths': r'[A-Za-z]:\\[^<>:"|?*\r\n]*|/[^<>:"|?*\r\n]*',
            'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mac_addresses': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'port_numbers': r'\b(?:port\s+)?(\d{1,5})\b',
            'process_names': r'\b(?:process|exe)\s+([a-zA-Z0-9_.-]+\.exe)\b',
            'registry_keys': r'HKEY_[A-Z_]+\\[^\\]+(?:\\[^\\]+)*',
            'mutex_names': r'\b(?:mutex|semaphore)\s+([a-zA-Z0-9_\\-]+)\b'
        }
        
        # Analysis type patterns
        self.analysis_types = {
            'malware': ['malware', 'virus', 'trojan', 'suspicious', 'malicious', 'backdoor', 'rootkit'],
            'vulnerability': ['vulnerability', 'vuln', 'scan', 'CVE', 'exploit', 'patch', 'security hole'],
            'forensics': ['forensic', 'investigate', 'timeline', 'evidence', 'artifact', 'recovery'],
            'network': ['network', 'traffic', 'pcap', 'packet', 'connection', 'protocol', 'firewall'],
            'threat_hunting': ['hunt', 'threat', 'ioc', 'apt', 'campaign', 'attribution', 'intelligence'],
            'incident_response': ['incident', 'breach', 'response', 'containment', 'eradication', 'recovery']
        }
        
        # Priority/urgency patterns
        self.priority_patterns = {
            'critical': ['critical', 'urgent', 'emergency', 'immediate', 'asap', 'now'],
            'high': ['high', 'important', 'priority', 'soon', 'today'],
            'medium': ['medium', 'normal', 'standard', 'regular'],
            'low': ['low', 'when possible', 'eventually', 'later']
        }
        
        # Time-based patterns
        self.time_patterns = {
            'last_hours': r'last\s+(\d+)\s+hours?',
            'last_days': r'last\s+(\d+)\s+days?',
            'last_weeks': r'last\s+(\d+)\s+weeks?',
            'since_date': r'since\s+(\d{4}-\d{2}-\d{2})',
            'between_dates': r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})',
            'today': r'today',
            'yesterday': r'yesterday',
            'this_week': r'this\s+week',
            'this_month': r'this\s+month'
        }
    
    def extract_all_parameters(self, text: str) -> ExtractedParameters:
        """
        Extract all cybersecurity parameters
        Returns: ExtractedParameters with all extracted data
        """
        text_lower = text.lower()
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Determine analysis type
        analysis_type = self._detect_analysis_type(text_lower)
        
        # Determine priority
        priority = self._detect_priority(text_lower)
        
        # Extract targets (main focus of analysis)
        targets = self._extract_targets(text, entities)
        
        # Extract time parameters
        time_params = self._extract_time_parameters(text_lower)
        
        # Extract file parameters
        file_params = self._extract_file_parameters(text, entities)
        
        # Extract network parameters
        network_params = self._extract_network_parameters(text, entities)
        
        # Calculate confidence
        confidence = self._calculate_confidence(entities, analysis_type, targets)
        
        return ExtractedParameters(
            entities=entities,
            analysis_type=analysis_type,
            priority=priority,
            targets=targets,
            confidence=confidence,
            time_parameters=time_params,
            file_parameters=file_params,
            network_parameters=network_params
        )
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entity types from text"""
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = self.extract_multiple_values(text, entity_type)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def extract_multiple_values(self, text: str, pattern_type: str) -> List[str]:
        """Extract multiple values of same type (e.g., multiple IPs)"""
        if pattern_type not in self.patterns:
            return []
        
        pattern = self.patterns[pattern_type]
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        # Clean and validate matches
        cleaned_matches = []
        for match in matches:
            if isinstance(match, tuple):
                # Handle capture groups
                match = match[0] if match[0] else match[1] if len(match) > 1 else str(match)
            
            match = str(match).strip()
            if match and self._validate_entity(pattern_type, match):
                cleaned_matches.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in cleaned_matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        
        return unique_matches
    
    def _validate_entity(self, entity_type: str, value: str) -> bool:
        """Validate extracted entity based on type"""
        try:
            if entity_type == 'ip_addresses':
                ipaddress.ip_address(value)
                return True
            elif entity_type == 'ipv6_addresses':
                ipaddress.ip_address(value)
                return True
            elif entity_type == 'domains':
                return len(value) > 3 and '.' in value
            elif entity_type == 'urls':
                return value.startswith(('http://', 'https://'))
            elif entity_type in ['md5_hashes', 'sha1_hashes', 'sha256_hashes']:
                return len(value) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in value)
            elif entity_type == 'cve_ids':
                return value.startswith('CVE-') and len(value.split('-')) == 3
            elif entity_type == 'email_addresses':
                return '@' in value and '.' in value.split('@')[1]
            elif entity_type == 'port_numbers':
                port = int(value)
                return 1 <= port <= 65535
            else:
                return len(value) > 0
        except (ValueError, AttributeError):
            return False
    
    def _detect_analysis_type(self, text_lower: str) -> str:
        """Detect the type of analysis requested"""
        type_scores = {}
        
        for analysis_type, keywords in self.analysis_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[analysis_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _detect_priority(self, text_lower: str) -> str:
        """Detect priority level from text"""
        for priority, keywords in self.priority_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority
        
        return 'medium'
    
    def _extract_targets(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract main targets for analysis"""
        targets = []
        
        # Add IP addresses as targets
        if 'ip_addresses' in entities:
            targets.extend(entities['ip_addresses'])
        
        # Add domains as targets
        if 'domains' in entities:
            targets.extend(entities['domains'])
        
        # Add file paths as targets
        if 'file_paths' in entities:
            targets.extend(entities['file_paths'])
        
        # Add sample IDs as targets
        if 'sample_ids' in entities:
            targets.extend(entities['sample_ids'])
        
        # Add CVE IDs as targets
        if 'cve_ids' in entities:
            targets.extend(entities['cve_ids'])
        
        return targets[:10]  # Limit to top 10 targets
    
    def _extract_time_parameters(self, text_lower: str) -> Dict[str, Any]:
        """Extract time-based parameters"""
        time_params = {}
        
        for pattern_name, pattern in self.time_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                if pattern_name == 'last_hours':
                    hours = int(match.group(1))
                    time_params['start_time'] = datetime.now() - timedelta(hours=hours)
                    time_params['end_time'] = datetime.now()
                elif pattern_name == 'last_days':
                    days = int(match.group(1))
                    time_params['start_time'] = datetime.now() - timedelta(days=days)
                    time_params['end_time'] = datetime.now()
                elif pattern_name == 'last_weeks':
                    weeks = int(match.group(1))
                    time_params['start_time'] = datetime.now() - timedelta(weeks=weeks)
                    time_params['end_time'] = datetime.now()
                elif pattern_name == 'since_date':
                    date_str = match.group(1)
                    time_params['start_time'] = datetime.strptime(date_str, '%Y-%m-%d')
                    time_params['end_time'] = datetime.now()
                elif pattern_name == 'between_dates':
                    start_date = datetime.strptime(match.group(1), '%Y-%m-%d')
                    end_date = datetime.strptime(match.group(2), '%Y-%m-%d')
                    time_params['start_time'] = start_date
                    time_params['end_time'] = end_date
                elif pattern_name == 'today':
                    today = datetime.now().date()
                    time_params['start_time'] = datetime.combine(today, datetime.min.time())
                    time_params['end_time'] = datetime.now()
                elif pattern_name == 'yesterday':
                    yesterday = datetime.now().date() - timedelta(days=1)
                    time_params['start_time'] = datetime.combine(yesterday, datetime.min.time())
                    time_params['end_time'] = datetime.combine(yesterday, datetime.max.time())
        
        return time_params
    
    def _extract_file_parameters(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract file-related parameters"""
        file_params = {}
        
        # File paths
        if 'file_paths' in entities:
            file_params['paths'] = entities['file_paths']
        
        # File hashes
        hashes = []
        for hash_type in ['md5_hashes', 'sha1_hashes', 'sha256_hashes']:
            if hash_type in entities:
                hashes.extend(entities[hash_type])
        if hashes:
            file_params['hashes'] = hashes
        
        # Sample IDs
        if 'sample_ids' in entities:
            file_params['sample_ids'] = entities['sample_ids']
        
        # File types (extract from file paths)
        file_types = set()
        if 'file_paths' in entities:
            for path in entities['file_paths']:
                if '.' in path:
                    ext = path.split('.')[-1].lower()
                    file_types.add(ext)
        if file_types:
            file_params['file_types'] = list(file_types)
        
        return file_params
    
    def _extract_network_parameters(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract network-related parameters"""
        network_params = {}
        
        # IP addresses
        if 'ip_addresses' in entities:
            network_params['ip_addresses'] = entities['ip_addresses']
        
        # IPv6 addresses
        if 'ipv6_addresses' in entities:
            network_params['ipv6_addresses'] = entities['ipv6_addresses']
        
        # Domains
        if 'domains' in entities:
            network_params['domains'] = entities['domains']
        
        # URLs
        if 'urls' in entities:
            network_params['urls'] = entities['urls']
        
        # Port numbers
        if 'port_numbers' in entities:
            network_params['ports'] = [int(port) for port in entities['port_numbers'] if port.isdigit()]
        
        # MAC addresses
        if 'mac_addresses' in entities:
            network_params['mac_addresses'] = entities['mac_addresses']
        
        return network_params
    
    def _calculate_confidence(self, entities: Dict[str, List[str]], analysis_type: str, targets: List[str]) -> float:
        """Calculate confidence in parameter extraction"""
        confidence = 0.0
        
        # Base confidence from entity count
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities > 0:
            confidence += min(0.5, total_entities * 0.1)
        
        # Boost confidence for specific analysis types
        if analysis_type != 'general':
            confidence += 0.2
        
        # Boost confidence for having targets
        if targets:
            confidence += 0.2
        
        # Boost confidence for having key entities
        key_entities = ['ip_addresses', 'domains', 'file_paths', 'cve_ids']
        if any(entity_type in entities for entity_type in key_entities):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def validate_extracted_params(self, params: ExtractedParameters) -> Tuple[bool, List[str]]:
        """Validate extracted parameters and return issues"""
        issues = []
        
        # Check if we have targets
        if not params.targets:
            issues.append("No clear targets identified for analysis")
        
        # Check if analysis type is too vague
        if params.analysis_type == 'general' and params.confidence < 0.3:
            issues.append("Analysis type unclear - may need clarification")
        
        # Check for conflicting parameters
        if 'ip_addresses' in params.entities and 'domains' in params.entities:
            if len(params.entities['ip_addresses']) > 5 and len(params.entities['domains']) > 5:
                issues.append("Many IPs and domains detected - may need scope clarification")
        
        # Check time parameters
        if params.time_parameters:
            if 'start_time' in params.time_parameters and 'end_time' in params.time_parameters:
                start = params.time_parameters['start_time']
                end = params.time_parameters['end_time']
                if start > end:
                    issues.append("Invalid time range - start time is after end time")
        
        return len(issues) == 0, issues


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the extractor
    extractor = CyberSecurityParameterExtractor()
    
    test_cases = [
        "analyze sample X-47 for malware",
        "scan 192.168.1.1 and 10.0.0.1 for vulnerabilities",
        "check files /tmp/suspicious.exe and /var/log/access.log for forensic evidence",
        "hunt for threats in network traffic from the last 24 hours",
        "investigate CVE-2023-1234 and CVE-2023-5678 on domain example.com"
    ]
    
    print("🧪 Testing Rule-Based Parameter Extractor")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        params = extractor.extract_all_parameters(text)
        
        print(f"Analysis Type: {params.analysis_type}")
        print(f"Priority: {params.priority}")
        print(f"Targets: {params.targets}")
        print(f"Confidence: {params.confidence:.2f}")
        
        if params.entities:
            print("Entities:")
            for entity_type, values in params.entities.items():
                print(f"  {entity_type}: {values}")
        
        if params.time_parameters:
            print(f"Time Parameters: {params.time_parameters}")
        
        # Validate parameters
        is_valid, issues = extractor.validate_extracted_params(params)
        if not is_valid:
            print(f"Issues: {issues}")
        else:
            print("✅ Parameters valid")
