"""
Public Cyber Intelligence Feed Tools for Cybersecurity Agent

Provides access to and processing of various public cyber intelligence feeds
including MITRE ATT&CK, CVE databases, threat actors, and other OSINT sources.
"""

import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import asyncio
import aiohttp
import time
from urllib.parse import urljoin, urlparse
import hashlib
import gzip
import io

from .enhanced_session_manager import EnhancedSessionManager
from .context_memory_manager import ContextMemoryManager
from .credential_vault import CredentialVault

logger = logging.getLogger(__name__)

class PublicIntelFeeds:
    """Comprehensive public cyber intelligence feed processor."""
    
    def __init__(self, session_manager: EnhancedSessionManager, 
                 memory_manager: ContextMemoryManager,
                 credential_vault: CredentialVault):
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.credential_vault = credential_vault
        
        # Feed configurations
        self.feed_configs = self._initialize_feed_configs()
        
        # Rate limiting and caching
        self.rate_limits = {}
        self.cache = {}
        self.last_fetch = {}
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Cybersecurity-Agent/1.0 (Public Intel Feed Processor)'
        })
    
    def _initialize_feed_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configurations for various intelligence feeds."""
        return {
            'mitre_attack': {
                'name': 'MITRE ATT&CK',
                'description': 'MITRE ATT&CK framework and techniques',
                'base_url': 'https://attack.mitre.org/api/',
                'endpoints': {
                    'techniques': 'techniques/enterprise/',
                    'tactics': 'tactics/enterprise/',
                    'groups': 'groups/',
                    'software': 'software/',
                    'campaigns': 'campaigns/',
                    'mitigations': 'mitigations/enterprise/'
                },
                'update_frequency': 'weekly',
                'format': 'json',
                'rate_limit': 100,  # requests per hour
                'requires_auth': False
            },
            'cve_database': {
                'name': 'CVE Database',
                'description': 'Common Vulnerabilities and Exposures',
                'base_url': 'https://cve.mitre.org/cgi-bin/cvekey.cgi',
                'api_url': 'https://services.nvd.nist.gov/rest/json/cves/2.0',
                'update_frequency': 'daily',
                'format': 'json',
                'rate_limit': 1000,  # requests per hour
                'requires_auth': False
            },
            'threatfox': {
                'name': 'ThreatFox',
                'description': 'Malware threat intelligence platform',
                'base_url': 'https://threatfox-api.abuse.ch/api/v1/',
                'endpoints': {
                    'malware': 'malware/',
                    'iocs': 'iocs/',
                    'recent': 'recent/',
                    'query': 'query/'
                },
                'update_frequency': 'hourly',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': False
            },
            'abuseipdb': {
                'name': 'AbuseIPDB',
                'description': 'IP reputation and abuse database',
                'base_url': 'https://api.abuseipdb.com/api/v2/',
                'endpoints': {
                    'check': 'check',
                    'report': 'report',
                    'blacklist': 'blacklist',
                    'clear': 'clear'
                },
                'update_frequency': 'real-time',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': True
            },
            'virustotal': {
                'name': 'VirusTotal',
                'description': 'File and URL reputation analysis',
                'base_url': 'https://www.virustotal.com/vtapi/v2/',
                'endpoints': {
                    'file_report': 'file/report',
                    'url_report': 'url/report',
                    'ip_report': 'ip-address/report',
                    'domain_report': 'domain/report'
                },
                'update_frequency': 'real-time',
                'format': 'json',
                'rate_limit': 500,
                'requires_auth': True
            },
            'shodan': {
                'name': 'Shodan',
                'description': 'Internet-connected device search engine',
                'base_url': 'https://api.shodan.io/',
                'endpoints': {
                    'search': 'shodan/host/search',
                    'host': 'shodan/host/',
                    'ports': 'shodan/ports',
                    'protocols': 'shodan/protocols'
                },
                'update_frequency': 'real-time',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': True
            },
            'alienvault_otx': {
                'name': 'AlienVault OTX',
                'description': 'Open Threat Exchange platform',
                'base_url': 'https://otx.alienvault.com/api/v1/',
                'endpoints': {
                    'indicators': 'indicators/',
                    'pulses': 'pulses/',
                    'users': 'users/',
                    'search': 'search/'
                },
                'update_frequency': 'daily',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': False
            },
            'urlhaus': {
                'name': 'URLhaus',
                'description': 'Malicious URL database',
                'base_url': 'https://urlhaus-api.abuse.ch/v1/',
                'endpoints': {
                    'url': 'url/',
                    'payload': 'payload/',
                    'tag': 'tag/',
                    'recent': 'recent/'
                },
                'update_frequency': 'hourly',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': False
            },
            'phishtank': {
                'name': 'PhishTank',
                'description': 'Phishing URL database',
                'base_url': 'https://data.phishtank.com/data/',
                'endpoints': {
                    'online': 'online-valid.json',
                    'verified': 'verified_online.json'
                },
                'update_frequency': 'hourly',
                'format': 'json',
                'rate_limit': 100,
                'requires_auth': False
            },
            'malware_bazaar': {
                'name': 'Malware Bazaar',
                'description': 'Malware sample database',
                'base_url': 'https://bazaar.abuse.ch/api/v1/',
                'endpoints': {
                    'query': 'query/',
                    'tag': 'tag/',
                    'signature': 'signature/',
                    'recent': 'recent/'
                },
                'update_frequency': 'hourly',
                'format': 'json',
                'rate_limit': 1000,
                'requires_auth': False
            }
        }
    
    def get_available_feeds(self) -> Dict[str, Any]:
        """Get list of available intelligence feeds."""
        return {
            'success': True,
            'feeds': self.feed_configs,
            'total_feeds': len(self.feed_configs)
        }
    
    def fetch_mitre_attack(self, feed_type: str = 'techniques', 
                          include_deprecated: bool = False) -> Dict[str, Any]:
        """
        Fetch MITRE ATT&CK data.
        
        Args:
            feed_type: Type of data to fetch (techniques, tactics, groups, software, campaigns, mitigations)
            include_deprecated: Whether to include deprecated items
            
        Returns:
            MITRE ATT&CK data
        """
        try:
            if feed_type not in self.feed_configs['mitre_attack']['endpoints']:
                return {
                    'success': False,
                    'error': f'Invalid feed type: {feed_type}. Valid types: {list(self.feed_configs["mitre_attack"]["endpoints"].keys())}'
                }
            
            # Check rate limiting
            if not self._check_rate_limit('mitre_attack'):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded for MITRE ATT&CK feed'
                }
            
            endpoint = self.feed_configs['mitre_attack']['endpoints'][feed_type]
            url = urljoin(self.feed_configs['mitre_attack']['base_url'], endpoint)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Filter deprecated items if requested
            if not include_deprecated and isinstance(data, list):
                data = [item for item in data if not item.get('revoked', False)]
            
            # Store in memory
            self._store_intel_data(f'mitre_attack_{feed_type}', data, 'mitre_attack')
            
            # Update rate limiting
            self._update_rate_limit('mitre_attack')
            
            return {
                'success': True,
                'feed_type': feed_type,
                'data': data,
                'count': len(data) if isinstance(data, list) else 1,
                'source': 'MITRE ATT&CK',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch MITRE ATT&CK {feed_type}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def fetch_cve_data(self, cve_id: Optional[str] = None, 
                       keyword: Optional[str] = None,
                       limit: int = 100) -> Dict[str, Any]:
        """
        Fetch CVE data from NVD database.
        
        Args:
            cve_id: Specific CVE ID to fetch
            keyword: Keyword to search for
            limit: Maximum number of results
            
        Returns:
            CVE data
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit('cve_database'):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded for CVE database'
                }
            
            if cve_id:
                # Fetch specific CVE
                url = f"{self.feed_configs['cve_database']['api_url']}?cveId={cve_id}"
            elif keyword:
                # Search by keyword
                url = f"{self.feed_configs['cve_database']['api_url']}?keywordSearch={keyword}&resultsPerPage={limit}"
            else:
                # Fetch recent CVEs
                url = f"{self.feed_configs['cve_database']['api_url']}?resultsPerPage={limit}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Store in memory
            self._store_intel_data('cve_data', data, 'cve_database')
            
            # Update rate limiting
            self._update_rate_limit('cve_database')
            
            return {
                'success': True,
                'cve_id': cve_id,
                'keyword': keyword,
                'data': data,
                'count': len(data.get('vulnerabilities', [])),
                'source': 'NVD CVE Database',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch CVE data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def fetch_threatfox_iocs(self, days: int = 1, 
                            malware_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch IOCs from ThreatFox.
        
        Args:
            days: Number of days to look back
            malware_type: Specific malware type to filter by
            
        Returns:
            ThreatFox IOC data
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit('threatfox'):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded for ThreatFox feed'
                }
            
            # Calculate timestamp
            timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            payload = {
                'query': 'get_recent',
                'days': days
            }
            
            if malware_type:
                payload['malware_type'] = malware_type
            
            response = self.session.post(
                self.feed_configs['threatfox']['base_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('query_status') == 'ok':
                # Store in memory
                self._store_intel_data('threatfox_iocs', data, 'threatfox')
                
                # Update rate limiting
                self._update_rate_limit('threatfox')
                
                return {
                    'success': True,
                    'days': days,
                    'malware_type': malware_type,
                    'data': data,
                    'count': len(data.get('data', [])),
                    'source': 'ThreatFox',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'ThreatFox query failed: {data.get("query_status")}'
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch ThreatFox IOCs: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_ip_reputation(self, ip_address: str, 
                           sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check IP reputation across multiple sources.
        
        Args:
            ip_address: IP address to check
            sources: List of sources to check (default: all available)
            
        Returns:
            IP reputation data
        """
        try:
            if sources is None:
                sources = ['abuseipdb', 'virustotal', 'shodan']
            
            results = {}
            
            # Check AbuseIPDB
            if 'abuseipdb' in sources:
                abuse_result = self._check_abuseipdb(ip_address)
                results['abuseipdb'] = abuse_result
            
            # Check VirusTotal
            if 'virustotal' in sources:
                vt_result = self._check_virustotal_ip(ip_address)
                results['virustotal'] = vt_result
            
            # Check Shodan
            if 'shodan' in sources:
                shodan_result = self._check_shodan_ip(ip_address)
                results['shodan'] = shodan_result
            
            # Aggregate results
            reputation_score = self._calculate_reputation_score(results)
            
            # Store in memory
            self._store_intel_data(f'ip_reputation_{ip_address}', {
                'ip_address': ip_address,
                'results': results,
                'reputation_score': reputation_score,
                'timestamp': datetime.now().isoformat()
            }, 'ip_reputation')
            
            return {
                'success': True,
                'ip_address': ip_address,
                'reputation_score': reputation_score,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to check IP reputation for {ip_address}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_abuseipdb(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation in AbuseIPDB."""
        try:
            # Get API key from vault
            api_key = self.credential_vault.get_api_key('abuseipdb')
            if not api_key:
                return {'error': 'AbuseIPDB API key not found in vault'}
            
            url = f"{self.feed_configs['abuseipdb']['base_url']}check"
            params = {
                'ipAddress': ip_address,
                'maxAgeInDays': '90'
            }
            headers = {
                'Key': api_key,
                'Accept': 'application/json'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': data,
                'abuse_confidence': data.get('data', {}).get('abuseConfidenceScore', 0),
                'country': data.get('data', {}).get('countryCode'),
                'usage_type': data.get('data', {}).get('usageType')
            }
            
        except Exception as e:
            logger.error(f"Failed to check AbuseIPDB for {ip_address}: {e}")
            return {'error': str(e)}
    
    def _check_virustotal_ip(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation in VirusTotal."""
        try:
            # Get API key from vault
            api_key = self.credential_vault.get_api_key('virustotal')
            if not api_key:
                return {'error': 'VirusTotal API key not found in vault'}
            
            url = f"{self.feed_configs['virustotal']['base_url']}ip-address/report"
            params = {
                'apikey': api_key,
                'ip': ip_address
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': data,
                'positives': data.get('positives', 0),
                'total': data.get('total', 0),
                'detection_ratio': data.get('positives', 0) / max(data.get('total', 1), 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to check VirusTotal for {ip_address}: {e}")
            return {'error': str(e)}
    
    def _check_shodan_ip(self, ip_address: str) -> Dict[str, Any]:
        """Check IP information in Shodan."""
        try:
            # Get API key from vault
            api_key = self.credential_vault.get_api_key('shodan')
            if not api_key:
                return {'error': 'Shodan API key not found in vault'}
            
            url = f"{self.feed_configs['shodan']['base_url']}shodan/host/{ip_address}"
            params = {'key': api_key}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': data,
                'ports': data.get('ports', []),
                'hostnames': data.get('hostnames', []),
                'country': data.get('country_name'),
                'os': data.get('os')
            }
            
        except Exception as e:
            logger.error(f"Failed to check Shodan for {ip_address}: {e}")
            return {'error': str(e)}
    
    def _calculate_reputation_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall reputation score from multiple sources."""
        try:
            score = 0.0
            total_weight = 0.0
            
            # AbuseIPDB scoring
            if 'abuseipdb' in results and results['abuseipdb'].get('success'):
                abuse_score = results['abuseipdb'].get('abuse_confidence', 0)
                # Convert to 0-100 scale where 0 is good, 100 is bad
                score += (100 - abuse_score) * 0.4  # 40% weight
                total_weight += 0.4
            
            # VirusTotal scoring
            if 'virustotal' in results and results['virustotal'].get('success'):
                vt_score = results['virustotal'].get('detection_ratio', 0)
                # Convert to 0-100 scale where 0 is good, 100 is bad
                score += (100 - (vt_score * 100)) * 0.4  # 40% weight
                total_weight += 0.4
            
            # Shodan scoring (neutral, just for information)
            if 'shodan' in results and results['shodan'].get('success'):
                # Shodan provides information, not reputation
                score += 50 * 0.2  # 20% weight, neutral score
                total_weight += 0.2
            
            if total_weight > 0:
                return score / total_weight
            else:
                return 50.0  # Neutral score if no data
                
        except Exception as e:
            logger.error(f"Failed to calculate reputation score: {e}")
            return 50.0
    
    def fetch_urlhaus_malicious_urls(self, days: int = 1, 
                                   tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch malicious URLs from URLhaus.
        
        Args:
            days: Number of days to look back
            tag: Specific tag to filter by
            
        Returns:
            URLhaus malicious URL data
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit('urlhaus'):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded for URLhaus feed'
                }
            
            payload = {
                'query': 'recent',
                'days': days
            }
            
            if tag:
                payload['tag'] = tag
            
            response = self.session.post(
                self.feed_configs['urlhaus']['base_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('query_status') == 'ok':
                # Store in memory
                self._store_intel_data('urlhaus_malicious_urls', data, 'urlhaus')
                
                # Update rate limiting
                self._update_rate_limit('urlhaus')
                
                return {
                    'success': True,
                    'days': days,
                    'tag': tag,
                    'data': data,
                    'count': len(data.get('urls', [])),
                    'source': 'URLhaus',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'URLhaus query failed: {data.get("query_status")}'
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch URLhaus malicious URLs: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def fetch_otx_pulses(self, limit: int = 100, 
                         modified_since: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch threat intelligence pulses from AlienVault OTX.
        
        Args:
            limit: Maximum number of pulses to fetch
            modified_since: ISO timestamp to fetch pulses modified since
            
        Returns:
            OTX pulse data
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit('alienvault_otx'):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded for AlienVault OTX feed'
                }
            
            url = f"{self.feed_configs['alienvault_otx']['base_url']}pulses/subscribed"
            params = {'limit': limit}
            
            if modified_since:
                params['modified_since'] = modified_since
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Store in memory
            self._store_intel_data('otx_pulses', data, 'alienvault_otx')
            
            # Update rate limiting
            self._update_rate_limit('alienvault_otx')
            
            return {
                'success': True,
                'limit': limit,
                'modified_since': modified_since,
                'data': data,
                'count': len(data),
                'source': 'AlienVault OTX',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch OTX pulses: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_rate_limit(self, feed_name: str) -> bool:
        """Check if rate limit has been exceeded for a feed."""
        try:
            if feed_name not in self.rate_limits:
                return True
            
            current_time = time.time()
            last_fetch = self.rate_limits[feed_name].get('last_fetch', 0)
            request_count = self.rate_limits[feed_name].get('request_count', 0)
            
            # Reset counter if hour has passed
            if current_time - last_fetch > 3600:
                self.rate_limits[feed_name] = {
                    'last_fetch': current_time,
                    'request_count': 0
                }
                return True
            
            # Check if limit exceeded
            max_requests = self.feed_configs[feed_name]['rate_limit']
            return request_count < max_requests
            
        except Exception as e:
            logger.error(f"Failed to check rate limit for {feed_name}: {e}")
            return True
    
    def _update_rate_limit(self, feed_name: str):
        """Update rate limit counter for a feed."""
        try:
            if feed_name not in self.rate_limits:
                self.rate_limits[feed_name] = {
                    'last_fetch': time.time(),
                    'request_count': 0
                }
            
            self.rate_limits[feed_name]['request_count'] += 1
            
        except Exception as e:
            logger.error(f"Failed to update rate limit for {feed_name}: {e}")
    
    def _store_intel_data(self, data_type: str, data: Any, source: str):
        """Store intelligence data in memory."""
        try:
            self.memory_manager.import_data(
                data_type,
                data,
                domain="public_intel",
                tier="long_term",
                ttl_days=90,
                metadata={
                    'source': source,
                    'data_type': data_type,
                    'imported_at': datetime.now().isoformat(),
                    'size': len(str(data))
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store intel data in memory: {e}")
    
    def get_feed_status(self) -> Dict[str, Any]:
        """Get status of all intelligence feeds."""
        try:
            status = {}
            
            for feed_name, config in self.feed_configs.items():
                feed_status = {
                    'name': config['name'],
                    'description': config['description'],
                    'update_frequency': config['update_frequency'],
                    'rate_limit': config['rate_limit'],
                    'requires_auth': config['requires_auth']
                }
                
                # Add rate limiting info
                if feed_name in self.rate_limits:
                    feed_status['current_requests'] = self.rate_limits[feed_name]['request_count']
                    feed_status['last_fetch'] = datetime.fromtimestamp(
                        self.rate_limits[feed_name]['last_fetch']
                    ).isoformat()
                
                # Add last fetch info
                if feed_name in self.last_fetch:
                    feed_status['last_successful_fetch'] = self.last_fetch[feed_name]
                
                status[feed_name] = feed_status
            
            return {
                'success': True,
                'feeds': status,
                'total_feeds': len(status)
            }
            
        except Exception as e:
            logger.error(f"Failed to get feed status: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# MCP Tools for Public Intelligence Feeds
class PublicIntelMCPTools:
    """MCP-compatible tools for public intelligence feeds."""
    
    def __init__(self, public_intel: PublicIntelFeeds):
        self.intel_feeds = public_intel
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for public intelligence feeds."""
        return {
            "get_available_feeds": {
                "name": "get_available_feeds",
                "description": "Get list of available intelligence feeds",
                "parameters": {},
                "returns": {"type": "object", "description": "List of available feeds"}
            },
            "fetch_mitre_attack": {
                "name": "fetch_mitre_attack",
                "description": "Fetch MITRE ATT&CK data",
                "parameters": {
                    "feed_type": {"type": "string", "description": "Type of data to fetch (techniques, tactics, groups, software, campaigns, mitigations)"},
                    "include_deprecated": {"type": "boolean", "description": "Whether to include deprecated items"}
                },
                "returns": {"type": "object", "description": "MITRE ATT&CK data"}
            },
            "fetch_cve_data": {
                "name": "fetch_cve_data",
                "description": "Fetch CVE data from NVD database",
                "parameters": {
                    "cve_id": {"type": "string", "description": "Specific CVE ID to fetch"},
                    "keyword": {"type": "string", "description": "Keyword to search for"},
                    "limit": {"type": "integer", "description": "Maximum number of results"}
                },
                "returns": {"type": "object", "description": "CVE data"}
            },
            "fetch_threatfox_iocs": {
                "name": "fetch_threatfox_iocs",
                "description": "Fetch IOCs from ThreatFox",
                "parameters": {
                    "days": {"type": "integer", "description": "Number of days to look back"},
                    "malware_type": {"type": "string", "description": "Specific malware type to filter by"}
                },
                "returns": {"type": "object", "description": "ThreatFox IOC data"}
            },
            "check_ip_reputation": {
                "name": "check_ip_reputation",
                "description": "Check IP reputation across multiple sources",
                "parameters": {
                    "ip_address": {"type": "string", "description": "IP address to check"},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "List of sources to check"}
                },
                "returns": {"type": "object", "description": "IP reputation data"}
            },
            "fetch_urlhaus_malicious_urls": {
                "name": "fetch_urlhaus_malicious_urls",
                "description": "Fetch malicious URLs from URLhaus",
                "parameters": {
                    "days": {"type": "integer", "description": "Number of days to look back"},
                    "tag": {"type": "string", "description": "Specific tag to filter by"}
                },
                "returns": {"type": "object", "description": "URLhaus malicious URL data"}
            },
            "fetch_otx_pulses": {
                "name": "fetch_otx_pulses",
                "description": "Fetch threat intelligence pulses from AlienVault OTX",
                "parameters": {
                    "limit": {"type": "integer", "description": "Maximum number of pulses to fetch"},
                    "modified_since": {"type": "string", "description": "ISO timestamp to fetch pulses modified since"}
                },
                "returns": {"type": "object", "description": "OTX pulse data"}
            },
            "get_feed_status": {
                "name": "get_feed_status",
                "description": "Get status of all intelligence feeds",
                "parameters": {},
                "returns": {"type": "object", "description": "Feed status information"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute public intelligence MCP tool."""
        if tool_name == "get_available_feeds":
            return self.intel_feeds.get_available_feeds()
        elif tool_name == "fetch_mitre_attack":
            return self.intel_feeds.fetch_mitre_attack(**kwargs)
        elif tool_name == "fetch_cve_data":
            return self.intel_feeds.fetch_cve_data(**kwargs)
        elif tool_name == "fetch_threatfox_iocs":
            return self.intel_feeds.fetch_threatfox_iocs(**kwargs)
        elif tool_name == "check_ip_reputation":
            return self.intel_feeds.check_ip_reputation(**kwargs)
        elif tool_name == "fetch_urlhaus_malicious_urls":
            return self.intel_feeds.fetch_urlhaus_malicious_urls(**kwargs)
        elif tool_name == "fetch_otx_pulses":
            return self.intel_feeds.fetch_otx_pulses(**kwargs)
        elif tool_name == "get_feed_status":
            return self.intel_feeds.get_feed_status()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

