"""
Enhanced Splunk Tools for Cybersecurity Agent

Provides comprehensive Splunk data exploration, discovery, enrichment, and analysis capabilities.
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import re
from pathlib import Path

from .splunk_integration import SplunkIntegration
from .enhanced_session_manager import EnhancedSessionManager
from .context_memory_manager import ContextMemoryManager

logger = logging.getLogger(__name__)

class EnhancedSplunkTools:
    """Enhanced Splunk tools for data discovery, enrichment, and analysis."""
    
    def __init__(self, splunk_integration: SplunkIntegration, 
                 session_manager: EnhancedSessionManager,
                 memory_manager: ContextMemoryManager):
        self.splunk = splunk_integration
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        
        # Predefined query templates for common data discovery patterns
        self.query_templates = self._initialize_query_templates()
        
        # Data enrichment patterns
        self.enrichment_patterns = self._initialize_enrichment_patterns()
    
    def _initialize_query_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined query templates for data discovery."""
        return {
            "data_discovery": {
                "name": "Data Discovery",
                "description": "Discover what data is available in Splunk indexes",
                "query": "index!=* splunk!=* earliest=-4h | stats latest(_time) as last_seen, latest(_raw) as sample_message by index, sourcetype, app",
                "parameters": {
                    "time_range": "-4h",
                    "exclude_indexes": ["*", "splunk*"],
                    "summary_fields": ["index", "sourcetype", "app"]
                }
            },
            "index_analysis": {
                "name": "Index Analysis",
                "description": "Analyze index usage and data patterns",
                "query": "| tstats count by index, sourcetype, app earliest=-24h | sort -count",
                "parameters": {
                    "time_range": "-24h",
                    "group_by": ["index", "sourcetype", "app"],
                    "sort_by": "count"
                }
            },
            "data_flow_analysis": {
                "name": "Data Flow Analysis",
                "description": "Analyze data flow and volume patterns",
                "query": "| tstats count, avg(_time) as avg_time by index, sourcetype, app, span=1h earliest=-24h | sort index, sourcetype, -count",
                "parameters": {
                    "time_range": "-24h",
                    "span": "1h",
                    "metrics": ["count", "avg_time"]
                }
            },
            "source_technology_discovery": {
                "name": "Source Technology Discovery",
                "description": "Discover what technologies are feeding into Splunk",
                "query": "index!=* splunk!=* earliest=-24h | stats latest(_time) as last_seen, latest(_raw) as sample_message, count by index, sourcetype, app, host, source",
                "parameters": {
                    "time_range": "-24h",
                    "exclude_indexes": ["*", "splunk*"],
                    "group_by": ["index", "sourcetype", "app", "host", "source"]
                }
            },
            "data_quality_assessment": {
                "name": "Data Quality Assessment",
                "description": "Assess data quality and completeness",
                "query": "index!=* splunk!=* earliest=-24h | stats count, dc(host) as unique_hosts, dc(source) as unique_sources, latest(_time) as last_seen by index, sourcetype, app",
                "parameters": {
                    "time_range": "-24h",
                    "exclude_indexes": ["*", "splunk*"],
                    "quality_metrics": ["count", "unique_hosts", "unique_sources", "last_seen"]
                }
            }
        }
    
    def _initialize_enrichment_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data enrichment patterns."""
        return {
            "ip_address": {
                "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                "description": "Extract and enrich IP addresses",
                "enrichment_fields": ["geo_location", "threat_intel", "network_info"]
            },
            "domain_names": {
                "pattern": r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
                "description": "Extract and enrich domain names",
                "enrichment_fields": ["dns_info", "reputation", "ssl_cert"]
            },
            "email_addresses": {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "description": "Extract and enrich email addresses",
                "enrichment_fields": ["domain_info", "breach_data", "reputation"]
            },
            "urls": {
                "pattern": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                "description": "Extract and enrich URLs",
                "enrichment_fields": ["domain_info", "category", "reputation"]
            },
            "file_paths": {
                "pattern": r'(?:[A-Za-z]:\\)?(?:[^<>:"/\\|?*\x00-\x1f]*[^<>:"/\\|?*\x00-\x1f\s])',
                "description": "Extract and enrich file paths",
                "enrichment_fields": ["file_type", "hash_info", "reputation"]
            }
        }
    
    def discover_data_sources(self, time_range: str = "-4h", 
                            exclude_indexes: Optional[List[str]] = None,
                            summary_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Discover what data sources are feeding into Splunk indexes.
        
        Args:
            time_range: Time range for the search (e.g., "-4h", "-24h", "-7d")
            exclude_indexes: Indexes to exclude from search
            summary_fields: Fields to group by in the summary
            
        Returns:
            Dictionary containing discovery results and analysis
        """
        try:
            # Build the discovery query
            if exclude_indexes is None:
                exclude_indexes = ["*", "splunk*"]
            
            if summary_fields is None:
                summary_fields = ["index", "sourcetype", "app"]
            
            # Create the query
            exclude_clause = " ".join([f"index!={idx}" for idx in exclude_indexes])
            group_by_clause = ", ".join(summary_fields)
            
            query = f"{exclude_clause} earliest={time_range} | stats latest(_time) as last_seen, latest(_raw) as sample_message, count by {group_by_clause}"
            
            # Execute the search
            results = self.splunk.execute_search(query, max_results=1000)
            
            if not results.get('success', False):
                return {
                    'success': False,
                    'error': results.get('error', 'Unknown error'),
                    'query': query
                }
            
            # Process and analyze results
            data_sources = self._analyze_data_sources(results.get('data', []))
            
            # Store results in memory for future reference
            self._store_discovery_results(data_sources, time_range)
            
            # Save results to session
            self._save_discovery_results(data_sources, query)
            
            return {
                'success': True,
                'query': query,
                'time_range': time_range,
                'total_sources': len(data_sources),
                'data_sources': data_sources,
                'analysis': self._generate_data_source_analysis(data_sources)
            }
            
        except Exception as e:
            logger.error(f"Failed to discover data sources: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_data_sources(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze raw data source discovery results."""
        try:
            analyzed_sources = []
            
            for item in raw_data:
                source_info = {
                    'index': item.get('index', 'unknown'),
                    'sourcetype': item.get('sourcetype', 'unknown'),
                    'app': item.get('app', 'unknown'),
                    'last_seen': item.get('last_seen', 'unknown'),
                    'sample_message': item.get('sample_message', ''),
                    'count': item.get('count', 0),
                    'data_volume': self._categorize_data_volume(item.get('count', 0)),
                    'technology_type': self._identify_technology_type(item.get('sourcetype', ''), item.get('app', '')),
                    'data_category': self._categorize_data_type(item.get('sourcetype', ''), item.get('app', '')),
                    'enrichment_opportunities': self._identify_enrichment_opportunities(item.get('sample_message', ''))
                }
                
                analyzed_sources.append(source_info)
            
            return analyzed_sources
            
        except Exception as e:
            logger.error(f"Failed to analyze data sources: {e}")
            return []
    
    def _categorize_data_volume(self, count: int) -> str:
        """Categorize data volume based on count."""
        if count < 1000:
            return "low"
        elif count < 10000:
            return "medium"
        elif count < 100000:
            return "high"
        else:
            return "very_high"
    
    def _identify_technology_type(self, sourcetype: str, app: str) -> str:
        """Identify the type of technology based on sourcetype and app."""
        technology_patterns = {
            'windows': ['win', 'windows', 'eventlog', 'wineventlog'],
            'linux': ['linux', 'syslog', 'rsyslog', 'systemd'],
            'network': ['netflow', 'pcap', 'firewall', 'ids', 'ips'],
            'security': ['security', 'audit', 'compliance', 'siem'],
            'application': ['web', 'app', 'database', 'middleware'],
            'infrastructure': ['infra', 'monitoring', 'metrics', 'logs']
        }
        
        combined_text = f"{sourcetype} {app}".lower()
        
        for tech_type, patterns in technology_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return tech_type
        
        return "unknown"
    
    def _categorize_data_type(self, sourcetype: str, app: str) -> str:
        """Categorize the type of data being collected."""
        data_patterns = {
            'security_events': ['security', 'audit', 'compliance', 'siem', 'ids', 'ips'],
            'system_logs': ['syslog', 'eventlog', 'system', 'os'],
            'network_traffic': ['netflow', 'pcap', 'network', 'traffic'],
            'application_logs': ['web', 'app', 'database', 'middleware'],
            'performance_metrics': ['metrics', 'performance', 'monitoring'],
            'user_activity': ['user', 'login', 'authentication', 'access']
        }
        
        combined_text = f"{sourcetype} {app}".lower()
        
        for data_type, patterns in data_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return data_type
        
        return "general_logs"
    
    def _identify_enrichment_opportunities(self, sample_message: str) -> List[str]:
        """Identify opportunities for data enrichment."""
        opportunities = []
        
        for pattern_name, pattern_info in self.enrichment_patterns.items():
            if re.search(pattern_info['pattern'], sample_message):
                opportunities.extend(pattern_info['enrichment_fields'])
        
        return list(set(opportunities))  # Remove duplicates
    
    def _generate_data_source_analysis(self, data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis of data sources."""
        try:
            analysis = {
                'summary': {
                    'total_indexes': len(set(source['index'] for source in data_sources)),
                    'total_sourcetypes': len(set(source['sourcetype'] for source in data_sources)),
                    'total_apps': len(set(source['app'] for source in data_sources)),
                    'total_sources': len(data_sources)
                },
                'volume_distribution': {},
                'technology_distribution': {},
                'data_category_distribution': {},
                'enrichment_opportunities': {},
                'recommendations': []
            }
            
            # Analyze volume distribution
            for source in data_sources:
                volume = source['data_volume']
                analysis['volume_distribution'][volume] = analysis['volume_distribution'].get(volume, 0) + 1
            
            # Analyze technology distribution
            for source in data_sources:
                tech_type = source['technology_type']
                analysis['technology_distribution'][tech_type] = analysis['technology_distribution'].get(tech_type, 0) + 1
            
            # Analyze data category distribution
            for source in data_sources:
                data_category = source['data_category']
                analysis['data_category_distribution'][data_category] = analysis['data_category_distribution'].get(data_category, 0) + 1
            
            # Analyze enrichment opportunities
            for source in data_sources:
                for opportunity in source['enrichment_opportunities']:
                    analysis['enrichment_opportunities'][opportunity] = analysis['enrichment_opportunities'].get(opportunity, 0) + 1
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate data source analysis: {e}")
            return {}
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on data source analysis."""
        recommendations = []
        
        # Volume-based recommendations
        if analysis['volume_distribution'].get('very_high', 0) > 0:
            recommendations.append("Consider implementing data retention policies for high-volume indexes")
        
        if analysis['volume_distribution'].get('low', 0) > 0:
            recommendations.append("Investigate low-volume data sources for potential data loss issues")
        
        # Technology-based recommendations
        if analysis['technology_distribution'].get('security', 0) > 0:
            recommendations.append("Security data sources detected - ensure proper correlation and alerting")
        
        if analysis['technology_distribution'].get('infrastructure', 0) > 0:
            recommendations.append("Infrastructure monitoring data available - consider performance dashboards")
        
        # Enrichment recommendations
        if analysis['enrichment_opportunities'].get('threat_intel', 0) > 0:
            recommendations.append("Threat intelligence enrichment opportunities identified - consider implementing enrichment workflows")
        
        if analysis['enrichment_opportunities'].get('geo_location', 0) > 0:
            recommendations.append("Geolocation enrichment opportunities available - consider implementing location-based analytics")
        
        return recommendations
    
    def _store_discovery_results(self, data_sources: List[Dict[str, Any]], time_range: str):
        """Store discovery results in memory for future reference."""
        try:
            self.memory_manager.import_data(
                "splunk_data_discovery",
                {
                    'data_sources': data_sources,
                    'discovery_time': datetime.now().isoformat(),
                    'time_range': time_range,
                    'total_sources': len(data_sources)
                },
                domain="splunk_integration",
                tier="medium_term",
                ttl_days=30,
                metadata={
                    'description': f"Splunk data source discovery results for {time_range}",
                    'type': 'data_discovery',
                    'source': 'splunk'
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store discovery results in memory: {e}")
    
    def _save_discovery_results(self, data_sources: List[Dict[str, Any]], query: str):
        """Save discovery results to session."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as DataFrame
            df = pd.DataFrame(data_sources)
            filename = f"splunk_data_discovery_{timestamp}"
            self.session_manager.save_dataframe(
                df,
                filename,
                f"Splunk data source discovery results - Query: {query}"
            )
            
            # Save analysis summary
            analysis = self._generate_data_source_analysis(data_sources)
            summary_filename = f"splunk_discovery_analysis_{timestamp}"
            self.session_manager.save_text_output(
                json.dumps(analysis, indent=2),
                summary_filename,
                f"Splunk data discovery analysis summary"
            )
            
        except Exception as e:
            logger.warning(f"Failed to save discovery results: {e}")
    
    def analyze_index_performance(self, time_range: str = "-24h") -> Dict[str, Any]:
        """Analyze Splunk index performance and usage patterns."""
        try:
            query = f"| tstats count, avg(_time) as avg_time, latest(_time) as last_seen by index, sourcetype, app, span=1h earliest={time_range} | sort index, sourcetype, -count"
            
            results = self.splunk.execute_search(query, max_results=1000)
            
            if not results.get('success', False):
                return {
                    'success': False,
                    'error': results.get('error', 'Unknown error'),
                    'query': query
                }
            
            # Process performance data
            performance_data = self._analyze_performance_data(results.get('data', []))
            
            # Save results
            self._save_performance_analysis(performance_data, query)
            
            return {
                'success': True,
                'query': query,
                'time_range': time_range,
                'performance_data': performance_data,
                'summary': self._generate_performance_summary(performance_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze index performance: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_performance_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze raw performance data."""
        try:
            analyzed_data = []
            
            for item in raw_data:
                performance_info = {
                    'index': item.get('index', 'unknown'),
                    'sourcetype': item.get('sourcetype', 'unknown'),
                    'app': item.get('app', 'unknown'),
                    'count': item.get('count', 0),
                    'avg_time': item.get('avg_time', 0),
                    'last_seen': item.get('last_seen', 'unknown'),
                    'performance_category': self._categorize_performance(item.get('count', 0)),
                    'data_freshness': self._assess_data_freshness(item.get('last_seen', 'unknown'))
                }
                
                analyzed_data.append(performance_info)
            
            return analyzed_data
            
        except Exception as e:
            logger.error(f"Failed to analyze performance data: {e}")
            return []
    
    def _categorize_performance(self, count: int) -> str:
        """Categorize performance based on event count."""
        if count < 100:
            return "low_volume"
        elif count < 1000:
            return "medium_volume"
        elif count < 10000:
            return "high_volume"
        else:
            return "very_high_volume"
    
    def _assess_data_freshness(self, last_seen: str) -> str:
        """Assess how fresh the data is."""
        try:
            if isinstance(last_seen, str):
                # Parse the timestamp
                if 'T' in last_seen:
                    last_time = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                else:
                    last_time = datetime.fromtimestamp(float(last_seen))
                
                time_diff = datetime.now() - last_time
                
                if time_diff < timedelta(hours=1):
                    return "very_fresh"
                elif time_diff < timedelta(hours=6):
                    return "fresh"
                elif time_diff < timedelta(days=1):
                    return "recent"
                else:
                    return "stale"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _generate_performance_summary(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary."""
        try:
            summary = {
                'total_indexes': len(set(item['index'] for item in performance_data)),
                'total_sourcetypes': len(set(item['sourcetype'] for item in performance_data)),
                'total_apps': len(set(item['app'] for item in performance_data)),
                'performance_distribution': {},
                'freshness_distribution': {},
                'top_performers': [],
                'recommendations': []
            }
            
            # Analyze distributions
            for item in performance_data:
                perf_cat = item['performance_category']
                freshness = item['data_freshness']
                
                summary['performance_distribution'][perf_cat] = summary['performance_distribution'].get(perf_cat, 0) + 1
                summary['freshness_distribution'][freshness] = summary['freshness_distribution'].get(freshness, 0) + 1
            
            # Identify top performers
            sorted_data = sorted(performance_data, key=lambda x: x.get('count', 0), reverse=True)
            summary['top_performers'] = sorted_data[:10]
            
            # Generate recommendations
            summary['recommendations'] = self._generate_performance_recommendations(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {}
    
    def _generate_performance_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance-based recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if summary['performance_distribution'].get('very_high_volume', 0) > 0:
            recommendations.append("High-volume indexes detected - consider implementing data retention and archiving policies")
        
        if summary['freshness_distribution'].get('stale', 0) > 0:
            recommendations.append("Stale data detected - investigate data collection issues and pipeline health")
        
        if summary['performance_distribution'].get('low_volume', 0) > 0:
            recommendations.append("Low-volume indexes detected - consider consolidating or investigating data loss")
        
        return recommendations
    
    def _save_performance_analysis(self, performance_data: List[Dict[str, Any]], query: str):
        """Save performance analysis results to session."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as DataFrame
            df = pd.DataFrame(performance_data)
            filename = f"splunk_performance_analysis_{timestamp}"
            self.session_manager.save_dataframe(
                df,
                filename,
                f"Splunk performance analysis results - Query: {query}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to save performance analysis: {e}")
    
    def get_query_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available query templates."""
        return self.query_templates
    
    def execute_custom_query(self, query: str, max_results: int = 1000) -> Dict[str, Any]:
        """Execute a custom Splunk query."""
        try:
            results = self.splunk.execute_search(query, max_results=max_results)
            
            if results.get('success', False):
                # Save results to session
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"custom_splunk_query_{timestamp}"
                
                if results.get('data'):
                    df = pd.DataFrame(results['data'])
                    self.session_manager.save_dataframe(
                        df,
                        filename,
                        f"Custom Splunk query results: {query}"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# MCP Tools for Enhanced Splunk Tools
class EnhancedSplunkMCPTools:
    """MCP-compatible tools for enhanced Splunk functionality."""
    
    def __init__(self, enhanced_splunk: EnhancedSplunkTools):
        self.splunk_tools = enhanced_splunk
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for enhanced Splunk tools."""
        return {
            "discover_data_sources": {
                "name": "discover_data_sources",
                "description": "Discover what data sources are feeding into Splunk indexes",
                "parameters": {
                    "time_range": {"type": "string", "description": "Time range for search (e.g., -4h, -24h, -7d)"},
                    "exclude_indexes": {"type": "array", "items": {"type": "string"}, "description": "Indexes to exclude from search"},
                    "summary_fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to group by in summary"}
                },
                "returns": {"type": "object", "description": "Data source discovery results and analysis"}
            },
            "analyze_index_performance": {
                "name": "analyze_index_performance",
                "description": "Analyze Splunk index performance and usage patterns",
                "parameters": {
                    "time_range": {"type": "string", "description": "Time range for analysis (e.g., -24h, -7d)"}
                },
                "returns": {"type": "object", "description": "Index performance analysis results"}
            },
            "execute_custom_query": {
                "name": "execute_custom_query",
                "description": "Execute a custom Splunk query",
                "parameters": {
                    "query": {"type": "string", "description": "Custom Splunk query to execute"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"}
                },
                "returns": {"type": "object", "description": "Query execution results"}
            },
            "get_query_templates": {
                "name": "get_query_templates",
                "description": "Get available Splunk query templates",
                "parameters": {},
                "returns": {"type": "object", "description": "Available query templates"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute enhanced Splunk MCP tool."""
        if tool_name == "discover_data_sources":
            return self.splunk_tools.discover_data_sources(**kwargs)
        elif tool_name == "analyze_index_performance":
            return self.splunk_tools.analyze_index_performance(**kwargs)
        elif tool_name == "execute_custom_query":
            return self.splunk_tools.execute_custom_query(**kwargs)
        elif tool_name == "get_query_templates":
            return self.splunk_tools.get_query_templates()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

