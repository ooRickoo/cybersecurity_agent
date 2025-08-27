#!/usr/bin/env python3
"""
Browser MCP Tools for Cybersecurity Frameworks

Provides lightweight browser functionality to download and process
cybersecurity frameworks like MITRE ATT&CK, D3FEND, NIST SP 800-53.
"""

import asyncio
import json
import logging
import sys
import os
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import re

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")

logger = logging.getLogger(__name__)

@dataclass
class FrameworkMetadata:
    """Metadata for cybersecurity frameworks."""
    name: str
    version: str
    source_url: str
    download_url: str
    format_type: str  # 'stix', 'json', 'xml', 'csv'
    last_updated: datetime
    file_size: int
    checksum: str
    description: str
    tags: List[str] = field(default_factory=list)

@dataclass
class FlattenedFramework:
    """Flattened framework data for easy querying."""
    framework_id: str
    framework_name: str
    flatten_date: datetime
    total_items: int
    flattened_data: Dict[str, Any]
    query_index: Dict[str, List[str]]  # Search index for fast lookups
    metadata: FrameworkMetadata

class BrowserMCPTools:
    """MCP tools for browser functionality and framework processing."""
    
    def __init__(self):
        """Initialize browser MCP tools."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Cybersecurity-Agent/1.0)'
        })
        
        # Framework registry
        self.framework_registry = self._initialize_framework_registry()
        
        # Download cache
        self.download_cache = {}
        
        # Processing history
        self.processing_history = []
        
        logger.info("🚀 Browser MCP Tools initialized")
    
    def _initialize_framework_registry(self) -> Dict[str, FrameworkMetadata]:
        """Initialize registry of known cybersecurity frameworks."""
        return {
            "mitre_attack": FrameworkMetadata(
                name="MITRE ATT&CK",
                version="latest",
                source_url="https://attack.mitre.org/",
                download_url="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
                format_type="json",  # STIX data model in JSON format
                last_updated=datetime.now(),
                file_size=0,
                checksum="",
                description="MITRE ATT&CK Enterprise framework for cyber threat intelligence",
                tags=["threat_intelligence", "attack_patterns", "enterprise"]
            ),
            "mitre_d3fend": FrameworkMetadata(
                name="MITRE D3FEND",
                version="latest",
                source_url="https://d3fend.mitre.org/",
                download_url="https://raw.githubusercontent.com/mitre/d3fend/master/ontology/d3fend.json",
                format_type="json",
                last_updated=datetime.now(),
                file_size=0,
                checksum="",
                description="MITRE D3FEND framework for defensive techniques",
                tags=["defense", "countermeasures", "security_controls"]
            ),
            "nist_sp800_53": FrameworkMetadata(
                name="NIST SP 800-53",
                version="Rev 5",
                source_url="https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final",
                download_url="https://raw.githubusercontent.com/usnistgov/oscal-content/master/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
                format_type="json",
                last_updated=datetime.now(),
                file_size=0,
                checksum="",
                description="NIST SP 800-53 Security and Privacy Controls",
                tags=["compliance", "security_controls", "privacy", "nist"]
            ),
            "cve_database": FrameworkMetadata(
                name="CVE Database",
                version="latest",
                source_url="https://cve.mitre.org/",
                download_url="https://cve.mitre.org/data/downloads/allitems.csv",
                format_type="csv",
                last_updated=datetime.now(),
                file_size=0,
                checksum="",
                description="Common Vulnerabilities and Exposures database",
                tags=["vulnerabilities", "cve", "security_advisories"]
            ),
            "capec_database": FrameworkMetadata(
                name="CAPEC Database",
                version="latest",
                source_url="https://capec.mitre.org/",
                download_url="https://raw.githubusercontent.com/mitre/capec/master/capec_v3.12.xml",
                format_type="xml",
                last_updated=datetime.now(),
                file_size=0,
                checksum="",
                description="Common Attack Pattern Enumeration and Classification",
                tags=["attack_patterns", "capec", "threat_modeling"]
            )
        }
    
    async def search_online(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search online for cybersecurity information."""
        try:
            logger.info(f"🔍 Searching online for: {query}")
            
            # For now, we'll use a simple search simulation
            # In production, this could integrate with search APIs
            
            search_results = []
            
            # Search through known frameworks
            for framework_id, framework in self.framework_registry.items():
                if query.lower() in framework.name.lower() or any(tag.lower() in query.lower() for tag in framework.tags):
                    search_results.append({
                        "title": framework.name,
                        "url": framework.source_url,
                        "description": framework.description,
                        "tags": framework.tags,
                        "framework_id": framework_id
                    })
            
            # Add some generic search results
            if "threat" in query.lower():
                search_results.append({
                    "title": "Threat Intelligence Resources",
                    "url": "https://www.threatintelligence.com",
                    "description": "Comprehensive threat intelligence resources and tools",
                    "tags": ["threat_intelligence", "security"],
                    "framework_id": None
                })
            
            if "compliance" in query.lower():
                search_results.append({
                    "title": "Cybersecurity Compliance Guide",
                    "url": "https://www.nist.gov/cyberframework",
                    "description": "NIST Cybersecurity Framework and compliance resources",
                    "tags": ["compliance", "nist", "cybersecurity"],
                    "framework_id": None
                })
            
            return {
                "success": True,
                "query": query,
                "results": search_results[:max_results],
                "total_found": len(search_results),
                "search_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Online search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def download_framework(self, framework_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Download a cybersecurity framework."""
        try:
            if framework_id not in self.framework_registry:
                return {"success": False, "error": f"Unknown framework: {framework_id}"}
            
            framework = self.framework_registry[framework_id]
            
            # Check cache
            cache_key = f"{framework_id}_{framework.version}"
            if not force_refresh and cache_key in self.download_cache:
                cached_data = self.download_cache[cache_key]
                if (datetime.now() - cached_data['download_time']).days < 1:  # 1 day cache
                    logger.info(f"📦 Using cached framework: {framework.name}")
                    return {
                        "success": True,
                        "framework": framework_id,
                        "cached": True,
                        "data": cached_data['data'],
                        "metadata": framework
                    }
            
            logger.info(f"📥 Downloading framework: {framework.name}")
            
            # Download the framework
            response = self.session.get(framework.download_url, timeout=30)
            response.raise_for_status()
            
            # Parse content based on format
            if framework.format_type == "json":
                data = response.json()
            elif framework.format_type == "xml":
                data = self._parse_xml_to_dict(response.text)
            elif framework.format_type == "csv":
                data = self._parse_csv_to_dict(response.text)
            else:
                data = response.text
            
            # Update framework metadata
            framework.file_size = len(response.content)
            framework.checksum = hashlib.sha256(response.content).hexdigest()
            framework.last_updated = datetime.now()
            
            # Cache the downloaded data
            self.download_cache[cache_key] = {
                "data": data,
                "download_time": datetime.now(),
                "file_size": framework.file_size,
                "checksum": framework.checksum
            }
            
            # Log processing history
            self.processing_history.append({
                "action": "download_framework",
                "framework_id": framework_id,
                "timestamp": datetime.now(),
                "file_size": framework.file_size,
                "success": True
            })
            
            return {
                "success": True,
                "framework": framework_id,
                "cached": False,
                "data": data,
                "metadata": framework,
                "download_time": datetime.now().isoformat(),
                "file_size": framework.file_size
            }
            
        except Exception as e:
            logger.error(f"❌ Framework download failed: {e}")
            
            # Log failure
            self.processing_history.append({
                "action": "download_framework",
                "framework_id": framework_id,
                "timestamp": datetime.now(),
                "error": str(e),
                "success": False
            })
            
            return {"success": False, "error": str(e)}
    
    def _parse_xml_to_dict(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML content to dictionary."""
        try:
            root = ET.fromstring(xml_content)
            return self._xml_element_to_dict(root)
        except Exception as e:
            logger.error(f"❌ XML parsing failed: {e}")
            return {"error": f"XML parsing failed: {e}", "raw_content": xml_content}
    
    def _xml_element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result["text"] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_element_to_dict(child)
            child_tag = child.tag
            
            if child_tag in result:
                if not isinstance(result[child_tag], list):
                    result[child_tag] = [result[child_tag]]
                result[child_tag].append(child_data)
            else:
                result[child_tag] = child_data
        
        return result
    
    def _parse_csv_to_dict(self, csv_content: str) -> Dict[str, Any]:
        """Parse CSV content to dictionary."""
        try:
            lines = csv_content.strip().split('\n')
            if not lines:
                return {"error": "Empty CSV content"}
            
            headers = lines[0].split(',')
            data = []
            
            for line in lines[1:]:
                values = line.split(',')
                row = {}
                for i, header in enumerate(headers):
                    if i < len(values):
                        row[header.strip()] = values[i].strip()
                    else:
                        row[header.strip()] = ""
                data.append(row)
            
            return {
                "headers": headers,
                "data": data,
                "total_rows": len(data)
            }
            
        except Exception as e:
            logger.error(f"❌ CSV parsing failed: {e}")
            return {"error": f"CSV parsing failed: {e}", "raw_content": csv_content}
    
    async def flatten_framework(self, framework_id: str, flatten_strategy: str = "auto") -> Dict[str, Any]:
        """Flatten a cybersecurity framework for easy querying."""
        try:
            # Download framework if not already cached
            download_result = await self.download_framework(framework_id)
            if not download_result["success"]:
                return download_result
            
            framework_data = download_result["data"]
            framework_meta = download_result["metadata"]
            
            logger.info(f"🔄 Flattening framework: {framework_meta.name} (format: {framework_meta.format_type})")
            
            # Add debug logging
            if isinstance(framework_data, dict):
                logger.info(f"📊 Framework data keys: {list(framework_data.keys())}")
                if "objects" in framework_data:
                    logger.info(f"📊 Found {len(framework_data['objects'])} objects")
            else:
                logger.info(f"📊 Framework data type: {type(framework_data)}")
            
            # Flatten based on format type
            if framework_meta.format_type == "stix":
                flattened_data = self._flatten_stix_framework(framework_data)
            elif framework_meta.format_type == "json":
                flattened_data = self._flatten_json_framework(framework_data, framework_id)
            elif framework_meta.format_type == "xml":
                flattened_data = self._flatten_xml_framework(framework_data, framework_id)
            elif framework_meta.format_type == "csv":
                flattened_data = self._flatten_csv_framework(framework_data, framework_id)
            else:
                flattened_data = self._flatten_generic_framework(framework_data, framework_id)
            
            # Create flattened framework object
            flattened_framework = FlattenedFramework(
                framework_id=framework_id,  # Use original framework_id for consistency
                framework_name=framework_meta.name,
                flatten_date=datetime.now(),
                total_items=flattened_data.get("total_items", 0),
                flattened_data=flattened_data,
                query_index=flattened_data.get("query_index", {}),
                metadata=framework_meta
            )
            
            # Log processing history
            self.processing_history.append({
                "action": "flatten_framework",
                "framework_id": framework_id,
                "timestamp": datetime.now(),
                "total_items": flattened_framework.total_items,
                "success": True
            })
            
            return {
                "success": True,
                "framework": framework_id,
                "flattened_framework": {
                    "id": flattened_framework.framework_id,
                    "name": flattened_framework.framework_name,
                    "flatten_date": flattened_framework.flatten_date.isoformat(),
                    "total_items": flattened_framework.total_items,
                    "items": flattened_framework.flattened_data.get("items", []),
                    "query_index": flattened_framework.query_index,  # Return full query index
                    "query_index_keys": list(flattened_framework.query_index.keys()),
                    "metadata": {
                        "name": framework_meta.name,
                        "version": framework_meta.version,
                        "format_type": framework_meta.format_type,
                        "tags": framework_meta.tags
                    }
                },
                "flatten_strategy": flatten_strategy
            }
            
        except Exception as e:
            logger.error(f"❌ Framework flattening failed: {e}")
            
            # Log failure
            self.processing_history.append({
                "action": "flatten_framework",
                "framework_id": framework_id,
                "timestamp": datetime.now(),
                "error": str(e),
                "success": False
            })
            
            return {"success": False, "error": str(e)}
    
    def _flatten_stix_framework(self, stix_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten STIX framework data."""
        flattened_items = []
        query_index = {}
        
        try:
            # Extract STIX objects
            objects = stix_data.get("objects", [])
            
            for obj in objects:
                obj_type = obj.get("type", "unknown")
                obj_id = obj.get("id", "unknown")
                
                # Flatten the object
                flattened_obj = {
                    "id": obj_id,
                    "type": obj_type,
                    "created": obj.get("created"),
                    "modified": obj.get("modified"),
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "properties": {}
                }
                
                # Extract all properties
                for key, value in obj.items():
                    if key not in ["id", "type", "created", "modified", "name", "description"]:
                        flattened_obj["properties"][key] = value
                
                flattened_items.append(flattened_obj)
                
                # Build query index
                if obj.get("name"):
                    name_lower = obj.get("name", "").lower()
                    for word in name_lower.split():
                        if word not in query_index:
                            query_index[word] = []
                        query_index[word].append(obj_id)
                
                if obj.get("description"):
                    desc_lower = obj.get("description", "").lower()
                    for word in re.findall(r'\b\w+\b', desc_lower):
                        if word not in query_index:
                            query_index[word] = []
                        query_index[word].append(obj_id)
            
            return {
                "total_items": len(flattened_items),
                "items": flattened_items,
                "query_index": query_index,
                "framework_type": "stix"
            }
            
        except Exception as e:
            logger.error(f"❌ STIX flattening failed: {e}")
            return {"error": f"STIX flattening failed: {e}", "total_items": 0}
    
    def _flatten_json_framework(self, json_data: Dict[str, Any], framework_id: str) -> Dict[str, Any]:
        """Flatten JSON framework data."""
        flattened_items = []
        query_index = {}
        
        try:
            # Handle different JSON structures
            if isinstance(json_data, dict):
                if "catalog" in json_data:  # NIST SP 800-53 style
                    items = json_data.get("catalog", {}).get("groups", [])
                    for group in items:
                        for control in group.get("controls", []):
                            flattened_items.append({
                                "id": control.get("id", ""),
                                "type": "control",
                                "title": control.get("title", ""),
                                "description": control.get("description", ""),
                                "properties": control
                            })
                
                elif "objects" in json_data:  # STIX-style objects (MITRE ATT&CK)
                    for obj in json_data["objects"]:
                        if isinstance(obj, dict):
                            flattened_items.append({
                                "id": obj.get("id", ""),
                                "type": obj.get("type", "unknown"),
                                "name": obj.get("name", ""),
                                "description": obj.get("description", ""),
                                "properties": obj
                            })
                
                else:  # Generic structure
                    for key, value in json_data.items():
                        if isinstance(value, (dict, list)):
                            flattened_items.append({
                                "id": key,
                                "type": "item",
                                "name": key,
                                "description": str(value)[:200],
                                "properties": value
                            })
            
            # Build query index
            for item in flattened_items:
                # Index by name
                if item.get("name"):
                    name_lower = item["name"].lower()
                    for word in name_lower.split():
                        if len(word) > 2:  # Only index words longer than 2 characters
                            if word not in query_index:
                                query_index[word] = []
                            if item["id"] not in query_index[word]:  # Avoid duplicates
                                query_index[word].append(item["id"])
                
                # Index by description
                if item.get("description"):
                    desc_lower = item["description"].lower()
                    for word in re.findall(r'\b\w+\b', desc_lower):
                        if len(word) > 2:  # Only index words longer than 2 characters
                            if word not in query_index:
                                query_index[word] = []
                            if item["id"] not in query_index[word]:  # Avoid duplicates
                                query_index[word].append(item["id"])
                
                # Index by type
                if item.get("type"):
                    type_lower = item["type"].lower()
                    if type_lower not in query_index:
                        query_index[type_lower] = []
                    if item["id"] not in query_index[type_lower]:  # Avoid duplicates
                        query_index[type_lower].append(item["id"])
            
            return {
                "total_items": len(flattened_items),
                "items": flattened_items,
                "query_index": query_index,
                "framework_type": "json"
            }
            
        except Exception as e:
            logger.error(f"❌ JSON flattening failed: {e}")
            return {"error": f"JSON flattening failed: {e}", "total_items": 0}
    
    def _flatten_xml_framework(self, xml_data: Dict[str, Any], framework_id: str) -> Dict[str, Any]:
        """Flatten XML framework data."""
        flattened_items = []
        query_index = {}
        
        try:
            # Handle XML structure
            if "capec" in framework_id.lower():
                # CAPEC specific flattening
                patterns = xml_data.get("capec", {}).get("attack_patterns", {}).get("attack_pattern", [])
                for pattern in patterns:
                    flattened_items.append({
                        "id": pattern.get("@attributes", {}).get("id", ""),
                        "type": "attack_pattern",
                        "name": pattern.get("name", ""),
                        "description": pattern.get("description", ""),
                        "properties": pattern
                    })
            else:
                # Generic XML flattening
                for key, value in xml_data.items():
                    if isinstance(value, dict):
                        flattened_items.append({
                            "id": key,
                            "type": "xml_item",
                            "name": key,
                            "description": str(value)[:200],
                            "properties": value
                        })
            
            # Build query index
            for item in flattened_items:
                # Index by name
                if item.get("name"):
                    name_lower = item["name"].lower()
                    for word in name_lower.split():
                        if len(word) > 2:  # Only index words longer than 2 characters
                            if word not in query_index:
                                query_index[word] = []
                            if item["id"] not in query_index[word]:  # Avoid duplicates
                                query_index[word].append(item["id"])
                
                # Index by description
                if item.get("description"):
                    desc_lower = item["description"].lower()
                    for word in re.findall(r'\b\w+\b', desc_lower):
                        if len(word) > 2:  # Only index words longer than 2 characters
                            if word not in query_index:
                                query_index[word] = []
                            if item["id"] not in query_index[word]:  # Avoid duplicates
                                query_index[word].append(item["id"])
                
                # Index by type
                if item.get("type"):
                    type_lower = item["type"].lower()
                    if type_lower not in query_index:
                        query_index[type_lower] = []
                    if item["id"] not in query_index[type_lower]:  # Avoid duplicates
                        query_index[type_lower].append(item["id"])
            
            return {
                "total_items": len(flattened_items),
                "items": flattened_items,
                "query_index": query_index,
                "framework_type": "xml"
            }
            
        except Exception as e:
            logger.error(f"❌ XML flattening failed: {e}")
            return {"error": f"XML flattening failed: {e}", "total_items": 0}
    
    def _flatten_csv_framework(self, csv_data: Dict[str, Any], framework_id: str) -> Dict[str, Any]:
        """Flatten CSV framework data."""
        try:
            items = csv_data.get("data", [])
            query_index = {}
            
            # Build query index
            for item in items:
                for key, value in item.items():
                    if value:
                        value_lower = str(value).lower()
                        for word in re.findall(r'\b\w+\b', value_lower):
                            if word not in query_index:
                                query_index[word] = []
                            query_index[word].append(item.get("id", str(item)))
            
            return {
                "total_items": len(items),
                "items": items,
                "query_index": query_index,
                "framework_type": "csv"
            }
            
        except Exception as e:
            logger.error(f"❌ CSV flattening failed: {e}")
            return {"error": f"CSV flattening failed: {e}", "total_items": 0}
    
    def _flatten_generic_framework(self, data: Any, framework_id: str) -> Dict[str, Any]:
        """Flatten generic framework data."""
        try:
            if isinstance(data, str):
                return {
                    "total_items": 1,
                    "items": [{"id": "content", "type": "text", "content": data}],
                    "query_index": {},
                    "framework_type": "text"
                }
            elif isinstance(data, (list, tuple)):
                return {
                    "total_items": len(data),
                    "items": [{"id": str(i), "type": "item", "content": item} for i, item in enumerate(data)],
                    "query_index": {},
                    "framework_type": "list"
                }
            else:
                return {
                    "total_items": 1,
                    "items": [{"id": "data", "type": "object", "content": data}],
                    "query_index": {},
                    "framework_type": "generic"
                }
                
        except Exception as e:
            logger.error(f"❌ Generic flattening failed: {e}")
            return {"error": f"Generic flattening failed: {e}", "total_items": 0}
    
    async def query_flattened_framework(self, framework_id: str, query: str, max_results: int = 50) -> Dict[str, Any]:
        """Query a flattened framework."""
        try:
            # This would query the actual flattened framework data
            # For now, return a placeholder
            return {
                "success": True,
                "framework": framework_id,
                "query": query,
                "results": [],
                "total_found": 0,
                "query_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Framework query failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_frameworks(self) -> Dict[str, Any]:
        """Get list of available frameworks."""
        return {
            "success": True,
            "frameworks": {
                framework_id: {
                    "name": framework.name,
                    "version": framework.version,
                    "description": framework.description,
                    "tags": framework.tags,
                    "format_type": framework.format_type,
                    "source_url": framework.source_url
                }
                for framework_id, framework in self.framework_registry.items()
            }
        }
    
    def get_processing_history(self) -> Dict[str, Any]:
        """Get processing history."""
        return {
            "success": True,
            "history": self.processing_history,
            "total_actions": len(self.processing_history),
            "successful_actions": len([h for h in self.processing_history if h.get("success", False)])
        }
    
    def clear_cache(self):
        """Clear download cache."""
        self.download_cache.clear()
        logger.info("🧹 Download cache cleared")

# Example usage
async def main():
    """Test Browser MCP Tools."""
    print("🚀 Testing Browser MCP Tools")
    print("=" * 50)
    
    # Create tools
    browser_tools = BrowserMCPTools()
    
    # Test online search
    print("🔍 Testing online search...")
    search_result = await browser_tools.search_online("threat intelligence")
    print(f"Search results: {len(search_result.get('results', []))} found")
    
    # Test framework download
    print("\n📥 Testing framework download...")
    download_result = await browser_tools.download_framework("mitre_attack")
    if download_result["success"]:
        print(f"Downloaded: {download_result['framework']} ({download_result['file_size']} bytes)")
    else:
        print(f"Download failed: {download_result['error']}")
    
    # Test framework flattening
    print("\n🔄 Testing framework flattening...")
    flatten_result = await browser_tools.flatten_framework("mitre_attack")
    if flatten_result["success"]:
        print(f"Flattened: {flatten_result['flattened_framework']['name']}")
        print(f"Total items: {flatten_result['flattened_framework']['total_items']}")
    else:
        print(f"Flattening failed: {flatten_result['error']}")
    
    # Get available frameworks
    print("\n📚 Available frameworks...")
    frameworks = browser_tools.get_available_frameworks()
    for fw_id, fw_info in frameworks["frameworks"].items():
        print(f"  • {fw_info['name']} ({fw_info['format_type']})")
    
    # Get processing history
    print("\n📊 Processing history...")
    history = browser_tools.get_processing_history()
    print(f"Total actions: {history['total_actions']}")
    print(f"Successful: {history['successful_actions']}")
    
    print(f"\n🎉 Browser MCP Tools test completed!")

if __name__ == "__main__":
    asyncio.run(main())
