"""
Enhanced Field Normalization for Resource Graph Context Memory

Provides flexible and robust field normalization schemes for better entity relationship
mapping and data connectivity across different resource types and sources.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from enum import Enum
import pandas as pd
from pathlib import Path
import hashlib
import uuid

from .enhanced_session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)

class NormalizationType(Enum):
    """Types of field normalization."""
    STANDARDIZE = "standardize"      # Standardize field names and values
    RELATIONSHIP = "relationship"     # Extract and normalize relationships
    ENTITY = "entity"                # Normalize entity identifiers
    TEMPORAL = "temporal"            # Normalize temporal fields
    GEOGRAPHIC = "geographic"        # Normalize geographic fields
    SECURITY = "security"            # Normalize security-related fields
    CUSTOM = "custom"                # Custom normalization rules

class EntityType(Enum):
    """Types of entities that can be normalized."""
    HOST = "host"
    NETWORK = "network"
    USER = "user"
    APPLICATION = "application"
    SERVICE = "service"
    RESOURCE = "resource"
    LOCATION = "location"
    ORGANIZATION = "organization"
    THREAT = "threat"
    VULNERABILITY = "vulnerability"
    INCIDENT = "incident"
    POLICY = "policy"

class FieldNormalizer:
    """Comprehensive field normalization system for resource graph data."""
    
    def __init__(self, session_manager: EnhancedSessionManager):
        self.session_manager = session_manager
        self.normalization_rules = {}
        self.entity_mappings = {}
        self.relationship_patterns = {}
        self.custom_rules = {}
        
        # Initialize default normalization rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default normalization rules."""
        
        # Standard field name mappings
        self.normalization_rules[NormalizationType.STANDARDIZE] = {
            # Azure Resource Graph fields
            "azure": {
                "id": "resource_id",
                "name": "resource_name",
                "type": "resource_type",
                "location": "region",
                "resourceGroup": "resource_group",
                "tags": "metadata",
                "properties": "attributes",
                "sku": "specifications",
                "identity": "authentication",
                "managedBy": "managed_by",
                "plan": "pricing_plan"
            },
            # Google Cloud fields
            "gcp": {
                "name": "resource_name",
                "project": "project_id",
                "location": "region",
                "labels": "metadata",
                "description": "description",
                "createTime": "created_at",
                "updateTime": "updated_at",
                "state": "status"
            },
            # Generic fields
            "generic": {
                "ip": "ip_address",
                "mac": "mac_address",
                "hostname": "host_name",
                "fqdn": "fully_qualified_domain_name",
                "url": "uniform_resource_locator",
                "uri": "uniform_resource_identifier",
                "email": "email_address",
                "phone": "phone_number",
                "timestamp": "created_at",
                "last_seen": "last_observed",
                "first_seen": "first_observed"
            }
        }
        
        # Entity relationship patterns
        self.relationship_patterns = {
            "network_contains": {
                "pattern": r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/(\d{1,2})",
                "entity_type": EntityType.NETWORK,
                "relationship": "contains",
                "extract": ["network_address", "subnet_mask"]
            },
            "host_belongs_to": {
                "pattern": r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                "entity_type": EntityType.HOST,
                "relationship": "belongs_to",
                "extract": ["ip_address"]
            },
            "service_runs_on": {
                "pattern": r"(\w+):(\d+)",
                "entity_type": EntityType.SERVICE,
                "relationship": "runs_on",
                "extract": ["protocol", "port"]
            },
            "user_authenticates_to": {
                "pattern": r"(\w+@[\w\.-]+\.\w+)",
                "entity_type": EntityType.USER,
                "relationship": "authenticates_to",
                "extract": ["email_address"]
            }
        }
        
        # Entity type mappings
        self.entity_mappings = {
            "azure": {
                "microsoft.compute/virtualmachines": EntityType.HOST,
                "microsoft.network/virtualnetworks": EntityType.NETWORK,
                "microsoft.network/networkinterfaces": EntityType.NETWORK,
                "microsoft.storage/storageaccounts": EntityType.RESOURCE,
                "microsoft.keyvault/vaults": EntityType.RESOURCE,
                "microsoft.web/sites": EntityType.APPLICATION,
                "microsoft.web/serverfarms": EntityType.RESOURCE,
                "microsoft.sql/servers": EntityType.SERVICE,
                "microsoft.insights/components": EntityType.SERVICE,
                "microsoft.operationalinsights/workspaces": EntityType.SERVICE
            },
            "gcp": {
                "compute.googleapis.com/Instance": EntityType.HOST,
                "compute.googleapis.com/Network": EntityType.NETWORK,
                "compute.googleapis.com/Subnetwork": EntityType.NETWORK,
                "storage.googleapis.com/Bucket": EntityType.RESOURCE,
                "bigquery.googleapis.com/Dataset": EntityType.RESOURCE,
                "cloudsql.googleapis.com/Instance": EntityType.SERVICE,
                "run.googleapis.com/Service": EntityType.APPLICATION,
                "container.googleapis.com/Cluster": EntityType.RESOURCE
            }
        }
    
    def normalize_resource_data(self, data: Union[Dict, List[Dict], pd.DataFrame], 
                              source_type: str = "generic",
                              normalization_types: Optional[List[NormalizationType]] = None) -> Dict[str, Any]:
        """
        Normalize resource data using comprehensive normalization schemes.
        
        Args:
            data: Resource data to normalize
            source_type: Type of source (azure, gcp, generic)
            normalization_types: Types of normalization to apply
            
        Returns:
            Normalized data with entity relationships and metadata
        """
        try:
            if normalization_types is None:
                normalization_types = [NormalizationType.STANDARDIZE, NormalizationType.RELATIONSHIP, 
                                    NormalizationType.ENTITY, NormalizationType.TEMPORAL]
            
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Apply normalization types
            normalized_data = {
                "original_shape": df.shape,
                "normalization_applied": [],
                "entities_extracted": [],
                "relationships_found": [],
                "normalized_data": df.copy()
            }
            
            for norm_type in normalization_types:
                if norm_type == NormalizationType.STANDARDIZE:
                    df = self._standardize_fields(df, source_type)
                    normalized_data["normalization_applied"].append("field_standardization")
                
                elif norm_type == NormalizationType.RELATIONSHIP:
                    relationships = self._extract_relationships(df, source_type)
                    normalized_data["relationships_found"].extend(relationships)
                    normalized_data["normalization_applied"].append("relationship_extraction")
                
                elif norm_type == NormalizationType.ENTITY:
                    entities = self._extract_entities(df, source_type)
                    normalized_data["entities_extracted"].extend(entities)
                    normalized_data["normalization_applied"].append("entity_extraction")
                
                elif norm_type == NormalizationType.TEMPORAL:
                    df = self._normalize_temporal_fields(df)
                    normalized_data["normalization_applied"].append("temporal_normalization")
                
                elif norm_type == NormalizationType.GEOGRAPHIC:
                    df = self._normalize_geographic_fields(df)
                    normalized_data["normalization_applied"].append("geographic_normalization")
                
                elif norm_type == NormalizationType.SECURITY:
                    df = self._normalize_security_fields(df)
                    normalized_data["normalization_applied"].append("security_normalization")
                
                elif norm_type == NormalizationType.CUSTOM:
                    df = self._apply_custom_rules(df)
                    normalized_data["normalization_applied"].append("custom_normalization")
            
            # Update normalized data
            normalized_data["normalized_data"] = df
            normalized_data["final_shape"] = df.shape
            normalized_data["normalization_timestamp"] = datetime.now().isoformat()
            
            # Generate entity relationship graph
            relationship_graph = self._generate_relationship_graph(
                normalized_data["entities_extracted"],
                normalized_data["relationships_found"]
            )
            normalized_data["relationship_graph"] = relationship_graph
            
            # Save normalized data to session
            self._save_normalized_data(normalized_data, source_type)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to normalize resource data: {e}")
            raise
    
    def _standardize_fields(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """Standardize field names and values."""
        try:
            df_std = df.copy()
            
            # Get standardization rules for source type
            rules = self.normalization_rules.get(NormalizationType.STANDARDIZE, {}).get(source_type, {})
            generic_rules = self.normalization_rules.get(NormalizationType.STANDARDIZE, {}).get("generic", {})
            
            # Apply source-specific rules
            for old_name, new_name in rules.items():
                if old_name in df_std.columns:
                    df_std[new_name] = df_std[old_name]
                    df_std = df_std.drop(columns=[old_name])
            
            # Apply generic rules
            for old_name, new_name in generic_rules.items():
                if old_name in df_std.columns:
                    df_std[new_name] = df_std[old_name]
                    df_std = df_std.drop(columns=[old_name])
            
            # Standardize column names
            df_std.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_std.columns]
            
            return df_std
            
        except Exception as e:
            logger.error(f"Failed to standardize fields: {e}")
            return df
    
    def _extract_relationships(self, df: pd.DataFrame, source_type: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        try:
            relationships = []
            
            for pattern_name, pattern_info in self.relationship_patterns.items():
                pattern = pattern_info["pattern"]
                entity_type = pattern_info["entity_type"]
                relationship_type = pattern_info["relationship"]
                extract_fields = pattern_info["extract"]
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        for idx, value in df[col].items():
                            if pd.notna(value):
                                matches = re.findall(pattern, str(value))
                                if matches:
                                    for match in matches:
                                        if isinstance(match, tuple):
                                            extracted_values = list(match)
                                        else:
                                            extracted_values = [match]
                                        
                                        relationship = {
                                            "id": str(uuid.uuid4()),
                                            "source_entity": {
                                                "id": str(idx),
                                                "type": "resource",
                                                "source": source_type
                                            },
                                            "target_entity": {
                                                "type": entity_type.value,
                                                "extracted_values": dict(zip(extract_fields, extracted_values))
                                            },
                                            "relationship_type": relationship_type,
                                            "pattern_used": pattern_name,
                                            "confidence": 0.8,
                                            "extracted_at": datetime.now().isoformat()
                                        }
                                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to extract relationships: {e}")
            return []
    
    def _extract_entities(self, df: pd.DataFrame, source_type: str) -> List[Dict[str, Any]]:
        """Extract entities from resource data."""
        try:
            entities = []
            
            # Get entity mappings for source type
            mappings = self.entity_mappings.get(source_type, {})
            
            for idx, row in df.iterrows():
                # Determine entity type from resource type
                resource_type = row.get('resource_type', row.get('type', 'unknown'))
                entity_type = mappings.get(resource_type, EntityType.RESOURCE)
                
                # Extract entity information
                entity = {
                    "id": str(uuid.uuid4()),
                    "source_id": str(idx),
                    "entity_type": entity_type.value,
                    "source_type": source_type,
                    "attributes": {},
                    "metadata": {},
                    "extracted_at": datetime.now().isoformat()
                }
                
                # Extract relevant attributes based on entity type
                if entity_type == EntityType.HOST:
                    entity["attributes"] = self._extract_host_attributes(row)
                elif entity_type == EntityType.NETWORK:
                    entity["attributes"] = self._extract_network_attributes(row)
                elif entity_type == EntityType.USER:
                    entity["attributes"] = self._extract_user_attributes(row)
                elif entity_type == EntityType.APPLICATION:
                    entity["attributes"] = self._extract_application_attributes(row)
                elif entity_type == EntityType.SERVICE:
                    entity["attributes"] = self._extract_service_attributes(row)
                else:
                    entity["attributes"] = self._extract_generic_attributes(row)
                
                # Extract metadata
                entity["metadata"] = self._extract_metadata(row)
                
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    def _extract_host_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract host-specific attributes."""
        attributes = {}
        
        # IP addresses
        ip_fields = ['ip_address', 'private_ip', 'public_ip', 'internal_ip', 'external_ip']
        for field in ip_fields:
            if field in row and pd.notna(row[field]):
                attributes['ip_addresses'] = attributes.get('ip_addresses', [])
                attributes['ip_addresses'].append(row[field])
        
        # Hostname
        hostname_fields = ['host_name', 'computer_name', 'vm_name', 'instance_name']
        for field in hostname_fields:
            if field in row and pd.notna(row[field]):
                attributes['hostname'] = row[field]
                break
        
        # Operating system
        os_fields = ['os_type', 'os_name', 'operating_system']
        for field in os_fields:
            if field in row and pd.notna(row[field]):
                attributes['operating_system'] = row[field]
                break
        
        # Resource specifications
        spec_fields = ['vm_size', 'machine_type', 'cpu_count', 'memory_gb']
        for field in spec_fields:
            if field in row and pd.notna(row[field]):
                attributes['specifications'] = attributes.get('specifications', {})
                attributes['specifications'][field] = row[field]
        
        return attributes
    
    def _extract_network_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract network-specific attributes."""
        attributes = {}
        
        # Network address
        network_fields = ['address_space', 'cidr', 'subnet_range']
        for field in network_fields:
            if field in row and pd.notna(row[field]):
                attributes['network_range'] = row[field]
                break
        
        # Network type
        type_fields = ['network_type', 'vnet_type', 'subnet_type']
        for field in type_fields:
            if field in row and pd.notna(row[field]):
                attributes['network_type'] = row[field]
                break
        
        # Security groups
        security_fields = ['nsg', 'security_groups', 'firewall_rules']
        for field in security_fields:
            if field in row and pd.notna(row[field]):
                attributes['security_groups'] = row[field]
                break
        
        return attributes
    
    def _extract_user_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract user-specific attributes."""
        attributes = {}
        
        # User identifiers
        id_fields = ['user_id', 'principal_id', 'object_id']
        for field in id_fields:
            if field in row and pd.notna(row[field]):
                attributes['user_id'] = row[field]
                break
        
        # Email address
        email_fields = ['email', 'email_address', 'upn']
        for field in email_fields:
            if field in row and pd.notna(row[field]):
                attributes['email'] = row[field]
                break
        
        # Display name
        name_fields = ['display_name', 'name', 'full_name']
        for field in name_fields:
            if field in row and pd.notna(row[field]):
                attributes['display_name'] = row[field]
                break
        
        return attributes
    
    def _extract_application_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract application-specific attributes."""
        attributes = {}
        
        # Application name
        name_fields = ['app_name', 'application_name', 'site_name']
        for field in name_fields:
            if field in row and pd.notna(row[field]):
                attributes['application_name'] = row[field]
                break
        
        # Application type
        type_fields = ['app_type', 'application_type', 'platform']
        for field in type_fields:
            if field in row and pd.notna(row[field]):
                attributes['application_type'] = row[field]
                break
        
        # Runtime environment
        runtime_fields = ['runtime', 'framework', 'stack']
        for field in runtime_fields:
            if field in row and pd.notna(row[field]):
                attributes['runtime'] = row[field]
                break
        
        return attributes
    
    def _extract_service_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract service-specific attributes."""
        attributes = {}
        
        # Service name
        name_fields = ['service_name', 'service_type', 'endpoint']
        for field in name_fields:
            if field in row and pd.notna(row[field]):
                attributes['service_name'] = row[field]
                break
        
        # Service configuration
        config_fields = ['configuration', 'settings', 'parameters']
        for field in config_fields:
            if field in row and pd.notna(row[field]):
                attributes['configuration'] = row[field]
                break
        
        # Service status
        status_fields = ['status', 'state', 'health']
        for field in status_fields:
            if field in row and pd.notna(row[field]):
                attributes['status'] = row[field]
                break
        
        return attributes
    
    def _extract_generic_attributes(self, row: pd.Series) -> Dict[str, Any]:
        """Extract generic attributes for any resource type."""
        attributes = {}
        
        # Resource identifier
        id_fields = ['id', 'resource_id', 'name', 'identifier']
        for field in id_fields:
            if field in row and pd.notna(row[field]):
                attributes['resource_id'] = row[field]
                break
        
        # Resource type
        type_fields = ['type', 'resource_type', 'kind']
        for field in type_fields:
            if field in row and pd.notna(row[field]):
                attributes['resource_type'] = row[field]
                break
        
        # Location
        location_fields = ['location', 'region', 'zone', 'datacenter']
        for field in location_fields:
            if field in row and pd.notna(row[field]):
                attributes['location'] = row[field]
                break
        
        return attributes
    
    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from resource row."""
        metadata = {}
        
        # Tags and labels
        tag_fields = ['tags', 'labels', 'metadata']
        for field in tag_fields:
            if field in row and pd.notna(row[field]):
                if isinstance(row[field], dict):
                    metadata['tags'] = row[field]
                break
        
        # Creation and modification times
        time_fields = ['created_at', 'created_time', 'creation_timestamp', 'last_modified', 'updated_at']
        for field in time_fields:
            if field in row and pd.notna(row[field]):
                metadata['timestamps'] = metadata.get('timestamps', {})
                if 'created' in field.lower():
                    metadata['timestamps']['created'] = row[field]
                elif 'modified' in field.lower() or 'updated' in field.lower():
                    metadata['timestamps']['modified'] = row[field]
        
        # Resource group and subscription
        group_fields = ['resource_group', 'project', 'subscription']
        for field in group_fields:
            if field in row and pd.notna(row[field]):
                metadata['organization'] = metadata.get('organization', {})
                if 'resource_group' in field.lower():
                    metadata['organization']['resource_group'] = row[field]
                elif 'project' in field.lower():
                    metadata['organization']['project'] = row[field]
                elif 'subscription' in field.lower():
                    metadata['organization']['subscription'] = row[field]
        
        return metadata
    
    def _normalize_temporal_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize temporal fields to standard format."""
        try:
            df_temp = df.copy()
            
            # Common temporal field patterns
            temporal_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(\d{4}/\d{2}/\d{2})',  # YYYY/MM/DD
                r'(\d{2}-\d{2}-\d{4})',  # MM-DD-YYYY
                r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
                r'(\d{10,13})',          # Unix timestamp
            ]
            
            for col in df_temp.columns:
                if df_temp[col].dtype == 'object':
                    # Check if column contains temporal data
                    temporal_count = 0
                    for value in df_temp[col].dropna():
                        if any(re.search(pattern, str(value)) for pattern in temporal_patterns):
                            temporal_count += 1
                    
                    # If more than 50% of values are temporal, normalize the column
                    if temporal_count > len(df_temp[col].dropna()) * 0.5:
                        df_temp[f"{col}_normalized"] = df_temp[col].apply(self._normalize_timestamp)
            
            return df_temp
            
        except Exception as e:
            logger.error(f"Failed to normalize temporal fields: {e}")
            return df
    
    def _normalize_timestamp(self, value) -> Optional[str]:
        """Normalize timestamp to ISO format."""
        try:
            if pd.isna(value):
                return None
            
            value_str = str(value)
            
            # Try to parse different timestamp formats
            if re.match(r'\d{4}-\d{2}-\d{2}', value_str):
                # Already in YYYY-MM-DD format
                return value_str
            
            elif re.match(r'\d{10,13}', value_str):
                # Unix timestamp
                if len(value_str) == 10:
                    timestamp = int(value_str)
                else:
                    timestamp = int(value_str) / 1000
                return datetime.fromtimestamp(timestamp).isoformat()
            
            else:
                # Try other formats
                for fmt in ['%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        dt = datetime.strptime(value_str, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
            
            return value_str
            
        except Exception as e:
            logger.warning(f"Failed to normalize timestamp {value}: {e}")
            return str(value)
    
    def _normalize_geographic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize geographic fields."""
        try:
            df_geo = df.copy()
            
            # Common geographic field patterns
            geo_fields = ['location', 'region', 'zone', 'country', 'city', 'datacenter']
            
            for col in df_geo.columns:
                if col.lower() in geo_fields or any(geo in col.lower() for geo in geo_fields):
                    # Standardize geographic values
                    df_geo[f"{col}_normalized"] = df_geo[col].apply(self._normalize_geographic_value)
            
            return df_geo
            
        except Exception as e:
            logger.error(f"Failed to normalize geographic fields: {e}")
            return df
    
    def _normalize_geographic_value(self, value) -> str:
        """Normalize geographic value."""
        try:
            if pd.isna(value):
                return "unknown"
            
            value_str = str(value).strip()
            
            # Common geographic normalizations
            geo_mappings = {
                'us': 'United States',
                'usa': 'United States',
                'uk': 'United Kingdom',
                'eu': 'European Union',
                'na': 'North America',
                'sa': 'South America',
                'asia': 'Asia',
                'emea': 'Europe, Middle East, and Africa',
                'apac': 'Asia Pacific'
            }
            
            return geo_mappings.get(value_str.lower(), value_str)
            
        except Exception as e:
            logger.warning(f"Failed to normalize geographic value {value}: {e}")
            return str(value)
    
    def _normalize_security_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize security-related fields."""
        try:
            df_sec = df.copy()
            
            # Security field patterns
            security_fields = ['security', 'auth', 'encryption', 'ssl', 'tls', 'firewall', 'nsg']
            
            for col in df_sec.columns:
                if any(sec in col.lower() for sec in security_fields):
                    # Normalize security values
                    df_sec[f"{col}_normalized"] = df_sec[col].apply(self._normalize_security_value)
            
            return df_sec
            
        except Exception as e:
            logger.error(f"Failed to normalize security fields: {e}")
            return df
    
    def _normalize_security_value(self, value) -> str:
        """Normalize security value."""
        try:
            if pd.isna(value):
                return "unknown"
            
            value_str = str(value).strip().lower()
            
            # Security value normalizations
            security_mappings = {
                'enabled': 'enabled',
                'true': 'enabled',
                '1': 'enabled',
                'yes': 'enabled',
                'disabled': 'disabled',
                'false': 'disabled',
                '0': 'disabled',
                'no': 'disabled',
                'ssl': 'ssl_enabled',
                'tls': 'tls_enabled',
                'encrypted': 'encrypted',
                'unencrypted': 'unencrypted'
            }
            
            return security_mappings.get(value_str, value_str)
            
        except Exception as e:
            logger.warning(f"Failed to normalize security value {value}: {e}")
            return str(value)
    
    def _apply_custom_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom normalization rules."""
        try:
            df_custom = df.copy()
            
            # Apply any custom rules that have been added
            for rule_name, rule_func in self.custom_rules.items():
                try:
                    df_custom = rule_func(df_custom)
                    logger.info(f"Applied custom rule: {rule_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply custom rule {rule_name}: {e}")
            
            return df_custom
            
        except Exception as e:
            logger.error(f"Failed to apply custom rules: {e}")
            return df
    
    def _generate_relationship_graph(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Generate a relationship graph from extracted entities and relationships."""
        try:
            graph = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "total_entities": len(entities),
                    "total_relationships": len(relationships),
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            # Add entity nodes
            for entity in entities:
                node = {
                    "id": entity["id"],
                    "label": entity.get("attributes", {}).get("resource_id", entity["source_id"]),
                    "type": entity["entity_type"],
                    "source": entity["source_type"],
                    "attributes": entity["attributes"],
                    "metadata": entity["metadata"]
                }
                graph["nodes"].append(node)
            
            # Add relationship edges
            for relationship in relationships:
                edge = {
                    "id": relationship["id"],
                    "source": relationship["source_entity"]["id"],
                    "target": relationship["target_entity"]["type"],
                    "type": relationship["relationship_type"],
                    "attributes": relationship["target_entity"]["extracted_values"],
                    "confidence": relationship["confidence"],
                    "pattern": relationship["pattern_used"]
                }
                graph["edges"].append(edge)
            
            return graph
            
        except Exception as e:
            logger.error(f"Failed to generate relationship graph: {e}")
            return {"nodes": [], "edges": [], "metadata": {"error": str(e)}}
    
    def _save_normalized_data(self, normalized_data: Dict[str, Any], source_type: str):
        """Save normalized data to session."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save normalized DataFrame
            if not normalized_data["normalized_data"].empty:
                filename = f"normalized_{source_type}_{timestamp}"
                self.session_manager.save_dataframe(
                    normalized_data["normalized_data"],
                    filename,
                    f"Normalized {source_type} resource data with {len(normalized_data['normalization_applied'])} normalization types applied"
                )
            
            # Save relationship graph
            if normalized_data.get("relationship_graph"):
                graph_filename = f"relationship_graph_{source_type}_{timestamp}"
                self.session_manager.save_text_output(
                    json.dumps(normalized_data["relationship_graph"], indent=2),
                    graph_filename,
                    f"Entity relationship graph for {source_type} resources"
                )
            
            # Save normalization summary
            summary_filename = f"normalization_summary_{source_type}_{timestamp}"
            summary_data = {
                "source_type": source_type,
                "normalization_applied": normalized_data["normalization_applied"],
                "entities_extracted": len(normalized_data["entities_extracted"]),
                "relationships_found": len(normalized_data["relationships_found"]),
                "original_shape": normalized_data["original_shape"],
                "final_shape": normalized_data["final_shape"],
                "timestamp": normalized_data["normalization_timestamp"]
            }
            self.session_manager.save_text_output(
                json.dumps(summary_data, indent=2),
                summary_filename,
                f"Normalization summary for {source_type} resources"
            )
            
        except Exception as e:
            logger.error(f"Failed to save normalized data: {e}")
    
    def add_custom_rule(self, rule_name: str, rule_function):
        """Add a custom normalization rule."""
        try:
            self.custom_rules[rule_name] = rule_function
            logger.info(f"Added custom normalization rule: {rule_name}")
        except Exception as e:
            logger.error(f"Failed to add custom rule {rule_name}: {e}")
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get statistics about normalization operations."""
        return {
            "total_rules": len(self.normalization_rules),
            "total_patterns": len(self.relationship_patterns),
            "total_entity_mappings": len(self.entity_mappings),
            "custom_rules": list(self.custom_rules.keys()),
            "normalization_types": [t.value for t in NormalizationType],
            "entity_types": [t.value for t in EntityType]
        }

# MCP Tools for Field Normalization
class FieldNormalizationMCPTools:
    """MCP-compatible tools for field normalization."""
    
    def __init__(self, field_normalizer: FieldNormalizer):
        self.normalizer = field_normalizer
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for field normalization."""
        return {
            "normalize_resource_data": {
                "name": "normalize_resource_data",
                "description": "Normalize resource data using comprehensive field normalization schemes",
                "parameters": {
                    "data": {"type": "object", "description": "Resource data to normalize"},
                    "source_type": {"type": "string", "description": "Type of source (azure, gcp, generic)"},
                    "normalization_types": {"type": "array", "items": {"type": "string"}, "description": "Types of normalization to apply"}
                },
                "returns": {"type": "object", "description": "Normalized data with entity relationships and metadata"}
            },
            "extract_entities": {
                "name": "extract_entities",
                "description": "Extract entities from resource data",
                "parameters": {
                    "data": {"type": "object", "description": "Resource data to extract entities from"},
                    "source_type": {"type": "string", "description": "Type of source (azure, gcp, generic)"}
                },
                "returns": {"type": "object", "description": "Extracted entities with attributes and metadata"}
            },
            "extract_relationships": {
                "name": "extract_relationships",
                "description": "Extract relationships between entities",
                "parameters": {
                    "data": {"type": "object", "description": "Resource data to extract relationships from"},
                    "source_type": {"type": "string", "description": "Type of source (azure, gcp, generic)"}
                },
                "returns": {"type": "object", "description": "Extracted relationships with confidence scores"}
            },
            "get_normalization_stats": {
                "name": "get_normalization_stats",
                "description": "Get statistics about normalization operations",
                "parameters": {},
                "returns": {"type": "object", "description": "Normalization statistics and available options"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute field normalization MCP tool."""
        if tool_name == "normalize_resource_data":
            return self.normalizer.normalize_resource_data(**kwargs)
        elif tool_name == "extract_entities":
            # Extract entities from data
            if isinstance(kwargs["data"], pd.DataFrame):
                df = kwargs["data"]
            else:
                df = pd.DataFrame([kwargs["data"]] if isinstance(kwargs["data"], dict) else kwargs["data"])
            return self.normalizer._extract_entities(df, kwargs["source_type"])
        elif tool_name == "extract_relationships":
            # Extract relationships from data
            if isinstance(kwargs["data"], pd.DataFrame):
                df = kwargs["data"]
            else:
                df = pd.DataFrame([kwargs["data"]] if isinstance(kwargs["data"], dict) else kwargs["data"])
            return self.normalizer._extract_relationships(df, kwargs["source_type"])
        elif tool_name == "get_normalization_stats":
            return self.normalizer.get_normalization_stats()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

