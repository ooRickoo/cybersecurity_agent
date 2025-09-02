"""
Google Resource Manager Integration for Cybersecurity Agent

Provides comprehensive access to Google Cloud Resource Manager for querying GCP resources,
managing credentials, and handling local data processing.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from google.cloud import resourcemanager_v3
from google.cloud import asset_v1
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

from .credential_vault import CredentialVault
from .context_memory_manager import ContextMemoryManager
from .enhanced_session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)

class GoogleResourceManagerIntegration:
    """Comprehensive Google Cloud Resource Manager integration with credential management and local data handling."""
    
    def __init__(self, session_manager: EnhancedSessionManager, credential_vault: CredentialVault, memory_manager: ContextMemoryManager):
        self.session_manager = session_manager
        self.credential_vault = credential_vault
        self.memory_manager = memory_manager
        self.client = None
        self.asset_client = None
        self.compute_client = None
        self.storage_client = None
        self.bigquery_client = None
        self.projects = []
        self.connection_info = {}
        
        # Initialize connection from memory or prompt user
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Google Cloud connection from memory or prompt user."""
        try:
            # Try to get connection info from long-term memory
            connection_data = self.memory_manager.search("gcp_connection_info", domain="gcp", max_entries=1)
            
            if connection_data:
                self.connection_info = connection_data[0].data
                self._connect_with_stored_credentials()
            else:
                self._prompt_for_credentials()
                
        except Exception as e:
            logger.warning(f"Failed to initialize GCP connection from memory: {e}")
            self._prompt_for_credentials()
    
    def _prompt_for_credentials(self):
        """Prompt user for Google Cloud credentials and store them."""
        print("\n🔐 Google Cloud Resource Manager Authentication Required")
        print("Please provide your GCP connection details:")
        
        # Get authentication method
        auth_method = input("Authentication method (1: Service Account Key, 2: Default Credentials): ").strip()
        
        if auth_method == "1":
            # Service Account authentication
            print("\nPlease provide the path to your service account JSON key file:")
            key_file_path = input("Service Account Key File Path: ").strip()
            
            # Read and store the key file
            try:
                with open(key_file_path, 'r') as f:
                    key_data = json.load(f)
                
                # Store in credential vault
                self.credential_vault.store_credential(
                    "gcp_service_account",
                    key_data
                )
                
                self.connection_info = {
                    "auth_method": "service_account",
                    "project_id": key_data.get("project_id"),
                    "client_email": key_data.get("client_email"),
                    "key_file_path": key_file_path,
                    "last_updated": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error reading service account key: {e}")
                raise
                
        else:
            # Default credentials
            project_id = input("Project ID: ").strip()
            
            self.connection_info = {
                "auth_method": "default_credentials",
                "project_id": project_id,
                "last_updated": datetime.now().isoformat()
            }
        
        # Store connection info in long-term memory
        self.memory_manager.import_data(
            "gcp_connection_info",
            self.connection_info,
            domain="gcp",
            tier="long_term",
            ttl_days=365,
            metadata={
                "description": "Google Cloud Resource Manager connection configuration",
                "type": "connection_info"
            }
        )
        
        self._connect_with_stored_credentials()
    
    def _connect_with_stored_credentials(self):
        """Connect to Google Cloud using stored credentials."""
        try:
            if self.connection_info["auth_method"] == "service_account":
                # Get credentials from vault
                key_data = self.credential_vault.get_credential("gcp_service_account")
                
                credentials = service_account.Credentials.from_service_account_info(key_data)
                
                # Initialize clients
                self.client = resourcemanager_v3.ProjectsClient(credentials=credentials)
                self.asset_client = asset_v1.AssetServiceClient(credentials=credentials)
                self.compute_client = compute_v1.InstancesClient(credentials=credentials)
                self.storage_client = storage.Client(credentials=credentials, project=key_data["project_id"])
                self.bigquery_client = bigquery.Client(credentials=credentials, project=key_data["project_id"])
                
                self.projects = [key_data["project_id"]]
                
            else:
                # Default credentials
                credentials, project_id = default()
                
                if not project_id:
                    project_id = self.connection_info["project_id"]
                
                # Initialize clients
                self.client = resourcemanager_v3.ProjectsClient(credentials=credentials)
                self.asset_client = asset_v1.AssetServiceClient(credentials=credentials)
                self.compute_client = compute_v1.InstancesClient(credentials=credentials)
                self.storage_client = storage.Client(credentials=credentials, project=project_id)
                self.bigquery_client = bigquery.Client(credentials=credentials, project=project_id)
                
                self.projects = [project_id]
            
            logger.info("Successfully connected to Google Cloud Resource Manager")
            
        except Exception as e:
            logger.error(f"Failed to connect to Google Cloud: {e}")
            raise
    
    def list_projects(self, parent: Optional[str] = None) -> Dict[str, Any]:
        """
        List all accessible projects.
        
        Args:
            parent: Parent resource (organization or folder)
            
        Returns:
            Dictionary containing project list and metadata
        """
        try:
            if not self.client:
                raise Exception("Google Cloud client not initialized")
            
            # Create request
            request = resourcemanager_v3.ListProjectsRequest(
                parent=parent or ""
            )
            
            # Execute request
            start_time = datetime.now()
            page_result = self.client.list_projects(request=request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            projects = []
            for project in page_result:
                projects.append({
                    "name": project.name,
                    "project_id": project.project_id,
                    "state": project.state.name,
                    "create_time": project.create_time.isoformat(),
                    "labels": dict(project.labels) if project.labels else {},
                    "parent": project.parent
                })
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(projects)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud projects list"
                )
            
            return {
                "success": True,
                "total_count": len(projects),
                "execution_time": execution_time,
                "data": projects,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to list GCP projects: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def search_resources(self, query: str, asset_types: Optional[List[str]] = None,
                        projects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for resources using Asset Inventory.
        
        Args:
            query: Search query string
            asset_types: List of asset types to search for
            projects: List of project IDs to search in
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            if not self.asset_client:
                raise Exception("Asset client not initialized")
            
            if not projects:
                projects = self.projects
            
            # Create search request
            request = asset_v1.SearchAllResourcesRequest(
                scope=f"projects/{projects[0]}" if len(projects) == 1 else f"organizations/{projects[0].split('/')[1]}",
                query=query,
                asset_types=asset_types or []
            )
            
            # Execute search
            start_time = datetime.now()
            page_result = self.asset_client.search_all_resources(request=request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            resources = []
            for resource in page_result:
                resources.append({
                    "name": resource.name,
                    "asset_type": resource.asset_type,
                    "project": resource.project,
                    "location": resource.location,
                    "labels": dict(resource.labels) if resource.labels else {},
                    "description": resource.description,
                    "display_name": resource.display_name,
                    "additional_attributes": resource.additional_attributes
                })
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(resources)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    f"Google Cloud resource search results: {query[:100]}..."
                )
            
            return {
                "success": True,
                "query": query,
                "projects": projects,
                "total_count": len(resources),
                "execution_time": execution_time,
                "data": resources,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GCP resource search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_compute_instances(self, project_id: Optional[str] = None, 
                             zone: Optional[str] = None) -> Dict[str, Any]:
        """Get all compute instances with detailed information."""
        try:
            if not self.compute_client:
                raise Exception("Compute client not initialized")
            
            if not project_id:
                project_id = self.projects[0]
            
            # List instances
            start_time = datetime.now()
            instances = []
            
            if zone:
                # Single zone
                request = compute_v1.ListInstancesRequest(
                    project=project_id,
                    zone=zone
                )
                page_result = self.compute_client.list(request=request)
                
                for instance in page_result:
                    instances.append({
                        "name": instance.name,
                        "id": instance.id,
                        "zone": instance.zone.split('/')[-1],
                        "machine_type": instance.machine_type.split('/')[-1],
                        "status": instance.status,
                        "network_interfaces": [
                            {
                                "network": ni.network.split('/')[-1],
                                "subnetwork": ni.subnetwork.split('/')[-1] if ni.subnetwork else None,
                                "internal_ip": ni.network_ip,
                                "external_ip": ni.access_configs[0].nat_ip if ni.access_configs else None
                            }
                            for ni in instance.network_interfaces
                        ],
                        "disks": [
                            {
                                "name": disk.device_name,
                                "size_gb": disk.boot_disk.disk_size_gb if disk.boot_disk else None,
                                "type": disk.boot_disk.disk_type.split('/')[-1] if disk.boot_disk else None
                            }
                            for disk in instance.disks
                        ],
                        "labels": dict(instance.labels) if instance.labels else {},
                        "creation_timestamp": instance.creation_timestamp
                    })
            else:
                # All zones
                for zone_name in ["us-central1-a", "us-central1-b", "us-east1-b", "us-west1-a"]:  # Common zones
                    try:
                        request = compute_v1.ListInstancesRequest(
                            project=project_id,
                            zone=zone_name
                        )
                        page_result = self.compute_client.list(request=request)
                        
                        for instance in page_result:
                            instances.append({
                                "name": instance.name,
                                "id": instance.id,
                                "zone": zone_name,
                                "machine_type": instance.machine_type.split('/')[-1],
                                "status": instance.status,
                                "network_interfaces": [
                                    {
                                        "network": ni.network.split('/')[-1],
                                        "subnetwork": ni.subnetwork.split('/')[-1] if ni.subnetwork else None,
                                        "internal_ip": ni.network_ip,
                                        "external_ip": ni.access_configs[0].nat_ip if ni.access_configs else None
                                    }
                                    for ni in instance.network_interfaces
                                ],
                                "disks": [
                                    {
                                        "name": disk.device_name,
                                        "size_gb": disk.boot_disk.disk_size_gb if disk.boot_disk else None,
                                        "type": disk.boot_disk.disk_type.split('/')[-1] if disk.boot_disk else None
                                    }
                                    for disk in instance.disks
                                ],
                                "labels": dict(instance.labels) if instance.labels else {},
                                "creation_timestamp": instance.creation_timestamp
                            })
                    except Exception as e:
                        logger.warning(f"Failed to list instances in zone {zone_name}: {e}")
                        continue
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(instances)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_compute_instances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud compute instances"
                )
            
            return {
                "success": True,
                "project_id": project_id,
                "zone": zone,
                "total_count": len(instances),
                "execution_time": execution_time,
                "data": instances,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCP compute instances: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_storage_buckets(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all storage buckets with detailed information."""
        try:
            if not self.storage_client:
                raise Exception("Storage client not initialized")
            
            if not project_id:
                project_id = self.projects[0]
            
            # List buckets
            start_time = datetime.now()
            buckets = []
            
            for bucket in self.storage_client.list_buckets():
                # Get bucket details
                bucket.reload()
                
                buckets.append({
                    "name": bucket.name,
                    "project_id": bucket.project_number,
                    "location": bucket.location,
                    "storage_class": bucket.storage_class,
                    "versioning_enabled": bucket.versioning_enabled,
                    "labels": bucket.labels,
                    "created": bucket.time_created.isoformat(),
                    "updated": bucket.updated.isoformat(),
                    "public_access_prevention": bucket.public_access_prevention,
                    "uniform_bucket_level_access": bucket.uniform_bucket_level_access_enabled,
                    "retention_policy": {
                        "retention_period": bucket.retention_period,
                        "effective_time": bucket.retention_policy_effective_time.isoformat() if bucket.retention_policy_effective_time else None
                    } if bucket.retention_policy else None
                })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(buckets)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_storage_buckets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud storage buckets"
                )
            
            return {
                "success": True,
                "project_id": project_id,
                "total_count": len(buckets),
                "execution_time": execution_time,
                "data": buckets,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCP storage buckets: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_bigquery_datasets(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all BigQuery datasets with detailed information."""
        try:
            if not self.bigquery_client:
                raise Exception("BigQuery client not initialized")
            
            if not project_id:
                project_id = self.projects[0]
            
            # List datasets
            start_time = datetime.now()
            datasets = []
            
            for dataset in self.bigquery_client.list_datasets(project_id):
                # Get dataset details
                dataset.reload()
                
                datasets.append({
                    "dataset_id": dataset.dataset_id,
                    "project_id": dataset.project,
                    "friendly_name": dataset.friendly_name,
                    "description": dataset.description,
                    "labels": dataset.labels,
                    "created": dataset.created.isoformat() if dataset.created else None,
                    "updated": dataset.modified.isoformat() if dataset.modified else None,
                    "default_table_expiration_ms": dataset.default_table_expiration_ms,
                    "default_partition_expiration_ms": dataset.default_partition_expiration_ms,
                    "location": dataset.location
                })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(datasets)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_bigquery_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud BigQuery datasets"
                )
            
            return {
                "success": True,
                "project_id": project_id,
                "total_count": len(datasets),
                "execution_time": execution_time,
                "data": datasets,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCP BigQuery datasets: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_network_resources(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get network resources including VPCs, subnets, and firewall rules."""
        try:
            if not self.asset_client:
                raise Exception("Asset client not initialized")
            
            if not project_id:
                project_id = self.projects[0]
            
            # Search for network resources
            request = asset_v1.SearchAllResourcesRequest(
                scope=f"projects/{project_id}",
                asset_types=[
                    "compute.googleapis.com/Network",
                    "compute.googleapis.com/Subnetwork",
                    "compute.googleapis.com/Firewall",
                    "compute.googleapis.com/Router"
                ]
            )
            
            # Execute search
            start_time = datetime.now()
            page_result = self.asset_client.search_all_resources(request=request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            resources = []
            for resource in page_result:
                resources.append({
                    "name": resource.name,
                    "asset_type": resource.asset_type,
                    "project": resource.project,
                    "location": resource.location,
                    "labels": dict(resource.labels) if resource.labels else {},
                    "description": resource.description,
                    "display_name": resource.display_name
                })
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(resources)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_network_resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud network resources"
                )
            
            return {
                "success": True,
                "project_id": project_id,
                "total_count": len(resources),
                "execution_time": execution_time,
                "data": resources,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCP network resources: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_iam_policies(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get IAM policies for resources."""
        try:
            if not self.asset_client:
                raise Exception("Asset client not initialized")
            
            if not project_id:
                project_id = self.projects[0]
            
            # Search for IAM policies
            request = asset_v1.SearchAllResourcesRequest(
                scope=f"projects/{project_id}",
                asset_types=["cloudresourcemanager.googleapis.com/Project"]
            )
            
            # Execute search
            start_time = datetime.now()
            page_result = self.asset_client.search_all_resources(request=request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            resources = []
            for resource in page_result:
                resources.append({
                    "name": resource.name,
                    "asset_type": resource.asset_type,
                    "project": resource.project,
                    "location": resource.location,
                    "labels": dict(resource.labels) if resource.labels else {},
                    "description": resource.description,
                    "display_name": resource.display_name
                })
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(resources)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"gcp_iam_policies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    "Google Cloud IAM policies"
                )
            
            return {
                "success": True,
                "project_id": project_id,
                "total_count": len(resources),
                "execution_time": execution_time,
                "data": resources,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCP IAM policies: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_to_local_scratch(self, query_results: Dict[str, Any], 
                               export_format: str = "csv") -> str:
        """
        Export query results to local scratch tools for processing.
        
        Args:
            query_results: Results from any GCP query method
            export_format: Export format (csv, json, parquet)
            
        Returns:
            Path to exported file
        """
        if not query_results.get("success") or query_results.get("dataframe") is None:
            raise Exception("No data to export")
        
        df = query_results["dataframe"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "csv":
            filename = f"gcp_scratch_{timestamp}.csv"
            self.session_manager.save_dataframe(df, filename.replace('.csv', ''), 
                                             f"Google Cloud data for local processing")
        elif export_format == "json":
            filename = f"gcp_scratch_{timestamp}.json"
            self.session_manager.save_text_output(
                df.to_json(orient='records', indent=2),
                filename.replace('.json', ''),
                "Google Cloud data in JSON format for local processing"
            )
        elif export_format == "parquet":
            filename = f"gcp_scratch_{timestamp}.parquet"
            df.to_parquet(f"session-outputs/{self.session_manager.session_id}/data/{filename}")
        
        return filename
    
    def get_available_queries(self) -> Dict[str, str]:
        """Get list of available pre-built queries."""
        return {
            "list_projects": "List all accessible GCP projects",
            "search_resources": "Search for resources using Asset Inventory",
            "compute_instances": "Get all compute instances with network and disk info",
            "storage_buckets": "Get storage buckets with security settings",
            "bigquery_datasets": "Get BigQuery datasets with configuration",
            "network_resources": "Get network resources including VPCs and firewalls",
            "iam_policies": "Get IAM policies and permissions"
        }
    
    def get_query_templates(self) -> Dict[str, str]:
        """Get search query templates for common operations."""
        return {
            "resources_by_type": "assetType:{asset_type}",
            "resources_by_location": "location:{location}",
            "resources_by_project": "project:{project_id}",
            "resources_with_labels": "labels.{label_key}:{label_value}",
            "security_resources": "assetType:(compute.googleapis.com/Firewall OR compute.googleapis.com/Network OR compute.googleapis.com/Subnetwork)",
            "storage_resources": "assetType:(storage.googleapis.com/Bucket OR compute.googleapis.com/Disk)",
            "compute_resources": "assetType:(compute.googleapis.com/Instance OR compute.googleapis.com/Disk OR compute.googleapis.com/Network)"
        }

# MCP Tools for Google Resource Manager
class GoogleResourceManagerMCPTools:
    """MCP-compatible tools for Google Cloud Resource Manager integration."""
    
    def __init__(self, gcp_integration: GoogleResourceManagerIntegration):
        self.gcp = gcp_integration
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for Google Cloud Resource Manager."""
        return {
            "gcp_list_projects": {
                "name": "gcp_list_projects",
                "description": "List all accessible Google Cloud projects",
                "parameters": {
                    "parent": {"type": "string", "description": "Parent resource (organization or folder)"}
                },
                "returns": {"type": "object", "description": "Project list with metadata"}
            },
            "gcp_search_resources": {
                "name": "gcp_search_resources",
                "description": "Search for GCP resources using Asset Inventory",
                "parameters": {
                    "query": {"type": "string", "description": "Search query string"},
                    "asset_types": {"type": "array", "items": {"type": "string"}, "description": "Asset types to search for"},
                    "projects": {"type": "array", "items": {"type": "string"}, "description": "Project IDs to search in"}
                },
                "returns": {"type": "object", "description": "Search results with data and metadata"}
            },
            "gcp_get_compute_instances": {
                "name": "gcp_get_compute_instances",
                "description": "Get all compute instances with detailed information",
                "parameters": {
                    "project_id": {"type": "string", "description": "Project ID (optional)"},
                    "zone": {"type": "string", "description": "Zone (optional)"}
                },
                "returns": {"type": "object", "description": "Compute instance data with metadata"}
            },
            "gcp_get_storage_buckets": {
                "name": "gcp_get_storage_buckets",
                "description": "Get all storage buckets with security settings",
                "parameters": {
                    "project_id": {"type": "string", "description": "Project ID (optional)"}
                },
                "returns": {"type": "object", "description": "Storage bucket data with metadata"}
            },
            "gcp_export_to_scratch": {
                "name": "gcp_export_to_scratch",
                "description": "Export Google Cloud data to local scratch tools for processing",
                "parameters": {
                    "query_results": {"type": "object", "description": "Results from GCP query"},
                    "export_format": {"type": "string", "description": "Export format (csv, json, parquet)"}
                },
                "returns": {"type": "string", "description": "Path to exported file"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute Google Cloud Resource Manager MCP tool."""
        if tool_name == "gcp_list_projects":
            return self.gcp.list_projects(**kwargs)
        elif tool_name == "gcp_search_resources":
            return self.gcp.search_resources(**kwargs)
        elif tool_name == "gcp_get_compute_instances":
            return self.gcp.get_compute_instances(**kwargs)
        elif tool_name == "gcp_get_storage_buckets":
            return self.gcp.get_storage_buckets(**kwargs)
        elif tool_name == "gcp_export_to_scratch":
            return self.gcp.export_to_local_scratch(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
