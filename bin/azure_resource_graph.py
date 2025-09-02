"""
Azure Resource Graph Integration for Cybersecurity Agent

Provides comprehensive access to Azure Resource Graph for querying Azure resources,
managing credentials, and handling local data processing.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest
from azure.core.exceptions import AzureError

from .credential_vault import CredentialVault
from .context_memory_manager import ContextMemoryManager
from .enhanced_session_manager import EnhancedSessionManager

logger = logging.getLogger(__name__)

class AzureResourceGraphIntegration:
    """Comprehensive Azure Resource Graph integration with credential management and local data handling."""
    
    def __init__(self, session_manager: EnhancedSessionManager, credential_vault: CredentialVault, memory_manager: ContextMemoryManager):
        self.session_manager = session_manager
        self.credential_vault = credential_vault
        self.memory_manager = memory_manager
        self.client = None
        self.subscriptions = []
        self.connection_info = {}
        
        # Initialize connection from memory or prompt user
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Azure connection from memory or prompt user."""
        try:
            # Try to get connection info from long-term memory
            connection_data = self.memory_manager.search("azure_connection_info", domain="azure", max_entries=1)
            
            if connection_data:
                self.connection_info = connection_data[0].data
                self._connect_with_stored_credentials()
            else:
                self._prompt_for_credentials()
                
        except Exception as e:
            logger.warning(f"Failed to initialize Azure connection from memory: {e}")
            self._prompt_for_credentials()
    
    def _prompt_for_credentials(self):
        """Prompt user for Azure credentials and store them."""
        print("\n🔐 Azure Resource Graph Authentication Required")
        print("Please provide your Azure connection details:")
        
        # Get authentication method
        auth_method = input("Authentication method (1: Service Principal, 2: Default Credentials): ").strip()
        
        if auth_method == "1":
            # Service Principal authentication
            tenant_id = input("Tenant ID: ").strip()
            client_id = input("Client ID: ").strip()
            client_secret = input("Client Secret: ").strip()
            subscription_id = input("Subscription ID: ").strip()
            
            # Store in credential vault
            self.credential_vault.store_credential(
                "azure_service_principal",
                {
                    "tenant_id": tenant_id,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "subscription_id": subscription_id
                }
            )
            
            self.connection_info = {
                "auth_method": "service_principal",
                "tenant_id": tenant_id,
                "client_id": client_id,
                "subscription_id": subscription_id,
                "last_updated": datetime.now().isoformat()
            }
            
        else:
            # Default credentials
            subscription_id = input("Subscription ID: ").strip()
            
            self.connection_info = {
                "auth_method": "default_credentials",
                "subscription_id": subscription_id,
                "last_updated": datetime.now().isoformat()
            }
        
        # Store connection info in long-term memory
        self.memory_manager.import_data(
            "azure_connection_info",
            self.connection_info,
            domain="azure",
            tier="long_term",
            ttl_days=365,
            metadata={
                "description": "Azure Resource Graph connection configuration",
                "type": "connection_info"
            }
        )
        
        self._connect_with_stored_credentials()
    
    def _connect_with_stored_credentials(self):
        """Connect to Azure using stored credentials."""
        try:
            if self.connection_info["auth_method"] == "service_principal":
                # Get credentials from vault
                creds = self.credential_vault.get_credential("azure_service_principal")
                
                credential = ClientSecretCredential(
                    tenant_id=creds["tenant_id"],
                    client_id=creds["client_id"],
                    client_secret=creds["client_secret"]
                )
                
                self.client = ResourceGraphClient(credential)
                self.subscriptions = [creds["subscription_id"]]
                
            else:
                # Default credentials
                credential = DefaultAzureCredential()
                self.client = ResourceGraphClient(credential)
                self.subscriptions = [self.connection_info["subscription_id"]]
            
            logger.info("Successfully connected to Azure Resource Graph")
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure: {e}")
            raise
    
    def query_resources(self, query: str, subscriptions: Optional[List[str]] = None, 
                       max_results: int = 1000) -> Dict[str, Any]:
        """
        Execute a KQL query against Azure Resource Graph.
        
        Args:
            query: KQL query string
            subscriptions: List of subscription IDs (defaults to stored subscriptions)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            if not self.client:
                raise Exception("Azure client not initialized")
            
            if not subscriptions:
                subscriptions = self.subscriptions
            
            # Create query request
            query_request = QueryRequest(
                query=query,
                subscriptions=subscriptions,
                options={"top": max_results}
            )
            
            # Execute query
            start_time = datetime.now()
            result = self.client.resources(query_request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            resources = []
            if hasattr(result, 'data') and result.data:
                for item in result.data:
                    resources.append(item)
            
            # Convert to DataFrame for local processing
            df = pd.DataFrame(resources)
            
            # Save to session for local scratch work
            if not df.empty:
                filename = f"azure_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.session_manager.save_dataframe(
                    df, 
                    filename.replace('.csv', ''), 
                    f"Azure Resource Graph query results: {query[:100]}..."
                )
            
            return {
                "success": True,
                "query": query,
                "subscriptions": subscriptions,
                "total_count": len(resources),
                "execution_time": execution_time,
                "data": resources,
                "dataframe": df,
                "local_file": filename if not df.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Azure Resource Graph query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_all_resources(self, resource_types: Optional[List[str]] = None, 
                         locations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get all resources or filtered subset.
        
        Args:
            resource_types: List of resource types to filter by
            locations: List of Azure regions to filter by
            
        Returns:
            Query results dictionary
        """
        # Build KQL query
        query_parts = ["Resources"]
        
        if resource_types:
            resource_filter = " or ".join([f"type =~ '{rt}'" for rt in resource_types])
            query_parts.append(f"| where {resource_filter}")
        
        if locations:
            location_filter = " or ".join([f"location =~ '{loc}'" for loc in locations])
            query_parts.append(f"| where {location_filter}")
        
        query = " ".join(query_parts)
        
        return self.query_resources(query)
    
    def get_virtual_machines(self, include_disks: bool = True, include_networking: bool = True) -> Dict[str, Any]:
        """Get all virtual machines with optional related resources."""
        query_parts = [
            "Resources",
            "| where type =~ 'microsoft.compute/virtualmachines'"
        ]
        
        if include_disks:
            query_parts.extend([
                "| extend diskCount = array_length(properties.storageProfile.dataDisks)",
                "| extend osDiskSize = properties.storageProfile.osDisk.diskSizeGB"
            ])
        
        if include_networking:
            query_parts.extend([
                "| extend networkInterfaces = properties.networkProfile.networkInterfaces",
                "| extend publicIPs = properties.networkProfile.networkInterfaces"
            ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_network_resources(self, include_security: bool = True) -> Dict[str, Any]:
        """Get network resources with security information."""
        query_parts = [
            "Resources",
            "| where type in~ ('microsoft.network/virtualnetworks', 'microsoft.network/networksecuritygroups', 'microsoft.network/loadbalancers')"
        ]
        
        if include_security:
            query_parts.extend([
                "| extend securityRules = case(type =~ 'microsoft.network/networksecuritygroups', properties.securityRules, [])",
                "| extend addressSpace = case(type =~ 'microsoft.network/virtualnetworks', properties.addressSpace.addressPrefixes, [])"
            ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_storage_accounts(self, include_containers: bool = True) -> Dict[str, Any]:
        """Get storage accounts with optional container information."""
        query_parts = [
            "Resources",
            "| where type =~ 'microsoft.storage/storageaccounts'"
        ]
        
        if include_containers:
            query_parts.extend([
                "| extend blobServices = properties.primaryEndpoints.blob",
                "| extend fileServices = properties.primaryEndpoints.file"
            ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_key_vaults(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get Key Vaults with optional secret information."""
        query_parts = [
            "Resources",
            "| where type =~ 'microsoft.keyvault/vaults'"
        ]
        
        if include_secrets:
            query_parts.extend([
                "| extend secretCount = array_length(properties.accessPolicies)",
                "| extend enabledForDeployment = properties.enabledForDeployment",
                "| extend enabledForDiskEncryption = properties.enabledForDiskEncryption"
            ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_app_service_plans(self, include_apps: bool = True) -> Dict[str, Any]:
        """Get App Service Plans with optional app information."""
        query_parts = [
            "Resources",
            "| where type =~ 'microsoft.web/serverfarms'"
        ]
        
        if include_apps:
            query_parts.extend([
                "| extend appCount = array_length(properties.numberOfSites)",
                "| extend sku = properties.sku.name",
                "| extend capacity = properties.sku.capacity"
            ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_cost_analysis(self, time_range: str = "P30D") -> Dict[str, Any]:
        """Get cost analysis data for resources."""
        query = f"""
        Resources
        | where type =~ 'microsoft.consumption/usageDetails'
        | where properties.usageStart >= ago({time_range})
        | extend cost = todouble(properties.pretaxCost)
        | extend resourceGroup = properties.resourceGroup
        | extend resourceType = properties.meterDetails.meterName
        | summarize totalCost = sum(cost), resourceCount = count() by resourceGroup, resourceType
        | order by totalCost desc
        """
        
        return self.query_resources(query)
    
    def get_compliance_status(self, policy_assignments: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get compliance status for resources."""
        query_parts = [
            "Resources",
            "| where type =~ 'microsoft.policyinsights/policystates'"
        ]
        
        if policy_assignments:
            policy_filter = " or ".join([f"properties.policyAssignmentId =~ '{pa}'" for pa in policy_assignments])
            query_parts.append(f"| where {policy_filter}")
        
        query_parts.extend([
            "| extend complianceState = properties.complianceState",
            "| extend policyDefinitionId = properties.policyDefinitionId",
            "| extend resourceId = properties.resourceId"
        ])
        
        query = " ".join(query_parts)
        return self.query_resources(query)
    
    def get_resource_relationships(self, resource_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get resource relationships and dependencies."""
        query = f"""
        Resources
        | where id =~ '{resource_id}' or id in (
            Resources
            | where id =~ '{resource_id}'
            | extend dependencies = properties.dependsOn
            | mv-expand dependencies
            | project dependencies
        )
        | extend resourceType = type
        | extend location = location
        | extend resourceGroup = resourceGroup
        | extend tags = tags
        """
        
        return self.query_resources(query)
    
    def export_to_local_scratch(self, query_results: Dict[str, Any], 
                               export_format: str = "csv") -> str:
        """
        Export query results to local scratch tools for processing.
        
        Args:
            query_results: Results from query_resources
            export_format: Export format (csv, json, parquet)
            
        Returns:
            Path to exported file
        """
        if not query_results.get("success") or query_results.get("dataframe") is None:
            raise Exception("No data to export")
        
        df = query_results["dataframe"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "csv":
            filename = f"azure_scratch_{timestamp}.csv"
            self.session_manager.save_dataframe(df, filename.replace('.csv', ''), 
                                             f"Azure data for local processing")
        elif export_format == "json":
            filename = f"azure_scratch_{timestamp}.json"
            self.session_manager.save_text_output(
                df.to_json(orient='records', indent=2),
                filename.replace('.json', ''),
                "Azure data in JSON format for local processing"
            )
        elif export_format == "parquet":
            filename = f"azure_scratch_{timestamp}.parquet"
            df.to_parquet(f"session-outputs/{self.session_manager.session_id}/data/{filename}")
        
        return filename
    
    def get_available_queries(self) -> Dict[str, str]:
        """Get list of available pre-built queries."""
        return {
            "all_vms": "Get all virtual machines with disk and network info",
            "network_resources": "Get network resources with security groups",
            "storage_accounts": "Get storage accounts with service endpoints",
            "key_vaults": "Get Key Vaults with access policies",
            "app_services": "Get App Service Plans with app counts",
            "cost_analysis": "Get cost analysis for last 30 days",
            "compliance_status": "Get policy compliance status",
            "resource_relationships": "Get resource dependencies and relationships"
        }
    
    def get_query_templates(self) -> Dict[str, str]:
        """Get KQL query templates for common operations."""
        return {
            "resource_by_type": "Resources | where type =~ '{resource_type}' | project name, type, location, resourceGroup",
            "resources_by_tag": "Resources | where tags.{tag_key} =~ '{tag_value}' | project name, type, location, tags",
            "resources_by_location": "Resources | where location =~ '{location}' | summarize count() by type",
            "resources_by_resource_group": "Resources | where resourceGroup =~ '{resource_group}' | project name, type, location",
            "cost_by_resource_group": "Resources | where type =~ 'microsoft.consumption/usageDetails' | where properties.resourceGroup =~ '{resource_group}' | summarize totalCost = sum(todouble(properties.pretaxCost))",
            "security_analysis": "Resources | where type in~ ('microsoft.network/networksecuritygroups', 'microsoft.keyvault/vaults', 'microsoft.storage/storageaccounts') | extend securityScore = case(type =~ 'microsoft.network/networksecuritygroups', array_length(properties.securityRules), 0)"
        }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate KQL query syntax and structure."""
        try:
            # Basic KQL validation
            if not query.strip():
                return {"valid": False, "error": "Query cannot be empty"}
            
            if not query.lower().startswith("resources"):
                return {"valid": False, "error": "Query must start with 'Resources'"}
            
            # Try to parse basic structure
            if "|" in query:
                parts = query.split("|")
                for part in parts[1:]:
                    if part.strip() and not any(op in part.lower() for op in ["where", "project", "extend", "summarize", "order", "limit"]):
                        return {"valid": False, "error": f"Invalid operator in: {part.strip()}"}
            
            return {"valid": True, "message": "Query syntax appears valid"}
            
        except Exception as e:
            return {"valid": False, "error": f"Query validation failed: {str(e)}"}

# MCP Tools for Azure Resource Graph
class AzureResourceGraphMCPTools:
    """MCP-compatible tools for Azure Resource Graph integration."""
    
    def __init__(self, azure_integration: AzureResourceGraphIntegration):
        self.azure = azure_integration
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for Azure Resource Graph."""
        return {
            "azure_query_resources": {
                "name": "azure_query_resources",
                "description": "Execute KQL query against Azure Resource Graph",
                "parameters": {
                    "query": {"type": "string", "description": "KQL query string"},
                    "subscriptions": {"type": "array", "items": {"type": "string"}, "description": "Subscription IDs (optional)"},
                    "max_results": {"type": "integer", "description": "Maximum results to return"}
                },
                "returns": {"type": "object", "description": "Query results with data and metadata"}
            },
            "azure_get_vms": {
                "name": "azure_get_vms",
                "description": "Get all virtual machines with optional disk and network information",
                "parameters": {
                    "include_disks": {"type": "boolean", "description": "Include disk information"},
                    "include_networking": {"type": "boolean", "description": "Include network information"}
                },
                "returns": {"type": "object", "description": "VM data with optional related resources"}
            },
            "azure_get_network_resources": {
                "name": "azure_get_network_resources",
                "description": "Get network resources with security information",
                "parameters": {
                    "include_security": {"type": "boolean", "description": "Include security group information"}
                },
                "returns": {"type": "object", "description": "Network resources with security details"}
            },
            "azure_export_to_scratch": {
                "name": "azure_export_to_scratch",
                "description": "Export Azure data to local scratch tools for processing",
                "parameters": {
                    "query_results": {"type": "object", "description": "Results from Azure query"},
                    "export_format": {"type": "string", "description": "Export format (csv, json, parquet)"}
                },
                "returns": {"type": "string", "description": "Path to exported file"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute Azure Resource Graph MCP tool."""
        if tool_name == "azure_query_resources":
            return self.azure.query_resources(**kwargs)
        elif tool_name == "azure_get_vms":
            return self.azure.get_virtual_machines(**kwargs)
        elif tool_name == "azure_get_network_resources":
            return self.azure.get_network_resources(**kwargs)
        elif tool_name == "azure_export_to_scratch":
            return self.azure.export_to_local_scratch(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
