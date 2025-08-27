#!/usr/bin/env python3
"""
Active Directory MCP Tools for Cybersecurity Agent
Provides NLP-friendly Active Directory operations with intelligent credential management
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys
from datetime import datetime

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

try:
    from active_directory_tools import get_ad_manager
    AD_MANAGER_AVAILABLE = True
except ImportError:
    AD_MANAGER_AVAILABLE = False

class ActiveDirectoryMCPTools:
    """MCP tools for Active Directory operations with NLP support."""
    
    def __init__(self):
        self.ad_manager = get_ad_manager() if AD_MANAGER_AVAILABLE else None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools."""
        if not AD_MANAGER_AVAILABLE:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "connect_to_active_directory",
                    "description": "Connect to an on-premises Active Directory domain with automatic credential management",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "Active Directory domain name (e.g., example.com)"
                            },
                            "server": {
                                "type": "string",
                                "description": "Domain controller server address or hostname"
                            },
                            "port": {
                                "type": "integer",
                                "description": "LDAP port (389 for LDAP, 636 for LDAPS)",
                                "default": 389
                            },
                            "username": {
                                "type": "string",
                                "description": "Domain user account for authentication (will be prompted if not provided)"
                            },
                            "password": {
                                "type": "string",
                                "description": "Domain user password (will be prompted if not provided)"
                            },
                            "base_dn": {
                                "type": "string",
                                "description": "Base distinguished name for searches (auto-generated from domain if not provided)"
                            },
                            "use_ssl": {
                                "type": "boolean",
                                "description": "Whether to use SSL/TLS for the connection",
                                "default": True
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for connecting (for logging and memory)"
                            }
                        },
                        "required": ["domain", "server", "username", "password"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_ad_users",
                    "description": "Retrieve all users from Active Directory with comprehensive properties",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection to query"
                            },
                            "include_properties": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific user properties to include (defaults to all common properties)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for user enumeration (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_ad_groups",
                    "description": "Retrieve all groups from Active Directory with member information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection to query"
                            },
                            "include_members": {
                                "type": "boolean",
                                "description": "Whether to include member details in the response",
                                "default": True
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for group enumeration (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ad_group_members",
                    "description": "Get all members of a specific Active Directory group",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection"
                            },
                            "group_name": {
                                "type": "string",
                                "description": "Name of the group to query (sAMAccountName)"
                            },
                            "include_user_details": {
                                "type": "boolean",
                                "description": "Whether to include full user details or just basic info",
                                "default": True
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for group member query (for logging)"
                            }
                        },
                        "required": ["connection_id", "group_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_ad_users",
                    "description": "Search for Active Directory users with flexible criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection"
                            },
                            "search_term": {
                                "type": "string",
                                "description": "Search term to look for in user attributes"
                            },
                            "search_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific fields to search in (defaults to common fields like name, email, etc.)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for user search (for logging)"
                            }
                        },
                        "required": ["connection_id", "search_term"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "export_ad_users_to_csv",
                    "description": "Export Active Directory users to a CSV file for analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path for the output CSV file"
                            },
                            "include_properties": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific user properties to include in the export"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for export (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_ad_user_permissions",
                    "description": "Analyze user permissions and group memberships for security assessment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection"
                            },
                            "username": {
                                "type": "string",
                                "description": "Username to analyze (sAMAccountName)"
                            },
                            "include_group_details": {
                                "type": "boolean",
                                "description": "Whether to include detailed group information",
                                "default": True
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for permission analysis (for logging)"
                            }
                        },
                        "required": ["connection_id", "username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_ad_connections",
                    "description": "List all available Active Directory connections with their status",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ad_connection_status",
                    "description": "Get detailed status of a specific Active Directory connection",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection to check"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "disconnect_from_active_directory",
                    "description": "Disconnect from Active Directory and clean up resources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "connection_id": {
                                "type": "string",
                                "description": "ID of the Active Directory connection to disconnect"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for disconnection (for logging)"
                            }
                        },
                        "required": ["connection_id"]
                    }
                }
            }
        ]
    
    def connect_to_active_directory(self, 
                                  domain: str,
                                  server: str,
                                  port: int = 389,
                                  username: str = None,
                                  password: str = None,
                                  base_dn: str = None,
                                  use_ssl: bool = True,
                                  reason: str = None) -> Dict[str, Any]:
        """Connect to Active Directory with intelligent credential management."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            # Create connection configuration
            connection_id = self.ad_manager.create_connection(
                domain=domain,
                server=server,
                port=port,
                username=username,
                password=password,
                base_dn=base_dn,
                use_ssl=use_ssl
            )
            
            if not connection_id:
                return {
                    "success": False,
                    "error": "Connection creation failed",
                    "message": "Failed to create Active Directory connection configuration"
                }
            
            # Attempt to connect
            if self.ad_manager.connect_to_ad(connection_id):
                cli_output = f"🔗 **Active Directory Connected Successfully!**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Domain:** {domain}\n"
                cli_output += f"   **Server:** {server}:{port}\n"
                cli_output += f"   **Username:** {username or 'N/A'}\n"
                cli_output += f"   **SSL/TLS:** {'Enabled' if use_ssl else 'Disabled'}\n"
                cli_output += f"   **Base DN:** {base_dn or 'Auto-generated'}\n"
                
                if reason:
                    cli_output += f"   **Reason:** {reason}\n"
                
                cli_output += f"\n💡 **Next Steps:**\n"
                cli_output += f"   • Use 'get_all_ad_users' to enumerate users\n"
                cli_output += f"   • Use 'get_all_ad_groups' to enumerate groups\n"
                cli_output += f"   • Use 'get_ad_group_members' to check group membership\n"
                cli_output += f"   • Use 'search_ad_users' to find specific users\n"
                cli_output += f"   • Use 'disconnect_from_active_directory' when finished\n"
                
                return {
                    "success": True,
                    "connection_id": connection_id,
                    "message": f"Successfully connected to Active Directory domain {domain}",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Connection failed",
                    "message": f"Failed to establish connection to Active Directory domain {domain}",
                    "connection_id": connection_id,
                    "cli_output": f"❌ **Active Directory Connection Failed**\n   **Domain:** {domain}\n   **Server:** {server}:{port}\n   **Error:** Connection establishment failed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error connecting to Active Directory: {e}",
                "cli_output": f"❌ **Active Directory Connection Error**\n   **Error:** {e}"
            }
    
    def get_all_ad_users(self, 
                        connection_id: str, 
                        include_properties: List[str] = None,
                        reason: str = None) -> Dict[str, Any]:
        """Retrieve all users from Active Directory."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            users = self.ad_manager.get_all_users(connection_id)
            
            if users is not None:
                cli_output = f"👥 **Active Directory Users Retrieved**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Total Users:** {len(users)}\n"
                
                if users:
                    cli_output += f"\n📋 **Sample Users:**\n"
                    for i, user in enumerate(users[:5]):  # Show first 5 users
                        cli_output += f"   **{i+1}.** {user.display_name} ({user.sam_account_name})\n"
                        cli_output += f"       Email: {user.email or 'N/A'}\n"
                        cli_output += f"       Department: {user.department or 'N/A'}\n"
                        cli_output += f"       Title: {user.title or 'N/A'}\n"
                        cli_output += f"       Groups: {len(user.member_of)} memberships\n"
                        cli_output += "\n"
                    
                    if len(users) > 5:
                        cli_output += f"   ... and {len(users) - 5} more users\n"
                else:
                    cli_output += f"   **Note:** No users found in the domain\n"
                
                if reason:
                    cli_output += f"\n💡 **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "users": [user.to_dict() for user in users],
                    "user_count": len(users),
                    "message": f"Retrieved {len(users)} users from Active Directory",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "User retrieval failed",
                    "message": "Failed to retrieve users from Active Directory",
                    "cli_output": f"❌ **User Retrieval Failed**\n   **Connection ID:** {connection_id}\n   **Error:** Could not retrieve users"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error retrieving users: {e}",
                "cli_output": f"❌ **User Retrieval Error**\n   **Error:** {e}"
            }
    
    def get_all_ad_groups(self, 
                          connection_id: str, 
                          include_members: bool = True,
                          reason: str = None) -> Dict[str, Any]:
        """Retrieve all groups from Active Directory."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            groups = self.ad_manager.get_all_groups(connection_id)
            
            if groups is not None:
                cli_output = f"👥 **Active Directory Groups Retrieved**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Total Groups:** {len(groups)}\n"
                
                if groups:
                    cli_output += f"\n📋 **Sample Groups:**\n"
                    for i, group in enumerate(groups[:5]):  # Show first 5 groups
                        cli_output += f"   **{i+1}.** {group.name} ({group.sam_account_name})\n"
                        cli_output += f"       Description: {group.description or 'N/A'}\n"
                        cli_output += f"       Scope: {group.scope}\n"
                        cli_output += f"       Members: {group.member_count} users\n"
                        cli_output += f"       Managed By: {group.managed_by or 'N/A'}\n"
                        cli_output += "\n"
                    
                    if len(groups) > 5:
                        cli_output += f"   ... and {len(groups) - 5} more groups\n"
                else:
                    cli_output += f"   **Note:** No groups found in the domain\n"
                
                if reason:
                    cli_output += f"\n💡 **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "groups": [group.to_dict() for group in groups],
                    "group_count": len(groups),
                    "message": f"Retrieved {len(groups)} groups from Active Directory",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Group retrieval failed",
                    "message": "Failed to retrieve groups from Active Directory",
                    "cli_output": f"❌ **Group Retrieval Failed**\n   **Connection ID:** {connection_id}\n   **Error:** Could not retrieve groups"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error retrieving groups: {e}",
                "cli_output": f"❌ **Group Retrieval Error**\n   **Error:** {e}"
            }
    
    def get_ad_group_members(self, 
                            connection_id: str, 
                            group_name: str,
                            include_user_details: bool = True,
                            reason: str = None) -> Dict[str, Any]:
        """Get all members of a specific Active Directory group."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            users = self.ad_manager.get_group_members(connection_id, group_name)
            
            cli_output = f"👥 **Group Members Retrieved**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Group Name:** {group_name}\n"
            cli_output += f"   **Total Members:** {len(users)}\n"
            
            if users:
                cli_output += f"\n📋 **Group Members:**\n"
                for i, user in enumerate(users[:10]):  # Show first 10 members
                    cli_output += f"   **{i+1}.** {user.display_name} ({user.sam_account_name})\n"
                    cli_output += f"       Email: {user.email or 'N/A'}\n"
                    cli_output += f"       Department: {user.department or 'N/A'}\n"
                    cli_output += f"       Title: {user.title or 'N/A'}\n"
                    cli_output += "\n"
                
                if len(users) > 10:
                    cli_output += f"   ... and {len(users) - 10} more members\n"
            else:
                cli_output += f"   **Note:** No members found in group {group_name}\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "group_name": group_name,
                "members": [user.to_dict() for user in users],
                "member_count": len(users),
                "message": f"Retrieved {len(users)} members from group {group_name}",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error retrieving group members: {e}",
                "cli_output": f"❌ **Group Member Retrieval Error**\n   **Error:** {e}"
            }
    
    def search_ad_users(self, 
                       connection_id: str, 
                       search_term: str,
                       search_fields: List[str] = None,
                       reason: str = None) -> Dict[str, Any]:
        """Search for Active Directory users with flexible criteria."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            users = self.ad_manager.search_users(connection_id, search_term, search_fields)
            
            cli_output = f"🔍 **User Search Results**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Search Term:** {search_term}\n"
            cli_output += f"   **Search Fields:** {', '.join(search_fields) if search_fields else 'Default fields'}\n"
            cli_output += f"   **Results Found:** {len(users)}\n"
            
            if users:
                cli_output += f"\n📋 **Search Results:**\n"
                for i, user in enumerate(users[:10]):  # Show first 10 results
                    cli_output += f"   **{i+1}.** {user.display_name} ({user.sam_account_name})\n"
                    cli_output += f"       Email: {user.email or 'N/A'}\n"
                    cli_output += f"       Department: {user.department or 'N/A'}\n"
                    cli_output += f"       Title: {user.title or 'N/A'}\n"
                    cli_output += "\n"
                
                if len(users) > 10:
                    cli_output += f"   ... and {len(users) - 10} more results\n"
            else:
                cli_output += f"   **Note:** No users found matching '{search_term}'\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "search_term": search_term,
                "search_fields": search_fields,
                "users": [user.to_dict() for user in users],
                "result_count": len(users),
                "message": f"Found {len(users)} users matching '{search_term}'",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error searching users: {e}",
                "cli_output": f"❌ **User Search Error**\n   **Error:** {e}"
            }
    
    def export_ad_users_to_csv(self, 
                              connection_id: str,
                              file_path: str = None,
                              include_properties: List[str] = None,
                              reason: str = None) -> Dict[str, Any]:
        """Export Active Directory users to a CSV file."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            # Get users
            users = self.ad_manager.get_all_users(connection_id)
            
            if not users:
                return {
                    "success": False,
                    "error": "No users to export",
                    "message": "No users found to export to CSV"
                }
            
            # Generate file path if not provided
            if not file_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"session-outputs/ad_users_{connection_id[:8]}_{timestamp}.csv"
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Export to CSV
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                if users:
                    # Get fieldnames from first user
                    fieldnames = list(users[0].to_dict().keys())
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for user in users:
                        user_dict = user.to_dict()
                        # Convert any remaining datetime objects
                        for key, value in user_dict.items():
                            if hasattr(value, 'isoformat'):
                                user_dict[key] = value.isoformat()
                        writer.writerow(user_dict)
            
            cli_output = f"💾 **Users Exported to CSV**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **File Path:** {file_path}\n"
            cli_output += f"   **Users Exported:** {len(users)}\n"
            cli_output += f"   **File Size:** {os.path.getsize(file_path) / 1024:.1f} KB\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "file_path": file_path,
                "users_exported": len(users),
                "file_size_kb": os.path.getsize(file_path) / 1024,
                "message": f"Successfully exported {len(users)} users to CSV",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error exporting users to CSV: {e}",
                "cli_output": f"❌ **CSV Export Error**\n   **Error:** {e}"
            }
    
    def analyze_ad_user_permissions(self, 
                                   connection_id: str,
                                   username: str,
                                   include_group_details: bool = True,
                                   reason: str = None) -> Dict[str, Any]:
        """Analyze user permissions and group memberships for security assessment."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            # Search for the specific user
            users = self.ad_manager.search_users(connection_id, username, ['sAMAccountName'])
            
            if not users:
                return {
                    "success": False,
                    "error": "User not found",
                    "message": f"User '{username}' not found in Active Directory"
                }
            
            user = users[0]
            
            # Get group details if requested
            group_details = []
            if include_group_details and user.member_of:
                for group_dn in user.member_of:
                    # Extract group name from DN
                    group_name = group_dn.split(',')[0].replace('CN=', '')
                    group_details.append({
                        'name': group_name,
                        'dn': group_dn
                    })
            
            cli_output = f"🔍 **User Permission Analysis**\n"
            cli_output += f"   **Connection ID:** {connection_id}\n"
            cli_output += f"   **Username:** {username}\n"
            cli_output += f"   **Display Name:** {user.display_name}\n"
            cli_output += f"   **Email:** {user.email or 'N/A'}\n"
            cli_output += f"   **Department:** {user.department or 'N/A'}\n"
            cli_output += f"   **Title:** {user.title or 'N/A'}\n"
            cli_output += f"   **Manager:** {user.manager or 'N/A'}\n"
            cli_output += f"   **Last Logon:** {user.last_logon.isoformat() if user.last_logon else 'Never'}\n"
            cli_output += f"   **Account Expires:** {user.account_expires.isoformat() if user.account_expires else 'Never'}\n"
            cli_output += f"   **Password Last Set:** {user.password_last_set.isoformat() if user.password_last_set else 'Unknown'}\n"
            cli_output += f"   **Groups:** {len(user.member_of)} memberships\n\n"
            
            if group_details:
                cli_output += f"📋 **Group Memberships:**\n"
                for i, group in enumerate(group_details[:10]):  # Show first 10 groups
                    cli_output += f"   **{i+1}.** {group['name']}\n"
                    cli_output += f"       DN: {group['dn']}\n"
                
                if len(group_details) > 10:
                    cli_output += f"   ... and {len(group_details) - 10} more groups\n"
            
            # Security analysis
            cli_output += f"\n🔒 **Security Analysis:**\n"
            
            # Check for common security issues
            if user.user_account_control & 0x00000002:  # ACCOUNTDISABLE
                cli_output += f"   ⚠️  **Account Disabled**\n"
            
            if user.user_account_control & 0x00000010:  # LOCKOUT
                cli_output += f"   ⚠️  **Account Locked Out**\n"
            
            if user.user_account_control & 0x00000020:  # PASSWD_NOTREQD
                cli_output += f"   ⚠️  **Password Not Required**\n"
            
            if user.user_account_control & 0x00000040:  # PASSWD_CANT_CHANGE
                cli_output += f"   ⚠️  **User Cannot Change Password**\n"
            
            if user.user_account_control & 0x00000080:  # ENCRYPTED_TEXT_PASSWORD_ALLOWED
                cli_output += f"   ⚠️  **Encrypted Text Passwords Allowed**\n"
            
            if user.user_account_control & 0x00000100:  # DONT_EXPIRE_PASSWD
                cli_output += f"   ⚠️  **Password Never Expires**\n"
            
            if user.user_account_control & 0x00000200:  # MNS_LOGON_ACCOUNT
                cli_output += f"   ⚠️  **MNS Logon Account**\n"
            
            if user.user_account_control & 0x00000400:  # SMARTCARD_REQUIRED
                cli_output += f"   ✅  **Smart Card Required**\n"
            
            if user.user_account_control & 0x00000800:  # TRUSTED_FOR_DELEGATION
                cli_output += f"   ⚠️  **Trusted for Delegation**\n"
            
            if user.user_account_control & 0x00001000:  # NOT_DELEGATED
                cli_output += f"   ✅  **Not Delegated**\n"
            
            if user.user_account_control & 0x00002000:  # USE_DES_KEY_ONLY
                cli_output += f"   ⚠️  **Use DES Key Only**\n"
            
            if user.user_account_control & 0x00004000:  # DONT_REQUIRE_PREAUTH
                cli_output += f"   ⚠️  **Pre-Authentication Not Required**\n"
            
            if user.user_account_control & 0x00008000:  # PASSWORD_EXPIRED
                cli_output += f"   ⚠️  **Password Expired**\n"
            
            if user.user_account_control & 0x00010000:  # TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION
                cli_output += f"   ⚠️  **Trusted to Authenticate for Delegation**\n"
            
            if reason:
                cli_output += f"\n💡 **Reason:** {reason}\n"
            
            return {
                "success": True,
                "username": username,
                "user_info": user.to_dict(),
                "group_details": group_details,
                "security_flags": user.user_account_control,
                "message": f"Permission analysis completed for user {username}",
                "cli_output": cli_output,
                "reason": reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error analyzing user permissions: {e}",
                "cli_output": f"❌ **Permission Analysis Error**\n   **Error:** {e}"
            }
    
    def list_ad_connections(self) -> Dict[str, Any]:
        """List all available Active Directory connections."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            connections = self.ad_manager.list_connections()
            
            cli_output = f"🔗 **Active Directory Connections**\n"
            cli_output += f"   **Total Connections:** {len(connections)}\n\n"
            
            if connections:
                for conn in connections:
                    status_icon = "🟢" if conn['is_connected'] else "🔴"
                    cli_output += f"{status_icon} **{conn['connection_id'][:8]}...**\n"
                    cli_output += f"   **Domain:** {conn['domain']}\n"
                    cli_output += f"   **Server:** {conn['server']}:{conn['port']}\n"
                    cli_output += f"   **User:** {conn['username']}\n"
                    cli_output += f"   **Status:** {'Connected' if conn['is_connected'] else 'Disconnected'}\n"
                    cli_output += f"   **Created:** {conn['created_at'][:10]}\n"
                    cli_output += f"   **Last Used:** {conn['last_used'][:10]}\n\n"
            else:
                cli_output += f"   **No connections available**\n"
                cli_output += f"   Use 'connect_to_active_directory' to create a new connection\n"
            
            return {
                "success": True,
                "connections": connections,
                "connection_count": len(connections),
                "message": f"Found {len(connections)} Active Directory connections",
                "cli_output": cli_output
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error listing connections: {e}",
                "cli_output": f"❌ **Connection List Error**\n   **Error:** {e}"
            }
    
    def get_ad_connection_status(self, connection_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific Active Directory connection."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            status = self.ad_manager.get_connection_status(connection_id)
            
            if status:
                cli_output = f"📊 **Connection Status**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Domain:** {status['domain']}\n"
                cli_output += f"   **Server:** {status['server']}:{status['port']}\n"
                cli_output += f"   **Connected:** {'Yes' if status['is_connected'] else 'No'}\n"
                cli_output += f"   **Working:** {'Yes' if status['connection_working'] else 'No'}\n"
                cli_output += f"   **Created:** {status['created_at'][:10]}\n"
                cli_output += f"   **Last Used:** {status['last_used'][:10]}\n"
                
                return {
                    "success": True,
                    "status": status,
                    "message": "Connection status retrieved successfully",
                    "cli_output": cli_output
                }
            else:
                return {
                    "success": False,
                    "error": "Connection not found",
                    "message": f"Connection ID {connection_id} not found",
                    "cli_output": f"❌ **Connection Not Found**\n   **Connection ID:** {connection_id}\n   **Error:** Connection does not exist"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error getting connection status: {e}",
                "cli_output": f"❌ **Status Check Error**\n   **Error:** {e}"
            }
    
    def disconnect_from_active_directory(self, connection_id: str, reason: str = None) -> Dict[str, Any]:
        """Disconnect from Active Directory and clean up resources."""
        if not AD_MANAGER_AVAILABLE:
            return {
                "success": False,
                "error": "Active Directory manager not available",
                "message": "Active Directory tools not available"
            }
        
        try:
            success = self.ad_manager.disconnect_from_ad(connection_id)
            
            if success:
                cli_output = f"🔌 **Active Directory Disconnected**\n"
                cli_output += f"   **Connection ID:** {connection_id}\n"
                cli_output += f"   **Status:** Successfully disconnected\n"
                cli_output += f"   **Resources:** Cleaned up and reclaimed\n"
                
                if reason:
                    cli_output += f"   **Reason:** {reason}\n"
                
                return {
                    "success": True,
                    "message": "Successfully disconnected from Active Directory",
                    "cli_output": cli_output,
                    "reason": reason
                }
            else:
                return {
                    "success": False,
                    "error": "Disconnection failed",
                    "message": "Failed to disconnect from Active Directory",
                    "cli_output": f"❌ **Disconnection Failed**\n   **Connection ID:** {connection_id}\n   **Error:** Could not disconnect"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error disconnecting from Active Directory: {e}",
                "cli_output": f"❌ **Disconnection Error**\n   **Error:** {e}"
            }

# Global instance
_ad_mcp_tools = None

def get_ad_mcp_tools() -> ActiveDirectoryMCPTools:
    """Get or create the global Active Directory MCP tools instance."""
    global _ad_mcp_tools
    if _ad_mcp_tools is None:
        _ad_mcp_tools = ActiveDirectoryMCPTools()
    return _ad_mcp_tools

if __name__ == "__main__":
    # Test the MCP tools
    tools = get_ad_mcp_tools()
    
    print("🧪 Testing Active Directory MCP Tools...")
    
    # Test getting tools
    available_tools = tools.get_tools()
    print(f"Available tools: {len(available_tools)}")
    for tool in available_tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    
    # Test connection listing
    result = tools.list_ad_connections()
    print(f"List connections: {result}")
