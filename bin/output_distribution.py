"""
Output Distribution Tools for Cybersecurity Agent

Provides capabilities to distribute output files to various destinations including
CIFS/SMB, NFS, object storage, SSH, FTP, SCP, and streaming to TCP/UDP ports.
"""

import json
import logging
import socket
import ftplib
import paramiko
import smbclient
import smbprotocol
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import asyncio
import aiofiles
import aiohttp
import zipfile
import tempfile
import shutil

from .enhanced_session_manager import EnhancedSessionManager
from .credential_vault import CredentialVault

logger = logging.getLogger(__name__)

class OutputDistributor:
    """Comprehensive output distribution system for cybersecurity agent."""
    
    def __init__(self, session_manager: EnhancedSessionManager, 
                 credential_vault: CredentialVault):
        self.session_manager = session_manager
        self.credential_vault = credential_vault
        
        # Supported output formats
        self.output_formats = {
            'json': self._format_as_json,
            'cef': self._format_as_cef,
            'csv': self._format_as_csv,
            'xml': self._format_as_xml,
            'syslog': self._format_as_syslog,
            'leef': self._format_as_leef,
            'raw': self._format_as_raw
        }
        
        # Distribution history
        self.distribution_history = []
    
    def distribute_to_cifs(self, file_path: str, share_path: str, 
                          destination_path: str, credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Distribute file to CIFS/SMB share.
        
        Args:
            file_path: Path to the file to distribute
            share_path: CIFS share path (e.g., //server/share)
            destination_path: Destination path within the share
            credentials: Optional credentials (username, password, domain)
            
        Returns:
            Distribution result
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Source file not found: {file_path}'
                }
            
            # Get credentials from vault if not provided
            if not credentials:
                credentials = self._get_credentials_for_share(share_path)
            
            # Connect to CIFS share
            if credentials:
                smbclient.ClientConfig(username=credentials.get('username'),
                                     password=credentials.get('password'),
                                     domain=credentials.get('domain'))
            
            # Copy file to share
            with smbclient.open_file(f"{share_path}/{destination_path}", mode='wb') as dest_file:
                with open(source_path, 'rb') as src_file:
                    shutil.copyfileobj(src_file, dest_file)
            
            # Record distribution
            self._record_distribution('cifs', file_path, share_path, destination_path, True)
            
            return {
                'success': True,
                'message': f'File distributed to CIFS share: {share_path}/{destination_path}',
                'source': str(source_path),
                'destination': f"{share_path}/{destination_path}",
                'file_size': source_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute to CIFS: {e}")
            self._record_distribution('cifs', file_path, share_path, destination_path, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def distribute_to_nfs(self, file_path: str, mount_point: str, 
                         destination_path: str) -> Dict[str, Any]:
        """
        Distribute file to NFS mount point.
        
        Args:
            file_path: Path to the file to distribute
            mount_point: NFS mount point
            destination_path: Destination path within the mount
            
        Returns:
            Distribution result
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Source file not found: {file_path}'
                }
            
            nfs_path = Path(mount_point) / destination_path
            
            # Ensure destination directory exists
            nfs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to NFS
            shutil.copy2(source_path, nfs_path)
            
            # Record distribution
            self._record_distribution('nfs', file_path, mount_point, destination_path, True)
            
            return {
                'success': True,
                'message': f'File distributed to NFS: {nfs_path}',
                'source': str(source_path),
                'destination': str(nfs_path),
                'file_size': source_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute to NFS: {e}")
            self._record_distribution('nfs', file_path, mount_point, destination_path, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def distribute_to_object_storage(self, file_path: str, bucket_name: str, 
                                   object_key: str, storage_type: str = 's3',
                                   credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Distribute file to object storage (S3, Azure Blob, GCS).
        
        Args:
            file_path: Path to the file to distribute
            bucket_name: Storage bucket/container name
            object_key: Object key within the bucket
            storage_type: Type of storage (s3, azure, gcs)
            credentials: Optional credentials
            
        Returns:
            Distribution result
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Source file not found: {file_path}'
                }
            
            # Get credentials from vault if not provided
            if not credentials:
                credentials = self._get_credentials_for_storage(storage_type, bucket_name)
            
            if storage_type.lower() == 's3':
                return self._distribute_to_s3(source_path, bucket_name, object_key, credentials)
            elif storage_type.lower() == 'azure':
                return self._distribute_to_azure_blob(source_path, bucket_name, object_key, credentials)
            elif storage_type.lower() == 'gcs':
                return self._distribute_to_gcs(source_path, bucket_name, object_key, credentials)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported storage type: {storage_type}'
                }
                
        except Exception as e:
            logger.error(f"Failed to distribute to object storage: {e}")
            self._record_distribution('object_storage', file_path, bucket_name, object_key, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def _distribute_to_s3(self, source_path: Path, bucket_name: str, 
                          object_key: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Distribute file to S3."""
        try:
            import boto3
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get('access_key'),
                aws_secret_access_key=credentials.get('secret_key'),
                region_name=credentials.get('region', 'us-east-1')
            )
            
            # Upload file
            s3_client.upload_file(str(source_path), bucket_name, object_key)
            
            # Record distribution
            self._record_distribution('s3', str(source_path), bucket_name, object_key, True)
            
            return {
                'success': True,
                'message': f'File uploaded to S3: s3://{bucket_name}/{object_key}',
                'source': str(source_path),
                'destination': f"s3://{bucket_name}/{object_key}",
                'file_size': source_path.stat().st_size
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'boto3 not installed. Install with: pip install boto3'
            }
        except Exception as e:
            logger.error(f"Failed to distribute to S3: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def distribute_via_ssh(self, file_path: str, host: str, port: int, 
                          remote_path: str, credentials: Optional[Dict[str, str]] = None,
                          use_scp: bool = True) -> Dict[str, Any]:
        """
        Distribute file via SSH/SCP.
        
        Args:
            file_path: Path to the file to distribute
            host: SSH host
            port: SSH port
            remote_path: Remote destination path
            credentials: Optional credentials (username, password, key_file)
            use_scp: Whether to use SCP (True) or SFTP (False)
            
        Returns:
            Distribution result
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Source file not found: {file_path}'
                }
            
            # Get credentials from vault if not provided
            if not credentials:
                credentials = self._get_credentials_for_host(host)
            
            if use_scp:
                return self._distribute_via_scp(source_path, host, port, remote_path, credentials)
            else:
                return self._distribute_via_sftp(source_path, host, port, remote_path, credentials)
                
        except Exception as e:
            logger.error(f"Failed to distribute via SSH: {e}")
            self._record_distribution('ssh', file_path, host, remote_path, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def _distribute_via_scp(self, source_path: Path, host: str, port: int, 
                           remote_path: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Distribute file via SCP."""
        try:
            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to host
            if credentials.get('key_file'):
                ssh_client.connect(
                    host, port=port, username=credentials['username'],
                    key_filename=credentials['key_file']
                )
            else:
                ssh_client.connect(
                    host, port=port, username=credentials['username'],
                    password=credentials['password']
                )
            
            # Use SCP to transfer file
            scp_client = ssh_client.open_sftp()
            scp_client.put(str(source_path), remote_path)
            scp_client.close()
            ssh_client.close()
            
            # Record distribution
            self._record_distribution('scp', str(source_path), host, remote_path, True)
            
            return {
                'success': True,
                'message': f'File transferred via SCP to {host}:{remote_path}',
                'source': str(source_path),
                'destination': f"{host}:{remote_path}",
                'file_size': source_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute via SCP: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def distribute_via_ftp(self, file_path: str, host: str, port: int, 
                          remote_path: str, credentials: Optional[Dict[str, str]] = None,
                          use_sftp: bool = False) -> Dict[str, Any]:
        """
        Distribute file via FTP/SFTP.
        
        Args:
            file_path: Path to the file to distribute
            host: FTP host
            port: FTP port
            remote_path: Remote destination path
            credentials: Optional credentials (username, password)
            use_sftp: Whether to use SFTP (True) or FTP (False)
            
        Returns:
            Distribution result
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {
                    'success': False,
                    'error': f'Source file not found: {file_path}'
                }
            
            # Get credentials from vault if not provided
            if not credentials:
                credentials = self._get_credentials_for_host(host)
            
            if use_sftp:
                return self._distribute_via_sftp(source_path, host, port, remote_path, credentials)
            else:
                return self._distribute_via_ftp_standard(source_path, host, port, remote_path, credentials)
                
        except Exception as e:
            logger.error(f"Failed to distribute via FTP: {e}")
            self._record_distribution('ftp', file_path, host, remote_path, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def _distribute_via_ftp_standard(self, source_path: Path, host: str, port: int, 
                                    remote_path: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Distribute file via standard FTP."""
        try:
            # Connect to FTP server
            ftp = ftplib.FTP()
            ftp.connect(host, port)
            ftp.login(credentials['username'], credentials['password'])
            
            # Upload file
            with open(source_path, 'rb') as file:
                ftp.storbinary(f'STOR {remote_path}', file)
            
            ftp.quit()
            
            # Record distribution
            self._record_distribution('ftp', str(source_path), host, remote_path, True)
            
            return {
                'success': True,
                'message': f'File uploaded via FTP to {host}:{remote_path}',
                'source': str(source_path),
                'destination': f"{host}:{remote_path}",
                'file_size': source_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to distribute via FTP: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def stream_to_network(self, data: Union[str, bytes, Dict[str, Any]], 
                         host: str, port: int, protocol: str = 'tcp',
                         format_type: str = 'json', credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Stream data to network endpoint via TCP/UDP.
        
        Args:
            data: Data to stream
            host: Target host
            port: Target port
            protocol: Protocol (tcp or udp)
            format_type: Output format (json, cef, syslog, etc.)
            credentials: Optional credentials for authentication
            
        Returns:
            Streaming result
        """
        try:
            # Format data according to specified format
            if format_type in self.output_formats:
                formatted_data = self.output_formats[format_type](data)
            else:
                formatted_data = str(data)
            
            # Convert to bytes if needed
            if isinstance(formatted_data, str):
                formatted_data = formatted_data.encode('utf-8')
            
            # Stream data
            if protocol.lower() == 'tcp':
                return self._stream_tcp(formatted_data, host, port)
            elif protocol.lower() == 'udp':
                return self._stream_udp(formatted_data, host, port)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported protocol: {protocol}'
                }
                
        except Exception as e:
            logger.error(f"Failed to stream to network: {e}")
            self._record_distribution('network_stream', str(data), f"{host}:{port}", protocol, False, str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def _stream_tcp(self, data: bytes, host: str, port: int) -> Dict[str, Any]:
        """Stream data via TCP."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
                sock.sendall(data)
                
                # Record distribution
                self._record_distribution('tcp_stream', str(data), f"{host}:{port}", 'tcp', True)
                
                return {
                    'success': True,
                    'message': f'Data streamed via TCP to {host}:{port}',
                    'bytes_sent': len(data),
                    'destination': f"{host}:{port}"
                }
                
        except Exception as e:
            logger.error(f"Failed to stream via TCP: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _stream_udp(self, data: bytes, host: str, port: int) -> Dict[str, Any]:
        """Stream data via UDP."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(data, (host, port))
                
                # Record distribution
                self._record_distribution('udp_stream', str(data), f"{host}:{port}", 'udp', True)
                
                return {
                    'success': True,
                    'message': f'Data streamed via UDP to {host}:{port}',
                    'bytes_sent': len(data),
                    'destination': f"{host}:{port}"
                }
                
        except Exception as e:
            logger.error(f"Failed to stream via UDP: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_as_json(self, data: Any) -> str:
        """Format data as JSON."""
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception:
            return str(data)
    
    def _format_as_cef(self, data: Any) -> str:
        """Format data as CEF (Common Event Format)."""
        try:
            if isinstance(data, dict):
                # Extract CEF fields
                device_vendor = data.get('device_vendor', 'Unknown')
                device_product = data.get('device_product', 'Unknown')
                device_version = data.get('device_version', 'Unknown')
                signature_id = data.get('signature_id', 'Unknown')
                name = data.get('name', 'Unknown')
                severity = data.get('severity', 'Unknown')
                
                # Build CEF string
                cef = f"CEF:0|{device_vendor}|{device_product}|{device_version}|{signature_id}|{name}|{severity}|"
                
                # Add extension fields
                extensions = []
                for key, value in data.items():
                    if key not in ['device_vendor', 'device_product', 'device_version', 'signature_id', 'name', 'severity']:
                        extensions.append(f"{key}={value}")
                
                if extensions:
                    cef += " ".join(extensions)
                
                return cef
            else:
                return str(data)
        except Exception:
            return str(data)
    
    def _format_as_csv(self, data: Any) -> str:
        """Format data as CSV."""
        try:
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    
                    return output.getvalue()
                else:
                    return ','.join(str(item) for item in data)
            elif isinstance(data, dict):
                return ','.join(f"{k},{v}" for k, v in data.items())
            else:
                return str(data)
        except Exception:
            return str(data)
    
    def _format_as_xml(self, data: Any) -> str:
        """Format data as XML."""
        try:
            if isinstance(data, dict):
                xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<root>']
                
                def dict_to_xml(d, indent=2):
                    xml = []
                    for key, value in d.items():
                        if isinstance(value, dict):
                            xml.append(f"{' ' * indent}<{key}>")
                            xml.extend(dict_to_xml(value, indent + 2))
                            xml.append(f"{' ' * indent}</{key}>")
                        elif isinstance(value, list):
                            xml.append(f"{' ' * indent}<{key}>")
                            for item in value:
                                if isinstance(item, dict):
                                    xml.extend(dict_to_xml(item, indent + 2))
                                else:
                                    xml.append(f"{' ' * (indent + 2)}<item>{item}</item>")
                            xml.append(f"{' ' * indent}</{key}>")
                        else:
                            xml.append(f"{' ' * indent}<{key}>{value}</{key}>")
                    return xml
                
                xml_parts.extend(dict_to_xml(data))
                xml_parts.append('</root>')
                
                return '\n'.join(xml_parts)
            else:
                return f"<root>{data}</root>"
        except Exception:
            return f"<root>{data}</root>"
    
    def _format_as_syslog(self, data: Any) -> str:
        """Format data as syslog message."""
        try:
            timestamp = datetime.now().strftime('%b %d %H:%M:%S')
            hostname = socket.gethostname()
            
            if isinstance(data, dict):
                message = json.dumps(data)
            else:
                message = str(data)
            
            return f"{timestamp} {hostname} cybersecurity_agent: {message}"
        except Exception:
            return str(data)
    
    def _format_as_leef(self, data: Any) -> str:
        """Format data as LEEF (Log Event Extended Format)."""
        try:
            if isinstance(data, dict):
                # Build LEEF string
                leef = "LEEF:2.0|CybersecurityAgent|1.0|"
                
                # Add event fields
                event_fields = []
                for key, value in data.items():
                    event_fields.append(f"{key}={value}")
                
                leef += "|".join(event_fields)
                
                return leef
            else:
                return f"LEEF:2.0|CybersecurityAgent|1.0|{data}"
        except Exception:
            return str(data)
    
    def _format_as_raw(self, data: Any) -> str:
        """Format data as raw string."""
        return str(data)
    
    def _get_credentials_for_share(self, share_path: str) -> Optional[Dict[str, str]]:
        """Get credentials for CIFS share from vault."""
        try:
            # Extract server from share path
            server = share_path.split('/')[2] if len(share_path.split('/')) > 2 else 'unknown'
            return self.credential_vault.get_credential(f"cifs_{server}")
        except Exception:
            return None
    
    def _get_credentials_for_storage(self, storage_type: str, bucket_name: str) -> Optional[Dict[str, str]]:
        """Get credentials for object storage from vault."""
        try:
            return self.credential_vault.get_credential(f"{storage_type}_{bucket_name}")
        except Exception:
            return None
    
    def _get_credentials_for_host(self, host: str) -> Optional[Dict[str, str]]:
        """Get credentials for host from vault."""
        try:
            return self.credential_vault.get_credential(f"ssh_{host}")
        except Exception:
            return None
    
    def _record_distribution(self, method: str, source: str, destination: str, 
                           success: bool, error: Optional[str] = None):
        """Record distribution attempt."""
        record = {
            'method': method,
            'source': source,
            'destination': destination,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
        
        self.distribution_history.append(record)
        
        # Keep only last 1000 records
        if len(self.distribution_history) > 1000:
            self.distribution_history = self.distribution_history[-1000:]
    
    def get_distribution_history(self) -> Dict[str, Any]:
        """Get distribution history."""
        return {
            'success': True,
            'distribution_history': self.distribution_history,
            'total_distributions': len(self.distribution_history)
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return list(self.output_formats.keys())

# MCP Tools for Output Distribution
class OutputDistributionMCPTools:
    """MCP-compatible tools for output distribution."""
    
    def __init__(self, output_distributor: OutputDistributor):
        self.distributor = output_distributor
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP tool definitions for output distribution."""
        return {
            "distribute_to_cifs": {
                "name": "distribute_to_cifs",
                "description": "Distribute file to CIFS/SMB share",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to distribute"},
                    "share_path": {"type": "string", "description": "CIFS share path (e.g., //server/share)"},
                    "destination_path": {"type": "string", "description": "Destination path within the share"},
                    "credentials": {"type": "object", "description": "Optional credentials (username, password, domain)"}
                },
                "returns": {"type": "object", "description": "Distribution result"}
            },
            "distribute_to_nfs": {
                "name": "distribute_to_nfs",
                "description": "Distribute file to NFS mount point",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to distribute"},
                    "mount_point": {"type": "string", "description": "NFS mount point"},
                    "destination_path": {"type": "string", "description": "Destination path within the mount"}
                },
                "returns": {"type": "object", "description": "Distribution result"}
            },
            "distribute_to_object_storage": {
                "name": "distribute_to_object_storage",
                "description": "Distribute file to object storage (S3, Azure Blob, GCS)",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to distribute"},
                    "bucket_name": {"type": "string", "description": "Storage bucket/container name"},
                    "object_key": {"type": "string", "description": "Object key within the bucket"},
                    "storage_type": {"type": "string", "description": "Type of storage (s3, azure, gcs)"},
                    "credentials": {"type": "object", "description": "Optional credentials"}
                },
                "returns": {"type": "object", "description": "Distribution result"}
            },
            "distribute_via_ssh": {
                "name": "distribute_via_ssh",
                "description": "Distribute file via SSH/SCP",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to distribute"},
                    "host": {"type": "string", "description": "SSH host"},
                    "port": {"type": "integer", "description": "SSH port"},
                    "remote_path": {"type": "string", "description": "Remote destination path"},
                    "credentials": {"type": "object", "description": "Optional credentials (username, password, key_file)"},
                    "use_scp": {"type": "boolean", "description": "Whether to use SCP (True) or SFTP (False)"}
                },
                "returns": {"type": "object", "description": "Distribution result"}
            },
            "distribute_via_ftp": {
                "name": "distribute_via_ftp",
                "description": "Distribute file via FTP/SFTP",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to distribute"},
                    "host": {"type": "string", "description": "FTP host"},
                    "port": {"type": "integer", "description": "FTP port"},
                    "remote_path": {"type": "string", "description": "Remote destination path"},
                    "credentials": {"type": "object", "description": "Optional credentials (username, password)"},
                    "use_sftp": {"type": "boolean", "description": "Whether to use SFTP (True) or FTP (False)"}
                },
                "returns": {"type": "object", "description": "Distribution result"}
            },
            "stream_to_network": {
                "name": "stream_to_network",
                "description": "Stream data to network endpoint via TCP/UDP",
                "parameters": {
                    "data": {"type": "object", "description": "Data to stream"},
                    "host": {"type": "string", "description": "Target host"},
                    "port": {"type": "integer", "description": "Target port"},
                    "protocol": {"type": "string", "description": "Protocol (tcp or udp)"},
                    "format_type": {"type": "string", "description": "Output format (json, cef, syslog, etc.)"},
                    "credentials": {"type": "object", "description": "Optional credentials for authentication"}
                },
                "returns": {"type": "object", "description": "Streaming result"}
            },
            "get_supported_formats": {
                "name": "get_supported_formats",
                "description": "Get list of supported output formats",
                "parameters": {},
                "returns": {"type": "object", "description": "List of supported formats"}
            },
            "get_distribution_history": {
                "name": "get_distribution_history",
                "description": "Get distribution history",
                "parameters": {},
                "returns": {"type": "object", "description": "Distribution history"}
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute output distribution MCP tool."""
        if tool_name == "distribute_to_cifs":
            return self.distributor.distribute_to_cifs(**kwargs)
        elif tool_name == "distribute_to_nfs":
            return self.distributor.distribute_to_nfs(**kwargs)
        elif tool_name == "distribute_to_object_storage":
            return self.distributor.distribute_to_object_storage(**kwargs)
        elif tool_name == "distribute_via_ssh":
            return self.distributor.distribute_via_ssh(**kwargs)
        elif tool_name == "distribute_via_ftp":
            return self.distributor.distribute_via_ftp(**kwargs)
        elif tool_name == "stream_to_network":
            return self.distributor.stream_to_network(**kwargs)
        elif tool_name == "get_supported_formats":
            return self.distributor.get_supported_formats()
        elif tool_name == "get_distribution_history":
            return self.distributor.get_distribution_history()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

