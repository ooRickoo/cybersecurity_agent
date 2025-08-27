#!/usr/bin/env python3
"""
Active Directory Tools for Cybersecurity Agent
Provides read-only access to on-premises Active Directory with NLP-friendly interfaces
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Add bin directory to path for imports
bin_path = Path(__file__).parent
if str(bin_path) not in sys.path:
    sys.path.insert(0, str(bin_path))

# Try to import LDAP libraries
try:
    import ldap3
    LDAP3_AVAILABLE = True
except ImportError:
    LDAP3_AVAILABLE = False

try:
    import ldap
    PYLDAP_AVAILABLE = True
except ImportError:
    PYLDAP_AVAILABLE = False

# Try to import credential vault
try:
    from credential_vault import CredentialVault
    CREDENTIAL_VAULT_AVAILABLE = True
except ImportError:
    CREDENTIAL_VAULT_AVAILABLE = False

# Try to import memory manager
try:
    from context_memory_manager import ContextMemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ADConnection:
    """Active Directory connection configuration."""
    connection_id: str
    domain: str
    server: str
    port: int
    username: str
    base_dn: str
    use_ssl: bool
    created_at: datetime
    last_used: datetime
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ADConnection':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)

@dataclass
class ADUser:
    """Active Directory user information."""
    sam_account_name: str
    user_principal_name: str
    display_name: str
    given_name: str
    surname: str
    email: str
    department: str
    title: str
    manager: str
    member_of: List[str]
    last_logon: datetime
    account_expires: datetime
    password_last_set: datetime
    user_account_control: int
    distinguished_name: str
    object_guid: str
    when_created: datetime
    when_changed: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data

@dataclass
class ADGroup:
    """Active Directory group information."""
    name: str
    sam_account_name: str
    description: str
    group_type: int
    scope: str
    member_count: int
    members: List[str]
    managed_by: str
    distinguished_name: str
    object_guid: str
    when_created: datetime
    when_changed: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data

@dataclass
class ADComputer:
    """Active Directory computer information."""
    name: str
    sam_account_name: str
    description: str
    operating_system: str
    operating_system_version: str
    last_logon: datetime
    when_created: datetime
    when_changed: datetime
    distinguished_name: str
    object_guid: str
    ip_addresses: List[str]
    location: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data

class ActiveDirectoryConnector:
    """Active Directory connector using LDAP."""
    
    def __init__(self, connection_config: ADConnection):
        self.connection_config = connection_config
        self.connection = None
        self.is_connected = False
        
        if not LDAP3_AVAILABLE and not PYLDAP_AVAILABLE:
            raise ImportError("No LDAP library available (ldap3 or python-ldap)")
    
    def connect(self) -> bool:
        """Establish connection to Active Directory."""
        try:
            if LDAP3_AVAILABLE:
                return self._connect_with_ldap3()
            elif PYLDAP_AVAILABLE:
                return self._connect_with_pyldap()
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to connect to AD: {e}")
            return False
    
    def _connect_with_ldap3(self) -> bool:
        """Connect using ldap3 library."""
        try:
            # Create server object
            server = ldap3.Server(
                self.connection_config.server,
                port=self.connection_config.port,
                use_ssl=self.connection_config.use_ssl,
                get_info=ldap3.ALL
            )
            
            # Create connection
            self.connection = ldap3.Connection(
                server,
                user=self.connection_config.username,
                password=self._get_password(),
                auto_bind=True,
                read_only=True
            )
            
            if self.connection.bound:
                self.is_connected = True
                self.connection_config.last_used = datetime.now()
                self.connection_config.is_active = True
                logger.info(f"Connected to AD: {self.connection_config.domain}")
                return True
            else:
                logger.error("Failed to bind to AD")
                return False
                
        except Exception as e:
            logger.error(f"LDAP3 connection failed: {e}")
            return False
    
    def _connect_with_pyldap(self) -> bool:
        """Connect using python-ldap library."""
        try:
            # Create LDAP object
            ldap_obj = ldap.initialize(f"ldap{'s' if self.connection_config.use_ssl else ''}://{self.connection_config.server}:{self.connection_config.port}")
            
            # Set options
            ldap_obj.set_option(ldap.OPT_REFERRALS, 0)
            ldap_obj.set_option(ldap.OPT_NETWORK_TIMEOUT, 10)
            
            # Bind
            ldap_obj.simple_bind_s(self.connection_config.username, self._get_password())
            
            self.connection = ldap_obj
            self.is_connected = True
            self.connection_config.last_used = datetime.now()
            self.connection_config.is_active = True
            logger.info(f"Connected to AD: {self.connection_config.domain}")
            return True
            
        except Exception as e:
            logger.error(f"Python-LDAP connection failed: {e}")
            return False
    
    def _get_password(self) -> str:
        """Get password from credential vault or environment."""
        if CREDENTIAL_VAULT_AVAILABLE:
            try:
                credential_key = f"ad_{self.connection_config.connection_id}"
                credential = self.credential_vault.get_credential(credential_key)
                if credential:
                    return credential['password']
            except Exception as e:
                logger.warning(f"Failed to get password from vault: {e}")
        
        # Fallback to environment variable
        env_key = f"AD_PASSWORD_{self.connection_config.connection_id.upper()}"
        return os.getenv(env_key, "")
    
    def disconnect(self) -> bool:
        """Close AD connection."""
        try:
            if self.connection:
                if LDAP3_AVAILABLE:
                    self.connection.unbind()
                elif PYLDAP_AVAILABLE:
                    self.connection.unbind_s()
                
                self.connection = None
                self.is_connected = False
                self.connection_config.is_active = False
                logger.info("AD connection closed")
                return True
        except Exception as e:
            logger.error(f"Error closing AD connection: {e}")
        return False
    
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            if not self.is_connected:
                return False
            
            if LDAP3_AVAILABLE:
                return self.connection.bound
            elif PYLDAP_AVAILABLE:
                # Try a simple search
                self.connection.search_s(self.connection_config.base_dn, ldap.SCOPE_BASE, '(objectClass=*)', ['distinguishedName'])
                return True
                
        except Exception as e:
            logger.error(f"AD connection test failed: {e}")
            return False
        
        return False
    
    def get_all_users(self, attributes: List[str] = None) -> List[ADUser]:
        """Get all users from Active Directory."""
        try:
            if not self.is_connected:
                return []
            
            # Default attributes to retrieve
            if not attributes:
                attributes = [
                    'sAMAccountName', 'userPrincipalName', 'displayName', 'givenName', 'surname',
                    'mail', 'department', 'title', 'manager', 'memberOf', 'lastLogon',
                    'accountExpires', 'pwdLastSet', 'userAccountControl', 'distinguishedName',
                    'objectGUID', 'whenCreated', 'whenChanged'
                ]
            
            # Build search filter
            search_filter = '(&(objectClass=user)(objectCategory=person))'
            
            if LDAP3_AVAILABLE:
                return self._get_users_ldap3(search_filter, attributes)
            elif PYLDAP_AVAILABLE:
                return self._get_users_pyldap(search_filter, attributes)
                
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            return []
    
    def _get_users_ldap3(self, search_filter: str, attributes: List[str]) -> List[ADUser]:
        """Get users using ldap3."""
        users = []
        
        try:
            self.connection.search(
                search_base=self.connection_config.base_dn,
                search_filter=search_filter,
                attributes=attributes
            )
            
            for entry in self.connection.entries:
                user = self._parse_user_entry_ldap3(entry)
                if user:
                    users.append(user)
                    
        except Exception as e:
            logger.error(f"LDAP3 user search failed: {e}")
        
        return users
    
    def _get_users_pyldap(self, search_filter: str, attributes: List[str]) -> List[ADUser]:
        """Get users using python-ldap."""
        users = []
        
        try:
            results = self.connection.search_s(
                self.connection_config.base_dn,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attributes
            )
            
            for dn, attrs in results:
                if dn:  # Skip None entries
                    user = self._parse_user_entry_pyldap(dn, attrs)
                    if user:
                        users.append(user)
                        
        except Exception as e:
            logger.error(f"Python-LDAP user search failed: {e}")
        
        return users
    
    def _parse_user_entry_ldap3(self, entry) -> Optional[ADUser]:
        """Parse user entry from ldap3."""
        try:
            # Helper function to get attribute value
            def get_attr(attr_name, default=None):
                if hasattr(entry, attr_name):
                    value = getattr(entry, attr_name)
                    if value and len(value) > 0:
                        return value[0] if isinstance(value, list) else value
                return default
            
            # Parse datetime fields
            def parse_datetime(value):
                if value:
                    if isinstance(value, str):
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            return None
                    elif isinstance(value, datetime):
                        return value
                return None
            
            # Parse user account control
            uac = get_attr('userAccountControl', 0)
            if isinstance(uac, str):
                uac = int(uac)
            
            # Parse memberOf
            member_of = []
            if hasattr(entry, 'memberOf'):
                member_of = [str(m) for m in entry.memberOf]
            
            return ADUser(
                sam_account_name=get_attr('sAMAccountName', ''),
                user_principal_name=get_attr('userPrincipalName', ''),
                display_name=get_attr('displayName', ''),
                given_name=get_attr('givenName', ''),
                surname=get_attr('surname', ''),
                email=get_attr('mail', ''),
                department=get_attr('department', ''),
                title=get_attr('title', ''),
                manager=get_attr('manager', ''),
                member_of=member_of,
                last_logon=parse_datetime(get_attr('lastLogon')),
                account_expires=parse_datetime(get_attr('accountExpires')),
                password_last_set=parse_datetime(get_attr('pwdLastSet')),
                user_account_control=uac,
                distinguished_name=str(entry.distinguishedName),
                object_guid=str(entry.objectGUID),
                when_created=parse_datetime(get_attr('whenCreated')),
                when_changed=parse_datetime(get_attr('whenChanged'))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse user entry: {e}")
            return None
    
    def _parse_user_entry_pyldap(self, dn: str, attrs: Dict) -> Optional[ADUser]:
        """Parse user entry from python-ldap."""
        try:
            # Helper function to get attribute value
            def get_attr(attr_name, default=None):
                if attr_name in attrs:
                    value = attrs[attr_name]
                    if value and len(value) > 0:
                        return value[0] if isinstance(value, list) else value
                return default
            
            # Parse datetime fields
            def parse_datetime(value):
                if value:
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if isinstance(value, str):
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            return None
                return None
            
            # Parse user account control
            uac = get_attr('userAccountControl', 0)
            if isinstance(uac, bytes):
                uac = int(uac.decode('utf-8'))
            elif isinstance(uac, str):
                uac = int(uac)
            
            # Parse memberOf
            member_of = []
            if 'memberOf' in attrs:
                member_of = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in attrs['memberOf']]
            
            return ADUser(
                sam_account_name=get_attr('sAMAccountName', ''),
                user_principal_name=get_attr('userPrincipalName', ''),
                display_name=get_attr('displayName', ''),
                given_name=get_attr('givenName', ''),
                surname=get_attr('surname', ''),
                email=get_attr('mail', ''),
                department=get_attr('department', ''),
                title=get_attr('title', ''),
                manager=get_attr('manager', ''),
                member_of=member_of,
                last_logon=parse_datetime(get_attr('lastLogon')),
                account_expires=parse_datetime(get_attr('accountExpires')),
                password_last_set=parse_datetime(get_attr('pwdLastSet')),
                user_account_control=uac,
                distinguished_name=dn,
                object_guid=get_attr('objectGUID', ''),
                when_created=parse_datetime(get_attr('whenCreated')),
                when_changed=parse_datetime(get_attr('whenChanged'))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse user entry: {e}")
            return None
    
    def get_all_groups(self, attributes: List[str] = None) -> List[ADGroup]:
        """Get all groups from Active Directory."""
        try:
            if not self.is_connected:
                return []
            
            # Default attributes to retrieve
            if not attributes:
                attributes = [
                    'name', 'sAMAccountName', 'description', 'groupType', 'scope',
                    'member', 'managedBy', 'distinguishedName', 'objectGUID',
                    'whenCreated', 'whenChanged'
                ]
            
            # Build search filter
            search_filter = '(&(objectClass=group)(objectCategory=group))'
            
            if LDAP3_AVAILABLE:
                return self._get_groups_ldap3(search_filter, attributes)
            elif PYLDAP_AVAILABLE:
                return self._get_groups_pyldap(search_filter, attributes)
                
        except Exception as e:
            logger.error(f"Failed to get groups: {e}")
            return []
    
    def _get_groups_ldap3(self, search_filter: str, attributes: List[str]) -> List[ADGroup]:
        """Get groups using ldap3."""
        groups = []
        
        try:
            self.connection.search(
                search_base=self.connection_config.base_dn,
                search_filter=search_filter,
                attributes=attributes
            )
            
            for entry in self.connection.entries:
                group = self._parse_group_entry_ldap3(entry)
                if group:
                    groups.append(group)
                    
        except Exception as e:
            logger.error(f"LDAP3 group search failed: {e}")
        
        return groups
    
    def _get_groups_pyldap(self, search_filter: str, attributes: List[str]) -> List[ADGroup]:
        """Get groups using python-ldap."""
        groups = []
        
        try:
            results = self.connection.search_s(
                self.connection_config.base_dn,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attributes
            )
            
            for dn, attrs in results:
                if dn:  # Skip None entries
                    group = self._parse_group_entry_pyldap(dn, attrs)
                    if group:
                        groups.append(group)
                        
        except Exception as e:
            logger.error(f"Python-LDAP group search failed: {e}")
        
        return groups
    
    def _parse_group_entry_ldap3(self, entry) -> Optional[ADGroup]:
        """Parse group entry from ldap3."""
        try:
            # Helper function to get attribute value
            def get_attr(attr_name, default=None):
                if hasattr(entry, attr_name):
                    value = getattr(entry, attr_name)
                    if value and len(value) > 0:
                        return value[0] if isinstance(value, list) else value
                return default
            
            # Parse datetime fields
            def parse_datetime(value):
                if value:
                    if isinstance(value, str):
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            return None
                    elif isinstance(value, datetime):
                        return value
                return None
            
            # Parse group type and scope
            group_type = get_attr('groupType', 0)
            if isinstance(group_type, str):
                group_type = int(group_type)
            
            # Determine scope
            scope = "Global"
            if group_type & 0x00000002:  # ADS_GROUP_TYPE_DOMAIN_LOCAL_GROUP
                scope = "Domain Local"
            elif group_type & 0x00000004:  # ADS_GROUP_TYPE_UNIVERSAL_GROUP
                scope = "Universal"
            
            # Parse members
            members = []
            if hasattr(entry, 'member'):
                members = [str(m) for m in entry.member]
            
            return ADGroup(
                name=get_attr('name', ''),
                sam_account_name=get_attr('sAMAccountName', ''),
                description=get_attr('description', ''),
                group_type=group_type,
                scope=scope,
                member_count=len(members),
                members=members,
                managed_by=get_attr('managedBy', ''),
                distinguished_name=str(entry.distinguishedName),
                object_guid=str(entry.objectGUID),
                when_created=parse_datetime(get_attr('whenCreated')),
                when_changed=parse_datetime(get_attr('whenChanged'))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse group entry: {e}")
            return None
    
    def _parse_group_entry_pyldap(self, dn: str, attrs: Dict) -> Optional[ADGroup]:
        """Parse group entry from python-ldap."""
        try:
            # Helper function to get attribute value
            def get_attr(attr_name, default=None):
                if attr_name in attrs:
                    value = attrs[attr_name]
                    if value and len(value) > 0:
                        return value[0] if isinstance(value, list) else value
                return default
            
            # Parse datetime fields
            def parse_datetime(value):
                if value:
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if isinstance(value, str):
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            return None
                return None
            
            # Parse group type and scope
            group_type = get_attr('groupType', 0)
            if isinstance(group_type, bytes):
                group_type = int(group_type.decode('utf-8'))
            elif isinstance(group_type, str):
                group_type = int(group_type)
            
            # Determine scope
            scope = "Global"
            if group_type & 0x00000002:  # ADS_GROUP_TYPE_DOMAIN_LOCAL_GROUP
                scope = "Domain Local"
            elif group_type & 0x00000004:  # ADS_GROUP_TYPE_UNIVERSAL_GROUP
                scope = "Universal"
            
            # Parse members
            members = []
            if 'member' in attrs:
                members = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in attrs['member']]
            
            return ADGroup(
                name=get_attr('name', ''),
                sam_account_name=get_attr('sAMAccountName', ''),
                description=get_attr('description', ''),
                group_type=group_type,
                scope=scope,
                member_count=len(members),
                members=members,
                managed_by=get_attr('managedBy', ''),
                distinguished_name=dn,
                object_guid=get_attr('objectGUID', ''),
                when_created=parse_datetime(get_attr('whenCreated')),
                when_changed=parse_datetime(get_attr('whenChanged'))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse group entry: {e}")
            return None
    
    def get_group_members(self, group_name: str) -> List[ADUser]:
        """Get all members of a specific group."""
        try:
            if not self.is_connected:
                return []
            
            # First, find the group
            group_filter = f'(&(objectClass=group)(sAMAccountName={group_name}))'
            
            if LDAP3_AVAILABLE:
                self.connection.search(
                    search_base=self.connection_config.base_dn,
                    search_filter=group_filter,
                    attributes=['member']
                )
                
                if not self.connection.entries:
                    logger.warning(f"Group not found: {group_name}")
                    return []
                
                group_entry = self.connection.entries[0]
                member_dns = []
                
                if hasattr(group_entry, 'member'):
                    member_dns = [str(m) for m in group_entry.member]
                
            elif PYLDAP_AVAILABLE:
                results = self.connection.search_s(
                    self.connection_config.base_dn,
                    ldap.SCOPE_SUBTREE,
                    group_filter,
                    ['member']
                )
                
                if not results or not results[0][1]:
                    logger.warning(f"Group not found: {group_name}")
                    return []
                
                member_dns = []
                if 'member' in results[0][1]:
                    member_dns = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in results[0][1]['member']]
            
            # Now get user details for each member
            users = []
            for member_dn in member_dns:
                user = self._get_user_by_dn(member_dn)
                if user:
                    users.append(user)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to get group members: {e}")
            return []
    
    def _get_user_by_dn(self, user_dn: str) -> Optional[ADUser]:
        """Get user by distinguished name."""
        try:
            user_filter = f'(&(objectClass=user)(objectCategory=person)(distinguishedName={user_dn}))'
            attributes = [
                'sAMAccountName', 'userPrincipalName', 'displayName', 'givenName', 'surname',
                'mail', 'department', 'title', 'manager', 'memberOf', 'lastLogon',
                'accountExpires', 'pwdLastSet', 'userAccountControl', 'distinguishedName',
                'objectGUID', 'whenCreated', 'whenChanged'
            ]
            
            if LDAP3_AVAILABLE:
                self.connection.search(
                    search_base=user_dn,
                    search_filter='(objectClass=*)',
                    search_scope=ldap3.BASE,
                    attributes=attributes
                )
                
                if self.connection.entries:
                    return self._parse_user_entry_ldap3(self.connection.entries[0])
                    
            elif PYLDAP_AVAILABLE:
                results = self.connection.search_s(
                    user_dn,
                    ldap.SCOPE_BASE,
                    '(objectClass=*)',
                    attributes
                )
                
                if results and results[0][1]:
                    return self._parse_user_entry_pyldap(results[0][0], results[0][1])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by DN: {e}")
            return None
    
    def search_users(self, search_term: str, search_fields: List[str] = None) -> List[ADUser]:
        """Search for users with flexible criteria."""
        try:
            if not self.is_connected:
                return []
            
            if not search_fields:
                search_fields = ['sAMAccountName', 'displayName', 'givenName', 'surname', 'mail']
            
            # Build search filter
            search_parts = []
            for field in search_fields:
                search_parts.append(f'({field}=*{search_term}*)')
            
            search_filter = f'(&(objectClass=user)(objectCategory=person)(|{"".join(search_parts)}))'
            
            attributes = [
                'sAMAccountName', 'userPrincipalName', 'displayName', 'givenName', 'surname',
                'mail', 'department', 'title', 'manager', 'memberOf', 'lastLogon',
                'accountExpires', 'pwdLastSet', 'userAccountControl', 'distinguishedName',
                'objectGUID', 'whenCreated', 'whenChanged'
            ]
            
            if LDAP3_AVAILABLE:
                return self._get_users_ldap3(search_filter, attributes)
            elif PYLDAP_AVAILABLE:
                return self._get_users_pyldap(search_filter, attributes)
                
        except Exception as e:
            logger.error(f"Failed to search users: {e}")
            return []

class ActiveDirectoryManager:
    """Main Active Directory manager for handling connections and operations."""
    
    def __init__(self):
        self.connections: Dict[str, ActiveDirectoryConnector] = {}
        self.connection_configs: Dict[str, ADConnection] = {}
        
        # Initialize credential vault and memory if available
        self.credential_vault = None
        self.memory_manager = None
        
        if CREDENTIAL_VAULT_AVAILABLE:
            try:
                self.credential_vault = CredentialVault()
                logger.info("Credential vault initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize credential vault: {e}")
        
        if MEMORY_AVAILABLE:
            try:
                self.memory_manager = ContextMemoryManager()
                logger.info("Memory manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory manager: {e}")
    
    def create_connection(self, 
                         domain: str,
                         server: str,
                         port: int,
                         username: str,
                         password: str,
                         base_dn: str = None,
                         use_ssl: bool = True) -> Optional[str]:
        """Create a new Active Directory connection."""
        try:
            # Generate connection ID
            connection_id = self._generate_connection_id(domain, server, port)
            
            # Store credentials in vault if available
            if self.credential_vault and password:
                credential_key = f"ad_{connection_id}"
                self.credential_vault.add_credential(
                    credential_key,
                    username,
                    password,
                    f"Active Directory connection for {domain} at {server}:{port}"
                )
            
            # Set default base DN if not provided
            if not base_dn:
                base_dn = self._build_default_base_dn(domain)
            
            # Create connection config
            connection_config = ADConnection(
                connection_id=connection_id,
                domain=domain,
                server=server,
                port=port,
                username=username,
                base_dn=base_dn,
                use_ssl=use_ssl,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            # Store connection config
            self.connection_configs[connection_id] = connection_config
            
            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.store_in_memory(
                    'ACTIVE_DIRECTORY_CONNECTIONS',
                    connection_id,
                    connection_config.to_dict(),
                    'ad_connection',
                    ttl_hours=24 * 7  # 1 week
                )
            
            logger.info(f"Created AD connection config: {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to create AD connection: {e}")
            return None
    
    def connect_to_ad(self, connection_id: str) -> bool:
        """Establish connection to Active Directory."""
        try:
            if connection_id not in self.connection_configs:
                logger.error(f"Connection ID not found: {connection_id}")
                return False
            
            connection_config = self.connection_configs[connection_id]
            
            # Create connector
            connector = ActiveDirectoryConnector(connection_config)
            
            # Establish connection
            if connector.connect():
                self.connections[connection_id] = connector
                logger.info(f"Connected to AD: {connection_id}")
                return True
            else:
                logger.error(f"Failed to connect to AD: {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to AD: {e}")
            return False
    
    def disconnect_from_ad(self, connection_id: str) -> bool:
        """Disconnect from Active Directory."""
        try:
            if connection_id in self.connections:
                connector = self.connections[connection_id]
                if connector.disconnect():
                    del self.connections[connection_id]
                    logger.info(f"Disconnected from AD: {connection_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error disconnecting from AD: {e}")
            return False
    
    def get_all_users(self, connection_id: str) -> List[ADUser]:
        """Get all users from Active Directory."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.get_all_users()
            
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
    
    def get_all_groups(self, connection_id: str) -> List[ADGroup]:
        """Get all groups from Active Directory."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.get_all_groups()
            
        except Exception as e:
            logger.error(f"Error getting groups: {e}")
            return []
    
    def get_group_members(self, connection_id: str, group_name: str) -> List[ADUser]:
        """Get all members of a specific group."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.get_group_members(group_name)
            
        except Exception as e:
            logger.error(f"Error getting group members: {e}")
            return []
    
    def search_users(self, connection_id: str, search_term: str, search_fields: List[str] = None) -> List[ADUser]:
        """Search for users with flexible criteria."""
        try:
            if connection_id not in self.connections:
                logger.error(f"Connection not found: {connection_id}")
                return []
            
            connector = self.connections[connection_id]
            return connector.search_users(search_term, search_fields)
            
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return []
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all available AD connections."""
        connections = []
        for connection_id, config in self.connection_configs.items():
            connection_info = {
                'connection_id': connection_id,
                'domain': config.domain,
                'server': config.server,
                'port': config.port,
                'username': config.username,
                'is_connected': connection_id in self.connections,
                'created_at': config.created_at.isoformat(),
                'last_used': config.last_used.isoformat()
            }
            connections.append(connection_info)
        
        return connections
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection status."""
        if connection_id not in self.connection_configs:
            return None
        
        config = self.connection_configs[connection_id]
        connector = self.connections.get(connection_id)
        
        status = {
            'connection_id': connection_id,
            'domain': config.domain,
            'server': config.server,
            'port': config.port,
            'is_connected': connector is not None,
            'connection_working': connector.test_connection() if connector else False,
            'created_at': config.created_at.isoformat(),
            'last_used': config.last_used.isoformat()
        }
        
        return status
    
    def _generate_connection_id(self, domain: str, server: str, port: int) -> str:
        """Generate unique connection ID."""
        unique_string = f"ad_{domain}_{server}_{port}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _build_default_base_dn(self, domain: str) -> str:
        """Build default base DN from domain."""
        parts = domain.split('.')
        return ','.join([f'DC={part}' for part in parts])
    
    def cleanup(self):
        """Cleanup all connections."""
        for connection_id in list(self.connections.keys()):
            self.disconnect_from_ad(connection_id)

# Global instance
_ad_manager = None

def get_ad_manager() -> ActiveDirectoryManager:
    """Get or create the global AD manager instance."""
    global _ad_manager
    if _ad_manager is None:
        _ad_manager = ActiveDirectoryManager()
    return _ad_manager

if __name__ == "__main__":
    # Test the AD manager
    manager = get_ad_manager()
    
    print("🧪 Testing Active Directory Manager...")
    print(f"LDAP3 Available: {LDAP3_AVAILABLE}")
    print(f"Python-LDAP Available: {PYLDAP_AVAILABLE}")
    print(f"Credential Vault: {CREDENTIAL_VAULT_AVAILABLE}")
    print(f"Memory Manager: {MEMORY_AVAILABLE}")
    
    # Test connection creation
    connection_id = manager.create_connection(
        domain="example.com",
        server="dc01.example.com",
        port=389,
        username="admin@example.com",
                    password="test_password"
    )
    
    if connection_id:
        print(f"✅ Connection created: {connection_id}")
        
        # List connections
        connections = manager.list_connections()
        print(f"Connections: {connections}")
        
        # Get status
        status = manager.get_connection_status(connection_id)
        print(f"Status: {status}")
    else:
        print("❌ Failed to create connection")
