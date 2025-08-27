#!/usr/bin/env python3
"""
Environment Variable Migration System
Moves credentials from environment variables to secure credential vault.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class EnvironmentMigration:
    """Migrates environment variables to secure credential vault."""
    
    def __init__(self):
        self.credential_vault = None
        self._initialize_vault()
        
        # Define migration mappings
        self.migration_mappings = {
            'credentials': [
                'ENCRYPTION_PASSWORD_HASH',
                'DATABASE_PASSWORD',
                'API_PASSWORD',
                'SERVICE_PASSWORD'
            ],
            'api_keys': [
                'API_KEY',
                'OPENAI_API_KEY',
                'AZURE_API_KEY',
                'AWS_ACCESS_KEY_ID',
                'GOOGLE_API_KEY'
            ],
            'secrets': [
                'SECRET_KEY',
                'JWT_SECRET',
                'ENCRYPTION_KEY',
                'DATABASE_URL'
            ],
            'web_credentials': [
                'WEB_USERNAME',
                'WEB_PASSWORD',
                'ADMIN_USERNAME',
                'ADMIN_PASSWORD'
            ]
        }
    
    def _initialize_vault(self):
        """Initialize credential vault if available."""
        try:
            from bin.credential_vault import CredentialVault
            self.credential_vault = CredentialVault()
        except ImportError:
            print("⚠️  Credential vault not available")
    
    def scan_environment_variables(self) -> Dict[str, List[str]]:
        """Scan for environment variables that should be migrated."""
        found_variables = {
            'credentials': [],
            'api_keys': [],
            'secrets': [],
            'web_credentials': [],
            'other_sensitive': []
        }
        
        # Check all environment variables
        for key, value in os.environ.items():
            if not value or value.strip() == '':
                continue
            
            # Check if it's a known credential type
            categorized = False
            for category, patterns in self.migration_mappings.items():
                for pattern in patterns:
                    if pattern.lower() in key.lower():
                        found_variables[category].append(key)
                        categorized = True
                        break
                if categorized:
                    break
            
            # Check for other potentially sensitive variables
            if not categorized:
                sensitive_patterns = [
                    'password', 'secret', 'key', 'token', 'auth',
                    'credential', 'login', 'passwd', 'pwd'
                ]
                
                if any(pattern in key.lower() for pattern in sensitive_patterns):
                    found_variables['other_sensitive'].append(key)
        
        return found_variables
    
    def migrate_environment_variables(self, variables: Dict[str, List[str]], 
                                   interactive: bool = True) -> bool:
        """Migrate environment variables to credential vault."""
        if not self.credential_vault or not self.credential_vault.cipher:
            print("❌ Credential vault not available for migration")
            return False
        
        migrated_count = 0
        total_count = sum(len(vars) for vars in variables.values())
        
        print(f"🔄 Migrating {total_count} environment variables to credential vault...")
        
        for category, var_list in variables.items():
            if not var_list:
                continue
            
            print(f"\n📁 Category: {category.title()}")
            
            for var_name in var_list:
                var_value = os.environ.get(var_name, '')
                
                if not var_value:
                    continue
                
                # Determine migration action
                if interactive:
                    action = self._prompt_for_migration_action(var_name, var_value, category)
                else:
                    action = 'migrate'  # Default action
                
                if action == 'migrate':
                    if self._migrate_variable(var_name, var_value, category):
                        migrated_count += 1
                        print(f"   ✅ Migrated: {var_name}")
                    else:
                        print(f"   ❌ Failed to migrate: {var_name}")
                
                elif action == 'skip':
                    print(f"   ⏭️  Skipped: {var_name}")
                
                elif action == 'delete':
                    if self._delete_environment_variable(var_name):
                        print(f"   🗑️  Deleted: {var_name}")
                    else:
                        print(f"   ❌ Failed to delete: {var_name}")
        
        print(f"\n✅ Migration complete: {migrated_count}/{total_count} variables migrated")
        return migrated_count > 0
    
    def _prompt_for_migration_action(self, var_name: str, var_value: str, category: str) -> str:
        """Prompt user for migration action for a specific variable."""
        print(f"\n🔐 Variable: {var_name}")
        print(f"   Category: {category}")
        print(f"   Value: {var_value[:20]}{'...' if len(var_value) > 20 else ''}")
        
        while True:
            print("\n   Actions:")
            print("     m - Migrate to vault")
            print("     s - Skip (keep in environment)")
            print("     d - Delete from environment")
            print("     v - View full value")
            
            choice = input("   Select action (m/s/d/v): ").strip().lower()
            
            if choice == 'm':
                return 'migrate'
            elif choice == 's':
                return 'skip'
            elif choice == 'd':
                return 'delete'
            elif choice == 'v':
                print(f"   Full value: {var_value}")
                continue
            else:
                print("   Invalid choice. Please select m, s, d, or v.")
    
    def _migrate_variable(self, var_name: str, var_value: str, category: str) -> bool:
        """Migrate a single environment variable to the vault."""
        try:
            # Generate a descriptive name
            name = self._generate_variable_name(var_name)
            description = f"Migrated from environment variable: {var_name}"
            
            if category == 'credentials':
                # Extract username if possible
                username = self._extract_username_from_varname(var_name)
                return self.credential_vault.add_credential(name, username, var_value, description)
            
            elif category == 'api_keys':
                return self.credential_vault.add_api_key(name, var_value, description)
            
            elif category == 'secrets':
                return self.credential_vault.add_secret(name, var_value, description)
            
            elif category == 'web_credentials':
                username = self._extract_username_from_varname(var_name)
                url = "https://example.com"  # Default URL
                return self.credential_vault.add_web_credential(name, url, username, var_value, description)
            
            else:
                # Default to secret
                return self.credential_vault.add_secret(name, var_value, description)
                
        except Exception as e:
            print(f"   Error migrating {var_name}: {e}")
            return False
    
    def _delete_environment_variable(self, var_name: str) -> bool:
        """Delete an environment variable (user confirmation required)."""
        try:
            # Note: We can't actually delete environment variables from the parent process
            # But we can show instructions
            print(f"   📝 To delete {var_name}, remove it from your shell profile:")
            print(f"      - ~/.bashrc, ~/.zshrc, ~/.profile, or ~/.bash_profile")
            print(f"      - Or unset it: unset {var_name}")
            return True
            
        except Exception as e:
            print(f"   Error with {var_name}: {e}")
            return False
    
    def _generate_variable_name(self, var_name: str) -> str:
        """Generate a human-readable name from environment variable name."""
        # Remove common prefixes
        name = var_name
        prefixes = ['ENCRYPTION_', 'DATABASE_', 'API_', 'WEB_', 'SERVICE_']
        
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Convert to title case
        name = name.replace('_', ' ').title()
        
        return name
    
    def _extract_username_from_varname(self, var_name: str) -> str:
        """Extract username from environment variable name if possible."""
        var_lower = var_name.lower()
        
        if 'username' in var_lower:
            return "username"
        elif 'user' in var_lower:
            return "user"
        elif 'admin' in var_lower:
            return "admin"
        else:
            return "user"
    
    def generate_migration_report(self, variables: Dict[str, List[str]]) -> str:
        """Generate a migration report."""
        report = []
        report.append("# Environment Variable Migration Report")
        report.append("")
        report.append(f"Generated: {os.popen('date').read().strip()}")
        report.append("")
        
        total_count = sum(len(vars) for vars in variables.values())
        report.append(f"Total variables found: {total_count}")
        report.append("")
        
        for category, var_list in variables.items():
            if not var_list:
                continue
            
            report.append(f"## {category.title()}")
            report.append("")
            
            for var_name in var_list:
                var_value = os.environ.get(var_name, '')
                masked_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
                report.append(f"- **{var_name}**: `{masked_value}`")
            
            report.append("")
        
        return "\n".join(report)
    
    def create_migration_script(self, variables: Dict[str, List[str]], 
                               output_path: str = "migrate_credentials.sh") -> bool:
        """Create a shell script for credential migration."""
        try:
            script_content = [
                "#!/bin/bash",
                "# Credential Migration Script",
                "# Generated automatically - review before running",
                "",
                "echo '🔐 Credential Migration Script'",
                "echo '============================='",
                "",
                "echo 'This script will help you migrate environment variables to the credential vault.'",
                "echo 'Review the script before running it.'",
                "echo ''",
                ""
            ]
            
            # Add migration commands
            for category, var_list in variables.items():
                if not var_list:
                    continue
                
                script_content.append(f"echo '📁 Category: {category.title()}'")
                
                for var_name in var_list:
                    script_content.append(f"echo '   Variable: {var_name}'")
                    script_content.append(f"echo '   Current value: ${{{var_name}:-[NOT SET]}}'")
                    script_content.append("")
            
            script_content.extend([
                "echo 'Migration complete!'",
                "echo 'Remember to:'",
                "echo '  1. Remove credentials from environment variables'",
                "echo '  2. Update your shell profiles'",
                "echo '  3. Test the system with vault-based credentials'",
                ""
            ])
            
            # Write script
            with open(output_path, 'w') as f:
                f.write('\n'.join(script_content))
            
            # Make executable
            os.chmod(output_path, 0o755)
            
            print(f"✅ Migration script created: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create migration script: {e}")
            return False

def main():
    """Test environment variable migration system."""
    print("🔄 Environment Variable Migration System")
    print("=" * 50)
    
    migration = EnvironmentMigration()
    
    if not migration.credential_vault:
        print("❌ Credential vault not available")
        print("   Set ENCRYPTION_ENABLED=true and ENCRYPTION_PASSWORD_HASH")
        return
    
    # Scan for variables
    print("🔍 Scanning environment variables...")
    variables = migration.scan_environment_variables()
    
    # Show found variables
    total_count = sum(len(vars) for vars in variables.values())
    if total_count == 0:
        print("✅ No sensitive environment variables found")
        return
    
    print(f"\n📊 Found {total_count} potentially sensitive variables:")
    for category, var_list in variables.items():
        if var_list:
            print(f"   {category.title()}: {len(var_list)}")
            for var_name in var_list[:3]:  # Show first 3
                print(f"     - {var_name}")
            if len(var_list) > 3:
                print(f"     ... and {len(var_list) - 3} more")
    
    # Generate report
    report = migration.generate_migration_report(variables)
    with open("migration_report.md", "w") as f:
        f.write(report)
    print("\n📝 Migration report saved to: migration_report.md")
    
    # Create migration script
    migration.create_migration_script(variables)
    
    # Ask if user wants to migrate
    if input("\n🔄 Start migration now? (y/N): ").strip().lower() == 'y':
        migration.migrate_environment_variables(variables)
    else:
        print("Migration skipped. Use the generated script when ready.")

if __name__ == "__main__":
    main()

