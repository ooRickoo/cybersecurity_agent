#!/usr/bin/env python3
"""
MITRE ATT&CK Framework Mapping Workflow

Automatically maps security policies, controls, and content to MITRE ATT&CK framework:
- Analyzes input content (CSV, text, policy documents)
- Maps to relevant MITRE ATT&CK tactics and techniques
- Provides reasoning for each mapping
- Enriches output with recommendations
- Supports batch processing of large catalogs
"""

import asyncio
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MitreMapping:
    """MITRE ATT&CK mapping result."""
    tactic_id: str
    tactic_name: str
    technique_id: str
    technique_name: str
    confidence_score: float
    reasoning: str
    recommendations: List[str]
    mitre_url: str

@dataclass
class MappingResult:
    """Result of MITRE mapping operation."""
    success: bool
    total_items: int
    mapped_items: int
    mapping_time: float
    output_file: str
    mappings: List[MitreMapping]
    error_message: Optional[str] = None

class MitreAttackFramework:
    """MITRE ATT&CK framework data and utilities."""
    
    def __init__(self):
        self.tactics = self._initialize_tactics()
        self.techniques = self._initialize_techniques()
        self.subtechniques = self._initialize_subtechniques()
        self.mappings = self._initialize_mappings()
    
    def _initialize_tactics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize MITRE ATT&CK tactics."""
        return {
            "TA0001": {
                "name": "Initial Access",
                "description": "The adversary is trying to get into your network.",
                "keywords": ["initial access", "entry point", "breach", "compromise", "infiltration"]
            },
            "TA0002": {
                "name": "Execution",
                "description": "The adversary is trying to run malicious code.",
                "keywords": ["execution", "code execution", "malware", "script", "command"]
            },
            "TA0003": {
                "name": "Persistence",
                "description": "The adversary is trying to maintain their foothold.",
                "keywords": ["persistence", "maintain access", "survive reboot", "backdoor"]
            },
            "TA0004": {
                "name": "Privilege Escalation",
                "description": "The adversary is trying to gain higher-level permissions.",
                "keywords": ["privilege escalation", "elevate", "admin", "root", "escalation"]
            },
            "TA0005": {
                "name": "Defense Evasion",
                "description": "The adversary is trying to avoid being detected.",
                "keywords": ["defense evasion", "evade detection", "bypass", "stealth", "hide"]
            },
            "TA0006": {
                "name": "Credential Access",
                "description": "The adversary is trying to steal account names and passwords.",
                "keywords": ["credential access", "password", "credential", "authentication", "login"]
            },
            "TA0007": {
                "name": "Discovery",
                "description": "The adversary is trying to figure out your environment.",
                "keywords": ["discovery", "reconnaissance", "enumeration", "mapping", "scanning"]
            },
            "TA0008": {
                "name": "Lateral Movement",
                "description": "The adversary is trying to move through your environment.",
                "keywords": ["lateral movement", "pivot", "move laterally", "network traversal"]
            },
            "TA0009": {
                "name": "Collection",
                "description": "The adversary is trying to gather data of interest to their goal.",
                "keywords": ["collection", "data theft", "exfiltration", "gather", "harvest"]
            },
            "TA0010": {
                "name": "Command and Control",
                "description": "The adversary is trying to communicate with compromised systems to control them.",
                "keywords": ["command and control", "c2", "communication", "control", "remote access"]
            },
            "TA0011": {
                "name": "Exfiltration",
                "description": "The adversary is trying to steal data.",
                "keywords": ["exfiltration", "data theft", "steal", "extract", "remove"]
            },
            "TA0040": {
                "name": "Impact",
                "description": "The adversary is trying to manipulate, interrupt, or destroy your systems and data.",
                "keywords": ["impact", "destruction", "manipulation", "interruption", "damage"]
            }
        }
    
    def _initialize_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize MITRE ATT&CK techniques."""
        return {
            "T1078": {
                "name": "Valid Accounts",
                "tactic": "TA0001",
                "description": "Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
                "keywords": ["valid accounts", "credentials", "authentication", "login", "user account"]
            },
            "T1071": {
                "name": "Application Layer Protocol",
                "tactic": "TA0010",
                "description": "Adversaries may communicate using application layer protocols to avoid detection/network filtering by blending in with existing traffic.",
                "keywords": ["application layer", "protocol", "communication", "network", "traffic"]
            },
            "T1055": {
                "name": "Process Injection",
                "tactic": "TA0002",
                "description": "Adversaries may inject code into processes in order to evade process-based defenses as well as to elevate privileges.",
                "keywords": ["process injection", "code injection", "memory injection", "dll injection"]
            },
            "T1021": {
                "name": "Remote Services",
                "tactic": "TA0008",
                "description": "Adversaries may use Valid Accounts to log into a service specifically designed to accept remote connections.",
                "keywords": ["remote services", "remote access", "ssh", "rdp", "telnet"]
            },
            "T1005": {
                "name": "Data from Local System",
                "tactic": "TA0009",
                "description": "Adversaries may search local system sources, such as file systems and configuration files or local databases, to find files of interest and sensitive data prior to Exfiltration.",
                "keywords": ["local system", "file system", "local data", "configuration", "database"]
            },
            "T1041": {
                "name": "Exfiltration Over C2 Channel",
                "tactic": "TA0011",
                "description": "Adversaries may steal data by exfiltrating it over an existing command and control channel.",
                "keywords": ["exfiltration", "c2 channel", "data theft", "command control", "steal"]
            },
            "T1565": {
                "name": "Data Manipulation",
                "tactic": "TA0040",
                "description": "Adversaries may insert, delete, or manipulate data in order to influence external outcomes or hide activity.",
                "keywords": ["data manipulation", "insert", "delete", "modify", "corrupt"]
            }
        }
    
    def _initialize_subtechniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize MITRE ATT&CK subtechniques."""
        return {
            "T1078.001": {
                "name": "Default Accounts",
                "technique": "T1078",
                "description": "Adversaries may obtain and abuse credentials of default accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
                "keywords": ["default accounts", "default credentials", "factory settings", "out of box"]
            },
            "T1078.002": {
                "name": "Domain Accounts",
                "technique": "T1078",
                "description": "Adversaries may obtain and abuse credentials of domain accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
                "keywords": ["domain accounts", "active directory", "domain user", "enterprise account"]
            }
        }
    
    def _initialize_mappings(self) -> Dict[str, List[str]]:
        """Initialize keyword to technique mappings."""
        return {
            "authentication": ["T1078", "T1078.001", "T1078.002"],
            "password": ["T1078", "T1078.001", "T1078.002"],
            "login": ["T1078", "T1078.001", "T1078.002"],
            "remote access": ["T1021"],
            "ssh": ["T1021"],
            "rdp": ["T1021"],
            "network": ["T1071", "T1021"],
            "communication": ["T1071", "T1021"],
            "injection": ["T1055"],
            "process": ["T1055"],
            "memory": ["T1055"],
            "file system": ["T1005"],
            "local data": ["T1005"],
            "configuration": ["T1005"],
            "exfiltration": ["T1041", "T1005"],
            "data theft": ["T1041", "T1005"],
            "manipulation": ["T1565"],
            "corruption": ["T1565"]
        }
    
    def get_tactic_info(self, tactic_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tactic."""
        return self.tactics.get(tactic_id)
    
    def get_technique_info(self, technique_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific technique."""
        return self.techniques.get(technique_id)
    
    def get_subtechnique_info(self, subtechnique_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific subtechnique."""
        return self.subtechniques.get(subtechnique_id)
    
    def search_techniques(self, query: str) -> List[Dict[str, Any]]:
        """Search for techniques based on query."""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Search in techniques
        for technique_id, technique_info in self.techniques.items():
            score = 0.0
            technique_name_lower = technique_info["name"].lower()
            technique_desc_lower = technique_info["description"].lower()
            technique_keywords = [kw.lower() for kw in technique_info.get("keywords", [])]
            
            # Check for word matches in name
            name_words = set(technique_name_lower.split())
            name_matches = query_words.intersection(name_words)
            if name_matches:
                score += len(name_matches) * 0.3
            
            # Check for word matches in description
            desc_words = set(technique_desc_lower.split())
            desc_matches = query_words.intersection(desc_words)
            if desc_matches:
                score += len(desc_matches) * 0.2
            
            # Check for keyword matches
            keyword_matches = 0
            for keyword in technique_keywords:
                if keyword in query_lower:
                    keyword_matches += 1
            score += keyword_matches * 0.4
            
            # Check for policy-specific term matches
            policy_terms = {
                'authentication': ['password', 'login', 'credential', 'mfa', '2fa', 'authentication'],
                'remote_access': ['vpn', 'ssh', 'rdp', 'remote', 'access'],
                'encryption': ['encrypt', 'cipher', 'key', 'secure', 'encryption'],
                'incident_response': ['incident', 'response', 'detection', 'containment'],
                'network_security': ['network', 'firewall', 'segmentation', 'traffic'],
                'application_security': ['application', 'code', 'injection', 'validation'],
                'endpoint_security': ['endpoint', 'edr', 'antivirus', 'patch'],
                'backup': ['backup', 'recovery', 'disaster', 'restore'],
                'training': ['training', 'awareness', 'phishing', 'social engineering'],
                'vendor': ['vendor', 'third-party', 'assessment', 'compliance'],
                'logging': ['log', 'monitoring', 'audit', 'detection'],
                'change_management': ['change', 'management', 'review', 'testing'],
                'physical_security': ['physical', 'badge', 'camera', 'access control'],
                'compliance': ['compliance', 'audit', 'regulation', 'standard'],
                'threat_intelligence': ['threat', 'intelligence', 'vulnerability', 'monitoring']
            }
            
            for policy_type, terms in policy_terms.items():
                if any(term in query_lower for term in terms):
                    score += 0.2
            
            # Add to results if score is above threshold
            if score >= 0.3:  # Lower threshold for more inclusive search
                results.append({
                    "id": technique_id,
                    "type": "technique",
                    "relevance_score": score,
                    **technique_info
                })
        
        # Search in subtechniques
        for subtechnique_id, subtechnique_info in self.subtechniques.items():
            score = 0.0
            subtechnique_name_lower = subtechnique_info["name"].lower()
            subtechnique_desc_lower = subtechnique_info["description"].lower()
            subtechnique_keywords = [kw.lower() for kw in subtechnique_info.get("keywords", [])]
            
            # Check for word matches in name
            name_words = set(subtechnique_name_lower.split())
            name_matches = query_words.intersection(name_words)
            if name_matches:
                score += len(name_matches) * 0.3
            
            # Check for word matches in description
            desc_words = set(subtechnique_desc_lower.split())
            desc_matches = query_words.intersection(desc_words)
            if desc_matches:
                score += len(desc_matches) * 0.2
            
            # Check for keyword matches
            keyword_matches = 0
            for keyword in subtechnique_keywords:
                if keyword in query_lower:
                    keyword_matches += 1
            score += keyword_matches * 0.4
            
            # Add to results if score is above threshold
            if score >= 0.3:  # Lower threshold for more inclusive search
                results.append({
                    "id": subtechnique_id,
                    "type": "subtechnique",
                    "relevance_score": score,
                    **subtechnique_info
                })
        
        # Sort results by relevance score
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return results
    
    def get_mitre_url(self, technique_id: str) -> str:
        """Generate MITRE ATT&CK URL for a technique."""
        base_url = "https://attack.mitre.org/techniques/"
        return f"{base_url}{technique_id.replace('.', '/')}"

class MitreAttackMappingWorkflow:
    """Executes MITRE ATT&CK mapping workflow."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.mitre_framework = MitreAttackFramework()
        self.dataframe = None
        self.mappings = []
    
    async def execute_mapping_workflow(self, input_file: str, output_file: str, 
                                     content_column: str = None, 
                                     batch_size: int = 100) -> MappingResult:
        """Execute the complete MITRE ATT&CK mapping workflow."""
        try:
            start_time = time.time()
            
            logger.info(f"🚀 Starting MITRE ATT&CK mapping workflow")
            logger.info(f"   Input: {input_file}")
            logger.info(f"   Output: {output_file}")
            logger.info(f"   Content column: {content_column}")
            logger.info(f"   Batch size: {batch_size}")
            
            # Step 1: Import and analyze data
            logger.info("📥 Step 1: Importing and analyzing data...")
            import_result = await self._import_and_analyze_data(input_file, content_column)
            if not import_result['success']:
                return MappingResult(
                    success=False,
                    total_items=0,
                    mapped_items=0,
                    mapping_time=0,
                    output_file="",
                    mappings=[],
                    error_message=import_result['error']
                )
            
            # Step 2: Perform MITRE mapping
            logger.info("🔍 Step 2: Performing MITRE ATT&CK mapping...")
            mapping_result = await self._perform_mitre_mapping(batch_size)
            
            # Step 3: Enrich with recommendations
            logger.info("💡 Step 3: Enriching with recommendations...")
            enrichment_result = await self._enrich_with_recommendations()
            
            # Step 4: Export results
            logger.info("💾 Step 4: Exporting results...")
            export_result = await self._export_results(output_file)
            
            mapping_time = time.time() - start_time
            
            result = MappingResult(
                success=True,
                total_items=len(self.dataframe),
                mapped_items=len(self.mappings),
                mapping_time=mapping_time,
                output_file=output_file,
                mappings=self.mappings
            )
            
            logger.info(f"✅ MITRE ATT&CK mapping completed successfully!")
            logger.info(f"   Total items: {result.total_items}")
            logger.info(f"   Mapped items: {result.mapped_items}")
            logger.info(f"   Mapping time: {mapping_time:.2f} seconds")
            logger.info(f"   Output file: {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in MITRE mapping workflow: {e}")
            return MappingResult(
                success=False,
                total_items=0,
                mapped_items=0,
                mapping_time=0,
                output_file="",
                mappings=[],
                error_message=str(e)
            )
    
    async def _import_and_analyze_data(self, input_file: str, content_column: str = None) -> Dict[str, Any]:
        """Import and analyze input data."""
        try:
            # Import data
            if input_file.endswith('.csv'):
                self.dataframe = pd.read_csv(input_file)
            elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
                self.dataframe = pd.read_excel(input_file)
            elif input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
                self.dataframe = pd.DataFrame(data)
            else:
                # Try to read as text file
                with open(input_file, 'r') as f:
                    lines = f.readlines()
                self.dataframe = pd.DataFrame({'content': lines})
            
            # Auto-detect content column if not specified
            if content_column is None:
                content_column = self._auto_detect_content_column()
            
            # Store content column for later use
            self.content_column = content_column
            
            # Debug logging for content column detection
            logger.info(f"🔍 Content column detection:")
            logger.info(f"   Detected column: {content_column}")
            logger.info(f"   Available columns: {list(self.dataframe.columns)}")
            logger.info(f"   Sample content from detected column: {self.dataframe[content_column].iloc[0][:100]}...")
            
            # Validate content column
            if content_column not in self.dataframe.columns:
                return {
                    'success': False,
                    'error': f"Content column '{content_column}' not found. Available columns: {list(self.dataframe.columns)}"
                }
            
            # Add MITRE mapping columns
            self.dataframe['mitre_tactic_id'] = None
            self.dataframe['mitre_tactic_name'] = None
            self.dataframe['mitre_technique_id'] = None
            self.dataframe['mitre_technique_name'] = None
            self.dataframe['mitre_confidence_score'] = None
            self.dataframe['mitre_reasoning'] = None
            self.dataframe['mitre_recommendations'] = None
            self.dataframe['mitre_url'] = None
            
            return {
                'success': True,
                'total_rows': len(self.dataframe),
                'content_column': content_column,
                'columns': list(self.dataframe.columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error importing data: {e}"
            }
    
    def _auto_detect_content_column(self) -> str:
        """Auto-detect the content column for analysis."""
        # Look for common content column names with priority order
        content_keywords_high = ['description', 'content', 'text', 'details', 'summary']
        content_keywords_medium = ['policy', 'control', 'requirement', 'item', 'name']
        content_keywords_low = ['id', 'code', 'number']
        
        # First, look for high-priority content columns
        for col in self.dataframe.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in content_keywords_high):
                logger.info(f"Found high-priority content column: {col}")
                return col
        
        # Then, look for medium-priority content columns
        for col in self.dataframe.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in content_keywords_medium):
                # Skip if it's a low-priority column (like policy_id)
                if not any(low_keyword in col_lower for low_keyword in content_keywords_low):
                    logger.info(f"Found medium-priority content column: {col}")
                    return col
        
        # If no obvious content column, find the column with the longest average text
        text_columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype == 'object']
        if text_columns:
            # Calculate average length for each text column
            column_lengths = {}
            for col in text_columns:
                avg_length = self.dataframe[col].astype(str).str.len().mean()
                column_lengths[col] = avg_length
            
            # Return the column with the longest average text
            longest_col = max(column_lengths, key=column_lengths.get)
            logger.info(f"Auto-detected content column by length: {longest_col} (avg length: {column_lengths[longest_col]:.1f})")
            return longest_col
        
        # Fallback to first column
        logger.info(f"Using fallback column: {self.dataframe.columns[0]}")
        return self.dataframe.columns[0]
    
    def _get_content_for_analysis(self, row: pd.Series, content_column: str) -> str:
        """Get content for analysis from a row."""
        if content_column in row:
            return str(row[content_column])
        
        # Fallback: try to find the most descriptive column
        text_columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype == 'object']
        if text_columns:
            # Use the longest text column
            longest_col = max(text_columns, key=lambda col: len(str(row.get(col, ''))))
            return str(row.get(longest_col, ''))
        
        # Last resort: concatenate all text columns
        text_content = []
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                text_content.append(str(row[col]))
        
        return ' '.join(text_content)
    
    async def _perform_mitre_mapping(self, batch_size: int) -> Dict[str, Any]:
        """Perform MITRE ATT&CK mapping on the data."""
        try:
            total_rows = len(self.dataframe)
            mapped_count = 0
            
            # Process in batches
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = self.dataframe.iloc[batch_start:batch_end]
                
                logger.info(f"📦 Processing batch {batch_start//batch_size + 1}: rows {batch_start+1}-{batch_end}")
                
                # Process each row in the batch
                for idx, row in batch_df.iterrows():
                    try:
                        # Get content for analysis using the detected content column
                        content = self._get_content_for_analysis(row, self.content_column)
                        
                        # Debug logging
                        if idx < 3:  # Log first 3 rows for debugging
                            logger.info(f"Row {idx} content: {content[:100]}...")
                        
                        # Perform MITRE mapping
                        mapping = await self._map_content_to_mitre(content)
                        
                        if mapping:
                            # Update DataFrame
                            self.dataframe.at[idx, 'mitre_tactic_id'] = mapping.tactic_id
                            self.dataframe.at[idx, 'mitre_tactic_name'] = mapping.tactic_name
                            self.dataframe.at[idx, 'mitre_technique_id'] = mapping.technique_id
                            self.dataframe.at[idx, 'mitre_technique_name'] = mapping.technique_name
                            self.dataframe.at[idx, 'mitre_confidence_score'] = mapping.confidence_score
                            self.dataframe.at[idx, 'mitre_reasoning'] = mapping.reasoning
                            self.dataframe.at[idx, 'mitre_recommendations'] = '; '.join(mapping.recommendations)
                            self.dataframe.at[idx, 'mitre_url'] = mapping.mitre_url
                            
                            self.mappings.append(mapping)
                            mapped_count += 1
                        
                        # Update progress
                        progress = (idx + 1) / total_rows * 100
                        if (idx + 1) % 10 == 0:  # Log every 10 rows
                            logger.info(f"📊 Progress: {progress:.1f}% - {mapped_count} items mapped")
                    
                    except Exception as e:
                        logger.warning(f"Failed to map row {idx}: {e}")
                        continue
            
            return {
                'success': True,
                'total_processed': total_rows,
                'mapped_count': mapped_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error in MITRE mapping: {e}"
            }
    
    async def _map_content_to_mitre(self, content: str) -> Optional[MitreMapping]:
        """Map content to MITRE ATT&CK framework."""
        try:
            # Clean content
            content_clean = self._clean_content(content)
            
            # Debug logging
            logger.info(f"Searching for techniques in content: {content_clean[:100]}...")
            
            # Search for relevant techniques
            search_results = self.mitre_framework.search_techniques(content_clean)
            
            logger.info(f"Found {len(search_results)} potential techniques")
            
            if not search_results:
                return None
            
            # Select best match (highest relevance)
            best_match = self._select_best_mitre_match(content_clean, search_results)
            
            if not best_match:
                return None
            
            # Generate reasoning
            reasoning = self._generate_mapping_reasoning(content_clean, best_match)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(best_match)
            
            # Get tactic info
            tactic_info = self.mitre_framework.get_tactic_info(best_match['tactic'])
            
            # Create mapping
            mapping = MitreMapping(
                tactic_id=best_match['tactic'],
                tactic_name=tactic_info['name'] if tactic_info else "Unknown",
                technique_id=best_match['id'],
                technique_name=best_match['name'],
                confidence_score=best_match.get('relevance_score', 0.7),
                reasoning=reasoning,
                recommendations=recommendations,
                mitre_url=self.mitre_framework.get_mitre_url(best_match['id'])
            )
            
            return mapping
            
        except Exception as e:
            logger.warning(f"Error mapping content to MITRE: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for analysis."""
        # Remove special characters and normalize whitespace
        content = re.sub(r'[^\w\s]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip().lower()
    
    def _select_best_mitre_match(self, content: str, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best MITRE technique match for the content."""
        if not search_results:
            return None
        
        # Score each result based on relevance
        scored_results = []
        for result in search_results:
            score = self._calculate_relevance_score(content, result)
            scored_results.append({**result, 'relevance_score': score})
        
        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Return best match if score is above threshold
        best_match = scored_results[0]
        if best_match['relevance_score'] > 0.2:  # Lower threshold for more inclusive mapping
            return best_match
        
        return None
    
    def _calculate_relevance_score(self, content: str, technique: Dict[str, Any]) -> float:
        """Calculate relevance score between content and technique."""
        score = 0.0
        content_lower = content.lower()
        
        # Check name relevance
        technique_name_lower = technique['name'].lower()
        if any(word in content_lower for word in technique_name_lower.split()):
            score += 0.4
        
        # Check description relevance
        technique_desc_lower = technique['description'].lower()
        content_words = set(content_lower.split())
        desc_words = set(technique_desc_lower.split())
        common_words = content_words.intersection(desc_words)
        if common_words:
            score += min(len(common_words) * 0.1, 0.3)
        
        # Check keyword relevance
        for keyword in technique.get('keywords', []):
            if keyword.lower() in content_lower:
                score += 0.2
        
        # Additional scoring for policy-specific terms
        policy_keywords = {
            'authentication': ['password', 'login', 'credential', 'mfa', '2fa'],
            'remote_access': ['vpn', 'ssh', 'rdp', 'remote', 'access'],
            'encryption': ['encrypt', 'cipher', 'key', 'secure'],
            'incident_response': ['incident', 'response', 'detection', 'containment'],
            'network_security': ['network', 'firewall', 'segmentation', 'traffic'],
            'application_security': ['application', 'code', 'injection', 'validation'],
            'endpoint_security': ['endpoint', 'edr', 'antivirus', 'patch'],
            'backup': ['backup', 'recovery', 'disaster', 'restore'],
            'training': ['training', 'awareness', 'phishing', 'social engineering'],
            'vendor': ['vendor', 'third-party', 'assessment', 'compliance'],
            'logging': ['log', 'monitoring', 'audit', 'detection'],
            'change_management': ['change', 'management', 'review', 'testing'],
            'physical_security': ['physical', 'badge', 'camera', 'access control'],
            'compliance': ['compliance', 'audit', 'regulation', 'standard'],
            'threat_intelligence': ['threat', 'intelligence', 'vulnerability', 'monitoring']
        }
        
        # Check for policy-specific keyword matches
        for policy_type, keywords in policy_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_mapping_reasoning(self, content: str, technique: Dict[str, Any]) -> str:
        """Generate reasoning for the MITRE mapping."""
        # Find matching keywords
        matching_keywords = []
        for keyword in technique.get('keywords', []):
            if keyword.lower() in content:
                matching_keywords.append(keyword)
        
        if matching_keywords:
            return f"Content matches {technique['name']} (T{technique['id']}) based on keywords: {', '.join(matching_keywords)}"
        else:
            return f"Content semantically relates to {technique['name']} (T{technique['id']}) based on content analysis"
    
    def _generate_recommendations(self, technique: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on the technique."""
        recommendations = []
        
        # Technique-specific recommendations
        if "T1078" in technique['id']:  # Valid Accounts
            recommendations.extend([
                "Implement strong password policies",
                "Enable multi-factor authentication",
                "Regular account access reviews",
                "Monitor for unusual login patterns"
            ])
        elif "T1021" in technique['id']:  # Remote Services
            recommendations.extend([
                "Restrict remote access to authorized users only",
                "Use VPN for remote connections",
                "Implement network segmentation",
                "Monitor remote access logs"
            ])
        elif "T1055" in technique['id']:  # Process Injection
            recommendations.extend([
                "Enable process monitoring",
                "Implement application whitelisting",
                "Monitor for unusual process behavior",
                "Use EDR solutions"
            ])
        else:
            # Generic recommendations
            recommendations.extend([
                "Implement defense in depth",
                "Regular security assessments",
                "Employee security awareness training",
                "Incident response planning"
            ])
        
        return recommendations
    
    async def _enrich_with_recommendations(self) -> Dict[str, Any]:
        """Enrich the data with additional recommendations and insights."""
        try:
            # This could include additional AI-powered analysis
            # For now, we'll add some basic enrichment
            
            # Add overall risk assessment
            self.dataframe['risk_level'] = self.dataframe['mitre_confidence_score'].apply(
                lambda x: 'High' if x and x > 0.8 else 'Medium' if x and x > 0.5 else 'Low' if x else 'Unknown'
            )
            
            # Add priority score
            self.dataframe['priority_score'] = self.dataframe['mitre_confidence_score'].apply(
                lambda x: round(x * 10, 1) if x else 0
            )
            
            return {
                'success': True,
                'enrichments_added': ['risk_level', 'priority_score']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error in enrichment: {e}"
            }
    
    async def _export_results(self, output_file: str) -> Dict[str, Any]:
        """Export the enriched results to output file."""
        try:
            # Export to CSV
            self.dataframe.to_csv(output_file, index=False)
            
            # Also export a summary report
            summary_file = output_file.replace('.csv', '_summary.txt')
            await self._export_summary_report(summary_file)
            
            return {
                'success': True,
                'output_file': output_file,
                'summary_file': summary_file,
                'rows_exported': len(self.dataframe)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error exporting results: {e}"
            }
    
    async def _export_summary_report(self, summary_file: str) -> None:
        """Export a summary report of the mapping results."""
        try:
            with open(summary_file, 'w') as f:
                f.write("MITRE ATT&CK Framework Mapping Summary Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Items Processed: {len(self.dataframe)}\n")
                f.write(f"Successfully Mapped: {len(self.mappings)}\n")
                f.write(f"Mapping Success Rate: {len(self.mappings)/len(self.dataframe)*100:.1f}%\n\n")
                
                # Tactic breakdown
                tactic_counts = {}
                for mapping in self.mappings:
                    tactic = mapping.tactic_name
                    tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
                
                f.write("Tactic Distribution:\n")
                f.write("-" * 20 + "\n")
                for tactic, count in sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{tactic}: {count} items\n")
                
                f.write(f"\nTop Techniques:\n")
                f.write("-" * 20 + "\n")
                technique_counts = {}
                for mapping in self.mappings:
                    technique = f"{mapping.technique_name} ({mapping.technique_id})"
                    technique_counts[technique] = technique_counts.get(technique, 0) + 1
                
                for technique, count in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"{technique}: {count} items\n")
                
                f.write(f"\nAverage Confidence Score: {self.dataframe['mitre_confidence_score'].mean():.2f}\n")
                f.write(f"High Risk Items: {len(self.dataframe[self.dataframe['risk_level'] == 'High'])}\n")
                f.write(f"Medium Risk Items: {len(self.dataframe[self.dataframe['risk_level'] == 'Medium'])}\n")
                f.write(f"Low Risk Items: {len(self.dataframe[self.dataframe['risk_level'] == 'Low'])}\n")
                
        except Exception as e:
            logger.warning(f"Error exporting summary report: {e}")

# Example usage
async def main():
    """Example usage of MITRE ATT&CK mapping workflow."""
    workflow = MitreAttackMappingWorkflow()
    
    # Example execution
    result = await workflow.execute_mapping_workflow(
        input_file="policy_catalog.csv",
        output_file="mitre_mapped_policies.csv",
        content_column="policy_description",
        batch_size=50
    )
    
    if result.success:
        print(f"✅ Mapping completed successfully!")
        print(f"   Total items: {result.total_items}")
        print(f"   Mapped items: {result.mapped_items}")
        print(f"   Output file: {result.output_file}")
    else:
        print(f"❌ Mapping failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
