#!/usr/bin/env python3
"""
Enhanced Training Data Generator for Workflow Detection

This module generates comprehensive, real-world training datasets based on actual
usage patterns and capabilities of the cybersecurity agent.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib

from bin.intelligent_workflow_detector import WorkflowType, InputComplexity
from bin.workflow_detection_trainer import TrainingExample, TrainingDataset

@dataclass
class RealWorldScenario:
    """A real-world training scenario based on actual agent capabilities."""
    user_input: str
    workflow_type: WorkflowType
    complexity: InputComplexity
    input_files: List[str]
    context_clues: List[str]
    expected_outputs: List[str]
    metadata: Dict[str, Any]

class EnhancedTrainingDataGenerator:
    """Generate comprehensive training datasets based on real agent capabilities."""
    
    def __init__(self):
        self.real_world_scenarios = self._initialize_real_world_scenarios()
        self.edge_cases = self._initialize_edge_cases()
        self.multi_step_workflows = self._initialize_multi_step_workflows()
    
    def _initialize_real_world_scenarios(self) -> List[RealWorldScenario]:
        """Initialize real-world scenarios based on actual agent capabilities."""
        return [
            # Patent Analysis - Real USPTO Integration
            RealWorldScenario(
                user_input="Analyze these patent numbers and get their full details from USPTO: US12345678, EP98765432, WO1234567890",
                workflow_type=WorkflowType.PATENT_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["patent_list.csv"],
                context_clues=["USPTO", "publication number", "inventor", "assignee", "abstract"],
                expected_outputs=["CSV with patent details", "PDF downloads", "LLM insights"],
                metadata={"domain": "intellectual_property", "api": "USPTO", "llm_enabled": True}
            ),
            
            # Malware Analysis - YARA Rules and PE Analysis
            RealWorldScenario(
                user_input="Analyze this suspicious executable file for malware using YARA rules and PE analysis",
                workflow_type=WorkflowType.MALWARE_ANALYSIS,
                complexity=InputComplexity.COMPLEX,
                input_files=["suspicious.exe", "malware_samples.zip"],
                context_clues=["YARA", "PE analysis", "hash", "signature", "behavior"],
                expected_outputs=["YARA scan results", "PE analysis report", "threat classification"],
                metadata={"domain": "malware_analysis", "tools": ["YARA", "PE"], "automated": True}
            ),
            
            # Network Analysis - PCAP Deep Dive
            RealWorldScenario(
                user_input="Perform deep packet analysis on this network capture to identify suspicious traffic patterns and potential threats",
                workflow_type=WorkflowType.NETWORK_ANALYSIS,
                complexity=InputComplexity.COMPLEX,
                input_files=["network_traffic.pcap", "baseline_traffic.pcap"],
                context_clues=["packet analysis", "traffic patterns", "protocols", "anomalies"],
                expected_outputs=["Traffic analysis report", "Anomaly detection", "Protocol breakdown"],
                metadata={"domain": "network_forensics", "tools": ["Scapy", "Wireshark"], "deep_analysis": True}
            ),
            
            # Vulnerability Scanning - Comprehensive Assessment
            RealWorldScenario(
                user_input="Run a comprehensive vulnerability scan on our network infrastructure and generate a detailed security assessment report",
                workflow_type=WorkflowType.VULNERABILITY_SCAN,
                complexity=InputComplexity.COMPLEX,
                input_files=["network_inventory.csv", "scan_config.json"],
                context_clues=["vulnerability scan", "security assessment", "CVE", "risk rating"],
                expected_outputs=["Vulnerability report", "Risk assessment", "Remediation recommendations"],
                metadata={"domain": "vulnerability_management", "comprehensive": True, "automated": True}
            ),
            
            # Incident Response - Full Investigation
            RealWorldScenario(
                user_input="Investigate this security incident: we detected suspicious activity on our web server. Analyze logs, timeline, and evidence to determine the scope and impact",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.EXPERT,
                input_files=["web_server_logs.csv", "network_logs.pcap", "system_logs.json"],
                context_clues=["incident response", "timeline analysis", "evidence", "scope", "impact"],
                expected_outputs=["Incident timeline", "Evidence analysis", "Impact assessment", "Response plan"],
                metadata={"domain": "incident_response", "severity": "high", "multi_source": True}
            ),
            
            # Threat Hunting - Proactive Search
            RealWorldScenario(
                user_input="Conduct proactive threat hunting across our network to identify potential APT activity and IOCs that might indicate a breach",
                workflow_type=WorkflowType.THREAT_HUNTING,
                complexity=InputComplexity.EXPERT,
                input_files=["network_logs.csv", "endpoint_logs.json", "ioc_database.csv"],
                context_clues=["threat hunting", "APT", "IOC", "proactive", "breach indicators"],
                expected_outputs=["Threat hunting report", "IOC analysis", "APT indicators", "Recommendations"],
                metadata={"domain": "threat_hunting", "apt_focus": True, "proactive": True}
            ),
            
            # Data Analysis - Security Metrics
            RealWorldScenario(
                user_input="Analyze our security metrics data to identify trends, anomalies, and provide insights for improving our security posture",
                workflow_type=WorkflowType.DATA_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["security_metrics.csv", "incident_data.json", "vulnerability_data.xlsx"],
                context_clues=["security metrics", "trends", "anomalies", "insights", "security posture"],
                expected_outputs=["Trend analysis", "Anomaly detection", "Security insights", "Recommendations"],
                metadata={"domain": "security_analytics", "metrics_focus": True, "insights": True}
            ),
            
            # File Forensics - Digital Evidence
            RealWorldScenario(
                user_input="Perform digital forensics analysis on this disk image to recover deleted files, analyze file system artifacts, and create a timeline of user activity",
                workflow_type=WorkflowType.FILE_FORENSICS,
                complexity=InputComplexity.EXPERT,
                input_files=["disk_image.dd", "memory_dump.dmp", "registry_hives"],
                context_clues=["digital forensics", "deleted files", "artifacts", "timeline", "user activity"],
                expected_outputs=["Forensic report", "Recovered files", "Activity timeline", "Evidence chain"],
                metadata={"domain": "digital_forensics", "evidence": True, "timeline": True}
            ),
            
            # Compliance Assessment - Regulatory Framework
            RealWorldScenario(
                user_input="Assess our compliance with GDPR, HIPAA, and SOC 2 requirements by analyzing our security controls and data handling practices",
                workflow_type=WorkflowType.COMPLIANCE_ASSESSMENT,
                complexity=InputComplexity.COMPLEX,
                input_files=["security_controls.csv", "data_inventory.json", "policy_documents.pdf"],
                context_clues=["compliance", "GDPR", "HIPAA", "SOC 2", "security controls"],
                expected_outputs=["Compliance report", "Gap analysis", "Remediation plan", "Risk assessment"],
                metadata={"domain": "compliance", "frameworks": ["GDPR", "HIPAA", "SOC2"], "comprehensive": True}
            ),
            
            # Threat Intelligence - IOC Analysis
            RealWorldScenario(
                user_input="Analyze these threat intelligence feeds and IOCs to identify potential threats to our organization and update our security controls",
                workflow_type=WorkflowType.THREAT_INTELLIGENCE,
                complexity=InputComplexity.MODERATE,
                input_files=["threat_feeds.json", "ioc_list.csv", "threat_actors.json"],
                context_clues=["threat intelligence", "IOC", "threat feeds", "security controls"],
                expected_outputs=["Threat analysis", "IOC correlation", "Control updates", "Threat landscape"],
                metadata={"domain": "threat_intelligence", "feeds": True, "ioc_analysis": True}
            ),
            
            # Multi-Domain Analysis - Complex Investigation
            RealWorldScenario(
                user_input="We suspect a sophisticated attack. Analyze network traffic, endpoint logs, and threat intelligence to determine if this is an APT campaign and what data may have been compromised",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.EXPERT,
                input_files=["network_traffic.pcap", "endpoint_logs.csv", "threat_intel.json", "user_activity.log"],
                context_clues=["sophisticated attack", "APT", "data compromise", "multi-source analysis"],
                expected_outputs=["Attack analysis", "APT assessment", "Data impact", "Response strategy"],
                metadata={"domain": "complex_investigation", "apt": True, "multi_domain": True}
            ),
            
            # Cloud Security Assessment
            RealWorldScenario(
                user_input="Assess the security posture of our AWS and Azure cloud infrastructure, including IAM policies, network configurations, and data encryption",
                workflow_type=WorkflowType.VULNERABILITY_SCAN,
                complexity=InputComplexity.COMPLEX,
                input_files=["aws_config.json", "azure_config.json", "iam_policies.csv"],
                context_clues=["cloud security", "AWS", "Azure", "IAM", "encryption"],
                expected_outputs=["Cloud security report", "Configuration analysis", "Risk assessment"],
                metadata={"domain": "cloud_security", "platforms": ["AWS", "Azure"], "comprehensive": True}
            ),
            
            # Mobile Security Analysis
            RealWorldScenario(
                user_input="Analyze these mobile device logs and app data to identify potential security risks and compliance violations in our BYOD program",
                workflow_type=WorkflowType.DATA_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["mobile_logs.csv", "app_data.json", "device_inventory.xlsx"],
                context_clues=["mobile security", "BYOD", "compliance", "app analysis"],
                expected_outputs=["Mobile security report", "Risk assessment", "Compliance analysis"],
                metadata={"domain": "mobile_security", "byod": True, "compliance": True}
            ),
            
            # Supply Chain Security
            RealWorldScenario(
                user_input="Evaluate the security of our software supply chain by analyzing third-party dependencies, build processes, and deployment pipelines",
                workflow_type=WorkflowType.VULNERABILITY_SCAN,
                complexity=InputComplexity.COMPLEX,
                input_files=["dependencies.json", "build_logs.csv", "deployment_config.yaml"],
                context_clues=["supply chain", "dependencies", "build process", "deployment"],
                expected_outputs=["Supply chain report", "Vulnerability analysis", "Risk assessment"],
                metadata={"domain": "supply_chain", "dependencies": True, "automation": True}
            ),
            
            # Insider Threat Detection
            RealWorldScenario(
                user_input="Analyze user behavior patterns and access logs to identify potential insider threats and unusual activity that might indicate malicious intent",
                workflow_type=WorkflowType.THREAT_HUNTING,
                complexity=InputComplexity.COMPLEX,
                input_files=["user_behavior.csv", "access_logs.json", "privilege_changes.log"],
                context_clues=["insider threat", "behavior analysis", "access patterns", "privilege escalation"],
                expected_outputs=["Behavior analysis", "Threat indicators", "Risk assessment"],
                metadata={"domain": "insider_threat", "behavioral": True, "privilege": True}
            )
        ]
    
    def _initialize_edge_cases(self) -> List[RealWorldScenario]:
        """Initialize edge cases and ambiguous scenarios."""
        return [
            # Ambiguous Cases
            RealWorldScenario(
                user_input="Check this file for any issues",
                workflow_type=WorkflowType.MALWARE_ANALYSIS,
                complexity=InputComplexity.SIMPLE,
                input_files=["unknown_file.bin"],
                context_clues=["file check", "issues"],
                expected_outputs=["File analysis"],
                metadata={"domain": "ambiguous", "type": "file_check"}
            ),
            
            RealWorldScenario(
                user_input="Analyze this data and tell me what you find",
                workflow_type=WorkflowType.DATA_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["mystery_data.csv"],
                context_clues=["data analysis", "findings"],
                expected_outputs=["Data insights"],
                metadata={"domain": "ambiguous", "type": "general_analysis"}
            ),
            
            RealWorldScenario(
                user_input="Investigate this security alert",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.MODERATE,
                input_files=["security_alert.json"],
                context_clues=["security alert", "investigation"],
                expected_outputs=["Alert analysis"],
                metadata={"domain": "ambiguous", "type": "alert_investigation"}
            ),
            
            # Multi-Domain Cases
            RealWorldScenario(
                user_input="This looks like malware but I also want to check if it's communicating over the network",
                workflow_type=WorkflowType.MALWARE_ANALYSIS,
                complexity=InputComplexity.COMPLEX,
                input_files=["suspicious.exe", "network_capture.pcap"],
                context_clues=["malware", "network communication", "multi-domain"],
                expected_outputs=["Malware analysis", "Network analysis"],
                metadata={"domain": "multi_domain", "type": "malware_network"}
            ),
            
            RealWorldScenario(
                user_input="I need to analyze this incident data and also check our threat intelligence for related IOCs",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.COMPLEX,
                input_files=["incident_data.csv", "threat_intel.json"],
                context_clues=["incident", "threat intelligence", "IOC"],
                expected_outputs=["Incident analysis", "IOC correlation"],
                metadata={"domain": "multi_domain", "type": "incident_intel"}
            ),
            
            # Context-Dependent Cases
            RealWorldScenario(
                user_input="Analyze this network traffic for security threats",
                workflow_type=WorkflowType.NETWORK_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["traffic.pcap"],
                context_clues=["network traffic", "security threats"],
                expected_outputs=["Threat analysis"],
                metadata={"domain": "context_dependent", "type": "network_threats"}
            ),
            
            RealWorldScenario(
                user_input="Analyze this network traffic for performance issues",
                workflow_type=WorkflowType.NETWORK_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                input_files=["traffic.pcap"],
                context_clues=["network traffic", "performance"],
                expected_outputs=["Performance analysis"],
                metadata={"domain": "context_dependent", "type": "network_performance"}
            )
        ]
    
    def _initialize_multi_step_workflows(self) -> List[RealWorldScenario]:
        """Initialize multi-step workflow scenarios."""
        return [
            # Sequential Analysis
            RealWorldScenario(
                user_input="First scan this file for malware, then if it's clean, analyze its network behavior",
                workflow_type=WorkflowType.MALWARE_ANALYSIS,
                complexity=InputComplexity.COMPLEX,
                input_files=["unknown_file.exe"],
                context_clues=["sequential", "malware scan", "network behavior"],
                expected_outputs=["Malware scan", "Network analysis"],
                metadata={"domain": "sequential", "steps": ["malware", "network"]}
            ),
            
            # Parallel Analysis
            RealWorldScenario(
                user_input="Analyze this incident from multiple angles: network traffic, endpoint logs, and user behavior",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.EXPERT,
                input_files=["network.pcap", "endpoint_logs.csv", "user_behavior.json"],
                context_clues=["multiple angles", "parallel analysis", "comprehensive"],
                expected_outputs=["Network analysis", "Endpoint analysis", "Behavior analysis"],
                metadata={"domain": "parallel", "angles": ["network", "endpoint", "behavior"]}
            ),
            
            # Iterative Analysis
            RealWorldScenario(
                user_input="Start with a broad threat hunt, then drill down into any suspicious findings for detailed analysis",
                workflow_type=WorkflowType.THREAT_HUNTING,
                complexity=InputComplexity.EXPERT,
                input_files=["network_logs.csv", "system_logs.json"],
                context_clues=["iterative", "broad hunt", "drill down"],
                expected_outputs=["Broad analysis", "Detailed findings"],
                metadata={"domain": "iterative", "approach": "broad_to_detailed"}
            )
        ]
    
    def generate_enhanced_training_data(self, num_examples: int = 1000) -> List[TrainingExample]:
        """Generate enhanced training data with real-world scenarios."""
        examples = []
        
        # Calculate distribution
        real_world_count = int(num_examples * 0.6)  # 60% real-world scenarios
        edge_case_count = int(num_examples * 0.25)  # 25% edge cases
        multi_step_count = int(num_examples * 0.15)  # 15% multi-step workflows
        
        # Generate real-world scenarios
        for _ in range(real_world_count):
            scenario = random.choice(self.real_world_scenarios)
            example = self._create_training_example_from_scenario(scenario)
            examples.append(example)
        
        # Generate edge cases
        for _ in range(edge_case_count):
            scenario = random.choice(self.edge_cases)
            example = self._create_training_example_from_scenario(scenario)
            examples.append(example)
        
        # Generate multi-step workflows
        for _ in range(multi_step_count):
            scenario = random.choice(self.multi_step_workflows)
            example = self._create_training_example_from_scenario(scenario)
            examples.append(example)
        
        # Add variations and complexity modifiers
        enhanced_examples = []
        for example in examples:
            # Add variations
            variations = self._generate_variations(example)
            enhanced_examples.extend(variations)
        
        # Shuffle and return
        random.shuffle(enhanced_examples)
        return enhanced_examples[:num_examples]
    
    def _create_training_example_from_scenario(self, scenario: RealWorldScenario) -> TrainingExample:
        """Create a training example from a real-world scenario."""
        return TrainingExample(
            user_input=scenario.user_input,
            input_files=scenario.input_files,
            correct_workflow=scenario.workflow_type,
            complexity=scenario.complexity,
            confidence_score=random.uniform(0.8, 1.0),
            timestamp=time.time(),
            metadata={
                **scenario.metadata,
                "context_clues": scenario.context_clues,
                "expected_outputs": scenario.expected_outputs,
                "source": "enhanced_real_world"
            }
        )
    
    def _generate_variations(self, example: TrainingExample) -> List[TrainingExample]:
        """Generate variations of a training example."""
        variations = [example]  # Include original
        
        # Add complexity variations
        if example.complexity == InputComplexity.SIMPLE:
            # Add moderate complexity version
            moderate_example = TrainingExample(
                user_input=f"Please {example.user_input.lower()}",
                input_files=example.input_files,
                correct_workflow=example.correct_workflow,
                complexity=InputComplexity.MODERATE,
                confidence_score=example.confidence_score,
                timestamp=time.time(),
                metadata={**example.metadata, "variation": "complexity_upgrade"}
            )
            variations.append(moderate_example)
        
        # Add urgency variations
        urgency_phrases = [
            "This is urgent - ",
            "ASAP: ",
            "High priority: ",
            "Critical: ",
            "Emergency: "
        ]
        
        if random.random() < 0.3:  # 30% chance
            urgency_phrase = random.choice(urgency_phrases)
            urgent_example = TrainingExample(
                user_input=f"{urgency_phrase}{example.user_input}",
                input_files=example.input_files,
                correct_workflow=example.correct_workflow,
                complexity=example.complexity,
                confidence_score=example.confidence_score,
                timestamp=time.time(),
                metadata={**example.metadata, "variation": "urgency"}
            )
            variations.append(urgent_example)
        
        # Add context variations
        context_additions = [
            " for our security team",
            " to help with our investigation",
            " as part of our security assessment",
            " for compliance purposes",
            " to improve our security posture"
        ]
        
        if random.random() < 0.2:  # 20% chance
            context_addition = random.choice(context_additions)
            context_example = TrainingExample(
                user_input=f"{example.user_input}{context_addition}",
                input_files=example.input_files,
                correct_workflow=example.correct_workflow,
                complexity=example.complexity,
                confidence_score=example.confidence_score,
                timestamp=time.time(),
                metadata={**example.metadata, "variation": "context"}
            )
            variations.append(context_example)
        
        return variations
    
    def generate_dataset(self, num_examples: int = 1000, version: str = "2.0") -> TrainingDataset:
        """Generate a complete enhanced training dataset."""
        print(f"Generating {num_examples} enhanced training examples...")
        
        examples = self.generate_enhanced_training_data(num_examples)
        
        dataset = TrainingDataset(
            examples=examples,
            created_at=time.time(),
            version=version,
            description=f"Enhanced training dataset with {len(examples)} real-world examples based on actual agent capabilities"
        )
        
        print(f"Generated {len(examples)} enhanced training examples")
        return dataset
    
    def save_dataset(self, dataset: TrainingDataset, filename: str = None):
        """Save enhanced dataset to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"enhanced_training_data_{timestamp}.json"
        
        filepath = Path("training_data") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)
        
        print(f"Enhanced dataset saved to {filepath}")
        return filepath

def main():
    """Main function for generating enhanced training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced training data for workflow detection")
    parser.add_argument("--examples", type=int, default=1000,
                       help="Number of training examples to generate")
    parser.add_argument("--version", default="2.0",
                       help="Dataset version")
    parser.add_argument("--output", default=None,
                       help="Output filename")
    
    args = parser.parse_args()
    
    # Generate enhanced dataset
    generator = EnhancedTrainingDataGenerator()
    dataset = generator.generate_dataset(args.examples, args.version)
    
    # Save dataset
    filepath = generator.save_dataset(dataset, args.output)
    
    # Print statistics
    stats = {
        "total_examples": len(dataset.examples),
        "workflow_distribution": {},
        "complexity_distribution": {},
        "source_distribution": {},
        "domain_distribution": {}
    }
    
    from collections import Counter
    stats["workflow_distribution"] = Counter(ex.correct_workflow.value for ex in dataset.examples)
    stats["complexity_distribution"] = Counter(ex.complexity.value for ex in dataset.examples)
    stats["source_distribution"] = Counter(ex.metadata.get("source", "unknown") for ex in dataset.examples)
    stats["domain_distribution"] = Counter(ex.metadata.get("domain", "unknown") for ex in dataset.examples)
    
    print(f"\nEnhanced Dataset Statistics:")
    print(f"Total Examples: {stats['total_examples']}")
    print(f"Workflow Distribution: {dict(stats['workflow_distribution'])}")
    print(f"Complexity Distribution: {dict(stats['complexity_distribution'])}")
    print(f"Source Distribution: {dict(stats['source_distribution'])}")
    print(f"Domain Distribution: {dict(stats['domain_distribution'])}")
    
    print(f"\nEnhanced training data generation completed!")
    print(f"Dataset saved to: {filepath}")

if __name__ == "__main__":
    main()
