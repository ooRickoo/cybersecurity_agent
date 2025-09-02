#!/usr/bin/env python3
"""
Training Data Generator for Workflow Detection

This module generates comprehensive training datasets for the workflow detection system,
including real-world examples, edge cases, and diverse scenarios.
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
class TrainingScenario:
    """A training scenario with multiple variations."""
    base_input: str
    workflow_type: WorkflowType
    complexity: InputComplexity
    variations: List[str]
    file_types: List[str]
    context_clues: List[str]
    metadata: Dict[str, Any]

class TrainingDataGenerator:
    """Generate comprehensive training datasets for workflow detection."""
    
    def __init__(self):
        self.scenarios = self._initialize_training_scenarios()
        self.file_type_samples = self._initialize_file_type_samples()
        self.complexity_modifiers = self._initialize_complexity_modifiers()
    
    def _initialize_training_scenarios(self) -> List[TrainingScenario]:
        """Initialize comprehensive training scenarios."""
        return [
            # Patent Analysis Scenarios
            TrainingScenario(
                base_input="Analyze these patent numbers",
                workflow_type=WorkflowType.PATENT_ANALYSIS,
                complexity=InputComplexity.SIMPLE,
                variations=[
                    "Analyze these patent numbers: {patents}",
                    "Look up patent information for {patents}",
                    "Get patent details from USPTO for {patents}",
                    "Find patent abstracts and inventors for {patents}",
                    "Patent analysis workflow for {patents}",
                    "Can you analyze these patents: {patents}?",
                    "I need patent information for {patents}",
                    "Please look up these patent numbers: {patents}",
                    "Patent lookup for {patents}",
                    "USPTO patent search for {patents}"
                ],
                file_types=["patent_list.csv", "patent_numbers.txt"],
                context_clues=["publication", "inventor", "assignee", "abstract", "claims"],
                metadata={"domain": "intellectual_property", "source": "patent_analysis"}
            ),
            
            # Malware Analysis Scenarios
            TrainingScenario(
                base_input="Analyze this suspicious file",
                workflow_type=WorkflowType.MALWARE_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                variations=[
                    "Analyze this suspicious file: {file}",
                    "Scan for malware in {file}",
                    "Detect threats in {file}",
                    "Malware analysis of {file}",
                    "Check if {file} is malicious",
                    "Is {file} a virus?",
                    "Can you analyze {file} for malware?",
                    "Threat detection for {file}",
                    "Security scan of {file}",
                    "Virus check for {file}"
                ],
                file_types=["malware.exe", "suspicious.pdf", "trojan.dll", "virus.zip"],
                context_clues=["hash", "signature", "behavior", "sandbox", "yara"],
                metadata={"domain": "malware_analysis", "source": "security_analysis"}
            ),
            
            # Network Analysis Scenarios
            TrainingScenario(
                base_input="Analyze network traffic",
                workflow_type=WorkflowType.NETWORK_ANALYSIS,
                complexity=InputComplexity.MODERATE,
                variations=[
                    "Analyze network traffic in {file}",
                    "PCAP analysis of {file}",
                    "Network forensics for {file}",
                    "Packet capture analysis {file}",
                    "Traffic analysis workflow {file}",
                    "Can you analyze this network capture: {file}?",
                    "Network packet analysis {file}",
                    "Traffic forensics {file}",
                    "Network investigation {file}",
                    "PCAP file analysis {file}"
                ],
                file_types=["traffic.pcap", "network.pcapng", "capture.pcap"],
                context_clues=["tcp", "udp", "icmp", "dns", "http", "https", "firewall"],
                metadata={"domain": "network_analysis", "source": "network_forensics"}
            ),
            
            # Vulnerability Scanning Scenarios
            TrainingScenario(
                base_input="Scan for vulnerabilities",
                workflow_type=WorkflowType.VULNERABILITY_SCAN,
                complexity=InputComplexity.MODERATE,
                variations=[
                    "Scan {target} for vulnerabilities",
                    "Security assessment of {target}",
                    "Vulnerability scan {target}",
                    "Penetration test {target}",
                    "Security audit {target}",
                    "Can you scan {target} for security issues?",
                    "Vulnerability assessment {target}",
                    "Security testing {target}",
                    "Pen test {target}",
                    "Security evaluation {target}"
                ],
                file_types=["scan_results.xml", "vulnerability_report.json"],
                context_clues=["port", "service", "exploit", "cve", "severity", "risk"],
                metadata={"domain": "vulnerability_assessment", "source": "security_testing"}
            ),
            
            # Incident Response Scenarios
            TrainingScenario(
                base_input="Incident response investigation",
                workflow_type=WorkflowType.INCIDENT_RESPONSE,
                complexity=InputComplexity.COMPLEX,
                variations=[
                    "Incident response for {incident}",
                    "Investigate security breach {incident}",
                    "Forensic analysis of {incident}",
                    "Incident timeline {incident}",
                    "Security incident {incident}",
                    "Can you help with incident response for {incident}?",
                    "Breach investigation {incident}",
                    "Security incident analysis {incident}",
                    "Incident forensics {incident}",
                    "Response to security incident {incident}"
                ],
                file_types=["incident_logs.csv", "breach_data.json", "forensics.log"],
                context_clues=["timeline", "evidence", "ioc", "triage", "containment", "eradication"],
                metadata={"domain": "incident_response", "source": "security_incident"}
            ),
            
            # Threat Hunting Scenarios
            TrainingScenario(
                base_input="Threat hunting analysis",
                workflow_type=WorkflowType.THREAT_HUNTING,
                complexity=InputComplexity.COMPLEX,
                variations=[
                    "Threat hunting in {data}",
                    "Proactive threat search {data}",
                    "Find anomalies in {data}",
                    "Hunt for threats {data}",
                    "Threat intelligence {data}",
                    "Can you help with threat hunting in {data}?",
                    "Proactive security analysis {data}",
                    "Threat detection {data}",
                    "Anomaly detection {data}",
                    "Threat hunting workflow {data}"
                ],
                file_types=["threat_data.csv", "hunting_logs.json", "ioc_list.txt"],
                context_clues=["ioc", "ttps", "mitre", "behavior", "anomaly", "baseline"],
                metadata={"domain": "threat_hunting", "source": "proactive_security"}
            ),
            
            # Data Analysis Scenarios
            TrainingScenario(
                base_input="Analyze data file",
                workflow_type=WorkflowType.DATA_ANALYSIS,
                complexity=InputComplexity.SIMPLE,
                variations=[
                    "Analyze data in {file}",
                    "Data analysis workflow {file}",
                    "Statistical analysis {file}",
                    "Data insights from {file}",
                    "Process data file {file}",
                    "Can you analyze this data: {file}?",
                    "Data processing {file}",
                    "Statistical insights {file}",
                    "Data exploration {file}",
                    "Data summary {file}"
                ],
                file_types=["data.csv", "dataset.json", "analysis.xlsx", "stats.txt"],
                context_clues=["columns", "rows", "statistics", "correlation", "trend", "summary"],
                metadata={"domain": "data_analysis", "source": "data_processing"}
            ),
            
            # File Forensics Scenarios
            TrainingScenario(
                base_input="Forensic analysis",
                workflow_type=WorkflowType.FILE_FORENSICS,
                complexity=InputComplexity.COMPLEX,
                variations=[
                    "Forensic analysis of {file}",
                    "File recovery from {file}",
                    "Metadata extraction {file}",
                    "Timeline analysis {file}",
                    "Digital forensics {file}",
                    "Can you perform forensics on {file}?",
                    "File system analysis {file}",
                    "Digital evidence {file}",
                    "Forensic investigation {file}",
                    "File carving {file}"
                ],
                file_types=["forensic_image.dd", "disk_image.e01", "file_system.img"],
                context_clues=["metadata", "timeline", "deleted", "recovery", "artifact", "hash"],
                metadata={"domain": "digital_forensics", "source": "forensic_analysis"}
            ),
            
            # Compliance Assessment Scenarios
            TrainingScenario(
                base_input="Compliance assessment",
                workflow_type=WorkflowType.COMPLIANCE_ASSESSMENT,
                complexity=InputComplexity.MODERATE,
                variations=[
                    "Compliance assessment for {system}",
                    "Check compliance with {standard}",
                    "Audit {system} for compliance",
                    "Compliance review {system}",
                    "Regulatory assessment {system}",
                    "Can you assess compliance for {system}?",
                    "Compliance audit {system}",
                    "Regulatory compliance {system}",
                    "Compliance check {system}",
                    "Standards assessment {system}"
                ],
                file_types=["compliance_data.csv", "audit_results.json", "standards.xlsx"],
                context_clues=["compliance", "audit", "regulatory", "standards", "requirements"],
                metadata={"domain": "compliance", "source": "regulatory_assessment"}
            ),
            
            # Threat Intelligence Scenarios
            TrainingScenario(
                base_input="Threat intelligence analysis",
                workflow_type=WorkflowType.THREAT_INTELLIGENCE,
                complexity=InputComplexity.MODERATE,
                variations=[
                    "Threat intelligence analysis {data}",
                    "IOC analysis {data}",
                    "Threat actor profiling {data}",
                    "Intelligence gathering {data}",
                    "Threat landscape analysis {data}",
                    "Can you analyze threat intelligence: {data}?",
                    "IOC enrichment {data}",
                    "Threat research {data}",
                    "Intelligence assessment {data}",
                    "Threat analysis {data}"
                ],
                file_types=["ioc_list.csv", "threat_intel.json", "actor_profiles.txt"],
                context_clues=["ioc", "threat_actor", "campaign", "ttps", "intelligence"],
                metadata={"domain": "threat_intelligence", "source": "intelligence_analysis"}
            )
        ]
    
    def _initialize_file_type_samples(self) -> Dict[str, List[str]]:
        """Initialize file type samples for each category."""
        return {
            "patent": ["US12345678", "EP98765432", "WO1234567890", "patent_list.csv"],
            "malware": ["malware.exe", "suspicious.pdf", "trojan.dll", "virus.zip", "ransomware.bin"],
            "network": ["traffic.pcap", "network.pcapng", "capture.pcap", "netflow.csv"],
            "vulnerability": ["scan_results.xml", "vulnerability_report.json", "nmap_output.txt"],
            "incident": ["incident_logs.csv", "breach_data.json", "forensics.log", "timeline.txt"],
            "threat_hunting": ["threat_data.csv", "hunting_logs.json", "ioc_list.txt", "anomalies.csv"],
            "data": ["data.csv", "dataset.json", "analysis.xlsx", "stats.txt", "metrics.csv"],
            "forensics": ["forensic_image.dd", "disk_image.e01", "file_system.img", "memory.dmp"],
            "compliance": ["compliance_data.csv", "audit_results.json", "standards.xlsx", "policies.pdf"],
            "intelligence": ["ioc_list.csv", "threat_intel.json", "actor_profiles.txt", "campaigns.csv"]
        }
    
    def _initialize_complexity_modifiers(self) -> Dict[InputComplexity, List[str]]:
        """Initialize complexity modifiers for each level."""
        return {
            InputComplexity.SIMPLE: [
                "quick", "simple", "basic", "fast", "easy", "straightforward"
            ],
            InputComplexity.MODERATE: [
                "detailed", "comprehensive", "thorough", "complete", "standard"
            ],
            InputComplexity.COMPLEX: [
                "advanced", "complex", "sophisticated", "deep", "extensive", "comprehensive"
            ],
            InputComplexity.EXPERT: [
                "expert", "professional", "enterprise", "advanced", "sophisticated", "cutting-edge"
            ]
        }
    
    def generate_training_examples(self, num_examples: int = 5000) -> List[TrainingExample]:
        """Generate comprehensive training examples."""
        examples = []
        
        # Calculate examples per scenario
        examples_per_scenario = num_examples // len(self.scenarios)
        
        for scenario in self.scenarios:
            for _ in range(examples_per_scenario):
                # Select random variation
                variation = random.choice(scenario.variations)
                
                # Replace placeholders with sample data
                user_input = variation
                input_files = []
                
                # Replace file placeholders
                if "{file}" in user_input:
                    file_sample = random.choice(self.file_type_samples.get(
                        scenario.workflow_type.value.split('_')[0], ["sample.txt"]
                    ))
                    user_input = user_input.replace("{file}", file_sample)
                    input_files = [file_sample]
                
                # Replace other placeholders
                if "{patents}" in user_input:
                    patents = random.sample(self.file_type_samples["patent"], 
                                          random.randint(1, 3))
                    user_input = user_input.replace("{patents}", ", ".join(patents))
                
                if "{target}" in user_input:
                    targets = ["192.168.1.1", "example.com", "server.local", "10.0.0.1"]
                    user_input = user_input.replace("{target}", random.choice(targets))
                
                if "{incident}" in user_input:
                    incidents = ["breach_2024", "attack_logs", "security_event", "incident_001"]
                    user_input = user_input.replace("{incident}", random.choice(incidents))
                
                if "{data}" in user_input:
                    data_sources = ["network_logs", "system_logs", "user_data", "security_logs"]
                    user_input = user_input.replace("{data}", random.choice(data_sources))
                
                if "{system}" in user_input:
                    systems = ["web_server", "database", "network", "application"]
                    user_input = user_input.replace("{system}", random.choice(systems))
                
                if "{standard}" in user_input:
                    standards = ["ISO 27001", "NIST", "PCI DSS", "GDPR", "HIPAA"]
                    user_input = user_input.replace("{standard}", random.choice(standards))
                
                # Add complexity modifiers
                if random.random() < 0.3:  # 30% chance to add complexity modifier
                    modifier = random.choice(self.complexity_modifiers[scenario.complexity])
                    user_input = f"{modifier} {user_input}"
                
                # Add context clues
                if random.random() < 0.2:  # 20% chance to add context clues
                    clue = random.choice(scenario.context_clues)
                    user_input = f"{user_input} (focus on {clue})"
                
                # Determine final complexity (may be modified by modifiers)
                final_complexity = scenario.complexity
                if any(mod in user_input for mod in self.complexity_modifiers[InputComplexity.COMPLEX]):
                    final_complexity = InputComplexity.COMPLEX
                elif any(mod in user_input for mod in self.complexity_modifiers[InputComplexity.EXPERT]):
                    final_complexity = InputComplexity.EXPERT
                
                # Create training example
                example = TrainingExample(
                    user_input=user_input,
                    input_files=input_files,
                    correct_workflow=scenario.workflow_type,
                    complexity=final_complexity,
                    confidence_score=random.uniform(0.8, 1.0),
                    timestamp=time.time(),
                    metadata={
                        **scenario.metadata,
                        "generated": True,
                        "scenario_id": hash(scenario.base_input) % 10000
                    }
                )
                
                examples.append(example)
        
        # Add some edge cases and difficult examples
        edge_cases = self._generate_edge_cases(num_examples // 10)
        examples.extend(edge_cases)
        
        # Shuffle examples
        random.shuffle(examples)
        
        return examples
    
    def _generate_edge_cases(self, num_cases: int) -> List[TrainingExample]:
        """Generate edge cases and difficult examples."""
        edge_cases = []
        
        # Ambiguous cases
        ambiguous_inputs = [
            "Analyze this file",  # Could be malware, network, or data analysis
            "Check for issues",   # Could be vulnerability scan or general analysis
            "Investigate this",   # Could be incident response or threat hunting
            "Look at this data",  # Could be data analysis or threat intelligence
            "Process this",       # Could be any workflow
            "Help me with this",  # Very ambiguous
            "What can you tell me about this?",  # General analysis
            "Is this normal?",    # Could be threat hunting or general analysis
        ]
        
        for _ in range(num_cases // 2):
            user_input = random.choice(ambiguous_inputs)
            
            # Randomly assign workflow (these are intentionally ambiguous)
            workflow = random.choice(list(WorkflowType))
            
            example = TrainingExample(
                user_input=user_input,
                input_files=[],
                correct_workflow=workflow,
                complexity=InputComplexity.MODERATE,
                confidence_score=random.uniform(0.3, 0.7),  # Lower confidence for ambiguous cases
                timestamp=time.time(),
                metadata={"source": "edge_case", "type": "ambiguous"}
            )
            
            edge_cases.append(example)
        
        # Multi-domain cases (could fit multiple workflows)
        multi_domain_inputs = [
            "Analyze this network traffic for malware indicators",
            "Check this file for vulnerabilities and malware",
            "Investigate this incident and hunt for threats",
            "Analyze this data for security anomalies",
            "Forensic analysis of this network capture",
            "Threat intelligence analysis of this incident data",
            "Compliance assessment of this security data",
            "Malware analysis with network forensics"
        ]
        
        for _ in range(num_cases // 2):
            user_input = random.choice(multi_domain_inputs)
            
            # Choose primary workflow (these cases have a primary focus)
            primary_workflows = [
                WorkflowType.NETWORK_ANALYSIS,
                WorkflowType.MALWARE_ANALYSIS,
                WorkflowType.INCIDENT_RESPONSE,
                WorkflowType.THREAT_HUNTING
            ]
            
            workflow = random.choice(primary_workflows)
            
            example = TrainingExample(
                user_input=user_input,
                input_files=[],
                correct_workflow=workflow,
                complexity=InputComplexity.COMPLEX,
                confidence_score=random.uniform(0.6, 0.9),
                timestamp=time.time(),
                metadata={"source": "edge_case", "type": "multi_domain"}
            )
            
            edge_cases.append(example)
        
        return edge_cases
    
    def generate_dataset(self, num_examples: int = 5000, version: str = "1.0") -> TrainingDataset:
        """Generate a complete training dataset."""
        print(f"Generating {num_examples} training examples...")
        
        examples = self.generate_training_examples(num_examples)
        
        dataset = TrainingDataset(
            examples=examples,
            created_at=time.time(),
            version=version,
            description=f"Comprehensive training dataset with {len(examples)} examples"
        )
        
        print(f"Generated {len(examples)} training examples")
        return dataset
    
    def save_dataset(self, dataset: TrainingDataset, filename: str = None):
        """Save dataset to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"comprehensive_training_data_{timestamp}.json"
        
        filepath = Path("training_data") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        return filepath

def main():
    """Main function for generating training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for workflow detection")
    parser.add_argument("--examples", type=int, default=5000,
                       help="Number of training examples to generate")
    parser.add_argument("--version", default="1.0",
                       help="Dataset version")
    parser.add_argument("--output", default=None,
                       help="Output filename")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = TrainingDataGenerator()
    dataset = generator.generate_dataset(args.examples, args.version)
    
    # Save dataset
    filepath = generator.save_dataset(dataset, args.output)
    
    # Print statistics
    stats = {
        "total_examples": len(dataset.examples),
        "workflow_distribution": {},
        "complexity_distribution": {},
        "source_distribution": {}
    }
    
    from collections import Counter
    stats["workflow_distribution"] = Counter(ex.correct_workflow.value for ex in dataset.examples)
    stats["complexity_distribution"] = Counter(ex.complexity.value for ex in dataset.examples)
    stats["source_distribution"] = Counter(ex.metadata.get("source", "unknown") for ex in dataset.examples)
    
    print(f"\nDataset Statistics:")
    print(f"Total Examples: {stats['total_examples']}")
    print(f"Workflow Distribution: {dict(stats['workflow_distribution'])}")
    print(f"Complexity Distribution: {dict(stats['complexity_distribution'])}")
    print(f"Source Distribution: {dict(stats['source_distribution'])}")
    
    print(f"\nTraining data generation completed!")
    print(f"Dataset saved to: {filepath}")

if __name__ == "__main__":
    main()
