#!/usr/bin/env python3
"""
Incident Response Tools - Comprehensive incident response and digital forensics capabilities

Provides tools for:
- Incident timeline reconstruction
- Evidence collection and preservation
- Chain of custody management
- Incident classification and prioritization
- Response playbook execution
- Forensic artifact analysis
"""

import os
import json
import logging
import hashlib
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status levels."""
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"

class EvidenceType(Enum):
    """Types of digital evidence."""
    FILE = "file"
    MEMORY = "memory"
    NETWORK = "network"
    LOG = "log"
    REGISTRY = "registry"
    DATABASE = "database"
    CLOUD = "cloud"
    MOBILE = "mobile"

@dataclass
class Incident:
    """Incident data structure."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_time: datetime
    assigned_to: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Evidence:
    """Digital evidence data structure."""
    evidence_id: str
    incident_id: str
    evidence_type: EvidenceType
    source: str
    collected_time: datetime
    collected_by: str
    hash_md5: Optional[str] = None
    hash_sha1: Optional[str] = None
    hash_sha256: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_of_custody: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class TimelineEvent:
    """Timeline event data structure."""
    event_id: str
    incident_id: str
    timestamp: datetime
    event_type: str
    description: str
    source: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class IncidentResponseTools:
    """Comprehensive incident response and digital forensics tools."""
    
    def __init__(self, base_path: str = "incident_response"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.incidents_dir = self.base_path / "incidents"
        self.evidence_dir = self.base_path / "evidence"
        self.timelines_dir = self.base_path / "timelines"
        self.playbooks_dir = self.base_path / "playbooks"
        
        for directory in [self.incidents_dir, self.evidence_dir, self.timelines_dir, self.playbooks_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize incident counter
        self.incident_counter = self._get_next_incident_number()
        
        logger.info("🚀 Incident Response Tools initialized")
    
    def _get_next_incident_number(self) -> int:
        """Get the next incident number."""
        try:
            existing_incidents = list(self.incidents_dir.glob("incident_*.json"))
            if existing_incidents:
                numbers = []
                for incident_file in existing_incidents:
                    try:
                        number = int(incident_file.stem.split('_')[1])
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) + 1 if numbers else 1
            return 1
        except Exception:
            return 1
    
    def create_incident(self, title: str, description: str, severity: IncidentSeverity,
                       affected_systems: List[str] = None, indicators: List[str] = None,
                       tags: List[str] = None, metadata: Dict[str, Any] = None) -> Incident:
        """Create a new incident."""
        incident_id = f"INC-{self.incident_counter:06d}"
        self.incident_counter += 1
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.NEW,
            created_time=datetime.now(timezone.utc),
            affected_systems=affected_systems or [],
            indicators=indicators or [],
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Save incident
        self._save_incident(incident)
        
        logger.info(f"✅ Created incident: {incident_id}")
        return incident
    
    def _save_incident(self, incident: Incident):
        """Save incident to file."""
        incident_file = self.incidents_dir / f"incident_{incident.incident_id}.json"
        with open(incident_file, 'w') as f:
            json.dump(asdict(incident), f, indent=2, default=str)
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        incident_file = self.incidents_dir / f"incident_{incident_id}.json"
        if incident_file.exists():
            with open(incident_file, 'r') as f:
                data = json.load(f)
                return Incident(**data)
        return None
    
    def update_incident_status(self, incident_id: str, status: IncidentStatus, 
                              notes: str = None, assigned_to: str = None) -> bool:
        """Update incident status."""
        incident = self.get_incident(incident_id)
        if incident:
            incident.status = status
            if assigned_to:
                incident.assigned_to = assigned_to
            
            # Add timeline event
            timeline_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "status_change",
                "description": f"Status changed to {status.value}",
                "source": "incident_response_tools",
                "confidence": 1.0,
                "metadata": {
                    "old_status": incident.status.value if hasattr(incident, 'old_status') else None,
                    "new_status": status.value,
                    "notes": notes,
                    "assigned_to": assigned_to
                }
            }
            incident.timeline.append(timeline_event)
            
            self._save_incident(incident)
            logger.info(f"✅ Updated incident {incident_id} status to {status.value}")
            return True
        return False
    
    def collect_evidence(self, incident_id: str, evidence_type: EvidenceType, 
                        source: str, collected_by: str, file_path: str = None,
                        metadata: Dict[str, Any] = None) -> Evidence:
        """Collect digital evidence."""
        evidence_id = f"EVID-{uuid.uuid4().hex[:8].upper()}"
        
        evidence = Evidence(
            evidence_id=evidence_id,
            incident_id=incident_id,
            evidence_type=evidence_type,
            source=source,
            collected_time=datetime.now(timezone.utc),
            collected_by=collected_by,
            file_path=file_path,
            metadata=metadata or {}
        )
        
        # Calculate hashes if file exists
        if file_path and Path(file_path).exists():
            evidence.hash_md5, evidence.hash_sha1, evidence.hash_sha256 = self._calculate_file_hashes(file_path)
        
        # Add to chain of custody
        evidence.chain_of_custody.append({
            "timestamp": evidence.collected_time.isoformat(),
            "action": "collected",
            "performed_by": collected_by,
            "location": file_path or source,
            "notes": f"Evidence collected from {source}"
        })
        
        # Save evidence
        self._save_evidence(evidence)
        
        # Update incident
        incident = self.get_incident(incident_id)
        if incident:
            incident.evidence.append({
                "evidence_id": evidence_id,
                "evidence_type": evidence_type.value,
                "collected_time": evidence.collected_time.isoformat(),
                "source": source
            })
            self._save_incident(incident)
        
        logger.info(f"✅ Collected evidence: {evidence_id}")
        return evidence
    
    def _calculate_file_hashes(self, file_path: str) -> Tuple[str, str, str]:
        """Calculate file hashes."""
        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha1_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha1_hash.hexdigest(), sha256_hash.hexdigest()
    
    def _save_evidence(self, evidence: Evidence):
        """Save evidence to file."""
        evidence_file = self.evidence_dir / f"evidence_{evidence.evidence_id}.json"
        with open(evidence_file, 'w') as f:
            json.dump(asdict(evidence), f, indent=2, default=str)
    
    def add_timeline_event(self, incident_id: str, event_type: str, description: str,
                          source: str, timestamp: datetime = None, confidence: float = 1.0,
                          tags: List[str] = None, metadata: Dict[str, Any] = None) -> TimelineEvent:
        """Add timeline event to incident."""
        event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        
        timeline_event = TimelineEvent(
            event_id=event_id,
            incident_id=incident_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            event_type=event_type,
            description=description,
            source=source,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Update incident timeline
        incident = self.get_incident(incident_id)
        if incident:
            incident.timeline.append({
                "event_id": event_id,
                "timestamp": timeline_event.timestamp.isoformat(),
                "event_type": event_type,
                "description": description,
                "source": source,
                "confidence": confidence,
                "tags": tags or [],
                "metadata": metadata or {}
            })
            self._save_incident(incident)
        
        logger.info(f"✅ Added timeline event: {event_id}")
        return timeline_event
    
    def generate_incident_report(self, incident_id: str) -> Dict[str, Any]:
        """Generate comprehensive incident report."""
        incident = self.get_incident(incident_id)
        if not incident:
            return {"error": f"Incident {incident_id} not found"}
        
        # Get all evidence for this incident
        evidence_files = list(self.evidence_dir.glob(f"*{incident_id}*"))
        evidence_list = []
        for evidence_file in evidence_files:
            with open(evidence_file, 'r') as f:
                evidence_data = json.load(f)
                evidence_list.append(evidence_data)
        
        # Generate report
        report = {
            "incident_id": incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_time": incident.created_time.isoformat(),
            "assigned_to": incident.assigned_to,
            "affected_systems": incident.affected_systems,
            "indicators": incident.indicators,
            "tags": incident.tags,
            "timeline": incident.timeline,
            "evidence_count": len(evidence_list),
            "evidence": evidence_list,
            "metadata": incident.metadata,
            "report_generated": datetime.now(timezone.utc).isoformat()
        }
        
        # Save report
        report_file = self.incidents_dir / f"report_{incident_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Generated incident report: {incident_id}")
        return report
    
    def search_incidents(self, query: str = None, severity: IncidentSeverity = None,
                        status: IncidentStatus = None, tags: List[str] = None) -> List[Incident]:
        """Search incidents by various criteria."""
        incidents = []
        
        for incident_file in self.incidents_dir.glob("incident_*.json"):
            try:
                with open(incident_file, 'r') as f:
                    data = json.load(f)
                    incident = Incident(**data)
                    
                    # Apply filters
                    if query and query.lower() not in incident.title.lower() and query.lower() not in incident.description.lower():
                        continue
                    
                    if severity and incident.severity != severity:
                        continue
                    
                    if status and incident.status != status:
                        continue
                    
                    if tags and not any(tag in incident.tags for tag in tags):
                        continue
                    
                    incidents.append(incident)
                    
            except Exception as e:
                logger.warning(f"Error loading incident file {incident_file}: {e}")
        
        return incidents
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident statistics."""
        incidents = []
        for incident_file in self.incidents_dir.glob("incident_*.json"):
            try:
                with open(incident_file, 'r') as f:
                    data = json.load(f)
                    incidents.append(Incident(**data))
            except Exception:
                continue
        
        # Calculate statistics
        total_incidents = len(incidents)
        severity_counts = {}
        status_counts = {}
        
        for incident in incidents:
            severity_counts[incident.severity.value] = severity_counts.get(incident.severity.value, 0) + 1
            status_counts[incident.status.value] = status_counts.get(incident.status.value, 0) + 1
        
        return {
            "total_incidents": total_incidents,
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "average_incidents_per_day": total_incidents / max(1, (datetime.now(timezone.utc) - datetime(2024, 1, 1, tzinfo=timezone.utc)).days)
        }
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Incident Response Tools cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize tools
    irt = IncidentResponseTools()
    
    # Create incident
    incident = irt.create_incident(
        title="Suspicious Network Activity",
        description="Detected unusual network traffic patterns",
        severity=IncidentSeverity.HIGH,
        affected_systems=["web-server-01", "db-server-01"],
        indicators=["192.168.1.100", "malicious-domain.com"],
        tags=["network", "suspicious"]
    )
    
    print(f"✅ Created incident: {incident.incident_id}")
    
    # Add timeline event
    irt.add_timeline_event(
        incident_id=incident.incident_id,
        event_type="detection",
        description="Anomalous network traffic detected",
        source="network_monitor",
        confidence=0.8
    )
    
    # Collect evidence
    evidence = irt.collect_evidence(
        incident_id=incident.incident_id,
        evidence_type=EvidenceType.NETWORK,
        source="network_capture.pcap",
        collected_by="analyst_john",
        metadata={"capture_duration": "1 hour", "packet_count": 10000}
    )
    
    print(f"✅ Collected evidence: {evidence.evidence_id}")
    
    # Update incident status
    irt.update_incident_status(incident.incident_id, IncidentStatus.IN_PROGRESS, "Investigation started")
    
    # Generate report
    report = irt.generate_incident_report(incident.incident_id)
    print(f"✅ Generated report with {report['evidence_count']} evidence items")
    
    # Get statistics
    stats = irt.get_incident_statistics()
    print(f"✅ Total incidents: {stats['total_incidents']}")
    
    irt.cleanup()
