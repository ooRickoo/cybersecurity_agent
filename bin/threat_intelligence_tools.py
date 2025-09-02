#!/usr/bin/env python3
"""
Threat Intelligence Tools - Comprehensive threat intelligence and IOC analysis capabilities

Provides tools for:
- IOC (Indicators of Compromise) analysis and correlation
- Threat actor profiling and attribution
- Campaign analysis and tracking
- Threat intelligence feed integration
- Malware family classification
- Attack pattern recognition
- Threat landscape analysis
"""

import os
import json
import logging
import hashlib
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import re

logger = logging.getLogger(__name__)

class IOCType(Enum):
    """Types of Indicators of Compromise."""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    EMAIL = "email"
    FILE_HASH = "file_hash"
    REGISTRY_KEY = "registry_key"
    MUTEX = "mutex"
    CERTIFICATE = "certificate"
    USER_AGENT = "user_agent"
    CVE = "cve"

class ThreatLevel(Enum):
    """Threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class IOC:
    """Indicator of Compromise data structure."""
    ioc_id: str
    ioc_type: IOCType
    value: str
    threat_level: ThreatLevel
    confidence: ConfidenceLevel
    first_seen: datetime
    last_seen: datetime
    source: str
    description: str
    tags: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    threat_actors: List[str] = field(default_factory=list)
    malware_families: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatActor:
    """Threat actor data structure."""
    actor_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    country: Optional[str] = None
    motivation: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    iocs: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Campaign:
    """Campaign data structure."""
    campaign_id: str
    name: str
    description: str
    threat_actors: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    targets: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    iocs: List[str] = field(default_factory=list)
    malware_families: List[str] = field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MalwareFamily:
    """Malware family data structure."""
    family_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    category: str = ""
    capabilities: List[str] = field(default_factory=list)
    iocs: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    threat_actors: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ThreatIntelligenceTools:
    """Comprehensive threat intelligence and IOC analysis tools."""
    
    def __init__(self, base_path: str = "threat_intelligence"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.iocs_dir = self.base_path / "iocs"
        self.actors_dir = self.base_path / "actors"
        self.campaigns_dir = self.base_path / "campaigns"
        self.malware_dir = self.base_path / "malware"
        self.feeds_dir = self.base_path / "feeds"
        
        for directory in [self.iocs_dir, self.actors_dir, self.campaigns_dir, self.malware_dir, self.feeds_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        self.ioc_counter = self._get_next_ioc_number()
        self.actor_counter = self._get_next_actor_number()
        self.campaign_counter = self._get_next_campaign_number()
        self.malware_counter = self._get_next_malware_number()
        
        # Load existing data
        self._load_existing_data()
        
        logger.info("🚀 Threat Intelligence Tools initialized")
    
    def _get_next_ioc_number(self) -> int:
        """Get the next IOC number."""
        try:
            existing_iocs = list(self.iocs_dir.glob("ioc_*.json"))
            if existing_iocs:
                numbers = []
                for ioc_file in existing_iocs:
                    try:
                        number = int(ioc_file.stem.split('_')[1])
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) + 1 if numbers else 1
            return 1
        except Exception:
            return 1
    
    def _get_next_actor_number(self) -> int:
        """Get the next actor number."""
        try:
            existing_actors = list(self.actors_dir.glob("actor_*.json"))
            if existing_actors:
                numbers = []
                for actor_file in existing_actors:
                    try:
                        number = int(actor_file.stem.split('_')[1])
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) + 1 if numbers else 1
            return 1
        except Exception:
            return 1
    
    def _get_next_campaign_number(self) -> int:
        """Get the next campaign number."""
        try:
            existing_campaigns = list(self.campaigns_dir.glob("campaign_*.json"))
            if existing_campaigns:
                numbers = []
                for campaign_file in existing_campaigns:
                    try:
                        number = int(campaign_file.stem.split('_')[1])
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) + 1 if numbers else 1
            return 1
        except Exception:
            return 1
    
    def _get_next_malware_number(self) -> int:
        """Get the next malware number."""
        try:
            existing_malware = list(self.malware_dir.glob("malware_*.json"))
            if existing_malware:
                numbers = []
                for malware_file in existing_malware:
                    try:
                        number = int(malware_file.stem.split('_')[1])
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) + 1 if numbers else 1
            return 1
        except Exception:
            return 1
    
    def _load_existing_data(self):
        """Load existing threat intelligence data."""
        # This would load existing data from files
        # For now, we'll just initialize empty structures
        self.iocs = {}
        self.actors = {}
        self.campaigns = {}
        self.malware_families = {}
    
    def add_ioc(self, ioc_type: IOCType, value: str, threat_level: ThreatLevel,
                confidence: ConfidenceLevel, source: str, description: str,
                tags: List[str] = None, campaigns: List[str] = None,
                threat_actors: List[str] = None, malware_families: List[str] = None,
                metadata: Dict[str, Any] = None) -> IOC:
        """Add a new IOC."""
        ioc_id = f"IOC-{self.ioc_counter:06d}"
        self.ioc_counter += 1
        
        # Validate IOC value based on type
        if not self._validate_ioc_value(ioc_type, value):
            raise ValueError(f"Invalid {ioc_type.value} value: {value}")
        
        ioc = IOC(
            ioc_id=ioc_id,
            ioc_type=ioc_type,
            value=value,
            threat_level=threat_level,
            confidence=confidence,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            source=source,
            description=description,
            tags=tags or [],
            campaigns=campaigns or [],
            threat_actors=threat_actors or [],
            malware_families=malware_families or [],
            metadata=metadata or {}
        )
        
        # Save IOC
        self._save_ioc(ioc)
        
        logger.info(f"✅ Added IOC: {ioc_id}")
        return ioc
    
    def _validate_ioc_value(self, ioc_type: IOCType, value: str) -> bool:
        """Validate IOC value based on type."""
        if ioc_type == IOCType.IP_ADDRESS:
            return self._is_valid_ip(value)
        elif ioc_type == IOCType.DOMAIN:
            return self._is_valid_domain(value)
        elif ioc_type == IOCType.URL:
            return self._is_valid_url(value)
        elif ioc_type == IOCType.EMAIL:
            return self._is_valid_email(value)
        elif ioc_type == IOCType.FILE_HASH:
            return self._is_valid_hash(value)
        elif ioc_type == IOCType.CVE:
            return self._is_valid_cve(value)
        else:
            return len(value) > 0
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name."""
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(pattern, domain))
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _is_valid_hash(self, hash_value: str) -> bool:
        """Validate file hash."""
        hash_lengths = [32, 40, 64]  # MD5, SHA1, SHA256
        return len(hash_value) in hash_lengths and all(c in '0123456789abcdefABCDEF' for c in hash_value)
    
    def _is_valid_cve(self, cve: str) -> bool:
        """Validate CVE identifier."""
        pattern = r'^CVE-\d{4}-\d{4,}$'
        return bool(re.match(pattern, cve))
    
    def _save_ioc(self, ioc: IOC):
        """Save IOC to file."""
        ioc_file = self.iocs_dir / f"ioc_{ioc.ioc_id}.json"
        with open(ioc_file, 'w') as f:
            json.dump(asdict(ioc), f, indent=2, default=str)
    
    def search_iocs(self, query: str = None, ioc_type: IOCType = None,
                   threat_level: ThreatLevel = None, confidence: ConfidenceLevel = None,
                   tags: List[str] = None, campaigns: List[str] = None) -> List[IOC]:
        """Search IOCs by various criteria."""
        iocs = []
        
        for ioc_file in self.iocs_dir.glob("ioc_*.json"):
            try:
                with open(ioc_file, 'r') as f:
                    data = json.load(f)
                    ioc = IOC(**data)
                    
                    # Apply filters
                    if query and query.lower() not in ioc.value.lower() and query.lower() not in ioc.description.lower():
                        continue
                    
                    if ioc_type and ioc.ioc_type != ioc_type:
                        continue
                    
                    if threat_level and ioc.threat_level != threat_level:
                        continue
                    
                    if confidence and ioc.confidence != confidence:
                        continue
                    
                    if tags and not any(tag in ioc.tags for tag in tags):
                        continue
                    
                    if campaigns and not any(campaign in ioc.campaigns for campaign in campaigns):
                        continue
                    
                    iocs.append(ioc)
                    
            except Exception as e:
                logger.warning(f"Error loading IOC file {ioc_file}: {e}")
        
        return iocs
    
    def correlate_iocs(self, ioc_values: List[str]) -> Dict[str, Any]:
        """Correlate multiple IOCs to find relationships."""
        correlations = {
            "shared_campaigns": {},
            "shared_threat_actors": {},
            "shared_malware_families": {},
            "shared_tags": {},
            "timeline": [],
            "confidence_score": 0.0
        }
        
        # Find IOCs
        found_iocs = []
        for value in ioc_values:
            iocs = self.search_iocs(query=value)
            found_iocs.extend(iocs)
        
        if len(found_iocs) < 2:
            return correlations
        
        # Analyze correlations
        all_campaigns = []
        all_actors = []
        all_malware = []
        all_tags = []
        
        for ioc in found_iocs:
            all_campaigns.extend(ioc.campaigns)
            all_actors.extend(ioc.threat_actors)
            all_malware.extend(ioc.malware_families)
            all_tags.extend(ioc.tags)
        
        # Count shared elements
        from collections import Counter
        campaign_counts = Counter(all_campaigns)
        actor_counts = Counter(all_actors)
        malware_counts = Counter(all_malware)
        tag_counts = Counter(all_tags)
        
        # Find shared elements (appearing in multiple IOCs)
        correlations["shared_campaigns"] = {k: v for k, v in campaign_counts.items() if v > 1}
        correlations["shared_threat_actors"] = {k: v for k, v in actor_counts.items() if v > 1}
        correlations["shared_malware_families"] = {k: v for k, v in malware_counts.items() if v > 1}
        correlations["shared_tags"] = {k: v for k, v in tag_counts.items() if v > 1}
        
        # Calculate confidence score
        total_shared = (len(correlations["shared_campaigns"]) + 
                       len(correlations["shared_threat_actors"]) + 
                       len(correlations["shared_malware_families"]) + 
                       len(correlations["shared_tags"]))
        
        correlations["confidence_score"] = min(1.0, total_shared / len(found_iocs))
        
        # Create timeline
        timeline_events = []
        for ioc in found_iocs:
            timeline_events.append({
                "timestamp": ioc.first_seen.isoformat(),
                "event": f"IOC {ioc.ioc_id} first seen",
                "ioc_type": ioc.ioc_type.value,
                "value": ioc.value,
                "threat_level": ioc.threat_level.value
            })
        
        correlations["timeline"] = sorted(timeline_events, key=lambda x: x["timestamp"])
        
        return correlations
    
    def add_threat_actor(self, name: str, aliases: List[str] = None, description: str = "",
                        country: str = None, motivation: List[str] = None,
                        capabilities: List[str] = None, tools: List[str] = None,
                        metadata: Dict[str, Any] = None) -> ThreatActor:
        """Add a new threat actor."""
        actor_id = f"ACTOR-{self.actor_counter:06d}"
        self.actor_counter += 1
        
        actor = ThreatActor(
            actor_id=actor_id,
            name=name,
            aliases=aliases or [],
            description=description,
            country=country,
            motivation=motivation or [],
            capabilities=capabilities or [],
            tools=tools or [],
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Save actor
        self._save_actor(actor)
        
        logger.info(f"✅ Added threat actor: {actor_id}")
        return actor
    
    def _save_actor(self, actor: ThreatActor):
        """Save threat actor to file."""
        actor_file = self.actors_dir / f"actor_{actor.actor_id}.json"
        with open(actor_file, 'w') as f:
            json.dump(asdict(actor), f, indent=2, default=str)
    
    def add_campaign(self, name: str, description: str, threat_actors: List[str] = None,
                    start_date: datetime = None, end_date: datetime = None,
                    targets: List[str] = None, techniques: List[str] = None,
                    iocs: List[str] = None, malware_families: List[str] = None,
                    metadata: Dict[str, Any] = None) -> Campaign:
        """Add a new campaign."""
        campaign_id = f"CAMP-{self.campaign_counter:06d}"
        self.campaign_counter += 1
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name=name,
            description=description,
            threat_actors=threat_actors or [],
            start_date=start_date,
            end_date=end_date,
            targets=targets or [],
            techniques=techniques or [],
            iocs=iocs or [],
            malware_families=malware_families or [],
            metadata=metadata or {}
        )
        
        # Save campaign
        self._save_campaign(campaign)
        
        logger.info(f"✅ Added campaign: {campaign_id}")
        return campaign
    
    def _save_campaign(self, campaign: Campaign):
        """Save campaign to file."""
        campaign_file = self.campaigns_dir / f"campaign_{campaign.campaign_id}.json"
        with open(campaign_file, 'w') as f:
            json.dump(asdict(campaign), f, indent=2, default=str)
    
    def add_malware_family(self, name: str, aliases: List[str] = None, description: str = "",
                          category: str = "", capabilities: List[str] = None,
                          iocs: List[str] = None, campaigns: List[str] = None,
                          threat_actors: List[str] = None, metadata: Dict[str, Any] = None) -> MalwareFamily:
        """Add a new malware family."""
        family_id = f"MAL-{self.malware_counter:06d}"
        self.malware_counter += 1
        
        malware_family = MalwareFamily(
            family_id=family_id,
            name=name,
            aliases=aliases or [],
            description=description,
            category=category,
            capabilities=capabilities or [],
            iocs=iocs or [],
            campaigns=campaigns or [],
            threat_actors=threat_actors or [],
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Save malware family
        self._save_malware_family(malware_family)
        
        logger.info(f"✅ Added malware family: {family_id}")
        return malware_family
    
    def _save_malware_family(self, malware_family: MalwareFamily):
        """Save malware family to file."""
        malware_file = self.malware_dir / f"malware_{malware_family.family_id}.json"
        with open(malware_file, 'w') as f:
            json.dump(asdict(malware_family), f, indent=2, default=str)
    
    def get_threat_landscape(self, days: int = 30) -> Dict[str, Any]:
        """Get threat landscape overview."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get recent IOCs
        recent_iocs = []
        for ioc_file in self.iocs_dir.glob("ioc_*.json"):
            try:
                with open(ioc_file, 'r') as f:
                    data = json.load(f)
                    ioc = IOC(**data)
                    if ioc.first_seen >= start_date:
                        recent_iocs.append(ioc)
            except Exception:
                continue
        
        # Analyze threat landscape
        ioc_types = {}
        threat_levels = {}
        confidence_levels = {}
        top_tags = {}
        top_campaigns = {}
        
        for ioc in recent_iocs:
            # IOC types
            ioc_types[ioc.ioc_type.value] = ioc_types.get(ioc.ioc_type.value, 0) + 1
            
            # Threat levels
            threat_levels[ioc.threat_level.value] = threat_levels.get(ioc.threat_level.value, 0) + 1
            
            # Confidence levels
            confidence_levels[ioc.confidence.value] = confidence_levels.get(ioc.confidence.value, 0) + 1
            
            # Tags
            for tag in ioc.tags:
                top_tags[tag] = top_tags.get(tag, 0) + 1
            
            # Campaigns
            for campaign in ioc.campaigns:
                top_campaigns[campaign] = top_campaigns.get(campaign, 0) + 1
        
        return {
            "period": f"{days} days",
            "total_iocs": len(recent_iocs),
            "ioc_types": ioc_types,
            "threat_levels": threat_levels,
            "confidence_levels": confidence_levels,
            "top_tags": dict(sorted(top_tags.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_campaigns": dict(sorted(top_campaigns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Threat Intelligence Tools cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize tools
    tit = ThreatIntelligenceTools()
    
    # Add IOCs
    ioc1 = tit.add_ioc(
        ioc_type=IOCType.IP_ADDRESS,
        value="192.168.1.100",
        threat_level=ThreatLevel.HIGH,
        confidence=ConfidenceLevel.HIGH,
        source="threat_feed",
        description="Malicious IP address",
        tags=["malware", "c2"],
        campaigns=["campaign_1"],
        threat_actors=["actor_1"]
    )
    
    ioc2 = tit.add_ioc(
        ioc_type=IOCType.DOMAIN,
        value="malicious-domain.com",
        threat_level=ThreatLevel.HIGH,
        confidence=ConfidenceLevel.MEDIUM,
        source="dns_analysis",
        description="Malicious domain",
        tags=["malware", "c2"],
        campaigns=["campaign_1"],
        threat_actors=["actor_1"]
    )
    
    print(f"✅ Added IOCs: {ioc1.ioc_id}, {ioc2.ioc_id}")
    
    # Correlate IOCs
    correlations = tit.correlate_iocs(["192.168.1.100", "malicious-domain.com"])
    print(f"✅ IOC correlation confidence: {correlations['confidence_score']:.2f}")
    
    # Add threat actor
    actor = tit.add_threat_actor(
        name="APT Group Alpha",
        aliases=["Alpha Group", "Group A"],
        description="Advanced persistent threat group",
        country="Unknown",
        motivation=["espionage", "financial"],
        capabilities=["spear_phishing", "lateral_movement", "data_exfiltration"]
    )
    
    print(f"✅ Added threat actor: {actor.actor_id}")
    
    # Add campaign
    campaign = tit.add_campaign(
        name="Operation Silent Strike",
        description="Long-term espionage campaign",
        threat_actors=[actor.actor_id],
        targets=["government", "defense"],
        techniques=["spear_phishing", "lateral_movement"]
    )
    
    print(f"✅ Added campaign: {campaign.campaign_id}")
    
    # Get threat landscape
    landscape = tit.get_threat_landscape(days=30)
    print(f"✅ Threat landscape: {landscape['total_iocs']} IOCs in last 30 days")
    
    tit.cleanup()
