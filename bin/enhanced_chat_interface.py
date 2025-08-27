#!/usr/bin/env python3
"""
Enhanced Chat Interface for Cybersecurity Agent
Provides multi-modal chat with visual feedback and interactive elements
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message type enumeration."""
    TEXT = "text"
    VISUAL = "visual"
    INTERACTIVE = "interactive"
    DATA = "data"
    COMMAND = "command"

class VisualType(Enum):
    """Visual element type enumeration."""
    CHART = "chart"
    GRAPH = "graph"
    PROGRESS = "progress"
    STATUS = "status"
    METRIC = "metric"
    TIMELINE = "timeline"

class InteractionType(Enum):
    """Interaction element type enumeration."""
    BUTTON = "button"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    INPUT = "input"
    SELECTION = "selection"

@dataclass
class ChatMessage:
    """Enhanced chat message with multi-modal support."""
    message_id: str
    content: str
    message_type: MessageType
    timestamp: datetime
    sender: str
    metadata: Dict[str, Any]
    visual_elements: List[Dict[str, Any]] = None
    interactive_elements: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class VisualElement:
    """Visual element for chat messages."""
    element_id: str
    visual_type: VisualType
    title: str
    description: str
    data: Any
    style: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['visual_type'] = self.visual_type.value
        return data

@dataclass
class InteractiveElement:
    """Interactive element for chat messages."""
    element_id: str
    interaction_type: InteractionType
    label: str
    options: List[str] = None
    default_value: Any = None
    callback: str = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['interaction_type'] = self.interaction_type.value
        return data

class MessageProcessor:
    """Process and enhance chat messages."""
    
    def __init__(self):
        self.processing_rules = self._load_processing_rules()
        self.enhancement_engines = self._load_enhancement_engines()
    
    def _load_processing_rules(self) -> Dict[str, Any]:
        """Load message processing rules."""
        return {
            "threat_analysis": {
                "visuals": ["threat_timeline", "attack_vectors", "risk_heatmap"],
                "interactions": ["threat_details", "mitigation_options", "risk_assessment"]
            },
            "incident_response": {
                "visuals": ["incident_timeline", "affected_systems", "response_progress"],
                "interactions": ["incident_details", "response_actions", "escalation"]
            },
            "compliance": {
                "visuals": ["compliance_score", "policy_coverage", "gap_analysis"],
                "interactions": ["policy_details", "compliance_check", "remediation"]
            },
            "data_processing": {
                "visuals": ["data_summary", "processing_progress", "quality_metrics"],
                "interactions": ["data_preview", "filter_options", "export_format"]
            }
        }
    
    def _load_enhancement_engines(self) -> List[Callable]:
        """Load message enhancement engines."""
        return [
            self._enhance_with_visuals,
            self._enhance_with_interactions,
            self._enhance_with_metadata
        ]
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> ChatMessage:
        """Process and enhance a chat message."""
        # Create base message
        chat_message = ChatMessage(
            message_id=f"msg_{hashlib.sha256(f'{message}_{datetime.now()}'.encode()).hexdigest()[:8]}",
            content=message,
            message_type=MessageType.TEXT,
            timestamp=datetime.now(),
            sender="user",
            metadata={},
            visual_elements=[],
            interactive_elements=[]
        )
        
        # Enhance message based on content and context
        enhanced_message = await self._enhance_message(chat_message, context)
        
        return enhanced_message
    
    async def _enhance_message(self, message: ChatMessage, context: Dict[str, Any]) -> ChatMessage:
        """Enhance message with visuals and interactions."""
        # Analyze message content to determine enhancement type
        enhancement_type = await self._determine_enhancement_type(message.content, context)
        
        # Apply enhancements
        for engine in self.enhancement_engines:
            try:
                message = await engine(message, enhancement_type, context)
            except Exception as e:
                logger.error(f"Enhancement engine failed: {e}")
        
        return message
    
    async def _determine_enhancement_type(self, content: str, context: Dict[str, Any]) -> str:
        """Determine what type of enhancement to apply."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["threat", "attack", "malware", "apt"]):
            return "threat_analysis"
        elif any(word in content_lower for word in ["incident", "breach", "response"]):
            return "incident_response"
        elif any(word in content_lower for word in ["compliance", "policy", "regulation"]):
            return "compliance"
        elif any(word in content_lower for word in ["data", "csv", "json", "process"]):
            return "data_processing"
        else:
            return "general"
    
    async def _enhance_with_visuals(self, message: ChatMessage, enhancement_type: str, 
                                   context: Dict[str, Any]) -> ChatMessage:
        """Enhance message with visual elements."""
        if enhancement_type in self.processing_rules:
            visual_types = self.processing_rules[enhancement_type].get("visuals", [])
            
            for visual_type in visual_types:
                visual_element = await self._create_visual_element(visual_type, context)
                if visual_element:
                    message.visual_elements.append(visual_element)
        
        return message
    
    async def _enhance_with_interactions(self, message: ChatMessage, enhancement_type: str, 
                                       context: Dict[str, Any]) -> ChatMessage:
        """Enhance message with interactive elements."""
        if enhancement_type in self.processing_rules:
            interaction_types = self.processing_rules[enhancement_type].get("interactions", [])
            
            for interaction_type in interaction_types:
                interactive_element = await self._create_interactive_element(interaction_type, context)
                if interactive_element:
                    message.interactive_elements.append(interactive_element)
        
        return message
    
    async def _enhance_with_metadata(self, message: ChatMessage, enhancement_type: str, 
                                    context: Dict[str, Any]) -> ChatMessage:
        """Enhance message with metadata."""
        message.metadata.update({
            "enhancement_type": enhancement_type,
            "processing_timestamp": datetime.now().isoformat(),
            "context_keys": list(context.keys()),
            "enhancement_applied": True
        })
        
        return message
    
    async def _create_visual_element(self, visual_type: str, context: Dict[str, Any]) -> Optional[VisualElement]:
        """Create a visual element based on type."""
        try:
            if visual_type == "threat_timeline":
                return await self._create_threat_timeline(context)
            elif visual_type == "attack_vectors":
                return await self._create_attack_vectors_chart(context)
            elif visual_type == "risk_heatmap":
                return await self._create_risk_heatmap(context)
            elif visual_type == "incident_timeline":
                return await self._create_incident_timeline(context)
            elif visual_type == "compliance_score":
                return await self._create_compliance_score_chart(context)
            elif visual_type == "data_summary":
                return await self._create_data_summary_chart(context)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create visual element {visual_type}: {e}")
            return None
    
    async def _create_interactive_element(self, interaction_type: str, 
                                        context: Dict[str, Any]) -> Optional[InteractiveElement]:
        """Create an interactive element based on type."""
        try:
            if interaction_type == "threat_details":
                return InteractiveElement(
                    element_id=f"interact_{interaction_type}_{hashlib.sha256(interaction_type.encode()).hexdigest()[:8]}",
                    interaction_type=InteractionType.BUTTON,
                    label="View Threat Details",
                    callback="show_threat_details",
                    metadata={"action": "expand_threat_info"}
                )
            elif interaction_type == "mitigation_options":
                return InteractiveElement(
                    element_id=f"interact_{interaction_type}_{hashlib.sha256(interaction_type.encode()).hexdigest()[:8]}",
                    interaction_type=InteractionType.DROPDOWN,
                    label="Select Mitigation Strategy",
                    options=["Immediate", "Short-term", "Long-term"],
                    default_value="Immediate",
                    callback="select_mitigation",
                    metadata={"action": "mitigation_selection"}
                )
            elif interaction_type == "data_preview":
                return InteractiveElement(
                    element_id=f"interact_{interaction_type}_{hashlib.sha256(interaction_type.encode()).hexdigest()[:8]}",
                    interaction_type=InteractionType.BUTTON,
                    label="Preview Data",
                    callback="show_data_preview",
                    metadata={"action": "data_preview"}
                )
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create interactive element {interaction_type}: {e}")
            return None
    
    async def _create_threat_timeline(self, context: Dict[str, Any]) -> VisualElement:
        """Create threat timeline visualization."""
        # Generate sample timeline data
        timeline_data = {
            "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "threat_levels": [0.3, 0.7, 0.9, 0.6],
            "events": ["Initial detection", "Escalation", "Peak threat", "Containment"]
        }
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(timeline_data["timestamps"])), timeline_data["threat_levels"], 
                marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Threat Level")
        ax.set_title("Threat Timeline")
        ax.set_xticks(range(len(timeline_data["timestamps"])))
        ax.set_xticklabels(timeline_data["timestamps"], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_threat_timeline_{hashlib.sha256('threat_timeline'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.TIMELINE,
            title="Threat Timeline",
            description="Timeline of threat events and levels",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "line", "data_source": "threat_intelligence"}
        )
    
    async def _create_attack_vectors_chart(self, context: Dict[str, Any]) -> VisualElement:
        """Create attack vectors visualization."""
        # Generate sample attack vector data
        attack_vectors = ["Phishing", "Malware", "Social Engineering", "DDoS", "SQL Injection"]
        frequencies = [35, 25, 20, 15, 5]
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(attack_vectors, frequencies, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
        ax.set_xlabel("Attack Vector")
        ax.set_ylabel("Frequency (%)")
        ax.set_title("Attack Vector Distribution")
        ax.set_ylim(0, max(frequencies) * 1.1)
        
        # Add value labels on bars
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{freq}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_attack_vectors_{hashlib.sha256('attack_vectors'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.CHART,
            title="Attack Vector Distribution",
            description="Distribution of attack vectors by frequency",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "bar", "data_source": "threat_intelligence"}
        )
    
    async def _create_risk_heatmap(self, context: Dict[str, Any]) -> VisualElement:
        """Create risk heatmap visualization."""
        # Generate sample risk data
        risk_matrix = np.array([
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.2, 0.4, 0.6, 0.8, 0.9],
            [0.3, 0.5, 0.7, 0.8, 0.9],
            [0.4, 0.6, 0.8, 0.9, 0.9],
            [0.5, 0.7, 0.9, 0.9, 0.9]
        ])
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Add labels
        ax.set_xlabel("Impact Level")
        ax.set_ylabel("Probability")
        ax.set_title("Risk Heatmap")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Risk Level", rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{risk_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black" if risk_matrix[i, j] < 0.5 else "white")
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_risk_heatmap_{hashlib.sha256('risk_heatmap'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.METRIC,
            title="Risk Heatmap",
            description="Risk assessment matrix showing probability vs impact",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "heatmap", "data_source": "risk_assessment"}
        )
    
    async def _create_incident_timeline(self, context: Dict[str, Any]) -> VisualElement:
        """Create incident timeline visualization."""
        # Generate sample incident data
        incident_data = {
            "phases": ["Detection", "Analysis", "Containment", "Eradication", "Recovery"],
            "durations": [2, 4, 6, 8, 12],
            "status": ["Completed", "Completed", "In Progress", "Pending", "Pending"]
        }
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(incident_data["phases"]))
        
        # Create horizontal bars
        bars = ax.barh(y_pos, incident_data["durations"], 
                      color=['#2ecc71', '#2ecc71', '#f39c12', '#e74c3c', '#e74c3c'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(incident_data["phases"])
        ax.set_xlabel("Duration (hours)")
        ax.set_title("Incident Response Timeline")
        
        # Add value labels
        for i, (bar, duration) in enumerate(zip(bars, incident_data["durations"])):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{duration}h', ha='left', va='center')
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_incident_timeline_{hashlib.sha256('incident_timeline'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.TIMELINE,
            title="Incident Response Timeline",
            description="Timeline of incident response phases and durations",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "horizontal_bar", "data_source": "incident_management"}
        )
    
    async def _create_compliance_score_chart(self, context: Dict[str, Any]) -> VisualElement:
        """Create compliance score visualization."""
        # Generate sample compliance data
        compliance_data = {
            "categories": ["Access Control", "Data Protection", "Network Security", "Incident Response"],
            "scores": [85, 92, 78, 88]
        }
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(compliance_data["categories"], compliance_data["scores"], 
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_xlabel("Compliance Category")
        ax.set_ylabel("Compliance Score (%)")
        ax.set_title("Compliance Score by Category")
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, compliance_data["scores"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_compliance_score_{hashlib.sha256('compliance_score'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.METRIC,
            title="Compliance Score by Category",
            description="Compliance scores across different security categories",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "bar", "data_source": "compliance_audit"}
        )
    
    async def _create_data_summary_chart(self, context: Dict[str, Any]) -> VisualElement:
        """Create data summary visualization."""
        # Generate sample data summary
        summary_data = {
            "metrics": ["Total Records", "Valid Records", "Missing Values", "Duplicates"],
            "values": [10000, 9500, 500, 200],
            "percentages": [100, 95, 5, 2]
        }
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart for absolute values
        bars1 = ax1.bar(summary_data["metrics"], summary_data["values"], 
                       color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax1.set_title("Data Summary - Absolute Values")
        ax1.set_ylabel("Count")
        
        # Add value labels
        for bar, value in zip(bars1, summary_data["values"]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{value:,}', ha='center', va='bottom')
        
        # Pie chart for percentages
        ax2.pie(summary_data["percentages"], labels=summary_data["metrics"], 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title("Data Summary - Percentages")
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return VisualElement(
            element_id=f"visual_data_summary_{hashlib.sha256('data_summary'.encode()).hexdigest()[:8]}",
            visual_type=VisualType.CHART,
            title="Data Summary",
            description="Summary of data quality metrics and statistics",
            data=img_str,
            style={"width": "100%", "height": "auto"},
            metadata={"chart_type": "mixed", "data_source": "data_analysis"}
        )

class InteractionHandler:
    """Handle interactive elements in chat messages."""
    
    def __init__(self):
        self.interaction_handlers = self._load_interaction_handlers()
    
    def _load_interaction_handlers(self) -> Dict[str, Callable]:
        """Load interaction handler functions."""
        return {
            "show_threat_details": self._handle_threat_details,
            "select_mitigation": self._handle_mitigation_selection,
            "show_data_preview": self._handle_data_preview,
            "expand_threat_info": self._handle_expand_threat_info
        }
    
    async def create_elements(self, response: str, context: Dict[str, Any]) -> List[InteractiveElement]:
        """Create interactive elements based on response content."""
        elements = []
        
        # Analyze response to determine what interactions would be useful
        if "threat" in response.lower():
            elements.append(InteractiveElement(
                element_id=f"interact_threat_analysis_{hashlib.sha256('threat'.encode()).hexdigest()[:8]}",
                interaction_type=InteractionType.BUTTON,
                label="Analyze Threat",
                callback="analyze_threat",
                metadata={"action": "threat_analysis"}
            ))
        
        if "data" in response.lower():
            elements.append(InteractiveElement(
                element_id=f"interact_data_export_{hashlib.sha256('data'.encode()).hexdigest()[:8]}",
                interaction_type=InteractionType.DROPDOWN,
                label="Export Format",
                options=["CSV", "JSON", "PDF", "Excel"],
                default_value="CSV",
                callback="export_data",
                metadata={"action": "data_export"}
            ))
        
        if "workflow" in response.lower():
            elements.append(InteractiveElement(
                element_id=f"interact_workflow_control_{hashlib.sha256('workflow'.encode()).hexdigest()[:8]}",
                interaction_type=InteractionType.BUTTON,
                label="Control Workflow",
                callback="workflow_control",
                metadata={"action": "workflow_management"}
            ))
        
        return elements
    
    async def _handle_threat_details(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle threat details interaction."""
        return {
            "action": "show_threat_details",
            "result": "Expanded threat information displayed",
            "metadata": {"interaction_type": "threat_details"}
        }
    
    async def _handle_mitigation_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mitigation strategy selection."""
        selected_strategy = context.get("selected_strategy", "Immediate")
        return {
            "action": "mitigation_selection",
            "result": f"Mitigation strategy '{selected_strategy}' selected",
            "metadata": {"interaction_type": "mitigation_selection", "strategy": selected_strategy}
        }
    
    async def _handle_data_preview(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data preview interaction."""
        return {
            "action": "show_data_preview",
            "result": "Data preview displayed",
            "metadata": {"interaction_type": "data_preview"}
        }
    
    async def _handle_expand_threat_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle expand threat information interaction."""
        return {
            "action": "expand_threat_info",
            "result": "Threat information expanded",
            "metadata": {"interaction_type": "expand_threat_info"}
        }

class EnhancedChatInterface:
    """Main enhanced chat interface with multi-modal support."""
    
    def __init__(self):
        self.message_processor = MessageProcessor()
        self.visual_generator = VisualGenerator()
        self.interaction_handler = InteractionHandler()
        self.chat_history: List[ChatMessage] = []
        self.chat_db_path = Path("knowledge-objects/enhanced_chat.db")
        self.chat_db_path.parent.mkdir(exist_ok=True)
        self._init_chat_db()
    
    def _init_chat_db(self):
        """Initialize chat database."""
        try:
            with sqlite3.connect(self.chat_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sender TEXT NOT NULL,
                        metadata TEXT,
                        visual_elements TEXT,
                        interactive_elements TEXT
                    )
                """)
        except Exception as e:
            logger.warning(f"Chat database initialization failed: {e}")
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> ChatResponse:
        """Process message with enhanced visual and interactive elements."""
        # Process text message
        text_response = await self.message_processor.process_message(message, context)
        
        # Generate visual elements
        visual_elements = await self.visual_generator.generate_elements(text_response, context)
        
        # Create interactive components
        interactive_elements = await self.interaction_handler.create_elements(text_response, context)
        
        # Create enhanced response
        response = ChatResponse(
            text=text_response,
            visuals=visual_elements,
            interactions=interactive_elements,
            metadata=await self._generate_metadata(context)
        )
        
        # Store in chat history
        self.chat_history.append(text_response)
        
        return response
    
    async def _generate_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for chat response."""
        return {
            "processing_timestamp": datetime.now().isoformat(),
            "context_size": len(context),
            "enhancement_applied": True,
            "response_type": "enhanced"
        }

class VisualGenerator:
    """Generate charts, graphs, and visual aids."""
    
    def __init__(self):
        self.visual_templates = self._load_visual_templates()
    
    def _load_visual_templates(self) -> Dict[str, Any]:
        """Load visual generation templates."""
        return {
            "threat": ["timeline", "heatmap", "distribution"],
            "incident": ["timeline", "status", "progress"],
            "compliance": ["score", "coverage", "gaps"],
            "data": ["summary", "quality", "trends"]
        }
    
    async def generate_elements(self, response: str, context: Dict[str, Any]) -> List[VisualElement]:
        """Generate relevant visual elements."""
        elements = []
        
        # Determine what type of visuals to generate based on response content
        if "threat" in response.lower():
            threat_elements = await self._generate_threat_visuals(context)
            elements.extend(threat_elements)
        
        if "incident" in response.lower():
            incident_elements = await self._generate_incident_visuals(context)
            elements.extend(incident_elements)
        
        if "compliance" in response.lower():
            compliance_elements = await self._generate_compliance_visuals(context)
            elements.extend(compliance_elements)
        
        if "data" in response.lower():
            data_elements = await self._generate_data_visuals(context)
            elements.extend(data_elements)
        
        return elements
    
    async def _generate_threat_visuals(self, context: Dict[str, Any]) -> List[VisualElement]:
        """Generate threat-related visual elements."""
        # This would integrate with the visual generation methods from MessageProcessor
        # For now, return empty list
        return []
    
    async def _generate_incident_visuals(self, context: Dict[str, Any]) -> List[VisualElement]:
        """Generate incident-related visual elements."""
        return []
    
    async def _generate_compliance_visuals(self, context: Dict[str, Any]) -> List[VisualElement]:
        """Generate compliance-related visual elements."""
        return []
    
    async def _generate_data_visuals(self, context: Dict[str, Any]) -> List[VisualElement]:
        """Generate data-related visual elements."""
        return []

@dataclass
class ChatResponse:
    """Enhanced chat response with multiple modalities."""
    text: str
    visuals: List[VisualElement]
    interactions: List[InteractiveElement]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "visuals": [v.to_dict() for v in self.visuals],
            "interactions": [i.to_dict() for i in self.interactions],
            "metadata": self.metadata
        }

# Global enhanced chat interface instance
enhanced_chat_interface = EnhancedChatInterface()

# Convenience functions
async def process_enhanced_message(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Convenience function for enhanced message processing."""
    return await enhanced_chat_interface.process_message(message, context)
