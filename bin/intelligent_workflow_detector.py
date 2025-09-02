#!/usr/bin/env python3
"""
Intelligent Workflow Detector - Local ML-based workflow selection and optimization

Uses local machine learning models and NLP techniques to:
1. Clean and preprocess user input
2. Analyze intent and context
3. Select optimal workflows
4. Provide confidence scores and reasoning
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import pickle
from collections import Counter, defaultdict

# Local ML libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Enumeration of available workflow types."""
    PATENT_ANALYSIS = "patent_analysis"
    MALWARE_ANALYSIS = "malware_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    VULNERABILITY_SCAN = "vulnerability_scan"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_HUNTING = "threat_hunting"
    COMPLIANCE_ASSESSMENT = "compliance_assessment"
    DATA_ANALYSIS = "data_analysis"
    FILE_FORENSICS = "file_forensics"
    THREAT_INTELLIGENCE = "threat_intelligence"
    GENERAL_ANALYSIS = "general_analysis"
    INTERACTIVE = "interactive"

class InputComplexity(Enum):
    """Input complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class WorkflowRecommendation:
    """Workflow recommendation with confidence and reasoning."""
    workflow_type: WorkflowType
    confidence_score: float
    reasoning: str
    alternative_workflows: List[WorkflowType] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    estimated_duration: str = "unknown"
    complexity_level: InputComplexity = InputComplexity.MODERATE
    preprocessing_steps: List[str] = field(default_factory=list)

@dataclass
class InputAnalysis:
    """Analysis of user input."""
    cleaned_text: str
    intent: str
    entities: List[str]
    file_types: List[str]
    complexity: InputComplexity
    keywords: List[str]
    context_clues: Dict[str, Any]

class IntelligentWorkflowDetector:
    """Local ML-based workflow detection and optimization."""
    
    def __init__(self, model_cache_dir: str = "models/workflow_detection"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.nlp_model = None
        self.vectorizer = None
        self.classifier = None
        self.workflow_patterns = self._initialize_workflow_patterns()
        self.file_type_patterns = self._initialize_file_type_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()
        
        # Load or initialize models
        self._initialize_models()
        
        # Performance tracking
        self.performance_history = []
        self.pattern_cache = {}
        
    def _initialize_workflow_patterns(self) -> Dict[WorkflowType, Dict[str, Any]]:
        """Initialize workflow detection patterns."""
        return {
            WorkflowType.PATENT_ANALYSIS: {
                "keywords": ["patent", "invention", "intellectual property", "uspto", "publication number", "patent number"],
                "file_patterns": [r"\.csv$", r"patent", r"invention"],
                "intent_patterns": [r"analyze.*patent", r"look.*up.*patent", r"patent.*information"],
                "context_clues": ["publication", "inventor", "assignee", "abstract", "claims"]
            },
            WorkflowType.MALWARE_ANALYSIS: {
                "keywords": ["malware", "virus", "trojan", "ransomware", "suspicious file", "threat", "infection"],
                "file_patterns": [r"\.exe$", r"\.dll$", r"\.pdf$", r"\.doc$", r"malware", r"virus"],
                "intent_patterns": [r"analyze.*malware", r"scan.*file", r"detect.*threat", r"malware.*analysis"],
                "context_clues": ["hash", "signature", "behavior", "sandbox", "yara"]
            },
            WorkflowType.NETWORK_ANALYSIS: {
                "keywords": ["network", "traffic", "pcap", "packet", "flow", "connection", "protocol"],
                "file_patterns": [r"\.pcap$", r"\.pcapng$", r"network", r"traffic"],
                "intent_patterns": [r"analyze.*network", r"traffic.*analysis", r"packet.*capture", r"network.*forensics"],
                "context_clues": ["tcp", "udp", "icmp", "dns", "http", "https", "firewall"]
            },
            WorkflowType.VULNERABILITY_SCAN: {
                "keywords": ["vulnerability", "scan", "security", "exploit", "cve", "assessment", "penetration"],
                "file_patterns": [r"\.xml$", r"\.json$", r"scan", r"vulnerability"],
                "intent_patterns": [r"scan.*vulnerability", r"security.*assessment", r"penetration.*test", r"vulnerability.*scan"],
                "context_clues": ["port", "service", "exploit", "cve", "severity", "risk"]
            },
            WorkflowType.INCIDENT_RESPONSE: {
                "keywords": ["incident", "response", "breach", "attack", "forensics", "investigation", "timeline"],
                "file_patterns": [r"\.log$", r"\.csv$", r"\.json$", r"incident", r"forensics"],
                "intent_patterns": [r"incident.*response", r"investigate.*breach", r"forensic.*analysis", r"incident.*timeline"],
                "context_clues": ["timeline", "evidence", "ioc", "triage", "containment", "eradication"]
            },
            WorkflowType.THREAT_HUNTING: {
                "keywords": ["threat hunting", "hunt", "proactive", "ioc", "indicator", "anomaly", "behavior"],
                "file_patterns": [r"\.csv$", r"\.json$", r"\.log$", r"threat", r"hunt"],
                "intent_patterns": [r"threat.*hunt", r"proactive.*search", r"find.*anomaly", r"hunt.*threat"],
                "context_clues": ["ioc", "ttps", "mitre", "behavior", "anomaly", "baseline"]
            },
            WorkflowType.DATA_ANALYSIS: {
                "keywords": ["analyze", "data", "csv", "json", "statistics", "insights", "patterns"],
                "file_patterns": [r"\.csv$", r"\.json$", r"\.xlsx?$", r"data", r"analysis"],
                "intent_patterns": [r"analyze.*data", r"data.*analysis", r"statistics", r"insights", r"patterns"],
                "context_clues": ["columns", "rows", "statistics", "correlation", "trend", "summary"]
            },
            WorkflowType.FILE_FORENSICS: {
                "keywords": ["forensics", "file", "metadata", "timeline", "deleted", "recovery", "artifact"],
                "file_patterns": [r"\.bin$", r"\.img$", r"\.dd$", r"forensics", r"image"],
                "intent_patterns": [r"forensic.*analysis", r"file.*recovery", r"metadata.*extraction", r"timeline.*analysis"],
                "context_clues": ["metadata", "timeline", "deleted", "recovery", "artifact", "hash"]
            }
        }
    
    def _initialize_file_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize file type detection patterns."""
        return {
            "network": [".pcap", ".pcapng", ".netflow", ".sflow"],
            "malware": [".exe", ".dll", ".pdf", ".doc", ".docx", ".zip", ".rar"],
            "logs": [".log", ".txt", ".csv", ".json", ".xml"],
            "data": [".csv", ".xlsx", ".xls", ".json", ".xml", ".yaml", ".yml"],
            "documents": [".pdf", ".doc", ".docx", ".txt", ".md", ".rtf"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
            "archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
            "forensics": [".bin", ".img", ".dd", ".e01", ".aff", ".vmdk"]
        }
    
    def _initialize_complexity_indicators(self) -> Dict[InputComplexity, List[str]]:
        """Initialize complexity detection indicators."""
        return {
            InputComplexity.SIMPLE: [
                "analyze", "check", "scan", "look up", "find", "show", "list"
            ],
            InputComplexity.MODERATE: [
                "compare", "correlate", "enrich", "enhance", "process", "transform", "convert"
            ],
            InputComplexity.COMPLEX: [
                "investigate", "forensic", "timeline", "reconstruct", "analyze patterns", "detect anomalies"
            ],
            InputComplexity.EXPERT: [
                "reverse engineer", "malware analysis", "threat hunting", "incident response", "penetration test"
            ]
        }
    
    def _initialize_models(self):
        """Initialize or load ML models."""
        try:
            if SPACY_AVAILABLE:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp_model = None
            
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.classifier = MultinomialNB()
                
                # Try to load existing models
                self._load_models()
                
        except Exception as e:
            logger.warning(f"Error initializing ML models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from cache."""
        try:
            vectorizer_path = self.model_cache_dir / "vectorizer.pkl"
            classifier_path = self.model_cache_dir / "classifier.pkl"
            
            if vectorizer_path.exists() and classifier_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded pre-trained workflow detection models")
        except Exception as e:
            logger.warning(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to cache."""
        try:
            if self.vectorizer and self.classifier:
                with open(self.model_cache_dir / "vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                with open(self.model_cache_dir / "classifier.pkl", 'wb') as f:
                    pickle.dump(self.classifier, f)
                logger.info("Saved workflow detection models")
        except Exception as e:
            logger.warning(f"Error saving models: {e}")
    
    def preprocess_input(self, user_input: str, input_files: List[str] = None) -> InputAnalysis:
        """Clean and preprocess user input."""
        # Clean text
        cleaned_text = self._clean_text(user_input)
        
        # Extract entities and keywords
        entities = self._extract_entities(cleaned_text)
        keywords = self._extract_keywords(cleaned_text)
        
        # Analyze file types
        file_types = self._analyze_file_types(input_files or [])
        
        # Determine complexity
        complexity = self._assess_complexity(cleaned_text, keywords)
        
        # Extract context clues
        context_clues = self._extract_context_clues(cleaned_text, entities, keywords)
        
        # Determine intent
        intent = self._determine_intent(cleaned_text, keywords, context_clues)
        
        return InputAnalysis(
            cleaned_text=cleaned_text,
            intent=intent,
            entities=entities,
            file_types=file_types,
            complexity=complexity,
            keywords=keywords,
            context_clues=context_clues
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
        
        # Normalize case
        text = text.lower()
        
        return text
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE']]
            except Exception as e:
                logger.warning(f"Error extracting entities: {e}")
        
        # Fallback: extract potential entities using patterns
        if not entities:
            # IP addresses
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            entities.extend(re.findall(ip_pattern, text))
            
            # File extensions
            ext_pattern = r'\b\w+\.\w{2,4}\b'
            entities.extend(re.findall(ext_pattern, text))
            
            # Patent numbers
            patent_pattern = r'\b(?:US|EP|WO)?\d{4,}\b'
            entities.extend(re.findall(patent_pattern, text))
        
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Split into words and filter
        words = re.findall(r'\b\w+\b', text)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency and return most common
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _analyze_file_types(self, file_paths: List[str]) -> List[str]:
        """Analyze file types from input files."""
        file_types = []
        
        for file_path in file_paths:
            file_ext = Path(file_path).suffix.lower()
            file_name = Path(file_path).name.lower()
            
            # Check against patterns
            for category, patterns in self.file_type_patterns.items():
                if file_ext in patterns or any(pattern.replace('.', '') in file_name for pattern in patterns):
                    file_types.append(category)
                    break
        
        return list(set(file_types))
    
    def _assess_complexity(self, text: str, keywords: List[str]) -> InputComplexity:
        """Assess input complexity level."""
        complexity_scores = defaultdict(int)
        
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    complexity_scores[complexity] += 1
                for keyword in keywords:
                    if indicator in keyword or keyword in indicator:
                        complexity_scores[complexity] += 0.5
        
        if complexity_scores:
            return max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return InputComplexity.MODERATE
    
    def _extract_context_clues(self, text: str, entities: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Extract context clues from input."""
        clues = {
            "has_file_references": any(ext in text for ext in ['.csv', '.json', '.pcap', '.log', '.exe']),
            "mentions_time": any(word in text for word in ['time', 'date', 'timeline', 'when', 'recent']),
            "mentions_network": any(word in text for word in ['network', 'ip', 'port', 'traffic', 'packet']),
            "mentions_security": any(word in text for word in ['security', 'threat', 'attack', 'vulnerability', 'malware']),
            "mentions_analysis": any(word in text for word in ['analyze', 'analysis', 'investigate', 'examine', 'study']),
            "entity_count": len(entities),
            "keyword_count": len(keywords)
        }
        
        return clues
    
    def _determine_intent(self, text: str, keywords: List[str], context_clues: Dict[str, Any]) -> str:
        """Determine user intent from input."""
        intent_patterns = {
            "analyze": ["analyze", "analysis", "examine", "study", "investigate"],
            "scan": ["scan", "check", "detect", "find", "look for"],
            "convert": ["convert", "transform", "change", "modify"],
            "enrich": ["enrich", "enhance", "add", "supplement", "augment"],
            "compare": ["compare", "contrast", "difference", "similarity"],
            "report": ["report", "summary", "overview", "status", "result"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text for pattern in patterns):
                return intent
        
        return "general"
    
    def detect_workflow(self, user_input: str, input_files: List[str] = None) -> WorkflowRecommendation:
        """Detect the most appropriate workflow for user input."""
        start_time = time.time()
        
        # Preprocess input
        analysis = self.preprocess_input(user_input, input_files)
        
        # Check cache first
        cache_key = self._generate_cache_key(analysis)
        if cache_key in self.pattern_cache:
            cached_result = self.pattern_cache[cache_key]
            cached_result.workflow_type = cached_result.workflow_type  # Ensure enum type
            return cached_result
        
        # Score each workflow
        workflow_scores = {}
        for workflow_type, patterns in self.workflow_patterns.items():
            score = self._score_workflow(analysis, patterns)
            workflow_scores[workflow_type] = score
        
        # Select best workflow
        best_workflow = max(workflow_scores.items(), key=lambda x: x[1])
        
        # Get alternative workflows
        alternatives = sorted(workflow_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
        
        # Generate recommendation
        recommendation = WorkflowRecommendation(
            workflow_type=best_workflow[0],
            confidence_score=best_workflow[1],
            reasoning=self._generate_reasoning(analysis, best_workflow[0], best_workflow[1]),
            alternative_workflows=[alt[0] for alt in alternatives if alt[1] > 0.3],
            required_inputs=self._determine_required_inputs(best_workflow[0], analysis),
            estimated_duration=self._estimate_duration(best_workflow[0], analysis),
            complexity_level=analysis.complexity,
            preprocessing_steps=self._determine_preprocessing_steps(analysis)
        )
        
        # Cache result
        self.pattern_cache[cache_key] = recommendation
        
        # Track performance
        processing_time = time.time() - start_time
        self.performance_history.append({
            "timestamp": time.time(),
            "processing_time": processing_time,
            "confidence": best_workflow[1],
            "workflow": best_workflow[0].value
        })
        
        return recommendation
    
    def _generate_cache_key(self, analysis: InputAnalysis) -> str:
        """Generate cache key for analysis."""
        key_data = f"{analysis.cleaned_text}:{':'.join(analysis.file_types)}:{analysis.complexity.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _score_workflow(self, analysis: InputAnalysis, patterns: Dict[str, Any]) -> float:
        """Score a workflow based on analysis."""
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in analysis.keywords if keyword in patterns.get("keywords", []))
        if patterns.get("keywords"):
            score += (keyword_matches / len(patterns["keywords"])) * 0.4
        
        # File type matching
        file_matches = sum(1 for file_type in analysis.file_types if file_type in patterns.get("file_patterns", []))
        if patterns.get("file_patterns"):
            score += (file_matches / len(patterns["file_patterns"])) * 0.3
        
        # Intent pattern matching
        intent_matches = sum(1 for pattern in patterns.get("intent_patterns", []) 
                           if re.search(pattern, analysis.cleaned_text, re.IGNORECASE))
        if patterns.get("intent_patterns"):
            score += (intent_matches / len(patterns["intent_patterns"])) * 0.2
        
        # Context clue matching
        context_matches = sum(1 for clue in patterns.get("context_clues", []) 
                            if clue in analysis.cleaned_text)
        if patterns.get("context_clues"):
            score += (context_matches / len(patterns["context_clues"])) * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_reasoning(self, analysis: InputAnalysis, workflow_type: WorkflowType, confidence: float) -> str:
        """Generate reasoning for workflow selection."""
        reasons = []
        
        if confidence > 0.7:
            reasons.append(f"High confidence match for {workflow_type.value}")
        elif confidence > 0.4:
            reasons.append(f"Moderate confidence match for {workflow_type.value}")
        else:
            reasons.append(f"Low confidence match for {workflow_type.value}")
        
        if analysis.keywords:
            reasons.append(f"Keywords detected: {', '.join(analysis.keywords[:3])}")
        
        if analysis.file_types:
            reasons.append(f"File types: {', '.join(analysis.file_types)}")
        
        if analysis.complexity != InputComplexity.MODERATE:
            reasons.append(f"Complexity level: {analysis.complexity.value}")
        
        return "; ".join(reasons)
    
    def _determine_required_inputs(self, workflow_type: WorkflowType, analysis: InputAnalysis) -> List[str]:
        """Determine required inputs for workflow."""
        base_inputs = {
            WorkflowType.PATENT_ANALYSIS: ["CSV file with patent numbers"],
            WorkflowType.MALWARE_ANALYSIS: ["Suspicious file or file path"],
            WorkflowType.NETWORK_ANALYSIS: ["PCAP file"],
            WorkflowType.VULNERABILITY_SCAN: ["Target IP addresses or hostnames"],
            WorkflowType.INCIDENT_RESPONSE: ["Log files or incident data"],
            WorkflowType.THREAT_HUNTING: ["Data sources for hunting"],
            WorkflowType.DATA_ANALYSIS: ["Data file (CSV, JSON, etc.)"],
            WorkflowType.FILE_FORENSICS: ["Forensic image or file system"]
        }
        
        return base_inputs.get(workflow_type, ["Input data"])
    
    def _estimate_duration(self, workflow_type: WorkflowType, analysis: InputAnalysis) -> str:
        """Estimate workflow duration."""
        base_durations = {
            WorkflowType.PATENT_ANALYSIS: "2-5 minutes",
            WorkflowType.MALWARE_ANALYSIS: "1-3 minutes",
            WorkflowType.NETWORK_ANALYSIS: "3-10 minutes",
            WorkflowType.VULNERABILITY_SCAN: "5-15 minutes",
            WorkflowType.INCIDENT_RESPONSE: "10-30 minutes",
            WorkflowType.THREAT_HUNTING: "15-45 minutes",
            WorkflowType.DATA_ANALYSIS: "1-5 minutes",
            WorkflowType.FILE_FORENSICS: "5-20 minutes"
        }
        
        duration = base_durations.get(workflow_type, "2-10 minutes")
        
        # Adjust based on complexity
        if analysis.complexity == InputComplexity.COMPLEX:
            duration = duration.replace("minutes", "minutes (complex)")
        elif analysis.complexity == InputComplexity.EXPERT:
            duration = duration.replace("minutes", "minutes (expert)")
        
        return duration
    
    def _determine_preprocessing_steps(self, analysis: InputAnalysis) -> List[str]:
        """Determine preprocessing steps needed."""
        steps = []
        
        if analysis.complexity == InputComplexity.SIMPLE:
            steps.append("Basic input validation")
        elif analysis.complexity == InputComplexity.MODERATE:
            steps.append("Input validation and cleaning")
            steps.append("Data format verification")
        elif analysis.complexity == InputComplexity.COMPLEX:
            steps.append("Comprehensive input validation")
            steps.append("Data preprocessing and normalization")
            steps.append("Context analysis")
        else:  # EXPERT
            steps.append("Advanced input validation")
            steps.append("Multi-stage data preprocessing")
            steps.append("Deep context analysis")
            steps.append("Expert-level optimization")
        
        return steps
    
    def train_on_feedback(self, user_input: str, selected_workflow: WorkflowType, 
                         was_correct: bool, actual_workflow: WorkflowType = None):
        """Train the model based on user feedback."""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            # This would be implemented with actual training data
            # For now, we'll just log the feedback
            logger.info(f"Feedback: Input='{user_input[:50]}...', Selected={selected_workflow.value}, "
                       f"Correct={was_correct}, Actual={actual_workflow.value if actual_workflow else 'N/A'}")
            
            # In a real implementation, you would:
            # 1. Add this to training data
            # 2. Retrain the model periodically
            # 3. Update the workflow patterns based on feedback
            
        except Exception as e:
            logger.warning(f"Error processing feedback: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_history = self.performance_history[-100:]  # Last 100 requests
        
        avg_processing_time = sum(h["processing_time"] for h in recent_history) / len(recent_history)
        avg_confidence = sum(h["confidence"] for h in recent_history) / len(recent_history)
        
        workflow_counts = Counter(h["workflow"] for h in recent_history)
        
        return {
            "total_requests": len(self.performance_history),
            "recent_requests": len(recent_history),
            "avg_processing_time": round(avg_processing_time, 3),
            "avg_confidence": round(avg_confidence, 3),
            "workflow_distribution": dict(workflow_counts.most_common()),
            "cache_hit_rate": len(self.pattern_cache) / max(len(self.performance_history), 1)
        }
