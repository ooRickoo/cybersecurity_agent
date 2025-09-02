#!/usr/bin/env python3
"""
Workflow Detection Model Trainer

This module provides comprehensive training capabilities for the intelligent workflow detection system.
It includes data collection, model training, evaluation, and continuous learning features.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import random

# ML libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from bin.intelligent_workflow_detector import WorkflowType, InputComplexity, WorkflowRecommendation, InputAnalysis

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """A single training example for workflow detection."""
    user_input: str
    input_files: List[str]
    correct_workflow: WorkflowType
    complexity: InputComplexity
    confidence_score: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TrainingDataset:
    """Training dataset for workflow detection."""
    examples: List[TrainingExample]
    created_at: float
    version: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        examples_dict = []
        for example in self.examples:
            example_dict = asdict(example)
            # Convert enum to string for JSON serialization
            example_dict["correct_workflow"] = example.correct_workflow.value
            example_dict["complexity"] = example.complexity.value
            examples_dict.append(example_dict)
        
        return {
            "examples": examples_dict,
            "created_at": self.created_at,
            "version": self.version,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingDataset':
        """Create from dictionary."""
        examples = []
        for ex_data in data["examples"]:
            ex_data["correct_workflow"] = WorkflowType(ex_data["correct_workflow"])
            ex_data["complexity"] = InputComplexity(ex_data["complexity"])
            examples.append(TrainingExample(**ex_data))
        
        return cls(
            examples=examples,
            created_at=data["created_at"],
            version=data["version"],
            description=data["description"]
        )

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
    cross_val_scores: List[float]
    training_time: float
    model_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class WorkflowDetectionTrainer:
    """Comprehensive trainer for workflow detection models."""
    
    def __init__(self, data_dir: str = "training_data", model_dir: str = "models/workflow_detection"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.training_data = []
        self.vectorizer = None
        self.classifier = None
        self.nlp_model = None
        
        # Model configurations
        self.model_configs = {
            "naive_bayes": {
                "classifier": MultinomialNB(),
                "params": {"alpha": [0.1, 0.5, 1.0, 2.0]}
            },
            "random_forest": {
                "classifier": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "logistic_regression": {
                "classifier": LogisticRegression(random_state=42, max_iter=1000),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            },
            "gradient_boosting": {
                "classifier": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }
        }
        
        # Load existing data and models
        self._load_training_data()
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP model."""
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_model = None
    
    def _load_training_data(self):
        """Load existing training data."""
        data_files = list(self.data_dir.glob("training_data_*.json"))
        for data_file in data_files:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    dataset = TrainingDataset.from_dict(data)
                    self.training_data.extend(dataset.examples)
                logger.info(f"Loaded {len(dataset.examples)} examples from {data_file}")
            except Exception as e:
                logger.warning(f"Error loading {data_file}: {e}")
        
        logger.info(f"Total training examples loaded: {len(self.training_data)}")
    
    def add_training_example(self, user_input: str, input_files: List[str], 
                           correct_workflow: WorkflowType, complexity: InputComplexity = None,
                           confidence_score: float = 1.0, metadata: Dict[str, Any] = None):
        """Add a new training example."""
        if complexity is None:
            complexity = InputComplexity.MODERATE
        
        example = TrainingExample(
            user_input=user_input,
            input_files=input_files or [],
            correct_workflow=correct_workflow,
            complexity=complexity,
            confidence_score=confidence_score,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.training_data.append(example)
        logger.info(f"Added training example: {correct_workflow.value}")
    
    def add_feedback_example(self, user_input: str, predicted_workflow: WorkflowType,
                           correct_workflow: WorkflowType, input_files: List[str] = None):
        """Add training example from user feedback."""
        # Determine complexity based on input
        complexity = self._assess_complexity_from_input(user_input)
        
        # Calculate confidence based on correctness
        confidence = 1.0 if predicted_workflow == correct_workflow else 0.0
        
        self.add_training_example(
            user_input=user_input,
            input_files=input_files or [],
            correct_workflow=correct_workflow,
            complexity=complexity,
            confidence_score=confidence,
            metadata={"source": "user_feedback", "predicted": predicted_workflow.value}
        )
    
    def _assess_complexity_from_input(self, user_input: str) -> InputComplexity:
        """Assess complexity from user input."""
        complexity_indicators = {
            InputComplexity.SIMPLE: ["analyze", "check", "scan", "look up", "find", "show", "list"],
            InputComplexity.MODERATE: ["compare", "correlate", "enrich", "enhance", "process", "transform"],
            InputComplexity.COMPLEX: ["investigate", "forensic", "timeline", "reconstruct", "analyze patterns"],
            InputComplexity.EXPERT: ["reverse engineer", "malware analysis", "threat hunting", "incident response"]
        }
        
        user_input_lower = user_input.lower()
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in user_input_lower for indicator in indicators):
                return complexity
        
        return InputComplexity.MODERATE
    
    def create_synthetic_training_data(self, num_examples: int = 1000):
        """Create synthetic training data for initial model training."""
        synthetic_examples = []
        
        # Define synthetic data templates
        templates = {
            WorkflowType.PATENT_ANALYSIS: [
                "Analyze these patent numbers: {patents}",
                "Look up patent information for {patents}",
                "Get patent details from USPTO for {patents}",
                "Find patent abstracts and inventors for {patents}",
                "Patent analysis workflow for {patents}"
            ],
            WorkflowType.MALWARE_ANALYSIS: [
                "Analyze this suspicious file: {file}",
                "Scan for malware in {file}",
                "Detect threats in {file}",
                "Malware analysis of {file}",
                "Check if {file} is malicious"
            ],
            WorkflowType.NETWORK_ANALYSIS: [
                "Analyze network traffic in {file}",
                "PCAP analysis of {file}",
                "Network forensics for {file}",
                "Packet capture analysis {file}",
                "Traffic analysis workflow {file}"
            ],
            WorkflowType.VULNERABILITY_SCAN: [
                "Scan {target} for vulnerabilities",
                "Security assessment of {target}",
                "Vulnerability scan {target}",
                "Penetration test {target}",
                "Security audit {target}"
            ],
            WorkflowType.INCIDENT_RESPONSE: [
                "Incident response for {incident}",
                "Investigate security breach {incident}",
                "Forensic analysis of {incident}",
                "Incident timeline {incident}",
                "Security incident {incident}"
            ],
            WorkflowType.THREAT_HUNTING: [
                "Threat hunting in {data}",
                "Proactive threat search {data}",
                "Find anomalies in {data}",
                "Hunt for threats {data}",
                "Threat intelligence {data}"
            ],
            WorkflowType.DATA_ANALYSIS: [
                "Analyze data in {file}",
                "Data analysis workflow {file}",
                "Statistical analysis {file}",
                "Data insights from {file}",
                "Process data file {file}"
            ],
            WorkflowType.FILE_FORENSICS: [
                "Forensic analysis of {file}",
                "File recovery from {file}",
                "Metadata extraction {file}",
                "Timeline analysis {file}",
                "Digital forensics {file}"
            ]
        }
        
        # Sample data for placeholders
        sample_data = {
            "patents": ["US12345678", "EP98765432", "WO1234567890"],
            "file": ["malware.exe", "suspicious.pdf", "traffic.pcap", "data.csv", "image.dd"],
            "target": ["192.168.1.1", "example.com", "server.local"],
            "incident": ["breach_2024", "attack_logs", "security_event"],
            "data": ["network_logs", "system_logs", "user_data"]
        }
        
        for workflow_type, template_list in templates.items():
            for _ in range(num_examples // len(templates)):
                template = random.choice(template_list)
                
                # Replace placeholders with sample data
                user_input = template
                for placeholder, samples in sample_data.items():
                    if f"{{{placeholder}}}" in user_input:
                        user_input = user_input.replace(f"{{{placeholder}}}", random.choice(samples))
                
                # Determine file types
                input_files = []
                if any(ext in user_input for ext in [".exe", ".pdf", ".pcap", ".csv", ".dd"]):
                    input_files = [f"sample_{random.choice(sample_data['file'])}"]
                
                # Determine complexity
                complexity = random.choice(list(InputComplexity))
                
                example = TrainingExample(
                    user_input=user_input,
                    input_files=input_files,
                    correct_workflow=workflow_type,
                    complexity=complexity,
                    confidence_score=random.uniform(0.7, 1.0),
                    timestamp=time.time(),
                    metadata={"source": "synthetic"}
                )
                
                synthetic_examples.append(example)
        
        self.training_data.extend(synthetic_examples)
        logger.info(f"Created {len(synthetic_examples)} synthetic training examples")
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare training data for model training."""
        if not self.training_data:
            raise ValueError("No training data available")
        
        X = []  # Input features
        y = []  # Target labels
        
        for example in self.training_data:
            # Create feature representation
            features = self._create_feature_vector(example)
            X.append(features)
            y.append(example.correct_workflow.value)
        
        return X, y
    
    def _create_feature_vector(self, example: TrainingExample) -> str:
        """Create feature vector from training example."""
        features = []
        
        # Add user input
        features.append(example.user_input)
        
        # Add file type information
        if example.input_files:
            file_types = []
            for file_path in example.input_files:
                ext = Path(file_path).suffix.lower()
                file_types.append(ext)
            features.append(f"file_types: {' '.join(file_types)}")
        
        # Add complexity information
        features.append(f"complexity: {example.complexity.value}")
        
        # Add metadata
        if example.metadata:
            for key, value in example.metadata.items():
                features.append(f"{key}: {value}")
        
        return " ".join(features)
    
    def train_model(self, model_name: str = "random_forest", use_grid_search: bool = True) -> ModelPerformance:
        """Train the workflow detection model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model training")
        
        if not self.training_data:
            raise ValueError("No training data available. Use create_synthetic_training_data() first.")
        
        logger.info(f"Training {model_name} model with {len(self.training_data)} examples")
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', self.model_configs[model_name]["classifier"])
        ])
        
        # Train model
        start_time = time.time()
        
        if use_grid_search and model_name in self.model_configs:
            # Use grid search for hyperparameter tuning
            param_grid = {}
            for param, values in self.model_configs[model_name]["params"].items():
                param_grid[f'classifier__{param}'] = values
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.classifier = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train without grid search
            pipeline.fit(X_train, y_train)
            self.classifier = pipeline
        
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Model size
        model_size = len(pickle.dumps(self.classifier))
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision={k: v['precision'] for k, v in report.items() if isinstance(v, dict)},
            recall={k: v['recall'] for k, v in report.items() if isinstance(v, dict)},
            f1_score={k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)},
            confusion_matrix=cm.tolist(),
            cross_val_scores=cv_scores.tolist(),
            training_time=training_time,
            model_size=model_size
        )
        
        # Add macro averages for compatibility
        performance.precision_macro = precision_macro
        performance.recall_macro = recall_macro
        performance.f1_macro = f1_macro
        
        # Save model
        self._save_model(model_name)
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        return performance
    
    def _save_model(self, model_name: str):
        """Save trained model."""
        if self.classifier is None:
            return
        
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(self.classifier, model_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "training_examples": len(self.training_data),
            "timestamp": time.time(),
            "workflow_types": [wt.value for wt in WorkflowType]
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = "random_forest"):
        """Load trained model."""
        model_path = self.model_dir / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.classifier = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def predict_workflow(self, user_input: str, input_files: List[str] = None) -> WorkflowRecommendation:
        """Predict workflow using trained model."""
        if self.classifier is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Create feature vector
        example = TrainingExample(
            user_input=user_input,
            input_files=input_files or [],
            correct_workflow=WorkflowType.GENERAL_ANALYSIS,  # Dummy value
            complexity=InputComplexity.MODERATE,
            confidence_score=0.0,
            timestamp=time.time()
        )
        
        features = self._create_feature_vector(example)
        
        # Predict
        prediction = self.classifier.predict([features])[0]
        probabilities = self.classifier.predict_proba([features])[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Get workflow type
        workflow_type = WorkflowType(prediction)
        
        # Create recommendation
        recommendation = WorkflowRecommendation(
            workflow_type=workflow_type,
            confidence_score=confidence,
            reasoning=f"ML model prediction with {confidence:.3f} confidence",
            alternative_workflows=[],
            required_inputs=[],
            estimated_duration="unknown",
            complexity_level=InputComplexity.MODERATE,
            preprocessing_steps=[]
        )
        
        return recommendation
    
    def save_training_data(self, version: str = "1.0", description: str = "Training dataset"):
        """Save training data to file."""
        if not self.training_data:
            logger.warning("No training data to save")
            return
        
        dataset = TrainingDataset(
            examples=self.training_data,
            created_at=time.time(),
            version=version,
            description=description
        )
        
        filename = f"training_data_{int(time.time())}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(dataset.to_dict(), f, indent=2)
        
        logger.info(f"Training data saved to {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training data statistics."""
        if not self.training_data:
            return {"message": "No training data available"}
        
        # Count by workflow type
        workflow_counts = Counter(ex.correct_workflow.value for ex in self.training_data)
        
        # Count by complexity
        complexity_counts = Counter(ex.complexity.value for ex in self.training_data)
        
        # Count by source
        source_counts = Counter(ex.metadata.get("source", "unknown") for ex in self.training_data)
        
        # Average confidence
        avg_confidence = sum(ex.confidence_score for ex in self.training_data) / len(self.training_data)
        
        return {
            "total_examples": len(self.training_data),
            "workflow_distribution": dict(workflow_counts),
            "complexity_distribution": dict(complexity_counts),
            "source_distribution": dict(source_counts),
            "average_confidence": round(avg_confidence, 3),
            "date_range": {
                "oldest": min(ex.timestamp for ex in self.training_data),
                "newest": max(ex.timestamp for ex in self.training_data)
            }
        }
    
    def evaluate_model_performance(self, test_data: List[TrainingExample] = None) -> ModelPerformance:
        """Evaluate model performance on test data."""
        if self.classifier is None:
            raise ValueError("No trained model available")
        
        if test_data is None:
            # Use 20% of training data as test data
            test_size = max(1, len(self.training_data) // 5)
            test_data = random.sample(self.training_data, test_size)
        
        X_test = [self._create_feature_vector(ex) for ex in test_data]
        y_test = [ex.correct_workflow.value for ex in test_data]
        
        # Predict
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision={k: v['precision'] for k, v in report.items() if isinstance(v, dict)},
            recall={k: v['recall'] for k, v in report.items() if isinstance(v, dict)},
            f1_score={k: v['f1-score'] for k, v in report.items() if isinstance(v, dict)},
            confusion_matrix=cm.tolist(),
            cross_val_scores=[],
            training_time=0.0,
            model_size=len(pickle.dumps(self.classifier))
        )
        
        return performance

def main():
    """Main function for training workflow detection models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train workflow detection models")
    parser.add_argument("--model", default="random_forest", 
                       choices=["naive_bayes", "random_forest", "logistic_regression", "gradient_boosting"],
                       help="Model type to train")
    parser.add_argument("--synthetic", type=int, default=1000,
                       help="Number of synthetic examples to create")
    parser.add_argument("--no-grid-search", action="store_true",
                       help="Disable grid search for hyperparameter tuning")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model performance after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = WorkflowDetectionTrainer()
    
    # Create synthetic data if needed
    if len(trainer.training_data) < 100:
        print(f"Creating {args.synthetic} synthetic training examples...")
        trainer.create_synthetic_training_data(args.synthetic)
    
    # Train model
    print(f"Training {args.model} model...")
    performance = trainer.train_model(args.model, not args.no_grid_search)
    
    print(f"\nTraining Results:")
    print(f"Accuracy: {performance.accuracy:.3f}")
    print(f"Training Time: {performance.training_time:.2f} seconds")
    print(f"Model Size: {performance.model_size} bytes")
    
    if args.evaluate:
        print("\nEvaluating model performance...")
        eval_performance = trainer.evaluate_model_performance()
        print(f"Test Accuracy: {eval_performance.accuracy:.3f}")
    
    # Save training data
    trainer.save_training_data()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
