# Workflow Detection Training Guide

## Overview

The Cybersecurity Agent includes an intelligent workflow detection system that uses machine learning to automatically determine the most appropriate workflow for a given user query. This system is trained on real-world cybersecurity scenarios and can handle complex, multi-domain workflows.

## Architecture

The training system consists of several key components:

### Core Components

- **`bin/intelligent_workflow_detector.py`** - Main workflow detection engine
- **`bin/workflow_detection_trainer.py`** - Model training and management
- **`bin/workflow_detection_evaluator.py`** - Model evaluation and performance metrics
- **`bin/training_data_generator.py`** - Synthetic training data generation
- **`bin/enhanced_training_data_generator.py`** - Real-world scenario training data
- **`bin/train_workflow_detection.py`** - Basic training pipeline
- **`bin/train_enhanced_workflow_detection.py`** - Enhanced training pipeline

### Training Data

- **`training_data/`** - Directory containing generated training datasets
- **`models/workflow_detection/`** - Directory containing trained models and metadata

## Quick Start

### Basic Training

To train the workflow detection model with synthetic data:

```bash
python bin/train_workflow_detection.py --examples 1000
```

### Enhanced Training

To train with real-world scenarios and enhanced data:

```bash
python bin/train_enhanced_workflow_detection.py --examples 2000
```

## Training Components

### 1. Intelligent Workflow Detector

**File:** `bin/intelligent_workflow_detector.py`

The core detection engine that:
- Analyzes user input for intent and complexity
- Extracts entities and keywords
- Determines workflow type and complexity level
- Provides confidence scores and recommendations

**Key Classes:**
- `WorkflowType` - Enumeration of available workflow types
- `InputComplexity` - Complexity levels (SIMPLE, MODERATE, COMPLEX)
- `WorkflowRecommendation` - Detection results with confidence scores
- `IntelligentWorkflowDetector` - Main detection class

### 2. Workflow Detection Trainer

**File:** `bin/workflow_detection_trainer.py`

Handles model training and management:
- Data preprocessing and feature extraction
- Multiple ML algorithms (RandomForest, NaiveBayes, LogisticRegression, GradientBoosting)
- Hyperparameter tuning with GridSearchCV
- Model persistence and loading
- Performance metrics calculation

**Key Classes:**
- `TrainingExample` - Individual training sample
- `TrainingDataset` - Collection of training examples
- `ModelPerformance` - Performance metrics container
- `WorkflowDetectionTrainer` - Main training class

### 3. Workflow Detection Evaluator

**File:** `bin/workflow_detection_evaluator.py`

Comprehensive model evaluation:
- Train-test splits and cross-validation
- Multiple performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrix analysis
- ROC curves and precision-recall curves
- Real-world scenario testing

**Key Classes:**
- `EvaluationResult` - Comprehensive evaluation results
- `WorkflowDetectionEvaluator` - Main evaluation class

### 4. Training Data Generators

#### Basic Generator
**File:** `bin/training_data_generator.py`

Generates synthetic training data with:
- Various cybersecurity scenarios
- Different complexity levels
- Multiple workflow types
- Realistic user queries

#### Enhanced Generator
**File:** `bin/enhanced_training_data_generator.py`

Generates real-world training scenarios:
- Patent analysis workflows
- Malware analysis scenarios
- Network forensics investigations
- Vulnerability assessments
- Incident response procedures
- Threat hunting operations
- Compliance assessments
- File forensics analysis

## Training Pipelines

### Basic Training Pipeline

**File:** `bin/train_workflow_detection.py`

```bash
python bin/train_workflow_detection.py [options]

Options:
  --examples N        Number of training examples to generate (default: 1000)
  --skip-data-gen     Skip data generation and use existing data
  --deploy            Deploy the best model after training
  --verbose           Enable verbose output
```

**Features:**
- Synthetic data generation
- Multiple model training
- Performance evaluation
- Model selection and deployment

### Enhanced Training Pipeline

**File:** `bin/train_enhanced_workflow_detection.py`

```bash
python bin/train_enhanced_workflow_detection.py [options]

Options:
  --examples N        Number of training examples to generate (default: 2000)
  --skip-data-gen     Skip data generation and use existing data
  --deploy            Deploy the best model after training
  --test-real-world   Run real-world scenario tests
  --verbose           Enable verbose output
```

**Features:**
- Real-world scenario data generation
- Enhanced training examples
- Real-world scenario testing
- Comprehensive performance evaluation
- Detailed reporting

## Workflow Types

The system supports the following workflow types:

1. **Patent Analysis** - US patent lookup and analysis
2. **Malware Analysis** - YARA rules, PE analysis, behavioral analysis
3. **Network Analysis** - PCAP analysis, traffic patterns, anomaly detection
4. **Vulnerability Assessment** - Port scanning, web vulnerabilities, SSL/TLS analysis
5. **Threat Hunting** - IOC analysis, pattern detection, APT hunting
6. **Incident Response** - Incident tracking, playbook execution, forensics
7. **Compliance Assessment** - GDPR, HIPAA, SOC2 compliance checking
8. **File Forensics** - Digital evidence recovery, metadata analysis
9. **Risk Assessment** - Security risk evaluation and mitigation
10. **Policy Analysis** - Security policy review and gap analysis

## Training Data Structure

### Training Example Format

```json
{
  "user_input": "Analyze this PCAP file for suspicious network activity",
  "file_type": "pcap",
  "complexity": "moderate",
  "correct_workflow": "network_analysis",
  "entities": ["PCAP", "network", "suspicious"],
  "keywords": ["analyze", "network", "activity"],
  "context": {
    "domain": "network_security",
    "urgency": "medium",
    "data_sources": ["network_traffic"]
  }
}
```

### Dataset Format

```json
{
  "examples": [...],
  "created_at": "2025-01-01T00:00:00Z",
  "version": "1.0",
  "description": "Enhanced training dataset with real-world scenarios"
}
```

## Model Performance

### Typical Performance Metrics

- **Accuracy:** 95-98% on synthetic data
- **F1-Score:** 90-97% across workflow types
- **Real-world Scenario Accuracy:** 60-80% (complex scenarios)
- **Cross-validation Score:** 85-95%

### Model Selection

The system automatically selects the best performing model based on:
- Overall accuracy
- F1-score
- Cross-validation performance
- Real-world scenario testing

## Usage Examples

### Training a New Model

```bash
# Generate 2000 training examples and train models
python bin/train_enhanced_workflow_detection.py --examples 2000

# Skip data generation and retrain with existing data
python bin/train_enhanced_workflow_detection.py --skip-data-gen

# Train and automatically deploy the best model
python bin/train_enhanced_workflow_detection.py --examples 1500 --deploy
```

### Evaluating Existing Models

```bash
# Evaluate the current model
python bin/workflow_detection_evaluator.py --model random_forest

# Generate evaluation report
python bin/workflow_detection_evaluator.py --model random_forest --report
```

### Using the Trained Model

```python
from bin.intelligent_workflow_detector import IntelligentWorkflowDetector

# Initialize detector
detector = IntelligentWorkflowDetector()

# Analyze user input
result = detector.analyze_input(
    "I need to analyze this malware sample for indicators of compromise",
    file_type="exe"
)

print(f"Recommended workflow: {result.recommended_workflow}")
print(f"Confidence: {result.confidence}")
print(f"Complexity: {result.complexity}")
```

## Configuration

### Model Configuration

Models are configured in `bin/workflow_detection_trainer.py`:

```python
# Available algorithms
ALGORITHMS = {
    'random_forest': RandomForestClassifier,
    'naive_bayes': MultinomialNB,
    'logistic_regression': LogisticRegression,
    'gradient_boosting': GradientBoostingClassifier
}

# Hyperparameter grids
PARAM_GRIDS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
}
```

### Training Configuration

Training parameters can be adjusted in the training scripts:

```python
# Data generation parameters
NUM_EXAMPLES = 2000
TRAIN_TEST_SPLIT = 0.8
CROSS_VALIDATION_FOLDS = 5

# Model selection criteria
SELECTION_METRIC = 'f1_macro'
MIN_ACCURACY_THRESHOLD = 0.85
```

## Troubleshooting

### Common Issues

1. **Low Accuracy on Real-world Scenarios**
   - Increase training examples
   - Add more diverse scenarios
   - Check data quality and labeling

2. **Model Training Failures**
   - Verify dependencies are installed
   - Check training data format
   - Ensure sufficient memory

3. **Poor Cross-validation Scores**
   - Increase training data size
   - Tune hyperparameters
   - Check for data leakage

### Debug Mode

Enable verbose output for debugging:

```bash
python bin/train_enhanced_workflow_detection.py --verbose
```

## Best Practices

1. **Regular Retraining**
   - Retrain monthly with new scenarios
   - Monitor performance degradation
   - Update training data with user feedback

2. **Data Quality**
   - Validate training examples
   - Ensure balanced dataset
   - Remove duplicate or low-quality examples

3. **Model Selection**
   - Test multiple algorithms
   - Use cross-validation
   - Evaluate on real-world scenarios

4. **Performance Monitoring**
   - Track accuracy over time
   - Monitor false positive/negative rates
   - Collect user feedback

## Integration

The trained models are automatically integrated into the main agent:

```python
# In bin/langgraph_cybersecurity_agent.py
from bin.intelligent_workflow_detector import IntelligentWorkflowDetector

class LangGraphCybersecurityAgent:
    def __init__(self):
        self.intelligent_workflow_detector = IntelligentWorkflowDetector()
        # ... other initialization
```

The detector is used throughout the agent to automatically route user queries to appropriate workflows.

## Future Enhancements

1. **Continuous Learning**
   - Online learning from user feedback
   - Adaptive model updates
   - Real-time performance monitoring

2. **Advanced Models**
   - Deep learning approaches
   - Transformer-based models
   - Multi-modal input processing

3. **Enhanced Evaluation**
   - A/B testing framework
   - User satisfaction metrics
   - Business impact measurement

## Support

For issues or questions about the training system:

1. Check the logs in `logs/` directory
2. Review the evaluation reports in `evaluation_results/`
3. Examine the training data in `training_data/`
4. Consult the model metadata in `models/workflow_detection/`

## Related Documentation

- [AI Tools Guide](AI_TOOLS_GUIDE.md) - Overview of all AI tools
- [Usage Examples Guide](USAGE_EXAMPLES_GUIDE.md) - Practical usage examples
- [Network Analysis Guide](NETWORK_ANALYSIS_GUIDE.md) - Network analysis workflows
- [Malware Analysis Guide](MALWARE_ANALYSIS_GUIDE.md) - Malware analysis workflows
