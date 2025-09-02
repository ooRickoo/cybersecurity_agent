#!/usr/bin/env python3
"""
Comprehensive Workflow Detection Training Script

This script provides a complete training pipeline for the workflow detection system,
including data generation, model training, evaluation, and deployment.
"""

import argparse
import logging
import time
from pathlib import Path
import json

# Import our training components
from bin.workflow_detection_trainer import WorkflowDetectionTrainer
from bin.training_data_generator import TrainingDataGenerator
from bin.workflow_detection_evaluator import WorkflowDetectionEvaluator
from bin.intelligent_workflow_detector import IntelligentWorkflowDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Setup required directories."""
    directories = [
        "training_data",
        "models/workflow_detection", 
        "evaluation_results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def generate_training_data(num_examples: int = 5000, version: str = "1.0"):
    """Generate comprehensive training data."""
    logger.info(f"Generating {num_examples} training examples...")
    
    generator = TrainingDataGenerator()
    dataset = generator.generate_dataset(num_examples, version)
    
    # Save dataset
    timestamp = int(time.time())
    filename = f"comprehensive_training_data_{timestamp}.json"
    filepath = generator.save_dataset(dataset, filename)
    
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
    
    logger.info(f"Generated {stats['total_examples']} training examples")
    logger.info(f"Workflow Distribution: {dict(stats['workflow_distribution'])}")
    logger.info(f"Complexity Distribution: {dict(stats['complexity_distribution'])}")
    logger.info(f"Source Distribution: {dict(stats['source_distribution'])}")
    
    return filepath, dataset

def train_models(trainer: WorkflowDetectionTrainer, models_to_train: list = None):
    """Train multiple models and compare performance."""
    if models_to_train is None:
        models_to_train = ["random_forest", "naive_bayes", "logistic_regression", "gradient_boosting"]
    
    logger.info(f"Training models: {models_to_train}")
    
    trained_models = {}
    performances = {}
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name} model...")
        
        try:
            performance = trainer.train_model(model_name, use_grid_search=True)
            trained_models[model_name] = trainer.classifier
            performances[model_name] = performance
            
            logger.info(f"{model_name} - Accuracy: {performance.accuracy:.3f}, F1: {performance.f1_macro:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue
    
    return trained_models, performances

def evaluate_models(trainer: WorkflowDetectionTrainer, evaluator: WorkflowDetectionEvaluator, 
                   trained_models: dict, test_size: float = 0.2):
    """Evaluate all trained models."""
    logger.info("Evaluating models...")
    
    # Prepare test data
    X, y = trainer.prepare_training_data()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Evaluate each model
    evaluation_results = {}
    
    for model_name, model in trained_models.items():
        logger.info(f"Evaluating {model_name}...")
        
        try:
            result = evaluator.evaluate_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = result
            
            logger.info(f"{model_name} - Test Accuracy: {result.accuracy:.3f}")
            
            # Save evaluation results
            evaluator.save_evaluation_results(result, model_name)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    return evaluation_results

def compare_models(evaluator: WorkflowDetectionEvaluator, trained_models: dict, 
                  X_test: list, y_test: list):
    """Compare model performance."""
    logger.info("Comparing models...")
    
    comparison = evaluator.compare_models(trained_models, X_test, y_test)
    
    logger.info(f"Best model: {comparison.best_model}")
    logger.info(f"Model accuracies: {dict(zip(comparison.model_names, comparison.accuracies))}")
    logger.info(f"Model F1 scores: {dict(zip(comparison.model_names, comparison.f1_scores))}")
    
    return comparison

def deploy_best_model(trainer: WorkflowDetectionTrainer, best_model_name: str):
    """Deploy the best performing model."""
    logger.info(f"Deploying best model: {best_model_name}")
    
    # Load the best model
    trainer.load_model(best_model_name)
    
    # Test the deployed model
    test_inputs = [
        "Analyze these patent numbers: US12345678, EP98765432",
        "Scan this file for malware: suspicious.exe",
        "Analyze network traffic in capture.pcap",
        "Check for vulnerabilities on 192.168.1.1",
        "Investigate this security incident",
        "Threat hunting in network logs",
        "Analyze this data file: data.csv",
        "Forensic analysis of disk image"
    ]
    
    logger.info("Testing deployed model...")
    
    for test_input in test_inputs:
        try:
            recommendation = trainer.predict_workflow(test_input)
            logger.info(f"Input: {test_input}")
            logger.info(f"Predicted: {recommendation.workflow_type.value} (confidence: {recommendation.confidence_score:.3f})")
            logger.info(f"Reasoning: {recommendation.reasoning}")
            logger.info("---")
        except Exception as e:
            logger.error(f"Error testing input '{test_input}': {e}")
    
    logger.info("Model deployment completed!")

def create_training_report(trainer: WorkflowDetectionTrainer, evaluator: WorkflowDetectionEvaluator,
                          performances: dict, evaluation_results: dict, comparison):
    """Create a comprehensive training report."""
    logger.info("Creating training report...")
    
    report = f"""
# Workflow Detection Model Training Report

## Training Summary
- **Training Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Training Examples**: {len(trainer.training_data)}
- **Models Trained**: {len(performances)}
- **Best Model**: {comparison.best_model}

## Training Data Statistics
"""
    
    # Add training data stats
    stats = trainer.get_training_stats()
    report += f"""
- **Total Examples**: {stats['total_examples']}
- **Workflow Distribution**: {stats['workflow_distribution']}
- **Complexity Distribution**: {stats['complexity_distribution']}
- **Source Distribution**: {stats['source_distribution']}
- **Average Confidence**: {stats['average_confidence']}

## Model Performance Comparison
"""
    
    # Add model comparison
    for i, model_name in enumerate(comparison.model_names):
        report += f"""
### {model_name}
- **Accuracy**: {comparison.accuracies[i]:.3f}
- **F1 Score**: {comparison.f1_scores[i]:.3f}
"""
    
    report += f"""
## Best Model Details
**Model**: {comparison.best_model}
**Accuracy**: {max(comparison.accuracies):.3f}
**F1 Score**: {max(comparison.f1_scores):.3f}

## Recommendations
"""
    
    # Add recommendations
    best_accuracy = max(comparison.accuracies)
    if best_accuracy > 0.9:
        report += "- ✅ **Excellent Performance**: Model is ready for production use\n"
    elif best_accuracy > 0.8:
        report += "- ✅ **Good Performance**: Model is suitable for production with monitoring\n"
    elif best_accuracy > 0.7:
        report += "- ⚠️ **Moderate Performance**: Consider additional training data or model tuning\n"
    else:
        report += "- ❌ **Poor Performance**: Significant improvements needed before deployment\n"
    
    if best_accuracy < 0.8:
        report += "- 📊 **Data Collection**: Collect more diverse training examples\n"
        report += "- 🔧 **Model Tuning**: Experiment with different hyperparameters\n"
        report += "- 🧪 **Feature Engineering**: Improve feature extraction methods\n"
    
    report += f"""
## Next Steps
1. **Deploy Model**: Use the best performing model in production
2. **Monitor Performance**: Track model performance in real-world usage
3. **Collect Feedback**: Gather user feedback for continuous improvement
4. **Retrain Periodically**: Update model with new training data
5. **A/B Testing**: Compare model performance against rule-based detection

## Files Generated
- **Training Data**: `training_data/comprehensive_training_data_*.json`
- **Trained Models**: `models/workflow_detection/*_model.pkl`
- **Evaluation Results**: `evaluation_results/evaluation_*.json`
- **Reports**: `evaluation_results/report_*.md`

---
*Report generated by Workflow Detection Training System*
"""
    
    # Save report
    timestamp = int(time.time())
    report_path = Path("training_report.md")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Training report saved to: {report_path}")
    return report_path

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train workflow detection models")
    parser.add_argument("--examples", type=int, default=5000,
                       help="Number of training examples to generate")
    parser.add_argument("--models", nargs="+", 
                       default=["random_forest", "naive_bayes", "logistic_regression"],
                       help="Models to train")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size for evaluation")
    parser.add_argument("--skip-data-generation", action="store_true",
                       help="Skip training data generation")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip model evaluation")
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy best model after training")
    parser.add_argument("--report", action="store_true",
                       help="Generate training report")
    
    args = parser.parse_args()
    
    logger.info("Starting workflow detection training pipeline...")
    
    # Setup directories
    setup_directories()
    
    # Initialize components
    trainer = WorkflowDetectionTrainer()
    evaluator = WorkflowDetectionEvaluator()
    
    # Generate training data
    if not args.skip_data_generation:
        data_file, dataset = generate_training_data(args.examples)
        logger.info(f"Training data generated: {data_file}")
        
        # Add generated data to trainer
        trainer.training_data.extend(dataset.examples)
        logger.info(f"Added {len(dataset.examples)} training examples to trainer")
    else:
        logger.info("Skipping training data generation")
    
    # Train models
    if not args.skip_training:
        trained_models, performances = train_models(trainer, args.models)
        logger.info(f"Trained {len(trained_models)} models")
    else:
        logger.info("Skipping model training")
        trained_models = {}
        performances = {}
    
    # Evaluate models
    if not args.skip_evaluation and trained_models:
        evaluation_results = evaluate_models(trainer, evaluator, trained_models, args.test_size)
        
        # Compare models
        X, y = trainer.prepare_training_data()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        comparison = compare_models(evaluator, trained_models, X_test, y_test)
    else:
        logger.info("Skipping model evaluation")
        evaluation_results = {}
        comparison = None
    
    # Deploy best model
    if args.deploy and comparison:
        deploy_best_model(trainer, comparison.best_model)
    
    # Generate report
    if args.report and comparison:
        report_path = create_training_report(trainer, evaluator, performances, 
                                           evaluation_results, comparison)
        logger.info(f"Training report generated: {report_path}")
    
    logger.info("Training pipeline completed successfully!")
    
    # Print summary
    if comparison:
        print(f"\n🎯 Training Summary:")
        print(f"   Best Model: {comparison.best_model}")
        print(f"   Best Accuracy: {max(comparison.accuracies):.3f}")
        print(f"   Best F1 Score: {max(comparison.f1_scores):.3f}")
        print(f"   Models Trained: {len(trained_models)}")
        print(f"   Training Examples: {len(trainer.training_data)}")

if __name__ == "__main__":
    main()
