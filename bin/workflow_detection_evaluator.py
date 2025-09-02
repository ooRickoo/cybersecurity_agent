#!/usr/bin/env python3
"""
Workflow Detection Model Evaluator

This module provides comprehensive evaluation capabilities for the workflow detection system,
including performance metrics, confusion matrices, and continuous evaluation.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import random

# ML libraries
try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from bin.intelligent_workflow_detector import WorkflowType, InputComplexity, WorkflowRecommendation
from bin.workflow_detection_trainer import TrainingExample, ModelPerformance

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    cross_val_scores: List[float]
    cross_val_mean: float
    cross_val_std: float
    evaluation_time: float
    num_test_examples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ModelComparison:
    """Comparison between multiple models."""
    model_names: List[str]
    accuracies: List[float]
    f1_scores: List[float]
    training_times: List[float]
    model_sizes: List[int]
    best_model: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class WorkflowDetectionEvaluator:
    """Comprehensive evaluator for workflow detection models."""
    
    def __init__(self, model_dir: str = "models/workflow_detection", results_dir: str = "evaluation_results"):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation history
        self.evaluation_history = []
        self.model_comparisons = []
    
    def evaluate_model(self, model, X_test: List[str], y_test: List[str], 
                      model_name: str = "model") -> EvaluationResult:
        """Evaluate a trained model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for model evaluation")
        
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        unique_labels = sorted(set(list(y_test) + list(y_pred)))
        for label in unique_labels:
            per_class_metrics[label] = {
                'precision': precision_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0] if label in y_test else 0.0,
                'recall': recall_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0] if label in y_test else 0.0,
                'f1_score': f1_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0] if label in y_test else 0.0
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_scores = []
            cv_mean = 0.0
            cv_std = 0.0
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm.tolist(),
            cross_val_scores=cv_scores.tolist(),
            cross_val_mean=cv_mean,
            cross_val_std=cv_std,
            evaluation_time=evaluation_time,
            num_test_examples=len(y_test)
        )
        
        # Store in history
        self.evaluation_history.append({
            "model_name": model_name,
            "timestamp": time.time(),
            "result": result.to_dict()
        })
        
        return result
    
    def compare_models(self, models: Dict[str, Any], X_test: List[str], y_test: List[str]) -> ModelComparison:
        """Compare multiple models."""
        model_names = []
        accuracies = []
        f1_scores = []
        training_times = []
        model_sizes = []
        
        for name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            
            model_names.append(name)
            accuracies.append(result.accuracy)
            f1_scores.append(result.f1_macro)
            training_times.append(0.0)  # Would need to track training time separately
            model_sizes.append(0)  # Would need to calculate model size separately
        
        # Find best model
        best_idx = np.argmax(accuracies)
        best_model = model_names[best_idx]
        
        comparison = ModelComparison(
            model_names=model_names,
            accuracies=accuracies,
            f1_scores=f1_scores,
            training_times=training_times,
            model_sizes=model_sizes,
            best_model=best_model
        )
        
        self.model_comparisons.append(comparison.to_dict())
        return comparison
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], 
                            labels: List[str], model_name: str = "model",
                            save_path: str = None):
        """Plot confusion matrix."""
        if not SKLEARN_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_comparison(self, comparison: ModelComparison, save_path: str = None):
        """Plot performance comparison between models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.bar(comparison.model_names, comparison.accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        ax2.bar(comparison.model_names, comparison.f1_scores, color='lightcoral')
        ax2.set_title('Model F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_evaluation_report(self, result: EvaluationResult, model_name: str = "model") -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Workflow Detection Model Evaluation Report

## Model: {model_name}
## Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- **Accuracy**: {result.accuracy:.3f}
- **Precision (Macro)**: {result.precision_macro:.3f}
- **Recall (Macro)**: {result.recall_macro:.3f}
- **F1 Score (Macro)**: {result.f1_macro:.3f}
- **Precision (Weighted)**: {result.precision_weighted:.3f}
- **Recall (Weighted)**: {result.recall_weighted:.3f}
- **F1 Score (Weighted)**: {result.f1_weighted:.3f}

## Cross-Validation Results
- **Mean CV Score**: {result.cross_val_mean:.3f}
- **CV Standard Deviation**: {result.cross_val_std:.3f}
- **CV Scores**: {[f'{score:.3f}' for score in result.cross_val_scores]}

## Test Dataset
- **Number of Examples**: {result.num_test_examples}
- **Evaluation Time**: {result.evaluation_time:.3f} seconds

## Per-Class Performance
"""
        
        for class_name, metrics in result.per_class_metrics.items():
            report += f"""
### {class_name}
- **Precision**: {metrics['precision']:.3f}
- **Recall**: {metrics['recall']:.3f}
- **F1 Score**: {metrics['f1_score']:.3f}
"""
        
        report += f"""
## Confusion Matrix
```
{np.array(result.confusion_matrix)}
```

## Recommendations
"""
        
        # Add recommendations based on performance
        if result.accuracy < 0.7:
            report += "- **Low Accuracy**: Consider collecting more training data or trying different models\n"
        
        if result.f1_macro < 0.6:
            report += "- **Low F1 Score**: Model may have class imbalance issues\n"
        
        if result.cross_val_std > 0.1:
            report += "- **High CV Variance**: Model may be overfitting or unstable\n"
        
        if result.accuracy > 0.9:
            report += "- **High Performance**: Model is performing well\n"
        
        return report
    
    def save_evaluation_results(self, result: EvaluationResult, model_name: str = "model"):
        """Save evaluation results to file."""
        timestamp = int(time.time())
        filename = f"evaluation_{model_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        data = {
            "model_name": model_name,
            "timestamp": time.time(),
            "result": result.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
        
        # Also save report
        report = self.generate_evaluation_report(result, model_name)
        report_path = self.results_dir / f"report_{model_name}_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return filepath, report_path
    
    def load_evaluation_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history
    
    def get_model_comparisons(self) -> List[Dict[str, Any]]:
        """Get model comparison history."""
        return self.model_comparisons
    
    def continuous_evaluation(self, model, test_data: List[TrainingExample], 
                            batch_size: int = 100) -> List[EvaluationResult]:
        """Perform continuous evaluation on streaming data."""
        results = []
        
        # Process data in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            
            # Prepare batch data
            X_batch = []
            y_batch = []
            
            for example in batch:
                # Create feature vector (simplified)
                features = f"{example.user_input} {' '.join(example.input_files)} {example.complexity.value}"
                X_batch.append(features)
                y_batch.append(example.correct_workflow.value)
            
            # Evaluate batch
            result = self.evaluate_model(model, X_batch, y_batch, f"batch_{i//batch_size}")
            results.append(result)
            
            logger.info(f"Batch {i//batch_size + 1}: Accuracy = {result.accuracy:.3f}")
        
        return results
    
    def analyze_error_patterns(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Analyze error patterns in predictions."""
        errors = defaultdict(list)
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                errors[f"{true_label} -> {pred_label}"].append(1)
        
        # Count errors
        error_counts = {error: len(counts) for error, counts in errors.items()}
        
        # Find most common errors
        most_common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_errors": sum(error_counts.values()),
            "error_rate": sum(error_counts.values()) / len(y_true),
            "most_common_errors": most_common_errors[:10],
            "error_distribution": dict(error_counts)
        }
    
    def generate_performance_summary(self) -> str:
        """Generate a summary of all evaluations."""
        if not self.evaluation_history:
            return "No evaluation history available."
        
        summary = "# Workflow Detection Model Performance Summary\n\n"
        
        # Overall statistics
        accuracies = [eval_data["result"]["accuracy"] for eval_data in self.evaluation_history]
        f1_scores = [eval_data["result"]["f1_macro"] for eval_data in self.evaluation_history]
        
        summary += f"## Overall Statistics\n"
        summary += f"- **Total Evaluations**: {len(self.evaluation_history)}\n"
        summary += f"- **Average Accuracy**: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n"
        summary += f"- **Average F1 Score**: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n"
        summary += f"- **Best Accuracy**: {max(accuracies):.3f}\n"
        summary += f"- **Best F1 Score**: {max(f1_scores):.3f}\n\n"
        
        # Model comparison
        if self.model_comparisons:
            summary += "## Model Comparisons\n"
            for comparison in self.model_comparisons:
                summary += f"### {comparison['best_model']} (Best Model)\n"
                summary += f"- **Accuracy**: {max(comparison['accuracies']):.3f}\n"
                summary += f"- **F1 Score**: {max(comparison['f1_scores']):.3f}\n\n"
        
        return summary

def main():
    """Main function for model evaluation."""
    import argparse
    from bin.workflow_detection_trainer import WorkflowDetectionTrainer
    
    parser = argparse.ArgumentParser(description="Evaluate workflow detection models")
    parser.add_argument("--model", default="random_forest",
                       help="Model to evaluate")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed report")
    
    args = parser.parse_args()
    
    # Initialize trainer and evaluator
    trainer = WorkflowDetectionTrainer()
    evaluator = WorkflowDetectionEvaluator()
    
    # Create synthetic data if needed
    if len(trainer.training_data) < 100:
        print("Creating synthetic training data...")
        trainer.create_synthetic_training_data(1000)
    
    # Prepare data
    X, y = trainer.prepare_training_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Train model
    print(f"Training {args.model} model...")
    trainer.train_model(args.model)
    
    # Evaluate model
    print("Evaluating model...")
    result = evaluator.evaluate_model(trainer.classifier, X_test, y_test, args.model)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"F1 Score: {result.f1_macro:.3f}")
    print(f"Precision: {result.precision_macro:.3f}")
    print(f"Recall: {result.recall_macro:.3f}")
    
    # Generate plots
    if args.plot:
        print("Generating plots...")
        labels = sorted(set(y_test))
        evaluator.plot_confusion_matrix(result.confusion_matrix, labels, args.model)
    
    # Generate report
    if args.report:
        print("Generating report...")
        filepath, report_path = evaluator.save_evaluation_results(result, args.model)
        print(f"Results saved to: {filepath}")
        print(f"Report saved to: {report_path}")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
