
# Workflow Detection Model Evaluation Report

## Model: random_forest
## Evaluation Date: 2025-09-01 22:25:48

## Overall Performance
- **Accuracy**: 0.980
- **Precision (Macro)**: 0.967
- **Recall (Macro)**: 0.980
- **F1 Score (Macro)**: 0.969
- **Precision (Weighted)**: 0.987
- **Recall (Weighted)**: 0.980
- **F1 Score (Weighted)**: 0.981

## Cross-Validation Results
- **Mean CV Score**: 0.805
- **CV Standard Deviation**: 0.105
- **CV Scores**: ['0.727', '0.800', '0.700', '1.000', '0.800']

## Test Dataset
- **Number of Examples**: 51
- **Evaluation Time**: 0.812 seconds

## Per-Class Performance

### compliance_assessment
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### data_analysis
- **Precision**: 1.000
- **Recall**: 0.800
- **F1 Score**: 0.889

### file_forensics
- **Precision**: 0.667
- **Recall**: 1.000
- **F1 Score**: 0.800

### incident_response
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### malware_analysis
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### network_analysis
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### patent_analysis
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### threat_hunting
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### threat_intelligence
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

### vulnerability_scan
- **Precision**: 1.000
- **Recall**: 1.000
- **F1 Score**: 1.000

## Confusion Matrix
```
[[3 0 0 0 0 0 0 0 0 0]
 [0 4 1 0 0 0 0 0 0 0]
 [0 0 2 0 0 0 0 0 0 0]
 [0 0 0 9 0 0 0 0 0 0]
 [0 0 0 0 9 0 0 0 0 0]
 [0 0 0 0 0 7 0 0 0 0]
 [0 0 0 0 0 0 2 0 0 0]
 [0 0 0 0 0 0 0 6 0 0]
 [0 0 0 0 0 0 0 0 2 0]
 [0 0 0 0 0 0 0 0 0 6]]
```

## Recommendations
- **High CV Variance**: Model may be overfitting or unstable
- **High Performance**: Model is performing well
