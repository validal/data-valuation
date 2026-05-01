# Experimental Scenarios for Data Valuation

## Overview

This folder contains scripts that evaluate data valuation methods across four key scenarios:
1. **MPerf Correlation** - Bootstrap sampling experiments
2. **Robustness** - Noise, diversity, and replication testing
3. **Diversity** - Cross-domain evaluation (placeholder)
4. **Replication** - Reproducibility studies (placeholder)

---

## MPerf Correlation (Model Performance Correlation)

**Scripts in this folder**: `boostrap.py`, `boostrap_tabular.py`, `boostrap_reg.py`

### Methodology
1. Generate 100 bootstrap samples from training data
2. Compute estimated coarse-grained data value for each sample
3. Measure ground truth validation performance for each bootstrap
4. Calculate **Pearson correlation** between estimated values and actual validation performance

### Scripts

**bootstrap.py** - Image Datasets (MNIST, CIFAR-10, ImageNet)
- Model: CNN/ResNet
- Evaluates correlation on image classification

**bootstrap_tabular.py** - Tabular Data (CovType)
- Model: MLP for tabular data
- Evaluates correlation on structured data

**bootstrap_reg.py** - Regression (CASP8 - Protein Structure)
- Model: Regression MLP
- Evaluates correlation on continuous prediction

### Metric
**Pearson Correlation**: Between estimated coarse-grained values and ground truth validation performance

Higher correlation = better at identifying valuable samples

---

## Robustness Testing

**Scripts in this folder**: `robustness.py`, `robustness_imagenet.py`

Tests three robustness dimensions:

### 1. Noise Robustness
**Methodology**:
1. Inject label noise from 2% to 100%
2. Compute coarse-grained data values under noisy conditions
3. Observe if values **decrease as noise increases**
4. Track metric changes across noise levels

**Expected Behavior**: Values should degrade monotonically with increasing noise

**Metric**: Value trend consistency - how well values reflect noise level

### 2. Diversity Sensitivity  
**Methodology**:
1. Increase diversity by covering more of the domain
2. Compute coarse-grained data values at different diversity levels
3. Measure if values **increase as diversity increases**

**Expected Behavior**: More diverse data → higher coarse-grained values

**Metric**: Spearman/Pearson correlation between diversity and estimated values

### 3. Replication Awareness
**Methodology**:
1. Introduce replicated/duplicate samples in training data
2. Compute coarse-grained data values with replicates
3. Verify values **stay the same** with duplicates

**Expected Behavior**: Duplicates don't add new information → values should be invariant

**Metric**: Coefficient of Variance (CoV) - lower is better

### Datasets Tested
- **robustness.py**: MNIST, CIFAR-10, tabular datasets
- **robustness_imagenet.py**: ImageNet (large-scale validation)

---

## Scenario Comparison

| Scenario | Script | Metric | What It Tests |
|----------|--------|--------|---------------|
| **MPerf Correlation** | boostrap.py | Pearson Correlation | Do values predict performance on images? |
| **MPerf Correlation** | boostrap_tabular.py | Pearson Correlation | Do values predict performance on tabular data? |
| **MPerf Correlation** | boostrap_reg.py | Pearson Correlation | Do values predict performance on regression? |
| **Noise Robustness** | robustness.py | Value Trend | Do values decrease with label noise? |
| **Diversity Sensitivity** | robustness.py | Spearman/Pearson | Do values increase with diversity? |
| **Replication Awareness** | robustness.py | Coefficient of Variance | Are values invariant to duplicates? |
| **Large-Scale Robustness** | robustness_imagenet.py | Multiple | Does robustness hold at ImageNet scale? |

---

## Bootstrap Variants

### Three Bootstrap Scripts for MPerf Correlation

1. **bootstrap.py** → General image datasets + MNIST, CIFAR-10
2. **bootstrap_tabular.py** → CovType dataset with MLP
3. **bootstrap_reg.py** → CASP dataset for regression tasks

All generate 100 bootstraps to measure Pearson correlation with performance.

---

## Data Valuation Methods Evaluated

- **DAVINZ** - Neural network based valuation
- **OT** - Optimal transport approach
- **RV** - Regression value baseline
- **NTK** - Neural tangent kernel
- **MMD** - Maximum mean discrepancy

---

## Key Insights

**Model Performance**: Validates that estimated values predict actual impact on model performance

**Robustness**: Ensures valuation methods maintain reliability under noisy/imperfect data

**Diversity**: Confirms methods properly reward diverse, informative samples

**Replication**: Validates methods don't overvalue redundant duplicates
