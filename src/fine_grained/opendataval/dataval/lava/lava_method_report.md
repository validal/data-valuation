# LAVA & OTDD Data Valuation: Detailed Method Report

> Components covered: `LavaEvaluator`, `LavaOOBEvaluator`, and underlying `DatasetDistance` (OTDD core) from `otdd.py`.

---
## 1. Conceptual Overview
LAVA (Label-Aware / Value-Aware) adapts Optimal Transport (OT) based Dataset Distance (OTDD) to produce per-point data values. Instead of training a predictive model, it builds an *augmented ground cost* between train and validation points combining feature dissimilarity and (optionally) label structure, then solves an entropic OT (Sinkhorn) problem to obtain dual potentials. A calibrated transformation of the train-side potentials yields a scalar value per training point. Lower values indicate potentially *detrimental* points (sign flipped for consistency with other evaluators).

OTDD supplies:
- A unified cost matrix mixing feature distances and label distances.
- Efficient computation of dual potentials via `geomloss.SamplesLoss` (entropic regularized OT / Sinkhorn divergence) producing potentials `(F_i, G_j)`.
- Optional feature embedding model (via `FeatureCost`) to replace raw feature space with learned representations (e.g., CNN features for images).

`LavaOOBEvaluator` approximates full LAVA by bootstrapping smaller subsets to reduce runtime for large datasets.

---
## 2. Mathematical Foundations
Let:  
- Train set: \(\{(x_i, y_i)\}_{i=1}^N\)  
- Validation set: \(\{(x'_j, y'_j)\}_{j=1}^M\)

### 2.1 Augmented Cost Construction
An augmented feature-label distance \(C_{ij}\) combines:
1. Feature term: \(C^{(x)}_{ij} = \tfrac{1}{p} \| x_i - x'_j \|^p\) (or learned embedding cost)  
2. Label term (if `lam_y > 0`): precomputed label-to-label Wasserstein distances \(W_{y_i, y'_j}\) scaled appropriately.

Overall ground cost:
\[ C_{ij} = \lambda_x C^{(x)}_{ij} + \lambda_y C^{(y)}_{ij}\, , \quad C^{(y)}_{ij} = \frac{W_{y_i, y'_j}}{p}. \]

Special Case (Regression Mode): If `lam_y == 0`, labels are appended as an extra feature dimension → pure feature OT; label term skipped.

### 2.2 Entropic Regularized OT / Sinkhorn
We seek coupling \(\pi \in \mathbb{R}^{N \times M}\) minimizing:
\[ \min_{\pi \ge 0} \sum_{i,j} \pi_{ij} C_{ij} + \varepsilon \sum_{i,j} \pi_{ij} (\log \pi_{ij} - 1) \] 
subject to \(\pi \mathbf{1} = a\), \(\pi^T \mathbf{1} = b\) with uniform marginals \(a_i = 1/N\), \(b_j = 1/M\).

`geomloss.SamplesLoss` solves the dual form with multi-scale or tensorized routines, producing potentials \(F_i, G_j\): they approximate gradients w.r.t. mass movement. Potentials are used as point-wise *scores*.

### 2.3 Calibration to Data Values
Raw train potentials \(F\) are transformed:  
Let \(n = N\), potentials flattened to \(f \in \mathbb{R}^n\). Calibration (in code):
\[ v_i = f_i \Big(1 + \frac{1}{n-1}\Big) - \frac{1}{n-1} \sum_{k} f_k. \]
Simplified (implementation variant): `f * (1 + 1/num_points) - mean(f)` (with `num_points = n-1`). Then multiplied by -1 to align semantics: *lower values → detrimental points*.

### 2.4 LavaOOB Approximation
Instead of full \(N \times M\) OT: sample subsets of size \(s_{tr}, s_{va}\) for `n_bootstrap` replicates; for each subset:
1. Compute OT potentials on subset.
2. Calibrate subset potentials.
3. Aggregate (sum + counts) back to original train indices.
4. Mean over counts → approximate full-data potential.
Reduces complexity from ~O(N M) cost matrix construction per run to multiple smaller problems.

---
## 3. Algorithms & Backends
### 3.1 GeomLoss `SamplesLoss`
Implements differentiable Sinkhorn divergence / entropic OT. Key parameters:
- `loss`: usually "sinkhorn" (entropic OT), can vary (e.g., "energy").
- `p`: cost exponent (often 2 for squared Euclidean scaled by 1/(2)).
- `blur`: sets \(\varepsilon \approx \text{blur}^p\). Larger blur → smoother, faster convergence, less sharp structure.
- `scaling`: geometric schedule between scales (<1). Lower scaling (e.g., 0.7) reduces scales → faster.
- `backend`: "tensorized" (exact pairwise cost), "multiscale" (hierarchical, needs KeOps), "online" (streaming large sets, needs KeOps). Code falls back to "tensorized" when KeOps unavailable.

### 3.2 Sinkhorn Plan (Transport Matrix) Construction
`transport_plan` method builds explicit cost matrix then runs stabilized iterative scaling (classic Sinkhorn):
```
for it in iterations:
    u = a / (K @ v)
    v = b / (K^T @ u)
```
with Gibbs kernel `K = exp(-C/epsilon)`; convergence when change in `u` below tolerance.

### 3.3 Label Distance Precomputation
`pwdist_exact` computes pairwise label distributions distances via nested OT over subsets grouped by label. This yields block matrix `W` of shape `(C_total, C_total)`. Important when `lam_y > 0` and classification tasks.

---
## 4. Parameters & Their Effects
| Parameter | Meaning | Impact / Guidance |
|-----------|---------|-------------------|
| `lam_x` | Feature weight | Higher emphasizes geometry; increase when labels noisy. |
| `lam_y` | Label weight | Emphasizes label-specific structure; set 0 for regression or to ignore labels. |
| `p` | OT order | Usually 2; p=1 (Manhattan) possible but less common. Influences sensitivity to outliers. |
| `entreg` | (Outer) entropy regularization | Controls smoothness if used directly; often superseded by `blur` in SamplesLoss path. |
| `blur` | Entropic scale | Larger blur → faster, more diffuse potentials; too large washes out fine-grain signal. |
| `loss` | GeomLoss mode | "sinkhorn" recommended. Divergence vs distance semantics differ with `debias`. |
| `feature_cost` | Feature metric | "euclidean" or custom embedding via `FeatureCost`; embeddings help with structured (image/text) data. |
| `scaling` | Multiscale epsilon schedule | Lower (~0.7) fewer scales, faster; too low may degrade accuracy. |
| `backend` | Computational backend | Force "tensorized" for small N (<10k). Use "multiscale" when KeOps installed and N large. |
| `truncate` | Multiscale radius | Prunes far-away interactions; smaller speeds up but risks accuracy loss. |
| `diameter` | Data diameter hint | Helps multiscale heuristics; supply if known to stabilize. |
| `outer_debias` | Debiased Sinkhorn | Use True for distance-like behavior; False yields raw potentials. |
| `lam_y==0` mode | Regression fallback | Appends y as feature; avoids label OT overhead entirely. |
| `n_bootstrap` (OOB) | Replication count | Higher reduces variance; start 20–50. |
| `train_sample_frac`, `valid_sample_frac` | Subset size fractions | Trade accuracy vs speed; 0.2 typical; increase if high variance. |
| `replace` (OOB) | Sampling mode | Bootstrap (True) enhances variability; without replacement lowers variance. |
| `blur` (OOB variant) | Local Sinkhorn smoothing | Keep near Lava blur for consistency. |

---
## 5. Complexity & Scaling
### 5.1 Full LAVA
- Cost matrix implicit via `SamplesLoss` unless transport plan requested.
- Tensorized complexity: O(N M D) for pairwise feature distances; memory O(N M).
- Multiscale/online reduces memory, near O((N+M) log(N+M)) with KeOps; requires installation.

### 5.2 LAVA OOB
- Each replicate complexity: O(s_tr s_va D).
- Total: O(n_bootstrap * s_tr * s_va * D). If `s_tr = αN`, `s_va = βM`, effective cost αβ times original plus multiplicative replicates vs full.
- Suitable for large N (>50k) when full matrix prohibitive.

### 5.3 Label Distances
Computing `W` involves OT between label-conditioned subsets; cost depends on number of classes and per-class sample sizes; can dominate when many classes with large imbalance.

---
## 6. When to Choose LAVA
Choose LAVA when:
1. You need model-free valuation (no predictive model training cost).  
2. Geometry + label alignment strongly influence utility (e.g., class separation matters).  
3. Dataset contains structured features (images) benefiting from embedding-based distances.  
4. You want interpretable potentials rather than stochastic sampling (as in subset-based Shapley approximations).  
5. Data size moderate (≤ 20k points) or KeOps available for scaling.

Choose LavaOOB when:
- Full OT cost/memory too high.  
- Accept approximate valuations via repeated subsets.  
- Need early coarse ranking before committing to full run.

Avoid LAVA when:
- Label noise extremely high and geometry alone insufficient (consider lam_y=0 regression mode + other evaluators).  
- Extremely large N without KeOps (use OOB or different scalable evaluator).  
- Very high-dimensional sparse features where Euclidean distances are poorly informative (consider embedding transformations first).

---
## 7. Tuning Guidelines
| Objective | Adjustment |
|-----------|------------|
| Reduce runtime | Increase `blur`; use "tensorized" for small N; use OOB with lower fractions. |
| Sharper differentiation | Decrease `blur`; ensure `lam_y` > 0 if labels informative; consider embeddings. |
| Mitigate label imbalance | Lower `lam_y`; or reweight classes externally before OTDD (not implemented here). |
| Regression task | Set `lam_y=0` (auto concatenation); scale features. |
| Large dataset & memory pressure | Install PyKeOps; set `backend="multiscale"`; moderate `truncate`. |
| High variance OOB values | Increase `n_bootstrap`, raise subset fractions, switch `replace=False`. |

Rule-of-thumb starting points:
- Classification, balanced, medium size (N~10k): `lam_x=1.0`, `lam_y=1.0`, `blur=0.05`, `p=2`, backend auto.
- Regression (append y): `lam_y=0`, `lam_x=1.0`, `blur=0.05`.
- OOB approximation: `n_bootstrap=30`, `train_sample_frac=0.25`, `valid_sample_frac=0.25`.

---
## 8. Pitfalls & Diagnostics
| Symptom | Cause | Remedy |
|---------|-------|--------|
| Uniform or near-zero values | Over-smoothed (blur too large) or lam_y=0 with low feature variance | Decrease `blur`; normalize features; consider embeddings. |
| Exploding memory | Full cost with large N×M | Use OOB or install KeOps for multiscale backend. |
| Slow convergence | Very small blur / high p | Increase `blur`; verify backend selection; raise scaling (e.g., 0.9). |
| No label influence | `lam_y=0` or class mapping failed | Ensure `lam_y>0`; confirm labels properly encoded (integers or one-hot). |
| Negative values confusion | Sign flip for detriment alignment | Accept convention; remove `-1` factor if preferring raw potentials. |
| OOB high variance | Too few bootstrap replicates | Increase `n_bootstrap`; ensure subset sizes ≥ 2 per class when possible. |

Debug suggestions (`debug=True`):
- Initial stats of embeddings & labels.  
- For OOB first subset sizes + sample costs.  
- Dual potentials shapes.  
- Cost matrix min/max/mean when generating transport plan.  

---
## 9. Calibration Rationale
Raw `F_i` potentials approximate gradient of the entropic OT objective w.r.t. mass at point i. They are relative, up to additive constants; calibration normalizes scale and centers distribution to highlight influence directions. The added `(1 + 1/n)` factor compensates for sample-size bias while mean subtraction centers global mass effect.

---
## 10. Regression Mode Specifics (`lam_y==0`)
- Label term skipped entirely → `_skip_labels=True`.  
- y appended to feature vector providing continuous target context without categorical OT overhead.  
- Feature cost forced to Euclidean (embedding cost would mix target dimension incorrectly).  
- Recommended: Scale target into similar range as features (e.g., standardize) to avoid domination.

---
## 11. Lava vs Other Valuators
| Method | Model-Free | Handles Labels Geometrically | Complexity | Stochasticity |
|--------|-----------|------------------------------|------------|--------------|
| LAVA | Yes | Yes (label OT) | O(NM) / multiscale | Low |
| KNNShapley | Yes | Indirect (matching labels only) | O(NM) distances | Low |
| DVRL | No (RL model) | Implicit via reward | Data/model training cost | Higher |
| AME / Bagging | No | Via model performance | Subset training repeated | Medium |
| InfluenceSubsample | No | Via model predictions | Many subset trainings | Medium/High |
| Random/OOB variants | Yes (subset OT) | Approximate | Configurable | Medium |

---
## 12. Practical Workflow
1. Prepare / normalize features; optionally supply `embedding_model`.  
2. Decide: full LAVA vs OOB (based on N, memory).  
3. Set `lam_x`, `lam_y` (start 1.0/1.0).  
4. Choose `blur` (0.05–0.1 typical); confirm backend fallback (KeOps?).  
5. Run `train_data_values()`; optionally save transport plan for analysis.  
6. Extract values from `evaluate_data_values()`.  
7. Diagnose distribution (histogram); tune blur or lam parameters for spread.  
8. (OOB) adjust fractions / bootstraps if variance high.  

---
## 13. Saving Artifacts
- Transport plan: `save_transport_plan(path)` → coupling matrix (N×M).  
- OOB indices: `save_bootstrap_indices(path)` (JSON or NPZ).  
- For interpretability: inspect cost matrix extremes to identify unusual points.

---
## 14. Extension Ideas
- Class weighting: modify label distances weighting per class count.  
- Adaptive blur: start large, anneal smaller for sharper potentials.  
- Hybrid OOB: large points use OOB; small subset do full.  
- Dimensionality reduction pipeline preceding OT (PCA, UMAP embeddings).  

---
## 15. Recommended Default Profiles
| Scenario | Settings |
|----------|----------|
| Balanced classification (≤10k pts) | lam_x=1, lam_y=1, p=2, blur=0.05, backend auto |
| Large classification (≥50k) with KeOps | lam_x=1, lam_y=1, blur=0.1, backend=multiscale |
| Regression | lam_y=0, lam_x=1, blur=0.05, feature scaling required |
| Fast approximate | Use LavaOOB: n_bootstrap=20, fractions=0.2 |
| High precision | Full LAVA, blur=0.03, scaling=0.8, outer_debias=True |

---
## 16. Validation Checks
Before trusting results:
- Ensure wide enough spread in calibrated values (not all near zero).  
- Confirm sign convention (negative = detrimental).  
- Correlate removal of most negative points with validation performance increase.  
- Use bootstrap (OOB) variance: compute std across replicates per point; points with unstable valuations may need more replicates.

---
## 17. Known Limitations
- Memory footprint for full tensorized cost at large N×M.  
- KeOps dependency required for efficient multiscale/online backends; fallback may slow large runs.  
- Label distance accuracy declines with severe class imbalance (few samples per class).  
- Entropic regularization can blur fine-grained feature differences (trade-off necessary).

---
## 18. Debugging Checklist
| Step | Check |
|------|-------|
| Pre-run | Shapes & dtypes printed with `debug=True`. |
| Label mapping | `y_train` and `y_valid` remapped contiguous; counts per class reasonable. |
| Backend | If KeOps missing, fallback message appears; confirm backend="tensorized". |
| Potentials | `dual_sol` returns two tensors; shapes (N), (M). |
| Calibration | Resulting distribution mean ~0 after transformation. |
| OOB variance | Increase `n_bootstrap` until per-point stderr stabilizes. |

---
## 19. Implementation Highlights
- Reindex labels to contiguous blocks ensures combined label distance matrix alignment.  
- Regression mode concatenation avoids label OT overhead.  
- Dual potentials obtained with `loss.potentials=True`; debias toggled off there to fetch potentials.  
- Calibration formula simple linear transformation; sign inversion standardizes interpretation vs other valuators.

---
## 20. Example Minimal Usage
```python
from opendataval.dataval.lava.lava import LavaEvaluator
lava = LavaEvaluator(lam_x=1.0, lam_y=1.0, p=2, blur=0.05, device=torch.device('cuda'))
lava.input_data(x_train, y_train, x_valid, y_valid)
lava.train_data_values()
values = lava.evaluate_data_values()
``` 
OOB variant:
```python
from opendataval.dataval.lava.lava import LavaOOBEvaluator
oob = LavaOOBEvaluator(n_bootstrap=30, train_sample_frac=0.25, valid_sample_frac=0.25)
oob.input_data(x_train, y_train, x_valid, y_valid)
oob.train_data_values()
values_oob = oob.evaluate_data_values()
```

---
## 21. Interpreting Outputs
- Sorted ascending values: candidates for removal or cleaning (potentially mislabeled / detrimental).  
- Sorted descending: highly beneficial, preserve or prioritize labeling/augmentation.  
- Combine with other evaluators (e.g., intersection of low LAVA & low KNNShapley strengthens suspicion).

---
## 22. Future Optimization Paths
- Vectorize calibration + aggregation across bootstraps.  
- GPU memory pooling for repeated OT computations.  
- Label distance caching across runs with same dataset.  
- Mixed-precision (fp16) for large tensorized cost computations.

---
## 23. Glossary
| Term | Definition |
|------|------------|
| OT | Optimal Transport; mapping mass between distributions minimizing cost. |
| Sinkhorn | Iterative scaling algorithm for entropic regularized OT. |
| Potentials | Dual variables of OT problem indicating per-point marginal contribution. |
| Blur | GeomLoss parameter controlling entropic strength. |
| KeOps | Library enabling symbolic lazy tensor operations for large-scale distances. |
| Debiased Sinkhorn | Adjustment removing entropic bias for sharper distance estimate. |
| FeatureCost | Wrapper applying embedding models during cost computation. |
| OOB | Out-of-Bag approximation via subset bootstrapping. |

---
## 24. Summary
LAVA leverages entropic OT (Sinkhorn) over a feature + label augmented cost to produce calibrated dual potentials as data values, offering a powerful model-free valuation method sensitive to geometric and label structure. Parameter choices balance fidelity, computational cost, and robustness. The OOB variant enables scaling to large datasets by trading exactness for statistical approximation. Proper tuning of `blur`, `lam_x/lam_y`, backend, and (for OOB) bootstrap configuration yields stable, interpretable valuations.

---
*End of Report*
