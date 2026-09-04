## High and Low Value Parameter Configurations (CIFAR10_R9)

### HIGH_CONFIG (high value)

- **DOOB**: `num_models=50`, `proportion=0.1`  
- **IR-DShapley**: `val_batch_size=512`  
- **Rand**  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **A-KShapley**: `k_neighbors=1000`, `eps=0.0001`, `alpha=0.5`, `n_hash_table=100`, `dist_rand=22.5189`, `t=2.331`, `valid_chunk=10000`  
- **Kairos**: `lambda_weight=0.97`, `unbiased=True`, `use_median_heuristic=True`, `num_samples=10000`  

---

### LOW_CONFIG (low value)

- **DOOB**: `num_models=100`, `proportion=0.1`  
- **IR-DShapley**: `val_batch_size=128`  
- **Rand**  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **A-KShapley**: `k_neighbors=1000`, `eps=0.0001`, `alpha=0.5`, `n_hash_table=100`, `dist_rand=22.5189`, `t=2.331`, `valid_chunk=10000`  
- **Kairos**: `lambda_weight=0.97`, `unbiased=True`, `use_median_heuristic=True`, `num_samples=10000`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
