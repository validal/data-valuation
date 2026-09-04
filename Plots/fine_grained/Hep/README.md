## High and Low Value Parameter Configurations (Hep)

### HIGH_CONFIG (high value)

- **A-KShapley**: `dist_rand=7.3622`, `alpha=0.5`, `eps=0.0001`, `k_neighbors=1000`, `n_hash_table=100`, `t=2.399`  
- **DOOB**: `num_models=10`, `proportion=0.1`  
- **DShapley**: `mc_epochs=100`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=64`  
- **InfSub**: `num_models=1000000`, `subset_size=16`  
- **IR-DShapley**: `batch_size=1024`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=100`  
- **LAVA**: `lam_y=100.0`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=10.0`  

---

### LOW_CONFIG (low value)

- **A-KShapley**: `dist_rand=7.3622`, `alpha=0.5`, `eps=0.0001`, `k_neighbors=1000`, `n_hash_table=100`, `t=2.399`  
- **DOOB**: `num_models=10`, `proportion=0.1`  
- **DShapley**: `mc_epochs=100`  
- **DVRL**: `rl_epochs=15000`, `rl_batch_size=64`  
- **InfSub**: `num_models=1000000`, `subset_size=16`  
- **IR-DShapley**: `batch_size=500000`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=100`  
- **LAVA**: `lam_y=10.0`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=10.0`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
