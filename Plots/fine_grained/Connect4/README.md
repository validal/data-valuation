## High and Low Value Parameter Configurations (Connect4)

### HIGH_CONFIG (high value)

- **A-KShapley**: `dist_rand=14.8177`, `alpha=0.5`, `eps=0.01`, `k_neighbors=1000`, `n_hash_table=100`, `t=2.062`  
- **DOOB**: `num_models=100`, `proportion=1.0`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=32`  
- **InfSub**: `num_models=1000000`, `subset_size=100`  
- **IR-DShapley**: `batch_size=100`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=1000`  
- **LAVA**: `lam_y=100.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=100.0`  

---

### LOW_CONFIG (low value)

- **A-KShapley**: `dist_rand=14.8177`, `alpha=0.1`, `eps=0.01`, `k_neighbors=10`, `n_hash_table=100`, `t=2.062`  
- **DOOB**: `num_models=100`, `proportion=1.0`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=32`  
- **InfSub**: `num_models=1000000`, `subset_size=16`  
- **IR-DShapley**: `batch_size=10000`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=5`  
- **LAVA**: `lam_y=10.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=100.0`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
