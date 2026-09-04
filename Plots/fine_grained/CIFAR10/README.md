## High and Low Value Parameter Configurations (CIFAR10)

### HIGH_CONFIG (high value)

- **A-KShapley**: `dist_rand=31.9286`, `alpha=0.5`, `eps=0.0005`, `k_neighbors=1000`, `n_hash_table=100`, `t=2.228`  
- **DOOB**: `num_models=1000`, `proportion=0.1`  
- **DShapley**: `mc_epochs=10`  
- **DVRL**: `rl_epochs=10000`, `rl_batch_size=32`  
- **InfSub**: `num_models=100`, `subset_size=200`  
- **IR-DShapley**: `batch_size=4096`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=100`  
- **LAVA**: `lam_x=1.0`, `lam_y=10.0`  
- **LoGra-NR**: `lora=none`, `hessian=none`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=10.0`  

---

### LOW_CONFIG (low value)

- **A-KShapley**: `dist_rand=31.9286`, `alpha=0.5`, `eps=0.0001`, `k_neighbors=1000`, `n_hash_table=100`, `t=2.228`  
- **DOOB**: `num_models=1000`, `proportion=0.1`  
- **DShapley**: `mc_epochs=10`  
- **DVRL**: `rl_epochs=100000`, `rl_batch_size=64`  
- **InfSub**: `num_models=10000`, `subset_size=200`  
- **IR-DShapley**: `batch_size=10000`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=1000`  
- **LAVA**: `lam_x=1.0`, `lam_y=10.0`  
- **LoGra-NR**: `lora=none`, `hessian=none`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=10.0`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
