## High and Low Value Parameter Configurations (Hep1K)

### HIGH_CONFIG (high value)

- **A-KShapley**: `dist_rand=7.2352`, `alpha=0.3`, `eps=0.01`, `k_neighbors=100`, `n_hash_table=50`, `t=2.251`  
- **DOOB**: `num_models=100`, `proportion=0.1`  
- **DShapley**: `mc_epochs=700`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=512`  
- **InfSub**: `num_models=100000`, `subset_size=128`  
- **IR-DShapley**: `batch_size=512`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=100`  
- **LAVA**: `lam_x=1.0`, `lam_y=100.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LOO**  
- **Rand**  
- **SAVA**: `batch_size=1024`, `lam_y=100.0`  

---

### LOW_CONFIG (low value)

- **A-KShapley**: `dist_rand=7.2352`, `alpha=0.1`, `eps=0.01`, `k_neighbors=10`, `n_hash_table=10`, `t=2.251`  
- **DOOB**: `num_models=100`, `proportion=1.0`  
- **DShapley**: `mc_epochs=700`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=512`  
- **InfSub**: `num_models=100000`, `subset_size=16`  
- **IR-DShapley**: `batch_size=32`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=10`  
- **LAVA**: `lam_x=1.0`, `lam_y=10.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LoGra-NR**: `lora=none`, `hessian=raw`  
- **LOO**  
- **Rand**  
- **SAVA**: `batch_size=1024`, `lam_y=10.0`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
