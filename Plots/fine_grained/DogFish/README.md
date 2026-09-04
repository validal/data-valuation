## High and Low Value Parameter Configurations (DogFish)

### HIGH_CONFIG (high value)

- **A-KShapley**: `dist_rand=20.0395`, `alpha=0.1`, `eps=0.01`, `k_neighbors=100`, `n_hash_table=50`, `t=2.329`  
- **DOOB**: `num_models=10`, `proportion=1.0`  
- **DShapley**: `mc_epochs=180`  
- **DVRL**: `rl_epochs=2000`, `rl_batch_size=128`  
- **InfSub**: `num_models=10000`, `subset_size=200`  
- **IR-DShapley**: `batch_size=32`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=10`  
- **LAVA**: `lam_y=10.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=10.0`  

---

### LOW_CONFIG (low value)

- **A-KShapley**: `dist_rand=20.0395`, `alpha=0.1`, `eps=0.01`, `k_neighbors=10`, `n_hash_table=50`, `t=2.329`  
- **DOOB**: `num_models=10`, `proportion=0.1`  
- **DShapley**: `mc_epochs=180`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=512`  
- **InfSub**: `num_models=10000`, `subset_size=200`  
- **IR-DShapley**: `batch_size=512`  
- **Kairos**: `lambda_weight=0.97`  
- **KShapley**: `k_neighbors=10`  
- **LAVA**: `lam_y=10.0`  
- **LoGra-PK**: `lora=pca`, `hessian=kfac`  
- **LOO**  
- **Rand**  
- **SAVA**: `lam_y=5.0`  

---

### Note

Plots in `evaluation/` (`high_value_removal.pdf`, `low_value_removal.pdf`, `mislabeled_detection.pdf`) correspond to the high- and low-value configurations above.
