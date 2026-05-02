## High and Low Value Parameter Configurations (Connect4)

### HIGH_CONFIG (high value)

- **DOOB**: `num_models=10`, `proportion=0.1`  
- **InfSub**: `num_models=500000`, `subset_size=4000`  
- **DBanzhaf**: `num_models=100000`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=128`  
- **KShapley**: `k_neighbors=5000`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=100`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **SAVA**: `batch_size=1024`, `lam_x=1.0`, `lam_y=10.0`, `p=2`, `blur=0.05`, `mode=cls`  
- **LOO**  
- **Rand**

---

### LOW_CONFIG (low value)

- **DOOB**: `num_models=10`, `proportion=0.1`  
- **InfSub**: `num_models=400000`, `subset_size=100`  
- **DBanzhaf**: `num_models=100000`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=256`  
- **KShapley**: `k_neighbors=10`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=100`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **SAVA**: `batch_size=1024`, `lam_x=1.0`, `lam_y=10.0`, `p=2`, `blur=0.05`, `mode=cls`  
- **LOO**  
- **Rand**

---
