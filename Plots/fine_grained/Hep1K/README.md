## High and Low Value Parameter Configurations (Hep1K)

### HIGH_CONFIG (high value)

- **DOOB**: `num_models=100`, `proportion=1.0`  
- **InfSub**: `num_models=300000`, `proportion=0.1`  
- **DBanzhaf**: `num_models=50000`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=32`  
- **DShapley**: `mc_epochs=5000`  
- **KShapley**: `k_neighbors=500`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=20`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **AME**: `num_models=30000`  
- **LOO**  
- **Rand**

---

### LOW_CONFIG (low value)

- **DOOB**: `num_models=100`, `proportion=1.0`  
- **InfSub**: `num_models=10000`, `proportion=0.1`  
- **DBanzhaf**: `num_models=100000`  
- **DVRL**: `rl_epochs=2000`, `rl_batch_size=32`  
- **DShapley**: `mc_epochs=1000`  
- **KShapley**: `k_neighbors=10`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=20`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **AME**: `num_models=15000`  
- **LOO**  
- **Rand**

---

### Note

Plots in `tuning/high` and `tuning/low` correspond to the high- and low-value configurations.
