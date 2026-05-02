## High and Low Value Parameter Configurations (Hep1M)

### HIGH_CONFIG (high value)

- **DOOB**: `num_models=10`, `proportion=0.1`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=512`  
- **AKShapley**: `k_neighbors=100`, `n_hash_table=100`, `eps=1e-04`, `alpha=0.5`  
- **SAVA**: `batch_size=1024`  
- **Rand**

---

### LOW_CONFIG (low value)

- **DOOB**: `num_models=5`, `proportion=0.1`  
- **InfSub**: `num_models=200000`, `proportion=0.0001`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=512`  
- **AKShapley**: `k_neighbors=100`, `n_hash_table=100`, `eps=1e-04`, `alpha=0.5`  
- **SAVA**: `batch_size=1024`  
- **Rand**

---


