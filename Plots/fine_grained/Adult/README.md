## High and Low value  Parameter Configurations

### HIGH_CONFIG (high value)

- **DataOOB**: `num_models=10`, `proportion=0.1`  
- **InfluenceSubsample**: `num_models=300000`, `subset_size=3000`  
- **DataBanzhaf**: `num_models=350000`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=32`  
- **KNNShapley**: `k_neighbors=5000`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=100`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **SAVA**: `batch_size=1024`, `lam_x=1.0`, `lam_y=10.0`, `p=2`, `blur=0.05`, `mode=cls`  
- **LeaveOneOut**  
- **RandomEvaluator**

---

### LOW_CONFIG (low value)

- **DataOOB**: `num_models=10`, `proportion=0.1`  
- **InfluenceSubsample**: `num_models=300000`, `subset_size=100`  
- **DataBanzhaf**: `num_models=100000`  
- **DVRL**: `rl_epochs=5000`, `rl_batch_size=64`  
- **KNNShapley**: `k_neighbors=10`  
- **AKShapley**: `k_neighbors=10`, `n_hash_table=100`, `eps=0.01`, `alpha=0.5`  
- **LAVA**: `lam_y=10`  
- **SAVA**: `batch_size=1024`, `lam_x=1.0`, `lam_y=10.0`, `p=2`, `blur=0.05`, `mode=cls`  
- **LeaveOneOut**  
- **RandomEvaluator**

---

### Note

The evaluation contain plots generated using these configurations.
