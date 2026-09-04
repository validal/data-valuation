## Value Configuration (Domain Adaptation)

Each method's single best-performing configuration (selected by highest `test_accuracy_mean` on a hyperparameter sweep; `LoGRA` is fixed to `pca+kfac` by convention). Unlike other datasets, there is no separate high/low removal split here.

### CONFIG

- **A-KShapley**: `k_neighbors=50`, `n_hash_table=20`, `eps=0.01`, `dist_rand=59.3879`, `t=2.51`, `alpha=0.1`  
- **DVRL**: `rl_epochs=3000`, `rl_batch_size=1024`  
- **DOOB**: `num_models=10000`, `proportion=0.5`  
- **DShapley**: `mc_epochs=70`  
- **IR-DShapley**: `epochs=10`, `batch_size=64`, `learning_rate=0.001`, `scheduler_type=none`  
- **InfSub**: `num_models=100000`, `proportion=0.1`  
- **Kairos**: `lambda_weight=0.97`, `unbiased=True`, `use_median_heuristic=True`, `num_samples=10000`, `batch_size=1024`  
- **KShapley**: `k_neighbors=10`, `batch_size=1024`  
- **LAVA**: `lam_x=1`, `lam_y=100`, `mode=cls`, `p=2`  
- **LOO**  
- **LoGra-PK**: `epochs=10`, `batch_size=64`, `learning_rate=0.001`, `lora=pca`, `hessian=kfac`  
- **Rand**  
- **SAVA**: `batch_size=1024`, `lam_x=1`, `lam_y=5`, `p=2`  

---

### Note

Plots in `evaluation/` correspond to the single per-method configuration above, selected from the tuning sweep in `tuning/`.
