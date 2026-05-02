HIGH_CONFIG = {
    "DataOOB":            {"num_models": 10,      "proportion": 0.1},
    "InfluenceSubsample": {"num_models": 500000,  "subset_size": 4000},
    "DataBanzhaf":        {"num_models": 100000},
    "DVRL":               {"rl_epochs": 3000,     "rl_batch_size": 128},
    "KNNShapley":         {"k_neighbors": 5000},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 100, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "SAVA":               {"batch_size": 1024, "lam_x": 1.0, "lam_y": 10.0, "p": 2, "blur": 0.05, "mode": "cls", "debug": True, "stratified_batches": True},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}

LOW_CONFIG = {
    "DataOOB":            {"num_models": 10,      "proportion": 0.1},
    "InfluenceSubsample": {"num_models": 400000,  "subset_size": 100},
    "DataBanzhaf":        {"num_models": 100000},
    "DVRL":               {"rl_epochs": 5000,     "rl_batch_size": 256},
    "KNNShapley":         {"k_neighbors": 10},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 100, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "SAVA":               {"batch_size": 1024, "lam_x": 1.0, "lam_y": 10.0, "p": 2, "blur": 0.05, "mode": "cls", "debug": True, "stratified_batches": True},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}
