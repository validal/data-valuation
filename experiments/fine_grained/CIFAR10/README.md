HIGH_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 0.1},
    "InfluenceSubsample": {"num_models": 100000, "subset_size": 4000},
    "DataBanzhaf":        {"num_models": 20000},
    "DVRL":               {"rl_epochs": 5000,  "rl_batch_size": 128},
    "DataShapley":        {"mc_epochs": 500},
    "KNNShapley":         {"k_neighbors": 50},
    "AKShapley":          {"k_neighbors": 100, "n_hash_table": 100, "eps": 0.001, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "AME":                {"num_models": 5000},
    "LeaveOneOut":        {},
    "SAVA":               {"batch_size": 1024},
    "RandomEvaluator":    {},
}

LOW_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 1.0},
    "InfluenceSubsample": {"num_models": 500000,  "proportion": 100},
    "DataBanzhaf":        {"num_models": 10000},
    "DVRL":               {"rl_epochs": 5000,  "rl_batch_size": 128},
    "KNNShapley":         {"k_neighbors": 10},
    "AKShapley":          {"k_neighbors": 100, "n_hash_table": 100, "eps": 0.001, "alpha": 0.5},
    "LAVA":               {"lam_y": 1},
    "SAVA":               {"batch_size": 1024},
    "RandomEvaluator":    {},
}
