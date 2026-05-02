HIGH_CONFIG = {
    "DataOOB":            {"num_models": 10,     "proportion": 0.2},
    "InfluenceSubsample": {"num_models": 1000000, "proportion": 0.1},
    "DataBanzhaf":        {"num_models": 1000000},
    "DVRL":               {"rl_epochs": 3000,   "rl_batch_size": 32},
    "DataShapley":        {"mc_epochs": 1000},
    "KNNShapley":         {"k_neighbors": 5000},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 20, "eps": 0.001, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "SAVA":               {"batch_size": 1024},
    "AME":                {"num_models": 50000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}

LOW_CONFIG = {
    "DataOOB":            {"num_models": 10,     "proportion": 0.2},
    "InfluenceSubsample": {"num_models": 100000, "proportion": 0.01},
    "DataBanzhaf":        {"num_models": 2000},
    "DVRL":               {"rl_epochs": 3000,   "rl_batch_size": 32},
    "DataShapley":        {"mc_epochs": 1000},
    "KNNShapley":         {"k_neighbors": 10},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 20, "eps": 0.001, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "SAVA":               {"batch_size": 1024},
    "AME":                {"num_models": 50000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}