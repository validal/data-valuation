
HIGH_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 0.1},
    "InfluenceSubsample": {"num_models": 100000, "proportion": 0.2},
    "DataBanzhaf":        {"num_models": 100000},
    "DVRL":               {"rl_epochs": 5000,  "rl_batch_size": 128},
    "DataShapley":        {"mc_epochs": 500},
    "KNNShapley":         {"k_neighbors": 5},
    "AKShapley":          {"k_neighbors": 5, "n_hash_table": 20, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 5},
    "AME":                {"num_models": 5000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}

LOW_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 0.1},
    "InfluenceSubsample": {"num_models": 15000,  "proportion": 0.1},
    "DataBanzhaf":        {"num_models": 15000},
    "DVRL":               {"rl_epochs": 2000,  "rl_batch_size": 128},
    "DataShapley":        {"mc_epochs": 500},
    "KNNShapley":         {"k_neighbors": 100},
    "AKShapley":          {"k_neighbors": 5, "n_hash_table": 20, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 5},
    "AME":                {"num_models": 5000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}
