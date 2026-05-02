# -------------------------------------------------------
HIGH_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 1.0},
    "InfluenceSubsample": {"num_models": 300000, "proportion": 0.1},
    "DataBanzhaf":        {"num_models": 50000},
    "DVRL":               {"rl_epochs": 5000,  "rl_batch_size": 32},
    "DataShapley":        {"mc_epochs": 5000},
    "KNNShapley":         {"k_neighbors": 500},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 20, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "AME":                {"num_models": 30000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}

LOW_CONFIG = {
    "DataOOB":            {"num_models": 100,    "proportion": 1.0},
    "InfluenceSubsample": {"num_models": 10000,  "proportion": 0.1},
    "DataBanzhaf":        {"num_models": 100000},
    "DVRL":               {"rl_epochs": 2000,  "rl_batch_size": 32},
    "DataShapley":        {"mc_epochs": 1000},
    "KNNShapley":         {"k_neighbors": 10},
    "AKShapley":          {"k_neighbors": 10, "n_hash_table": 20, "eps": 0.01, "alpha": 0.5},
    "LAVA":               {"lam_y": 10},
    "AME":                {"num_models": 15000},
    "LeaveOneOut":        {},
    "RandomEvaluator":    {},
}

High and low value parameter chosen: the `tuning/high` and `tuning/low` directories
contain plots produced for the high and low parameter settings used in the paper.
