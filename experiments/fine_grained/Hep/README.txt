HIGH_CONFIG = {
    "DataOOB":         {"num_models": 10, "proportion": 0.1},
    "DVRL":            {"rl_epochs": 10000, "rl_batch_size": 1024},
    "KNNShapley":      {"k_neighbors": 10000, "dist_rand": 7.3622, "n_hash_table": 100,
                           "eps": 0.0001,        "alpha": 0.5,        "t": 2.399,
                           "random_state": 1},
    "AKShapley":       {"k_neighbors": 10000, "n_hash_table": 100, "eps": 0.0001, "alpha": 0.5},
    "SAVA":            {"batch_size": 1024, "lam_x": 1.0, "lam_y": 10.0, "p": 2, "blur": 0.05,
                          "mode": "cls", "debug": True, "random_state": 5},
    "RandomEvaluator": {},
}

LOW_CONFIG = {
    "DataOOB":         {"num_models": 10, "proportion": 0.1},
    "DVRL":            {"rl_epochs": 10000, "rl_batch_size": 1024},
    "KNNShapley":      {"k_neighbors": 1000, "dist_rand": 7.3622, "n_hash_table": 100,
                           "eps": 0.0001,        "alpha": 0.5,        "t": 2.399,
                           "random_state": 1},
    "AKShapley":       {"k_neighbors": 10000, "n_hash_table": 100, "eps": 0.0001, "alpha": 0.5},
    "SAVA":            {"batch_size": 1024, "lam_x": 1.0, "lam_y": 10.0, "p": 2, "blur": 0.05,
                          "mode": "cls", "debug": True, "random_state": 5},
    "RandomEvaluator": {},
}
