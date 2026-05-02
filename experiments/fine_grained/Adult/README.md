HIGH\_CONFIG = {

&#x20;   "DataOOB":            {"num\_models": 10,      "proportion": 0.1},

&#x20;   "InfluenceSubsample": {"num\_models": 300000,  "subset\_size": 3000},

&#x20;   "DataBanzhaf":        {"num\_models": 350000},

&#x20;   "DVRL":               {"rl\_epochs": 3000,     "rl\_batch\_size": 32},

&#x20;   "KNNShapley":         {"k\_neighbors": 5000},

&#x20;   "AKShapley":          {"k\_neighbors": 10, "n\_hash\_table": 100, "eps": 0.01, "alpha": 0.5},

&#x20;   "LAVA":               {"lam\_y": 10},

&#x20;   "SAVA":               {"batch\_size": 1024, "lam\_x": 1.0, "lam\_y": 10.0, "p": 2, "blur": 0.05, "mode": "cls", "debug": True, "stratified\_batches": True},

&#x20;   "LeaveOneOut":        {},

&#x20;   "RandomEvaluator":    {},

}



LOW\_CONFIG = {

&#x20;   "DataOOB":            {"num\_models": 10,      "proportion": 0.1},

&#x20;   "InfluenceSubsample": {"num\_models": 300000,  "subset\_size": 100},

&#x20;   "DataBanzhaf":        {"num\_models": 100000},

&#x20;   "DVRL":               {"rl\_epochs": 5000,     "rl\_batch\_size": 64},

&#x20;   "KNNShapley":         {"k\_neighbors": 10},

&#x20;   "AKShapley":          {"k\_neighbors": 10, "n\_hash\_table": 100, "eps": 0.01, "alpha": 0.5},

&#x20;   "LAVA":               {"lam\_y": 10},

&#x20;   "SAVA":               {"batch\_size": 1024, "lam\_x": 1.0, "lam\_y": 10.0, "p": 2, "blur": 0.05, "mode": "cls", "debug": True, "stratified\_batches": True},

&#x20;   "LeaveOneOut":        {},

&#x20;   "RandomEvaluator":    {},

}



