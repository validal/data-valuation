# Domain Adaptation - Data Valuation Methods

This Readme contains configuration settings for evaluating data valuation methods in domain adaptation scenarios.


## Best Configurations

The following configurations have been optimized for domain adaptation tasks. These settings balance computational efficiency with valuation accuracy.

### Configuration Parameters

```python
TUNING_CONFIG = {
    "InfluenceSubsample": {"num_models": 100000, "proportion": 0.2},
    "AME": {"num_models": 10000},
    "DataShapley": {"mc_epochs": 135},
    "DataBanzhaf": {"num_models": 80000},
    "KNNShapley": {"k_neighbors": 10},
    "DVRL": {"rl_epochs": 3000, "rl_batch_size": 256},
    "LAVA": {"lam_y": 10},
    "DataOOB": {"num_models": 100, "proportion": 0.2},
    "LeaveOneOut": {},
}