from typing import Union, Optional

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from opendataval.dataloader.fetcher import DataFetcher
from opendataval.dataloader.util import IndexTransformDataset
from opendataval.util import FuncEnum


def mix_labels(
    fetcher: DataFetcher,
    noise_rate: float = 0.2,
    rng: Optional[np.random.RandomState] = None,
    noise_random_state: Optional[np.random.RandomState] = None,
) -> dict[str, np.ndarray]:
    """Mixes y_train labels of a DataFetcher, adding noise to data.

    For a given set of unique labels, we shift the label forward up to n-1 steps. This
    prevents selecting the same label when noise is added.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher object housing the data to have noise added to
    noise_rate : float
        Proportion of labels to add noise to

    Returns
    -------
    dict[str, np.ndarray]
        dictionary of updated data points

        - **"y_train"** -- Updated training labels mixed
        - **"y_valid"** -- Updated validation labels mixed
        - **"noisy_train_indices"** -- Indices of training data set with mixed labels
    """
    # Choose RNG precedence: explicit noise_random_state > rng > fetcher.noise_random_state > fetcher.random_state
    chosen_rng = (
        noise_random_state
        if noise_random_state is not None
        else rng
        if rng is not None
        else getattr(fetcher, "noise_random_state", None)
    )
    rs = check_random_state(chosen_rng if chosen_rng is not None else fetcher.random_state)

    y_train, y_valid = fetcher.y_train, fetcher.y_valid
    num_train, num_valid = len(y_train), len(y_valid)

    train_replace = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    valid_replace = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)

    # Gets unique classes and mapping of training data set to those classes
    train_classes, train_mapping = np.unique(y_train, return_inverse=True, axis=0)
    valid_classes, valid_mapping = np.unique(y_valid, return_inverse=True, axis=0)

    # For each label, we determine a shift to pick a new label
    # The new label cannot be the same as the prior, therefore start at 1
    train_shift = rs.choice(len(train_classes) - 1, round(num_train * noise_rate)) + 1
    valid_shift = rs.choice(len(valid_classes) - 1, round(num_valid * noise_rate)) + 1

    train_noise = (train_mapping[train_replace] + train_shift) % len(train_classes)
    valid_noise = (valid_mapping[valid_replace] + valid_shift) % len(valid_classes)

    y_train[train_replace] = train_classes[train_noise]
    #y_valid[valid_replace] = valid_classes[valid_noise]

    return {
        "y_train": y_train,
        "y_valid": y_valid,
        "noisy_train_indices": train_replace,
    }


# def add_gauss_noise(
#     fetcher: DataFetcher,
#     noise_rate: float = 0.1,
#     mu: float = 0.0,
#     sigma: float = 10.0,
#     noise_features: bool = False,
#     noise_targets: bool = True,
#     target_outlier_scale: float = 1.0,
# ) -> dict[str, Union[Dataset, np.ndarray]]:
#     """Add Gaussian noise to training covariates and/or targets. Validation data unchanged.

#     Parameters
#     ----------
#     fetcher : DataFetcher
#         DataFetcher object housing the data to have noise added to
#     noise_rate : float
#         Proportion of training samples to add noise to
#     mu : float, optional
#         Mean of Gaussian distribution for noise, by default 0
#     sigma : float, optional
#         Standard deviation of Gaussian distribution, by default 100.0
#     noise_features : bool, optional
#         Whether to add noise to features, by default False
#     noise_targets : bool, optional
#         Whether to add noise to targets, by default True
#     target_outlier_scale : float, optional
#         Multiplier for target noise (larger = more extreme outliers), by default 10.0

#     Returns
#     -------
#     dict[str, Union[Dataset, np.ndarray]]
#         Dictionary with updated data:
#         - "x_train": Updated training covariates (with noise)
#         - "x_valid": Original validation covariates (unchanged)
#         - "y_train": Updated training targets (with noise, if noise_targets=True)
#         - "y_valid": Original validation targets (unchanged)
#         - "noisy_train_indices": Indices of corrupted training samples
#     """
#     rs = check_random_state(fetcher.random_state)

#     # Convert data to numpy arrays
#     x_train = np.array(fetcher.x_train, dtype=np.float64)
#     x_valid = np.array(fetcher.x_valid, dtype=np.float64)  # Keep original
#     num_train = len(x_train)
#     feature_dim = fetcher.covar_dim

#     # Select random training indices to corrupt (validation unchanged)
#     noisy_train_idx = rs.choice(num_train, round(num_train * noise_rate), replace=False)

#     # Add noise to training features if requested (validation unchanged)
#     if noise_features:
#         # Generate large Gaussian noise (scaled by target_outlier_scale)
#         noise_train = rs.normal(mu, sigma * target_outlier_scale, 
#                                 size=(len(noisy_train_idx), *feature_dim))
        
#         # Apply noise only to training feature arrays
#         x_train[noisy_train_idx] = x_train[noisy_train_idx] + noise_train

#     # Add noise to training targets if requested (validation unchanged)
#     if noise_targets:
#         y_train = np.array(fetcher.y_train, dtype=np.float64)
#         y_valid = fetcher.y_valid  # Keep original validation targets
        
#         # Generate large Gaussian noise for training targets
#         if y_train.ndim == 1:
#             # 1D targets (e.g., regression with single output)
#             noise_y_train = rs.normal(mu, sigma * target_outlier_scale, 
#                                       size=len(noisy_train_idx))
#         else:
#             # Multi-dimensional targets
#             target_dim = y_train.shape[1:]
#             noise_y_train = rs.normal(mu, sigma * target_outlier_scale, 
#                                       size=(len(noisy_train_idx), *target_dim))
        
#         # Apply noise only to training target arrays (additive)
#         y_train[noisy_train_idx] = y_train[noisy_train_idx] + noise_y_train
#     else:
#         y_train = fetcher.y_train
#         y_valid = fetcher.y_valid

#     # Prepare return dictionary
#     out = {
#         "x_train": x_train,              # Noisy training features
#         "x_valid": x_valid,               # Original validation features (unchanged)
#         "y_train": y_train,               # Noisy training targets
#         "y_valid": y_valid,                # Original validation targets (unchanged)
#         "noisy_train_indices": noisy_train_idx,
#     }
    
#     return out
def add_gauss_noise(
    fetcher: DataFetcher, noise_rate: float = 0.2, mu: float = 0.0, sigma: float = 1.0
) -> dict[str, Union[Dataset, np.ndarray]]:
    """Add gaussian noise to covariates.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher object housing the data to have noise added to
    noise_rate : float
        Proportion of labels to add noise to
    mu : float, optional
        Center of gaussian distribution which noise is generated from, by default 0
    sigma : float, optional
        Standard deviation of gaussian distribution, by default 1

    Returns
    -------
    dict[str, np.ndarray]
        dictionary of updated data points

        - **"x_train"** -- Updated training covariates with added gaussian noise
        - **"noisy_train_indices"** -- Indices of training data set with mixed labels
    """
    rs = check_random_state(fetcher.random_state)

    x_train = np.array(fetcher.x_train, dtype=np.float64)
    x_valid = np.array(fetcher.x_valid, dtype=np.float64)
    x_test = np.array(fetcher.x_test, dtype=np.float64)
    num_train, num_valid, _ = len(x_train), len(x_valid), len(x_test)
    feature_dim = fetcher.covar_dim

    noisy_train_idx = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    #noisy_valid_idx = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)
    noise_train = rs.normal(mu, sigma, size=(len(noisy_train_idx), *feature_dim))
    #noise_valid = rs.normal(mu, sigma, size=(len(noisy_valid_idx), *feature_dim))

    if isinstance(x_train, Dataset):
        # We add a zero tensor at the top because noise only some indices have noise
        # added. For those that do not, they have the zero tensor added -> no change
        padded_noise_train = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_train])
       # padded_noise_valid = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_valid])
        noise_add_train = torch.tensor(padded_noise_train, dtype=torch.float)
       #noise_add_valid = torch.tensor(padded_noise_valid, dtype=torch.float)

        # A remapping to noisy index, in noise array, offset by 1 for non-noisy data
        # as the 0th index is the zero tensor from above
        remap_train = np.zeros((num_train,), dtype=int)
        remap_valid = np.zeros((num_valid,), dtype=int)
        remap_train[noisy_train_idx] = range(1, len(noisy_train_idx) + 1)
        #remap_valid[noisy_valid_idx] = range(1, len(noisy_valid_idx) + 1)

        x_train = IndexTransformDataset(
            x_train, lambda data, ind: (data + noise_add_train[remap_train[ind]])
        )
        #x_valid = IndexTransformDataset(
        #    x_valid, lambda data, ind: (data + noise_add_valid[remap_valid[ind]])
        #)
    else:
        x_train[noisy_train_idx] = x_train[noisy_train_idx] + noise_train
        #x_valid[noisy_valid_idx] = x_valid[noisy_valid_idx] + noise_valid

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,  # just to keep numpy structure
        "noisy_train_indices": noisy_train_idx,
    }

class NoiseFunc(FuncEnum):
    MIX_LABELS = FuncEnum.wrap(mix_labels)
    ADD_GAUSS_NOISE = FuncEnum.wrap(add_gauss_noise)
