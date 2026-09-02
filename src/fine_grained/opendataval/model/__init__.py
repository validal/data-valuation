r"""Prediction models to be trained, predict, and evaluated.

Models
======

.. currentmodule:: opendataval.model

:py:class:`Model` is an ABC used to take an existing model and make it compatible with
the :py:class:`~opendataval.dataval.DataEvaluator` and other related objects.

API
---
.. autosummary::
    :toctree: generated/

    Model
    GradientModel
    ModelFactory

Torch Mixins
------------
.. autosummary::
    :toctree: generated/

    TorchClassMixin
    TorchRegressMixin
    TorchPredictMixin
    TorchGradMixin

Sci-kit learn wrappers
----------------------
.. autosummary::
    :toctree: generated/

    ClassifierSkLearnWrapper
    ClassifierUnweightedSkLearnWrapper
    RegressionSkLearnWrapper


Default Hyperparameters
-----------------------

.. math::

    \newcommand\T{\Rule{0pt}{1em}{.3em}}

    \begin{array}{llll}
    \hline
    \textbf{Algorithm} & \textbf{Hyperparameter} & \textbf{Default Value} & \textbf{Key word argument} \\
    \hline
    \mbox{Logistic Regression}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\ & \mbox{loss function}
        & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{MLP Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{BERT Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.001 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAMW} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{LeNet-5 Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{MLP Regression}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Mean Square Error} & \mbox{no} \\
    \hline
    \end{array}
"""
# Model Factory imports
from typing import Optional

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

from opendataval.dataloader import DataFetcher
from opendataval.model.api import (
    ClassifierSkLearnWrapper,
    ClassifierUnweightedSkLearnWrapper,
    Model,
    RegressionSkLearnWrapper,
    TorchClassMixin,
    TorchPredictMixin,
    TorchRegressMixin,
)
from opendataval.model.bert import BertClassifier
from opendataval.model.grad import GradientModel, TorchGradMixin
from opendataval.model.lenet import LeNet
from opendataval.model.logistic_regression import LogisticRegression
from opendataval.model.mlp import ClassifierMLP, RegressionMLP, SimpleMLP
from opendataval.model.resnet import ResNet9, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet110

# Optional third-party model backends (installed separately)
try:  # XGBoost (sklearn-compatible API)
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

try:  # LightGBM (sklearn-compatible API)
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore
    LGBMRegressor = None  # type: ignore


def ModelFactory(  # noqa: C901 model factory tries to match name with long if-else
    model_name: str,
    fetcher: Optional[DataFetcher] = None,
    device: torch.device = torch.device("cpu"),
    *args,
    **kwargs,
) -> Model:
    """Factory to create prediction models from specified presets

    Model Factory that creates a specified mode, based on the input parameters, it is
    recommended to import the specific model and specify additional arguments instead of
    relying on the factory.

    Parameters
    ----------
    model_name : str
        Name of prediction model
    covar_dim : tuple[int, ...]
        Dimensions of the covariates, typically the shape besides first dimension
    label_dim : tuple[int, ...]
        Dimensions of the labels, typically the shape besides first dimension
    device : torch.device, optional
        Tensor device for acceleration, some models do not use this argument,
        by default torch.device("cpu")
    args : tuple[Any]
        Additional positional arguments passed to the Model constructor
    kwargs : tuple[Any]
        Additional key word arguments passed to the Model constructor

    Returns
    -------
    Model
        Preset model with the specified dimensions on the specified tensor device

    Raises
    ------
    ValueError
        Raises exception when model name is not matched
    """
    covar_dim, label_dim = fetcher.covar_dim, fetcher.label_dim
    # Robust num_classes computation for classification with optional integer labels
    num_classes = None
    if getattr(fetcher, "one_hot", False):
        # One-hot labels will have label_dim like (C,), integer labels yield (1,)
        if len(label_dim) == 1 and label_dim[0] > 1:
            num_classes = int(label_dim[0])
        else:
            # Try to infer from training labels if provided
            try:
                y_tr = getattr(fetcher, "y_train", None)
                if y_tr is None:
                    y_tr = getattr(fetcher, "labels", None)
                if y_tr is not None:
                    import numpy as _np
                    y_tr = _np.asarray(y_tr)
                    if y_tr.ndim == 1:
                        num_classes = int(_np.max(y_tr)) + 1 if y_tr.size else 1
                    elif y_tr.ndim > 1:
                        num_classes = int(y_tr.shape[1])
            except Exception:
                pass
    # Fallback if still None
    if num_classes is None:
        num_classes = int(label_dim[0]) if len(label_dim) else 1
    if model_name is None:
        return None
    model_name = model_name.lower()

    if model_name == "logisticregression":
        input_dim = int(np.prod(covar_dim))
        return LogisticRegression(input_dim, num_classes, *args, **kwargs).to(device)
    elif model_name == "classifiermlp":
        return ClassifierMLP(*covar_dim, num_classes, *args, **kwargs).to(device)
    elif model_name in ("simplemlp", "mlp_mnist"):
        return SimpleMLP(*args, **kwargs).to(device)
    elif model_name == "regressionmlp":
        return RegressionMLP(*covar_dim, *label_dim, *args, **kwargs).to(device)
    elif model_name == "bertclassifier":
        return BertClassifier(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name == "lenet":
        return LeNet(
            num_classes=num_classes,
            gray_scale=covar_dim[0] == 1,  # 1 means grey
            *args,
            **kwargs,
        ).to(device)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name == "resnet34":
         return ResNet34(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name == "resnet50":
         return ResNet50(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name in ("resnet101", "rn101"):
         return ResNet101(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name in ("resnet152", "rn152"):
         return ResNet152(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name in ("resnet110", "rn110"):
         return ResNet110(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name in ("resnet9", "rn9"):
        return ResNet9(num_classes=num_classes, *args, **kwargs).to(device)
    elif model_name == "sklogreg":
        return ClassifierSkLearnWrapper(SkLogReg, num_classes, *args, **kwargs)
    elif model_name == "skmlp":
        return ClassifierUnweightedSkLearnWrapper(
            MLPClassifier, num_classes, *args, **kwargs
        )
    elif model_name == "skknn":
        return ClassifierUnweightedSkLearnWrapper(
            KNeighborsClassifier, num_classes, num_classes, *args, **kwargs
        )
    elif model_name == "sklinreg":
        return RegressionSkLearnWrapper(LinearRegression, *args, **kwargs)
    # Decision Trees / Random Forests / SVMs (scikit-learn)
    elif model_name in (
        "decisiontree",
        "decisiontreeclassifier",
        "skdecisiontree",
        "skdt",
        "dt",
    ):
        return ClassifierSkLearnWrapper(DecisionTreeClassifier, num_classes, *args, **kwargs)
    elif model_name in (
        "decisiontreeregressor",
        "skdecisiontreeregressor",
        "skdtr",
        "dtr",
    ):
        return RegressionSkLearnWrapper(DecisionTreeRegressor, *args, **kwargs)
    elif model_name in (
        "randomforest",
        "randomforestclassifier",
        "skrandomforest",
        "skrf",
        "rf",
    ):
        return ClassifierSkLearnWrapper(RandomForestClassifier, num_classes, *args, **kwargs)
    elif model_name in (
        "randomforestregressor",
        "skrandomforestregressor",
        "skrfr",
        "rfr",
    ):
        return RegressionSkLearnWrapper(RandomForestRegressor, *args, **kwargs)
    elif model_name in (
        "svm",
        "svc",
        "sksvm",
        "sksvc",
    ):
        # `ClassifierSkLearnWrapper` uses `predict_proba`, so ensure it's enabled.
        svm_kwargs = {**kwargs}
        svm_kwargs.setdefault("probability", True)
        return ClassifierSkLearnWrapper(SVC, num_classes, *args, **svm_kwargs)
    elif model_name in (
        "svr",
        "svmregressor",
        "svmregression",
        "sksvr",
    ):
        return RegressionSkLearnWrapper(SVR, *args, **kwargs)
    # XGBoost family (requires `xgboost` package)
    elif model_name in ("xgboost", "xgb", "xgbclassifier"):
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        return ClassifierSkLearnWrapper(XGBClassifier, num_classes, *args, **kwargs)
    elif model_name in ("xgbregressor", "xgbregression"):
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        return RegressionSkLearnWrapper(XGBRegressor, *args, **kwargs)
    # LightGBM family (requires `lightgbm` package)
    elif model_name in ("lightgbm", "lgbm", "lgbmclassifier"):
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        return ClassifierSkLearnWrapper(LGBMClassifier, num_classes, *args, **kwargs)
    elif model_name in ("lgbmregressor", "lightgbmregressor"):
        if LGBMRegressor is None:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        return RegressionSkLearnWrapper(LGBMRegressor, *args, **kwargs)
    else:
        raise ValueError(f"{model_name} is not a valid predefined model")
