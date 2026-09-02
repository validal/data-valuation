"""Create :py:class:`~opendataval.dataval.DataEvaluator` to quantify the value of data.

Data Evaluator
==============

.. currentmodule:: opendataval.dataval

Provides an ABC for DataEvaluator to inherit from. The work flow is as follows:
:py:class:`~opendataval.dataloader.Register`,
:py:class:`~opendataval.dataloader.DataFetcher`
-> :py:class:`~opendataval.dataval.DataEvaluator`
-> :py:mod:`~opendataval.experiment.exper_methods`



Catalog
-------
.. autosummary::
    :toctree: generated/

    DataEvaluator
    ModelMixin
    ModelLessMixin
    AME
    DVRL
    InfluenceFunction
    InfluenceSubsample
    TracIn
    TRAK
    InRunDataShapleyGhost
    LoGRA
    InRunDataShapley
    Kairos
    KNNShapley
    DataOob
    DataBanzhaf
    BetaShapley
    DataShapley
    LavaEvaluator
    LavaOOBEvaluator
    LeaveOneOut
    ShapEvaluator
    RandomEvaluator
    RobustVolumeShapley
    Sampler
    TMCSampler
    GrTMCSampler
"""
from opendataval.dataval.ame import AME
from opendataval.dataval.api import DataEvaluator, ModelLessMixin, ModelMixin
from opendataval.dataval.progress import ProgressBar, SimpleProgressBar, progress_range
from opendataval.dataval.csshap import ClassWiseShapley
from opendataval.dataval.dvrl import DVRL
from opendataval.dataval.dvrlshap import DVRLShap
from opendataval.dataval.influence import (
    InfluenceFunction,
    InfluenceSubsample,
    TracIn,
    TRAK,
    InRunDataShapleyGhost,
    LoGRA,
)
from opendataval.dataval.inrun_shapley import InRunDataShapley
from opendataval.dataval.knnshap import KNNShapley
from opendataval.dataval.gava import GAVA
from opendataval.dataval.loo import LOORemovalRanker
from opendataval.dataval.fem import ForgettingEvents

#from opendataval.dataval.lava import LavaEvaluator

from opendataval.dataval.lava import LavaEvaluator, LavaOOBEvaluator,ParallelLavaOOBEvaluator
from opendataval.dataval.lava import BatchwiseLavaEvaluator,HierarchicalLavaEvaluator
from opendataval.dataval.margcontrib import (
    BetaShapley,
    DataBanzhaf,
    DataBanzhafMargContrib,
    DataShapley,
    GrTMCSampler,
    LeaveOneOut,
    Sampler,
    ShapEvaluator,
    TMCSampler,
)
from opendataval.dataval.oob import DataOob
from opendataval.dataval.random import RandomEvaluator
from opendataval.dataval.volume import RobustVolumeShapley
from opendataval.dataval.kairos import Kairos
