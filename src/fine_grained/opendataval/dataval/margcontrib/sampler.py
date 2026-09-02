from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, TypeVar

import numpy as np
import torch
from opendataval.dataval.progress import ProgressBar, progress_range
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.util import ReprMixin

Self = TypeVar("Self", bound="Sampler")


class Sampler(ABC, ReprMixin):
    """Abstract Sampler class for marginal contribution based data evaluators.

    Many marginal contribution based data evaluators depend on a sampling method as
    they typically can be very computationally expensive. The Sampler class provides
    a blue print of required methods to be used and the following samplers provide ways
    of caching computed marginal contributions if given a `"cache_name"`.
    """

    def set_evaluator(self, value_func: Callable[..., float]):
        """Sets the evaluator function to evaluate the utility of a coalition


        Parameters
        ----------
    value_func : Callable[..., float]
            T his function sets the utility function  which computes the utility for a
            given coalition of indices.

        The following is an example of how the api would work in a DataEvaluator:
        ::
            self.sampler.set_evaluator(self._evaluate_model)
        """
        self.compute_utility = value_func

    @abstractmethod
    def set_coalition(self, coalition: torch.Tensor) -> Self:
        """Given the coalition, initializes data structures to compute marginal contrib.

        Parameters
        ----------
        coalition : torch.Tensor
            Coalition of data to compute the marginal contribution of each data point.
        """

    @abstractmethod
    def compute_marginal_contribution(self, *args, **kwargs) -> np.ndarray:
        """Given args and kwargs for the value func, computes marginal contribution.

        Returns
        -------
        np.ndarray
            Marginal contribution array per data point for each coalition size. Dim 0 is
            the index of the added data point, Dim 1 is the cardinality when the data
            point is added.
        """


class MonteCarloSampler(Sampler):
    """Monte Carlo sampler for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler. We take
    mc_epochs permutations and compute the marginal contributions. Simplest
    implementation but the least practical.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of outer epochs of MCMC sampling, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 1000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = "",
        random_state: Optional[RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.cache_name = None if cache_name is None else (cache_name or id(self))
        self.random_state = check_random_state(random_state)

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8

        return self

    def compute_marginal_contribution(self, *args, **kwargs):
        """Trains model to predict data values.

        Uses permutation sampling to find the marginal contribution of each data point,
        takes self.mc_epochs number of permutations.
        """
        # Checks if data values have already been computed
        if self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        if getattr(self, "marginal_contribution", None) is not None:
            return self.marginal_contribution

        for _ in progress_range(self.mc_epochs):
            self._calculate_marginal_contributions(*args, **kwargs)

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count

        if self.cache_name is not None:
            self.CACHE[self.cache_name] = self.marginal_contribution
        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs):
        """Compute marginal contribution through MC sampling.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_utility(coalition, *args, **kwargs)
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # to consider additional zero contributions
                self.marginal_count[
                    subset[(cutoff + 1) :], np.arange(cutoff + 1, len(subset))
                ] += 1
                break

            # update performance
            prev_perf = curr_perf

        return


class TMCSampler(Sampler):
    """TMCShapley sampler for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of outer epochs of MCMC sampling, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 1000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = "",
        random_state: Optional[RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.random_state = check_random_state(random_state)
        self.cache_name = None if cache_name is None else (cache_name or id(self))

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8

        return self

    def compute_marginal_contribution(self, *args, **kwargs):
        """Computes marginal contribution through TMC Shapley.

        Uses TMC-Shapley sampling to find the marginal contribution of each data point,
        takes self.mc_epochs number of samples.
        """
        # Checks if data values have already been computed
        if self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        for _ in progress_range(self.mc_epochs):
            self._calculate_marginal_contributions(*args, **kwargs)

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count

        if self.cache_name is not None:
            self.CACHE[self.cache_name] = self.marginal_contribution
        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs):
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_utility(coalition, *args, **kwargs)
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # to consider additional zero contributions
                self.marginal_count[
                    subset[(cutoff + 1) :], np.arange(cutoff + 1, len(subset))
                ] += 1
                break

            # update performance
            prev_perf = curr_perf

        return


class TMCSamplerBYnum_models(Sampler):
    """TMC-Shapley sampler that targets a total number of utility evaluations.

    Many methods report "num_models" (i.e., the number of utility evaluations or
    model fits/evaluations). In standard TMC, a single permutation performs many
    utility evaluations (baseline + one per added point until truncation), so the
    number of permutations is not equal to the number of models.

    This variant iterates permutations until the cumulative number of utility
    evaluations reaches or exceeds ``max_models``. It mirrors the logic of
    :class:`TMCSampler` but stops by a target count of utility calls instead of a
    fixed number of permutations.

    Parameters
    ----------
    max_models : int
        Target number of utility evaluations to perform in total. The actual
        count may slightly exceed this target because we complete the current
        permutation once started.
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for an object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        max_models: int = 10000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = "",
        random_state: Optional[RandomState] = None,
    ):
        if max_models is None or int(max_models) < 0:
            raise ValueError("max_models must be a non-negative integer")
        self.max_models = int(max_models)
        self.min_cardinality = min_cardinality
        self.random_state = check_random_state(random_state)
        self.cache_name = None if cache_name is None else (cache_name or id(self))
        # Counters (for transparency/fairness)
        self.total_permutations = 0
        self.total_utility_evals = 0

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8
        # Reset counters for a fresh run
        self.total_permutations = 0
        self.total_utility_evals = 0
        return self

    def compute_marginal_contribution(self, *args, **kwargs):
        """Compute marginal contribution through TMC with a utility-eval budget.

        Iteratively samples permutations and accumulates marginal increments until
        the cumulative number of utility evaluations (model fits/evals) reaches
        ``max_models``.
        """
        # Return cached result if available
        if self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        # If budget is zero, return zeros (counts prevent division-by-zero)
        if self.max_models == 0:
            self.marginal_contribution = np.zeros((self.num_points, self.num_points))
            if self.cache_name is not None:
                self.CACHE[self.cache_name] = self.marginal_contribution
            return self.marginal_contribution

        # Draw permutations until we exhaust the model budget
        while self.total_utility_evals < self.max_models:
            self._calculate_marginal_contributions(*args, **kwargs)
            self.total_permutations += 1

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count

        if self.cache_name is not None:
            self.CACHE[self.cache_name] = self.marginal_contribution
        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs):
        """Compute marginal contribution through TMC-Shapley algorithm.

        This increments ``self.total_utility_evals`` for each call to
        ``self.compute_utility(...)`` (baseline + per-added-point), and stops the
        permutation early if the budget is reached mid-way. When stopped early due to
        budget, we conservatively count the remaining positions as zeros (by
        incrementing ``marginal_count``) to keep the estimator unbiased relative to
        partial permutations, similar to the truncation handling.
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)
        self.total_utility_evals += 1

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # If we've already met/exceeded the budget, finish this permutation by
            # counting remaining positions as zeros and break.
            if self.total_utility_evals >= self.max_models:
                self.marginal_count[
                    subset[(cutoff + 1) :], np.arange(cutoff + 1, len(subset))
                ] += 1
                break

            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_utility(coalition, *args, **kwargs)
            self.total_utility_evals += 1
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n'
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # to consider additional zero contributions
                self.marginal_count[
                    subset[(cutoff + 1) :], np.arange(cutoff + 1, len(subset))
                ] += 1
                break

            # update performance
            prev_perf = curr_perf

        return

class GrTMCSampler(Sampler):
    """TMC Sampler with terminator for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.05
    max_mc_epochs : int, optional
        Max number of outer epochs of MCMC sampling, by default 100
    models_per_epoch : int, optional
        Number of model fittings to take per epoch prior to checking GR convergence,
        by default 100
    min_models : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default "" which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    GR_MAX = 100
    """Default maximum Gelman-Rubin statistic. Used for burn-in."""

    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_mc_epochs: int = 100,
        models_per_epoch: int = 100,
        min_models: int = 1000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = "",
        random_state: Optional[RandomState] = None,
        debug: bool = False,
        chain_dump_dir: Optional[str] = None,
        max_chain_dumps: int = 1,
    ):
        self.max_mc_epochs = max_mc_epochs
        self.gr_threshold = gr_threshold
        self.models_per_epoch = models_per_epoch
        self.min_models = min_models
        self.min_cardinality = min_cardinality

        self.cache_name = None if cache_name is None else (cache_name or id(self))
        self.random_state = check_random_state(random_state)
        self.debug = bool(debug)
        # Default dump directory uses the actual path of this file so no args are required
        if chain_dump_dir is None:
            try:
                import os
                self.chain_dump_dir = os.path.join(os.path.dirname(__file__), "mcmc_chains")
            except Exception:
                # Fallback to None if for any reason we cannot resolve the path
                self.chain_dump_dir = None
        else:
            self.chain_dump_dir = chain_dump_dir
        self.max_chain_dumps = int(max_chain_dumps)
        self._chain_dump_count = 0

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8


        # Used for computing the GR-statistic
        self.marginal_increment_array_stack = np.zeros((0, self.num_points))
        return self

    def compute_marginal_contribution(self, *args, **kwargs):
        """Compute the marginal contributions for semivalue based data evaluators.

        Computes the marginal contribution by sampling.
        Checks MCMC convergence every 100 iterations using Gelman-Rubin Statistic.
        NOTE if the marginal contribution has not been calculated, will look it up in
        a cache of already trained ShapEvaluators, otherwise will train from scratch.

        Parameters
        ----------
        args : tuple[Any], optional
             Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Notes
        -----
        marginal_increment_array_stack : np.ndarray
            Marginal increments when one data point is added.
        """
        # Checks cache if model name has been computed prior
        if self.cache_name is not None and self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        print("Start: marginal contribution computation", flush=True)
        if getattr(self, "debug", False):
            print(
                f"[gr-tmc] config: max_mc_epochs={self.max_mc_epochs}, models_per_epoch={self.models_per_epoch}, min_models={self.min_models}, min_cardinality={self.min_cardinality}",
                flush=True,
            )

        gr_stat = GrTMCSampler.GR_MAX  # Converges when < gr_threshold
        iteration = 0  # Iteration wise terminator, in case MCMC goes on for too long

        while iteration < self.max_mc_epochs and gr_stat > self.gr_threshold:
            # we check the convergence every 100 random samples.
            # we terminate iteration if Shapley value is converged.
            samples_array = [
                self._calculate_marginal_contributions(*args, **kwargs)
                for _ in progress_range(self.models_per_epoch)
            ]
            self.marginal_increment_array_stack = np.vstack(
                [self.marginal_increment_array_stack, *samples_array],
            )

            gr_stat = self._compute_gr_statistic(self.marginal_increment_array_stack)
            iteration += 1  # Update terminating conditions
            if getattr(self, "debug", False):
                total_samples = len(self.marginal_increment_array_stack)
                print(
                    f"[gr-tmc] iter={iteration}, total_samples={total_samples}, r_hat={gr_stat:.6f}",
                    flush=True,
                )
            else:
                print(f"{gr_stat=}")

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count
        print("Done: marginal contribution computation", flush=True)

        if self.cache_name is not None:
            self.CACHE[self.cache_name] = self.marginal_contribution
        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)
        if getattr(self, "debug", False):
            print(
                f"[gr-tmc] calc: num_points={self.num_points}, min_card={self.min_cardinality}",
                flush=True,
            )

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_utility(coalition, *args, **kwargs)
            delta = curr_perf - prev_perf
            marginal_increment[idx] = delta
         
    
            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1


            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # to consider additional zero contributions
                self.marginal_count[
                    subset[(cutoff + 1) :], np.arange(cutoff + 1, len(subset))
                ] += 1
                if getattr(self, "debug", False):
                    print(
                        f"[gr-tmc] truncation break at cutoff={cutoff}, remaining counted as zero",
                        flush=True,
                    )
                break

            # update performance
            prev_perf = curr_perf

        return marginal_increment.reshape(1, -1)

    def _compute_gr_statistic(self, samples: np.ndarray, num_chains: int = 10) -> float:
        """Compute Gelman-Rubin statistic of the marginal contributions.

        References
        ----------
        .. [1] Y. Kwon and J. Zou,
            Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
            Machine Learning,
            arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

        .. [2] D. Vats and C. Knudson,
            Revisiting the Gelman-Rubin Diagnostic,
            arXiv.org, 2018. Available: https://arxiv.org/abs/1812.09384.

        Parameters
        ----------
        samples : np.ndarray
            Marginal incremental stack, used to find values for the num_chains variances
        num_chains : int, optional
            Number of chains to be made from the incremental stack, by default 10

        Returns
        -------
        float
            Gelman-Rubin statistic
        """
        if len(samples) < self.min_models:
            if getattr(self, "debug", False):
                print(
                    f"[gr] burn-in: samples={len(samples)} < min_models={self.min_models}; return={GrTMCSampler.GR_MAX}",
                    flush=True,
                )
            return GrTMCSampler.GR_MAX  # If not burn-in, returns a high GR value

        # Set up
        num_samples, num_datapoints = samples.shape
        num_samples_per_chain, offset = divmod(num_samples, num_chains)
        if getattr(self, "debug", False):
            print(
                f"[gr] samples_shape={samples.shape}, num_chains={num_chains}, per_chain={num_samples_per_chain}, offset={offset}",
                flush=True,
            )
        if num_samples_per_chain == 0:
            if getattr(self, "debug", False):
                print("[gr] not enough samples to form chains; returning GR_MAX", flush=True)
            return GrTMCSampler.GR_MAX
        samples = samples[offset:]  # Remove remainders from initial
        print(f"[gr] Using {len(samples)} samples for GR computation", flush=True)

        # Divides total sample into num_chains parallel chains
        mcmc_chains = samples.reshape(num_chains, num_samples_per_chain, num_datapoints)
        print(f"[gr] mcmc_chains shape: {mcmc_chains.shape}", flush=True)

        # Dump chains to CSVs for debugging/inspection using default path if available
        # Unconditional save: do not gate on debug or additional arguments
        if self.chain_dump_dir:
            try:
                import os, datetime
                os.makedirs(self.chain_dump_dir, exist_ok=True)
                ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                base = os.path.join(self.chain_dump_dir, f"chains_iter{self._chain_dump_count}_{ts}")
                # Save one CSV per chain: rows=samples in chain, cols=datapoints
                for c in range(num_chains):
                    path = f"{base}_c{c}.csv"
                    np.savetxt(path, mcmc_chains[c], delimiter=",")
                self._chain_dump_count += 1
                if getattr(self, "debug", False):
                    print(f"[gr] dumped {num_chains} chain CSVs to {self.chain_dump_dir} (dump_count={self._chain_dump_count})", flush=True)
            except Exception as e:
                if getattr(self, "debug", False):
                    print(f"[gr][warn] failed to dump chains: {e}", flush=True)

        # Computes the average of the intra-chain sample variances
        s_term = np.mean(np.var(mcmc_chains, axis=1, ddof=1), axis=0)

        # Computes the variance of the sample_means of the chain
        sampling_mean = np.mean(mcmc_chains, axis=1, keepdims=False)
        b_term = num_samples_per_chain * np.var(sampling_mean, axis=0, ddof=1)

        gr_stats = np.sqrt(
            (num_samples_per_chain - 1) / num_samples_per_chain
            + (b_term / (s_term * num_samples_per_chain))
        )  # Ref. https://arxiv.org/pdf/1812.09384 (p.7, Eq.4)
        if getattr(self, "debug", False):
            # Basic stats
            def _stats(x):
                return dict(min=float(np.min(x)), max=float(np.max(x)), mean=float(np.mean(x)))

            s_stats = _stats(s_term)
            b_stats = _stats(b_term)
            r_stats = _stats(gr_stats)
            worst_idx = int(np.argmax(gr_stats))
            # Percentiles for R-hat distribution
            rhat_p = np.percentile(gr_stats, [50, 90, 95, 99]).tolist()
            print(
                "[gr] W(within) stats:", s_stats,
                "\n[gr] B(between) stats:", b_stats,
                "\n[gr] R-hat stats:", r_stats,
                f"\n[gr] R-hat percentiles (50/90/95/99): {rhat_p}",
                f"\n[gr] worst_idx={worst_idx}, R-hat_worst={float(gr_stats[worst_idx]):.6f}, W_worst={float(s_term[worst_idx]):.6g}, B_worst={float(b_term[worst_idx]):.6g}",
                sep=" ",
                flush=True,
            )
        return np.max(gr_stats)
    
class GroupAwareTMCSampler(GrTMCSampler):
    """Extended GR-TMC Sampler that tracks co-contributions between points to support group-aware valuation."""

    def set_coalition(self, coalition: torch.Tensor):
        """Initialize coalition setup and co-contribution matrix."""
        super().set_coalition(coalition)
        self.co_contrib_matrix = np.zeros((self.num_points, self.num_points))  # symmetric
        self.contribution_trace = []  # to store marginal contributions
        return self

    def _calculate_marginal_contributions(self, *args, **kwargs) -> np.ndarray:
        """Override marginal contribution calculation to include co-contribution tracking."""
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(subset[self.min_cardinality:], start=self.min_cardinality):
            coalition.append(idx)
            curr_perf = self.compute_utility(coalition, *args, **kwargs)
            delta = curr_perf - prev_perf
            marginal_increment[idx] = delta

            self.marginal_contrib_sum[idx, cutoff] += delta
            self.marginal_count[idx, cutoff] += 1

            self.contribution_trace.append((int(idx), int(cutoff), prev_perf, curr_perf))

            # Track co-contribution between idx and each point in the current coalition (before adding idx)
            for j in coalition[:-1]:
                self.co_contrib_matrix[idx, j] += delta
                self.co_contrib_matrix[j, idx] += delta  # symmetry

            # Truncation logic
            if abs(delta) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:
                self.marginal_count[
                    subset[(cutoff + 1):], np.arange(cutoff + 1, len(subset))
                ] += 1
                break

            prev_perf = curr_perf

        return marginal_increment.reshape(1, -1)
