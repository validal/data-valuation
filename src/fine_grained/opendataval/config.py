"""Global configuration for OpenDataVal.

Use this to customize behavior across all evaluators and experiments.
"""

# ==============================================================================
# PROGRESS BAR CONFIGURATION
# ==============================================================================

# Choose progress bar style:
# - "fancy": tqdm.notebook with Jupyter widgets (default, requires Jupyter)
# - "simple": tqdm text-based progress bars
# - "none": No progress bars at all
PROGRESS_BAR_STYLE = "fancy"

# Whether to use clear_output for main experiment progress bar
# Set to False if progress bars are being cleared unexpectedly
CLEAR_OUTPUT_ON_PROGRESS = False

# ==============================================================================
# DEBUG CONFIGURATION
# ==============================================================================

# Print debug information for evaluators
DEBUG = False

# Print detailed timing information
VERBOSE_TIMINGS = False

# ==============================================================================
# PERFORMANCE CONFIGURATION
# ==============================================================================

# Default number of workers for parallel evaluators
NUM_WORKERS = 1

# Default batch size for evaluators
DEFAULT_BATCH_SIZE = 32

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_progress_bar_class():
    """Get the progress bar class based on configuration.

    Returns
    -------
    type
        Either tqdm notebook, tqdm text, or NoProgressBar class
    """
    if PROGRESS_BAR_STYLE == "fancy":
        # Use tqdm.notebook for fancy Jupyter widget in notebook
        try:
            from tqdm.notebook import tqdm
            return tqdm
        except ImportError:
            # Fallback to text tqdm if notebook version not available
            from tqdm import tqdm
            return tqdm
    elif PROGRESS_BAR_STYLE == "simple":
        # Use tqdm text-based for simple style
        from tqdm import tqdm
        return tqdm
    elif PROGRESS_BAR_STYLE == "none":
        return NoProgressBar
    else:
        # Default to fancy
        try:
            from tqdm.notebook import tqdm
            return tqdm
        except ImportError:
            from tqdm import tqdm
            return tqdm


class NoProgressBar:
    """Dummy progress bar that does nothing (for PROGRESS_BAR_STYLE='none')."""

    def __init__(self, iterable=None, desc="Progress", total=None, clear_output=True, **kwargs):
        self.iterable = list(iterable) if iterable is not None else []

    def __iter__(self):
        for item in self.iterable:
            yield item

    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
