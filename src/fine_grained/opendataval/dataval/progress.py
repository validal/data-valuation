"""Unified progress bar for all data evaluators.

Uses in-place updating for active bars while preserving completed bar logs.
Works across all methods without external dependencies beyond IPython.
"""

from IPython.display import clear_output
import time
import sys


class ProgressBar:
    """One-line progress bar that updates in-place.

    Works as context manager, iterable, or standalone:

    Examples
    --------
    # As context manager (recommended)
    with ProgressBar(total=100, desc="Training") as pbar:
        for i in range(100):
            # do work
            pbar.update()

    # As iterable (for list iteration)
    pbar = ProgressBar(iterable=[1, 2, 3], desc="Processing")
    for item in pbar:
        # do work with item

    # Standalone
    pbar = ProgressBar(total=100, desc="Training")
    for i in range(100):
        # do work
        pbar.update()
    pbar.close()
    """

    def __init__(self, total=None, desc="Progress", bar_length=40, iterable=None, clear_output=False):
        """Initialize progress bar.

        Parameters
        ----------
        total : int, optional
            Total iterations (not needed if iterable is provided)
        desc : str, optional
            Description label, by default "Progress"
        bar_length : int, optional
            Length of progress bar, by default 40
        iterable : list, optional
            Iterable to wrap with progress bar
        clear_output : bool, optional
            Whether to use clear_output (set False for better results), by default False
        """
        if iterable is not None:
            self.iterable = list(iterable)
            self.total = len(self.iterable)
            self._iter_index = 0
        else:
            self.iterable = None
            self.total = total if total is not None else 100

        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.clear_output_enabled = clear_output
        self._last_line = ""  # Track last printed line for overwriting
        self._finished = False

    def update(self, n=1):
        """Update progress by n steps.

        Parameters
        ----------
        n : int, optional
            Steps to advance, by default 1
        """
        self.current = min(self.current + n, self.total)
        if self.current <= self.total:
            self._render()

    def _render(self, final=False):
        """Render progress bar (updates in-place).

        Parameters
        ----------
        final : bool, optional
            If True, prints final state and moves to new line, by default False
        """
        percent = (self.current / self.total) * 100
        filled = int(self.bar_length * self.current // self.total)
        bar = '█' * filled + '░' * (self.bar_length - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            time_str = f"{elapsed:.0f}s<{remaining:.0f}s"
        else:
            time_str = "00s<??s"

        # Build the progress line
        line = f"{self.desc}: {percent:5.1f}% |{bar}| {self.current}/{self.total} [{time_str}]"

        if final:
            # Final state - print with newline so it stays
            if self._last_line:
                # Overwrite the last line with final state
                sys.stdout.write('\r' + ' ' * len(self._last_line))
                sys.stdout.write('\r')
                sys.stdout.flush()
            print(line)
            sys.stdout.flush()
            self._last_line = ""
            self._finished = True
        else:
            # Update in place using carriage return
            sys.stdout.write('\r' + line)
            sys.stdout.flush()
            self._last_line = line

    def close(self):
        """Close and finalize progress bar (keeps as log)."""
        if not self._finished:
            self.current = self.total
            self._render(final=True)
        else:
            # If already finished, just print summary if not done
            pass

        # Print completion summary if not already printed
        if not hasattr(self, '_summary_printed'):
            elapsed = time.time() - self.start_time
            print(f"✓ {self.desc} completed in {elapsed:.1f}s")
            sys.stdout.flush()
            self._summary_printed = True

    def __iter__(self):
        """Iterate over items (if iterable provided) or range (if total provided)."""
        if self.iterable is not None:
            self._iter_index = 0
            return self
        else:
            self._range_iter = iter(range(self.total))
            return self

    def __next__(self):
        """Get next item and update progress."""
        if self.iterable is not None:
            if self._iter_index >= len(self.iterable):
                self.close()
                raise StopIteration
            item = self.iterable[self._iter_index]
            self._iter_index += 1
            self.update()
            return item
        else:
            try:
                next(self._range_iter)
                self.update()
                return self.current - 1
            except StopIteration:
                self.close()
                raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


class SimpleProgressBar:
    """Simple text-only progress bar (prints each update as plain text).

    Use this if you prefer plain text output without fancy in-place updates.
    Each update prints immediately.
    """

    def __init__(self, iterable=None, desc="Progress", total=None, clear_output=True):
        """Initialize simple progress bar.

        Parameters
        ----------
        iterable : list, optional
            Iterable to wrap with progress bar
        desc : str, optional
            Description label, by default "Progress"
        total : int, optional
            Total iterations (auto-detected from iterable if provided)
        clear_output : bool, optional
            Ignored for compatibility with ProgressBar, by default True
        """
        self.iterable = list(iterable) if iterable is not None else None
        self.total = len(self.iterable) if self.iterable is not None else total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        """Update progress by n steps - prints immediately."""
        self.current += n
        if self.total > 0:
            print(f"{self.desc}: {self.current}/{self.total}")

    def close(self):
        """Close and print final summary."""
        elapsed = time.time() - self.start_time
        print(f"✓ {self.desc} completed in {elapsed:.1f}s")

    def __iter__(self):
        """Iterate over items."""
        if self.iterable is not None:
            for i, item in enumerate(self.iterable):
                self.current = i + 1
                print(f"{self.desc}: {self.current}/{self.total}")
                yield item
            self.close()
        else:
            raise ValueError("SimpleProgressBar requires an iterable")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Convenience function for quick usage
def progress_range(total, desc="Progress"):
    """Simple progress iterator.

    Parameters
    ----------
    total : int
        Number of iterations
    desc : str, optional
        Description, by default "Progress"

    Yields
    ------
    int
        Iteration number (0-indexed)

    Examples
    --------
    for i in progress_range(100, "Training"):
        # do work
    """
    pbar = ProgressBar(total=total, desc=desc)
    for i in range(total):
        yield i
        pbar.update()
    pbar.close()