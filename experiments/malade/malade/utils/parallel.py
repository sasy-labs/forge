"""
Bounded-concurrency parallel map for I/O- or CPU-bound work.

Adapted from regurates/utils/parallel.py — dropped the Langroid
redis-pool cleanup (redis pools are not a SASY concern; consumers
that need it can add an after_each hook), kept dual thread/process
executors, ordered results, and atexit/signal-driven shutdown of
process pools so KeyboardInterrupt doesn't orphan workers.

Use threads for LLM/network-bound work (lighter, shared memory,
no pickling) and processes for CPU-bound work that needs to escape
the GIL.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import traceback
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from typing import Any, Callable, Iterable, Literal, Optional


# Track active process executors so we can shut them down on
# unexpected exit. Threads die with the process so they don't
# need this.
_active_executors: list[ProcessPoolExecutor] = []
_active_executors_lock = threading.Lock()


def _cleanup_executors() -> None:
    with _active_executors_lock:
        executors = list(_active_executors)
        _active_executors.clear()
    for executor in executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(_cleanup_executors)


def _signal_handler(signum, frame):  # noqa: ARG001
    """Clean up process pools on SIGTERM/SIGINT, then re-raise."""
    _cleanup_executors()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Only install signal handler in the main thread of the main
# process — workers and non-main threads would raise.
if threading.current_thread() is threading.main_thread():
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _signal_handler)
        except (OSError, ValueError):
            pass


def _execute_func(
    pickled_func: bytes,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Worker entry point for process-mode execution.

    Module-level so multiprocessing can pickle it. Cloudpickle
    handles the actual function so closures, lambdas, and dynamic
    classes all work.
    """
    import cloudpickle

    func = cloudpickle.loads(pickled_func)
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Print before re-raising; the parent only sees a pickled
        # exception, which loses the worker-side traceback otherwise.
        traceback.print_exception(e)
        raise


def map_concurrent(
    func: Callable[..., Any],
    work_items: Iterable[
        tuple[tuple[Any, ...], dict[str, Any]] | dict[str, Any] | Any
    ],
    max_in_flight: Optional[int] = None,
    progress: bool = False,
    desc: Optional[str] = None,
    mode: Literal["processes", "threads"] = "threads",
) -> list[Any]:
    """Execute ``func`` over ``work_items`` with bounded concurrency.

    Each work item may be:
      - ``(args, kwargs)`` tuple
      - dict (passed as kwargs)
      - any other value (passed as a single positional arg)

    Results are returned in the original order regardless of which
    worker finishes first.

    Args:
        func: Callable to apply. In process mode it must be
            cloudpickle-serialisable.
        work_items: Iterable of items as described above.
        max_in_flight: Concurrency cap. Defaults to ``os.cpu_count()
            or 4``.
        progress: If True, show a tqdm progress bar.
        desc: Optional progress-bar description (defaults to
            ``func.__name__``).
        mode: ``"threads"`` (default) for I/O-bound work, ``"processes"``
            for CPU-bound work that needs to bypass the GIL.

    Returns:
        List of results in the order of ``work_items``.
    """
    if max_in_flight is None:
        max_in_flight = os.cpu_count() or 4

    work_list = list(work_items)
    queued: list[tuple[int, tuple[Any, ...], dict[str, Any]]] = []

    for i, item in enumerate(work_list):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
            # ``(args_tuple, kwargs_dict)`` shape — short-circuit so
            # a 2-tuple wrapper doesn't get auto-unpacked.
            args, kwargs = item
            if not isinstance(args, tuple):
                args = tuple(args) if isinstance(args, (list, tuple)) else (args,)
        elif isinstance(item, tuple):
            # N-tuple (N != 2 or second element not a dict): treat
            # the whole tuple as positional args. This matches the
            # regurates original behaviour for 1-tuples but is more
            # useful for 3+ args than the previous fallthrough,
            # which wrapped the tuple as a single argument.
            args, kwargs = item, {}
        elif isinstance(item, dict):
            args, kwargs = (), item
        else:
            args, kwargs = (item,), {}
        queued.append((i, args, kwargs))

    results: dict[int, Any] = {}
    total = len(queued)

    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc=desc or func.__name__, unit="item")
        except ImportError:
            pass

    use_threads = mode == "threads"

    if use_threads:
        executor_class = ThreadPoolExecutor

        def submit(executor, args, kwargs):
            return executor.submit(func, *args, **kwargs)
    else:
        executor_class = ProcessPoolExecutor

        # Try plain pickle first (works for module-level functions
        # and matches ProcessPoolExecutor.submit's own protocol).
        # Fall back to cloudpickle for closures / lambdas. Plain
        # pickle's qualified-name approach also dodges spurious
        # cloudpickle failures when imported instrumentation puts
        # un-picklable globals (e.g. ContextVars) in the function's
        # module-level scope.
        import pickle as _pickle
        try:
            _pickle.dumps(func)
            pickled_func = None
        except Exception:
            try:
                import cloudpickle
                pickled_func = cloudpickle.dumps(func)
            except Exception as e:
                raise ValueError(
                    f"Function {func!r} cannot be pickled: {e}"
                ) from e

        if pickled_func is None:
            def submit(executor, args, kwargs):
                return executor.submit(func, *args, **kwargs)
        else:
            def submit(executor, args, kwargs):
                return executor.submit(
                    _execute_func, pickled_func, args, kwargs,
                )

    executor = None
    try:
        with executor_class(max_workers=max_in_flight) as executor:
            if isinstance(executor, ProcessPoolExecutor):
                with _active_executors_lock:
                    _active_executors.append(executor)
            in_flight: dict[Future[Any], int] = {}

            while queued:
                while len(in_flight) < max_in_flight and queued:
                    idx, args, kwargs = queued.pop(0)
                    future = submit(executor, args, kwargs)
                    in_flight[future] = idx

                done, pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    idx = in_flight.pop(future)
                    results[idx] = future.result()
                    if pbar is not None:
                        pbar.update(1)

                in_flight = {f: idx for f, idx in in_flight.items() if f in pending}

            if in_flight:
                done, _ = wait(in_flight.keys(), return_when=ALL_COMPLETED)
                for future in done:
                    idx = in_flight[future]
                    results[idx] = future.result()
                    if pbar is not None:
                        pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        if executor is not None and isinstance(executor, ProcessPoolExecutor):
            with _active_executors_lock:
                _active_executors[:] = [e for e in _active_executors if e is not executor]

    return [results.get(i) for i in range(total)]
