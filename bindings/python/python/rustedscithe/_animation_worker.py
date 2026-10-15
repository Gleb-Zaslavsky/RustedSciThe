"""Private child-process entry point for one interactive Bevy animation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from . import _native

_ERROR_PREFIX = "RUSTEDSCITHE_ANIMATION_ERROR:"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dimension", required=True, type=int, choices=(2, 3))
    parser.add_argument("--axis-names", required=True)
    parser.add_argument("--speed", required=True, type=float)
    return parser.parse_args()


def main() -> int:
    """Load one detached trajectory and run Bevy on this process's main thread."""

    arguments = _parse_arguments()
    try:
        axis_names: Any = json.loads(arguments.axis_names)
        if axis_names is not None:
            axis_names = tuple(axis_names)
        with np.load(arguments.input, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            times = np.asarray(data["times"], dtype=np.float64)

        if arguments.dimension == 2:
            _native.animate_trajectory_2d(points, times, axis_names, arguments.speed)
        else:
            _native.animate_trajectory_3d(points, times, axis_names, arguments.speed)
    except BaseException as error:
        # PyO3's panic exception derives from BaseException, so translate it
        # too. KeyboardInterrupt and SystemExit should retain normal semantics.
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        payload = {"type": type(error).__name__, "message": str(error)}
        print(_ERROR_PREFIX + json.dumps(payload), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
