"""pytest bootstrap for the kbound_edge test suite.

Adds ``edge/src`` to sys.path so ``import kbound_edge`` resolves when running
``pytest`` from the ``edge`` directory (no install needed).  The ``kbound``
package itself is located by :mod:`kbound_edge._bridge` (it is already installed
in ``~/.venv_wilds`` on the host, or found via the sibling ``kbound_pkg``).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
