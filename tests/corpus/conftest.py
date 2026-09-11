"""Test setup for the corpus build scripts."""

from __future__ import annotations

import sys
import types

if "datasets" not in sys.modules:
    stub = types.ModuleType("datasets")
    stub.load_dataset = lambda *_args, **_kwargs: []
    sys.modules["datasets"] = stub
