"""Test setup for the H1 pipeline scripts."""

from __future__ import annotations

import sys
import types

if "datasets" not in sys.modules:
    stub = types.ModuleType("datasets")
    stub.load_dataset = lambda *args, **kwargs: []
    sys.modules["datasets"] = stub
