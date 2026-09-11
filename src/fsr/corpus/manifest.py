"""A record of what a corpus build produced."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DIGEST_CHUNK_BYTES = 1 << 20
UNKNOWN_VERSION = "unknown"


def package_version() -> str:
    """Return the installed package version, or a placeholder."""
    try:
        return version("format-sensitivity-reranking")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def digest_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in chunks.

    Args:
        path: The file to digest.

    Returns:
        The digest as lowercase hexadecimal.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(DIGEST_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Output:
    """One file a stage wrote.

    Attributes:
        path: The path, relative to the data root.
        sha256: The digest of the file contents.
        bytes: The size of the file.
        records: The number of records the file holds, when it holds records.
    """

    path: str
    sha256: str
    bytes: int
    records: int | None = None

    @classmethod
    def describe(cls, path: Path, root: Path, records: int | None = None) -> Output:
        """Describe a written file.

        Args:
            path: The file to describe.
            root: The directory that `path` is reported relative to.
            records: The number of records the file holds.

        Returns:
            The description.
        """
        return cls(
            path=str(path.relative_to(root)),
            sha256=digest_file(path),
            bytes=path.stat().st_size,
            records=records,
        )


@dataclass
class Stage:
    """One completed build stage and the files it wrote."""

    name: str
    outputs: list[Output] = field(default_factory=list)


@dataclass
class Manifest:
    """The configuration and outputs of one corpus build."""

    config: dict[str, Any]
    stages: list[Stage] = field(default_factory=list)
    schema: int = SCHEMA_VERSION
    fsr_version: str = field(default_factory=package_version)

    def record(self, name: str, outputs: list[Output]) -> None:
        """Add a stage, replacing any earlier entry of the same name.

        A replaced stage keeps its original position. The stage order is the
        build order.

        Args:
            name: The stage name.
            outputs: The files the stage wrote.
        """
        stage = Stage(name=name, outputs=outputs)
        for i, existing in enumerate(self.stages):
            if existing.name == name:
                self.stages[i] = stage
                return
        self.stages.append(stage)

    def stage(self, name: str) -> Stage | None:
        """Return the named stage, or None when it has not run."""
        return next((s for s in self.stages if s.name == name), None)

    def has(self, name: str) -> bool:
        """Report whether the named stage has run."""
        return self.stage(name) is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-ready mapping."""
        return {
            "schema": self.schema,
            "fsr_version": self.fsr_version,
            "config": self.config,
            "stages": [
                {
                    "name": s.name,
                    "outputs": [
                        {
                            "path": o.path,
                            "sha256": o.sha256,
                            "bytes": o.bytes,
                            "records": o.records,
                        }
                        for o in s.outputs
                    ],
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Rebuild a manifest from a mapping.

        Args:
            data: A mapping produced by `to_dict`.

        Returns:
            The manifest.

        Raises:
            ValueError: If the schema version is not supported.
        """
        schema = data.get("schema")
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema {schema!r}, expected {SCHEMA_VERSION}"
            )
        return cls(
            config=data["config"],
            stages=[
                Stage(name=s["name"], outputs=[Output(**o) for o in s["outputs"]])
                for s in data["stages"]
            ],
            schema=schema,
            fsr_version=data.get("fsr_version", UNKNOWN_VERSION),
        )

    def save(self, path: Path) -> None:
        """Write the manifest as indented JSON, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read a manifest from a JSON file."""
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def load_or_new(cls, path: Path, config: dict[str, Any]) -> Manifest:
        """Read a manifest, or start one when the file is absent.

        Args:
            path: The manifest file.
            config: The configuration to start a new manifest with.

        Returns:
            The manifest.
        """
        return cls.load(path) if path.exists() else cls(config=config)
