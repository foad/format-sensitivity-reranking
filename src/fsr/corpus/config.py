"""The parameters that define the corpus.

Every value that changes what the built corpus contains lives here. One object
describes the artefact and is recorded alongside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

NQ_DATASET = "google-research-datasets/natural_questions"
NQ_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"


@dataclass(frozen=True)
class CorpusConfig:
    """The full parameter set of one corpus build.

    Attributes:
        dataset: The Hugging Face dataset to stream.
        dataset_revision: The dataset revision to pin, as a commit SHA, tag, or
            branch. The default is the revision the published corpus was built
            from. None resolves to the default branch at build time.
        body_chars: The length at which body collection stops.
        min_paragraph_chars: The shortest paragraph kept during extraction.
        max_key_chars: The longest metadata key kept.
        max_value_chars: The longest metadata value kept.
        min_pairs: The fewest pairs a record may hold to pass the gate.
        min_body_chars: The shortest body a record may hold to pass the gate.
        min_infobox_chars: The shortest raw infobox HTML to pass the gate.
        split_seed: The seed for the article-level split.
        frac_train: The share of articles in the train split.
        frac_dev: The share of articles in the dev split.
        negatives_seed: The seed for negative sampling.
        cache_k: The number of hard negatives cached per query.
    """

    dataset: str = NQ_DATASET
    dataset_revision: str | None = NQ_REVISION

    body_chars: int = 3000
    min_paragraph_chars: int = 30
    max_key_chars: int = 80
    max_value_chars: int = 400

    min_pairs: int = 3
    min_body_chars: int = 100
    min_infobox_chars: int = 200

    split_seed: int = 42
    frac_train: float = 0.80
    frac_dev: float = 0.10

    negatives_seed: int = 42
    cache_k: int = 25

    def __post_init__(self) -> None:
        """Reject a configuration that cannot produce three splits.

        Raises:
            ValueError: If either fraction is outside the open unit interval,
                or if the two together leave nothing for the test split.
        """
        for name in ("frac_train", "frac_dev"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.frac_train + self.frac_dev >= 1.0:
            raise ValueError(
                "frac_train + frac_dev must leave a test split, got "
                f"{self.frac_train} + {self.frac_dev}"
            )

    @property
    def frac_test(self) -> float:
        """The share of articles left for the test split."""
        return 1.0 - self.frac_train - self.frac_dev

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a mapping, including the test share."""
        return {**asdict(self), "frac_test": self.frac_test}


DEFAULT = CorpusConfig()
