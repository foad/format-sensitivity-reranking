"""Synthetic first-stage pools for the pipeline demonstration.

Each pool holds one question about a fictional entity and ten candidates, in
the shape of the candidate lists `fsr.standardisation.scoring` scores:

- the gold passage, first, the only candidate that carries the answer, and
  only in its metadata block;
- one wrong-answer passage about a confusable sibling entity, which states a
  plausible wrong value for the asked attribute;
- three near-misses about the same entity that leave the asked attribute out;
- five fillers about other entities of the same kind.

A pool that loses its gold passage therefore cannot answer the question, by
construction. The entities are invented, so a generator cannot answer from
what it already knows either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.scoring import MAX_TOKENS
from fsr.standardisation.conditions import mixed_assignment

GOLD_ROLE = "gold"
WRONG_ROLE = "wrong"
NEAR_ROLE = "near_miss"
FILLER_ROLE = "filler"
N_NEAR = 3
N_FILLER = 5
ROLES = (GOLD_ROLE, WRONG_ROLE) + (NEAR_ROLE,) * N_NEAR + (FILLER_ROLE,) * N_FILLER
N_CANDIDATES = len(ROLES)

NAME_KEY = "name"
MAX_ATTEMPTS = 50

Draw = Callable[[np.random.Generator], str]

SYLLABLES = (
    "var", "holt", "kes", "tren", "mor", "ath", "lin", "quo", "bel", "dra",
    "sor", "vik", "ost", "rel", "pan", "dur", "cal", "wen", "ish", "tor",
    "mal", "eph", "zel", "sul",
)  # fmt: skip
FIRST_NAMES = (
    "Ada", "Tomas", "Ilse", "Marek", "Noor", "Petra", "Oskar", "Lene",
    "Ravi", "Sanne", "Emil", "Yara", "Kaito", "Mira", "Anton", "Liv",
)  # fmt: skip
ADJECTIVES = ("Glass", "Silent", "Northern", "Hollow", "Amber", "Iron", "Pale")
NOUNS = ("Tide", "Lantern", "Orchard", "Harbour", "Choir", "Meridian", "Gate")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)  # fmt: skip


def pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
    """Return one of the options, drawn uniformly."""
    return options[int(rng.integers(len(options)))]


def stem(rng: np.random.Generator) -> str:
    """Return an invented two-syllable proper name."""
    return (pick(rng, SYLLABLES) + pick(rng, SYLLABLES)).capitalize()


def person(rng: np.random.Generator) -> str:
    """Return an invented personal name, some with punctuation in the surname."""
    surname = pick(
        rng,
        (
            lambda: stem(rng),
            lambda: "O'" + stem(rng),
            lambda: f"{stem(rng)}-{stem(rng)}",
            lambda: "de " + stem(rng),
        ),
    )()
    return f"{pick(rng, FIRST_NAMES)} {surname}"


def place(rng: np.random.Generator) -> str:
    """Return an invented place name, some with a comma or a hyphen."""
    return pick(
        rng,
        (
            lambda: f"Port {stem(rng)}",
            lambda: f"{stem(rng)}ford",
            lambda: f"{stem(rng)} Bay",
            lambda: f"{stem(rng)}, {stem(rng)} Province",
            lambda: f"{stem(rng)}-on-{stem(rng)}",
        ),
    )()


def year(low: int, high: int) -> Draw:
    """Return a draw of a year between the bounds, inclusive."""
    return lambda rng: str(int(rng.integers(low, high + 1)))


def count(low: int, high: int, unit: str = "") -> Draw:
    """Return a draw of a comma-grouped number, with an optional unit."""
    suffix = f" {unit}" if unit else ""
    return lambda rng: f"{int(rng.integers(low, high + 1)):,}{suffix}"


def one_of(*options: str) -> Draw:
    """Return a draw of one of the fixed options."""
    return lambda rng: pick(rng, options)


def work(rng: np.random.Generator) -> str:
    """Return an invented title, some quoted and some with a colon."""
    adjective, noun = pick(rng, ADJECTIVES), pick(rng, NOUNS)
    return pick(
        rng,
        (
            f'"The {adjective} {noun}"',
            f"Opus {int(rng.integers(2, 60))}: {adjective} {noun}",
            f'Suite No. {int(rng.integers(1, 9))}, "{noun}"',
        ),
    )


def date(rng: np.random.Generator) -> str:
    """Return a calendar date between 1990 and 2024."""
    day, month = int(rng.integers(1, 29)), pick(rng, MONTHS)
    return f"{day} {month} {int(rng.integers(1990, 2025))}"


def agency(rng: np.random.Generator) -> str:
    """Return an invented space agency or operator."""
    return pick(
        rng,
        (
            lambda: f"the {stem(rng)} Space Agency",
            lambda: f"{stem(rng)} Orbital Systems",
            lambda: f"the {stem(rng)} Institute of Astronautics",
        ),
    )()


def launch_site(rng: np.random.Generator) -> str:
    """Return an invented launch site, some with a comma."""
    return pick(
        rng,
        (
            lambda: f"{stem(rng)} Launch Complex",
            lambda: f"{stem(rng)} Spaceport",
            lambda: f"Pad {int(rng.integers(1, 40))}, {stem(rng)} Range",
        ),
    )()


def countries(rng: np.random.Generator) -> str:
    """Return a comma-separated list of two or three invented countries."""
    n = int(rng.integers(2, 4))
    return ", ".join(
        pick(rng, (f"{stem(rng)}ia", f"{stem(rng)}land")) for _ in range(n)
    )


def source(rng: np.random.Generator) -> str:
    """Return an invented river source."""
    return pick(
        rng, (f"Mount {stem(rng)}", f"Lake {stem(rng)}", f"the {stem(rng)} Hills")
    )


def mouth(rng: np.random.Generator) -> str:
    """Return an invented body of water a river flows into."""
    return pick(
        rng, (f"the {stem(rng)} Sea", f"{stem(rng)} Bay", f"the Gulf of {stem(rng)}")
    )


def tributaries(rng: np.random.Generator) -> str:
    """Return two invented tributary names."""
    return f"the {stem(rng)} and the {stem(rng)}"


@dataclass(frozen=True)
class EntityKind:
    """One kind of fictional entity, with its infobox and its prose.

    Attributes:
        kind: The name of the kind.
        draw_name: Draws a fresh entity name.
        sibling: Derives a confusable name from an entity name.
        attributes: The infobox keys, each with its value draw.
        questions: The question template of each askable attribute.
        intro: The opening sentence of every passage.
        sentences: One sentence template per attribute it states.
        generic: Sentences that state no attribute.
    """

    kind: str
    draw_name: Callable[[np.random.Generator], str]
    sibling: Callable[[str, np.random.Generator], str]
    attributes: dict[str, Draw]
    questions: dict[str, str]
    intro: str
    sentences: dict[str, str]
    generic: tuple[str, ...]


COMPANY_SUFFIXES = (
    "Instruments",
    "Holdings",
    "Freight",
    "Analytics",
    "Ceramics",
    "Optics",
)


def _company_name(rng: np.random.Generator) -> str:
    return f"{stem(rng)} {pick(rng, COMPANY_SUFFIXES)}"


def _company_sibling(name: str, rng: np.random.Generator) -> str:
    head, suffix = name.rsplit(" ", 1)
    others = [s for s in COMPANY_SUFFIXES if s != suffix]
    return f"{head} {pick(rng, others)}"


def _satellite_name(rng: np.random.Generator) -> str:
    return f"{stem(rng)}-{int(rng.integers(1, 9))}"


def _satellite_sibling(name: str, _rng: np.random.Generator) -> str:
    head, number = name.rsplit("-", 1)
    return f"{head}-{int(number) + 1}"


def _composer_sibling(name: str, rng: np.random.Generator) -> str:
    first, surname = name.split(" ", 1)
    return f"{pick(rng, [f for f in FIRST_NAMES if f != first])} {surname}"


KINDS: tuple[EntityKind, ...] = (
    EntityKind(
        kind="company",
        draw_name=_company_name,
        sibling=_company_sibling,
        attributes={
            "industry": one_of(
                "precision optics",
                "freight logistics",
                "industrial ceramics",
                "agricultural robotics",
                "marine sensors",
                "specialty glass",
            ),
            "founded": year(1880, 2015),
            "founder": person,
            "headquarters": place,
            "chief_executive": person,
            "employees": count(40, 90000),
        },
        questions={
            "founder": "Who founded {name}?",
            "headquarters": "Where is {name} headquartered?",
            "chief_executive": "Who is the chief executive of {name}?",
        },
        intro="{name} is a company in the {industry} sector.",
        sentences={
            "founded": "It was founded in {founded}.",
            "founder": "The company was founded by {founder}.",
            "headquarters": "{name} is headquartered in {headquarters}.",
            "chief_executive": "Its chief executive is {chief_executive}.",
            "employees": "The company employs about {employees} people.",
        },
        generic=(
            "The firm supplies customers across several regions.",
            "It has grown steadily through a series of acquisitions.",
            "Most of its products are sold through regional distributors.",
            "The company publishes an annual report each spring.",
        ),
    ),
    EntityKind(
        kind="river",
        draw_name=lambda rng: f"{stem(rng)} River",
        sibling=lambda name, _rng: f"Little {name}",
        attributes={
            "basin_countries": countries,
            "source": source,
            "mouth": mouth,
            "length": count(40, 2400, "km"),
            "discharge": count(20, 9000, "m3/s"),
            "tributaries": tributaries,
        },
        questions={
            "source": "Where does the {name} rise?",
            "mouth": "Into which body of water does the {name} flow?",
            "length": "How long is the {name}?",
        },
        intro="The {name} is a river that flows through {basin_countries}.",
        sentences={
            "source": "It rises at {source}.",
            "mouth": "It flows into {mouth}.",
            "length": "The river is {length} long.",
            "discharge": "Its mean discharge is about {discharge}.",
            "tributaries": "Its main tributaries are {tributaries}.",
        },
        generic=(
            "The river is navigable for part of its course.",
            "Several towns along its banks depend on it for irrigation.",
            "Its valley is known for seasonal flooding.",
            "A number of dams regulate its flow.",
        ),
    ),
    EntityKind(
        kind="satellite",
        draw_name=_satellite_name,
        sibling=_satellite_sibling,
        attributes={
            "mission": one_of(
                "weather observation",
                "ocean altimetry",
                "communications relay",
                "atmospheric sounding",
                "navigation",
            ),
            "operator": agency,
            "launch_date": date,
            "launch_site": launch_site,
            "orbit": one_of(
                "low Earth orbit", "sun-synchronous orbit", "geostationary orbit"
            ),
            "mass": count(80, 6500, "kg"),
        },
        questions={
            "operator": "Who operates the {name} satellite?",
            "launch_site": "From where was the {name} satellite launched?",
            "launch_date": "When was the {name} satellite launched?",
        },
        intro="{name} is a satellite used for {mission}.",
        sentences={
            "operator": "It is operated by {operator}.",
            "launch_date": "It was launched on {launch_date}.",
            "launch_site": "It was launched from {launch_site}.",
            "orbit": "It travels in {orbit}.",
            "mass": "It has a mass of about {mass}.",
        },
        generic=(
            "Its data are distributed to research groups at no cost.",
            "The spacecraft carries several redundant instruments.",
            "Its mission has been extended more than once.",
            "Ground stations receive its telemetry several times a day.",
        ),
    ),
    EntityKind(
        kind="composer",
        draw_name=person,
        sibling=_composer_sibling,
        attributes={
            "era": one_of(
                "late Romantic", "early modern", "Baroque revival", "post-minimalist"
            ),
            "born": year(1700, 1980),
            "birthplace": place,
            "teacher": person,
            "notable_work": work,
            "instrument": one_of("piano", "cello", "organ", "violin", "clarinet"),
        },
        questions={
            "teacher": "Who taught the composer {name}?",
            "birthplace": "Where was the composer {name} born?",
            "notable_work": "What is the best-known work of the composer {name}?",
        },
        intro="{name} is a composer associated with the {era} style.",
        sentences={
            "born": "The composer was born in {born}.",
            "birthplace": "{name} was born in {birthplace}.",
            "teacher": "{name} studied under {teacher}.",
            "notable_work": "The best-known work is {notable_work}.",
            "instrument": "{name} wrote mainly for the {instrument}.",
        },
        generic=(
            "Many of the works were first performed in small chamber settings.",
            "Several manuscripts survive in private collections.",
            "The music is noted for its unusual harmonic language.",
            "Recordings of the major works appeared decades later.",
        ),
    ),
)


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


KEY_STYLES: dict[str, Callable[[str], str]] = {
    "snake": lambda key: key,
    "title": lambda key: key.replace("_", " ").capitalize(),
    "camel": _camel,
}


@dataclass(frozen=True)
class Entity:
    """One drawn entity: its name and a value for every attribute."""

    name: str
    values: dict[str, str]


def draw_entity(kind: EntityKind, rng: np.random.Generator, name: str = "") -> Entity:
    """Draw a value for every attribute of a kind.

    Args:
        kind: The kind to draw.
        rng: The source of the draw.
        name: The entity name. A fresh one is drawn when empty.

    Returns:
        The entity.
    """
    return Entity(
        name=name or kind.draw_name(rng),
        values={key: draw(rng) for key, draw in kind.attributes.items()},
    )


def metadata(
    entity: Entity,
    keys: Sequence[str],
    style: Callable[[str], str],
    rng: np.random.Generator,
) -> list[list[str]]:
    """Return the infobox of an entity: its name, then the keys in a drawn order.

    Args:
        entity: The entity described.
        keys: The attributes to include.
        style: The key naming of the pool.
        rng: The source of the order.

    Returns:
        The key and value pairs.
    """
    order = [keys[i] for i in rng.permutation(len(keys))]
    return [[style(NAME_KEY), entity.name]] + [
        [style(key), entity.values[key]] for key in order
    ]


def body(
    kind: EntityKind,
    entity: Entity,
    stated: Sequence[str],
    rng: np.random.Generator,
    n_generic: int,
) -> str:
    """Return the prose of a passage: the intro, the stated attributes, filler.

    Args:
        kind: The kind of the entity.
        entity: The entity described.
        stated: The attributes the prose states, in order.
        rng: The source of the filler choice.
        n_generic: How many attribute-free sentences to add.

    Returns:
        The prose.
    """
    slots = {NAME_KEY: entity.name, **entity.values}
    lines = [kind.intro.format(**slots)]
    lines += [kind.sentences[key].format(**slots) for key in stated]
    chosen = rng.choice(len(kind.generic), size=n_generic, replace=False)
    lines += [kind.generic[int(i)] for i in chosen]
    return " ".join(lines)


def _subset(rng: np.random.Generator, keys: Sequence[str], low: int) -> list[str]:
    """Return between low and all of the keys, in a drawn order."""
    n = int(rng.integers(min(low, len(keys)), len(keys) + 1))
    return [keys[int(i)] for i in rng.choice(len(keys), size=n, replace=False)]


def _passage(
    role: str,
    entity: Entity,
    pairs: list[list[str]],
    text: str,
) -> dict[str, Any]:
    """Return one candidate in the shape the scorer reads.

    The body is held under `truncated_body` to match the corpus candidate
    lists. It is never truncated here, since `overlong` checks every pair
    fits.
    """
    return {"role": role, "entity": entity.name, "pairs": pairs, "truncated_body": text}


def draw_pool(
    kind: EntityKind, target: str, style_name: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Draw one pool, without checking it for leaks.

    Args:
        kind: The kind of entity the question asks about.
        target: The asked attribute.
        style_name: The key naming of the pool, a key of KEY_STYLES.
        rng: The source of the draw.

    Returns:
        The question, the answer, the wrong answer and the candidates.
    """
    style = KEY_STYLES[style_name]
    stateable = [key for key in kind.sentences if key != target]
    others = [key for key in kind.attributes if key != target]

    subject = draw_entity(kind, rng)
    sibling = draw_entity(kind, rng, kind.sibling(subject.name, rng))

    gold = _passage(
        GOLD_ROLE,
        subject,
        metadata(subject, [target, *_subset(rng, others, 1)], style, rng),
        body(kind, subject, _subset(rng, stateable, 1), rng, int(rng.integers(1, 3))),
    )
    wrong = _passage(
        WRONG_ROLE,
        sibling,
        metadata(sibling, [target, *_subset(rng, others, 1)], style, rng),
        body(
            kind,
            sibling,
            [target, *_subset(rng, stateable, 0)],
            rng,
            int(rng.integers(0, 2)),
        ),
    )
    near = [
        _passage(
            NEAR_ROLE,
            subject,
            metadata(subject, _subset(rng, others, 2), style, rng),
            body(
                kind, subject, _subset(rng, stateable, 1), rng, int(rng.integers(1, 3))
            ),
        )
        for _ in range(N_NEAR)
    ]
    fillers = []
    for _ in range(N_FILLER):
        other = draw_entity(kind, rng)
        fillers.append(
            _passage(
                FILLER_ROLE,
                other,
                metadata(other, [target, *_subset(rng, others, 1)], style, rng),
                body(
                    kind,
                    other,
                    _subset(rng, list(kind.sentences), 1),
                    rng,
                    int(rng.integers(1, 3)),
                ),
            )
        )
    return {
        "kind": kind.kind,
        "target": target,
        "key_style": style_name,
        "question": kind.questions[target].format(name=subject.name),
        "answer": subject.values[target],
        "wrong_answer": sibling.values[target],
        "candidates": [gold, wrong, *near, *fillers],
    }


def passage_text(candidate: dict[str, Any]) -> str:
    """Return every value and the prose of a candidate, for a leak search."""
    return " ".join(
        [*(value for _, value in candidate["pairs"]), candidate["truncated_body"]]
    )


def leaks(pool: dict[str, Any]) -> list[str]:
    """Return the ways the answer could be reached without the gold passage.

    Args:
        pool: One drawn pool.

    Returns:
        One message per problem. Empty when the pool is sound.
    """
    problems = []
    answer, wrong = pool["answer"].lower(), pool["wrong_answer"].lower()
    if answer in wrong or wrong in answer:
        problems.append("the answer and the wrong answer overlap")
    if answer in pool["question"].lower():
        problems.append("the question carries the answer")
    gold_entity = pool["candidates"][0]["entity"]
    for index, candidate in enumerate(pool["candidates"]):
        text = passage_text(candidate).lower()
        if index and answer in text:
            problems.append(
                f"candidate {index} ({candidate['role']}) carries the answer"
            )
        if candidate["role"] != WRONG_ROLE and wrong in text:
            problems.append(
                f"candidate {index} ({candidate['role']}) carries the wrong answer"
            )
        if candidate["role"] == FILLER_ROLE and candidate["entity"] == gold_entity:
            problems.append(f"filler {index} shares the subject's name")
    gold_body = pool["candidates"][0]["truncated_body"].lower()
    if answer in gold_body:
        problems.append("the gold prose states the answer")
    return problems


def build_scenarios(n: int, seed: int) -> list[dict[str, Any]]:
    """Draw the pools of the demonstration.

    The kinds, asked attributes and key namings cycle through every
    combination in a drawn order, so a small run still covers each. Every
    pool also draws one mixed-format assignment, which every arm shares.

    Args:
        n: The number of pools.
        seed: The seed of the whole draw.

    Returns:
        The pools, each with an id and its mixed assignment by format name.

    Raises:
        RuntimeError: If a pool keeps leaking its answer.
    """
    rng = np.random.default_rng(seed)
    combos = [
        (kind, target, style)
        for kind in KINDS
        for target in kind.questions
        for style in KEY_STYLES
    ]
    order = rng.permutation(len(combos))
    pools = []
    for i in range(n):
        kind, target, style = combos[int(order[i % len(combos)])]
        for _ in range(MAX_ATTEMPTS):
            pool = draw_pool(kind, target, style, rng)
            if not leaks(pool):
                break
        else:
            raise RuntimeError(
                f"pool {i} ({kind.kind}, {target}) leaked its answer "
                f"{MAX_ATTEMPTS} times"
            )
        pools.append({"id": f"demo{i:05d}", **pool})
    mixed = mixed_assignment(n, N_CANDIDATES, len(FORMAT_NAMES), rng)
    for pool, row in zip(pools, mixed, strict=True):
        pool["mixed"] = [FORMAT_NAMES[int(index)] for index in row]
    return pools


def role_index(pool: dict[str, Any], role: str) -> int:
    """Return the position of the first candidate with the role."""
    return next(i for i, c in enumerate(pool["candidates"]) if c["role"] == role)


def overlong(
    pools: Sequence[dict[str, Any]], tokenizer: Any, max_tokens: int = MAX_TOKENS
) -> list[tuple[str, int, str]]:
    """Return every pair the scorer would truncate.

    Args:
        pools: The pools to check.
        tokenizer: The tokenizer of the model that will score them.
        max_tokens: The encoding length at which the scorer truncates.

    Returns:
        The pool id, candidate position and format of each long pair.
    """
    found = []
    for pool in pools:
        for index, candidate in enumerate(pool["candidates"]):
            for name in FORMAT_NAMES:
                text = FORMATS[name](candidate["pairs"], candidate["truncated_body"])
                if len(tokenizer(pool["question"], text)["input_ids"]) > max_tokens:
                    found.append((pool["id"], index, name))
    return found
