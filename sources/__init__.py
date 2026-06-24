"""Source registry. Add a new site by implementing a Source subclass and
appending an instance to `_all`."""
from .base import LocationResult, ScrapedListing, SearchField, Source
from .immoscout import ImmoScoutSource
from .kleinanzeigen import KleinanzeigenSource

_all: list[Source] = [
    KleinanzeigenSource(),
    ImmoScoutSource(),
]

REGISTRY: dict[str, Source] = {s.name: s for s in _all}


def get(name: str) -> Source:
    return REGISTRY[name]


def all_sources() -> list[Source]:
    return list(REGISTRY.values())


def search_sources() -> list[Source]:
    return [s for s in _all if s.supports_search]


def source_for_url(url: str) -> Source | None:
    """Return the source a pasted single-listing URL belongs to, if any."""
    for s in _all:
        if s.supports_listing and s.matches_url(url):
            return s
    return None
