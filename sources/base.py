"""Source abstraction: a tracked site (Kleinanzeigen, ImmoScout, …).

A Source knows how to (optionally) resolve locations, run a search, and fetch a
single listing. Capabilities are declared per source so the rest of the app can
adapt the UI and poller without special-casing any particular site.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class ScrapedListing:
    """Normalised listing returned by every source."""
    ad_id: str                          # source-local id (namespaced by source in the DB)
    url: str
    title: str | None = None
    location: str | None = None
    image_url: str | None = None
    price: float | None = None          # numeric currency value, None when not numeric
    price_text: str | None = None       # raw label, e.g. "85.000 €", "VB", "560 € Kaltmiete"


@dataclass
class LocationResult:
    """A location autocomplete candidate. `id` is whatever the source needs to
    run a search for that place (a numeric id, a geocode path, …)."""
    id: str
    name: str


@dataclass
class SearchField:
    """Describes one input of a source's search form so the frontend can render
    it generically. `type` is one of: 'text', 'select', 'location'."""
    name: str
    label: str
    type: str
    options: list = field(default_factory=list)   # [[value, label], …] for 'select'
    placeholder: str = ''
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class Source(ABC):
    """Abstract base for a trackable site. Subclass, set the class attributes and
    implement the methods your site supports, then register it in __init__.py."""

    name: str                  # stable slug + DB key, e.g. 'kleinanzeigen'
    display_name: str          # shown in the UI, e.g. 'Kleinanzeigen'
    supports_search: bool = True
    supports_listing: bool = True

    # ── single listings ────────────────────────────────────────────────────
    def matches_url(self, url: str) -> bool:
        """True if a pasted single-listing URL belongs to this source."""
        return False

    def extract_ad_id(self, url: str) -> str | None:
        """Extract the source-local ad id from a listing URL, if possible."""
        return None

    def fetch_listing(self, url: str) -> ScrapedListing | None:
        """Fetch one listing, or None if it has been removed / is unavailable."""
        raise NotImplementedError

    # ── search ───────────────────────────────────────────────────────────────
    def search_fields(self) -> list[SearchField]:
        """The inputs of this source's search form (rendered by the frontend)."""
        return []

    def search_locations(self, query: str) -> list[LocationResult]:
        """Autocomplete for any 'location' search field."""
        return []

    def search_label(self, params: dict) -> str:
        """Human-readable label for a configured search."""
        return self.display_name

    def fetch_search(self, params: dict, max_pages: int = 5) -> list[ScrapedListing]:
        """Run the search described by *params* and return every listing found."""
        raise NotImplementedError
