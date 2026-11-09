"""Source abstraction: a tracked site (Kleinanzeigen, ImmoScout, …).

Everything is URL-driven: the user pastes one URL, the source recognises whether
it is a single listing or a search and extracts all parameters from the URL
itself. No site-specific input fields are needed.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    # Archive material — only filled when the source has it (full ad pages):
    description: str | None = None
    images: list[str] = field(default_factory=list)   # all image URLs of the ad
    raw_html: str | None = None                       # original page, archived verbatim


@dataclass
class UrlInfo:
    """What a pasted URL represents, derived from the URL alone (no network)."""
    type: str                           # 'listing' | 'search'
    supported: bool = True              # False → recognised but can't be tracked
    note: str = ''                      # reason shown to the user when unsupported
    label: str = ''                     # best-effort human label from the URL
    ad_id: str | None = None            # source-local ad id, for 'listing' URLs


class Source(ABC):
    """Abstract base for a trackable site. Subclass, set `name`/`display_name`,
    and implement URL recognition + fetching, then register in __init__.py."""

    name: str                  # stable slug + DB key, e.g. 'kleinanzeigen'
    archive_details = False    # True → search hits get their full ad page fetched + archived daily
    display_name: str          # shown in the UI, e.g. 'Kleinanzeigen'

    @abstractmethod
    def matches_url(self, url: str) -> bool:
        """True if *url* belongs to this site (any type)."""

    @abstractmethod
    def classify(self, url: str) -> UrlInfo | None:
        """Recognise the URL (listing vs search) from the URL alone, or None if
        it doesn't look like a trackable page on this site."""

    def full_image_url(self, url: str) -> str:
        """URL of the largest available version of an ad image (previews/thumbnails
        map to the full-size original). Default: the URL itself."""
        return url

    def fetch_listing(self, url: str) -> ScrapedListing | None:
        """Fetch one listing, or None if removed / unavailable."""
        raise NotImplementedError

    def fetch_search(self, url: str, max_pages: int = 5) -> list[ScrapedListing]:
        """Run the search described by *url* and return every listing found."""
        raise NotImplementedError

    def describe_search(self, url: str) -> tuple[int | None, str]:
        """Quick preview for a search URL: (total result count, refined label).
        Count may be None if unknown. Used by the paste-time check."""
        return None, ''
