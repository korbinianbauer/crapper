"""Kleinanzeigen (kleinanzeigen.de) source — general classifieds.

URL-driven. Plain browser-like HTTP is enough; no JS rendering needed.
- Listing URL:  …/s-anzeige/<slug>/<adId>-<cat>-<x>
- Search  URL:  …/s-<keyword>/k0l<locId>r<radius>  (also …/s-<ort>/<keyword>/k0…);
                pagination injects "seite:N" right after "/s-".
- Listing fields: #viewad-title / #viewad-price / #viewad-locality / og:image.
"""
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base import ScrapedListing, Source, UrlInfo

logger = logging.getLogger(__name__)

_BASE = 'https://www.kleinanzeigen.de'
_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64; rv:120.0) '
                   'Gecko/20100101 Firefox/120.0'),
    'Accept-Language': 'de-DE,de;q=0.9',
    'Accept': ('text/html,application/xhtml+xml,application/xml;q=0.9,'
               'image/avif,image/webp,*/*;q=0.8'),
}
_TIMEOUT = 20
# lxml is far more tolerant of the broken markup KA emits on some category result
# pages (the stdlib html.parser silently drops most <article> nodes there).
_PARSER = 'lxml'
_AD_ID_RE = re.compile(r'/(\d{6,})-\d+-\d+\b')
# Result count from the "<from> - <to> von <total> …" heading (works across
# verticals: "… von 233 Ergebnissen", "… von 94 Immobilien", …).
_COUNT_RE = re.compile(r'\d+\s*[-–]\s*\d+\s+von\s+([\d.]+)')
_REMOVED_MARKERS = (
    'Die gewünschte Anzeige ist nicht mehr verfügbar',
    'Anzeige nicht gefunden',
    'ist nicht mehr verfügbar',
)


def _parse_price(text: str | None) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    raw = ' '.join(text.split())
    m = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*€', raw)
    if not m:
        return None, raw
    clean = raw[:m.end()].strip()
    if 'VB' in raw[m.end():] or raw.rstrip().endswith('VB'):
        clean += ' VB'
    num = m.group(1).replace('.', '').replace(',', '.')
    try:
        return float(num), clean
    except ValueError:
        return None, clean


class KleinanzeigenSource(Source):
    name = 'kleinanzeigen'
    display_name = 'Kleinanzeigen'

    def _get(self, url: str) -> requests.Response:
        return requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

    def _extract_ad_id(self, url: str) -> str | None:
        m = _AD_ID_RE.search(url)
        return m.group(1) if m else None

    # ── URL recognition ──────────────────────────────────────────────────────
    def matches_url(self, url: str) -> bool:
        return 'kleinanzeigen.de' in url

    def classify(self, url: str) -> UrlInfo | None:
        path = urlparse(url).path
        ad_id = self._extract_ad_id(url)
        if '/s-anzeige/' in url or ad_id:
            return UrlInfo('listing', label=ad_id or 'Anzeige', ad_id=ad_id)
        if path.startswith('/s-'):       # all KA search paths start with /s-
            return UrlInfo('search', label='Kleinanzeigen-Suche')
        return None

    # ── single listing ───────────────────────────────────────────────────────
    def fetch_listing(self, url: str) -> ScrapedListing | None:
        ad_id = self._extract_ad_id(url) or url
        try:
            resp = self._get(url)
        except Exception as e:
            logger.error('KA listing fetch error %s: %s', url, e)
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.warning('KA listing %s HTTP %s', ad_id, resp.status_code)
            return None
        html = resp.text
        if any(m in html for m in _REMOVED_MARKERS):
            return None
        soup = BeautifulSoup(html, _PARSER)
        title_el = soup.select_one('#viewad-title')
        if title_el is None:
            return None
        price_el = soup.select_one('#viewad-price') or soup.select_one('h2.boxedarticle--price')
        loc_el = soup.select_one('#viewad-locality')
        img_el = soup.select_one('meta[property="og:image"]')
        price, price_text = _parse_price(price_el.get_text(strip=True) if price_el else None)
        return ScrapedListing(
            ad_id=str(ad_id),
            url=resp.url,
            title=title_el.get_text(strip=True),
            location=loc_el.get_text(' ', strip=True) if loc_el else None,
            image_url=img_el['content'] if img_el and img_el.get('content') else None,
            price=price,
            price_text=price_text,
        )

    # ── search ───────────────────────────────────────────────────────────────
    def _page_url(self, url: str, page: int) -> str:
        if page <= 1:
            return url
        # Inject "seite:N" right after the leading "/s-".
        return re.sub(r'/s-', f'/s-seite:{page}/', url, count=1)

    def _label_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, _PARSER)
        h1 = soup.select_one('h1')
        if not h1:
            return ''
        text = ' '.join(h1.get_text(' ', strip=True).split())
        m = re.search(r'Ergebnis(?:se|sen)?\s+für\s+(.*)$', text)
        return m.group(1).strip() if m else text

    def _parse_total(self, html: str) -> int | None:
        m = _COUNT_RE.search(html)
        return int(m.group(1).replace('.', '')) if m else None

    def describe_search(self, url: str) -> tuple[int | None, str]:
        try:
            resp = self._get(self._page_url(url, 1))
            resp.raise_for_status()
        except Exception as e:
            logger.error('KA describe_search failed: %s', e)
            return None, ''
        return self._parse_total(resp.text), self._label_from_html(resp.text)

    def _parse_page(self, html: str) -> list[ScrapedListing]:
        # Match by data-adid (not a CSS class) so this survives KA's old
        # `article.aditem` cards and the newer Tailwind-styled result cards alike.
        soup = BeautifulSoup(html, _PARSER)
        out: list[ScrapedListing] = []
        for art in soup.select('article[data-adid]'):
            ad_id = art.get('data-adid')
            link = art.select_one('a[href*="/s-anzeige/"]')
            href = art.get('data-href') or (link.get('href', '') if link else '')
            if not ad_id and href:
                ad_id = self._extract_ad_id(href)
            if not ad_id or not href:
                continue

            title_el = art.select_one('.text-module-begin a, h2 a, .ellipsis') or link
            loc_el = art.select_one('.aditem-main--top--left')

            # Price: dedicated element if present, else the first € amount found
            # in the card text (kept to just that snippet, not the whole card).
            price_el = art.select_one(
                '.aditem-main--middle--price-shipping--price, .aditem-main--middle--price')
            if price_el:
                price, price_text = _parse_price(price_el.get_text(strip=True))
            else:
                m = re.search(r'\d[\d.]*(?:,\d+)?\s*€(?:\s*VB)?', art.get_text(' ', strip=True))
                price, price_text = _parse_price(m.group(0)) if m else (None, None)

            img_el = art.select_one('img')
            image_url = None
            if img_el:
                image_url = img_el.get('src') or img_el.get('data-imgsrc') or img_el.get('srcset')

            out.append(ScrapedListing(
                ad_id=str(ad_id), url=urljoin(_BASE, href),
                title=title_el.get_text(strip=True) if title_el else None,
                location=' '.join(loc_el.get_text(' ', strip=True).split()) if loc_el else None,
                image_url=image_url,
                price=price, price_text=price_text,
            ))
        return out

    def fetch_search(self, url: str, max_pages: int = 5) -> list[ScrapedListing]:
        seen: dict[str, ScrapedListing] = {}
        total: int | None = None
        for page in range(1, max_pages + 1):
            try:
                resp = self._get(self._page_url(url, page))
                resp.raise_for_status()
            except Exception as e:
                logger.error('KA search failed (p%d): %s', page, e)
                break
            if total is None:
                total = self._parse_total(resp.text)
            page_results = self._parse_page(resp.text)
            if not page_results:
                break
            new = sum(1 for r in page_results if r.ad_id not in seen)
            for r in page_results:
                seen.setdefault(r.ad_id, r)
            logger.info('KA search p%d: %d ads (%d new, %d/%s total)',
                        page, len(page_results), new, len(seen), total)
            # Stop when we've covered the reported total. Don't stop on new==0:
            # KA throttling can return a duplicate page transiently, so keep
            # going (a slower cadence avoids that) until the total is reached.
            if total is not None and len(seen) >= total:
                break
            if total is None and new == 0:
                break          # unknown total → fall back to "no new ads" stop
            if page < max_pages:
                time.sleep(1.5)
        return list(seen.values())
