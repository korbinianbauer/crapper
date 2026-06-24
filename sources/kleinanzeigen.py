"""Kleinanzeigen (kleinanzeigen.de) source — general classifieds.

Plain browser-like HTTP is enough; no JS rendering needed.
- Location ids: GET /s-ort-empfehlungen.json?query=…
- Search URL: /s-{keyword}/k0l{locId}r{radius}, paged via /s-seite:{n}/…
- Listing fields: #viewad-title / #viewad-price / #viewad-locality / og:image.
"""
import logging
import re
import time
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

from .base import LocationResult, ScrapedListing, SearchField, Source

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
_AD_ID_RE = re.compile(r'/(\d{6,})-\d+-\d+\b')
_REMOVED_MARKERS = (
    'Die gewünschte Anzeige ist nicht mehr verfügbar',
    'Anzeige nicht gefunden',
    'ist nicht mehr verfügbar',
)
_RADIUS_OPTIONS = [['0', 'nur Ort'], ['5', '+5 km'], ['10', '+10 km'],
                   ['20', '+20 km'], ['50', '+50 km'], ['100', '+100 km']]


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
    supports_search = True
    supports_listing = True

    def _get(self, url: str) -> requests.Response:
        return requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

    # ── single listings ────────────────────────────────────────────────────
    def matches_url(self, url: str) -> bool:
        return 'kleinanzeigen.de' in url and ('/s-anzeige/' in url or bool(self.extract_ad_id(url)))

    def extract_ad_id(self, url: str) -> str | None:
        m = _AD_ID_RE.search(url)
        return m.group(1) if m else None

    def fetch_listing(self, url: str) -> ScrapedListing | None:
        ad_id = self.extract_ad_id(url) or url
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
        soup = BeautifulSoup(html, 'html.parser')
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
    def search_fields(self) -> list[SearchField]:
        return [
            SearchField('query', 'Suchbegriff', 'text', placeholder='z.B. wald'),
            SearchField('location', 'Ort', 'location', placeholder='z.B. Miesbach'),
            SearchField('radius', 'Umkreis', 'select', options=_RADIUS_OPTIONS),
        ]

    def search_locations(self, query: str) -> list[LocationResult]:
        if len(query.strip()) < 2:
            return []
        url = f'{_BASE}/s-ort-empfehlungen.json?query={quote(query.strip())}'
        try:
            resp = self._get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error('KA location lookup failed for %r: %s', query, e)
            return []
        out = []
        for key, name in data.items():
            loc_id = key.lstrip('_')
            if loc_id == '0':           # whole of Germany
                continue
            out.append(LocationResult(id=loc_id, name=name))
        return out[:10]

    def search_label(self, params: dict) -> str:
        radius = params.get('radius', '0')
        loc = params.get('location_label', '')
        suffix = f' +{radius} km' if radius and radius != '0' else ''
        return f'„{params.get("query", "")}" · {loc}{suffix}'

    def _build_url(self, query: str, location_id: str, radius: int, page: int) -> str:
        kw = quote(query.strip().lower().replace(' ', '-'))
        loc = f'l{location_id}r{int(radius)}'
        if page > 1:
            return f'{_BASE}/s-seite:{page}/{kw}/k0{loc}'
        return f'{_BASE}/s-{kw}/k0{loc}'

    def _parse_page(self, html: str) -> list[ScrapedListing]:
        soup = BeautifulSoup(html, 'html.parser')
        out: list[ScrapedListing] = []
        for art in soup.select('article.aditem'):
            ad_id = art.get('data-adid')
            href = art.get('data-href') or ''
            link = art.select_one('a[href*="/s-anzeige/"]')
            if not href and link:
                href = link.get('href', '')
            if not ad_id and href:
                ad_id = self.extract_ad_id(href)
            if not ad_id or not href:
                continue
            title_el = art.select_one('.text-module-begin a, h2 a, .ellipsis')
            price_el = art.select_one(
                '.aditem-main--middle--price-shipping--price, .aditem-main--middle--price')
            loc_el = art.select_one('.aditem-main--top--left')
            img_el = art.select_one('.aditem-image img, .imagebox img')
            image_url = None
            if img_el:
                image_url = img_el.get('src') or img_el.get('data-imgsrc') or img_el.get('srcset')
            location = ' '.join(loc_el.get_text(' ', strip=True).split()) if loc_el else None
            price, price_text = _parse_price(price_el.get_text(strip=True) if price_el else None)
            out.append(ScrapedListing(
                ad_id=str(ad_id), url=urljoin(_BASE, href),
                title=title_el.get_text(strip=True) if title_el else None,
                location=location, image_url=image_url,
                price=price, price_text=price_text,
            ))
        return out

    def fetch_search(self, params: dict, max_pages: int = 5) -> list[ScrapedListing]:
        query = params.get('query', '')
        location_id = params.get('location', '')
        try:
            radius = int(params.get('radius') or 0)
        except (ValueError, TypeError):
            radius = 0
        seen: dict[str, ScrapedListing] = {}
        for page in range(1, max_pages + 1):
            url = self._build_url(query, location_id, radius, page)
            try:
                resp = self._get(url)
                resp.raise_for_status()
            except Exception as e:
                logger.error('KA search failed (%s p%d): %s', query, page, e)
                break
            page_results = self._parse_page(resp.text)
            if not page_results:
                break
            new = sum(1 for r in page_results if r.ad_id not in seen)
            for r in page_results:
                seen.setdefault(r.ad_id, r)
            logger.info('KA search %r p%d: %d ads (%d new)', query, page, len(page_results), new)
            if new == 0:
                break
            if page < max_pages:
                time.sleep(1.0)
        return list(seen.values())
