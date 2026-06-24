"""ImmoScout24 (immobilienscout24.de) source — real estate.

URL-driven. The website HTML is bot-walled, but the mobile app's JSON gateway is
usable without auth, so a pasted web search URL is parsed and translated into a
mobile-API search:
- Region search:  /Suche/de/<land>/<kreis>[/<ort>]/<objektart-slug>
                  → searchType=region&geocodes=/de/<land>/<kreis>[/<ort>]
- Radius search:  /Suche/radius/<objektart-slug>?geocoordinates=<lat>;<lon>;<r>…
                  → searchType=radius&geocoordinates=<lat>;<lon>;<r>
Single-expose detail (/expose/<id>) is WAF-blocked → listings unsupported.
"""
import logging
import re
import time
from urllib.parse import parse_qs, urlparse

import requests

from .base import ScrapedListing, Source, UrlInfo

logger = logging.getLogger(__name__)

_API = 'https://api.mobile.immobilienscout24.de'
_API_HEADERS = {'User-Agent': 'ImmoScout24_1410_30_._', 'Accept': 'application/json'}
_TIMEOUT = 25
_PAGE_SIZE = 20

# Web URL slug → mobile-API realestatetype
_SLUG_TO_TYPE = {
    'wohnung-mieten': 'apartmentrent',
    'wohnung-kaufen': 'apartmentbuy',
    'haus-mieten': 'houserent',
    'haus-kaufen': 'housebuy',
    'grundstueck-kaufen': 'livingsitebuy',
}
_TYPE_LABEL = {
    'apartmentrent': 'Wohnung mieten', 'apartmentbuy': 'Wohnung kaufen',
    'houserent': 'Haus mieten', 'housebuy': 'Haus kaufen',
    'livingsitebuy': 'Grundstück kaufen',
}


def _parse_price(text: str | None) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    raw = ' '.join(text.split())
    m = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*€', raw)
    if not m:
        return None, raw
    num = m.group(1).replace('.', '').replace(',', '.')
    try:
        return float(num), raw
    except ValueError:
        return None, raw


def _prettify(segment: str) -> str:
    return segment.replace('-', ' ').title()


class ImmoScoutSource(Source):
    name = 'immoscout'
    display_name = 'ImmoScout24'

    # ── URL recognition ──────────────────────────────────────────────────────
    def matches_url(self, url: str) -> bool:
        return 'immobilienscout24.de' in url

    def _parse_search_url(self, url: str) -> dict | None:
        """Translate a web search URL into mobile-API query parameters.
        Returns {'query': <api query string>, 'type': <realestatetype>,
        'location': <human location>, 'radius': <km|0>} or None."""
        u = urlparse(url)
        parts = [p for p in u.path.split('/') if p]
        ret = None
        slug_idx = None
        for i, p in enumerate(parts):
            if p in _SLUG_TO_TYPE:
                ret = _SLUG_TO_TYPE[p]
                slug_idx = i
                break
        if ret is None:
            return None

        qs = parse_qs(u.query)
        geocoords = (qs.get('geocoordinates') or [''])[0]
        if geocoords:  # radius search — coordinates carry "lat;lon;radius"
            bits = geocoords.split(';')
            radius = 0
            try:
                radius = int(float(bits[2])) if len(bits) >= 3 else 0
            except ValueError:
                radius = 0
            center = (qs.get('centerofsearchaddress') or qs.get('centerOfSearchAddress') or [''])[0]
            location = center.split(';')[0] if center else 'Umkreis'
            return {'query': f'searchType=radius&geocoordinates={geocoords}',
                    'type': ret, 'location': location, 'radius': radius}

        # region search — geocode is the path between "/Suche" and the slug
        geo_parts = parts[1:slug_idx]            # ['de','bayern','miesbach-kreis', …]
        if not geo_parts:
            return None
        geocodes = '/' + '/'.join(geo_parts)
        location = _prettify(geo_parts[-1])
        return {'query': f'searchType=region&geocodes={geocodes}',
                'type': ret, 'location': location, 'radius': 0}

    def _label(self, parsed: dict) -> str:
        rt = _TYPE_LABEL.get(parsed['type'], parsed['type'])
        suffix = f' +{parsed["radius"]} km' if parsed.get('radius') else ''
        return f'{rt} · {parsed["location"]}{suffix}'

    def classify(self, url: str) -> UrlInfo | None:
        if '/expose/' in url:
            return UrlInfo('listing', supported=False,
                           note='ImmoScout-Einzelanzeigen werden nicht unterstützt — '
                                'nur Suchlisten (WAF-Schutz auf der Detailseite).')
        if '/Suche/' in url or '/suche/' in url:
            parsed = self._parse_search_url(url)
            if parsed is None:
                return UrlInfo('search', supported=False,
                               note='Diese ImmoScout-Suche wird nicht unterstützt '
                                    '(Objektart unbekannt).')
            return UrlInfo('search', label=self._label(parsed))
        return None

    # ── search ───────────────────────────────────────────────────────────────
    def _api_get(self, parsed: dict, page: int) -> dict | None:
        url = (f'{_API}/search?{parsed["query"]}&realestatetype={parsed["type"]}'
               f'&pagesize={_PAGE_SIZE}&pagenumber={page}')
        try:
            resp = requests.get(url, headers=_API_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error('IS24 API failed (p%d): %s', page, e)
            return None

    def describe_search(self, url: str) -> tuple[int | None, str]:
        parsed = self._parse_search_url(url)
        if parsed is None:
            return None, ''
        data = self._api_get(parsed, 1)
        if data is None:
            return None, self._label(parsed)
        return data.get('totalResults'), self._label(parsed)

    def fetch_search(self, url: str, max_pages: int = 5) -> list[ScrapedListing]:
        parsed = self._parse_search_url(url)
        if parsed is None:
            return []
        seen: dict[str, ScrapedListing] = {}
        for page in range(1, max_pages + 1):
            data = self._api_get(parsed, page)
            if data is None:
                break
            results = data.get('results') or []
            if not results:
                break
            for item in results:
                listing = self._parse_item(item)
                if listing:
                    seen.setdefault(listing.ad_id, listing)
            logger.info('IS24 search p%d: %d items (%d total)', page, len(results), len(seen))
            if page >= int(data.get('numberOfPages', page)):
                break
            time.sleep(1.0)
        return list(seen.values())

    def _parse_item(self, item: dict) -> ScrapedListing | None:
        ad_id = item.get('id')
        if not ad_id:
            return None
        price_text = None
        for attr in item.get('attributes', []):
            val = attr.get('value', '')
            if '€' in val:
                price_text = val
                break
        price, price_text = _parse_price(price_text)
        pic = item.get('titlePicture') or {}
        image_url = pic.get('preview') or pic.get('full')
        if not image_url:
            pics = item.get('pictures') or []
            if pics:
                image_url = (pics[0].get('urlScaleAndCrop') or '').replace('%WIDTH%', '400').replace('%HEIGHT%', '300')
        address = (item.get('address') or {}).get('line')
        return ScrapedListing(
            ad_id=str(ad_id),
            url=f'https://www.immobilienscout24.de/expose/{ad_id}',
            title=item.get('title', '').strip() or None,
            location=address,
            image_url=image_url or None,
            price=price,
            price_text=price_text,
        )
