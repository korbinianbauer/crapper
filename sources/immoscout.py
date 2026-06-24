"""ImmoScout24 (immobilienscout24.de) source — real estate.

The website HTML is behind a bot wall, but the mobile app's JSON gateway is
usable without auth:
- Search: GET https://api.mobile.immobilienscout24.de/search?searchType=region
          &geocodes=<geopath>&realestatetype=<type>&pagenumber=&pagesize=
          (mobile User-Agent required).
- Locations: GET https://www.immobilienscout24.de/geoautocomplete/v3/locations.json?i=…
             → entities with a `geopath.uri` used as the geocode.

Single-expose detail (/expose/{id}) is WAF-blocked, so this source is
search-only (`supports_listing = False`).
"""
import logging
import re
import time

import requests

from .base import LocationResult, ScrapedListing, SearchField, Source

logger = logging.getLogger(__name__)

_API = 'https://api.mobile.immobilienscout24.de'
_GEO = 'https://www.immobilienscout24.de/geoautocomplete/v3/locations.json'
_API_HEADERS = {'User-Agent': 'ImmoScout24_1410_30_._', 'Accept': 'application/json'}
_GEO_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
_TIMEOUT = 25
_PAGE_SIZE = 20

_REALESTATE_TYPES = [
    ['apartmentrent', 'Wohnung mieten'],
    ['apartmentbuy', 'Wohnung kaufen'],
    ['houserent', 'Haus mieten'],
    ['housebuy', 'Haus kaufen'],
    ['livingsitebuy', 'Grundstück kaufen'],
]
_TYPE_LABELS = {v: l for v, l in _REALESTATE_TYPES}


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


class ImmoScoutSource(Source):
    name = 'immoscout'
    display_name = 'ImmoScout24'
    supports_search = True
    supports_listing = False        # /expose detail is WAF-blocked

    # ── search ───────────────────────────────────────────────────────────────
    def search_fields(self) -> list[SearchField]:
        return [
            SearchField('realestatetype', 'Objektart', 'select', options=_REALESTATE_TYPES),
            SearchField('location', 'Ort / Region', 'location', placeholder='z.B. Miesbach'),
        ]

    def search_locations(self, query: str) -> list[LocationResult]:
        if len(query.strip()) < 2:
            return []
        try:
            resp = requests.get(_GEO, params={'i': query.strip()},
                                headers=_GEO_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error('IS24 location lookup failed for %r: %s', query, e)
            return []
        out = []
        for item in data:
            ent = item.get('entity', {})
            uri = (ent.get('geopath') or {}).get('uri')
            label = ent.get('label')
            if uri and label:
                out.append(LocationResult(id=uri, name=label))
        return out[:10]

    def search_label(self, params: dict) -> str:
        rt = _TYPE_LABELS.get(params.get('realestatetype', ''), params.get('realestatetype', ''))
        return f'{rt} · {params.get("location_label", "")}'

    def fetch_search(self, params: dict, max_pages: int = 5) -> list[ScrapedListing]:
        geocodes = params.get('location', '')
        ret = params.get('realestatetype', 'apartmentrent')
        if not geocodes:
            return []
        seen: dict[str, ScrapedListing] = {}
        for page in range(1, max_pages + 1):
            url = (f'{_API}/search?searchType=region&geocodes={geocodes}'
                   f'&realestatetype={ret}&pagesize={_PAGE_SIZE}&pagenumber={page}')
            try:
                resp = requests.get(url, headers=_API_HEADERS, timeout=_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error('IS24 search failed (%s p%d): %s', ret, page, e)
                break
            results = data.get('results') or []
            if not results:
                break
            for item in results:
                listing = self._parse_item(item)
                if listing:
                    seen.setdefault(listing.ad_id, listing)
            logger.info('IS24 search %s/%s p%d: %d items (%d total)',
                        ret, geocodes, page, len(results), len(seen))
            if page >= int(data.get('numberOfPages', page)):
                break
            time.sleep(1.0)
        return list(seen.values())

    def _parse_item(self, item: dict) -> ScrapedListing | None:
        ad_id = item.get('id')
        if not ad_id:
            return None
        # Price is the first attribute carrying a € amount.
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
