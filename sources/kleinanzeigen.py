"""Kleinanzeigen (kleinanzeigen.de) source — general classifieds.

URL-driven. Plain browser-like HTTP is enough; no JS rendering needed.
- Listing URL:  …/s-anzeige/<slug>/<adId>-<cat>-<x>
- Search  URL:  …/s-<keyword>/k0l<locId>r<radius>  (also …/s-<ort>/<keyword>/k0…);
                pagination injects "seite:N" right after "/s-".
- Listing fields: #viewad-title / #viewad-price / #viewad-locality / og:image.
"""
import json
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base import RateLimited, ScrapedListing, SearchIncomplete, Source, UrlInfo

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


def _best_img_url(img) -> str | None:
    """First real http(s) image URL of an <img>: lazy-loaded cards carry a
    placeholder (data: URI / empty) in src and the real one in data-* / srcset."""
    if img is None:
        return None
    cands = [img.get('src'), img.get('data-imgsrc'), img.get('data-src')]
    srcset = img.get('srcset') or img.get('data-srcset')
    if srcset:
        cands.append(srcset.split(',')[0].strip().split(' ')[0])
    for c in cands:
        if c and c.strip().startswith(('http://', 'https://')):
            return c.strip()
    return None


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


# ── tracker titles from search parameters ─────────────────────────────────────
_CATEGORY_NAMES = {
    'immobilien': 'Immobilien', 'grundstuecke-garten': 'Grundstücke & Gärten',
    'auto-rad-boot': 'Auto, Rad & Boot', 'elektronik': 'Elektronik',
    'haus-garten': 'Haus & Garten', 'mode-beauty': 'Mode & Beauty',
    'familie-kind-baby': 'Familie, Kind & Baby', 'freizeit-hobby-nachbarschaft':
    'Freizeit, Hobby & Nachbarschaft', 'musik-film-buecher': 'Musik, Filme & Bücher',
    'haustiere': 'Haustiere', 'jobs': 'Jobs', 'dienstleistungen': 'Dienstleistungen',
    'eintrittskarten-tickets': 'Eintrittskarten & Tickets',
    'unentgeltlich-zu-verschenken': 'Zu verschenken', 'verschenken-tauschen':
    'Verschenken & Tauschen', 'fahrzeuge': 'Fahrzeuge',
}
_CODE_RE = re.compile(r'^(?:k0|(?=c\d))(?:c(\d+))?(?:l(\d+))?(?:r(\d+))?(?P<attrs>\+.*)?$')
_H1_COUNT_RE = re.compile(r'^\d[\d.]*\s*-\s*\d[\d.]*\s+von\s+[\d.]+\s+')


def _pretty(slug: str) -> str:
    return slug.replace('_', ' ').replace('-', ' ').strip().capitalize()


def _price_filter(v: str) -> str:
    lo, _, hi = v.partition(':')
    if lo and hi:
        return f'{lo}–{hi} €'
    return f'ab {lo} €' if lo else (f'bis {hi} €' if hi else '')


def _h1_parts(h1: str) -> tuple[str | None, str | None]:
    """(keyword, place) from a result heading like
    '1 - 25 von 47 Immobilien für „wald“ in Schliersee - Bayern und Umgebung'.
    The count is dropped on purpose (KA's count is inflated and unstable)."""
    text = ' '.join((h1 or '').split())
    text = re.sub(r'\d[\d.]*\s*-\s*\d[\d.]*\s+von\s+[\d.]+', ' ', text)
    kw = re.search(r'„(.+?)“', text)
    m = (re.search(r'„.+?“\s+in\s+(.+?)(?:\s+-\s+.*)?$', text)
         or re.search(r'.*\bin\s+(.+?)(?:\s+-\s+.*)?$', text))
    place = ' '.join(m.group(1).split()) if m else None
    return (kw.group(1) if kw else None), place


def search_label(url: str, h1: str | None = None) -> str:
    """Describe what a Kleinanzeigen search tracks, e.g.
    '„wald“ · Grundstücke & Gärten · Miesbach +30 km · kaufen'.
    Everything comes from the URL (category, radius, filters); keyword and place
    names additionally from the page heading when available."""
    path, parts, flt = urlparse(url).path, [], []
    segs = [x for x in path.split('/') if x]
    if not segs or not segs[0].startswith('s-'):
        return 'Kleinanzeigen-Suche'
    segs[0] = segs[0][2:]
    code = _CODE_RE.match(segs[-1]) if segs else None
    names = []
    for seg in segs[:-1] if code else segs:
        if seg.startswith('seite:') or not seg:
            continue
        if ':' in seg:
            k, _, v = seg.partition(':')
            if k == 'preis':
                if f := _price_filter(v):
                    flt.append(f)
            elif k == 'anzeige':
                if v != 'angebote':               # offers are the default, not worth showing
                    flt.append(_pretty(v))
            elif k == 'versand':
                flt.append('mit Versand' if v == 'ja' else 'ohne Versand')
            else:
                flt.append(f'{_pretty(k)}: {_pretty(v)}')
        else:
            names.append(seg)
    cat_id = code.group(1) if code else None
    radius = code.group(3) if code else None
    has_loc = bool(code and code.group(2))
    attrs = code.group('attrs') if code else None
    for a in (attrs or '').split('+'):
        if ':' in a:
            flt.append(a.split(':', 1)[1].replace('_', ' '))

    kw, place = _h1_parts(h1) if h1 else (None, None)
    category = _CATEGORY_NAMES.get(names[0], _pretty(names[0])) if cat_id and names else None
    if kw is None and not h1:
        # offline: the keyword slug is the last name segment (after category [/ place])
        extra = names[(1 if cat_id else 0):]
        if has_loc and extra:
            place = place or _pretty(extra[0])    # city slug
            extra = extra[1:]
        if extra:
            kw = extra[-1].replace('-', ' ')

    out = []
    if kw:
        out.append(f'„{kw}“')
    if category:
        out.append(category)
    if has_loc:
        where = place or 'Umkreis'
        out.append(f'{where} +{radius} km' if radius else where)
    elif place:
        out.append(place)
    else:
        out.append('Deutschland')
    out += flt
    return ' · '.join(out)


class KleinanzeigenSource(Source):
    name = 'kleinanzeigen'
    display_name = 'Kleinanzeigen'
    archive_details = True

    def _get(self, url: str) -> requests.Response:
        return requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

    def _extract_ad_id(self, url: str) -> str | None:
        m = _AD_ID_RE.search(url)
        return m.group(1) if m else None

    def full_image_url(self, url: str) -> str:
        # img.kleinanzeigen.de serves size variants via ?rule=$_N.EXT; $_57 is the largest
        # (1600x1200, the stored original), $_59 only 960x720, $_1 a 400x300 thumbnail.
        if urlparse(url).netloc == 'img.kleinanzeigen.de' and 'rule=' in url:
            return re.sub(r'rule=[^&]*', 'rule=$_57.JPG', url)
        return url

    # ── URL recognition ──────────────────────────────────────────────────────
    def matches_url(self, url: str) -> bool:
        return 'kleinanzeigen.de' in url

    def classify(self, url: str) -> UrlInfo | None:
        path = urlparse(url).path
        ad_id = self._extract_ad_id(url)
        if '/s-anzeige/' in url or ad_id:
            return UrlInfo('listing', label=ad_id or 'Anzeige', ad_id=ad_id)
        if path.startswith('/s-'):       # all KA search paths start with /s-
            return UrlInfo('search', label=search_label(url))
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
        if resp.status_code in (403, 429):
            logger.warning('KA listing %s HTTP %s (rate-limited/blocked)', ad_id, resp.status_code)
            raise RateLimited(f'KA HTTP {resp.status_code}')
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
        desc_el = soup.select_one('#viewad-description-text')
        details_el = soup.select_one('#viewad-details')
        images: list[str] = []
        for u in ([img_el['content']] if img_el and img_el.get('content') else []) + [
                _best_img_url(i) for i in soup.select('#viewad-product img, .galleryimage-element img')]:
            if u and u not in images:
                images.append(u)
        price, price_text = _parse_price(price_el.get_text(strip=True) if price_el else None)
        return ScrapedListing(
            ad_id=str(ad_id),
            url=resp.url,
            title=title_el.get_text(strip=True),
            location=loc_el.get_text(' ', strip=True) if loc_el else None,
            image_url=images[0] if images else None,
            price=price,
            price_text=price_text,
            description=self._full_text(desc_el, details_el),
            images=images,
            raw_html=html,
        )

    @staticmethod
    def _full_text(desc_el, details_el) -> str | None:
        parts = []
        if desc_el:
            parts.append(desc_el.get_text('\n', strip=True))
        if details_el:
            rows = [' '.join(li.get_text(' ', strip=True).split())
                    for li in details_el.select('li')]
            if rows:
                parts.append('Details:\n' + '\n'.join(rows))
        return '\n\n'.join(parts) or None

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
        h1 = BeautifulSoup(resp.text, _PARSER).select_one('h1')
        label = search_label(url, h1.get_text(' ', strip=True) if h1 else None)
        return self._parse_total(resp.text), label

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

            # Newer card layout: the link only wraps the image (its text is the photo
            # count, e.g. "30"); title and image URL live in an embedded ld+json
            # ImageObject. Prefer that, then the classic title elements.
            title, ld_img = '', None
            for sc in art.select('script[type="application/ld+json"]'):
                try:
                    ld = json.loads(sc.string or '')
                except ValueError:
                    continue
                if isinstance(ld, dict):
                    title = title or (ld.get('title') or '').strip()
                    ld_img = ld_img or ld.get('contentUrl')
            if not title:
                el = art.select_one('.text-module-begin a, h2 a, .ellipsis') or link
                title = el.get_text(strip=True) if el else ''
                if title.isdigit():
                    title = ''
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

            image_url = next((u for u in map(_best_img_url, art.select('img')) if u), None) or ld_img

            out.append(ScrapedListing(
                ad_id=str(ad_id), url=urljoin(_BASE, href),
                title=title or None,
                location=' '.join(loc_el.get_text(' ', strip=True).split()) if loc_el else None,
                image_url=image_url,
                price=price, price_text=price_text,
            ))
        return out

    def fetch_search(self, url: str, max_pages: int = 5) -> list[ScrapedListing]:
        seen: dict[str, ScrapedListing] = {}
        total: int | None = None
        stale = 0
        for page in range(1, max_pages + 1):
            try:
                resp = self._get(self._page_url(url, page))
                resp.raise_for_status()
            except Exception as e:
                logger.error('KA search failed (p%d): %s', page, e)
                raise SearchIncomplete(f'KA search page {page} failed: {e}', list(seen.values()))
            if total is None:
                total = self._parse_total(resp.text)
            page_results = self._parse_page(resp.text)
            if not page_results:
                if page == 1 and (total is None or total > 0):      # block page / layout change, not "no ads"
                    raise SearchIncomplete('KA search page 1 had no ads', [])
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
            # Past the last real page KA keeps serving the last page again; the
            # reported total is inflated so it never triggers. Two consecutive
            # pages without new ads (tolerating one transient duplicate) = end.
            stale = stale + 1 if new == 0 else 0
            if stale >= 2:
                break
            if page < max_pages:
                time.sleep(1.5)
        return list(seen.values())
