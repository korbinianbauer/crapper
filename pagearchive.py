"""Archive everything a stored ad page needs to render as it looked back then:
stylesheets (recursively: @import, url()), fonts and images. The page HTML itself is
kept verbatim in listing_versions.raw_html; each version maps its original asset URLs
to content-addressed files in archive/assets/ (version_assets). Stylesheets are stored
with their references rewritten to `archive-asset:<hash>` tokens."""
import hashlib
import ipaddress
import logging
import os
import re
import socket
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import archive
import db

logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'archive', 'assets')
TOKEN = 'archive-asset:'
MAX_BYTES = 10 * 1024 * 1024
MAX_ASSETS_PER_PAGE = 300
MAX_CSS_DEPTH = 4
_UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                     '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'}

_FONT_EXT = {'.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
             '.otf': 'font/otf', '.eot': 'application/vnd.ms-fontobject'}
_IMG_EXT = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp', '.avif': 'image/avif',
            '.svg': 'image/svg+xml', '.ico': 'image/x-icon'}

_CSS_URL = re.compile(r'''url\(\s*(['"]?)(?!data:|#|archive-asset:)(.+?)\1\s*\)''', re.I)
_CSS_IMPORT = re.compile(r'''@import\s+(['"])(?!archive-asset:)(.+?)\1''', re.I)


# ── URL / type helpers ──────────────────────────────────────────────────────────
def _is_public(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ('http', 'https') or not p.hostname:
        return False
    try:
        return all(ipaddress.ip_address(ai[4][0]).is_global
                   for ai in socket.getaddrinfo(p.hostname, None))
    except Exception:
        return False


def _ctype(header: str, url: str) -> str | None:
    """Normalised content type if it is one we archive (css/image/font), else None."""
    ct = (header or '').split(';')[0].strip().lower()
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ct == 'text/css' or (ext == '.css' and ct in ('', 'text/plain', 'application/octet-stream')):
        return 'text/css'
    if ct.startswith(('image/', 'font/')) or ct in (
            'application/font-woff', 'application/font-woff2', 'application/x-font-woff',
            'application/x-font-ttf', 'application/x-font-otf', 'application/vnd.ms-fontobject'):
        return ct
    if ct in ('', 'application/octet-stream', 'binary/octet-stream'):
        return _FONT_EXT.get(ext) or _IMG_EXT.get(ext)
    return None


def _fetch(url: str) -> tuple[bytes, str] | None:
    for _ in range(4):                      # follow redirects manually (SSRF check per hop)
        if not _is_public(url):
            return None
        r = requests.get(url, headers=_UA, timeout=20, allow_redirects=False, stream=True)
        if r.is_redirect and r.headers.get('location'):
            url = urljoin(url, r.headers['location'])
            continue
        if r.status_code != 200:
            return None
        data = r.raw.read(MAX_BYTES + 1, decode_content=True)
        ct = _ctype(r.headers.get('content-type', ''), url)
        if len(data) > MAX_BYTES or not data or ct is None:
            return None
        return data, ct
    return None


# ── CSS / HTML reference walking ────────────────────────────────────────────────
def css_rewrite(text: str, base: str, fn) -> str:
    """Replace every url()/@import target by fn(absolute_url) (None = keep as is)."""
    def sub(m, fmt):
        ref = m.group(2).strip()
        new = fn(urljoin(base, ref)) if ref else None
        return fmt(m.group(1), new) if new else m.group(0)
    text = _CSS_URL.sub(lambda m: sub(m, lambda q, n: f'url({q}{n}{q})'), text)
    return _CSS_IMPORT.sub(lambda m: sub(m, lambda q, n: f'@import {q}{n}{q}'), text)


_SKIP = ('data:', '#', 'javascript:', 'mailto:', 'about:', 'blob:', 'tel:')


def walk_page(soup: BeautifulSoup, base: str, fn):
    """Call fn(absolute_url) for every asset reference in the page (stylesheet, icon,
    img/source/video src+srcset, inline style url()s) and substitute the result
    (None = leave untouched)."""
    def conv(u):
        u = (u or '').strip()
        return None if not u or u.lower().startswith(_SKIP) else fn(urljoin(base, u))

    for tag in soup.find_all(True):
        name = tag.name
        if name == 'link' and tag.get('href'):
            rel = ' '.join(tag.get('rel') or []).lower()
            if 'stylesheet' in rel or 'icon' in rel or (
                    rel == 'preload' and tag.get('as') in ('image', 'font', 'style')):
                if new := conv(tag['href']):
                    tag['href'] = new
                    tag.attrs.pop('integrity', None)
        elif name in ('img', 'source', 'video', 'audio', 'input'):
            for a in ('src', 'data-src', 'data-lazy-src', 'poster'):
                if tag.get(a) and (new := conv(tag[a])):
                    tag[a] = new
            for a in ('srcset', 'data-srcset'):
                if tag.get(a):
                    parts = []
                    for part in re.split(r',\s+|,(?=\S+\s+\d)', tag[a].strip()):
                        bits = part.strip().split(None, 1)
                        if bits:
                            bits[0] = conv(bits[0]) or bits[0]
                            parts.append(' '.join(bits))
                    tag[a] = ', '.join(parts)
        elif name == 'style' and tag.get_text():
            tag.string = css_rewrite(tag.get_text(), base, conv)
        if tag.get('style'):
            tag['style'] = css_rewrite(tag['style'], base, conv)


# ── archiving ───────────────────────────────────────────────────────────────────
def _store(data: bytes, ctype: str) -> str:
    h = hashlib.sha256(data).hexdigest()[:40]
    os.makedirs(ASSET_DIR, exist_ok=True)
    path = os.path.join(ASSET_DIR, h)
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(data)
    db.put_asset(h, ctype, len(data))
    return h


def _archive_asset(url: str, cache: dict, depth: int = 0) -> str | None:
    if url in cache:
        return cache[url]
    cache[url] = None
    got = None
    if (local := archive.local_name(url)):          # ad image we already downloaded earlier
        with open(os.path.join(archive.IMG_DIR, local), 'rb') as f:
            got = (f.read(), _IMG_EXT.get(os.path.splitext(local)[1], 'image/jpeg'))
    else:
        try:
            got = _fetch(url)
        except Exception as e:
            logger.debug('asset fetch failed %s: %s', url, e)
    if got is None:
        return None
    data, ctype = got
    if ctype == 'text/css':
        if depth >= MAX_CSS_DEPTH:
            return None
        text = data.decode('utf-8', 'replace')
        text = css_rewrite(text, url, lambda u: (TOKEN + h) if (h := _archive_asset(u, cache, depth + 1)) else None)
        data = text.encode()
    cache[url] = _store(data, ctype)
    return cache[url]


def archive_version(version_id: int, html: str, page_url: str, cache: dict) -> tuple[int, int]:
    """Archive the assets of one stored page version. Returns (archived, failed)."""
    soup = BeautifulSoup(html, 'lxml')
    refs: list[str] = []
    walk_page(soup, page_url, lambda u: refs.append(u))
    mapping, failed = {}, 0
    for u in list(dict.fromkeys(refs))[:MAX_ASSETS_PER_PAGE]:
        if (h := _archive_asset(u, cache)):
            mapping[u] = h
        else:
            failed += 1
    db.save_version_assets(version_id, mapping)
    return len(mapping), failed


def archive_pending(time_budget: float = 1800) -> int:
    """Archive assets for every stored page version that has none yet (new versions,
    and the backlog from before this existed). Newest first."""
    start, done, cache = time.time(), 0, {}
    for v in db.versions_pending_assets():
        if time.time() - start > time_budget:
            logger.info('Asset archiving: time budget used up, continuing next run')
            break
        html = v['html']
        if html.lstrip().startswith('{'):        # JSON source (ImmoScout): nothing to render
            db.mark_assets_done(v['id'])
            continue
        try:
            ok, failed = archive_version(v['id'], html, v['url'], cache)
        except Exception as e:
            logger.warning('Asset archiving failed for version %s: %s', v['id'], e)
            continue
        db.mark_assets_done(v['id'])
        done += 1
        logger.info('Archived %d assets (%d unavailable) for listing %s v%s',
                    ok, failed, v['listing_id'], v['version'])
    return done


# ── serving ─────────────────────────────────────────────────────────────────────
def read_asset(h: str) -> tuple[bytes, str] | None:
    meta = db.get_asset(h)
    if not meta:
        return None
    try:
        with open(os.path.join(ASSET_DIR, h), 'rb') as f:
            return f.read(), meta['content_type']
    except OSError:
        return None


def resolve_tokens(css: bytes, root: str) -> bytes:
    return re.sub(rb'archive-asset:([0-9a-f]{40})',
                  lambda m: (root + 'archive/asset/').encode() + m.group(1), css)
