import difflib
import hmac
import re
import logging
import math
import os
import secrets
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import wraps

from bs4 import BeautifulSoup
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)

import archive
import pagearchive
import db
import ml
import sources
from poller import poll_tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '')
if not app.secret_key:
    # Personal single-user app: fall back to an ephemeral key so it still boots
    # without SECRET_KEY set (sessions reset on restart). Set SECRET_KEY in prod.
    app.secret_key = secrets.token_hex(32)
    logger.warning('SECRET_KEY not set — using an ephemeral key (sessions reset on restart)')

_ADMIN_USER = os.environ.get('CRAPPER_USER', 'admin')
_ADMIN_PASSWORD = os.environ.get('CRAPPER_PASSWORD', '')
if not _ADMIN_PASSWORD:
    logger.warning('CRAPPER_PASSWORD not set — login will always fail')

db.init_db()


# ── file logging ──────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
_file_handler = logging.FileHandler(os.path.join(_DIR, 'app.log'))
_file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s — %(message)s'))
logging.getLogger().addHandler(_file_handler)


# ── poll process management ───────────────────────────────────────────────────

def _ensure_poll_running():
    pidfile = os.path.join(_DIR, 'poll.pid')
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return  # already running
    except (OSError, ValueError):
        pass
    with open(os.path.join(_DIR, 'poll.log'), 'a') as log_fh:
        subprocess.Popen(
            [sys.executable, os.path.join(_DIR, 'poll.py')],
            cwd=_DIR, stdout=log_fh, stderr=subprocess.STDOUT,
        )
    logger.info('Started poll.py process')


_ensure_poll_running()


# ── CSRF ──────────────────────────────────────────────────────────────────────

def _csrf_token() -> str:
    if '_csrf' not in session:
        session['_csrf'] = secrets.token_hex(16)
    return session['_csrf']

app.jinja_env.globals['csrf_token'] = _csrf_token


def csrf_protect(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.form.get('_csrf') or request.headers.get('X-CSRF-Token') or ''
        expected = session.get('_csrf', '')
        if not expected or not hmac.compare_digest(token, expected):
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ── rate limiting / auth ────────────────────────────────────────────────────────

_login_attempts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW = 300
_RATE_LIMIT_MAX = 10


def _is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    recent = [t for t in _login_attempts[ip] if now - t < _RATE_LIMIT_WINDOW]
    _login_attempts[ip] = recent
    return len(recent) >= _RATE_LIMIT_MAX


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        ip = request.remote_addr or ''
        if _is_rate_limited(ip):
            error = 'Too many attempts. Try again later.'
        else:
            u = request.form.get('username', '')
            p = request.form.get('password', '')
            user_ok = hmac.compare_digest(u, _ADMIN_USER)
            pass_ok = bool(_ADMIN_PASSWORD) and hmac.compare_digest(p, _ADMIN_PASSWORD)
            if user_ok and pass_ok:
                session['logged_in'] = True
                next_url = request.args.get('next') or url_for('index')
                if not next_url.startswith('/') or next_url.startswith('//'):
                    next_url = url_for('index')
                return redirect(next_url)
            _login_attempts[ip].append(time.monotonic())
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/logout', methods=['POST'])
@csrf_protect
def logout():
    session.clear()
    return redirect(url_for('index'))


# ── refresh tracking ────────────────────────────────────────────────────────────

_active_refreshes: set[int] = set()


def _refresh_async(tracker_id: int):
    tracker = db.get_tracker(tracker_id)
    if not tracker:
        return
    _active_refreshes.add(tracker_id)

    def _run():
        try:
            poll_tracker(tracker)
        except Exception as e:
            logger.error('Refresh failed for tracker %d: %s', tracker_id, e)
        finally:
            _active_refreshes.discard(tracker_id)

    threading.Thread(target=_run, daemon=True).start()


# ── index ───────────────────────────────────────────────────────────────────────

GONE_HIDE_DAYS = 7          # vanished listings are hidden by default after this long
MIN_INDEX_PRICE = 5.0       # ignore placeholder prices (1 €, 0 €) in the index
MAX_DAILY_MOVE = 2.0        # clamp a single listing's daily ratio to [1/2, 2]


def _median_price(cards: list[dict]) -> float | None:
    """Median current numeric price of the *active* listings (price level, as
    opposed to the chain-linked index which measures price movement)."""
    prices = [c['current_price_num'] for c in cards
              if c['active'] and c['current_price_num'] is not None
              and c['current_price_num'] >= MIN_INDEX_PRICE]
    return statistics.median(prices) if prices else None


def _price_index(items: list[dict], today: date | None = None,
                 gap_days: set[str] | None = None) -> list[dict]:
    """Chain-linked daily price index (Jevons / geometric mean), base 100 = 1.0.

    `items`: dicts with `history` (`{'t': iso, 'price': float|None}`), `active`
    and `last_seen`. Per day the index moves by the geometric mean of the
    day-over-day price ratios of the listings that had a numeric price on both
    days, so new listings enter without causing a jump and vanished ones simply
    drop out (a listing counts until the day it was last seen). Prices are
    forward-filled between observations (history only stores changes).
    Returns `[{'t': 'YYYY-MM-DD', 'index': float, 'median': float|None}]`, extended
    up to today; `median` is the median price of the listings alive that day
    (price level, in contrast to the movement-only index)."""
    today = today or date.today()
    gap_days = gap_days or set()
    series = []                                    # (start, end, [(day, price)])
    for it in items:
        pts = [(date.fromisoformat(h['t'][:10]), h['price'])
               for h in it['history'] if h['price'] is not None]
        if not pts:
            continue
        end = today if it.get('active', True) else \
            min(today, date.fromisoformat((it.get('last_seen') or '')[:10] or today.isoformat()))
        start = pts[0][0]
        if end < start:
            end = start
        series.append((start, end, pts))
    if not series:
        return []

    first = min(s[0] for s in series)
    days = [first + timedelta(d) for d in range((today - first).days + 1)]

    def price_on(pts, day, cache):
        # pts are chronological; last observation on/before `day`
        i = cache[0]
        while i + 1 < len(pts) and pts[i + 1][0] <= day:
            i += 1
        cache[0] = i
        return pts[i][1]

    caches = [[0] for _ in series]
    prev: list[float | None] = [None] * len(series)
    level = 1.0
    def med(vals):
        v = [x for x in vals if x is not None and x >= MIN_INDEX_PRICE]
        return statistics.median(v) if v else None

    out = []
    for k, day in enumerate(days):
        cur: list[float | None] = []
        for (start, end, pts), c in zip(series, caches):
            cur.append(price_on(pts, day, c) if start <= day <= end else None)
        if k == 0:
            out.append({'t': day.isoformat(), 'index': level, 'median': None if day.isoformat() in gap_days else med(cur)})
        else:
            logs = []
            for a, b in zip(prev, cur):
                if a is None or b is None or a < MIN_INDEX_PRICE or b < MIN_INDEX_PRICE:
                    continue
                r = min(max(b / a, 1 / MAX_DAILY_MOVE), MAX_DAILY_MOVE)
                logs.append(math.log(r))
            if logs:
                level *= math.exp(sum(logs) / len(logs))
            out.append({'t': day.isoformat(), 'index': level, 'median': None if day.isoformat() in gap_days else med(cur)})
        prev = cur
    return out


@app.route('/', defaults={'tracker_id': None})
@app.route('/tracker/<int:tracker_id>')
def index(tracker_id):
    """Front page. `/tracker/<id>` is the same page with that tracker's detail
    view open, so browser history and shareable URLs reflect where you are."""
    trackers = db.get_trackers()
    if tracker_id is not None and not any(t['id'] == tracker_id for t in trackers):
        abort(404)
    listings = db.get_all_listings()
    history = db.get_price_history_map()
    memberships = db.get_tracker_memberships()
    fair_map = db.get_fair_price_map()

    source_names = {s.name: s.display_name for s in sources.all_sources()}

    gone_cutoff = (datetime.utcnow() - timedelta(days=GONE_HIDE_DAYS)).isoformat()
    cards: dict[int, dict] = {}
    for l in listings:
        hist = history.get(l['id'], [])
        latest = next((h for h in reversed(hist) if h['price_text']), None)
        latest_num = next((h for h in reversed(hist) if h['price'] is not None), None)
        types = (l['tracker_types'] or '').split(',')
        fair = fair_map.get(l['id'])
        price_num = latest_num['price'] if latest_num else None
        fair_price = fair_delta = None
        if fair:
            fair_price = round(fair['fair_price'])
            if price_num and ml.is_plausible(price_num, fair['features'], fair['per']):
                fair_delta = round((price_num / fair['fair_price'] - 1) * 100)   # + = above fair
        cards[l['id']] = {
            'id': l['id'],
            'source': l['source'],
            'source_name': source_names.get(l['source'], l['source']),
            'tracker_labels': l['tracker_labels'] or '—',
            'is_search': 'search' in types,
            'is_listing': 'listing' in types,
            'ad_id': l['ad_id'],
            'url': l['url'],
            'title': l['title'] or '(untitled)',
            'location': l['location'],
            'image_url': l['image_url'],
            'image_file': l['image_file'],
            'active': bool(l['active']),
            # gone for more than a week → hidden in tracker views unless asked for
            'stale_gone': not l['active'] and (l['last_seen'] or '') < gone_cutoff,
            'first_seen': l['first_seen'],
            'last_seen': l['last_seen'],
            'current_price': latest['price_text'] if latest else '—',
            'current_price_num': price_num,
            'fair_price': fair_price,
            'fair_delta': fair_delta,
            'history': [
                {'t': h['observed_at'], 'price': h['price'], 'label': h['price_text']}
                for h in hist
            ],
        }

    counts = db.get_tracker_listing_counts()
    problems = db.get_price_problems()
    gaps = db.get_gap_days()
    tracker_cards = []
    for t in trackers:
        t['listing_count'] = counts.get(t['id'], 0)
        t['source_name'] = source_names.get(t['source'], t['source'])

        ids = [lid for lid in memberships.get(t['id'], []) if lid in cards]
        tcards = [cards[i] for i in ids]
        prices = [c['current_price_num'] for c in tcards if c['current_price_num'] is not None]
        active = sum(1 for c in tcards if c['active'])
        tracker_cards.append({
            'id': t['id'],
            'source_name': t['source_name'],
            'label': t['label'],
            'type': t['type'],
            'url': t['url'],
            'listing_ids': ids,
            'listing_count': len(tcards),
            'active_count': active,
            'gone_count': len(tcards) - active,
            'avg_price': sum(prices) / len(prices) if prices else None,
            'min_price': min(prices) if prices else None,
            'max_price': max(prices) if prices else None,
            'median_price': _median_price(tcards),
            'price_problem': problems.get(t['id']),
            'trend': _price_index([cards[i] for i in ids], gap_days=gaps.get(t['id'])),
        })

    hidden = [{
        'id': l['id'],
        'source_name': source_names.get(l['source'], l['source']),
        'title': l['title'] or '(untitled)',
        'url': l['url'],
        'location': l['location'],
    } for l in db.get_hidden_listings()]

    return render_template(
        'index.html',
        trackers=trackers,
        tracker_cards=tracker_cards,
        cards=cards,
        hidden_listings=hidden,
        listing_count=len(cards),
        active_count=sum(1 for c in cards.values() if c['active']),
        gone_count=sum(1 for c in cards.values() if not c['active']),
        refreshing=list(_active_refreshes),
        open_tracker=tracker_id,
    )


@app.route('/archive/img/<path:name>')
def archive_image(name):
    return send_from_directory(archive.IMG_DIR, name, max_age=86400 * 30)


@app.route('/archive/asset/<h>')
def archive_asset(h):
    if not re.fullmatch(r'[0-9a-f]{40}', h):
        abort(404)
    got = pagearchive.read_asset(h)
    if not got:
        abort(404)
    data, ctype = got
    if ctype == 'text/css':
        data = pagearchive.resolve_tokens(data, '//' + request.host + '/')
        ctype += '; charset=utf-8'
    return Response(data, mimetype=None, content_type=ctype, headers={
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Access-Control-Allow-Origin': '*',      # fonts from the sandboxed (opaque-origin) page
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "sandbox; default-src 'none'; style-src 'unsafe-inline'"})


@app.route('/listing/<int:listing_id>/versions')
@login_required
def listing_versions(listing_id):
    return jsonify(db.get_versions(listing_id))


@app.route('/listing/<int:listing_id>/archive')
def listing_archive(listing_id):
    """Wayback-style view: newest archived version by default, timeline of all versions."""
    listing = db.get_listing(listing_id)
    if not listing:
        abort(404)
    versions = sorted(db.get_versions(listing_id), key=lambda v: v['version'])
    versions = [v for v in versions if v['has_html']] or versions   # snapshots = stored pages
    if not versions:
        abort(404)
    want = request.args.get('v', type=int)
    idx = next((i for i, v in enumerate(versions) if v['version'] == want), len(versions) - 1)
    cur = versions[idx]
    prev = versions[idx - 1] if idx > 0 else None
    prices = db.get_price_history(listing_id)

    def price_at(i):    # price valid at version i = last observation before the next version
        end = versions[i + 1]['fetched_at'] if i + 1 < len(versions) else '9999'
        seen = [p['price_text'] for p in prices if p['observed_at'] < end]
        return seen[-1] if seen else None

    src = sources.get(listing['source'])

    def image_urls(v):    # {full-size URL: [stored variants]} – one entry per photo
        urls = db.get_version_images(listing_id, v['version']) or (
            [v['image_url']] if v['image_url'] else [])
        photos = {}
        for u in urls:
            photos.setdefault(src.full_image_url(u), []).append(u)
        return photos

    def img(full, variants):    # prefer the archived full-size file, else any archived variant
        name = next(filter(None, (archive.local_name(u) for u in [full, *variants])), None)
        return {'url': full, 'src': ('/archive/img/' + name) if name else full}

    price = price_at(idx)
    cur_imgs = image_urls(cur)
    prev_imgs = image_urls(prev) if prev else {}
    images = [dict(img(u, vs), new=bool(prev) and u not in prev_imgs) for u, vs in cur_imgs.items()]
    removed_images = [img(u, vs) for u, vs in prev_imgs.items() if u not in cur_imgs] if prev else []
    diff = None
    if prev:
        diff = {k: _wdiff(o, n) for k, o, n in (
            ('title', prev['title'], cur['title']),
            ('location', prev['location'], cur['location']),
            ('description', prev['description'], cur['description']),
            ('price', price_at(idx - 1), price))}
        diff = {k: {'segs': v, 'changed': any(op != 'eq' for op, _ in v)} for k, v in diff.items()}
        diff['images'] = {'changed': bool(removed_images or any(i['new'] for i in images))}
    fair = db.get_fair_price(listing_id)
    if fair:
        cur_price = next((p['price'] for p in reversed(prices) if p['price'] is not None), None)
        fair['current'] = cur_price
        fair['plausible'] = ml.is_plausible(cur_price, fair['features'], fair['metrics'].get('per'))
        fair['delta_pct'] = round((cur_price / fair['fair_price'] - 1) * 100) if cur_price else None
        m = fair['metrics']
        fair['mae'] = m['candidates'][m['chosen']]['mae']
        fair['median_ape'] = round(m['candidates'][m['chosen']]['median_ape'] * 100)
        feats = {f['key']: f for f in fair['definition']['features']}
        def show(f, v):
            if v is None:
                return 'unbekannt'
            if f['type'] == 'bool':
                return 'Ja' if v else 'Nein'
            return int(v) if isinstance(v, float) and v.is_integer() else v
        fair['facts'] = [(f['label'], show(f, fair['features'].get(f['key'])))
                         for f in fair['definition']['features']]
    source_names = {s.name: s.display_name for s in sources.all_sources()}
    raw = db.get_version_html(listing_id, cur['version']) if cur['has_html'] else None
    has_page = bool(raw) and not raw.lstrip().startswith('{')   # ImmoScout archives JSON, not a page
    view = request.args.get('view')
    if view not in ('page', 'data') or (view == 'page' and not has_page):
        view = 'page' if has_page else 'data'
    return render_template('archive.html', listing=listing, versions=versions, cur=cur, prev=prev,
                           idx=idx, images=images, removed_images=removed_images, diff=diff,
                           price=price, has_page=has_page, view=view, fair=fair,
                           source_name=source_names.get(listing['source'], listing['source']))


def _wdiff(old, new):
    """Word-level diff as [(op, text)], op in eq/ins/del."""
    a, b = re.findall(r'\s+|\S+', old or ''), re.findall(r'\s+|\S+', new or '')
    segs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == 'equal':
            segs.append(('eq', ''.join(a[i1:i2])))
            continue
        if i2 > i1:
            segs.append(('del', ''.join(a[i1:i2])))
        if j2 > j1:
            segs.append(('ins', ''.join(b[j1:j2])))
    return segs


@app.route('/listing/<int:listing_id>/version/<int:version>/page')
def listing_version_page(listing_id, version):
    """The archived original page, as it was fetched, for the archive iframe. Scripts
    are stripped and the response is sandboxed; stylesheets, fonts and images come from
    our own asset archive (state at archive time), not from the origin."""
    raw = db.get_version_html(listing_id, version)
    listing = db.get_listing(listing_id)
    if raw is None or not listing:
        abort(404)
    soup = BeautifulSoup(raw, 'lxml')
    mapping = db.get_version_assets(listing_id, version)
    root = '//' + request.host + '/'    # scheme-relative: behind a proxy host_url may say http

    def local(u):    # archived copy of this asset (as of this version), else the old image archive
        if u in mapping:
            return root + 'archive/asset/' + mapping[u]
        if name := archive.local_name(u):
            return root + 'archive/img/' + name
        return None
    pagearchive.walk_page(soup, listing['url'], local)
    for t in soup.find_all(['script', 'iframe', 'object', 'embed']):
        t.decompose()
    for t in soup.find_all('meta', attrs={'http-equiv': re.compile('refresh', re.I)}):
        t.decompose()
    for t in soup.find_all(True):
        for a in [a for a in t.attrs if a.lower().startswith('on')]:
            del t.attrs[a]
    head = soup.head or soup.new_tag('head')
    if soup.head is None and soup.html is not None:
        soup.html.insert(0, head)
    base = soup.new_tag('base', href=listing['url'], target='_blank')
    head.insert(0, base)
    return Response(str(soup), mimetype='text/html', headers={
        'Content-Security-Policy': "sandbox allow-popups allow-popups-to-escape-sandbox; "
                                   "script-src 'none'; object-src 'none'",
        'X-Content-Type-Options': 'nosniff'})


@app.route('/listing/<int:listing_id>/version/<int:version>/html')
def listing_version_html(listing_id, version):
    html = db.get_version_html(listing_id, version)
    if html is None:
        abort(404)
    # Archived third-party page: serve as inert text/plain-sandboxed document.
    mime = 'application/json' if html.lstrip().startswith('{') else 'text/html'  # ImmoScout archives raw JSON
    return Response(html, mimetype=mime,
                    headers={'Content-Security-Policy': 'sandbox; script-src \'none\'',
                             'X-Content-Type-Options': 'nosniff'})


def _safe_next() -> str | None:
    """Same-site path from the form's `next` field (never an external URL)."""
    n = request.form.get('next') or ''
    return n if n.startswith('/') and not n.startswith('//') else None


@app.route('/listing/<int:listing_id>/hide', methods=['POST'])
@login_required
@csrf_protect
def hide_listing(listing_id):
    db.set_listing_hidden(listing_id, True)
    return redirect(_safe_next() or url_for('index'))


@app.route('/listing/<int:listing_id>/unhide', methods=['POST'])
@login_required
@csrf_protect
def unhide_listing(listing_id):
    db.set_listing_hidden(listing_id, False)
    return redirect(_safe_next() or url_for('index'))


# ── add / manage trackers ───────────────────────────────────────────────────────

def _inspect_url(url: str) -> dict:
    """Recognise a pasted URL and (for searches) count how many ads it would
    track / (for listings) fetch a quick summary. Returns a JSON-able dict."""
    url = url.strip()
    if not url:
        return {'ok': False, 'error': 'Bitte eine URL einfügen.'}
    source = sources.source_for_url(url)
    if source is None:
        known = ', '.join(s.display_name for s in sources.all_sources())
        return {'ok': False, 'error': f'URL nicht erkannt. Unterstützte Seiten: {known}.'}
    info = source.classify(url)
    if info is None:
        return {'ok': False, 'error': f'URL als {source.display_name}-Seite erkannt, '
                                      f'aber weder Anzeige noch Suche.'}
    if not info.supported:
        return {'ok': False, 'source': source.name, 'source_name': source.display_name,
                'type': info.type, 'error': info.note}

    base = {'ok': True, 'source': source.name, 'source_name': source.display_name,
            'type': info.type, 'label': info.label}
    if info.type == 'listing':
        scraped = source.fetch_listing(url)
        if scraped is None:
            return {'ok': False, 'error': 'Anzeige nicht abrufbar (entfernt oder gesperrt).'}
        base['label'] = scraped.title or info.label
        base['count'] = 1
        base['summary'] = (f'Einzelanzeige · {scraped.title or "?"} · '
                           f'{scraped.price_text or "—"}')
    else:
        count, label = source.describe_search(url)
        if label:
            base['label'] = label
        base['count'] = count
        n = f'~{count}' if count is not None else 'mehrere'
        base['summary'] = f'Suche · {base["label"]} · {n} Anzeigen'
    return base


@app.route('/inspect')
@login_required
def inspect():
    return jsonify(_inspect_url(request.args.get('url', '')))


@app.route('/add', methods=['POST'])
@login_required
@csrf_protect
def add_tracker():
    url = request.form.get('url', '').strip()
    label_in = request.form.get('label', '').strip()
    source = sources.source_for_url(url)
    if source is None:
        flash('URL nicht erkannt — keine unterstützte Quelle.', 'error')
        return redirect(url_for('index'))
    info = source.classify(url)
    if info is None or not info.supported:
        flash(info.note if info else 'URL-Typ nicht erkannt.', 'error')
        return redirect(url_for('index'))

    if info.type == 'listing':
        existing = db.find_listing_tracker(source.name, info.ad_id, url)
    else:
        existing = db.find_search_tracker(source.name, url)
    if existing:
        flash(f'Wird bereits getrackt: {existing["label"]}.', 'error')
        return redirect(url_for('index'))

    label = label_in or info.label or source.display_name
    tracker_id = db.add_tracker(source.name, info.type, label, url)
    _refresh_async(tracker_id)
    return redirect(url_for('index'))


@app.route('/tracker/<int:tracker_id>/delete', methods=['POST'])
@login_required
@csrf_protect
def delete_tracker(tracker_id):
    db.delete_tracker(tracker_id)
    return redirect(url_for('index'))


@app.route('/tracker/<int:tracker_id>/refresh', methods=['POST'])
@login_required
@csrf_protect
def refresh_tracker(tracker_id):
    if tracker_id in _active_refreshes:
        return jsonify({'status': 'running'}), 202
    _refresh_async(tracker_id)
    return jsonify({'status': 'started'}), 202


@app.route('/refresh_all', methods=['POST'])
@login_required
@csrf_protect
def refresh_all():
    for t in db.get_trackers(enabled_only=True):
        _refresh_async(t['id'])
    return redirect(url_for('index'))


# ── settings ───────────────────────────────────────────────────────────────────

SETTINGS_PAGES = [('fair_pricing', 'Fair-Pricing-Modell')]
_trainings: dict[int, dict] = {}      # feature_set_id -> {status: running|done|error, msg}


@app.route('/settings')
@login_required
def settings_index():
    return redirect(url_for('settings_fair_pricing'))


def _fair_stats(fs: dict, fair_map: dict) -> dict:
    """How the asking prices of the feature set's listings relate to the *raw* ML prediction
    (median price / prediction - 1, in %) — the number to look at when choosing an offset."""
    ratios = []
    per = (fs['models'][0]['metrics'].get('per') if fs['models'] else None)
    for r in db.get_training_rows(fs['id']):
        f = fair_map.get(r['listing_id'])
        if f and r['price'] and ml.is_plausible(r['price'], r['features'], per):
            ratios.append(r['price'] / f['raw'])
    if not ratios:
        return {}
    return {'n': len(ratios), 'median_pct': round((statistics.median(ratios) - 1) * 100, 1)}


@app.route('/settings/fair-pricing')
@login_required
def settings_fair_pricing():
    sets = db.get_feature_sets_overview()
    fair_map = db.get_fair_price_map()
    for fs in sets:
        fs['train'] = _trainings.get(fs['id'])
        fs['stats'] = _fair_stats(fs, fair_map)
        for m in fs['models']:
            c = m['metrics']['candidates'][m['metrics']['chosen']]
            m['mae'], m['median_ape'], m['r2'] = round(c['mae']), round(c['median_ape'] * 100, 1), c['r2']
    return render_template('settings_fair_pricing.html', sets=sets, pages=SETTINGS_PAGES,
                           page='fair_pricing')


@app.route('/settings/fair-pricing/status')
@login_required
def settings_fair_pricing_status():
    return jsonify({str(k): v for k, v in _trainings.items()})


@app.route('/settings/fair-pricing/<int:fs_id>/train', methods=['POST'])
@login_required
@csrf_protect
def settings_train_model(fs_id):
    fs = db.get_feature_set_by_id(fs_id)
    if not fs:
        abort(404)
    if _trainings.get(fs_id, {}).get('status') == 'running':
        flash('Training läuft bereits.', 'error')
        return redirect(url_for('settings_fair_pricing'))
    activate = request.form.get('activate') == '1'
    _trainings[fs_id] = {'status': 'running', 'msg': '', 'started': datetime.utcnow().isoformat(timespec='seconds')}

    def _run():
        try:
            metrics = ml.train(fs['name'], activate=activate)
            c = metrics['candidates'][metrics['chosen']]
            _trainings[fs_id] = {'status': 'done', 'msg': f"Modell {metrics['model_id']} ({metrics['chosen'].replace('_', ' ')}, "
                                 f"Ø-Fehler {c['mae']:.0f} €) trainiert" + (' und aktiviert.' if activate else '.')}
        except BaseException as e:          # ml.train signals "too few listings" via SystemExit
            logger.exception('Training of feature set %s failed', fs['name'])
            _trainings[fs_id] = {'status': 'error', 'msg': str(e) or e.__class__.__name__}

    threading.Thread(target=_run, daemon=True).start()
    return redirect(url_for('settings_fair_pricing'))


@app.route('/settings/fair-pricing/<int:fs_id>/model', methods=['POST'])
@login_required
@csrf_protect
def settings_select_model(fs_id):
    fs = db.get_feature_set_by_id(fs_id)
    try:
        model_id = int(request.form.get('model_id', ''))
    except ValueError:
        abort(400)
    if not fs or not db.set_active_model(fs_id, model_id):
        abort(404)
    try:        # listings added since that model was trained still need a prediction from it
        ml.predict(fs['name'], model_id=model_id)
    except SystemExit as e:
        flash(str(e), 'error')
    flash(f'Modell {model_id} wird jetzt für den fairen Preis verwendet.', 'ok')
    return redirect(url_for('settings_fair_pricing'))


@app.route('/settings/fair-pricing/<int:fs_id>/offset', methods=['POST'])
@login_required
@csrf_protect
def settings_set_offset(fs_id):
    if not db.get_feature_set_by_id(fs_id):
        abort(404)
    raw = request.form.get('offset', '').strip().replace(',', '.').replace('%', '')
    try:
        pct = float(raw or 0)
    except ValueError:
        pct = math.nan
    if not math.isfinite(pct) or not -90 <= pct <= 200:
        flash('Offset muss eine Zahl zwischen -90 und 200 (%) sein.', 'error')
    else:
        db.set_fair_offset(fs_id, round(pct, 2))
        flash(f'Offset {pct:+g} % gespeichert.', 'ok')
    return redirect(url_for('settings_fair_pricing'))


@app.route('/refresh_status')
def refresh_status():
    return jsonify({'running': list(_active_refreshes)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
