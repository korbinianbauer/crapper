import hmac
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import defaultdict
from functools import wraps

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, session, url_for)

import db
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

@app.route('/')
def index():
    trackers = db.get_trackers()
    listings = db.get_all_listings()
    history = db.get_price_history_map()

    source_names = {s.name: s.display_name for s in sources.all_sources()}

    cards = []
    for l in listings:
        hist = history.get(l['id'], [])
        latest = next((h for h in reversed(hist) if h['price_text']), None)
        types = (l['tracker_types'] or '').split(',')
        cards.append({
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
            'active': bool(l['active']),
            'first_seen': l['first_seen'],
            'last_seen': l['last_seen'],
            'current_price': latest['price_text'] if latest else '—',
            'history': [
                {'t': h['observed_at'], 'price': h['price'], 'label': h['price_text']}
                for h in hist
            ],
        })

    counts = db.get_tracker_listing_counts()
    for t in trackers:
        t['listing_count'] = counts.get(t['id'], 0)
        t['source_name'] = source_names.get(t['source'], t['source'])

    return render_template(
        'index.html',
        trackers=trackers,
        cards=cards,
        search_sources=sources.search_sources(),
        active_count=sum(1 for c in cards if c['active']),
        gone_count=sum(1 for c in cards if not c['active']),
        refreshing=list(_active_refreshes),
    )


# ── add / manage trackers ───────────────────────────────────────────────────────

@app.route('/location_search')
def location_search():
    q = request.args.get('q', '')
    source_name = request.args.get('source', '')
    if source_name not in sources.REGISTRY:
        return jsonify([])
    results = sources.get(source_name).search_locations(q)
    return jsonify([{'id': r.id, 'name': r.name} for r in results])


@app.route('/source_fields/<source_name>')
def source_fields(source_name):
    if source_name not in sources.REGISTRY:
        return jsonify({'error': 'unknown source'}), 404
    source = sources.get(source_name)
    return jsonify({
        'name': source.name,
        'display_name': source.display_name,
        'supports_search': source.supports_search,
        'fields': [f.to_dict() for f in source.search_fields()],
    })


@app.route('/add_listing', methods=['POST'])
@login_required
@csrf_protect
def add_listing():
    url = request.form.get('url', '').strip()
    label = request.form.get('label', '').strip()
    source = sources.source_for_url(url)
    if source is None:
        supported = ', '.join(s.display_name for s in sources.all_sources() if s.supports_listing)
        flash(f'Diese URL gehört zu keiner unterstützten Einzelanzeigen-Quelle '
              f'({supported}). Für Suchen nutze „+ Suchbegriff".', 'error')
        return redirect(url_for('index'))
    ad_id = source.extract_ad_id(url)
    existing = db.find_listing_tracker(source.name, ad_id, url)
    if existing:
        flash(f'Diese Anzeige wird bereits getrackt: {existing["label"]}.', 'error')
        return redirect(url_for('index'))
    if not label:
        label = ad_id or url
    tracker_id = db.add_listing_tracker(source.name, label, url)
    _refresh_async(tracker_id)
    return redirect(url_for('index'))


@app.route('/add_search', methods=['POST'])
@login_required
@csrf_protect
def add_search():
    source_name = request.form.get('source', '').strip()
    if source_name not in sources.REGISTRY:
        flash('Unbekannte Quelle.', 'error')
        return redirect(url_for('index'))
    source = sources.get(source_name)

    # Collect the values for this source's declared search fields (+ the *_label
    # companions that location fields submit for display).
    params: dict = {}
    for f in source.search_fields():
        val = request.form.get(f.name, '').strip()
        if f.required and not val:
            flash(f'Bitte „{f.label}" ausfüllen.', 'error')
            return redirect(url_for('index'))
        params[f.name] = val
        if f.type == 'location':
            params[f.name + '_label'] = request.form.get(f.name + '_label', '').strip()

    existing = db.find_search_tracker(source_name, params)
    if existing:
        flash(f'Diese Suche wird bereits getrackt: {existing["label"]}.', 'error')
        return redirect(url_for('index'))
    label = source.search_label(params)
    tracker_id = db.add_search_tracker(source_name, label, params)
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


@app.route('/refresh_status')
def refresh_status():
    return jsonify({'running': list(_active_refreshes)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
