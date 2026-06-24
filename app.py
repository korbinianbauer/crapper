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
        cards=cards,
        hidden_listings=hidden,
        active_count=sum(1 for c in cards if c['active']),
        gone_count=sum(1 for c in cards if not c['active']),
        refreshing=list(_active_refreshes),
    )


@app.route('/listing/<int:listing_id>/hide', methods=['POST'])
@login_required
@csrf_protect
def hide_listing(listing_id):
    db.set_listing_hidden(listing_id, True)
    return redirect(url_for('index'))


@app.route('/listing/<int:listing_id>/unhide', methods=['POST'])
@login_required
@csrf_protect
def unhide_listing(listing_id):
    db.set_listing_hidden(listing_id, False)
    return redirect(url_for('index'))


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


@app.route('/refresh_status')
def refresh_status():
    return jsonify({'running': list(_active_refreshes)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
