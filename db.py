"""All SQLite access for crapper. `init_db()` creates the schema and wipes the
data tables if it detects an older (pre-multi-source) schema."""
import hashlib
import json
import os
import sqlite3
import zlib
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'crapper.db')


# ── init / migration ──────────────────────────────────────────────────────────

def init_db():
    with get_db() as conn:
        existing = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

        # Wipe the data tables if the schema predates the multi-source model
        # (trackers.source is the marker). No backward compatibility is required.
        if 'trackers' in existing:
            cols = {r[1] for r in conn.execute('PRAGMA table_info(trackers)')}
            if 'source' not in cols:
                conn.executescript('''
                    DROP TABLE IF EXISTS price_history;
                    DROP TABLE IF EXISTS tracker_listings;
                    DROP TABLE IF EXISTS listings;
                    DROP TABLE IF EXISTS trackers;
                ''')

        conn.executescript('''
            -- A tracker is one thing the user asked us to watch on one source:
            --   type='listing'  → a single ad (url)
            --   type='search'   → a saved search; `params` is a source-specific
            --                      JSON blob (query/location/radius/objektart/…)
            CREATE TABLE IF NOT EXISTS trackers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source        TEXT NOT NULL,                 -- source slug, e.g. 'kleinanzeigen'
                type          TEXT NOT NULL,                 -- 'listing' | 'search'
                label         TEXT NOT NULL DEFAULT '',
                url           TEXT,                          -- listing type
                params        TEXT,                          -- search type (JSON)
                enabled       INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT DEFAULT (datetime('now'))
            );

            -- One row per distinct ad, unique by (source, ad_id), so the same ad
            -- surfaced by several trackers is stored (and shown) once.
            -- `active=0` means no referencing tracker found it in its latest poll.
            CREATE TABLE IF NOT EXISTS listings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                ad_id       TEXT NOT NULL,
                url         TEXT NOT NULL,
                title       TEXT,
                location    TEXT,
                image_url   TEXT,
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                UNIQUE(source, ad_id)
            );

            -- Which trackers surfaced which listing (many-to-many). Per-tracker
            -- last_seen/active drive deactivation; a listing is active iff at
            -- least one of its memberships is active.
            CREATE TABLE IF NOT EXISTS tracker_listings (
                tracker_id  INTEGER NOT NULL
                                REFERENCES trackers(id) ON DELETE CASCADE,
                listing_id  INTEGER NOT NULL
                                REFERENCES listings(id) ON DELETE CASCADE,
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (tracker_id, listing_id)
            );

            -- Immutable price archive attached to the (global) listing; one row
            -- appended per poll that returned a price. `price` is NULL when the ad
            -- has no numeric price (e.g. "VB"); `price_text` keeps the raw label.
            CREATE TABLE IF NOT EXISTS price_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id  INTEGER NOT NULL
                                REFERENCES listings(id) ON DELETE CASCADE,
                observed_at TEXT NOT NULL,
                price       REAL,
                price_text  TEXT
            );

            -- Append-only archive of the ad's content. A new row is written only
            -- when title/location/description/image changes (versioned per listing).
            -- raw_html is the original page, zlib-compressed (single-ad fetches only).
            CREATE TABLE IF NOT EXISTS listing_versions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id   INTEGER NOT NULL
                                 REFERENCES listings(id) ON DELETE CASCADE,
                version      INTEGER NOT NULL,
                fetched_at   TEXT NOT NULL,
                title        TEXT,
                location     TEXT,
                description  TEXT,
                image_url    TEXT,
                image_file   TEXT,
                images       TEXT,          -- JSON list of all image URLs
                content_hash TEXT NOT NULL,
                raw_html     BLOB,
                UNIQUE(listing_id, version)
            );

            -- Content-addressed page assets (css/fonts/images); files in archive/assets/<hash>.
            CREATE TABLE IF NOT EXISTS assets (
                hash         TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                size         INTEGER NOT NULL
            );
            -- Which asset an original URL resolved to when a page version was archived.
            CREATE TABLE IF NOT EXISTS version_assets (
                version_id INTEGER NOT NULL REFERENCES listing_versions(id) ON DELETE CASCADE,
                url        TEXT NOT NULL,
                hash       TEXT NOT NULL,
                PRIMARY KEY (version_id, url)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_tl_tracker ON tracker_listings(tracker_id);
            CREATE INDEX IF NOT EXISTS idx_tl_listing ON tracker_listings(listing_id);
            CREATE INDEX IF NOT EXISTS idx_price_listing ON price_history(listing_id, observed_at);
        ''')
        _add_columns_if_missing(conn, 'listings', {
            'hidden': 'INTEGER NOT NULL DEFAULT 0',   # hidden from the frontpage by the user
            'detail_at': 'TEXT',                      # last full-page archive (ISO UTC)
            'image_file': 'TEXT',                     # locally archived copy of image_url
        })
        _add_columns_if_missing(conn, 'listing_versions', {
            'assets_at': 'TEXT',                      # when this page's assets were archived
        })
        # Seed v1 for listings that predate versioning.
        conn.execute('''
            INSERT INTO listing_versions (listing_id, version, fetched_at, title, location,
                                          image_url, images, content_hash)
            SELECT id, 1, last_seen, title, location, image_url, '[]', 'legacy'
              FROM listings l
             WHERE NOT EXISTS (SELECT 1 FROM listing_versions v WHERE v.listing_id = l.id)''')
        _prune_phantom_versions(conn)
        _init_default_settings(conn)


def _prune_phantom_versions(conn):
    """Remove versions created by search cards after a full page was archived, and
    consecutive full-page versions with identical content (flip-flop leftovers)."""
    conn.execute('''DELETE FROM listing_versions
                    WHERE raw_html IS NULL AND EXISTS (
                      SELECT 1 FROM listing_versions o
                       WHERE o.listing_id = listing_versions.listing_id
                         AND o.raw_html IS NOT NULL AND o.version < listing_versions.version)''')
    prev = {}
    for r in conn.execute('''SELECT id, listing_id, content_hash FROM listing_versions
                             WHERE raw_html IS NOT NULL ORDER BY listing_id, version''').fetchall():
        if r['content_hash'] != 'legacy' and prev.get(r['listing_id']) == r['content_hash']:
            conn.execute('DELETE FROM listing_versions WHERE id = ?', (r['id'],))
        else:
            prev[r['listing_id']] = r['content_hash']


def _add_columns_if_missing(conn, table: str, columns: dict[str, str]):
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}
    for col, typedef in columns.items():
        if col not in existing:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {col} {typedef}')


def _init_default_settings(conn):
    defaults = {
        'poll_cron':    '0 4 * * *',   # daily at 04:00 server time
        'search_pages': '5',           # max search result pages to walk per poll
    }
    for key, val in defaults.items():
        conn.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, val))


# ── connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── settings ──────────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = '') -> str:
    with get_db() as conn:
        row = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
        return row[0] if row else default


def set_setting(key: str, value: str):
    with get_db() as conn:
        conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))


def get_all_settings() -> dict[str, str]:
    with get_db() as conn:
        return {r[0]: r[1] for r in conn.execute('SELECT key, value FROM settings')}


# ── trackers ──────────────────────────────────────────────────────────────────

def add_tracker(source: str, type_: str, label: str, url: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            'INSERT INTO trackers (source, type, label, url) VALUES (?, ?, ?, ?)',
            (source, type_, label, url),
        )
        return cur.lastrowid


def _row_to_tracker(row) -> dict:
    return dict(row)


def _normalize_url(url: str) -> str:
    return url.strip().rstrip('/').split('#')[0].lower()


def find_listing_tracker(source: str, ad_id: str | None, url: str) -> dict | None:
    """Return an existing listing tracker on this source for the same ad, matched
    by ad_id (preferred — the same ad can have different URL slugs) or exact url."""
    with get_db() as conn:
        if ad_id:
            row = conn.execute(
                '''SELECT t.* FROM trackers t
                   JOIN tracker_listings tl ON tl.tracker_id = t.id
                   JOIN listings l ON l.id = tl.listing_id
                   WHERE t.type = 'listing' AND t.source = ? AND l.ad_id = ? LIMIT 1''',
                (source, ad_id),
            ).fetchone()
            if row:
                return _row_to_tracker(row)
        norm = _normalize_url(url)
        for row in conn.execute(
            "SELECT * FROM trackers WHERE type = 'listing' AND source = ?", (source,)
        ):
            if _normalize_url(row['url'] or '') == norm:
                return _row_to_tracker(row)
        return None


def find_search_tracker(source: str, url: str) -> dict | None:
    """Return an existing search tracker on this source with the same (normalized) URL."""
    norm = _normalize_url(url)
    with get_db() as conn:
        for row in conn.execute(
            "SELECT * FROM trackers WHERE type = 'search' AND source = ?", (source,)
        ):
            if _normalize_url(row['url'] or '') == norm:
                return _row_to_tracker(row)
    return None


def get_tracker(tracker_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute('SELECT * FROM trackers WHERE id = ?', (tracker_id,)).fetchone()
        return _row_to_tracker(row) if row else None


def get_trackers(enabled_only: bool = False) -> list[dict]:
    clause = 'WHERE enabled = 1' if enabled_only else ''
    with get_db() as conn:
        return [_row_to_tracker(r) for r in conn.execute(
            f'SELECT * FROM trackers {clause} ORDER BY id'
        )]


def delete_tracker(tracker_id: int):
    """Delete a tracker (its memberships cascade), then drop any listing left
    with no tracker referencing it, and refresh active flags."""
    with get_db() as conn:
        conn.execute('DELETE FROM trackers WHERE id = ?', (tracker_id,))
        conn.execute(
            'DELETE FROM listings WHERE id NOT IN '
            '(SELECT listing_id FROM tracker_listings)'
        )
        _recompute_active(conn)


# ── listings ──────────────────────────────────────────────────────────────────

def _clean(v: str | None) -> str | None:
    v = (v or '').strip()
    return v or None


def _clean_img(v: str | None) -> str | None:
    v = _clean(v)
    return v if v and v.startswith(('http://', 'https://')) else None


def upsert_listing(source: str, ad_id: str, url: str, title: str | None,
                   location: str | None, image_url: str | None, seen_at: str,
                   description: str | None = None, images: list[str] | None = None,
                   raw_html: str | None = None) -> int:
    """Insert or refresh the listing for an ad (unique by source + ad_id). Bumps
    last_seen and marks it active. Empty/placeholder values never overwrite good
    stored ones. Records a new archive version if the content changed. Returns the
    listing id. Tracker membership is recorded separately via link_listing()."""
    title, location, image_url = _clean(title), _clean(location), _clean_img(image_url)
    with get_db() as conn:
        row = conn.execute(
            'SELECT id FROM listings WHERE source = ? AND ad_id = ?', (source, ad_id)
        ).fetchone()
        if row:
            lid = row[0]
            conn.execute(
                '''UPDATE listings
                   SET last_seen = ?, active = 1,
                       title = COALESCE(?, title),
                       location = COALESCE(?, location),
                       image_url = COALESCE(?, image_url),
                       image_file = CASE WHEN ? IS NOT NULL AND ? IS NOT image_url
                                         THEN NULL ELSE image_file END,
                       url = ?
                   WHERE id = ?''',
                (seen_at, title, location, image_url, image_url, image_url, url, lid),
            )
        else:
            lid = conn.execute(
                '''INSERT INTO listings
                   (source, ad_id, url, title, location, image_url, first_seen, last_seen, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)''',
                (source, ad_id, url, title, location, image_url, seen_at, seen_at),
            ).lastrowid
        _record_version(conn, lid, seen_at, title, location, _clean(description),
                        image_url, images, raw_html)
        return lid


def _record_version(conn, lid, seen_at, title, location, description, image_url,
                    images, raw_html):
    """Append a version when the content differs from the latest one. Fields the
    scrape did not provide (e.g. description from a search card) are carried over
    from the previous version so partial scrapes don't create phantom changes."""
    last = conn.execute(
        'SELECT * FROM listing_versions WHERE listing_id = ? ORDER BY version DESC LIMIT 1',
        (lid,)).fetchone()
    if last:
        title = title or last['title']
        location = location or last['location']
        description = description or last['description']
        image_url = image_url or last['image_url']
    images = [u for u in (images or []) if _clean_img(u)]
    if not raw_html and conn.execute(
            'SELECT 1 FROM listing_versions WHERE listing_id = ? AND raw_html IS NOT NULL',
            (lid,)).fetchone():
        return     # a search card never makes a version once a full page is archived: its
                   # fields differ slightly from the full page and would flip-flop
    h = hashlib.sha256(json.dumps(
        [title, location, description, image_url], ensure_ascii=False).encode()).hexdigest()
    last_hash = last['content_hash'] if last else None
    if last_hash == 'legacy':      # seeded pre-versioning row: hash its fields now
        last_hash = hashlib.sha256(json.dumps(
            [last['title'], last['location'], last['description'], last['image_url']],
            ensure_ascii=False).encode()).hexdigest()
    if last and last_hash == h:
        # Same content; still keep a first original page if we only had a card before.
        if raw_html and not last['raw_html']:
            conn.execute('UPDATE listing_versions SET raw_html = ?, images = ? WHERE id = ?',
                         (zlib.compress(raw_html.encode()), json.dumps(images), last['id']))
        return
    conn.execute(
        '''INSERT INTO listing_versions
           (listing_id, version, fetched_at, title, location, description, image_url,
            image_file, images, content_hash, raw_html)
           VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)''',
        (lid, (last['version'] + 1) if last else 1, seen_at, title, location, description,
         image_url, json.dumps(images), h,
         zlib.compress(raw_html.encode()) if raw_html else None))


def listing_needs_image(listing_id: int) -> str | None:
    """image_url of the listing if it has no archived local copy yet, else None."""
    with get_db() as conn:
        r = conn.execute('SELECT image_url, image_file FROM listings WHERE id = ?',
                         (listing_id,)).fetchone()
    return r['image_url'] if r and r['image_url'] and not r['image_file'] else None


def set_image_file(listing_id: int, image_url: str, image_file: str):
    with get_db() as conn:
        conn.execute('UPDATE listings SET image_file = ? WHERE id = ? AND image_url = ?',
                     (image_file, listing_id, image_url))
        conn.execute(
            '''UPDATE listing_versions SET image_file = ?
               WHERE image_url = ? AND listing_id = ? AND image_file IS NULL''',
            (image_file, image_url, listing_id))


def get_versions(listing_id: int) -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            '''SELECT version, fetched_at, title, location, description, image_url,
                      image_file, raw_html IS NOT NULL AS has_html
               FROM listing_versions WHERE listing_id = ? ORDER BY version DESC''',
            (listing_id,))]


def get_listing(listing_id: int) -> dict | None:
    with get_db() as conn:
        r = conn.execute('SELECT * FROM listings WHERE id = ?', (listing_id,)).fetchone()
    return dict(r) if r else None


def get_version_images(listing_id: int, version: int) -> list[str]:
    with get_db() as conn:
        r = conn.execute('SELECT images FROM listing_versions WHERE listing_id = ? '
                         'AND version = ?', (listing_id, version)).fetchone()
    try:
        return json.loads(r[0]) if r and r[0] else []
    except ValueError:
        return []


def put_asset(h: str, content_type: str, size: int):
    with get_db() as conn:
        conn.execute('INSERT OR IGNORE INTO assets (hash, content_type, size) VALUES (?, ?, ?)',
                     (h, content_type, size))


def get_asset(h: str) -> dict | None:
    with get_db() as conn:
        r = conn.execute('SELECT * FROM assets WHERE hash = ?', (h,)).fetchone()
    return dict(r) if r else None


def save_version_assets(version_id: int, mapping: dict[str, str]):
    with get_db() as conn:
        conn.executemany(
            'INSERT OR REPLACE INTO version_assets (version_id, url, hash) VALUES (?, ?, ?)',
            [(version_id, u, h) for u, h in mapping.items()])


def mark_assets_done(version_id: int):
    with get_db() as conn:
        conn.execute('UPDATE listing_versions SET assets_at = ? WHERE id = ?',
                     (datetime.utcnow().isoformat(), version_id))


def versions_pending_assets():
    """Yield stored page versions whose assets have not been archived yet, newest
    first (loaded one at a time: the pages are big)."""
    with get_db() as conn:
        ids = [r[0] for r in conn.execute(
            'SELECT id FROM listing_versions WHERE raw_html IS NOT NULL AND assets_at IS NULL '
            'ORDER BY fetched_at DESC')]
    for vid in ids:
        with get_db() as conn:
            r = conn.execute(
                '''SELECT v.id, v.listing_id, v.version, v.raw_html, l.url
                   FROM listing_versions v JOIN listings l ON l.id = v.listing_id
                   WHERE v.id = ?''', (vid,)).fetchone()
        if r and r['raw_html']:
            yield {'id': r['id'], 'listing_id': r['listing_id'], 'version': r['version'],
                   'url': r['url'], 'html': zlib.decompress(r['raw_html']).decode()}


def get_version_assets(listing_id: int, version: int) -> dict[str, str]:
    with get_db() as conn:
        return {r['url']: r['hash'] for r in conn.execute(
            '''SELECT a.url, a.hash FROM version_assets a
               JOIN listing_versions v ON v.id = a.version_id
               WHERE v.listing_id = ? AND v.version = ?''', (listing_id, version))}


def get_version_html(listing_id: int, version: int) -> str | None:
    with get_db() as conn:
        r = conn.execute('SELECT raw_html FROM listing_versions WHERE listing_id = ? '
                         'AND version = ?', (listing_id, version)).fetchone()
    return zlib.decompress(r[0]).decode() if r and r[0] else None


def link_listing(tracker_id: int, listing_id: int, seen_at: str):
    """Record (or refresh) that this tracker surfaced this listing in this poll."""
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO tracker_listings (tracker_id, listing_id, first_seen, last_seen, active)
               VALUES (?, ?, ?, ?, 1)
               ON CONFLICT(tracker_id, listing_id)
               DO UPDATE SET last_seen = excluded.last_seen, active = 1''',
            (tracker_id, listing_id, seen_at, seen_at),
        )


def deactivate_missing(tracker_id: int, seen_at: str):
    """For one tracker, flag memberships not refreshed in this poll as inactive,
    then recompute each listing's global active flag (active iff any membership is)."""
    with get_db() as conn:
        conn.execute(
            'UPDATE tracker_listings SET active = 0 WHERE tracker_id = ? AND last_seen < ?',
            (tracker_id, seen_at),
        )
        _recompute_active(conn)


def mark_tracker_listings_inactive(tracker_id: int):
    """Flag all of a tracker's memberships inactive (e.g. its single ad is gone),
    then recompute global active flags."""
    with get_db() as conn:
        conn.execute(
            'UPDATE tracker_listings SET active = 0 WHERE tracker_id = ?', (tracker_id,))
        _recompute_active(conn)


def _recompute_active(conn):
    """A listing is active iff at least one tracker membership is active."""
    conn.execute('''
        UPDATE listings SET active = COALESCE(
            (SELECT MAX(active) FROM tracker_listings WHERE listing_id = listings.id), 0)
    ''')


def get_tracker_listing_counts() -> dict[int, int]:
    with get_db() as conn:
        return {r[0]: r[1] for r in conn.execute(
            'SELECT tracker_id, COUNT(*) FROM tracker_listings GROUP BY tracker_id')}


def get_tracker_memberships() -> dict[int, list[int]]:
    """tracker_id → list of its listing_ids (which trackers surfaced which listing)."""
    with get_db() as conn:
        rows = conn.execute(
            'SELECT tracker_id, listing_id FROM tracker_listings ORDER BY tracker_id, listing_id'
        ).fetchall()
    out: dict[int, list[int]] = {}
    for r in rows:
        out.setdefault(r[0], []).append(r[1])
    return out


_LISTING_SELECT = '''
    SELECT l.*,
           (SELECT GROUP_CONCAT(t.label, ' · ')
              FROM tracker_listings tl JOIN trackers t ON t.id = tl.tracker_id
             WHERE tl.listing_id = l.id)          AS tracker_labels,
           (SELECT GROUP_CONCAT(DISTINCT t.type)
              FROM tracker_listings tl JOIN trackers t ON t.id = tl.tracker_id
             WHERE tl.listing_id = l.id)          AS tracker_types
    FROM listings l
'''


def get_all_listings() -> list[dict]:
    """Visible (non-hidden) distinct listings, active-first, each annotated with
    the labels/types of every tracker that surfaced it."""
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            _LISTING_SELECT + ' WHERE COALESCE(l.hidden, 0) = 0 ORDER BY l.active DESC, l.id')]


def get_hidden_listings() -> list[dict]:
    """Listings the user has hidden from the frontpage (for the manage panel)."""
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            _LISTING_SELECT + ' WHERE l.hidden = 1 ORDER BY l.id')]


def set_listing_hidden(listing_id: int, hidden: bool):
    with get_db() as conn:
        conn.execute('UPDATE listings SET hidden = ? WHERE id = ?',
                     (1 if hidden else 0, listing_id))


# ── price history ─────────────────────────────────────────────────────────────

def add_price(listing_id: int, observed_at: str, price: float | None, price_text: str | None):
    """Append a price observation, skipping consecutive duplicates so the chart
    stays a clean step function instead of one point per poll."""
    with get_db() as conn:
        last = conn.execute(
            'SELECT price, price_text FROM price_history WHERE listing_id = ? '
            'ORDER BY observed_at DESC LIMIT 1',
            (listing_id,),
        ).fetchone()
        if last and last[0] == price and last[1] == price_text:
            return
        conn.execute(
            'INSERT INTO price_history (listing_id, observed_at, price, price_text) '
            'VALUES (?, ?, ?, ?)',
            (listing_id, observed_at, price, price_text),
        )


def get_price_history(listing_id: int) -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT observed_at, price, price_text FROM price_history '
            'WHERE listing_id = ? ORDER BY observed_at',
            (listing_id,),
        )]


def get_price_history_map() -> dict[int, list[dict]]:
    """All price history grouped by listing_id, chronological."""
    with get_db() as conn:
        rows = conn.execute(
            'SELECT listing_id, observed_at, price, price_text FROM price_history '
            'ORDER BY listing_id, observed_at'
        ).fetchall()
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r[0], []).append(
            {'observed_at': r[1], 'price': r[2], 'price_text': r[3]}
        )
    return out


def needs_detail(listing_id: int, older_than: str) -> bool:
    """True if the full ad page was never archived or not since *older_than*."""
    with get_db() as conn:
        r = conn.execute('SELECT detail_at FROM listings WHERE id = ?', (listing_id,)).fetchone()
    return bool(r) and (not r[0] or r[0] < older_than)


def mark_detail(listing_id: int, at: str):
    with get_db() as conn:
        conn.execute('UPDATE listings SET detail_at = ? WHERE id = ?', (at, listing_id))
