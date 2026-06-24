"""All SQLite access for crapper. `init_db()` creates the schema and wipes the
data tables if it detects an older (pre-multi-source) schema."""
import json
import os
import sqlite3
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

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_tl_tracker ON tracker_listings(tracker_id);
            CREATE INDEX IF NOT EXISTS idx_tl_listing ON tracker_listings(listing_id);
            CREATE INDEX IF NOT EXISTS idx_price_listing ON price_history(listing_id, observed_at);
        ''')
        _init_default_settings(conn)


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

def add_listing_tracker(source: str, label: str, url: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO trackers (source, type, label, url) VALUES (?, 'listing', ?, ?)",
            (source, label, url),
        )
        return cur.lastrowid


def add_search_tracker(source: str, label: str, params: dict) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO trackers (source, type, label, params) VALUES (?, 'search', ?, ?)",
            (source, label, json.dumps(params)),
        )
        return cur.lastrowid


def _row_to_tracker(row) -> dict:
    t = dict(row)
    t['params'] = json.loads(t['params']) if t.get('params') else {}
    return t


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
        row = conn.execute(
            "SELECT * FROM trackers WHERE type = 'listing' AND source = ? AND url = ? LIMIT 1",
            (source, url),
        ).fetchone()
        return _row_to_tracker(row) if row else None


def _search_signature(params: dict) -> dict:
    """The defining parameters of a search, ignoring display-only *_label keys."""
    return {k: v for k, v in params.items() if not k.endswith('_label')}


def find_search_tracker(source: str, params: dict) -> dict | None:
    """Return an existing search tracker on this source with identical defining
    parameters (display-only labels ignored)."""
    sig = _search_signature(params)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trackers WHERE type = 'search' AND source = ?", (source,)
        ).fetchall()
    for row in rows:
        t = _row_to_tracker(row)
        if _search_signature(t['params']) == sig:
            return t
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

def upsert_listing(source: str, ad_id: str, url: str, title: str | None,
                   location: str | None, image_url: str | None, seen_at: str) -> int:
    """Insert or refresh the listing for an ad (unique by source + ad_id). Bumps
    last_seen and marks it active. Returns the listing id. Tracker membership is
    recorded separately via link_listing()."""
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
                       url = ?
                   WHERE id = ?''',
                (seen_at, title, location, image_url, url, lid),
            )
            return lid
        cur = conn.execute(
            '''INSERT INTO listings
               (source, ad_id, url, title, location, image_url, first_seen, last_seen, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)''',
            (source, ad_id, url, title, location, image_url, seen_at, seen_at),
        )
        return cur.lastrowid


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


def get_all_listings() -> list[dict]:
    """All distinct listings, active-first, each annotated with the labels/types
    of every tracker that surfaced it."""
    with get_db() as conn:
        return [dict(r) for r in conn.execute('''
            SELECT l.*,
                   (SELECT GROUP_CONCAT(t.label, ' · ')
                      FROM tracker_listings tl JOIN trackers t ON t.id = tl.tracker_id
                     WHERE tl.listing_id = l.id)          AS tracker_labels,
                   (SELECT GROUP_CONCAT(DISTINCT t.type)
                      FROM tracker_listings tl JOIN trackers t ON t.id = tl.tracker_id
                     WHERE tl.listing_id = l.id)          AS tracker_types
            FROM listings l
            ORDER BY l.active DESC, l.id
        ''')]


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
