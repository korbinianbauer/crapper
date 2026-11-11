"""All SQLite access for crapper. `init_db()` creates the schema and wipes the
data tables if it detects an older (pre-multi-source) schema."""
import hashlib
import json
import os
import sqlite3
import zlib
from datetime import datetime, timedelta
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

            -- One row per search-tracker poll: how many ads were found and how many
            -- carried a price in the search results (`priced`) or were rescued from
            -- the ad's detail page (`rescued`). Detects price-extraction outages.
            CREATE TABLE IF NOT EXISTS poll_stats (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tracker_id INTEGER NOT NULL
                               REFERENCES trackers(id) ON DELETE CASCADE,
                polled_at  TEXT NOT NULL,
                found      INTEGER NOT NULL,
                priced     INTEGER NOT NULL,
                rescued    INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_stats_tracker ON poll_stats(tracker_id, polled_at);

            -- ML "fair price": a feature set defines up to 30 named features (JSON list of
            -- {key,label,type}) for the listings of one tracker; values are extracted
            -- manually by an LLM agent and stored per listing; models are trained from them.
            CREATE TABLE IF NOT EXISTS feature_sets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                tracker_id  INTEGER REFERENCES trackers(id) ON DELETE SET NULL,
                definition  TEXT NOT NULL,        -- JSON: {"features": [...], "target": "price"}
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS listing_features (
                listing_id     INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
                feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
                data           TEXT NOT NULL,     -- JSON {feature_key: value|null}
                source_version INTEGER,           -- listing_versions.version it was read from
                extracted_at   TEXT NOT NULL,
                PRIMARY KEY (listing_id, feature_set_id)
            );
            CREATE TABLE IF NOT EXISTS ml_models (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
                algo           TEXT NOT NULL,
                trained_at     TEXT NOT NULL,
                n_train        INTEGER NOT NULL,
                metrics        TEXT NOT NULL,     -- JSON: CV scores of all candidates + chosen one
                artifact       BLOB NOT NULL      -- joblib (encoder + fitted estimator)
            );
            -- Prediction per listing and model, always made without the model having seen
            -- the listing's price: out-of-fold for training listings, the stored model for new ones.
            CREATE TABLE IF NOT EXISTS fair_prices (
                listing_id   INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
                model_id     INTEGER NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
                fair_price   REAL NOT NULL,
                oof          INTEGER NOT NULL DEFAULT 0,
                predicted_at TEXT NOT NULL,
                PRIMARY KEY (listing_id, model_id)
            );

            -- results of each feature set's ACTIVE model per listing (fair price, heating energy,
            -- heating/operating costs …) as JSON; rewritten whenever the active model changes
            CREATE TABLE IF NOT EXISTS listing_predictions (
                listing_id     INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
                feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
                model_id       INTEGER NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
                data           TEXT NOT NULL,
                predicted_at   TEXT NOT NULL,
                PRIMARY KEY (listing_id, feature_set_id)
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
            'detail_tried_at': 'TEXT',                # last full-page fetch attempt, ok or not
            'image_file': 'TEXT',                     # locally archived copy of image_url
        })
        _add_columns_if_missing(conn, 'listing_versions', {
            'assets_at': 'TEXT',                      # when this page's assets were archived
        })
        if 'position' not in {r[1] for r in conn.execute('PRAGMA table_info(trackers)')}:
            conn.execute('ALTER TABLE trackers ADD COLUMN position INTEGER NOT NULL DEFAULT 0')
            conn.execute('UPDATE trackers SET position = id')   # keep the existing order
        # Groups: a tracker row with type='group' (never polled); its members point to it via parent_id.
        _add_columns_if_missing(conn, 'trackers', {
            'parent_id': 'INTEGER REFERENCES trackers(id) ON DELETE SET NULL',
        })
        # Only gradient boosting is supported: drop models of any other algorithm (+ their predictions).
        conn.execute("DELETE FROM fair_prices WHERE model_id IN "
                     "(SELECT id FROM ml_models WHERE algo != 'gradient_boosting')")
        conn.execute("DELETE FROM ml_models WHERE algo != 'gradient_boosting'")
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
        'search_pages': '50',          # max search result pages to walk per poll
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


def update_setting_atomic(key: str, fn):
    """Read-modify-write of one setting under SQLite's write lock (BEGIN IMMEDIATE),
    so concurrent threads *and* processes (web server, scheduled poll) serialise.
    *fn(old_value)* returns (new_value or None to leave it, result); returns result."""
    conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level=None)
    try:
        conn.execute('BEGIN IMMEDIATE')
        r = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
        new, result = fn(r[0] if r else '')
        if new is not None:
            conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, new))
        conn.execute('COMMIT')
        return result
    except Exception:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise
    finally:
        conn.close()


def get_all_settings() -> dict[str, str]:
    with get_db() as conn:
        return {r[0]: r[1] for r in conn.execute('SELECT key, value FROM settings')}


# ── trackers ──────────────────────────────────────────────────────────────────

def add_tracker(source: str, type_: str, label: str, url: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            'INSERT INTO trackers (source, type, label, url, position) '
            'VALUES (?, ?, ?, ?, (SELECT COALESCE(MAX(position), 0) + 1 FROM trackers))',
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


def get_trackers(enabled_only: bool = False, groups: bool = True) -> list[dict]:
    """All trackers. `enabled_only` is the polling view and never includes groups
    (they have nothing to poll); `groups=False` leaves them out otherwise."""
    conds = []
    if enabled_only:
        conds.append('enabled = 1')
    if enabled_only or not groups:
        conds.append("type != 'group'")
    clause = 'WHERE ' + ' AND '.join(conds) if conds else ''
    with get_db() as conn:
        return [_row_to_tracker(r) for r in conn.execute(
            f'SELECT * FROM trackers {clause} ORDER BY position, id'
        )]


def set_tracker_enabled(tracker_id: int, enabled: bool):
    """Pause (enabled=False) / resume a tracker. Paused trackers are not polled and
    are shown collapsed at the bottom of the frontpage; all data is kept."""
    with get_db() as conn:
        conn.execute('UPDATE trackers SET enabled = ? WHERE id = ?',
                     (1 if enabled else 0, tracker_id))


def reorder_trackers(ids: list[int]):
    """Put `ids` into the given order. They are redistributed over the positions they
    occupied before, so trackers not in `ids` (e.g. the paused group) keep their place."""
    with get_db() as conn:
        marks = ','.join('?' * len(ids))
        rows = conn.execute(
            f'SELECT id, position FROM trackers WHERE id IN ({marks})', ids).fetchall()
        known = {r['id'] for r in rows}
        ids = [i for i in ids if i in known]
        slots = sorted(r['position'] for r in rows)
        # equal positions would make the order ambiguous → fall back to a dense range
        if len(set(slots)) != len(slots):
            slots = list(range(slots[0], slots[0] + len(slots)))
        for tid, pos in zip(ids, slots):
            conn.execute('UPDATE trackers SET position = ? WHERE id = ?', (pos, tid))


def create_group(label: str, member_ids: list[int]) -> int | None:
    """Merge the given (non-group, ungrouped or grouped) trackers into a new group.
    Returns the group id, or None if fewer than 2 valid trackers were given."""
    with get_db() as conn:
        marks = ','.join('?' * len(member_ids))
        ids = [r['id'] for r in conn.execute(
            f"SELECT id FROM trackers WHERE type != 'group' AND id IN ({marks})", member_ids)]
        if len(ids) < 2:
            return None
        cur = conn.execute(
            "INSERT INTO trackers (source, type, label, position) "
            "VALUES ('', 'group', ?, (SELECT COALESCE(MAX(position), 0) + 1 FROM trackers))",
            (label.strip() or 'Gruppe',))
        gid = cur.lastrowid
        conn.execute(f'UPDATE trackers SET parent_id = ? WHERE id IN ({",".join("?" * len(ids))})',
                     [gid, *ids])
        _drop_empty_groups(conn)
        return gid


def set_tracker_group(tracker_id: int, group_id: int | None) -> bool:
    """Put a tracker into an existing group (None = take it out)."""
    with get_db() as conn:
        if group_id is not None and not conn.execute(
                "SELECT 1 FROM trackers WHERE id = ? AND type = 'group'", (group_id,)).fetchone():
            return False
        conn.execute("UPDATE trackers SET parent_id = ? WHERE id = ? AND type != 'group'",
                     (group_id, tracker_id))
        _drop_empty_groups(conn)
        return True


def _drop_empty_groups(conn):
    conn.execute("DELETE FROM trackers WHERE type = 'group' AND id NOT IN "
                 "(SELECT parent_id FROM trackers WHERE parent_id IS NOT NULL)")


def delete_tracker(tracker_id: int):
    """Delete a tracker (its memberships cascade), then drop any listing left
    with no tracker referencing it, and refresh active flags. Deleting a group
    only dissolves it: its members stay."""
    with get_db() as conn:
        conn.execute('DELETE FROM trackers WHERE id = ?', (tracker_id,))
        _drop_empty_groups(conn)
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


# ── ML fair price ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec='seconds')


def upsert_feature_set(name: str, tracker_id: int | None, definition: dict) -> int:
    import json
    with get_db() as conn:
        conn.execute('''INSERT INTO feature_sets (name, tracker_id, definition) VALUES (?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET tracker_id = excluded.tracker_id,
                                                        definition = excluded.definition''',
                     (name, tracker_id, json.dumps(definition, ensure_ascii=False)))
        return conn.execute('SELECT id FROM feature_sets WHERE name = ?', (name,)).fetchone()[0]


def get_feature_set(name: str) -> dict | None:
    import json
    with get_db() as conn:
        r = conn.execute('SELECT * FROM feature_sets WHERE name = ?', (name,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d['definition'] = json.loads(d['definition'])
    return d


def save_listing_features(feature_set_id: int, rows: dict[int, dict]) -> int:
    """rows: {listing_id: {feature_key: value}}. Replaces existing values. Returns #stored."""
    import json
    now = _now()
    n = 0
    with get_db() as conn:
        for lid, vals in rows.items():
            if not conn.execute('SELECT 1 FROM listings WHERE id = ?', (lid,)).fetchone():
                continue
            ver = conn.execute('SELECT MAX(version) FROM listing_versions WHERE listing_id = ?',
                               (lid,)).fetchone()[0]
            conn.execute('''INSERT OR REPLACE INTO listing_features
                            (listing_id, feature_set_id, data, source_version, extracted_at)
                            VALUES (?, ?, ?, ?, ?)''',
                         (lid, feature_set_id, json.dumps(vals, ensure_ascii=False), ver, now))
            n += 1
    return n


def get_training_rows(feature_set_id: int) -> list[dict]:
    """[{listing_id, features: {...}, price}] — price = latest numeric observation or None."""
    import json
    with get_db() as conn:
        rows = conn.execute('''
            SELECT lf.listing_id, lf.data,
                   (SELECT price FROM price_history WHERE listing_id = lf.listing_id
                      AND price IS NOT NULL ORDER BY observed_at DESC, id DESC LIMIT 1) AS price
            FROM listing_features lf WHERE lf.feature_set_id = ?''', (feature_set_id,)).fetchall()
    out = []
    for r in rows:
        data = json.loads(r['data'])
        out.append({'listing_id': r['listing_id'], 'price': r['price'], 'cost': data.get('_cost'),
                    'features': {k: v for k, v in data.items() if not k.startswith('_')}})
    return out


def get_heat_rows() -> list[dict]:
    """[{listing_id, features, cost}] of every listing in a feature set that has an `energieklasse`
    feature (flats, rent and sale) — training data of the heating-energy sub-models (ml.fit_cost_models)."""
    import json
    with get_db() as conn:
        sets = [r['id'] for r in conn.execute('SELECT id, definition FROM feature_sets')
                if any(f['key'] == 'energieklasse' for f in json.loads(r['definition'])['features'])]
        rows = conn.execute(f'''SELECT listing_id, data FROM listing_features
                               WHERE feature_set_id IN ({",".join("?" * len(sets))})''', sets).fetchall() if sets else []
    out = {}
    for r in rows:
        data = json.loads(r['data'])
        out.setdefault(r['listing_id'], {'listing_id': r['listing_id'], 'cost': data.get('_cost'),
                                         'features': {k: v for k, v in data.items() if not k.startswith('_')}})
    return list(out.values())


def get_cost_assumptions(feature_set_id: int) -> dict:
    """Monthly € assumptions used to normalise rents (see ml.effective_price)."""
    import ml
    r = get_setting(f'fair_cost_{feature_set_id}', '')
    try:
        return {**ml.DEFAULT_COSTS, **json.loads(r)} if r else dict(ml.DEFAULT_COSTS)
    except ValueError:
        return dict(ml.DEFAULT_COSTS)


def set_cost_assumptions(feature_set_id: int, values: dict):
    set_setting(f'fair_cost_{feature_set_id}', json.dumps(values))


KEEP_MODELS = 10          # trained models kept per feature set (the active one is never pruned)


def save_model(feature_set_id: int, algo: str, n_train: int, metrics: dict, artifact: bytes,
               predictions: dict[int, float], activate: bool = True) -> int:
    """Store a trained model and its predictions {listing_id: raw fair_price}. Older models
    stay selectable (settings page); only the oldest beyond KEEP_MODELS are pruned."""
    now = _now()
    with get_db() as conn:
        if not activate:    # keep the current model (default = newest) from being displaced
            cur_active = _active_model_id(conn, feature_set_id)
            if cur_active:
                conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
                             (f'fair_model_{feature_set_id}', str(cur_active)))
        cur = conn.execute('''INSERT INTO ml_models (feature_set_id, algo, trained_at, n_train,
                                                     metrics, artifact) VALUES (?, ?, ?, ?, ?, ?)''',
                           (feature_set_id, algo, now, n_train, json.dumps(metrics), artifact))
        mid = cur.lastrowid
        conn.executemany('''INSERT INTO fair_prices (listing_id, model_id, fair_price, oof, predicted_at)
                            VALUES (?, ?, ?, 1, ?)''',
                         [(lid, mid, p, now) for lid, p in predictions.items()])
        if activate:
            conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
                         (f'fair_model_{feature_set_id}', str(mid)))
        active = _active_model_id(conn, feature_set_id)
        conn.execute('''DELETE FROM ml_models WHERE feature_set_id = ? AND id != ? AND id NOT IN (
                          SELECT id FROM ml_models WHERE feature_set_id = ? ORDER BY id DESC LIMIT ?)''',
                     (feature_set_id, active, feature_set_id, KEEP_MODELS))
    return mid


def _active_model_id(conn, feature_set_id: int) -> int | None:
    """The model used for display: the one chosen on the settings page, else the newest."""
    r = conn.execute('SELECT value FROM settings WHERE key = ?',
                     (f'fair_model_{feature_set_id}',)).fetchone()
    if r and r[0].isdigit() and conn.execute(
            'SELECT 1 FROM ml_models WHERE id = ? AND feature_set_id = ?',
            (int(r[0]), feature_set_id)).fetchone():
        return int(r[0])
    r = conn.execute('SELECT MAX(id) FROM ml_models WHERE feature_set_id = ?',
                     (feature_set_id,)).fetchone()
    return r[0]


def _fair_offset(conn, feature_set_id: int) -> float:
    r = conn.execute('SELECT value FROM settings WHERE key = ?',
                     (f'fair_offset_{feature_set_id}',)).fetchone()
    try:
        return float(r[0]) if r else 0.0
    except ValueError:
        return 0.0


def get_fair_offset(feature_set_id: int) -> float:
    """Manual correction of the displayed fair price in percent (-10 = show 10 % less than
    the ML prediction), to bridge asking prices vs. actual deals."""
    with get_db() as conn:
        return _fair_offset(conn, feature_set_id)


def set_fair_offset(feature_set_id: int, pct: float):
    set_setting(f'fair_offset_{feature_set_id}', repr(float(pct)))


def delete_model(feature_set_id: int, model_id: int) -> bool:
    """Delete a model and its predictions; if it was the chosen one the newest remaining takes over."""
    with get_db() as conn:
        if not conn.execute('SELECT 1 FROM ml_models WHERE id = ? AND feature_set_id = ?',
                            (model_id, feature_set_id)).fetchone():
            return False
        conn.execute('DELETE FROM fair_prices WHERE model_id = ?', (model_id,))
        conn.execute('DELETE FROM ml_models WHERE id = ?', (model_id,))
        conn.execute('DELETE FROM settings WHERE key = ? AND value = ?',
                     (f'fair_model_{feature_set_id}', str(model_id)))
    return True


def set_active_model(feature_set_id: int, model_id: int) -> bool:
    with get_db() as conn:
        if not conn.execute('SELECT 1 FROM ml_models WHERE id = ? AND feature_set_id = ?',
                            (model_id, feature_set_id)).fetchone():
            return False
        conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
                     (f'fair_model_{feature_set_id}', str(model_id)))
    return True


def get_latest_model(feature_set_id: int) -> dict | None:
    """The active model (see _active_model_id) incl. its artifact."""
    with get_db() as conn:
        mid = _active_model_id(conn, feature_set_id)
        r = conn.execute('SELECT id, algo, artifact FROM ml_models WHERE id = ?',
                         (mid,)).fetchone() if mid else None
    return dict(r) if r else None


def get_model(model_id: int) -> dict | None:
    with get_db() as conn:
        r = conn.execute('SELECT id, feature_set_id, algo, artifact FROM ml_models WHERE id = ?',
                         (model_id,)).fetchone()
    return dict(r) if r else None


def get_feature_set_by_id(feature_set_id: int) -> dict | None:
    with get_db() as conn:
        r = conn.execute('SELECT id, name, tracker_id FROM feature_sets WHERE id = ?',
                         (feature_set_id,)).fetchone()
    return dict(r) if r else None


def get_feature_sets_overview() -> list[dict]:
    """Every feature set with its tracker, #listings with features, models (newest first),
    the active model id and the display offset — for the settings page."""
    out = []
    with get_db() as conn:
        for fs in conn.execute('''SELECT f.id, f.name, f.tracker_id, f.definition, t.label, t.url, t.source
                                  FROM feature_sets f LEFT JOIN trackers t ON t.id = f.tracker_id
                                  ORDER BY f.id''').fetchall():
            d = dict(fs)
            d['definition'] = json.loads(d['definition'])
            d['n_features'] = conn.execute(
                'SELECT COUNT(*) FROM listing_features WHERE feature_set_id = ?', (fs['id'],)).fetchone()[0]
            d['models'] = []
            for m in conn.execute('''SELECT id, algo, trained_at, n_train, metrics FROM ml_models
                                     WHERE feature_set_id = ? ORDER BY id DESC''', (fs['id'],)):
                md = dict(m)
                md['metrics'] = json.loads(md['metrics'])
                d['models'].append(md)
            d['active_model_id'] = _active_model_id(conn, fs['id'])
            d['offset'] = _fair_offset(conn, fs['id'])
            out.append(d)
    return out


def save_predictions(model_id: int, predictions: dict[int, float]) -> None:
    """Predictions of an existing model for listings it was not trained on."""
    now = _now()
    with get_db() as conn:
        conn.executemany('''INSERT OR REPLACE INTO fair_prices
                            (listing_id, model_id, fair_price, oof, predicted_at) VALUES (?, ?, ?, 1, ?)''',
                         [(lid, model_id, p, now) for lid, p in predictions.items()])


def save_listing_predictions(feature_set_id: int, model_id: int | None, rows: dict[int, dict],
                             replace_all: bool = False) -> None:
    """Store the active model's results {listing_id: {...}}; `replace_all` first drops every stored
    result of the feature set (active model changed / none left)."""
    now = _now()
    with get_db() as conn:
        if replace_all:
            conn.execute('DELETE FROM listing_predictions WHERE feature_set_id = ?', (feature_set_id,))
        if model_id:
            conn.executemany('''INSERT OR REPLACE INTO listing_predictions
                                (listing_id, feature_set_id, model_id, data, predicted_at) VALUES (?, ?, ?, ?, ?)''',
                             [(lid, feature_set_id, model_id, json.dumps(d, ensure_ascii=False), now)
                              for lid, d in rows.items()])


def get_listing_predictions(listing_id: int) -> list[dict]:
    """Stored model results of a listing: [{feature_set_id, model_id, data, predicted_at, trained_at}]."""
    with get_db() as conn:
        rows = conn.execute('''SELECT lp.feature_set_id, lp.model_id, lp.data, lp.predicted_at, m.trained_at
                               FROM listing_predictions lp JOIN ml_models m ON m.id = lp.model_id
                               WHERE lp.listing_id = ?''', (listing_id,)).fetchall()
    return [dict(r, data=json.loads(r['data'])) for r in rows]


def active_model_id(feature_set_id: int) -> int | None:
    with get_db() as conn:
        return _active_model_id(conn, feature_set_id)


def get_fair_price_map() -> dict[int, dict]:
    """{listing_id: {fair_price, raw, features, per}} from each feature set's active model.
    `fair_price` includes the manual offset, `raw` is the pure ML prediction."""
    out = {}
    with get_db() as conn:
        for fs in conn.execute('SELECT id FROM feature_sets').fetchall():
            mid = _active_model_id(conn, fs['id'])
            if not mid:
                continue
            factor = 1 + _fair_offset(conn, fs['id']) / 100
            for r in conn.execute('''
                    SELECT fp.listing_id, fp.fair_price, lf.data, m.metrics, m.id AS model_id, m.trained_at
                    FROM fair_prices fp
                    JOIN ml_models m ON m.id = fp.model_id
                    LEFT JOIN listing_features lf ON lf.listing_id = fp.listing_id
                                                 AND lf.feature_set_id = m.feature_set_id
                    WHERE m.id = ?''', (mid,)):
                data = json.loads(r['data']) if r['data'] else {}
                met = json.loads(r['metrics'])
                out[r['listing_id']] = {'fair_price': r['fair_price'] * factor, 'raw': r['fair_price'],
                                        'features': {k: v for k, v in data.items() if not k.startswith('_')},
                                        'cost': data.get('_cost'), 'assume': met.get('cost_assumptions'),
                                        'model_id': r['model_id'], 'trained_at': r['trained_at'],
                                        'per': met.get('per'), 'per_range': met.get('per_range')}
    return out


def get_fair_price(listing_id: int) -> dict | None:
    with get_db() as conn:
        rows = conn.execute('''
            SELECT fp.fair_price, fp.oof, fp.predicted_at, m.id AS model_id, m.algo, m.trained_at,
                   m.n_train, m.metrics, fs.id AS fs_id, fs.name AS feature_set,
                   lf.data AS features, fs.definition
            FROM fair_prices fp
            JOIN ml_models m ON m.id = fp.model_id
            JOIN feature_sets fs ON fs.id = m.feature_set_id
            LEFT JOIN listing_features lf ON lf.listing_id = fp.listing_id
                                         AND lf.feature_set_id = fs.id
            WHERE fp.listing_id = ?''', (listing_id,)).fetchall()
        r = next((x for x in rows if x['model_id'] == _active_model_id(conn, x['fs_id'])), None)
        offset = _fair_offset(conn, r['fs_id']) if r else 0.0
    if not r:
        return None
    d = dict(r)
    d['metrics'] = json.loads(d['metrics'])
    data = json.loads(d['features']) if d['features'] else {}
    d['features'] = {k: v for k, v in data.items() if not k.startswith('_')}
    d['cost'] = data.get('_cost')
    d['definition'] = json.loads(d['definition'])
    d['raw'] = d['fair_price']
    d['offset'] = offset
    d['fair_price'] = d['raw'] * (1 + offset / 100)
    return d


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
    stays a clean step function instead of one point per poll.

    An observation with neither a price nor a price label means "the scrape did
    not find a price element" (layout change, partial page) — not "the price was
    removed" — so it is not stored; the last known price stays valid."""
    if price is None and not price_text:
        return
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


PROBLEM_RATIO = 0.5      # below this share of priced ads a poll counts as a price-extraction failure
PROBLEM_MIN_ADS = 5      # ignore tiny searches


def record_poll_stats(tracker_id: int, polled_at: str, found: int, priced: int, rescued: int = 0):
    with get_db() as conn:
        conn.execute(
            'INSERT INTO poll_stats (tracker_id, polled_at, found, priced, rescued) '
            'VALUES (?, ?, ?, ?, ?)', (tracker_id, polled_at, found, priced, rescued))


def get_price_problems() -> dict[int, dict]:
    """tracker_id → stats of its latest poll, for trackers whose latest poll
    found few prices in the search results. Only polls from the last 3 days
    count, so backfilled history never shows up as a current problem."""
    cutoff = (datetime.utcnow() - timedelta(days=3)).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            'SELECT tracker_id, polled_at, found, priced, rescued FROM poll_stats '
            'WHERE id IN (SELECT MAX(id) FROM poll_stats GROUP BY tracker_id)').fetchall()
    return {r[0]: {'polled_at': r[1], 'found': r[2], 'priced': r[3], 'rescued': r[4]}
            for r in rows
            if r[1] >= cutoff and r[2] >= PROBLEM_MIN_ADS and r[3] < PROBLEM_RATIO * r[2]}


def get_gap_days() -> dict[int, set[str]]:
    """tracker_id → days (YYYY-MM-DD) on which too few ads had a usable price
    (search + detail-page rescue), i.e. days without trustworthy price data."""
    with get_db() as conn:
        rows = conn.execute(
            'SELECT tracker_id, substr(polled_at, 1, 10), SUM(found), SUM(priced + rescued) '
            'FROM poll_stats GROUP BY 1, 2').fetchall()
    out: dict[int, set[str]] = {}
    for tid, day, found, ok in rows:
        if found >= PROBLEM_MIN_ADS and ok < PROBLEM_RATIO * found:
            out.setdefault(tid, set()).add(day)
    return out


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


def detail_queue(listing_ids, older_than: str) -> list[int]:
    """Those of *listing_ids* whose full ad page is due (never archived or not since
    *older_than*), least recently *attempted* first (never attempted before all).
    Sorting by attempt, not success, gives a round-robin: an ad whose page keeps
    failing moves to the back instead of eating the request budget every run."""
    ids = list(dict.fromkeys(listing_ids))
    due = []
    with get_db() as conn:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            due += conn.execute(
                f'''SELECT id, detail_tried_at FROM listings
                     WHERE id IN ({",".join("?" * len(chunk))})
                       AND (detail_at IS NULL OR detail_at < ?)''',
                (*chunk, older_than)).fetchall()
    due.sort(key=lambda r: (r[1] is not None, r[1] or ''))
    return [r[0] for r in due]


def listing_tracker_ads(tracker_ids) -> dict[int, tuple[int, str | None]]:
    """tracker_id → (listing_id, detail_tried_at) for single-listing trackers that
    have already found their ad (trackers without a listing yet are absent)."""
    ids = list(tracker_ids)
    if not ids:
        return {}
    with get_db() as conn:
        rows = conn.execute(
            f'''SELECT tl.tracker_id, l.id, l.detail_tried_at
                  FROM tracker_listings tl JOIN listings l ON l.id = tl.listing_id
                 WHERE tl.tracker_id IN ({",".join("?" * len(ids))})''', ids).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def mark_detail_tried(listing_id: int, at: str):
    with get_db() as conn:
        conn.execute('UPDATE listings SET detail_tried_at = ? WHERE id = ?', (at, listing_id))


def mark_detail(listing_id: int, at: str):
    with get_db() as conn:
        conn.execute('UPDATE listings SET detail_at = ?, detail_tried_at = ? WHERE id = ?',
                     (at, at, listing_id))
