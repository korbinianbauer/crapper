"""One-off cleanup after the Kleinanzeigen price-extraction outage (2026-09-02..10-06).

1. Backs the database up (sqlite online backup) to <db>.bak-priceclean.
2. Backfills `poll_stats` for outage days (a tracker's day with >= 80 % price-less
   observations), so the median chart can hide them and the data stays explainable.
3. Deletes price_history rows that have neither price nor price label — they mean
   "price not found", not "price removed" (the poller no longer stores them).

Usage: python scripts/cleanup_missing_prices.py [--apply] [path/to/crapper.db]
Without --apply it only reports what it would do.
"""
import os
import sqlite3
import sys

args = [a for a in sys.argv[1:] if not a.startswith('--')]
apply = '--apply' in sys.argv
path = args[0] if args else os.path.join(os.path.dirname(__file__), '..', 'crapper.db')

conn = sqlite3.connect(path)
if apply:
    bak = sqlite3.connect(path + '.bak-priceclean')
    conn.backup(bak)
    bak.close()
    print('backup written:', path + '.bak-priceclean')

conn.executescript('''
    CREATE TABLE IF NOT EXISTS poll_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_id INTEGER NOT NULL REFERENCES trackers(id) ON DELETE CASCADE,
        polled_at TEXT NOT NULL, found INTEGER NOT NULL, priced INTEGER NOT NULL,
        rescued INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX IF NOT EXISTS idx_stats_tracker ON poll_stats(tracker_id, polled_at);
''')
empty = "price IS NULL AND (price_text IS NULL OR price_text = '')"
n_empty = conn.execute(f'SELECT COUNT(*) FROM price_history WHERE {empty}').fetchone()[0]
rows = conn.execute(f'''
    SELECT tl.tracker_id, substr(p.observed_at, 1, 10) AS d, COUNT(*) AS n,
           SUM({empty.replace('price', 'p.price')}) AS empties
    FROM price_history p JOIN tracker_listings tl ON tl.listing_id = p.listing_id
    GROUP BY 1, 2''').fetchall()
gaps = [(t, d, n, n - e) for t, d, n, e in rows if n >= 5 and e >= 0.8 * n]
print(f'{n_empty} price-less rows; {len(gaps)} outage tracker-days to backfill')
if apply:
    conn.executemany(
        'INSERT INTO poll_stats (tracker_id, polled_at, found, priced, rescued) VALUES (?,?,?,?,0)',
        [(t, d + 'T04:00:00', n, ok) for t, d, n, ok in gaps])
    conn.execute(f'DELETE FROM price_history WHERE {empty}')
    conn.commit()
    print('done:', n_empty, 'rows deleted,', len(gaps), 'outage days recorded')
