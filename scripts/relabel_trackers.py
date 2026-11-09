"""Re-generate the titles of Kleinanzeigen search trackers from their search
parameters (see sources.kleinanzeigen.search_label).

Usage: python scripts/relabel_trackers.py [--apply]   (dry run without --apply)
Needs network access (the result heading supplies keyword/place names).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import db
import sources

apply = '--apply' in sys.argv
for t in db.get_trackers():
    if t['source'] != 'kleinanzeigen' or t['type'] != 'search':
        continue
    _, label = sources.get('kleinanzeigen').describe_search(t['url'])
    if not label:
        print(f"#{t['id']}: no label (fetch failed), kept: {t['label']}")
        continue
    print(f"#{t['id']}: {t['label'][:70]!r}\n     -> {label!r}")
    if apply and label != t['label']:
        with db.get_db() as conn:
            conn.execute('UPDATE trackers SET label = ? WHERE id = ?', (label, t['id']))
