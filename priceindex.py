"""Hedonic price index (fair-price model) — computed per tracker and stored per day.

Per day: geometric mean of (normalised asking price / model fair price) over all listings alive that
day (see `index_by_day`). The values are persisted in `price_index` so that

* the daily poll appends today's value when it is finished (`update_daily`, called by
  `poller.poll_all_due`) — listings that have no features/prediction yet are missing from it;
* once those are extracted and predicted (or the model changes), the history is corrected
  retroactively with `recompute(since=…)` — called by `ml.predict` / `ml.refresh_predictions`, or by hand:
  `python priceindex.py recompute [tracker_id] [--since YYYY-MM-DD]`, `python priceindex.py update`.
"""
import logging
import math
import sys
from datetime import date, datetime, timedelta

import db
import ml

logger = logging.getLogger(__name__)

MIN_INDEX_N = 3             # fewer comparable listings on a day → no index value
_IDX_CACHE: dict = {}


def index_series(card: dict, fair: dict | None):
    """(start, end, [(day, price/fair)], model trained_at) of one listing for the model-based index.
    The ratio is the *normalised* asking price (rents incl. Nebenkosten/Heizung …, `ml.effective_price`)
    divided by the raw model prediction. Implausible prices (excluded from training) are skipped."""
    if not fair or not fair.get('raw'):
        return None
    key = (card['id'], fair.get('model_id'), len(card['history']))
    pts = _IDX_CACHE.get(key)
    if pts is None:
        cm = ml.cost_models_for(fair.get('model_id'))
        pts, memo = [], {}
        for h in card['history']:
            p = h['price']
            if p is not None and p not in memo:
                v = None
                if ml.is_plausible(p, fair['features'], fair['per'], fair.get('per_range')):
                    v = ml.effective_price(p, fair.get('cost'), fair['features'], fair.get('assume'), cm) or p
                memo[p] = v / fair['raw'] if v else None
            pts.append((date.fromisoformat(h['t'][:10]), None if p is None else memo[p]))
        _IDX_CACHE[key] = pts
    if not any(x[1] is not None for x in pts):
        return None
    # `end` depends on today / last_seen, so it must NOT be cached together with the points
    end = date.today() if card.get('active', True) else \
        min(date.today(), date.fromisoformat((card.get('last_seen') or '')[:10] or date.today().isoformat()))
    return (pts[0][0], max(end, pts[0][0]), pts, fair.get('trained_at'))


def index_by_day(items: list[dict], fair_map: dict, today: date | None = None):
    """({day iso: (index | None, n)}, trained dates) for the listings `items`
    (cards with `id`, `history`, `active`, `last_seen`)."""
    today = today or date.today()
    series = [x for x in (index_series(it, fair_map.get(it['id'])) for it in items) if x]
    if not series:
        return {}, []
    first = min(s[0] for s in series)
    cur = [[0] for _ in series]
    out = {}
    for d in range((today - first).days + 1):
        day = first + timedelta(d)
        logs = []
        for (start, end, pts, _), c in zip(series, cur):
            if start <= day <= end:
                i = c[0]
                while i + 1 < len(pts) and pts[i + 1][0] <= day:
                    i += 1
                c[0] = i
                r = pts[i][1] if pts[i][0] <= day else None
                if r:
                    logs.append(math.log(r))
        out[day.isoformat()] = (math.exp(sum(logs) / len(logs)) if len(logs) >= MIN_INDEX_N else None, len(logs))
    trained = sorted({(s[3] or '')[:10] for s in series if s[3]})
    return out, trained


def _tracker_items():
    """[(tracker_id, [cards])] for every tracker; a group also contains its members' listings."""
    trackers = db.get_trackers()
    memberships = db.get_tracker_memberships()
    hist = db.get_price_history_map()
    cards = {l['id']: {'id': l['id'], 'active': bool(l['active']), 'last_seen': l['last_seen'],
                       'history': [{'t': h['observed_at'], 'price': h['price']} for h in hist.get(l['id'], [])]}
             for l in db.get_all_listings()}
    res = []
    for t in trackers:
        tids = [t['id']] + ([m['id'] for m in trackers if m.get('parent_id') == t['id']]
                            if t['type'] == 'group' else [])
        ids = list(dict.fromkeys(i for tid in tids for i in memberships.get(tid, []) if i in cards))
        res.append((t['id'], [cards[i] for i in ids]))
    return res


def recompute(tracker_id: int | None = None, since: str | None = None) -> int:
    """(Re)write the stored index of one tracker or all, for the days from `since` (ISO date; default:
    the whole history). Days before `since` stay as they are. Returns the number of rows written."""
    fair_map = db.get_fair_price_map()
    now, n = datetime.now().isoformat(timespec='seconds'), 0
    for tid, items in _tracker_items():
        if tracker_id is not None and tid != tracker_id:
            continue
        days, _ = index_by_day(items, fair_map)
        rows = [(tid, d, v, cnt, now) for d, (v, cnt) in days.items() if since is None or d >= since]
        db.save_price_index(tid, rows, since)
        n += len(rows)
    logger.info('Price index: %d day values (re)written', n)
    return n


def update_daily() -> int:
    """After a poll: append today's value (and re-write yesterday's, whose listings may have just
    changed) for every tracker."""
    return recompute(since=(date.today() - timedelta(days=1)).isoformat())


def safe_recompute(tracker_id=None, since=None):
    """For hooks: never let an index problem break the caller."""
    try:
        return recompute(tracker_id, since)
    except Exception as e:
        logger.error('Price index recompute failed: %s', e)
        return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    db.init_db()
    a = sys.argv[1:]
    if not a or a[0] not in ('recompute', 'update'):
        sys.exit(__doc__)
    if a[0] == 'update':
        update_daily()
    else:
        since = a[a.index('--since') + 1] if '--since' in a else None
        tid = next((int(x) for x in a[1:] if x.isdigit()), None)
        recompute(tid, since)
