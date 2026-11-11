"""Poll logic: refresh every tracker via its source, archive prices, flag
disappeared listings."""
import json
import logging
import time
from datetime import datetime, timedelta

import archive
import db
import pagearchive
import sources
from sources.base import RateLimited

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _upsert(source, r, seen_at) -> int:
    lid = db.upsert_listing(
        source.name, r.ad_id, r.url, r.title, r.location, r.image_url, seen_at,
        description=r.description, images=r.images, raw_html=r.raw_html)
    url = db.listing_needs_image(lid)
    if url and (fname := archive.download_image(url)):
        db.set_image_file(lid, url, fname)
    archive_full_images(source, [*r.images, r.image_url])
    return lid


def archive_full_images(source, urls):
    """Download every image in its largest available size (never just the preview).
    Files are keyed by URL and reused, so repeated calls cost nothing."""
    seen = set()
    for u in urls:
        if not u:
            continue
        full = source.full_image_url(u)
        if full in seen:
            continue
        seen.add(full)
        if not archive.download_image(full) and full != u:
            archive.download_image(u)      # no larger variant available: keep what we have


DETAIL_INTERVAL = timedelta(hours=20)    # "at least daily" with slack for poll jitter

# Ad-page fetches of a source are paced by one persistent interval: the time between
# two requests. It doubles after every 403/429 and halves after every
# successful fetch, always staying within BACKOFF_MIN … BACKOFF_MAX.
BACKOFF_MIN = timedelta(seconds=1)       # also the start value
BACKOFF_MAX = timedelta(hours=4)
BACKOFF_GROWTH = 2.0
BACKOFF_DECAY = 0.5
LIMITED_TO_BACK = 3                      # an ad blocked this often in one poll is retried last
MAX_POLL_WAIT = timedelta(hours=4)       # a poll gives up on items not due within this of its start
IN_FLIGHT = timedelta(minutes=2)         # slot reserved for a running request (crash safety)

# All reads/writes of the pace state go through db.update_setting_atomic (SQLite write
# lock), so parallel polls — threads of the web server or the scheduled poll process —
# share one interval per source and never send at the same time.


def _block_key(source) -> str:
    return f'detail_block_{source.name}'


def _parse_pace(raw: str) -> dict:
    """{'delay': interval in s, 'next': ISO time the next request may go out (or None)}.
    Defaults for a source with no history."""
    try:
        st = json.loads(raw or 'null') or {}
        delay = float(st['delay']) if 'delay' in st else BACKOFF_MIN.total_seconds()
        nxt = st.get('next') or st.get('until')          # 'until': pre-interval format
        datetime.fromisoformat(nxt) if nxt else None
        return {'delay': delay, 'next': nxt}
    except (ValueError, TypeError, AttributeError):
        return {'delay': BACKOFF_MIN.total_seconds(), 'next': None}


def _pace_state(source) -> dict:
    return _parse_pace(db.get_setting(_block_key(source), ''))


def _ready_at(source) -> datetime:
    """When this source's next ad-page request may go out (≤ now: right away)."""
    nxt = _pace_state(source)['next']
    return datetime.fromisoformat(nxt) if nxt else datetime.min


def _reserve(source) -> bool:
    """Atomically claim the source's next request slot. False if it is not free yet
    (another poll got it first). The slot is held IN_FLIGHT until the request's
    outcome reschedules it (`_pace`)."""
    def fn(raw):
        st = _parse_pace(raw)
        now = datetime.utcnow()
        if st['next'] and datetime.fromisoformat(st['next']) > now:
            return None, False
        st['next'] = (now + IN_FLIGHT).isoformat()
        return json.dumps(st), True
    return db.update_setting_atomic(_block_key(source), fn)


def _pace(source, factor: float) -> dict:
    """Scale the interval by *factor* (clamped) and schedule the next request."""
    def fn(raw):
        st = _parse_pace(raw)
        st['delay'] = min(max(st['delay'] * factor, BACKOFF_MIN.total_seconds()),
                          BACKOFF_MAX.total_seconds())
        st['next'] = (datetime.utcnow() + timedelta(seconds=st['delay'])).isoformat()
        return json.dumps(st), st
    return db.update_setting_atomic(_block_key(source), fn)


def _register_block(source):
    """Rate-limited (403/429): double the interval; the next request waits for it."""
    st = _pace(source, BACKOFF_GROWTH)
    logger.warning('%s ad pages rate-limited — interval now %s (next request %s UTC)',
                   source.name, timedelta(seconds=round(st['delay'])), st['next'][11:19])


def _note_success(source):
    """An ad-page request that was not rate-limited: halve the interval."""
    _pace(source, BACKOFF_DECAY)


def _note_attempt(source):
    """A request that failed otherwise (network error …): keep the interval."""
    _pace(source, 1.0)


class _DetailJob:
    """A search hit whose full ad page should be archived (phase 2 of a poll)."""
    __slots__ = ('source', 'lid', 'url', 'seen_at', 'priceless')

    def __init__(self, source, lid, url, seen_at):
        self.source, self.lid, self.url, self.seen_at = source, lid, url, seen_at
        self.priceless: set[int] = set()   # trackers whose search card had no price


class _Limited(Exception):
    """A fetch hit HTTP 403/429: the item stays queued and is retried."""


def _fetch_detail(job: _DetailJob):
    """Fetch the whole ad page and archive it (all images, description, details,
    original HTML). A failed/removed fetch changes nothing — activity is decided by
    the search, not by this. Every attempt is recorded (`detail_tried_at`) so the
    queue rotates; a 403/429 is not the ad's fault and is not recorded."""
    source, now = job.source, _now()
    try:
        full = job.source.fetch_listing(job.url)
    except RateLimited:
        _register_block(source)
        raise _Limited()
    except Exception as e:
        logger.warning('Detail fetch failed %s: %s', job.url, e)
        _note_attempt(source)
        db.mark_detail_tried(job.lid, now)
        return None
    _note_success(source)
    if full is None:
        db.mark_detail_tried(job.lid, now)
        return None
    lid = _upsert(source, full, now)
    db.mark_detail(lid, now)
    return full


def _detail_item(job: _DetailJob, stats: dict):
    full = _fetch_detail(job)
    # If a search card carried no price (extraction problem), fall back to the
    # price on the ad's own page.
    if job.priceless and full and (full.price is not None or full.price_text):
        db.add_price(job.lid, job.seen_at, full.price, full.price_text)
        for tid in job.priceless:
            stats[tid]['rescued'] += 1


def _drain(queue: list, started: datetime):
    """Work through *queue* (list of (source, label, fn)) in order, each source at its
    own pace (`_ready_at`): the next item is the first one whose source may send
    again; if none may, sleep until the earliest does. A rate-limited item stays in
    place (retried first); after LIMITED_TO_BACK blocks in this poll it moves to the
    end, in case the ad itself, not our pace, triggers them. Items whose source is not ready within MAX_POLL_WAIT of the poll's start
    are left for the next poll (where they come first, as least recently attempted)."""
    queue = list(queue)
    limited: dict[int, int] = {}
    logged_wait = None
    while queue:
        ready = {src: _ready_at(src) for src in {i[0] for i in queue}}
        now = datetime.utcnow()
        idx = next((k for k, item in enumerate(queue) if ready[item[0]] <= now), None)
        if idx is None:
            soonest = min(ready.values())
            if soonest > started + MAX_POLL_WAIT:
                logger.warning('%d ad pages left for the next poll — next request allowed at '
                               '%s UTC, beyond this poll\'s wait limit', len(queue),
                               soonest.isoformat(timespec='seconds'))
                return
            wait = (soonest - now).total_seconds()
            if wait >= 60 and soonest != logged_wait:
                logged_wait = soonest
                logger.info('Waiting %s for the rate-limit interval (%d ad pages left)',
                            timedelta(seconds=round(wait)), len(queue))
            # Re-check at least every second: a slot held IN_FLIGHT by another poll
            # is released as soon as its request is done.
            time.sleep(min(wait, 1.0))
            continue
        if not _reserve(queue[idx][0]):
            continue                      # another poll took this source's slot
        item = queue.pop(idx)
        try:
            item[2]()
        except _Limited:
            limited[id(item)] = n = limited.get(id(item), 0) + 1
            queue.insert(len(queue) if n % LIMITED_TO_BACK == 0 else idx, item)
        except Exception as e:
            logger.error('%s failed: %s', item[1], e)
            _note_attempt(item[0])        # release the reserved slot


def poll_tracker(tracker: dict):
    """Refresh a single tracker. Used by the 'refresh now' button and the initial
    poll on add."""
    _poll([tracker])


def poll_trackers(trackers: list[dict]):
    """Refresh several trackers as one poll (shared, prioritised ad-page queue).
    Used by 'refresh all' — one call per source, each in its own thread."""
    _poll(trackers)


def poll_all_due():
    _poll(db.get_trackers(enabled_only=True))
    try:    # stylesheets/fonts/images of all newly stored pages (and the old backlog)
        pagearchive.archive_pending()
    except Exception as e:
        logger.error('Asset archiving failed: %s', e)


def _poll(trackers: list[dict]):
    """Phase 1: every search tracker (prices, memberships, deactivation; no ad pages).
    Phase 2: the single-listing trackers, then the due ad pages of all search hits —
    each group least recently attempted first — with rate-limit blocks waited out
    (`_drain`)."""
    started = datetime.utcnow()
    jobs: dict[int, _DetailJob] = {}
    stats: dict[int, dict] = {}
    singles = []
    for tracker in trackers:
        if tracker['type'] not in ('search', 'listing'):
            logger.warning('Unknown type %r (tracker %s)', tracker['type'], tracker['id'])
            continue
        try:
            source = sources.get(tracker['source'])
        except KeyError:
            logger.warning('Unknown source %r (tracker %s)', tracker['source'], tracker['id'])
            continue
        if tracker['type'] == 'listing':
            singles.append((tracker, source))
            continue
        try:
            stats[tracker['id']] = _poll_search_tracker(tracker, source, jobs)
        except Exception as e:
            logger.error('Poll failed for tracker %d: %s', tracker['id'], e)

    ads = db.listing_tracker_ads(t['id'] for t, _ in singles)
    singles.sort(key=lambda ts: (ads.get(ts[0]['id'], (0, None))[1] is not None,
                                 ads.get(ts[0]['id'], (0, None))[1] or ''))
    queue = [(src, f'Poll of tracker {t["id"]}',
              lambda t=t, src=src: _poll_listing_tracker(t, src, ads.get(t['id'], (None,))[0]))
             for t, src in singles]
    try:
        cutoff = (datetime.utcnow() - DETAIL_INTERVAL).isoformat()
        queue += [(jobs[lid].source, f'Detail fetch of listing {lid}',
                   lambda job=jobs[lid]: _detail_item(job, stats))
                  for lid in db.detail_queue(jobs, cutoff)]
        _drain(queue, started)
    except Exception as e:
        logger.error('Detail archiving failed: %s', e)
    for tid, s in stats.items():
        db.record_poll_stats(tid, s['seen_at'], s['found'], s['priced'], s['rescued'])
        if s['found'] >= db.PROBLEM_MIN_ADS and s['priced'] < db.PROBLEM_RATIO * s['found']:
            logger.warning('Search tracker %d: only %d of %d ads had a price in the search '
                           'results (%d rescued from ad pages) — price extraction broken?',
                           tid, s['priced'], s['found'], s['rescued'])


def _poll_listing_tracker(tracker: dict, source, lid: int | None):
    """Fetch the tracked ad. A 403/429 raises `_Limited` (retried after the block,
    nothing is deactivated). *lid* = the tracker's listing if already known."""
    seen_at = _now()
    try:
        scraped = source.fetch_listing(tracker['url'])
    except RateLimited:
        _register_block(source)
        raise _Limited()
    _note_success(source)

    if scraped is None:
        # Ad removed / unavailable — flag this tracker's membership inactive (the
        # listing stays inactive unless another tracker still finds it).
        if lid:
            db.mark_detail_tried(lid, seen_at)
        db.mark_tracker_listings_inactive(tracker['id'])
        logger.info('Listing tracker %d (%s): ad unavailable, marked inactive',
                    tracker['id'], source.name)
        return

    lid = _upsert(source, scraped, seen_at)
    db.mark_detail(lid, seen_at)
    db.link_listing(tracker['id'], lid, seen_at)
    db.add_price(lid, seen_at, scraped.price, scraped.price_text)
    logger.info('Listing tracker %d (%s): %s — %s',
                tracker['id'], source.name, scraped.title, scraped.price_text)


def _poll_search_tracker(tracker: dict, source, jobs: dict) -> dict:
    """Prices, memberships and deactivation from the search pages; the hits' ad
    pages are only queued in *jobs* (fetched in phase 2). Returns the poll stats."""
    seen_at = _now()
    try:
        max_pages = int(db.get_setting('search_pages', '5'))
    except ValueError:
        max_pages = 5

    results = source.fetch_search(tracker['url'], max_pages=max_pages)
    priced = queued = 0
    for r in results:
        lid = _upsert(source, r, seen_at)
        db.link_listing(tracker['id'], lid, seen_at)
        has_price = r.price is not None or bool(r.price_text)
        if has_price:
            priced += 1
            db.add_price(lid, seen_at, r.price, r.price_text)
        if source.archive_details:
            queued += 1
            job = jobs.setdefault(lid, _DetailJob(source, lid, r.url, seen_at))
            if not has_price:
                job.priceless.add(tracker['id'])

    # Anything not refreshed this run has dropped out of the search → inactive.
    db.deactivate_missing(tracker['id'], seen_at)
    logger.info('Search tracker %d (%s): %d listings found, %d queued for ad-page check',
                tracker['id'], source.name, len(results), queued)
    return {'seen_at': seen_at, 'found': len(results), 'priced': priced, 'rescued': 0}
