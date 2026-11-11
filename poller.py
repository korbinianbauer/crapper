"""Poll logic: refresh every tracker via its source, archive prices, flag
disappeared listings."""
import json
import logging
import random
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
DETAIL_DELAY = (1.0, 3.0)                # random pause range (s) between ad-page requests

BACKOFF_START = timedelta(minutes=10)    # block after the first 403/429 …
BACKOFF_MAX = timedelta(hours=24)        # … doubling with every further one, up to this cap


def _block_key(source) -> str:
    return f'detail_block_{source.name}'


def _detail_blocked_until(source) -> datetime | None:
    """Persistent circuit breaker: until when ad-page fetches of this source are
    suspended after a 403/429 (None = not suspended)."""
    try:
        st = json.loads(db.get_setting(_block_key(source), '') or 'null')
        until = datetime.fromisoformat(st['until']) if st else None
    except (ValueError, KeyError, TypeError):
        return None
    return until if until and until > datetime.utcnow() else None


def _register_block(source, lid: int | None = None) -> bool:
    """Rate-limited: double the suspension (10 min, 20 min, … max 24 h). The level
    survives restarts and only resets after a successful fetch. Returns True if the
    same listing *lid* also hit the previous block (→ maybe the ad itself, not us)."""
    try:
        st = json.loads(db.get_setting(_block_key(source), '') or 'null') or {}
        level = int(st['level'])
    except (ValueError, KeyError, TypeError, AttributeError):
        st, level = {}, 0
    repeat = lid is not None and st.get('lid') == lid
    level += 1
    delay = min(BACKOFF_START * 2 ** (level - 1), BACKOFF_MAX)
    until = datetime.utcnow() + delay
    db.set_setting(_block_key(source), json.dumps(
        {'level': level, 'until': until.isoformat(), 'lid': lid}))
    logger.warning('%s ad pages rate-limited (#%d) — detail fetches suspended for %s (until %s UTC)',
                   source.name, level, delay, until.isoformat(timespec='seconds'))
    return repeat


def _clear_block(source):
    if db.get_setting(_block_key(source), ''):
        db.set_setting(_block_key(source), '')


class _DetailJob:
    """A search hit whose full ad page should be archived (phase 2 of a poll)."""
    __slots__ = ('source', 'lid', 'url', 'seen_at', 'priceless')

    def __init__(self, source, lid, url, seen_at):
        self.source, self.lid, self.url, self.seen_at = source, lid, url, seen_at
        self.priceless: set[int] = set()   # trackers whose search card had no price


def _fetch_detail(job: _DetailJob):
    """Fetch the whole ad page and archive it (all images, description, details,
    original HTML). A failed/removed fetch changes nothing — activity is decided by
    the search, not by this. Every attempt is recorded (`detail_tried_at`) so the
    queue rotates. A 403/429 is not the ad's fault and is not recorded — unless the
    same ad also hit the previous block, then it goes to the back of the queue."""
    source, now = job.source, _now()
    time.sleep(random.uniform(*DETAIL_DELAY))
    try:
        full = job.source.fetch_listing(job.url)
    except RateLimited:
        if _register_block(source, job.lid):
            db.mark_detail_tried(job.lid, now)
        return None
    except Exception as e:
        logger.warning('Detail fetch failed %s: %s', job.url, e)
        db.mark_detail_tried(job.lid, now)
        return None
    _clear_block(source)
    if full is None:
        db.mark_detail_tried(job.lid, now)
        return None
    lid = _upsert(source, full, now)
    db.mark_detail(lid, now)
    return full


def _run_details(jobs: dict, stats: dict):
    """Phase 2: archive the due ad pages of all trackers polled in phase 1, least
    recently attempted first, so ads left over by a rate-limit block are the first
    to be fetched next time. Stops per source once it is blocked."""
    cutoff = (datetime.utcnow() - DETAIL_INTERVAL).isoformat()
    for lid in db.detail_queue(jobs, cutoff):
        job = jobs[lid]
        if _detail_blocked_until(job.source):
            continue
        full = _fetch_detail(job)
        # If a search card carried no price (extraction problem), fall back to
        # the price on the ad's own page.
        if job.priceless and full and (full.price is not None or full.price_text):
            db.add_price(lid, job.seen_at, full.price, full.price_text)
            for tid in job.priceless:
                stats[tid]['rescued'] += 1


def poll_tracker(tracker: dict):
    """Refresh a single tracker. Used by the 'refresh now' button and the initial
    poll on add."""
    _poll([tracker])


def poll_all_due():
    _poll(db.get_trackers(enabled_only=True))
    try:    # stylesheets/fonts/images of all newly stored pages (and the old backlog)
        pagearchive.archive_pending()
    except Exception as e:
        logger.error('Asset archiving failed: %s', e)


def _poll(trackers: list[dict]):
    """Phase 1: every search tracker (prices, memberships, deactivation), then the
    single-listing trackers, least recently attempted first. Phase 2: the full ad
    pages of all search hits, in one shared, prioritised queue."""
    jobs: dict[int, _DetailJob] = {}
    stats: dict[int, dict] = {}
    searches = [t for t in trackers if t['type'] == 'search']
    singles = [t for t in trackers if t['type'] == 'listing']
    for t in trackers:
        if t['type'] not in ('search', 'listing'):
            logger.warning('Unknown type %r (tracker %s)', t['type'], t['id'])
    ads = db.listing_tracker_ads(t['id'] for t in singles)
    singles.sort(key=lambda t: (t['id'] in ads and ads[t['id']][1] is not None,
                                ads.get(t['id'], (0, ''))[1] or ''))
    for tracker in searches + singles:
        try:
            source = sources.get(tracker['source'])
        except KeyError:
            logger.warning('Unknown source %r (tracker %s)', tracker['source'], tracker['id'])
            continue
        try:
            if tracker['type'] == 'listing':
                _poll_listing_tracker(tracker, source, ads.get(tracker['id'], (None,))[0])
            else:
                stats[tracker['id']] = _poll_search_tracker(tracker, source, jobs)
        except Exception as e:
            logger.error('Poll failed for tracker %d: %s', tracker['id'], e)
    try:
        _run_details(jobs, stats)
    except Exception as e:
        logger.error('Detail archiving failed: %s', e)
    for tid, s in stats.items():
        db.record_poll_stats(tid, s['seen_at'], s['found'], s['priced'], s['rescued'])
        if s['found'] >= db.PROBLEM_MIN_ADS and s['priced'] < db.PROBLEM_RATIO * s['found']:
            logger.warning('Search tracker %d: only %d of %d ads had a price in the search '
                           'results (%d rescued from ad pages) — price extraction broken?',
                           tid, s['priced'], s['found'], s['rescued'])


def _poll_listing_tracker(tracker: dict, source, lid: int | None):
    """Fetch the tracked ad. Subject to the source's rate-limit backoff like the
    detail queue: while blocked it is skipped (nothing is deactivated), a 403/429
    starts/extends the block. *lid* = the tracker's listing if already known."""
    if until := _detail_blocked_until(source):
        logger.info('Listing tracker %d (%s): skipped, ad pages blocked until %s UTC',
                    tracker['id'], source.name, until.isoformat(timespec='seconds'))
        return
    seen_at = _now()
    try:
        scraped = source.fetch_listing(tracker['url'])
    except RateLimited:
        if _register_block(source, lid) and lid:
            db.mark_detail_tried(lid, seen_at)
        return
    _clear_block(source)

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
