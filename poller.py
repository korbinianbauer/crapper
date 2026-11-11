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


def _register_block(source):
    """Rate-limited: double the suspension (10 min, 20 min, … max 24 h). The level
    survives restarts and only resets after a successful fetch."""
    try:
        level = int(json.loads(db.get_setting(_block_key(source), '') or 'null')['level'])
    except (ValueError, KeyError, TypeError):
        level = 0
    level += 1
    delay = min(BACKOFF_START * 2 ** (level - 1), BACKOFF_MAX)
    until = datetime.utcnow() + delay
    db.set_setting(_block_key(source), json.dumps({'level': level, 'until': until.isoformat()}))
    logger.warning('%s ad pages rate-limited (#%d) — detail fetches suspended for %s (until %s UTC)',
                   source.name, level, delay, until.isoformat(timespec='seconds'))


def _clear_block(source):
    if db.get_setting(_block_key(source), ''):
        db.set_setting(_block_key(source), '')


def _archive_full(source, lid: int, url: str, seen_at: str):
    """Fetch the whole ad page and archive it (all images, description, details,
    original HTML). A failed/removed fetch changes nothing — activity is decided
    by the search, not by this. Suspended while the source's backoff is running."""
    cutoff = (datetime.utcnow() - DETAIL_INTERVAL).isoformat()
    if not db.needs_detail(lid, cutoff) or _detail_blocked_until(source):
        return None
    time.sleep(random.uniform(*DETAIL_DELAY))
    try:
        full = source.fetch_listing(url)
    except RateLimited:
        _register_block(source)          # also ends detail fetching for the rest of the run
        return None
    except Exception as e:
        logger.warning('Detail fetch failed %s: %s', url, e)
        return None
    _clear_block(source)
    if full is None:
        return None
    lid = _upsert(source, full, seen_at)
    db.mark_detail(lid, seen_at)
    return full


def poll_tracker(tracker: dict):
    """Refresh a single tracker. Used by the scheduler, the 'refresh now' button
    and the initial poll on add."""
    try:
        source = sources.get(tracker['source'])
    except KeyError:
        logger.warning('Unknown source %r (tracker %s)', tracker['source'], tracker['id'])
        return
    if tracker['type'] == 'listing':
        _poll_listing_tracker(tracker, source)
    elif tracker['type'] == 'search':
        _poll_search_tracker(tracker, source)
    else:
        logger.warning('Unknown tracker type %r (id=%s)', tracker['type'], tracker['id'])


def _poll_listing_tracker(tracker: dict, source):
    seen_at = _now()
    scraped = source.fetch_listing(tracker['url'])

    if scraped is None:
        # Ad removed / unavailable — flag this tracker's membership inactive (the
        # listing stays inactive unless another tracker still finds it).
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


def _poll_search_tracker(tracker: dict, source):
    seen_at = _now()
    try:
        max_pages = int(db.get_setting('search_pages', '5'))
    except ValueError:
        max_pages = 5

    results = source.fetch_search(tracker['url'], max_pages=max_pages)
    priced = rescued = 0
    for r in results:
        lid = _upsert(source, r, seen_at)
        db.link_listing(tracker['id'], lid, seen_at)
        has_price = r.price is not None or bool(r.price_text)
        if has_price:
            priced += 1
            db.add_price(lid, seen_at, r.price, r.price_text)
        # The page is archived either way. If the search card carried no price
        # (extraction problem), fall back to the price on the ad's own page.
        full = _archive_full(source, lid, r.url, seen_at) if source.archive_details else None
        if not has_price and full and (full.price is not None or full.price_text):
            rescued += 1
            db.add_price(lid, seen_at, full.price, full.price_text)

    db.record_poll_stats(tracker['id'], seen_at, len(results), priced, rescued)
    if len(results) >= db.PROBLEM_MIN_ADS and priced < db.PROBLEM_RATIO * len(results):
        logger.warning('Search tracker %d (%s): only %d of %d ads had a price in the '
                       'search results (%d rescued from ad pages) — price extraction broken?',
                       tracker['id'], source.name, priced, len(results), rescued)

    # Anything not refreshed this run has dropped out of the search → inactive.
    db.deactivate_missing(tracker['id'], seen_at)
    logger.info('Search tracker %d (%s): %d listings found',
                tracker['id'], source.name, len(results))


def poll_all_due():
    for tracker in db.get_trackers(enabled_only=True):
        try:
            poll_tracker(tracker)
        except Exception as e:
            logger.error('Poll failed for tracker %d: %s', tracker['id'], e)
    try:    # stylesheets/fonts/images of all newly stored pages (and the old backlog)
        pagearchive.archive_pending()
    except Exception as e:
        logger.error('Asset archiving failed: %s', e)
