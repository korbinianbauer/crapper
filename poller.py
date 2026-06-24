"""Poll logic: refresh every tracker via its source, archive prices, flag
disappeared listings."""
import logging
from datetime import datetime

import db
import sources

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.utcnow().isoformat()


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

    lid = db.upsert_listing(
        source.name, scraped.ad_id, scraped.url, scraped.title,
        scraped.location, scraped.image_url, seen_at,
    )
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

    results = source.fetch_search(tracker['params'], max_pages=max_pages)
    for r in results:
        lid = db.upsert_listing(
            source.name, r.ad_id, r.url, r.title, r.location, r.image_url, seen_at,
        )
        db.link_listing(tracker['id'], lid, seen_at)
        db.add_price(lid, seen_at, r.price, r.price_text)

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
