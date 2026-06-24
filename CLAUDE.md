# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Running the app

```bash
pip install -r requirements.txt
SECRET_KEY=... CRAPPER_PASSWORD=... python app.py    # Flask dev server on :5000
```

In production it runs under gunicorn (see Deployment). Auth is single-user:
username from `CRAPPER_USER` (default `admin`), password from `CRAPPER_PASSWORD`
(login always fails if unset). `SECRET_KEY` signs sessions; if unset an ephemeral
key is generated (sessions drop on restart).

## What it is

**crapper** is a multi-source price tracker. The user tracks either single ads
(by URL) or saved searches on a chosen **source** (site); a daily poll scrapes
each, archives every price observation, and the frontpage plots each listing's
price history. Listings that disappear from their source are kept and shown with
a red border. Kleinanzeigen and ImmoScout24 are implemented; adding a site is a
matter of writing one `Source` subclass.

## Architecture

### Sources (the extension point)

A **Source** (`sources/base.py`) is one trackable site. It declares its
capabilities (`supports_search`, `supports_listing`) and implements only what the
site offers:

- `matches_url(url)` / `extract_ad_id(url)` / `fetch_listing(url)` — single ads
- `search_fields()` — describes the search form (list of `SearchField`:
  type `text` / `select` / `location`) so the frontend renders it generically
- `search_locations(query)` — autocomplete for `location` fields
- `search_label(params)` / `fetch_search(params, max_pages)` — run a saved search

`search` results and `fetch_listing` both return `ScrapedListing`. Register a new
source by appending an instance to `_all` in `sources/__init__.py`.

Implemented sources:
- **`kleinanzeigen`** — classifieds; search (query + location + radius) **and**
  single listings. Plain browser-like HTTP, BeautifulSoup parsing.
- **`immoscout`** — real estate; **search only** (`supports_listing=False`)
  because `/expose/{id}` detail is WAF-blocked. Uses the mobile JSON gateway.

### Data flow

1. `poll.py` (standalone process, auto-started by `app.py`) runs an APScheduler
   cron job — schedule from the `poll_cron` setting, reloaded via SIGHUP. Each
   fire runs `poll_all_due()` in a fresh subprocess so it uses the latest code.
2. `poller.poll_tracker()` looks up the tracker's source, then:
   - **listing** → `source.fetch_listing(url)`; if gone,
     `db.mark_tracker_listings_inactive()` flags this tracker's membership.
   - **search** → `source.fetch_search(params)`; each ad is upserted; memberships
     not seen this run are flagged via `db.deactivate_missing()`.
3. `db.upsert_listing(source, ad_id, …)` refreshes/creates the listing (unique by
   `(source, ad_id)`); `db.link_listing()` records the tracker→listing
   membership; `db.add_price()` appends a `price_history` row (skipping
   consecutive duplicate prices).
4. Deactivation recomputes each listing's `active` flag: active iff ≥1 membership
   is active. Adding a tracker triggers an immediate background poll
   (`_refresh_async`).

### Key files

| File | Role |
|------|------|
| `app.py` | Flask routes, auth/CSRF, poll-process management, index payload |
| `db.py` | All SQLite access; `init_db()` (wipe-migrates pre-multi-source schemas) |
| `sources/base.py` | `Source` ABC + `ScrapedListing` / `LocationResult` / `SearchField` |
| `sources/__init__.py` | `REGISTRY`, `get`, `all_sources`, `search_sources`, `source_for_url` |
| `sources/kleinanzeigen.py`, `sources/immoscout.py` | the two sources |
| `poller.py` | `poll_tracker()` / `poll_all_due()` — orchestrates source → DB |
| `poll.py` | APScheduler cron loop (separate process) |
| `templates/index.html` | Listing grid + Plotly charts + add modals (source-driven search form, per-source location autocomplete) |

### Database schema (`crapper.db`)

- **`trackers`** — one per user request; `source` slug + `type` = `listing`
  (uses `url`) or `search` (uses `params`, a source-specific JSON blob).
- **`listings`** — discovered ads, **unique by `(source, ad_id)`** so an ad
  surfaced by several trackers is stored/shown once. `active=0` means no
  referencing tracker found it in its latest poll.
- **`tracker_listings`** — many-to-many membership (which trackers surfaced which
  listing) with per-tracker `last_seen`/`active`. Deleting a tracker cascades its
  memberships and orphan listings (no remaining membership) are removed.
- **`price_history`** — append-only price observations per listing; `price` is
  NULL for non-numeric prices (VB / "Zu verschenken"), raw label in `price_text`.
- **`settings`** — `poll_cron` (default `0 4 * * *`), `search_pages`.

`init_db()` drops & recreates the data tables if it detects a pre-multi-source
schema (no `trackers.source` column). No backward compatibility is maintained.

## Source scraping notes

### Kleinanzeigen
- Location ids: `GET /s-ort-empfehlungen.json?query=...` (Miesbach → `10929`).
- Search URL: `/s-{keyword}/k0l{locId}r{radius}`, paged via `/s-seite:{n}/...`.
- Listing fields: `#viewad-title`, `#viewad-price`, `#viewad-locality`,
  `og:image`. Removed ad 404s or lacks `#viewad-title`. Search cards:
  `article.aditem` with `data-adid` / `data-href`. Plain headers suffice.

### ImmoScout24
- Locations: `GET https://www.immobilienscout24.de/geoautocomplete/v3/locations.json?i=…`
  → `entity.geopath.uri` (e.g. `/de/bayern/miesbach-kreis/miesbach`) used as the
  geocode.
- Search: `GET https://api.mobile.immobilienscout24.de/search?searchType=region&geocodes=<uri>&realestatetype=<type>&pagenumber=&pagesize=`
  with a **mobile User-Agent** (`ImmoScout24_1410_30_._`); paginate via
  `numberOfPages`. Item fields: `id`, `title`, `address.line`, first `attributes`
  entry with `€`, `titlePicture.preview`.
- The website HTML (`/expose`, `/Suche`) and the mobile `/expose/{id}` detail are
  bot/WAF-blocked → single listings unsupported.

## Deployment (Ubuntu VPS, webapps@168.119.115.32)

Runs alongside other apps under `/var/www`, each as a gunicorn server bound
directly to a port (8000, 8001 taken → crapper uses **8002**). Example:

```bash
cd /var/www/crapper && python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/gunicorn --bind 0.0.0.0:8002 --workers 1 \
  --access-logfile access.log --error-logfile error.log app:app
```

`app.py` auto-spawns `poll.py` on first import, so the scheduler starts with the
web server.

Run with **`--workers 1`**: the "refresh in progress" set (`_active_refreshes`,
used by `/refresh_status` for the frontpage auto-reload) lives in process memory,
so multiple workers would report it inconsistently. The app is light enough that
one worker is fine; scaling out would need a shared store (e.g. the DB).
