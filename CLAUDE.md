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

**crapper** is a multi-source price tracker. The user just **pastes a URL** — a
single ad or a search-results page on any supported site — and crapper auto-detects
the source and type and creates a tracker. A daily poll scrapes each, archives
every price observation, and the frontpage plots each listing's price history.
Listings that disappear from their source are kept and shown with a red border.
Kleinanzeigen and ImmoScout24 are implemented; adding a site is a matter of
writing one `Source` subclass.

## Architecture

### Sources (the extension point)

Everything is **URL-driven**: the user pastes one URL and the source recognises
whether it's a single listing or a search and extracts all parameters from the
URL itself — no site-specific input fields. A **Source** (`sources/base.py`)
implements:

- `matches_url(url)` — does this URL belong to the site
- `classify(url) -> UrlInfo` — listing vs search, from the URL alone (no network);
  `UrlInfo` carries `type` / `supported` / `note` / `label` / `ad_id`
- `fetch_listing(url)` — single ad → `ScrapedListing`
- `fetch_search(url, max_pages)` — walk the search → `list[ScrapedListing]`
- `describe_search(url) -> (count, label)` — quick paste-time preview

Register a new source by appending an instance to `_all` in
`sources/__init__.py`. A source that can't support a URL type returns
`UrlInfo(supported=False, note=…)`.

Implemented sources:
- **`kleinanzeigen`** — classifieds; search **and** single listings. Plain
  browser-like HTTP + BeautifulSoup. Search pagination injects `seite:N` after
  `/s-`; count comes from the results `<h1>`.
- **`immoscout`** — real estate; search **and** single listings (via the mobile
  `/expose/{id}` API; the website expose pages are WAF-blocked).
  A pasted web search URL (`/Suche/de/<land>/<kreis>[/<ort>]/<objektart-slug>` or
  a `/Suche/radius/…?geocoordinates=lat;lon;r` URL) is parsed and translated into
  a mobile-API call (`api.mobile.immobilienscout24.de/search`, mobile UA);
  `totalResults` gives the exact preview count.

### Add flow

`GET /inspect?url=` recognises the URL and returns `{ok, source, type, label,
count, summary}` (or an error) — the frontend calls it live as the user pastes.
`POST /add` re-validates and creates the tracker, then kicks off a background poll.

### Data flow

1. `poll.py` (standalone process, auto-started by `app.py`) runs an APScheduler
   cron job — schedule from the `poll_cron` setting, reloaded via SIGHUP. Each
   fire runs `poll_all_due()` in a fresh subprocess so it uses the latest code.
2. `poller.poll_tracker()` looks up the tracker's source, then:
   - **listing** → `source.fetch_listing(url)`; if gone,
     `db.mark_tracker_listings_inactive()` flags this tracker's membership.
   - **search** → `source.fetch_search(url)`; each ad is upserted; memberships
     not seen this run are flagged via `db.deactivate_missing()`. Ad pages of the
     hits are fetched afterwards in a shared, prioritised queue (see Kleinanzeigen notes).
3. `db.upsert_listing(source, ad_id, …)` refreshes/creates the listing (unique by
   `(source, ad_id)`); `db.link_listing()` records the tracker→listing
   membership; `db.add_price()` appends a `price_history` row (skipping
   consecutive duplicate prices).
4. Deactivation recomputes each listing's `active` flag: active iff ≥1 membership
   is active. Adding a tracker triggers an immediate background poll
   (`_refresh_async`).

### Tracker detail view (`/tracker/<id>`)

Rendered client-side from `CARDS`/`TRACKERS` (index payload). Each listing card shows its
price and the ML rating (`fair_delta` = asking price vs. fair price in %, negative = cheaper;
`None` when the price is implausible per `ml.is_plausible`; fair price from
`db.get_fair_price_map()`). Sorting is URL-driven: `?sort=date|price|rating&dir=asc|desc`
(defaults: date desc, price/rating asc; items without a value always last) and combines with
`?gone=1`; all changes go through `setParams()`.

### Key files

| File | Role |
|------|------|
| `app.py` | Flask routes, auth/CSRF, poll-process management, index payload |
| `db.py` | All SQLite access; `init_db()` (wipe-migrates pre-multi-source schemas) |
| `sources/base.py` | `Source` ABC + `ScrapedListing` / `UrlInfo` |
| `sources/__init__.py` | `REGISTRY`, `get`, `all_sources`, `source_for_url` |
| `sources/kleinanzeigen.py`, `sources/immoscout.py` | the two sources |
| `pagearchive.py` | archives a stored page's CSS/fonts/images (Wayback-style) + helpers to rewrite them |
| `ml.py` | fair-price model: `Encoder`, CV model selection, `train(feature_set)`; CLI `python ml.py train <set>` |
| `poller.py` | `poll_tracker()` / `poll_all_due()` — orchestrates source → DB |
| `poll.py` | APScheduler cron loop (separate process) |
| `templates/settings_fair_pricing.html` | Settings page (login only, `/settings`, sub-pages in `SETTINGS_PAGES`): train a model per feature set/tracker (background thread), choose the active model, set the display offset |
| `templates/index.html` | Listing grid + Plotly charts + single add-by-URL modal with live `/inspect` preview |

### Database schema (`crapper.db`)

- **`trackers`** — one per user request; `source` slug + `type` = `listing` /
  `search`, both defined by their `url` (the search URL is paginated directly).
- **`listings`** — discovered ads, **unique by `(source, ad_id)`** so an ad
  surfaced by several trackers is stored/shown once. `active=0` means no
  referencing tracker found it in its latest poll.
- **`tracker_listings`** — many-to-many membership (which trackers surfaced which
  listing) with per-tracker `last_seen`/`active`. Deleting a tracker cascades its
  memberships and orphan listings (no remaining membership) are removed.
- **`price_history`** — append-only price observations per listing; `price` is
  NULL for non-numeric prices (VB / "Zu verschenken"), raw label in `price_text`.
- **`listing_versions`** — append-only archive of the ad content (title, location,
  description, image, zlib'd original HTML for single-ad fetches). A new version is
  written only when content changes; fields a scrape lacks are carried over from the
  previous version. Empty titles / placeholder images never overwrite stored values
  (`db.upsert_listing`). Images are downloaded to `archive/images/` (`archive.py`)
  and served via `/archive/img/<file>`; the UI falls back to the remote URL.
- **`assets`** / **`version_assets`** — the archived page is *self-contained*
  (`pagearchive.py`): after each poll `archive_pending()` fetches every stylesheet
  (recursively: `@import`, `url()`), font and image a stored `raw_html` references into
  content-addressed `archive/assets/<hash>` (stylesheets stored with refs rewritten to
  `archive-asset:<hash>` tokens). `version_assets` maps each version's original URLs to
  hashes, so old versions keep the look they had. `/listing/<id>/version/<v>/page`
  serves the page (scripts stripped, sandboxed) with URLs rewritten to
  `/archive/asset/<hash>`. Only css/image/font types are stored; fetches are SSRF-guarded.
- **`feature_sets`** / **`listing_features`** / **`ml_models`** / **`fair_prices`** — the ML
  "fair price" (`ml.py`). A feature set (≤30 features, JSON definition: key/label/type
  num|bool|cat) belongs to one tracker; values per listing are extracted **by an LLM agent
  reading each ad by hand — there is deliberately no extraction script/regex**; the agent writes the values straight into the DB
  via `db.save_listing_features(feature_set_id, {listing_id: {feature: value}})` (inline Python on the server, venv).
  `python ml.py train <set>` trains a gradient boosting model (the only candidate; ridge/SVR/random forest were removed), scored by
  repeated 5-fold CV on log(price), stores the model (last 10 per set are kept) and writes
  `fair_prices`: training listings get their out-of-fold prediction, all others (excluded
  or new) the stored model's — so every stored fair price is made without knowing that
  listing's price (`oof` is always 1; kept for schema compatibility). Listings with implausible prices (€/m² outside 4–60) are excluded from training.
  Training is triggered by hand (settings page or CLI); new listings are scored with the **active** model (`ml.predict`,
  no retraining) — run `ml.predict(<set>, ids)` right after saving their features
  (or `python ml.py predict <set>` for any with features but no prediction). The archive view
  (`/listing/<id>/archive`) shows fair vs. current price plus a collapsible table of all
  evaluated features. Currently only set `wohnung_miete_v1` (KA tracker 10, rental flats)
  exists; a listing gets a fair price once its features have been extracted and imported.
  Models are pickled via joblib; `ml.py` is always run as an importable module
  (`_ml.train`) and `_load_artifact` also loads older models pickled under `__main__`.
  Workflow for new listings (run on the server, venv):
  read the new listings' texts (ad description + Details block) yourself, save them with
  `db.save_listing_features`, then `ml.predict`. Read negations carefully ("kein Aufzug", "ohne
  Einbauküche"), ignore wish lists in Tausch ads, skip wanted ads ("Suche …"); place→km distances are
  estimated by the agent from the town.
- **Normalised rent (`_cost`)**: for rent feature sets the agent also stores, per listing, next to the features
  `data["_cost"] = {basis: "kalt"|"warm", add: €NK+Heizung on top of the price (null = unknown), incl: [strom|internet|stellplatz]
  contained in the price, extra: [heizung|wasser|muell] NOT contained, sp: separate parking price,
  heat: {carrier: gas|oel|fernwaerme|strom|waermepumpe|pellets|holz|null, kwh: Endenergie kWh/(m²·a) from the ad|null}}`. Listings are always
  compared as **Warmmiete incl. Heizung, Strom, Internet, Wasser, Müll, 1 PKW-Stellplatz**: `ml.effective_price` adds the
  missing items. Heating = kWh/m² (from the ad, else a GB sub-model estimates the energy demand: `kwh` with Energieklasse, `kwh_baujahr` without it — Baujahr/saniert/renoviert/neubau/Energieträger; both trained on the flats of **all** feature sets with an `energieklasse` feature, rent and sale (`db.get_heat_rows`), `kwh_baujahr` also on Energieklasse-only ads with the class's nominal kWh as target) × m² × energy price of the carrier;
  unknown Nebenkosten = ML-estimated Betriebskosten (sub-model) + Heizung; Strom = Verbrauch × Preis. Energy prices are entered
  per billing unit (€/kWh, €/Liter, €/kg, €/Raummeter) on the settings page (`fair_cost_<fs_id>`, defaults in `ml.DEFAULT_COSTS`).
  The two sub-models are trained with each price model (`ml.fit_cost_models`) and stored in its artifact (`cost_models`). The model is
  trained on that normalised price, stores the assumptions in its metrics, and ratings (`fair_delta`, archive) use the same
  assumptions — so changed assumptions only apply from the next training. Sale feature sets have no `_cost`.
- **`listing_predictions`** — persisted results of each feature set's **active** model per listing (PK listing+feature set,
  JSON: `fair_price` raw, `heat_kwh_m2` + `heat_kwh_src` (GB estimate — always the model's own value, even if the ad states one),
  `heat_used` (anzeige|prognose), `heat_eur`, `heat_eur_kwh`, `nk_eur`/`nk_src`). Written by `ml.refresh_predictions(fs_id[, ids])`:
  after `ml.train`, after `ml.predict` (new listings), on choosing/deleting the active model (settings); CLI `python ml.py refresh [set]`.
  Pricing/rating use extracted values (`_cost`) where available, these predictions as fallback. Shown on the archive page next to
  the extracted features.
- **Fair-price settings** (`/settings/fair-pricing`, keys in `settings`): `fair_model_<feature_set_id>` =
  active model (default: newest), `fair_offset_<feature_set_id>` = manual offset in % (-10 → shown fair
  price is 10 % below the ML prediction, to bridge asking prices vs. real deals). The offset is applied
  at read time (`db.get_fair_price_map` / `get_fair_price`); `fair_prices` always stores the raw prediction.
- **`settings`** — `poll_cron` (default `0 4 * * *`), `search_pages`, `detail_block_<source>` (JSON
  `{level, until}`: persistent backoff for ad-page fetches after HTTP 403/429 — `RateLimited` from the source;
  suspension 10 min · 2^(level-1), max 24 h, rest of the poll continues; reset by the next successful fetch).

`init_db()` drops & recreates the data tables if it detects a pre-multi-source
schema (no `trackers.source` column). No backward compatibility is maintained.

## Source scraping notes

### Kleinanzeigen
- Location ids: `GET /s-ort-empfehlungen.json?query=...` (Miesbach → `10929`).
- Search URL: `/s-{keyword}/k0l{locId}r{radius}`, paged via `/s-seite:{n}/...`.
- Listing fields: `#viewad-title`, `#viewad-price`, `#viewad-locality`,
  `og:image`. Removed ad 404s or lacks `#viewad-title`. Search cards are matched
  by `article[data-adid]` (covers old `article.aditem` and the newer card layout).
  Parse with **lxml** — the stdlib `html.parser` silently drops most cards on some
  category pages. Plain headers suffice.
- **Result count is fake**: the `… von N …` heading is artificially inflated and
  does not match the number of distinct ads — this affects real browser users too,
  not just scrapers (KA over-counts, mainly via repeated promoted/Top ads). So
  `fetch_search` returning fewer than the preview count is **expected, not a bug**
  — the gap is non-existent ads; do not try to "close" it.

- Search cards (newer layout): the `<a>` only wraps the image (text = photo count);
  title + image URL come from the embedded `ld+json` ImageObject.
- Search hits get their full ad page fetched + archived at most every 20 h
  (`Source.archive_details`, `poller._fetch_detail`): all images, description,
  details, original HTML → `listing_versions` + `archive/images/`.
  A poll runs in **two phases** (`poller._poll`): first all trackers' searches
  (prices, memberships, deactivation), then one queue (`poller._drain`) of the
  single-listing trackers followed by the due ad pages (`db.detail_queue`), each
  ordered by `listings.detail_tried_at` ascending (never tried first) — a
  round-robin, so permanently failing pages move to the back. A 403/429 does not
  count as an attempt, unless the same ad also hit the previous block, and never
  marks an ad inactive. While a source is blocked its items are held back and the
  other sources continue; when only held items are left the poll **waits for the
  block to end and resumes** (blocked ad first), unless the block ends more than
  `MAX_POLL_WAIT` (4 h) after the poll started — then the rest is left for the next
  poll, where it comes first. Price rescue for search cards without a price
  (`poll_stats.rescued`) happens in phase 2. Both sources raise `RateLimited` on
  HTTP 403/429 from ad pages.

### ImmoScout24
- Locations: `GET https://www.immobilienscout24.de/geoautocomplete/v3/locations.json?i=…`
  → `entity.geopath.uri` (e.g. `/de/bayern/miesbach-kreis/miesbach`) used as the
  geocode.
- Search: `GET https://api.mobile.immobilienscout24.de/search?searchType=region&geocodes=<uri>&realestatetype=<type>&pagenumber=&pagesize=`
  with a **mobile User-Agent** (`ImmoScout24_1410_30_._`); paginate via
  `numberOfPages`. Item fields: `id`, `title`, `address.line`, first `attributes`
  entry with `€`, `titlePicture.preview`.
- The website HTML (`/expose`, `/Suche`) is bot/WAF-blocked, but the mobile
  `https://api.mobile.immobilienscout24.de/expose/{id}` (same mobile UA) works and
  returns the full expose JSON (MEDIA, TOP_ATTRIBUTES, ATTRIBUTE_LIST, TEXT_AREA, MAP, …);
  it is archived verbatim as `raw_html` (JSON).

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
