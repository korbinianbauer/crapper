"""Fair-price model: features of a listing -> expected price.

Features live in `listing_features` (extracted manually by an LLM agent, no script).
`train(feature_set_name)` encodes them, picks the best of a few small regressors by
repeated cross-validation, fits it on all usable listings and stores model +
predictions (`ml_models`, `fair_prices`). Training listings get their *out-of-fold*
prediction, so "fair price vs. asking price" is never an in-sample fit.

Training is triggered on the settings page (or the CLI); every model is kept and one is
chosen as active there. New listings are scored with the active model by `predict()`
as soon as their features have been extracted and imported (no retraining).

    python ml.py train wohnung_miete_v1      # one-off
    python ml.py predict wohnung_miete_v1    # score listings that have features but no prediction
"""
import io
import math
import sys

import numpy as np

import db

MIN_TRAIN = 30                      # refuse to train on fewer priced listings
# Plausible €/unit window for target-price sanity filtering, keyed by feature-set
# `price_per` (a numeric feature the price is normally proportional to).
SANE_PER_UNIT = {'wohnflaeche': (4.0, 60.0)}      # default (rent); a feature set can override via definition['plausible']


# ── normalised rent ("Warmmiete inkl. Heizung, Strom, Internet, Wasser, Müll, 1 PKW-Stellplatz") ──
# Per listing the LLM agent stores `_cost` next to the features:
#   {"basis": "kalt"|"warm", "add": € NK+Heizung on top of the price (basis kalt; null = unknown),
#    "incl": ["strom","internet","stellplatz"] already contained in price(+add),
#    "extra": ["heizung","wasser","muell"] NOT contained although normally part of the NK,
#    "sp": monthly price of a parking space if it is rented separately,
#    "heat": {"carrier": gas|oel|fernwaerme|strom|waermepumpe|pellets|holz|null, "kwh": Endenergie kWh/(m²·a) from the ad|null}}
# Missing items are added so every listing is compared on the same footing:
#   * Heizung = kWh/m² (ad, else ML model from Energieklasse/Baujahr/Energieträger) × m² × energy price / 12
#   * unknown Nebenkosten = ML-estimated Betriebskosten + Heizung
#   * Strom = Verbrauch × Strompreis; Internet/Stellplatz/Wasser/Müll = fixed amounts
# Energy prices are given per billing unit (kWh, Liter, kg, Raummeter) on the settings page.
DEFAULT_COSTS = {'strom_kwh': 0.35, 'strom_verbrauch': 2000.0, 'gas_kwh': 0.11, 'oel_l': 1.10,
                 'fernwaerme_kwh': 0.13, 'pellets_kg': 0.40, 'holz_rm': 120.0,
                 'internet': 30.0, 'stellplatz': 60.0, 'wasser': 25.0, 'muell': 10.0}
KWH_PER = {'oel_l': 10.0, 'pellets_kg': 4.8, 'holz_rm': 2000.0}      # kWh per billing unit
COST_LABELS = {'strom': 'Strom', 'internet': 'Internet', 'stellplatz': 'PKW-Stellplatz', 'wasser': 'Wasser',
               'muell': 'Müllabfuhr', 'heizung': 'Heizung'}
CLASS_KWH = {0: 15, 1: 40, 2: 65, 3: 90, 4: 115, 5: 145, 6: 180, 7: 225, 8: 270}   # Energieklasse -> kWh/(m²a)
HEAT_FEATURES = [{'key': 'energieklasse', 'type': 'num'}, {'key': 'baujahr', 'type': 'num'},
                 {'key': 'neubau', 'type': 'bool'}, {'key': 'saniert', 'type': 'bool'},
                 {'key': 'carrier', 'type': 'cat'}]
NK_FEATURES = [{'key': 'wohnflaeche', 'type': 'num'}, {'key': 'zimmer', 'type': 'num'}, {'key': 'baujahr', 'type': 'num'},
               {'key': 'etage', 'type': 'num'}, {'key': 'typ', 'type': 'cat'}, {'key': 'aufzug', 'type': 'bool'},
               {'key': 'garten', 'type': 'bool'}, {'key': 'balkon', 'type': 'bool'}, {'key': 'neubau', 'type': 'bool'},
               {'key': 'hochwertig', 'type': 'bool'}, {'key': 'carrier', 'type': 'cat'}]
MIN_HEAT, MIN_NK = 12, 20


def kwh_price(carrier, a):
    """€ per kWh for an energy carrier (billing-unit prices from the settings converted to kWh)."""
    p = {'gas': a.get('gas_kwh'), 'fernwaerme': a.get('fernwaerme_kwh'),
         'strom': a.get('strom_kwh'), 'waermepumpe': a.get('strom_kwh'),
         'oel': (a.get('oel_l') or 0) / KWH_PER['oel_l'] if a.get('oel_l') else None,
         'pellets': (a.get('pellets_kg') or 0) / KWH_PER['pellets_kg'] if a.get('pellets_kg') else None,
         'holz': (a.get('holz_rm') or 0) / KWH_PER['holz_rm'] if a.get('holz_rm') else None}.get(carrier)
    if p is None:                            # unknown carrier: average of gas and oil
        gas = a.get('gas_kwh') or 0.11
        oil = (a.get('oel_l') or 1.10) / KWH_PER['oel_l']
        p = (gas + oil) / 2
    return p


def _cost_row(features, cost):
    return {**(features or {}), 'carrier': ((cost or {}).get('heat') or {}).get('carrier')}


def _fit_small(rows, y, features, minimum):
    """Tiny regression for the cost sub-models: gradient boosting, CV MAE."""
    from sklearn.model_selection import KFold, cross_val_predict
    if len(rows) < minimum:
        return None
    enc = Encoder(features).fit(rows)
    X, y = enc.transform(rows), np.asarray(y, float)
    best = None
    for name, make in _candidates().items():
        pred = cross_val_predict(make(), X, np.log(y), cv=KFold(5, shuffle=True, random_state=0))
        mae = float(np.mean(np.abs(np.exp(pred) - y)))
        if best is None or mae < best[0]:
            best = (mae, name, make)
    mae, name, make = best
    return {'enc': enc, 'model': make().fit(X, np.log(y)), 'algo': name, 'n': len(rows), 'mae': round(mae, 2)}


def _predict_small(cm, row):
    if not cm:
        return None
    return float(np.exp(cm['model'].predict(cm['enc'].transform([row]))[0]))


def heat_kwh_m2(cost, features, cmodels):
    """kWh/(m²·a): stated in the ad, else ML estimate, else the nominal value of the Energieklasse, else None."""
    h = (cost or {}).get('heat') or {}
    if h.get('kwh'):
        return float(h['kwh'])
    est = _predict_small((cmodels or {}).get('kwh'), _cost_row(features, cost))
    if est:
        return est
    ek = (features or {}).get('energieklasse')
    return float(CLASS_KWH[int(ek)]) if ek is not None else None


def heating_eur(cost, features, a, cmodels):
    wf = (features or {}).get('wohnflaeche') or 0
    kwh = heat_kwh_m2(cost, features, cmodels) or 130.0       # unknown everything: typical stock value
    carrier = ((cost or {}).get('heat') or {}).get('carrier')
    return kwh * wf * kwh_price(carrier, a) / 12


def fit_cost_models(rows, a):
    """ML sub-models for the normalisation: heating energy (kWh/m²a) and Betriebskosten ohne Heizung."""
    heat_rows = [(_cost_row(r['features'], r['cost']), r['cost']['heat']['kwh']) for r in rows
                 if r.get('cost') and (r['cost'].get('heat') or {}).get('kwh')]
    kwh = _fit_small([x for x, _ in heat_rows], [y for _, y in heat_rows], HEAT_FEATURES, MIN_HEAT)
    cm = {'kwh': kwh}
    nk_rows, nk_y = [], []
    for r in rows:
        c = r.get('cost')
        if not c or c.get('basis') != 'kalt' or c.get('add') is None or c['add'] <= 0:
            continue
        y = c['add'] if 'heizung' in (c.get('extra') or []) else c['add'] - heating_eur(c, r['features'], a, cm)
        nk_rows.append(_cost_row(r['features'], c)); nk_y.append(max(y, 15.0))
    cm['nk'] = _fit_small(nk_rows, nk_y, NK_FEATURES, MIN_NK)
    return cm


def effective_breakdown(price, cost, features, assume, cmodels=None):
    """(total, [(label, €)]) — the normalised monthly rent, or (price, []) without cost data."""
    if not cost or price is None or not assume:
        return price, []
    a = {**DEFAULT_COSTS, **assume}
    legacy = 'strom' in assume and 'strom_kwh' not in assume          # models trained before energy prices
    wf = (features or {}).get('wohnflaeche') or 0
    parts = [('Angebotspreis', price)]
    incl, extra = set(cost.get('incl') or []), set(cost.get('extra') or [])

    def heat():
        return a['heizung_m2'] * wf if legacy else heating_eur(cost, features, a, cmodels)

    add = cost.get('add')
    if cost.get('basis') == 'kalt':
        if add is None:
            if legacy:
                parts.append(('Nebenkosten (geschätzt)', round(a['nk_m2'] * wf)))
            else:
                nk = _predict_small((cmodels or {}).get('nk'), _cost_row(features, cost))
                parts.append(('Betriebskosten (geschätzt)', round(nk if nk else 2.0 * wf)))
                parts.append(('Heizung (geschätzt)', round(heat())))
        elif add:
            parts.append(('Nebenkosten/Heizung lt. Anzeige', add))
    for k in ('strom', 'internet', 'stellplatz'):
        if k in incl:
            continue
        if k == 'stellplatz' and cost.get('sp') is not None:
            amt = cost['sp']
        elif k == 'strom' and not legacy:
            amt = a['strom_verbrauch'] * a['strom_kwh'] / 12
        else:
            amt = a[k]
        parts.append((COST_LABELS[k] + ' (zzgl.)', round(amt)))
    if 'heizung' in extra:
        parts.append(('Heizung (zzgl.)', round(heat())))
    for k in ('wasser', 'muell'):
        if k in extra:
            parts.append((COST_LABELS[k] + ' (zzgl.)', a[k]))
    return sum(v for _, v in parts), parts


def effective_price(price, cost, features, assume, cmodels=None):
    return effective_breakdown(price, cost, features, assume, cmodels)[0]


_CM_CACHE = {}


def cost_models_for(model_id):
    """Cost sub-models stored inside a trained price model (cached)."""
    if model_id not in _CM_CACHE:
        row = db.get_model(model_id) if model_id else None
        try:
            _CM_CACHE[model_id] = (_load_artifact(row['artifact']).get('cost_models') if row else None)
        except Exception:
            _CM_CACHE[model_id] = None
    return _CM_CACHE[model_id]


class Encoder:
    """Feature dicts -> numeric matrix. num: median-imputed (+ missing flag when
    sometimes missing); bool: 0/1; cat: one-hot (unknown/None -> all zeros)."""

    def __init__(self, features):
        self.features = features

    def fit(self, rows):
        self.median, self.cats, self.flag = {}, {}, []
        for f in self.features:
            vals = [r.get(f['key']) for r in rows]
            if f['type'] == 'num':
                have = [v for v in vals if v is not None]
                self.median[f['key']] = float(np.median(have)) if have else 0.0
                if len(have) < len(vals) and have:
                    self.flag.append(f['key'])
            elif f['type'] == 'cat':
                self.cats[f['key']] = sorted({v for v in vals if v is not None})
        self.names = []
        for f in self.features:
            k = f['key']
            if f['type'] == 'cat':
                self.names += [f'{k}={c}' for c in self.cats[k]]
            else:
                self.names.append(k)
                if k in self.flag:
                    self.names.append(f'{k}?')
        return self

    def transform(self, rows):
        out = []
        for r in rows:
            x = []
            for f in self.features:
                k, v = f['key'], r.get(f['key'])
                if f['type'] == 'cat':
                    x += [1.0 if v == c else 0.0 for c in self.cats[k]]
                elif f['type'] == 'bool':
                    x.append(1.0 if v else 0.0)
                else:
                    x.append(self.median[k] if v is None else float(v))
                    if k in self.flag:
                        x.append(1.0 if v is None else 0.0)
            out.append(x)
        return np.array(out, dtype=float)


def _candidates():
    from sklearn.ensemble import GradientBoostingRegressor
    return {
        'gradient_boosting': lambda: GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=3, subsample=0.8,
            min_samples_leaf=3, random_state=0),
    }


def _sane(row, per, rng=None):
    p = row['price']
    if not p or p <= 0:
        return False
    if per:
        base = row['features'].get(per)
        lo, hi = rng or SANE_PER_UNIT.get(per, (0, math.inf))
        return bool(base) and lo <= p / base <= hi
    return True


def train(feature_set_name: str, per: str | None = None, activate: bool = True) -> dict:
    import joblib
    from sklearn.model_selection import KFold, cross_val_predict

    fs = db.get_feature_set(feature_set_name)
    if not fs:
        raise SystemExit(f'unknown feature set {feature_set_name!r}')
    features = fs['definition']['features']
    # plausibility window (€ per unit of `per`, e.g. €/m²): from the feature set, else rent default
    pl = fs['definition'].get('plausible') or {}
    per = per or pl.get('per') or 'wohnflaeche'
    rng = (pl['min'], pl['max']) if pl.get('per') == per and 'min' in pl else None
    rows = db.get_training_rows(fs['id'])
    # rows with cost data are trained on the normalised rent (see effective_price)
    assume = db.get_cost_assumptions(fs['id']) if any(r.get('cost') for r in rows) else None
    cmodels = fit_cost_models(rows, assume) if assume else None
    for r in rows:
        r['raw_price'] = r['price']
        r['price'] = (effective_price(r['price'], r.get('cost'), r['features'], assume, cmodels)
                      if r['price'] else r['price'])
    train_rows = [r for r in rows if _sane({**r, 'price': r['raw_price']}, per, rng)]
    if len(train_rows) < MIN_TRAIN:
        raise SystemExit(f'only {len(train_rows)} usable priced listings (<{MIN_TRAIN})')

    enc = Encoder(features).fit([r['features'] for r in train_rows])
    X = enc.transform([r['features'] for r in train_rows])
    y = np.log([r['price'] for r in train_rows])
    prices = np.exp(y)

    # Model selection: 5-fold CV, 3 repeats; score = mean abs error in € on out-of-fold preds.
    scores, oofs = {}, {}
    for name, make in _candidates().items():
        reps = []
        for seed in range(3):
            pred = np.exp(cross_val_predict(make(), X, y, cv=KFold(5, shuffle=True, random_state=seed)))
            reps.append(pred)
        pred = np.mean(reps, axis=0)
        mae = float(np.mean([np.mean(np.abs(p - prices)) for p in reps]))
        mape = float(np.mean([np.median(np.abs(p - prices) / prices) for p in reps]))
        r2 = float(np.mean([1 - np.sum((p - prices) ** 2) / np.sum((prices - prices.mean()) ** 2)
                            for p in reps]))
        scores[name] = {'mae': round(mae, 1), 'median_ape': round(mape, 4), 'r2': round(r2, 3)}
        oofs[name] = pred
    best = min(scores, key=lambda n: scores[n]['mae'])

    model = _candidates()[best]().fit(X, y)
    # Predictions: honest out-of-fold for training listings, full model for all others.
    preds = {}
    oof_by_id = {r['listing_id']: float(oofs[best][i]) for i, r in enumerate(train_rows)}
    for r in rows:
        lid = r['listing_id']
        preds[lid] = oof_by_id[lid] if lid in oof_by_id else float(
            np.exp(model.predict(enc.transform([r['features']]))[0]))

    imp = _importances(model, best, X, y, enc.names)
    metrics = {'chosen': best, 'per': per, 'per_range': rng, 'cost_assumptions': assume,
               'cost_models': {k: ({'n': v['n'], 'mae': v['mae'], 'algo': v['algo']} if v else None)
                               for k, v in (cmodels or {}).items()}, 'candidates': scores, 'n_features': len(enc.names),
               'n_excluded': len(rows) - len(train_rows), 'importance': imp}
    buf = io.BytesIO()
    joblib.dump({'encoder': enc, 'model': model, 'algo': best, 'cost_models': cmodels}, buf)
    mid = db.save_model(fs['id'], best, len(train_rows), metrics, buf.getvalue(), preds, activate)
    metrics['model_id'] = mid
    print(f'model {mid}: {best}  n={len(train_rows)} (excluded {len(rows) - len(train_rows)})')
    for n, s in scores.items():
        print(f'  {n:18} MAE {s["mae"]:7.1f} €  median |err| {s["median_ape"]*100:4.1f} %  R² {s["r2"]:.3f}'
              + ('  <- chosen' if n == best else ''))
    print('  top features:', ', '.join(f'{k} ({v:.3f})' for k, v in imp[:8]))
    return metrics


def is_plausible(price, features, per, rng=None) -> bool:
    return _sane({'price': price, 'features': features}, per, tuple(rng) if rng else None)


def _load_artifact(blob: bytes) -> dict:
    """joblib load. Models trained via `python ml.py train` (before the CLI imported
    this module) were pickled with `Encoder` under `__main__`; provide it while loading."""
    import joblib
    main = sys.modules['__main__']
    added = not hasattr(main, 'Encoder')
    if added:
        main.Encoder = Encoder
    try:
        return joblib.load(io.BytesIO(blob))
    finally:
        if added:
            del main.Encoder


def predict(feature_set_name: str, listing_ids=None, model_id: int | None = None) -> int:
    """Score listings with the stored model. Default: every listing that has features
    but no prediction yet. Explicit ids are re-scored (except training listings, whose
    out-of-fold prediction is kept). Returns the number of predictions written."""
    import joblib
    fs = db.get_feature_set(feature_set_name)
    model_row = (db.get_model(model_id) if model_id else db.get_latest_model(fs['id'])) if fs else None
    if not model_row:
        raise SystemExit(f'no trained model for {feature_set_name!r} — run: python ml.py train')
    art = _load_artifact(model_row['artifact'])
    with db.get_db() as conn:
        have = {r[0] for r in conn.execute(
            'SELECT listing_id FROM fair_prices WHERE model_id = ?', (model_row['id'],))}
        trained = {r[0] for r in conn.execute(
            'SELECT listing_id FROM fair_prices WHERE model_id = ? AND predicted_at = '
            '(SELECT trained_at FROM ml_models WHERE id = ?)', (model_row['id'], model_row['id']))}
    rows = db.get_training_rows(fs['id'])
    if listing_ids is None:
        todo = [r for r in rows if r['listing_id'] not in have]
    else:
        wanted = set(listing_ids)
        todo = [r for r in rows if r['listing_id'] in wanted and r['listing_id'] not in trained]
    if not todo:
        return 0
    X = art['encoder'].transform([r['features'] for r in todo])
    pred = np.exp(art['model'].predict(X))
    db.save_predictions(model_row['id'], {r['listing_id']: float(p) for r, p in zip(todo, pred)})
    return len(todo)


def _importances(model, algo, X, y, names):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=8, random_state=0,
                               scoring='neg_mean_absolute_error')
    order = np.argsort(-r.importances_mean)
    return [(names[i], round(float(r.importances_mean[i]), 4)) for i in order[:15]]


if __name__ == '__main__':
    import ml as _ml      # not __main__: classes in the pickled model must be importable as ml.*
    if len(sys.argv) == 3 and sys.argv[1] == 'train':
        db.init_db()
        _ml.train(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'predict':
        db.init_db()
        print(_ml.predict(sys.argv[2]), 'predictions written')
    else:
        print(__doc__)
