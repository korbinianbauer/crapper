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
SANE_PER_UNIT = {'wohnflaeche': (4.0, 40.0)}      # default (rent); a feature set can override via definition['plausible']


# ── normalised rent ("Warmmiete inkl. Heizung, Strom, Internet, Wasser, Müll, 1 PKW-Stellplatz") ──
# Per listing the LLM agent stores `_cost` next to the features:
#   {"basis": "kalt"|"warm", "add": € NK+Heizung on top of the price (basis kalt; null = unknown),
#    "incl": ["strom","internet","stellplatz"] already contained in price(+add),
#    "extra": ["heizung","wasser","muell"] NOT contained although normally part of the NK,
#    "sp": monthly price of a parking space if it is rented separately}
# Missing items are added with the feature set's assumptions (settings page), so every listing is
# compared on the same footing. The model is trained on this normalised price.
DEFAULT_COSTS = {'strom': 50.0, 'internet': 30.0, 'stellplatz': 60.0, 'wasser': 25.0, 'muell': 10.0,
                 'heizung_m2': 1.5, 'nk_m2': 3.0}
COST_LABELS = {'strom': 'Strom', 'internet': 'Internet', 'stellplatz': 'PKW-Stellplatz', 'wasser': 'Wasser',
               'muell': 'Müllabfuhr', 'heizung': 'Heizung'}


def effective_breakdown(price, cost, features, assume):
    """(total, [(label, €)]) — the normalised monthly rent, or (price, []) without cost data."""
    if not cost or price is None or not assume:
        return price, []
    a = {**DEFAULT_COSTS, **assume}
    wf = (features or {}).get('wohnflaeche') or 0
    parts = [('Angebotspreis', price)]
    add = cost.get('add')
    if cost.get('basis') == 'kalt':
        if add is None:
            parts.append(('Nebenkosten (geschätzt)', round(a['nk_m2'] * wf)))
        elif add:
            parts.append(('Nebenkosten/Heizung lt. Anzeige', add))
    incl, extra = set(cost.get('incl') or []), set(cost.get('extra') or [])
    for k in ('strom', 'internet', 'stellplatz'):
        if k in incl:
            continue
        amt = cost.get('sp') if k == 'stellplatz' and cost.get('sp') is not None else a[k]
        parts.append((COST_LABELS[k] + ' (zzgl.)', amt))
    if 'heizung' in extra:
        parts.append(('Heizung (zzgl.)', round(a['heizung_m2'] * wf)))
    for k in ('wasser', 'muell'):
        if k in extra:
            parts.append((COST_LABELS[k] + ' (zzgl.)', a[k]))
    return sum(v for _, v in parts), parts


def effective_price(price, cost, features, assume):
    return effective_breakdown(price, cost, features, assume)[0]


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
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    return {
        'ridge': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'svr': lambda: make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.05, gamma='scale')),
        'random_forest': lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                       random_state=0, n_jobs=1),
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
    for r in rows:
        r['raw_price'] = r['price']
        r['price'] = effective_price(r['price'], r.get('cost'), r['features'], assume) if r['price'] else r['price']
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
    metrics = {'chosen': best, 'per': per, 'per_range': rng, 'cost_assumptions': assume, 'candidates': scores, 'n_features': len(enc.names),
               'n_excluded': len(rows) - len(train_rows), 'importance': imp}
    buf = io.BytesIO()
    joblib.dump({'encoder': enc, 'model': model, 'algo': best}, buf)
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
