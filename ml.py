"""Fair-price model: features of a listing -> expected price.

Features live in `listing_features` (extracted manually, see scripts/features/).
`train(feature_set_name)` encodes them, picks the best of a few small regressors by
repeated cross-validation, fits it on all usable listings and stores model +
predictions (`ml_models`, `fair_prices`). Training listings get their *out-of-fold*
prediction, so "fair price vs. asking price" is never an in-sample fit.

    python ml.py train wohnung_miete_v1
"""
import io
import math
import sys

import numpy as np

import db

MIN_TRAIN = 30                      # refuse to train on fewer priced listings
# Plausible €/unit window for target-price sanity filtering, keyed by feature-set
# `price_per` (a numeric feature the price is normally proportional to).
SANE_PER_UNIT = {'wohnflaeche': (4.0, 40.0)}


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


def _sane(row, per):
    p = row['price']
    if not p or p <= 0:
        return False
    if per:
        base = row['features'].get(per)
        lo, hi = SANE_PER_UNIT.get(per, (0, math.inf))
        return bool(base) and lo <= p / base <= hi
    return True


def train(feature_set_name: str, per: str | None = 'wohnflaeche') -> dict:
    import joblib
    from sklearn.model_selection import KFold, cross_val_predict

    fs = db.get_feature_set(feature_set_name)
    if not fs:
        raise SystemExit(f'unknown feature set {feature_set_name!r}')
    features = fs['definition']['features']
    rows = db.get_training_rows(fs['id'])
    train_rows = [r for r in rows if _sane(r, per)]
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
        if lid in oof_by_id:
            preds[lid] = (oof_by_id[lid], True)
        else:
            preds[lid] = (float(np.exp(model.predict(enc.transform([r['features']]))[0])), False)

    imp = _importances(model, best, X, y, enc.names)
    metrics = {'chosen': best, 'candidates': scores, 'n_features': len(enc.names),
               'n_excluded': len(rows) - len(train_rows), 'importance': imp}
    buf = io.BytesIO()
    joblib.dump({'encoder': enc, 'model': model, 'algo': best}, buf)
    mid = db.save_model(fs['id'], best, len(train_rows), metrics, buf.getvalue(), preds)
    print(f'model {mid}: {best}  n={len(train_rows)} (excluded {len(rows) - len(train_rows)})')
    for n, s in scores.items():
        print(f'  {n:18} MAE {s["mae"]:7.1f} €  median |err| {s["median_ape"]*100:4.1f} %  R² {s["r2"]:.3f}'
              + ('  <- chosen' if n == best else ''))
    print('  top features:', ', '.join(f'{k} ({v:.3f})' for k, v in imp[:8]))
    return metrics


def _importances(model, algo, X, y, names):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=8, random_state=0,
                               scoring='neg_mean_absolute_error')
    order = np.argsort(-r.importances_mean)
    return [(names[i], round(float(r.importances_mean[i]), 4)) for i in order[:15]]


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'train':
        db.init_db()
        train(sys.argv[2])
    else:
        print(__doc__)
