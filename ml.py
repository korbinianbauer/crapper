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
import json
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
                 {'key': 'renoviert', 'type': 'bool'}, {'key': 'carrier', 'type': 'cat'}]
HEAT_FEATURES_NO_EK = [f for f in HEAT_FEATURES if f['key'] != 'energieklasse']   # ads without Energieklasse
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


def heat_estimate(cost, features, cmodels):
    """(kWh/(m²·a), source): stated in the ad ('anzeige'), else GB estimate — with Energieklasse if the
    ad has one ('ml'), else from Baujahr/Sanierung/Energieträger ('ml_baujahr') —, else the nominal
    value of the Energieklasse ('energieklasse'), else (None, None)."""
    h = (cost or {}).get('heat') or {}
    if h.get('kwh'):
        return float(h['kwh']), 'anzeige'
    cm, row = cmodels or {}, _cost_row(features, cost)
    ek = (features or {}).get('energieklasse')
    if ek is not None and cm.get('kwh'):
        return _predict_small(cm['kwh'], row), 'ml'
    if ek is None and (cm.get('kwh_baujahr') or cm.get('kwh')):
        return _predict_small(cm.get('kwh_baujahr') or cm['kwh'], row), 'ml_baujahr' if cm.get('kwh_baujahr') else 'ml'
    return (float(CLASS_KWH[int(ek)]), 'energieklasse') if ek is not None else (None, None)


def heat_kwh_m2(cost, features, cmodels):
    return heat_estimate(cost, features, cmodels)[0]


def nk_estimate(cost, features, cmodels):
    """(Betriebskosten ohne Heizung €/Monat, source 'ml'|'pauschal') — GB sub-model, else 2 €/m²."""
    nk = _predict_small((cmodels or {}).get('nk'), _cost_row(features, cost))
    return (nk, 'ml') if nk else (2.0 * ((features or {}).get('wohnflaeche') or 0), 'pauschal')


def heating_eur(cost, features, a, cmodels):
    wf = (features or {}).get('wohnflaeche') or 0
    kwh = heat_kwh_m2(cost, features, cmodels) or 130.0       # unknown everything: typical stock value
    carrier = ((cost or {}).get('heat') or {}).get('carrier')
    return kwh * wf * kwh_price(carrier, a) / 12


def fit_cost_models(rows, a, heat_rows=None):
    """ML sub-models for the normalisation: heating energy (kWh/m²a) and Betriebskosten ohne Heizung.
    The heating models learn from `heat_rows` (all flats of all feature sets, rent and sale – the building
    is what matters; default: `rows`): `kwh` from ads stating kWh/m², `kwh_baujahr` (no Energieklasse,
    for ads without one) additionally from ads with only an Energieklasse (target = its nominal kWh)."""
    stated, nominal = [], []
    for r in (heat_rows if heat_rows is not None else rows):
        x, kwh = _cost_row(r['features'], r.get('cost')), ((r.get('cost') or {}).get('heat') or {}).get('kwh')
        ek = r['features'].get('energieklasse')
        if kwh:
            stated.append((x, kwh))
        elif ek is not None:
            nominal.append((x, CLASS_KWH[int(ek)]))
    cm = {'kwh': _fit_small([x for x, _ in stated], [y for _, y in stated], HEAT_FEATURES, MIN_HEAT),
          'kwh_baujahr': _fit_small([x for x, _ in stated + nominal], [y for _, y in stated + nominal],
                                    HEAT_FEATURES_NO_EK, MIN_HEAT)}
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
                if 'heizung' not in extra:            # else added below as 'Heizung (zzgl.)'
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


def rent_breakdown(price, cost, features, assume, cmodels=None):
    """The normalised Warmmiete split up from the Kaltmiete, for the archive bar chart:
    {kalt, kalt_est, parts: [{key, label, eur, assumed}], flags: [{label, eur}], total} or None.
    Basis kalt: Kaltmiete = asking price. Basis warm: asking price minus the (estimated) Nebenkosten,
    Heizung, Strom and Internet it contains. `parts` are ordered so that everything contained in the
    asking price (+ stated Nebenkosten) comes first; `flags` are the amounts stated in the ad, at their
    cumulative position. `assumed` = not priced in the ad (flat-rate assumption or ML estimate).
    `total` equals effective_breakdown()'s total."""
    if not cost or price is None or not assume:
        return None
    a = {**DEFAULT_COSTS, **assume}
    legacy = 'strom' in assume and 'strom_kwh' not in assume          # models trained before energy prices
    wf = (features or {}).get('wohnflaeche') or 0
    incl, extra = set(cost.get('incl') or []), set(cost.get('extra') or [])
    warm = cost.get('basis') != 'kalt'
    add = cost.get('add')
    strom = a['strom'] if legacy else a['strom_verbrauch'] * a['strom_kwh'] / 12

    def heat():
        return a['heizung_m2'] * wf if legacy else heating_eur(cost, features, a, cmodels)

    def part(key, label, eur, assumed=True):
        return {'key': key, 'label': label, 'eur': eur, 'assumed': assumed}

    inner = []                                # contained in price (+ stated Nebenkosten)
    if warm or add is None:
        if legacy:
            inner.append(part('nk', 'Nebenkosten', round(a['nk_m2'] * wf)))
        else:
            nk = _predict_small((cmodels or {}).get('nk'), _cost_row(features, cost))
            nk = nk if nk else 2.0 * wf
            if warm:                          # Wasser/Müll billed separately are not in this price
                nk -= sum(a[k] for k in ('wasser', 'muell') if k in extra)
            inner.append(part('nk', 'Betriebskosten', round(nk)))
            if 'heizung' not in extra:
                inner.append(part('heizung', 'Heizung', round(heat())))
    elif add:
        inner.append(part('nk', 'Nebenkosten ohne Heizung' if 'heizung' in extra else 'Nebenkosten inkl. Heizung',
                          add, False))
    if warm:
        if 'strom' in incl:
            inner.append(part('strom', 'Strom', round(strom)))
        if 'internet' in incl:
            inner.append(part('internet', 'Internet', round(a['internet'])))
        kalt = price - sum(p['eur'] for p in inner)
        flags = [{'label': 'Warmmiete lt. Anzeige', 'eur': price}]
    else:
        kalt = price
        flags = [{'label': 'Kaltmiete lt. Anzeige', 'eur': price}]
        if add:
            flags.append({'label': 'Warmmiete lt. Anzeige', 'eur': price + add})
    outer = []                                # added on top to normalise
    if 'heizung' in extra:
        outer.append(part('heizung', 'Heizung', round(heat())))
    for k in ('wasser', 'muell'):
        if k in extra:
            outer.append(part(k, COST_LABELS[k], a[k]))
    for k in ('strom', 'internet', 'stellplatz'):
        if k in incl:
            continue
        if k == 'stellplatz' and cost.get('sp') is not None:
            outer.append(part(k, COST_LABELS[k], round(cost['sp']), False))
        else:
            outer.append(part(k, COST_LABELS[k], round(strom if k == 'strom' else a[k])))
    if kalt <= 0:
        return None
    parts = inner + outer
    return {'kalt': kalt, 'kalt_est': warm, 'parts': parts, 'flags': flags,
            'total': kalt + sum(p['eur'] for p in parts)}


def kalt_rent(price, cost, features, assume, cmodels=None):
    """(Kaltmiete, estimated) for a rent listing; (None, False) without cost data."""
    rb = rent_breakdown(price, cost, features, assume, cmodels)
    return (round(rb['kalt']), rb['kalt_est']) if rb else (None, False)


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


def is_unified(definition: dict) -> bool:
    """A unified feature set covers flats AND houses, rent AND sale in ONE model (see `train`)."""
    return bool(definition.get('unified'))


OFFERS = ('miete', 'kauf')       # output order of a unified (multi-output) model


class MaskedBoost:
    """Gradient boosting with SEVERAL outputs (here: log rent, log purchase price) and a mask: every listing
    only has a real price for ONE output and only counts for that one (weight 0 for the other, so there is no
    synthetic ground truth and no Kauf/Miete factor). All outputs share the tree structure (so everything
    learnt about location, size, condition … helps both), the leaf values are fitted per output from the
    listings that have a label for it. Trees are chosen on the residuals of both outputs (missing = 0),
    then each leaf gets, per output, the mean residual of the labelled listings in it (shrunk by n/(n+1))."""

    def __init__(self, n_outputs=2, n_estimators=250, learning_rate=0.04, max_depth=3, subsample=0.8,
                 min_samples_leaf=3, random_state=0):
        self.n_outputs, self.n_estimators, self.learning_rate = n_outputs, n_estimators, learning_rate
        self.max_depth, self.subsample, self.min_samples_leaf = max_depth, subsample, min_samples_leaf
        self.random_state = random_state

    def fit(self, X, Y, M):
        """X (n, d); Y (n, k) log prices (anything where M is False); M (n, k) bool: label present."""
        from sklearn.tree import DecisionTreeRegressor
        n, k = Y.shape
        rng = np.random.RandomState(self.random_state)
        self.base_ = np.array([Y[M[:, j], j].mean() if M[:, j].any() else 0.0 for j in range(k)])
        F = np.tile(self.base_, (n, 1))
        self.trees_, self.values_ = [], []
        for t in range(self.n_estimators):
            R = np.where(M, Y - F, 0.0)
            idx = np.where(rng.rand(n) < self.subsample)[0]
            if len(idx) < 2 * self.min_samples_leaf:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         random_state=int(rng.randint(1 << 30))).fit(X[idx], R[idx])
            leaf = tree.apply(X)
            val = np.zeros((tree.tree_.node_count, k))
            for j in range(k):
                use = idx[M[idx, j]]
                cnt = np.bincount(leaf[use], minlength=val.shape[0])
                tot = np.bincount(leaf[use], weights=R[use, j], minlength=val.shape[0])
                val[:, j] = tot / (cnt + 1.0)
            F += self.learning_rate * val[leaf]
            self.trees_.append(tree)
            self.values_.append(val)
        return self

    def predict(self, X):
        """(n, k) log prices — every output for every row, whatever the row's own offer is."""
        F = np.tile(self.base_, (len(X), 1))
        for tree, val in zip(self.trees_, self.values_):
            F += self.learning_rate * val[tree.apply(X)]
        return F


def _masked_importances(model, X, Y, M, names, repeats=6):
    """Permutation importance: increase of the mean absolute log error (labelled outputs only)."""
    rng = np.random.RandomState(0)
    base = float(np.abs(np.where(M, model.predict(X) - Y, 0.0)).sum() / M.sum())
    out = []
    for j in range(X.shape[1]):
        errs = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(X)), j]
            errs.append(float(np.abs(np.where(M, model.predict(Xp) - Y, 0.0)).sum() / M.sum()) - base)
        out.append(float(np.mean(errs)))
    order = np.argsort(-np.array(out))
    return [(names[i], round(out[i], 4)) for i in order[:15]]


def train_unified(fs: dict, activate: bool = True) -> dict:
    """ONE model for rent AND sale (flats and houses): two outputs (log normalised Warmmiete, log purchase
    price), see `MaskedBoost`. Each listing has a label for its own offer only (`_angebot`); the other output
    is not trained on it. Every listing is predicted for BOTH offers (`fair_prices.fair_price` = own offer,
    `fair_other` = the other one); training listings get out-of-fold values for both."""
    import joblib
    from sklearn.model_selection import KFold

    d = fs['definition']
    pl = d.get('plausible') or {}
    per = pl.get('per') or 'wohnflaeche'
    ranges = {k: tuple(pl[k]) for k in OFFERS if k in pl}
    exclude = set(d.get('exclude') or [])        # flags that take a listing out of training and rating
    features = [f for f in d['features'] if f['key'] not in exclude]
    rows = [r for r in db.get_training_rows(fs['id']) if not r.get('skip')]
    for r in rows:
        if r['offer'] not in OFFERS:
            raise SystemExit(f'listing {r["listing_id"]} has no _angebot (miete|kauf)')
    assume = db.get_cost_assumptions(fs['id']) if any(r.get('cost') for r in rows) else None
    cmodels = fit_cost_models(rows, assume, db.get_heat_rows()) if assume else None
    for r in rows:
        r['raw_price'] = r['price']
        r['price'] = (effective_price(r['price'], r.get('cost'), r['features'], assume, cmodels)
                      if r['price'] else r['price'])
    train_rows = [r for r in rows if not any(r['features'].get(k) for k in exclude)
                  and _sane({**r, 'price': r['raw_price']}, per, ranges[r['offer']])]
    if len(train_rows) < MIN_TRAIN:
        raise SystemExit(f'only {len(train_rows)} usable priced listings (<{MIN_TRAIN})')

    enc = Encoder(features).fit([r['features'] for r in train_rows])
    X = enc.transform([r['features'] for r in train_rows])
    own = np.array([OFFERS.index(r['offer']) for r in train_rows])
    P = np.array([r['price'] for r in train_rows])
    Y = np.zeros((len(P), 2))
    M = np.zeros((len(P), 2), dtype=bool)
    Y[np.arange(len(P)), own] = np.log(P)
    M[np.arange(len(P)), own] = True

    reps = []                                  # out-of-fold predictions of BOTH outputs, 3 × 5-fold
    for seed in range(3):
        oof = np.zeros((len(P), 2))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            oof[te] = np.exp(MaskedBoost(random_state=seed).fit(X[tr], Y[tr], M[tr]).predict(X[te]))
        reps.append(oof)
    oof = np.mean(reps, axis=0)
    ar = np.arange(len(P))
    by_offer, apes = {}, []
    for k, off in enumerate(OFFERS):
        idx = np.where(own == k)[0]
        if len(idx):
            err = [np.abs(p[idx, k] - P[idx]) for p in reps]
            ape = [e / P[idx] for e in err]
            by_offer[off] = {'n': int(len(idx)), 'mae': round(float(np.mean([e.mean() for e in err])), 1),
                             'median_ape': round(float(np.mean([np.median(a) for a in ape])), 4)}
    own_pred = [p[ar, own] for p in reps]
    median_ape = float(np.mean([np.median(np.abs(p - P) / P) for p in own_pred]))
    ly = np.log(P)
    mu = np.array([ly[own == k].mean() if (own == k).any() else 0.0 for k in range(2)])[own]       # mean of its own offer
    r2 = float(np.mean([1 - np.sum((np.log(p) - ly) ** 2) / np.sum((ly - mu) ** 2) for p in own_pred]))
    scores = {'masked_boosting': {'mae': None, 'median_ape': round(median_ape, 4), 'r2': round(r2, 3)}}

    model = MaskedBoost().fit(X, Y, M)
    imp = _masked_importances(model, X, Y, M, enc.names)
    # predictions of every listing (also excluded / implausible ones) for both offers
    pred_all = np.exp(model.predict(enc.transform([r['features'] for r in rows])))
    oof_by_id = {r['listing_id']: oof[i] for i, r in enumerate(train_rows)}
    preds = {}
    for i, r in enumerate(rows):
        v = oof_by_id.get(r['listing_id'])
        v = v if v is not None else pred_all[i]
        o = OFFERS.index(r['offer'])
        preds[r['listing_id']] = (float(v[o]), float(v[1 - o]))
    ratio = pred_all[:, 1] / pred_all[:, 0]      # implied purchase price / rent of the model, per listing
    metrics = {'chosen': 'masked_boosting', 'multi': True, 'unified': True, 'per': per, 'per_range': None,
               'ranges': {k: list(v) for k, v in ranges.items()}, 'exclude': sorted(exclude),
               'cost_assumptions': assume,
               'cost_models': {k: ({'n': v['n'], 'mae': v['mae'], 'algo': v['algo']} if v else None)
                               for k, v in (cmodels or {}).items()},
               'candidates': scores, 'by_offer': by_offer, 'n_features': len(enc.names),
               'n_excluded': len(rows) - len(train_rows), 'importance': imp,
               'ratio': {k: round(float(np.percentile(ratio, q)), 1) for k, q in (('p10', 10), ('median', 50), ('p90', 90))}}
    buf = io.BytesIO()
    joblib.dump({'encoder': enc, 'model': model, 'algo': 'masked_boosting', 'multi': True, 'cost_models': cmodels}, buf)
    mid = db.save_model(fs['id'], 'masked_boosting', len(train_rows), metrics, buf.getvalue(), preds, activate)
    metrics['model_id'] = mid
    refresh_predictions(fs['id'])
    print(f'model {mid}: masked_boosting  n={len(train_rows)} (excluded {len(rows) - len(train_rows)})')
    for off, v in by_offer.items():
        print(f'  {off:6} n={v["n"]:4}  MAE {v["mae"]:9.1f} €  median |err| {v["median_ape"]*100:4.1f} %')
    print(f'  implied purchase price / monthly rent over all listings: median {metrics["ratio"]["median"]:.0f} '
          f'(10-90 %: {metrics["ratio"]["p10"]:.0f}-{metrics["ratio"]["p90"]:.0f})')
    print('  top features:', ', '.join(f'{k} ({v:.3f})' for k, v in imp[:8]))
    return metrics


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
    if is_unified(fs['definition']):
        return train_unified(fs, activate)
    rows = db.get_training_rows(fs['id'])
    # rows with cost data are trained on the normalised rent (see effective_price)
    assume = db.get_cost_assumptions(fs['id']) if any(r.get('cost') for r in rows) else None
    cmodels = fit_cost_models(rows, assume, db.get_heat_rows()) if assume else None
    for r in rows:
        r['raw_price'] = r['price']
        r['price'] = (effective_price(r['price'], r.get('cost'), r['features'], assume, cmodels)
                      if r['price'] else r['price'])
    train_rows = [r for r in rows if _sane({**r, 'price': r['raw_price']}, per, rng)]
    for r in train_rows:
        r['target'] = r['price']
    if len(train_rows) < MIN_TRAIN:
        raise SystemExit(f'only {len(train_rows)} usable priced listings (<{MIN_TRAIN})')

    enc = Encoder(features).fit([r['features'] for r in train_rows])
    X = enc.transform([r['features'] for r in train_rows])
    y = np.log([r['target'] for r in train_rows])
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
    refresh_predictions(fs['id'])            # results of the (possibly new) active model
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
    rows = [r for r in db.get_training_rows(fs['id']) if not r.get('skip')]
    if listing_ids is None:
        todo = [r for r in rows if r['listing_id'] not in have]
    else:
        wanted = set(listing_ids)
        todo = [r for r in rows if r['listing_id'] in wanted and r['listing_id'] not in trained]
    if not todo:
        return 0
    X = art['encoder'].transform([r['features'] for r in todo])
    pred = np.exp(art['model'].predict(X))
    if art.get('multi'):          # (own offer, other offer) per listing
        out = {}
        for r, p in zip(todo, pred):
            o = OFFERS.index(r['offer'])
            out[r['listing_id']] = (float(p[o]), float(p[1 - o]))
        db.save_predictions(model_row['id'], out)
    else:
        db.save_predictions(model_row['id'], {r['listing_id']: float(p) for r, p in zip(todo, pred)})
    if model_row['id'] == db.active_model_id(fs['id']):
        refresh_predictions(fs['id'], [r['listing_id'] for r in todo])
    return len(todo)


def listing_results(fair_raw, cost, features, assume, cmodels, impacts=None, other=None, offer=None):
    """Everything the active model (incl. its cost sub-models) predicts for one listing — stored in
    `listing_predictions` and shown on the archive page. `fair_raw` = prediction for the listing's own offer;
    unified model: `other` = prediction for the other offer, so both are stored."""
    out = {'fair_price': round(fair_raw, 2)}
    if other is not None and offer in OFFERS:
        mi, ka = (fair_raw, other) if offer == 'miete' else (other, fair_raw)
        out.update({'fair_miete': round(mi, 2), 'fair_kauf': round(ka, 2)})
    if impacts:      # effect of each feature value, as fraction of the model's own prediction
        out['impacts'] = impacts
    if cost is None or not assume:
        return out
    a = {**DEFAULT_COSTS, **assume}
    if 'strom' in assume and 'strom_kwh' not in assume:      # legacy model without energy prices
        return out
    # Energiebedarf: always the model's own estimate (also when the ad states one). Pricing/rating use
    # the extracted value from the ad if available, this prediction only as fallback (heat_estimate).
    heat = cost.get('heat') or {}
    kwh, src = heat_estimate({**cost, 'heat': {**heat, 'kwh': None}}, features, cmodels)
    out.update({'heat_kwh_m2': round(kwh, 1) if kwh else 130.0, 'heat_kwh_src': src or 'pauschal',
                'heat_used': 'anzeige' if heat.get('kwh') else 'prognose',
                'heat_carrier': heat.get('carrier'), 'heat_eur_kwh': round(kwh_price(heat.get('carrier'), a), 4),
                'heat_eur': round(heating_eur(cost, features, a, cmodels), 2)})   # with the value actually used
    nk, nsrc = nk_estimate(cost, features, cmodels)
    out.update({'nk_eur': round(nk, 2), 'nk_src': nsrc})
    return out


def refresh_predictions(feature_set_id: int, listing_ids=None) -> int:
    """(Re)write `listing_predictions` of a feature set from its ACTIVE model — all listings (the
    active model changed: old results are overwritten/dropped) or just `listing_ids`."""
    mid = db.active_model_id(feature_set_id)
    if not mid:
        db.save_listing_predictions(feature_set_id, None, {}, replace_all=True)
        return 0
    with db.get_db() as conn:
        fair = {r[0]: (r[1], r[2]) for r in conn.execute(
            'SELECT listing_id, fair_price, fair_other FROM fair_prices WHERE model_id = ?', (mid,))}
        metrics = json.loads(conn.execute('SELECT metrics FROM ml_models WHERE id = ?', (mid,)).fetchone()[0])
    assume, cm = metrics.get('cost_assumptions'), cost_models_for(mid)
    wanted = None if listing_ids is None else set(listing_ids)
    def impacts(r):
        try:
            own, imp = feature_impacts(mid, feature_set_id, r['features'], r.get('offer'))
            return {k: round(v / own, 4) for k, v in imp.items() if own and abs(v / own) >= 0.0005}
        except Exception as e:
            print(f'feature_impacts failed for listing {r["listing_id"]}: {e}')
            return None
    rows = {r['listing_id']: listing_results(fair[r['listing_id']][0], r.get('cost'), r['features'], assume, cm,
                                            impacts(r), fair[r['listing_id']][1], r.get('offer'))
            for r in db.get_training_rows(feature_set_id)
            if r['listing_id'] in fair and (wanted is None or r['listing_id'] in wanted)}
    db.save_listing_predictions(feature_set_id, mid, rows, replace_all=wanted is None)
    return len(rows)


_IMPACT_CACHE = {}


def feature_impacts(model_id, fs_id, features, offer=None, n_ref=120):
    """Per-feature effect (in model units, e.g. € of the fair price) for ONE listing: prediction with
    the listing's own value minus the mean prediction when that feature is replaced by the values of
    reference listings (random sample of the training data). Not additive in general. -> (own prediction, {key: €})"""
    if model_id not in _IMPACT_CACHE:
        _IMPACT_CACHE.clear()
        row = db.get_model(model_id)
        art = _load_artifact(row['artifact'])
        rows = [r['features'] for r in db.get_training_rows(fs_id)]
        rng = np.random.RandomState(0)
        if len(rows) > n_ref:
            rows = [rows[i] for i in rng.choice(len(rows), n_ref, replace=False)]
        _IMPACT_CACHE[model_id] = (art['encoder'], art['model'], rows)
    enc, model, refs = _IMPACT_CACHE[model_id]
    col = OFFERS.index(offer) if offer in OFFERS else 0      # multi-output model: the listing's own offer

    def run(rows):
        p = np.exp(model.predict(enc.transform(rows)))
        return p[:, col] if p.ndim == 2 else p
    own = float(run([features])[0])
    keys = [f['key'] for f in enc.features]
    batch = [{**features, k: r.get(k)} for k in keys for r in refs]
    pred = run(batch).reshape(len(keys), len(refs)).mean(axis=1)
    return own, {k: own - float(p) for k, p in zip(keys, pred)}


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
    elif len(sys.argv) in (2, 3) and sys.argv[1] == 'refresh':     # rewrite listing_predictions
        db.init_db()
        for fs in db.get_feature_sets_overview():
            if len(sys.argv) == 2 or fs['name'] == sys.argv[2]:
                print(fs['name'], _ml.refresh_predictions(fs['id']), 'listings')
    else:
        print(__doc__)
