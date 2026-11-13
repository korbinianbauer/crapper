"""One-off: create the UNIFIED feature set `immobilie_v1` and copy the already extracted values into it.

One feature set / one model for flats AND houses, rent AND sale (see ml.train, `definition['unified']`).
Every listing carries `_angebot` ('miete'|'kauf'); rents keep `_cost`. Sources of the copy:
  * wohnung_miete_gesamt_v1 (13)  -> _angebot = miete   (already holds the KA and ImmoScout rents)
  * wohnung_kaufen_v1 (6)         -> _angebot = kauf
Not copied: wohnung_miete_v1 / wohnung_miete_is_v1 (their values are in the Gesamt set).

Changed on the way: sale `saniert` -> `renoviert`, `hochwertig` dropped (replaced by `ausstattung`),
`dist_schliersee` added for every listing from its PLZ (straight line × 1.25, see PLZ_COORDS).
New features that an LLM agent still has to read from the ad (never a script) are listed in each listing's
`_todo`; the agent removes them when it has filled them in:
  seelage, denkmal, niessbrauch (all), ausstattung + zustand (sale only), grundstueck (houses only).

    venv/bin/python scripts/migrate_unified_features.py [--dry]
Idempotent: existing values of the unified set are kept (only missing listings are copied).
"""
import json
import math
import sys

sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0])
import db

NAME = 'immobilie_v1'


def F(key, label, typ):
    return {'key': key, 'label': label, 'type': typ}


FEATURES = [
    F('wohnflaeche', 'Wohnfläche (m²)', 'num'), F('nutzflaeche', 'Nutzfläche (m²)', 'num'),
    F('grundstueck', 'Grundstücksfläche (m², nur Häuser)', 'num'),
    F('zimmer', 'Zimmer', 'num'), F('schlafzimmer', 'Schlafzimmer', 'num'), F('badezimmer', 'Badezimmer', 'num'),
    F('etage', 'Etage (EG=0)', 'num'),
    F('typ', 'Objekttyp (Etagenwohnung, Dachgeschosswohnung, …, Einfamilienhaus, Doppelhaushälfte, Reihenhaus, …)', 'cat'),
    F('plz', 'PLZ', 'num'),
    F('dist_muenchen', 'Entfernung München (km)', 'num'), F('dist_tegernsee', 'Entfernung Tegernsee (km)', 'num'),
    F('dist_schliersee', 'Entfernung Schliersee (km)', 'num'),
    F('baujahr', 'Baujahr', 'num'), F('energieklasse', 'Energieeffizienzklasse (A+=0 … H=8)', 'num'),
    F('ausstattung', 'Ausstattungsqualität (Einfach=0, Normal=1, Gehoben=2, Luxus=3)', 'num'),
    F('zustand', 'Zustand (renovierungsbed.=0, gepflegt=1, modernisiert/saniert/renoviert=2, neuwertig/Erstbezug n. Sanierung=3, Erstbezug Neubau=4)', 'num'),
    F('hausgeld', 'Hausgeld (€/Monat, nur Kauf)', 'num'),
    F('balkon', 'Balkon/Terrasse/Loggia', 'bool'), F('garten', 'Garten', 'bool'),
    F('einbaukueche', 'Einbauküche', 'bool'), F('aufzug', 'Aufzug', 'bool'),
    F('stellplatz', 'Garage/Stellplatz', 'bool'), F('keller', 'Keller/Abstellraum', 'bool'),
    F('neubau', 'Neubau/Erstbezug', 'bool'), F('renoviert', 'Renoviert/saniert/modernisiert', 'bool'),
    F('sanierungsbedarf', 'Renovierungs-/Sanierungsbedarf', 'bool'),
    F('moebliert', 'Möbliert', 'bool'), F('befristet', 'Befristet/auf Zeit/Ferienwohnung (Miete)', 'bool'),
    F('gewerblich', 'Gewerblicher Anbieter/Makler/Bauträger', 'bool'),
    F('aussicht', 'See-/Berg-/Panoramablick', 'bool'),
    F('seelage', 'Seelage (Seezugang, Seegrundstück, unmittelbar am See)', 'bool'),
    F('barrierefrei', 'Barrierefrei', 'bool'), F('fussbodenheizung', 'Fußbodenheizung', 'bool'),
    F('vermietet', 'Aktuell vermietet (Kauf)', 'bool'), F('ferienwohnung', 'Ferienwohnung/Zweitwohnsitz-Nutzung', 'bool'),
    F('erbbaurecht', 'Erbbaurecht', 'bool'), F('denkmal', 'Denkmalgeschützt/historisches Gebäude', 'bool'),
    F('provisionsfrei', 'Provisionsfrei für Käufer', 'bool'),
    # exclusion flags: listings with one of them are neither trained on nor rated (definition['exclude'])
    F('tausch', 'Tauschwohnung (Preis ist keine Miete)', 'bool'), F('wg', 'WG/Untermiete/Zimmer', 'bool'),
    F('niessbrauch', 'Nießbrauch/Wohnrecht belastet (Kauf)', 'bool'),
]
DEFINITION = {
    'features': FEATURES, 'target': 'price', 'unified': True,
    'exclude': ['tausch', 'wg', 'niessbrauch'],
    'plausible': {'per': 'wohnflaeche', 'miete': [4, 60], 'kauf': [1500, 15000]},
}
NEW_ALL = ['seelage', 'denkmal', 'niessbrauch']
NEW_SALE = ['ausstattung', 'zustand']

# Ortsmitte per PLZ (lat, lon), estimated by hand — only used for distances, ±2 km is enough.
SCHLIERSEE = (47.735, 11.860)
PLZ_COORDS = {
    6335: (47.58, 12.10), 81479: (48.07, 11.52), 82008: (48.065, 11.61), 82024: (48.05, 11.62),
    82031: (48.04, 11.52), 82041: (48.02, 11.59), 82049: (48.06, 11.52), 82054: (47.97, 11.65),
    82057: (47.98, 11.42), 82064: (47.96, 11.50), 82065: (48.04, 11.49), 82069: (48.00, 11.45),
    82515: (47.91, 11.43), 82538: (47.86, 11.48), 82544: (47.89, 11.55), 82549: (47.82, 11.47),
    83022: (47.857, 12.12), 83024: (47.86, 12.14), 83026: (47.84, 12.10), 83043: (47.863, 12.01),
    83052: (47.88, 11.93), 83059: (47.85, 12.07), 83064: (47.79, 12.10), 83071: (47.84, 12.19),
    83075: (47.78, 12.00), 83080: (47.65, 12.17), 83083: (47.82, 12.19), 83088: (47.62, 12.18),
    83098: (47.74, 12.10), 83101: (47.80, 12.20), 83104: (47.93, 11.97), 83109: (47.90, 12.06),
    83115: (47.77, 12.14), 83126: (47.69, 12.14), 83128: (47.93, 12.28), 83131: (47.74, 12.14),
    83134: (47.91, 12.17), 83135: (47.91, 12.12), 83139: (47.97, 12.22), 83512: (48.06, 12.23),
    83533: (48.05, 12.12), 83539: (48.04, 12.07), 83543: (47.99, 12.10), 83549: (48.03, 12.20),
    83553: (47.99, 12.01), 83556: (48.01, 12.28), 83569: (47.89, 12.20), 83607: (47.88, 11.70),
    83620: (47.88, 11.83), 83623: (47.82, 11.59), 83624: (47.88, 11.74), 83626: (47.84, 11.77),
    83627: (47.82, 11.73), 83629: (47.86, 11.80), 83646: (47.76, 11.56), 83661: (47.68, 11.57),
    83666: (47.78, 11.65), 83670: (47.73, 11.47), 83674: (47.73, 11.55), 83677: (47.75, 11.62),
    83684: (47.71, 11.76), 83700: (47.69, 11.78), 83703: (47.74, 11.73), 83707: (47.71, 11.73),
    83708: (47.64, 11.75), 83714: (47.79, 11.84), 83727: (47.735, 11.86), 83730: (47.72, 11.95),
    83734: (47.75, 11.84), 83735: (47.68, 12.00), 83737: (47.85, 11.89), 85521: (48.07, 11.66),
    85560: (48.08, 11.97), 85567: (48.04, 11.97), 85625: (47.96, 11.89), 85630: (48.07, 11.72),
    85635: (48.01, 11.72), 85649: (47.99, 11.73), 85653: (47.96, 11.78), 85658: (48.01, 11.84),
    85662: (48.04, 11.72), 85665: (48.02, 11.97), 85667: (48.00, 11.86),
}
ROAD = 1.25      # straight line -> road


def dist_schliersee(plz):
    c = PLZ_COORDS.get(int(plz)) if plz else None
    if not c:
        return None
    (la1, lo1), (la2, lo2) = c, SCHLIERSEE
    p = math.pi / 180
    a = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return round(2 * 6371 * math.asin(math.sqrt(a)) * ROAD)


# sale listings whose titles name a Nießbrauch/Wohnrecht (read by me in the analysis of 13.11.2026)
NIESSBRAUCH = {5260, 5506, 7995}


def convert(data: dict, offer: str, lid: int) -> dict:
    d = {k: v for k, v in data.items()}
    if offer == 'kauf':
        if 'saniert' in d:
            d['renoviert'] = bool(d.pop('saniert'))
        d.pop('hochwertig', None)
    d['dist_schliersee'] = dist_schliersee(d.get('plz'))
    d['_angebot'] = offer
    todo = list(NEW_ALL) + (NEW_SALE if offer == 'kauf' else [])
    if lid in NIESSBRAUCH:
        d['niessbrauch'] = True
        todo.remove('niessbrauch')
    d['_todo'] = todo
    return d


def main(dry=False):
    with db.get_db() as conn:
        row = conn.execute('SELECT id FROM feature_sets WHERE name = ?', (NAME,)).fetchone()
        if row:
            fs_id = row[0]
            if not dry:
                conn.execute('UPDATE feature_sets SET definition = ? WHERE id = ?',
                             (json.dumps(DEFINITION, ensure_ascii=False), fs_id))
        else:
            fs_id = None if dry else conn.execute(
                'INSERT INTO feature_sets (name, tracker_id, definition, created_at) VALUES (?, NULL, ?, ?)',
                (NAME, json.dumps(DEFINITION, ensure_ascii=False), db._now())).lastrowid
        have = {r[0] for r in conn.execute('SELECT listing_id FROM listing_features WHERE feature_set_id = ?', (fs_id,))} if fs_id else set()
        rent = conn.execute('SELECT listing_id, data FROM listing_features WHERE feature_set_id = (SELECT id FROM feature_sets WHERE name = "wohnung_miete_gesamt_v1")').fetchall()
        sale = conn.execute('SELECT listing_id, data FROM listing_features WHERE feature_set_id = (SELECT id FROM feature_sets WHERE name = "wohnung_kaufen_v1")').fetchall()
        rows = {}
        for offer, src in (('miete', rent), ('kauf', sale)):
            for lid, data in src:
                if lid not in have:
                    rows[lid] = convert(json.loads(data), offer, lid)
        print(f'feature set {NAME} id={fs_id}: copy {len(rows)} listings '
              f'({sum(r["_angebot"] == "miete" for r in rows.values())} Miete, {sum(r["_angebot"] == "kauf" for r in rows.values())} Kauf), '
              f'{len(have)} already there; dist_schliersee missing: {sum(r["dist_schliersee"] is None for r in rows.values())}')
        if dry:
            return
        # keep the settings that belong to the old joint rent set (cost assumptions); the offset is NOT copied
        old = conn.execute('SELECT id FROM feature_sets WHERE name = "wohnung_miete_gesamt_v1"').fetchone()[0]
        v = conn.execute('SELECT value FROM settings WHERE key = ?', (f'fair_cost_{old}',)).fetchone()
        if v:
            conn.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (f'fair_cost_{fs_id}', v[0]))
    db.save_listing_features(fs_id, rows)
    print('saved')


if __name__ == '__main__':
    main(dry='--dry' in sys.argv)
