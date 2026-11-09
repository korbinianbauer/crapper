"""Manual feature extraction for 'Wohnung mieten' trackers on Kleinanzeigen.

NOT part of the running app: run by hand against a DB, review the output, then
load it with scripts/import_features.py. Writes <out>.json
    {"feature_set": {...definition...}, "rows": {listing_id: {feature: value}}}

    python scripts/features/extract_wohnung_miete.py crapper.db 10 out.json
"""
import json, math, re, sqlite3, sys

FEATURES = [  # key, label, type ('num' | 'bool' | 'cat')
    ('wohnflaeche', 'Wohnfläche (m²)', 'num'),
    ('zimmer', 'Zimmer', 'num'),
    ('schlafzimmer', 'Schlafzimmer', 'num'),
    ('badezimmer', 'Badezimmer', 'num'),
    ('etage', 'Etage (EG=0)', 'num'),
    ('typ', 'Wohnungstyp', 'cat'),
    ('plz', 'PLZ', 'num'),
    ('dist_muenchen', 'Entfernung München (km)', 'num'),
    ('dist_tegernsee', 'Entfernung Tegernsee (km)', 'num'),
    ('baujahr', 'Baujahr', 'num'),
    ('energieklasse', 'Energieeffizienzklasse (A+=0 … H=8)', 'num'),
    ('balkon', 'Balkon/Terrasse/Loggia', 'bool'),
    ('garten', 'Garten', 'bool'),
    ('einbaukueche', 'Einbauküche', 'bool'),
    ('aufzug', 'Aufzug', 'bool'),
    ('stellplatz', 'Garage/Stellplatz', 'bool'),
    ('keller', 'Keller/Abstellraum', 'bool'),
    ('neubau', 'Neubau/Erstbezug', 'bool'),
    ('renoviert', 'Renoviert/saniert', 'bool'),
    ('moebliert', 'Möbliert', 'bool'),
    ('befristet', 'Befristet/auf Zeit/Ferienwohnung', 'bool'),
    ('tausch', 'Tauschwohnung', 'bool'),
    ('gewerblich', 'Gewerblicher Anbieter', 'bool'),
    ('wg', 'WG/Untermiete/Zimmer', 'bool'),
    ('aussicht', 'See-/Berg-/Panoramablick', 'bool'),
    ('barrierefrei', 'Barrierefrei', 'bool'),
    ('fussbodenheizung', 'Fußbodenheizung', 'bool'),
    ('hochwertig', 'Hochwertig/luxuriös', 'bool'),
]

# Approximate centre coordinates (lat, lon) of the places seen in tracker 10 (±3 km is fine).
PLACES = {
 'Taufkirchen':(48.050,11.615),'Grünwald':(48.040,11.520),'Deisenhofen':(48.020,11.590),'Furth':(48.005,11.580),
 'Oberbiberg':(48.005,11.600),'Pullach':(48.060,11.520),'Höllriegelskreuth':(48.060,11.500),'Wolfratshausen':(47.910,11.430),
 'Geretsried':(47.860,11.480),'Königsdorf':(47.830,11.470),'Rosenheim':(47.856,12.128),'Bad Aibling':(47.863,12.010),
 'Bruckmühl':(47.880,11.930),'Kolbermoor':(47.848,12.063),'Raubling':(47.790,12.110),'Stephanskirchen':(47.840,12.150),
 'Bad Feilnbach':(47.780,12.010),'Oberaudorf':(47.640,12.170),'Riedering':(47.800,12.240),'Kiefersfelden':(47.613,12.190),
 'Brannenburg':(47.740,12.100),'Rohrdorf':(47.800,12.180),'Tuntenhausen':(47.970,11.900),'Großkarolinenfeld':(47.870,12.050),
 'Flintsbach':(47.720,12.110),'Halfing':(47.910,12.240),'Nußdorf':(47.750,12.160),'Prutting':(47.900,12.170),
 'Schechen':(47.900,12.200),'Wasserburg':(48.060,12.230),'Edling':(48.020,12.150),'Pfaffing':(48.030,12.170),
 'Rott am Inn':(47.990,12.110),'Frauenneuharting':(47.990,11.950),'Vogtareuth':(47.950,12.150),'Holzkirchen':(47.880,11.698),
 'Feldkirchen-Westerham':(47.900,11.820),'Dietramszell':(47.810,11.570),'Otterfing':(47.880,11.730),'Valley':(47.870,11.790),
 'Warngau':(47.820,11.750),'Weyarn':(47.870,11.770),'Bad Tölz':(47.760,11.560),'Wackersberg':(47.700,11.510),
 'Waakirchen':(47.790,11.690),'Bad Heilbrunn':(47.740,11.520),'Gaißach':(47.700,11.570),'Tegernsee':(47.710,11.760),
 'Rottach-Egern':(47.690,11.760),'Gmund':(47.750,11.730),'Bad Wiessee':(47.710,11.720),'Kreuth':(47.640,11.760),
 'Miesbach':(47.790,11.830),'Schliersee':(47.730,11.860),'Fischbachau':(47.710,11.960),'Hausham':(47.770,11.830),
 'Bayrischzell':(47.670,12.010),'Irschenberg':(47.800,11.900),'Ottobrunn':(48.070,11.660),'Riemerling':(48.050,11.700),
 'Grafing':(48.050,11.970),'Glonn':(47.990,11.880),'Höhenkirchen-Siegertsbrunn':(48.020,11.720),'Brunnthal':(47.990,11.740),
 'Aying':(47.970,11.780),'Großhelfendorf':(47.960,11.800),'Egmating':(48.000,11.820),'Hohenbrunn':(48.040,11.710),
 'Moosach':(47.970,11.950),'Oberpframmern':(48.010,11.800),
}
MUC, TEG = (48.137, 11.575), (47.710, 11.760)

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return round(12742 * math.asin(math.sqrt(h)), 1)

def place(loc):
    for name, xy in PLACES.items():
        if name.lower() in (loc or '').lower():
            return xy
    return None

NEG = re.compile(r'(kein|keine|keinen|keiner|ohne|nicht|un)\W{0,3}$', re.I)
def has(text, pattern):
    """Keyword present and not negated directly before it."""
    for m in re.finditer(pattern, text, re.I):
        if not NEG.search(text[max(0, m.start()-14):m.start()]):
            return True
    return False

def num(s):
    return float(s.replace('.', '').replace(',', '.'))

def detail(desc, key):
    m = re.search(r'^%s\s+([\d.,]+)' % key, desc.split('\nDetails:\n')[-1], re.M)
    return num(m.group(1)) if m else None

def extract(title, location, desc):
    d = desc or ''
    t = f'{title}\n{d}'
    body = d.split('\nDetails:\n')[0]
    f = {}
    f['wohnflaeche'] = detail(d, 'Wohnfläche')
    f['zimmer'] = detail(d, 'Zimmer')
    f['schlafzimmer'] = detail(d, 'Schlafzimmer')
    f['badezimmer'] = detail(d, 'Badezimmer')
    typ = re.search(r'^Wohnungstyp\s+(.+)$', d, re.M)
    typ = typ.group(1).strip() if typ else None
    f['typ'] = typ
    f['etage'] = detail(d, 'Etage')
    if f['etage'] is None and typ and 'Erdgeschoss' in typ:
        f['etage'] = 0.0
    m = re.match(r'(\d{5})', location or '')
    f['plz'] = float(m.group(1)) if m else None
    xy = place(location)
    f['dist_muenchen'] = km(xy, MUC) if xy else None
    f['dist_tegernsee'] = km(xy, TEG) if xy else None
    m = (re.search(r'Baujahr[^\d\n]{0,40}((?:18|19|20)\d\d)', t, re.I)
         or re.search(r'(?:erbaut|errichtet|gebaut)[^\d\n]{0,25}((?:18|19|20)\d\d)', t, re.I))
    f['baujahr'] = float(m.group(1)) if m else None
    m = re.search(r'Energieeffizienzklasse\W{0,4}(A\+|[A-H])\b', t)
    f['energieklasse'] = float(['A+','A','B','C','D','E','F','G','H'].index(m.group(1))) if m else None
    f['balkon'] = has(t, r'balkon|terrasse|loggia|dachterrasse')
    f['garten'] = has(t, r'garten|gartenanteil|gartennutzung')
    f['einbaukueche'] = has(t, r'einbauk[üu]che|\bEBK\b|k[üu]che (ist )?vorhanden|k[üu]chenzeile')
    f['aufzug'] = has(t, r'aufzug|fahrstuhl|lift\b')
    f['stellplatz'] = has(t, r'garage|stellplatz|tiefgarage|parkplatz|carport')
    f['keller'] = has(t, r'keller|abstellraum|abstellkammer|dachbodenanteil')
    f['neubau'] = has(t, r'neubau|erstbezug|neu gebaut')
    f['renoviert'] = has(t, r'renovier|saniert|modernisiert|kernsaniert|neu ?renoviert')
    f['moebliert'] = has(t, r'(?<!teil)(?<!un)m[öo]blier|voll ?ausgestattet(e)? wohnung|komplett eingerichtet')
    f['befristet'] = has(t, r'befristet|auf zeit|zwischenmiete|zwischenvermietung|\bTage?\b.{0,12}\bmonate?\b|ferienwohnung|zeitmiete')
    f['tausch'] = has(t, r'tauschwohnung|wohnungstausch|nur tausch|tausch\b')
    f['gewerblich'] = ('Anbieter-Objekt-ID' in d) or has(t, r'provision(?!sfrei)|courtage|\bGmbH\b|hausverwaltung|immobilien(?!\w)|verantwortliche maklerin|\bKG\b')
    f['wg'] = has(t, r'\bWG\b|wohngemeinschaft|untermiete|zimmer zu vermieten|mitbewohner|\bwohnen gegen')
    f['aussicht'] = has(t, r'seeblick|bergblick|alpenblick|panorama|blick auf (den )?(see|berge|alpen)|traumblick|fernblick')
    f['barrierefrei'] = has(t, r'barrierefrei|rollstuhl')
    f['fussbodenheizung'] = has(t, r'fu(ß|ss)bodenheizung')
    f['hochwertig'] = has(t, r'hochwertig|luxur|exklusiv|gehoben|premium|edel\b')
    return f

def main(db, tracker_id, out):
    c = sqlite3.connect(db)
    rows = c.execute('''
        SELECT l.id, l.title, l.location, v.description
        FROM tracker_listings tl JOIN listings l ON l.id = tl.listing_id
        JOIN listing_versions v ON v.listing_id = l.id
             AND v.version = (SELECT MAX(version) FROM listing_versions WHERE listing_id = l.id)
        WHERE tl.tracker_id = ? AND tl.active = 1''', (tracker_id,)).fetchall()
    res = {str(i): extract(t, loc, d) for i, t, loc, d in rows}
    fs = {'name': 'wohnung_miete_v1', 'tracker_id': tracker_id, 'target': 'price',
          'features': [{'key': k, 'label': l, 'type': ty} for k, l, ty in FEATURES]}
    json.dump({'feature_set': fs, 'rows': res}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(len(res), 'listings,', len(FEATURES), 'features ->', out)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
