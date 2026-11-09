"""Load manually extracted features (JSON from scripts/features/*.py) into the DB.

    python scripts/import_features.py features.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import db

d = json.load(open(sys.argv[1]))
fs = d['feature_set']
assert len(fs['features']) <= 30, 'max 30 features per feature set'
db.init_db()
fid = db.upsert_feature_set(fs['name'], fs.get('tracker_id'),
                            {'features': fs['features'], 'target': fs.get('target', 'price')})
n = db.save_listing_features(fid, {int(k): v for k, v in d['rows'].items()})
print(f'feature set {fs["name"]} (id {fid}): stored features for {n}/{len(d["rows"])} listings')

import ml
ids = [int(k) for k in d['rows']]
try:
    print(f'scored {ml.predict(fs["name"], ids)} listings with the stored model')
except SystemExit as e:        # no model trained yet — fine, train once afterwards
    print(e)
