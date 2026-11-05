"""Local archive of listing images (ad images vanish when the ad is removed)."""
import hashlib
import logging
import os

import requests

logger = logging.getLogger(__name__)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'archive', 'images')
_TYPES = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif'}


def download_image(url: str) -> str | None:
    """Store *url* under archive/images/<sha1(url)>.<ext>; returns the file name
    (reuses an existing file). None on failure."""
    key = hashlib.sha1(url.encode()).hexdigest()
    os.makedirs(IMG_DIR, exist_ok=True)
    for ext in _TYPES.values():
        if os.path.exists(os.path.join(IMG_DIR, key + ext)):
            return key + ext
    try:
        r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        ctype = r.headers.get('content-type', '').split(';')[0].strip()
        if r.status_code != 200 or ctype not in _TYPES or len(r.content) < 200:
            return None
    except Exception as e:
        logger.warning('Image download failed %s: %s', url, e)
        return None
    name = key + _TYPES[ctype]
    with open(os.path.join(IMG_DIR, name), 'wb') as f:
        f.write(r.content)
    return name
