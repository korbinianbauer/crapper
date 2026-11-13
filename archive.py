"""Local archive of listing images (ad images vanish when the ad is removed)."""
import hashlib
import io
import logging
import os
import threading

import requests

import db

try:
    from PIL import Image, ImageOps
except ImportError:                     # without Pillow images are stored unchanged
    Image = None

logger = logging.getLogger(__name__)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'archive', 'images')
JPEG_QUALITY = 85
THUMB_DIR = os.path.join(os.path.dirname(__file__), 'archive', 'thumbs')
THUMB_EDGE = 256                # px; longest side of the card thumbnails
MAX_EDGE = 2000                 # px; larger photos are scaled down (aspect ratio kept)
_TYPES = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif'}


_REF_TABLES: list | None = None


def _ref_tables() -> list:
    """Quantisation tables (luminance, chrominance) of our own output quality."""
    global _REF_TABLES
    if _REF_TABLES is None:
        out = io.BytesIO()
        Image.new('RGB', (16, 16)).save(out, 'JPEG', quality=JPEG_QUALITY)
        q = Image.open(io.BytesIO(out.getvalue())).quantization
        _REF_TABLES = [list(q[i]) for i in sorted(q)]
    return _REF_TABLES


def _already_small_enough(src) -> bool:
    """A JPEG whose quantisation is at least as coarse as ours (quality <= JPEG_QUALITY, in every
    coefficient), within MAX_EDGE and upright: re-encoding cannot help, so it is kept as is."""
    if src.format != 'JPEG' or max(src.size) > MAX_EDGE:
        return False
    if src.getexif().get(0x0112, 1) != 1:
        return False
    tabs = getattr(src, 'quantization', None)
    if not tabs:
        return False
    return all(i in tabs and all(a >= b for a, b in zip(tabs[i], ref))
               for i, ref in enumerate(_ref_tables()))


def _has_transparency(im) -> bool:
    if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
        return im.convert('RGBA').getchannel('A').getextrema()[0] < 255
    return False


def _to_jpeg(data: bytes, ctype: str) -> tuple[bytes, str]:
    """Re-encode *data* as JPEG (quality 85) with EXIF orientation applied. GIFs and images that really
    use transparency stay untouched; a JPEG that does not get smaller (or an undecodable image) too."""
    if Image is None or ctype == 'image/gif':
        return data, ctype
    try:
        src = Image.open(io.BytesIO(data))
        if _has_transparency(src) or _already_small_enough(src):
            return data, ctype
        im = ImageOps.exif_transpose(src)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        if max(im.size) > MAX_EDGE:
            im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=JPEG_QUALITY, optimize=True)
    except Exception as e:
        logger.warning('Image re-encode failed: %s', e)
        return data, ctype
    if out.tell() >= len(data):                 # e.g. flat graphics: keep the smaller original
        return data, ctype
    return out.getvalue(), 'image/jpeg'


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
    data, ctype = _to_jpeg(r.content, ctype)
    name = key + _TYPES[ctype]
    with open(os.path.join(IMG_DIR, name), 'wb') as f:
        f.write(data)
    db.mark_image_checked(name)                 # already processed, never re-encode it again
    return name


def local_name(url: str) -> str | None:
    """File name of an already archived copy of *url*, if any (no download)."""
    key = hashlib.sha1(url.encode()).hexdigest()
    for ext in _TYPES.values():
        if os.path.exists(os.path.join(IMG_DIR, key + ext)):
            return key + ext
    return None


_EXT_TYPES = {v: k for k, v in _TYPES.items()}


def recompress(name: str) -> str:
    """Bring an already archived image to the current storage format (see `_to_jpeg`) and return its
    (possibly new) file name. A converted PNG/WebP becomes `<key>.jpg`; the DB references are updated
    before the old file is removed. Every file is processed once (`db.image_checked`)."""
    if not name:
        return name
    v = db.image_checked_version(name)
    if v >= db.IMAGE_PIPELINE:
        return name
    path = os.path.join(IMG_DIR, name)
    key, ext = os.path.splitext(name)
    if v and Image is not None:                 # processed by an older version: only the new size cap is missing
        try:
            with Image.open(path) as im:
                if max(im.size) <= MAX_EDGE:
                    db.mark_image_checked(name)
                    return name
        except Exception:
            return name
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return name
    new_data, ctype = _to_jpeg(data, _EXT_TYPES.get(ext, 'image/jpeg'))
    if new_data is data:                        # nothing to gain / must stay as is
        db.mark_image_checked(name)
        return name
    new_name = key + '.jpg'
    new_path = os.path.join(IMG_DIR, new_name)
    if new_name != name and os.path.exists(new_path):   # a jpg of this URL already exists: leave it
        db.mark_image_checked(name)
        return name
    tmp = f'{new_path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(new_data)
    os.replace(tmp, new_path)
    db.mark_image_checked(new_name)
    if new_name != name:
        db.rename_image_file(name, new_name)
        try:
            os.remove(path)
        except OSError:
            pass
    logger.info('Image %s: %d → %d bytes', new_name, len(data), len(new_data))
    return new_name


def thumbnail(name: str) -> str | None:
    """Path of the card thumbnail (longest side THUMB_EDGE, JPEG) of archived image *name*, created on first
    use from the archived file and cached in archive/thumbs/<stem>.jpg (the stem survives a PNG→JPG
    conversion of the original). Transparency is flattened onto white. None if it cannot be made."""
    if Image is None or not name or '/' in name or name.startswith('.'):
        return None
    out = os.path.join(THUMB_DIR, os.path.splitext(name)[0] + '.jpg')
    if os.path.exists(out):
        return out
    try:
        im = ImageOps.exif_transpose(Image.open(os.path.join(IMG_DIR, name)))
        if _has_transparency(im):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel('A'))
            im = bg
        im = im.convert('RGB')
        im.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.LANCZOS)
        os.makedirs(THUMB_DIR, exist_ok=True)
        tmp = f'{out}.{os.getpid()}.{threading.get_ident()}.tmp'
        im.save(tmp, 'JPEG', quality=JPEG_QUALITY, optimize=True)
        os.replace(tmp, out)
        return out
    except Exception as e:
        logger.warning('Thumbnail failed for %s: %s', name, e)
        return None


def best_image_file(source, image_url: str | None, image_file: str | None) -> str | None:
    """File the card thumbnail is made from: the large variant of the ad's picture if archived (search
    cards only carry a tiny one), else the archived card file."""
    if image_url:
        try:
            if name := local_name(source.full_image_url(image_url)):
                return name
        except Exception:
            pass
    return image_file
