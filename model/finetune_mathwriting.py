"""Fine-tune the CoMER handwriting model for the calculator.

Training data, all rendered exactly like the app renders a line of ink
(app/src/main/java/.../Recognizer.kt: ~45 px character height, 3.6 px white
strokes on black, binary):

  * MathWriting (Google, 2024): ~230k human-written online formulas, plus a
    part of its synthetic split. Labels are converted to the CROHME token set
    the model already knows; formulas with other symbols are skipped.
  * synthetic calculator lines assembled from MathWriting's isolated glyphs:
    arithmetic with ×, ·, ÷, decimal points, x and y as variables, powers,
    fractions, brackets, equations. These target the confusions seen in the
    app (× vs x, · vs ., 4 vs y, exponents).
  * CROHME train images from the CoMER repository (keeps what the model knows).

Training starts from the published CROHME checkpoint and stops after a time
budget. The best checkpoint (by exact-match rate on MathWriting validation
inks) is written to OUT_DIR/comer_mathwriting.ckpt in the same format as the
original checkpoint, so model/export_onnx.py can export it unchanged
(COMER_CKPT=<file>).

Environment variables
  COMER_DIR   CoMER repository (git clone https://github.com/Green-Wood/CoMER)
  OUT_DIR     where checkpoints and metrics go               (default: ./out)
  MW_URL      MathWriting archive                            (default: full 2.9 GB archive)
  HOURS       training time budget in hours                  (default: 9.5)
  SMOKE=1     tiny CPU run on the 1.5 MB excerpt (used by CI to check this script)
  INIT_CKPT   start from this checkpoint instead of the CROHME one (to continue training)
"""

import io
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from xml.etree import ElementTree

import numpy as np
import torch
from PIL import Image, ImageDraw

SMOKE = os.environ.get("SMOKE") == "1"
COMER_DIR = os.path.abspath(os.environ.get("COMER_DIR", "CoMER"))
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", "out"))
MW_URL = os.environ.get(
    "MW_URL",
    "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz" if SMOKE
    else "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz",
)
HOURS = float(os.environ.get("HOURS", "0.05" if SMOKE else "9.5"))
INIT_CKPT = os.environ.get("INIT_CKPT") or os.path.join(
    COMER_DIR, "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt")

MW_SYNTHETIC_MAX = 400 if SMOKE else 120_000   # how many of MathWriting's synthetic inks to use
CALC_PER_EPOCH = 64 if SMOKE else 70_000        # generated calculator lines per epoch
CROHME_REPEAT = 1 if SMOKE else 2
VALID_MAX = 40 if SMOKE else 600                # MathWriting validation inks used for the score
EVAL_EVERY_MIN = 1 if SMOKE else 50
MAX_BATCH = 4 if SMOKE else 16
MAX_PIXELS = 2e5 if SMOKE else 3.6e5             # per batch (sum over images of padded H*W)
LR = 2e-4
WARMUP = 20 if SMOKE else 1000

sys.path.insert(0, COMER_DIR)
from comer.datamodule import vocab  # noqa: E402
from comer.model.comer import CoMER  # noqa: E402
from comer.utils.utils import ce_loss, to_bi_tgt_out  # noqa: E402

VOCAB = set(vocab.word2idx)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()


def log(*a):
    print("[%6.1f min]" % ((time.time() - T0) / 60), *a, flush=True)


# ---------------------------------------------------------------------------
# Rendering: the same as Recognizer.render() + typicalHeight() in the app
# ---------------------------------------------------------------------------

def typical_height(strokes):
    """Median height of character-like strokes (Ink.kt: typicalHeight), in ink units."""
    sizes = [(np.ptp(s[:, 0]), np.ptp(s[:, 1])) for s in strokes]
    hs = sorted(h for w, h in sizes if h > w * 0.35 and h > 4)
    if not hs:
        a = sorted(max(h, w * 0.5) for w, h in sizes)
        return max(a[len(a) // 2], 8.0)
    return max(hs[len(hs) // 2], 8.0)


def normalise_units(strokes):
    """Scale an ink so a character is ~100 units tall (like screen pixels on a phone),
    so that the pixel thresholds of typical_height() mean the same as in the app."""
    sizes = [max(np.ptp(s[:, 1]), 1e-6) for s in strokes]
    ref = sorted(sizes)[len(sizes) // 2]
    k = 100.0 / max(ref, 1e-6)
    return [s * k for s in strokes]


def render(strokes, target=45.0, width=3.6, jitter=None):
    """Strokes ([n, 2] arrays, y down) -> uint8 image, white ink on black, like the app."""
    ch = typical_height(strokes)
    pts = np.concatenate(strokes)
    left, top = pts.min(0)
    bw, bh = pts.max(0) - pts.min(0)
    margin = 11.0
    scale = target / ch
    max_scale = min((256 - 2 * margin) / max(bh, 1.0), (1024 - 2 * margin) / max(bw, 1.0))
    scale = min(scale, max_scale)
    w = max(16, int(math.ceil(bw * scale + 2 * margin)))
    h = max(16, int(math.ceil(bh * scale + 2 * margin)))
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    lw = max(1, int(round(width)))
    r = width / 2
    for s in strokes:
        p = (s - (left, top)) * scale + margin
        if np.ptp(s[:, 0]) * scale < 1.5 and np.ptp(s[:, 1]) * scale < 1.5:
            x, y = p[0]
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
            continue
        d.line([tuple(q) for q in p], fill=255, width=lw, joint="curve")
        for x, y in (p[0], p[-1]):  # round caps
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    a = np.asarray(img)
    return np.where(a > 127, 255, 0).astype(np.uint8)


def augment(strokes, rng):
    """Small affine distortion of a whole ink (slant, rotation, aspect)."""
    shear = rng.uniform(-0.18, 0.18)
    ang = math.radians(rng.uniform(-2.5, 2.5))
    sx = rng.uniform(0.9, 1.12)
    c, s_ = math.cos(ang), math.sin(ang)
    m = np.array([[c * sx, -s_], [s_ * sx, c]], dtype=np.float32) @ np.array([[1, shear], [0, 1]], dtype=np.float32)
    return [st @ m.T for st in strokes]


def render_train(strokes, rng):
    return render(augment(strokes, rng), target=rng.uniform(39, 52), width=rng.choice([3.0, 3.6, 4.0, 4.5]))


def size_estimate(strokes, target=45.0):
    ch = typical_height(strokes)
    pts = np.concatenate(strokes)
    bw, bh = pts.max(0) - pts.min(0)
    margin = 11.0
    scale = min(target / ch, (256 - 2 * margin) / max(bh, 1.0), (1024 - 2 * margin) / max(bw, 1.0))
    return max(16, int(bh * scale + 2 * margin)), max(16, int(bw * scale + 2 * margin))


# ---------------------------------------------------------------------------
# MathWriting labels -> CROHME tokens
# ---------------------------------------------------------------------------

CMD_MAP = {"\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq", "\\to": "\\rightarrow", "\\lt": "<", "\\gt": ">",
           "\\dots": "\\ldots", "\\longrightarrow": "\\rightarrow", "\\lbrace": "\\{", "\\rbrace": "\\}"}
SKIP = {"\\,", "\\;", "\\!", "\\ ", "\\quad", "\\qquad", "\\left", "\\right", "\\displaystyle"}
FUNC_WORDS = [("sin", "\\sin"), ("cos", "\\cos"), ("tan", "\\tan"), ("log", "\\log"), ("lim", "\\lim")]


def tokenize_latex(s):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            t = m.group(0)
            i += len(t)
            if t not in SKIP:
                toks.append(CMD_MAP.get(t, t))
            continue
        toks.append(c)
        i += 1
    return toks


def to_crohme(label):
    """MathWriting normalizedLabel -> CROHME token list, or None when it uses unknown symbols."""
    if not label:
        return None
    toks = tokenize_latex(label)
    out = []
    i = 0
    while i < len(toks):
        for word, cmd in FUNC_WORDS:  # MathWriting writes functions as letters, CROHME as commands
            n = len(word)
            if toks[i:i + n] == list(word) and not (out and out[-1].isalpha() and len(out[-1]) == 1) \
                    and not (i + n < len(toks) and toks[i + n].isalpha() and len(toks[i + n]) == 1):
                out.append(cmd)
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    if not out or len(out) > 150 or any(t not in VOCAB for t in out):
        return None
    return out


# ---------------------------------------------------------------------------
# Reading MathWriting straight from the .tgz stream (no 12 GB on disk)
# ---------------------------------------------------------------------------

def parse_inkml(data):
    root = ElementTree.fromstring(data)
    strokes, ann = [], {}
    for el in root:
        tag = el.tag.split("}")[-1]
        if tag == "annotation":
            ann[el.attrib.get("type")] = el.text
        elif tag == "trace":
            pts = [p.split() for p in el.text.strip().split(",")]
            a = np.array([[float(p[0]), float(p[1])] for p in pts if len(p) >= 2], dtype=np.float32)
            if len(a):
                strokes.append(a)
    return strokes, ann


def load_mathwriting():
    log("downloading and parsing", MW_URL)
    rng = random.Random(1)
    train, synth, valid, symbols_lines = [], [], [], []
    train_inks = {}  # sampleId -> strokes (for the isolated glyphs)
    n_seen = n_skip = 0
    with urllib.request.urlopen(MW_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name
                if name.endswith("symbols.jsonl"):
                    symbols_lines = tar.extractfile(m).read().decode().splitlines()
                    continue
                if not name.endswith(".inkml"):
                    continue
                split = name.split("/")[-2]
                if split in ("test", "symbols"):
                    continue
                if split == "synthetic" and (len(synth) >= MW_SYNTHETIC_MAX or rng.random() > 0.4):
                    continue
                if split == "valid" and len(valid) >= VALID_MAX * 3:
                    continue
                strokes, ann = parse_inkml(tar.extractfile(m).read())
                if not strokes:
                    continue
                n_seen += 1
                if split == "train":
                    train_inks[ann.get("sampleId", os.path.basename(name)[:-6])] = strokes
                toks = to_crohme(ann.get("normalizedLabel") or ann.get("label"))
                if toks is None:
                    n_skip += 1
                    continue
                strokes = normalise_units(strokes)
                item = (strokes, toks)
                {"train": train, "synthetic": synth, "valid": valid}[split].append(item)
                if n_seen % 20000 == 0:
                    log("  %d inks read, %d kept" % (n_seen, n_seen - n_skip))
    log("MathWriting: train %d, synthetic %d, valid %d (skipped %d with symbols the model does not know)"
        % (len(train), len(synth), len(valid), n_skip))
    glyphs = {}
    for line in symbols_lines:
        o = json.loads(line)
        src = train_inks.get(o["sourceSampleId"])
        if src is None or max(o["strokeIndices"]) >= len(src):
            continue
        glyphs.setdefault(o["label"], []).append([src[i] for i in o["strokeIndices"]])
    log("isolated glyphs:", {k: len(v) for k, v in sorted(glyphs.items()) if k in CALC_GLYPHS})
    log("all glyph labels:", " ".join(sorted(glyphs)))
    return train, synth, valid[:VALID_MAX], glyphs


# ---------------------------------------------------------------------------
# Synthetic calculator lines from isolated glyphs
# ---------------------------------------------------------------------------

# glyph box in "em" (digit height = 1, y down, baseline at 1): (top, bottom) or special handling
SHORT = set("acemnorsuvwxz")
DESC = set("gpqy")
BOXES = {".": (0.86, 1.0), "\\cdot": (0.53, 0.67), "+": (0.25, 0.9), "\\times": (0.32, 0.88), "\\div": (0.25, 0.92), "<": (0.3, 0.9), ">": (0.3, 0.9),
         "(": (-0.12, 1.15), ")": (-0.12, 1.15), "/": (-0.05, 1.08)}
FLAT = {"-": 0.6, "=": 0.6}  # centre height of flat glyphs
CALC_GLYPHS = set("0123456789xyabcnt+-=()/<>.") | {"\\times", "\\div", "\\cdot"}


class Layout:
    """Strokes in em units plus the matching CROHME tokens."""

    def __init__(self):
        self.strokes = []
        self.tokens = []
        self.x = 0.0


def _fit(ink, x0, top, bottom, rng, width=None):
    p = np.concatenate(ink)
    lo, hi = p.min(0), p.max(0)
    w, h = max(hi[0] - lo[0], 1e-6), max(hi[1] - lo[1], 1e-6)
    if width is not None:
        k = width / w
        cy = (top + bottom) / 2
        return [np.stack([(s[:, 0] - lo[0]) * k + x0, (s[:, 1] - (lo[1] + hi[1]) / 2) * k + cy], 1) for s in ink], width
    k = (bottom - top) / h
    return [np.stack([(s[:, 0] - lo[0]) * k + x0, (s[:, 1] - lo[1]) * k + top], 1) for s in ink], w * k


def _dot(x, y, r, rng):
    n = 8
    a = np.linspace(0, 2 * math.pi, n)
    return [np.stack([x + r * np.cos(a) * rng.uniform(0.7, 1.2), y + r * np.sin(a)], 1).astype(np.float32)]


def put(lay, tok, glyphs, rng, scale=1.0, dy=0.0):
    """Append one glyph (a CROHME token) at the cursor."""
    j = rng.uniform(-0.04, 0.04)
    if tok in ("\\cdot", ".") and (tok not in glyphs or rng.random() < 0.5):
        y = (0.62 if tok == "\\cdot" else 0.95) * scale + dy + j
        r = rng.uniform(0.035, 0.07) * scale
        lay.strokes += _dot(lay.x + r, y, r, rng)
        lay.x += 2 * r + rng.uniform(0.12, 0.25) * scale
        lay.tokens.append(tok)
        return
    ink = rng.choice(glyphs[tok])
    sv = rng.uniform(0.88, 1.12)  # writers vary glyph sizes
    if tok in FLAT:
        cy = FLAT[tok] * scale + dy + j
        st, w = _fit(ink, lay.x, cy, cy, rng, width=rng.uniform(0.45, 0.75) * scale * sv)
    else:
        if tok in BOXES:
            t, b = BOXES[tok]
        elif tok in SHORT:
            t, b = 0.4, 1.0
        elif tok in DESC:
            t, b = 0.4, 1.33
        else:
            t, b = 0.0, 1.0
        bottom = b * scale + dy + j
        st, w = _fit(ink, lay.x, bottom - (b - t) * scale * sv, bottom, rng)
    lay.strokes += st
    lay.x += w + rng.uniform(0.12, 0.32) * scale
    lay.tokens.append(tok)


def put_number(lay, glyphs, rng, scale=1.0, dy=0.0, decimal_p=0.15):
    n = rng.choice([1, 1, 1, 2, 2, 3])
    digits = [str(rng.randint(1, 9))] + [str(rng.randint(0, 9)) for _ in range(n - 1)]
    for d in digits:
        put(lay, d, glyphs, rng, scale, dy)
    if rng.random() < decimal_p:
        put(lay, ".", glyphs, rng, scale, dy)
        put(lay, str(rng.randint(0, 9)), glyphs, rng, scale, dy)


def put_power(lay, glyphs, rng, base_fn, scale=1.0, dy=0.0):
    base_fn()
    lay.tokens += ["^", "{"]
    lay.x -= 0.08 * scale
    put(lay, rng.choice("2223n"), glyphs, rng, scale * 0.6, dy - 0.42 * scale)
    lay.tokens.append("}")


def put_fraction(lay, glyphs, rng, scale=1.0):
    sub = 0.8 * scale
    num, den = Layout(), Layout()
    put_number(num, glyphs, rng, sub, decimal_p=0)
    if rng.random() < 0.3:
        put(den, rng.choice("xy"), glyphs, rng, sub)
    else:
        put_number(den, glyphs, rng, sub, decimal_p=0)
    wn, wd = num.x, den.x
    width = max(wn, wd) + 0.2 * scale
    bar_y = 0.6 * scale
    x0 = lay.x
    lay.strokes.append(np.array([[x0, bar_y], [x0 + width / 2, bar_y + rng.uniform(-0.03, 0.03)], [x0 + width, bar_y]],
                                dtype=np.float32))
    for part, w, top in ((num, wn, bar_y - 0.15 * scale - 1.0 * sub), (den, wd, bar_y + 0.15 * scale)):
        dx = x0 + (width - w) / 2
        lay.strokes += [s + (dx, top) for s in part.strokes]
    lay.tokens += ["\\frac", "{"] + num.tokens + ["}", "{"] + den.tokens + ["}"]
    lay.x = x0 + width + rng.uniform(0.15, 0.3) * scale


def calc_line(glyphs, rng):
    """A random calculator line: arithmetic, an equation, powers, fractions, brackets."""
    lay = Layout()
    var = rng.choice(["x", "x", "y", "a", "n", "t"])
    ops = ["+", "-", "\\times", "\\times", "\\cdot", "\\div", "/"]

    def operand():
        r = rng.random()
        if r < 0.45:
            put_number(lay, glyphs, rng)
        elif r < 0.6:
            if rng.random() < 0.5:
                put_number(lay, glyphs, rng, decimal_p=0)
            put(lay, var, glyphs, rng)
        elif r < 0.72:
            put_power(lay, glyphs, rng, lambda: put(lay, var, glyphs, rng) if rng.random() < 0.6
                      else put_number(lay, glyphs, rng, decimal_p=0))
        elif r < 0.85:
            put_fraction(lay, glyphs, rng)
        else:
            put(lay, "(", glyphs, rng)
            put_number(lay, glyphs, rng)
            put(lay, rng.choice(["+", "-"]), glyphs, rng)
            put_number(lay, glyphs, rng) if rng.random() < 0.6 else put(lay, var, glyphs, rng)
            put(lay, ")", glyphs, rng)

    for k in range(rng.choice([1, 2, 2, 3, 3, 4])):
        if k:
            put(lay, rng.choice(ops), glyphs, rng)
        operand()
    if rng.random() < 0.4:  # equation: "... = number"
        put(lay, "=", glyphs, rng)
        put_number(lay, glyphs, rng)
    return lay


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_crohme_train():
    out = []
    with zipfile.ZipFile(os.path.join(COMER_DIR, "data.zip")) as zf:
        lines = zf.read("data/train/caption.txt").decode().strip().split("\n")
        for line in lines[: 60 if SMOKE else None]:
            parts = line.strip().split()
            toks = parts[1:]
            if any(t not in VOCAB for t in toks):
                continue
            with zf.open("data/train/img/%s.bmp" % parts[0]) as f:
                img = np.array(Image.open(f).convert("L"))
            out.append((img, toks))
    log("CROHME train images:", len(out))
    return out


def fit_range(img):
    """CoMER's ScaleToLimitRange for the CROHME bitmaps."""
    h, w = img.shape
    k = min(256 / h, 1024 / w)
    if k < 1:
        img = np.array(Image.fromarray(img).resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR))
    k = max(16 / img.shape[0], 16 / img.shape[1])
    if k > 1:
        img = np.array(Image.fromarray(img).resize((int(img.shape[1] * k + 0.5), int(img.shape[0] * k + 0.5)), Image.BILINEAR))
    return img


class Mixed(torch.utils.data.Dataset):
    """Index space: MathWriting inks | CROHME images (repeated) | generated calculator lines."""

    def __init__(self, inks, crohme, glyphs, epoch=0):
        self.inks, self.crohme, self.glyphs = inks, crohme, glyphs
        self.n_calc = CALC_PER_EPOCH if glyphs else 0
        self.epoch = epoch
        self.sizes = [size_estimate(s) for s, _ in inks]
        self.sizes += [im.shape for im, _ in crohme] * CROHME_REPEAT
        self.sizes += [(110, 420)] * self.n_calc

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        rng = random.Random(hash((self.epoch, i)))
        if i < len(self.inks):
            strokes, toks = self.inks[i]
            img = render_train(strokes, rng)
        elif i < len(self.inks) + len(self.crohme) * CROHME_REPEAT:
            img, toks = self.crohme[(i - len(self.inks)) % len(self.crohme)]
            img = fit_range(img)
        else:
            for _ in range(10):
                lay = calc_line(self.glyphs, rng)
                if lay.strokes:
                    break
            strokes = normalise_units([s.astype(np.float32) for s in lay.strokes])
            img, toks = render_train(strokes, rng), lay.tokens
        return img, vocab.words2indices(toks)


def batches(sizes, rng):
    """Group indices of similar size; each batch stays under MAX_PIXELS of padded area."""
    order = sorted(range(len(sizes)), key=lambda i: (sizes[i][0] * sizes[i][1], rng.random()))
    out, cur, mh, mw = [], [], 0, 0
    for i in order:
        h, w = sizes[i]
        nh, nw = max(mh, h), max(mw, w)
        if cur and (len(cur) >= MAX_BATCH or nh * nw * (len(cur) + 1) > MAX_PIXELS):
            out.append(cur)
            cur, nh, nw = [], h, w
        cur.append(i)
        mh, mw = nh, nw
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(items):
    imgs = [torch.from_numpy(im).float().div(255) for im, _ in items]
    hs, ws = [im.shape[0] for im in imgs], [im.shape[1] for im in imgs]
    x = torch.zeros(len(imgs), 1, max(hs), max(ws))
    mask = torch.ones(len(imgs), max(hs), max(ws), dtype=torch.bool)
    for k, im in enumerate(imgs):
        x[k, 0, : hs[k], : ws[k]] = im
        mask[k, : hs[k], : ws[k]] = False
    return x, mask, [idx for _, idx in items]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    model = CoMER(**{k: hp[k] for k in ("d_model", "growth_rate", "num_layers", "nhead", "num_decoder_layers",
                                        "dim_feedforward", "dropout", "dc", "cross_coverage", "self_coverage")})
    sd = {k[len("comer_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("comer_model.")}
    model.load_state_dict(sd)
    return model, hp


def save_model(model, hp, path, meta):
    sd = {"comer_model." + k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"hyper_parameters": hp, "state_dict": sd, "meta": meta}, path)


@torch.no_grad()
def exprate(model, items, beam=3 if SMOKE else 5):
    """Share of formulas recognised exactly (rendered like the app, no augmentation)."""
    model.eval()
    ok = 0
    for k in range(0, len(items), 8):
        chunk = items[k:k + 8]
        x, mask, _ = collate([(render(s), vocab.words2indices(t)) for s, t in chunk])
        hyps = model.beam_search(x.to(DEVICE), mask.to(DEVICE), beam_size=beam, max_len=160, alpha=1.0,
                                 early_stopping=False, temperature=1.0)
        for h, (_, t) in zip(hyps, chunk):
            ok += vocab.indices2words(h.seq) == t
    model.train()
    return ok / max(1, len(items))


def main():
    log("device:", DEVICE, "| smoke" if SMOKE else "")
    train, synth, valid, glyphs = load_mathwriting()
    missing = CALC_GLYPHS - set(glyphs)
    if missing:
        log("no isolated glyphs for", sorted(missing), "- calculator lines are generated without them")
    usable = {k: v for k, v in glyphs.items() if k in CALC_GLYPHS}
    calc_ok = all(t in usable for t in "0123456789+-=") and "\\times" in usable and ("x" in usable)
    if not calc_ok:
        log("not enough glyphs for calculator lines; they are skipped")
    crohme = load_crohme_train()

    rng = random.Random(0)
    calc_valid = []
    if calc_ok:
        for _ in range(16 if SMOKE else 300):
            lay = calc_line(usable, rng)
            calc_valid.append((normalise_units(lay.strokes), lay.tokens))
    if not valid:  # the excerpt may have no valid inks
        valid = train[: VALID_MAX]

    model, hp = load_model(INIT_CKPT)
    model.to(DEVICE).train()
    base_mw = exprate(model, valid)
    base_calc = exprate(model, calc_valid) if calc_valid else float("nan")
    log("before training: ExpRate MathWriting valid %.1f%%, calculator lines %.1f%%" % (100 * base_mw, 100 * base_calc))
    metrics = {"before": {"mathwriting_valid": base_mw, "calculator": base_calc}, "evals": []}
    best = base_mw
    save_model(model, hp, os.path.join(OUT_DIR, "comer_mathwriting.ckpt"), {"step": 0, "valid": base_mw})

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    budget = max(60.0, HOURS * 3600 - (time.time() - T0))
    t_start = time.time()
    step, epoch, last_eval = 0, 0, time.time()
    done = False
    while not done:
        ds = Mixed(train + synth, crohme, usable if calc_ok else None, epoch)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches(ds.sizes, random.Random(epoch)),
                                             collate_fn=collate, num_workers=0 if SMOKE else 4,
                                             persistent_workers=False, prefetch_factor=None if SMOKE else 4)
        log("epoch %d: %d samples" % (epoch, len(ds)))
        run_loss, run_n = 0.0, 0
        for x, mask, idx in loader:
            elapsed = time.time() - t_start
            frac = min(1.0, elapsed / budget)
            lr = LR * min(1.0, (step + 1) / WARMUP) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, mask = x.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
            tgt, out = to_bi_tgt_out(idx, DEVICE)
            try:
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                    out_hat = model(x, mask, tgt)
                    loss = ce_loss(out_hat.float(), out)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                log("out of memory on a batch of", tuple(x.shape), "- skipped")
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            step += 1
            if torch.isfinite(loss):
                run_loss += loss.item()
                run_n += 1
            if step % (5 if SMOKE else 200) == 0:
                log("step %d  loss %.4f  lr %.2e" % (step, run_loss / max(1, run_n), lr))
                run_loss, run_n = 0.0, 0
            out_of_time = time.time() - t_start > budget
            if out_of_time or time.time() - last_eval > EVAL_EVERY_MIN * 60 or (SMOKE and step % 6 == 0):
                v = exprate(model, valid)
                c = exprate(model, calc_valid) if calc_valid else float("nan")
                log("eval step %d: MathWriting valid %.1f%%, calculator %.1f%%" % (step, 100 * v, 100 * c))
                metrics["evals"].append({"step": step, "mathwriting_valid": v, "calculator": c,
                                         "minutes": (time.time() - T0) / 60})
                if v >= best:
                    best = v
                    save_model(model, hp, os.path.join(OUT_DIR, "comer_mathwriting.ckpt"), {"step": step, "valid": v})
                    log("  saved best checkpoint")
                save_model(model, hp, os.path.join(OUT_DIR, "last.ckpt"), {"step": step, "valid": v})
                with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=1)
                last_eval = time.time()
            if out_of_time or (SMOKE and step >= 12):
                done = True
                break
        epoch += 1
    metrics["best_mathwriting_valid"] = best
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    log("done: best MathWriting valid ExpRate %.1f%% (was %.1f%%)" % (100 * best, 100 * base_mw))


if __name__ == "__main__":
    main()
