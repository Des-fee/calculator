"""End-to-end test on an Android emulator.

Synthesises handwriting for a few examples, sends it to the app through the
`ink_b64` intent extra, waits for the "DONE" log line and stores the results
and a screenshot of each case in the output folder.

usage: python tests/device_test.py <apk> <out dir>
"""
import base64
import json
import math
import os
import subprocess
import sys
import time

PKG = "io.github.desfee.calculator"
H = 46.0  # character height, dp

# glyph templates in a unit box (x right, y down); each glyph is a list of strokes
G = {
    "0": [[(0.5, 0), (0.15, 0.2), (0.1, 0.5), (0.2, 0.85), (0.5, 1), (0.8, 0.85), (0.9, 0.5), (0.85, 0.15), (0.5, 0)]],
    "1": [[(0.25, 0.25), (0.55, 0), (0.55, 1)]],
    "2": [[(0.1, 0.25), (0.3, 0.04), (0.6, 0), (0.85, 0.2), (0.8, 0.45), (0.1, 1), (0.9, 1)]],
    "3": [[(0.1, 0.1), (0.5, 0), (0.85, 0.2), (0.45, 0.47), (0.88, 0.72), (0.5, 1), (0.1, 0.9)]],
    "4": [[(0.7, 1), (0.7, 0), (0.05, 0.68), (0.95, 0.68)]],
    "5": [[(0.85, 0), (0.2, 0), (0.15, 0.45), (0.6, 0.38), (0.9, 0.65), (0.6, 1), (0.1, 0.9)]],
    "7": [[(0.1, 0), (0.9, 0), (0.35, 1)]],
    "+": [[(0.5, 0.2), (0.5, 0.85)], [(0.15, 0.52), (0.85, 0.52)]],
    "-": [[(0.15, 0.52), (0.85, 0.52)]],
    "=": [[(0.12, 0.4), (0.88, 0.4)], [(0.12, 0.66), (0.88, 0.66)]],
    "x": [[(0.1, 0.38), (0.9, 1)], [(0.9, 0.38), (0.1, 1)]],
    "y": [[(0.1, 0.38), (0.5, 0.75)], [(0.9, 0.38), (0.25, 1.3)]],
    ".": [[(0.3, 0.9), (0.7, 0.9), (0.7, 1.0), (0.3, 1.0), (0.3, 0.9)]],
    "·": [[(0.3, 0.47), (0.7, 0.47), (0.7, 0.57), (0.3, 0.57), (0.3, 0.47)]],
    "×": [[(0.15, 0.3), (0.85, 0.8)], [(0.85, 0.3), (0.15, 0.8)]],
}
WIDTH = {"1": 0.45, "-": 0.7, "+": 0.75, "=": 0.75, ".": 0.25, "·": 0.25, "×": 0.6}


def densify(pts, step=2.0):
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(1, n + 1):
            out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    return out


def write(text, x, y, h=H):
    """Strokes for `text` with the top-left at (x, y); returns (strokes, right edge)."""
    strokes = []
    for ch in text:
        if ch == " ":
            x += 0.4 * h
            continue
        w = WIDTH.get(ch, 0.6) * h
        for st in G[ch]:
            pts = [(x + px * w, y + py * h) for px, py in st]
            strokes.append(densify(pts))
        x += w + 0.28 * h
    return strokes, x


def fraction(num, den, x, y, h=H):
    """Stacked fraction whose bar is at y."""
    sh = 0.8 * h
    n_strokes, n_right = write(num, 0, 0, sh)
    d_strokes, d_right = write(den, 0, 0, sh)
    width = max(n_right, d_right) + 0.2 * h
    out = [densify([(x, y), (x + width, y)])]
    for strokes, right, top in ((n_strokes, n_right, y - 0.25 * h - sh), (d_strokes, d_right, y + 0.25 * h)):
        dx = x + (width - right) / 2
        out += [[(px + dx, py + top) for px, py in s] for s in strokes]
    return out, x + width + 0.3 * h


def case_simple(text):
    return write(text, 24, 140)[0]


def case_fraction():
    s1, x = fraction("1", "2", 24, 160)
    s2, x = write("+", x, 160 - 0.52 * H)
    s3, x = fraction("1", "4", x, 160)
    s4, x = write("=", x, 160 - 0.52 * H)
    return s1 + s2 + s3 + s4


def case_system(pitch=1.9, two_strokes=False, brace_frac=1.0, first="x+y=5", second="x-y=1"):
    """Two equations joined by a brace; `pitch` is the line distance in character heights."""
    top = 120
    l1, r1 = write(first, 60, top)
    l2, r2 = write(second, 60, top + pitch * H)
    span = (pitch + 1.0) * H
    bh = span * brace_frac
    bw = 0.45 * H
    bx, by = 24, top + (span - bh) / 2
    brace = [(1, 0), (0.55, 0.05), (0.5, 0.42), (0, 0.5), (0.5, 0.58), (0.55, 0.95), (1, 1)]
    pts = [(bx + px * bw, by + py * bh) for px, py in brace]
    parts = [pts[:4], pts[3:]] if two_strokes else [pts]
    return [densify(p) for p in parts] + l1 + l2


CASES = [
    ("sum", lambda: case_simple("2+2="), "4"),
    ("equation", lambda: case_simple("2x+1=7"), "x = 3"),
    ("fraction", case_fraction, "3/4"),
    ("system", case_system, "x = 3, y = 2"),
    ("system_close", lambda: case_system(pitch=1.25, first="2x+y=5"), "x = 2, y = 1"),
    ("system_2strokes", lambda: case_system(two_strokes=True), "x = 3, y = 2"),
    ("system_short", lambda: case_system(pitch=1.6, brace_frac=0.65), "x = 3, y = 2"),
    ("check", lambda: case_simple("3+4=7"), "✓"),
    ("times", lambda: case_simple("12×3="), "36"),
    ("mul_dot", lambda: case_simple("4·5="), "20"),
    ("decimal", lambda: case_simple("2.5+1="), "3.5"),
]


def adb(*args, check=True):
    return subprocess.run(["adb", *args], capture_output=True, text=True, check=check).stdout


def run_case(name, strokes, out):
    flat = [[round(v, 1) for p in s for v in p] for s in strokes]
    b64 = base64.b64encode(json.dumps(flat).encode()).decode()
    adb("shell", "am", "force-stop", PKG)
    adb("logcat", "-c")
    adb("shell", "am", "start", "-n", PKG + "/.MainActivity", "--es", "ink_b64", b64)
    results = []
    deadline = time.time() + 240
    log = ""
    while time.time() < deadline:
        time.sleep(3)
        log = adb("logcat", "-d", "-s", "Calc:*", "AndroidRuntime:E", "python.stderr:*", check=False)
        if "DONE" in log or "FATAL" in log or "load failed" in log:
            break
    for line in log.splitlines():
        if "RESULT " in line:
            results.append(json.loads(line.split("RESULT ", 1)[1]))
    time.sleep(1.5)
    with open(os.path.join(out, name + ".png"), "wb") as f:
        f.write(subprocess.run(["adb", "exec-out", "screencap", "-p"], capture_output=True).stdout)
    return results, log


def main():
    apk, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    adb("install", "-r", apk)
    report = []
    passed = 0
    for name, make, expected in CASES:
        t0 = time.time()
        results, log = run_case(name, make(), out)
        answers = [r.get("answer") for r in results]
        ok = any(a is not None and expected in a for a in answers)
        passed += ok
        report.append({"case": name, "expected": expected, "ok": ok, "seconds": round(time.time() - t0, 1),
                       "results": results, "log_tail": log.splitlines()[-25:]})
        print("%-9s %-5s %s" % (name, "OK" if ok else "FAIL", json.dumps(results, ensure_ascii=False)))
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("passed %d/%d" % (passed, len(CASES)))


if __name__ == "__main__":
    main()
