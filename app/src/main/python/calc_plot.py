"""Graphs: sampling of what a line draws (see calc_engine.plot_spec) for the Android plot views.

scene(specs, xmin, xmax, ymin, ymax, w, h) -> JSON {"layers": [...], "points": [...]}:
  layer {"type": "lines", "label", "paths": [[x0, y0, x1, y1, ...], ...]}       curves, implicit curves, fields
  layer {"type": "region", "label", "nx", "ny", "mask": "0110..."}              cells (row 0 = top) that satisfy
  point {"x", "y", "label", "kind": zero|extremum|intersection}
surface(spec, cx, cy, r, n) -> JSON {"z": [[...]] (row = y, col = x; null = undefined), "zmin", "zmax", "label"}
"""
import json
import math

import sympy as sp

import calc_engine as ce

X, Y = sp.Symbol("x"), sp.Symbol("y")


def _expr(text):
    return sp.sympify(text, locals={"E": sp.E, "I": sp.I, "pi": sp.pi})


def _fn(expr, args):
    f = sp.lambdify(args, expr, modules=["math"])

    def safe(*v):
        try:
            r = f(*v)
            if isinstance(r, complex):
                return r.real if abs(r.imag) < 1e-12 else None
            r = float(r)
            return r if math.isfinite(r) else None
        except (ValueError, ZeroDivisionError, OverflowError, TypeError, AttributeError):
            return None
    return safe


def _f(e):
    return ce.X(e)[0]


def _label(spec):
    if spec.get("l"):
        return spec["l"].replace("**", "^").replace("*", "·")
    t = spec["t"]
    if t == "y":
        return "y = " + _f(_expr(spec["e"]))
    if t == "param":
        return "x = %s, y = %s" % (_f(_expr(spec["x"])), _f(_expr(spec["y"])))
    if t == "polar":
        return "r = " + _f(_expr(spec["r"]))
    if t == "implicit":
        return _f(_expr(spec["e"])) + " = 0"
    if t == "region":
        return ", ".join(_f(_expr(c.rsplit(" ", 2)[0])) + " " + c.rsplit(" ", 2)[1] + " 0" for c in spec["c"])
    if t == "ode":
        return "y′ = " + _f(_expr(spec["e"]))
    if t == "surface":
        return "z = " + _f(_expr(spec["e"]))
    return ""


def _polyline(points, yspan):
    """Split [(x, y|None)] at gaps and at jumps (asymptotes) into flat paths."""
    paths, cur, prev = [], [], None
    for x, y in points:
        if y is None or (prev is not None and abs(y - prev) > yspan * 3):
            if len(cur) >= 4:
                paths.append(cur)
            cur = []
        if y is not None:
            cur += [round(x, 6), round(max(min(y, 1e9), -1e9), 6)]
        prev = y
    if len(cur) >= 4:
        paths.append(cur)
    return paths


def _bisect(g, a, b, ga, steps=50):
    for _ in range(steps):
        m = (a + b) / 2
        gm = g(m)
        if gm is None:
            return None
        if (gm > 0) == (ga > 0):
            a, ga = m, gm
        else:
            b = m
    return (a + b) / 2


def _num(v):
    if abs(v) < 1e-9:
        return "0"
    s = "%.4g" % v
    return s.replace("-", "−")


def _special_points(curves, xmin, xmax, ymin, ymax):
    """Zeros, extrema and intersections of y = f(x) curves inside the view."""
    pts = []
    n = 400
    xs = [xmin + (xmax - xmin) * k / n for k in range(n + 1)]

    def roots_of(g):
        out = []
        prev_x, prev_v = None, None
        for x in xs:
            v = g(x)
            if v is not None and prev_v is not None and (v > 0) != (prev_v > 0) and abs(v - prev_v) < (ymax - ymin):
                r = _bisect(g, prev_x, x, prev_v)
                if r is not None:
                    out.append(r)
            elif v == 0:
                out.append(x)
            prev_x, prev_v = x, v
        return out

    for f, expr in curves:
        for r in roots_of(f):
            pts.append({"x": r, "y": 0.0, "kind": "zero", "label": "(%s; 0)" % _num(r)})
        d = _fn(sp.diff(expr, X), (X,))
        for r in roots_of(d):
            yv = f(r)
            if yv is not None and ymin <= yv <= ymax:
                pts.append({"x": r, "y": yv, "kind": "extremum", "label": "(%s; %s)" % (_num(r), _num(yv))})
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            fi, fj = curves[i][0], curves[j][0]

            def g(x, fi=fi, fj=fj):
                a, b = fi(x), fj(x)
                return None if a is None or b is None else a - b
            for r in roots_of(g):
                yv = fi(r)
                if yv is not None and ymin <= yv <= ymax:
                    pts.append({"x": r, "y": yv, "kind": "intersection", "label": "(%s; %s)" % (_num(r), _num(yv))})
    # one label per place
    out = []
    for p in pts:
        if all(abs(p["x"] - q["x"]) > (xmax - xmin) / 200 or abs(p["y"] - q["y"]) > (ymax - ymin) / 200 for q in out):
            out.append(p)
    return out[:40]


def _marching(F, xmin, xmax, ymin, ymax, nx, ny):
    """Segments of F(x, y) = 0 on a grid (marching squares), as 4-number paths."""
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    vals = [[F(xmin + i * dx, ymin + j * dy) for i in range(nx + 1)] for j in range(ny + 1)]
    segs = []

    def edge(i, j, k):
        # point on edge k of cell (i, j): 0 bottom, 1 right, 2 top, 3 left
        (i0, j0), (i1, j1) = [((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                              ((i, j + 1), (i + 1, j + 1)), ((i, j), (i, j + 1))][k]
        a, b = vals[j0][i0], vals[j1][i1]
        t = a / (a - b) if a != b else 0.5
        return xmin + (i0 + (i1 - i0) * t) * dx, ymin + (j0 + (j1 - j0) * t) * dy

    for j in range(ny):
        for i in range(nx):
            c = [vals[j][i], vals[j][i + 1], vals[j + 1][i + 1], vals[j + 1][i]]
            if any(v is None for v in c):
                continue
            # skip cells where the sign change is a pole (1/x = 0), not a zero
            if max(abs(v) for v in c) > 1e6:
                continue
            crossing = [k for k, (a, b) in enumerate([(c[0], c[1]), (c[1], c[2]), (c[3], c[2]), (c[0], c[3])])
                        if (a > 0) != (b > 0)]
            pts = [edge(i, j, k) for k in crossing]
            for a in range(0, len(pts) - 1, 2):
                segs.append([pts[a][0], pts[a][1], pts[a + 1][0], pts[a + 1][1]])
    return segs


def _join(segs, eps):
    """Join touching segments into longer paths (fewer draw calls, smoother lines)."""
    paths = []
    for s in segs:
        if paths and abs(paths[-1][-2] - s[0]) < eps and abs(paths[-1][-1] - s[1]) < eps:
            paths[-1] += s[2:]
        else:
            paths.append(list(s))
    return [[round(v, 6) for v in p] for p in paths]


def _layer(spec, xmin, xmax, ymin, ymax, w, h):
    t = spec["t"]
    label = _label(spec)
    yspan = ymax - ymin
    if t == "y":
        f = _fn(_expr(spec["e"]), (X,))
        n = max(200, min(1200, int(w)))
        pts = [(xmin + (xmax - xmin) * k / n, None) for k in range(n + 1)]
        pts = [(x, f(x)) for x, _ in pts]
        return {"type": "lines", "label": label, "paths": _polyline(pts, yspan)}
    if t in ("param", "polar"):
        v = sp.Symbol(spec["v"])
        if t == "param":
            fx, fy = _fn(_expr(spec["x"]), (v,)), _fn(_expr(spec["y"]), (v,))
            ex = _expr(spec["x"]) + _expr(spec["y"])
        else:
            r = _fn(_expr(spec["r"]), (v,))
            ex = _expr(spec["r"])

            def fx(a):
                rv = r(a)
                return None if rv is None else rv * math.cos(a)

            def fy(a):
                rv = r(a)
                return None if rv is None else rv * math.sin(a)
        periodic = ex.has(sp.sin, sp.cos, sp.tan) or t == "polar"
        t0, t1 = (0.0, 2 * math.pi) if periodic else (-10.0, 10.0)
        if t == "polar":
            # r = cos(φ/2) and the like close only after 4π
            t1 = 4 * math.pi if any(isinstance(a, sp.Rational) and a.q > 1 for a in ex.atoms(sp.Rational)) else t1
        n = 1500
        paths, cur = [], []
        for k in range(n + 1):
            a = t0 + (t1 - t0) * k / n
            px, py = fx(a), fy(a)
            if px is None or py is None:
                if len(cur) >= 4:
                    paths.append(cur)
                cur = []
                continue
            cur += [round(px, 6), round(py, 6)]
        if len(cur) >= 4:
            paths.append(cur)
        return {"type": "lines", "label": label, "paths": paths}
    nx = 140
    ny = max(20, int(nx * h / max(w, 1)))
    if t == "implicit":
        F = _fn(_expr(spec["e"]), (X, Y))
        segs = _marching(F, xmin, xmax, ymin, ymax, nx, ny)
        return {"type": "lines", "label": label, "paths": _join(segs, (xmax - xmin) / nx / 100)}
    if t == "region":
        conds = []
        for c in spec["c"]:
            e, op, _ = c.rsplit(" ", 2)
            conds.append((_fn(_expr(e), (X, Y)), op))
        nx, ny = 120, max(20, int(120 * h / max(w, 1)))
        dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
        rows = []
        for j in range(ny):
            yv = ymax - (j + 0.5) * dy
            row = []
            for i in range(nx):
                xv = xmin + (i + 0.5) * dx
                ok = True
                for f, op in conds:
                    v = f(xv, yv)
                    if v is None or (v <= 0 if op == ">" else v >= 0):
                        ok = False
                        break
                row.append("1" if ok else "0")
            rows.append("".join(row))
        edges = []
        for c in spec["c"]:
            e = c.rsplit(" ", 2)[0]
            edges += _join(_marching(_fn(_expr(e), (X, Y)), xmin, xmax, ymin, ymax, nx, ny), dx / 100)
        return {"type": "region", "label": label, "nx": nx, "ny": ny, "mask": "".join(rows), "paths": edges}
    if t == "ode":
        f = _fn(_expr(spec["e"]), (X, Y))
        cols = 22
        rows_n = max(8, int(cols * h / max(w, 1)))
        dx, dy = (xmax - xmin) / cols, (ymax - ymin) / rows_n
        seg = min(dx, dy) * 0.35
        field = []
        for j in range(rows_n):
            for i in range(cols):
                xv, yv = xmin + (i + 0.5) * dx, ymin + (j + 0.5) * dy
                k = f(xv, yv)
                if k is None:
                    continue
                # equal length on screen: scale the slope by the aspect of the view
                sx, sy = 1.0 / dx, k / dy
                norm = math.hypot(sx, sy)
                ux, uy = sx / norm * dx * 0.35, sy / norm * dy * 0.35
                field.append([xv - ux, yv - uy, xv + ux, yv + uy])
        layers = {"type": "lines", "label": label, "paths": field, "thin": True}
        if "ic" in spec:
            x0, y0 = spec["ic"]
            sol = []
            for direction in (1, -1):
                pts = [(x0, y0)]
                xv, yv = x0, y0
                hstep = direction * (xmax - xmin) / 600
                while xmin - abs(hstep) <= xv <= xmax + abs(hstep) and ymin - 10 * yspan < yv < ymax + 10 * yspan:
                    k1 = f(xv, yv)
                    k2 = f(xv + hstep / 2, yv + hstep / 2 * k1) if k1 is not None else None
                    k3 = f(xv + hstep / 2, yv + hstep / 2 * k2) if k2 is not None else None
                    k4 = f(xv + hstep, yv + hstep * k3) if k3 is not None else None
                    if k4 is None:
                        break
                    yv += hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    xv += hstep
                    pts.append((xv, yv))
                sol = (list(reversed(pts)) + sol[1:]) if direction == -1 else pts
            layers["solution"] = [v for p in sol for v in (round(p[0], 6), round(p[1], 6))]
            layers["ic"] = [x0, y0]
        return layers
    raise ce.CalcError("этот график рисуется в 3D")


def scene(specs_json, xmin, xmax, ymin, ymax, w, h):
    def run():
        specs = [json.loads(s) if s.startswith("{") else {"t": "y", "e": s} for s in json.loads(specs_json)]
        layers, curves = [], []
        for spec in specs:
            try:
                layers.append(_layer(spec, xmin, xmax, ymin, ymax, w, h))
            except ce.CalcError as ex:
                layers.append({"type": "lines", "label": str(ex), "paths": []})
            except Exception:  # noqa: BLE001 - one bad graph must not hide the others
                layers.append({"type": "lines", "label": _label(spec) + " — не удалось построить", "paths": []})
            if spec["t"] == "y":
                try:
                    e = _expr(spec["e"])
                    curves.append((_fn(e, (X,)), e))
                except Exception:  # noqa: BLE001
                    pass
        points = []
        try:
            points = ce.run_limited(lambda: {"p": _special_points(curves, xmin, xmax, ymin, ymax)}, 3)
            points = points.get("p", []) if isinstance(points, dict) else []
        except Exception:  # noqa: BLE001
            points = []
        return {"layers": layers, "points": points}

    return json.dumps(ce._safe(run), ensure_ascii=False)


def initial_view(specs_json, w, h):
    """[xmin, xmax, ymin, ymax] that shows the graphs well, with equal units on both axes when shapes matter."""
    specs = [json.loads(s) if s.startswith("{") else {"t": "y", "e": s} for s in json.loads(specs_json)]
    aspect = h / max(w, 1)
    kinds = {sp_["t"] for sp_ in specs}

    def robust(vals, cut=0.03):
        vals = sorted(vals)
        return vals[int(len(vals) * cut)], vals[int((len(vals) - 1) * (1 - cut))]

    if kinds & {"param", "polar"}:
        xs, ys = [], []
        for spec in specs:
            if spec["t"] in ("param", "polar"):
                try:
                    for p in _layer(spec, -1e9, 1e9, -1e9, 1e9, 400, 400)["paths"]:
                        xs += p[0::2]
                        ys += p[1::2]
                except Exception:  # noqa: BLE001
                    pass
        if xs:
            (x0, x1), (y0, y1) = robust(xs, 0.002), robust(ys, 0.002)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max((x1 - x0) / 2, (y1 - y0) / 2 / aspect, 0.5) * 1.25
            return json.dumps([cx - half, cx + half, cy - half * aspect, cy + half * aspect])
    if kinds <= {"y", "ode"} and "y" in kinds:
        ys = []
        for spec in specs:
            if spec["t"] == "y":
                try:
                    f = _fn(_expr(spec["e"]), (X,))
                    ys += [v for v in (f(-10 + 20 * k / 300) for k in range(301)) if v is not None]
                except Exception:  # noqa: BLE001
                    pass
        if ys:
            # equal units on both axes (as on paper) when the graph is mostly visible that way
            if sum(1 for v in ys if abs(v) <= 10 * aspect) >= 0.2 * len(ys):
                return json.dumps([-10, 10, -10 * aspect, 10 * aspect])
            lo, hi = robust(ys)
            if hi - lo < 1e-9:
                lo, hi = lo - 1, hi + 1
            pad = (hi - lo) * 0.15
            lo, hi = lo - pad, hi + pad
            return json.dumps([-10, 10, lo, hi])
    half = 5.0
    if any("ic" in sp_ for sp_ in specs):
        x0, y0 = next(sp_["ic"] for sp_ in specs if "ic" in sp_)
        return json.dumps([x0 - half, x0 + half, y0 - half * aspect, y0 + half * aspect])
    return json.dumps([-half, half, -half * aspect, half * aspect])


def surface(spec_json, cx, cy, r, n):
    def run():
        spec = json.loads(spec_json)
        f = _fn(_expr(spec["e"]), (X, Y))
        z = []
        vals = []
        for j in range(n + 1):
            yv = cy - r + 2 * r * j / n
            row = []
            for i in range(n + 1):
                xv = cx - r + 2 * r * i / n
                v = f(xv, yv)
                row.append(None if v is None else round(v, 6))
                if v is not None:
                    vals.append(v)
            z.append(row)
        if not vals:
            raise ce.CalcError("функция не определена в этой области")
        vals.sort()
        zmin, zmax = vals[int(len(vals) * 0.02)], vals[int((len(vals) - 1) * 0.98)]
        if zmax - zmin < 1e-9:
            zmin, zmax = zmin - 1, zmax + 1
        return {"z": z, "zmin": zmin, "zmax": zmax, "label": _label(spec)}

    return json.dumps(ce._safe(run), ensure_ascii=False)
