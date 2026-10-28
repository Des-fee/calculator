"""Higher mathematics for calc_engine (registered into its LOCALS at the end of calc_engine).

Vector calculus, multiple integrals, integral transforms, recurrences, systems of ODEs,
optimisation of several variables, probability distributions, regression, logic and sets, residues.
"""
import itertools
import re

import sympy as sp

import calc_engine as ce

X, W, O = ce.X, ce.W, ce.O
x, y, z, t, s, n, k = sp.symbols("x y z t s n k")


def _vars(expr, given=None):
    if given:
        return list(given)
    syms = set()
    for e in (expr if isinstance(expr, (list, tuple, sp.MatrixBase)) else [expr]):
        syms |= sp.sympify(e).free_symbols
    return ce.order_vars(syms) or [x]


def _vec(v):
    if isinstance(v, sp.MatrixBase):
        return list(v)
    return [sp.sympify(a) for a in v]


def _row(items):
    return sp.Matrix([list(items)])


# ---------------------------------------------------------------- vector calculus

def grad(f, *vs):
    f = sp.sympify(f)
    vs = _vars(f, vs or (None if len(f.free_symbols) > 1 else [x, y, z][:max(1, len(f.free_symbols))]))
    return _row(sp.simplify(sp.diff(f, v)) for v in vs)


def div(F, *vs):
    F = _vec(F)
    vs = list(vs) or [x, y, z][:len(F)]
    return sp.simplify(sum(sp.diff(c, v) for c, v in zip(F, vs)))


def curl(F, *vs):
    F = _vec(F)
    if len(F) == 2:
        F = F + [sp.S.Zero]
    if len(F) != 3:
        raise ce.CalcError("ротор считается для поля из 3 компонент")
    a, b, c = list(vs) or [x, y, z]
    P, Q, R = F
    return _row(sp.simplify(e) for e in (sp.diff(R, b) - sp.diff(Q, c), sp.diff(P, c) - sp.diff(R, a),
                                         sp.diff(Q, a) - sp.diff(P, b)))


def jacobian(F, *vs):
    F = _vec(F)
    vs = list(vs) or _vars(F)
    return sp.Matrix(F).jacobian(vs).applyfunc(sp.simplify)


def hessian(f, *vs):
    f = sp.sympify(f)
    vs = list(vs) or _vars(f)
    return sp.hessian(f, vs).applyfunc(sp.simplify)


def laplacian(f, *vs):
    f = sp.sympify(f)
    vs = list(vs) or _vars(f)
    return sp.simplify(sum(sp.diff(f, v, 2) for v in vs))


# ---------------------------------------------------------------- multiple integrals

def int2(f, a, a0, a1, b, b0, b1):
    """∬ f: the outer variable with constant limits first, the inner one may depend on it."""
    return sp.integrate(f, (b, b0, b1), (a, a0, a1))


def int3(f, a, a0, a1, b, b0, b1, c, c0, c1):
    return sp.integrate(f, (c, c0, c1), (b, b0, b1), (a, a0, a1))


# ---------------------------------------------------------------- integral transforms

def _var_of(f, default):
    syms = sp.sympify(f).free_symbols
    return default if default in syms or not syms else ce.order_vars(syms)[0]


def laplace(f, var=None, to=None):
    var = var or _var_of(f, t)
    to = to or s
    tp = sp.Symbol(var.name, positive=True)
    res = sp.laplace_transform(sp.sympify(f).subs(var, tp), tp, to, noconds=True)
    if isinstance(res, sp.LaplaceTransform):
        raise ce.CalcError("преобразование Лапласа не находится в формулах")
    return sp.simplify(res)


def ilaplace(F, var=None, to=None):
    var = var or _var_of(F, s)
    to = to or t
    tp = sp.Symbol(to.name, positive=True)
    res = sp.inverse_laplace_transform(sp.sympify(F), var, tp)
    if isinstance(res, sp.InverseLaplaceTransform):
        raise ce.CalcError("обратное преобразование не находится в формулах")
    return sp.simplify(res.subs(sp.Heaviside(tp), 1)).subs(tp, to)


def fourier(f, var=None, to=None):
    var = var or _var_of(f, x)
    to = to or k
    res = sp.fourier_transform(sp.sympify(f), var, to)
    if isinstance(res, sp.FourierTransform):
        raise ce.CalcError("преобразование Фурье не находится в формулах")
    return sp.simplify(res)


def ifourier(F, var=None, to=None):
    var = var or _var_of(F, k)
    to = to or x
    res = sp.inverse_fourier_transform(sp.sympify(F), var, to)
    if isinstance(res, sp.InverseFourierTransform):
        raise ce.CalcError("обратное преобразование не находится в формулах")
    return sp.simplify(res)


def ztrans(f, var=None, to=None):
    var = var or _var_of(f, n)
    to = to or z
    res = sp.summation(sp.sympify(f) * to ** (-var), (var, 0, sp.oo))
    if isinstance(res, sp.Piecewise):
        res = res.args[0][0]
    if res.has(sp.Sum):
        raise ce.CalcError("Z-преобразование не находится в формулах")
    return sp.simplify(res)


def iztrans(F, var=None, to=None):
    """Inverse Z-transform of a rational function: partial fractions in z⁻¹ (simple poles)."""
    var = var or _var_of(F, z)
    to = to or n
    F = sp.together(sp.sympify(F))
    G = sp.apart(sp.simplify(F / var), var)  # F(z)/z = Σ A/(z − p)  ->  F = Σ A·z/(z − p)  ->  Σ A·pⁿ
    out = sp.S.Zero
    for term in sp.Add.make_args(G):
        num, den = sp.fraction(sp.together(term))
        poly = sp.Poly(den, var)
        if poly.degree() == 0:
            raise ce.CalcError("обратное Z-преобразование: только дробно-рациональные функции")
        roots = sp.roots(poly)
        if poly.degree() == 1:
            p = list(roots)[0]
            out += sp.simplify(num / poly.LC()) * p ** to
        elif poly.degree() == 2 and len(roots) == 1:  # A/(z − p)²  ->  A·n·pⁿ⁻¹
            p = list(roots)[0]
            out += sp.simplify(num / poly.LC()) * to * p ** (to - 1)
        else:
            raise ce.CalcError("обратное Z-преобразование: только простые полюсы")
    return sp.simplify(out)


# ---------------------------------------------------------------- recurrences: a(n+1) = 2a(n) + 1; a(0) = 1

REC_TERM = re.compile(r"\b([a-zA-Z])\s*\(\s*n\s*([+-]\s*\d+)?\s*\)")
REC_INIT = re.compile(r"^\s*([a-zA-Z])\s*\(\s*(-?\d+)\s*\)\s*=\s*(.+?)\s*$")


def process_recurrence(texts):
    """None when the statements are not a recurrence."""
    rec = [tx for tx in texts if any(m.group(2) for m in REC_TERM.finditer(tx))]
    inits = [tx for tx in texts if REC_INIT.match(tx)]
    if len(rec) != 1 or len(rec) + len(inits) != len(texts):
        return None
    name = REC_TERM.search(rec[0]).group(1)
    a = sp.Function(name)
    loc = dict(ce.LOCALS, **{name: a, "n": n})
    pieces, ops = ce.split_relation(rec[0])
    if ops != ["="]:
        return None
    try:
        lhs = ce.parse_expr(ce.normalize_text(pieces[0]), local_dict=loc, transformations=ce.TRANSFORMS)
        rhs = ce.parse_expr(ce.normalize_text(pieces[1]), local_dict=loc, transformations=ce.TRANSFORMS)
    except Exception as ex:  # noqa: BLE001
        raise ce.CalcError("не удалось разобрать рекуррентное соотношение") from ex
    ics = {}
    for tx in inits:
        m = REC_INIT.match(tx)
        if m.group(1) != name:
            return None
        ics[a(int(m.group(2)))] = ce.parse(m.group(3))
    try:
        sol = sp.rsolve(lhs - rhs, a(n), ics or None)
    except (ValueError, NotImplementedError) as ex:
        raise ce.CalcError("это рекуррентное соотношение не решается в формулах") from ex
    if sol is None:
        raise ce.CalcError("это рекуррентное соотношение не решается в формулах")
    sol = sp.simplify(sol)
    line = [("%s(n)" % name, r"%s_{n}" % name), O("="), X(sol)]
    return [line]


# ---------------------------------------------------------------- systems of ODEs in t: x' = y; y' = −x

ODE_LHS = re.compile(r"^\s*([a-zA-Z])\s*('+|′+)\s*$")


def process_ode_system(texts):
    eqs_text = []
    ics_text = []
    for tx in texts:
        pieces, ops = ce.split_relation(tx)
        if ops != ["="]:
            return None
        if ODE_LHS.match(pieces[0]):
            eqs_text.append((ODE_LHS.match(pieces[0]), pieces[1]))
        elif re.match(r"^\s*[a-zA-Z]\s*\(\s*[^()]+\)\s*$", pieces[0]):
            ics_text.append((pieces[0], pieces[1]))
        else:
            return None
    names = [m.group(1) for m, _ in eqs_text]
    if len(eqs_text) < 2 or len(set(names)) != len(names) or "t" in names:
        return None
    funcs = {nm: sp.Function(nm)(t) for nm in names}
    loc = dict(ce.LOCALS, **funcs, t=t)
    eqs = []
    try:
        for m, rhs in eqs_text:
            order = len(m.group(2))
            r = ce.parse_expr(ce.normalize_text(rhs), local_dict=loc, transformations=ce.TRANSFORMS)
            eqs.append(sp.Eq(sp.diff(funcs[m.group(1)], t, order), r))
    except Exception as ex:  # noqa: BLE001
        raise ce.CalcError("не удалось разобрать систему ОДУ") from ex
    ics = {}
    for lhs, rhs in ics_text:
        m = re.match(r"^\s*([a-zA-Z])\s*\(\s*([^()]+)\)\s*$", lhs)
        if m.group(1) not in funcs:
            return None
        ics[sp.Function(m.group(1))(ce.parse(m.group(2)))] = ce.parse(rhs)
    try:
        sols = sp.dsolve(eqs, list(funcs.values()), ics=ics or None)
    except (NotImplementedError, ValueError) as ex:
        raise ce.CalcError("эта система ОДУ не решается в формулах") from ex
    if not isinstance(sols, list):
        sols = [sols]
    if sols and isinstance(sols[0], list):
        sols = sols[0]
    return [[(str(sol.lhs.func), str(sol.lhs.func)), O("="), X(sp.simplify(sol.rhs))] for sol in sols]


# ---------------------------------------------------------------- optimisation of several variables

def _classify(H):
    try:
        ev = [sp.N(e) for e in H.eigenvals()]
    except Exception:  # noqa: BLE001
        return "?"
    if all(e > 0 for e in ev):
        return "минимум"
    if all(e < 0 for e in ev):
        return "максимум"
    if any(e > 0 for e in ev) and any(e < 0 for e in ev):
        return "седло"
    return "?"


def _point(vs, sol):
    return "(" + "; ".join(ce.fmt(sol[v]) for v in vs) + ")", r"\left(" + r";\ ".join(ce.to_tex(sol[v]) for v in vs) + r"\right)"


def extrema_n(f, *vs):
    f = sp.sympify(f)
    vs = list(vs) or _vars(f)
    if len(vs) == 1:
        return ce._extrema(f, vs[0])
    sols = sp.solve([sp.diff(f, v) for v in vs], vs, dict=True)
    sols = [sol for sol in sols if all(sol.get(v, v).is_real is not False and not sp.sympify(sol.get(v, v)).has(sp.I) for v in vs)]
    sols = [sol for sol in sols if all(v in sol for v in vs)]
    if not sols:
        return ce.Answer([W("критических точек нет")])
    H = sp.hessian(f, vs)
    lines = []
    for sol in sols[:8]:
        kind = _classify(H.subs(sol))
        lines.append([_point(vs, sol), (": " + kind + " ", r":\ \text{%s}\ " % kind), ("f", "f"), O("="),
                      X(sp.simplify(f.subs(sol)))])
    return ce.MultiAnswer(lines)


def lagrange(f, *constraints):
    """Extrema of f under constraints g = 0 (Lagrange multipliers)."""
    f = sp.sympify(f)
    gs = []
    for g in constraints:
        g = sp.sympify(g)
        gs.append(g.lhs - g.rhs if isinstance(g, sp.Equality) else g)
    vs = _vars([f] + gs)
    lams = sp.symbols("lambda1:%d" % (len(gs) + 1))
    L = f - sum(lm * g for lm, g in zip(lams, gs))
    eqs = [sp.diff(L, v) for v in vs] + gs
    sols = sp.solve(eqs, list(vs) + list(lams), dict=True)
    sols = [sol for sol in sols if all(v in sol and not sp.sympify(sol[v]).has(sp.I) for v in vs)]
    if not sols:
        return ce.Answer([W("условных экстремумов не найдено")])
    vals = [sp.simplify(f.subs(sol)) for sol in sols]
    numeric = [float(sp.N(v)) if v.is_number else None for v in vals]
    hi = max((v for v in numeric if v is not None), default=None)
    lo = min((v for v in numeric if v is not None), default=None)
    lines = []
    for sol, val, num in list(zip(sols, vals, numeric))[:8]:
        kind = "максимум" if num is not None and num == hi and hi != lo else "минимум" if num is not None and num == lo and hi != lo else "экстремум"
        lines.append([_point(vs, sol), (": " + kind + " ", r":\ \text{%s}\ " % kind), ("f", "f"), O("="), X(val)])
    return ce.MultiAnswer(lines)


# ---------------------------------------------------------------- probability

def _phi(v):
    return (1 + sp.erf(v / sp.sqrt(2))) / 2


def normpdf(v, mu=0, sigma=1):
    return sp.exp(-(sp.sympify(v) - mu) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))


def normcdf(a, b=None, mu=0, sigma=1):
    """normcdf(x) = P(X < x); normcdf(a, b, μ, σ) = P(a < X < b)."""
    if b is None:
        return _phi((sp.sympify(a) - mu) / sigma)
    return sp.simplify(_phi((sp.sympify(b) - mu) / sigma) - _phi((sp.sympify(a) - mu) / sigma))


def invnorm(p, mu=0, sigma=1):
    import mpmath
    q = float(p)
    if not 0 < q < 1:
        raise ce.CalcError("вероятность должна быть между 0 и 1")
    return sp.Float(mu + sigma * mpmath.sqrt(2) * mpmath.erfinv(2 * q - 1), 12)


def binompdf(nn, p, kk):
    return sp.binomial(nn, kk) * sp.sympify(p) ** kk * (1 - sp.sympify(p)) ** (nn - kk)


def binomcdf(nn, p, kk):
    return sp.simplify(sum(binompdf(nn, p, j) for j in range(0, int(kk) + 1)))


def poissonpdf(lam, kk):
    return sp.sympify(lam) ** kk * sp.exp(-sp.sympify(lam)) / sp.factorial(kk)


def poissoncdf(lam, kk):
    return sp.simplify(sum(poissonpdf(lam, j) for j in range(0, int(kk) + 1)))


def expcdf(v, lam=1):
    return 1 - sp.exp(-sp.sympify(lam) * v)


# ---------------------------------------------------------------- regression

def _pairs(xs, ys):
    xs, ys = [sp.nsimplify(v) for v in _vec(xs)], [sp.nsimplify(v) for v in _vec(ys)]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ce.CalcError("нужны два списка одной длины: значения x и значения y")
    return xs, ys


def polyfit(xs, ys, deg=1):
    xs, ys = _pairs(xs, ys)
    deg = int(deg)
    A = sp.Matrix([[xv ** j for j in range(deg + 1)] for xv in xs])
    coef = (A.T * A).solve(A.T * sp.Matrix(ys))
    return sp.expand(sum(c * x ** j for j, c in enumerate(coef)))


def linreg(xs, ys):
    xs, ys = _pairs(xs, ys)
    line = polyfit(xs, ys, 1)
    nn = len(xs)
    mx, my = sum(xs) / nn, sum(ys) / nn
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    r = sxy / sp.sqrt(sxx * syy) if sxx and syy else sp.S.One
    return ce.MultiAnswer([[("y", "y"), O("=")] + ce.value_line(line, False),
                           [("r", "r"), O("=")] + ce.value_line(sp.nsimplify(r), False)])


def expreg(xs, ys):
    """y = a·e^(bx) by a straight line through (x, ln y)."""
    xs, ys = _pairs(xs, ys)
    if any(v <= 0 for v in ys):
        raise ce.CalcError("для показательной регрессии все y должны быть положительными")
    import math
    lx = [float(v) for v in xs]
    ly = [math.log(float(v)) for v in ys]
    m = len(lx)
    mx, my = sum(lx) / m, sum(ly) / m
    sxx = sum((u - mx) ** 2 for u in lx)
    if sxx == 0:
        raise ce.CalcError("все x одинаковые")
    b = sum((u - mx) * (v - my) for u, v in zip(lx, ly)) / sxx
    a = my - b * mx
    return sp.Float(math.exp(a), 8) * sp.exp(sp.Float(b, 8) * x)


# ---------------------------------------------------------------- logic and sets

def truth(expr):
    """Truth table of a boolean expression: x & y, x | ~y, implies(x, y), xor(x, y)."""
    e = sp.sympify(expr)
    vs = ce.order_vars(e.free_symbols)
    if not vs or len(vs) > 5:
        raise ce.CalcError("таблица истинности — для 1–5 переменных")
    head = "  ".join(str(v) for v in vs) + "  │  f"
    lines = [[(head, r"\text{%s}" % head.replace("│", "|"))]]
    for vals in itertools.product([0, 1], repeat=len(vs)):
        val = e.subs(dict(zip(vs, [bool(v) for v in vals])))
        row = "  ".join(str(v) for v in vals) + "  │  " + ("1" if val == sp.true else "0")
        lines.append([(row, r"\text{%s}" % row.replace("│", "|"))])
    return ce.MultiAnswer(lines)


def simplify_bool(expr):
    return sp.simplify_logic(sp.sympify(expr))


def _set(a):
    if isinstance(a, (sp.Set,)):
        return a
    if isinstance(a, (list, tuple, set, frozenset)):
        return sp.FiniteSet(*a)
    if isinstance(a, sp.MatrixBase):
        return sp.FiniteSet(*list(a))
    raise ce.CalcError("множество пишется списком [1, 2, 3] или промежутком interval(0, 1)")


def union(*a):
    return sp.Union(*[_set(v) for v in a])


def intersect(*a):
    return sp.Intersection(*[_set(v) for v in a])


def setminus(a, b):
    return sp.Complement(_set(a), _set(b))


def interval(a, b, left_open=False, right_open=False):
    return sp.Interval(a, b, bool(left_open), bool(right_open))


# ---------------------------------------------------------------- complex analysis

def residue(f, var=None, at=0):
    if var is not None and not isinstance(var, sp.Symbol):
        var, at = None, var
    var = var or _var_of(f, z)
    return sp.simplify(sp.residue(sp.sympify(f), var, at))


def laurent(f, var=None, at=0, order=4):
    if var is not None and not isinstance(var, sp.Symbol):
        var, at = None, var
    var = var or _var_of(f, z)
    return sp.series(sp.sympify(f), var, at, int(order)).removeO()


def register():
    return {
        "grad": grad, "div": div, "curl": curl, "rot": curl, "jacobian": jacobian, "hessian": hessian,
        "laplacian": laplacian,
        "int2": int2, "int3": int3,
        "laplace": laplace, "ilaplace": ilaplace, "fourier": fourier, "ifourier": ifourier,
        "ztrans": ztrans, "iztrans": iztrans,
        "extrema": extrema_n, "extremum": extrema_n, "lagrange": lagrange,
        "normpdf": normpdf, "normcdf": normcdf, "invnorm": invnorm, "binompdf": binompdf, "binomcdf": binomcdf,
        "poissonpdf": poissonpdf, "poissoncdf": poissoncdf, "expcdf": expcdf,
        "linreg": linreg, "polyfit": polyfit, "expreg": expreg,
        "truth": truth, "simplify_bool": simplify_bool, "simplify_logic": simplify_bool,
        "xor": sp.Xor, "implies": sp.Implies,
        "union": union, "intersect": intersect, "setminus": setminus, "interval": interval,
        "residue": residue, "laurent": laurent,
    }
