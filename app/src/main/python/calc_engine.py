"""SymPy engine for the handwriting calculator.

Two inputs reach this module:
  * token sequences from the handwriting recogniser (CROHME LaTeX tokens,
    e.g. ``\\frac { 1 } { 2 } + x ^ { 2 } =``), converted to linear text by
    :func:`tokens_to_text`;
  * linear text typed on the keyboard (``2x^2 + sin x =``).

Both end up as linear text that is parsed by SymPy's ``parse_expr`` and then
evaluated, solved or checked by :func:`process`.
The Android side talks to :func:`evaluate_document` and :func:`plot_points`
using JSON strings.
"""

import ctypes
import json
import sys
import math
import re
import threading
from collections import OrderedDict

import mpmath

import sympy as sp
from sympy.core.relational import Relational
from sympy.printing.latex import LatexPrinter
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


class CalcError(Exception):
    """Error with a message that can be shown to the user."""


# ---------------------------------------------------------------------------
# Handwriting tokens -> linear text
# ---------------------------------------------------------------------------

FUNC_TOKENS = {
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\log": "log",
}
INVERSE = {"sin": "asin", "cos": "acos", "tan": "atan", "cot": "acot"}
# letter sequences written by hand that mean a function
LETTER_FUNCS = [
    (["a", "r", "c", "s", "i", "n"], "asin"),
    (["a", "r", "c", "c", "o", "s"], "acos"),
    (["a", "r", "c", "t", "g"], "atan"),
    (["c", "t", "g"], "cot"),
    (["c", "o", "t"], "cot"),
    (["e", "x", "p"], "exp"),
    (["l", "n"], "log"),
    (["l", "g"], "lg"),
    (["t", "g"], "tan"),
    # functions read letter by letter (the recogniser often does this instead of \\sin)
    (["s", "i", "n"], "sin"),
    (["c", "o", "s"], "cos"),
    (["t", "a", "n"], "tan"),
    (["l", "o", "g"], "log"),
]
GREEK = {
    "\\alpha": "α",
    "\\beta": "β",
    "\\gamma": "γ",
    "\\theta": "θ",
    "\\lambda": "λ",
    "\\mu": "μ",
    "\\sigma": "σ",
    "\\phi": "φ",
    "\\Delta": "Δ",
    "\\Pi": "Π",
}
RELOPS = {"=": "=", "<": "<", ">": ">", "\\leq": "<=", "\\geq": ">=", "\\neq": "!="}
BINOPS = {"+": "+", "-": "-", "\\times": "*", "\\cdot": "*", "\\div": "/", "/": "/"}
UNSUPPORTED = {"\\exists", "\\forall", "\\in", "\\ldots", "\\cdots", "\\pm", "\\rightarrow"}
OPEN = {"(": ")", "[": "]", "{": "}", "\\{": "\\}"}


class TokenConverter:
    def __init__(self, tokens):
        self.t = [tok for tok in tokens if tok not in ("\\limits", "\\prime")]
        self.i = 0

    # -- helpers -----------------------------------------------------------
    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def group_tokens(self):
        """Return the tokens of the next group: ``{ ... }`` or a single item."""
        tok = self.peek()
        if tok is None:
            raise CalcError("неполная запись")
        if tok in OPEN:
            close = OPEN[tok]
            self.take()
            depth = 1
            start = self.i
            while self.i < len(self.t):
                cur = self.t[self.i]
                if cur in OPEN:
                    depth += 1
                elif cur in OPEN.values():
                    depth -= 1
                    if depth == 0:
                        inner = self.t[start:self.i]
                        self.i += 1
                        if tok == "(" or tok == "[":
                            return ["("] + inner + [")"]
                        return inner
                self.i += 1
            raise CalcError("не закрыта скобка")
        # single item (with its own scripts, e.g. x^2 written as ^ x)
        start = self.i
        self.skip_item()
        return self.t[start:self.i]

    def skip_item(self):
        """Advance over one primary item (used for un-braced arguments)."""
        tok = self.take()
        if tok is None:
            raise CalcError("неполная запись")
        if tok in OPEN:
            self.i -= 1
            self.group_tokens()
            return
        if tok.isdigit() or tok == ".":
            while self.peek() is not None and (self.peek().isdigit() or self.peek() == "."):
                self.take()
        if tok in ("\\frac",):
            self.group_tokens()
            self.group_tokens()
        elif tok == "\\sqrt":
            if self.peek() == "[":
                self.group_tokens()
            self.group_tokens()

    @staticmethod
    def sub(tokens):
        return TokenConverter(tokens).convert()

    def operand_end(self, stop_at_func=True):
        """Index where an un-parenthesised operand (function argument, sum body) ends."""
        j = self.i
        depth = 0
        while j < len(self.t):
            tok = self.t[j]
            if tok in OPEN:
                depth += 1
            elif tok in OPEN.values():
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if tok in ("+", "-", ",", "|", "\\pm") or tok in RELOPS:
                    if j > self.i:
                        break
                if stop_at_func and j > self.i and (tok in FUNC_TOKENS or tok in ("\\int", "\\sum", "\\lim")):
                    break
            j += 1
        return j

    def operand(self, stop_at_func=True):
        if self.peek() == "(":
            toks = self.group_tokens()
            # keep postfix powers and factorials: (1 + 1/n)^n
            while self.peek() in ("^", "!"):
                if self.take() == "^":
                    toks = toks + ["^", "{"] + self.group_tokens() + ["}"]
                else:
                    toks = toks + ["!"]
            return self.sub(toks)
        end = self.operand_end(stop_at_func)
        toks = self.t[self.i:end]
        self.i = end
        if not toks:
            raise CalcError("нет аргумента функции")
        return self.sub(toks)

    # -- main conversion ---------------------------------------------------
    def convert(self):
        out = []
        abs_open = 0
        prev_kind = "start"  # start | op | value

        def emit(text, kind):
            nonlocal prev_kind
            out.append(text)
            prev_kind = kind

        while self.peek() is not None:
            tok = self.take()
            if tok == "\\bmod":
                emit(" mod ", "op")
                continue

            if tok.isdigit() or tok == ".":
                num = tok
                while self.peek() is not None and (self.peek().isdigit() or self.peek() == "."):
                    num += self.take()
                if num.count(".") > 1:
                    raise CalcError("неверное число")
                if num.startswith("."):
                    num = "0" + num
                if prev_kind == "value":
                    out.append(" ")
                emit(num, "value")
            elif len(tok) == 1 and tok.isalpha():
                fn = self.letter_function(tok)
                if fn:
                    emit(self.function(fn), "value")
                else:
                    name = tok
                    if self.peek() == "_":
                        self.take()
                        idx = self.group_tokens()
                        if all(x.isalnum() and len(x) == 1 for x in idx):
                            name = tok + "_" + "".join(idx)
                    emit(" " + name, "value")
            elif tok in GREEK:
                emit(" " + GREEK[tok], "value")
            elif tok == "\\pi":
                emit(" pi", "value")
            elif tok == "\\infty":
                emit(" oo", "value")
            elif tok in FUNC_TOKENS:
                emit(self.function(FUNC_TOKENS[tok]), "value")
            elif tok == "\\frac":
                emit(self.frac(), "value")
            elif tok == "\\sqrt":
                if self.peek() == "[":
                    self.take()
                    j = self.i
                    while j < len(self.t) and self.t[j] != "]":
                        j += 1
                    index = self.sub(self.t[self.i:j])
                    self.i = j + 1
                    emit(" root(%s, %s)" % (self.sub(self.group_tokens()), index), "value")
                else:
                    emit(" sqrt(%s)" % self.sub(self.group_tokens()), "value")
            elif tok == "^":
                emit("^(%s)" % self.sub(self.group_tokens()), "value")
            elif tok == "_":
                # a subscript on something that is not a letter: ignore it
                self.group_tokens()
            elif tok == "\\int":
                emit(self.integral(), "value")
            elif tok == "\\sum":
                emit(self.summation(), "value")
            elif tok == "\\lim":
                emit(self.limit(), "value")
            elif tok == "|":
                if abs_open and prev_kind == "value":
                    abs_open -= 1
                    emit(")", "value")
                else:
                    abs_open += 1
                    emit(" Abs(", "op")
            elif tok in ("(", "["):
                emit(" (", "op")
            elif tok in (")", "]"):
                emit(")", "value")
            elif tok == "{":
                emit(" (", "op")
            elif tok == "}":
                emit(")", "value")
            elif tok in ("\\{", "\\}"):
                continue  # system brace that slipped into a line
            elif tok == "!":
                emit("!", "value")
            elif tok in BINOPS:
                emit(" %s " % BINOPS[tok], "op")
            elif tok in RELOPS:
                emit(" %s " % RELOPS[tok], "op")
            elif tok == ",":
                emit(", ", "op")
            elif tok in UNSUPPORTED:
                raise CalcError("символ %s не поддерживается" % tok.lstrip("\\"))
            else:
                raise CalcError("неизвестный символ %s" % tok)
        if abs_open:
            out.append(")" * abs_open)
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def letter_function(self, first):
        for letters, name in LETTER_FUNCS:
            n = len(letters)
            if [first] + self.t[self.i:self.i + n - 1] == letters:
                nxt = self.peek(n - 1)
                # only a function when an argument follows
                if nxt is not None and nxt not in RELOPS and nxt not in BINOPS and nxt not in (")", "}", ","):
                    self.i += n - 1
                    return name
        return None

    def function(self, name):
        power = None
        base = None
        if self.peek() == "^":
            self.take()
            power_toks = self.group_tokens()
            if power_toks == ["-", "1"] and name in INVERSE:
                name = INVERSE[name]
            else:
                power = self.sub(power_toks)
        if self.peek() == "_" and name == "log":
            self.take()
            base = self.sub(self.group_tokens())
        if self.peek() == "^" and power is None:
            self.take()
            power = self.sub(self.group_tokens())
        arg = self.operand()
        if base is not None:
            text = " log(%s, %s)" % (arg, base)
        elif name == "lg":
            text = " log(%s, 10)" % arg
        else:
            text = " %s(%s)" % (name, arg)
        if power is not None:
            text += "^(%s)" % power
        return text

    def frac(self):
        num = self.group_tokens()
        den = self.group_tokens()
        # derivative d/dx f
        if num == ["d"] and len(den) == 2 and den[0] == "d" and den[1].isalpha():
            return " diff(%s, %s)" % (self.operand(stop_at_func=False), den[1])
        if (
            len(num) == 4 and num[:2] == ["d", "^"] and len(den) == 4
            and den[0] == "d" and den[1].isalpha() and den[2] == "^"
        ):
            return " diff(%s, %s, %s)" % (self.operand(stop_at_func=False), den[1], num[3])
        return " ((%s)/(%s))" % (self.sub(num), self.sub(den))

    def limits(self):
        lo = hi = None
        for _ in range(2):
            if self.peek() == "_":
                self.take()
                lo = self.group_tokens()
            elif self.peek() == "^":
                self.take()
                hi = self.group_tokens()
        return lo, hi

    def integral(self):
        lo, hi = self.limits()
        # find the differential "d x" at depth 0
        j = self.i
        depth = 0
        found = None
        while j < len(self.t):
            tok = self.t[j]
            if tok in OPEN:
                depth += 1
            elif tok in OPEN.values():
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if tok in RELOPS or tok == ",":
                    break
                if tok == "d" and j + 1 < len(self.t) and len(self.t[j + 1]) == 1 and self.t[j + 1].isalpha():
                    found = j
            j += 1
        if found is not None:
            body = self.t[self.i:found]
            var = self.t[found + 1]
            self.i = found + 2
        else:
            body = self.t[self.i:j]
            var = "x"
            self.i = j
        body_text = self.sub(body) if body else "1"
        if lo is not None and hi is not None:
            return " integrate(%s, (%s, %s, %s))" % (body_text, var, self.sub(lo), self.sub(hi))
        return " integrate(%s, %s)" % (body_text, var)

    def summation(self):
        lo, hi = self.limits()
        if lo is None or hi is None:
            raise CalcError("у суммы нужны пределы")
        if "=" in lo:
            k = lo.index("=")
            var, start = lo[:k], lo[k + 1:]
        else:
            var, start = ["n"], ["1"]
        body = self.operand(stop_at_func=False)
        return " summation(%s, (%s, %s, %s))" % (body, self.sub(var), self.sub(start), self.sub(hi))

    def limit(self):
        lo, _ = self.limits()
        if lo is None or "\\rightarrow" not in lo:
            raise CalcError("у предела нужно x → a")
        k = lo.index("\\rightarrow")
        var = self.sub(lo[:k])
        target = lo[k + 1:]
        direction = None
        if len(target) >= 3 and target[-3] == "^" and target[-1] in ("+", "-"):
            direction = target[-1]
            target = target[:-3] + [target[-2]] if target[-2] not in ("+", "-") else target[:-3]
        elif len(target) >= 2 and target[-2] == "^" and target[-1] in ("+", "-"):
            direction = target[-1]
            target = target[:-2]
        body = self.operand(stop_at_func=False)
        if direction:
            return " limit(%s, %s, %s, '%s')" % (body, var, self.sub(target), direction)
        return " limit(%s, %s, %s)" % (body, var, self.sub(target))


_CLOSE = {")": "(", "]": "[", "}": "{"}


def _operand_start(toks, end):
    """Index where the operand ending at toks[end] starts: a bracket group or a single token."""
    t = toks[end]
    if t in _CLOSE:
        depth = 0
        for k in range(end, -1, -1):
            if toks[k] == t:
                depth += 1
            elif toks[k] == _CLOSE[t]:
                depth -= 1
                if depth == 0:
                    # a "{...}" group may belong to a command like \\sqrt or \\frac before it
                    while k > 0 and toks[k - 1] in ("}", "]") and toks[k] == "{":
                        k = _operand_start(toks, k - 1)
                    if k > 0 and toks[k - 1].startswith("\\") and toks[k - 1] not in ("\\times", "\\cdot", "\\div"):
                        k -= 1
                    elif k > 1 and toks[k - 1] in ("^", "_"):  # an exponent: the operand includes its base
                        k = _operand_start(toks, k - 2)
                    return k
        return 0
    return end


def rewrite_primes(toks):
    """(x^3)' -> d/dx (x^3); a double prime is the second derivative."""
    toks = list(toks)
    while "\\prime" in toks:
        i = toks.index("\\prime")
        j = i
        while j + 1 < len(toks) and toks[j + 1] == "\\prime":
            j += 1
        n = j - i + 1
        if i >= 2 and toks[i - 2:i] == ["^", "{"] and j + 1 < len(toks) and toks[j + 1] == "}":
            start_tail, end_tail, base_end = i - 2, j + 2, i - 3  # x ^ { \\prime }
        else:
            start_tail, end_tail, base_end = i, j + 1, i - 1  # x \\prime
        if base_end < 0:
            del toks[i:j + 1]
            continue
        b = _operand_start(toks, base_end)
        operand = toks[b:base_end + 1]
        d = ["\\frac", "{", "d", "}", "{", "d", "x", "}"]
        toks[b:end_tail] = d * n + ["("] + operand + [")"]
    return toks


def rewrite_words(toks):
    """'m o d' -> \\bmod;  dy/dx (a fraction d y over d x) -> d/dx (y)."""
    out = []
    i = 0
    while i < len(toks):
        if toks[i:i + 3] == ["m", "o", "d"]:
            out.append("\\bmod")
            i += 3
            continue
        if toks[i:i + 9][:3] == ["\\frac", "{", "d"] and len(toks) >= i + 9 and toks[i + 4:i + 9] == ["}", "{", "d", "x", "}"] \
                and toks[i + 3].isalpha() and len(toks[i + 3]) == 1:
            out += ["\\frac", "{", "d", "}", "{", "d", "x", "}", "(", toks[i + 3], ")"]
            i += 9
            continue
        out.append(toks[i])
        i += 1
    return out


def tokens_to_text(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    return TokenConverter(rewrite_primes(rewrite_words(list(tokens)))).convert()


# ---------------------------------------------------------------------------
# Linear text -> SymPy
# ---------------------------------------------------------------------------

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


# keyboard forms: int(f, x), int(f, x, a, b); lim(f, x, a); sum(f, n, a, b) — besides SymPy's tuple forms

def _kb_int(fn):
    def f(expr, x=None, a=None, b=None, **kw):
        if x is None:
            fs = sorted(sp.sympify(expr).free_symbols, key=str)
            x = sp.Symbol("x") if sp.Symbol("x") in fs or not fs else fs[0]
        return fn(expr, (x, a, b), **kw) if a is not None and b is not None else fn(expr, x, **kw)
    return f


def _kb_lim(fn):
    def f(expr, x=None, a=None, *rest, **kw):
        if a is None:
            a, x = x, sp.Symbol("x")
        return fn(expr, x, a, *rest, **kw)
    return f


def _kb_sum(fn):
    def f(expr, n, a=None, b=None, **kw):
        return fn(expr, (n, a, b), **kw) if a is not None else fn(expr, n, **kw)
    return f


# ---------------------------------------------------------------------------
# More functions: number theory, combinatorics, bases, statistics, linear algebra, analysis
# ---------------------------------------------------------------------------

class Answer:
    """A ready answer line (text + LaTeX) returned by a function, for results that are not one SymPy value."""

    def __init__(self, line):
        self.line = line


def _ints(*args):
    out = []
    for a in args:
        a = sp.sympify(a)
        if not a.is_Integer:
            raise CalcError("нужно целое число")
        out.append(int(a))
    return out


def _factorint(n):
    (n,) = _ints(n)
    if n in (0, 1, -1):
        return Answer([NUM(n)])
    f = sp.factorint(n)
    sign = "−" if n < 0 else ""
    text = sign + "·".join(("%d%s" % (p, str(e).translate(SUP) if e > 1 else "")) for p, e in sorted(f.items()) if p != -1)
    tex = ("-" if n < 0 else "") + r" \cdot ".join(("%d^{%d}" % (p, e) if e > 1 else str(p)) for p, e in sorted(f.items()) if p != -1)
    return Answer([(text, tex)])


def _isprime(n):
    (n,) = _ints(n)
    if n < 2:
        return Answer([W("не простое")])
    if sp.isprime(n):
        return Answer([W("простое")])
    p = min(sp.factorint(n))
    return Answer([W("составное, делится на %d" % p)])


def _divisors(n):
    (n,) = _ints(n)
    ds = sp.divisors(abs(n))
    shown = ", ".join(str(d) for d in ds[:40]) + (", …" if len(ds) > 40 else "")
    return Answer([(shown + "  (%d шт.)" % len(ds), shown + r"\quad (%d\ \text{шт.})" % len(ds))])


def _arrangements(n, k):
    return sp.ff(n, k)


def _perm(n, k=None):
    return sp.factorial(n) if k is None else sp.ff(n, k)


def _in_base(n, b):
    (n, b) = _ints(n, b)
    if not 2 <= b <= 36:
        raise CalcError("основание от 2 до 36")
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    m, out = abs(n), ""
    while True:
        out = digits[m % b] + out
        m //= b
        if not m:
            break
    out = ("−" if n < 0 else "") + out
    return Answer([(out + str(b).translate(SUB), r"\mathrm{%s}_{%d}" % (out.replace("−", "-"), b))])


def _round(x, n=0):
    from decimal import Decimal, ROUND_HALF_UP
    (n,) = _ints(n)
    d = Decimal(str(sp.N(x, 40))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
    return sp.Integer(int(d)) if n <= 0 else sp.Float(str(d), 40)


def _data(args):
    if len(args) == 1 and isinstance(args[0], (list, tuple, sp.MatrixBase)):
        args = list(args[0])
    vals = [sp.nsimplify(sp.sympify(a), rational=True) if isinstance(sp.sympify(a), sp.Float) else sp.sympify(a) for a in args]
    if not vals:
        raise CalcError("нет данных")
    return vals


def _mean(*a):
    v = _data(a)
    return sp.Add(*v) / len(v)


def _median(*a):
    v = sorted(_data(a))
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _mode(*a):
    v = _data(a)
    counts = {}
    for x in v:
        counts[x] = counts.get(x, 0) + 1
    top = max(counts.values())
    modes = [x for x in v if counts[x] == top]
    return modes[0] if len(set(modes)) == 1 else Answer([X(sorted(set(modes)))])


def _var(*a, sample=False):
    v = _data(a)
    m = sp.Add(*v) / len(v)
    return sp.Add(*[(x - m) ** 2 for x in v]) / (len(v) - (1 if sample else 0))


def _matrix(m):
    return m if isinstance(m, sp.MatrixBase) else sp.Matrix(m)


def _eigenvals(m):
    ev = _matrix(m).eigenvals()
    line = []
    for k, (val, mult) in enumerate(sorted(ev.items(), key=lambda t: sp.default_sort_key(t[0]))):
        if k:
            line.append(("; ", r";\quad "))
        line += [("λ = ", r"\lambda = "), X(val)] + ([(" (×%d)" % mult, r"\ (\times %d)" % mult)] if mult > 1 else [])
    return Answer(line)


def _eigenvects(m):
    line = []
    for k, (val, mult, vecs) in enumerate(_matrix(m).eigenvects()):
        if k:
            line.append(("; ", r";\quad "))
        line += [("λ = ", r"\lambda = "), X(val), (": ", r":\ ")]
        for j, v in enumerate(vecs):
            if j:
                line.append((", ", ",\ "))
            line.append(X(v.T))
    return Answer(line)


def _dot(u, v):
    return (_matrix(u).T * _matrix(v))[0] if _matrix(u).shape[1] == 1 else _matrix(u).dot(_matrix(v))


def _cross(u, v):
    return _matrix(u).cross(_matrix(v)).T


def _taylor(f, x=None, a=0, n=6):
    x = x if x is not None else sp.Symbol("x")
    return sp.series(f, x, a, n).removeO()


def _maxmin(fn_sympy, util):
    def f(*args):
        if len(args) == 4 and isinstance(args[1], sp.Symbol):  # max(f, x, a, b): on a segment
            return util(args[0], args[1], sp.Interval(args[2], args[3]))
        if len(args) == 2 and isinstance(args[1], sp.Symbol):  # max(f, x): on the whole line
            return util(args[0], args[1], sp.S.Reals)
        return fn_sympy(*args)
    return f


def _extrema(f, x=None):
    """Critical points of f(x) with their kind: x = …: минимум f = …"""
    x = x if x is not None else (sorted(sp.sympify(f).free_symbols, key=str) or [sp.Symbol("x")])[0]
    f = sp.sympify(f)
    d1 = sp.diff(f, x)
    pts = [p for p in sp.solve(d1, x) if p.is_real is not False and not p.has(sp.I)]
    if not pts:
        return Answer([W("экстремумов нет")])
    d2 = sp.diff(d1, x)
    line = []
    for k, p in enumerate(sorted(pts, key=lambda q: float(sp.N(q)) if q.is_number else 0)):
        sec = sp.simplify(d2.subs(x, p))
        kind = "минимум" if sec.is_positive else "максимум" if sec.is_negative else "перегиб?"
        if k:
            line.append(("; ", r";\quad "))
        line += [X(x), O("="), X(p), (": " + kind + " ", r":\ \text{%s}\ " % kind), ("f", "f"), O("="), X(sp.simplify(f.subs(x, p)))]
    return Answer(line)


def _local_dict():
    d = {}
    for ch in "abcdfghjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        d[ch] = sp.Symbol(ch)
    d["e"] = sp.E
    d["i"] = sp.I  # the imaginary unit; a summation index i is handled in parse()
    d["lg"] = lambda x: sp.log(x, 10)
    d["ln"] = sp.log
    d["tg"] = sp.tan
    d["ctg"] = sp.cot
    d["arcsin"] = sp.asin
    d["arccos"] = sp.acos
    d["arctg"] = sp.atan
    d["arctan"] = sp.atan
    d["int"] = _kb_int(sp.integrate)
    d["lim"] = _kb_lim(sp.limit)
    d["sum"] = _kb_sum(sp.summation)
    d["det"] = lambda m: _matrix(m).det()
    d["inv"] = lambda m: _matrix(m).inv()
    d["T"] = sp.Symbol("T")
    d.update({
        "factorint": _factorint, "factorize": _factorint, "isprime": _isprime, "divisors": _divisors,
        "totient": sp.totient, "phi": sp.totient, "gcd": sp.gcd, "lcm": sp.lcm, "nod": sp.gcd, "nok": sp.lcm,
        "binomial": sp.binomial, "_arr": _arrangements, "_perm": _perm,
        "bin": lambda n: _in_base(n, 2), "oct": lambda n: _in_base(n, 8), "hex": lambda n: _in_base(n, 16),
        "base": _in_base, "round": _round,
        "mean": _mean, "average": _mean, "median": _median, "mode": _mode,
        "var": lambda *a: _var(*a), "std": lambda *a: sp.sqrt(_var(*a)),
        "svar": lambda *a: _var(*a, sample=True), "sstd": lambda *a: sp.sqrt(_var(*a, sample=True)),
        "eigenvals": _eigenvals, "eigenvects": _eigenvects, "rank": lambda m: _matrix(m).rank(),
        "transpose": lambda m: _matrix(m).T, "trace": lambda m: _matrix(m).trace(),
        "rref": lambda m: _matrix(m).rref()[0], "nullspace": lambda m: Answer([X([v.T for v in _matrix(m).nullspace()])]),
        "charpoly": lambda m: _matrix(m).charpoly(sp.Symbol("lambda")).as_expr(),
        "dot": _dot, "cross": _cross,
        "taylor": _taylor,
        "max": _maxmin(sp.Max, __import__("sympy.calculus.util", fromlist=["maximum"]).maximum),
        "min": _maxmin(sp.Min, __import__("sympy.calculus.util", fromlist=["minimum"]).minimum),
        "extrema": _extrema, "extremum": _extrema,
    })
    return d


LOCALS = _local_dict()


def normalize_text(text):
    s = text.strip()
    # primes typed on the keyboard: y'' and (x^3)' are derivatives in x
    s = s.replace("′", "'").replace("″", "''")
    s = re.sub(r"\b([A-Za-z])''", r"diff(\1, x, 2)", s)
    s = re.sub(r"\b([A-Za-z])'", r"diff(\1, x)", s)
    s = _paren_primes(s)
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    s = s.replace("÷", "/").replace(":", "/").replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    s = s.replace("π", "pi").replace("∞", "oo").replace("∫", "int")
    # degrees: sin 30° = sin(30·π/180)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°", r"(\1*pi/180)", s).replace("°", "*pi/180")
    # percent: 200 + 15% = 230, 15% = 0.15 (a "%" followed by a number is the remainder: 17 % 5)
    s = re.sub(r"^(.*\S)\s*([+\-])\s*(\d+(?:\.\d+)?)\s*%\s*$", r"(\1)*(1\2\3/100)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\w(.])", r"(\1/100)", s)
    # combinatorics in school notation: C(n, k), A(n, k), P(n)
    s = re.sub(r"\bC\s*\(", "binomial(", s)
    s = re.sub(r"\bA\s*\(", "_arr(", s)
    s = re.sub(r"\bP\s*\(", "_perm(", s)
    # number bases: "100 in binary" / "в двоичной"
    s = re.sub(r"(\S+)\s+(?:in|в)\s+(?:binary|двоичн\w*)", r"bin(\1)", s)
    s = re.sub(r"(\S+)\s+(?:in|в)\s+(?:hex|шестнадцатеричн\w*)", r"hex(\1)", s)
    s = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", lambda m: "^(" + m.group(1).translate(SUPERSCRIPTS) + ")", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z])", r"sqrt(\1)", s)
    s = s.replace("√", "sqrt")
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    # matrices: [[1,2],[3,4]] -> Matrix([[1,2],[3,4]])
    s = re.sub(r"(?<![A-Za-z(])\[\[", "Matrix([[", s)
    s = re.sub(r"\]\](?!\))", "]])", s)
    s = s.replace(",", ", ")
    return s


MAX_LOG10 = 100_000  # numbers with more digits than this are only estimated, never computed
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(MAX_LOG10 + 100)  # "Точно" shows every digit


class TooBig(CalcError):
    """A numeric result too large to compute exactly; `log10` is its decimal logarithm."""

    def __init__(self, log10):
        super().__init__("число слишком большое")
        self.log10 = log10


def _to_mpf(e):
    if isinstance(e, sp.Integer):
        return mpmath.mpf(int(e.p))
    if isinstance(e, sp.Rational):
        return mpmath.mpf(int(e.p)) / int(e.q)
    if isinstance(e, sp.Float):
        return mpmath.mpf(e._mpf_)
    if e is sp.pi:
        return +mpmath.pi
    if e is sp.E:
        return +mpmath.e
    return None


def _magnitude(e):
    """Approximate value of a numeric expression written with + · ^ !, computed with floating exponents
    (cheap even for towers like 2^2^2^2^2^2). Raises TooBig when the exact value would be enormous."""
    v = _to_mpf(e)
    if v is not None:
        return v
    args = [_magnitude(a) for a in e.args]
    if any(a is None for a in args):
        return None
    if isinstance(e, sp.Pow):
        b, x = args
        if b == 0:
            return mpmath.mpf(0)
        lg = x * mpmath.log10(abs(b))
        if lg > MAX_LOG10:
            raise TooBig(lg)
        return mpmath.power(b, x) if b > 0 else mpmath.power(abs(b), x) * (-1 if x % 2 == 1 else 1)
    if isinstance(e, sp.Mul):
        v = mpmath.mpf(1)
        for a in args:
            v *= a
    elif isinstance(e, sp.Add):
        v = mpmath.fsum(args)
    elif isinstance(e, sp.factorial):
        n = args[0]
        if n > 25_000:
            raise TooBig(mpmath.loggamma(n + 1) / mpmath.log(10))
        return mpmath.factorial(n)
    else:
        return None
    if v != 0 and mpmath.log10(abs(v)) > MAX_LOG10:
        raise TooBig(mpmath.log10(abs(v)))
    return v


def _paren_primes(s):
    """(x^3)' -> diff((x^3), x); (…)'' -> second derivative."""
    while True:
        m = re.search(r"\)('+)", s)
        if not m:
            return s
        end = m.start()
        depth = 0
        for k in range(end, -1, -1):
            depth += s[k] == ")"
            depth -= s[k] == "("
            if depth == 0:
                break
        inner = s[k:end + 1]
        n = len(m.group(1))
        s = s[:k] + "diff(%s, x%s)" % (inner, ", %d" % n if n > 1 else "") + s[m.end():]


def editable_text(text):
    """The engine's linear text in the friendlier keyboard form: x^2, 1/2, int(f, x, 0, 1), lim(f, x, 0)."""
    t = text

    def calls(t, name, rewrite):
        out, i = "", 0
        while True:
            j = t.find(name + "(", i)
            if j < 0 or (j > 0 and (t[j - 1].isalnum() or t[j - 1] == "_")):
                if j < 0:
                    return out + t[i:]
                out += t[i:j + 1]
                i = j + 1
                continue
            depth, k = 0, j + len(name)
            for k in range(j + len(name), len(t)):
                depth += t[k] == "("
                depth -= t[k] == ")"
                if depth == 0:
                    break
            args = [a.strip() for a in split_top(t[j + len(name) + 1:k], ",")]
            out += t[i:j] + rewrite(args)
            i = k + 1

    def tuple_args(args):
        if len(args) == 2 and args[1].startswith("(") and args[1].endswith(")"):
            return [args[0]] + [a.strip() for a in split_top(args[1][1:-1], ",")]
        return args

    t = calls(t, "integrate", lambda a: "int(%s)" % ", ".join(tuple_args(a)))
    t = calls(t, "summation", lambda a: "sum(%s)" % ", ".join(tuple_args(a)))
    t = calls(t, "limit", lambda a: "lim(%s)" % ", ".join(a))
    for _ in range(4):  # (2) -> 2, ((1)/(2)) -> (1/2), x^(2) -> x^2
        t = re.sub(r"(?<![\w)])\(\s*([A-Za-z0-9.]+)\s*\)", r"\1", t)
    t = re.sub(r"\(\s*\(([^()]*)\)\s*\)", r"(\1)", t)
    t = re.sub(r"(?<=\w)\s*\*\s*(?=\w)", "·", t)
    t = re.sub(r"\bAbs\(\s*([^()]*?)\s*\)", r"|\1|", t)
    t = re.sub(r"\s+", " ", t).replace("( ", "(").replace(" )", ")")
    return t.strip()


def parse(text):
    if not text.strip():
        raise CalcError("пусто")
    norm = normalize_text(text)
    try:
        # look before computing: an exact 2^2^2^2^2^2 would never finish
        _magnitude(parse_expr(norm, local_dict=DISPLAY_LOCALS, transformations=TRANSFORMS, evaluate=False))
    except TooBig:
        raise
    except Exception:  # noqa: BLE001 - only an early check
        pass
    loc = LOCALS
    if re.search(r"\(\s*i\s*,", norm) or re.search(r",\s*i\s*,", norm):  # i is a summation index here
        loc = dict(LOCALS, i=sp.Symbol("i"))
    try:
        return parse_expr(norm, local_dict=loc, transformations=TRANSFORMS)
    except CalcError:
        raise
    except Exception as ex:  # noqa: BLE001 - every parser error is a user error here
        raise CalcError("не удалось разобрать: %s" % text) from ex


def split_top(text, seps):
    """Split text on separators that are not inside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in seps:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


REL_RE = re.compile(r"(<=|>=|!=|==|=|<|>)")


def split_relation(text):
    """'a = b' -> (['a', 'b'], ['='])  (relations only at bracket depth 0)."""
    pieces, ops = [], []
    depth = 0
    cur = ""
    k = 0
    while k < len(text):
        ch = text[k]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0:
            m = REL_RE.match(text, k)
            if m:
                pieces.append(cur)
                ops.append(m.group(1).replace("==", "="))
                cur = ""
                k = m.end()
                continue
        cur += ch
        k += 1
    pieces.append(cur)
    return pieces, ops


# ---------------------------------------------------------------------------
# Pretty printing for the canvas
# ---------------------------------------------------------------------------

SUP = str.maketrans("0123456789-+()nixyk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾ⁿⁱˣʸᵏ")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
NAMES = {"pi": "π", "alpha": "α", "beta": "β", "gamma": "γ", "theta": "θ", "lambda": "λ",
         "mu": "μ", "sigma": "σ", "phi": "φ"}
FUNC_NAMES = {"log": "ln", "asin": "arcsin", "acos": "arccos", "atan": "arctg", "tan": "tg",
              "cot": "ctg", "acot": "arcctg"}


BIG_DIGITS = 30  # integers longer than this are shown as m·10ⁿ


def _mant_exp(x, digits):
    """(mantissa text, exponent) of a real number, also far outside the float range."""
    with mpmath.workdps(digits + 5):
        if isinstance(x, sp.Integer):
            v = mpmath.mpf(int(x.p))
        elif isinstance(x, sp.Rational):
            v = mpmath.mpf(int(x.p)) / int(x.q)
        elif isinstance(x, sp.Basic):
            v = mpmath.mpf(sp.Float(x, digits + 3)._mpf_)
        else:
            v = mpmath.mpf(x)
        mant, _, exp = mpmath.nstr(v, digits, min_fixed=1, max_fixed=0).partition("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return mant, int(exp or 0)


def fmt_big_int(n, digits=8):
    mant, exp = _mant_exp(n, digits)
    return "%s·10%s" % (mant.replace("-", "−"), str(exp).translate(SUP))


def is_big_int(e):
    return isinstance(e, sp.Integer) and abs(int(e.p)).bit_length() > BIG_DIGITS * 3.33


def fmt_number(x, digits=10):
    if x == 0:
        return "0"
    try:
        f = float(x)
        if math.isinf(f) or (f != 0 and abs(f) < 1e-300):
            raise OverflowError
    except (OverflowError, TypeError, ValueError):
        mant, exp = _mant_exp(x, digits)
        return "%s·10%s" % (mant.replace("-", "−"), str(exp).translate(SUP))
    s = "%.*g" % (digits, f)
    if "e" in s:
        mant, exp = s.split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return "%s·10%s" % (mant, str(int(exp)).translate(SUP))
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace("-", "−")


def _sym_name(name):
    m = re.fullmatch(r"C(\d+)", name)  # integration constants of differential equations
    if m:
        return "C" + m.group(1).translate(SUB)
    if "_" in name:
        base, idx = name.split("_", 1)
        if idx.isdigit():
            return NAMES.get(base, base) + idx.translate(SUB)
        return NAMES.get(base, base) + "_" + idx
    return NAMES.get(name, name)


def _needs_parens(e):
    return isinstance(e, (sp.Add, sp.Mul)) or (isinstance(e, sp.Rational) and not e.is_Integer) \
        or (e.is_number and e.is_negative) or isinstance(e, sp.Pow) or isinstance(e, sp.Float) and e < 0


def _wrap(e):
    s = fmt(e)
    return "(%s)" % s if _needs_parens(e) else s


def _power(base_text, exp):
    if exp.is_Integer or (isinstance(exp, sp.Symbol) and len(exp.name) == 1 and exp.name in "nixyk"):
        return base_text + str(exp).translate(SUP)
    e_text = fmt(exp)
    if isinstance(exp, (sp.Add, sp.Mul, sp.Pow)) or (isinstance(exp, sp.Rational) and not exp.is_Integer):
        e_text = "(%s)" % e_text
    return base_text + "^" + e_text


def _fmt_mul(e):
    num, den = sp.fraction(e)
    if den != 1:
        n_text = fmt(num)
        d_text = fmt(den)
        if isinstance(num, sp.Add):
            n_text = "(%s)" % n_text
        if isinstance(den, (sp.Add, sp.Mul)) or (isinstance(den, sp.Pow) and not den.exp.is_Integer and den.exp != sp.S.Half):
            d_text = "(%s)" % d_text
        return "%s/%s" % (n_text, d_text)
    coeff, rest = e.as_coeff_Mul()
    factors = sp.Mul.make_args(rest)
    parts = []
    for f in factors:
        if isinstance(f, sp.Add):
            parts.append("(%s)" % fmt(f))
        else:
            parts.append(fmt(f))
    body = ""
    for p in parts:
        if body and (body[-1].isdigit() and (p[0].isdigit() or p[0] == "−")):
            body += "·"
        body += p
    if coeff == 1:
        return body
    if coeff == -1:
        return "−" + body
    c = fmt(coeff)
    if isinstance(coeff, sp.Rational) and not coeff.is_Integer:
        c = "(%s)" % c
    if body and (body[0].isdigit()):
        return c + "·" + body
    return c + body


def fmt(e):
    """Compact Unicode text of a SymPy object for drawing on the canvas."""
    if isinstance(e, (list, tuple)):
        return ", ".join(fmt(x) for x in e)
    if isinstance(e, sp.MatrixBase):
        rows = ["  ".join(fmt(x) for x in e.row(r)) for r in range(e.rows)]
        return "[" + "; ".join(rows) + "]"
    if e is sp.true:
        return "верно"
    if e is sp.false:
        return "неверно"
    if e is sp.S.Infinity:
        return "∞"
    if e is sp.S.NegativeInfinity:
        return "−∞"
    if e is sp.S.ComplexInfinity or e is sp.S.NaN:
        return "не определено"
    if e is sp.S.Pi:
        return "π"
    if e is sp.E:
        return "e"
    if e is sp.I:
        return "i"
    if isinstance(e, sp.Integer):
        if is_big_int(e):
            return fmt_big_int(e)
        return str(e).replace("-", "−")
    if isinstance(e, sp.Rational):
        return ("%d/%d" % (abs(e.p), e.q)) if e.p > 0 else "−%d/%d" % (abs(e.p), e.q)
    if isinstance(e, sp.Float):
        return fmt_number(e)
    if isinstance(e, sp.Symbol):
        return _sym_name(e.name)
    if isinstance(e, sp.Add):
        terms = e.as_ordered_terms()
        out = fmt(terms[0])
        for t in terms[1:]:
            if t.could_extract_minus_sign():
                out += " − " + fmt(-t)
            else:
                out += " + " + fmt(t)
        return out
    if isinstance(e, sp.Mul):
        if e.could_extract_minus_sign():
            inner = fmt(-e)
            return "−" + inner
        return _fmt_mul(e)
    if isinstance(e, sp.Pow):
        b, x = e.base, e.exp
        if x == sp.S.Half:
            inner = fmt(b)
            return "√" + (inner if (b.is_Atom and not (b.is_number and b.is_negative)) else "(%s)" % inner)
        if isinstance(x, sp.Rational) and x.p == 1 and x.q > 2:
            return str(x.q).translate(SUP) + "√" + _wrap(b) if not b.is_Atom else str(x.q).translate(SUP) + "√" + fmt(b)
        if x.is_number and x.is_negative:
            return "1/" + fmt(sp.Pow(b, -x, evaluate=False)) if not isinstance(b, sp.Add) else "1/(%s)" % fmt(sp.Pow(b, -x))
        return _power(_wrap(b), x)
    if isinstance(e, sp.exp):
        return _power("e", e.args[0])
    if isinstance(e, sp.Abs):
        return "|%s|" % fmt(e.args[0])
    if isinstance(e, sp.factorial):
        return _wrap(e.args[0]) + "!"
    if isinstance(e, sp.log) and len(e.args) == 1:
        return "ln" + _func_arg(e.args[0])
    if isinstance(e, sp.Function):
        name = type(e).__name__
        name = FUNC_NAMES.get(name, name)
        return name + "(%s)" % ", ".join(fmt(a) for a in e.args)
    if isinstance(e, Relational):
        op = {"==": "=", "<=": "≤", ">=": "≥", "!=": "≠"}.get(e.rel_op, e.rel_op)
        return "%s %s %s" % (fmt(e.lhs), op, fmt(e.rhs))
    if isinstance(e, sp.And):
        return " и ".join(fmt(a) for a in _ordered(e.args))
    if isinstance(e, sp.Or):
        return " или ".join(fmt(a) for a in _ordered(e.args))
    if isinstance(e, sp.Interval):
        return "%s%s; %s%s" % ("(" if e.left_open else "[", fmt(e.start), fmt(e.end), ")" if e.right_open else "]")
    if isinstance(e, sp.Integral):
        return "∫" + fmt(e.function) + " d" + ",".join(fmt(v) for v in e.variables)
    if isinstance(e, sp.Derivative):
        return "(%s)'" % fmt(e.expr)
    s = sp.sstr(e)
    s = s.replace("**", "^").replace("*", "·").replace("sqrt", "√")
    return re.sub(r"\boo\b", "∞", re.sub(r"\bpi\b", "π", s))


def _ordered(args):
    try:
        return sorted(args, key=lambda a: float(a.rhs) if hasattr(a, "rhs") and a.rhs.is_number else 0)
    except TypeError:
        return list(args)


def _func_arg(a):
    if a.is_Atom:
        return " " + fmt(a) if isinstance(a, sp.Symbol) else "(%s)" % fmt(a)
    return "(%s)" % fmt(a)


# ---------------------------------------------------------------------------
# LaTeX for typeset display (rendered with JLatexMath on Android)
# ---------------------------------------------------------------------------
# An answer line is a list of pieces; every piece is a (text, tex) pair, so the
# same line can be shown as compact Unicode text and as typeset math.

TEX_REL = {"=": "=", "≈": r"\approx", "<": "<", ">": ">", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
           "<=": r"\leq", ">=": r"\geq", "!=": r"\neq"}
TEX_FUNCS = [(r"\tan", r"\operatorname{tg}"), (r"\cot", r"\operatorname{ctg}"),
             (r"\operatorname{atan}", r"\operatorname{arctg}"), (r"\arctan", r"\operatorname{arctg}"),
             (r"\operatorname{acot}", r"\operatorname{arcctg}"), (r"\operatorname{acos}", r"\arccos"),
             (r"\operatorname{asin}", r"\arcsin")]


def to_tex(e):
    if isinstance(e, (list, tuple)):
        return ",\\ ".join(to_tex(x) for x in e)
    if e is sp.true:
        return r"\text{верно}"
    if e is sp.false:
        return r"\text{неверно}"
    if e is sp.S.ComplexInfinity or e is sp.S.NaN:
        return r"\text{не определено}"
    if isinstance(e, sp.Float):
        return num_tex(e)
    if is_big_int(e):
        mant, exp = _mant_exp(e, 8)
        return r"%s \cdot 10^{%d}" % (mant, exp)
    if isinstance(e, sp.And):
        return r"\ \text{и}\ ".join(to_tex(a) for a in _ordered(e.args))
    if isinstance(e, sp.Or):
        return r"\ \text{или}\ ".join(to_tex(a) for a in _ordered(e.args))
    return _russian_funcs(sp.latex(e, ln_notation=True, inv_trig_style="full", fold_func_brackets=True))


def _russian_funcs(s):
    for a, b in TEX_FUNCS:
        s = s.replace(a, b)
    return s


class _AsWrittenPrinter(LatexPrinter):
    """LaTeX of an unevaluated expression that keeps the written order and shows a/b as a fraction."""

    def _print_Mul(self, expr):
        args = list(expr.args)
        den = [a for a in args if isinstance(a, sp.Pow) and a.exp == -1]
        if not den:
            args = [a for a in args if a != 1] or [sp.S.One]
            if len(args) == 1:
                return self._print(args[0])
            return super()._print_Mul(sp.Mul(*args, evaluate=False))
        num = [a for a in args if not (isinstance(a, sp.Pow) and a.exp == -1) and a != 1]
        n = self._print(sp.Mul(*num, evaluate=False)) if len(num) > 1 else (self._print(num[0]) if num else "1")
        bases = [d.base for d in den]
        d = self._print(sp.Mul(*bases, evaluate=False)) if len(bases) > 1 else self._print(bases[0])
        return r"\frac{%s}{%s}" % (n, d)


_AS_WRITTEN = _AsWrittenPrinter({"order": "none", "ln_notation": True, "inv_trig_style": "full",
                                 "fold_func_brackets": True})


def num_tex(x, digits=10):
    s = fmt_number(x, digits).replace("−", "-")
    if "·10" in s:
        mant, exp = s.split("·10")
        return r"%s \cdot 10^{%s}" % (mant, exp.translate(SUPERSCRIPTS))
    return s


def X(e):
    """A SymPy object."""
    return (fmt(e), to_tex(e))


def W(words):
    """Words (Russian text)."""
    return (words, r"\text{%s}" % words)


def O(op):
    """A relation sign with spaces around it."""
    return (" %s " % op, " %s " % TEX_REL[op])


def NUM(x, digits=10):
    return (fmt_number(x, digits), num_tex(x, digits))


def ANY(var):
    """'x — любое'."""
    return [X(var), (" — любое", r" \in \mathbb{R}")]


def line_text(line):
    return "".join(p[0] for p in line)


def line_tex(line):
    return "".join(p[1] for p in line)


class _LogBase(sp.Function):
    """log_b(x) kept unevaluated for display."""
    nargs = 2

    def _latex(self, printer):
        return r"\log_{%s}{%s}" % (printer._print(self.args[1]), printer._print(self.args[0]))


def _display_log(x, b=None, **_):
    return sp.log(x, evaluate=False) if b is None else _LogBase(x, b)


def _display_limit(e, x, x0, *rest, **_):
    # a two-sided limit is printed without the "0⁺" direction mark
    return sp.Limit(e, x, x0, dir="+-") if sp.sympify(x0).is_finite else sp.Limit(e, x, x0)


def _display_root(a, n, **_):
    n = sp.sympify(n)
    return sp.Pow(a, sp.Rational(1, n) if n.is_Integer else sp.Pow(n, -1, evaluate=False), evaluate=False)


def _display_locals():
    d = dict(LOCALS)
    d.update({
        "integrate": sp.Integral, "int": _kb_int(sp.Integral),
        "limit": _display_limit, "lim": _kb_lim(_display_limit),
        "diff": sp.Derivative, "summation": sp.Sum, "sum": _kb_sum(sp.Sum),
        "log": _display_log, "ln": _display_log, "lg": lambda x, **_: _LogBase(x, 10),
        "root": _display_root,
        "factorial": lambda n, **_: sp.factorial(n, evaluate=False),
        "det": lambda m, **_: sp.Determinant(sp.ImmutableMatrix(m)),
        "inv": lambda m, **_: sp.Inverse(sp.ImmutableMatrix(m)),
    })
    return d


DISPLAY_LOCALS = _display_locals()


def expr_tex(text):
    """Typeset form of linear text as written (nothing is computed)."""
    e = parse_expr(normalize_text(text), local_dict=DISPLAY_LOCALS, transformations=TRANSFORMS, evaluate=False)
    return _russian_funcs(_AS_WRITTEN.doprint(e))


def text_to_tex(text):
    """'2x+1=7' -> '2 x + 1 = 7' in LaTeX; None when the text cannot be parsed."""
    try:
        pieces, ops = split_relation(text)
        out = []
        for k, piece in enumerate(pieces):
            if piece.strip():
                out.append(expr_tex(piece))
            if k < len(ops):
                out.append(" %s " % TEX_REL.get(ops[k], ops[k]))
        return "".join(out).strip() or None
    except Exception:  # noqa: BLE001 - display only
        return None


def display_tex(text):
    """Typeset preview of typed text (several statements separated by ";" are a system)."""
    parts = statements(text)
    if len(parts) > 1:
        return system_tex(parts)
    return text_to_tex(parts[0]) if parts else None


def training_label(text):
    """LaTeX label of a corrected line for fine-tuning (as written; the trailing "=" is cut off like the
    app cuts it off the ink). None when the text cannot be parsed."""
    t = re.sub(r"\bmod\b", "%", text.strip())
    t = re.sub(r"=\s*$", "", t).strip()
    if not t:
        return None
    tex = text_to_tex(t)
    if tex and "×" in text:  # the user wrote a cross, not a dot
        tex = tex.replace(r"\cdot", r"\times")
    return tex


def system_tex(texts):
    lines = [text_to_tex(t) for t in texts]
    if any(t is None for t in lines):
        return None
    return r"\begin{cases}" + r"\\".join(lines) + r"\end{cases}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VAR_ORDER = ["x", "y", "z", "t", "u", "v", "w", "n", "m", "k", "a", "b", "c"]


def order_vars(symbols):
    def key(s):
        base = s.name.split("_")[0]
        return (VAR_ORDER.index(base) if base in VAR_ORDER else len(VAR_ORDER), s.name)

    return sorted(symbols, key=key)


def has_float(text):
    return re.search(r"\d\.\d", text) is not None


def value_line(expr, decimal_only=False):
    """Exact value plus a decimal approximation when it helps (an answer line)."""
    if isinstance(expr, sp.MatrixBase):
        return [X(expr)]
    if expr.is_number and not expr.free_symbols:
        if expr.has(sp.zoo, sp.nan):
            return [W("не определено")]
        if expr in (sp.oo, -sp.oo):
            return [X(expr)]
        try:
            approx = sp.N(expr, 15)
        except Exception:  # noqa: BLE001
            approx = None
        is_real = approx is not None and approx.is_real
        if decimal_only and is_real:
            return [NUM(approx)]
        if expr.is_Integer:
            return [X(expr)]
        if isinstance(expr, sp.Float):
            return [NUM(expr)]
        if expr.has(sp.CRootOf) and is_real:  # a root with no formula: only its value
            return [("≈ ", r"\approx "), NUM(approx, 8)]
        exact = fmt(expr)
        if is_real:
            dec = NUM(approx, 8)
            if len(exact) > 24:
                return [("≈ ", r"\approx "), dec]
            if isinstance(expr, sp.Rational) and _is_finite_decimal(expr):
                return [X(expr), O("="), NUM(approx, 12)]
            return [X(expr), O("≈"), dec]
        if approx is not None and approx.is_complex and len(exact) > 24:
            re_, im_ = approx.as_real_imag()
            sign = (" − ", " - ") if im_ < 0 else (" + ", " + ")
            return [("≈ ", r"\approx "), NUM(re_, 6), sign, NUM(abs(im_), 6), ("i", "i")]
        return [X(expr)]
    return [X(expr)]


def too_big_line(log10):
    """≈ 10^(5.9·10¹⁹⁷²⁷) for a number with that many digits."""
    mant, exp = _mant_exp(log10, 3)
    inner_text = "%s·10%s" % (mant, str(exp).translate(SUP)) if exp >= 6 else fmt_number(log10, 8)
    inner_tex = r"%s \cdot 10^{%d}" % (mant, exp) if exp >= 6 else num_tex(log10, 8)
    return [("≈ 10^(%s)" % inner_text, r"\approx 10^{%s}" % inner_tex)]


def value_text(expr, decimal_only=False):
    return line_text(value_line(expr, decimal_only))


def _is_finite_decimal(r):
    q = r.q
    for p in (2, 5):
        while q % p == 0:
            q //= p
    return q == 1


def simplify_value(expr):
    if isinstance(expr, sp.MatrixBase):
        return expr.applyfunc(sp.simplify)
    if expr.is_number:
        r = sp.nsimplify(expr) if isinstance(expr, sp.Float) and False else sp.simplify(expr)
        return r
    r = sp.simplify(expr)
    return r


MAX_ROOTS = 6
EXACT_SOLVE_SECONDS = 3.0


def _solve_or_none(fn):
    """sp.solve with a short time limit: None when there is no formula or it takes too long."""
    box = {}

    def target():
        try:
            box["r"] = fn()
        except NotImplementedError:
            box["r"] = None
        except _Timeout:
            box["r"] = None
        except BaseException as ex:  # noqa: BLE001
            box["e"] = ex

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(EXACT_SOLVE_SECONDS)
    if t.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(t.ident), ctypes.py_object(_Timeout))
        t.join(1.0)
        return None
    if "e" in box:
        raise box["e"]
    return box.get("r")


def _real_fn(expr, var):
    f = sp.lambdify(var, expr, modules="mpmath")

    def g(x):
        try:
            v = f(mpmath.mpf(x))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return None
        if isinstance(v, mpmath.mpc):
            if abs(v.imag) > 1e-12 * max(1, abs(v.real)):
                return None
            v = v.real
        return v if mpmath.isfinite(v) else None
    return g


def numeric_roots(expr, var, limit=1e4):
    """Real roots of expr = 0: sign changes on a dense grid (denser near 0), refined with bisection-safe
    secant steps; plus local minima of |f| touching zero (double roots)."""
    g = _real_fn(expr, var)
    xs = sorted(set([0.0] + [s * 10 ** (k / 40.0) for k in range(-160, int(40 * math.log10(limit)) + 1) for s in (1, -1)]))
    vals = [(x, g(x)) for x in xs]
    roots = []

    def add(r):
        if all(abs(r - q) > 1e-7 * max(1, abs(r)) for q in roots):
            roots.append(r)

    for (x0, f0), (x1, f1) in zip(vals, vals[1:]):
        if f0 is None or f1 is None:
            continue
        if f0 == 0:
            add(float(x0))
        elif f0 * f1 < 0:
            lo, hi, flo = mpmath.mpf(x0), mpmath.mpf(x1), f0
            ok = True
            for _ in range(200):  # bisection: slow but never jumps to another root
                mid = (lo + hi) / 2
                fm = g(mid)
                if fm is None:
                    ok = False
                    break
                if fm == 0 or hi - lo < 1e-25 * max(1, abs(mid)):
                    break
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            if ok and abs(g((lo + hi) / 2) or 0) < 1e-6 * max(1, abs(f0), abs(f1)):  # not a pole like 1/x
                add(float((lo + hi) / 2))
    for (xa, fa), (xb, fb), (xc, fc) in zip(vals, vals[1:], vals[2:]):  # touching roots (x^2 = 0)
        if None not in (fa, fb, fc) and abs(fb) < 1e-9 and abs(fb) <= abs(fa) and abs(fb) <= abs(fc):
            add(float(xb))
    return sorted(roots)


def _solve_linear(A, b):
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            k = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= k * M[c][j]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))) / M[r][r]
    return x


def _newton(F, J, x, steps=60):
    """Damped Newton's method in real floating point; None when it fails from this start."""
    try:
        fx = F(x)
        for _ in range(steps):
            norm = max(abs(v) for v in fx)
            if norm < 1e-13:
                return x
            dx = _solve_linear(J(x), [-v for v in fx])
            if dx is None:
                return None
            t = 1.0
            while t > 1e-6:  # halve the step until the residual drops
                nx = [a + t * d for a, d in zip(x, dx)]
                if max(abs(v) for v in nx) > 1e8:
                    return None
                try:
                    nf = F(nx)
                except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                    nf = None
                if nf is not None and max(abs(v) for v in nf) < norm:
                    x, fx = nx, nf
                    break
                t /= 2
            else:
                return None
        return x if max(abs(v) for v in fx) < 1e-9 else None
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def numeric_system(eqs, symbols, tries=60, seconds=5.0):
    """Real solutions of a system without a formula: Newton's method from many starting points."""
    import random
    import time
    exprs = [e.lhs - e.rhs for e in eqs]
    # plain floats: a negative number to a fractional power is an error (not a complex number), so only
    # real solutions are followed, and every step takes microseconds
    fs = [sp.lambdify(symbols, e, modules="math") for e in exprs]
    jac = [[sp.lambdify(symbols, sp.diff(e, v), modules="math") for v in symbols] for e in exprs]

    def real(v):
        if isinstance(v, complex):
            raise ValueError
        return float(v)

    F = lambda x: [real(f(*x)) for f in fs]  # noqa: E731
    J = lambda x: [[real(d(*x)) for d in row] for row in jac]  # noqa: E731
    rng = random.Random(7)
    starts = [[c] * len(symbols) for c in (1.0, 2.0, 0.5, -1.0, 3.0, -2.0)]
    starts += [[rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 1.2) for _ in symbols] for _ in range(tries)]
    found = []
    t0 = time.time()
    idle = 0  # starts in a row that found nothing new
    for st in starts:
        if time.time() - t0 > seconds or len(found) >= MAX_ROOTS or (found and idle >= 20):
            break
        vals = _newton(F, J, st)
        if vals is not None and all(max(abs(a - b) for a, b in zip(vals, f)) > 1e-6 for f in found):
            found.append(vals)
            idle = 0
        else:
            idle += 1
    return sorted(found)


def _periodic_solutions(eq, var):
    """All solutions of a trigonometric equation: x = π/6 + 2πk, k ∈ ℤ (None if SymPy gives no such form)."""
    box = {}

    def run():
        box["s"] = sp.solveset(eq.lhs - eq.rhs, var, domain=sp.S.Reals)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(EXACT_SOLVE_SECONDS)
    st = box.get("s")
    if st is None or t.is_alive():
        return None
    if st is sp.S.EmptySet:
        return [[W("нет действительных решений")]]
    parts = list(st.args) if isinstance(st, sp.Union) else [st]
    if not any(isinstance(p, sp.ImageSet) for p in parts):
        return None
    k = sp.Symbol("k", integer=True)
    lines = []
    for p in parts:
        if isinstance(p, sp.ImageSet) and p.base_sets[0] == sp.S.Integers:
            n = p.lamda.variables[0]
            expr = p.lamda.expr.subs(n, k)
            expr = expr.xreplace({f: sp.Float(f, 10) for f in expr.atoms(sp.Float)})
            lines.append([X(var), O("="), X(expr), (", k ∈ ℤ", r",\ k \in \mathbb{Z}")])
        elif isinstance(p, sp.FiniteSet):
            lines += [[X(var), O("=")] + value_line(v) for v in p]
        else:
            return None
    return lines[:MAX_ROOTS]


def assign(name, value):
    """'x = value', or 'x ≈ value' when the value is only approximate."""
    if value and value[0][0] == "≈ ":
        return [name, O("≈")] + value[1:]
    return [name, O("=")] + value


def solve_equation(eq, decimal):
    """Answer lines, one per root."""
    symbols = order_vars(eq.free_symbols)
    var = symbols[0]
    if eq.has(sp.Abs) or eq.has(sp.sign):
        # |x − 2| = 5: SymPy needs to know x is real to open the modulus
        real = sp.Symbol(var.name, real=True)
        sols = _solve_or_none(lambda: [s.subs(real, var) for s in sp.solve(eq.subs(var, real), real, dict=False)])
    else:
        sols = _solve_or_none(lambda: sp.solve(eq, var, dict=False))
    if len(symbols) == 1 and eq.has(sp.sin, sp.cos, sp.tan, sp.cot):
        periodic = _periodic_solutions(eq, var)
        if periodic:
            return periodic
    if sols is None:
        # no formula (2^x = x^2 + 1, cos x = x): find the real roots numerically
        if len(symbols) > 1:
            raise CalcError("для такого уравнения нет формулы; численно решается только с одной переменной")
        roots = numeric_roots(eq.lhs - eq.rhs, var)
        if not roots:
            return [[W("действительных корней не найдено")]]
        more = 0
        if len(roots) > MAX_ROOTS + 3:  # periodic functions have endlessly many: show those nearest to 0
            more = len(roots) - MAX_ROOTS
            roots = sorted(sorted(roots, key=abs)[:MAX_ROOTS])
        name = lambda k: (fmt(var) + (str(k).translate(SUB) if len(roots) > 1 else ""),
                          "{%s}_{%d}" % (to_tex(var), k) if len(roots) > 1 else to_tex(var))
        lines = [[name(k), O("≈"), NUM(r, 10)] for k, r in enumerate(roots, 1)]
        if more:
            lines.append([W("… ещё %d в пределах ±10000" % more)])
        return lines
    if not sols:
        # identity?
        diff = sp.simplify(eq.lhs - eq.rhs)
        if diff == 0:
            return [ANY(var)]
        return [[W("нет решений")]]
    # real roots first: with letters in the equation SymPy cannot tell, so roots written with i are dropped
    real = [s for s in sols if s.is_real is not False and not s.has(sp.I)]
    if real:
        sols = real
    sols = _sort_solutions(sols)
    if len(sols) == 1:
        return [assign(X(var), value_line(sols[0], decimal))]
    if len(sols) == 2 and sp.simplify(sols[0] + sols[1]) == 0:
        pos = sols[1] if not sols[1].could_extract_minus_sign() else sols[0]
        return [[X(var), O("="), ("±", r"\pm ")] + value_line(pos, decimal)]
    lines = []
    shown = sols[:MAX_ROOTS]
    for k, s in enumerate(shown, 1):
        name = (fmt(var) + str(k).translate(SUB), "{%s}_{%d}" % (to_tex(var), k))
        lines.append(assign(name, value_line(s, decimal)))
    if len(sols) > len(shown):
        lines.append([W("… ещё %d" % (len(sols) - len(shown)))])
    return lines


def _sort_solutions(sols):
    def key(s):
        try:
            return (0, float(sp.re(sp.N(s))), float(sp.im(sp.N(s))))
        except (TypeError, ValueError):
            return (1, 0.0, 0.0)

    return sorted(sols, key=key)


def solve_inequality(rel):
    symbols = order_vars(rel.free_symbols)
    if len(symbols) != 1:
        raise CalcError("неравенство должно содержать одну переменную")
    var = symbols[0]
    res = sp.reduce_inequalities(rel, var)
    if res is sp.true:
        return ANY(var)
    if res is sp.false:
        return [W("нет решений")]
    return fmt_ineq(_clean_ineq(res, var), var)


def _clean_ineq(res, var):
    # drop the -oo < x < oo bounds sympy adds and put the variable on the left
    if isinstance(res, (sp.And, sp.Or)):
        args = [a for a in res.args if not (isinstance(a, Relational) and (a.has(sp.oo) or a.has(-sp.oo)))]
        args = [_clean_ineq(a, var) for a in args]
        if not args:
            return sp.true
        return type(res)(*args, evaluate=False) if len(args) > 1 else args[0]
    if isinstance(res, Relational) and res.rhs == var and res.lhs != var:
        return res.reversed
    return res


def fmt_ineq(res, var):
    """'x > -2 и x < 2' -> '−2 < x < 2'."""
    if isinstance(res, sp.And) and len(res.args) == 2:
        lo = [a for a in res.args if isinstance(a, (sp.StrictGreaterThan, sp.GreaterThan)) and a.lhs == var]
        hi = [a for a in res.args if isinstance(a, (sp.StrictLessThan, sp.LessThan)) and a.lhs == var]
        if lo and hi:
            lop = ">" if isinstance(lo[0], sp.StrictGreaterThan) else "≥"
            hop = "<" if isinstance(hi[0], sp.StrictLessThan) else "≤"
            lop = "<" if lop == ">" else "≤"
            return [X(lo[0].rhs), O(lop), X(var), O(hop), X(hi[0].rhs)]
    if isinstance(res, sp.Or):
        out = []
        for k, a in enumerate(_ordered(res.args)):
            if k:
                out.append((" или ", r"\ \text{или}\ "))
            out.extend(fmt_ineq(a, var))
        return out
    return [X(res)]


def solve_system(eqs, decimal):
    """Answer lines, one per solution."""
    rels = [e for e in eqs if not isinstance(e, sp.Equality)]
    if rels:
        symbols = order_vars(set().union(*[e.free_symbols for e in eqs]))
        if len(symbols) != 1:
            raise CalcError("неравенства с двумя переменными задают область на плоскости — её покажет график (скоро)")
        res = sp.reduce_inequalities(eqs, symbols[0])
        if res is sp.false:
            return [[W("нет решений")]]
        return [fmt_ineq(_clean_ineq(res, symbols[0]), symbols[0])]
    symbols = order_vars(set().union(*[e.free_symbols for e in eqs]))
    if not symbols:
        raise CalcError("в системе нет переменных")
    sols = _solve_or_none(lambda: sp.solve(eqs, symbols, dict=True))
    if sols is None:
        # no formula (x^x = 11 together with x^y …): solve numerically
        if len(eqs) != len(symbols):
            raise CalcError("для такой системы нет формулы; численно решается, когда уравнений столько же, сколько неизвестных")
        found = numeric_system(eqs, symbols)
        if not found:
            return [[W("действительных решений не найдено")]]
        lines = []
        for vals in found:
            line = []
            for v, val in zip(symbols, vals):
                if line:
                    line.append((", ", r",\quad "))
                line += [X(v), O("≈"), NUM(val, 8)]
            lines.append(line)
        return lines
    if not sols:
        return [[W("нет решений")]]
    real = [s for s in sols if all(v.is_real is not False for v in s.values())]
    if real:
        sols = real
    lines = []
    for s in sols:
        line = []
        for v in symbols:
            if v in s:
                if line:
                    line.append((", ", r",\quad "))
                line += assign(X(v), value_line(s[v], decimal))
        lines.append(line or [W("любые значения")])
    return lines


def is_function_definition(lhs, rhs):
    return isinstance(lhs, sp.Symbol) and lhs.name in ("y", "f") and rhs.free_symbols <= {sp.Symbol("x")}


def evaluate_trailing(lhs_text, forced):
    """Value of the left part of `expr =` as an answer line (or None), and the plot text."""
    decimal = has_float(lhs_text)
    try:
        raw = parse(lhs_text)
    except TooBig as big:
        return too_big_line(big.log10), None
    if isinstance(raw, Answer):
        return raw.line, None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = sp.sympify(raw)
    if isinstance(raw, (list, tuple, dict, bool)):
        return [X(sp.sympify(raw) if not isinstance(raw, dict) else list(raw.items()))], None
    if isinstance(raw, sp.MatrixBase):
        return value_line(raw.applyfunc(sp.simplify)), None
    if re.search(r"\b(apart|expand|factor|together|cancel|collect|expand_trig|trigsimp|series|taylor|rref|charpoly)\s*\(",
                 normalize_text(lhs_text)):
        # the user asked for this form: do not simplify it back
        return [X(raw)], (lhs_text if raw.free_symbols <= {sp.Symbol("x")} and raw.free_symbols else None)
    result = simplify_value(raw)
    plot = None
    if result.free_symbols:
        if result.free_symbols <= {sp.Symbol("x")}:
            plot = lhs_text
        unevaluated = None
        try:
            unevaluated = parse_expr(normalize_text(lhs_text), local_dict=LOCALS,
                                     transformations=TRANSFORMS, evaluate=False)
        except Exception:  # noqa: BLE001
            pass
        calls = re.search(r"[A-Za-z]{2,}\s*\(", normalize_text(lhs_text)) is not None
        same = not calls and unevaluated is not None and (sp.srepr(unevaluated) == sp.srepr(result) or str(unevaluated) == str(result))
        if same or (result == raw and not calls):
            alt = sp.expand(result)
            if alt != result:
                return [X(alt)], plot
            alt = sp.factor(result)
            if alt != result:
                return [X(alt)], plot
            if not forced:
                return None, plot
        return [X(result)], plot
    if re.fullmatch(r"\s*-?\s*\d+(\.\d+)?\s*", lhs_text) and not forced:
        return None, None
    line = Line(value_line(result, decimal))
    line.exact = exact_text(result)
    return line, None


class Line(list):
    """An answer line; `exact` holds the full value when the line shows a shortened one."""
    exact = None


EXACT_DIGITS = 50


def exact_text(expr):
    """All digits of a big integer, or 50 significant digits of an irrational value; None if nothing is hidden."""
    try:
        if isinstance(expr, sp.Integer):
            return str(int(expr.p)).replace("-", "−") if is_big_int(expr) else None
        if isinstance(expr, sp.Rational) and _is_finite_decimal(expr) and expr.q < 10 ** 12:
            return None
        if expr.is_number and not expr.free_symbols and not isinstance(expr, sp.MatrixBase):
            v = sp.N(expr, EXACT_DIGITS + 5)
            if v.is_real:
                with mpmath.workdps(EXACT_DIGITS + 5):
                    m = mpmath.mpf(v._mpf_)
                    text = mpmath.nstr(m, EXACT_DIGITS, min_fixed=-6, max_fixed=EXACT_DIGITS)
                mant, _, exp = text.partition("e")
                if "." in mant:
                    mant = mant.rstrip("0").rstrip(".")
                if exp:
                    return "%s·10%s" % (mant.replace("-", "−"), str(int(exp)).translate(SUP))
                return mant.replace("-", "−")
    except Exception:  # noqa: BLE001 - optional
        return None
    return None


def answer(kind, lines=None, plot=None, sep="; "):
    """Result dict: `answer` is Unicode text (lines joined by `sep`), `tex` the typeset lines,
    `exact` the full value behind a shortened one (one-line answers only)."""
    if not lines:
        return {"kind": kind, "answer": None, "tex": None, "plot": plot}
    # ✓ / ✗ marks read better as handwriting than as typeset symbols
    tex = None if kind == "check" else [line_tex(l) for l in lines]
    res = {"kind": kind, "answer": sep.join(line_text(l) for l in lines), "tex": tex, "plot": plot}
    exact = getattr(lines[0], "exact", None) if len(lines) == 1 else None
    if exact:
        res["exact"] = exact
    return res


def process(text, forced=False):
    """Process one statement (one line) of linear text.

    Returns a dict: kind (value|solve|check|inequality|function|none), answer, tex, plot.
    `forced` means the user explicitly asked for a result (keyboard '=' key).
    """
    text = text.strip()
    if re.search(r"\bmod\b", text):
        return process_mod(text, forced)
    pieces, ops = split_relation(text)
    if ops == ["="] and pieces[1].strip() and "diff(" in normalize_text(text) and re.search(r"\by\b", text):
        return process_ode(pieces[0], pieces[1])
    if not ops:
        if forced:
            line, plot = evaluate_trailing(text, True)
            return answer("value", [line] if line else None, plot)
        return answer("none")
    if len(ops) > 1:
        raise CalcError("слишком много знаков сравнения")
    lhs_text, rhs_text = pieces[0].strip(), pieces[1].strip()
    op = ops[0]
    if not lhs_text:
        return answer("none")
    if not rhs_text:
        if op != "=":
            return answer("none")
        line, plot = evaluate_trailing(lhs_text, forced)
        return answer("value" if line else "none", [line] if line else None, plot)

    decimal = has_float(text)
    lhs, rhs = parse(lhs_text), parse(rhs_text)
    if op == "=":
        if is_function_definition(lhs, rhs) and not isinstance(rhs, sp.MatrixBase):
            return answer("function", plot=rhs_text)
        eq = sp.Eq(lhs, rhs, evaluate=False)
        symbols = eq.free_symbols
        if not symbols:
            ok = sp.simplify(lhs - rhs) == 0
            if ok:
                return answer("check", [[("✓", "")]])
            return answer("check", [[("✗  (= ", "")] + value_line(sp.simplify(lhs), decimal) + [(")", "")]])
        return answer("solve", solve_equation(eq, decimal))
    rel = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge, "!=": sp.Ne}[op](lhs, rhs, evaluate=False)
    if not rel.free_symbols:
        val = bool(sp.simplify({"<": lhs < rhs, ">": lhs > rhs, "<=": lhs <= rhs, ">=": lhs >= rhs}.get(op, sp.Ne(lhs, rhs))))
        return answer("check", [[("✓", "")]] if val else [[("✗", "")]])
    if op == "!=":
        return answer("none")
    return answer("inequality", [solve_inequality(rel)])


Y_OF_X = sp.Function("y")(sp.Symbol("x"))


def process_ode(lhs_text, rhs_text):
    """A differential equation in y(x): y' = 2x, y'' + y = 0, dy/dx = y."""
    loc = dict(LOCALS, y=Y_OF_X)
    try:
        lhs = parse_expr(normalize_text(lhs_text), local_dict=loc, transformations=TRANSFORMS)
        rhs = parse_expr(normalize_text(rhs_text), local_dict=loc, transformations=TRANSFORMS)
    except Exception as ex:  # noqa: BLE001
        raise CalcError("не удалось разобрать: %s = %s" % (lhs_text, rhs_text)) from ex
    eq = sp.Eq(lhs, rhs)
    if not eq.has(sp.Derivative):
        raise CalcError("нет производной от y")
    try:
        sols = sp.dsolve(eq, Y_OF_X)
    except (NotImplementedError, ValueError) as ex:
        raise CalcError("это дифференциальное уравнение не решается в формулах") from ex
    if not isinstance(sols, list):
        sols = [sols]
    y = sp.Symbol("y")
    lines = []
    for sol in sols[:MAX_ROOTS]:
        lines.append([X(y), O("=")] + [X(sp.simplify(sol.rhs) if sol.rhs.count_ops() > 30 else sol.rhs)])
    return answer("solve", lines)


def process_mod(text, forced):
    """Arithmetic modulo n: '17 mod 5 =', '3x = 4 mod 7' (a congruence, solved by trying every residue)."""
    pieces, ops = split_relation(text)
    if len(ops) > 1:
        raise CalcError("слишком много знаков сравнения")
    if not ops or not pieces[1].strip():  # a value: mod is the remainder
        expr_text = re.sub(r"\bmod\b", "%", pieces[0])
        if not ops and not forced:
            return answer("none")
        val = sp.simplify(parse(expr_text))
        return answer("value", [value_line(val)])
    sides = [p.strip() for p in pieces]
    mod_side = next((k for k in (1, 0) if re.search(r"\bmod\b", sides[k])), None)
    m = re.match(r"(.*)\bmod\b\s*\(?\s*(\d+)\s*\)?\s*$", sides[mod_side])
    if not m:
        raise CalcError("пишите модуль в конце: 3x = 4 mod 7")
    n = int(m.group(2))
    if n < 2:
        raise CalcError("модуль должен быть больше 1")
    sides[mod_side] = m.group(1).strip() or "0"
    diff = sp.expand(parse(sides[0]) - parse(sides[1]))
    syms = order_vars(diff.free_symbols)
    if len(syms) > 1:
        raise CalcError("сравнение по модулю решается для одной переменной")
    n_tex = r"\ (\operatorname{mod}\ %d)" % n
    n_txt = " (mod %d)" % n
    if not syms:
        ok = diff.is_Integer and int(diff) % n == 0
        return answer("check", [[("✓" if ok else "✗", "")]])
    if n > 1_000_000:
        raise CalcError("модуль слишком большой")
    v = syms[0]
    num, den = sp.fraction(sp.together(diff))
    if not (sp.Poly(num, v).is_ground or all(c.is_Integer for c in sp.Poly(num, v).all_coeffs())) or not den.is_Integer:
        raise CalcError("по модулю решаются уравнения с целыми коэффициентами")
    coeffs = [int(c) for c in sp.Poly(num, v).all_coeffs()]
    roots = []
    for r in range(n):
        acc = 0
        for c in coeffs:  # Horner's rule, everything modulo n
            acc = (acc * r + c) % n
        if acc == 0:
            roots.append(r)
    if not roots:
        return answer("solve", [[W("нет решений") ]])
    shown = roots[:12]
    vals = ", ".join(str(r) for r in shown) + (", …" if len(roots) > len(shown) else "")
    return answer("solve", [[X(v), (" ≡ " + vals + n_txt, r" \equiv " + vals + n_tex)]])


IC_RE = re.compile(r"^\s*y\s*((?:'|′)*)\s*\(\s*([^()]+?)\s*\)\s*=\s*(.+?)\s*$")


def process_ivp(texts):
    """A differential equation with initial conditions: y' = y; y(0) = 1 (None when the texts are not that)."""
    ode = [t for t in texts if not IC_RE.match(t) and re.search(r"\by\b", t)
           and ("'" in t or "′" in t or "diff(" in normalize_text(t))]
    ics_text = [t for t in texts if IC_RE.match(t)]
    if len(ode) != 1 or not ics_text or len(ode) + len(ics_text) != len(texts):
        return None
    x = sp.Symbol("x")
    ics = {}
    for t in ics_text:
        m = IC_RE.match(t)
        order = len(m.group(1).replace("′", "'"))
        at, val = parse(m.group(2)), parse(m.group(3))
        target = Y_OF_X.diff(x, order).subs(x, at) if order else Y_OF_X.subs(x, at)
        ics[target] = val
    pieces, ops = split_relation(ode[0])
    if ops != ["="]:
        return None
    loc = dict(LOCALS, y=Y_OF_X)
    lhs = parse_expr(normalize_text(pieces[0]), local_dict=loc, transformations=TRANSFORMS)
    rhs = parse_expr(normalize_text(pieces[1]), local_dict=loc, transformations=TRANSFORMS)
    try:
        sol = sp.dsolve(sp.Eq(lhs, rhs), Y_OF_X, ics=ics)
    except (NotImplementedError, ValueError) as ex:
        raise CalcError("задача Коши не решается в формулах") from ex
    sols = sol if isinstance(sol, list) else [sol]
    return [[X(sp.Symbol("y")), O("="), X(s.rhs)] for s in sols[:MAX_ROOTS]]


def process_system(texts):
    """Answer lines of a system written one equation per line."""
    ivp = process_ivp(texts)
    if ivp is not None:
        return ivp
    eqs = []
    for t in texts:
        pieces, ops = split_relation(t)
        if len(ops) != 1 or not pieces[0].strip() or not pieces[1].strip():
            raise CalcError("в системе каждая строка должна быть уравнением")
        lhs, rhs = parse(pieces[0]), parse(pieces[1])
        op = ops[0]
        if op == "=":
            eqs.append(sp.Eq(lhs, rhs))
        else:
            eqs.append({"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}[op](lhs, rhs))
    decimal = any(has_float(t) for t in texts)
    return solve_system(eqs, decimal)


def statements(text):
    """Split keyboard text into statements by ';' or newlines (',' too when it splits relations)."""
    parts = []
    for line in text.replace(";", "\n").split("\n"):
        if not line.strip():
            continue
        sub = split_top(line, ",")
        if len(sub) > 1 and all(split_relation(s)[1] for s in sub):
            parts.extend(s for s in sub if s.strip())
        else:
            parts.append(line)
    return parts


# ---------------------------------------------------------------------------
# JSON API used from Kotlin
# ---------------------------------------------------------------------------

def _safe(fn):
    try:
        return fn()
    except CalcError as ex:
        return {"error": str(ex)}
    except RecursionError:
        return {"error": "слишком сложное выражение"}
    except Exception as ex:  # noqa: BLE001
        return {"error": "не удалось вычислить (%s)" % type(ex).__name__}


# ---------------------------------------------------------------------------
# Choosing among recognition candidates
# ---------------------------------------------------------------------------

# handwritten letters that are easily read instead of digits
DIGIT_LOOKALIKES = {"y": "4", "z": "2", "l": "1", "I": "1", "o": "0", "O": "0", "s": "5", "S": "5",
                    "g": "9", "q": "9", "b": "6", "B": "8"}
_EDGE = set(BINOPS) | set(RELOPS) | {"(", "{", "[", "^", "_", ","}


def fix_times(toks):
    """A '×' with no operand on one side was a letter x: '2 × + 3 = 7' -> '2 x + 3 = 7'."""
    out = list(toks)
    for i, t in enumerate(out):
        if t == "\\times":
            prev = out[i - 1] if i else None
            nxt = out[i + 1] if i + 1 < len(out) else None
            if prev is None or prev in _EDGE or nxt is None or nxt in _EDGE or nxt in (")", "}", "]"):
                out[i] = "x"
    return out


def numeric_variant(toks):
    """For 'expr =': x between numbers is ×, letters that look like digits are digits. None if nothing changes."""
    out = list(toks)
    changed = False
    # letters are only swapped for digits in plain arithmetic: "y^2 - 1 =" or "sin(y) =" keep their letters
    plain = not any(t in ("^", "(", "\\frac", "\\sqrt") or (t.startswith("\\") and t not in ("\\times", "\\cdot", "\\div"))
                    for t in toks)
    for i, t in enumerate(out):
        prev = out[i - 1] if i else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        if t == "x" and prev is not None and nxt is not None and \
                (prev.isdigit() or prev in (")", "}")) and (nxt.isdigit() or nxt in ("(", "\\frac", "\\sqrt")):
            out[i] = "\\times"
            changed = True
        elif plain and t in DIGIT_LOOKALIKES:
            out[i] = DIGIT_LOOKALIKES[t]
            changed = True
    return out if changed else None


def _is_value_request(text):
    pieces, ops = split_relation(text)
    return ops == ["="] and not pieces[1].strip()


def _has_symbols(text):
    try:
        return bool(parse(split_relation(text)[0][0]).free_symbols)
    except CalcError:
        return True


def _valid(text):
    """Balanced brackets and every part between relation signs parses."""
    depth = 0
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth < 0:
            return False
    if depth:
        return False
    pieces, _ = split_relation(re.sub(r"\bmod\b", "%", text))
    for p in pieces:
        if p.strip():
            try:
                parse(p)
            except TooBig:
                pass  # well-formed, just enormous
            except CalcError:
                return False
    return True


def evaluate_line(candidates, forced=False):
    """Evaluate the first reading that makes sense. Returns (text, result dict)."""
    first_error = None
    for toks in candidates[:6]:
        toks = fix_times(toks)
        try:
            text = tokens_to_text(toks)
            if not _valid(text):
                raise CalcError("не удалось разобрать: %s" % text)
            res = process(text, forced=forced)
        except CalcError as ex:
            if first_error is None:
                first_error = (" ".join(toks), {"error": str(ex)})
            continue
        except Exception as ex:  # noqa: BLE001
            if first_error is None:
                first_error = (" ".join(toks), {"error": "не удалось вычислить (%s)" % type(ex).__name__})
            continue
        # "2 x 3 =" or "2 + y =": a value was asked for, so a reading without letters is meant
        if _is_value_request(text) and _has_symbols(text) and not re.search(r"\bmod\b", text):
            alt = numeric_variant(toks)
            if alt is not None:
                try:
                    alt_text = tokens_to_text(alt)
                    if not _has_symbols(alt_text):
                        return alt_text, process(alt_text, forced=forced)
                except Exception:  # noqa: BLE001
                    pass
        return text, res
    if first_error is not None:
        return first_error
    return "", answer("none")


def _line_candidates(entry):
    """A line entry: text, a token list, or {"tokens": [...], "alts": [[...], ...]}."""
    if isinstance(entry, dict):
        return [entry.get("tokens") or []] + list(entry.get("alts") or [])
    return [entry]


def system_line_text(entry):
    """Text of one line of a system: the first reading that is a single parsable relation."""
    cands = _line_candidates(entry)
    if isinstance(cands[0], str):
        return cands[0]
    first = None
    for toks in cands[:6]:
        try:
            text = tokens_to_text(fix_times(toks))
        except CalcError:
            continue
        if first is None:
            first = text
        pieces, ops = split_relation(text)
        if len(ops) == 1 and pieces[0].strip() and pieces[1].strip():
            try:
                parse(pieces[0])
                parse(pieces[1])
                return text
            except CalcError:
                continue
    return first if first is not None else tokens_to_text(cands[0])


def evaluate_item(item):
    """item: {id, type: line|system|text, tokens (+ alts) | lines | text, forced}.

    The result has `input` (linear text), `input_tex`, `kind`, `answer`, `tex`, `plot` or `error`.
    """
    kind = item.get("type", "line")
    out = {"id": item.get("id")}
    if kind == "line":
        def run():
            text, res = evaluate_line(_line_candidates({"tokens": item["tokens"], "alts": item.get("alts")}),
                                      forced=item.get("forced", False))
            out["input"] = text
            out["input_tex"] = text_to_tex(text) if "error" not in res else None
            add_preview(text, res)
            return res
    elif kind == "system":
        def run():
            texts = [system_line_text(entry) for entry in item["lines"]]
            out["input"] = "; ".join(texts)
            complete = [t for t in texts if t.strip()]
            out["input_tex"] = system_tex(complete)
            if len(complete) < 2:
                return answer("none")
            return answer("system", process_system(complete), sep="\n")
    else:
        def run():
            text = item["text"]
            out["input"] = text
            parts = statements(text)
            if len(parts) > 1:
                out["input_tex"] = system_tex(parts)
                return answer("system", process_system(parts), sep="\n")
            out["input_tex"] = text_to_tex(parts[0]) if parts else None
            return process(parts[0] if parts else "", forced=item.get("forced", True))
    res = _safe(run)
    out.update(res)
    return out


def add_preview(text, res):
    """Without "=" there is no answer on the sheet, but the panel shows the value right away."""
    if res.get("kind") != "none" or "error" in res or not text.strip():
        return
    pieces, ops = split_relation(text)
    if ops:
        return
    try:
        line, _ = evaluate_trailing(re.sub(r"\bmod\b", "%", text), True)
    except Exception:  # noqa: BLE001 - a preview is optional
        return
    if line:
        res["preview"] = line_text(line)
        res["preview_tex"] = line_tex(line)


# ---------------------------------------------------------------------------
# Time limit and cache
# ---------------------------------------------------------------------------

class _Timeout(BaseException):
    """Raised inside a computation that ran out of time (BaseException: SymPy must not swallow it)."""


def run_limited(fn, seconds):
    """Run fn() in a thread; after `seconds` the thread is interrupted. Returns fn()'s dict or an error dict."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except _Timeout:
            pass
        except BaseException as ex:  # noqa: BLE001
            box["error"] = ex

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        # pure-Python SymPy code checks for async exceptions between bytecodes, so this stops it
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(t.ident), ctypes.py_object(_Timeout))
        t.join(2.0)
        return {"error": "считается дольше %s с — остановлено" % _fmt_seconds(seconds), "timeout": True}
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _fmt_seconds(x):
    return ("%g" % x).replace(".", ",")


_CACHE = OrderedDict()
_CACHE_SIZE = 300


def evaluate_item_cached(item, limit):
    """Results are cached by the item's content, so a slow line is not recomputed on every stroke."""
    key = json.dumps({k: v for k, v in item.items() if k != "id"}, sort_keys=True, ensure_ascii=False) + "|%g" % limit
    hit = _CACHE.get(key)
    if hit is None:
        if limit and limit > 0:
            hit = run_limited(lambda: evaluate_item(item), limit)
            if hit.get("timeout"):
                hit = dict(hit, input=_input_of(item))
        else:
            hit = evaluate_item(item)
        _CACHE[key] = hit
        if len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return dict(hit, id=item.get("id"))


def _input_of(item):
    try:
        if item.get("type") == "text":
            return item["text"]
        if item.get("type") == "system":
            return "; ".join(system_line_text(e) for e in item["lines"])
        return tokens_to_text(item["tokens"])
    except Exception:  # noqa: BLE001
        return ""


def evaluate_document(json_text, limit=10.0):
    """Evaluate all items; each one may take at most `limit` seconds (0 = no limit)."""
    items = json.loads(json_text)
    return json.dumps([evaluate_item_cached(it, float(limit)) for it in items], ensure_ascii=False)


def tokens_to_text_json(json_tokens):
    try:
        return tokens_to_text(json.loads(json_tokens))
    except CalcError:
        return " ".join(json.loads(json_tokens))


def plot_points(text, xmin, xmax, n):
    """Sample y = f(x). Returns JSON {"xs":[..], "ys":[..]} (null for gaps) or {"error": ..}."""
    def run():
        pieces, ops = split_relation(text)
        expr_text = pieces[-1] if ops and pieces[-1].strip() else pieces[0]
        expr = parse(expr_text)
        x = sp.Symbol("x")
        extra = expr.free_symbols - {x}
        if extra:
            raise CalcError("для графика нужна функция только от x")
        f = sp.lambdify(x, expr, modules=["math"])
        xs, ys = [], []
        step = (xmax - xmin) / (n - 1)
        for k in range(n):
            xv = xmin + k * step
            try:
                yv = f(xv)
                if isinstance(yv, complex) or yv is None:
                    yv = None
                else:
                    yv = float(yv)
                    if math.isnan(yv) or math.isinf(yv):
                        yv = None
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                yv = None
            xs.append(xv)
            ys.append(yv)
        return {"xs": xs, "ys": ys, "label": "y = " + fmt(expr)}

    return json.dumps(_safe(run), ensure_ascii=False)
