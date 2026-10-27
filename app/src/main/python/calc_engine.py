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


def tokens_to_text(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    return TokenConverter(list(tokens)).convert()


# ---------------------------------------------------------------------------
# Linear text -> SymPy
# ---------------------------------------------------------------------------

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _local_dict():
    d = {}
    for ch in "abcdfghjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        d[ch] = sp.Symbol(ch)
    d["e"] = sp.E
    d["i"] = sp.Symbol("i")
    d["lg"] = lambda x: sp.log(x, 10)
    d["ln"] = sp.log
    d["tg"] = sp.tan
    d["ctg"] = sp.cot
    d["arcsin"] = sp.asin
    d["arccos"] = sp.acos
    d["arctg"] = sp.atan
    d["arctan"] = sp.atan
    d["int"] = sp.integrate
    d["lim"] = sp.limit
    d["det"] = lambda m: sp.Matrix(m).det()
    d["inv"] = lambda m: sp.Matrix(m).inv()
    d["T"] = sp.Symbol("T")
    return d


LOCALS = _local_dict()


def normalize_text(text):
    s = text.strip()
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    s = s.replace("÷", "/").replace(":", "/").replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    s = s.replace("π", "pi").replace("∞", "oo").replace("∫", "int")
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


def parse(text):
    if not text.strip():
        raise CalcError("пусто")
    norm = normalize_text(text)
    try:
        # look before computing: an exact 2^2^2^2^2^2 would never finish
        _magnitude(parse_expr(norm, local_dict=DISPLAY_LOCALS, transformations=TRANSFORMS, evaluate=False))
    except CalcError:
        raise
    except Exception:  # noqa: BLE001 - only an early check
        pass
    try:
        return parse_expr(norm, local_dict=LOCALS, transformations=TRANSFORMS)
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
    return s.replace("**", "^").replace("*", "·").replace("pi", "π").replace("sqrt", "√").replace("oo", "∞")


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
        "integrate": sp.Integral, "int": sp.Integral,
        "limit": _display_limit, "lim": _display_limit,
        "diff": sp.Derivative, "summation": sp.Sum,
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


def solve_equation(eq, decimal):
    """Answer lines, one per root."""
    symbols = order_vars(eq.free_symbols)
    var = symbols[0]
    sols = sp.solve(eq, var, dict=False)
    if not sols:
        # identity?
        diff = sp.simplify(eq.lhs - eq.rhs)
        if diff == 0:
            return [ANY(var)]
        return [[W("нет решений")]]
    real = [s for s in sols if s.is_real is not False]
    if real:
        sols = real
    sols = _sort_solutions(sols)
    if len(sols) == 1:
        return [[X(var), O("=")] + value_line(sols[0], decimal)]
    lines = []
    for k, s in enumerate(sols, 1):
        name = (fmt(var) + str(k).translate(SUB), "{%s}_{%d}" % (to_tex(var), k))
        lines.append([name, O("=")] + value_line(s, decimal))
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
            raise CalcError("система неравенств должна содержать одну переменную")
        res = sp.reduce_inequalities(eqs, symbols[0])
        if res is sp.false:
            return [[W("нет решений")]]
        return [fmt_ineq(_clean_ineq(res, symbols[0]), symbols[0])]
    symbols = order_vars(set().union(*[e.free_symbols for e in eqs]))
    if not symbols:
        raise CalcError("в системе нет переменных")
    sols = sp.solve(eqs, symbols, dict=True)
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
                line += [X(v), O("=")] + value_line(s[v], decimal)
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
    if isinstance(raw, sp.MatrixBase):
        return value_line(raw.applyfunc(sp.simplify)), None
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
    return value_line(result, decimal), None


def answer(kind, lines=None, plot=None, sep="; "):
    """Result dict: `answer` is Unicode text (lines joined by `sep`), `tex` the typeset lines."""
    if not lines:
        return {"kind": kind, "answer": None, "tex": None, "plot": plot}
    # ✓ / ✗ marks read better as handwriting than as typeset symbols
    tex = None if kind == "check" else [line_tex(l) for l in lines]
    return {"kind": kind, "answer": sep.join(line_text(l) for l in lines), "tex": tex, "plot": plot}


def process(text, forced=False):
    """Process one statement (one line) of linear text.

    Returns a dict: kind (value|solve|check|inequality|function|none), answer, tex, plot.
    `forced` means the user explicitly asked for a result (keyboard '=' key).
    """
    text = text.strip()
    pieces, ops = split_relation(text)
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


def process_system(texts):
    """Answer lines of a system written one equation per line."""
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
    pieces, _ = split_relation(text)
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
        if _is_value_request(text) and _has_symbols(text):
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
        line, _ = evaluate_trailing(text, True)
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
