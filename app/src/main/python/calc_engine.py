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
    # symbols of the extended recogniser (model/finetune_mathwriting.py: EXTRA_VOCAB)
    "\\ln": "log",
    "\\cot": "cot",
    "\\arcsin": "asin",
    "\\arccos": "acos",
    "\\arctan": "atan",
    "\\sinh": "sinh",
    "\\cosh": "cosh",
    "\\tanh": "tanh",
    "\\exp": "exp",
    "\\det": "det",
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
    "\\delta": "δ",
    "\\epsilon": "ε",
    "\\varepsilon": "ε",
    "\\varphi": "φ",
    "\\omega": "ω",
    "\\rho": "ρ",
    "\\tau": "τ",
    "\\psi": "ψ",
    "\\eta": "η",
    "\\kappa": "κ",
    "\\nu": "ν",
    "\\xi": "ξ",
    "\\zeta": "ζ",
    "\\chi": "χ",
    "\\Omega": "Ω",
    "\\Sigma": "Σ",
    "\\Phi": "Φ",
    "\\Gamma": "Γ",
    "\\Psi": "Ψ",
    "\\Lambda": "Λ",
}
RELOPS = {"=": "=", "<": "<", ">": ">", "\\leq": "<=", "\\geq": ">=", "\\neq": "!=",
          "\\approx": "=", "\\equiv": "="}
BINOPS = {"+": "+", "-": "-", "\\times": "*", "\\cdot": "*", "\\div": "/", "/": "/", ":": "/", "*": "*",
          "\\wedge": "∧", "\\vee": "∨", "\\Rightarrow": ">>", "\\cup": "∪", "\\cap": "∩"}
UNSUPPORTED = {"\\exists", "\\forall", "\\in", "\\ldots", "\\cdots", "\\pm", "\\rightarrow",
               "\\notin", "\\subset", "\\subseteq", "\\supset", "\\supseteq", "\\emptyset", "\\perp",
               "\\parallel", "\\angle", "\\Leftrightarrow", "\\mapsto", "\\sim", "\\simeq", "\\oint",
               "\\mathbb", "\\circ"}
MATRIX_ENVS = {"\\begin{matrix}", "\\begin{pmatrix}", "\\begin{bmatrix}"}
OPEN = {"(": ")", "[": "]", "{": "}", "\\{": "\\}"}


class TokenConverter:
    def __init__(self, tokens):
        self.t = [tok for tok in tokens if tok not in ("\\limits", "\\prime", "\\hat", "\\dot", "\\vec")]
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
            if tok in ("\\bmod", "\\mod"):
                emit(" mod ", "op")
                continue
            if tok in MATRIX_ENVS:
                emit(self.matrix(tok), "value")
                continue
            if tok == "\\begin{cases}":
                emit(self.cases(), "value")
                continue
            if tok in ("&", "\\\\") or tok.startswith("\\end{"):
                continue
            if tok == "\\%":
                emit("%", "value")
                continue
            if tok == ";":
                emit("; ", "op")
                continue
            if tok == "\\neg":
                emit(" ~", "op")
                continue
            if tok == "\\nabla":
                emit(self.nabla(), "value")
                continue
            if tok == "\\overline":
                emit(" conjugate(%s)" % self.sub(self.group_tokens()), "value")
                continue
            if tok in ("\\lfloor", "\\lceil"):
                close = "\\rfloor" if tok == "\\lfloor" else "\\rceil"
                j = self.i
                while j < len(self.t) and self.t[j] != close:
                    j += 1
                inner = self.sub(self.t[self.i:j])
                self.i = j + 1
                emit(" %s(%s)" % ("floor" if tok == "\\lfloor" else "ceiling", inner), "value")
                continue
            if tok in ("\\max", "\\min"):
                emit(" " + tok[1:], "value")
                continue
            if tok == "\\prod":
                emit(self.summation(product=True), "value")
                continue
            if tok in ("\\iint", "\\iiint"):
                emit(self.integral(multiple=True), "value")
                continue
            if tok == "\\mid":
                tok = "|"

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
                power = self.group_tokens()
                if power == ["\\circ"]:
                    emit("°", "value")  # 30^{\circ}: degrees
                else:
                    emit("^(%s)" % self.sub(power), "value")
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
            elif tok == "\\partial":
                raise CalcError("∂ пишется дробью: ∂f/∂x")
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
        if num and num[0] == "\\partial" and den and den[0] == "\\partial":
            return self.partial(num, den)
        return " ((%s)/(%s))" % (self.sub(num), self.sub(den))

    def partial(self, num, den):
        """∂f/∂x, ∂²f/∂x², ∂²f/∂x∂y, and ∂/∂x f."""
        def power_at(toks, k):
            """'^ 2' or '^ { 2 }' at toks[k]: (exponent, tokens used); (1, 0) when there is none."""
            if toks[k:k + 1] != ["^"]:
                return 1, 0
            if toks[k + 1:k + 2] == ["{"] and k + 3 < len(toks) and toks[k + 2].isdigit() and toks[k + 3] == "}":
                return int(toks[k + 2]), 4
            if k + 1 < len(toks) and toks[k + 1].isdigit():
                return int(toks[k + 1]), 2
            return 1, 0

        _, used = power_at(num, 1)
        rest = num[1 + used:]
        # variables in the denominator: ∂x ∂y, ∂x^2
        vars_ = []
        k = 0
        while k < len(den):
            if den[k] == "\\partial" and k + 1 < len(den):
                v = den[k + 1]
                times, used = power_at(den, k + 2)
                vars_ += [v] * times
                k += 2 + used
            else:
                k += 1
        if not vars_:
            raise CalcError("непонятная частная производная")
        body = self.sub(rest) if rest else self.operand(stop_at_func=False)
        return " diff(%s, %s)" % (body, ", ".join(vars_))

    def nabla(self):
        """∇f (gradient), ∇·F (divergence), ∇×F (curl), ∇²f (Laplacian)."""
        if self.peek() == "^":
            self.take()
            self.group_tokens()
            return " laplacian(%s)" % self.operand(stop_at_func=False)
        if self.peek() in ("\\cdot", "\\times"):
            kind = "div" if self.take() == "\\cdot" else "curl"
            return " %s(%s)" % (kind, self.operand(stop_at_func=False))
        return " grad(%s)" % self.operand(stop_at_func=False)

    def matrix(self, begin):
        end = begin.replace("begin", "end")
        j = self.i
        depth = 0
        while j < len(self.t):
            if self.t[j] in MATRIX_ENVS:
                depth += 1
            elif self.t[j] == end and depth == 0:
                break
            elif self.t[j].startswith("\\end{") and self.t[j] != end:
                pass
            elif self.t[j] == end:
                depth -= 1
            j += 1
        inner = self.t[self.i:j]
        self.i = j + 1
        rows, row, cell = [], [], []
        for tok in inner + ["\\\\"]:
            if tok == "&":
                row.append(cell)
                cell = []
            elif tok == "\\\\":
                row.append(cell)
                cell = []
                if any(row_cell for row_cell in row):
                    rows.append(row)
                row = []
            else:
                cell.append(tok)
        if not rows:
            raise CalcError("пустая матрица")
        return " [%s]" % ", ".join("[%s]" % ", ".join(self.sub(c) if c else "0" for c in r) for r in rows)

    def cases(self):
        j = self.i
        while j < len(self.t) and self.t[j] != "\\end{cases}":
            j += 1
        inner = self.t[self.i:j]
        self.i = j + 1
        parts, cur = [], []
        for tok in inner + ["\\\\"]:
            if tok == "\\\\":
                if cur:
                    parts.append(self.sub([t for t in cur if t != "&"]))
                cur = []
            else:
                cur.append(tok)
        return "; ".join(parts)

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

    def integral(self, multiple=False):
        lo, hi = self.limits()
        if multiple:
            # ∬ f dx dy: every differential at the end; limits (rare by hand) are not supported here
            j = len(self.t)
            k = self.i
            depth = 0
            while k < len(self.t):
                if self.t[k] in OPEN:
                    depth += 1
                elif self.t[k] in OPEN.values():
                    depth -= 1
                elif depth == 0 and (self.t[k] in RELOPS or self.t[k] == ","):
                    j = k
                    break
                k += 1
            seg = self.t[self.i:j]
            diffs = []
            while len(seg) >= 2 and seg[-2] == "d" and len(seg[-1]) == 1 and seg[-1].isalpha():
                diffs.insert(0, seg[-1])
                seg = seg[:-2]
            self.i = j
            if not diffs:
                raise CalcError("у кратного интеграла нужны dx dy")
            return " integrate(%s, %s)" % (self.sub(seg) if seg else "1", ", ".join(diffs))
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

    def summation(self, product=False):
        lo, hi = self.limits()
        if lo is None or hi is None:
            raise CalcError("у произведения нужны пределы" if product else "у суммы нужны пределы")
        if "=" in lo:
            k = lo.index("=")
            var, start = lo[:k], lo[k + 1:]
        else:
            var, start = ["n"], ["1"]
        body = self.operand(stop_at_func=False)
        return " %s(%s, (%s, %s, %s))" % ("product" if product else "summation", body, self.sub(var), self.sub(start),
                                           self.sub(hi))

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
    text = TokenConverter(rewrite_primes(rewrite_words(list(tokens)))).convert()
    # x ≡ 3 (mod 7): the modulus in brackets
    return re.sub(r"\(\s*mod\s+([^()]+?)\s*\)", r"mod \1", text)


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


def _study(f, x=None):
    """Investigate a function of one variable: domain, zeros, extrema, monotonicity, asymptotes."""
    f = sp.sympify(f)
    x = x if x is not None else (sorted(f.free_symbols, key=str) or [sp.Symbol("x")])[0]
    lines = []
    from sympy.calculus.util import continuous_domain
    dom = continuous_domain(f, x, sp.S.Reals)
    lines.append([W("область определения: "), X(dom) if dom != sp.S.Reals else ("ℝ", r"\mathbb{R}")])
    zeros = sp.solveset(f, x, dom)
    if isinstance(zeros, sp.FiniteSet):
        lines.append([W("нули: ")] + ([X(sorted(zeros, key=lambda q: float(sp.N(q))))] if zeros else [W("нет")]))
    d = sp.simplify(sp.diff(f, x))
    crit = sp.solveset(d, x, dom)
    sing = sp.solveset(1 / sp.simplify(f), x, sp.S.Reals) if not dom == sp.S.Reals else sp.S.EmptySet
    if isinstance(crit, sp.FiniteSet) and (isinstance(sing, sp.FiniteSet) or sing is sp.S.EmptySet):
        cut = sorted(set(crit) | set(sing), key=lambda q: float(sp.N(q)))
        bounds = [-sp.oo] + cut + [sp.oo]
        up, down = [], []
        for a, b in zip(bounds, bounds[1:]):
            mid = (a + b) / 2 if a.is_finite and b.is_finite else (b - 1 if a == -sp.oo and b.is_finite else a + 1 if b == sp.oo and a.is_finite else 0)
            if mid not in dom:
                continue
            sign = d.subs(x, mid)
            (up if sign.is_positive else down if sign.is_negative else []).append(sp.Interval.open(a, b))
        if up:
            lines.append([W("возрастает на "), X(sp.Union(*up))])
        if down:
            lines.append([W("убывает на "), X(sp.Union(*down))])
        ext = _extrema(f, x).line
        lines.append([W("экстремумы: ")] + ext)
    vert = [p for p in (sing if isinstance(sing, sp.FiniteSet) else [])
            if sp.limit(f, x, p, "+").is_infinite or sp.limit(f, x, p, "-").is_infinite]
    if vert:
        lines.append([W("вертикальные асимптоты: ")] + sum(([X(x), O("="), X(p), (";  ", r";\ ")] for p in vert), [])[:-1])
    found = {}
    for side, inf in (("+∞", sp.oo), ("−∞", -sp.oo)):
        try:
            k = sp.limit(f / x, x, inf)
            if k.is_finite:
                b = sp.limit(f - k * x, x, inf)
                if b.is_finite:
                    found.setdefault(sp.simplify(k * x + b), []).append(side)
        except Exception:  # noqa: BLE001
            pass
    for line, sides in found.items():
        where = "x → ±∞" if len(sides) == 2 else "x → " + sides[0]
        lines.append([W("асимптота при %s: " % where), ("y", "y"), O("="), X(line)])
    return MultiAnswer(lines)


class MultiAnswer:
    """Several answer lines returned by one function (shown one under another)."""

    def __init__(self, lines):
        self.lines = lines


def _lsolve(a, b):
    """A·X = B, like MathCad's lsolve."""
    a, b = _matrix(a), _matrix(b)
    if b.shape[0] != a.shape[0] and b.shape[1] == a.shape[0]:
        b = b.T
    sol, params = a.gauss_jordan_solve(b)
    if params.shape[0]:
        sol = sol.subs({p: sp.Symbol("t%d" % (k + 1)) for k, p in enumerate(params)})
    return sol.T


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
        "extrema": _extrema, "extremum": _extrema, "study": _study, "investigate": _study,
        "lsolve": _lsolve,
    })
    return d


LOCALS = _local_dict()


def _wrap_matrices(s):
    """[[1,2],[3,4]] -> Matrix([[1,2],[3,4]]) unless the brackets are an argument of Matrix(...) already."""
    out, i, closers = [], 0, []
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "[":
            if s.startswith("[[", i) and not re.search(r"Matrix\(\s*$", s[:i]):
                out.append("Matrix(")
                closers.append(depth)
            depth += 1
        elif c == "]":
            depth -= 1
            out.append(c)
            if closers and closers[-1] == depth:
                closers.pop()
                out.append(")")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


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
    s = _wrap_matrices(s)
    # logic typed with symbols (after |x| became Abs)
    s = s.replace("∧", " & ").replace("∨", " | ").replace("¬", "~").replace("⊕", " ^ ")
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
    text = _ctx_primes(text)
    norm = normalize_text(text)
    try:
        # look before computing: an exact 2^2^2^2^2^2 would never finish
        _magnitude(parse_expr(norm, local_dict=DISPLAY_LOCALS, transformations=TRANSFORMS, evaluate=False))
    except TooBig:
        raise
    except Exception:  # noqa: BLE001 - only an early check
        pass
    loc = dict(LOCALS, **_CTX) if _CTX else LOCALS
    if re.search(r"\(\s*i\s*,", norm) or re.search(r",\s*i\s*,", norm):  # i is a summation index here
        loc = dict(loc, i=sp.Symbol("i"))
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
    if exp == 1:
        return base_text
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
    if isinstance(e, sp.Union):
        return " ∪ ".join(fmt(a) for a in e.args)
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


    def _print_AppliedUndef(self, expr):
        # user functions f(3), f'(x): always with brackets
        name = expr.func.__name__
        primes = re.match(r"(.+)__d(\d)$", name)
        if primes:
            name = primes.group(1) + "'" * int(primes.group(2))
        return r"%s\left(%s\right)" % (name, ", ".join(self._print(a) for a in expr.args))


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
    loc = DISPLAY_LOCALS
    funcs = [k for k, v in _CTX.items() if isinstance(v, sp.Lambda)]
    if funcs:
        loc = dict(loc, **{k: sp.Function(k) for k in funcs})
        text = _ctx_primes(text)
    e = parse_expr(normalize_text(text), local_dict=loc, transformations=TRANSFORMS, evaluate=False)
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
    if isinstance(raw, MultiAnswer):
        return raw, None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = sp.sympify(raw)
    if isinstance(raw, (list, tuple, dict, bool)):
        return [X(sp.sympify(raw) if not isinstance(raw, dict) else list(raw.items()))], None
    if isinstance(raw, sp.MatrixBase):
        return value_line(raw.applyfunc(sp.simplify)), None
    if re.search(r"\b(apart|expand|factor|together|cancel|collect|expand_trig|trigsimp|series|taylor|rref|charpoly|laurent|polyfit)\s*\(",
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


# ---------------------------------------------------------------- units of measurement

def _unit_table():
    import sympy.physics.units as u
    m, s_, kg, g = u.meter, u.second, u.kilogram, u.gram
    J, W, N, Pa, L = u.joule, u.watt, u.newton, u.pascal, u.liter
    t = {
        "m": m, "cm": u.centimeter, "mm": u.millimeter, "km": u.kilometer, "um": u.micrometer, "μm": u.micrometer,
        "nm": u.nanometer, "dm": m / 10, "in": u.inch, "inch": u.inch, "ft": u.foot, "yd": u.yard, "mi": u.mile,
        "mile": u.mile, "nmi": u.nautical_mile, "AU": u.astronomical_unit, "ly": 9460730472580800 * m,
        "s": s_, "sec": s_, "ms": u.millisecond, "us": u.microsecond, "ns": u.nanosecond, "min": u.minute,
        "h": u.hour, "hr": u.hour, "d": u.day, "day": u.day, "week": 7 * u.day, "yr": u.year, "year": u.year,
        "kg": kg, "g": g, "mg": u.milligram, "t": u.tonne, "lb": u.pound, "oz": u.pound / 16,
        "N": N, "kN": 1000 * N, "J": J, "kJ": 1000 * J, "MJ": 10**6 * J, "cal": sp.Rational(4184, 1000) * J,
        "kcal": 4184 * J, "eV": u.electronvolt, "Wh": W * u.hour, "kWh": 1000 * W * u.hour,
        "W": W, "kW": 1000 * W, "MW": 10**6 * W, "GW": 10**9 * W, "hp": sp.Rational(7354987, 10000) * W,
        "Pa": Pa, "kPa": 1000 * Pa, "MPa": 10**6 * Pa, "bar": u.bar, "atm": u.atmosphere, "mmHg": sp.Rational(101325, 760) * Pa,
        "psi": u.psi, "L": L, "l": L, "mL": u.milliliter, "ml": u.milliliter, "gal": sp.Rational(3785411784, 10**12) * m**3,
        "ha": u.hectare, "V": u.volt, "A": u.ampere, "mA": u.ampere / 1000, "Ohm": u.ohm, "Ω": u.ohm,
        "C": u.coulomb, "Hz": u.hertz, "kHz": 1000 * u.hertz, "MHz": 10**6 * u.hertz, "GHz": 10**9 * u.hertz,
        "K": u.kelvin, "B": u.byte, "KB": 1024 * u.byte, "MB": 1024**2 * u.byte, "GB": 1024**3 * u.byte,
        "bit": u.bit, "kmh": u.kilometer / u.hour, "kph": u.kilometer / u.hour, "mph": u.mile / u.hour,
        "rad": sp.S.One, "deg": sp.pi / 180,
    }
    ru = {
        "м": "m", "см": "cm", "мм": "mm", "км": "km", "мкм": "um", "нм": "nm", "дм": "dm", "дюйм": "in",
        "фут": "ft", "миля": "mi", "с": "s", "сек": "s", "мс": "ms", "мин": "min", "ч": "h", "час": "h",
        "сут": "d", "нед": "week", "год": "yr", "кг": "kg", "г": "g", "мг": "mg", "т": "t", "фунт": "lb",
        "Н": "N", "кН": "kN", "Дж": "J", "кДж": "kJ", "МДж": "MJ", "кал": "cal", "ккал": "kcal", "эВ": "eV",
        "Вт": "W", "кВт": "kW", "МВт": "MW", "ГВт": "GW", "Втч": "Wh", "кВтч": "kWh", "лс": "hp",
        "Па": "Pa", "кПа": "kPa", "МПа": "MPa", "бар": "bar", "атм": "atm", "л": "L", "мл": "mL", "га": "ha",
        "В": "V", "А": "A", "мА": "mA", "Ом": "Ohm", "Кл": "C", "Гц": "Hz", "кГц": "kHz", "МГц": "MHz",
        "ГГц": "GHz", "К": "K", "Б": "B", "КБ": "KB", "МБ": "MB", "ГБ": "GB", "бит": "bit",
        "рад": "rad", "град": "deg",
    }
    t.update({k: t[v] for k, v in ru.items()})
    return t


_UNITS = None
CONVERT_RE = re.compile(r"^(.+?)\s+(?:in|to|в|->|→)\s+([^=]+?)\s*=?\s*$")
TEMP_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*°?\s*([CFKСК]|℃|℉)\s+(?:in|to|в|->|→)\s+°?\s*([CFKСК]|℃|℉)\s*=?\s*$")


def _units():
    global _UNITS
    if _UNITS is None:
        _UNITS = _unit_table()
    return _UNITS


def _unit_expr(text):
    t = text.replace("·", "*").replace("²", "^2").replace("³", "^3").replace(",", ".")
    t = re.sub(r"\b(к?Вт)\s*[*·]?\s*ч\b", r"\1ч", t)
    t = re.sub(r"\bкм\s*/\s*ч\b", "km/h", t)
    t = re.sub(r"(?<=\d)(?=[A-Za-zА-Яа-яμΩ])", " ", t)
    names = re.findall(r"[A-Za-zА-Яа-яμΩ]+", t)
    table = _units()
    if not names or any(n not in table for n in names):
        return None
    try:
        return parse_expr(t, local_dict=dict(table), transformations=TRANSFORMS)
    except Exception:  # noqa: BLE001
        return None


def _si(q):
    import sympy.physics.units as u
    return u.convert_to(q, [u.meter, u.second, u.kilogram, u.ampere, u.kelvin, u.byte])


def process_units(text):
    """'30 km/h in m/s' -> 25/3 m/s ≈ 8.333333 m/s; '20 °C in F' -> 68 °F. None when not a conversion."""
    m = TEMP_RE.match(text)
    if m:
        v = sp.nsimplify(m.group(1).replace(",", "."))
        norm = {"С": "C", "К": "K", "℃": "C", "℉": "F"}
        a, b = norm.get(m.group(2), m.group(2)), norm.get(m.group(3), m.group(3))
        kelvin = {"C": v + sp.Rational(27315, 100), "F": (v - 32) * sp.Rational(5, 9) + sp.Rational(27315, 100), "K": v}[a]
        out = {"C": kelvin - sp.Rational(27315, 100), "F": (kelvin - sp.Rational(27315, 100)) * sp.Rational(9, 5) + 32,
               "K": kelvin}[b]
        unit = {"C": "°C", "F": "°F", "K": "K"}[b]
        return answer("value", [value_line(out, False) + [(" " + unit, r"\ \text{%s}" % unit)]])
    m = CONVERT_RE.match(text)
    if not m:
        # 5 км + 300 м =   (the answer in the first unit written)
        expr_text = text.rstrip().rstrip("=").strip()
        names = re.findall(r"[A-Za-zА-Яа-яμΩ]+", expr_text)
        if not text.rstrip().endswith("=") or not names or not re.search(r"\d", expr_text) or not (
                any(re.search(r"[А-Яа-я]", n) for n in names) or any(len(n) > 1 for n in names)):
            return None
        if _unit_expr(expr_text) is None:
            return None
        cyr = any(re.search(r"[А-Яа-я]", n) for n in names)
        firsts = [names[0]] + (["Н", "Дж", "Вт", "Па", "м/с", "м/с^2", "кг", "м", "с", "м^2", "м^3"] if cyr
                               else ["N", "J", "W", "Pa", "m/s", "m/s^2", "kg", "m", "s", "m^2", "m^3", "C", "V", "Ohm"])
        for target in firsts:
            try:
                return process_units(expr_text + " in " + target)
            except CalcError:
                continue
        raise CalcError("единицы не складываются в известную величину")
    if re.search(r"\b(binary|hex|octal|двоичн|шестнадцат|восьмерич)", m.group(2)):
        return None
    target = _unit_expr(m.group(2))
    src = _unit_expr(m.group(1))
    if target is None or src is None or not target.free_symbols and not target.atoms(sp.physics.units.Quantity):
        return None
    ratio = sp.simplify(_si(src) / _si(target))
    if ratio.atoms(sp.physics.units.Quantity) or ratio.free_symbols:
        raise CalcError("единицы несовместимы: %s и %s" % (m.group(1).strip(), m.group(2).strip()))
    unit = m.group(2).strip().replace("^2", "²").replace("^3", "³")
    unit_tex = r"\ \text{%s}" % unit
    if ratio.is_Rational and not sp.factorint(ratio.q).keys() - {2, 5}:
        ratio = sp.Float(ratio, 15)  # 1.609344 km rather than 25146/15625
    line = value_line(ratio, False)
    out = []
    for piece in line:  # "25/3 ≈ 8.33" -> "25/3 m/s ≈ 8.33 m/s"
        if out and piece[0].strip() in ("=", "≈"):
            out.append((" " + unit, unit_tex))
        out.append(piece)
    return answer("value", [out + [(" " + unit, unit_tex)]])


def process(text, forced=False):
    """Process one statement (one line) of linear text.

    Returns a dict: kind (value|solve|check|inequality|function|none), answer, tex, plot.
    `forced` means the user explicitly asked for a result (keyboard '=' key).
    """
    text = text.strip()
    if re.search(r"\s(in|to|в|->|→)\s", text) or re.search(r"\d\s*[A-Za-zА-Яа-я]{2,}|[А-Яа-я]", text):
        res = process_units(text)
        if res is not None:
            return res
    if re.search(r"\bmod\b", text):
        return process_mod(text, forced)
    if calc_higher.REC_TERM.search(text):
        rec = calc_higher.process_recurrence([text])
        if rec is not None:
            return answer("solve", rec)
    pieces, ops = split_relation(text)
    if ops == ["="] and pieces[1].strip() and "diff(" in normalize_text(text) and re.search(r"\by\b", text):
        return process_ode(pieces[0], pieces[1])
    if not ops:
        if forced:
            line, plot = evaluate_trailing(text, True)
            if isinstance(line, MultiAnswer):
                return answer("system", line.lines, sep="\n")
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
        if isinstance(line, MultiAnswer):
            return answer("system", line.lines, sep="\n")
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
    sol = congruence_roots(pieces)
    if sol is True or sol is False:
        return answer("check", [[("✓" if sol else "✗", "")]])
    v, roots, n = sol
    if not roots:
        return answer("solve", [[W("нет решений")]])
    return answer("solve", [_congruence_line(v, roots, n)])


def _congruence_line(v, roots, n):
    shown = roots[:12]
    vals = ", ".join(str(r) for r in shown) + (", …" if len(roots) > len(shown) else "")
    return [X(v), (" ≡ " + vals + " (mod %d)" % n, r" \equiv " + vals + r"\ (\operatorname{mod}\ %d)" % n)]


def congruence_roots(pieces):
    """'3x = 4 mod 7' split at '=' -> (x, [6], 7); True/False when there is no variable."""
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
    if not syms:
        return bool(diff.is_Integer and int(diff) % n == 0)
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
    return v, roots, n


def process_congruences(texts):
    """x = 2 mod 3; x = 3 mod 5 -> x ≡ 8 (mod 15) (Chinese remainder theorem). None if not such a system."""
    if not all(re.search(r"\bmod\b", t) for t in texts):
        return None
    from itertools import product
    from sympy.ntheory.modular import solve_congruence
    parts = []
    for t in texts:
        pieces, ops = split_relation(t)
        if ops != ["="]:
            return None
        sol = congruence_roots(pieces)
        if sol is True:
            continue
        if sol is False or not sol[1]:
            return [[W("нет решений")]]
        parts.append(sol)
    if not parts:
        return [[("✓", "")]]
    if len({str(p[0]) for p in parts}) != 1:
        raise CalcError("в системе сравнений должна быть одна переменная")
    found, modulus = set(), None
    for combo in product(*[[(r, p[2]) for r in p[1]] for p in parts][:8]):
        res = solve_congruence(*combo)
        if res:
            found.add(int(res[0]))
            modulus = int(res[1])
    if not found:
        return [[W("нет решений")]]
    return [_congruence_line(parts[0][0], sorted(found), modulus)]


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
    except (NotImplementedError, ValueError, TypeError) as ex:
        num = _ivp_numeric(lhs, rhs, ics)
        if num is None:
            raise CalcError("задача Коши не решается в формулах") from ex
        return num
    sols = sol if isinstance(sol, list) else [sol]
    return [[X(sp.Symbol("y")), O("="), X(s.rhs)] for s in sols[:MAX_ROOTS]]


# ---------------------------------------------------------------- what a line draws (graphs)

PARAM_NAMES = ("t", "φ", "θ", "phi", "theta")


def _rel_parts(t):
    pieces, ops = split_relation(t)
    if len(ops) != 1 or not pieces[0].strip() or not pieces[1].strip():
        return None
    return pieces[0].strip(), ops[0], pieces[1].strip()


def _expr_text(e):
    return sp.sstr(e)


def plot_spec(texts):
    """JSON text describing what the statements draw, or None.

    {"t": "y", "e": f(x)} | {"t": "param", "x": .., "y": .., "v": "t"} | {"t": "polar", "r": .., "v": "φ"}
    | {"t": "implicit", "e": F(x, y)} | {"t": "region", "c": ["y > x^2", ...]} | {"t": "surface", "e": f(x, y)}
    | {"t": "ode", "e": f(x, y), "ic": [x0, y0]}
    """
    spec = _plot_spec(texts)
    if spec is None:
        return None
    d = json.loads(spec)
    d["l"] = "; ".join(re.sub(r"\s+", " ", t.strip()) for t in texts)
    return json.dumps(d, ensure_ascii=False)


def _plot_spec(texts):
    x, y = sp.Symbol("x"), sp.Symbol("y")
    parts = [_rel_parts(t) for t in texts]
    if not parts or any(p is None for p in parts):
        return None
    # a differential equation y' = f(x, y) (with y(x0) = y0): direction field
    odes = [t for t in texts if not IC_RE.match(t) and re.search(r"\by\b", t) and re.search(r"'|′|diff\(", t)]
    if len(odes) == 1:
        lhs, op, rhs = _rel_parts(odes[0])
        if op == "=" and re.fullmatch(r"y\s*('|′)", lhs):
            try:
                f = parse(rhs)
            except CalcError:
                return None
            if f.free_symbols <= {x, y}:
                spec = {"t": "ode", "e": _expr_text(f)}
                ics = [IC_RE.match(t) for t in texts if IC_RE.match(t)]
                if len(ics) == 1 and not ics[0].group(1):
                    try:
                        spec["ic"] = [float(parse(ics[0].group(2))), float(parse(ics[0].group(3)))]
                    except (CalcError, TypeError):
                        pass
                return json.dumps(spec)
        return None
    try:
        sides = [(parse(a), op, parse(b)) for a, op, b in parts]
    except CalcError:
        return None
    if any(isinstance(a, sp.MatrixBase) or isinstance(b, sp.MatrixBase) for a, _, b in sides):
        return None
    syms = set()
    for a, _, b in sides:
        syms |= a.free_symbols | b.free_symbols
    names = {str(v) for v in syms}
    if len(sides) == 2 and all(op == "=" for _, op, _ in sides):
        lhs = {str(a): b for a, _, b in sides if isinstance(a, sp.Symbol)}
        if set(lhs) == {"x", "y"}:
            free = lhs["x"].free_symbols | lhs["y"].free_symbols
            if len(free) == 1 and str(next(iter(free))) in PARAM_NAMES:
                v = str(next(iter(free)))
                return json.dumps({"t": "param", "x": _expr_text(lhs["x"]), "y": _expr_text(lhs["y"]), "v": v})
    if len(sides) == 1:
        a, op, b = sides[0]
        if op == "=" and isinstance(a, sp.Symbol):
            if a.name == "r" and len(b.free_symbols) == 1 and str(next(iter(b.free_symbols))) in PARAM_NAMES:
                return json.dumps({"t": "polar", "r": _expr_text(b), "v": str(next(iter(b.free_symbols)))})
            if a.name == "z" and b.free_symbols <= {x, y} and b.free_symbols:
                return json.dumps({"t": "surface", "e": _expr_text(b)})
            if a.name in ("y", "f") and b.free_symbols <= {x}:
                return json.dumps({"t": "y", "e": _expr_text(b)})
        if op == "=" and names == {"x", "y"}:
            return json.dumps({"t": "implicit", "e": _expr_text(a - b)})
    if names <= {"x", "y"} and "y" in names and all(op in ("<", ">", "<=", ">=") for _, op, _ in sides):
        return json.dumps({"t": "region", "c": [_expr_text(a - b) + (" > 0" if op in (">", ">=") else " < 0")
                                                for a, op, b in sides]})
    return None


def drawn_answer(texts):
    """Statements that are only drawn (a parametric curve, a polar curve, a surface, a region); else None."""
    spec = plot_spec(texts)
    if spec is None:
        return None
    t = json.loads(spec)["t"]
    if t in ("param", "polar", "surface", "region"):
        return answer("function", plot=spec)
    return None


def _ivp_numeric(lhs, rhs, ics):
    """y' = f(x, y), y(x0) = y0 without a formula: Runge–Kutta, y at a few points (the graph shows the curve)."""
    x = sp.Symbol("x")
    yv = sp.Symbol("y")
    if len(ics) != 1:
        return None
    (cond, y0), = ics.items()
    if cond.has(sp.Derivative):
        return None
    x0 = cond.args[0]
    try:
        f_expr = sp.solve(sp.Eq(lhs, rhs), Y_OF_X.diff(x))
    except NotImplementedError:
        return None
    if len(f_expr) != 1:
        return None
    f = sp.lambdify((x, yv), f_expr[0].subs(Y_OF_X, yv), modules=["math"])
    xs, ys = float(x0), float(y0)
    h = 0.001
    marks = {round(float(x0) + d, 6): None for d in (0.5, 1, 2)}
    try:
        while xs < float(x0) + 2 - 1e-9:
            k1 = f(xs, ys)
            k2 = f(xs + h / 2, ys + h / 2 * k1)
            k3 = f(xs + h / 2, ys + h / 2 * k2)
            k4 = f(xs + h, ys + h * k3)
            ys += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xs += h
            key = round(xs, 6)
            if key in marks:
                marks[key] = ys
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    lines = [[W("формулы нет, численно (Рунге — Кутта):")]]
    for at, val in marks.items():
        if val is not None and math.isfinite(val):
            lines.append([("y(%s)" % fmt(sp.nsimplify(at)), r"y\left(%s\right)" % to_tex(sp.nsimplify(at))),
                          (" ≈ ", r" \approx ")] + [X(sp.Float(val, 8))])
    return lines if len(lines) > 1 else None


def process_system(texts):
    """Answer lines of a system written one equation per line."""
    rec = calc_higher.process_recurrence(texts)
    if rec is not None:
        return rec
    odes = calc_higher.process_ode_system(texts)
    if odes is not None:
        return odes
    ivp = process_ivp(texts)
    if ivp is not None:
        return ivp
    crt = process_congruences(texts)
    if crt is not None:
        return crt
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
    if ";" in text:  # a system from \begin{cases}
        return all(_valid(part) for part in text.split(";") if part.strip())
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
            if ";" in text:  # \begin{cases}: answered as a system by evaluate_item
                return text, answer("none")
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
            if ";" in text:  # a system written with \begin{cases}
                parts = statements(text)
                out["input_tex"] = system_tex(parts)
                return _drawn_or(parts, lambda: answer("system", process_system(parts), sep="\n"))
            if text.strip():
                res = _drawn_or([text], lambda: res)
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
            return _drawn_or(complete, lambda: answer("system", process_system(complete), sep="\n"))
    else:
        def run():
            text = item["text"]
            out["input"] = text
            parts = statements(text)
            if len(parts) > 1:
                out["input_tex"] = system_tex(parts)
                return _drawn_or(parts, lambda: answer("system", process_system(parts), sep="\n"))
            out["input_tex"] = text_to_tex(parts[0]) if parts else None
            if not parts:
                return answer("none")
            return _drawn_or(parts, lambda: process(parts[0], forced=item.get("forced", True)))
    res = _safe(run)
    out.update(res)
    return out


def _drawn_or(texts, compute):
    """A line that is only drawn gets its graph; any other line its answer plus, when it has one, its graph."""
    try:
        drawn = drawn_answer(texts)
    except Exception:  # noqa: BLE001 - a graph is a bonus
        drawn = None
    if drawn is not None:
        return drawn
    res = compute()
    try:
        spec = plot_spec(texts)
    except Exception:  # noqa: BLE001
        spec = None
    if spec is not None and isinstance(res, dict) and "error" not in res:
        res = dict(res, plot=spec)
    elif isinstance(res, dict) and res.get("plot") and not str(res["plot"]).startswith("{"):
        # "x^2 - 1 =" plots its value: store the expression itself (names defined above are substituted)
        try:
            e = parse(res["plot"])
            res = dict(res, plot=json.dumps({"t": "y", "e": _expr_text(e)}) if e.free_symbols <= {sp.Symbol("x")} else None)
        except CalcError:
            pass
    return res


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


# ---------------------------------------------------------------- worksheet: definitions across lines

_CTX = {}  # names defined higher up on the sheet: a = 3 (value), f(x) = x^2 + 1 (Lambda)
_LAST_CTX = {}  # everything defined on the whole sheet (used by the keyboard editor)
_SHEET_LOCK = threading.Lock()  # the sheet and the keyboard editor are evaluated from different threads
NAME_RE = r"[A-Za-zА-Яа-яα-ωΑ-Ω](?:_?\d+)?"
DEF_RE = re.compile(r"^\s*(%s)\s*:?=\s*(.+?)\s*$" % NAME_RE)
FUNC_DEF_RE = re.compile(r"^\s*(%s)\s*\(\s*([a-z](?:\s*,\s*[a-z])*)\s*\)\s*:?=\s*(.+?)\s*$" % NAME_RE)
NOT_DEFINABLE = {"x", "e", "i", "E", "I", "π"}


def _ctx_primes(text):
    """f'(2) with a user function f -> f__d1(2)."""
    if not _CTX:
        return text

    def sub(m):
        name, n = m.group(1), len(m.group(2))
        return name + "__d%d(" % n if isinstance(_CTX.get(name), sp.Lambda) else m.group(0)
    return re.sub(r"(?<![A-Za-z_])([A-Za-z])('+|′+)\s*\(", sub, text)


def definition_of(text):
    """'a = 3' -> ('a', 3); 'f(x) = x^2' -> ('f', Lambda). None when the line is not a definition."""
    if not text or text.rstrip().endswith("=") or ";" in text:
        return None
    t = text.replace("≔", ":=")
    m = FUNC_DEF_RE.match(t)
    if m and len(split_relation(t)[1]) == 1:
        name, args, rhs = m.group(1), [a.strip() for a in m.group(2).split(",")], m.group(3)
        if (name in LOCALS and not isinstance(LOCALS[name], sp.Symbol)) or name in NOT_DEFINABLE or name in args:
            return None
        syms = [sp.Symbol(a) for a in args]
        saved = _CTX.pop(name, None)
        try:
            body = parse(rhs)
        finally:
            if saved is not None:
                _CTX[name] = saved
        return name, sp.Lambda(tuple(syms), body)
    m = DEF_RE.match(t)
    if not m or len(split_relation(t.replace(":=", "="))[1]) != 1:
        return None
    name, rhs = m.group(1), m.group(2)
    if name in NOT_DEFINABLE or (name in LOCALS and not isinstance(LOCALS[name], sp.Symbol)):
        return None
    saved = _CTX.pop(name, None)
    try:
        value = parse(rhs)
    except CalcError:
        return None
    finally:
        if saved is not None:
            _CTX[name] = saved
    if isinstance(value, (sp.Basic, sp.MatrixBase)) and sp.Symbol(name) not in getattr(value, "free_symbols", set()):
        return name, value
    return None


def _define(name, value):
    _CTX[name] = value
    if isinstance(value, sp.Lambda):
        for k in (1, 2, 3):
            var = value.variables[0]
            _CTX[name + "__d%d" % k] = sp.Lambda(value.variables, sp.diff(value.expr, var, k))


def _ctx_key():
    return "|".join("%s=%s" % (k, sp.srepr(v)) for k, v in sorted(_CTX.items()) if "__d" not in k)


def _define_result(res, name, value):
    """A definition shows nothing on the sheet (as in MathCad); the panel shows its value."""
    out = {k: v for k, v in res.items() if k in ("id", "input", "input_tex", "plot")}
    out["kind"] = "define"
    out["input_tex"] = text_to_tex(res.get("input", "").replace(":=", "=")) or res.get("input_tex")
    if isinstance(value, sp.Lambda):
        args = ", ".join(str(a) for a in value.variables)
        out["preview"] = "%s(%s) = %s" % (name, args, fmt(value.expr))
        try:
            out["preview_tex"] = r"%s\left(%s\right) = %s" % (name, args, sp.latex(value.expr))
        except Exception:  # noqa: BLE001
            pass
        if value.variables == (sp.Symbol("x"),) and not value.expr.free_symbols - {sp.Symbol("x")}:
            out["plot"] = json.dumps({"t": "y", "e": sp.sstr(value.expr)})
            out["kind"] = "function"
        elif len(value.variables) == 2 and not value.expr.free_symbols - set(value.variables):
            a, b = value.variables
            e = value.expr.subs({a: sp.Symbol("x"), b: sp.Symbol("y")}, simultaneous=True)
            out["plot"] = json.dumps({"t": "surface", "e": sp.sstr(e)})
            out["kind"] = "function"
    else:
        try:
            line = value_line(value, False) if not isinstance(value, sp.MatrixBase) else [X(value)]
            out["preview"] = name + " = " + line_text(line)
            out["preview_tex"] = name + " = " + line_tex(line)
        except Exception:  # noqa: BLE001
            out["preview"] = name + " = " + fmt(value)
        if res.get("kind") == "function":
            out["kind"] = "function"
    return out


def evaluate_item_cached(item, limit):
    """Results are cached by the item's content, so a slow line is not recomputed on every stroke."""
    key = json.dumps({k: v for k, v in item.items() if k not in ("id", "top")}, sort_keys=True,
                     ensure_ascii=False) + "|%g" % limit + "|" + _ctx_key()
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
    # the sheet is read from top to bottom: a = 3 above, a^2 = below
    order = sorted(range(len(items)), key=lambda k: (items[k].get("top", 0), k))
    results = [None] * len(items)
    with _SHEET_LOCK:
        return _evaluate_sheet(items, order, results, limit)


def _evaluate_sheet(items, order, results, limit):
    _CTX.clear()
    try:
        for k in order:
            res = evaluate_item_cached(items[k], float(limit))
            if items[k].get("type") != "system" and not res.get("timeout"):
                try:
                    d = run_limited(lambda: definition_of(res.get("input", "")), 3) if limit else definition_of(res.get("input", ""))
                except Exception:  # noqa: BLE001
                    d = None
                if isinstance(d, tuple):
                    _define(*d)
                    res = _define_result(res, *d)
            results[k] = res
        _LAST_CTX.clear()
        _LAST_CTX.update(_CTX)
    finally:
        _CTX.clear()
    return json.dumps(results, ensure_ascii=False)


def evaluate_text_in_sheet(text):
    """The keyboard editor: one line, with everything defined on the sheet."""
    with _SHEET_LOCK:
        _CTX.update(_LAST_CTX)
        try:
            return _evaluate_kb(text)
        finally:
            _CTX.clear()


def _evaluate_kb(text):
    res = evaluate_item_cached({"id": "kb", "type": "text", "text": text, "forced": True}, 10.0)
    d = definition_of(res.get("input", "")) if not res.get("timeout") else None
    return json.dumps([_define_result(res, *d) if d else res], ensure_ascii=False)


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


# higher mathematics lives in its own module; it needs the helpers above
import calc_higher  # noqa: E402

LOCALS.update(calc_higher.register())
