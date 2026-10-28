"""Tests for the SymPy engine. Run: python tests/test_engine.py  (or pytest)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "python"))

import calc_engine as ce  # noqa: E402

T2T = [
    ("\\frac { 1 } { 2 } + \\frac { 1 } { 3 } =", "((1)/(2)) + ((1)/(3)) ="),
    ("x ^ { 2 } - 4 = 0", "x^(2) - 4 = 0"),
    ("\\sqrt { 1 6 } =", "sqrt(16) ="),
    ("\\sin ^ { 2 } x + \\cos ^ { 2 } x =", "sin(x)^(2) + cos(x)^(2) ="),
    ("\\int _ { 0 } ^ { 1 } x ^ { 2 } d x =", "integrate(x^(2), (x, 0, 1)) ="),
    ("\\lim _ { x \\rightarrow 0 } \\frac { \\sin x } { x } =", "limit(((sin(x))/(x)), x, 0) ="),
    ("\\frac { d } { d x } x ^ { 3 } =", "diff(x^(3), x) ="),
    ("\\sum _ { i = 1 } ^ { n } i =", "summation(i, (i, 1, n)) ="),
    ("| - 3 | + 2 =", "Abs( - 3) + 2 ="),
    ("l n e =", "log(e) ="),
    ("\\log _ { 2 } 8 =", "log(8, 2) ="),
    ("x _ { 1 } + x _ { 2 } = 3", "x_1 + x_2 = 3"),
    ("1 2 . 5 \\times 2 =", "12.5 * 2 ="),
    ("\\sqrt [ 3 ] { 2 7 } =", "root(27, 3) ="),
    ("\\lim _ { n \\rightarrow \\infty } ( 1 + \\frac { 1 } { n } ) ^ { n } =", "limit((1 + ((1)/(n)))^(n), n, oo) ="),
]

PROCESS = [
    ("2 + 2 =", "4"),
    ("((1)/(2)) + ((1)/(3)) =", "5/6 ≈ 0.83333333"),
    ("1/4 =", "1/4 = 0.25"),
    ("sqrt(16) =", "4"),
    ("sqrt(2) =", "√2 ≈ 1.4142136"),
    ("sin(x)^(2) + cos(x)^(2) =", "1"),
    ("integrate(x^(2), (x, 0, 1)) =", "1/3 ≈ 0.33333333"),
    ("limit(((sin(x))/(x)), x, 0) =", "1"),
    ("diff(x^(3), x) =", "3x²"),
    ("2x + 3 = 7", "x = 2"),
    ("x^2 - 4 = 0", "x = ±2"),
    ("x^2 + 1 = 0", "x = ±i"),
    ("x^3 - 6x^2 + 11x - 6 = 0", "x₁ = 1; x₂ = 2; x₃ = 3"),
    ("x^5 - x - 1 = 0", "x ≈ 1.167304"),
    ("x^16 + y^z = 0", "x = ±¹⁶√(−y^z)"),
    ("2 + 2 = 4", "✓"),
    ("2 + 2 = 5", "✗  (= 4)"),
    ("12.5 * 2 =", "25"),
    ("0.1 + 0.2 =", "0.3"),
    ("(x+1)^2 =", "x² + 2x + 1"),
    ("x^2 - 1 =", "(x + 1)(x − 1)"),
    ("2x + 3 =", None),
    ("5 =", None),
    ("x > 2", None),  # inequality answer checked separately
    ("2x - 4 > 0", "x > 2"),
    ("x^2 < 4", "−2 < x < 2"),
    ("summation(i, (i, 1, 100)) =", "5050"),
    ("5! =", "120"),
    ("log(8, 2) =", "3"),
    ("log(e) =", "1"),
    ("2pi =", "2π ≈ 6.2831853"),
    ("integrate(1/x, x) =", "ln x"),
]


def test_tokens_to_text():
    for toks, expected in T2T:
        got = ce.tokens_to_text(toks)
        assert got == expected, (toks, got, expected)


def test_process():
    for text, expected in PROCESS:
        res = ce.process(text)
        if text == "x > 2":
            continue
        assert res.get("answer") == expected, (text, res, expected)


def test_function_definition():
    res = ce.process("y = x^2 - 1")
    assert res["kind"] == "function" and res["plot"] == "x^2 - 1", res


def test_system():
    out = [ce.line_text(l) for l in ce.process_system(["x + y = 3", "x - y = 1"])]
    assert out == ["x = 2, y = 1"], out
    out = [ce.line_text(l) for l in ce.process_system(["x^2 + y^2 = 25", "x - y = 1"])]
    assert out == ["x = −3, y = −4", "x = 4, y = 3"], out


def test_document_json():
    items = [
        {"id": "a", "type": "line", "tokens": "2 + 2 =".split()},
        {"id": "b", "type": "system", "lines": ["x + y = 3".split(), "x - y = 1".split()]},
        {"id": "c", "type": "text", "text": "2x+1=5"},
        {"id": "d", "type": "text", "text": "x+y=3, x-y=1"},
        {"id": "e", "type": "line", "tokens": "\\exists x".split()},
        {"id": "f", "type": "text", "text": "det([[1,2],[3,4]])"},
        {"id": "g", "type": "text", "text": "√2·√8"},
        {"id": "h", "type": "system", "lines": ["x + y = 3", "x - y = 1".split()]},
    ]
    out = {o["id"]: o for o in json.loads(ce.evaluate_document(json.dumps(items)))}
    assert out["a"]["answer"] == "4", out["a"]
    assert out["b"]["answer"] == "x = 2, y = 1", out["b"]
    assert out["c"]["answer"] == "x = 2", out["c"]
    assert out["d"]["answer"] == "x = 2, y = 1", out["d"]
    assert "error" in out["e"], out["e"]
    assert out["f"]["answer"] == "−2", out["f"]
    assert out["g"]["answer"] == "4", out["g"]
    assert out["h"]["answer"] == "x = 2, y = 1", out["h"]


def test_tex():
    items = [
        {"id": "a", "type": "line", "tokens": "\\frac { 1 } { 2 } + \\frac { 1 } { 3 } =".split()},
        {"id": "b", "type": "line", "tokens": "x ^ { 2 } - 2 = 0".split()},
        {"id": "c", "type": "system", "lines": ["x + y = 3".split(), "x - y = 1".split()]},
        {"id": "d", "type": "line", "tokens": "\\sqrt [ 3 ] { 2 7 } =".split()},
        {"id": "e", "type": "line", "tokens": "2 + 2 = 5".split()},
        {"id": "f", "type": "text", "text": "x^2 > 4"},
    ]
    out = {o["id"]: o for o in json.loads(ce.evaluate_document(json.dumps(items)))}
    assert out["a"]["input_tex"] == "\\frac{1}{2} + \\frac{1}{3} =", out["a"]
    assert out["a"]["tex"] == ["\\frac{5}{6} \\approx 0.83333333"], out["a"]
    assert out["b"]["tex"] == ["x = \\pm \\sqrt{2} \\approx 1.4142136"], out["b"]
    assert out["c"]["input_tex"] == "\\begin{cases}x + y = 3\\\\x - y = 1\\end{cases}", out["c"]
    assert out["c"]["tex"] == ["x = 2,\\quad y = 1"], out["c"]
    assert out["d"]["input_tex"] == "\\sqrt[3]{27} =", out["d"]
    assert out["e"]["tex"] is None and out["e"]["answer"].startswith("✗"), out["e"]
    assert out["f"]["tex"] == ["x < -2\\ \\text{или}\\ x > 2"], out["f"]


def test_candidates():
    def line(best, *alts):
        return {"id": "l", "type": "line", "tokens": best.split(), "alts": [a.split() for a in alts]}

    def run(item):
        return json.loads(ce.evaluate_document(json.dumps([item])))[0]

    # × read as the letter x between numbers
    r = run(line("2 x 3 ="))
    assert r["answer"] == "6", r
    # 4 read as y in a value request
    r = run(line("2 + y ="))
    assert r["answer"] == "6", r
    # expressions with powers keep their letters (factorisation was asked for)
    r = run(line("y ^ { 2 } - 1 ="))
    assert r["answer"] == "(y + 1)(y − 1)", r
    # but a real equation keeps its letters
    r = run(line("2 y = 8"))
    assert r["answer"] == "y = 4", r
    # a × with nothing on one side is the letter x
    r = run(line("2 \\times + 3 = 7"))
    assert r["answer"] == "x = 2", r
    # unparsable best reading: the next candidate is used
    r = run(line("2 + ( =", "2 + 2 ="))
    assert r["answer"] == "4" and r["input"] == "2 + 2 =", r
    # system lines choose readings that are equations
    r = run({"id": "s", "type": "system", "lines": [
        {"tokens": "x + y 5".split(), "alts": ["x + y = 5".split()]}, "x - y = 1".split()]})
    assert r["answer"] == "x = 3, y = 2", r


def test_preview_limit_big():
    def run(item, limit=10.0):
        return json.loads(ce.evaluate_document(json.dumps([item]), limit))[0]

    # no "=": no answer on the sheet, but a preview for the panel
    r = run({"id": "p", "type": "line", "tokens": "\\frac { 1 } { 2 } + \\frac { 1 } { 3 }".split()})
    assert r["answer"] is None and r["preview"] == "5/6 ≈ 0.83333333", r
    assert r["preview_tex"] == "\\frac{5}{6} \\approx 0.83333333", r
    # huge numbers are shown as m·10ⁿ
    r = run({"id": "b", "type": "line", "tokens": "2 ^ { 2 ^ { 2 ^ { 2 ^ { 2 } } } } =".split()})
    assert r["answer"] == "2.0035299·10¹⁹⁷²⁸", r
    assert r["tex"] == ["2.0035299 \\cdot 10^{19728}"], r
    # one more storey: estimated, never computed
    r = run({"id": "b2", "type": "line", "tokens": "2 ^ { 2 ^ { 2 ^ { 2 ^ { 2 ^ { 2 } } } } } =".split()})
    assert r["answer"] == "≈ 10^(6.03·10¹⁹⁷²⁷)", r
    r = run({"id": "b3", "type": "text", "text": "100000! ="})
    assert r["answer"].startswith("≈ 10^(456573.4"), r
    # a computation that takes too long is stopped
    import time
    t0 = time.time()
    r = run({"id": "t", "type": "text", "text": "factorint(2^251 - 1)"}, 1.0)
    assert r.get("timeout") and time.time() - t0 < 4, (r, time.time() - t0)
    # and the result is cached (no second wait)
    t0 = time.time()
    r = run({"id": "t2", "type": "text", "text": "factorint(2^251 - 1)"}, 1.0)
    assert r.get("timeout") and r["id"] == "t2" and time.time() - t0 < 0.5, r


def test_letters_primes_exact():
    def run(toks):
        return json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "line", "tokens": toks.split()}])))[0]

    # sin written letter by letter is the function, not s·i·n
    assert run("\\lim _ { x \\rightarrow 0 } \\frac { s i n ( x ) } { x } =")["answer"] == "1"
    assert run("c o s ( 0 ) =")["answer"] == "1"
    # derivatives with a prime
    assert run("( x ^ { 3 } ) ^ { \\prime } =")["answer"] == "3x²"
    assert run("x ^ { 3 } \\prime =")["answer"] == "3x²"
    assert run("( s i n x ) ^ { \\prime \\prime } =")["answer"] == "−sin(x)"
    # the full value behind a shortened answer
    r = run("2 ^ { 1 0 0 0 } =")
    assert r["answer"] == "1.0715086·10³⁰¹" and len(r["exact"]) == 302 and r["exact"].startswith("10715086071862673"), r
    assert run("\\sqrt { 2 } =")["exact"] == "1.4142135623730950488016887242096980785696718753769"
    assert "exact" not in run("2 + 2 =")


def test_numeric():
    # no formula: real roots found numerically
    assert ce.process("cos(x) = x")["answer"] == "x ≈ 0.7390851332"
    assert ce.process("2^x = x^2 + 1")["answer"] == "x₁ ≈ 0; x₂ ≈ 1; x₃ ≈ 4.257461914"
    out = [ce.line_text(l) for l in ce.process_system(["x^3 + z = y", "x^y + z = 1", "x^x = 11"])]
    assert out == ["x ≈ 2.5556046, y ≈ 2.8731345, z ≈ −13.817813"], out


def test_mod_and_ode():
    def run(toks):
        return json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "line", "tokens": toks.split()}])))[0]

    assert run("3 x = 4 m o d 7")["answer"] == "x ≡ 6 (mod 7)"
    assert run("x ^ { 2 } = 1 m o d 8")["answer"] == "x ≡ 1, 3, 5, 7 (mod 8)"
    assert run("1 7 m o d 5 =")["answer"] == "2"
    assert run("1 7 m o d 5")["preview"] == "2"
    # y' = 2x is a differential equation in y(x), not 0 = 2x
    assert run("y \\prime = 2 x")["answer"] == "y = C₁ + x²"
    assert run("y ^ { \\prime \\prime } + y = 0")["answer"] == "y = C₁sin(x) + C₂cos(x)"
    assert run("\\frac { d y } { d x } = y")["answer"] == "y = C₁eˣ"


def test_keyboard_forms():
    assert ce.editable_text("integrate(x^(2), (x, 0, 1)) =") == "int(x^2, x, 0, 1) ="
    assert ce.editable_text("((1)/(2)) + ((1)/(3)) =") == "(1/2) + (1/3) ="
    assert ce.process("int(x^2, x, 0, 1) =")["answer"] == "1/3 ≈ 0.33333333"
    assert ce.process("lim(sin(x)/x, x, 0) =")["answer"] == "1"
    assert ce.process("sum(1/n^2, n, 1, oo) =")["answer"] == "π²/6 ≈ 1.6449341"
    assert ce.process("(x^3)' =")["answer"] == "3x²"
    assert ce.process("y' = 2x")["answer"] == "y = C₁ + x²"
    assert ce.display_tex("x+y=3; x-y=1") == "\\begin{cases}x + y = 3\\\\x - y = 1\\end{cases}"


def test_phase1():
    def ans(t):
        r = json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "text", "text": t, "forced": True}])))[0]
        assert "answer" in r and r["answer"], (t, r)
        return r["answer"]

    cases = {
        "(1+2i)*(3-i)=": "5 + 5i", "abs(3+4i)=": "5", "x^2+2x+5=0": "x₁ = −1 − 2i; x₂ = −1 + 2i",
        "sum(i, i, 1, 10)=": "55", "sin(30°)=": "1/2 = 0.5", "200+15%=": "230", "15% * 200=": "30",
        "17 % 5=": "2", "factorint(360)=": "2³·3²·5", "isprime(91)=": "составное, делится на 7",
        "C(5,2)=": "10", "A(5,2)=": "20", "P(5)=": "120", "100 in binary=": "1100100₂", "hex(255)=": "FF₁₆",
        "0xFF=": "255", "round(2.567, 2)=": "2.57", "apart(1/(x^2-1))=": "−1/(2x + 2) + 1/(2x − 2)",
        "mean(1,2,3,4)=": "5/2 = 2.5", "median(1,3,2)=": "2", "std(2,4,4,4,5,5,7,9)=": "2",
        "eigenvals([[2,1],[1,2]])=": "λ = 1; λ = 3", "rank([[1,2],[2,4]])=": "1", "dot([1,2,3],[4,5,6])=": "32",
        "max(x^2-4x, x, 0, 5)=": "5", "extrema(x^3-3x)=": "x = −1: максимум f = 2; x = 1: минимум f = −2",
        "|x-2|=5": "x₁ = −3; x₂ = 7", "tg(x)=1": "x = πk + π/4, k ∈ ℤ", "sin(x)=2": "нет действительных решений",
        "y'=y; y(0)=1": "y = eˣ", "y''+y=0; y(0)=0; y'(0)=1": "y = sin(x)",
        "x = 2 mod 3; x = 3 mod 5": "x ≡ 8 (mod 15)", "x = 2 mod 3; x = 3 mod 5; x = 2 mod 7": "x ≡ 23 (mod 105)",
        "lsolve([[2,1],[1,3]], [3,5])=": "[4/5  7/5]",
        "study(1/x)=": "область определения: (−∞; 0) ∪ (0; ∞)\nвертикальные асимптоты: x = 0\nасимптота при x → ±∞: y = 0",
    }
    for t, expected in cases.items():
        got = ans(t)
        assert got == expected, (t, got, expected)


def test_worksheet():
    def sheet(*lines):
        items = [{"id": str(k), "type": "text", "text": t, "forced": True, "top": k} for k, t in enumerate(lines)]
        # given out of order on purpose: the sheet is read from top to bottom
        res = json.loads(ce.evaluate_document(json.dumps(items[::-1])))[::-1]
        return [r.get("answer") or r.get("preview") or r.get("error") for r in res]

    assert sheet("a = 3", "a^2 + 1 =") == ["a = 3", "10"]
    assert sheet("a = 2", "b = a + 1", "a*b =") == ["a = 2", "b = 3", "6"]
    assert sheet("f(x) = x^2 + 1", "f(3) =", "f'(2) =", "int(f(x), x, 0, 3) =") == ["f(x) = x² + 1", "10", "4", "12"]
    assert sheet("k = 5", "x^2 = k")[1] == "x = ±√5 ≈ 2.236068"
    assert sheet("a^2 =", "a = 3") == ["a²", "a = 3"]  # only lines above count
    assert sheet("x = 3", "x^2 =")[1] == "x²"  # x stays the unknown


def test_units():
    def ans(t):
        r = json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "text", "text": t, "forced": True}])))[0]
        return r.get("answer") or r.get("error")

    assert ans("30 km/h in m/s =") == "25/3 m/s ≈ 8.3333333 m/s"
    assert ans("5 м в см") == "500 см"
    assert ans("1 mi in km") == "1.609344 km"
    assert ans("20 °C in F") == "68 °F"
    assert ans("1 atm in mmHg") == "760 mmHg"
    assert ans("5 kg in m").startswith("единицы несовместимы")
    assert ans("100 in binary =") == "1100100₂"


def test_graphs():
    import calc_plot as cp

    def spec(t):
        r = json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "text", "text": t, "forced": True}])))[0]
        return r["kind"], json.loads(r["plot"])["t"] if r.get("plot") else None

    assert spec("y = x^2 - 1") == ("function", "y")
    assert spec("x = cos(t); y = sin(t)") == ("function", "param")
    assert spec("r = 1 + cos(φ)") == ("function", "polar")
    assert spec("x^2 + y^2 = 1") == ("solve", "implicit")
    assert spec("y > x^2; x + y < 3") == ("function", "region")
    assert spec("z = x^2 - y^2") == ("function", "surface")
    assert spec("y' = x - y; y(0) = 1") == ("system", "ode")
    assert spec("2x + 1 = 5") == ("solve", None)
    specs = json.dumps([json.dumps({"t": "y", "e": "x**2 - 1"}), json.dumps({"t": "y", "e": "x + 1"})])
    sc = json.loads(cp.scene(specs, -5, 5, -5, 5, 800, 600))
    labels = {p["label"] for p in sc["points"]}
    assert {"(−1; 0)", "(1; 0)", "(0; −1)", "(2; 3)"} <= labels, labels
    su = json.loads(cp.surface(json.dumps({"t": "surface", "e": "x*y"}), 0, 0, 2, 10))
    assert len(su["z"]) == 11 and su["z"][0][0] == 4.0


def test_higher():
    def ans(t):
        r = json.loads(ce.evaluate_document(json.dumps([{"id": "a", "type": "text", "text": t, "forced": True}]), 30))[0]
        assert r.get("answer"), (t, r)
        return r["answer"]

    cases = {
        "grad(x^2*y + z)=": "[2xy  x²  1]", "div([x^2, y^2, z])=": "2x + 2y + 1", "curl([-y, x, 0])=": "[0  0  2]",
        "hessian(x^3+y^2)=": "[6x  0; 0  2]", "int2(x*y, x, 0, 1, y, 0, x)=": "1/8 = 0.125",
        "laplace(sin(t))=": "1/(s² + 1)", "ilaplace(1/(s^2+1))=": "sin(t)", "ztrans(2^n)=": "z/(z − 2)",
        "iztrans(z/(z-2))=": "2ⁿ", "a(n+1) = 2a(n) + 1; a(0) = 1": "a(n) = 2^(n + 1) − 1",
        "x' = y; y' = -x; x(0)=1; y(0)=0": "x = cos(t)\ny = −sin(t)",
        "extrema(x^3 - 3x + y^2)=": "(−1; 0): седло f = 2\n(1; 0): минимум f = −2",
        "lagrange(x+y, x^2+y^2-1)=": "(−√2/2; −√2/2): минимум f = −√2\n(√2/2; √2/2): максимум f = √2",
        "invnorm(0.975)=": "1.959963985", "binomcdf(10, 1/2, 3)=": "11/64 = 0.171875",
        "polyfit([0,1,2],[1,2,5],2)=": "x² + 1", "simplify_bool((x & y) | (x & ~y))=": "x",
        "union([1,2],[2,3])=": "{1, 2, 3}", "residue(1/(z^2+1), z, i)=": "−i/2",
    }
    for t, expected in cases.items():
        got = ans(t)
        assert got == expected, (t, got, expected)
    assert ans("y' = sin(x*y); y(0)=1").startswith("формулы нет, численно")


def test_plot():
    data = json.loads(ce.plot_points("y = 1/x", -1.0, 1.0, 5))
    assert data["ys"][2] is None and data["ys"][0] == -1.0, data
    data = json.loads(ce.plot_points("sqrt(x) =", -1.0, 1.0, 3))
    assert data["ys"] == [None, 0.0, 1.0], data


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as ex:
                failed += 1
                print("FAIL", name, ex)
    sys.exit(1 if failed else 0)
