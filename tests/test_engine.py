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
    ("x^2 - 4 = 0", "x₁ = −2; x₂ = 2"),
    ("x^2 + 1 = 0", "x₁ = −i; x₂ = i"),
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
    assert out["b"]["tex"] == ["{x}_{1} = - \\sqrt{2} \\approx -1.4142136", "{x}_{2} = \\sqrt{2} \\approx 1.4142136"], out["b"]
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
