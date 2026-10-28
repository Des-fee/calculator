package io.github.desfee.calculator

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject

/** Bridge to the Python/SymPy engine (app/src/main/python/calc_engine.py). */
class Engine(context: Context) {
    private val module: PyObject

    init {
        if (!Python.isStarted()) Python.start(AndroidPlatform(context.applicationContext))
        module = Python.getInstance().getModule("calc_engine")
        // warm up SymPy so that the first real answer is fast
        module.callAttr("evaluate_document", "[{\"id\":\"warm\",\"type\":\"text\",\"text\":\"x^2-1=0\"}]")
    }

    data class Result(
        val id: String,
        val input: String,
        val kind: String,
        val answer: String?,
        val plot: String?,
        val error: String?,
        /** Typeset input (LaTeX), when it could be produced. */
        val inputTex: String? = null,
        /** Typeset answer, one LaTeX string per answer line. */
        val tex: List<String>? = null,
        /** Value of a line without "=", shown only in the panel. */
        val preview: String? = null,
        val previewTex: String? = null,
        val timedOut: Boolean = false,
        /** Every digit of a shortened answer (2^1000, √2 to 50 digits). */
        val exact: String? = null,
    )

    /** Seconds one item may take (0 = no limit). */
    @Volatile
    var limitSeconds = 10.0

    fun evaluate(items: JSONArray): Map<String, Result> {
        val out = JSONArray(module.callAttr("evaluate_document", items.toString(), limitSeconds).toString())
        val map = HashMap<String, Result>()
        for (i in 0 until out.length()) {
            val r = parse(out.getJSONObject(i))
            map[r.id] = r
        }
        return map
    }

    private fun parse(o: JSONObject): Result =
            Result(
                id = o.optString("id"),
                input = o.optString("input", ""),
                kind = o.optString("kind", "none"),
                answer = o.optStringOrNull("answer"),
                plot = o.optStringOrNull("plot"),
                error = o.optStringOrNull("error"),
                inputTex = o.optStringOrNull("input_tex"),
                tex = o.optJSONArray("tex")?.let { a -> List(a.length()) { a.getString(it) } },
                preview = o.optStringOrNull("preview"),
                previewTex = o.optStringOrNull("preview_tex"),
                timedOut = o.optBoolean("timeout", false),
                exact = o.optStringOrNull("exact"),
            )

    fun evaluateText(text: String): Result {
        val o = JSONArray(module.callAttr("evaluate_text_in_sheet", text).toString()).getJSONObject(0)
        return parse(o)
    }

    /** Typeset preview of typed text, or null when it does not parse (yet). */
    fun texOf(text: String): String? = module.callAttr("display_tex", text)?.toString()?.takeIf { it != "None" }

    /** The engine's linear text in the keyboard form shown for corrections (x^2, int(f, x, 0, 1)). */
    fun editableText(text: String): String = module.callAttr("editable_text", text).toString()

    /** LaTeX label of a corrected line for fine-tuning, or null. */
    fun trainingLabel(text: String): String? =
        module.callAttr("training_label", text)?.toString()?.takeIf { it != "None" }

    private val plots: PyObject by lazy { Python.getInstance().getModule("calc_plot") }

    class Layer(
        val type: String,
        val label: String,
        val paths: List<DoubleArray>,
        val thin: Boolean,
        val nx: Int = 0,
        val ny: Int = 0,
        val mask: String? = null,
        val solution: DoubleArray? = null,
        val ic: DoubleArray? = null,
    )

    class Point(val x: Double, val y: Double, val label: String, val kind: String)

    class Scene(val layers: List<Layer>, val points: List<Point>)

    private fun JSONArray.doubles() = DoubleArray(length()) { optDouble(it) }

    /** Everything [specs] draw in the given window (page units), sampled for a [w]×[h] px view. */
    fun scene(specs: List<String>, xmin: Double, xmax: Double, ymin: Double, ymax: Double, w: Int, h: Int): Scene {
        val o = JSONObject(plots.callAttr("scene", JSONArray(specs).toString(), xmin, xmax, ymin, ymax, w, h).toString())
        if (o.has("error")) throw IllegalArgumentException(o.getString("error"))
        val layers = o.getJSONArray("layers").let { a ->
            List(a.length()) { i ->
                val l = a.getJSONObject(i)
                val paths = l.optJSONArray("paths")?.let { p -> List(p.length()) { p.getJSONArray(it).doubles() } } ?: emptyList()
                Layer(
                    l.optString("type"), l.optString("label"), paths, l.optBoolean("thin", false),
                    l.optInt("nx"), l.optInt("ny"), l.optStringOrNull("mask"),
                    l.optJSONArray("solution")?.doubles(), l.optJSONArray("ic")?.doubles(),
                )
            }
        }
        val points = o.getJSONArray("points").let { a ->
            List(a.length()) { i ->
                val p = a.getJSONObject(i)
                Point(p.getDouble("x"), p.getDouble("y"), p.optString("label"), p.optString("kind"))
            }
        }
        return Scene(layers, points)
    }

    /** A good first window [xmin, xmax, ymin, ymax] for these graphs in a [w]×[h] view. */
    fun initialView(specs: List<String>, w: Int, h: Int): DoubleArray =
        JSONArray(plots.callAttr("initial_view", JSONArray(specs).toString(), w, h).toString()).doubles()

    class Surface(val z: Array<DoubleArray>, val zmin: Double, val zmax: Double, val label: String)

    /** z = f(x, y) on the square [cx ± r] × [cy ± r], (n + 1)² points; NaN where undefined. */
    fun surface(spec: String, cx: Double, cy: Double, r: Double, n: Int): Surface {
        val o = JSONObject(plots.callAttr("surface", spec, cx, cy, r, n).toString())
        if (o.has("error")) throw IllegalArgumentException(o.getString("error"))
        val rows = o.getJSONArray("z")
        val z = Array(rows.length()) { j ->
            val row = rows.getJSONArray(j)
            DoubleArray(row.length()) { if (row.isNull(it)) Double.NaN else row.getDouble(it) }
        }
        return Surface(z, o.getDouble("zmin"), o.getDouble("zmax"), o.optString("label"))
    }

    class Plot(val xs: DoubleArray, val ys: Array<Double?>, val label: String)

    fun plot(expr: String, xmin: Double, xmax: Double, n: Int): Plot {
        val o = JSONObject(module.callAttr("plot_points", expr, xmin, xmax, n).toString())
        if (o.has("error")) throw IllegalArgumentException(o.getString("error"))
        val xs = o.getJSONArray("xs")
        val ys = o.getJSONArray("ys")
        return Plot(
            DoubleArray(xs.length()) { xs.getDouble(it) },
            Array(ys.length()) { if (ys.isNull(it)) null else ys.getDouble(it) },
            o.optString("label", ""),
        )
    }
}

private fun JSONObject.optStringOrNull(name: String): String? =
    if (!has(name) || isNull(name)) null else getString(name)
