package io.github.desfee.calculator

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Typing and correcting formulas: a live typeset preview of what is typed, the answer under it,
 * and a math keypad with pages (digits, functions, letters, calculus). The system keyboard is
 * available with "ABC".
 *
 * Text uses the keyboard forms the engine understands: x^2, sqrt(2), 1/2, int(x^2, x, 0, 1),
 * lim(sin(x)/x, x, 0), sum(1/n^2, n, 1, oo), diff(x^3, x), y', 17 mod 5, systems separated by ";".
 */
@SuppressLint("ViewConstructor")
class MathEditor(context: Context, private val colors: Palette, private val listener: Listener) : LinearLayout(context) {
    // GradientDrawable has its own `colors`, so drawables use this name for the palette
    private val pal = colors

    interface Listener {
        /** The text changed: compute its preview and answer (debounce in the caller). */
        fun onEditorText(text: String)
        fun onEditorApply(text: String)
        fun onEditorReset()
        fun onEditorClose()
        fun onEditorCopy()
        fun onEditorGraph()
    }

    enum class Mode { CALC, FIX }

    private class Key(val label: String, val insert: String? = null, val action: String? = null, val wide: Float = 1f)

    var mode = Mode.CALC
        private set
    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = (v * density).toInt()

    private val title = TextView(context)
    private val applyButton: TextView
    private val resetButton: TextView
    val input: EditText
    private val preview = MathView(context)
    val result = MathView(context)
    private val copyChip: TextView
    private val graphChip: TextView
    private val pages = LinkedHashMap<String, LinearLayout>()
    private val tabs = LinkedHashMap<String, TextView>()
    private var page = "123"
    private var systemKeyboard = false

    init {
        orientation = VERTICAL
        setBackgroundColor(colors.panel)
        setPadding(dp(12f), dp(6f), dp(12f), dp(8f))

        // header: what is being edited and the actions
        val header = LinearLayout(context).apply { orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        title.apply { textSize = 14f; typeface = Typeface.DEFAULT_BOLD; setTextColor(colors.text) }
        header.addView(title, LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        resetButton = chip(context.getString(R.string.fix_reset)) { listener.onEditorReset() }
        header.addView(resetButton)
        header.addView(chip(context.getString(R.string.close)) { listener.onEditorClose() })
        applyButton = chip(context.getString(R.string.apply), accent = true) { listener.onEditorApply(input.text.toString()) }
        header.addView(applyButton)
        addView(header)

        // live preview of the formula
        preview.textSizePx = dp(20f).toFloat()
        preview.color = colors.text
        preview.setPadding(0, dp(6f), 0, dp(2f))
        addView(HorizontalScrollView(context).apply { isHorizontalScrollBarEnabled = false; addView(preview) })

        input = EditText(context).apply {
            textSize = 17f
            typeface = Typeface.MONOSPACE
            setTextColor(colors.text)
            setHintTextColor(colors.muted)
            hint = context.getString(R.string.editor_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            maxLines = 3
            showSoftInputOnFocus = false
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) = listener.onEditorText(s.toString())
            })
        }
        addView(input, LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        // the answer, with copy / graph
        val resultRow = LinearLayout(context).apply { orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        result.textSizePx = dp(20f).toFloat()
        result.color = colors.accent
        result.typeface = Typeface.create("casual", Typeface.NORMAL)
        resultRow.addView(HorizontalScrollView(context).apply { isHorizontalScrollBarEnabled = false; addView(result) },
            LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        copyChip = chip("⧉") { listener.onEditorCopy() }
        graphChip = chip("📈") { listener.onEditorGraph() }
        resultRow.addView(copyChip)
        resultRow.addView(graphChip)
        addView(resultRow)

        // tabs
        val tabRow = LinearLayout(context).apply { orientation = HORIZONTAL; setPadding(0, dp(4f), 0, dp(4f)) }
        for (name in listOf("123", "f(x)", "x y", "∫ Σ", "ещё")) {
            val t = TextView(context).apply {
                text = name
                textSize = 14f
                gravity = Gravity.CENTER
                setPadding(dp(4f), dp(7f), dp(4f), dp(7f))
                setOnClickListener { showPage(name) }
            }
            tabs[name] = t
            tabRow.addView(t, LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        tabRow.addView(TextView(context).apply {
            text = "ABC"
            textSize = 14f
            gravity = Gravity.CENTER
            setTextColor(colors.muted)
            setPadding(dp(4f), dp(7f), dp(4f), dp(7f))
            setOnClickListener { toggleSystemKeyboard() }
        }, LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.8f))
        addView(tabRow)

        val back = Key("⌫", action = "back")
        val enter = Key("⏎", action = "enter")
        val left = Key("←", action = "left")
        val right = Key("→", action = "right")
        addPage("123", listOf(
            listOf(Key("7", "7"), Key("8", "8"), Key("9", "9"), Key("÷", "÷"), Key("(", "("), Key(")", ")")),
            listOf(Key("4", "4"), Key("5", "5"), Key("6", "6"), Key("×", "×"), Key("xⁿ", "^"), Key("√", "sqrt(|)")),
            listOf(Key("1", "1"), Key("2", "2"), Key("3", "3"), Key("−", "-"), Key("=", "="), Key("x", "x")),
            listOf(Key("0", "0"), Key(".", "."), Key("a/b", "(|)/()"), Key("+", "+"), back, enter),
        ))
        addPage("f(x)", listOf(
            listOf(Key("sin", "sin(|)"), Key("cos", "cos(|)"), Key("tg", "tg(|)"), Key("ctg", "ctg(|)"), Key("ln", "ln(|)"), Key("lg", "lg(|)")),
            listOf(Key("arcsin", "arcsin(|)"), Key("arccos", "arccos(|)"), Key("arctg", "arctg(|)"), Key("logₐ", "log(|, 2)"), Key("eˣ", "e^(|)"), Key("|x|", "abs(|)")),
            listOf(Key("π", "π"), Key("e", "e"), Key("n!", "!"), Key("∛", "root(|, 3)"), Key("ⁿ√", "root(|, n)"), Key("x²", "^2")),
            listOf(Key("x⁻¹", "^(-1)"), Key("(", "("), Key(")", ")"), left, right, back),
        ))
        addPage("x y", listOf(
            listOf(Key("x", "x"), Key("y", "y"), Key("z", "z"), Key("a", "a"), Key("b", "b"), Key("c", "c")),
            listOf(Key("n", "n"), Key("t", "t"), Key("k", "k"), Key("m", "m"), Key("p", "p"), Key("q", "q")),
            listOf(Key("<", "<"), Key(">", ">"), Key("≤", "<="), Key("≥", ">="), Key("≠", "!="), Key(";", "; ")),
            listOf(left, right, Key(",", ", "), Key("=", "="), back, enter),
        ))
        addPage("∫ Σ", listOf(
            listOf(Key("∫ dx", "int(|, x)"), Key("∫ₐᵇ", "int(|, x, 0, 1)"), Key("d/dx", "diff(|, x)"), Key("′", "'"), Key("lim", "lim(|, x, 0)"), Key("Σ", "sum(|, n, 1, oo)")),
            listOf(Key("∞", "oo"), Key("mod", " mod "), Key("y′", "y'"), Key("y″", "y''"), Key("det", "det([[|, 0], [0, 1]])"), Key("[[ ]]", "[[|, 0], [0, 1]]")),
            listOf(Key("x", "x"), Key("y", "y"), Key("n", "n"), Key(",", ", "), Key(";", "; "), Key("=", "=")),
            listOf(left, right, Key("(", "("), Key(")", ")"), back, enter),
        ))
        addPage("ещё", listOf(
            listOf(Key("i", "i"), Key("°", "°"), Key("%", "%"), Key("C(n,k)", "C(|, )"), Key("A(n,k)", "A(|, )"), Key("n!", "!")),
            listOf(Key("на множ.", "factorint(|)"), Key("простое?", "isprime(|)"), Key("делители", "divisors(|)"), Key("НОД", "gcd(|, )"), Key("НОК", "lcm(|, )"), Key("в 2-ю", "bin(|)")),
            listOf(Key("среднее", "mean(|)"), Key("медиана", "median(|)"), Key("σ", "std(|)"), Key("λ матр.", "eigenvals(|)"), Key("ранг", "rank(|)"), Key("экстр.", "extrema(|)")),
            listOf(Key("max", "max(|, x, 0, 1)"), Key("min", "min(|, x, 0, 1)"), Key("Тейлор", "taylor(|, x, 0, 6)"), Key("исслед.", "study(|)"), Key("A·X=B", "lsolve([[|, 0], [0, 1]], [1, 1])"), back),
        ))
        showPage("123")
    }

    private fun chip(text: String, accent: Boolean = false, onClick: () -> Unit) = TextView(context).apply {
        this.text = text
        textSize = 14f
        setTextColor(if (accent) colors.paper else colors.text)
        gravity = Gravity.CENTER
        setPadding(dp(12f), dp(7f), dp(12f), dp(7f))
        background = GradientDrawable().apply {
            cornerRadius = dp(16f).toFloat()
            setColor(if (accent) pal.accent else pal.paper)
            setStroke(dp(1f), if (accent) pal.accent else pal.rule)
        }
        layoutParams = LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(6f) }
        setOnClickListener { onClick() }
    }

    private fun addPage(name: String, rows: List<List<Key>>) {
        val grid = LinearLayout(context).apply { orientation = VERTICAL }
        for (row in rows) {
            val r = LinearLayout(context).apply { orientation = HORIZONTAL }
            for (k in row) {
                val v = TextView(context).apply {
                    text = k.label
                    textSize = if (k.label.length > 4) 13f else 18f
                    gravity = Gravity.CENTER
                    setTextColor(if (k.action == "enter") colors.paper else colors.text)
                    background = GradientDrawable().apply {
                        cornerRadius = dp(8f).toFloat()
                        setColor(if (k.action == "enter") pal.accent else pal.paper)
                    }
                    setOnClickListener { press(k) }
                    if (k.action == "back") setOnLongClickListener { input.setText(""); true }
                }
                r.addView(v, LayoutParams(0, dp(46f), k.wide).apply { setMargins(dp(2f), dp(2f), dp(2f), dp(2f)) })
            }
            grid.addView(r)
        }
        pages[name] = grid
        addView(grid)
    }

    private fun showPage(name: String) {
        page = name
        for ((n, v) in pages) v.visibility = if (n == name) View.VISIBLE else View.GONE
        for ((n, t) in tabs) {
            t.setTextColor(if (n == name) colors.accent else colors.muted)
            t.typeface = if (n == name) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
        if (systemKeyboard) toggleSystemKeyboard()
    }

    private fun press(k: Key) {
        val e = input.text
        val start = minOf(input.selectionStart, input.selectionEnd).coerceAtLeast(0)
        val end = maxOf(input.selectionStart, input.selectionEnd).coerceAtLeast(0)
        when (k.action) {
            "back" -> if (end > start) e.delete(start, end) else if (start > 0) e.delete(start - 1, start)
            "left" -> input.setSelection((start - 1).coerceAtLeast(0))
            "right" -> input.setSelection((end + 1).coerceAtMost(e.length))
            "enter" -> listener.onEditorApply(e.toString())
            else -> {
                val s = k.insert ?: return
                val cursor = s.indexOf('|')
                val text = s.replace("|", "")
                e.replace(start, end, text)
                input.setSelection(start + if (cursor >= 0) cursor else text.length)
            }
        }
    }

    private fun toggleSystemKeyboard() {
        systemKeyboard = !systemKeyboard
        val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        for (v in pages.values) v.visibility = if (systemKeyboard) View.GONE else if (v === pages[page]) View.VISIBLE else View.GONE
        input.showSoftInputOnFocus = systemKeyboard
        input.requestFocus()
        if (systemKeyboard) imm.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT) else imm.hideSoftInputFromWindow(input.windowToken, 0)
    }

    /** Open for typing a new calculation, or for correcting a recognised line ([text] prefilled). */
    fun open(mode: Mode, text: String, titleText: String) {
        this.mode = mode
        title.text = titleText
        resetButton.visibility = if (mode == Mode.FIX) View.VISIBLE else View.GONE
        applyButton.text = context.getString(if (mode == Mode.FIX) R.string.apply else R.string.compute)
        input.setText(text)
        input.setSelection(text.length)
        input.requestFocus()
        showPage("123")
        visibility = View.VISIBLE
    }

    fun close() {
        if (systemKeyboard) toggleSystemKeyboard()
        visibility = View.GONE
    }

    fun showPreview(tex: String?, text: String) {
        preview.show(tex?.let { listOf(it) }, text)
    }

    fun showResult(tex: List<String>?, text: String?, muted: Boolean, canCopy: Boolean, canGraph: Boolean) {
        result.color = if (muted) colors.muted else colors.accent
        result.show(tex, text ?: "")
        copyChip.visibility = if (canCopy) View.VISIBLE else View.GONE
        graphChip.visibility = if (canGraph) View.VISIBLE else View.GONE
    }

    @Suppress("unused")
    private fun sp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics)
}
