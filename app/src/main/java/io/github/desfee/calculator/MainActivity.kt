package io.github.desfee.calculator

import android.app.Activity
import android.app.AlertDialog
import android.app.Dialog
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min

class MainActivity : Activity(), InkView.Listener {
    private lateinit var colors: Palette
    private lateinit var ink: InkView
    private lateinit var inputView: TextView
    private lateinit var answerView: TextView
    private lateinit var editButton: TextView
    private lateinit var graphButton: TextView
    private lateinit var eraserButton: TextView
    private lateinit var keyboardRow: LinearLayout
    private lateinit var keyboardInput: EditText

    private val ui = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private var engine: Engine? = null
    private var recognizer: Recognizer? = null
    private var loadError: String? = null

    private val version = AtomicInteger(0)
    private val tokenCache = HashMap<String, List<String>>()
    private val overrides = HashMap<String, String>()

    /** Geometry and last result of each item on the page, by item id. */
    private class Item(val id: String, val lineKey: String?, val bounds: android.graphics.RectF, val charHeight: Float)

    private var items: Map<String, Item> = emptyMap()
    private var results: Map<String, Engine.Result> = emptyMap()
    private var focusedId: String? = null
    private var keyboardResult: Engine.Result? = null
    private var lastStrokeX = 0f
    private var lastStrokeY = 0f

    private val recognizeLater = Runnable { scheduleRecognition() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val night = (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        colors = if (night) Palette.dark else Palette.light
        window.statusBarColor = colors.panel
        window.navigationBarColor = colors.panel
        if (!night) {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or
                (if (android.os.Build.VERSION.SDK_INT >= 26) View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR else 0)
        }
        window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        setContentView(buildLayout())
        setStatus(getString(R.string.loading), null)
        loadTestInk(intent)
        worker.execute { load() }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        loadTestInk(intent)
        if (engine != null) scheduleRecognition()
    }

    /** Automated tests pass handwriting as base64 JSON: [[x0,y0,x1,y1,…], …] in dp. */
    private fun loadTestInk(intent: Intent?) {
        val b64 = intent?.getStringExtra("ink_b64") ?: return
        try {
            val arr = JSONArray(String(android.util.Base64.decode(b64, android.util.Base64.DEFAULT)))
            val d = resources.displayMetrics.density
            val list = ArrayList<List<Float>>()
            for (i in 0 until arr.length()) {
                val pts = arr.getJSONArray(i)
                list.add(List(pts.length()) { (pts.getDouble(it) * d).toFloat() })
            }
            ink.setStrokes(list)
            overrides.clear()
        } catch (e: Exception) {
            android.util.Log.e(TAG, "bad test ink", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdownNow()
    }

    // ------------------------------------------------------------------ layout

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)
    private fun dpi(v: Float) = dp(v).toInt()

    private fun toolButton(text: String, description: String, onClick: () -> Unit) = TextView(this).apply {
        this.text = text
        contentDescription = description
        textSize = 20f
        setTextColor(colors.text)
        gravity = Gravity.CENTER
        minWidth = dpi(44f)
        setPadding(dpi(10f), dpi(8f), dpi(10f), dpi(8f))
        val tv = TypedValue()
        context.theme.resolveAttribute(android.R.attr.selectableItemBackgroundBorderless, tv, true)
        setBackgroundResource(tv.resourceId)
        setOnClickListener { onClick() }
    }

    private fun chip(text: String, onClick: () -> Unit) = TextView(this).apply {
        this.text = text
        textSize = 15f
        setTextColor(colors.text)
        gravity = Gravity.CENTER
        minWidth = dpi(40f)
        setPadding(dpi(12f), dpi(8f), dpi(12f), dpi(8f))
        val pal = this@MainActivity.colors
        background = GradientDrawable().apply {
            cornerRadius = dp(18f)
            setColor(pal.paper)
            setStroke(dpi(1f), pal.rule)
        }
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            marginEnd = dpi(6f)
        }
        setOnClickListener { onClick() }
    }

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(colors.panel)
        }

        // toolbar
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dpi(12f), dpi(2f), dpi(4f), dpi(2f))
        }
        bar.addView(TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(colors.text)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(toolButton("↶", getString(R.string.undo)) { ink.undo() })
        eraserButton = toolButton("⌫", getString(R.string.eraser)) { toggleEraser() }
        bar.addView(eraserButton)
        bar.addView(toolButton("✕", getString(R.string.clear)) { clearAll() })
        bar.addView(toolButton("⌨", getString(R.string.keyboard)) { toggleKeyboard() })
        val menuButton = toolButton("⋮", getString(R.string.menu)) {}
        menuButton.setOnClickListener { showMenu(menuButton) }
        bar.addView(menuButton)
        root.addView(bar)

        ink = InkView(this, colors)
        ink.listener = this
        root.addView(ink, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        // result panel
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpi(16f), dpi(10f), dpi(16f), dpi(12f))
        }
        inputView = TextView(this).apply {
            textSize = 14f
            setTextColor(colors.muted)
            setTextIsSelectable(true)
        }
        answerView = TextView(this).apply {
            textSize = 22f
            setTextColor(colors.accent)
            typeface = Typeface.create("casual", Typeface.NORMAL)
            setTextIsSelectable(true)
            setPadding(0, dpi(2f), 0, dpi(4f))
        }
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        editButton = chip("✎ " + getString(R.string.fix)) { editFocused() }
        graphButton = chip("📈 " + getString(R.string.graph)) { openGraphForFocused() }
        actions.addView(editButton)
        actions.addView(graphButton)
        panel.addView(inputView)
        panel.addView(answerView)
        panel.addView(actions)

        // keyboard input (hidden by default)
        keyboardRow = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding(0, dpi(8f), 0, 0)
        }
        val entry = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        keyboardInput = EditText(this).apply {
            hint = getString(R.string.keyboard_hint)
            setTextColor(colors.text)
            setHintTextColor(colors.muted)
            textSize = 18f
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            imeOptions = EditorInfo.IME_ACTION_DONE
            maxLines = 4
            setOnEditorActionListener { _, actionId, event ->
                if (actionId == EditorInfo.IME_ACTION_DONE || event?.keyCode == KeyEvent.KEYCODE_ENTER) {
                    computeKeyboard(); true
                } else false
            }
        }
        entry.addView(keyboardInput, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        entry.addView(chip(" = ") { computeKeyboard() })
        val keys = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (k in listOf("x", "y", "^", "√", "π", "(", ")", "sin", "cos", "ln", "∫", "d/dx", ",")) {
            keys.addView(chip(k) { insertKey(k) })
        }
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(keys)
            setPadding(0, dpi(6f), 0, 0)
        }
        keyboardRow.addView(entry)
        keyboardRow.addView(scroll)
        panel.addView(keyboardRow)
        root.addView(panel)
        return root
    }

    private fun showMenu(anchor: View) {
        val menu = PopupMenu(this, anchor)
        menu.menu.add(0, 1, 0, getString(R.string.menu_update))
        menu.menu.add(0, 2, 1, getString(R.string.menu_help))
        menu.setOnMenuItemClickListener {
            when (it.itemId) {
                1 -> startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(DOWNLOAD_URL)))
                2 -> showHelp()
            }
            true
        }
        menu.show()
    }

    private fun showHelp() {
        val version = try {
            packageManager.getPackageInfo(packageName, 0).versionName
        } catch (e: Exception) {
            "?"
        }
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.app_name) + " " + version)
            .setMessage(getString(R.string.help_text))
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    // ------------------------------------------------------------------ actions

    private fun toggleEraser() {
        ink.eraser = !ink.eraser
        eraserButton.setTextColor(if (ink.eraser) colors.accent else colors.text)
    }

    private fun clearAll() {
        ink.clear()
        overrides.clear()
        focusedId = null
        results = emptyMap()
        items = emptyMap()
        updatePanel()
    }

    private fun toggleKeyboard() {
        val show = keyboardRow.visibility != View.VISIBLE
        keyboardRow.visibility = if (show) View.VISIBLE else View.GONE
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        if (show) {
            keyboardInput.requestFocus()
            imm.showSoftInput(keyboardInput, InputMethodManager.SHOW_IMPLICIT)
        } else {
            imm.hideSoftInputFromWindow(keyboardInput.windowToken, 0)
            keyboardResult = null
            updatePanel()
        }
    }

    private fun insertKey(k: String) {
        val text = when (k) {
            "sin", "cos", "ln" -> "$k("
            "∫" -> "int("
            "d/dx" -> "diff("
            else -> k
        }
        val start = max(keyboardInput.selectionStart, 0)
        val end = max(keyboardInput.selectionEnd, 0)
        keyboardInput.text.replace(min(start, end), max(start, end), text)
    }

    private fun computeKeyboard() {
        val text = keyboardInput.text.toString()
        if (text.isBlank()) return
        val eng = engine ?: return
        worker.execute {
            val r = try {
                eng.evaluateText(text)
            } catch (e: Exception) {
                Engine.Result("kb", text, "error", null, null, e.message)
            }
            ui.post {
                keyboardResult = r
                updatePanel()
            }
        }
    }

    private fun editFocused() {
        val kb = keyboardResult
        if (kb != null && keyboardRow.visibility == View.VISIBLE) {
            keyboardInput.requestFocus()
            return
        }
        val id = focusedId ?: return
        val item = items[id] ?: return
        val key = item.lineKey
        val current = results[id]?.input ?: ""
        if (key == null) {
            // a system: let the user retype it on the keyboard
            if (keyboardRow.visibility != View.VISIBLE) toggleKeyboard()
            keyboardInput.setText(current)
            return
        }
        val field = EditText(this).apply {
            setText(overrides[key] ?: current)
            setSelection(text.length)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        val box = LinearLayout(this).apply {
            setPadding(dpi(20f), dpi(8f), dpi(20f), 0)
            addView(field, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.fix_title)
            .setView(box)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                overrides[key] = field.text.toString()
                scheduleRecognition()
            }
            .setNeutralButton(R.string.fix_reset) { _, _ ->
                overrides.remove(key)
                scheduleRecognition()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun openGraphForFocused() {
        val r = if (keyboardRow.visibility == View.VISIBLE && keyboardResult != null) keyboardResult else focusedId?.let { results[it] }
        val expr = r?.plot ?: return
        openGraph(expr)
    }

    private fun openGraph(expr: String) {
        val eng = engine ?: return
        val night = colors === Palette.dark
        val dialog = Dialog(this, if (night) android.R.style.Theme_Material_NoActionBar else android.R.style.Theme_Material_Light_NoActionBar)
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(colors.panel)
        }
        val top = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dpi(12f), dpi(4f), dpi(4f), dpi(4f))
        }
        top.addView(TextView(this).apply {
            text = getString(R.string.graph)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(colors.text)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        top.addView(toolButton("✕", getString(R.string.close)) { dialog.dismiss() })
        box.addView(top)
        val graph = GraphView(this, colors) { xmin, xmax, n, done ->
            worker.execute {
                val (p, err) = try {
                    eng.plot(expr, xmin, xmax, n) to null
                } catch (e: Exception) {
                    null to (e.message ?: "ошибка")
                }
                ui.post { done(p, err) }
            }
        }
        box.addView(graph, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        box.addView(TextView(this).apply {
            text = getString(R.string.graph_hint)
            textSize = 12f
            setTextColor(colors.muted)
            setPadding(dpi(16f), dpi(6f), dpi(16f), dpi(10f))
        })
        dialog.setContentView(box)
        dialog.show()
    }

    // ------------------------------------------------------------------ ink events

    override fun onStrokeFinished(stroke: Stroke) {
        lastStrokeX = stroke.bounds.centerX()
        lastStrokeY = stroke.bounds.centerY()
    }

    override fun onInkChanged() {
        version.incrementAndGet()
        ui.removeCallbacks(recognizeLater)
        ui.postDelayed(recognizeLater, PAUSE_MS)
    }

    override fun onAnnotationTapped(itemId: String) {
        focusedId = itemId
        updatePanel()
        val plot = results[itemId]?.plot
        if (plot != null && results[itemId]?.kind == "function") openGraph(plot)
    }

    // ------------------------------------------------------------------ recognition

    private fun load() {
        try {
            recognizer = Recognizer(this)
            engine = Engine(this)
            ui.post {
                setStatus(getString(R.string.ready), null)
                if (ink.hasInk()) scheduleRecognition()
            }
        } catch (e: Throwable) {
            loadError = e.toString()
            android.util.Log.e(TAG, "load failed", e)
            ui.post { setStatus(getString(R.string.load_failed), e.message) }
        }
    }

    private fun scheduleRecognition() {
        val v = version.incrementAndGet()
        val strokes = ink.snapshot()
        val ov = HashMap(overrides)
        val focusX = lastStrokeX
        val focusY = lastStrokeY
        worker.execute { recognize(v, strokes, ov, focusX, focusY) }
    }

    private fun recognize(v: Int, strokes: List<Stroke>, ov: Map<String, String>, fx: Float, fy: Float) {
        val rec = recognizer ?: return
        val eng = engine ?: return
        if (v != version.get()) return
        if (strokes.isEmpty()) {
            ui.post {
                items = emptyMap(); results = emptyMap(); focusedId = null
                ink.setAnnotations(emptyList()); updatePanel()
            }
            return
        }
        val page = Segmenter.segment(strokes)
        val json = JSONArray()
        val newItems = HashMap<String, Item>()

        fun lineJson(line: InkLine): Any {
            ov[line.key]?.let { return it }
            val toks = tokenCache.getOrPut(line.key) {
                try {
                    rec.recognize(line)
                } catch (e: Exception) {
                    emptyList()
                }
            }
            return JSONArray(toks)
        }

        for (line in page.lines) {
            val id = "L:" + line.key
            newItems[id] = Item(id, line.key, line.bounds, line.charHeight)
            val o = JSONObject().put("id", id)
            val l = lineJson(line)
            if (l is String) o.put("type", "text").put("text", l).put("forced", false) else o.put("type", "line").put("tokens", l)
            json.put(o)
            if (v != version.get()) return
        }
        for (sys in page.systems) {
            val id = "S:" + sys.lines.joinToString("|") { it.key }
            val h = sys.lines.map { it.charHeight }.sorted()[sys.lines.size / 2]
            newItems[id] = Item(id, null, sys.bounds, h)
            val lines = JSONArray()
            for (line in sys.lines.sortedBy { it.bounds.top }) lines.put(lineJson(line))
            json.put(JSONObject().put("id", id).put("type", "system").put("lines", lines))
            if (v != version.get()) return
        }
        // forget cached tokens of lines that no longer exist
        val alive = page.lines.map { it.key }.toSet() + page.systems.flatMap { s -> s.lines.map { it.key } }
        tokenCache.keys.retainAll(alive)

        val res = try {
            eng.evaluate(json)
        } catch (e: Exception) {
            emptyMap()
        }
        if (v != version.get()) return
        val focus = newItems.values.firstOrNull { it.bounds.contains(fx, fy) }?.id
            ?: newItems.values.minByOrNull { kotlin.math.abs(it.bounds.centerY() - fy) }?.id
        ui.post {
            if (v != version.get()) return@post
            items = newItems
            results = res
            focusedId = focus
            ink.setAnnotations(buildAnnotations())
            updatePanel()
            for (r in res.values) {
                android.util.Log.i(TAG, "RESULT " + JSONObject().put("id", r.id).put("input", r.input).put("kind", r.kind)
                    .put("answer", r.answer ?: JSONObject.NULL).put("error", r.error ?: JSONObject.NULL))
            }
            android.util.Log.i(TAG, "DONE " + res.size)
        }
    }

    private fun buildAnnotations(): List<Annotation> {
        val out = ArrayList<Annotation>()
        for (item in items.values) {
            val r = results[item.id] ?: continue
            val size = (item.charHeight * 0.9f).coerceIn(dp(18f), dp(56f))
            val x = item.bounds.right + max(item.charHeight * 0.3f, dp(8f))
            val lines: List<String>
            var muted = false
            val ans = r.answer
            when {
                r.error != null -> {
                    if (!r.input.trim().endsWith("=")) continue
                    lines = listOf("?")
                    muted = true
                }
                r.kind == "function" -> {
                    lines = listOf("📈")
                    muted = true
                }
                ans == null -> continue
                r.kind == "solve" || r.kind == "inequality" -> lines = ans.split("; ").mapIndexed { i, s -> if (i == 0) "⇒ $s" else "   $s" }
                r.kind == "system" -> lines = ans.split("\n").map { "⇒ $it" }
                else -> lines = listOf(ans)
            }
            out.add(Annotation(item.id, lines, x, item.bounds.centerY(), size, muted))
        }
        return out
    }

    // ------------------------------------------------------------------ panel

    private fun setStatus(main: String, detail: String?) {
        inputView.text = detail ?: ""
        answerView.text = main
        editButton.visibility = View.GONE
        graphButton.visibility = View.GONE
    }

    private fun updatePanel() {
        if (engine == null) return
        val kb = keyboardResult
        val r = if (keyboardRow.visibility == View.VISIBLE && kb != null) kb else focusedId?.let { results[it] }
        if (r == null) {
            setStatus(if (ink.hasInk()) "" else getString(R.string.ready), null)
            return
        }
        inputView.text = if (r.input.isNotBlank()) getString(R.string.recognized, r.input) else ""
        answerView.text = when {
            r.error != null -> r.error
            r.kind == "function" -> getString(R.string.function_hint)
            r.answer != null -> r.answer
            else -> getString(R.string.no_answer)
        }
        editButton.visibility = if (r.id != "kb") View.VISIBLE else View.GONE
        graphButton.visibility = if (r.plot != null) View.VISIBLE else View.GONE
    }

    companion object {
        const val PAUSE_MS = 700L
        const val TAG = "Calc"
        const val DOWNLOAD_URL = "https://github.com/Des-fee/calculator/releases/latest/download/calculator.apk"
    }
}
