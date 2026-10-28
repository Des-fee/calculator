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

class MainActivity : Activity(), InkView.Listener, MathEditor.Listener {
    private lateinit var colors: Palette
    private lateinit var ink: InkView
    private lateinit var titleView: TextView
    private lateinit var inputLabel: TextView
    private lateinit var inputView: MathView
    private lateinit var answerView: MathView
    private lateinit var editButton: TextView
    private lateinit var copyButton: TextView
    private lateinit var exactButton: TextView

    /** Answer shown with all its digits (by result id and value), null = shortened. */
    private var exactShown: String? = null
    private lateinit var timerView: TextView
    private lateinit var graphButton: TextView
    private lateinit var eraserButton: TextView
    private lateinit var panel: LinearLayout
    private lateinit var editor: MathEditor

    /** The answer for the editor's text, and the item being corrected (null when typing a new calculation). */
    private var editorResult: Engine.Result? = null
    private var editorItemId: String? = null
    private var editorText = ""
    private val editorEval = Runnable { evaluateEditor() }

    private val ui = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private var engine: Engine? = null
    private var recognizer: Recognizer? = null
    private var loadError: String? = null

    private val version = AtomicInteger(0)
    /** Recognition candidates per line (best first), by line key. */
    private val tokenCache = HashMap<String, List<List<String>>>()
    private val overrides = HashMap<String, String>()

    /** Geometry and last result of each item on the page, by item id. */
    /** A line ([lineKey]) or a system ([lineKeys] of its lines, top to bottom) on the page. */
    private class Item(
        val id: String,
        val lineKey: String?,
        val bounds: android.graphics.RectF,
        val charHeight: Float,
        val lineKeys: List<String> = emptyList(),
    )

    private var items: Map<String, Item> = emptyMap()
    private var results: Map<String, Engine.Result> = emptyMap()
    private var focusedId: String? = null
    private var lastStrokeX = 0f
    private var lastStrokeY = 0f

    private val recognizeLater = Runnable { scheduleRecognition() }

    /** The open sheet (files/sheets/<id>.json); saved a moment after every change. */
    private var sheetId = ""
    private var sheetName = ""
    private val saveLater = Runnable { saveSheet() }
    private val io = Executors.newSingleThreadExecutor()

    private val prefs by lazy { getSharedPreferences("calc", MODE_PRIVATE) }

    /** When the current background computation started (uptime ms), 0 when idle. */
    private val busySince = java.util.concurrent.atomic.AtomicLong(0)
    private val tick = object : Runnable {
        override fun run() {
            val since = busySince.get()
            if (since == 0L) {
                timerView.visibility = View.GONE
                return
            }
            val ms = android.os.SystemClock.uptimeMillis() - since
            if (ms >= 2000) {
                val limit = engine?.limitSeconds ?: 0.0
                timerView.text = if (limit > 0) getString(R.string.computing_limit, ms / 1000, limit.toInt())
                else getString(R.string.computing, ms / 1000)
                timerView.visibility = View.VISIBLE
            }
            ui.postDelayed(this, 250)
        }
    }

    /** Run [work] on the worker thread with the "computing… N s" timer shown after 2 s. */
    private fun background(work: () -> Unit) {
        worker.execute {
            busySince.set(android.os.SystemClock.uptimeMillis())
            ui.post { ui.removeCallbacks(tick); ui.post(tick) }
            try {
                work()
            } finally {
                busySince.set(0)
                ui.post(tick)
            }
        }
    }

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
        MathRender.init(this)
        setContentView(buildLayout())
        setStatus(getString(R.string.loading), null)
        if (!prefs.getBoolean("help_seen", false)) {
            prefs.edit().putBoolean("help_seen", true).apply()
            ui.post { showHelp() }
        }
        if (intent?.getStringExtra("ink_b64") == null) openSheet(prefs.getString("sheet", null))
        loadTestInk(intent)
        worker.execute { load() }
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(saveLater)
        saveSheet()
    }

    // ------------------------------------------------------------------ sheets

    private fun sheetData(): JSONObject {
        val ov = JSONObject()
        for ((k, v) in overrides) ov.put(k, v)
        return JSONObject().put("name", sheetName).put("strokes", ink.strokesJson()).put("overrides", ov)
    }

    private fun saveSheet() {
        if (sheetId.isEmpty()) return
        val id = sheetId
        val data = sheetData()
        // an empty new sheet is not worth a file
        if (data.getJSONArray("strokes").length() == 0 && Sheets.load(this, id) == null) return
        io.execute {
            try {
                Sheets.save(this, id, data)
            } catch (e: Exception) {
                android.util.Log.e(TAG, "save failed", e)
            }
        }
    }

    private fun saveSoon() {
        ui.removeCallbacks(saveLater)
        ui.postDelayed(saveLater, 1500)
    }

    /** Open a saved sheet, or start a new one when [id] is null or missing. */
    private fun openSheet(id: String?) {
        val data = id?.let { Sheets.load(this, it) }
        overrides.clear()
        tokenCache.clear()
        focusedId = null
        results = emptyMap()
        items = emptyMap()
        if (data != null) {
            sheetId = id!!
            sheetName = data.optString("name", getString(R.string.sheet_default, 1))
            data.optJSONObject("overrides")?.let { ov -> for (k in ov.keys()) overrides[k] = ov.getString(k) }
            ink.loadStrokes(data.optJSONArray("strokes") ?: JSONArray())
        } else {
            sheetId = Sheets.newId()
            sheetName = getString(R.string.sheet_default, Sheets.list(this).size + 1)
            ink.loadStrokes(JSONArray())
        }
        prefs.edit().putString("sheet", sheetId).apply()
        titleView.text = sheetName
        updatePanel()
    }

    private fun showSheets() {
        saveSheet()
        val list = Sheets.list(this).filter { it.id != sheetId }
        val fmt = java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.SHORT)
        val labels = listOf("＋ " + getString(R.string.sheet_new)) + list.map { it.name + "   ·   " + fmt.format(java.util.Date(it.updated)) }
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.sheets_title, sheetName))
            .setItems(labels.toTypedArray()) { _, i ->
                saveSheet()
                openSheet(if (i == 0) null else list[i - 1].id)
            }
            .setNeutralButton(R.string.sheet_rename) { _, _ -> renameSheet() }
            .setNegativeButton(R.string.sheet_delete) { _, _ -> deleteSheet() }
            .show()
    }

    private fun renameSheet() {
        val field = EditText(this).apply {
            setText(sheetName)
            setSelection(text.length)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.sheet_rename)
            .setView(LinearLayout(this).apply {
                setPadding(dpi(20f), dpi(8f), dpi(20f), 0)
                addView(field, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            })
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val name = field.text.toString().trim()
                if (name.isNotEmpty()) {
                    sheetName = name
                    titleView.text = name
                    if (Sheets.load(this, sheetId) == null && !ink.hasInk()) Sheets.save(this, sheetId, sheetData())
                    saveSheet()
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun deleteSheet() {
        AlertDialog.Builder(this)
            .setMessage(getString(R.string.sheet_delete_confirm, sheetName))
            .setPositiveButton(R.string.sheet_delete) { _, _ ->
                ui.removeCallbacks(saveLater)
                Sheets.delete(this, sheetId)
                openSheet(Sheets.list(this).firstOrNull()?.id)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    // ------------------------------------------------------------------ export

    private fun showExport() {
        if (!ink.hasInk()) {
            android.widget.Toast.makeText(this, R.string.export_empty, android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        val labels = arrayOf(getString(R.string.export_png), getString(R.string.export_pdf), getString(R.string.export_text))
        AlertDialog.Builder(this)
            .setTitle(R.string.menu_export)
            .setItems(labels) { _, i ->
                when (i) {
                    0 -> createDocument("image/png", "$sheetName.png", REQUEST_PNG)
                    1 -> createDocument("application/pdf", "$sheetName.pdf", REQUEST_PDF)
                    else -> shareText()
                }
            }
            .show()
    }

    private fun createDocument(mime: String, title: String, request: Int) {
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = mime
            putExtra(Intent.EXTRA_TITLE, title)
        }
        @Suppress("DEPRECATION")
        startActivityForResult(intent, request)
    }

    /** The sheet as a picture: everything written plus the answers, with a margin. */
    private fun sheetBitmap(): android.graphics.Bitmap? {
        val area = ink.contentBounds() ?: return null
        area.inset(-dp(24f), -dp(24f))
        val scale = min(1f, 4096f / max(area.width(), area.height()))
        val bmp = android.graphics.Bitmap.createBitmap(
            (area.width() * scale).toInt().coerceAtLeast(1), (area.height() * scale).toInt().coerceAtLeast(1),
            android.graphics.Bitmap.Config.ARGB_8888,
        )
        ink.renderPage(android.graphics.Canvas(bmp), area, scale)
        return bmp
    }

    private fun writeExport(request: Int, uri: Uri) {
        val bmp = sheetBitmap() ?: return
        contentResolver.openOutputStream(uri)?.use { out ->
            if (request == REQUEST_PNG) {
                bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            } else {
                val pdf = android.graphics.pdf.PdfDocument()
                // A4 width (595 pt); the page is as long as the sheet
                val k = 595f / bmp.width
                val page = pdf.startPage(android.graphics.pdf.PdfDocument.PageInfo.Builder(595, (bmp.height * k).toInt().coerceAtLeast(1), 1).create())
                page.canvas.scale(k, k)
                page.canvas.drawBitmap(bmp, 0f, 0f, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
                pdf.finishPage(page)
                pdf.writeTo(out)
                pdf.close()
            }
        }
        android.widget.Toast.makeText(this, R.string.export_done, android.widget.Toast.LENGTH_SHORT).show()
    }

    /** Every line as text, top to bottom: "2 + 2 = 4", "x^2 = 4 ⇒ x = ±2". */
    private fun sheetText(): String {
        val sb = StringBuilder(sheetName).append("\n\n")
        for (item in items.values.sortedWith(compareBy({ it.bounds.top }, { it.bounds.left }))) {
            val r = results[item.id] ?: continue
            val input = r.input.trim()
            if (input.isEmpty()) continue
            val ans = answerTextOf(r)
            sb.append(
                when {
                    r.kind == "define" || ans == null -> input
                    r.kind == "value" -> (if (input.endsWith("=")) "$input " else "$input = ") + ans
                    else -> "$input  ⇒  " + ans.replace("\n", "; ")
                },
            ).append("\n")
        }
        return sb.toString()
    }

    private fun shareText() {
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, sheetName)
            putExtra(Intent.EXTRA_TEXT, sheetText())
        }
        startActivity(Intent.createChooser(send, getString(R.string.export_text)))
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
        io.shutdown()
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
        titleView = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(colors.text)
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
            // the sheet's name; tap for the list of sheets
            setOnClickListener { showSheets() }
        }
        bar.addView(titleView, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(toolButton("↶", getString(R.string.undo)) { ink.undo() })
        eraserButton = toolButton("⌫", getString(R.string.eraser)) { toggleEraser() }
        bar.addView(eraserButton)
        bar.addView(toolButton("✕", getString(R.string.clear)) { clearAll() })
        bar.addView(toolButton("⌨", getString(R.string.keyboard)) { openCalcEditor() })
        val menuButton = toolButton("⋮", getString(R.string.menu)) {}
        menuButton.setOnClickListener { showMenu(menuButton) }
        bar.addView(menuButton)
        root.addView(bar)

        ink = InkView(this, colors)
        ink.listener = this
        root.addView(ink, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        // result panel
        panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpi(16f), dpi(10f), dpi(16f), dpi(12f))
        }
        inputLabel = TextView(this).apply {
            textSize = 13f
            setTextColor(colors.muted)
        }
        inputView = MathView(this).apply {
            textSizePx = dp(17f)
            color = colors.muted
        }
        answerView = MathView(this).apply {
            textSizePx = dp(24f)
            color = colors.accent
            typeface = Typeface.create("casual", Typeface.NORMAL)
            setPadding(0, dpi(4f), 0, dpi(6f))
        }
        // long formulas scroll sideways instead of being cut
        fun scrolled(v: View) = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(v)
        }
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        editButton = chip("✎ " + getString(R.string.fix)) { openFixEditor() }
        copyButton = chip("⧉ " + getString(R.string.copy)) { copyFocused() }
        exactButton = chip(getString(R.string.exact)) { toggleExact() }
        graphButton = chip("📈 " + getString(R.string.graph)) { openGraphForFocused() }
        actions.addView(copyButton)
        actions.addView(exactButton)
        actions.addView(editButton)
        actions.addView(graphButton)
        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        inputRow.addView(inputLabel)
        inputRow.addView(scrolled(inputView))
        panel.addView(inputRow)
        panel.addView(scrolled(answerView))
        timerView = TextView(this).apply {
            textSize = 13f
            setTextColor(colors.muted)
            visibility = View.GONE
            setPadding(0, 0, 0, dpi(6f))
        }
        panel.addView(timerView)
        panel.addView(scrolled(actions))

        editor = MathEditor(this, colors, this).apply { visibility = View.GONE }
        root.addView(panel)
        root.addView(editor)
        return root
    }

    private fun showMenu(anchor: View) {
        val menu = PopupMenu(this, anchor)
        menu.menu.add(0, 6, 0, getString(R.string.menu_sheets))
        menu.menu.add(0, 7, 0, getString(R.string.menu_export))
        menu.menu.add(0, 3, 0, getString(R.string.menu_home))
        menu.menu.add(0, 4, 1, getString(R.string.menu_limit))
        menu.menu.add(0, 5, 2, getString(R.string.menu_samples))
        menu.menu.add(0, 2, 3, getString(R.string.menu_help))
        menu.menu.add(0, 1, 4, getString(R.string.menu_update))
        menu.setOnMenuItemClickListener {
            when (it.itemId) {
                1 -> startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(DOWNLOAD_URL)))
                2 -> showHelp()
                3 -> ink.scrollToStart()
                4 -> chooseLimit()
                5 -> showSamples()
                6 -> showSheets()
                7 -> showExport()
            }
            true
        }
        menu.show()
    }

    private fun chooseLimit() {
        val values = floatArrayOf(3f, 5f, 10f, 20f, 30f, 60f, 0f)
        val labels = values.map { if (it == 0f) getString(R.string.limit_none) else getString(R.string.limit_seconds, it.toInt()) }
        val current = prefs.getFloat("limit", 10f)
        AlertDialog.Builder(this)
            .setTitle(R.string.menu_limit)
            .setSingleChoiceItems(labels.toTypedArray(), values.indexOfFirst { it == current }) { d, i ->
                prefs.edit().putFloat("limit", values[i]).apply()
                engine?.limitSeconds = values[i].toDouble()
                d.dismiss()
                scheduleRecognition()
            }
            .show()
    }

    private fun shownResult(): Engine.Result? = focusedId?.let { results[it] }

    private fun exactKey(r: Engine.Result) = r.id + "|" + r.exact

    private fun toggleExact() {
        val r = shownResult() ?: return
        if (r.exact == null) return
        exactShown = if (exactShown == exactKey(r)) null else exactKey(r)
        updatePanel()
    }

    /** The answer of the focused item (or its preview) as plain text. */
    private fun answerTextOf(r: Engine.Result?): String? = when {
        r == null || r.error != null -> null
        r.exact != null && exactShown == exactKey(r) -> r.exact
        r.answer != null -> r.answer.replace("\n", "; ")
        r.preview != null -> r.preview
        else -> null
    }

    private fun copyText(text: String) {
        val cm = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
        cm.setPrimaryClip(android.content.ClipData.newPlainText(getString(R.string.app_name), text))
        if (android.os.Build.VERSION.SDK_INT < 33) {
            android.widget.Toast.makeText(this, getString(R.string.copied, text), android.widget.Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyFocused() {
        answerTextOf(shownResult())?.let { copyText(it) }
    }

    /** Full-screen guide: what the calculator understands and how to write it. */
    private fun showHelp() {
        val version = try {
            packageManager.getPackageInfo(packageName, 0).versionName
        } catch (e: Exception) {
            "?"
        }
        val night = colors === Palette.dark
        val dialog = Dialog(this, if (night) android.R.style.Theme_Material_NoActionBar else android.R.style.Theme_Material_Light_NoActionBar)
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(colors.panel)
        }
        val top = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dpi(16f), dpi(4f), dpi(4f), dpi(4f))
        }
        top.addView(TextView(this).apply {
            text = getString(R.string.help_title)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(colors.text)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        top.addView(toolButton("✕", getString(R.string.close)) { dialog.dismiss() })
        box.addView(top)
        val body = TextView(this).apply {
            @Suppress("DEPRECATION")
            text = android.text.Html.fromHtml(getString(R.string.help_html) + "<p><small>Версия $version</small></p>")
            textSize = 15f
            setTextColor(colors.text)
            setLineSpacing(0f, 1.15f)
            setPadding(dpi(18f), dpi(4f), dpi(18f), dpi(24f))
            setTextIsSelectable(true)
        }
        box.addView(android.widget.ScrollView(this).apply { addView(body) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        dialog.setContentView(box)
        dialog.show()
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

    // ------------------------------------------------------------------ editor (typing and corrections)

    private fun openCalcEditor() {
        editorItemId = null
        editorResult = null
        editor.open(MathEditor.Mode.CALC, editorText.takeIf { editor.mode == MathEditor.Mode.CALC } ?: "",
            getString(R.string.editor_calc))
        panel.visibility = View.GONE
    }

    /** Correct the focused line or system: its recognised text in keyboard form, editable with the math keypad. */
    private fun openFixEditor() {
        val id = focusedId ?: return
        val item = items[id] ?: return
        val r = results[id]
        val eng = engine ?: return
        val raw = item.lineKey?.let { overrides[it] } ?: r?.input ?: ""
        background {
            val text = try {
                if (item.lineKey == null) raw.split(";").joinToString("; ") { eng.editableText(it.trim()) } else eng.editableText(raw)
            } catch (e: Exception) {
                raw
            }
            ui.post {
                editorItemId = id
                editorResult = null
                editor.open(MathEditor.Mode.FIX, text,
                    getString(if (item.lineKey == null) R.string.editor_fix_system else R.string.editor_fix_line))
                panel.visibility = View.GONE
            }
        }
    }

    private fun closeEditor() {
        editor.close()
        editorItemId = null
        panel.visibility = View.VISIBLE
        updatePanel()
    }

    override fun onEditorText(text: String) {
        if (editor.mode == MathEditor.Mode.CALC) editorText = text
        ui.removeCallbacks(editorEval)
        ui.postDelayed(editorEval, 250)
    }

    /** Live preview: the typeset formula and its answer while typing. */
    private fun evaluateEditor() {
        val text = editor.input.text.toString()
        val eng = engine ?: return
        if (text.isBlank()) {
            editor.showPreview(null, "")
            editor.showResult(null, "", muted = true, canCopy = false, canGraph = false)
            editorResult = null
            return
        }
        background {
            val tex = try { eng.texOf(text) } catch (e: Exception) { null }
            val r = try {
                eng.evaluateText(text)
            } catch (e: Exception) {
                Engine.Result("kb", text, "error", null, null, e.message)
            }
            ui.post {
                if (editor.input.text.toString() != text) return@post
                editorResult = r
                editor.showPreview(tex, if (tex == null) text else "")
                val (rTex, rText, muted) = when {
                    r.error != null -> Triple(null, if (tex == null) getString(R.string.editor_incomplete) else r.error, true)
                    r.kind == "function" -> Triple(null, getString(R.string.function_hint), true)
                    r.kind == "define" -> Triple(r.previewTex?.let { listOf(it) }, (r.preview ?: "") + "\n" + getString(R.string.defined_hint), true)
                    r.answer != null -> Triple(r.tex, r.answer.replace("\n", "\n"), false)
                    else -> Triple(null, "", true)
                }
                editor.showResult(rTex?.map { "= $it".takeIf { _ -> r.kind == "value" } ?: it }, rText, muted,
                    canCopy = answerTextOf(r) != null, canGraph = r.plot != null)
            }
        }
    }

    override fun onEditorApply(text: String) {
        if (editor.mode == MathEditor.Mode.CALC) {
            evaluateEditor()
            return
        }
        val id = editorItemId ?: return closeEditor()
        val item = items[id] ?: return closeEditor()
        val recognized = results[id]?.input ?: ""
        val single = item.lineKey
        if (single != null) {
            overrides[single] = text
            saveSample(single, text, recognized)
        } else {
            // a system: one statement per line of the system, separated by ";"
            val parts = text.split(";").map { it.trim() }.filter { it.isNotEmpty() }
            if (parts.size != item.lineKeys.size) {
                android.widget.Toast.makeText(this, getString(R.string.editor_system_lines, item.lineKeys.size), android.widget.Toast.LENGTH_LONG).show()
                return
            }
            val before = recognized.split(";").map { it.trim() }
            for ((k, key) in item.lineKeys.withIndex()) {
                overrides[key] = parts[k]
                saveSample(key, parts[k], before.getOrElse(k) { "" })
            }
        }
        closeEditor()
        saveSoon()
        scheduleRecognition()
    }

    override fun onEditorReset() {
        val id = editorItemId
        val item = id?.let { items[it] }
        item?.lineKey?.let { overrides.remove(it) }
        item?.lineKeys?.forEach { overrides.remove(it) }
        closeEditor()
        saveSoon()
        scheduleRecognition()
    }

    override fun onEditorClose() = closeEditor()

    override fun onEditorCopy() {
        answerTextOf(editorResult)?.let { copyText(it) }
    }

    override fun onEditorGraph() {
        editorResult?.plot?.let { openGraph(it) }
    }

    @Deprecated("Back handling of android.app.Activity")
    override fun onBackPressed() {
        if (editor.visibility == View.VISIBLE) closeEditor() else @Suppress("DEPRECATION") super.onBackPressed()
    }

    /** Keep the ink of a corrected line with the right answer, for fine-tuning on this handwriting. */
    private fun saveSample(lineKey: String, corrected: String, recognized: String) {
        if (corrected.isBlank() || corrected.trim() == recognized.trim()) return
        val ids = lineKey.split(",").mapNotNull { it.toLongOrNull() }.toSet()
        val strokes = ink.snapshot().filter { it.id in ids }
        if (strokes.isEmpty()) return
        val eng = engine
        worker.execute {
            // the model never sees a trailing "=" (it is cut off before recognition), so neither does the sample
            val eq = InkGeometry.trailingEquals(InkLine(strokes))
            val body = if (eq != null) strokes.filter { it !in eq } else strokes
            val label = try {
                eng?.trainingLabel(corrected)
            } catch (e: Exception) {
                null
            }
            Samples.save(this, body, corrected, label, recognized)
            val n = Samples.count(this)
            ui.post {
                android.widget.Toast.makeText(this, getString(R.string.sample_saved, n), android.widget.Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showSamples() {
        val n = Samples.count(this)
        val d = AlertDialog.Builder(this)
            .setTitle(R.string.menu_samples)
            .setMessage(getString(R.string.samples_text, n))
            .setNegativeButton(R.string.close, null)
        if (n > 0) {
            d.setPositiveButton(R.string.samples_export) { _, _ ->
                val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "application/octet-stream"
                    putExtra(Intent.EXTRA_TITLE, "calculator_samples.jsonl")
                }
                @Suppress("DEPRECATION")
                startActivityForResult(intent, REQUEST_EXPORT)
            }
            d.setNeutralButton(R.string.samples_clear) { _, _ ->
                AlertDialog.Builder(this)
                    .setMessage(getString(R.string.samples_clear_confirm, n))
                    .setPositiveButton(R.string.samples_clear) { _, _ -> Samples.clear(this) }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
            }
        }
        d.show()
    }

    @Deprecated("Activity result API of android.app.Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null) return
        if (requestCode == REQUEST_PNG || requestCode == REQUEST_PDF) {
            try {
                writeExport(requestCode, uri)
            } catch (e: Exception) {
                android.widget.Toast.makeText(this, e.message ?: "ошибка", android.widget.Toast.LENGTH_LONG).show()
            }
            return
        }
        if (requestCode != REQUEST_EXPORT) return
        try {
            contentResolver.openOutputStream(uri)?.use { it.write(Samples.read(this)) }
            android.widget.Toast.makeText(this, getString(R.string.samples_exported, Samples.count(this)), android.widget.Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            android.widget.Toast.makeText(this, e.message ?: "ошибка", android.widget.Toast.LENGTH_LONG).show()
        }
    }

    private fun openGraphForFocused() {
        val expr = shownResult()?.plot ?: return
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
        saveSoon()
        ui.removeCallbacks(recognizeLater)
        ui.postDelayed(recognizeLater, PAUSE_MS)
    }

    override fun onAnnotationLongPressed(itemId: String) {
        focusedId = itemId
        updatePanel()
        answerTextOf(results[itemId])?.let { copyText(it) }
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
            engine = Engine(this).also { it.limitSeconds = prefs.getFloat("limit", 10f).toDouble() }
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
        background { recognize(v, strokes, ov, focusX, focusY) }
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

        /** A corrected line as text, or {"tokens": best, "alts": [other readings]}. */
        fun lineJson(line: InkLine): Any {
            ov[line.key]?.let { return it }
            val cands = tokenCache.getOrPut(line.key) {
                try {
                    rec.candidates(line)
                } catch (e: Exception) {
                    android.util.Log.e(TAG, "recognition failed", e)
                    emptyList()
                }
            }
            val alts = JSONArray()
            for (c in cands.drop(1)) alts.put(JSONArray(c))
            return JSONObject().put("tokens", JSONArray(cands.firstOrNull() ?: emptyList<String>())).put("alts", alts)
        }

        for (line in page.lines) {
            val id = "L:" + line.key
            newItems[id] = Item(id, line.key, line.bounds, line.charHeight)
            val o = JSONObject().put("id", id).put("top", line.bounds.top.toDouble())
            val l = lineJson(line)
            if (l is String) {
                o.put("type", "text").put("text", l).put("forced", false)
            } else {
                l as JSONObject
                o.put("type", "line").put("tokens", l.getJSONArray("tokens")).put("alts", l.getJSONArray("alts"))
            }
            json.put(o)
            if (v != version.get()) return
        }
        for (sys in page.systems) {
            val id = "S:" + sys.lines.joinToString("|") { it.key }
            val h = sys.lines.map { it.charHeight }.sorted()[sys.lines.size / 2]
            newItems[id] = Item(id, null, sys.bounds, h, sys.lines.sortedBy { it.bounds.top }.map { it.key })
            val lines = JSONArray()
            for (line in sys.lines.sortedBy { it.bounds.top }) lines.put(lineJson(line))
            json.put(JSONObject().put("id", id).put("type", "system").put("lines", lines).put("top", sys.bounds.top.toDouble()))
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
                    .put("answer", r.answer ?: JSONObject.NULL).put("error", r.error ?: JSONObject.NULL)
                    .put("preview", r.preview ?: JSONObject.NULL))
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
            var lines: List<String>
            var tex: List<String?>? = null
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
                r.kind == "solve" || r.kind == "inequality" || r.kind == "system" -> {
                    val sep = if (r.kind == "system") "\n" else "; "
                    lines = ans.split(sep).mapIndexed { i, t -> (if (i == 0) "⇒ " else "   ") + t }
                    tex = r.tex?.mapIndexed { i, t -> (if (i == 0) "\\Rightarrow\\ " else "\\phantom{\\Rightarrow}\\ ") + t }
                }
                else -> {
                    lines = listOf(ans)
                    tex = r.tex
                }
            }
            if (tex != null && tex.size != lines.size) tex = null
            out.add(
                Annotation(
                    item.id, lines, tex, x, item.bounds.centerY(),
                    item.bounds.left, item.bounds.bottom + item.charHeight * 0.35f, size, muted,
                ),
            )
        }
        return out
    }

    // ------------------------------------------------------------------ panel

    private fun setStatus(main: String, detail: String?) {
        copyButton.visibility = View.GONE
        exactButton.visibility = View.GONE
        inputLabel.text = ""
        inputView.show(null, detail ?: "")
        answerView.show(null, main)
        editButton.visibility = View.GONE
        graphButton.visibility = View.GONE
    }

    private fun updatePanel() {
        if (engine == null) return
        val r = shownResult()
        if (r == null) {
            setStatus(if (ink.hasInk()) "" else getString(R.string.ready), null)
            return
        }
        if (r.input.isNotBlank()) {
            inputLabel.text = getString(R.string.recognized_label)
            inputView.show(r.inputTex?.let { listOf(it) }, r.input)
        } else {
            inputLabel.text = ""
            inputView.show(null, "")
        }
        when {
            r.error != null -> answerView.show(null, r.error)
            r.kind == "function" -> answerView.show(null, getString(R.string.function_hint))
            r.exact != null && exactShown == exactKey(r) -> {
                // all digits, 40 per line; the panel scrolls sideways for the rest
                answerView.color = colors.accent
                answerView.show(null, r.exact.chunked(40).joinToString("\n"))
            }
            r.answer != null -> {
                val sep = if (r.kind == "system") "\n" else "; "
                val lines = r.answer.split(sep).joinToString("\n")
                answerView.color = colors.accent
                answerView.show(r.tex, lines)
            }
            r.kind == "define" && r.preview != null -> {
                // a = 3: remembered for the lines below, nothing is written on the sheet
                answerView.color = colors.muted
                answerView.show(r.previewTex?.let { listOf(it) }, r.preview + "\n" + getString(R.string.defined_hint))
                answerView.color = colors.accent
            }
            r.preview != null -> {
                // without "=" nothing is written on the sheet, the value is only shown here
                answerView.color = colors.muted
                answerView.show(r.previewTex?.let { listOf("= $it") }, "= " + r.preview)
                answerView.color = colors.accent
            }
            else -> answerView.show(null, getString(R.string.no_answer))
        }
        copyButton.visibility = if (answerTextOf(r) != null) View.VISIBLE else View.GONE
        exactButton.visibility = if (r.exact != null && r.error == null) View.VISIBLE else View.GONE
        exactButton.text = getString(if (r.exact != null && exactShown == exactKey(r)) R.string.short_value else R.string.exact)
        editButton.visibility = if (r.id != "kb") View.VISIBLE else View.GONE
        graphButton.visibility = if (r.plot != null) View.VISIBLE else View.GONE
    }

    companion object {
        const val REQUEST_EXPORT = 42
        const val REQUEST_PNG = 43
        const val REQUEST_PDF = 44
        const val PAUSE_MS = 700L
        const val TAG = "Calc"
        const val DOWNLOAD_URL = "https://github.com/Des-fee/calculator/releases/latest/download/calculator.apk"
    }
}
