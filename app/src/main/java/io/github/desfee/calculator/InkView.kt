package io.github.desfee.calculator

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Text drawn on the page next to the handwriting (an answer). */
class Annotation(
    val itemId: String,
    val lines: List<String>,
    val x: Float,
    val centerY: Float,
    val textSize: Float,
    val muted: Boolean,
) {
    val hitRect = RectF()
}

/** The writing surface: one finger writes, two fingers scroll. */
@SuppressLint("ViewConstructor")
class InkView(context: Context, private val colors: Palette) : View(context) {
    interface Listener {
        fun onInkChanged()
        fun onStrokeFinished(stroke: Stroke)
        fun onAnnotationTapped(itemId: String)
    }

    var listener: Listener? = null
    var eraser = false

    private val strokes = mutableListOf<Stroke>()
    private val undoStack = ArrayDeque<List<Stroke>>()  // snapshots
    private var nextId = 1L
    private var current: Stroke? = null
    private var annotations: List<Annotation> = emptyList()
    private var offsetY = 0f
    private var panning = false
    private var lastPanY = 0f
    private var downTime = 0L
    private val density = resources.displayMetrics.density

    private val inkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.6f * density
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = colors.ink
    }
    private val answerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create("casual", Typeface.NORMAL)
        color = colors.accent
    }
    private val rulePaint = Paint().apply {
        color = colors.rule
        strokeWidth = 1f
    }
    private val hintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colors.muted
        textSize = 16f * density
        textAlign = Paint.Align.CENTER
    }
    private val path = Path()

    fun snapshot(): List<Stroke> = synchronized(strokes) { strokes.toList() }

    /** Replace the page with strokes given as flat [x0, y0, x1, y1, …] lists (used by tests). */
    fun setStrokes(list: List<List<Float>>) {
        synchronized(strokes) {
            strokes.clear()
            for (pts in list) {
                val s = Stroke(nextId++)
                var i = 0
                while (i + 1 < pts.size) {
                    s.add(pts[i], pts[i + 1])
                    i += 2
                }
                if (s.size > 0) strokes.add(s)
            }
        }
        offsetY = 0f
        invalidate()
        listener?.onInkChanged()
    }

    fun hasInk(): Boolean = strokes.isNotEmpty()

    fun setAnnotations(list: List<Annotation>) {
        annotations = list
        invalidate()
    }

    fun undo() {
        val prev = undoStack.removeLastOrNull() ?: return
        synchronized(strokes) {
            strokes.clear()
            strokes.addAll(prev)
        }
        invalidate()
        listener?.onInkChanged()
    }

    fun clear() {
        if (strokes.isEmpty()) return
        pushUndo()
        synchronized(strokes) { strokes.clear() }
        offsetY = 0f
        annotations = emptyList()
        invalidate()
        listener?.onInkChanged()
    }

    private fun pushUndo() {
        undoStack.addLast(strokes.toList())
        if (undoStack.size > 100) undoStack.removeFirst()
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(colors.paper)
        val step = 56f * density
        var y = step - (offsetY % step)
        while (y < height) {
            canvas.drawLine(0f, y, width.toFloat(), y, rulePaint)
            y += step
        }
        canvas.save()
        canvas.translate(0f, -offsetY)
        synchronized(strokes) {
            for (s in strokes) drawStroke(canvas, s)
        }
        current?.let { drawStroke(canvas, it) }
        for (a in annotations) drawAnnotation(canvas, a)
        canvas.restore()
        if (strokes.isEmpty() && current == null) {
            canvas.drawText(context.getString(R.string.hint_write), width / 2f, height / 2.4f, hintPaint)
            canvas.drawText(context.getString(R.string.hint_write2), width / 2f, height / 2.4f + 24f * density, hintPaint)
        }
    }

    private fun drawStroke(canvas: Canvas, s: Stroke) {
        if (s.size == 1) {
            canvas.drawPoint(s.xs[0], s.ys[0], inkPaint)
            return
        }
        path.reset()
        path.moveTo(s.xs[0], s.ys[0])
        for (i in 1 until s.size) {
            // smooth with quadratic segments through midpoints
            val mx = (s.xs[i - 1] + s.xs[i]) / 2
            val my = (s.ys[i - 1] + s.ys[i]) / 2
            path.quadTo(s.xs[i - 1], s.ys[i - 1], mx, my)
        }
        path.lineTo(s.xs[s.size - 1], s.ys[s.size - 1])
        canvas.drawPath(path, inkPaint)
    }

    private fun drawAnnotation(canvas: Canvas, a: Annotation) {
        answerPaint.textSize = a.textSize
        answerPaint.color = if (a.muted) colors.muted else colors.accent
        val lineH = a.textSize * 1.15f
        val total = lineH * a.lines.size
        var baseline = a.centerY - total / 2 + a.textSize * 0.8f
        var maxW = 0f
        for (line in a.lines) {
            canvas.drawText(line, a.x, baseline, answerPaint)
            maxW = max(maxW, answerPaint.measureText(line))
            baseline += lineH
        }
        a.hitRect.set(a.x - 8 * density, a.centerY - total / 2 - 8 * density, a.x + maxW + 8 * density, a.centerY + total / 2 + 8 * density)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val x = e.x
        val y = e.y + offsetY
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                panning = false
                downTime = e.eventTime
                if (eraser) {
                    pushUndo()
                    eraseAt(x, y)
                } else {
                    current = Stroke(nextId++).also { it.add(x, y) }
                }
                invalidate()
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                // second finger: scrolling, drop the stroke that was just started
                current = null
                panning = true
                lastPanY = avgY(e)
                invalidate()
            }
            MotionEvent.ACTION_MOVE -> {
                if (panning) {
                    val ay = avgY(e)
                    offsetY = max(0f, offsetY - (ay - lastPanY))
                    lastPanY = ay
                    invalidate()
                } else if (eraser) {
                    for (h in 0 until e.historySize) eraseAt(e.getHistoricalX(h), e.getHistoricalY(h) + offsetY)
                    eraseAt(x, y)
                } else {
                    current?.let { s ->
                        for (h in 0 until e.historySize) s.add(e.getHistoricalX(h), e.getHistoricalY(h) + offsetY)
                        s.add(x, y)
                        invalidate()
                    }
                }
            }
            MotionEvent.ACTION_POINTER_UP -> {
                if (panning) lastPanY = avgY(e, e.actionIndex)
            }
            MotionEvent.ACTION_UP -> {
                val s = current
                current = null
                if (s != null && !panning) {
                    val tap = s.width < 6 * density && s.height < 6 * density && e.eventTime - downTime < 250
                    val hit = if (tap) annotations.firstOrNull { it.hitRect.contains(x, y) } else null
                    if (hit != null) {
                        listener?.onAnnotationTapped(hit.itemId)
                    } else {
                        pushUndo()
                        synchronized(strokes) { strokes.add(s) }
                        listener?.onStrokeFinished(s)
                        listener?.onInkChanged()
                    }
                } else if (eraser && !panning) {
                    listener?.onInkChanged()
                }
                panning = false
                invalidate()
            }
            MotionEvent.ACTION_CANCEL -> {
                current = null
                panning = false
                invalidate()
            }
        }
        return true
    }

    private fun avgY(e: MotionEvent, skip: Int = -1): Float {
        var sum = 0f
        var n = 0
        for (i in 0 until e.pointerCount) {
            if (i == skip) continue
            sum += e.getY(i)
            n++
        }
        return if (n == 0) 0f else sum / n
    }

    private fun eraseAt(x: Float, y: Float) {
        val r = 14f * density
        val removed = synchronized(strokes) {
            strokes.removeAll { s ->
                s.bounds.left - r <= x && x <= s.bounds.right + r && s.bounds.top - r <= y && y <= s.bounds.bottom + r &&
                    (0 until s.size).any { i -> hypot(s.xs[i] - x, s.ys[i] - y) < r || nearSegment(s, i, x, y, r) }
            }
        }
        if (removed) invalidate()
    }

    private fun nearSegment(s: Stroke, i: Int, x: Float, y: Float, r: Float): Boolean {
        if (i == 0) return false
        val x1 = s.xs[i - 1]
        val y1 = s.ys[i - 1]
        val x2 = s.xs[i]
        val y2 = s.ys[i]
        val dx = x2 - x1
        val dy = y2 - y1
        val len2 = dx * dx + dy * dy
        if (len2 < 1e-3f) return false
        val t = min(1f, max(0f, ((x - x1) * dx + (y - y1) * dy) / len2))
        return abs(hypot(x1 + t * dx - x, y1 + t * dy - y)) < r
    }
}

/** Colours for light and dark themes. */
class Palette(
    val paper: Int,
    val ink: Int,
    val accent: Int,
    val muted: Int,
    val rule: Int,
    val panel: Int,
    val text: Int,
) {
    companion object {
        val light = Palette(
            paper = Color.rgb(0xFB, 0xFA, 0xF6),
            ink = Color.rgb(0x1F, 0x2A, 0x44),
            accent = Color.rgb(0xD9, 0x4F, 0x1E),
            muted = Color.rgb(0x8A, 0x8F, 0x99),
            rule = Color.rgb(0xEC, 0xEA, 0xE2),
            panel = Color.rgb(0xF1, 0xEF, 0xE8),
            text = Color.rgb(0x1F, 0x23, 0x2B),
        )
        val dark = Palette(
            paper = Color.rgb(0x17, 0x19, 0x1D),
            ink = Color.rgb(0xE8, 0xEA, 0xEE),
            accent = Color.rgb(0xFF, 0x9A, 0x5C),
            muted = Color.rgb(0x7D, 0x83, 0x8D),
            rule = Color.rgb(0x22, 0x25, 0x2B),
            panel = Color.rgb(0x1F, 0x22, 0x27),
            text = Color.rgb(0xE8, 0xEA, 0xEE),
        )
    }
}
