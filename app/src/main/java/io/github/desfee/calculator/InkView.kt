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
import android.view.VelocityTracker
import android.view.View
import android.view.ViewConfiguration
import android.widget.OverScroller
import ru.noties.jlatexmath.JLatexMathDrawable
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * An answer drawn on the page next to the handwriting: one or more lines, each
 * typeset from LaTeX when available, otherwise drawn as handwriting-like text.
 */
class Annotation(
    val itemId: String,
    val lines: List<String>,
    val tex: List<String?>?,
    /** Preferred place: to the right of the item, vertically centred on [centerY]. */
    val x: Float,
    val centerY: Float,
    /** Fallback place under the item (left, top) when there is no room on the right. */
    val belowX: Float,
    val belowY: Float,
    val textSize: Float,
    val muted: Boolean,
) {
    val hitRect = RectF()
    internal var rows: List<Row>? = null

    internal class Row(val drawable: JLatexMathDrawable?, val text: String?, val w: Float, val h: Float)
}

/** The writing surface: an endless sheet. One finger writes, two fingers move the sheet in any direction. */
@SuppressLint("ViewConstructor")
class InkView(context: Context, private val colors: Palette) : View(context) {
    interface Listener {
        fun onInkChanged()
        fun onStrokeFinished(stroke: Stroke)
        fun onAnnotationTapped(itemId: String)
        fun onAnnotationLongPressed(itemId: String)
    }

    var listener: Listener? = null
    var eraser = false

    private val strokes = mutableListOf<Stroke>()
    private val undoStack = ArrayDeque<List<Stroke>>()  // snapshots
    private var nextId = 1L
    private var current: Stroke? = null
    private var annotations: List<Annotation> = emptyList()

    // sheet position: page coordinate shown at the top-left corner of the view
    private var offsetX = 0f
    private var offsetY = 0f
    private var panning = false
    private var lastPanX = 0f
    private var lastPanY = 0f
    private val scroller = OverScroller(context)
    private var velocity: VelocityTracker? = null

    private var downTime = 0L
    private var downX = 0f
    private var downY = 0f
    private var pressedAnnotation: String? = null
    private val longPress = Runnable {
        val id = pressedAnnotation ?: return@Runnable
        pressedAnnotation = null
        current = null  // the press was not writing
        performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS)
        listener?.onAnnotationLongPressed(id)
        invalidate()
    }
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
        scrollToStart()
        listener?.onInkChanged()
    }

    fun hasInk(): Boolean = strokes.isNotEmpty()

    fun setAnnotations(list: List<Annotation>) {
        annotations = list
        invalidate()
    }

    /** Back to the top-left corner of the sheet. */
    fun scrollToStart() {
        scroller.forceFinished(true)
        offsetX = 0f
        offsetY = 0f
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
        annotations = emptyList()
        scrollToStart()
        listener?.onInkChanged()
    }

    private fun pushUndo() {
        undoStack.addLast(strokes.toList())
        if (undoStack.size > 100) undoStack.removeFirst()
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(colors.paper)
        val step = 56f * density
        var y = step - (((offsetY % step) + step) % step)
        while (y < height) {
            canvas.drawLine(0f, y, width.toFloat(), y, rulePaint)
            y += step
        }
        canvas.save()
        canvas.translate(-offsetX, -offsetY)
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

    override fun computeScroll() {
        if (scroller.computeScrollOffset()) {
            offsetX = scroller.currX.toFloat()
            offsetY = scroller.currY.toFloat()
            postInvalidateOnAnimation()
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

    private fun rowsOf(a: Annotation): List<Annotation.Row> {
        a.rows?.let { return it }
        answerPaint.textSize = a.textSize
        val color = if (a.muted) colors.muted else colors.accent
        val fm = answerPaint.fontMetrics
        val rows = a.lines.indices.map { i ->
            val d = a.tex?.getOrNull(i)?.let { MathRender.drawable(it, a.textSize, color) }
            if (d != null) {
                Annotation.Row(d, null, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
            } else {
                Annotation.Row(null, a.lines[i], answerPaint.measureText(a.lines[i]), fm.descent - fm.ascent)
            }
        }
        a.rows = rows
        return rows
    }

    private fun drawAnnotation(canvas: Canvas, a: Annotation) {
        val rows = rowsOf(a)
        val gap = a.textSize * 0.15f
        val total = rows.sumOf { it.h.toDouble() }.toFloat() + gap * (rows.size - 1).coerceAtLeast(0)
        val maxW = rows.maxOfOrNull { it.w } ?: 0f
        // to the right of the item, unless that runs past the right edge of what is on screen now:
        // then under the item, kept inside the visible part of the sheet, if nothing is written there
        var left = a.x
        var top = a.centerY - total / 2
        val visLeft = offsetX + 8 * density
        val visRight = offsetX + width - 8 * density
        if (width > 0 && a.x + maxW > visRight && maxW < visRight - visLeft) {
            val belowLeft = a.belowX.coerceIn(visLeft, visRight - maxW)
            if (isFree(belowLeft, a.belowY, maxW, total)) {
                left = belowLeft
                top = a.belowY
            }
        }
        answerPaint.textSize = a.textSize
        answerPaint.color = if (a.muted) colors.muted else colors.accent
        val fm = answerPaint.fontMetrics
        var y = top
        for (r in rows) {
            val d = r.drawable
            if (d != null) {
                canvas.save()
                canvas.translate(left, y)
                d.setBounds(0, 0, r.w.toInt(), r.h.toInt())
                d.draw(canvas)
                canvas.restore()
            } else if (r.text != null) {
                canvas.drawText(r.text, left, y - fm.ascent, answerPaint)
            }
            y += r.h + gap
        }
        a.hitRect.set(left - 8 * density, top - 8 * density, left + maxW + 8 * density, top + total + 8 * density)
    }

    /** True when no ink lies in the given page rectangle. */
    private fun isFree(left: Float, top: Float, w: Float, h: Float): Boolean {
        val r = RectF(left, top, left + w, top + h)
        return synchronized(strokes) { strokes.none { RectF.intersects(it.bounds, r) } }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val x = e.x + offsetX
        val y = e.y + offsetY
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                scroller.forceFinished(true)
                panning = false
                downTime = e.eventTime
                downX = e.x
                downY = e.y
                pressedAnnotation = if (eraser) null else annotations.firstOrNull { it.hitRect.contains(x, y) }?.itemId
                if (pressedAnnotation != null) postDelayed(longPress, ViewConfiguration.getLongPressTimeout().toLong())
                velocity?.recycle()
                velocity = VelocityTracker.obtain().also { it.addMovement(e) }
                if (eraser) {
                    pushUndo()
                    eraseAt(x, y)
                } else {
                    current = Stroke(nextId++).also { it.add(x, y) }
                }
                invalidate()
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                // second finger: move the sheet, drop the stroke that was just started
                cancelAnswerPress()
                current = null
                panning = true
                lastPanX = avg(e, true)
                lastPanY = avg(e, false)
                velocity?.addMovement(e)
                invalidate()
            }
            MotionEvent.ACTION_MOVE -> {
                velocity?.addMovement(e)
                if (pressedAnnotation != null && hypot(e.x - downX, e.y - downY) > ViewConfiguration.get(context).scaledTouchSlop) {
                    cancelAnswerPress()
                }
                if (panning) {
                    val ax = avg(e, true)
                    val ay = avg(e, false)
                    offsetX -= ax - lastPanX
                    offsetY -= ay - lastPanY
                    lastPanX = ax
                    lastPanY = ay
                    invalidate()
                } else if (eraser) {
                    for (h in 0 until e.historySize) eraseAt(e.getHistoricalX(h) + offsetX, e.getHistoricalY(h) + offsetY)
                    eraseAt(x, y)
                } else {
                    current?.let { s ->
                        for (h in 0 until e.historySize) s.add(e.getHistoricalX(h) + offsetX, e.getHistoricalY(h) + offsetY)
                        s.add(x, y)
                        invalidate()
                    }
                }
            }
            MotionEvent.ACTION_POINTER_UP -> {
                if (panning) {
                    velocity?.addMovement(e)
                    if (e.pointerCount == 2) fling(e)
                    lastPanX = avg(e, true, e.actionIndex)
                    lastPanY = avg(e, false, e.actionIndex)
                }
            }
            MotionEvent.ACTION_UP -> {
                cancelAnswerPress()
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
                velocity?.recycle()
                velocity = null
                invalidate()
            }
            MotionEvent.ACTION_CANCEL -> {
                cancelAnswerPress()
                current = null
                panning = false
                velocity?.recycle()
                velocity = null
                invalidate()
            }
        }
        return true
    }

    private fun cancelAnswerPress() {
        pressedAnnotation = null
        removeCallbacks(longPress)
    }

    /** Keep the sheet moving after a quick two-finger swipe. */
    private fun fling(e: MotionEvent) {
        val v = velocity ?: return
        v.computeCurrentVelocity(1000, ViewConfiguration.get(context).scaledMaximumFlingVelocity.toFloat())
        var vx = 0f
        var vy = 0f
        for (i in 0 until e.pointerCount) {
            vx += v.getXVelocity(e.getPointerId(i))
            vy += v.getYVelocity(e.getPointerId(i))
        }
        vx /= e.pointerCount
        vy /= e.pointerCount
        val minV = ViewConfiguration.get(context).scaledMinimumFlingVelocity
        if (abs(vx) < minV && abs(vy) < minV) return
        val far = 10_000_000
        scroller.fling(offsetX.toInt(), offsetY.toInt(), -vx.toInt(), -vy.toInt(), -far, far, -far, far)
        postInvalidateOnAnimation()
    }

    private fun avg(e: MotionEvent, xAxis: Boolean, skip: Int = -1): Float {
        var sum = 0f
        var n = 0
        for (i in 0 until e.pointerCount) {
            if (i == skip) continue
            sum += if (xAxis) e.getX(i) else e.getY(i)
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
