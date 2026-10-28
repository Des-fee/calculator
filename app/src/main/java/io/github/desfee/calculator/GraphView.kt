package io.github.desfee.calculator

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.pow

/**
 * 2D graphs: y = f(x), parametric and polar curves, implicit curves, regions of inequalities, direction fields.
 * Drag to move, pinch to zoom, tap a marked point (zero, extremum, intersection) to see its coordinates.
 * The engine samples everything for the current window via [sampler]; [initial] picks the first window.
 */
@SuppressLint("ViewConstructor")
class GraphView(
    context: Context,
    private val colors: Palette,
    private val initial: (w: Int, h: Int, done: (DoubleArray?) -> Unit) -> Unit,
    private val sampler: (xmin: Double, xmax: Double, ymin: Double, ymax: Double, w: Int, h: Int,
                          done: (Engine.Scene?, String?) -> Unit) -> Unit,
) : View(context) {
    private var xmin = -10.0
    private var xmax = 10.0
    private var ymin = -10.0
    private var ymax = 10.0
    private var ready = false
    private var scene: Engine.Scene? = null
    /** The window the scene was sampled for: xmin, xmax, ymin, ymax. */
    private var sceneWindow = DoubleArray(4)
    private var message: String? = null
    private var requested = false
    private var again = false
    private var selected: Engine.Point? = null
    private val density = resources.displayMetrics.density

    private val palette = intArrayOf(colors.accent, 0xFF2E7DD7.toInt(), 0xFF2EA44F.toInt(), 0xFFE07B00.toInt(),
        0xFF8E44AD.toInt(), 0xFF00897B.toInt())

    private val axis = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; strokeWidth = 1.5f * density }
    private val grid = Paint().apply { color = colors.rule; strokeWidth = 1f }
    private val curve = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 2.5f * density; style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND; strokeCap = Paint.Cap.ROUND
    }
    private val fill = Paint()
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG)
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; textSize = 11f * density }
    private val pointLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.text; textSize = 13f * density }
    private val title = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 14f * density }
    private val titleBack = Paint().apply { color = colors.paper and 0x00FFFFFF or (0xD0 shl 24) }
    private val path = Path()

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val f = 1.0 / d.scaleFactor
            val cx = wx(d.focusX)
            val cy = wy(d.focusY)
            xmin = cx + (xmin - cx) * f
            xmax = cx + (xmax - cx) * f
            ymin = cy + (ymin - cy) * f
            ymax = cy + (ymax - cy) * f
            invalidate()
            return true
        }
    })
    private val taps = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onSingleTapUp(e: MotionEvent): Boolean {
            val pts = scene?.points ?: return false
            val near = pts.minByOrNull { hypot((sx(it.x) - e.x).toDouble(), (sy(it.y) - e.y).toDouble()) }
            selected = near?.takeIf { hypot((sx(it.x) - e.x).toDouble(), (sy(it.y) - e.y).toDouble()) < 28 * density }
            invalidate()
            return true
        }
    })
    private var lastX = 0f
    private var lastY = 0f

    /** Draw again from scratch (other graphs were added): the window stays. */
    fun reload() {
        if (ready) refresh()
    }

    private fun refresh() {
        if (requested) {
            again = true
            return
        }
        requested = true
        val window = doubleArrayOf(xmin, xmax, ymin, ymax)
        sampler(xmin, xmax, ymin, ymax, width, height) { s, err ->
            requested = false
            if (s != null) {
                scene = s
                sceneWindow = window
            }
            message = err
            invalidate()
            if (again) {
                again = false
                refresh()
            }
        }
    }

    private fun sx(x: Double) = ((x - xmin) / (xmax - xmin) * width).toFloat()
    private fun sy(y: Double) = ((ymax - y) / (ymax - ymin) * height).toFloat()
    private fun wx(px: Float) = xmin + (xmax - xmin) * px / width
    private fun wy(py: Float) = ymax - (ymax - ymin) * py / height

    private fun colorOf(i: Int) = palette[i % palette.size]

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(colors.paper)
        if (!ready) return
        drawGrid(canvas)
        val s = scene
        if (s != null) {
            for ((i, layer) in s.layers.withIndex()) drawLayer(canvas, layer, colorOf(i))
            drawPoints(canvas, s.points)
            drawLegend(canvas, s.layers)
        }
        message?.let {
            title.color = colors.text
            canvas.drawText(it, 12 * density, height - 16 * density, title)
        }
    }

    private fun strokePath(canvas: Canvas, p: DoubleArray, paint: Paint) {
        if (p.size < 4) return
        path.reset()
        path.moveTo(sx(p[0]), sy(p[1]).coerceIn(-10f * height, 11f * height))
        var k = 2
        while (k + 1 < p.size) {
            path.lineTo(sx(p[k]), sy(p[k + 1]).coerceIn(-10f * height, 11f * height))
            k += 2
        }
        canvas.drawPath(path, paint)
    }

    private fun drawLayer(canvas: Canvas, layer: Engine.Layer, color: Int) {
        if (layer.type == "region" && layer.mask != null && layer.nx > 0) {
            // cells of the window the scene was sampled for, placed where that window is now
            fill.color = color and 0x00FFFFFF or (0x48 shl 24)
            val w = sceneWindow
            val cw = (w[1] - w[0]) / layer.nx
            val ch = (w[3] - w[2]) / layer.ny
            for (j in 0 until layer.ny) {
                var i = 0
                while (i < layer.nx) {
                    if (layer.mask[j * layer.nx + i] == '1') {
                        var e = i
                        while (e + 1 < layer.nx && layer.mask[j * layer.nx + e + 1] == '1') e++
                        canvas.drawRect(sx(w[0] + i * cw), sy(w[3] - j * ch), sx(w[0] + (e + 1) * cw), sy(w[3] - (j + 1) * ch) + 0.5f, fill)
                        i = e + 1
                    } else {
                        i++
                    }
                }
            }
        }
        curve.color = if (layer.thin) colors.muted else color
        curve.strokeWidth = (if (layer.thin) 1.3f else if (layer.type == "region") 1.6f else 2.5f) * density
        for (p in layer.paths) strokePath(canvas, p, curve)
        layer.solution?.let {
            curve.color = color
            curve.strokeWidth = 2.5f * density
            strokePath(canvas, it, curve)
        }
        layer.ic?.let {
            dot.color = color
            canvas.drawCircle(sx(it[0]), sy(it[1]), 5 * density, dot)
        }
    }

    private fun drawPoints(canvas: Canvas, points: List<Engine.Point>) {
        val showAll = points.size <= 8
        for (p in points) {
            val x = sx(p.x)
            val y = sy(p.y)
            if (x < -20 || x > width + 20 || y < -20 || y > height + 20) continue
            dot.color = colors.paper
            canvas.drawCircle(x, y, 5.5f * density, dot)
            dot.color = if (p === selected) colors.accent else colors.text
            canvas.drawCircle(x, y, 3.5f * density, dot)
            if (showAll || p === selected) {
                val w = pointLabel.measureText(p.label)
                val lx = (x + 6 * density).coerceAtMost(width - w - 4 * density)
                val ly = if (y < 40 * density) y + 18 * density else y - 8 * density
                canvas.drawRect(lx - 2 * density, ly - 13 * density, lx + w + 2 * density, ly + 4 * density, titleBack)
                pointLabel.color = if (p === selected) colors.accent else colors.text
                canvas.drawText(p.label, lx, ly, pointLabel)
            }
        }
    }

    private fun drawLegend(canvas: Canvas, layers: List<Engine.Layer>) {
        var y = 22 * density
        for ((i, l) in layers.withIndex()) {
            if (l.label.isEmpty()) continue
            val w = title.measureText(l.label)
            canvas.drawRect(8 * density, y - 15 * density, 34 * density + w, y + 6 * density, titleBack)
            fill.color = colorOf(i)
            canvas.drawRect(12 * density, y - 6 * density, 26 * density, y - 3 * density, fill)
            title.color = colors.text
            canvas.drawText(l.label, 30 * density, y, title)
            y += 22 * density
        }
    }

    private fun drawGrid(canvas: Canvas) {
        val stepX = niceStep((xmax - xmin) / 6)
        val axisY = sy(0.0).coerceIn(14 * density, height - 4 * density)
        var gx = ceil(xmin / stepX) * stepX
        while (gx <= xmax) {
            val px = sx(gx)
            canvas.drawLine(px, 0f, px, height.toFloat(), grid)
            if (abs(gx) > stepX / 2) canvas.drawText(fmt(gx), px + 3 * density, axisY - 3 * density, label)
            gx += stepX
        }
        val stepY = niceStep((ymax - ymin) / 8)
        val axisX = sx(0.0).coerceIn(2 * density, width - 40 * density)
        var gy = floor(ymin / stepY) * stepY
        while (gy <= ymax) {
            val py = sy(gy)
            canvas.drawLine(0f, py, width.toFloat(), py, grid)
            if (abs(gy) > stepY / 2) canvas.drawText(fmt(gy), axisX + 3 * density, py - 3 * density, label)
            gy += stepY
        }
        canvas.drawLine(0f, sy(0.0), width.toFloat(), sy(0.0), axis)
        canvas.drawLine(sx(0.0), 0f, sx(0.0), height.toFloat(), axis)
    }

    private fun niceStep(raw: Double): Double {
        val p = 10.0.pow(floor(log10(raw)))
        val m = raw / p
        return p * when {
            m < 1.5 -> 1.0
            m < 3.5 -> 2.0
            m < 7.5 -> 5.0
            else -> 10.0
        }
    }

    private fun fmt(v: Double): String {
        val s = "%.4g".format(v)
        return if (s.contains('.') && !s.contains('e')) s.trimEnd('0').trimEnd('.') else s
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (!ready) return true
        scaler.onTouchEvent(e)
        taps.onTouchEvent(e)
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastX = e.x
                lastY = e.y
            }
            MotionEvent.ACTION_MOVE -> if (!scaler.isInProgress && e.pointerCount == 1) {
                val dx = (e.x - lastX) / width * (xmax - xmin)
                val dy = (e.y - lastY) / height * (ymax - ymin)
                xmin -= dx
                xmax -= dx
                ymin += dy
                ymax += dy
                lastX = e.x
                lastY = e.y
                invalidate()
            }
            MotionEvent.ACTION_POINTER_UP -> {
                val i = if (e.actionIndex == 0) 1 else 0
                lastX = e.getX(i)
                lastY = e.getY(i)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> refresh()
        }
        return true
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (ready || w == 0 || h == 0) return
        initial(w, h) { v ->
            if (v != null && v.size == 4) {
                xmin = v[0]; xmax = v[1]; ymin = v[2]; ymax = v[3]
            } else {
                ymax = 10.0 * h / w
                ymin = -ymax
            }
            ready = true
            refresh()
        }
    }
}
