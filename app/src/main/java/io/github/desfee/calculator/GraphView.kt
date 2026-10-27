package io.github.desfee.calculator

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow

/** Plot of y = f(x); drag to move, pinch to zoom. Points come from SymPy via [sampler]. */
@SuppressLint("ViewConstructor")
class GraphView(
    context: Context,
    private val colors: Palette,
    private val sampler: (xmin: Double, xmax: Double, n: Int, done: (Engine.Plot?, String?) -> Unit) -> Unit,
) : View(context) {
    private var xmin = -10.0
    private var xmax = 10.0
    private var ymid = 0.0
    private var yspan = 0.0  // 0 = not chosen yet, derived from the data
    private var plot: Engine.Plot? = null
    private var message: String? = null
    private var requested = false
    private val density = resources.displayMetrics.density

    private val axis = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; strokeWidth = 1.5f * density }
    private val grid = Paint().apply { color = colors.rule; strokeWidth = 1f }
    private val curve = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colors.accent; strokeWidth = 2.5f * density; style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; textSize = 11f * density }
    private val title = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.text; textSize = 16f * density }

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val f = 1.0 / d.scaleFactor
            val cx = xmin + (xmax - xmin) * d.focusX / width
            xmin = cx + (xmin - cx) * f
            xmax = cx + (xmax - cx) * f
            yspan *= f
            invalidate()
            return true
        }
    })
    private var lastX = 0f
    private var lastY = 0f

    fun refresh() {
        requested = true
        val n = 400
        sampler(xmin, xmax, n) { p, err ->
            requested = false
            plot = p
            message = err
            if (p != null && yspan == 0.0) autoY(p)
            invalidate()
        }
    }

    private fun autoY(p: Engine.Plot) {
        val ys = p.ys.filterNotNull().sorted()
        if (ys.isEmpty()) {
            yspan = xmax - xmin
            return
        }
        // robust range: ignore the extreme 5% (asymptotes)
        val lo = ys[(ys.size * 0.05).toInt()]
        val hi = ys[((ys.size - 1) * 0.95).toInt()]
        ymid = (lo + hi) / 2
        yspan = maxOf((hi - lo) * 1.3, 1e-6)
        if (yspan < 2) yspan = maxOf(yspan, 2.0)
    }

    private fun sx(x: Double) = ((x - xmin) / (xmax - xmin) * width).toFloat()
    private fun sy(y: Double) = (height / 2.0 - (y - ymid) / yspan * height).toFloat()

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(colors.paper)
        if (yspan == 0.0) yspan = (xmax - xmin) * height / maxOf(width, 1)
        drawGrid(canvas)
        val p = plot
        if (p != null) {
            val path = Path()
            var pen = false
            var prevY: Double? = null
            for (i in p.xs.indices) {
                val y = p.ys[i]
                if (y == null || (prevY != null && abs(y - prevY) > yspan * 2)) {
                    pen = false
                    prevY = y
                    if (y == null) continue
                }
                val px = sx(p.xs[i])
                val py = sy(y).coerceIn(-10f * height, 11f * height)
                if (pen) path.lineTo(px, py) else path.moveTo(px, py)
                pen = true
                prevY = y
            }
            canvas.drawPath(path, curve)
            canvas.drawText(p.label, 12 * density, 24 * density, title)
        }
        message?.let { canvas.drawText(it, 12 * density, 24 * density, title) }
    }

    private fun drawGrid(canvas: Canvas) {
        val stepX = niceStep((xmax - xmin) / 6)
        var gx = ceil(xmin / stepX) * stepX
        while (gx <= xmax) {
            val px = sx(gx)
            canvas.drawLine(px, 0f, px, height.toFloat(), grid)
            if (abs(gx) > stepX / 2) canvas.drawText(fmt(gx), px + 3 * density, sy(0.0).coerceIn(14 * density, height - 4 * density) - 3 * density, label)
            gx += stepX
        }
        val ymin = ymid - yspan / 2
        val ymax = ymid + yspan / 2
        val stepY = niceStep(yspan / 8)
        var gy = floor(ymin / stepY) * stepY
        while (gy <= ymax) {
            val py = sy(gy)
            canvas.drawLine(0f, py, width.toFloat(), py, grid)
            if (abs(gy) > stepY / 2) canvas.drawText(fmt(gy), sx(0.0).coerceIn(2 * density, width - 40 * density) + 3 * density, py - 3 * density, label)
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
        scaler.onTouchEvent(e)
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastX = e.x
                lastY = e.y
            }
            MotionEvent.ACTION_MOVE -> if (!scaler.isInProgress && e.pointerCount == 1) {
                val dx = (e.x - lastX) / width * (xmax - xmin)
                val dy = (e.y - lastY) / height * yspan
                xmin -= dx
                xmax -= dx
                ymid += dy
                lastX = e.x
                lastY = e.y
                invalidate()
            }
            MotionEvent.ACTION_POINTER_UP -> {
                val i = if (e.actionIndex == 0) 1 else 0
                lastX = e.getX(i)
                lastY = e.getY(i)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> if (!requested) refresh()
        }
        return true
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (plot == null && !requested) refresh()
    }
}
