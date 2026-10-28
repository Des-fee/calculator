package io.github.desfee.calculator

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * z = f(x, y) as a shaded surface. One finger turns it, two fingers zoom (the area x, y ∈ [−r; r] grows or
 * shrinks and the engine samples it again via [sampler]).
 */
@SuppressLint("ViewConstructor")
class SurfaceView3D(
    context: Context,
    private val colors: Palette,
    private val sampler: (r: Double, n: Int, done: (Engine.Surface?, String?) -> Unit) -> Unit,
) : View(context) {
    private var r = 3.0
    private var yaw = -0.6  // around the vertical axis
    private var pitch = 0.55  // tilt towards the viewer
    private var zoom = 1f
    private var data: Engine.Surface? = null
    private var dataR = r
    private var message: String? = null
    private var requested = false
    private val n = 36
    private val density = resources.displayMetrics.density

    private val face = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val edge = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 0.6f * density }
    private val axis = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; strokeWidth = 1.2f * density }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.text; textSize = 14f * density }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colors.muted; textSize = 12f * density }
    private val path = Path()

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            zoom = (zoom * d.scaleFactor).coerceIn(0.05f, 20f)
            invalidate()
            return true
        }

        override fun onScaleEnd(d: ScaleGestureDetector) {
            // zooming in shows a smaller area in more detail
            r = (dataR / zoom).coerceIn(1e-3, 1e4)
            refresh()
        }
    })
    private var lastX = 0f
    private var lastY = 0f

    private fun refresh() {
        if (requested) return
        requested = true
        val want = r
        sampler(want, n) { s, err ->
            requested = false
            if (s != null) {
                data = s
                dataR = want
                zoom = 1f
            }
            message = err
            invalidate()
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (data == null && !requested) refresh()
    }

    private class Quad(val xs: FloatArray, val ys: FloatArray, val depth: Float, val color: Int)

    /** Rotate the normalised point (x, y, z ∈ [−1; 1]) and project it: screen x, screen y, depth. */
    private fun project(x: Double, y: Double, z: Double, out: FloatArray) {
        val cy = cos(yaw)
        val sy = sin(yaw)
        val x1 = x * cy - y * sy
        val y1 = x * sy + y * cy
        val cp = cos(pitch)
        val sp = sin(pitch)
        // camera looks along (0, cos p, −sin p): from the front and above
        val depth = y1 * cp - z * sp  // larger = farther from the viewer
        val up = y1 * sp + z * cp
        val scale = min(width, height) * 0.36f * zoom
        val persp = 1.0 / (1.0 + depth * 0.18)
        out[0] = (width / 2f + x1 * scale * persp).toFloat()
        out[1] = (height / 2f - up * scale * persp).toFloat()
        out[2] = depth.toFloat()
    }

    private fun heightColor(t: Double, shade: Double): Int {
        // blue (low) -> teal -> yellow -> orange (high), darker where the surface faces away from the light
        val stops = arrayOf(intArrayOf(40, 90, 200), intArrayOf(30, 170, 160), intArrayOf(240, 200, 60), intArrayOf(230, 90, 40))
        val p = t.coerceIn(0.0, 1.0) * (stops.size - 1)
        val i = p.toInt().coerceAtMost(stops.size - 2)
        val f = p - i
        val k = 0.45 + 0.55 * shade
        fun ch(c: Int) = ((stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f) * k).toInt().coerceIn(0, 255)
        return Color.rgb(ch(0), ch(1), ch(2))
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(colors.paper)
        val d = data
        if (d == null) {
            message?.let { canvas.drawText(it, 16 * density, 40 * density, text) }
            return
        }
        val rows = d.z.size
        val cols = d.z[0].size
        val span = (d.zmax - d.zmin).takeIf { it > 0 } ?: 1.0
        fun nz(v: Double) = ((v - d.zmin) / span * 2 - 1).coerceIn(-1.4, 1.4) * 0.8
        val p = FloatArray(3)
        val sxs = Array(rows) { FloatArray(cols) }
        val sys = Array(rows) { FloatArray(cols) }
        val dep = Array(rows) { FloatArray(cols) }
        for (j in 0 until rows) for (i in 0 until cols) {
            val v = d.z[j][i]
            val x = -1.0 + 2.0 * i / (cols - 1)
            val y = -1.0 + 2.0 * j / (rows - 1)
            project(x, y, if (v.isNaN()) 0.0 else nz(v), p)
            sxs[j][i] = p[0]; sys[j][i] = p[1]; dep[j][i] = p[2]
        }
        val quads = ArrayList<Quad>((rows - 1) * (cols - 1))
        val h = 2.0 / (cols - 1)
        for (j in 0 until rows - 1) for (i in 0 until cols - 1) {
            val a = d.z[j][i]; val b = d.z[j][i + 1]; val c = d.z[j + 1][i + 1]; val e = d.z[j + 1][i]
            if (a.isNaN() || b.isNaN() || c.isNaN() || e.isNaN()) continue
            // normal from the two diagonals; light from above and a little to the side
            val dzx = (nz(b) + nz(c) - nz(a) - nz(e)) / 2
            val dzy = (nz(e) + nz(c) - nz(a) - nz(b)) / 2
            val nx = -dzx * h; val ny = -dzy * h; val nzc = h * h
            val len = sqrt(nx * nx + ny * ny + nzc * nzc)
            val shade = abs((nx * 0.3 + ny * -0.4 + nzc * 0.87) / len)
            val t = ((a + b + c + e) / 4 - d.zmin) / span
            quads.add(
                Quad(
                    floatArrayOf(sxs[j][i], sxs[j][i + 1], sxs[j + 1][i + 1], sxs[j + 1][i]),
                    floatArrayOf(sys[j][i], sys[j][i + 1], sys[j + 1][i + 1], sys[j + 1][i]),
                    (dep[j][i] + dep[j][i + 1] + dep[j + 1][i + 1] + dep[j + 1][i]) / 4,
                    heightColor(t, shade),
                ),
            )
        }
        drawAxes(canvas, d)
        // painter's algorithm: far quads first
        quads.sortByDescending { it.depth }
        edge.color = colors.paper and 0x00FFFFFF or (0x50 shl 24)
        for (q in quads) {
            path.reset()
            path.moveTo(q.xs[0], q.ys[0])
            for (k in 1..3) path.lineTo(q.xs[k], q.ys[k])
            path.close()
            face.color = q.color
            canvas.drawPath(path, face)
            canvas.drawPath(path, edge)
        }
        canvas.drawText(d.label, 16 * density, 28 * density, text)
        val rr = fmt(dataR)
        canvas.drawText("x, y ∈ [−$rr; $rr]   z ∈ [${fmt(d.zmin)}; ${fmt(d.zmax)}]", 16 * density, height - 16 * density, small)
        message?.let { canvas.drawText(it, 16 * density, 50 * density, small) }
    }

    private fun drawAxes(canvas: Canvas, d: Engine.Surface) {
        val a = FloatArray(3)
        val b = FloatArray(3)
        val zb = -1.0 * 0.8
        // floor square and the three axes from its far corner
        val corners = arrayOf(doubleArrayOf(-1.0, -1.0), doubleArrayOf(1.0, -1.0), doubleArrayOf(1.0, 1.0), doubleArrayOf(-1.0, 1.0))
        for (k in 0..3) {
            project(corners[k][0], corners[k][1], zb, a)
            project(corners[(k + 1) % 4][0], corners[(k + 1) % 4][1], zb, b)
            canvas.drawLine(a[0], a[1], b[0], b[1], axis)
        }
        project(-1.0, -1.0, zb, a)
        project(-1.0, -1.0, 0.8, b)
        canvas.drawLine(a[0], a[1], b[0], b[1], axis)
        canvas.drawText("z", b[0] + 4 * density, b[1], small)
        project(1.1, -1.0, zb, b)
        canvas.drawText("x", b[0], b[1], small)
        project(-1.0, 1.1, zb, b)
        canvas.drawText("y", b[0], b[1], small)
    }

    private fun fmt(v: Double): String {
        val s = "%.3g".format(v)
        return (if (s.contains('.') && !s.contains('e')) s.trimEnd('0').trimEnd('.') else s).replace("-", "−")
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
                yaw -= (e.x - lastX) / width * 3.5
                pitch = (pitch + (e.y - lastY) / height * 3.0).coerceIn(-1.5, 1.5)
                lastX = e.x
                lastY = e.y
                invalidate()
            }
            MotionEvent.ACTION_POINTER_UP -> {
                val i = if (e.actionIndex == 0) 1 else 0
                lastX = e.getX(i)
                lastY = e.getY(i)
            }
        }
        return true
    }
}
