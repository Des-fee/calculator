package io.github.desfee.calculator

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.util.LruCache
import android.view.View
import ru.noties.jlatexmath.JLatexMathAndroid
import ru.noties.jlatexmath.JLatexMathDrawable
import kotlin.math.ceil
import kotlin.math.max

/** Typeset formulas (LaTeX → drawable) with a small cache; null when the LaTeX cannot be laid out. */
object MathRender {
    private val cache = LruCache<String, JLatexMathDrawable>(96)
    private var initialised = false

    fun init(context: Context) {
        if (!initialised) {
            JLatexMathAndroid.init(context.applicationContext)
            initialised = true
        }
    }

    fun drawable(tex: String, sizePx: Float, color: Int): JLatexMathDrawable? {
        if (tex.isBlank()) return null
        val key = "$sizePx|$color|$tex"
        cache.get(key)?.let { return it }
        return try {
            JLatexMathDrawable.builder(tex)
                .textSize(sizePx)
                .color(color)
                .align(JLatexMathDrawable.ALIGN_LEFT)
                .build()
                .also { cache.put(key, it) }
        } catch (e: Throwable) {
            android.util.Log.w("Calc", "latex failed: $tex", e)
            null
        }
    }
}

/**
 * One block of math: several lines, each typeset from LaTeX or, when there is
 * no LaTeX (or it fails), drawn as plain text.
 */
class MathView(context: Context) : View(context) {
    private class Row(val drawable: JLatexMathDrawable?, val text: String?, val w: Int, val h: Int)

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var rows: List<Row> = emptyList()
    private val gap = 4f * resources.displayMetrics.density

    var textSizePx = 16f * resources.displayMetrics.scaledDensity
    var color = 0xFF000000.toInt()
    var typeface: Typeface = Typeface.DEFAULT

    /** `tex[i]` (if any) is used for line i, `text` lines are the fallback. */
    fun show(tex: List<String?>?, text: String?) {
        val textLines = text?.split("\n") ?: emptyList()
        val n = max(tex?.size ?: 0, textLines.size)
        paint.textSize = textSizePx
        paint.color = color
        paint.typeface = typeface
        val fm = paint.fontMetrics
        rows = (0 until n).map { i ->
            val d = tex?.getOrNull(i)?.let { MathRender.drawable(it, textSizePx, color) }
            if (d != null) {
                Row(d, null, d.intrinsicWidth, d.intrinsicHeight)
            } else {
                val t = textLines.getOrNull(i) ?: textLines.joinToString(" ")
                Row(null, t, ceil(paint.measureText(t)).toInt(), ceil(fm.descent - fm.ascent).toInt())
            }
        }
        requestLayout()
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = (rows.maxOfOrNull { it.w } ?: 0) + paddingLeft + paddingRight
        val h = rows.sumOf { it.h } + (gap * max(0, rows.size - 1)).toInt() + paddingTop + paddingBottom
        setMeasuredDimension(resolveSize(w, widthMeasureSpec), resolveSize(h, heightMeasureSpec))
    }

    override fun onDraw(canvas: Canvas) {
        var y = paddingTop.toFloat()
        paint.color = color
        val fm = paint.fontMetrics
        for (r in rows) {
            val d = r.drawable
            if (d != null) {
                canvas.save()
                canvas.translate(paddingLeft.toFloat(), y)
                d.setBounds(0, 0, r.w, r.h)
                d.draw(canvas)
                canvas.restore()
            } else if (r.text != null) {
                canvas.drawText(r.text, paddingLeft.toFloat(), y - fm.ascent, paint)
            }
            y += r.h + gap
        }
    }
}
