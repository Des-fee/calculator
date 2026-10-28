package io.github.desfee.calculator

import kotlin.math.abs
import kotlin.math.min

/**
 * What the shape of a handwritten line says, independently of the recognition model:
 * where the middle and the baseline of the writing are, whether something is raised
 * (an exponent), how many dots there are and at which height, and how many fraction bars.
 *
 * The recogniser produces several readings; readings that contradict the geometry are
 * moved down, and dots are re-labelled by their height (· on the middle of the line is
 * multiplication, . on the baseline is a decimal point).
 */
class InkGeometry(
    val raised: Boolean,
    /** Dots from left to right: true = at mid-height (multiplication), false = on the baseline. */
    val dots: List<Boolean>,
    val fractionBars: Int,
) {
    companion object {
        private val DOT_TOKENS = setOf(".", "\\cdot")

        fun of(line: InkLine): InkGeometry {
            val h = line.charHeight
            val strokes = line.strokes
            val isDot = { s: Stroke -> s.width < 0.28f * h && s.height < 0.28f * h }
            val isBar = { s: Stroke -> s.width > 2.5f * s.height && s.width > 0.5f * h }

            // fraction bars: a bar with non-bar ink both above and below inside its horizontal range
            val bars = strokes.filter(isBar).filter { bar ->
                val inRange = strokes.filter {
                    it !== bar && !isBar(it) && it.bounds.centerX() > bar.bounds.left && it.bounds.centerX() < bar.bounds.right
                }
                inRange.any { it.bounds.bottom < bar.bounds.top + 0.1f * h } && inRange.any { it.bounds.top > bar.bounds.bottom - 0.1f * h }
            }
            val overBar = { s: Stroke ->
                bars.any { b -> s.bounds.centerX() > b.bounds.left && s.bounds.centerX() < b.bounds.right && s !== b }
            }

            // the main body of the line: character-sized strokes that are not part of a fraction
            val body = strokes.filter { !isDot(it) && !isBar(it) && it.height > 0.5f * h && !overBar(it) }
            val ref = body.ifEmpty { strokes.filter { !isDot(it) } }.ifEmpty { strokes }
            val tops = ref.map { it.bounds.top }.sorted()
            val bottoms = ref.map { it.bounds.bottom }.sorted()
            val top = tops[tops.size / 2]
            val bottom = bottoms[bottoms.size / 2]
            val mid = (top + bottom) / 2

            val raised = strokes.any {
                // a real character (not the flag of a 5 or the bar of a t), smaller than the body
                !isDot(it) && !overBar(it) && it.height > 0.3f * h && it.height < 0.8f * (bottom - top) &&
                    it.bounds.bottom < mid - 0.15f * h && it.bounds.top < top + 0.1f * h
            }
            // free-standing dots only: the dot of i/j or of ÷ has ink right above or below it
            val attached = { d: Stroke ->
                strokes.any {
                    it !== d && !isDot(it) && d.bounds.centerX() > it.bounds.left - 0.1f * h &&
                        d.bounds.centerX() < it.bounds.right + 0.1f * h &&
                        (it.bounds.top > d.bounds.bottom && it.bounds.top - d.bounds.bottom < 0.6f * h ||
                            it.bounds.bottom < d.bounds.top && d.bounds.top - it.bounds.bottom < 0.6f * h)
                }
            }
            val dots = strokes.filter { isDot(it) && !overBar(it) && !attached(it) }
                .sortedBy { it.bounds.centerX() }
                .map { it.bounds.centerY() < bottom - 0.22f * (bottom - top) }
            return InkGeometry(raised, dots, bars.size)
        }

        /**
         * Readings ordered by the model's confidence (mean log-probability per token), nudged by the
         * geometry: a reading that contradicts the shape of the ink only loses to one the model finds
         * nearly as likely. The model is trusted when it is sure; the geometry decides close calls.
         */
        fun rank(hyps: List<Pair<List<String>, Float>>, g: InkGeometry): List<List<String>> =
            hyps.sortedBy { (toks, score) -> -score + 0.08f * penalty(toks, g) }.map { it.first }

        fun penalty(toks: List<String>, g: InkGeometry): Int {
            var p = 0
            val hasPower = "^" in toks
            // limits of ∫ and Σ are raised too, so a "^" there is expected
            if (g.raised != hasPower) p += 2
            p += abs(toks.count { it == "\\frac" } - g.fractionBars) * 2
            p += abs(toks.count { it in DOT_TOKENS } - g.dots.size)
            return p
        }

        /** Re-label "." / "\cdot" by the height of the dots when their number matches. */
        fun fixDots(toks: List<String>, g: InkGeometry): List<String> {
            val idx = toks.indices.filter { toks[it] in DOT_TOKENS }
            if (idx.isEmpty() || idx.size != g.dots.size) return toks
            val out = toks.toMutableList()
            for ((k, i) in idx.withIndex()) out[i] = if (g.dots[k]) "\\cdot" else "."
            return out
        }

        /** The two strokes of an "=" written at the very end of the line, or null. */
        fun trailingEquals(line: InkLine): List<Stroke>? {
            val ss = line.strokes
            if (ss.size < 3) return null
            val h = line.charHeight
            val byX = ss.sortedByDescending { it.bounds.centerX() }
            val a = byX[0]
            val b = byX[1]
            fun bar(s: Stroke) = s.width > 2.2f * kotlin.math.max(s.height, 1f) && s.width > 0.3f * h && s.width < 2.5f * h
            if (!bar(a) || !bar(b)) return null
            val overlap = min(a.bounds.right, b.bounds.right) - kotlin.math.max(a.bounds.left, b.bounds.left)
            if (overlap < 0.55f * min(a.width, b.width)) return null
            val gap = abs(a.centerY - b.centerY)
            if (gap < 0.12f * h || gap > 0.9f * h) return null
            // nothing else reaches into the sign: rules out fractions, exponents and "≤"-like symbols
            val left = kotlin.math.min(a.bounds.left, b.bounds.left)
            val w = min(a.width, b.width)
            if (ss.any { it !== a && it !== b && it.bounds.right > left + 0.25f * w }) return null
            return listOf(a, b)
        }
    }
}
