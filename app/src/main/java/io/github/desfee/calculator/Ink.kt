package io.github.desfee.calculator

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/** One pen stroke in canvas coordinates. */
class Stroke(val id: Long) {
    val xs = FloatList()
    val ys = FloatList()
    val bounds = RectF(Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)

    fun add(x: Float, y: Float) {
        xs.add(x)
        ys.add(y)
        bounds.left = min(bounds.left, x)
        bounds.top = min(bounds.top, y)
        bounds.right = max(bounds.right, x)
        bounds.bottom = max(bounds.bottom, y)
    }

    val size: Int get() = xs.size
    val width: Float get() = bounds.width()
    val height: Float get() = bounds.height()
    val centerY: Float get() = bounds.centerY()
}

class FloatList {
    private var data = FloatArray(64)
    var size = 0
        private set

    fun add(v: Float) {
        if (size == data.size) data = data.copyOf(size * 2)
        data[size++] = v
    }

    operator fun get(i: Int): Float = data[i]
}

/** A line of handwriting: strokes that form one expression. */
class InkLine(val strokes: List<Stroke>) {
    val bounds = RectF().also { r ->
        r.set(strokes[0].bounds)
        for (s in strokes) r.union(s.bounds)
    }

    /** Stable identity used to cache recognition results. */
    val key: String = strokes.map { it.id }.sorted().joinToString(",")

    /** Typical character height in this line. */
    val charHeight: Float = typicalHeight(strokes)
}

/** A group of lines joined by a left curly brace: a system of equations. */
class InkSystem(val brace: Stroke, val lines: List<InkLine>) {
    val bounds = RectF(brace.bounds).also { r -> for (l in lines) r.union(l.bounds) }
}

class Page(val lines: List<InkLine>, val systems: List<InkSystem>)

fun typicalHeight(strokes: List<Stroke>): Float {
    // heights of "character-like" strokes; flat strokes (minus, fraction bars) and dots are ignored
    val hs = strokes.filter { it.height > it.width * 0.35f && it.height > 4f }.map { it.height }.sorted()
    if (hs.isEmpty()) {
        val all = strokes.map { max(it.height, it.width * 0.5f) }.sorted()
        return max(all[all.size / 2], 8f)
    }
    return max(hs[hs.size / 2], 8f)
}

/**
 * Splits the page into lines and systems.
 *
 * 1. Tall, narrow strokes are set aside as possible system braces.
 * 2. The remaining strokes are cut into horizontal bands separated by empty space.
 * 3. Bands are merged back when a fraction bar in one band has a numerator or
 *    denominator in the neighbouring band.
 * 4. A brace candidate that covers two or more bands and sits to their left
 *    becomes a system; any other candidate joins the band it overlaps most.
 */
object Segmenter {
    fun segment(input: List<Stroke>): Page {
        if (input.isEmpty()) return Page(emptyList(), emptyList())
        val h = typicalHeight(input)
        val all = mergeBraceHalves(input, h)

        val candidates = all.filter { isBraceCandidate(it, h) }
        val rest = all.filter { it !in candidates }

        var bands = splitBands(rest, h)
        bands = mergeFractions(bands, h)

        val systems = mutableListOf<InkSystem>()
        val bandLists = bands.map { it.toMutableList() }.toMutableList()
        val usedInSystem = HashSet<Int>()
        for (c in candidates.sortedBy { it.bounds.left }) {
            // a line belongs to the brace when its middle (ignoring descenders and exponents)
            // lies within the brace's height and it starts to the right of the brace
            val margin = 0.35f * h
            fun inside(i: Int): Boolean {
                val band = bandLists[i]
                if (i in usedInSystem || band.isEmpty()) return false
                val cy = coreCenter(band)
                return cy >= c.bounds.top - margin && cy <= c.bounds.bottom + margin &&
                    band.maxOf { it.bounds.right } - band.minOf { it.bounds.left } > 1.2f * h &&
                    c.bounds.right <= band.minOf { it.bounds.left } + 0.5f * h
            }
            var covered = bandLists.indices.filter { inside(it) }
            if (covered.size == 1) {
                // lines written close together end up in one band: split it by stroke centres
                val i = covered[0]
                val band = bandLists[i]
                if (band.maxOf { it.bounds.bottom } - band.minOf { it.bounds.top } > 1.8f * h) {
                    val parts = mergeFractions(splitByCentres(band, h), h)
                    if (parts.size >= 2) {
                        bandLists.removeAt(i)
                        bandLists.addAll(i, parts.map { it.toMutableList() })
                        // indices after i shift; usedInSystem only holds bands above earlier braces
                        val shifted = usedInSystem.map { if (it > i) it + parts.size - 1 else it }
                        usedInSystem.clear(); usedInSystem.addAll(shifted)
                        covered = bandLists.indices.filter { inside(it) }
                    }
                }
            }
            if (covered.size >= 2) {
                usedInSystem.addAll(covered)
                systems += InkSystem(c, covered.map { InkLine(bandLists[it]) })
            } else {
                // ordinary tall symbol (bracket, integral sign, …): attach to the closest band
                val best = bandLists.indices.maxByOrNull { overlap(c, bandLists[it]) }
                if (best != null && bandLists[best].isNotEmpty() && overlap(c, bandLists[best]) > 0f) {
                    bandLists[best].add(c)
                } else {
                    bandLists.add(mutableListOf(c))
                }
            }
        }
        val lines = bandLists.indices.filter { it !in usedInSystem && bandLists[it].isNotEmpty() }
            .map { InkLine(bandLists[it]) }
        // lines inside systems are recomputed so that strokes added later are included
        return Page(lines, systems)
    }

    private fun isBraceCandidate(s: Stroke, h: Float) =
        s.height > 1.7f * h && s.height > 2.5f * s.width && isBraceShape(s)

    /** A brace drawn as two strokes (upper and lower half) is joined into one stroke. */
    private fun mergeBraceHalves(strokes: List<Stroke>, h: Float): List<Stroke> {
        val out = strokes.toMutableList()
        var i = 0
        while (i < out.size - 1) {
            val a = out[i]
            val b = out[i + 1]  // strokes are in drawing order: the halves are consecutive
            val upper = if (a.bounds.top < b.bounds.top) a else b
            val lower = if (upper === a) b else a
            val hOverlap = min(a.bounds.right, b.bounds.right) - max(a.bounds.left, b.bounds.left)
            if (a.height > 0.6f * h && b.height > 0.6f * h && a.height > 1.3f * a.width && b.height > 1.3f * b.width &&
                kotlin.math.abs(lower.bounds.top - upper.bounds.bottom) < 0.35f * h && hOverlap > -0.2f * h &&
                !isBraceCandidate(a, h) && !isBraceCandidate(b, h)
            ) {
                val joined = Stroke(a.id)
                for (s in listOf(upper, lower)) for (k in 0 until s.size) joined.add(s.xs[k], s.ys[k])
                if (isBraceCandidate(joined, h)) {
                    out[i] = joined
                    out.removeAt(i + 1)
                    continue
                }
            }
            i++
        }
        return out
    }

    /** Vertical middle of a band, robust to descenders and exponents: the median stroke centre. */
    private fun coreCenter(band: List<Stroke>): Float {
        val c = band.map { it.centerY }.sorted()
        return c[c.size / 2]
    }

    /** Split strokes into rows where consecutive stroke centres jump by more than 0.7 of a character. */
    private fun splitByCentres(strokes: List<Stroke>, h: Float): List<List<Stroke>> {
        val sorted = strokes.sortedBy { it.centerY }
        val rows = mutableListOf(mutableListOf(sorted[0]))
        for (k in 1 until sorted.size) {
            if (sorted[k].centerY - sorted[k - 1].centerY > 0.7f * h) rows.add(mutableListOf(sorted[k]))
            else rows.last().add(sorted[k])
        }
        return rows
    }

    /** A "{": the point at mid-height is further left than both ends. An integral sign fails this. */
    private fun isBraceShape(s: Stroke): Boolean {
        var topI = 0
        var botI = 0
        var midI = 0
        val cy = s.bounds.centerY()
        for (i in 0 until s.size) {
            if (s.ys[i] < s.ys[topI]) topI = i
            if (s.ys[i] > s.ys[botI]) botI = i
            if (kotlin.math.abs(s.ys[i] - cy) < kotlin.math.abs(s.ys[midI] - cy)) midI = i
        }
        val margin = 0.15f * s.width
        return s.xs[midI] + margin < s.xs[topI] && s.xs[midI] + margin < s.xs[botI]
    }

    private fun splitBands(strokes: List<Stroke>, h: Float): List<List<Stroke>> {
        if (strokes.isEmpty()) return emptyList()
        val sorted = strokes.sortedBy { it.bounds.top }
        val result = mutableListOf<MutableList<Stroke>>()
        var current = mutableListOf(sorted[0])
        var bottom = sorted[0].bounds.bottom
        for (s in sorted.drop(1)) {
            // a small tolerance keeps touching superscripts in the same band
            if (s.bounds.top <= bottom + 0.12f * h) {
                current.add(s)
                bottom = max(bottom, s.bounds.bottom)
            } else {
                result.add(current)
                current = mutableListOf(s)
                bottom = s.bounds.bottom
            }
        }
        result.add(current)
        return result
    }

    private fun isBar(s: Stroke, h: Float) = s.width > 1.6f * s.height && s.width > 0.6f * h

    /** Merge neighbouring bands joined by a fraction bar. */
    private fun mergeFractions(input: List<List<Stroke>>, h: Float): List<List<Stroke>> {
        val bands = input.map { it.toMutableList() }.toMutableList()
        var changed = true
        while (changed) {
            changed = false
            loop@ for (i in 0 until bands.size - 1) {
                val upper = bands[i]
                val lower = bands[i + 1]
                val gap = lower.minOf { it.bounds.top } - upper.maxOf { it.bounds.bottom }
                if (gap > 1.0f * h) continue
                // bar in the upper band with a denominator below it
                val barUp = upper.filter { isBar(it, h) }.any { bar -> covers(bar, lower) && isLowestBar(bar, upper) }
                // bar in the lower band with a numerator above it
                val barDown = lower.filter { isBar(it, h) }.any { bar -> covers(bar, upper) && isHighestBar(bar, lower) }
                if (barUp || barDown) {
                    upper.addAll(lower)
                    bands.removeAt(i + 1)
                    changed = true
                    break@loop
                }
            }
        }
        return bands
    }

    /** True when most of `other` lies horizontally within the bar. */
    private fun covers(bar: Stroke, other: List<Stroke>): Boolean {
        val l = other.minOf { it.bounds.left }
        val r = other.maxOf { it.bounds.right }
        val inside = min(r, bar.bounds.right + 0.15f * bar.width) - max(l, bar.bounds.left - 0.15f * bar.width)
        return inside > 0.7f * (r - l)
    }

    private fun isLowestBar(bar: Stroke, band: List<Stroke>): Boolean {
        // nothing of the band hangs below the bar inside its horizontal range
        return band.none { it !== bar && it.bounds.top > bar.bounds.bottom &&
            it.bounds.centerX() > bar.bounds.left && it.bounds.centerX() < bar.bounds.right }
    }

    private fun isHighestBar(bar: Stroke, band: List<Stroke>): Boolean {
        return band.none { it !== bar && it.bounds.bottom < bar.bounds.top &&
            it.bounds.centerX() > bar.bounds.left && it.bounds.centerX() < bar.bounds.right }
    }

    private fun overlap(c: Stroke, band: List<Stroke>): Float {
        val top = band.minOf { it.bounds.top }
        val bottom = band.maxOf { it.bounds.bottom }
        return min(bottom, c.bounds.bottom) - max(top, c.bounds.top)
    }
}
