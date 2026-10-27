package io.github.desfee.calculator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Handwritten expression recognition with the CoMER model (ONNX, on device).
 * A line of strokes is rendered like the CROHME training images (white ink on
 * black, ~45 px character height) and decoded with beam search into LaTeX tokens.
 */
class Recognizer(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession
    private val vocab: List<String>

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(max(1, min(4, Runtime.getRuntime().availableProcessors())))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val assets = context.assets
        encoder = env.createSession(assets.open("model/encoder.onnx").use { it.readBytes() }, opts)
        decoder = env.createSession(assets.open("model/decoder.onnx").use { it.readBytes() }, opts)
        vocab = assets.open("model/vocab.txt").bufferedReader().readLines().map { it.trim() }
    }

    /** Best reading of a line (see [candidates]). */
    fun recognize(line: InkLine): List<String> = candidates(line).firstOrNull() ?: emptyList()

    /**
     * Several readings of a line, most likely first: the model's beam-search hypotheses re-ranked by how
     * well they agree with the geometry of the ink (raised exponents, dots, fraction bars, trailing "=").
     */
    fun candidates(line: InkLine, beam: Int = 6, maxLen: Int = 100): List<List<String>> {
        // a trailing "=" is cut off before recognition: the model was trained on CROHME, where formulas
        // almost never end with "=", and tends to drop it
        val eq = InkGeometry.trailingEquals(line)
        val body = if (eq != null) line.strokes.filter { it !in eq } else line.strokes
        if (body.isEmpty()) return listOf(listOf("="))
        val bodyLine = if (eq != null) InkLine(body) else line
        val hyps = beamSearch(encode(render(bodyLine)), beam, maxLen)
        val geom = InkGeometry.of(bodyLine)
        return InkGeometry.rank(hyps, geom).map { toks ->
            val fixed = InkGeometry.fixDots(toks, geom)
            if (eq != null && fixed.lastOrNull() != "=") fixed + "=" else fixed
        }.distinct()
    }

    class Image(val w: Int, val h: Int, val data: FloatArray)

    private class Feature(val h: Int, val w: Int, val d: Int, val data: FloatArray)

    fun render(line: InkLine): Image {
        val target = 45f
        var scale = target / line.charHeight
        val margin = 11f
        val b = line.bounds
        // keep inside the training range: height <= 256, width <= 1024
        val maxScale = min((256f - 2 * margin) / max(b.height(), 1f), (1024f - 2 * margin) / max(b.width(), 1f))
        scale = min(scale, maxScale)
        val w = max(16, ceil(b.width() * scale + 2 * margin).toInt())
        val h = max(16, ceil(b.height() * scale + 2 * margin).toInt())
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.BLACK)
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 3.6f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = false
        }
        val ox = margin - b.left * scale
        val oy = margin - b.top * scale
        for (s in line.strokes) {
            if (s.size == 1) {
                canvas.drawPoint(s.xs[0] * scale + ox, s.ys[0] * scale + oy, paint)
                continue
            }
            val path = Path()
            path.moveTo(s.xs[0] * scale + ox, s.ys[0] * scale + oy)
            for (i in 1 until s.size) path.lineTo(s.xs[i] * scale + ox, s.ys[i] * scale + oy)
            if (s.width * scale < 1.5f && s.height * scale < 1.5f) {
                canvas.drawPoint(s.xs[0] * scale + ox, s.ys[0] * scale + oy, paint)
            } else {
                canvas.drawPath(path, paint)
            }
        }
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        bmp.recycle()
        val data = FloatArray(w * h) { if ((pixels[it] and 0xff) > 127) 1f else 0f }
        return Image(w, h, data)
    }

    private fun encode(img: Image): Feature {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(img.data), longArrayOf(1, 1, img.h.toLong(), img.w.toLong())).use { t ->
            encoder.run(mapOf("image" to t)).use { res ->
                val out = res[0] as OnnxTensor
                val shape = out.info.shape
                val buf = out.floatBuffer
                val arr = FloatArray(buf.remaining())
                buf.get(arr)
                return Feature(shape[1].toInt(), shape[2].toInt(), shape[3].toInt(), arr)
            }
        }
    }

    private fun decodeStep(feat: Feature, seqs: List<IntArray>): Array<FloatArray> {
        val b = seqs.size
        val len = seqs[0].size
        val fb = FloatBuffer.allocate(b * feat.data.size)
        repeat(b) { fb.put(feat.data) }
        fb.rewind()
        val tb = LongBuffer.allocate(b * len)
        for (s in seqs) for (t in s) tb.put(t.toLong())
        tb.rewind()
        OnnxTensor.createTensor(env, fb, longArrayOf(b.toLong(), feat.h.toLong(), feat.w.toLong(), feat.d.toLong())).use { ft ->
            OnnxTensor.createTensor(env, tb, longArrayOf(b.toLong(), len.toLong())).use { tt ->
                decoder.run(mapOf("feature" to ft, "tokens" to tt)).use { res ->
                    val out = res[0] as OnnxTensor
                    val v = out.info.shape[1].toInt()
                    val buf = out.floatBuffer
                    return Array(b) { FloatArray(v).also { row -> buf.get(row) } }
                }
            }
        }
    }

    private class Hyp(val seq: IntArray, val score: Float)

    /** Left-to-right beam search; mirrors beam_search() in model/export_onnx.py. Best hypothesis first. */
    private fun beamSearch(feat: Feature, beam: Int, maxLen: Int): List<List<String>> {
        var beams = listOf(Hyp(intArrayOf(SOS), 0f))
        val done = mutableListOf<Pair<IntArray, Float>>()
        for (step in 0 until maxLen) {
            val logp = decodeStep(feat, beams.map { it.seq })
            val cand = ArrayList<Hyp>(beams.size * beam)
            for ((bi, h) in beams.withIndex()) {
                for (t in topK(logp[bi], beam)) {
                    cand.add(Hyp(h.seq + t, h.score + logp[bi][t]))
                }
            }
            cand.sortByDescending { it.score }
            val next = ArrayList<Hyp>(beam)
            for (c in cand) {
                if (c.seq.last() == EOS) {
                    done.add(c.seq.copyOfRange(1, c.seq.size - 1) to c.score / (c.seq.size - 1))
                } else {
                    next.add(c)
                }
                if (next.size == beam) break
            }
            beams = next
            if (done.size >= beam || beams.isEmpty()) break
            val bestDone = done.maxOfOrNull { it.second }
            if (bestDone != null && beams.all { it.score / it.seq.size < bestDone }) break
        }
        if (done.isEmpty()) beams.forEach { done.add(it.seq.copyOfRange(1, it.seq.size) to it.score / it.seq.size) }
        return done.sortedByDescending { it.second }
            .map { h -> h.first.map { vocab.getOrElse(it) { "" } }.filter { it.isNotEmpty() } }
            .distinct()
    }

    private fun topK(row: FloatArray, k: Int): IntArray {
        val idx = IntArray(k) { -1 }
        val vals = FloatArray(k) { Float.NEGATIVE_INFINITY }
        for (i in row.indices) {
            val v = row[i]
            if (v <= vals[k - 1]) continue
            var j = k - 1
            while (j > 0 && vals[j - 1] < v) {
                vals[j] = vals[j - 1]
                idx[j] = idx[j - 1]
                j--
            }
            vals[j] = v
            idx[j] = i
        }
        return idx.filter { it >= 0 }.toIntArray()
    }

    companion object {
        const val SOS = 1
        const val EOS = 2
    }
}
