package io.github.desfee.calculator

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Handwriting samples for fine-tuning the recogniser on the user's own writing.
 * Every correction made with "Исправить" stores the ink of the line and the corrected text,
 * one JSON object per line in files/samples.jsonl. The file is exported from the menu and
 * read by the Kaggle notebook (model/finetune_mathwriting.py).
 */
object Samples {
    private fun file(ctx: Context) = File(ctx.filesDir, "samples.jsonl")

    fun count(ctx: Context): Int = file(ctx).takeIf { it.exists() }?.useLines { s -> s.count { it.isNotBlank() } } ?: 0

    fun save(ctx: Context, strokes: List<Stroke>, text: String, latex: String?, recognized: String) {
        val ink = JSONArray()
        for (s in strokes) {
            val pts = JSONArray()
            for (i in 0 until s.size) {
                pts.put(Math.round(s.xs[i] * 10) / 10.0)
                pts.put(Math.round(s.ys[i] * 10) / 10.0)
            }
            ink.put(pts)
        }
        val o = JSONObject()
            .put("ink", ink)
            .put("text", text)
            .put("latex", latex ?: JSONObject.NULL)
            .put("recognized", recognized)
            .put("time", System.currentTimeMillis())
        synchronized(this) { file(ctx).appendText(o.toString() + "\n") }
    }

    fun read(ctx: Context): ByteArray = file(ctx).takeIf { it.exists() }?.readBytes() ?: ByteArray(0)

    fun clear(ctx: Context) {
        synchronized(this) { file(ctx).delete() }
    }
}
