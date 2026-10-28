package io.github.desfee.calculator

import android.content.Context
import org.json.JSONObject
import java.io.File

/** Saved sheets: files/sheets/<id>.json = {name, updated, strokes: [{id, p: [x0, y0, …]}], overrides: {lineKey: text}}. */
object Sheets {
    class Meta(val id: String, val name: String, val updated: Long)

    private fun dir(ctx: Context) = File(ctx.filesDir, "sheets").apply { mkdirs() }
    private fun file(ctx: Context, id: String) = File(dir(ctx), "$id.json")

    fun list(ctx: Context): List<Meta> =
        (dir(ctx).listFiles { f -> f.name.endsWith(".json") } ?: emptyArray()).mapNotNull { f ->
            try {
                val o = JSONObject(f.readText())
                Meta(f.name.removeSuffix(".json"), o.optString("name", "?"), o.optLong("updated", f.lastModified()))
            } catch (e: Exception) {
                null
            }
        }.sortedByDescending { it.updated }

    fun load(ctx: Context, id: String): JSONObject? = try {
        file(ctx, id).takeIf { it.exists() }?.let { JSONObject(it.readText()) }
    } catch (e: Exception) {
        null
    }

    fun save(ctx: Context, id: String, data: JSONObject) {
        data.put("updated", System.currentTimeMillis())
        val f = file(ctx, id)
        val tmp = File(f.parentFile, f.name + ".tmp")
        tmp.writeText(data.toString())
        tmp.renameTo(f)
    }

    fun delete(ctx: Context, id: String) {
        file(ctx, id).delete()
    }

    fun newId(): String = "s" + System.currentTimeMillis()
}
