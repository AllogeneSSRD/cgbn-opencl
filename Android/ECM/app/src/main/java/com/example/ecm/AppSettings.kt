package com.example.ecm

import android.content.Context
import androidx.appcompat.app.AppCompatDelegate
import java.io.File
import java.util.concurrent.atomic.AtomicReference

object AppSettings {

    private const val LEGACY_PREFS = "ecm_app_settings"
    private const val KEY_FOLLOW_SYSTEM = "follow_system"
    private const val KEY_DARK_MODE = "dark_mode"
    private const val KEY_LOG_TO_FILE = "log_to_file"
    private const val KEY_LOG_SAVED_TOAST = "log_saved_toast"

    private val defaults = mapOf(
        KEY_FOLLOW_SYSTEM to true,
        KEY_DARK_MODE to false,
        KEY_LOG_TO_FILE to true,
        KEY_LOG_SAVED_TOAST to true,
    )

    private val lock = Any()
    private val loadedContext = AtomicReference<Context?>(null)
    private var values: MutableMap<String, Boolean> = defaults.toMutableMap()

    fun settingsFile(context: Context): File = AppStoragePaths.settingsFile(context)

    fun isFollowSystem(context: Context): Boolean = getBoolean(context, KEY_FOLLOW_SYSTEM)

    fun isDarkMode(context: Context): Boolean = getBoolean(context, KEY_DARK_MODE)

    fun isLogToFileEnabled(context: Context): Boolean = getBoolean(context, KEY_LOG_TO_FILE)

    fun isLogSavedToastEnabled(context: Context): Boolean = getBoolean(context, KEY_LOG_SAVED_TOAST)

    fun setFollowSystem(context: Context, enabled: Boolean) {
        setBoolean(context, KEY_FOLLOW_SYSTEM, enabled)
        applyTheme(context)
    }

    fun setDarkMode(context: Context, enabled: Boolean) {
        setBoolean(context, KEY_DARK_MODE, enabled)
        applyTheme(context)
    }

    fun setLogToFileEnabled(context: Context, enabled: Boolean) {
        setBoolean(context, KEY_LOG_TO_FILE, enabled)
    }

    fun setLogSavedToastEnabled(context: Context, enabled: Boolean) {
        setBoolean(context, KEY_LOG_SAVED_TOAST, enabled)
    }

    fun applyTheme(context: Context) {
        AppCompatDelegate.setDefaultNightMode(resolveNightMode(context))
    }

    private fun resolveNightMode(context: Context): Int {
        if (isFollowSystem(context)) {
            return AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
        }
        return if (isDarkMode(context)) {
            AppCompatDelegate.MODE_NIGHT_YES
        } else {
            AppCompatDelegate.MODE_NIGHT_NO
        }
    }

    private fun getBoolean(context: Context, key: String): Boolean {
        ensureLoaded(context)
        return values[key] ?: defaults[key] ?: false
    }

    private fun setBoolean(context: Context, key: String, enabled: Boolean) {
        synchronized(lock) {
            ensureLoadedLocked(context)
            values[key] = enabled
            persistLocked(context)
        }
    }

    private fun ensureLoaded(context: Context) {
        synchronized(lock) {
            ensureLoadedLocked(context)
        }
    }

    private fun ensureLoadedLocked(context: Context) {
        val app = context.applicationContext
        if (loadedContext.get() === app) {
            return
        }
        values = loadOrMigrate(app).toMutableMap()
        loadedContext.set(app)
    }

    private fun loadOrMigrate(context: Context): Map<String, Boolean> {
        val file = settingsFile(context)
        if (file.isFile) {
            return parseSettingsXml(file.readText())
        }
        val legacy = context.getSharedPreferences(LEGACY_PREFS, Context.MODE_PRIVATE)
        val merged = defaults.toMutableMap()
        for ((key, value) in legacy.all) {
            if (value is Boolean) {
                merged[key] = value
            }
        }
        if (legacy.all.isNotEmpty()) {
            persistToFile(file, merged)
            legacy.edit().clear().apply()
        }
        return merged
    }

    private fun persistLocked(context: Context) {
        persistToFile(settingsFile(context), values)
    }

    private fun persistToFile(file: File, data: Map<String, Boolean>) {
        file.parentFile?.mkdirs()
        val body = buildString {
            append("<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n")
            append("<map>\n")
            for (key in defaults.keys) {
                val value = data[key] ?: defaults[key] ?: false
                append("    <boolean name=\"")
                append(key)
                append("\" value=\"")
                append(value)
                append("\" />\n")
            }
            append("</map>\n")
        }
        file.writeText(body)
    }

    private fun parseSettingsXml(xml: String): Map<String, Boolean> {
        val merged = defaults.toMutableMap()
        val pattern = Regex("""<boolean\s+name="([^"]+)"\s+value="(true|false)"\s*/>""")
        for (match in pattern.findAll(xml)) {
            merged[match.groupValues[1]] = match.groupValues[2] == "true"
        }
        return merged
    }
}
