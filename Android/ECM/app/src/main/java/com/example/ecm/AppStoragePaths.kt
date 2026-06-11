package com.example.ecm

import android.content.Context
import java.io.File

/**
 * Writable app data under `/Android/data/<package>/` (siblings of `files/`), for logs,
 * settings XML, and OpenCL compile cache — visible in file managers when permitted.
 */
object AppStoragePaths {

    const val DIR_LOGS = "logs"
    const val DIR_CONFIG = "config"
    const val SETTINGS_FILE = "ecm_app_settings.xml"

    /** `/Android/data/com.example.ecm/` or null if external storage unavailable. */
    fun externalPackageRoot(context: Context): File? {
        return context.applicationContext.getExternalFilesDir(null)?.parentFile
    }

    fun resolveDir(context: Context, dirName: String): File {
        val root = externalPackageRoot(context)
        val dir = if (root != null) {
            File(root, dirName)
        } else {
            File(context.applicationContext.filesDir, dirName)
        }
        dir.mkdirs()
        return dir
    }

    fun logsDir(context: Context): File = resolveDir(context, DIR_LOGS)

    fun configDir(context: Context): File = resolveDir(context, DIR_CONFIG)

    fun settingsFile(context: Context): File = File(configDir(context), SETTINGS_FILE)

    /**
     * Root passed to native `set_opencl_cache_dir`; native appends `opencl_cache/`.
     * Result: `/Android/data/<package>/opencl_cache/`.
     */
    fun openClCacheRoot(context: Context): File {
        val root = externalPackageRoot(context)
        return if (root != null) {
            root.mkdirs()
            root
        } else {
            context.applicationContext.filesDir
        }
    }
}
