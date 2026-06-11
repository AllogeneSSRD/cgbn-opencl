package com.example.ecm

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Append-only log files for ECM runs and micro-benchmarks (separate files). */
class RunLogStore(context: Context) {

    enum class Channel {
        ECM,
        BENCH,
    }

    private val logDir: File = AppStoragePaths.logsDir(context.applicationContext)

    private val stampFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    private val lock = Any()

    fun logFile(channel: Channel): File {
        return File(logDir, fileName(channel))
    }

    fun displayPath(channel: Channel): String = logFile(channel).absolutePath

    fun beginSession(channel: Channel, header: String? = null) {
        synchronized(lock) {
            val file = logFile(channel)
            file.appendText("\n=== ${stampFormat.format(Date())} ===\n")
            if (!header.isNullOrBlank()) {
                file.appendText(header.trim())
                if (!header.endsWith("\n")) {
                    file.appendText("\n")
                }
            }
        }
    }

    fun append(channel: Channel, text: String) {
        if (text.isEmpty()) {
            return
        }
        synchronized(lock) {
            logFile(channel).appendText(text)
        }
    }

    private fun fileName(channel: Channel): String {
        return when (channel) {
            Channel.ECM -> ECM_FILE_NAME
            Channel.BENCH -> BENCH_FILE_NAME
        }
    }

    companion object {
        private const val ECM_FILE_NAME = "ecm.log"
        private const val BENCH_FILE_NAME = "bench.log"
    }
}
