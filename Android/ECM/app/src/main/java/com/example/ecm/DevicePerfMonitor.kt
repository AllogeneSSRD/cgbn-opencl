package com.example.ecm

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import java.io.File
import java.io.FileInputStream

/**
 * Reads device performance hints from sysfs (Qualcomm kgsl / thermal zones).
 * GPU busy/freq prefer native scanner (direct fopen, directory walk).
 */
object DevicePerfMonitor {

    private const val SRC_NONE = 0
    private const val SRC_PROPERTY = 1
    private const val SRC_SYSFS = 2

    data class Snapshot(
        val gpuBusyPercent: String,
        val gpuFreqMhz: String,
        val gpuTempC: String,
        val cpuTempC: String,
        val skinTempC: String,
        val batteryTempC: String,
        val gpuSourceHint: String = "",
    )

    @JvmStatic
    external fun nativeGpuStats(): LongArray?

    fun sample(context: Context): Snapshot {
        val gpu = readGpuStats()
        return Snapshot(
            gpuBusyPercent = gpu.busy,
            gpuFreqMhz = gpu.freq,
            gpuTempC = readThermalByKeyword("gpu", "g3d", "kgsl"),
            cpuTempC = readThermalByKeyword("cpu", "cluster", "soc"),
            skinTempC = readThermalByKeyword("skin", "shell", "board", "quiet"),
            batteryTempC = readBatteryTempC(context),
            gpuSourceHint = gpu.sourceHint,
        )
    }

    fun format(snapshot: Snapshot): String {
        return buildString {
            appendLine("GPU 占用: ${snapshot.gpuBusyPercent}")
            appendLine("GPU 频率: ${snapshot.gpuFreqMhz}")
            if (snapshot.gpuSourceHint.isNotEmpty()) {
                appendLine("GPU 数据源: ${snapshot.gpuSourceHint}")
            }
            appendLine("GPU 温度: ${snapshot.gpuTempC}")
            appendLine("CPU 温度: ${snapshot.cpuTempC}")
            appendLine("机身温度: ${snapshot.skinTempC}")
            append("电池温度: ${snapshot.batteryTempC}")
        }
    }

    private data class GpuSample(val busy: String, val freq: String, val sourceHint: String)

    private fun readGpuStats(): GpuSample {
        try {
            val arr = nativeGpuStats()
            if (arr != null && arr.size >= 2) {
                val busy = arr[0].toInt()
                val freqHz = arr[1]
                val busySrc = if (arr.size >= 3) arr[2].toInt() else SRC_NONE
                val freqSrc = if (arr.size >= 4) arr[3].toInt() else SRC_NONE

                val maxFreqHz = if (arr.size >= 5) arr[4] else -1L

                var busyStr = if (busy >= 0) "$busy %" else unavailable()
                var freqStr = when {
                    freqHz > 0 -> formatFreqHz(freqHz)
                    maxFreqHz > 0 -> "— (max ${formatFreqHz(maxFreqHz)})"
                    else -> unavailable()
                }

                if (busyStr == unavailable()) {
                    busyStr = readGpuBusyPercentFallback()
                }
                if (freqStr == unavailable()) {
                    freqStr = readGpuFreqFallback()
                }

                val hint = buildSourceHint(busySrc, freqSrc)
                return GpuSample(busyStr, freqStr, hint)
            }
        } catch (_: UnsatisfiedLinkError) {
        } catch (_: Exception) {
        }
        return GpuSample(
            readGpuBusyPercentFallback(),
            readGpuFreqFallback(),
            "",
        )
    }

    private fun buildSourceHint(busySrc: Int, freqSrc: Int): String {
        fun label(src: Int): String? = when (src) {
            SRC_PROPERTY -> "系统属性"
            SRC_SYSFS -> "sysfs"
            else -> null
        }
        val parts = mutableListOf<String>()
        label(busySrc)?.let { parts.add("占用=$it") }
        label(freqSrc)?.let { parts.add("频率=$it") }
        return parts.joinToString(", ")
    }

    private fun readGpuBusyPercentFallback(): String {
        val paths = mutableListOf(
            "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
            "/sys/class/kgsl/kgsl-3d0/busy_percentage",
            "/sys/class/kgsl/kgsl-3d0/gpubusy",
            "/sys/devices/virtual/kgsl/kgsl-3d0/gpu_busy_percentage",
            "/sys/kernel/gpu/gpu_busy_percentage",
            "/sys/class/misc/mali0/device/utilization",
        )
        appendKgslChildren(paths, listOf("gpu_busy_percentage", "busy_percentage", "gpubusy", "utilization"))
        scanForLeaf(paths, listOf("gpu_busy_percentage", "busy_percentage", "gpubusy", "utilization"))
        for (path in paths.distinct()) {
            val raw = readTextFile(path) ?: continue
            parseBusy(raw, path.contains("gpubusy"))?.let { return "$it %" }
        }
        return unavailable()
    }

    private fun readGpuFreqFallback(): String {
        val paths = mutableListOf(
            "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
            "/sys/class/kgsl/kgsl-3d0/gpuclk",
            "/sys/class/kgsl/kgsl-3d0/clk_freq",
            "/sys/kernel/gpu/gpu_clock",
            "/sys/class/misc/mali0/device/clock",
        )
        collectDevfreqFreqPaths(paths)
        appendKgslChildren(paths, listOf("cur_freq", "gpuclk", "clk_freq"))
        scanForLeaf(paths, listOf("cur_freq", "gpuclk", "clk_freq", "clock"))
        for (path in paths.distinct()) {
            parseFreqHz(readTextFile(path))?.let { return formatFreqHz(it) }
        }
        return unavailable()
    }

    private fun collectDevfreqFreqPaths(out: MutableList<String>) {
        File("/sys/class/devfreq").listFiles()?.forEach { entry ->
            if (entry.name.contains("kgsl", true) || entry.name.contains("gpu", true) ||
                entry.name.contains("mali", true)
            ) {
                out.add("${entry.absolutePath}/cur_freq")
                out.add("${entry.absolutePath}/userspace/set_freq")
            }
        }
    }

    private fun appendKgslChildren(out: MutableList<String>, leafNames: List<String>) {
        File("/sys/class/kgsl").listFiles()?.forEach { entry ->
            for (leaf in leafNames) {
                out.add("${entry.absolutePath}/$leaf")
                out.add("${entry.absolutePath}/devfreq/$leaf")
            }
        }
    }

    private fun scanForLeaf(out: MutableList<String>, leafNames: List<String>) {
        val roots = listOf("/sys/class/kgsl", "/sys/class/devfreq", "/sys/devices/virtual/kgsl")
        for (root in roots) {
            walk(root, 0, leafNames, out)
        }
    }

    private fun walk(dirPath: String, depth: Int, leafNames: List<String>, out: MutableList<String>) {
        if (depth > 4) return
        val dir = File(dirPath)
        val files = dir.listFiles() ?: return
        val gpuCtx = dirPath.contains("kgsl", true) || dirPath.contains("gpu", true) ||
            dirPath.contains("mali", true) || dirPath == "/sys/class/devfreq"
        for (f in files) {
            if (f.name.startsWith(".")) continue
            if (gpuCtx && leafNames.any { it == f.name }) {
                out.add(f.absolutePath)
            }
            if (f.isDirectory) {
                walk(f.absolutePath, depth + 1, leafNames, out)
            }
        }
    }

    private fun parseBusy(raw: String?, gpubusyPair: Boolean): Int? {
        if (raw == null) return null
        if (gpubusyPair) {
            val parts = raw.split(Regex("\\s+")).filter { it.isNotEmpty() }
            if (parts.size >= 2) {
                val busy = parts[0].toLongOrNull() ?: return null
                val total = parts[1].toLongOrNull() ?: return null
                if (total > 0) {
                    return ((busy * 100) / total).toInt().coerceIn(0, 100)
                }
            }
            return null
        }
        val v = raw.toLongOrNull() ?: return null
        return when {
            v in 0..100 -> v.toInt()
            v in 0..1000 -> (v / 10).toInt()
            else -> 100
        }
    }

    private fun parseFreqHz(raw: String?): Long? {
        var v = raw?.toLongOrNull() ?: return null
        if (v <= 0) return null
        if (v < 1_000_000L) v *= 1_000_000L
        return v
    }

    private fun formatFreqHz(hz: Long): String {
        val mhz = hz / 1_000_000.0
        return if (mhz >= 1000) {
            String.format("%.2f GHz", mhz / 1000.0)
        } else {
            String.format("%.0f MHz", mhz)
        }
    }

    private fun readThermalByKeyword(vararg keywords: String): String {
        val dir = File("/sys/class/thermal")
        if (!dir.isDirectory) {
            return unavailable()
        }
        var best: Pair<String, Int>? = null
        for (zone in dir.listFiles()?.sortedBy { it.name } ?: emptyList()) {
            val type = readTextFile(zone.resolve("type").absolutePath)?.lowercase() ?: continue
            if (keywords.none { type.contains(it) }) {
                continue
            }
            val milliC = readTextFile(zone.resolve("temp").absolutePath)?.toIntOrNull() ?: continue
            val priority = keywords.indexOfFirst { type.contains(it) }
            if (best == null || priority < best.second) {
                best = formatTempMilliC(milliC) to priority
            }
        }
        return best?.first ?: unavailable()
    }

    private fun readBatteryTempC(context: Context): String {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val batteryStatus = context.registerReceiver(null, filter) ?: return unavailable()
        val temp = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        if (temp == Int.MIN_VALUE) {
            return unavailable()
        }
        return String.format("%.1f °C", temp / 10.0)
    }

    private fun formatTempMilliC(milliC: Int): String {
        return String.format("%.1f °C", milliC / 1000.0)
    }

    /** Do not use [File.canRead]; sysfs often rejects canRead but allows open. */
    private fun readTextFile(path: String): String? {
        try {
            FileInputStream(path).use { fis ->
                val buf = ByteArray(128)
                val n = fis.read(buf)
                if (n <= 0) return null
                return String(buf, 0, n).trim().lineSequence().firstOrNull()?.trim()
            }
        } catch (_: Exception) {
            return try {
                File(path).readText().trim().lineSequence().firstOrNull()?.trim()
            } catch (_: Exception) {
                null
            }
        }
    }

    private fun unavailable(): String = "—"
}
