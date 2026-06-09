package com.example.ecm

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var outputText: TextView
    private lateinit var perfText: TextView
    private lateinit var perfUpdated: TextView
    private lateinit var progress: ProgressBar
    private lateinit var scroll: NestedScrollView
    private lateinit var inputBits: TextInputEditText
    private lateinit var inputKernelIters: TextInputEditText
    private lateinit var inputInstances: TextInputEditText
    private lateinit var inputLaunchRepeats: TextInputEditText
    private val benchExecutor = Executors.newSingleThreadExecutor()
    private val perfExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    @Volatile
    private var perfSampleRunning = false

    private val perfRefreshRunnable = object : Runnable {
        override fun run() {
            refreshPerfStats()
            mainHandler.postDelayed(this, PERF_REFRESH_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<MaterialToolbar>(R.id.toolbar).apply {
            subtitle = getString(R.string.subtitle)
        }

        outputText = findViewById(R.id.output_text)
        perfText = findViewById(R.id.perf_text)
        perfUpdated = findViewById(R.id.perf_updated)
        progress = findViewById(R.id.progress)
        scroll = findViewById(R.id.nestedScrollView)
        inputBits = findViewById(R.id.input_bits)
        inputKernelIters = findViewById(R.id.input_kernel_iters)
        inputInstances = findViewById(R.id.input_instances)
        inputLaunchRepeats = findViewById(R.id.input_launch_repeats)

        nativeInitAssets(assets)

        findViewById<MaterialButton>(R.id.btn_probe).setOnClickListener {
            runNative { nativeProbe(openClLoadError) }
        }
        findViewById<MaterialButton>(R.id.btn_short).setOnClickListener {
            runNative { nativeShortTest() }
        }
        findViewById<MaterialButton>(R.id.btn_addsub_bench).setOnClickListener {
            runAddSubBench(limbBits = 32)
        }
        findViewById<MaterialButton>(R.id.btn_addsub_bench_24).setOnClickListener {
            runAddSubBench(limbBits = 24)
        }
        findViewById<MaterialButton>(R.id.btn_bench_16).setOnClickListener {
            runBench(16)
        }
        findViewById<MaterialButton>(R.id.btn_bench_24).setOnClickListener {
            runBench(24)
        }
        findViewById<MaterialButton>(R.id.btn_bench_32).setOnClickListener {
            runBench(32)
        }

        runNative { nativeProbe(openClLoadError) }
    }

    override fun onResume() {
        super.onResume()
        mainHandler.removeCallbacks(perfRefreshRunnable)
        refreshPerfStats()
        mainHandler.postDelayed(perfRefreshRunnable, PERF_REFRESH_MS)
    }

    override fun onPause() {
        mainHandler.removeCallbacks(perfRefreshRunnable)
        super.onPause()
    }

    private fun refreshPerfStats() {
        if (perfSampleRunning) {
            return
        }
        perfSampleRunning = true
        perfExecutor.execute {
            try {
                val snapshot = DevicePerfMonitor.sample(applicationContext)
                val body = DevicePerfMonitor.format(snapshot)
                val stamp = getString(R.string.perf_updated, timeFormat.format(Date()))
                runOnUiThread {
                    perfText.text = body
                    perfUpdated.text = stamp
                }
            } finally {
                perfSampleRunning = false
            }
        }
    }

    private fun parsePositiveInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value > 0) value else null
    }

    private fun runAddSubBench(limbBits: Int) {
        val defaultBits = if (limbBits == 24) 504 else 512
        val bits = parsePositiveInt(inputBits, defaultBits) ?: run {
            toastInvalid()
            return
        }
        val kernelIters = parsePositiveInt(inputKernelIters, 10000) ?: run {
            toastInvalid()
            return
        }
        val instances = parsePositiveInt(inputInstances, 64) ?: run {
            toastInvalid()
            return
        }
        val launchRepeats = parsePositiveInt(inputLaunchRepeats, 1) ?: run {
            toastInvalid()
            return
        }
        if (bits % limbBits != 0) {
            Toast.makeText(
                this,
                "bits 必须是 $limbBits 的倍数",
                Toast.LENGTH_SHORT,
            ).show()
            return
        }
        runNative {
            nativeAddSubBench(bits, kernelIters, instances, launchRepeats, limbBits)
        }
    }

    private fun toastInvalid() {
        Toast.makeText(this, R.string.invalid_params, Toast.LENGTH_SHORT).show()
    }

    private fun runBench(bits: Int) {
        runNative {
            nativeBitBench(bits, elements = 1 shl 18, kernelIters = 64, launchRepeats = 8)
        }
    }

    private fun runNative(block: () -> String) {
        setBusy(true)
        benchExecutor.execute {
            val result = try {
                block()
            } catch (e: Exception) {
                "Error: ${e.message}"
            }
            runOnUiThread {
                outputText.text = result
                setBusy(false)
                scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        findViewById<MaterialButton>(R.id.btn_probe).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_short).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_addsub_bench).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_addsub_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_16).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_32).isEnabled = !busy
        inputBits.isEnabled = !busy
        inputKernelIters.isEnabled = !busy
        inputInstances.isEnabled = !busy
        inputLaunchRepeats.isEnabled = !busy
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(perfRefreshRunnable)
        benchExecutor.shutdownNow()
        perfExecutor.shutdownNow()
        super.onDestroy()
    }

    private external fun nativeInitAssets(assetManager: android.content.res.AssetManager)
    private external fun nativeProbe(openClLoadError: String?): String
    private external fun nativeShortTest(): String
    private external fun nativeAddSubBench(
        bits: Int,
        kernelIters: Int,
        instances: Int,
        launchRepeats: Int,
        limbBits: Int,
    ): String
    private external fun nativeBitBench(
        limbBits: Int,
        elements: Int,
        kernelIters: Int,
        launchRepeats: Int,
    ): String

    companion object {
        private const val PERF_REFRESH_MS = 1500L

        @JvmField
        var openClLoadError: String? = null

        init {
            try {
                System.loadLibrary("OpenCL")
            } catch (e: UnsatisfiedLinkError) {
                openClLoadError = e.message
            }
            System.loadLibrary("ecm")
        }
    }
}
