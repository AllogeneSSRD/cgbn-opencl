package com.example.ecm

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
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
    private lateinit var inputNExpr: TextInputEditText
    private lateinit var inputNPreset: AutoCompleteTextView
    private lateinit var inputB1: TextInputEditText
    private lateinit var inputB2: TextInputEditText
    private lateinit var inputGpuCurves: TextInputEditText
    private lateinit var inputDeviceIndex: TextInputEditText
    private lateinit var inputGpuCkpt: TextInputEditText
    private lateinit var inputSigma: TextInputEditText
    private lateinit var inputMulPath: TextInputEditText
    private lateinit var inputSqrPath: TextInputEditText
    private lateinit var inputAddPath: TextInputEditText
    private lateinit var inputSubPath: TextInputEditText
    private lateinit var chkVerbose: MaterialCheckBox
    private lateinit var ecmAdvancedPanel: LinearLayout
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
        inputNExpr = findViewById(R.id.input_n_expr)
        inputNPreset = findViewById(R.id.input_n_preset)
        inputB1 = findViewById(R.id.input_b1)
        inputB2 = findViewById(R.id.input_b2)
        inputGpuCurves = findViewById(R.id.input_gpu_curves)
        inputDeviceIndex = findViewById(R.id.input_device_index)
        inputGpuCkpt = findViewById(R.id.input_gpu_ckpt)
        inputSigma = findViewById(R.id.input_sigma)
        inputMulPath = findViewById(R.id.input_mul_path)
        inputSqrPath = findViewById(R.id.input_sqr_path)
        inputAddPath = findViewById(R.id.input_add_path)
        inputSubPath = findViewById(R.id.input_sub_path)
        chkVerbose = findViewById(R.id.chk_verbose)
        ecmAdvancedPanel = findViewById(R.id.ecm_advanced_panel)

        nativeInitAssets(assets, codeCacheDir.absolutePath)

        setupEcmPresets()
        findViewById<MaterialButton>(R.id.btn_toggle_advanced).setOnClickListener {
            val show = ecmAdvancedPanel.visibility != View.VISIBLE
            ecmAdvancedPanel.visibility = if (show) View.VISIBLE else View.GONE
        }
        findViewById<MaterialButton>(R.id.btn_run_ecm).setOnClickListener {
            runEcm()
        }

        findViewById<MaterialButton>(R.id.btn_probe).setOnClickListener {
            runNative { nativeProbe(openClLoadError) }
        }
        findViewById<MaterialButton>(R.id.btn_short).setOnClickListener {
            runNative { nativeShortTest() }
        }
        findViewById<MaterialButton>(R.id.btn_addsub_32).setOnClickListener {
            runAddSubBench(limbBits = 32)
        }
        findViewById<MaterialButton>(R.id.btn_addsub_24).setOnClickListener {
            runAddSubBench(limbBits = 24)
        }
        findViewById<MaterialButton>(R.id.btn_mont_32).setOnClickListener {
            runMontSqrBench(limbBits = 32)
        }
        findViewById<MaterialButton>(R.id.btn_mont_24).setOnClickListener {
            runMontSqrBench(limbBits = 24)
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

    private fun setupEcmPresets() {
        val presets = resources.getStringArray(R.array.ecm_n_presets)
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, presets)
        inputNPreset.setAdapter(adapter)
        inputNPreset.setText(presets[0], false)
        inputNPreset.setOnItemClickListener { _, _, position, _ ->
            if (position < presets.size - 1) {
                inputNExpr.setText(presets[position])
            }
        }
    }

    private fun parseNonNegativeInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value >= 0) value else null
    }

    private fun parseDoubleField(edit: TextInputEditText, fallback: Double, allowZero: Boolean): Double? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toDoubleOrNull() ?: return null
        return if (value > 0.0 || (allowZero && value >= 0.0)) value else null
    }

    private fun runEcm() {
        val nExpr = inputNExpr.text?.toString()?.trim().orEmpty()
        if (nExpr.isEmpty()) {
            Toast.makeText(this, R.string.toast_ecm_invalid, Toast.LENGTH_SHORT).show()
            return
        }
        val b1 = parseDoubleField(inputB1, 2000.0, allowZero = false) ?: run {
            toastInvalid()
            return
        }
        val b2 = parseDoubleField(inputB2, 0.0, allowZero = true) ?: run {
            toastInvalid()
            return
        }
        val gpuCurves = parsePositiveInt(inputGpuCurves, 64) ?: run {
            toastInvalid()
            return
        }
        val deviceIndex = parseNonNegativeInt(inputDeviceIndex, 0) ?: run {
            toastInvalid()
            return
        }
        val gpuCkpt = parseDoubleField(inputGpuCkpt, 600.0, allowZero = true) ?: run {
            toastInvalid()
            return
        }

        setBusy(true)
        outputText.text = ""
        val logCallback = EcmLogCallback { line ->
            mainHandler.post {
                outputText.append(line)
                scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
            }
        }
        benchExecutor.execute {
            val tail = try {
                nativeRunEcm(
                    nExpr,
                    b1,
                    b2,
                    gpuCurves,
                    deviceIndex,
                    chkVerbose.isChecked,
                    gpuCkpt,
                    inputSigma.text?.toString()?.trim().orEmpty(),
                    inputMulPath.text?.toString()?.trim().orEmpty(),
                    inputSqrPath.text?.toString()?.trim().orEmpty(),
                    inputAddPath.text?.toString()?.trim().orEmpty(),
                    inputSubPath.text?.toString()?.trim().orEmpty(),
                    logCallback,
                )
            } catch (e: Exception) {
                "Error: ${e.message}\n"
            }
            runOnUiThread {
                if (tail.isNotEmpty()) {
                    outputText.append(tail)
                }
                setBusy(false)
                scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
            }
        }
    }

    private fun parsePositiveInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value > 0) value else null
    }

    private fun runMontSqrBench(limbBits: Int) {
        val defaultBits = 512
        val bits = parsePositiveInt(inputBits, defaultBits) ?: run {
            toastInvalid()
            return
        }
        val kernelIters = parsePositiveInt(inputKernelIters, 1000) ?: run {
            toastInvalid()
            return
        }
        val instances = parsePositiveInt(inputInstances, 128) ?: run {
            toastInvalid()
            return
        }
        val launchRepeats = parsePositiveInt(inputLaunchRepeats, 1) ?: run {
            toastInvalid()
            return
        }
        if (limbBits == 24) {
            if (bits != 512 && bits % 24 != 0) {
                Toast.makeText(this, R.string.toast_mont_i24_bits, Toast.LENGTH_SHORT).show()
                return
            }
        } else if (bits % 32 != 0) {
            Toast.makeText(this, getString(R.string.toast_bits_multiple, 32), Toast.LENGTH_SHORT).show()
            return
        }
        val useWg = limbBits == 32 && bits != 512
        runNative {
            nativeMontSqrBench(bits, kernelIters, instances, launchRepeats, useWg, tpi = 4, limbBits)
        }
    }

    private fun runAddSubBench(limbBits: Int) {
        val defaultBits = if (limbBits == 24) 504 else 512
        val bits = parsePositiveInt(inputBits, defaultBits) ?: run {
            toastInvalid()
            return
        }
        val kernelIters = parsePositiveInt(inputKernelIters, 1000) ?: run {
            toastInvalid()
            return
        }
        val instances = parsePositiveInt(inputInstances, 128) ?: run {
            toastInvalid()
            return
        }
        val launchRepeats = parsePositiveInt(inputLaunchRepeats, 1) ?: run {
            toastInvalid()
            return
        }
        if (bits % limbBits != 0) {
            Toast.makeText(this, getString(R.string.toast_bits_multiple, limbBits), Toast.LENGTH_SHORT).show()
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
        findViewById<MaterialButton>(R.id.btn_addsub_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_addsub_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_mont_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_mont_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_16).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_32).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_run_ecm).isEnabled = !busy
        inputBits.isEnabled = !busy
        inputKernelIters.isEnabled = !busy
        inputInstances.isEnabled = !busy
        inputLaunchRepeats.isEnabled = !busy
        inputNExpr.isEnabled = !busy
        inputNPreset.isEnabled = !busy
        inputB1.isEnabled = !busy
        inputB2.isEnabled = !busy
        inputGpuCurves.isEnabled = !busy
        inputDeviceIndex.isEnabled = !busy
        inputGpuCkpt.isEnabled = !busy
        inputSigma.isEnabled = !busy
        inputMulPath.isEnabled = !busy
        inputSqrPath.isEnabled = !busy
        inputAddPath.isEnabled = !busy
        inputSubPath.isEnabled = !busy
        chkVerbose.isEnabled = !busy
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(perfRefreshRunnable)
        benchExecutor.shutdownNow()
        perfExecutor.shutdownNow()
        super.onDestroy()
    }

    private external fun nativeInitAssets(
        assetManager: android.content.res.AssetManager,
        cacheDir: String,
    )
    private external fun nativeProbe(openClLoadError: String?): String
    private external fun nativeShortTest(): String
    private external fun nativeMontSqrBench(
        bits: Int,
        kernelIters: Int,
        instances: Int,
        launchRepeats: Int,
        useWg: Boolean,
        tpi: Int,
        limbBits: Int,
    ): String

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

    private external fun nativeRunEcm(
        nExpr: String,
        b1: Double,
        b2: Double,
        gpuCurves: Int,
        deviceIndex: Int,
        verbose: Boolean,
        gpuCkptSec: Double,
        sigma: String,
        mulPath: String,
        sqrPath: String,
        addPath: String,
        subPath: String,
        logCallback: EcmLogCallback?,
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
