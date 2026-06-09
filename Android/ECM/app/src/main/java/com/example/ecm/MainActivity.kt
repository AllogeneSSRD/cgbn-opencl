package com.example.ecm

import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var outputText: TextView
    private lateinit var progress: ProgressBar
    private lateinit var scroll: NestedScrollView
    private lateinit var inputBits: TextInputEditText
    private lateinit var inputKernelIters: TextInputEditText
    private lateinit var inputInstances: TextInputEditText
    private lateinit var inputLaunchRepeats: TextInputEditText
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        findViewById<MaterialToolbar>(R.id.toolbar).apply {
            subtitle = getString(R.string.subtitle)
        }

        outputText = findViewById(R.id.output_text)
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
            runAddSubBench()
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

    private fun parsePositiveInt(edit: TextInputEditText, fallback: Int): Int? {
        val raw = edit.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return fallback
        val value = raw.toIntOrNull() ?: return null
        return if (value > 0) value else null
    }

    private fun runAddSubBench() {
        val bits = parsePositiveInt(inputBits, 512) ?: run {
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
        if (bits % 32 != 0) {
            Toast.makeText(this, "bits 必须是 32 的倍数", Toast.LENGTH_SHORT).show()
            return
        }
        runNative {
            nativeAddSubBench(bits, kernelIters, instances, launchRepeats)
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
        executor.execute {
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
        findViewById<MaterialButton>(R.id.btn_bench_16).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_32).isEnabled = !busy
        inputBits.isEnabled = !busy
        inputKernelIters.isEnabled = !busy
        inputInstances.isEnabled = !busy
        inputLaunchRepeats.isEnabled = !busy
    }

    override fun onDestroy() {
        executor.shutdownNow()
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
    ): String
    private external fun nativeBitBench(
        limbBits: Int,
        elements: Int,
        kernelIters: Int,
        launchRepeats: Int,
    ): String

    companion object {
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
