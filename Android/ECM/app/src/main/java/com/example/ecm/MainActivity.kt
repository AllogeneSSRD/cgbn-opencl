package com.example.ecm

import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var outputText: TextView
    private lateinit var progress: ProgressBar
    private lateinit var scroll: NestedScrollView
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

        findViewById<MaterialButton>(R.id.btn_probe).setOnClickListener {
            runNative { nativeProbe(openClLoadError) }
        }
        findViewById<MaterialButton>(R.id.btn_short).setOnClickListener {
            runNative { nativeShortTest() }
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
        findViewById<MaterialButton>(R.id.btn_bench_16).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_24).isEnabled = !busy
        findViewById<MaterialButton>(R.id.btn_bench_32).isEnabled = !busy
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }

    private external fun nativeProbe(openClLoadError: String?): String
    private external fun nativeShortTest(): String
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
