package com.example.ecm

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.example.ecm.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.sampleText.text = try {
            stringFromJNI(openClLoadError)
        } catch (e: UnsatisfiedLinkError) {
            "Native library error:\n${e.message}"
        } catch (e: Exception) {
            "Error:\n${e.message}"
        }
    }

    /**
     * A native method that is implemented by the 'ecm' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(openClLoadError: String?): String

    companion object {
        /** Set when vendor libOpenCL is not exposed to this app (see uses-native-library). */
        @JvmField
        var openClLoadError: String? = null

        init {
            // Must load before ecm: uses-native-library whitelists vendor libOpenCL.so (API 31+).
            try {
                System.loadLibrary("OpenCL")
            } catch (e: UnsatisfiedLinkError) {
                openClLoadError = e.message
            }
            System.loadLibrary("ecm")
        }
    }
}