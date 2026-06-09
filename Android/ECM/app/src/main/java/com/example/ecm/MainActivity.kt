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
            stringFromJNI()
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
    external fun stringFromJNI(): String

    companion object {
        init {
            // OpenCL loads from /vendor/lib64 at runtime (not packaged — 16 KB page safe).
            System.loadLibrary("ecm")
        }
    }
}