package com.example.ecm

import android.graphics.Color
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updatePadding
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.switchmaterial.SwitchMaterial

class SettingsActivity : AppCompatActivity() {

    private lateinit var switchFollowSystem: SwitchMaterial
    private lateinit var switchDarkMode: SwitchMaterial
    private lateinit var switchLogToFile: SwitchMaterial
    private lateinit var switchLogToast: SwitchMaterial
    private var suppressCallbacks = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_settings)
        setupWindowInsets()

        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar)
        toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }

        switchFollowSystem = findViewById(R.id.switch_follow_system)
        switchDarkMode = findViewById(R.id.switch_dark_mode)
        switchLogToFile = findViewById(R.id.switch_log_to_file)
        switchLogToast = findViewById(R.id.switch_log_toast)

        bindFromPrefs()
        setupListeners()
    }

    private fun setupWindowInsets() {
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT
        WindowInsetsControllerCompat(window, window.decorView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = !AppSettings.isDarkMode(this@SettingsActivity) &&
                !AppSettings.isFollowSystem(this@SettingsActivity)
        }
        val appBar = findViewById<AppBarLayout>(R.id.app_bar)
        ViewCompat.setOnApplyWindowInsetsListener(appBar) { view, insets ->
            val top = insets.getInsets(
                WindowInsetsCompat.Type.statusBars() or WindowInsetsCompat.Type.displayCutout(),
            ).top
            view.updatePadding(top = top)
            insets
        }
    }

    private fun bindFromPrefs() {
        suppressCallbacks = true
        switchFollowSystem.isChecked = AppSettings.isFollowSystem(this)
        switchDarkMode.isChecked = AppSettings.isDarkMode(this)
        switchLogToFile.isChecked = AppSettings.isLogToFileEnabled(this)
        switchLogToast.isChecked = AppSettings.isLogSavedToastEnabled(this)
        updateDarkModeEnabled()
        updateLogToastEnabled()
        suppressCallbacks = false
    }

    private fun setupListeners() {
        switchFollowSystem.setOnCheckedChangeListener { _, checked ->
            if (suppressCallbacks) return@setOnCheckedChangeListener
            AppSettings.setFollowSystem(this, checked)
            updateDarkModeEnabled()
            onThemePreferenceChanged()
        }
        switchDarkMode.setOnCheckedChangeListener { _, checked ->
            if (suppressCallbacks) return@setOnCheckedChangeListener
            AppSettings.setDarkMode(this, checked)
            onThemePreferenceChanged()
        }
        switchLogToFile.setOnCheckedChangeListener { _, checked ->
            if (suppressCallbacks) return@setOnCheckedChangeListener
            AppSettings.setLogToFileEnabled(this, checked)
            updateLogToastEnabled()
            setResult(RESULT_OK)
        }
        switchLogToast.setOnCheckedChangeListener { _, checked ->
            if (suppressCallbacks) return@setOnCheckedChangeListener
            AppSettings.setLogSavedToastEnabled(this, checked)
            setResult(RESULT_OK)
        }
    }

    private fun updateDarkModeEnabled() {
        val follow = switchFollowSystem.isChecked
        switchDarkMode.isEnabled = !follow
        findViewById<View>(R.id.row_dark_mode).alpha = if (follow) 0.5f else 1f
    }

    private fun updateLogToastEnabled() {
        val enabled = switchLogToFile.isChecked
        switchLogToast.isEnabled = enabled
        findViewById<View>(R.id.row_log_toast).alpha = if (enabled) 1f else 0.5f
    }

    private fun onThemePreferenceChanged() {
        setResult(RESULT_OK)
        recreate()
    }
}
